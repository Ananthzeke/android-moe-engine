#include "layered_store.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <utility>

namespace ame::residency {
namespace {

bool power_of_two(std::uint32_t value) { return value != 0 && (value & (value - 1)) == 0; }

}  // namespace

LayerLease::LayerLease(LayerLease&& other) noexcept
    : pins_(other.pins_), count_(other.count_), layer_(other.layer_), store_(other.store_) {
  other.count_ = 0;
  other.store_ = nullptr;
}

LayerLease& LayerLease::operator=(LayerLease&& other) noexcept {
  if (this != &other) {
    release();
    pins_ = other.pins_;
    count_ = other.count_;
    layer_ = other.layer_;
    store_ = other.store_;
    other.count_ = 0;
    other.store_ = nullptr;
  }
  return *this;
}

LayerLease::~LayerLease() { release(); }

void LayerLease::release() noexcept {
  if (store_ != nullptr && count_ != 0) {
    store_->release(*this);
  }
  count_ = 0;
  store_ = nullptr;
}

std::optional<std::uint32_t> LayerLease::slot_index(std::uint32_t expert) const {
  for (std::uint32_t index = 0; index < count_; ++index) {
    if (pins_[index].expert == expert) {
      return pins_[index].slot;
    }
  }
  return std::nullopt;
}

LayeredStore::LayeredStore(FileSource source, std::vector<Pool> pools, LayeredOptions options)
    : source_(std::move(source)), pools_(std::move(pools)), options_(options) {
  std::uint64_t total = 0;
  for (auto& pool : pools_) {
    pool.base_offset = total;
    total += pool.slot_bytes * pool.slots;
  }
  for (const auto& pool : pools_) {
    metadata_bytes_ += pool.experts.size() * sizeof(LayerExpert) +
                       pool.slot_of.size() * sizeof(std::int32_t) +
                       pool.slot_state.size() * sizeof(Slot) + pool.taken.size();
  }
  metadata_bytes_ += pools_.size() * sizeof(Pool);
  if (!options.external_arena.empty()) {
    arena_ = options.external_arena.first(static_cast<std::size_t>(total));
    return;
  }
  void* raw = nullptr;
  if (::posix_memalign(&raw, options.alignment, static_cast<std::size_t>(total)) == 0) {
    arena_owner_ = std::unique_ptr<std::byte, void (*)(void*)>(
        static_cast<std::byte*>(raw), [](void* pointer) { std::free(pointer); });
    arena_ = std::span<std::byte>(arena_owner_.get(), static_cast<std::size_t>(total));
  }
}

LayeredStore::~LayeredStore() = default;

Result<std::unique_ptr<LayeredStore>, StoreError> LayeredStore::create(
    FileSource source, std::vector<LayerSpec> layers, LayeredOptions options) {
  if (layers.empty() || !power_of_two(options.alignment) ||
      options.alignment < model::kTensorAlignment || options.alignment > 4096) {
    return StoreError::InvalidOptions;
  }
  std::uint64_t total = 0;
  std::vector<Pool> pools;
  pools.reserve(layers.size());
  for (auto& layer : layers) {
    if (layer.slots == 0 || layer.experts.empty() || layer.slots > layer.experts.size() ||
        layer.slot_bytes == 0 || layer.experts.size() > 65536) {
      return StoreError::InvalidOptions;
    }
    if (layer.slot_bytes % options.alignment != 0) {
      return StoreError::Misaligned;
    }
    for (const auto& expert : layer.experts) {
      if (expert.slice_count == 0 || expert.slice_count > kMaxSlicesPerSlot) {
        return StoreError::InvalidOptions;
      }
      for (std::uint32_t index = 0; index < expert.slice_count; ++index) {
        const SliceExtent& slice = expert.slices[index];
        if (slice.length.value == 0 || slice.offset.value % model::kTensorAlignment != 0) {
          return StoreError::Misaligned;
        }
        if (slice.offset.value > source.size_bytes() ||
            slice.length.value > source.size_bytes() - slice.offset.value) {
          return StoreError::EntryBeyondFile;
        }
        if (slice.slot_offset > layer.slot_bytes ||
            slice.length.value > layer.slot_bytes - slice.slot_offset) {
          return StoreError::Overflow;
        }
        // Slices of one expert must not overlap inside the slot.
        for (std::uint32_t other = 0; other < index; ++other) {
          const SliceExtent& previous = expert.slices[other];
          if (slice.slot_offset < previous.slot_offset + previous.length.value &&
              previous.slot_offset < slice.slot_offset + slice.length.value) {
            return StoreError::Overflow;
          }
        }
      }
    }
    if (layer.slot_bytes > std::numeric_limits<std::uint64_t>::max() / layer.slots ||
        layer.slot_bytes * layer.slots > std::numeric_limits<std::uint64_t>::max() - total) {
      return StoreError::Overflow;
    }
    total += layer.slot_bytes * layer.slots;
  }
  if (!options.external_arena.empty() &&
      (options.external_arena.size() < total ||
       reinterpret_cast<std::uintptr_t>(options.external_arena.data()) % options.alignment != 0)) {
    return StoreError::Misaligned;
  }
  for (auto& layer : layers) {
    Pool pool;
    pool.slot_of.assign(layer.experts.size(), -1);
    pool.slot_state.assign(layer.slots, Slot{});
    pool.taken.assign(layer.slots, 0);
    pool.slot_bytes = layer.slot_bytes;
    pool.slots = layer.slots;
    pool.experts = std::move(layer.experts);
    pools.push_back(std::move(pool));
  }
  if (options.fadvise_random) {
    source.advise_random();
  }
  auto store =
      std::unique_ptr<LayeredStore>(new LayeredStore(std::move(source), std::move(pools), options));
  if (store->arena_.empty() && total != 0) {
    return StoreError::ArenaAllocationFailed;
  }
  return store;
}

LayeredStats LayeredStore::stats() const noexcept {
  LayeredStats result = stats_;
  result.pread_calls = source_.pread_calls();
  result.pread_bytes = source_.pread_bytes();
  return result;
}

std::span<std::byte> LayeredStore::pool_arena(std::uint32_t layer) noexcept {
  const Pool& pool = pools_[layer];
  return arena_.subspan(static_cast<std::size_t>(pool.base_offset),
                        static_cast<std::size_t>(pool.slot_bytes * pool.slots));
}

std::uint32_t LayeredStore::resident_slots() const noexcept {
  std::uint32_t count = 0;
  for (const auto& pool : pools_) {
    for (const auto& slot : pool.slot_state) {
      if (slot.expert >= 0) ++count;
    }
  }
  return count;
}

void LayeredStore::list_remove(Pool& pool, std::uint32_t slot_index) noexcept {
  Slot& slot = pool.slot_state[slot_index];
  if (!slot.listed) return;
  if (slot.prev >= 0) {
    pool.slot_state[static_cast<std::size_t>(slot.prev)].next = slot.next;
  } else {
    pool.head = slot.next;
  }
  if (slot.next >= 0) {
    pool.slot_state[static_cast<std::size_t>(slot.next)].prev = slot.prev;
  } else {
    pool.tail = slot.prev;
  }
  slot.prev = slot.next = -1;
  slot.listed = false;
}

void LayeredStore::list_push_back(Pool& pool, std::uint32_t slot_index) noexcept {
  list_remove(pool, slot_index);
  Slot& slot = pool.slot_state[slot_index];
  slot.prev = pool.tail;
  slot.next = -1;
  if (pool.tail >= 0) {
    pool.slot_state[static_cast<std::size_t>(pool.tail)].next =
        static_cast<std::int32_t>(slot_index);
  } else {
    pool.head = static_cast<std::int32_t>(slot_index);
  }
  pool.tail = static_cast<std::int32_t>(slot_index);
  slot.listed = true;
}

// The oldest unpinned, untaken resident slot; for LFU the one with the lowest frequency,
// oldest first among equals. The walk only passes pinned slots, at most one lease's worth.
std::int32_t LayeredStore::choose_victim(Pool& pool, const std::vector<std::uint8_t>& taken) const {
  std::int32_t best = -1;
  std::uint32_t best_frequency = std::numeric_limits<std::uint32_t>::max();
  for (std::int32_t index = pool.head; index >= 0;
       index = pool.slot_state[static_cast<std::size_t>(index)].next) {
    const Slot& slot = pool.slot_state[static_cast<std::size_t>(index)];
    if (taken[static_cast<std::size_t>(index)] != 0 || slot.pins != 0 || slot.expert < 0) {
      continue;
    }
    if (options_.policy == EvictionPolicy::Lru) {
      return index;
    }
    if (slot.frequency < best_frequency) {
      best_frequency = slot.frequency;
      best = index;
    }
  }
  return best;
}

Result<LayerLease, StoreError> LayeredStore::acquire(std::uint32_t layer,
                                                     std::span<const std::uint32_t> experts,
                                                     const Cancellation& stop) {
  if (layer >= pools_.size()) {
    return StoreError::UnknownKey;
  }
  Pool& pool = pools_[layer];
  if (experts.empty()) {
    return LayerLease{};
  }
  if (experts.size() > kMaxLeaseExperts) {
    return StoreError::TooManyKeys;
  }
  std::array<std::uint32_t, kMaxLeaseExperts> unique{};
  std::uint32_t unique_count = 0;
  for (const std::uint32_t expert : experts) {
    if (expert >= pool.experts.size()) {
      return StoreError::UnknownKey;
    }
    bool seen = false;
    for (std::uint32_t existing = 0; existing < unique_count; ++existing) {
      if (unique[existing] == expert) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      unique[unique_count++] = expert;
    }
  }

  std::vector<std::uint8_t>& taken = pool.taken;
  std::fill(taken.begin(), taken.end(), static_cast<std::uint8_t>(0));
  std::array<std::uint32_t, kMaxLeaseExperts> hit_experts{};
  std::uint32_t hit_count = 0;
  std::array<std::uint32_t, kMaxLeaseExperts> miss_experts{};
  std::uint32_t miss_count = 0;
  for (std::uint32_t index = 0; index < unique_count; ++index) {
    const std::int32_t slot = pool.slot_of[unique[index]];
    if (slot >= 0) {
      hit_experts[hit_count++] = unique[index];
      taken[static_cast<std::size_t>(slot)] = 1;
    } else {
      miss_experts[miss_count++] = unique[index];
    }
  }
  if (miss_count > pool.slots) {
    return StoreError::ResourceLimit;
  }

  std::array<std::uint32_t, kMaxLeaseExperts> chosen{};
  std::uint32_t chosen_count = 0;
  for (std::uint32_t slot = 0; slot < pool.slots && chosen_count < miss_count; ++slot) {
    if (pool.slot_state[slot].expert < 0 && taken[slot] == 0) {
      chosen[chosen_count++] = slot;
      taken[slot] = 1;
    }
  }
  while (chosen_count < miss_count) {
    const std::int32_t victim = choose_victim(pool, taken);
    if (victim < 0) {
      return StoreError::ResourceLimit;
    }
    taken[static_cast<std::size_t>(victim)] = 1;
    chosen[chosen_count++] = static_cast<std::uint32_t>(victim);
  }

  ++pool.acquires;
  if (options_.policy == EvictionPolicy::Lfu && options_.lfu_halve_every != 0 &&
      pool.acquires % options_.lfu_halve_every == 0) {
    for (auto& slot : pool.slot_state) slot.frequency /= 2;
  }

  LayerLease lease;
  lease.store_ = this;
  lease.layer_ = layer;
  for (std::uint32_t index = 0; index < hit_count; ++index) {
    const auto slot = static_cast<std::uint32_t>(pool.slot_of[hit_experts[index]]);
    ++pool.slot_state[slot].pins;
    ++pool.slot_state[slot].frequency;
    lease.pins_[lease.count_++] = LayerLease::Pin{hit_experts[index], slot};
    ++stats_.hits;
  }
  for (std::uint32_t index = 0; index < miss_count; ++index) {
    const std::uint32_t slot = chosen[index];
    Slot& state = pool.slot_state[slot];
    if (state.expert >= 0) {
      pool.slot_of[static_cast<std::size_t>(state.expert)] = -1;
      list_remove(pool, slot);
      ++stats_.evictions;
    }
    state.expert = static_cast<std::int32_t>(miss_experts[index]);
    state.pins = 1;
    state.frequency = 1;
    pool.slot_of[miss_experts[index]] = static_cast<std::int32_t>(slot);
    lease.pins_[lease.count_++] = LayerLease::Pin{miss_experts[index], slot};
    ++stats_.misses;
  }

  // On failure the loaded slots are emptied, since their bytes are incomplete, and the hit
  // slots keep their place in the recency order.
  const auto rollback = [&]() noexcept {
    for (std::uint32_t index = hit_count; index < lease.count_; ++index) {
      const std::uint32_t slot = lease.pins_[index].slot;
      Slot& state = pool.slot_state[slot];
      pool.slot_of[lease.pins_[index].expert] = -1;
      list_remove(pool, slot);
      state = Slot{};
    }
    for (std::uint32_t index = 0; index < hit_count; ++index) {
      Slot& state = pool.slot_state[lease.pins_[index].slot];
      if (state.pins > 0) --state.pins;
    }
    lease.count_ = 0;
    lease.store_ = nullptr;
  };

  for (std::uint32_t index = 0; index < miss_count; ++index) {
    if (stop.requested()) {
      rollback();
      return StoreError::Cancelled;
    }
    const std::uint32_t slot = chosen[index];
    const LayerExpert& expert = pool.experts[miss_experts[index]];
    std::byte* slot_base = arena_.data() + pool.base_offset + slot * pool.slot_bytes;
    for (std::uint32_t part = 0; part < expert.slice_count; ++part) {
      const SliceExtent& slice = expert.slices[part];
      const auto destination = std::span<std::byte>(slot_base + slice.slot_offset,
                                                    static_cast<std::size_t>(slice.length.value));
      const auto read_begin = std::chrono::steady_clock::now();
      const auto read = source_.read_exact(slice.offset.value, destination, stop);
      stats_.read_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - read_begin).count();
      if (!read.ok()) {
        rollback();
        return read.error() == FileError::Cancelled ? StoreError::Cancelled
                                                    : StoreError::ReadFailed;
      }
      ++stats_.slice_reads;
      stats_.read_bytes += slice.length.value;
      if (options_.fadvise_dontneed) {
        source_.advise_dontneed(slice.offset.value, slice.length.value);
      }
    }
    list_push_back(pool, slot);
  }
  // Hits and loaded slots are pinned for the lease; recency is stamped on release.
  return lease;
}

void LayeredStore::release(const LayerLease& lease) noexcept {
  Pool& pool = pools_[lease.layer_];
  for (std::uint32_t index = 0; index < lease.count_; ++index) {
    const std::uint32_t slot = lease.pins_[index].slot;
    Slot& state = pool.slot_state[slot];
    if (state.pins > 0) --state.pins;
    list_push_back(pool, slot);
  }
}

}  // namespace ame::residency
