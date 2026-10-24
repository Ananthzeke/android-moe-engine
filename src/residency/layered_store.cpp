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
  readers_ = std::make_unique<ReaderPool>(source_, options.readers, options.reader_cpus,
                                          options.background_readers);
  read_jobs_.reserve(kMaxLeaseExperts * kMaxSlicesPerSlot);
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
  metadata_bytes_ +=
      pools_.size() * sizeof(Pool) + kMaxLeaseExperts * kMaxSlicesPerSlot * sizeof(ReadJob);
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

LayeredStore::~LayeredStore() { abandon_all_pending(); }

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

Result<LayerLease, StoreError> LayeredStore::reserve(Pool& pool, std::uint32_t layer,
                                                     std::span<const std::uint32_t> experts,
                                                     Reservation& reservation) {
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
  reservation = Reservation{};
  for (std::uint32_t index = 0; index < hit_count; ++index) {
    const auto slot = static_cast<std::uint32_t>(pool.slot_of[hit_experts[index]]);
    ++pool.slot_state[slot].pins;
    ++pool.slot_state[slot].frequency;
    lease.pins_[lease.count_++] = LayerLease::Pin{hit_experts[index], slot};
    reservation.hit_slots[reservation.hit_count++] = slot;
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
    reservation.loaded[reservation.loaded_count++] = Loaded{slot, miss_experts[index]};
    ++stats_.misses;
  }
  return lease;
}

// One read per slice of every missing expert, grouped by slice index so a caller can wait for
// one role at a time.
void LayeredStore::build_jobs(Pool& pool, const Reservation& reservation) {
  read_jobs_.clear();
  // Role-major: every expert's first slice, then every expert's second, and so on, so the first
  // role can be waited for before the others have been read.
  for (std::uint32_t part = 0; part < kMaxSlicesPerSlot; ++part) {
    for (std::uint32_t index = 0; index < reservation.loaded_count; ++index) {
      const Loaded& loaded = reservation.loaded[index];
      const LayerExpert& expert = pool.experts[loaded.expert];
      if (part >= expert.slice_count) continue;
      const SliceExtent& slice = expert.slices[part];
      std::byte* slot_base = arena_.data() + pool.base_offset + loaded.slot * pool.slot_bytes;
      ReadJob job;
      job.offset = slice.offset.value;
      job.destination = std::span<std::byte>(slot_base + slice.slot_offset,
                                             static_cast<std::size_t>(slice.length.value));
      job.drop_cache = options_.fadvise_dontneed;
      job.group = static_cast<std::uint8_t>(part);
      read_jobs_.push_back(job);
    }
  }
}

void LayeredStore::rollback(Pool& pool, const Reservation& reservation, bool unpin_hits) noexcept {
  for (std::uint32_t index = 0; index < reservation.loaded_count; ++index) {
    const Loaded& loaded = reservation.loaded[index];
    Slot& state = pool.slot_state[loaded.slot];
    if (state.expert != static_cast<std::int32_t>(loaded.expert)) continue;
    pool.slot_of[loaded.expert] = -1;
    list_remove(pool, loaded.slot);
    state = Slot{};
  }
  if (unpin_hits) {
    for (std::uint32_t index = 0; index < reservation.hit_count; ++index) {
      Slot& state = pool.slot_state[reservation.hit_slots[index]];
      if (state.pins > 0) --state.pins;
    }
  }
}

void LayeredStore::publish(Pool& pool, const Reservation& reservation) {
  for (std::uint32_t index = 0; index < reservation.loaded_count; ++index) {
    const Loaded& loaded = reservation.loaded[index];
    const LayerExpert& expert = pool.experts[loaded.expert];
    stats_.slice_reads += expert.slice_count;
    for (std::uint32_t part = 0; part < expert.slice_count; ++part) {
      stats_.read_bytes += expert.slices[part].length.value;
    }
    list_push_back(pool, loaded.slot);
  }
}

Result<LayerLease, StoreError> LayeredStore::acquire(std::uint32_t layer,
                                                     std::span<const std::uint32_t> experts,
                                                     const Cancellation& stop) {
  if (layer >= pools_.size()) {
    return StoreError::UnknownKey;
  }
  Pool& pool = pools_[layer];
  abandon_pending(pool);
  Reservation reservation;
  auto reserved = reserve(pool, layer, experts, reservation);
  if (!reserved.ok()) {
    return reserved;
  }
  LayerLease lease = std::move(reserved).value();
  // The lease only exists to hand pins back; on failure it is dropped without release.
  const auto fail = [&](StoreError error) {
    rollback(pool, reservation, true);
    lease.count_ = 0;
    lease.store_ = nullptr;
    return error;
  };
  if (stop.requested()) {
    return fail(StoreError::Cancelled);
  }
  // Every miss is reserved and pinned. read_all returns only after every read has ended, so a
  // failure or cancellation rolls back with nothing still writing into a slot.
  build_jobs(pool, reservation);
  const auto begin = std::chrono::steady_clock::now();
  const auto read = readers_->read_all(read_jobs_, stop);
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  stats_.read_seconds += elapsed;
  stats_.read_wait_seconds += elapsed;
  if (!read.ok()) {
    return fail(read.error() == FileError::Cancelled ? StoreError::Cancelled
                                                     : StoreError::ReadFailed);
  }
  publish(pool, reservation);
  return lease;
}

Result<LayerLease, StoreError> LayeredStore::begin_acquire(std::uint32_t layer,
                                                           std::span<const std::uint32_t> experts,
                                                           const Cancellation& stop) {
  if (layer >= pools_.size()) {
    return StoreError::UnknownKey;
  }
  if (!readers_->background()) {
    return StoreError::InvalidOptions;
  }
  Pool& pool = pools_[layer];
  abandon_pending(pool);
  Reservation reservation;
  auto reserved = reserve(pool, layer, experts, reservation);
  if (!reserved.ok()) {
    return reserved;
  }
  LayerLease lease = std::move(reserved).value();
  if (stop.requested()) {
    rollback(pool, reservation, true);
    lease.count_ = 0;
    lease.store_ = nullptr;
    return StoreError::Cancelled;
  }
  build_jobs(pool, reservation);
  pool.pending.reservation = reservation;
  pool.pending.started = std::chrono::steady_clock::now();
  pool.pending.batch = readers_->submit(read_jobs_, stop);
  pool.pending.active = true;
  ++stats_.async_acquires;
  return lease;
}

Result<void, StoreError> LayeredStore::wait_role(std::uint32_t layer, std::uint32_t role) {
  Pool& pool = pools_[layer];
  if (!pool.pending.active) {
    return {};
  }
  const auto begin = std::chrono::steady_clock::now();
  const auto waited = ReaderPool::wait_group(pool.pending.batch, role);
  stats_.read_wait_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  if (!waited.ok()) {
    // Drain the rest, drop the slots that were loading; the lease still unpins the hits.
    const FileError error = waited.error();
    abandon_pending(pool);
    return error == FileError::Cancelled ? StoreError::Cancelled : StoreError::ReadFailed;
  }
  return {};
}

Result<void, StoreError> LayeredStore::finish_pending(std::uint32_t layer) {
  Pool& pool = pools_[layer];
  if (!pool.pending.active) {
    return {};
  }
  const auto begin = std::chrono::steady_clock::now();
  const auto waited = ReaderPool::wait_all(pool.pending.batch);
  const auto end = std::chrono::steady_clock::now();
  stats_.read_wait_seconds += std::chrono::duration<double>(end - begin).count();
  stats_.read_seconds += std::chrono::duration<double>(end - pool.pending.started).count();
  if (!waited.ok()) {
    const FileError error = waited.error();
    rollback(pool, pool.pending.reservation, false);
    pool.pending = Pending{};
    return error == FileError::Cancelled ? StoreError::Cancelled : StoreError::ReadFailed;
  }
  publish(pool, pool.pending.reservation);
  pool.pending = Pending{};
  return {};
}

void LayeredStore::abandon_pending(Pool& pool) noexcept {
  if (!pool.pending.active) {
    return;
  }
  (void)ReaderPool::wait_all(pool.pending.batch);  // every read has ended once this returns
  stats_.read_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - pool.pending.started)
          .count();
  rollback(pool, pool.pending.reservation, false);
  pool.pending = Pending{};
}

void LayeredStore::abandon_all_pending() noexcept {
  for (auto& pool : pools_) {
    abandon_pending(pool);
  }
}

void LayeredStore::release(const LayerLease& lease) noexcept {
  Pool& pool = pools_[lease.layer_];
  abandon_pending(pool);
  for (std::uint32_t index = 0; index < lease.count_; ++index) {
    const std::uint32_t slot = lease.pins_[index].slot;
    Slot& state = pool.slot_state[slot];
    // A slot that was dropped while loading no longer holds this expert.
    if (state.expert != static_cast<std::int32_t>(lease.pins_[index].expert)) continue;
    if (state.pins > 0) --state.pins;
    list_push_back(pool, slot);
  }
}

}  // namespace ame::residency
