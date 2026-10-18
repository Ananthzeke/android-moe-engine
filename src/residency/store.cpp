#include "store.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <utility>

namespace ame::residency {
namespace {

struct FreeDeleter {
  void operator()(void* pointer) const noexcept { std::free(pointer); }
};

bool power_of_two(std::uint32_t value) { return value != 0 && (value & (value - 1)) == 0; }

}  // namespace

Lease::Lease(Lease&& other) noexcept
    : pins_(other.pins_), count_(other.count_), store_(other.store_) {
  other.count_ = 0;
  other.store_ = nullptr;
}

Lease& Lease::operator=(Lease&& other) noexcept {
  if (this != &other) {
    release();
    pins_ = other.pins_;
    count_ = other.count_;
    store_ = other.store_;
    other.count_ = 0;
    other.store_ = nullptr;
  }
  return *this;
}

Lease::~Lease() { release(); }

void Lease::release() noexcept {
  if (store_ != nullptr && count_ != 0) {
    store_->release(*this);
  }
  count_ = 0;
  store_ = nullptr;
}

std::span<const std::byte> Lease::payload(const model::ExpertKey& key) const {
  if (store_ == nullptr) {
    return {};
  }
  for (std::uint32_t index = 0; index < count_; ++index) {
    const auto& entry = store_->entries_[pins_[index].entry];
    if (entry.key == key) {
      return store_->payload(pins_[index].entry, pins_[index].slot);
    }
  }
  return {};
}

std::optional<std::uint32_t> Lease::slot_index(const model::ExpertKey& key) const {
  if (store_ == nullptr) {
    return std::nullopt;
  }
  for (std::uint32_t index = 0; index < count_; ++index) {
    if (store_->entries_[pins_[index].entry].key == key) {
      return pins_[index].slot;
    }
  }
  return std::nullopt;
}

Store::Store(FileSource source, std::vector<EntryState> entries, StoreOptions options,
             std::uint64_t slot_bytes)
    : source_(std::move(source)),
      entries_(std::move(entries)),
      slots_(options.slot_count),
      slot_taken_(options.slot_count, 0),
      slot_bytes_(slot_bytes) {
  void* raw = nullptr;
  const std::size_t bytes = static_cast<std::size_t>(slot_bytes * options.slot_count);
  if (::posix_memalign(&raw, options.alignment, bytes) == 0) {
    arena_owner_ = std::unique_ptr<std::byte, void (*)(void*)>(
        static_cast<std::byte*>(raw), [](void* pointer) { std::free(pointer); });
    arena_ = std::span<std::byte>(arena_owner_.get(), bytes);
  }
}

Store::~Store() = default;

Result<std::unique_ptr<Store>, StoreError> Store::create(FileSource source,
                                                         std::span<const ExpertEntry> entries,
                                                         StoreOptions options) {
  if (options.slot_count == 0 || entries.empty()) {
    return StoreError::InvalidOptions;
  }
  if (!power_of_two(options.alignment) || options.alignment < model::kTensorAlignment ||
      options.alignment > 4096) {
    return StoreError::InvalidOptions;
  }
  if (entries.size() > 65536) {
    return StoreError::InvalidOptions;
  }
  const std::uint64_t payload_bytes = entries.front().length.value;
  if (payload_bytes == 0 || payload_bytes % model::kTensorAlignment != 0) {
    return StoreError::Misaligned;
  }
  if (payload_bytes > std::numeric_limits<std::uint64_t>::max() / options.slot_count) {
    return StoreError::Overflow;
  }
  std::vector<EntryState> states;
  states.reserve(entries.size());
  for (const auto& entry : entries) {
    if (entry.length.value != payload_bytes) {
      return StoreError::NonUniformPayload;
    }
    if (entry.offset.value % model::kTensorAlignment != 0) {
      return StoreError::Misaligned;
    }
    if (entry.offset.value > source.size_bytes() ||
        entry.length.value > source.size_bytes() - entry.offset.value) {
      return StoreError::EntryBeyondFile;
    }
    states.push_back(EntryState{entry.key, entry.offset, entry.length, -1});
  }
  std::sort(states.begin(), states.end(),
            [](const EntryState& left, const EntryState& right) { return left.key < right.key; });
  for (std::size_t index = 1; index < states.size(); ++index) {
    if (states[index - 1].key == states[index].key) {
      return StoreError::DuplicateKey;
    }
  }
  auto store = std::unique_ptr<Store>(
      new Store(std::move(source), std::move(states), options, payload_bytes));
  if (store->arena_.empty()) {
    return StoreError::ArenaAllocationFailed;
  }
  return store;
}

std::optional<std::uint32_t> Store::entry_index(const model::ExpertKey& key) const {
  const auto found = std::lower_bound(
      entries_.begin(), entries_.end(), key,
      [](const EntryState& entry, const model::ExpertKey& value) { return entry.key < value; });
  if (found == entries_.end() || !(found->key == key)) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(std::distance(entries_.begin(), found));
}

std::span<const std::byte> Store::payload(std::uint32_t entry, std::uint32_t slot) const {
  const auto offset = static_cast<std::size_t>(slot) * static_cast<std::size_t>(slot_bytes_);
  const auto length = static_cast<std::size_t>(entries_[entry].length.value);
  return arena_.subspan(offset, length);
}

Result<Lease, StoreError> Store::acquire(std::span<const model::ExpertKey> keys,
                                         const Cancellation& stop) {
  if (keys.empty()) {
    return Lease{};
  }
  if (keys.size() > kMaxLeaseExperts) {
    return StoreError::TooManyKeys;
  }
  std::array<std::uint32_t, kMaxLeaseExperts> unique{};
  std::uint32_t unique_count = 0;
  for (const auto& key : keys) {
    const auto index = entry_index(key);
    if (!index.has_value()) {
      return StoreError::UnknownKey;
    }
    bool seen = false;
    for (std::uint32_t existing = 0; existing < unique_count; ++existing) {
      if (unique[existing] == *index) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      unique[unique_count] = *index;
      ++unique_count;
    }
  }

  std::fill(slot_taken_.begin(), slot_taken_.end(), static_cast<std::uint8_t>(0));
  std::array<std::uint32_t, kMaxLeaseExperts> hit_slots{};
  std::uint32_t hit_count = 0;
  std::array<std::uint32_t, kMaxLeaseExperts> miss_entries{};
  std::uint32_t miss_count = 0;
  for (std::uint32_t index = 0; index < unique_count; ++index) {
    const auto entry_index_value = unique[index];
    auto& entry = entries_[entry_index_value];
    if (entry.slot >= 0) {
      auto& slot = slots_[static_cast<std::size_t>(entry.slot)];
      if (slot.resident && slot.entry == static_cast<std::int32_t>(entry_index_value)) {
        hit_slots[hit_count++] = static_cast<std::uint32_t>(entry.slot);
        slot_taken_[static_cast<std::size_t>(entry.slot)] = 1;
        continue;
      }
    }
    miss_entries[miss_count++] = entry_index_value;
  }
  if (miss_count > slot_count()) {
    return StoreError::ResourceLimit;
  }

  std::array<std::uint32_t, kMaxLeaseExperts> chosen{};
  std::uint32_t chosen_count = 0;
  for (std::uint32_t slot_index = 0; slot_index < slot_count() && chosen_count < miss_count;
       ++slot_index) {
    if (slots_[slot_index].entry < 0) {
      chosen[chosen_count++] = slot_index;
      slot_taken_[slot_index] = 1;
    }
  }
  while (chosen_count < miss_count) {
    std::int32_t best_slot = -1;
    std::uint64_t best_tick = std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t slot_index = 0; slot_index < slot_count(); ++slot_index) {
      const auto& slot = slots_[slot_index];
      if (slot_taken_[slot_index] != 0 || slot.pins != 0 || !slot.resident || slot.entry < 0) {
        continue;
      }
      if (slot.tick < best_tick) {
        best_tick = slot.tick;
        best_slot = static_cast<std::int32_t>(slot_index);
      }
    }
    if (best_slot < 0) {
      return StoreError::ResourceLimit;
    }
    slot_taken_[static_cast<std::size_t>(best_slot)] = 1;
    chosen[chosen_count++] = static_cast<std::uint32_t>(best_slot);
  }

  Lease lease;
  lease.store_ = this;
  for (std::uint32_t index = 0; index < hit_count; ++index) {
    auto& slot = slots_[hit_slots[index]];
    ++slot.pins;
    slot.tick = ++clock_;
    lease.pins_[lease.count_++] =
        Lease::Pin{hit_slots[index], static_cast<std::uint32_t>(slot.entry)};
    ++stats_.hits;
  }
  std::array<std::uint32_t, kMaxLeaseExperts> loading_slots{};
  std::uint32_t loading_count = 0;
  for (std::uint32_t index = 0; index < miss_count; ++index) {
    const std::uint32_t entry_index_value = miss_entries[index];
    auto& entry = entries_[entry_index_value];
    const std::uint32_t slot_index = chosen[index];
    auto& slot = slots_[slot_index];
    if (slot.entry >= 0) {
      entries_[static_cast<std::size_t>(slot.entry)].slot = -1;
      ++stats_.evictions;
    }
    slot.entry = static_cast<std::int32_t>(entry_index_value);
    slot.resident = false;
    slot.pins = 1;
    entry.slot = static_cast<std::int32_t>(slot_index);
    lease.pins_[lease.count_++] = Lease::Pin{slot_index, entry_index_value};
    loading_slots[loading_count++] = slot_index;
    ++stats_.misses;
  }

  const auto rollback = [&]() noexcept {
    for (std::uint32_t index = 0; index < lease.count_; ++index) {
      auto& slot = slots_[lease.pins_[index].slot];
      if (!slot.resident) {
        entries_[static_cast<std::size_t>(slot.entry)].slot = -1;
        slot.entry = -1;
        slot.pins = 0;
        slot.tick = 0;
      } else if (slot.pins > 0) {
        --slot.pins;
      }
    }
    lease.count_ = 0;
    lease.store_ = nullptr;
  };

  for (std::uint32_t index = 0; index < loading_count; ++index) {
    if (stop.requested()) {
      rollback();
      return StoreError::Cancelled;
    }
    const std::uint32_t slot_index = loading_slots[index];
    auto& entry = entries_[static_cast<std::size_t>(slots_[slot_index].entry)];
    auto destination =
        arena_.subspan(static_cast<std::size_t>(slot_index) * static_cast<std::size_t>(slot_bytes_),
                       static_cast<std::size_t>(entry.length.value));
    const auto read_begin = std::chrono::steady_clock::now();
    const auto read = source_.read_exact(entry.offset.value, destination, stop);
    stats_.read_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - read_begin).count();
    if (!read.ok()) {
      rollback();
      return read.error() == FileError::Cancelled ? StoreError::Cancelled : StoreError::ReadFailed;
    }
    ++stats_.read_calls;
    stats_.read_bytes += entry.length.value;
    auto& slot = slots_[slot_index];
    slot.resident = true;
    slot.tick = ++clock_;
  }
  return lease;
}

void Store::release(const Lease& lease) noexcept {
  for (std::uint32_t index = 0; index < lease.count_; ++index) {
    auto& slot = slots_[lease.pins_[index].slot];
    if (slot.pins > 0) {
      --slot.pins;
    }
    slot.tick = ++clock_;
  }
}

StoreStats Store::stats() const {
  StoreStats result = stats_;
  for (const auto& slot : slots_) {
    if (slot.pins != 0) {
      ++result.pinned_slots;
    }
    if (slot.resident) {
      ++result.resident_slots;
    }
  }
  return result;
}

}  // namespace ame::residency