#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "cancellation.hpp"
#include "expert_layout.hpp"
#include "file_source.hpp"
#include "result.hpp"
#include "store.hpp"

namespace ame::residency {

enum class EvictionPolicy { Lru, Lfu };

inline constexpr std::size_t kMaxSlicesPerSlot = 4;

// One role's bytes (for example the gate matrices) of one expert, and where they go inside
// the expert's slot.
struct SliceExtent {
  model::FileOffset offset;
  model::ByteCount length;
  std::uint64_t slot_offset{0};
};

struct LayerExpert {
  std::array<SliceExtent, kMaxSlicesPerSlot> slices{};
  std::uint32_t slice_count{0};
};

// One layer's pool. `experts[i]` describes expert i, so ids index it directly. Every slot
// holds all of an expert's slices contiguously, `slot_bytes` apart.
struct LayerSpec {
  std::vector<LayerExpert> experts;
  std::uint64_t slot_bytes{0};
  std::uint32_t slots{0};
};

struct LayeredOptions {
  EvictionPolicy policy{EvictionPolicy::Lru};
  // LFU only: halve a pool's frequency counts every N successful acquisitions of that pool.
  std::uint32_t lfu_halve_every{32};
  // Ask the kernel to drop each extent's page-cache pages right after the copy. The arena
  // is then the only copy of an expert in memory.
  bool fadvise_dontneed{false};
  // Ask the kernel not to read ahead on the read descriptor.
  bool fadvise_random{false};
  std::uint32_t alignment{64};
  std::span<std::byte> external_arena{};
};

struct LayeredStats {
  std::uint64_t hits{0};
  std::uint64_t misses{0};
  std::uint64_t evictions{0};
  std::uint64_t slice_reads{0};  // expert slices read, not syscalls
  std::uint64_t read_bytes{0};   // application bytes for those slices
  std::uint64_t pread_calls{0};
  std::uint64_t pread_bytes{0};
  double read_seconds{0.0};
};

class LayeredStore;

// Move-only pin set for one layer. It must be destroyed before its store.
class LayerLease {
 public:
  LayerLease() = default;
  LayerLease(LayerLease&& other) noexcept;
  LayerLease& operator=(LayerLease&& other) noexcept;
  LayerLease(const LayerLease&) = delete;
  LayerLease& operator=(const LayerLease&) = delete;
  ~LayerLease();

  [[nodiscard]] bool valid() const noexcept { return store_ != nullptr; }
  [[nodiscard]] std::uint32_t size() const noexcept { return count_; }
  [[nodiscard]] std::uint32_t layer() const noexcept { return layer_; }
  // Slot index inside the layer's pool, or nullopt when the expert is not in the lease.
  [[nodiscard]] std::optional<std::uint32_t> slot_index(std::uint32_t expert) const;

 private:
  friend class LayeredStore;
  void release() noexcept;

  struct Pin {
    std::uint32_t expert{0};
    std::uint32_t slot{0};
  };
  std::array<Pin, kMaxLeaseExperts> pins_{};
  std::uint32_t count_{0};
  std::uint32_t layer_{0};
  LayeredStore* store_{nullptr};
};

// A fixed arena divided into one pool per layer. A slot holds every role's slice of one
// expert, so one routing decision is one reservation and one read batch. Eviction inside a
// pool is least recently used or least frequently used with periodic halving. A slot is
// never evicted while a lease pins it. A request whose misses exceed the pool's evictable
// slots is refused with ResourceLimit.
class LayeredStore {
 public:
  static Result<std::unique_ptr<LayeredStore>, StoreError> create(FileSource source,
                                                                  std::vector<LayerSpec> layers,
                                                                  LayeredOptions options);
  ~LayeredStore();
  LayeredStore(const LayeredStore&) = delete;
  LayeredStore& operator=(const LayeredStore&) = delete;

  // `layer` is the position in the vector given to create(). Experts are expert ids.
  [[nodiscard]] Result<LayerLease, StoreError> acquire(std::uint32_t layer,
                                                       std::span<const std::uint32_t> experts,
                                                       const Cancellation& stop);

  [[nodiscard]] std::uint32_t layer_count() const noexcept {
    return static_cast<std::uint32_t>(pools_.size());
  }
  [[nodiscard]] std::uint32_t slots(std::uint32_t layer) const { return pools_[layer].slots; }
  [[nodiscard]] std::uint64_t slot_bytes(std::uint32_t layer) const {
    return pools_[layer].slot_bytes;
  }
  // The bytes of the layer's pool; slot s starts at s * slot_bytes(layer).
  [[nodiscard]] std::span<std::byte> pool_arena(std::uint32_t layer) noexcept;
  [[nodiscard]] std::uint64_t arena_bytes() const noexcept { return arena_.size(); }
  // Bookkeeping the store allocates besides the arena: slot records, expert tables, maps.
  [[nodiscard]] std::uint64_t metadata_bytes() const noexcept { return metadata_bytes_; }
  [[nodiscard]] LayeredStats stats() const noexcept;
  // Slots currently holding an expert, across all pools.
  [[nodiscard]] std::uint32_t resident_slots() const noexcept;
  void reload_fault_injection() const noexcept { source_.reload_fault_injection(); }

 private:
  friend class LayerLease;
  struct Slot {
    std::int32_t expert{-1};
    std::uint32_t pins{0};
    std::uint32_t frequency{0};
    std::int32_t prev{-1};  // recency list, oldest first; -1 when not listed
    std::int32_t next{-1};
    bool listed{false};
  };
  struct Pool {
    std::vector<LayerExpert> experts;
    std::vector<std::int32_t> slot_of;  // expert -> slot, -1 when absent
    std::vector<Slot> slot_state;
    std::vector<std::uint8_t> taken;  // scratch for one acquire, kept to avoid allocating per call
    std::uint64_t slot_bytes{0};
    std::uint32_t slots{0};
    std::uint64_t base_offset{0};
    std::int32_t head{-1};  // least recently used
    std::int32_t tail{-1};  // most recently used
    std::uint64_t acquires{0};
  };

  LayeredStore(FileSource source, std::vector<Pool> pools, LayeredOptions options);

  void list_remove(Pool& pool, std::uint32_t slot) noexcept;
  void list_push_back(Pool& pool, std::uint32_t slot) noexcept;
  void release(const LayerLease& lease) noexcept;
  std::int32_t choose_victim(Pool& pool, const std::vector<std::uint8_t>& taken) const;

  mutable FileSource source_;
  std::vector<Pool> pools_;
  LayeredOptions options_;
  std::span<std::byte> arena_;
  std::unique_ptr<std::byte, void (*)(void*)> arena_owner_{nullptr, nullptr};
  std::uint64_t metadata_bytes_{0};
  LayeredStats stats_{};
};

}  // namespace ame::residency
