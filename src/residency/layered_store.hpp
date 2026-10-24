#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "cancellation.hpp"
#include "expert_layout.hpp"
#include "file_source.hpp"
#include "reader_pool.hpp"
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
  // Reader threads for the misses of one acquire (1 to 4). One reads on the calling thread.
  unsigned readers{1};
  std::vector<int> reader_cpus{};
  // Keep reader threads even for one reader, which begin_acquire needs.
  bool background_readers{false};
};

struct LayeredStats {
  std::uint64_t hits{0};
  std::uint64_t misses{0};
  std::uint64_t evictions{0};
  std::uint64_t slice_reads{0};  // expert slices read, not syscalls
  std::uint64_t read_bytes{0};   // application bytes for those slices
  std::uint64_t pread_calls{0};
  std::uint64_t pread_bytes{0};
  double read_seconds{0.0};       // wall time from starting a batch of reads to its end
  double read_wait_seconds{0.0};  // time the caller was blocked waiting for reads
  std::uint64_t async_acquires{0};
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

  // Asynchronous acquisition for overlapping reads with compute. begin_acquire reserves and pins
  // exactly as acquire does, starts the reads of the missing experts' slices grouped by slice
  // index (role), and returns at once. The lease answers slot_index immediately, but a slot's
  // bytes for role r are valid only after wait_role(layer, r) succeeds. There is at most one
  // pending acquisition per layer, and the store must have been created with background readers.
  // finish_pending waits for every remaining read and publishes the loaded slots. Releasing the
  // lease, or beginning another acquisition of the layer, while reads are pending drains them
  // and drops the slots that were loading. After an error, the failed slots are dropped and the
  // lease must still be released.
  [[nodiscard]] Result<LayerLease, StoreError> begin_acquire(std::uint32_t layer,
                                                             std::span<const std::uint32_t> experts,
                                                             const Cancellation& stop);
  [[nodiscard]] Result<void, StoreError> wait_role(std::uint32_t layer, std::uint32_t role);
  [[nodiscard]] Result<void, StoreError> finish_pending(std::uint32_t layer);
  [[nodiscard]] bool pending(std::uint32_t layer) const noexcept {
    return pools_[layer].pending.active;
  }
  // Drain and drop every pending acquisition. Call before the arena or the source goes away.
  void abandon_all_pending() noexcept;

  [[nodiscard]] std::uint32_t layer_count() const noexcept {
    return static_cast<std::uint32_t>(pools_.size());
  }
  [[nodiscard]] std::uint32_t slots(std::uint32_t layer) const { return pools_[layer].slots; }
  [[nodiscard]] std::size_t expert_count(std::uint32_t layer) const {
    return pools_[layer].experts.size();
  }
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
  struct Loaded {
    std::uint32_t slot{0};
    std::uint32_t expert{0};
  };
  // The slots of one acquisition: hits that were pinned, and misses that are being read.
  struct Reservation {
    std::array<Loaded, kMaxLeaseExperts> loaded{};
    std::uint32_t loaded_count{0};
    std::array<std::uint32_t, kMaxLeaseExperts> hit_slots{};
    std::uint32_t hit_count{0};
  };
  struct Pending {
    bool active{false};
    ReaderPool::BatchHandle batch;
    Reservation reservation;
    std::chrono::steady_clock::time_point started;
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
    Pending pending;
  };

  LayeredStore(FileSource source, std::vector<Pool> pools, LayeredOptions options);

  void list_remove(Pool& pool, std::uint32_t slot) noexcept;
  void list_push_back(Pool& pool, std::uint32_t slot) noexcept;
  void release(const LayerLease& lease) noexcept;
  // Pin the slots for an acquisition: hits first, then misses (evicting victims). Nothing is read.
  [[nodiscard]] Result<LayerLease, StoreError> reserve(Pool& pool, std::uint32_t layer,
                                                       std::span<const std::uint32_t> experts,
                                                       Reservation& reservation);
  void build_jobs(Pool& pool, const Reservation& reservation);
  // Empty the slots that were loading. unpin_hits is true when no lease was handed out.
  void rollback(Pool& pool, const Reservation& reservation, bool unpin_hits) noexcept;
  void publish(Pool& pool, const Reservation& reservation);
  void abandon_pending(Pool& pool) noexcept;
  std::int32_t choose_victim(Pool& pool, const std::vector<std::uint8_t>& taken) const;

  mutable FileSource source_;
  std::unique_ptr<ReaderPool> readers_;  // after source_: joined before the source closes
  std::vector<ReadJob> read_jobs_;       // scratch: one per slice of one acquire
  std::vector<Pool> pools_;
  LayeredOptions options_;
  std::span<std::byte> arena_;
  std::unique_ptr<std::byte, void (*)(void*)> arena_owner_{nullptr, nullptr};
  std::uint64_t metadata_bytes_{0};
  LayeredStats stats_{};
};

}  // namespace ame::residency
