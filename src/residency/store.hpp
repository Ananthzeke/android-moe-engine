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
#include "reader_pool.hpp"
#include "result.hpp"

namespace ame::residency {

inline constexpr std::uint32_t kMaxLeaseExperts = 128;

struct ExpertEntry {
  model::ExpertKey key;
  model::FileOffset offset;
  model::ByteCount length;
};

struct StoreOptions {
  std::uint32_t slot_count{0};
  std::uint32_t alignment{64};
  // Optional caller-owned arena. When non-empty it must be aligned and at least
  // slot_count * slot_bytes. The store never frees borrowed memory.
  std::span<std::byte> external_arena{};
  // Reader threads for the misses of one acquire (1 to 4). One reads on the calling thread.
  unsigned readers{1};
  std::vector<int> reader_cpus{};
};

struct StoreStats {
  std::uint64_t hits{0};
  std::uint64_t misses{0};
  std::uint64_t evictions{0};
  std::uint64_t read_calls{0};
  std::uint64_t read_bytes{0};
  std::uint64_t pread_calls{0};
  std::uint64_t pread_bytes{0};
  double read_seconds{0.0};
  std::uint64_t pinned_slots{0};
  std::uint64_t resident_slots{0};
};

enum class StoreError {
  InvalidOptions,
  NonUniformPayload,
  DuplicateKey,
  EntryBeyondFile,
  Misaligned,
  Overflow,
  ArenaAllocationFailed,
  UnknownKey,
  ResourceLimit,
  Cancelled,
  ReadFailed,
  TooManyKeys,
};

class Store;

// Move-only pin set. It must be destroyed before its Store. It exposes only the
// experts acquired by this lease; a caller cannot retain a view after release.
class Lease {
 public:
  Lease() = default;
  Lease(Lease&& other) noexcept;
  Lease& operator=(Lease&& other) noexcept;
  Lease(const Lease&) = delete;
  Lease& operator=(const Lease&) = delete;
  ~Lease();

  [[nodiscard]] bool valid() const noexcept { return store_ != nullptr && count_ != 0; }
  [[nodiscard]] std::uint32_t size() const noexcept { return count_; }
  [[nodiscard]] std::span<const std::byte> payload(const model::ExpertKey& key) const;
  [[nodiscard]] std::optional<std::uint32_t> slot_index(const model::ExpertKey& key) const;

 private:
  friend class Store;
  struct Pin {
    std::uint32_t slot{0};
    std::uint32_t entry{0};
  };
  void release() noexcept;

  std::array<Pin, kMaxLeaseExperts> pins_{};
  std::uint32_t count_{0};
  Store* store_{nullptr};
};

// A fixed aligned arena of equally sized slots plus a byte-based LRU. Slots are
// only evicted when unpinned. A unit whose distinct expert count exceeds the slot
// count is refused instead of being scheduled or silently widened.
class Store {
 public:
  static Result<std::unique_ptr<Store>, StoreError> create(FileSource source,
                                                           std::span<const ExpertEntry> entries,
                                                           StoreOptions options);
  ~Store();
  Store(const Store&) = delete;
  Store& operator=(const Store&) = delete;
  Store(Store&&) = delete;
  Store& operator=(Store&&) = delete;

  [[nodiscard]] Result<Lease, StoreError> acquire(std::span<const model::ExpertKey> keys,
                                                  const Cancellation& stop);
  [[nodiscard]] std::span<std::byte> arena() noexcept { return arena_; }
  [[nodiscard]] std::span<const std::byte> arena() const noexcept { return arena_; }
  [[nodiscard]] std::uint64_t slot_bytes() const noexcept { return slot_bytes_; }
  [[nodiscard]] std::uint32_t slot_count() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
  }
  [[nodiscard]] StoreStats stats() const;
  // Re-read the injected-fault environment. Only when no acquire is in flight.
  void reload_fault_injection() const noexcept { source_.reload_fault_injection(); }

 private:
  friend class Lease;
  struct EntryState {
    model::ExpertKey key;
    model::FileOffset offset;
    model::ByteCount length;
    std::int32_t slot{-1};
  };
  struct SlotState {
    std::int32_t entry{-1};
    std::uint32_t pins{0};
    std::uint64_t tick{0};
    bool resident{false};
  };

  Store(FileSource source, std::vector<EntryState> entries, StoreOptions options,
        std::uint64_t slot_bytes);

  [[nodiscard]] std::optional<std::uint32_t> entry_index(const model::ExpertKey& key) const;
  [[nodiscard]] std::span<const std::byte> payload(std::uint32_t entry, std::uint32_t slot) const;
  void release(const Lease& lease) noexcept;

  FileSource source_;
  std::unique_ptr<ReaderPool> readers_;  // after source_: joined before the source closes
  std::vector<EntryState> entries_;
  std::vector<SlotState> slots_;
  std::vector<std::uint8_t> slot_taken_;
  std::span<std::byte> arena_;
  std::unique_ptr<std::byte, void (*)(void*)> arena_owner_{nullptr, nullptr};
  std::uint64_t slot_bytes_{0};
  std::uint64_t clock_{0};
  StoreStats stats_{};
};

}  // namespace ame::residency