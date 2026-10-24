#include "layered_store.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "cancellation.hpp"
#include "file_source.hpp"

namespace {

using ame::residency::EvictionPolicy;
using ame::residency::FileSource;
using ame::residency::LayeredOptions;
using ame::residency::LayeredStore;
using ame::residency::LayerExpert;
using ame::residency::LayerSpec;
using ame::residency::SliceExtent;
using ame::residency::StoreError;

int failures = 0;
unsigned g_readers = 1;  // reader threads the store under test uses
void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Layered store test failure: " << message << '\n';
    ++failures;
  }
}

// Three roles per expert with different sizes, stored role-major like a GGUF: all gate
// matrices, then all up, then all down. A slot holds one expert's three slices together.
struct Fixture {
  std::filesystem::path directory;
  std::filesystem::path file;
  std::uint32_t experts{0};
  std::array<std::uint64_t, 3> role_bytes{};
  std::array<std::uint64_t, 3> role_base{};
  std::uint64_t slot_bytes{0};
  std::vector<std::byte> contents;

  [[nodiscard]] std::byte expected(std::uint32_t expert, std::uint32_t role,
                                   std::uint64_t index) const {
    return static_cast<std::byte>((expert * 37U + role * 11U + index * 3U) & 0xFFU);
  }
};

Fixture make_fixture(const std::string& name, std::uint32_t experts,
                     std::array<std::uint64_t, 3> role_bytes) {
  Fixture fixture;
  fixture.directory =
      std::filesystem::temp_directory_path() /
      ("ame_layered_" + name + "_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(fixture.directory);
  std::filesystem::create_directories(fixture.directory);
  fixture.file = fixture.directory / "model.bin";
  fixture.experts = experts;
  fixture.role_bytes = role_bytes;
  std::uint64_t cursor = 0;
  for (std::uint32_t role = 0; role < 3; ++role) {
    fixture.role_base[role] = cursor;
    cursor += role_bytes[role] * experts;
  }
  fixture.contents.assign(static_cast<std::size_t>(cursor), std::byte{0});
  for (std::uint32_t role = 0; role < 3; ++role) {
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      for (std::uint64_t index = 0; index < role_bytes[role]; ++index) {
        fixture.contents[static_cast<std::size_t>(fixture.role_base[role] +
                                                  expert * role_bytes[role] + index)] =
            fixture.expected(expert, role, index);
      }
    }
  }
  std::ofstream stream(fixture.file, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(fixture.contents.data()),
               static_cast<std::streamsize>(fixture.contents.size()));
  for (const auto bytes : role_bytes) fixture.slot_bytes += bytes;
  return fixture;
}

LayerSpec make_spec(const Fixture& fixture, std::uint32_t slots) {
  LayerSpec spec;
  spec.slots = slots;
  spec.slot_bytes = fixture.slot_bytes;
  for (std::uint32_t expert = 0; expert < fixture.experts; ++expert) {
    LayerExpert entry;
    std::uint64_t slot_offset = 0;
    for (std::uint32_t role = 0; role < 3; ++role) {
      entry.slices[role] = SliceExtent{
          ame::model::FileOffset{fixture.role_base[role] + expert * fixture.role_bytes[role]},
          ame::model::ByteCount{fixture.role_bytes[role]}, slot_offset};
      slot_offset += fixture.role_bytes[role];
    }
    entry.slice_count = 3;
    spec.experts.push_back(entry);
  }
  return spec;
}

std::unique_ptr<LayeredStore> open_store(const Fixture& fixture, std::uint32_t layers,
                                         std::uint32_t slots, LayeredOptions options = {}) {
  auto source = FileSource::open(fixture.file.string());
  if (!source.ok()) return nullptr;
  options.readers = g_readers;
  std::vector<LayerSpec> specs(layers, make_spec(fixture, slots));
  auto store = LayeredStore::create(std::move(source).value(), std::move(specs), options);
  return store.ok() ? std::move(store).value() : nullptr;
}

std::vector<std::uint32_t> ids(std::initializer_list<std::uint32_t> list) { return list; }

void check_slot_bytes_and_views() {
  const auto fixture = make_fixture("bytes", 6, {256, 192, 128});
  auto store = open_store(fixture, 2, 4);
  expect(store != nullptr, "store opens");
  if (store == nullptr) return;
  expect(store->layer_count() == 2 && store->slots(1) == 4 && store->slot_bytes(0) == 576,
         "pool geometry");
  expect(store->arena_bytes() == 2U * 4U * 576U, "arena is the sum of the pools");
  expect(store->metadata_bytes() > 0, "metadata is accounted");
  ame::Cancellation stop;
  // The arena is uninitialized memory; keep a copy of layer 0 to prove it is not touched.
  const auto other_before =
      std::vector<std::byte>(store->pool_arena(0).begin(), store->pool_arena(0).end());
  const auto wanted = ids({3, 1, 5, 3});
  auto lease = store->acquire(1, wanted, stop);
  expect(lease.ok(), "acquire succeeds");
  if (!lease.ok()) return;
  expect(lease.value().size() == 3, "duplicates collapse");
  const auto pool = store->pool_arena(1);
  expect(pool.size() == 4U * 576U, "pool view size");
  for (const std::uint32_t expert : {1U, 3U, 5U}) {
    const auto slot = lease.value().slot_index(expert);
    expect(slot.has_value(), "every acquired expert has a slot");
    if (!slot) continue;
    std::uint64_t slot_offset = 0;
    for (std::uint32_t role = 0; role < 3; ++role) {
      bool equal = true;
      for (std::uint64_t index = 0; index < fixture.role_bytes[role]; ++index) {
        equal = equal && pool[static_cast<std::size_t>(*slot * 576U + slot_offset + index)] ==
                             fixture.expected(expert, role, index);
      }
      expect(equal, "a role's slice sits at its offset inside the slot, equal to the file");
      slot_offset += fixture.role_bytes[role];
    }
  }
  const auto stats = store->stats();
  expect(stats.misses == 3 && stats.hits == 0 && stats.slice_reads == 9,
         "three experts read as nine slices");
  expect(stats.read_bytes == 3U * 576U, "application bytes");
  // The other pool was not touched.
  const auto other_after = store->pool_arena(0);
  expect(
      std::equal(other_before.begin(), other_before.end(), other_after.begin(), other_after.end()),
      "other layers' pools are not written");
  std::filesystem::remove_all(fixture.directory);
}

void check_lru_pins_and_refusal() {
  const auto fixture = make_fixture("lru", 8, {64, 64, 64});
  auto store = open_store(fixture, 1, 3);
  if (store == nullptr) return expect(false, "lru store opens");
  ame::Cancellation stop;
  {
    auto first = store->acquire(0, ids({0, 1, 2}), stop);
    expect(first.ok(), "fills the pool");
    // Everything is pinned: a miss cannot evict, and the pinned bytes stay intact.
    auto refused = store->acquire(0, ids({3}), stop);
    expect(!refused.ok() && refused.error() == StoreError::ResourceLimit,
           "a pinned slot is never evicted");
    expect(first.value().slot_index(0).has_value(), "pins survive the refusal");
    auto too_many = store->acquire(0, ids({4, 5, 6, 7}), stop);
    expect(!too_many.ok() && too_many.error() == StoreError::ResourceLimit,
           "more misses than slots is refused");
  }
  // Released: 0,1,2 in recency order. A hit on 0 keeps it; the miss evicts the oldest, 1.
  auto second = store->acquire(0, ids({0, 3}), stop);
  expect(second.ok(), "second acquire");
  const auto stats = store->stats();
  expect(stats.hits == 1 && stats.evictions == 1, "hit pinned first, one eviction");
  expect(!second.value().slot_index(1).has_value(), "expert 1 is not in the lease");
  auto third = store->acquire(0, ids({1}), stop);
  expect(third.ok(), "oldest unpinned expert comes back as a miss");
  second = ame::residency::LayerLease{};
  third = ame::residency::LayerLease{};
  auto again = store->acquire(0, ids({3}), stop);
  expect(again.ok() && store->stats().hits == 2, "expert 3 stayed resident");
  std::filesystem::remove_all(fixture.directory);
}

void check_lfu_keeps_the_frequent_expert() {
  const auto fixture = make_fixture("lfu", 8, {64, 64, 64});
  for (const auto policy : {EvictionPolicy::Lru, EvictionPolicy::Lfu}) {
    LayeredOptions options;
    options.policy = policy;
    options.lfu_halve_every = 0;
    auto store = open_store(fixture, 1, 2, options);
    if (store == nullptr) return expect(false, "policy store opens");
    ame::Cancellation stop;
    for (int round = 0; round < 3; ++round) {
      auto lease = store->acquire(0, ids({0}), stop);  // expert 0 is used often
    }
    (void)store->acquire(0, ids({1}), stop);
    (void)store->acquire(0, ids({2}), stop);  // evicts 0 under LRU, 1 under LFU
    auto probe = store->acquire(0, ids({0}), stop);
    const bool kept = probe.ok() && store->stats().hits == 3;  // 0's 2 reuses + this one
    expect(policy == EvictionPolicy::Lfu ? kept : !kept,
           policy == EvictionPolicy::Lfu ? "LFU keeps the frequently used expert"
                                         : "LRU evicts the oldest even when frequent");
  }
  std::filesystem::remove_all(fixture.directory);
}

void check_cancellation_and_read_failure_rollback() {
  // 1 MiB slices so the injected mid-batch cancel (after the first 1 MiB chunk) lands
  // between two reads of one acquisition.
  const auto fixture = make_fixture("fault", 4, {1U << 20, 1U << 20, 1U << 20});
  auto store = open_store(fixture, 1, 3);
  if (store == nullptr) return expect(false, "fault store opens");
  ame::Cancellation stop;
  {
    auto warm = store->acquire(0, ids({0}), stop);  // expert 0 resident
    expect(warm.ok(), "warm-up");
  }
  const auto before = store->stats();

  // Cancellation before any read: nothing changes.
  ame::Cancellation cancelled;
  cancelled.request();
  auto none = store->acquire(0, ids({0, 1}), cancelled);
  expect(!none.ok() && none.error() == StoreError::Cancelled, "cancelled before reading");

  // Cancellation between reads of the batch: partial loads are dropped, hits unpinned.
  expect(::setenv("AME_IO_CANCEL_AFTER_CHUNKS", "1", 1) == 0, "set cancel injection");
  store->reload_fault_injection();
  ame::Cancellation mid;
  auto partial = store->acquire(0, ids({0, 1, 2}), mid);
  expect(!partial.ok() && partial.error() == StoreError::Cancelled, "cancelled mid-batch");
  (void)::unsetenv("AME_IO_CANCEL_AFTER_CHUNKS");
  store->reload_fault_injection();

  // Read failure: ReadFailed, and the pool is usable and consistent afterwards.
  const auto bytes_before = store->stats().pread_bytes;
  expect(::setenv("AME_IO_FAIL_AFTER_BYTES", std::to_string(bytes_before + 100).c_str(), 1) == 0,
         "set fault limit");
  store->reload_fault_injection();
  ame::Cancellation clear;
  auto failed = store->acquire(0, ids({1, 2}), clear);
  expect(!failed.ok() && failed.error() == StoreError::ReadFailed, "injected read failure");
  (void)::unsetenv("AME_IO_FAIL_AFTER_BYTES");
  store->reload_fault_injection();

  auto recovered = store->acquire(0, ids({0, 1, 2}), clear);
  expect(recovered.ok(), "the store works after the failures");
  if (recovered.ok()) {
    const auto pool = store->pool_arena(0);
    bool equal = true;
    for (const std::uint32_t expert : {0U, 1U, 2U}) {
      const auto slot = *recovered.value().slot_index(expert);
      for (std::uint64_t index = 0; index < 4096; ++index) {
        equal = equal && pool[static_cast<std::size_t>(slot * fixture.slot_bytes + index)] ==
                             fixture.expected(expert, 0, index);
      }
    }
    expect(equal, "reloaded slices equal the file after rollback");
    const auto stats = store->stats();
    expect(stats.hits >= before.hits + 1, "the resident expert was still a hit");
  }
  std::filesystem::remove_all(fixture.directory);
}

void check_rejected_layouts() {
  const auto fixture = make_fixture("layout", 4, {64, 64, 64});
  const auto try_spec = [&](auto edit) {
    auto source = FileSource::open(fixture.file.string());
    std::vector<LayerSpec> specs{make_spec(fixture, 2)};
    edit(specs[0]);
    auto store = LayeredStore::create(std::move(source).value(), std::move(specs), {});
    return store.ok() ? std::optional<StoreError>{} : std::optional<StoreError>{store.error()};
  };
  expect(!try_spec([](LayerSpec&) {}).has_value(), "the baseline layout is accepted");
  expect(try_spec([](LayerSpec& spec) { spec.slots = 5; }) == StoreError::InvalidOptions,
         "more slots than experts is refused");
  expect(try_spec([](LayerSpec& spec) { spec.slots = 0; }) == StoreError::InvalidOptions,
         "an empty pool is refused");
  expect(try_spec([](LayerSpec& spec) { spec.slot_bytes = 100; }) == StoreError::Misaligned,
         "a slot size that breaks the arena alignment is refused");
  expect(try_spec([](LayerSpec& spec) {
           spec.experts[0].slices[1].slot_offset = 32;  // overlaps the first slice
         }) == StoreError::Overflow,
         "overlapping slices inside a slot are refused");
  expect(try_spec([](LayerSpec& spec) {
           spec.experts[0].slices[2].offset = ame::model::FileOffset{1U << 30};
         }) == StoreError::EntryBeyondFile,
         "a slice beyond the file is refused");
  expect(try_spec([](LayerSpec& spec) { spec.experts[1].slice_count = 0; }) ==
             StoreError::InvalidOptions,
         "an expert without slices is refused");
  std::filesystem::remove_all(fixture.directory);
}

void check_dontneed_and_random_flags() {
  const auto fixture = make_fixture("advise", 4, {64, 64, 64});
  LayeredOptions options;
  options.fadvise_dontneed = true;
  options.fadvise_random = true;
  auto store = open_store(fixture, 1, 2, options);
  if (store == nullptr) return expect(false, "advise store opens");
  ame::Cancellation stop;
  auto lease = store->acquire(0, ids({2, 3}), stop);
  expect(lease.ok(), "hints do not change what is read");
  if (lease.ok()) {
    const auto slot = *lease.value().slot_index(3);
    expect(store->pool_arena(0)[static_cast<std::size_t>(slot * fixture.slot_bytes)] ==
               fixture.expected(3, 0, 0),
           "bytes are intact with page-cache hints on");
  }
  std::filesystem::remove_all(fixture.directory);
}

}  // namespace

int main() {
  for (const unsigned readers : {1U, 2U, 4U}) {
    g_readers = readers;
    check_slot_bytes_and_views();
    check_lru_pins_and_refusal();
    check_lfu_keeps_the_frequent_expert();
    check_cancellation_and_read_failure_rollback();
    check_rejected_layouts();
    check_dontneed_and_random_flags();
  }
  return failures == 0 ? 0 : 1;
}
