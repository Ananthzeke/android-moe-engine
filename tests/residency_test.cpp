#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "cancellation.hpp"
#include "expert_layout.hpp"
#include "file_source.hpp"
#include "store.hpp"

namespace {

using ame::model::ByteCount;
using ame::model::ExpertKey;
using ame::model::ExpertTable;
using ame::model::FileOffset;
using ame::residency::ExpertEntry;
using ame::residency::FileSource;
using ame::residency::Lease;
using ame::residency::Store;
using ame::residency::StoreError;
using ame::residency::StoreOptions;
using ame::residency::StoreStats;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Residency test failure: " << message << '\n';
    ++failures;
  }
}

std::filesystem::path temporary_directory() {
  const auto path = std::filesystem::temp_directory_path() /
                    ("ame_residency_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

void write_pattern(const std::filesystem::path& path, std::uint32_t experts, std::uint32_t stride,
                   std::uint32_t payload) {
  std::vector<std::byte> bytes(static_cast<std::size_t>(experts) * stride, std::byte{0});
  for (std::uint32_t expert = 0; expert < experts; ++expert) {
    for (std::uint32_t index = 0; index < payload; ++index) {
      bytes[static_cast<std::size_t>(expert) * stride + index] =
          static_cast<std::byte>((expert * 17U + index) & 0xFFU);
    }
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

std::vector<ExpertEntry> make_entries(std::uint32_t experts, std::uint32_t stride,
                                      std::uint32_t payload, std::uint32_t layer = 0) {
  std::vector<ExpertEntry> entries;
  for (std::uint32_t expert = 0; expert < experts; ++expert) {
    entries.push_back(ExpertEntry{ExpertKey{layer, expert},
                                  FileOffset{static_cast<std::uint64_t>(expert) * stride},
                                  ByteCount{payload}});
  }
  return entries;
}

void check_layout() {
  const auto table = ExpertTable::create(FileOffset{64}, ByteCount{128}, ByteCount{64}, 4, 0,
                                         ByteCount{64 + 3 * 128 + 64});
  expect(table.ok(), "valid table creation");
  if (table.ok()) {
    const auto first = table.value().extent(ExpertKey{0, 0});
    const auto last = table.value().extent(ExpertKey{0, 3});
    expect(first.ok() && first.value().offset.value == 64, "first extent offset");
    expect(last.ok() && last.value().offset.value == 64 + 3 * 128, "last extent offset");
    expect(!table.value().extent(ExpertKey{1, 0}).ok(), "layer mismatch rejected");
    expect(!table.value().extent(ExpertKey{0, 4}).ok(), "expert outside table rejected");
  }
  expect(ExpertTable::create(FileOffset{0}, ByteCount{128}, ByteCount{128}, 0, 0, ByteCount{128})
                 .error() == ame::model::LayoutError::ZeroCount,
         "zero count rejected");
  expect(ExpertTable::create(FileOffset{0}, ByteCount{128}, ByteCount{256}, 1, 0, ByteCount{512})
                 .error() == ame::model::LayoutError::PayloadExceedsStride,
         "payload over stride rejected");
  expect(ExpertTable::create(FileOffset{3}, ByteCount{128}, ByteCount{64}, 1, 0, ByteCount{512})
                 .error() == ame::model::LayoutError::Misaligned,
         "misaligned base rejected");
  expect(ExpertTable::create(FileOffset{0}, ByteCount{128}, ByteCount{64}, 4, 0, ByteCount{128})
                 .error() == ame::model::LayoutError::BeyondFile,
         "table beyond file rejected");
  constexpr std::uint64_t max = UINT64_MAX;
  const auto overflow =
      ExpertTable::create(FileOffset{max - 31}, ByteCount{64}, ByteCount{64}, 2, 0, ByteCount{max});
  expect(!overflow.ok(), "offset overflow rejected");
}

void check_store() {
  const auto directory = temporary_directory();
  const auto path = directory / "experts.bin";
  constexpr std::uint32_t kExperts = 6;
  constexpr std::uint32_t kStride = 128;
  constexpr std::uint32_t kPayload = 64;
  write_pattern(path, kExperts, kStride, kPayload);

  auto source = FileSource::open(path.string());
  expect(source.ok(), "file source opens");
  if (!source.ok()) {
    return;
  }
  StoreOptions options;
  options.slot_count = 2;
  const auto entries = make_entries(kExperts, kStride, kPayload);
  auto created = Store::create(std::move(source).value(), entries, options);
  expect(created.ok(), "store creation");
  if (!created.ok()) {
    return;
  }
  Store& store = *created.value();
  expect(store.slot_bytes() == kPayload, "slot bytes");
  expect(store.arena().size() == static_cast<std::size_t>(kPayload) * 2, "arena bytes");

  ame::Cancellation stop;
  const ExpertKey first{0, 0};
  const ExpertKey second{0, 1};
  const ExpertKey third{0, 2};
  {
    auto lease = store.acquire(std::span<const ExpertKey>(&first, 1), stop);
    expect(lease.ok() && lease.value().valid(), "single acquire");
    if (lease.ok()) {
      const auto bytes = lease.value().payload(first);
      expect(bytes.size() == kPayload, "payload size");
      expect(static_cast<unsigned char>(bytes[0]) == 0, "expert zero first byte");
      expect(static_cast<unsigned char>(bytes[17]) == 17, "expert zero pattern");
      Lease moved = std::move(lease).value();
      expect(moved.valid(), "move keeps lease valid");
      expect(moved.payload(first).size() == kPayload, "moved payload");
    }
  }
  StoreStats stats = store.stats();
  expect(stats.hits == 0 && stats.misses == 1, "first acquire miss");
  expect(stats.pinned_slots == 0, "released pins");
  expect(stats.read_calls == 1 && stats.read_bytes == kPayload, "read accounting");

  const ExpertKey pair[] = {first, second};
  const ExpertKey triple[] = {first, second, third};
  {
    auto lease = store.acquire(pair, stop);
    expect(lease.ok(), "pair acquire");
    expect(store.stats().hits == 1, "pair hits first");
    expect(store.stats().misses == 2, "pair misses second");
    auto too_large = store.acquire(triple, stop);
    expect(!too_large.ok() && too_large.error() == StoreError::ResourceLimit,
           "unit larger than arena refused");
    auto pinned = store.acquire(std::span<const ExpertKey>(&third, 1), stop);
    expect(!pinned.ok() && pinned.error() == StoreError::ResourceLimit,
           "pinned slots are not evicted");
  }
  {
    auto lease = store.acquire(std::span<const ExpertKey>(&third, 1), stop);
    expect(lease.ok(), "acquire after release evicts");
    expect(store.stats().evictions >= 1, "eviction counted");
  }
  {
    const auto duplicate = store.acquire(pair, stop);
    expect(duplicate.ok(), "duplicate keys deduplicated");
    expect(duplicate.value().size() == 2, "deduplicated lease size");
  }
  {
    const ExpertKey unknown{3, 9};
    const auto missing = store.acquire(std::span<const ExpertKey>(&unknown, 1), stop);
    expect(!missing.ok() && missing.error() == StoreError::UnknownKey, "unknown key rejected");
  }
  {
    ame::Cancellation cancelled;
    cancelled.request();
    const auto stopped = store.acquire(std::span<const ExpertKey>(&third, 1), cancelled);
    expect(!stopped.ok() && stopped.error() == StoreError::Cancelled, "pre-requested stop");
    expect(store.stats().pinned_slots == 0, "cancellation leaves no pins");
  }

  // A truncated file produces a read failure and leaves no pins behind.
  const auto broken_path = directory / "broken.bin";
  write_pattern(broken_path, 2, kStride, kPayload);
  auto broken_source = FileSource::open(broken_path.string());
  expect(broken_source.ok(), "broken source opens");
  if (broken_source.ok()) {
    const auto broken_entries = make_entries(2, kStride, kPayload);
    auto broken = Store::create(std::move(broken_source).value(), broken_entries, options);
    expect(broken.ok(), "broken store creates");
    if (broken.ok()) {
      expect(::truncate(broken_path.c_str(), 0) == 0, "truncate fixture");
      const ExpertKey key{0, 0};
      const auto failed = broken.value()->acquire(std::span<const ExpertKey>(&key, 1), stop);
      expect(!failed.ok() && failed.error() == StoreError::ReadFailed, "short read rejected");
      expect(broken.value()->stats().pinned_slots == 0, "failed read leaves no pins");
      const auto retry = broken.value()->acquire(std::span<const ExpertKey>(&key, 1), stop);
      expect(!retry.ok(), "truncated retry remains refused");
    }
  }

  // Validation errors.
  {
    auto duplicate_entries = make_entries(2, kStride, kPayload);
    duplicate_entries[1].key = duplicate_entries[0].key;
    auto duplicate_source = FileSource::open(path.string());
    auto result = Store::create(std::move(duplicate_source).value(), duplicate_entries, options);
    expect(!result.ok() && result.error() == StoreError::DuplicateKey, "duplicate entry rejected");
  }
  {
    auto mixed = make_entries(2, kStride, kPayload);
    mixed[1].length = ByteCount{32};
    auto mixed_source = FileSource::open(path.string());
    auto result = Store::create(std::move(mixed_source).value(), mixed, options);
    expect(!result.ok() && result.error() == StoreError::NonUniformPayload,
           "mixed payload rejected");
  }
  {
    auto misaligned = make_entries(2, kStride, kPayload);
    misaligned[1].offset = FileOffset{3};
    auto aligned_source = FileSource::open(path.string());
    auto result = Store::create(std::move(aligned_source).value(), misaligned, options);
    expect(!result.ok() && result.error() == StoreError::Misaligned, "misaligned entry rejected");
  }
  {
    auto beyond = make_entries(2, kStride, kPayload);
    beyond[1].offset = FileOffset{1024};
    auto beyond_source = FileSource::open(path.string());
    auto result = Store::create(std::move(beyond_source).value(), beyond, options);
    expect(!result.ok() && result.error() == StoreError::EntryBeyondFile,
           "entry beyond file rejected");
  }
  std::filesystem::remove_all(directory);
}

void check_sparse_large_file() {
  const auto directory = temporary_directory();
  const auto path = directory / "sparse.bin";
  const int descriptor = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  expect(descriptor >= 0, "sparse file creation");
  if (descriptor < 0) {
    return;
  }
  constexpr std::uint64_t kBase = (std::uint64_t{4} << 30) + 128;
  constexpr std::uint64_t kFileSize = std::uint64_t{5} << 30;
  expect(::ftruncate(descriptor, static_cast<off_t>(kFileSize)) == 0, "sparse truncate");
  std::vector<std::byte> pattern(64);
  for (std::size_t index = 0; index < pattern.size(); ++index) {
    pattern[index] = static_cast<std::byte>((index + 5U) & 0xFFU);
  }
  expect(::pwrite(descriptor, pattern.data(), pattern.size(), static_cast<off_t>(kBase)) ==
             static_cast<ssize_t>(pattern.size()),
         "sparse write");
  ::close(descriptor);

  auto source = FileSource::open(path.string());
  expect(source.ok() && source.value().size_bytes() == kFileSize, "sparse source size");
  if (!source.ok()) {
    return;
  }
  std::vector<ExpertEntry> entries{ExpertEntry{ExpertKey{0, 0}, FileOffset{kBase}, ByteCount{64}}};
  StoreOptions options;
  options.slot_count = 1;
  auto store = Store::create(std::move(source).value(), entries, options);
  expect(store.ok(), "sparse store creates");
  if (store.ok()) {
    ame::Cancellation stop;
    const ExpertKey key{0, 0};
    auto lease = store.value()->acquire(std::span<const ExpertKey>(&key, 1), stop);
    expect(lease.ok(), "sparse acquire");
    if (lease.ok()) {
      const auto bytes = lease.value().payload(key);
      expect(bytes.size() == 64 && std::memcmp(bytes.data(), pattern.data(), 64) == 0,
             "sparse payload above 4 GiB");
    }
  }
  std::filesystem::remove_all(directory);
}

}  // namespace

static_assert(!std::is_copy_constructible_v<Lease>);
static_assert(std::is_move_constructible_v<Lease>);

int main() {
  check_layout();
  check_store();
  check_sparse_large_file();
  return failures == 0 ? 0 : 1;
}
