#include "reader_pool.hpp"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "cancellation.hpp"
#include "file_source.hpp"

namespace {

using ame::residency::FileError;
using ame::residency::FileSource;
using ame::residency::ReaderPool;
using ame::residency::ReadJob;

int failures = 0;
void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Reader pool test failure: " << message << '\n';
    ++failures;
  }
}

constexpr std::size_t kJobBytes = 1U << 20;  // one read chunk

std::byte pattern(std::uint64_t index) {
  return static_cast<std::byte>((index * 31U + index / 251U) & 0xFFU);
}

struct Fixture {
  std::filesystem::path directory, file;
  std::uint32_t jobs{0};
};

Fixture make_fixture(const std::string& name, std::uint32_t jobs) {
  Fixture fixture;
  fixture.jobs = jobs;
  fixture.directory =
      std::filesystem::temp_directory_path() /
      ("ame_readers_" + name + "_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(fixture.directory);
  std::filesystem::create_directories(fixture.directory);
  fixture.file = fixture.directory / "data.bin";
  std::vector<std::byte> bytes(static_cast<std::size_t>(jobs) * kJobBytes);
  for (std::size_t index = 0; index < bytes.size(); ++index) bytes[index] = pattern(index);
  std::ofstream(fixture.file, std::ios::binary)
      .write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  return fixture;
}

std::vector<ReadJob> make_jobs(std::vector<std::byte>& destination, std::uint32_t count,
                               bool reversed = false) {
  destination.assign(static_cast<std::size_t>(count) * kJobBytes, std::byte{0xEE});
  std::vector<ReadJob> jobs;
  for (std::uint32_t index = 0; index < count; ++index) {
    const std::uint32_t source = reversed ? count - 1 - index : index;
    jobs.push_back(
        {static_cast<std::uint64_t>(source) * kJobBytes,
         std::span<std::byte>(destination.data() + static_cast<std::size_t>(index) * kJobBytes,
                              kJobBytes),
         false});
  }
  return jobs;
}

bool bytes_match(const std::vector<std::byte>& destination, std::uint32_t count, bool reversed) {
  for (std::uint32_t index = 0; index < count; ++index) {
    const std::uint32_t source = reversed ? count - 1 - index : index;
    for (std::size_t offset = 0; offset < kJobBytes; offset += 4099) {
      if (destination[static_cast<std::size_t>(index) * kJobBytes + offset] !=
          pattern(static_cast<std::uint64_t>(source) * kJobBytes + offset)) {
        return false;
      }
    }
  }
  return true;
}

void check_byte_equality() {
  const auto fixture = make_fixture("equal", 12);
  for (const unsigned threads : {1U, 2U, 4U}) {
    auto source = FileSource::open(fixture.file.string());
    if (!source.ok()) return expect(false, "source opens");
    ReaderPool pool(source.value(), threads);
    expect(pool.threads() == threads, "thread count is honoured");
    std::vector<std::byte> destination;
    ame::Cancellation stop;
    for (const bool reversed : {false, true}) {
      auto jobs = make_jobs(destination, 12, reversed);
      const auto result = pool.read_all(jobs, stop);
      expect(result.ok(), "every job completes");
      expect(bytes_match(destination, 12, reversed), "every job's bytes equal the file");
    }
    expect(source.value().pread_bytes() == 2U * 12U * kJobBytes,
           "syscall bytes are counted exactly");
  }
  expect(ReaderPool(FileSource::open(fixture.file.string()).value(), 99).threads() == 4,
         "the pool never exceeds four readers");
  std::filesystem::remove_all(fixture.directory);
}

// After read_all returns no worker may still be writing: poll the destination and require
// it to stay frozen.
bool frozen_after_return(const std::vector<std::byte>& destination) {
  const std::vector<std::byte> snapshot = destination;
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  return snapshot == destination;
}

void check_failure_with_jobs_in_flight() {
  const auto fixture = make_fixture("fail", 16);
  for (const unsigned threads : {2U, 4U}) {
    auto source = FileSource::open(fixture.file.string());
    if (!source.ok()) return expect(false, "source opens");
    ReaderPool pool(source.value(), threads);
    // Allow 5 MiB in total: some jobs finish, one crosses the limit, others are in flight.
    expect(::setenv("AME_IO_FAIL_AFTER_BYTES", std::to_string(5U * kJobBytes).c_str(), 1) == 0,
           "set fault limit");
    source.value().reload_fault_injection();
    std::vector<std::byte> destination;
    auto jobs = make_jobs(destination, 16);
    ame::Cancellation stop;
    const auto result = pool.read_all(jobs, stop);
    expect(!result.ok() && result.error() == FileError::ReadFailed,
           "a failing job fails the batch with its error");
    expect(frozen_after_return(destination), "no reader writes after the failure returns");
    expect(source.value().pread_bytes() <= 5U * kJobBytes + 4U * kJobBytes,
           "queued jobs stop being read after the failure");
    (void)::unsetenv("AME_IO_FAIL_AFTER_BYTES");
    source.value().reload_fault_injection();
    // The pool is reusable once the fault is gone.
    auto again = make_jobs(destination, 4);
    expect(pool.read_all(again, stop).ok() && bytes_match(destination, 4, false),
           "the pool works again after a failed batch");
  }
  std::filesystem::remove_all(fixture.directory);
}

void check_cancellation_mid_batch() {
  const auto fixture = make_fixture("cancel", 16);
  for (const unsigned threads : {1U, 2U, 4U}) {
    auto source = FileSource::open(fixture.file.string());
    if (!source.ok()) return expect(false, "source opens");
    ReaderPool pool(source.value(), threads);
    expect(::setenv("AME_IO_CANCEL_AFTER_CHUNKS", "1", 1) == 0, "set cancel injection");
    source.value().reload_fault_injection();
    std::vector<std::byte> destination;
    auto jobs = make_jobs(destination, 16);
    ame::Cancellation stop;
    const auto result = pool.read_all(jobs, stop);
    expect(!result.ok() && result.error() == FileError::Cancelled, "cancellation is reported");
    expect(frozen_after_return(destination), "no reader writes after cancellation returns");
    (void)::unsetenv("AME_IO_CANCEL_AFTER_CHUNKS");
    source.value().reload_fault_injection();
    ame::Cancellation fresh;
    auto again = make_jobs(destination, 4);
    expect(pool.read_all(again, fresh).ok() && bytes_match(destination, 4, false),
           "a fresh cancellation token works after a cancelled batch");
  }
  std::filesystem::remove_all(fixture.directory);
}

void check_unexpected_end_and_empty_batches() {
  const auto fixture = make_fixture("end", 2);
  auto source = FileSource::open(fixture.file.string());
  if (!source.ok()) return expect(false, "source opens");
  ReaderPool pool(source.value(), 2);
  std::vector<std::byte> destination(kJobBytes);
  ame::Cancellation stop;
  const std::vector<ReadJob> past = {
      {2U * kJobBytes - 100, std::span<std::byte>(destination), false}};
  const auto result = pool.read_all(past, stop);
  expect(!result.ok() && result.error() == FileError::UnexpectedEnd, "a read past the end fails");
  expect(pool.read_all({}, stop).ok(), "an empty batch is a no-op");
  std::filesystem::remove_all(fixture.directory);
}

}  // namespace

int main() {
  check_byte_equality();
  check_failure_with_jobs_in_flight();
  check_cancellation_mid_batch();
  check_unexpected_end_and_empty_batches();
  return failures == 0 ? 0 : 1;
}
