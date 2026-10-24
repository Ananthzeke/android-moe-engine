#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "cancellation.hpp"
#include "result.hpp"

namespace ame::residency {

enum class FileError {
  OpenFailed,
  NotRegularFile,
  StatFailed,
  ReadFailed,
  UnexpectedEnd,
  Cancelled,
};

// Read-only, move-only file source. Reads use bounded pread chunks and never
// allocate or map the whole file. Cancellation is checked between chunks; a single
// storage read has no wall-clock deadline.
class FileSource {
 public:
  static Result<FileSource, FileError> open(const std::string& path);
  // Duplicate an already-open read-only FD. This is how every expert size-class
  // source is bound to the single verified artifact handle.
  static Result<FileSource, FileError> open_fd(int descriptor);
  FileSource(FileSource&& other) noexcept;
  FileSource& operator=(FileSource&& other) noexcept;
  FileSource(const FileSource&) = delete;
  FileSource& operator=(const FileSource&) = delete;
  ~FileSource();

  [[nodiscard]] std::uint64_t size_bytes() const noexcept { return size_bytes_; }
  [[nodiscard]] Result<void, FileError> read_exact(std::uint64_t offset,
                                                   std::span<std::byte> destination,
                                                   const Cancellation& stop) const;
  [[nodiscard]] bool valid() const noexcept { return descriptor_ >= 0; }

  // Test-only fault injection, read from AME_IO_FAIL_AFTER_BYTES,
  // AME_IO_CANCEL_AFTER_CHUNKS, and AME_IO_INJECT_EINTR_ONCE. The environment is read
  // when the source opens and again on each call to this function, never per chunk. Call
  // it only when no read is in flight. It keeps the source's cumulative byte count.
  void reload_fault_injection() const noexcept;

  // Page-cache hints on the read descriptor. They are advisory: the kernel may ignore them.
  // advise_random asks for no readahead; advise_dontneed drops clean cached pages of one
  // extent. Both are no-ops on failure.
  void advise_random() const noexcept;
  void advise_dontneed(std::uint64_t offset, std::uint64_t length) const noexcept;

  // Actual pread syscalls and the bytes they returned. These differ from the expert
  // read counts kept by the store, which count completed expert slices.
  [[nodiscard]] std::uint64_t pread_calls() const noexcept { return pread_calls_; }
  [[nodiscard]] std::uint64_t pread_bytes() const noexcept { return pread_bytes_; }

 private:
  FileSource(int descriptor, std::uint64_t size_bytes) noexcept;
  int descriptor_ = -1;
  std::uint64_t size_bytes_ = 0;
  mutable std::uint64_t bytes_read_ = 0;
  mutable bool eintr_consumed_ = false;
  mutable bool cancel_injected_ = false;
  mutable std::uint64_t pread_calls_ = 0;
  mutable std::uint64_t pread_bytes_ = 0;
  mutable std::uint64_t fail_limit_ = UINT64_MAX;
  mutable bool cancel_after_chunks_ = false;
  mutable bool inject_eintr_once_ = false;
};

}  // namespace ame::residency