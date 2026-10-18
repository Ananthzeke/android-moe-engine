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

 private:
  FileSource(int descriptor, std::uint64_t size_bytes) noexcept;
  int descriptor_ = -1;
  std::uint64_t size_bytes_ = 0;
};

}  // namespace ame::residency
