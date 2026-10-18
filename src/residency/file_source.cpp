#include "file_source.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>

namespace ame::residency {
namespace {

constexpr std::size_t kReadChunk = 1024U * 1024U;

}  // namespace

FileSource::FileSource(int descriptor, std::uint64_t size_bytes) noexcept
    : descriptor_(descriptor), size_bytes_(size_bytes) {}

FileSource::FileSource(FileSource&& other) noexcept
    : descriptor_(other.descriptor_), size_bytes_(other.size_bytes_) {
  other.descriptor_ = -1;
  other.size_bytes_ = 0;
}

FileSource& FileSource::operator=(FileSource&& other) noexcept {
  if (this != &other) {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
    descriptor_ = other.descriptor_;
    size_bytes_ = other.size_bytes_;
    other.descriptor_ = -1;
    other.size_bytes_ = 0;
  }
  return *this;
}

FileSource::~FileSource() {
  if (descriptor_ >= 0) {
    ::close(descriptor_);
  }
}

Result<FileSource, FileError> FileSource::open(const std::string& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return FileError::OpenFailed;
  }
  struct stat status {};
  if (::fstat(descriptor, &status) != 0) {
    ::close(descriptor);
    return FileError::StatFailed;
  }
  if (!S_ISREG(status.st_mode)) {
    ::close(descriptor);
    return FileError::NotRegularFile;
  }
  if (status.st_size < 0) {
    ::close(descriptor);
    return FileError::StatFailed;
  }
  return FileSource(descriptor, static_cast<std::uint64_t>(status.st_size));
}

Result<void, FileError> FileSource::read_exact(std::uint64_t offset,
                                               std::span<std::byte> destination,
                                               const Cancellation& stop) const {
  if (descriptor_ < 0) {
    return FileError::ReadFailed;
  }
  if (offset > size_bytes_ || destination.size() > size_bytes_ - offset) {
    return FileError::UnexpectedEnd;
  }
  std::size_t completed = 0;
  while (completed < destination.size()) {
    if (stop.requested()) {
      return FileError::Cancelled;
    }
    const std::size_t remaining = destination.size() - completed;
    const std::size_t chunk = std::min(remaining, kReadChunk);
    const ssize_t count = ::pread(descriptor_, destination.data() + completed, chunk,
                                  static_cast<off_t>(offset + completed));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return FileError::ReadFailed;
    }
    if (count == 0) {
      return FileError::UnexpectedEnd;
    }
    completed += static_cast<std::size_t>(count);
  }
  return {};
}

}  // namespace ame::residency
