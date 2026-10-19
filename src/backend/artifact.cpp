#include "artifact.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <vector>

#include "gguf.h"
#include "sha256.hpp"

namespace ame::backend {
namespace {

bool is_hex_digest(const std::string& text) {
  if (text.size() != 64) {
    return false;
  }
  for (const char character : text) {
    if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

}  // namespace

Artifact::~Artifact() {
  if (gguf_ != nullptr) {
    gguf_free(gguf_);
    gguf_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

Artifact::Artifact(Artifact&& other) noexcept
    : fd_(other.fd_),
      size_(other.size_),
      sha256_(std::move(other.sha256_)),
      hash_seconds_(other.hash_seconds_),
      proc_path_(std::move(other.proc_path_)),
      device_(other.device_),
      inode_(other.inode_),
      mtime_sec_(other.mtime_sec_),
      mtime_nsec_(other.mtime_nsec_),
      gguf_(other.gguf_),
      tensors_(std::move(other.tensors_)) {
  other.fd_ = -1;
  other.gguf_ = nullptr;
}

Artifact& Artifact::operator=(Artifact&& other) noexcept {
  if (this != &other) {
    if (gguf_ != nullptr) {
      gguf_free(gguf_);
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
    fd_ = other.fd_;
    size_ = other.size_;
    sha256_ = std::move(other.sha256_);
    hash_seconds_ = other.hash_seconds_;
    proc_path_ = std::move(other.proc_path_);
    device_ = other.device_;
    inode_ = other.inode_;
    mtime_sec_ = other.mtime_sec_;
    mtime_nsec_ = other.mtime_nsec_;
    gguf_ = other.gguf_;
    tensors_ = std::move(other.tensors_);
    other.fd_ = -1;
    other.gguf_ = nullptr;
  }
  return *this;
}

Result<std::unique_ptr<Artifact>, ArtifactError> Artifact::open(const std::string& path,
                                                                const ExpectedIdentity& expected,
                                                                std::string& error) {
  if (!is_hex_digest(expected.sha256) || expected.bytes == 0) {
    error = "expected identity must be a 64-hex digest and a positive byte count";
    return ArtifactError::InvalidExpectedIdentity;
  }
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    error = "cannot open artifact";
    return ArtifactError::IoError;
  }
  struct stat before {};
  if (::fstat(fd, &before) != 0 || before.st_size < 0) {
    error = "cannot stat artifact";
    ::close(fd);
    return ArtifactError::IoError;
  }
  const auto size = static_cast<std::uint64_t>(before.st_size);
  if (size != expected.bytes) {
    error = "artifact size does not match the enrolled identity";
    ::close(fd);
    return ArtifactError::SizeMismatch;
  }
  ame::crypto::Sha256Result digest;
  if (!ame::crypto::sha256_fd(fd, size, digest, error)) {
    ::close(fd);
    return ArtifactError::IoError;
  }
  if (digest.hex != expected.sha256) {
    error = "artifact SHA-256 does not match the enrolled identity";
    ::close(fd);
    return ArtifactError::HashMismatch;
  }
  struct stat after {};
  if (::fstat(fd, &after) != 0 || after.st_size != before.st_size ||
      after.st_ino != before.st_ino || after.st_dev != before.st_dev ||
      after.st_mtim.tv_sec != before.st_mtim.tv_sec ||
      after.st_mtim.tv_nsec != before.st_mtim.tv_nsec) {
    error = "artifact changed during identity verification";
    ::close(fd);
    return ArtifactError::IoError;
  }

  auto artifact = std::unique_ptr<Artifact>(new Artifact());
  artifact->fd_ = fd;
  artifact->size_ = size;
  artifact->sha256_ = digest.hex;
  artifact->hash_seconds_ = digest.seconds;
  artifact->proc_path_ = "/proc/self/fd/" + std::to_string(fd);
  artifact->device_ = static_cast<std::uint64_t>(before.st_dev);
  artifact->inode_ = static_cast<std::uint64_t>(before.st_ino);
  artifact->mtime_sec_ = static_cast<std::int64_t>(before.st_mtim.tv_sec);
  artifact->mtime_nsec_ = static_cast<std::int64_t>(before.st_mtim.tv_nsec);

  gguf_init_params gguf_params{};
  gguf_params.no_alloc = true;
  gguf_params.ctx = nullptr;
  artifact->gguf_ = gguf_init_from_file(artifact->proc_path_.c_str(), gguf_params);
  if (artifact->gguf_ == nullptr) {
    error = "cannot parse GGUF metadata";
    return ArtifactError::GgufError;
  }
  const std::uint64_t data_offset =
      static_cast<std::uint64_t>(gguf_get_data_offset(artifact->gguf_));
  const std::int64_t count = gguf_get_n_tensors(artifact->gguf_);
  if (count <= 0 || data_offset > size) {
    error = "GGUF tensor table is empty or outside the file";
    return ArtifactError::TensorTableError;
  }
  for (std::int64_t index = 0; index < count; ++index) {
    const char* raw_name = gguf_get_tensor_name(artifact->gguf_, index);
    if (raw_name == nullptr) {
      error = "GGUF tensor has no name";
      return ArtifactError::TensorTableError;
    }
    TensorInfo info;
    info.name = raw_name;
    const std::uint64_t relative =
        static_cast<std::uint64_t>(gguf_get_tensor_offset(artifact->gguf_, index));
    if (relative > size || data_offset > size - relative) {
      error = "GGUF tensor offset is outside the file";
      return ArtifactError::TensorTableError;
    }
    info.absolute_offset = data_offset + relative;
    info.nbytes = static_cast<std::uint64_t>(gguf_get_tensor_size(artifact->gguf_, index));
    if (info.nbytes > size - info.absolute_offset) {
      error = "GGUF tensor extent is outside the file";
      return ArtifactError::TensorTableError;
    }
    info.type = static_cast<std::uint32_t>(gguf_get_tensor_type(artifact->gguf_, index));
    const std::int64_t* ne = gguf_get_tensor_ne(artifact->gguf_, index);
    if (ne == nullptr) {
      error = "GGUF tensor has no dimensions";
      return ArtifactError::TensorTableError;
    }
    for (std::size_t dim = 0; dim < info.ne.size(); ++dim) {
      info.ne[dim] = ne[dim];
    }
    if (!artifact->tensors_.emplace(info.name, std::move(info)).second) {
      error = "duplicate GGUF tensor name";
      return ArtifactError::TensorTableError;
    }
  }
  return artifact;
}

bool Artifact::recheck(std::string& error) const {
  struct stat status {};
  if (::fstat(fd_, &status) != 0 || status.st_size < 0) {
    error = "cannot re-stat artifact";
    return false;
  }
  if (static_cast<std::uint64_t>(status.st_size) != size_ ||
      static_cast<std::uint64_t>(status.st_dev) != device_ ||
      static_cast<std::uint64_t>(status.st_ino) != inode_ ||
      static_cast<std::int64_t>(status.st_mtim.tv_sec) != mtime_sec_ ||
      static_cast<std::int64_t>(status.st_mtim.tv_nsec) != mtime_nsec_) {
    error = "artifact changed after identity verification";
    return false;
  }
  return true;
}

const TensorInfo* Artifact::find(std::string_view name) const {
  const auto found = tensors_.find(std::string(name));
  return found == tensors_.end() ? nullptr : &found->second;
}

}  // namespace ame::backend