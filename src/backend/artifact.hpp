#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include "result.hpp"

struct gguf_context;

namespace ame::backend {

struct ExpectedIdentity {
  std::string sha256;
  std::uint64_t bytes{0};
};

struct TensorInfo {
  std::string name;
  std::uint64_t absolute_offset{0};
  std::uint64_t nbytes{0};
  std::uint32_t type{0};
  std::array<std::int64_t, 4> ne{1, 1, 1, 1};
};

enum class ArtifactError {
  IoError,
  SizeMismatch,
  HashMismatch,
  InvalidExpectedIdentity,
  GgufError,
  TensorTableError,
};

// One read-only artifact FD, one verified digest, and one parsed GGUF tensor
// table. The backend model load and every expert size-class source must use this
// same FD (or its /proc/self/fd path); separate pathname opens are not bound.
class Artifact {
 public:
  Artifact() = default;
  ~Artifact();
  Artifact(Artifact&&) noexcept;
  Artifact& operator=(Artifact&&) noexcept;
  Artifact(const Artifact&) = delete;
  Artifact& operator=(const Artifact&) = delete;

  [[nodiscard]] static Result<std::unique_ptr<Artifact>, ArtifactError> open(
      const std::string& path, const ExpectedIdentity& expected, std::string& error);

  [[nodiscard]] int fd() const noexcept { return fd_; }
  [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
  [[nodiscard]] const std::string& sha256() const noexcept { return sha256_; }
  [[nodiscard]] double hash_seconds() const noexcept { return hash_seconds_; }
  [[nodiscard]] const std::string& proc_path() const noexcept { return proc_path_; }
  [[nodiscard]] const std::map<std::string, TensorInfo>& tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] const TensorInfo* find(std::string_view name) const;
  // Detect pathname-independent mutation/truncation of the retained inode before
  // the artifact is used. This is a metadata check, not content authentication.
  [[nodiscard]] bool recheck(std::string& error) const;

 private:
  int fd_{-1};
  std::uint64_t size_{0};
  std::string sha256_;
  double hash_seconds_{0.0};
  std::string proc_path_;
  std::uint64_t device_{0};
  std::uint64_t inode_{0};
  std::int64_t mtime_sec_{0};
  std::int64_t mtime_nsec_{0};
  gguf_context* gguf_{nullptr};
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace ame::backend