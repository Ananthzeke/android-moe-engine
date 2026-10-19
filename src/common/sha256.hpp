#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ame::crypto {

struct Sha256Result {
  std::string hex;
  std::uint64_t bytes{0};
  double seconds{0.0};
};

// Portable SHA-256 over an already-open regular file. The caller owns the FD and
// must supply the expected byte count; short reads and early EOF are errors.
[[nodiscard]] bool sha256_fd(int fd, std::uint64_t expected_bytes, Sha256Result& result,
                             std::string& error);

// Test/helper entry point for small in-memory values.
[[nodiscard]] std::string sha256_hex(std::string_view data);

}  // namespace ame::crypto
