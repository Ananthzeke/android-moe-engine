#include "sha256.hpp"

#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <vector>

namespace ame::crypto {
#if defined(__aarch64__)
bool sha256_arm_available();
void sha256_arm_compress(std::uint32_t state[8], const std::uint8_t* block);
#endif

namespace {

constexpr std::uint32_t kInitial[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                       0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};

constexpr std::uint32_t kRound[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U};

constexpr std::uint32_t rotr(std::uint32_t value, int count) {
  return (value >> count) | (value << (32 - count));
}

void compress(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
#if defined(__aarch64__)
  if (sha256_arm_available()) {
    sha256_arm_compress(state.data(), block);
    return;
  }
#endif
  std::array<std::uint32_t, 64> words{};
  for (int index = 0; index < 16; ++index) {
    words[static_cast<std::size_t>(index)] =
        (static_cast<std::uint32_t>(block[index * 4]) << 24U) |
        (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16U) |
        (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8U) |
        static_cast<std::uint32_t>(block[index * 4 + 3]);
  }
  for (int index = 16; index < 64; ++index) {
    const std::uint32_t s0 = rotr(words[static_cast<std::size_t>(index - 15)], 7) ^
                             rotr(words[static_cast<std::size_t>(index - 15)], 18) ^
                             (words[static_cast<std::size_t>(index - 15)] >> 3U);
    const std::uint32_t s1 = rotr(words[static_cast<std::size_t>(index - 2)], 17) ^
                             rotr(words[static_cast<std::size_t>(index - 2)], 19) ^
                             (words[static_cast<std::size_t>(index - 2)] >> 10U);
    words[static_cast<std::size_t>(index)] = words[static_cast<std::size_t>(index - 16)] + s0 +
                                             words[static_cast<std::size_t>(index - 7)] + s1;
  }
  std::uint32_t a = state[0];
  std::uint32_t b = state[1];
  std::uint32_t c = state[2];
  std::uint32_t d = state[3];
  std::uint32_t e = state[4];
  std::uint32_t f = state[5];
  std::uint32_t g = state[6];
  std::uint32_t h = state[7];
  for (int index = 0; index < 64; ++index) {
    const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const std::uint32_t choice = (e & f) ^ (~e & g);
    const std::uint32_t temp1 =
        h + s1 + choice + kRound[index] + words[static_cast<std::size_t>(index)];
    const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = s0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

class Hasher {
 public:
  void update(const std::uint8_t* data, std::size_t size) {
    total_ += size;
    while (size != 0) {
      const std::size_t space = 64 - buffer_size_;
      const std::size_t take = size < space ? size : space;
      std::memcpy(buffer_.data() + buffer_size_, data, take);
      buffer_size_ += take;
      data += take;
      size -= take;
      if (buffer_size_ == 64) {
        compress(state_, buffer_.data());
        buffer_size_ = 0;
      }
    }
  }

  std::string final_hex() {
    const std::uint64_t bit_count = total_ * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56) {
      while (buffer_size_ < 64) {
        buffer_[buffer_size_++] = 0;
      }
      compress(state_, buffer_.data());
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56) {
      buffer_[buffer_size_++] = 0;
    }
    for (int index = 7; index >= 0; --index) {
      buffer_[buffer_size_++] = static_cast<std::uint8_t>((bit_count >> (index * 8)) & 0xFFU);
    }
    compress(state_, buffer_.data());
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output;
    output.reserve(64);
    for (const std::uint32_t word : state_) {
      for (int shift = 28; shift >= 0; shift -= 4) {
        output += kHex[(word >> shift) & 0xFU];
      }
    }
    return output;
  }

 private:
  std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_{0};
  std::uint64_t total_{0};
};

}  // namespace

std::string sha256_hex(std::string_view data) {
  Hasher hasher;
  hasher.update(reinterpret_cast<const std::uint8_t*>(data.data()), data.size());
  return hasher.final_hex();
}

bool sha256_fd(int fd, std::uint64_t expected_bytes, Sha256Result& result, std::string& error) {
  const auto start = std::chrono::steady_clock::now();
  Hasher hasher;
  std::vector<std::uint8_t> buffer(4U * 1024U * 1024U);
  std::uint64_t offset = 0;
  while (offset < expected_bytes) {
    const std::uint64_t remaining = expected_bytes - offset;
    const std::size_t chunk =
        remaining < buffer.size() ? static_cast<std::size_t>(remaining) : buffer.size();
    ssize_t count = ::pread(fd, buffer.data(), chunk, static_cast<off_t>(offset));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = "pread failed while hashing";
      return false;
    }
    if (count == 0) {
      error = "unexpected EOF while hashing";
      return false;
    }
    hasher.update(buffer.data(), static_cast<std::size_t>(count));
    offset += static_cast<std::uint64_t>(count);
  }
  result.hex = hasher.final_hex();
  result.bytes = offset;
  result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return true;
}

}  // namespace ame::crypto