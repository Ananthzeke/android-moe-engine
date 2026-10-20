#include "sha256.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>

namespace ame::crypto {
namespace {
constexpr std::uint32_t kK[64] = {
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

inline std::uint32_t rotr(std::uint32_t value, int count) {
  return (value >> count) | (value << (32 - count));
}
}  // namespace

bool sha256_arm_available() {
  return (getauxval(AT_HWCAP) & (1UL << 6)) != 0;  // HWCAP_SHA2
}

void sha256_arm_compress(std::uint32_t state[8], const std::uint8_t* block) {
  std::uint32_t words[64];
  for (int index = 0; index < 16; ++index) {
    words[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24U) |
                   (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16U) |
                   (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8U) |
                   static_cast<std::uint32_t>(block[index * 4 + 3]);
  }
  for (int index = 16; index < 64; ++index) {
    const std::uint32_t s0 =
        rotr(words[index - 15], 7) ^ rotr(words[index - 15], 18) ^ (words[index - 15] >> 3U);
    const std::uint32_t s1 =
        rotr(words[index - 2], 17) ^ rotr(words[index - 2], 19) ^ (words[index - 2] >> 10U);
    words[index] = words[index - 16] + s0 + words[index - 7] + s1;
  }
  uint32x4_t abcd = vld1q_u32(state);
  uint32x4_t efgh = vld1q_u32(state + 4);
  for (int group = 0; group < 16; ++group) {
    const uint32x4_t wk = vaddq_u32(vld1q_u32(words + group * 4), vld1q_u32(kK + group * 4));
    const uint32x4_t next_abcd = vsha256hq_u32(abcd, efgh, wk);
    efgh = vsha256h2q_u32(efgh, abcd, wk);
    abcd = next_abcd;
  }
  abcd = vaddq_u32(abcd, vld1q_u32(state));
  efgh = vaddq_u32(efgh, vld1q_u32(state + 4));
  vst1q_u32(state, abcd);
  vst1q_u32(state + 4, efgh);
}
}  // namespace ame::crypto
#endif