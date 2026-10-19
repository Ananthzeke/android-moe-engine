#include "prefill_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "Prefill test failure: " << message << '\n';
    ++failures;
  }
}

void check(std::uint32_t experts, std::uint32_t used, std::uint32_t slots, std::uint32_t batch,
           std::uint32_t lease, std::uint32_t expected) {
  const auto plan = ame::prefill::make_plan(experts, used, slots, batch, lease);
  expect(plan.ok() && plan.value().unit_tokens == expected, "unexpected unit");
}
}  // namespace

int main() {
  check(256, 8, 64, 64, 128, 8);
  check(256, 8, 8, 64, 128, 1);
  check(256, 8, 9, 64, 128, 1);
  check(256, 8, 64, 4, 128, 4);
  check(256, 8, 64, 64, 8, 1);
  const auto too_small = ame::prefill::make_plan(256, 8, 7, 64, 128);
  expect(!too_small.ok() && too_small.error() == ame::prefill::PlanError::Capacity,
         "slots below top-k refused");
  const auto zero_batch = ame::prefill::make_plan(256, 8, 64, 0, 128);
  expect(!zero_batch.ok() && zero_batch.error() == ame::prefill::PlanError::InvalidBatch,
         "zero batch refused");
  const auto lease_small = ame::prefill::make_plan(256, 8, 64, 64, 4);
  expect(!lease_small.ok() && lease_small.error() == ame::prefill::PlanError::LeaseCapacity,
         "lease capacity refused");
  const auto plan = ame::prefill::make_plan(256, 8, 64, 64, 128);
  if (plan.ok()) {
    expect(ame::prefill::chunks_for(63, plan.value().unit_tokens) == 8, "63 chunks");
    expect(ame::prefill::chunks_for(64, plan.value().unit_tokens) == 8, "64 chunks");
    expect(ame::prefill::chunks_for(65, plan.value().unit_tokens) == 9, "65 chunks");
    expect(63u * 8u > 64u, "63-token union bound exceeds slots only without chunking");
    for (const std::uint64_t tokens : {63ULL, 64ULL, 65ULL}) {
      const std::uint32_t chunks = ame::prefill::chunks_for(tokens, plan.value().unit_tokens);
      for (std::uint32_t index = 0; index < chunks; ++index) {
        const std::uint64_t begin = static_cast<std::uint64_t>(index) * plan.value().unit_tokens;
        const std::uint64_t count =
            std::min<std::uint64_t>(plan.value().unit_tokens, tokens - begin);
        expect(std::min<std::uint64_t>(256, count * 8) <= 64, "chunk union bound");
      }
    }
  }
  return failures == 0 ? 0 : 1;
}