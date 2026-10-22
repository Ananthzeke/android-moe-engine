#include <cstdio>
#include <vector>

#include "prefill_plan.hpp"

namespace {
bool check(bool condition, int line) {
  return condition ? true : (std::fprintf(stderr, "line %d\n", line), false);
}
#define CHECK(x)                         \
  do {                                   \
    if (!check((x), __LINE__)) return 1; \
  } while (0)
}  // namespace
int main() {
  using ame::prefill::ConsumerBound;
  using ame::prefill::PlanError;
  std::vector<ConsumerBound> two = {{8, 2, 4, 64}, {8, 2, 4, 64}};
  auto plan = ame::prefill::make_contract_plan(two, 8);
  CHECK(plan.ok() && plan.value().unit_tokens == 2);
  std::vector<ConsumerBound> different = {{8, 2, 4, 64}, {16, 4, 8, 64}};
  plan = ame::prefill::make_contract_plan(different, 8);
  CHECK(plan.ok() && plan.value().unit_tokens == 2);
  std::vector<ConsumerBound> below = {{8, 4, 3, 64}};
  plan = ame::prefill::make_contract_plan(below, 8);
  CHECK(!plan.ok() && plan.error() == PlanError::Capacity);
  std::vector<ConsumerBound> lease = {{8, 2, 8, 1}};
  plan = ame::prefill::make_contract_plan(lease, 8);
  CHECK(!plan.ok() && plan.error() == PlanError::LeaseCapacity);
  std::vector<ConsumerBound> invalid = {{8, 0, 8, 64}};
  plan = ame::prefill::make_contract_plan(invalid, 8);
  CHECK(!plan.ok() && plan.error() == PlanError::InvalidExpertUsed);
  std::vector<ConsumerBound> zero = {{0, 1, 8, 64}};
  plan = ame::prefill::make_contract_plan(zero, 8);
  CHECK(!plan.ok() && plan.error() == PlanError::InvalidExpertCount);
  plan = ame::prefill::make_contract_plan(two, 0);
  CHECK(!plan.ok() && plan.error() == PlanError::InvalidBatch);
  std::vector<ConsumerBound> capped = {{256, 1, 256, 256}};
  plan = ame::prefill::make_contract_plan(capped, 256);
  CHECK(plan.ok() && plan.value().unit_tokens == 128);
  return 0;
}