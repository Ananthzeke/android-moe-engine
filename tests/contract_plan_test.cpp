#include <cassert>
#include <cstdint>
#include <vector>

#include "prefill_plan.hpp"

int main() {
  using ame::prefill::ConsumerBound;
  using ame::prefill::PlanError;
  std::vector<ConsumerBound> two = {{8, 2, 4, 64}, {8, 2, 4, 64}};
  auto plan = ame::prefill::make_contract_plan(two, 8);
  assert(plan.ok() && plan.value().unit_tokens == 2);

  std::vector<ConsumerBound> different = {{8, 2, 4, 64}, {16, 4, 8, 64}};
  plan = ame::prefill::make_contract_plan(different, 8);
  assert(plan.ok() && plan.value().unit_tokens == 2);

  std::vector<ConsumerBound> below = {{8, 4, 3, 64}};
  plan = ame::prefill::make_contract_plan(below, 8);
  assert(!plan.ok() && plan.error() == PlanError::Capacity);

  std::vector<ConsumerBound> lease = {{8, 2, 8, 1}};
  plan = ame::prefill::make_contract_plan(lease, 8);
  assert(!plan.ok() && plan.error() == PlanError::LeaseCapacity);

  std::vector<ConsumerBound> invalid = {{8, 0, 8, 64}};
  plan = ame::prefill::make_contract_plan(invalid, 8);
  assert(!plan.ok() && plan.error() == PlanError::InvalidExpertUsed);

  std::vector<ConsumerBound> zero = {{0, 1, 8, 64}};
  plan = ame::prefill::make_contract_plan(zero, 8);
  assert(!plan.ok() && plan.error() == PlanError::InvalidExpertCount);

  plan = ame::prefill::make_contract_plan(two, 0);
  assert(!plan.ok() && plan.error() == PlanError::InvalidBatch);
  return 0;
}