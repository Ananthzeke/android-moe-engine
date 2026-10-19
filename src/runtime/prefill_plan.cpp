#include "prefill_plan.hpp"

#include <algorithm>

namespace ame::prefill {

Result<Plan, PlanError> make_plan(std::uint32_t expert_count, std::uint32_t expert_used,
                                  std::uint32_t slots, std::uint32_t batch_size,
                                  std::uint32_t lease_limit) {
  if (expert_count == 0) {
    return PlanError::InvalidExpertCount;
  }
  if (expert_used == 0 || expert_used > expert_count) {
    return PlanError::InvalidExpertUsed;
  }
  if (slots == 0) {
    return PlanError::InvalidSlots;
  }
  if (batch_size == 0) {
    return PlanError::InvalidBatch;
  }
  if (slots < expert_used) {
    return PlanError::Capacity;
  }
  if (lease_limit < expert_used) {
    return PlanError::LeaseCapacity;
  }
  const std::uint64_t capacity = std::min<std::uint64_t>(expert_count, slots);
  std::uint64_t unit = capacity / expert_used;
  if (unit == 0) {
    return PlanError::Capacity;
  }
  unit = std::min<std::uint64_t>(unit, batch_size);
  unit = std::min<std::uint64_t>(unit, lease_limit / expert_used);
  if (unit == 0) {
    return PlanError::LeaseCapacity;
  }
  Plan plan;
  plan.supported = true;
  plan.expert_count = expert_count;
  plan.expert_used = expert_used;
  plan.slots = slots;
  plan.batch_size = batch_size;
  plan.unit_tokens = static_cast<std::uint32_t>(unit);
  return plan;
}

std::uint32_t chunks_for(std::uint64_t tokens, std::uint32_t unit_tokens) {
  if (tokens == 0 || unit_tokens == 0) {
    return 0;
  }
  const std::uint64_t chunks = (tokens + unit_tokens - 1) / unit_tokens;
  return chunks > UINT32_MAX ? UINT32_MAX : static_cast<std::uint32_t>(chunks);
}

}  // namespace ame::prefill
