#pragma once

#include <cstdint>
#include <string>

#include "result.hpp"

namespace ame::prefill {

struct Plan {
  bool supported{false};
  std::uint32_t expert_count{0};
  std::uint32_t expert_used{0};
  std::uint32_t slots{0};
  std::uint32_t batch_size{0};
  std::uint32_t unit_tokens{0};
  std::string reason;
};

enum class PlanError {
  InvalidExpertCount,
  InvalidExpertUsed,
  InvalidSlots,
  InvalidBatch,
  Capacity,
  LeaseCapacity,
};

// Conservative union bound: min(expert_count, unit_tokens * expert_used) must
// fit both slot capacity and the fixed lease limit. The unit is also capped by
// the caller's batch size.
[[nodiscard]] Result<Plan, PlanError> make_plan(std::uint32_t expert_count,
                                                std::uint32_t expert_used, std::uint32_t slots,
                                                std::uint32_t batch_size,
                                                std::uint32_t lease_limit);

[[nodiscard]] std::uint32_t chunks_for(std::uint64_t tokens, std::uint32_t unit_tokens);

}  // namespace ame::prefill
