#pragma once

#include <cstdint>
#include <span>
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
struct ConsumerBound {
  std::uint32_t expert_count{0};
  std::uint32_t max_selected{0};
  std::uint32_t slots{0};
  std::uint32_t lease_limit{0};
};

// The contract plan takes every routed consumer's source-derived cardinality and
// selection bound. The admitted token unit must satisfy every consumer pool and the
// fixed lease limit; no manifest scalar can enlarge it.
[[nodiscard]] Result<Plan, PlanError> make_contract_plan(std::span<const ConsumerBound> consumers,
                                                         std::uint32_t batch_size);

[[nodiscard]] Result<Plan, PlanError> make_plan(std::uint32_t expert_count,
                                                std::uint32_t expert_used, std::uint32_t slots,
                                                std::uint32_t batch_size,
                                                std::uint32_t lease_limit);

[[nodiscard]] std::uint32_t chunks_for(std::uint64_t tokens, std::uint32_t unit_tokens);

}  // namespace ame::prefill