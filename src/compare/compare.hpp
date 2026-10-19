#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "result.hpp"

namespace ame::compare {

struct Metrics {
  std::optional<double> ttft_seconds;
  std::optional<double> decode_tokens_per_second;
  std::optional<std::uint64_t> peak_rss_kib;
};

struct Comparison {
  bool same_status{false};
  bool same_artifact{false};
  bool same_file_bytes{false};
  bool same_settings{false};
  bool same_prompt_tokens{false};
  bool same_output_tokens{false};
  bool same_finish_reason{false};
  Metrics a;
  Metrics b;
  std::vector<std::string> differences;
};

enum class CompareError {
  IoError,
  FileTooLarge,
  InvalidJson,
  NotAnObject,
  InvalidSchema,
  InvalidMeasurement,
};

// Compare two harness measurement JSON files. Required fields and integer token IDs
// are validated; malformed measurements are errors, not "different" runs. The
// comparison does not execute a model.
[[nodiscard]] Result<Comparison, CompareError> compare_files(const std::string& left_path,
                                                             const std::string& right_path);
[[nodiscard]] std::string to_json(const Comparison& comparison);

}  // namespace ame::compare