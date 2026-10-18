#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "result.hpp"

namespace ame::runtime {

struct RunOptions {
  std::string runner;
  std::filesystem::path run_dir;
  std::uint32_t timeout_seconds{600};
  std::string tag;
  std::vector<std::string> arguments;
};

struct RunResult {
  std::string status;  // exited, timeout, signaled
  int exit_code{-1};
  int signal{0};
  bool timed_out{false};
  double duration_seconds{0.0};
  std::string started_at;
  std::string ended_at;
};

enum class RunError {
  IoError,
  RunnerMissing,
  RecordExists,
  SpawnFailed,
};

// Supervise one backend runner process. The supervisor owns lifecycle records, log
// capture, timeout enforcement, and outcome classification; it does not run a model.
[[nodiscard]] Result<RunResult, RunError> supervise(const RunOptions& options);
[[nodiscard]] std::string result_json(const RunResult& result);

}  // namespace ame::runtime
