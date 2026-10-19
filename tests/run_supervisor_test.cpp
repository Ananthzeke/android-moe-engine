#include "run_supervisor.hpp"

#include <unistd.h>

#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {
int failures = 0;
void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Run supervisor test failure: " << message << '\n';
    ++failures;
  }
}
}  // namespace

int main() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("ame_run_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  ame::runtime::RunOptions success;
  success.runner = "/bin/sh";
  success.run_dir = root / "success";
  success.timeout_seconds = 5;
  success.tag = "unit";
  success.arguments = {"-c", "exit 0"};
  const auto ok_result = ame::runtime::supervise(success);
  expect(ok_result.ok(), "successful runner completes");
  if (ok_result.ok()) {
    expect(ok_result.value().status == "exited" && ok_result.value().exit_code == 0,
           "success status");
    expect(std::filesystem::exists(success.run_dir / "started.json"), "started record");
    expect(std::filesystem::exists(success.run_dir / "result.json"), "result record");
    const auto json = ame::runtime::result_json(ok_result.value());
    expect(json.find("\"status\": \"exited\"") != std::string::npos, "result json");
  }

  ame::runtime::RunOptions timeout;
  timeout.runner = "/bin/sh";
  timeout.run_dir = root / "timeout";
  timeout.timeout_seconds = 1;
  timeout.arguments = {"-c", "sleep 5"};
  const auto timeout_result = ame::runtime::supervise(timeout);
  expect(timeout_result.ok(), "timeout runner returns a record");
  if (timeout_result.ok()) {
    expect(timeout_result.value().timed_out, "timeout flagged");
    expect(timeout_result.value().status == "timeout", "timeout status");
    expect(timeout_result.value().duration_seconds < 4.0, "timeout enforcement");
  }

  ame::runtime::RunOptions aborting;
  aborting.runner = "/bin/sh";
  aborting.run_dir = root / "abort";
  aborting.timeout_seconds = 5;
  aborting.arguments = {"-c", std::string("kill -ABRT $") + "$"};
  const auto abort_result = ame::runtime::supervise(aborting);
  expect(abort_result.ok(), "aborting runner returns a record");
  if (abort_result.ok()) {
    expect(abort_result.value().status == "signaled", "signal classified");
    expect(abort_result.value().signal == SIGABRT, "SIGABRT recorded");
  }

  ame::runtime::RunOptions missing;
  missing.runner = "/definitely/not/a/runner";
  missing.run_dir = root / "missing";
  const auto missing_result = ame::runtime::supervise(missing);
  expect(!missing_result.ok() && missing_result.error() == ame::runtime::RunError::RunnerMissing,
         "missing runner rejected");

  ame::runtime::RunOptions exists;
  exists.runner = "/bin/sh";
  exists.run_dir = root / "exists";
  exists.arguments = {"-c", "exit 0"};
  std::filesystem::create_directories(exists.run_dir);
  std::ofstream(exists.run_dir / "started.json") << "{}";
  const auto exists_result = ame::runtime::supervise(exists);
  expect(!exists_result.ok() && exists_result.error() == ame::runtime::RunError::RecordExists,
         "existing record rejected");

  std::filesystem::remove_all(root);
  return failures == 0 ? 0 : 1;
}