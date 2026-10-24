#include "run_supervisor.hpp"

#include <unistd.h>

#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>

namespace {
int failures = 0;
void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Run supervisor test failure: " << message << '\n';
    ++failures;
  }
}

// A process is gone when it has no /proc entry or is a zombie awaiting a reaper.
bool process_stopped(long pid) {
  std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
  std::string line;
  if (!std::getline(stat, line)) {
    return true;
  }
  const auto close = line.rfind(')');
  return close != std::string::npos && close + 2 < line.size() && line[close + 2] == 'Z';
}

long read_pid(const std::filesystem::path& path) {
  std::ifstream stream(path);
  long pid = 0;
  stream >> pid;
  return pid;
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

  // A TERM-ignoring descendant must not outlive a timed-out run.
  ame::runtime::RunOptions descendant;
  descendant.runner = "/bin/sh";
  descendant.run_dir = root / "descendant-timeout";
  descendant.timeout_seconds = 1;
  const auto timeout_pid_file = root / "timeout.pid";
  descendant.arguments = {
      "-c", "(trap '' TERM; exec sleep 30) & echo $! > " + timeout_pid_file.string() + "; wait"};
  const auto descendant_result = ame::runtime::supervise(descendant);
  expect(descendant_result.ok() && descendant_result.value().timed_out,
         "descendant timeout returns a record");
  const long timeout_child = read_pid(timeout_pid_file);
  expect(timeout_child > 0, "descendant pid recorded");
  expect(timeout_child > 0 && process_stopped(timeout_child), "timeout drains descendants");
  if (descendant_result.ok()) {
    expect(descendant_result.value().descendants_killed, "timeout reports killed descendants");
  }

  // A leader that exits while a descendant keeps running leaves nothing behind.
  ame::runtime::RunOptions orphan;
  orphan.runner = "/bin/sh";
  orphan.run_dir = root / "descendant-exit";
  orphan.timeout_seconds = 5;
  const auto exit_pid_file = root / "exit.pid";
  orphan.arguments = {"-c", "(trap '' TERM; exec sleep 30) & echo $! > " + exit_pid_file.string()};
  const auto orphan_result = ame::runtime::supervise(orphan);
  expect(orphan_result.ok() && orphan_result.value().status == "exited" &&
             orphan_result.value().exit_code == 0,
         "leader exit status is preserved");
  const long exit_child = read_pid(exit_pid_file);
  expect(exit_child > 0 && process_stopped(exit_child), "normal exit drains descendants");
  expect(orphan_result.ok() && orphan_result.value().descendants_killed,
         "normal exit reports killed descendants");

  // With SIGCHLD ignored the kernel discards exit status; that is unknown, not success.
  ame::runtime::RunOptions ignored;
  ignored.runner = "/bin/sh";
  ignored.run_dir = root / "ignored-sigchld";
  ignored.timeout_seconds = 5;
  ignored.arguments = {"-c", "exit 3"};
  const auto previous = std::signal(SIGCHLD, SIG_IGN);
  const auto ignored_result = ame::runtime::supervise(ignored);
  std::signal(SIGCHLD, previous);
  expect(ignored_result.ok() && ignored_result.value().status == "unknown",
         "discarded child status is unknown");
  expect(ignored_result.ok() && ignored_result.value().exit_code != 0,
         "discarded child status is never reported as exit zero");

  std::filesystem::remove_all(root);
  return failures == 0 ? 0 : 1;
}