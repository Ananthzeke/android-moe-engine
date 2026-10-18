#include "run_supervisor.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string_view>
#include <thread>
#include <vector>

namespace ame::runtime {
namespace {

std::string escape(std::string_view text) {
  std::string output;
  for (const char character : text) {
    switch (character) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        output += character;
        break;
    }
  }
  return output;
}

std::string now_iso() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds).count();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&time, &utc);
  char buffer[32] = {};
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &utc);
  char suffix[8] = {};
  std::snprintf(suffix, sizeof(suffix), ".%03lldZ", static_cast<long long>(millis));
  return std::string(buffer) + suffix;
}

bool atomic_write(const std::filesystem::path& path, const std::string& contents) {
  const std::filesystem::path temporary =
      path.string() + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      return false;
    }
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.flush();
    if (!stream.good()) {
      std::filesystem::remove(temporary);
      return false;
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary);
    return false;
  }
  return true;
}

std::string started_json(const RunOptions& options, const std::string& started_at) {
  std::string output = "{\n  \"schema_version\": 1,\n";
  output += "  \"tag\": \"" + escape(options.tag) + "\",\n";
  output += "  \"runner\": \"" + escape(options.runner) + "\",\n";
  output += "  \"arguments\": [";
  for (std::size_t index = 0; index < options.arguments.size(); ++index) {
    if (index != 0) {
      output += ", ";
    }
    output += "\"" + escape(options.arguments[index]) + "\"";
  }
  output += "],\n";
  output += "  \"started_at\": \"" + started_at + "\",\n";
  output += "  \"timeout_seconds\": " + std::to_string(options.timeout_seconds) + ",\n";
  output += "  \"stdout\": \"stdout.log\",\n  \"stderr\": \"stderr.log\"\n}\n";
  return output;
}

}  // namespace

Result<RunResult, RunError> supervise(const RunOptions& options) {
  if (options.runner.empty() || options.run_dir.empty()) {
    return RunError::IoError;
  }
  if (::access(options.runner.c_str(), X_OK) != 0) {
    return RunError::RunnerMissing;
  }
  std::error_code error;
  std::filesystem::create_directories(options.run_dir, error);
  if (error) {
    return RunError::IoError;
  }
  if (std::filesystem::exists(options.run_dir / "started.json")) {
    return RunError::RecordExists;
  }
  const auto stdout_path = options.run_dir / "stdout.log";
  const auto stderr_path = options.run_dir / "stderr.log";
  const int stdout_fd = ::open(stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  const int stderr_fd = ::open(stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (stdout_fd < 0 || stderr_fd < 0) {
    if (stdout_fd >= 0) {
      ::close(stdout_fd);
    }
    if (stderr_fd >= 0) {
      ::close(stderr_fd);
    }
    return RunError::IoError;
  }

  const std::string started_at = now_iso();
  if (!atomic_write(options.run_dir / "started.json", started_json(options, started_at))) {
    ::close(stdout_fd);
    ::close(stderr_fd);
    return RunError::IoError;
  }

  const auto started = std::chrono::steady_clock::now();
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(stdout_fd);
    ::close(stderr_fd);
    return RunError::SpawnFailed;
  }
  if (pid == 0) {
    ::dup2(stdout_fd, STDOUT_FILENO);
    ::dup2(stderr_fd, STDERR_FILENO);
    ::close(stdout_fd);
    ::close(stderr_fd);
    std::vector<char*> argv;
    argv.reserve(options.arguments.size() + 2);
    argv.push_back(const_cast<char*>(options.runner.c_str()));
    for (const auto& argument : options.arguments) {
      argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    ::execvp(options.runner.c_str(), argv.data());
    _exit(127);
  }
  ::close(stdout_fd);
  ::close(stderr_fd);

  int status = 0;
  bool timed_out = false;
  bool reaped = false;
  while (true) {
    const pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      reaped = true;
      break;
    }
    if (result < 0 && errno == ECHILD) {
      reaped = true;
      break;
    }
    if (options.timeout_seconds != 0 && std::chrono::steady_clock::now() - started >=
                                            std::chrono::seconds(options.timeout_seconds)) {
      timed_out = true;
      ::kill(pid, SIGTERM);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < deadline) {
        if (::waitpid(pid, &status, WNOHANG) == pid) {
          reaped = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      if (!reaped) {
        ::kill(pid, SIGKILL);
        (void)::waitpid(pid, &status, 0);
        reaped = true;
      }
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!reaped) {
    (void)::waitpid(pid, &status, 0);
  }

  const auto ended = std::chrono::steady_clock::now();
  RunResult result;
  result.started_at = started_at;
  result.ended_at = now_iso();
  result.duration_seconds = std::chrono::duration<double>(ended - started).count();
  result.timed_out = timed_out;
  if (timed_out) {
    result.status = "timeout";
    if (WIFSIGNALED(status)) {
      result.signal = WTERMSIG(status);
    }
  } else if (WIFEXITED(status)) {
    result.status = "exited";
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.status = "signaled";
    result.signal = WTERMSIG(status);
  } else {
    result.status = "unknown";
  }

  if (!atomic_write(options.run_dir / "result.json", result_json(result))) {
    return RunError::IoError;
  }
  return result;
}

std::string result_json(const RunResult& result) {
  std::string output = "{\n  \"schema_version\": 1,\n";
  output += "  \"status\": \"" + escape(result.status) + "\",\n";
  output += "  \"exit_code\": " + std::to_string(result.exit_code) + ",\n";
  output += "  \"signal\": " + std::to_string(result.signal) + ",\n";
  output += std::string("  \"timed_out\": ") + (result.timed_out ? "true" : "false") + ",\n";
  char duration[64] = {};
  std::snprintf(duration, sizeof(duration), "%.6f", result.duration_seconds);
  output += "  \"duration_seconds\": " + std::string(duration) + ",\n";
  output += "  \"started_at\": \"" + result.started_at + "\",\n";
  output += "  \"ended_at\": \"" + result.ended_at + "\",\n";
  output += "  \"stdout\": \"stdout.log\",\n  \"stderr\": \"stderr.log\"\n}\n";
  return output;
}

}  // namespace ame::runtime