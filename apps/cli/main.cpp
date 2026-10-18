#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

#include "compare.hpp"
#include "probe.hpp"
#include "run_supervisor.hpp"

namespace {

constexpr std::string_view kVersion = "ame 0.1.0";

void print_help() {
  std::cout << "Usage:\n"
               "  ame probe [--output PATH] [--data-dir DIR] [--root DIR]\n"
               "  ame compare --left PATH --right PATH [--output PATH]\n"
               "  ame run --runner PATH --run-dir DIR [--timeout N] [--tag NAME] -- ARGS...\n"
               "  ame --help\n"
               "  ame --version\n"
               "\n"
               "probe performs a read-only observation of memory, CPU, storage, OS, thermal,\n"
               "and power data. It does not read serials or application data. Missing\n"
               "observations are null with a reason. --root DIR resolves /proc and /sys\n"
               "beneath DIR for fixture testing; ordinary runs use /.\n"
               "\n"
               "compare checks two harness measurement JSON files for status, artifact,\n"
               "prompt/output token, and finish-reason equality. It never runs a model.\n"
               "Exit 0 means all compared fields match; 1 means a difference; 3 means a\n"
               "file or JSON failure.\n"
               "\n"
               "run supervises one backend runner process, writes started/result records and\n"
               "stdout/stderr logs, and enforces the timeout. It does not execute a model\n"
               "itself. Exit 0 means the runner exited 0; 1 means a non-zero exit, signal, or\n"
               "timeout; 2 means usage; 3 means a record or spawn failure.\n";
}

int usage_error(std::string_view message) {
  std::cerr << message << "\nUse ame --help.\n";
  return 2;
}

struct ProbeArguments {
  std::string output;
  std::string data_dir;
  std::string root = "/";
};

bool write_atomic(const std::string& path, const std::string& contents, std::string& error) {
  const std::string temporary = path + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
  const int descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (descriptor < 0) {
    error = std::strerror(errno);
    return false;
  }
  std::size_t written = 0;
  while (written < contents.size()) {
    const ssize_t count = ::write(descriptor, contents.data() + written, contents.size() - written);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = std::strerror(errno);
      ::close(descriptor);
      ::unlink(temporary.c_str());
      return false;
    }
    written += static_cast<std::size_t>(count);
  }
  if (::fsync(descriptor) != 0) {
    error = std::strerror(errno);
    ::close(descriptor);
    ::unlink(temporary.c_str());
    return false;
  }
  if (::close(descriptor) != 0) {
    error = std::strerror(errno);
    ::unlink(temporary.c_str());
    return false;
  }
  if (::rename(temporary.c_str(), path.c_str()) != 0) {
    error = std::strerror(errno);
    ::unlink(temporary.c_str());
    return false;
  }
  const std::size_t slash = path.find_last_of('/');
  const std::string directory = slash == std::string::npos ? std::string(".")
                                : slash == 0               ? std::string("/")
                                                           : path.substr(0, slash);
  const int directory_fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory_fd >= 0) {
    (void)::fsync(directory_fd);
    ::close(directory_fd);
  }
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
    print_help();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--version") {
    std::cout << kVersion << '\n';
    return 0;
  }
  if (argc >= 2 && std::string_view(argv[1]) == "run") {
    ame::runtime::RunOptions options;
    std::string tag;
    bool after_separator = false;
    for (int index = 2; index < argc; ++index) {
      const std::string_view argument = argv[index];
      if (after_separator) {
        options.arguments.emplace_back(argument);
        continue;
      }
      if (argument == "--") {
        after_separator = true;
        continue;
      }
      if (argument != "--runner" && argument != "--run-dir" && argument != "--timeout" &&
          argument != "--tag") {
        return usage_error("Unsupported run argument: " + std::string(argument));
      }
      if (index + 1 >= argc) {
        return usage_error("Missing value for " + std::string(argument));
      }
      const std::string value = argv[++index];
      if (argument == "--runner") {
        options.runner = value;
      } else if (argument == "--run-dir") {
        options.run_dir = value;
      } else if (argument == "--tag") {
        options.tag = value;
      } else {
        std::uint32_t timeout = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), timeout);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
          return usage_error("Invalid --timeout value");
        }
        options.timeout_seconds = timeout;
      }
    }
    if (options.runner.empty() || options.run_dir.empty()) {
      return usage_error("run requires --runner and --run-dir");
    }
    const auto result = ame::runtime::supervise(options);
    if (!result.ok()) {
      switch (result.error()) {
        case ame::runtime::RunError::RunnerMissing:
          std::cerr << "run failed: runner is missing or not executable\n";
          return 2;
        case ame::runtime::RunError::RecordExists:
          std::cerr << "run failed: started.json already exists in the run directory\n";
          return 3;
        case ame::runtime::RunError::SpawnFailed:
          std::cerr << "run failed: cannot spawn runner\n";
          return 3;
        case ame::runtime::RunError::IoError:
          std::cerr << "run failed: cannot create run records or logs\n";
          return 3;
      }
    }
    std::cout << ame::runtime::result_json(result.value());
    if (!std::cout.good()) {
      std::cerr << "Failed to write run result.\n";
      return 3;
    }
    return result.value().status == "exited" && result.value().exit_code == 0 ? 0 : 1;
  }
  if (argc >= 2 && std::string_view(argv[1]) == "compare") {
    std::string left;
    std::string right;
    std::string output;
    for (int index = 2; index < argc; ++index) {
      const std::string_view option = argv[index];
      if (option != "--left" && option != "--right" && option != "--output") {
        return usage_error("Unsupported compare argument: " + std::string(option));
      }
      if (index + 1 >= argc) {
        return usage_error("Missing value for " + std::string(option));
      }
      const std::string value = argv[++index];
      if (value.empty()) {
        return usage_error("Empty value for " + std::string(option));
      }
      if (option == "--left") {
        left = value;
      } else if (option == "--right") {
        right = value;
      } else {
        output = value;
      }
    }
    if (left.empty() || right.empty()) {
      return usage_error("compare requires --left and --right");
    }
    const auto comparison = ame::compare::compare_files(left, right);
    if (!comparison.ok()) {
      std::cerr << "compare failed: ";
      switch (comparison.error()) {
        case ame::compare::CompareError::IoError:
          std::cerr << "cannot read a measurement file";
          break;
        case ame::compare::CompareError::FileTooLarge:
          std::cerr << "measurement file is too large";
          break;
        case ame::compare::CompareError::InvalidJson:
          std::cerr << "invalid measurement JSON";
          break;
        case ame::compare::CompareError::NotAnObject:
          std::cerr << "measurement JSON is not an object";
          break;
      }
      std::cerr << '\n';
      return 3;
    }
    const auto& value = comparison.value();
    const std::string json = ame::compare::to_json(value);
    if (output.empty()) {
      std::cout << json;
      if (!std::cout.good()) {
        std::cerr << "Failed to write compare output.\n";
        return 3;
      }
    } else {
      std::string error;
      if (!write_atomic(output, json, error)) {
        std::cerr << "Failed to write " << output << ": " << error << '\n';
        return 3;
      }
    }
    const bool same = value.same_status && value.same_artifact && value.same_prompt_tokens &&
                      value.same_output_tokens && value.same_finish_reason;
    return same ? 0 : 1;
  }
  if (argc >= 2 && std::string_view(argv[1]) == "probe") {
    ProbeArguments arguments;
    for (int index = 2; index < argc; ++index) {
      const std::string_view option = argv[index];
      const bool has_value = index + 1 < argc;
      if (option == "--output" || option == "--data-dir" || option == "--root") {
        if (!has_value) {
          return usage_error("Missing value for " + std::string(option));
        }
        const std::string value = argv[++index];
        if (value.empty()) {
          return usage_error("Empty value for " + std::string(option));
        }
        if (option == "--output") {
          arguments.output = value;
        } else if (option == "--data-dir") {
          arguments.data_dir = value;
        } else {
          arguments.root = value;
        }
      } else {
        return usage_error("Unsupported probe argument: " + std::string(option));
      }
    }
    ame::probe::ObservationPaths paths;
    paths.root = arguments.root;
#if defined(__ANDROID__)
    paths.storage_path = arguments.data_dir.empty() ? "/data" : arguments.data_dir;
#else
    paths.storage_path = arguments.data_dir.empty() ? "/" : arguments.data_dir;
#endif
    const ame::probe::Snapshot snapshot = ame::probe::observe(paths);
    const std::string json = ame::probe::to_json(snapshot);
    if (arguments.output.empty()) {
      std::cout << json;
      if (!std::cout.good()) {
        std::cerr << "Failed to write probe output to stdout.\n";
        return 3;
      }
      return 0;
    }
    std::string error;
    if (!write_atomic(arguments.output, json, error)) {
      std::cerr << "Failed to write " << arguments.output << ": " << error << '\n';
      return 3;
    }
    return 0;
  }
  return usage_error("Unsupported arguments.");
}