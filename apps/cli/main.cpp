#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

#include "probe.hpp"

namespace {

constexpr std::string_view kVersion = "ame 0.1.0";

void print_help() {
  std::cout << "Usage:\n"
               "  ame probe [--output PATH] [--data-dir DIR] [--root DIR]\n"
               "  ame --help\n"
               "  ame --version\n"
               "\n"
               "probe performs a read-only observation of memory, CPU, storage, OS, thermal,\n"
               "and power data. It does not read serials or application data. Missing\n"
               "observations are null with a reason. --root DIR resolves /proc and /sys\n"
               "beneath DIR for fixture testing; ordinary runs use /.\n";
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
