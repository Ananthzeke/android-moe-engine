// Separates sequential file-read time from SHA-256 time on a resident buffer.
// Bounded to an explicit prefix; this is a diagnostic, not an acceptance gate.

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "sha256.hpp"

namespace {
double elapsed(std::chrono::steady_clock::time_point begin,
               std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

std::string cpu_flags() {
  std::ifstream stream("/proc/cpuinfo");
  std::string line;
  while (std::getline(stream, line)) {
    if (line.rfind("flags", 0) == 0 || line.rfind("Features", 0) == 0) {
      return line;
    }
  }
  return "unavailable";
}
}  // namespace

int main(int argc, char* argv[]) {
  std::string path;
  std::uint64_t limit = 1024ULL * 1024ULL * 1024ULL;
  std::uint32_t repeats = 4;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--file" && index + 1 < argc) {
      path = argv[++index];
    } else if (arg == "--bytes" && index + 1 < argc) {
      limit = std::strtoull(argv[++index], nullptr, 10);
    } else if (arg == "--repeats" && index + 1 < argc) {
      repeats = static_cast<std::uint32_t>(std::strtoul(argv[++index], nullptr, 10));
    } else {
      std::cerr << "usage: hash_diagnostic --file PATH [--bytes N] [--repeats N]\n";
      return 2;
    }
  }
  if (path.empty() || limit == 0 || repeats == 0) {
    std::cerr << "missing arguments\n";
    return 2;
  }
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    std::cerr << "cannot open file\n";
    return 1;
  }
  const std::uint64_t file_bytes = static_cast<std::uint64_t>(::lseek(fd, 0, SEEK_END));
  const std::uint64_t bytes = std::min(file_bytes, limit);
  static constexpr std::size_t kBuffer = 64U * 1024U * 1024U;
  std::vector<std::uint8_t> buffer(std::min<std::size_t>(kBuffer, static_cast<std::size_t>(bytes)));
  if (buffer.empty()) {
    std::cerr << "empty file\n";
    ::close(fd);
    return 1;
  }

  std::uint64_t read_total = 0;
  const auto read_begin = std::chrono::steady_clock::now();
  std::uint64_t offset = 0;
  while (offset < bytes) {
    const std::size_t chunk = static_cast<std::size_t>(
        std::min<std::uint64_t>(buffer.size(), bytes - offset));
    const ssize_t count = ::pread(fd, buffer.data(), chunk, static_cast<off_t>(offset));
    if (count <= 0) {
      break;
    }
    read_total += static_cast<std::uint64_t>(count);
    offset += static_cast<std::uint64_t>(count);
  }
  const auto read_end = std::chrono::steady_clock::now();

  std::string resident(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  std::string hash;
  const auto hash_begin = std::chrono::steady_clock::now();
  for (std::uint32_t iteration = 0; iteration < repeats; ++iteration) {
    hash = ame::crypto::sha256_hex(resident);
  }
  const auto hash_end = std::chrono::steady_clock::now();

  ame::crypto::Sha256Result combined;
  std::string error;
  const auto combined_begin = std::chrono::steady_clock::now();
  if (!ame::crypto::sha256_fd(fd, bytes, combined, error)) {
    std::cerr << "combined hash failed: " << error << '\n';
    ::close(fd);
    return 1;
  }
  const auto combined_end = std::chrono::steady_clock::now();

  const double read_seconds = elapsed(read_begin, read_end);
  const double hash_seconds = elapsed(hash_begin, hash_end) / repeats;
  const double combined_seconds = elapsed(combined_begin, combined_end);
  const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  const double resident_mib = static_cast<double>(buffer.size()) / (1024.0 * 1024.0);
  std::cout << "{\n"
            << "  \"file\": \"" << path << "\",\n"
            << "  \"bytes\": " << bytes << ",\n"
            << "  \"read_bytes\": " << read_total << ",\n"
            << "  \"resident_sha256\": \"" << hash << "\",\n"
            << "  \"buffer_bytes\": " << buffer.size() << ",\n"
            << "  \"repeats\": " << repeats << ",\n"
            << "  \"read_seconds\": " << read_seconds << ",\n"
            << "  \"read_mib_per_second\": " << (mib / read_seconds) << ",\n"
            << "  \"resident_sha256_seconds\": " << hash_seconds << ",\n"
            << "  \"resident_sha256_mib_per_second\": " << (resident_mib / hash_seconds) << ",\n"
            << "  \"combined_seconds\": " << combined_seconds << ",\n"
            << "  \"combined_mib_per_second\": " << (mib / combined_seconds) << ",\n"
#ifdef NDEBUG
            << "  \"build\": \"release\",\n"
#else
            << "  \"build\": \"debug\",\n"
#endif
            << "  \"compiler\": \"" << __VERSION__ << "\",\n"
            << "  \"hash_implementation\": \"portable scalar SHA-256\",\n"
            << "  \"cpu\": \"" << cpu_flags() << "\",\n"
            << "  \"cache_condition\": \"uncontrolled OS page cache; bounded prefix\"\n"
            << "}\n";
  ::close(fd);
  return 0;
}