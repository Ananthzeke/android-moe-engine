#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "artifact.hpp"
#include "moe_stream.hpp"

namespace {
void write_byte(const std::string& path, std::uint64_t offset, unsigned char value) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (fd < 0) {
    return;
  }
  (void)::pwrite(fd, &value, 1, static_cast<off_t>(offset));
  ::close(fd);
}
}  // namespace

int main(int argc, char* argv[]) {
  std::string model;
  std::string manifest;
  std::string sha;
  std::string replace_after_open;
  std::string rename_after_open;
  std::string link_after_open;
  std::uint64_t mutate_offset = 0;
  int mutate_value = -1;
  bool truncate_after_open = false;
  std::uint32_t slots = 4;
  std::uint64_t arena_mib = 64;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    auto next = [&](std::string& value) {
      if (index + 1 >= argc) {
        return false;
      }
      value = argv[++index];
      return true;
    };
    if (arg == "--model") {
      if (!next(model)) return 2;
    } else if (arg == "--manifest") {
      if (!next(manifest)) return 2;
    } else if (arg == "--sha") {
      if (!next(sha)) return 2;
    } else if (arg == "--replace-after-open") {
      if (!next(replace_after_open)) return 2;
    } else if (arg == "--rename-after-open") {
      if (!next(rename_after_open)) return 2;
    } else if (arg == "--link-after-open") {
      if (!next(link_after_open)) return 2;
    } else if (arg == "--truncate-after-open") {
      truncate_after_open = true;
    } else if (arg == "--mutate-after-open") {
      std::string offset;
      std::string value;
      if (!next(offset) || !next(value)) return 2;
      mutate_offset = std::strtoull(offset.c_str(), nullptr, 10);
      mutate_value = static_cast<int>(std::strtol(value.c_str(), nullptr, 0));
    } else if (arg == "--slots") {
      std::string value;
      if (!next(value)) return 2;
      slots = static_cast<std::uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
    } else if (arg == "--arena-mib") {
      std::string value;
      if (!next(value)) return 2;
      arena_mib = std::strtoull(value.c_str(), nullptr, 10);
    } else {
      std::cerr << "unsupported argument: " << arg << '\n';
      return 2;
    }
  }
  if (model.empty() || manifest.empty() || sha.empty()) {
    std::cerr << "usage: artifact_gate --model PATH --manifest PATH --sha HEX\n";
    return 2;
  }

  std::error_code size_error;
  const auto bytes = std::filesystem::file_size(model, size_error);
  if (size_error) {
    std::cerr << "cannot stat model\n";
    return 2;
  }
  std::string error;
  const ame::backend::ExpectedIdentity expected{sha, bytes};
  auto artifact = ame::backend::Artifact::open(model, expected, error);
  if (!artifact.ok()) {
    std::cout << "{\"stage\": \"open\", \"ok\": false, \"error\": \"" << error << "\"}\n";
    return 1;
  }
  if (!replace_after_open.empty()) {
    std::error_code copy_error;
    std::filesystem::rename(replace_after_open, model, copy_error);
    if (copy_error) {
      std::cerr << "replace failed\n";
      return 2;
    }
  }
  if (!rename_after_open.empty()) {
    std::error_code rename_error;
    std::filesystem::rename(model, rename_after_open, rename_error);
    if (rename_error) {
      std::cerr << "rename failed\n";
      return 2;
    }
  }
  if (!link_after_open.empty()) {
    std::error_code link_error;
    std::filesystem::create_hard_link(model, link_after_open, link_error);
    if (link_error) {
      std::cerr << "hard link failed\n";
      return 2;
    }
  }
  if (truncate_after_open) {
    std::error_code truncate_error;
    std::filesystem::resize_file(model, 1, truncate_error);
    if (truncate_error) {
      std::cerr << "truncate failed\n";
      return 2;
    }
  }
  if (mutate_value >= 0) {
    write_byte(model, mutate_offset, static_cast<unsigned char>(mutate_value));
  }

  if (!ame::backend::install_moe_stream(*artifact.value(), manifest, slots,
                                        arena_mib * 1024ULL * 1024ULL)) {
    const auto stats = ame::backend::moe_stream_stats();
    std::cout << "{\"stage\": \"install\", \"ok\": false, \"error\": \"" << stats.error
              << "\"}\n";
    return 1;
  }
  const auto stats = ame::backend::moe_stream_stats();
  std::cout << "{\"stage\": \"install\", \"ok\": true, \"sha\": \"" << stats.verified_artifact_sha256
            << "\", \"hash_seconds\": " << stats.artifact_hash_seconds << "}\n";
  ame::backend::uninstall_moe_stream();
  return 0;
}