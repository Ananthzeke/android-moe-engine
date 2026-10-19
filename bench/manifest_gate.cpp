#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "moe_stream.hpp"

int main(int argc, char* argv[]) {
  std::string model;
  std::string manifest;
  std::uint32_t slots = 4;
  std::uint64_t arena_mib = 4;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--model" && index + 1 < argc) {
      model = argv[++index];
    } else if (argument == "--manifest" && index + 1 < argc) {
      manifest = argv[++index];
    } else if (argument == "--slots" && index + 1 < argc) {
      slots = static_cast<std::uint32_t>(std::strtoul(argv[++index], nullptr, 10));
    } else if (argument == "--arena-mib" && index + 1 < argc) {
      arena_mib = std::strtoull(argv[++index], nullptr, 10);
    } else {
      std::cerr << "usage: manifest_gate --model PATH --manifest PATH [--slots N] [--arena-mib N]\n";
      return 2;
    }
  }
  if (model.empty() || manifest.empty()) {
    std::cerr << "usage: manifest_gate --model PATH --manifest PATH\n";
    return 2;
  }
  if (!ame::backend::install_moe_stream(model, manifest, slots, arena_mib * 1024ULL * 1024ULL)) {
    const auto stats = ame::backend::moe_stream_stats();
    std::cout << "{\"installed\": false, \"error\": \"" << stats.error << "\"}\n";
    return 1;
  }
  const auto stats = ame::backend::moe_stream_stats();
  std::cout << "{\"installed\": true, \"manifest_version\": " << stats.manifest_version
            << ", \"identity_checked\": " << (stats.manifest_identity_checked ? "true" : "false")
            << ", \"layout\": \"" << stats.manifest_layout_hash << "\"}\n";
  ame::backend::uninstall_moe_stream();
  return 0;
}
