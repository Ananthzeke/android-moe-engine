// Drives the real Store with a scenario and prints what it did, so tools/replay_routing.py
// can be checked against it. Scenario: "capacity N" then one line of expert ids per
// acquisition. A lease is released just before the next acquisition, as the materializer
// does. Output per step: hits misses evictions then the slot of each distinct key.
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cancellation.hpp"
#include "file_source.hpp"
#include "store.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: store_replay_driver SCENARIO\n";
    return 2;
  }
  std::ifstream scenario(argv[1]);
  std::string line;
  std::uint32_t capacity = 0;
  std::vector<std::vector<std::uint32_t>> steps;
  std::uint32_t max_id = 0;
  while (std::getline(scenario, line)) {
    std::istringstream fields(line);
    std::string head;
    fields >> head;
    if (head == "capacity") {
      fields >> capacity;
      continue;
    }
    std::vector<std::uint32_t> ids;
    if (!head.empty()) {
      ids.push_back(static_cast<std::uint32_t>(std::stoul(head)));
      std::uint32_t id = 0;
      while (fields >> id) ids.push_back(id);
    }
    for (const auto id : ids) max_id = std::max(max_id, id);
    steps.push_back(std::move(ids));
  }
  if (capacity == 0) {
    std::cerr << "scenario needs a capacity\n";
    return 2;
  }
  constexpr std::uint64_t kSlot = 64;
  const auto directory = std::filesystem::temp_directory_path() /
                         ("ame_store_replay_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::create_directories(directory);
  const auto path = directory / "experts.bin";
  {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const std::string block(kSlot, 'x');
    for (std::uint32_t id = 0; id <= max_id; ++id) file.write(block.data(), kSlot);
  }
  auto source = ame::residency::FileSource::open(path.string());
  if (!source.ok()) return 3;
  std::vector<ame::residency::ExpertEntry> entries;
  for (std::uint32_t id = 0; id <= max_id; ++id) {
    entries.push_back({ame::model::ExpertKey{0, id}, ame::model::FileOffset{id * kSlot},
                       ame::model::ByteCount{kSlot}});
  }
  ame::residency::StoreOptions options;
  options.slot_count = capacity;
  auto store = ame::residency::Store::create(std::move(source).value(), entries, options);
  if (!store.ok()) return 3;
  ame::Cancellation stop;
  ame::residency::Lease lease;
  for (const auto& ids : steps) {
    lease = ame::residency::Lease{};
    const auto before = store.value()->stats();
    std::vector<ame::model::ExpertKey> keys;
    for (const auto id : ids) keys.push_back({0, id});
    auto acquired = store.value()->acquire(keys, stop);
    if (!acquired.ok()) {
      std::cout << "refused\n";
      continue;
    }
    lease = std::move(acquired).value();
    const auto after = store.value()->stats();
    std::cout << (after.hits - before.hits) << ' ' << (after.misses - before.misses) << ' '
              << (after.evictions - before.evictions);
    std::vector<std::uint32_t> seen;
    for (const auto id : ids) {
      if (std::find(seen.begin(), seen.end(), id) != seen.end()) continue;
      seen.push_back(id);
      std::cout << ' ' << lease.slot_index({0, id}).value_or(9999);
    }
    std::cout << '\n';
  }
  lease = ame::residency::Lease{};
  std::filesystem::remove_all(directory);
  return 0;
}
