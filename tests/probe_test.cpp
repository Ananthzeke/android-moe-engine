#include "probe.hpp"

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Probe test failure: " << message << '\n';
    ++failures;
  }
}

void write_file(const std::filesystem::path& path, std::string_view contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  stream.close();
}

bool contains(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

std::filesystem::path build_fixture() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("ame_probe_fixture_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(root);
  write_file(root / "proc/meminfo",
             "MemTotal:        7834636 kB\n"
             "MemFree:          355160 kB\n"
             "MemAvailable:    5335020 kB\n"
             "Buffers:           13352 kB\n"
             "Cached:           1000 kB\n"
             "SwapTotal:        2000 kB\n"
             "SwapFree:         1500 kB\n");
  write_file(root / "proc/cpuinfo",
             "processor\t: 0\n"
             "BogoMIPS\t: 38.40\n"
             "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid\n"
             "CPU implementer\t: 0x41\n"
             "CPU architecture: 8\n"
             "CPU variant\t: 0x4\n"
             "CPU part\t: 0xd03\n"
             "CPU revision\t: 0x0\n"
             "\n"
             "processor\t: 1\n"
             "Features\t: fp asimd aes\n"
             "CPU implementer\t: 0x41\n");
  write_file(root / "sys/devices/system/cpu/online", "0-1\n");
  // Mount table with the fixture root on f2fs, which advertises verity.
  write_file(root / "proc/self/mounts", "rootfs / rootfs ro 0 0\n/dev/block/dm-4 " + root.string() +
                                            " f2fs rw,seclabel 0 0\n");
  write_file(root / "sys/fs/f2fs/features/verity", "");
  write_file(root / "sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "1804800\n");
  write_file(root / "sys/class/thermal/thermal_zone0/type", "cpu-0-0\n");
  write_file(root / "sys/class/thermal/thermal_zone0/temp", "31000\n");
  write_file(root / "sys/class/thermal/thermal_zone1/type", "battery \"x\"\n");
  write_file(root / "sys/class/thermal/thermal_zone1/temp", "42000\n");
  write_file(root / "sys/class/power_supply/battery/type", "Battery\n");
  write_file(root / "sys/class/power_supply/battery/status", "Charging\n");
  write_file(root / "sys/class/power_supply/battery/capacity", "87\n");
  return root;
}

void check_fixture() {
  const std::filesystem::path root = build_fixture();
  ame::probe::ObservationPaths paths;
  paths.root = root.string();
  paths.storage_path = root.string();
  const ame::probe::Snapshot snapshot = ame::probe::observe(paths);

  expect(snapshot.memory.total == 7834636, "MemTotal");
  expect(snapshot.memory.available == 5335020, "MemAvailable");
  expect(snapshot.memory.cached == 1000, "Cached");
  expect(snapshot.memory.swap_total == 2000, "SwapTotal");
  expect(snapshot.memory.swap_free == 1500, "SwapFree");

  expect(snapshot.cpu.processors == 2, "processor count");
  expect(snapshot.cpu.online == 2, "online CPU count");
  expect(snapshot.cpu.implementer == 0x41, "implementer");
  expect(snapshot.cpu.part == 0xd03, "part");
  expect(snapshot.cpu.variant == 0x4, "variant");
  expect(snapshot.cpu.revision == 0x0, "revision");
  expect(snapshot.cpu.max_khz == 1804800, "cpufreq");
  expect(snapshot.cpu.features.has_value() && contains(*snapshot.cpu.features, "asimd"),
         "features");

  expect(snapshot.storage.path == root.string(), "storage path");
  expect(snapshot.storage.total_bytes.has_value() && *snapshot.storage.total_bytes > 0,
         "storage total");
  expect(snapshot.storage.available_bytes.has_value(), "storage available");
  expect(snapshot.storage.filesystem == "f2fs", "filesystem from the longest matching mount");
  expect(snapshot.storage.verity_advertised == true, "verity advertised by the feature file");
  expect(contains(to_json(snapshot), "\"verity_advertised\": true"), "verity in the JSON");

  expect(snapshot.thermal.size() == 2, "thermal zone count");
  expect(snapshot.thermal[0].name == "thermal_zone0", "thermal name");
  expect(snapshot.thermal[0].temp_millidegrees == 31000, "thermal temp");
  expect(snapshot.thermal[1].type.has_value() && *snapshot.thermal[1].type == "battery \"x\"",
         "thermal type");

  expect(snapshot.power.size() == 1, "power supply count");
  expect(snapshot.power[0].capacity_percent == 87, "battery capacity");
  expect(snapshot.power[0].status.has_value() && *snapshot.power[0].status == "Charging",
         "battery status");

  expect(snapshot.os.machine.has_value() && !snapshot.os.machine->empty(), "machine");
  expect(snapshot.os.page_size_bytes.has_value() && *snapshot.os.page_size_bytes > 0, "page size");
  expect(snapshot.process.uid == static_cast<std::uint64_t>(::getuid()), "uid");
  expect(snapshot.execution_tag == "host", "execution tag");

  const std::string json = ame::probe::to_json(snapshot);
  expect(json.starts_with("{\n  \"schema_version\": 1"), "json prefix");
  expect(contains(json, "\"execution_tag\": \"host\""), "json tag");
  expect(contains(json, "\"total\": 7834636"), "json memtotal");
  expect(contains(json, "\"name\": \"thermal_zone1\""), "json thermal name");
  expect(contains(json, "battery \\\"x\\\""), "json escaping");
  expect(json.ends_with("\n"), "json newline");
  std::filesystem::remove_all(root);
}

void check_missing_and_oversized() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("ame_probe_missing_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(root);
  ame::probe::ObservationPaths paths;
  paths.root = root.string();
  const ame::probe::Snapshot missing = ame::probe::observe(paths);
  expect(!missing.memory.total.has_value(), "missing memory is null");
  bool reason_seen = false;
  for (const auto& entry : missing.unavailable) {
    reason_seen = reason_seen || (entry.field == "memory" && entry.reason == "not_present");
  }
  expect(reason_seen, "missing memory reason");

  write_file(root / "proc/meminfo", std::string(128U * 1024U, 'x'));
  const ame::probe::Snapshot oversized = ame::probe::observe(paths);
  expect(!oversized.memory.total.has_value(), "oversized memory is null");
  reason_seen = false;
  for (const auto& entry : oversized.unavailable) {
    reason_seen =
        reason_seen || (entry.field == "memory" && entry.reason == "exceeds_observation_limit");
  }
  expect(reason_seen, "oversized memory reason");
  std::filesystem::remove_all(root);
}

}  // namespace

int main() {
  check_fixture();
  check_missing_and_oversized();
  return failures == 0 ? 0 : 1;
}
