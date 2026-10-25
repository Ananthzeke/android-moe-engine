#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ame::probe {

// Root-relative observation paths. Tests pass a fixture root; the CLI uses "/".
struct ObservationPaths {
  std::string root{"/"};
  std::string storage_path{};
};

struct Unavailable {
  std::string field;
  std::string reason;
};

struct MemoryKib {
  std::optional<std::uint64_t> total;
  std::optional<std::uint64_t> free;
  std::optional<std::uint64_t> available;
  std::optional<std::uint64_t> buffers;
  std::optional<std::uint64_t> cached;
  std::optional<std::uint64_t> swap_total;
  std::optional<std::uint64_t> swap_free;
};

struct CpuSnapshot {
  std::optional<std::uint64_t> processors;
  std::optional<std::uint64_t> online;
  std::optional<std::uint64_t> implementer;
  std::optional<std::uint64_t> part;
  std::optional<std::uint64_t> variant;
  std::optional<std::uint64_t> revision;
  std::optional<std::uint64_t> max_khz;
  std::optional<std::string> hardware;
  std::optional<std::string> model_name;
  std::optional<std::string> features;
};

struct StorageSnapshot {
  std::string path;
  std::optional<std::uint64_t> total_bytes;
  std::optional<std::uint64_t> available_bytes;
  std::optional<std::uint64_t> free_bytes;
  std::optional<std::uint64_t> fragment_size_bytes;
  // Filesystem type of the mount holding `path`, from the mount table.
  std::optional<std::string> filesystem;
  // Whether the kernel advertises fs-verity for that filesystem type (a feature file under
  // /sys/fs/<type>/features). It says nothing about whether a given file can have verity
  // enabled or whether the shell user may enable it; nothing is enabled or written.
  std::optional<bool> verity_advertised;
};

struct OsSnapshot {
  std::optional<std::string> sysname;
  std::optional<std::string> release;
  std::optional<std::string> version;
  std::optional<std::string> machine;
  std::optional<std::uint64_t> page_size_bytes;
  std::optional<std::uint32_t> android_api_level;
  std::vector<std::pair<std::string, std::string>> properties;
};

struct ProcessSnapshot {
  std::uint64_t pid{0};
  std::uint64_t uid{0};
  std::optional<std::uint64_t> rss_kib;
  std::optional<std::uint64_t> threads;
};

struct ThermalZone {
  std::string name;
  std::optional<std::string> type;
  std::optional<std::int64_t> temp_millidegrees;
};

struct PowerSupply {
  std::string name;
  std::optional<std::string> type;
  std::optional<std::string> status;
  std::optional<std::int64_t> capacity_percent;
};

struct Snapshot {
  std::string observed_at;
  std::string execution_tag;
  MemoryKib memory;
  CpuSnapshot cpu;
  StorageSnapshot storage;
  OsSnapshot os;
  ProcessSnapshot process;
  std::vector<ThermalZone> thermal;
  std::vector<PowerSupply> power;
  std::vector<Unavailable> unavailable;
};

[[nodiscard]] Snapshot observe(const ObservationPaths& paths);
[[nodiscard]] std::string to_json(const Snapshot& snapshot);
[[nodiscard]] std::string_view schema_version() noexcept;

}  // namespace ame::probe
