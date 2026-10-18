#include "probe.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

namespace ame::probe {
namespace {

constexpr std::size_t kFileLimit = 64U * 1024U;
constexpr std::size_t kZoneLimit = 32U;
constexpr std::size_t kSupplyLimit = 8U;

enum class ReadError { NotPresent, Unreadable, TooLarge };

std::string under_root(const std::string& root, std::string_view relative) {
  if (root.empty() || root == "/") {
    return "/" + std::string(relative);
  }
  if (root.back() == '/') {
    return root + std::string(relative);
  }
  return root + "/" + std::string(relative);
}

std::variant<std::string, ReadError> read_bounded(const std::string& path, std::size_t limit) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return errno == ENOENT ? ReadError::NotPresent : ReadError::Unreadable;
  }
  std::string contents;
  char buffer[4096];
  for (;;) {
    const ssize_t count = ::read(descriptor, buffer, sizeof(buffer));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      ::close(descriptor);
      return ReadError::Unreadable;
    }
    if (count == 0) {
      break;
    }
    const auto chunk = static_cast<std::size_t>(count);
    if (contents.size() > limit || chunk > limit - contents.size()) {
      ::close(descriptor);
      return ReadError::TooLarge;
    }
    contents.append(buffer, chunk);
  }
  ::close(descriptor);
  return contents;
}

std::string_view trim(std::string_view text) {
  const auto blank = [](char character) {
    return character == ' ' || character == '\t' || character == '\r' || character == '\n';
  };
  while (!text.empty() && blank(text.front())) {
    text.remove_prefix(1);
  }
  while (!text.empty() && blank(text.back())) {
    text.remove_suffix(1);
  }
  return text;
}

std::optional<std::uint64_t> parse_leading_number(std::string_view text) {
  text = trim(text);
  if (text.empty()) {
    return std::nullopt;
  }
  std::uint64_t value = 0;
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
    base = 16;
  }
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (result.ec != std::errc{}) {
    return std::nullopt;
  }
  return value;
}

std::optional<std::uint64_t> parse_exact_number(std::string_view text) {
  text = trim(text);
  if (text.empty()) {
    return std::nullopt;
  }
  std::uint64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

std::optional<std::int64_t> parse_signed_number(std::string_view text) {
  text = trim(text);
  if (text.empty()) {
    return std::nullopt;
  }
  std::int64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (result.ec != std::errc{}) {
    return std::nullopt;
  }
  return value;
}

struct LineField {
  std::string_view key;
  std::string_view value;
};

std::optional<LineField> find_line_field(std::string_view text, std::string_view key) {
  std::size_t position = 0;
  while (position < text.size()) {
    std::size_t end = text.find('\n', position);
    if (end == std::string_view::npos) {
      end = text.size();
    }
    const std::string_view line = text.substr(position, end - position);
    const std::size_t colon = line.find(':');
    if (colon != std::string_view::npos && trim(line.substr(0, colon)) == key) {
      return LineField{trim(line.substr(0, colon)), trim(line.substr(colon + 1))};
    }
    position = end + 1;
  }
  return std::nullopt;
}

std::optional<std::uint64_t> meminfo_value(std::string_view text, std::string_view key) {
  const auto field = find_line_field(text, key);
  if (!field.has_value()) {
    return std::nullopt;
  }
  return parse_leading_number(field->value);
}

std::uint64_t count_lines_beginning(std::string_view text, std::string_view prefix) {
  std::uint64_t count = 0;
  std::size_t position = 0;
  while (position < text.size()) {
    std::size_t end = text.find('\n', position);
    if (end == std::string_view::npos) {
      end = text.size();
    }
    if (trim(text.substr(position, end - position)).starts_with(prefix)) {
      ++count;
    }
    position = end + 1;
  }
  return count;
}

std::optional<std::uint64_t> parse_cpu_list(std::string_view text) {
  text = trim(text);
  if (text.empty()) {
    return std::nullopt;
  }
  std::uint64_t total = 0;
  while (!text.empty()) {
    const std::size_t comma = text.find(',');
    std::string_view item = text.substr(0, comma);
    if (item.empty()) {
      return std::nullopt;
    }
    std::uint64_t first = 0;
    std::uint64_t last = 0;
    const std::size_t dash = item.find('-');
    if (dash == std::string_view::npos) {
      const auto single = parse_exact_number(item);
      if (!single.has_value()) {
        return std::nullopt;
      }
      first = *single;
      last = *single;
    } else {
      const auto begin = parse_exact_number(item.substr(0, dash));
      const auto finish = parse_exact_number(item.substr(dash + 1));
      if (!begin.has_value() || !finish.has_value() || *begin > *finish) {
        return std::nullopt;
      }
      first = *begin;
      last = *finish;
    }
    const std::uint64_t span = last - first + 1;
    if (span > 4096 || total > 4096 - span) {
      return std::nullopt;
    }
    total += span;
    if (comma == std::string_view::npos) {
      break;
    }
    text.remove_prefix(comma + 1);
  }
  return total;
}

#if defined(__ANDROID__)
std::optional<std::string> property(const char* name) {
  char value[PROP_VALUE_MAX] = {};
  const int length = __system_property_get(name, value);
  if (length > 0) {
    return std::string(value, static_cast<std::size_t>(length));
  }
  return std::nullopt;
}
#endif

std::string execution_tag() {
#if defined(__ANDROID__)
  const auto kernel_qemu = property("ro.kernel.qemu");
  const auto boot_qemu = property("ro.boot.qemu");
  if ((kernel_qemu.has_value() && *kernel_qemu == "1") ||
      (boot_qemu.has_value() && *boot_qemu == "1")) {
    return "emulator";
  }
  return "physical_target";
#else
  return "host";
#endif
}

std::string utc_now() {
  const std::time_t now = std::time(nullptr);
  std::tm parts{};
  if (::gmtime_r(&now, &parts) == nullptr) {
    return "";
  }
  char buffer[32] = {};
  const std::size_t written = std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &parts);
  return std::string(buffer, written);
}

void note(std::vector<Unavailable>& list, std::string field, std::string reason) {
  list.push_back(Unavailable{std::move(field), std::move(reason)});
}

std::string read_error_reason(ReadError error) {
  switch (error) {
    case ReadError::NotPresent:
      return "not_present";
    case ReadError::Unreadable:
      return "unreadable";
    case ReadError::TooLarge:
      return "exceeds_observation_limit";
  }
  return "unknown";
}

std::optional<std::string> read_text(const std::string& path, std::vector<Unavailable>& list,
                                     std::string field) {
  auto result = read_bounded(path, kFileLimit);
  if (const auto* error = std::get_if<ReadError>(&result)) {
    note(list, std::move(field), read_error_reason(*error));
    return std::nullopt;
  }
  return std::get<std::string>(std::move(result));
}

void observe_memory(const ObservationPaths& paths, MemoryKib& memory,
                    std::vector<Unavailable>& list) {
  const auto text = read_text(under_root(paths.root, "proc/meminfo"), list, "memory");
  if (!text.has_value()) {
    return;
  }
  memory.total = meminfo_value(*text, "MemTotal");
  memory.free = meminfo_value(*text, "MemFree");
  memory.available = meminfo_value(*text, "MemAvailable");
  memory.buffers = meminfo_value(*text, "Buffers");
  memory.cached = meminfo_value(*text, "Cached");
  memory.swap_total = meminfo_value(*text, "SwapTotal");
  memory.swap_free = meminfo_value(*text, "SwapFree");
}

void observe_cpu(const ObservationPaths& paths, CpuSnapshot& cpu, std::vector<Unavailable>& list) {
  const auto text = read_text(under_root(paths.root, "proc/cpuinfo"), list, "cpu.cpuinfo");
  if (text.has_value()) {
    cpu.processors = count_lines_beginning(*text, "processor");
    if (const auto field = find_line_field(*text, "CPU implementer"); field.has_value()) {
      cpu.implementer = parse_leading_number(field->value);
    }
    if (const auto field = find_line_field(*text, "CPU part"); field.has_value()) {
      cpu.part = parse_leading_number(field->value);
    }
    if (const auto field = find_line_field(*text, "CPU variant"); field.has_value()) {
      cpu.variant = parse_leading_number(field->value);
    }
    if (const auto field = find_line_field(*text, "CPU revision"); field.has_value()) {
      cpu.revision = parse_leading_number(field->value);
    }
    if (const auto field = find_line_field(*text, "Features"); field.has_value()) {
      cpu.features = std::string(field->value);
    }
    if (const auto field = find_line_field(*text, "Hardware"); field.has_value()) {
      cpu.hardware = std::string(field->value);
    }
    if (const auto field = find_line_field(*text, "model name"); field.has_value()) {
      cpu.model_name = std::string(field->value);
    }
  }
  if (const auto online =
          read_text(under_root(paths.root, "sys/devices/system/cpu/online"), list, "cpu.online");
      online.has_value()) {
    cpu.online = parse_cpu_list(*online);
  }
  if (const auto max_frequency =
          read_text(under_root(paths.root, "sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),
                    list, "cpu.max_khz");
      max_frequency.has_value()) {
    cpu.max_khz = parse_exact_number(*max_frequency);
  }
}

void observe_storage(const ObservationPaths& paths, StorageSnapshot& storage,
                     std::vector<Unavailable>& list) {
  storage.path = paths.storage_path.empty() ? std::string("/") : paths.storage_path;
  struct statvfs status {};
  if (::statvfs(storage.path.c_str(), &status) != 0) {
    note(list, "storage", std::string("statvfs_failed:") + std::strerror(errno));
    return;
  }
  const std::uint64_t fragment = status.f_frsize != 0 ? status.f_frsize : status.f_bsize;
  storage.fragment_size_bytes = fragment;
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  if (fragment != 0 && static_cast<std::uint64_t>(status.f_blocks) <= kMax / fragment) {
    storage.total_bytes = static_cast<std::uint64_t>(status.f_blocks) * fragment;
  } else {
    note(list, "storage.total_bytes", "arithmetic_overflow");
  }
  if (fragment != 0 && static_cast<std::uint64_t>(status.f_bavail) <= UINT64_MAX / fragment) {
    storage.available_bytes = static_cast<std::uint64_t>(status.f_bavail) * fragment;
  } else {
    note(list, "storage.available_bytes", "arithmetic_overflow");
  }
  if (fragment != 0 && static_cast<std::uint64_t>(status.f_bfree) <= UINT64_MAX / fragment) {
    storage.free_bytes = static_cast<std::uint64_t>(status.f_bfree) * fragment;
  } else {
    note(list, "storage.free_bytes", "arithmetic_overflow");
  }
}

void observe_os(OsSnapshot& os, std::vector<Unavailable>& list) {
  struct utsname info {};
  if (::uname(&info) == 0) {
    os.sysname = std::string(info.sysname);
    os.release = std::string(info.release);
    os.version = std::string(info.version);
    os.machine = std::string(info.machine);
  } else {
    note(list, "os.uname", std::string("uname_failed:") + std::strerror(errno));
  }
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size > 0) {
    os.page_size_bytes = static_cast<std::uint64_t>(page_size);
  } else {
    note(list, "os.page_size_bytes", "sysconf_unavailable");
  }
#if defined(__ANDROID__)
  static constexpr const char* kPropertyNames[] = {
      "ro.product.model",         "ro.product.device",    "ro.board.platform",
      "ro.build.version.release", "ro.build.version.sdk", "ro.product.cpu.abilist",
      "ro.product.cpu.abilist64", "ro.kernel.qemu",       "ro.boot.qemu",
  };
  for (const char* name : kPropertyNames) {
    const auto value = property(name);
    if (value.has_value()) {
      os.properties.emplace_back(name, *value);
    }
  }
  const auto sdk = property("ro.build.version.sdk");
  if (sdk.has_value()) {
    const auto parsed = parse_exact_number(*sdk);
    if (parsed.has_value() && *parsed <= std::numeric_limits<std::uint32_t>::max()) {
      os.android_api_level = static_cast<std::uint32_t>(*parsed);
    }
  }
#else
  note(list, "os.android_properties", "not_android");
#endif
}

void observe_process(ProcessSnapshot& process, std::vector<Unavailable>& list) {
  process.pid = static_cast<std::uint64_t>(::getpid());
  process.uid = static_cast<std::uint64_t>(::getuid());
  const auto text = read_text("/proc/self/status", list, "process.status");
  if (!text.has_value()) {
    return;
  }
  if (const auto field = find_line_field(*text, "VmRSS"); field.has_value()) {
    process.rss_kib = parse_leading_number(field->value);
  }
  if (const auto field = find_line_field(*text, "Threads"); field.has_value()) {
    process.threads = parse_exact_number(field->value);
  }
}

void observe_thermal(const ObservationPaths& paths, std::vector<ThermalZone>& zones,
                     std::vector<Unavailable>& list) {
  const std::string directory = under_root(paths.root, "sys/class/thermal");
  std::error_code error;
  std::filesystem::directory_iterator iterator(directory, error);
  if (error) {
    note(list, "thermal", "directory_unavailable");
    return;
  }
  std::vector<std::string> names;
  const std::filesystem::directory_iterator end;
  for (; !error && iterator != end; iterator.increment(error)) {
    const std::string name = iterator->path().filename().string();
    if (name.starts_with("thermal_zone")) {
      names.push_back(name);
    }
  }
  std::sort(names.begin(), names.end());
  if (names.empty()) {
    note(list, "thermal", "no_thermal_zones");
    return;
  }
  const std::size_t limit = std::min(names.size(), kZoneLimit);
  for (std::size_t index = 0; index < limit; ++index) {
    ThermalZone zone;
    zone.name = names[index];
    const auto type =
        read_text(directory + "/" + zone.name + "/type", list, "thermal." + zone.name + ".type");
    if (type.has_value()) {
      zone.type = std::string(trim(*type));
    }
    const auto temperature =
        read_text(directory + "/" + zone.name + "/temp", list, "thermal." + zone.name + ".temp");
    if (temperature.has_value()) {
      zone.temp_millidegrees = parse_signed_number(*temperature);
    }
    zones.push_back(std::move(zone));
  }
  if (names.size() > limit) {
    note(list, "thermal", "zone_count_truncated");
  }
}

void observe_power(const ObservationPaths& paths, std::vector<PowerSupply>& supplies,
                   std::vector<Unavailable>& list) {
  const std::string directory = under_root(paths.root, "sys/class/power_supply");
  std::error_code error;
  std::filesystem::directory_iterator iterator(directory, error);
  if (error) {
    note(list, "power_supply", "directory_unavailable");
    return;
  }
  std::vector<std::string> names;
  const std::filesystem::directory_iterator end;
  for (; !error && iterator != end; iterator.increment(error)) {
    names.push_back(iterator->path().filename().string());
  }
  std::sort(names.begin(), names.end());
  if (names.empty()) {
    note(list, "power_supply", "no_supplies");
    return;
  }
  const std::size_t limit = std::min(names.size(), kSupplyLimit);
  for (std::size_t index = 0; index < limit; ++index) {
    PowerSupply supply;
    supply.name = names[index];
    const auto type = read_text(directory + "/" + supply.name + "/type", list,
                                "power_supply." + supply.name + ".type");
    if (type.has_value()) {
      supply.type = std::string(trim(*type));
    }
    const auto status = read_text(directory + "/" + supply.name + "/status", list,
                                  "power_supply." + supply.name + ".status");
    if (status.has_value()) {
      supply.status = std::string(trim(*status));
    }
    const auto capacity = read_text(directory + "/" + supply.name + "/capacity", list,
                                    "power_supply." + supply.name + ".capacity");
    if (capacity.has_value()) {
      supply.capacity_percent = parse_signed_number(*capacity);
    }
    supplies.push_back(std::move(supply));
  }
  if (names.size() > limit) {
    note(list, "power_supply", "supply_count_truncated");
  }
}

class JsonWriter {
 public:
  void begin_object() { open('{'); }
  void end_object() { close('}'); }
  void begin_array() { open('['); }
  void end_array() { close(']'); }
  void key(std::string_view text) {
    separator();
    write_string(text);
    out_ += ": ";
    after_key_ = true;
  }
  void null_value() {
    element();
    out_ += "null";
  }
  void string_value(std::string_view text) {
    element();
    write_string(text);
  }
  void unsigned_value(std::uint64_t value) {
    element();
    char buffer[32] = {};
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    out_.append(buffer, result.ptr);
  }
  void signed_value(std::int64_t value) {
    element();
    char buffer[32] = {};
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    out_.append(buffer, result.ptr);
  }
  void bool_value(bool value) {
    element();
    out_ += value ? "true" : "false";
  }
  [[nodiscard]] const std::string& text() const { return out_; }

 private:
  void open(char bracket) {
    element();
    out_ += bracket;
    ++depth_;
    first_.push_back(true);
    after_key_ = false;
  }
  void close(char bracket) {
    --depth_;
    const bool was_empty = first_.back();
    first_.pop_back();
    if (!was_empty) {
      newline();
    }
    out_ += bracket;
    after_key_ = false;
  }
  void element() {
    if (!after_key_) {
      separator();
    }
    after_key_ = false;
  }
  void separator() {
    if (first_.empty()) {
      return;
    }
    if (first_.back()) {
      first_.back() = false;
      newline();
    } else {
      out_ += ',';
      newline();
    }
  }
  void newline() {
    out_ += '\n';
    out_.append(depth_ * 2U, ' ');
  }
  void write_string(std::string_view text) {
    static constexpr char kHex[] = "0123456789abcdef";
    out_ += '"';
    for (const char character : text) {
      const auto byte = static_cast<unsigned char>(character);
      if (character == '"' || character == '\\') {
        out_ += '\\';
        out_ += character;
      } else if (byte < 0x20) {
        out_ += "\\u00";
        out_ += kHex[byte >> 4U];
        out_ += kHex[byte & 0x0FU];
      } else {
        out_ += character;
      }
    }
    out_ += '"';
  }
  std::string out_;
  std::vector<bool> first_;
  std::size_t depth_ = 0;
  bool after_key_ = false;
};

void write_optional_string(JsonWriter& json, const std::optional<std::string>& value) {
  if (value.has_value()) {
    json.string_value(*value);
  } else {
    json.null_value();
  }
}

void write_optional_unsigned(JsonWriter& json, const std::optional<std::uint64_t>& value) {
  if (value.has_value()) {
    json.unsigned_value(*value);
  } else {
    json.null_value();
  }
}

void write_optional_signed(JsonWriter& json, const std::optional<std::int64_t>& value) {
  if (value.has_value()) {
    json.signed_value(*value);
  } else {
    json.null_value();
  }
}

}  // namespace

std::string_view schema_version() noexcept { return "1"; }

Snapshot observe(const ObservationPaths& paths) {
  Snapshot snapshot;
  snapshot.observed_at = utc_now();
  snapshot.execution_tag = execution_tag();
  observe_memory(paths, snapshot.memory, snapshot.unavailable);
  observe_cpu(paths, snapshot.cpu, snapshot.unavailable);
  observe_storage(paths, snapshot.storage, snapshot.unavailable);
  observe_os(snapshot.os, snapshot.unavailable);
  observe_process(snapshot.process, snapshot.unavailable);
  observe_thermal(paths, snapshot.thermal, snapshot.unavailable);
  observe_power(paths, snapshot.power, snapshot.unavailable);
  return snapshot;
}

std::string to_json(const Snapshot& snapshot) {
  JsonWriter json;
  json.begin_object();
  json.key("schema_version");
  json.unsigned_value(1);
  json.key("observed_at");
  json.string_value(snapshot.observed_at);
  json.key("execution_tag");
  json.string_value(snapshot.execution_tag);

  json.key("os");
  json.begin_object();
  json.key("sysname");
  write_optional_string(json, snapshot.os.sysname);
  json.key("release");
  write_optional_string(json, snapshot.os.release);
  json.key("version");
  write_optional_string(json, snapshot.os.version);
  json.key("machine");
  write_optional_string(json, snapshot.os.machine);
  json.key("page_size_bytes");
  write_optional_unsigned(json, snapshot.os.page_size_bytes);
  json.key("android_api_level");
  if (snapshot.os.android_api_level.has_value()) {
    json.unsigned_value(*snapshot.os.android_api_level);
  } else {
    json.null_value();
  }
  json.key("properties");
  json.begin_object();
  for (const auto& [name, value] : snapshot.os.properties) {
    json.key(name);
    json.string_value(value);
  }
  json.end_object();
  json.end_object();

  json.key("cpu");
  json.begin_object();
  json.key("processors");
  write_optional_unsigned(json, snapshot.cpu.processors);
  json.key("online");
  write_optional_unsigned(json, snapshot.cpu.online);
  json.key("implementer");
  write_optional_unsigned(json, snapshot.cpu.implementer);
  json.key("part");
  write_optional_unsigned(json, snapshot.cpu.part);
  json.key("variant");
  write_optional_unsigned(json, snapshot.cpu.variant);
  json.key("revision");
  write_optional_unsigned(json, snapshot.cpu.revision);
  json.key("max_khz");
  write_optional_unsigned(json, snapshot.cpu.max_khz);
  json.key("hardware");
  write_optional_string(json, snapshot.cpu.hardware);
  json.key("model_name");
  write_optional_string(json, snapshot.cpu.model_name);
  json.key("features");
  write_optional_string(json, snapshot.cpu.features);
  json.end_object();

  json.key("memory_kib");
  json.begin_object();
  json.key("total");
  write_optional_unsigned(json, snapshot.memory.total);
  json.key("free");
  write_optional_unsigned(json, snapshot.memory.free);
  json.key("available");
  write_optional_unsigned(json, snapshot.memory.available);
  json.key("buffers");
  write_optional_unsigned(json, snapshot.memory.buffers);
  json.key("cached");
  write_optional_unsigned(json, snapshot.memory.cached);
  json.key("swap_total");
  write_optional_unsigned(json, snapshot.memory.swap_total);
  json.key("swap_free");
  write_optional_unsigned(json, snapshot.memory.swap_free);
  json.end_object();

  json.key("storage");
  json.begin_object();
  json.key("path");
  json.string_value(snapshot.storage.path);
  json.key("total_bytes");
  write_optional_unsigned(json, snapshot.storage.total_bytes);
  json.key("available_bytes");
  write_optional_unsigned(json, snapshot.storage.available_bytes);
  json.key("free_bytes");
  write_optional_unsigned(json, snapshot.storage.free_bytes);
  json.key("fragment_size_bytes");
  write_optional_unsigned(json, snapshot.storage.fragment_size_bytes);
  json.end_object();

  json.key("process");
  json.begin_object();
  json.key("pid");
  json.unsigned_value(snapshot.process.pid);
  json.key("uid");
  json.unsigned_value(snapshot.process.uid);
  json.key("rss_kib");
  write_optional_unsigned(json, snapshot.process.rss_kib);
  json.key("threads");
  write_optional_unsigned(json, snapshot.process.threads);
  json.end_object();

  json.key("thermal");
  json.begin_array();
  for (const auto& zone : snapshot.thermal) {
    json.begin_object();
    json.key("name");
    json.string_value(zone.name);
    json.key("type");
    write_optional_string(json, zone.type);
    json.key("temp_millidegrees");
    write_optional_signed(json, zone.temp_millidegrees);
    json.end_object();
  }
  json.end_array();

  json.key("power_supply");
  json.begin_array();
  for (const auto& supply : snapshot.power) {
    json.begin_object();
    json.key("name");
    json.string_value(supply.name);
    json.key("type");
    write_optional_string(json, supply.type);
    json.key("status");
    write_optional_string(json, supply.status);
    json.key("capacity_percent");
    write_optional_signed(json, supply.capacity_percent);
    json.end_object();
  }
  json.end_array();

  json.key("unavailable");
  json.begin_array();
  for (const auto& entry : snapshot.unavailable) {
    json.begin_object();
    json.key("field");
    json.string_value(entry.field);
    json.key("reason");
    json.string_value(entry.reason);
    json.end_object();
  }
  json.end_array();

  json.end_object();
  return json.text() + "\n";
}

}  // namespace ame::probe
