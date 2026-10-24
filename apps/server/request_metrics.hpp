#pragma once

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ame::server {

// A counter is a value or a reason it is unavailable. Zero is never used to mean
// "could not be read".
struct Counter {
  std::optional<std::uint64_t> value;
  std::string reason;

  static Counter of(std::uint64_t number) { return {number, {}}; }
  static Counter missing(std::string why) { return {std::nullopt, std::move(why)}; }
};

// Process and system counters kept as distinct quantities. rchar counts bytes the process
// asked read calls for; read_bytes counts bytes the block layer fetched for it. Their ratio
// is not a cache-hit rate and is never computed here.
struct ProcCounters {
  Counter rchar;
  Counter read_bytes;
  Counter majflt;
  Counter pswpin;
};

struct CoreFrequency {
  int cpu{0};
  Counter khz;
};

struct Sample {
  std::chrono::steady_clock::time_point time;
  ProcCounters proc;
  std::vector<CoreFrequency> big_cores;
};

struct ExpertCounters {
  std::uint64_t hits{0};
  std::uint64_t misses{0};
  std::uint64_t expert_reads{0};  // expert slices read, not syscalls
  std::uint64_t expert_bytes{0};  // application bytes for those slices
  std::uint64_t pread_calls{0};
  std::uint64_t pread_bytes{0};
  double read_seconds{0.0};
};

inline ExpertCounters operator-(const ExpertCounters& end, const ExpertCounters& start) {
  const auto sub = [](std::uint64_t a, std::uint64_t b) { return a >= b ? a - b : 0; };
  return {sub(end.hits, start.hits),
          sub(end.misses, start.misses),
          sub(end.expert_reads, start.expert_reads),
          sub(end.expert_bytes, start.expert_bytes),
          sub(end.pread_calls, start.pread_calls),
          sub(end.pread_bytes, start.pread_bytes),
          end.read_seconds >= start.read_seconds ? end.read_seconds - start.read_seconds : 0.0};
}

inline Counter delta(const Counter& start, const Counter& end) {
  if (!start.value.has_value()) return Counter::missing(start.reason);
  if (!end.value.has_value()) return Counter::missing(end.reason);
  if (*end.value < *start.value) return Counter::missing("counter_decreased");
  return Counter::of(*end.value - *start.value);
}

inline ProcCounters delta(const ProcCounters& start, const ProcCounters& end) {
  return {delta(start.rchar, end.rchar), delta(start.read_bytes, end.read_bytes),
          delta(start.majflt, end.majflt), delta(start.pswpin, end.pswpin)};
}

struct PhaseRecord {
  std::string name;
  double seconds{0.0};
  ProcCounters proc_delta;
  std::optional<ExpertCounters> expert_delta;
  std::vector<CoreFrequency> khz_start;
  std::vector<CoreFrequency> khz_end;
};

inline std::optional<std::uint64_t> parse_unsigned(std::string_view text) {
  std::uint64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) return std::nullopt;
  return value;
}

// "name: value" lines, as in /proc/self/io.
inline Counter parse_keyed_colon(std::string_view text, std::string_view key) {
  std::size_t position = 0;
  while (position < text.size()) {
    std::size_t end = text.find('\n', position);
    if (end == std::string_view::npos) end = text.size();
    const std::string_view line = text.substr(position, end - position);
    position = end + 1;
    if (line.size() > key.size() + 1 && line.substr(0, key.size()) == key &&
        line[key.size()] == ':') {
      std::string_view rest = line.substr(key.size() + 1);
      while (!rest.empty() && rest.front() == ' ') rest.remove_prefix(1);
      const auto number = parse_unsigned(rest);
      return number ? Counter::of(*number) : Counter::missing("field_malformed");
    }
  }
  return Counter::missing("field_missing");
}

// "name value" lines, as in /proc/vmstat.
inline Counter parse_keyed_space(std::string_view text, std::string_view key) {
  std::size_t position = 0;
  while (position < text.size()) {
    std::size_t end = text.find('\n', position);
    if (end == std::string_view::npos) end = text.size();
    const std::string_view line = text.substr(position, end - position);
    position = end + 1;
    if (line.size() > key.size() + 1 && line.substr(0, key.size()) == key &&
        line[key.size()] == ' ') {
      const auto number = parse_unsigned(line.substr(key.size() + 1));
      return number ? Counter::of(*number) : Counter::missing("field_malformed");
    }
  }
  return Counter::missing("field_missing");
}

// Field 12 of /proc/<pid>/stat. The command name may contain spaces and parentheses, so
// parse after the last ')'.
inline Counter parse_majflt(std::string_view text) {
  const auto close = text.rfind(')');
  if (close == std::string_view::npos) return Counter::missing("field_malformed");
  std::istringstream rest{std::string(text.substr(close + 1))};
  std::string field;
  for (int index = 3; index <= 12; ++index) {
    if (!(rest >> field)) return Counter::missing("field_missing");
  }
  const auto number = parse_unsigned(field);
  return number ? Counter::of(*number) : Counter::missing("field_malformed");
}

// Big cores are those whose cpuinfo_max_freq equals the highest one. A homogeneous
// system has no distinguishable big cluster; all cores are returned and the caller notes it.
struct BigCores {
  std::vector<int> cpus;
  bool homogeneous{false};
};

inline BigCores choose_big_cores(const std::vector<std::pair<int, std::uint64_t>>& max_khz) {
  BigCores result;
  if (max_khz.empty()) return result;
  std::uint64_t highest = 0;
  std::uint64_t lowest = UINT64_MAX;
  for (const auto& [cpu, khz] : max_khz) {
    (void)cpu;
    highest = std::max(highest, khz);
    lowest = std::min(lowest, khz);
  }
  result.homogeneous = highest == lowest;
  for (const auto& [cpu, khz] : max_khz) {
    if (khz == highest) result.cpus.push_back(cpu);
  }
  return result;
}

inline std::optional<std::string> read_text(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) return std::nullopt;
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

class SystemSampler {
 public:
  // root is the directory holding proc/ and sys/; tests point it at a fixture tree.
  explicit SystemSampler(std::filesystem::path root = "/") : root_(std::move(root)) {
    std::vector<std::pair<int, std::uint64_t>> max_khz;
    for (int cpu = 0; cpu < 256; ++cpu) {
      const auto text = read_text(cpu_file(cpu, "cpuinfo_max_freq"));
      if (!text) {
        if (cpu > 0) break;
        continue;
      }
      std::string_view view(*text);
      while (!view.empty() && (view.back() == '\n' || view.back() == ' ')) view.remove_suffix(1);
      if (const auto number = parse_unsigned(view)) max_khz.emplace_back(cpu, *number);
    }
    const BigCores big = choose_big_cores(max_khz);
    big_cores_ = big.cpus;
    homogeneous_ = big.homogeneous;
    discovery_reason_ = max_khz.empty() ? "cpuinfo_max_freq_unreadable" : "";
  }

  [[nodiscard]] const std::vector<int>& big_cores() const { return big_cores_; }
  [[nodiscard]] bool homogeneous() const { return homogeneous_; }
  [[nodiscard]] const std::string& discovery_reason() const { return discovery_reason_; }

  [[nodiscard]] Sample sample() const {
    Sample result;
    result.time = std::chrono::steady_clock::now();
    const auto io = read_text(root_ / "proc/self/io");
    result.proc.rchar = io ? parse_keyed_colon(*io, "rchar") : Counter::missing("cannot_open");
    result.proc.read_bytes =
        io ? parse_keyed_colon(*io, "read_bytes") : Counter::missing("cannot_open");
    const auto stat = read_text(root_ / "proc/self/stat");
    result.proc.majflt = stat ? parse_majflt(*stat) : Counter::missing("cannot_open");
    const auto vmstat = read_text(root_ / "proc/vmstat");
    result.proc.pswpin =
        vmstat ? parse_keyed_space(*vmstat, "pswpin") : Counter::missing("cannot_open");
    for (const int cpu : big_cores_) {
      CoreFrequency frequency{cpu, Counter::missing("cannot_open")};
      if (const auto text = read_text(cpu_file(cpu, "scaling_cur_freq"))) {
        std::string_view view(*text);
        while (!view.empty() && (view.back() == '\n' || view.back() == ' ')) view.remove_suffix(1);
        const auto number = parse_unsigned(view);
        frequency.khz = number ? Counter::of(*number) : Counter::missing("field_malformed");
      }
      result.big_cores.push_back(std::move(frequency));
    }
    return result;
  }

 private:
  [[nodiscard]] std::filesystem::path cpu_file(int cpu, const char* name) const {
    return root_ / "sys/devices/system/cpu" / ("cpu" + std::to_string(cpu)) / "cpufreq" / name;
  }

  std::filesystem::path root_;
  std::vector<int> big_cores_;
  bool homogeneous_{false};
  std::string discovery_reason_;
};

// Phases run one after another. Each records its own deltas from its own start sample, so
// lifetime counters never leak from one request into the next.
class PhaseRecorder {
 public:
  using ExpertSource = std::function<ExpertCounters()>;

  PhaseRecorder(const SystemSampler& sampler, ExpertSource experts)
      : sampler_(sampler), experts_(std::move(experts)) {}

  void begin(std::string name) {
    end();
    name_ = std::move(name);
    start_ = sampler_.sample();
    if (experts_) start_experts_ = experts_();
    open_ = true;
  }

  void end() {
    if (!open_) return;
    open_ = false;
    const Sample stop = sampler_.sample();
    PhaseRecord record;
    record.name = std::move(name_);
    record.seconds = std::chrono::duration<double>(stop.time - start_.time).count();
    record.proc_delta = delta(start_.proc, stop.proc);
    if (experts_) record.expert_delta = experts_() - start_experts_;
    record.khz_start = std::move(start_.big_cores);
    record.khz_end = stop.big_cores;
    phases_.push_back(std::move(record));
  }

  [[nodiscard]] std::vector<PhaseRecord> take() {
    end();
    return std::exchange(phases_, {});
  }

 private:
  const SystemSampler& sampler_;
  ExpertSource experts_;
  std::string name_;
  Sample start_;
  ExpertCounters start_experts_;
  bool open_{false};
  std::vector<PhaseRecord> phases_;
};

class PhaseScope {
 public:
  PhaseScope(PhaseRecorder& recorder, std::string name) : recorder_(recorder) {
    recorder_.begin(std::move(name));
  }
  PhaseScope(const PhaseScope&) = delete;
  PhaseScope& operator=(const PhaseScope&) = delete;
  ~PhaseScope() { recorder_.end(); }

 private:
  PhaseRecorder& recorder_;
};

}  // namespace ame::server
