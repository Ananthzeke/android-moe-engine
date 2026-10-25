#include "request_metrics.hpp"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace ame::server;

int failures = 0;
void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Request metrics test failure: " << message << '\n';
    ++failures;
  }
}

void write(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path, std::ios::trunc) << text;
}

void set_system(const std::filesystem::path& root, std::uint64_t rchar, std::uint64_t read_bytes,
                std::uint64_t majflt, std::uint64_t pswpin) {
  write(root / "proc/self/io", "rchar: " + std::to_string(rchar) +
                                   "\nwchar: 7\nsyscr: 3\nsyscw: 1\nread_bytes: " +
                                   std::to_string(read_bytes) + "\nwrite_bytes: 0\n");
  write(root / "proc/self/stat", "4242 (ame (server) x) S 1 4242 4242 0 -1 4194560 100 0 " +
                                     std::to_string(majflt) +
                                     " 0 5 6 0 0 20 0 8 0 100 123456 789 18446744073709551615\n");
  write(root / "proc/vmstat",
        "nr_free_pages 12\npswpin " + std::to_string(pswpin) + "\npswpout 9\n");
}

void set_cpu(const std::filesystem::path& root, int cpu, unsigned max_khz, unsigned cur_khz) {
  const auto base = root / "sys/devices/system/cpu" / ("cpu" + std::to_string(cpu)) / "cpufreq";
  write(base / "cpuinfo_max_freq", std::to_string(max_khz) + "\n");
  write(base / "scaling_cur_freq", std::to_string(cur_khz) + "\n");
}

void check_parsers() {
  const std::string io = "rchar: 100\nread_bytes: 4096\n";
  expect(parse_keyed_colon(io, "rchar").value == 100u, "rchar parsed");
  expect(parse_keyed_colon(io, "read_bytes").value == 4096u, "read_bytes parsed");
  const auto absent = parse_keyed_colon(io, "cancelled_write_bytes");
  expect(!absent.value.has_value() && absent.reason == "field_missing", "missing field has reason");
  const auto bad = parse_keyed_colon("rchar: abc\n", "rchar");
  expect(!bad.value.has_value() && bad.reason == "field_malformed", "malformed field has reason");
  expect(parse_keyed_space("pswpin 9\npswpout 1\n", "pswpin").value == 9u, "vmstat key");
  expect(!parse_keyed_space("pswpout 1\n", "pswpin").value.has_value(), "vmstat key absent");
  expect(parse_majflt("1 (a b) c) S 1 2 3 4 5 6 7 8 77 9").value == 77u,
         "majflt parsed after the last parenthesis");
  expect(!parse_majflt("1 (a) S 1 2").value.has_value(), "short stat line has no majflt");
}

void check_big_cores() {
  const auto mixed = choose_big_cores(
      {{0, 1612800}, {1, 1612800}, {2, 1612800}, {3, 1612800}, {4, 1804800}, {5, 1804800}});
  expect(!mixed.homogeneous && mixed.cpus == std::vector<int>({4, 5}),
         "big cores are those with the highest max frequency");
  const auto same = choose_big_cores({{0, 2000000}, {1, 2000000}});
  expect(same.homogeneous && same.cpus.size() == 2, "homogeneous systems are flagged");
  expect(choose_big_cores({}).cpus.empty(), "no frequency data selects nothing");
}

void check_sampler_and_recorder(const std::filesystem::path& root) {
  for (int cpu = 0; cpu < 4; ++cpu) set_cpu(root, cpu, 1612800, 1000000);
  for (int cpu = 4; cpu < 8; ++cpu) set_cpu(root, cpu, 1804800, 1700000);
  set_system(root, 1000, 4000, 10, 2);
  SystemSampler sampler(root);
  expect(sampler.big_cores() == std::vector<int>({4, 5, 6, 7}), "sampler finds the big cores");
  expect(!sampler.homogeneous(), "heterogeneous cluster detected");
  const Sample first = sampler.sample();
  expect(first.proc.rchar.value == 1000u && first.proc.read_bytes.value == 4000u &&
             first.proc.majflt.value == 10u && first.proc.pswpin.value == 2u,
         "sample reads each counter separately");
  expect(first.big_cores.size() == 4 && first.big_cores[0].khz.value == 1700000u,
         "big-core frequency sampled");

  // Lifetime counters only grow. Each request must report only its own increments.
  ExpertCounters lifetime;
  PhaseRecorder recorder(sampler, [&] { return lifetime; });
  const auto run_request = [&](std::uint64_t io_step, std::uint64_t expert_step, bool fail) {
    std::uint64_t rchar = *sampler.sample().proc.rchar.value;
    try {
      PhaseScope prefill(recorder, "prefill");
      rchar += io_step;
      set_system(root, rchar, 4000 + (rchar - 1000) * 2, 10, 2);
      lifetime.misses += expert_step;
      lifetime.expert_bytes += expert_step * 100;
      if (fail) throw std::runtime_error("injected failure");
    } catch (const std::runtime_error&) {
    }
    return recorder.take();
  };
  const auto success = run_request(500, 3, false);
  const auto failed = run_request(70, 1, true);
  const auto cancelled = run_request(0, 0, true);
  expect(success.size() == 1 && failed.size() == 1 && cancelled.size() == 1,
         "a phase is recorded on success and when the request throws");
  expect(success[0].proc_delta.rchar.value == 500u && failed[0].proc_delta.rchar.value == 70u &&
             cancelled[0].proc_delta.rchar.value == 0u,
         "rchar deltas reset between requests");
  expect(success[0].proc_delta.read_bytes.value == 1000u &&
             failed[0].proc_delta.read_bytes.value == 140u,
         "read_bytes stays a separate counter");
  expect(success[0].expert_delta->misses == 3u && failed[0].expert_delta->misses == 1u &&
             cancelled[0].expert_delta->misses == 0u,
         "expert deltas reset between requests");
  expect(
      success[0].expert_delta->expert_bytes == 300u && failed[0].expert_delta->expert_bytes == 100u,
      "expert byte deltas are per request");
  expect(cancelled[0].proc_delta.rchar.value.has_value(), "zero is a real delta, not unavailable");

  // A counter that moves backwards or cannot be read is null with a reason.
  {
    PhaseScope phase(recorder, "decode");
    set_system(root, 1, 1, 10, 2);
  }
  const auto backwards = recorder.take();
  expect(!backwards[0].proc_delta.rchar.value.has_value() &&
             backwards[0].proc_delta.rchar.reason == "counter_decreased",
         "a decreasing counter is reported unavailable");
  std::filesystem::remove(root / "proc/vmstat");
  const Sample no_vmstat = sampler.sample();
  expect(!no_vmstat.proc.pswpin.value.has_value() && no_vmstat.proc.pswpin.reason == "cannot_open",
         "an unreadable file is null with a reason");
}

void check_missing_frequency_data() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("ame_metrics_empty_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  SystemSampler sampler(root);
  expect(sampler.big_cores().empty() && sampler.discovery_reason() == "cpuinfo_max_freq_unreadable",
         "missing cpufreq data is reported, not guessed");
  const Sample sample = sampler.sample();
  expect(!sample.proc.rchar.value.has_value() && sample.big_cores.empty(),
         "counters are unavailable without a proc tree");
  std::filesystem::remove_all(root);
}
}  // namespace

int main() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("ame_metrics_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(root);
  check_parsers();
  check_big_cores();
  check_sampler_and_recorder(root);
  check_missing_frequency_data();
  std::filesystem::remove_all(root);
  return failures == 0 ? 0 : 1;
}
