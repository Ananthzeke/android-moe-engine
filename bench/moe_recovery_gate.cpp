// Same-process recovery after a failed or cancelled streamed MUL_MAT_ID graph.
//
// A graph that fails inside the materializer latches an error in the patched backend.
// Without a reset the next graph, even a dense one, aborts. This gate runs: failed MoE
// graph, unrecovered dense graph (documents the latch), begin_moe_stream_request(),
// dense graph, then the same MoE graph again, which must match the resident reference
// bit for bit. It covers a read failure and a cancellation.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "moe_stream.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

namespace {

constexpr int kEmbd = 16;
constexpr int kFf = 32;
constexpr int kExperts = 8;
constexpr int kSelected = 4;
constexpr std::uint64_t kExpertBytes = kEmbd * kFf * sizeof(float);

float value(int expert, int index) {
  return static_cast<float>((expert * 131 + index * 17) % 1009) / 1009.0F;
}

struct Graph {
  ggml_status status{GGML_STATUS_FAILED};
  std::vector<float> output;
};

Graph run_moe(const std::vector<float>& experts, bool named, int threads) {
  ggml_init_params params{};
  params.mem_size = 32U * 1024U * 1024U;
  ggml_context* context = ggml_init(params);
  ggml_tensor* weights = ggml_new_tensor_3d(context, GGML_TYPE_F32, kEmbd, kFf, kExperts);
  std::memcpy(weights->data, experts.data(), experts.size() * sizeof(float));
  if (named) {
    ggml_set_name(weights, "fake.exps");
  }
  ggml_tensor* input = ggml_new_tensor_3d(context, GGML_TYPE_F32, kEmbd, 1, 1);
  for (int index = 0; index < kEmbd; ++index) {
    static_cast<float*>(input->data)[index] = value(99, index);
  }
  ggml_tensor* ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, kSelected, 1);
  for (int index = 0; index < kSelected; ++index) {
    static_cast<std::int32_t*>(ids->data)[index] = index;
  }
  ggml_tensor* result = ggml_mul_mat_id(context, weights, input, ids);
  ggml_cgraph* graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, result);
  Graph outcome;
  outcome.status = ggml_graph_compute_with_ctx(context, graph, threads);
  if (outcome.status == GGML_STATUS_SUCCESS) {
    outcome.output.resize(ggml_nbytes(result) / sizeof(float));
    std::memcpy(outcome.output.data(), result->data, ggml_nbytes(result));
  }
  ggml_free(context);
  return outcome;
}

// A graph with no MUL_MAT_ID: it can only fail through the leftover latch.
Graph run_dense(int threads) {
  ggml_init_params params{};
  params.mem_size = 1U * 1024U * 1024U;
  ggml_context* context = ggml_init(params);
  ggml_tensor* left = ggml_new_tensor_1d(context, GGML_TYPE_F32, 8);
  ggml_tensor* right = ggml_new_tensor_1d(context, GGML_TYPE_F32, 8);
  for (int index = 0; index < 8; ++index) {
    static_cast<float*>(left->data)[index] = static_cast<float>(index);
    static_cast<float*>(right->data)[index] = 1.0F;
  }
  ggml_tensor* sum = ggml_add(context, left, right);
  ggml_cgraph* graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, sum);
  Graph outcome;
  outcome.status = ggml_graph_compute_with_ctx(context, graph, threads);
  if (outcome.status == GGML_STATUS_SUCCESS) {
    outcome.output.assign(static_cast<float*>(sum->data), static_cast<float*>(sum->data) + 8);
  }
  ggml_free(context);
  return outcome;
}

bool dense_correct(const Graph& graph) {
  if (graph.status != GGML_STATUS_SUCCESS || graph.output.size() != 8) return false;
  for (int index = 0; index < 8; ++index) {
    if (graph.output[static_cast<std::size_t>(index)] != static_cast<float>(index) + 1.0F) {
      return false;
    }
  }
  return true;
}

bool identical(const Graph& left, const Graph& right) {
  return left.status == GGML_STATUS_SUCCESS && right.status == GGML_STATUS_SUCCESS &&
         left.output.size() == right.output.size() &&
         std::memcmp(left.output.data(), right.output.data(),
                     left.output.size() * sizeof(float)) == 0;
}

// Lifetime counters only grow. A request's own work is the difference between two
// snapshots, so a failed or cancelled request cannot leak into the next one.
struct Counted {
  std::uint64_t hits{0};
  std::uint64_t misses{0};
  std::uint64_t bytes{0};
};

Counted counted_since(const ame::backend::MoeStreamStats& start) {
  const auto now = ame::backend::moe_stream_stats();
  return {now.hits - start.hits, now.misses - start.misses, now.read_bytes - start.read_bytes};
}

struct Case {
  std::string fault;
  Counted failed_work;
  Counted recovered_work;
  bool poisoned{false};
  bool recovered_dense{false};
  bool recovered_moe_identical{false};
  bool failure_reported{false};
  bool outcome_cleared{false};
};

Case run_case(const std::string& fault, const std::vector<float>& experts, const Graph& reference,
              int threads) {
  Case result;
  result.fault = fault;
  if (fault == "read") {
    (void)::setenv("AME_IO_FAIL_AFTER_BYTES", "1", 1);
    ame::backend::begin_moe_stream_request();  // re-reads the injected-fault environment
  } else {
    ame::backend::request_moe_stream_cancel();
  }
  const auto before_failed = ame::backend::moe_stream_stats();
  const Graph failed = run_moe(experts, true, threads);
  result.failed_work = counted_since(before_failed);
  const auto stats = ame::backend::moe_stream_stats();
  result.failure_reported = failed.status != GGML_STATUS_SUCCESS &&
                            (fault == "read" ? stats.failed && !stats.cancelled
                                             : stats.cancelled && !stats.failed);
  result.poisoned = run_dense(threads).status != GGML_STATUS_SUCCESS;
  (void)::unsetenv("AME_IO_FAIL_AFTER_BYTES");
  ame::backend::begin_moe_stream_request();
  const auto cleared = ame::backend::moe_stream_stats();
  result.outcome_cleared = !cleared.failed && !cleared.cancelled && cleared.error.empty();
  result.recovered_dense = dense_correct(run_dense(threads));
  const auto before_recovered = ame::backend::moe_stream_stats();
  result.recovered_moe_identical = identical(run_moe(experts, true, threads), reference);
  result.recovered_work = counted_since(before_recovered);
  return result;
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)::setenv("AME_ALLOW_UNVERIFIED_ARTIFACT", "1", 0);
  std::string workdir = ".";
  int threads = 4;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--workdir" && index + 1 < argc) {
      workdir = argv[++index];
    } else if (argument == "--threads" && index + 1 < argc) {
      threads = std::atoi(argv[++index]);
    } else {
      std::cerr << "usage: moe_recovery_gate [--workdir DIR] [--threads N]\n";
      return 2;
    }
  }
  std::filesystem::create_directories(workdir);
  const auto file = std::filesystem::path(workdir) / "experts.bin";
  const auto manifest = std::filesystem::path(workdir) / "experts.manifest";
  std::vector<float> experts(static_cast<std::size_t>(kExperts) * kEmbd * kFf);
  for (int expert = 0; expert < kExperts; ++expert) {
    for (int index = 0; index < kEmbd * kFf; ++index) {
      experts[static_cast<std::size_t>(expert) * kEmbd * kFf + static_cast<std::size_t>(index)] =
          value(expert, index);
    }
  }
  {
    std::ofstream stream(file, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(experts.data()),
                 static_cast<std::streamsize>(experts.size() * sizeof(float)));
    std::ofstream list(manifest, std::ios::trunc);
    list << "# ame-expert-manifest 1\n"
         << "tensor fake.exps 0 role 0 f32 " << kExperts << ' ' << kExpertBytes << ' '
         << kExpertBytes << " 0 " << (kExpertBytes * kExperts) << '\n';
  }

  ame::backend::uninstall_moe_stream();
  const Graph reference = run_moe(experts, false, threads);
  if (reference.status != GGML_STATUS_SUCCESS) {
    std::cerr << "reference compute failed\n";
    return 1;
  }
  if (!ame::backend::install_moe_stream(file.string(), manifest.string(), kSelected,
                                        1024U * 1024U)) {
    std::cerr << "install failed\n";
    return 1;
  }
  const Case read_case = run_case("read", experts, reference, threads);
  const Case cancel_case = run_case("cancel", experts, reference, threads);

  // A new installation also starts with a clean latch.
  (void)::setenv("AME_IO_FAIL_AFTER_BYTES", "1", 1);
  ame::backend::begin_moe_stream_request();
  (void)run_moe(experts, true, threads);
  (void)::unsetenv("AME_IO_FAIL_AFTER_BYTES");
  ame::backend::uninstall_moe_stream();
  const bool reinstall_clears = dense_correct(run_dense(threads));

  bool passed = reinstall_clears;
  std::cout << "{\n  \"cases\": [\n";
  const Case* cases[] = {&read_case, &cancel_case};
  for (std::size_t index = 0; index < 2; ++index) {
    const Case& item = *cases[index];
    // Read fault: the failed request reserves four missing slots (counted as misses) but
    // reads no bytes; the recovery request misses and reads all four again. Cancellation:
    // the failed request touched no store; the recovery request finds the four experts
    // already resident from the previous recovery.
    const bool read_fault = item.fault == "read";
    const bool deltas =
        item.failed_work.bytes == 0 && item.failed_work.misses == (read_fault ? kSelected : 0) &&
        item.failed_work.hits == 0 &&
        item.recovered_work.misses == (read_fault ? kSelected : 0) &&
        item.recovered_work.hits == (read_fault ? 0 : kSelected) &&
        item.recovered_work.bytes == (read_fault ? kSelected * kExpertBytes : 0);
    const bool ok = item.failure_reported && item.poisoned && item.outcome_cleared &&
                    item.recovered_dense && item.recovered_moe_identical && deltas;
    passed = passed && ok;
    std::cout << "    {\"fault\": \"" << item.fault
              << "\", \"failure_reported\": " << (item.failure_reported ? "true" : "false")
              << ", \"poisoned_without_reset\": " << (item.poisoned ? "true" : "false")
              << ", \"outcome_cleared\": " << (item.outcome_cleared ? "true" : "false")
              << ", \"dense_after_reset\": " << (item.recovered_dense ? "true" : "false")
              << ", \"moe_identical_after_reset\": "
              << (item.recovered_moe_identical ? "true" : "false")
              << ", \"per_request_deltas\": " << (deltas ? "true" : "false")
              << ", \"recovered_misses\": " << item.recovered_work.misses
              << ", \"recovered_hits\": " << item.recovered_work.hits
              << ", \"recovered_bytes\": " << item.recovered_work.bytes
              << ", \"passed\": " << (ok ? "true" : "false") << "}" << (index == 0 ? "," : "")
              << "\n";
  }
  std::cout << "  ],\n  \"uninstall_clears_latch\": " << (reinstall_clears ? "true" : "false")
            << ",\n  \"passed\": " << (passed ? "true" : "false") << "\n}\n";
  return passed ? 0 : 1;
}
