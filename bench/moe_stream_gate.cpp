// End-to-end test of the patched llama.cpp MUL_MAT_ID materializer contract.
//
// A full resident expert tensor is the reference. The candidate installs the
// project materializer against a manifest and file, then runs the same ggml op.
// Successful staging must be bitwise identical. Injected read, capacity, and
// cancellation failures must return an aborted graph without aborting the host
// process or falling back to resident expert tensors.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
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
constexpr std::uint64_t kExpertBytes = kEmbd * kFf * sizeof(float);

float value(int expert, int index) {
  return static_cast<float>((expert * 131 + index * 17) % 1009) / 1009.0F;
}

void write_file(const std::filesystem::path& path, const std::vector<float>& data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
}

struct RunResult {
  ggml_status status{GGML_STATUS_FAILED};
  std::vector<float> output;
};

RunResult run(ggml_context* context, ggml_tensor* weights, ggml_tensor* input,
              ggml_tensor* ids, int threads) {
  ggml_tensor* result = ggml_mul_mat_id(context, weights, input, ids);
  ggml_cgraph* graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, result);
  RunResult outcome;
  outcome.status = ggml_graph_compute_with_ctx(context, graph, threads);
  if (outcome.status == GGML_STATUS_SUCCESS) {
    outcome.output.resize(ggml_nbytes(result) / sizeof(float));
    std::memcpy(outcome.output.data(), result->data, ggml_nbytes(result));
  }
  return outcome;
}

std::string escape(std::string_view text) {
  std::string out;
  for (const char character : text) {
    if (character == '"' || character == '\\') {
      out += '\\';
    }
    out += character;
  }
  return out;
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)::setenv("AME_ALLOW_UNVERIFIED_ARTIFACT", "1", 0);
  std::string workdir = ".";
  std::string fault = "none";
  int threads = 4;
  int slots = 4;
  int selected_count = 4;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--workdir" && index + 1 < argc) {
      workdir = argv[++index];
    } else if (argument == "--fault" && index + 1 < argc) {
      fault = argv[++index];
    } else if (argument == "--threads" && index + 1 < argc) {
      threads = std::atoi(argv[++index]);
    } else if (argument == "--slots" && index + 1 < argc) {
      slots = std::atoi(argv[++index]);
    } else if (argument == "--selected" && index + 1 < argc) {
      selected_count = std::atoi(argv[++index]);
    } else {
      std::cerr << "usage: moe_stream_gate [--workdir DIR] [--threads N] [--slots N] "
                   "[--selected N] [--fault none|read|cancel|capacity]\n";
      return 2;
    }
  }
  if (threads < 1 || slots < 1 || selected_count < 1 || selected_count > kExperts) {
    std::cerr << "invalid test dimensions\n";
    return 2;
  }
  if (fault == "capacity") {
    selected_count = slots + 1;
    if (selected_count > kExperts) {
      std::cerr << "capacity fault needs slots below expert count\n";
      return 2;
    }
  }
  const bool expect_failure = fault != "none";

  std::filesystem::create_directories(workdir);
  const std::filesystem::path file = std::filesystem::path(workdir) / "experts.bin";
  const std::filesystem::path manifest = std::filesystem::path(workdir) / "experts.manifest";
  std::vector<float> full_bytes(static_cast<std::size_t>(kExperts) * kEmbd * kFf);
  for (int expert = 0; expert < kExperts; ++expert) {
    for (int index = 0; index < kEmbd * kFf; ++index) {
      full_bytes[static_cast<std::size_t>(expert) * kEmbd * kFf + index] =
          value(expert, index);
    }
  }
  write_file(file, full_bytes);
  {
    std::ofstream stream(manifest, std::ios::trunc);
    stream << "# ame-expert-manifest 1\n";
    stream << "tensor fake.exps 0 role 0 f32 " << kExperts << ' ' << kExpertBytes << ' '
           << kExpertBytes << " 0 " << (kExpertBytes * kExperts) << '\n';
  }

  std::vector<std::int32_t> selected(static_cast<std::size_t>(selected_count));
  for (int index = 0; index < selected_count; ++index) {
    selected[static_cast<std::size_t>(index)] = index;
  }

  ggml_init_params params{};
  params.mem_size = 32U * 1024U * 1024U;
  params.mem_buffer = nullptr;
  params.no_alloc = false;

  // Reference: no materializer installed.
  ame::backend::uninstall_moe_stream();
  ggml_context* reference_context = ggml_init(params);
  ggml_tensor* reference_weights =
      ggml_new_tensor_3d(reference_context, GGML_TYPE_F32, kEmbd, kFf, kExperts);
  std::memcpy(reference_weights->data, full_bytes.data(),
              full_bytes.size() * sizeof(float));
  ggml_tensor* reference_input =
      ggml_new_tensor_3d(reference_context, GGML_TYPE_F32, kEmbd, 1, 1);
  for (int index = 0; index < kEmbd; ++index) {
    static_cast<float*>(reference_input->data)[index] = value(99, index);
  }
  ggml_tensor* reference_ids =
      ggml_new_tensor_2d(reference_context, GGML_TYPE_I32, selected_count, 1);
  std::memcpy(reference_ids->data, selected.data(),
              selected.size() * sizeof(std::int32_t));
  const RunResult reference =
      run(reference_context, reference_weights, reference_input, reference_ids, threads);
  ggml_free(reference_context);
  if (reference.status != GGML_STATUS_SUCCESS) {
    std::cerr << "reference compute failed\n";
    return 1;
  }

  if (fault == "read") {
    (void)::setenv("AME_IO_FAIL_AFTER_BYTES", "1", 1);
  }
  if (!ame::backend::install_moe_stream(file.string(), manifest.string(),
                                        static_cast<std::uint32_t>(slots),
                                        1024U * 1024U)) {
    std::cerr << "install failed: " << ame::backend::moe_stream_stats().error << '\n';
    return 1;
  }
  if (fault == "cancel") {
    ame::backend::request_moe_stream_cancel();
  }
  if (fault == "eof") {
    std::filesystem::resize_file(file, 1);
  }
  ggml_context* candidate_context = ggml_init(params);
  ggml_tensor* candidate_weights =
      ggml_new_tensor_3d(candidate_context, GGML_TYPE_F32, kEmbd, kFf, kExperts);
  ggml_set_name(candidate_weights, "fake.exps");
  ggml_tensor* candidate_input =
      ggml_new_tensor_3d(candidate_context, GGML_TYPE_F32, kEmbd, 1, 1);
  for (int index = 0; index < kEmbd; ++index) {
    static_cast<float*>(candidate_input->data)[index] = value(99, index);
  }
  ggml_tensor* candidate_ids =
      ggml_new_tensor_2d(candidate_context, GGML_TYPE_I32, selected_count, 1);
  std::memcpy(candidate_ids->data, selected.data(),
              selected.size() * sizeof(std::int32_t));
  const RunResult candidate =
      run(candidate_context, candidate_weights, candidate_input, candidate_ids, threads);
  const auto stats = ame::backend::moe_stream_stats();
  ggml_free(candidate_context);
  ame::backend::uninstall_moe_stream();
  if (fault == "read") {
    (void)::unsetenv("AME_IO_FAIL_AFTER_BYTES");
  }

  std::size_t mismatched = 0;
  double max_difference = 0.0;
  const std::size_t compared =
      std::min(reference.output.size(), candidate.output.size());
  for (std::size_t index = 0; index < compared; ++index) {
    if (std::memcmp(&reference.output[index], &candidate.output[index], sizeof(float)) != 0) {
      ++mismatched;
      max_difference = std::max<double>(
          max_difference,
          std::abs(static_cast<double>(reference.output[index]) - candidate.output[index]));
    }
  }
  const bool identical = candidate.status == GGML_STATUS_SUCCESS &&
                         reference.output.size() == candidate.output.size() &&
                         mismatched == 0;
  bool passed = false;
  if (!expect_failure) {
    passed = identical && stats.materialize_calls == 1 &&
             stats.reads == static_cast<std::uint64_t>(selected_count);
  } else if (fault == "cancel") {
    passed = candidate.status != GGML_STATUS_SUCCESS && stats.cancelled &&
             !stats.failed;
  } else {
    passed = candidate.status != GGML_STATUS_SUCCESS && stats.failed &&
             !stats.cancelled && !stats.error.empty() && !identical;
  }
  std::cout << "{\n"
            << "  \"fault\": \"" << escape(fault) << "\",\n"
            << "  \"threads\": " << threads << ",\n"
            << "  \"slots\": " << stats.slots << ",\n"
            << "  \"selected\": " << selected_count << ",\n"
            << "  \"candidate_status\": " << static_cast<int>(candidate.status) << ",\n"
            << "  \"identical\": " << (identical ? "true" : "false") << ",\n"
            << "  \"failed\": " << (stats.failed ? "true" : "false") << ",\n"
            << "  \"cancelled\": " << (stats.cancelled ? "true" : "false") << ",\n"
            << "  \"error\": \"" << escape(stats.error) << "\",\n"
            << "  \"materialize_calls\": " << stats.materialize_calls << ",\n"
            << "  \"reads\": " << stats.reads << ",\n"
            << "  \"read_bytes\": " << stats.read_bytes << ",\n"
            << "  \"arena_bytes\": " << stats.arena_bytes << ",\n"
            << "  \"mismatched\": " << mismatched << ",\n"
            << "  \"max_difference\": " << max_difference << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";
  return passed ? 0 : 1;
}