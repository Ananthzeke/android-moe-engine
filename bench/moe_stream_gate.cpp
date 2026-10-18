// End-to-end test of the patched llama.cpp MUL_MAT_ID materializer hook.
//
// A full resident expert tensor is the reference. The candidate installs the
// project materializer against a manifest and file, then runs the same ggml op.
// Bitwise equality is required.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "moe_stream.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

namespace {

constexpr int kEmbd = 16;
constexpr int kFf = 32;
constexpr int kExperts = 8;
constexpr int kTopK = 4;
constexpr std::uint64_t kExpertBytes = kEmbd * kFf * sizeof(float);

float value(int expert, int index) {
  return static_cast<float>((expert * 131 + index * 17) % 1009) / 1009.0F;
}

void write_file(const std::filesystem::path& path, const std::vector<float>& data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
}

std::vector<float> run(ggml_context* context, ggml_tensor* weights, ggml_tensor* input,
                       ggml_tensor* ids) {
  ggml_tensor* result = ggml_mul_mat_id(context, weights, input, ids);
  ggml_cgraph* graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, result);
  const ggml_status status = ggml_graph_compute_with_ctx(context, graph, 4);
  if (status != GGML_STATUS_SUCCESS) {
    std::cerr << "compute failed\n";
    std::exit(1);
  }
  std::vector<float> output(ggml_nbytes(result) / sizeof(float));
  std::memcpy(output.data(), result->data, ggml_nbytes(result));
  return output;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string workdir = ".";
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--workdir" && index + 1 < argc) {
      workdir = argv[++index];
    }
  }
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

  const std::int32_t selected[kTopK] = {1, 3, 5, 7};
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
      ggml_new_tensor_2d(reference_context, GGML_TYPE_I32, kTopK, 1);
  std::memcpy(reference_ids->data, selected, sizeof(selected));
  const std::vector<float> reference =
      run(reference_context, reference_weights, reference_input, reference_ids);
  ggml_free(reference_context);

  // Candidate: install, then the kernel must stage and remap.
  if (!ame::backend::install_moe_stream(file.string(), manifest.string(), kTopK,
                                        1024U * 1024U)) {
    std::cerr << "install failed: " << ame::backend::moe_stream_stats().error << '\n';
    return 1;
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
      ggml_new_tensor_2d(candidate_context, GGML_TYPE_I32, kTopK, 1);
  std::memcpy(candidate_ids->data, selected, sizeof(selected));
  const std::vector<float> candidate =
      run(candidate_context, candidate_weights, candidate_input, candidate_ids);
  const auto stats = ame::backend::moe_stream_stats();
  ggml_free(candidate_context);
  ame::backend::uninstall_moe_stream();

  std::size_t mismatched = 0;
  double max_difference = 0.0;
  for (std::size_t index = 0; index < std::min(reference.size(), candidate.size()); ++index) {
    if (std::memcmp(&reference[index], &candidate[index], sizeof(float)) != 0) {
      ++mismatched;
      max_difference = std::max<double>(
          max_difference, std::abs(static_cast<double>(reference[index]) - candidate[index]));
    }
  }
  const bool identical = reference.size() == candidate.size() && mismatched == 0;
  std::cout << "{\n"
            << "  \"identical\": " << (identical ? "true" : "false") << ",\n"
            << "  \"reference_values\": " << reference.size() << ",\n"
            << "  \"materialize_calls\": " << stats.materialize_calls << ",\n"
            << "  \"tensor_switches\": " << stats.tensor_switches << ",\n"
            << "  \"reads\": " << stats.reads << ",\n"
            << "  \"read_bytes\": " << stats.read_bytes << ",\n"
            << "  \"arena_bytes\": " << stats.arena_bytes << ",\n"
            << "  \"slots\": " << stats.slots << ",\n"
            << "  \"mismatched\": " << mismatched << ",\n"
            << "  \"max_difference\": " << max_difference << ",\n"
            << "  \"reference_first\": [" << reference[0] << ", " << reference[1] << ", "
            << reference[2] << ", " << reference[3] << "],\n"
            << "  \"candidate_first\": [" << candidate[0] << ", " << candidate[1] << ", "
            << candidate[2] << ", " << candidate[3] << "]\n"
            << "}\n";
  return identical && stats.materialize_calls == 1 && stats.reads == kTopK ? 0 : 1;
}