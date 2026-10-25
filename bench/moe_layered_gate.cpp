// End-to-end test of the layered expert cache against the patched MUL_MAT_ID.
//
// Two layers, each with separate gate, up and down expert tensors stored role-major like a
// GGUF. The reference keeps every tensor resident. The candidate installs the layered cache
// (one pool per layer, one slot holding the three role slices of an expert) and runs the
// same ops in model order. Staged results must match the reference bit for bit, for float
// and Q4_K experts, for decode and multi-token prefill, across hits and evictions. A fused
// gate_up tensor, a union above the pool, and a read failure must each fail cleanly.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "moe_stream.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

namespace {

constexpr int kLayers = 2;
constexpr int kExperts = 8;
constexpr int kFf = 16;
constexpr const char* kRoles[3] = {"gate", "up", "down"};

struct Format {
  const char* name;
  ggml_type type;
  int embd;
};

float value(int tag, int index) {
  return static_cast<float>(((tag * 131 + index * 17) % 1009) - 504) / 211.0F;
}

struct Model {
  Format format;
  std::uint64_t expert_bytes{0};
  std::vector<std::vector<std::byte>> tensors;  // [layer * 3 + role]: all experts of one tensor
  std::filesystem::path file, manifest;
};

Model build_model(const Format& format, const std::filesystem::path& directory,
                  const std::string& role_name_prefix, bool fused) {
  Model model;
  model.format = format;
  model.expert_bytes = ggml_row_size(format.type, format.embd) * kFf;
  const int rows = kFf * kExperts;
  std::ofstream file(directory / "model.bin", std::ios::binary | std::ios::trunc);
  std::ofstream manifest(directory / "experts.manifest", std::ios::trunc);
  manifest << "# ame-expert-manifest 1\n";
  std::uint64_t offset = 0;
  std::uint32_t tag = 1;
  for (int layer = 0; layer < kLayers; ++layer) {
    for (int role = 0; role < 3; ++role) {
      std::vector<float> floats(static_cast<std::size_t>(format.embd) * rows);
      for (std::size_t index = 0; index < floats.size(); ++index) {
        floats[index] = value(static_cast<int>(tag), static_cast<int>(index));
      }
      std::vector<std::byte> bytes(model.expert_bytes * kExperts);
      if (format.type == GGML_TYPE_F32) {
        std::memcpy(bytes.data(), floats.data(), bytes.size());
      } else {
        ggml_quantize_chunk(format.type, floats.data(), bytes.data(), 0, rows, format.embd,
                            nullptr);
      }
      file.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      const std::string role_name = fused && role == 1
                                        ? "gate_up"
                                        : std::string(kRoles[role]);
      manifest << "tensor blk." << layer << ".ffn_" << role_name << "_exps.weight " << layer
               << " " << role_name_prefix << role_name << "_exps " << static_cast<int>(format.type)
               << " " << format.name << " " << kExperts << " " << model.expert_bytes << " "
               << model.expert_bytes << " " << offset << " " << bytes.size() << "\n";
      offset += bytes.size();
      model.tensors.push_back(std::move(bytes));
      ++tag;
    }
  }
  model.file = directory / "model.bin";
  model.manifest = directory / "experts.manifest";
  return model;
}

struct Output {
  ggml_status status{GGML_STATUS_FAILED};
  std::vector<float> values;
};

Output run_op(const Model& model, int layer, int role, bool named,
              const std::vector<std::int32_t>& ids, int tokens, int top_k, int threads) {
  ggml_init_params params{};
  params.mem_size = 64U * 1024U * 1024U;
  ggml_context* context = ggml_init(params);
  ggml_tensor* weights =
      ggml_new_tensor_3d(context, model.format.type, model.format.embd, kFf, kExperts);
  const auto& bytes = model.tensors[static_cast<std::size_t>(layer * 3 + role)];
  std::memcpy(weights->data, bytes.data(), bytes.size());
  if (named) {
    const std::string name =
        "blk." + std::to_string(layer) + ".ffn_" + kRoles[role] + "_exps.weight";
    ggml_set_name(weights, name.c_str());
  }
  ggml_tensor* input = ggml_new_tensor_3d(context, GGML_TYPE_F32, model.format.embd, 1, tokens);
  for (int token = 0; token < tokens; ++token) {
    for (int index = 0; index < model.format.embd; ++index) {
      static_cast<float*>(input->data)[token * model.format.embd + index] =
          value(99 + token, index);
    }
  }
  ggml_tensor* selected = ggml_new_tensor_2d(context, GGML_TYPE_I32, top_k, tokens);
  std::memcpy(selected->data, ids.data(), ids.size() * sizeof(std::int32_t));
  ggml_tensor* result = ggml_mul_mat_id(context, weights, input, selected);
  ggml_cgraph* graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, result);
  Output output;
  output.status = ggml_graph_compute_with_ctx(context, graph, threads);
  if (output.status == GGML_STATUS_SUCCESS) {
    output.values.resize(ggml_nbytes(result) / sizeof(float));
    std::memcpy(output.values.data(), result->data, ggml_nbytes(result));
  }
  ggml_free(context);
  return output;
}

// FNV-1a over the bytes of every result, to compare runs of this program with each other.
std::uint64_t fold(std::uint64_t hash, const Output& output) {
  for (const float value : output.values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 32; shift += 8) {
      hash = (hash ^ ((bits >> shift) & 0xFFU)) * 0x100000001B3ULL;
    }
  }
  return hash;
}

bool same(const Output& a, const Output& b) {
  return a.status == GGML_STATUS_SUCCESS && b.status == GGML_STATUS_SUCCESS &&
         a.values.size() == b.values.size() &&
         std::memcmp(a.values.data(), b.values.data(), a.values.size() * sizeof(float)) == 0;
}

// A slot holds the three role slices, each starting on a 64-byte boundary, and ends aligned.
std::uint64_t slot_bytes_for(std::uint64_t expert_bytes) {
  const auto align = [](std::uint64_t value) { return (value + 63) / 64 * 64; };
  return align(align(align(expert_bytes) + expert_bytes) + expert_bytes);
}

// What the llama scheduler does between graph segments: ask whether the router's top-k tensor
// is wanted, then hand it over once computed. The tensor name carries the layer.
bool deliver_topk(int layer, const std::vector<std::int32_t>& ids, int tokens, int top_k,
                  bool* asked) {
  ggml_init_params params{};
  params.mem_size = 1U * 1024U * 1024U;
  ggml_context* context = ggml_init(params);
  ggml_tensor* topk = ggml_new_tensor_2d(context, GGML_TYPE_I32, top_k, tokens);
  const std::string name = "ffn_moe_topk-" + std::to_string(layer);
  ggml_set_name(topk, name.c_str());
  std::memcpy(topk->data, ids.data(), ids.size() * sizeof(std::int32_t));
  *asked = ame::backend::moe_eval_callback(topk, true, nullptr);
  const bool delivered = ame::backend::moe_eval_callback(topk, false, nullptr);
  ggml_free(context);
  return delivered;
}

struct Step {
  int tokens;
  std::vector<std::int32_t> ids;  // tokens x top_k
};

// A tiny per-layer LRU that predicts what the store must do, so the gate checks the counters
// as well as the numbers. After a visit the order is: untouched slots, hits, then misses.
struct Predictor {
  std::vector<int> order;  // oldest first
  int slots;
  std::uint64_t hits{0}, misses{0}, evictions{0};

  void visit(const std::vector<std::int32_t>& ids) {
    std::vector<int> unique;
    for (const int id : ids) {
      if (std::find(unique.begin(), unique.end(), id) == unique.end()) unique.push_back(id);
    }
    std::vector<int> hit, miss;
    for (const int id : unique) {
      (std::find(order.begin(), order.end(), id) != order.end() ? hit : miss).push_back(id);
    }
    std::vector<int> untouched;
    for (const int id : order) {
      if (std::find(hit.begin(), hit.end(), id) == hit.end()) untouched.push_back(id);
    }
    const int free_slots = slots - static_cast<int>(order.size());
    const int need = std::max(0, static_cast<int>(miss.size()) - free_slots);
    if (need > static_cast<int>(untouched.size())) return;  // the pool would refuse
    untouched.erase(untouched.begin(), untouched.begin() + need);
    hits += hit.size();
    misses += miss.size();
    evictions += static_cast<std::uint64_t>(need);
    order = untouched;
    order.insert(order.end(), hit.begin(), hit.end());
    order.insert(order.end(), miss.begin(), miss.end());
  }
};

}  // namespace

int main(int argc, char* argv[]) {
  (void)::setenv("AME_ALLOW_UNVERIFIED_ARTIFACT", "1", 0);
  std::string workdir = ".";
  int threads = 4;
  unsigned readers = 1;
  bool overlap = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--workdir" && index + 1 < argc) {
      workdir = argv[++index];
    } else if (argument == "--threads" && index + 1 < argc) {
      threads = std::atoi(argv[++index]);
    } else if (argument == "--overlap") {
      overlap = true;
    } else if (argument == "--readers" && index + 1 < argc) {
      readers = static_cast<unsigned>(std::atoi(argv[++index]));
    } else {
      std::cerr << "usage: moe_layered_gate [--workdir DIR] [--threads N] [--readers N] [--overlap]\n";
      return 2;
    }
  }
  const Format formats[] = {{"f32", GGML_TYPE_F32, 16},
                            {"q4_K", GGML_TYPE_Q4_K, 256},
                            {"q6_K", GGML_TYPE_Q6_K, 256}};
  // Decode steps with overlap, a set that evicts, then a three-token prefill unit.
  const std::vector<Step> steps = {{1, {0, 1, 2, 3}},
                                   {1, {2, 3, 4, 5}},
                                   {1, {0, 1, 6, 7}},
                                   {3, {0, 1, 2, 3, 3, 4, 5, 0, 1, 2, 4, 5}}};
  constexpr int kTopK = 4;
  constexpr int kPoolSlots = 6;

  bool passed = true;
  std::ostringstream report;
  report << "{\n  \"readers\": " << readers << ",\n  \"overlap\": "
         << (overlap ? "true" : "false") << ",\n  \"formats\": [\n";
  bool first_format = true;
  for (const Format& format : formats) {
    const auto directory = std::filesystem::path(workdir) / format.name;
    std::filesystem::create_directories(directory);
    const Model model = build_model(format, directory, "ffn_", false);

    // Reference results with the materializer removed.
    ame::backend::uninstall_moe_stream();
    std::vector<Output> references;
    for (const Step& step : steps) {
      for (int layer = 0; layer < kLayers; ++layer) {
        for (int role = 0; role < 3; ++role) {
          references.push_back(
              run_op(model, layer, role, true, step.ids, step.tokens, kTopK, threads));
        }
      }
    }

    ame::backend::StreamOptions options;
    options.layout = ame::backend::CacheLayout::Layered;
    options.layer_slots = kPoolSlots;
    options.readers = readers;
    options.overlap = overlap;
    if (!ame::backend::install_moe_stream(model.file.string(), model.manifest.string(), 1,
                                          64U * 1024U * 1024U, options)) {
      std::cerr << "layered install failed: " << ame::backend::moe_stream_stats().error << '\n';
      return 1;
    }
    const auto installed = ame::backend::moe_stream_stats();
    std::vector<Predictor> predictors(kLayers, Predictor{{}, kPoolSlots});
    std::size_t compared = 0;
    std::size_t mismatched = 0;
    std::uint64_t digest = 0xCBF29CE484222325ULL;
    bool counters_ok = true;
    bool callbacks_ok = true;
    std::size_t reference_index = 0;
    for (const Step& step : steps) {
      for (int layer = 0; layer < kLayers; ++layer) {
        const auto before = ame::backend::moe_stream_stats();
        Predictor& predictor = predictors[static_cast<std::size_t>(layer)];
        const Predictor mark = predictor;
        predictor.visit(step.ids);
        if (overlap) {
          bool asked = false;
          const bool delivered = deliver_topk(layer, step.ids, step.tokens, kTopK, &asked);
          callbacks_ok = callbacks_ok && asked && delivered;
        }
        for (int role = 0; role < 3; ++role) {
          const Output candidate =
              run_op(model, layer, role, true, step.ids, step.tokens, kTopK, threads);
          ++compared;
          digest = fold(digest, candidate);
          if (!same(references[reference_index++], candidate)) ++mismatched;
        }
        const auto after = ame::backend::moe_stream_stats();
        // One reservation per layer visit, three slices read per missing expert, and three
        // staged operators.
        counters_ok = counters_ok && after.hits - before.hits == predictor.hits - mark.hits &&
                      after.misses - before.misses == predictor.misses - mark.misses &&
                      after.evictions - before.evictions == predictor.evictions - mark.evictions &&
                      after.reads - before.reads == 3 * (predictor.misses - mark.misses) &&
                      after.read_bytes - before.read_bytes ==
                          3 * model.expert_bytes * (predictor.misses - mark.misses) &&
                      after.materialize_calls - before.materialize_calls == 3 &&
                      after.prefetches_started - before.prefetches_started == (overlap ? 1U : 0U) &&
                      after.prefetches_consumed - before.prefetches_consumed ==
                          (overlap ? 3U : 0U) &&
                      after.prefetch_fallbacks == before.prefetch_fallbacks;
      }
    }
    const auto final_stats = ame::backend::moe_stream_stats();
    const bool geometry_ok =
        final_stats.layered && final_stats.layer_slots == kPoolSlots &&
        final_stats.arena_bytes ==
            static_cast<std::uint64_t>(kLayers) * kPoolSlots * slot_bytes_for(model.expert_bytes) &&
        final_stats.cache_metadata_bytes > 0 && final_stats.arena_bytes == installed.arena_bytes;
    const bool format_ok = mismatched == 0 && counters_ok && callbacks_ok && geometry_ok &&
                           !final_stats.failed;
    passed = passed && format_ok;
    report << (first_format ? "" : ",\n") << "    {\"format\": \"" << format.name
           << "\", \"operators\": " << compared << ", \"mismatched\": " << mismatched
           << ", \"counters_match_prediction\": " << (counters_ok ? "true" : "false")
           << ", \"geometry_ok\": " << (geometry_ok ? "true" : "false")
           << ", \"output_digest\": \"" << std::hex << digest << std::dec << "\""
           << ", \"arena_bytes\": " << final_stats.arena_bytes
           << ", \"metadata_bytes\": " << final_stats.cache_metadata_bytes
           << ", \"passed\": " << (format_ok ? "true" : "false") << "}";
    first_format = false;
    ame::backend::uninstall_moe_stream();
  }
  report << "\n  ],\n";

  // Refusals and faults, on the float model.
  const auto directory = std::filesystem::path(workdir) / "f32";
  const Model model = build_model(formats[0], directory, "ffn_", false);
  ame::backend::StreamOptions options;
  options.layout = ame::backend::CacheLayout::Layered;
  options.layer_slots = kPoolSlots;
  options.readers = readers;
  options.overlap = overlap;

  // A fused gate_up tensor cannot be expressed as three role views: refuse at install.
  const auto fused_dir = std::filesystem::path(workdir) / "fused";
  std::filesystem::create_directories(fused_dir);
  const Model fused = build_model(formats[0], fused_dir, "ffn_", true);
  ame::backend::uninstall_moe_stream();
  const bool fused_refused =
      !ame::backend::install_moe_stream(fused.file.string(), fused.manifest.string(), 1,
                                        64U * 1024U * 1024U, options) &&
      ame::backend::moe_stream_stats().error.find("separate gate, up and down") !=
          std::string::npos;

  // An arena allowance below one slot per layer is refused, not shrunk silently.
  ame::backend::StreamOptions derived;
  derived.layout = ame::backend::CacheLayout::Layered;
  const bool tiny_arena_refused =
      !ame::backend::install_moe_stream(model.file.string(), model.manifest.string(), 1, 1024,
                                        derived);

  // A derived pool is the largest that fits: arena plus bookkeeping stay inside the allowance
  // and one more slot per layer would not fit (unless every expert already has a slot).
  constexpr std::uint64_t kAllowance = 64U * 1024U;
  bool derived_ok = ame::backend::install_moe_stream(model.file.string(), model.manifest.string(),
                                                     1, kAllowance, derived);
  if (derived_ok) {
    const auto stats = ame::backend::moe_stream_stats();
    const std::uint64_t covered = stats.arena_bytes + stats.cache_metadata_bytes;
    const std::uint64_t one_more = stats.arena_bytes / stats.layer_slots * (stats.layer_slots + 1) +
                                   stats.cache_metadata_bytes;
    derived_ok = covered <= kAllowance && stats.layer_slots >= 1 &&
                 (stats.layer_slots == kExperts || one_more > kAllowance);
  }

  // Union above the pool fails the graph; so does a read failure. A fresh request then
  // matches the resident reference.
  ame::backend::uninstall_moe_stream();
  ame::backend::StreamOptions small = options;
  small.layer_slots = 3;
  const bool installed_small = ame::backend::install_moe_stream(
      model.file.string(), model.manifest.string(), 1, 64U * 1024U * 1024U, small);
  const std::vector<std::int32_t> four = {0, 1, 2, 3};
  const Output over = run_op(model, 0, 0, true, four, 1, kTopK, threads);
  const bool union_refused = installed_small && over.status != GGML_STATUS_SUCCESS &&
                             ame::backend::moe_stream_stats().failed;

  ame::backend::uninstall_moe_stream();
  (void)ame::backend::install_moe_stream(model.file.string(), model.manifest.string(), 1,
                                         64U * 1024U * 1024U, options);
  (void)::setenv("AME_IO_FAIL_AFTER_BYTES", "1", 1);
  ame::backend::begin_moe_stream_request();
  const Output failed = run_op(model, 0, 0, true, four, 1, kTopK, threads);
  const bool read_failed = failed.status != GGML_STATUS_SUCCESS &&
                           ame::backend::moe_stream_stats().failed;
  (void)::unsetenv("AME_IO_FAIL_AFTER_BYTES");
  ame::backend::begin_moe_stream_request();
  const Output recovered = run_op(model, 0, 0, true, four, 1, kTopK, threads);
  ame::backend::uninstall_moe_stream();
  const Output reference = run_op(model, 0, 0, true, four, 1, kTopK, threads);
  const bool recovery_ok = read_failed && same(reference, recovered);

  ame::backend::uninstall_moe_stream();
  (void)ame::backend::install_moe_stream(model.file.string(), model.manifest.string(), 1,
                                         64U * 1024U * 1024U, options);
  ame::backend::request_moe_stream_cancel();
  const Output cancelled = run_op(model, 0, 0, true, four, 1, kTopK, threads);
  const bool cancel_ok = cancelled.status != GGML_STATUS_SUCCESS &&
                         ame::backend::moe_stream_stats().cancelled;
  ame::backend::uninstall_moe_stream();

  // Overlap only: the callback ignores other tensors and unknown layers, an armed layer that is
  // never consumed is drained and dropped at request start, a read failure surfaces at the
  // operator, and cancellation starts nothing.
  bool overlap_checks = true;
  if (overlap) {
    ame::backend::uninstall_moe_stream();
    (void)ame::backend::install_moe_stream(model.file.string(), model.manifest.string(), 1,
                                           64U * 1024U * 1024U, options);
    ggml_init_params params{};
    params.mem_size = 1U * 1024U * 1024U;
    ggml_context* context = ggml_init(params);
    ggml_tensor* other = ggml_new_tensor_2d(context, GGML_TYPE_I32, kTopK, 1);
    ggml_set_name(other, "ffn_moe_weights-0");
    overlap_checks = overlap_checks && !ame::backend::moe_eval_callback(other, true, nullptr);
    ggml_free(context);
    bool asked = false;
    (void)deliver_topk(7, four, 1, kTopK, &asked);  // a layer the model does not have
    overlap_checks = overlap_checks && asked &&
                     ame::backend::moe_stream_stats().prefetches_started == 0;

    // Armed but never consumed.
    (void)deliver_topk(0, four, 1, kTopK, &asked);
    overlap_checks = overlap_checks && ame::backend::moe_stream_stats().prefetches_started == 1;
    ame::backend::begin_moe_stream_request();
    overlap_checks = overlap_checks && ame::backend::moe_stream_stats().resident_slots == 0;
    const Output after_drop = run_op(model, 0, 0, true, four, 1, kTopK, threads);
    ame::backend::uninstall_moe_stream();
    const Output resident_reference = run_op(model, 0, 0, true, four, 1, kTopK, threads);
    overlap_checks = overlap_checks && same(after_drop, resident_reference);

    // A read failure after the callback armed the layer.
    (void)ame::backend::install_moe_stream(model.file.string(), model.manifest.string(), 1,
                                           64U * 1024U * 1024U, options);
    (void)::setenv("AME_IO_FAIL_AFTER_BYTES", "1", 1);
    ame::backend::begin_moe_stream_request();
    (void)deliver_topk(0, four, 1, kTopK, &asked);
    const Output armed_failure = run_op(model, 0, 0, true, four, 1, kTopK, threads);
    overlap_checks = overlap_checks && armed_failure.status != GGML_STATUS_SUCCESS &&
                     ame::backend::moe_stream_stats().failed;
    (void)::unsetenv("AME_IO_FAIL_AFTER_BYTES");
    ame::backend::begin_moe_stream_request();
    (void)deliver_topk(0, four, 1, kTopK, &asked);
    const Output armed_recovery = run_op(model, 0, 0, true, four, 1, kTopK, threads);
    overlap_checks = overlap_checks && same(armed_recovery, resident_reference);

    // Cancellation: nothing is started.
    ame::backend::begin_moe_stream_request();
    const auto started_before = ame::backend::moe_stream_stats().prefetches_started;
    ame::backend::request_moe_stream_cancel();
    (void)deliver_topk(1, four, 1, kTopK, &asked);
    overlap_checks = overlap_checks &&
                     ame::backend::moe_stream_stats().prefetches_started == started_before;
    ame::backend::uninstall_moe_stream();
  }

  const bool refusals = overlap_checks && fused_refused && tiny_arena_refused && derived_ok && union_refused &&
                        recovery_ok && cancel_ok;
  passed = passed && refusals;
  report << "  \"fused_gate_up_refused\": " << (fused_refused ? "true" : "false")
         << ",\n  \"arena_below_one_slot_per_layer_refused\": "
         << (tiny_arena_refused ? "true" : "false")
         << ",\n  \"derived_pool_within_allowance\": " << (derived_ok ? "true" : "false")
         << ",\n  \"union_above_pool_refused\": " << (union_refused ? "true" : "false")
         << ",\n  \"read_failure_then_identical_recovery\": " << (recovery_ok ? "true" : "false")
         << ",\n  \"cancellation_reported\": " << (cancel_ok ? "true" : "false")
         << ",\n  \"overlap_checks\": " << (overlap_checks ? "true" : "false")
         << ",\n  \"passed\": " << (passed ? "true" : "false") << "\n}\n";
  std::cout << report.str();
  return passed ? 0 : 1;
}
