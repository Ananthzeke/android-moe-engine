// Quantized expert-remap gate against the pinned CPU MUL_MAT_ID kernel.
//
// The reference keeps the complete quantized expert table resident. The candidate
// stores identical quantized bytes in a bounded arena, acquires the selected
// experts, backs a compact tensor with the arena, and remaps IDs to arena slots.
// Bitwise equality is required; no tolerance is applied.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "expert_layout.hpp"
#include "file_source.hpp"
#include "store.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

namespace {
using ame::model::ByteCount;
using ame::model::ExpertKey;
using ame::model::FileOffset;
using ame::residency::ExpertEntry;
using ame::residency::FileSource;
using ame::residency::Store;
using ame::residency::StoreOptions;
using ame::residency::StoreStats;

struct Options {
  std::string type_name = "q4_k";
  std::string role = "gate";
  std::uint32_t embd = 256;
  std::uint32_t ff = 256;
  std::uint32_t experts = 8;
  std::uint32_t top_k = 4;
  std::uint32_t prefill_tokens = 8;
  std::uint32_t threads = 4;
  std::uint32_t seed = 1234;
  std::string workdir = ".";
  std::string output;
  std::string backend_commit;
  bool corrupt = false;
};

std::uint64_t next_random(std::uint64_t& state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return state >> 33U;
}

float deterministic_weight(std::uint64_t seed, std::uint32_t expert, std::uint64_t index) {
  std::uint64_t state = seed ^ (static_cast<std::uint64_t>(expert) << 32U) ^ index;
  const std::uint64_t random = next_random(state);
  return static_cast<float>(static_cast<std::int64_t>(random % 2001U) - 1000) / 1000.0F;
}

bool parse_uint(std::string_view text, std::uint32_t& value) {
  std::uint64_t parsed = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || parsed > UINT32_MAX) {
    return false;
  }
  value = static_cast<std::uint32_t>(parsed);
  return true;
}

ggml_type parse_type(std::string_view text) {
  if (text == "q4_k") {
    return GGML_TYPE_Q4_K;
  }
  if (text == "q6_k") {
    return GGML_TYPE_Q6_K;
  }
  return GGML_TYPE_COUNT;
}

struct Comparison {
  bool identical{false};
  std::uint64_t mismatched{0};
  double max_difference{0.0};
};

Comparison compare(const std::vector<float>& left, const std::vector<float>& right) {
  Comparison result;
  if (left.size() != right.size()) {
    result.mismatched = std::max(left.size(), right.size());
    result.max_difference = -1.0;
    return result;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (std::memcmp(&left[index], &right[index], sizeof(float)) != 0) {
      ++result.mismatched;
      result.max_difference = std::max(
          result.max_difference, std::abs(static_cast<double>(left[index]) - right[index]));
    }
  }
  result.identical = result.mismatched == 0;
  return result;
}

struct Run {
  std::vector<float> values;
  bool ok{false};
  std::string error;
};

Run run_kernel(ggml_context* context, ggml_tensor* weights, ggml_tensor* input,
               ggml_tensor* ids, std::uint32_t threads) {
  Run result;
  ggml_tensor* output = ggml_mul_mat_id(context, weights, input, ids);
  ggml_cgraph* graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, output);
  if (ggml_graph_compute_with_ctx(context, graph, static_cast<int>(threads)) !=
      GGML_STATUS_SUCCESS) {
    result.error = "graph compute failed";
    return result;
  }
  result.values.resize(ggml_nbytes(output) / sizeof(float));
  std::memcpy(result.values.data(), output->data, ggml_nbytes(output));
  result.ok = true;
  return result;
}

std::vector<std::int32_t> select_ids(std::uint32_t tokens, std::uint32_t top_k,
                                     std::uint32_t experts, std::uint64_t& state,
                                     std::uint32_t offset = 0) {
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * top_k, 0);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    std::vector<std::uint32_t> available(experts);
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      available[expert] = expert;
    }
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      const std::uint32_t picked = static_cast<std::uint32_t>(
          next_random(state) % static_cast<std::uint64_t>(available.size()));
      ids[static_cast<std::size_t>(token) * top_k + slot] =
          static_cast<std::int32_t>(available[picked]);
      available[picked] = available.back();
      available.pop_back();
    }
    (void)offset;
  }
  return ids;
}

std::pair<std::uint32_t, std::uint32_t> layout_dims(const Options& options) {
  if (options.role == "down") {
    return {options.ff, options.embd};
  }
  return {options.embd, options.ff};
}

std::uint64_t fingerprint(const std::vector<float>& values) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
  for (std::size_t index = 0; index < values.size() * sizeof(float); ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

struct Round {
  std::string name;
  std::uint32_t tokens{0};
  std::uint32_t union_size{0};
  bool acquired{false};
  Comparison comparison;
  StoreStats stats{};
  std::uint64_t reference_fingerprint{0};
  std::uint64_t candidate_fingerprint{0};
  std::string error;
};

Round run_round(const std::string& name, const Options& options, ggml_type type,
                std::uint32_t row_dim, std::uint32_t out_dim,
                std::uint32_t tokens, std::span<const std::int32_t> ids,
                const std::vector<std::uint8_t>& expert_bytes, std::uint64_t expert_size,
                std::uint32_t store_slots, const std::string& file_path, bool corrupt,
                Store* shared_store = nullptr) {
  Round result;
  result.name = name;
  result.tokens = tokens;
  std::vector<ExpertKey> keys;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    const ExpertKey key{0, static_cast<std::uint32_t>(ids[index])};
    if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
      keys.push_back(key);
    }
  }
  result.union_size = static_cast<std::uint32_t>(keys.size());
  if (keys.size() > store_slots) {
    result.error = "selected union exceeds slots";
    return result;
  }

  std::vector<ExpertEntry> entries;
  for (std::uint32_t expert = 0; expert < options.experts; ++expert) {
    entries.push_back(ExpertEntry{ExpertKey{0, expert},
                                  FileOffset{static_cast<std::uint64_t>(expert) * expert_size},
                                  ByteCount{expert_size}});
  }
  std::unique_ptr<Store> owned_store;
  Store* store = shared_store;
  if (store == nullptr) {
    auto source = FileSource::open(file_path);
    if (!source.ok()) {
      result.error = "cannot open fixture";
      return result;
    }
    StoreOptions store_options;
    store_options.slot_count = store_slots;
    auto store_result = Store::create(std::move(source).value(), entries, store_options);
    if (!store_result.ok()) {
      result.error = "cannot create store";
      return result;
    }
    owned_store = std::move(store_result).value();
    store = owned_store.get();
  }
  ame::Cancellation stop;
  auto lease = store->acquire(keys, stop);
  if (!lease.ok()) {
    result.error = "store acquire failed";
    return result;
  }
  result.acquired = true;
  if (corrupt) {
    const auto slot = lease.value().slot_index(keys[0]);
    if (slot.has_value()) {
      std::memset(store->arena().data() + static_cast<std::size_t>(*slot) * expert_size,
                  0xA5, static_cast<std::size_t>(expert_size));
    }
  }

  ggml_init_params params{};
  params.mem_size = 64U * 1024U * 1024U;
  ggml_context* reference_context = ggml_init(params);
  ggml_context* candidate_context = ggml_init(params);
  ggml_init_params stage_params{};
  stage_params.mem_size = 1024U * 1024U;
  stage_params.no_alloc = true;
  ggml_context* stage_context = ggml_init(stage_params);
  if (reference_context == nullptr || candidate_context == nullptr ||
      stage_context == nullptr) {
    result.error = "ggml context creation failed";
    ggml_free(reference_context);
    ggml_free(candidate_context);
    ggml_free(stage_context);
    return result;
  }

  ggml_tensor* reference_weights =
      ggml_new_tensor_3d(reference_context, type, row_dim, out_dim, options.experts);
  std::memcpy(reference_weights->data,
              expert_bytes.data(),
              static_cast<std::size_t>(expert_size) * options.experts);
  ggml_tensor* reference_input =
      ggml_new_tensor_3d(reference_context, GGML_TYPE_F32, row_dim, 1, tokens);
  for (std::size_t index = 0; index < static_cast<std::size_t>(row_dim) * tokens; ++index) {
    static_cast<float*>(reference_input->data)[index] =
        static_cast<float>(static_cast<std::int64_t>(index % 17U) - 8) / 8.0F;
  }
  ggml_tensor* reference_ids =
      ggml_new_tensor_2d(reference_context, GGML_TYPE_I32, options.top_k, tokens);
  std::memcpy(reference_ids->data, ids.data(), ids.size() * sizeof(std::int32_t));
  const Run reference =
      run_kernel(reference_context, reference_weights, reference_input, reference_ids,
                 options.threads);

  ggml_tensor* staged_weights =
      ggml_new_tensor_3d(stage_context, type, row_dim, out_dim, store_slots);
  staged_weights->data = store->arena().data();
  ggml_tensor* candidate_input =
      ggml_new_tensor_3d(candidate_context, GGML_TYPE_F32, row_dim, 1, tokens);
  std::memcpy(candidate_input->data, reference_input->data,
              static_cast<std::size_t>(row_dim) * tokens * sizeof(float));
  ggml_tensor* staged_ids =
      ggml_new_tensor_2d(candidate_context, GGML_TYPE_I32, options.top_k, tokens);
  auto* remapped = static_cast<std::int32_t*>(staged_ids->data);
  bool remap_ok = true;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    const ExpertKey key{0, static_cast<std::uint32_t>(ids[index])};
    const auto slot = lease.value().slot_index(key);
    if (!slot.has_value()) {
      remap_ok = false;
      break;
    }
    remapped[index] = static_cast<std::int32_t>(*slot);
  }
  Run candidate;
  if (remap_ok && reference.ok) {
    candidate = run_kernel(candidate_context, staged_weights, candidate_input, staged_ids,
                           options.threads);
  } else {
    candidate.error = remap_ok ? reference.error : "slot remap failed";
  }
  result.comparison = compare(reference.values, candidate.values);
  result.reference_fingerprint = fingerprint(reference.values);
  result.candidate_fingerprint = fingerprint(candidate.values);
  if (!reference.ok || !candidate.ok) {
    result.error = !reference.ok ? reference.error : candidate.error;
  }
  result.stats = store->stats();
  ggml_free(reference_context);
  ggml_free(candidate_context);
  ggml_free(stage_context);
  return result;
}

bool write_quantized_fixture(const std::string& path, const Options& options, ggml_type type,
                             std::uint32_t row_dim, std::uint32_t out_dim,
                             std::uint64_t& expert_size, std::vector<std::uint8_t>& bytes) {
  const std::size_t row_size = ggml_row_size(type, row_dim);
  expert_size = row_size * out_dim;
  bytes.resize(static_cast<std::size_t>(expert_size) * options.experts);
  std::vector<float> source(static_cast<std::size_t>(row_dim) * out_dim);
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return false;
  }
  for (std::uint32_t expert = 0; expert < options.experts; ++expert) {
    for (std::size_t index = 0; index < source.size(); ++index) {
      source[index] = deterministic_weight(options.seed, expert, index);
    }
    const std::size_t written = ggml_quantize_chunk(
        type, source.data(), bytes.data() + static_cast<std::size_t>(expert) * expert_size,
        0, out_dim, row_dim, nullptr);
    if (written != expert_size) {
      return false;
    }
  }
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  return stream.good();
}

}  // namespace

int main(int argc, char* argv[]) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&](std::string& output) {
      if (index + 1 >= argc) {
        return false;
      }
      output = argv[++index];
      return true;
    };
    if (argument == "--help") {
      std::cout << "usage: moe_quantized_gate [--type q4_k|q6_k] [--embd N] [--ff N]\n"
                   "       [--experts N] [--top-k N] [--prefill-tokens N] [--threads N]\n"
                   "       [--role gate|up|down] [--workdir DIR] [--output PATH] [--corrupt]\n";
      return 0;
    }
    std::string text;
    if (argument == "--role") {
      if (!value(text)) return 2;
      options.role = text;
    } else if (argument == "--type") {
      if (!value(text)) return 2;
      options.type_name = text;
    } else if (argument == "--workdir") {
      if (!value(options.workdir)) return 2;
    } else if (argument == "--output") {
      if (!value(options.output)) return 2;
    } else if (argument == "--backend-commit") {
      if (!value(options.backend_commit)) return 2;
    } else if (argument == "--corrupt") {
      options.corrupt = true;
    } else if (argument == "--embd" || argument == "--ff" || argument == "--experts" ||
               argument == "--top-k" || argument == "--prefill-tokens" ||
               argument == "--threads" || argument == "--seed") {
      if (!value(text)) return 2;
      std::uint32_t parsed = 0;
      if (!parse_uint(text, parsed)) return 2;
      if (argument == "--embd") options.embd = parsed;
      else if (argument == "--ff") options.ff = parsed;
      else if (argument == "--experts") options.experts = parsed;
      else if (argument == "--top-k") options.top_k = parsed;
      else if (argument == "--prefill-tokens") options.prefill_tokens = parsed;
      else if (argument == "--threads") options.threads = parsed;
      else options.seed = parsed;
    } else {
      std::cerr << "unsupported argument: " << argument << '\n';
      return 2;
    }
  }
  const ggml_type type = parse_type(options.type_name);
  if (options.role != "gate" && options.role != "up" && options.role != "down") {
    std::cerr << "invalid role\n";
    return 2;
  }
  const auto [row_dim, out_dim] = layout_dims(options);
  if (type == GGML_TYPE_COUNT || options.top_k == 0 || options.top_k > options.experts ||
      row_dim % 256 != 0 || out_dim == 0 || options.threads == 0) {
    std::cerr << "invalid dimensions or quantization type\n";
    return 2;
  }
  std::filesystem::create_directories(options.workdir);
  const std::string path = options.workdir + "/quantized-experts.bin";
  std::uint64_t expert_size = 0;
  std::vector<std::uint8_t> expert_bytes;
  if (!write_quantized_fixture(path, options, type, row_dim, out_dim, expert_size,
                               expert_bytes)) {
    std::cerr << "cannot write quantized fixture\n";
    return 1;
  }

  std::uint64_t random_state = options.seed;
  const auto decode_a = select_ids(1, options.top_k, options.experts, random_state);
  const auto decode_a_again = decode_a;
  const auto decode_b = select_ids(1, options.top_k, options.experts, random_state);
  const auto prefill = select_ids(options.prefill_tokens, options.top_k, options.experts,
                                  random_state);

  std::vector<ExpertEntry> entries;
  for (std::uint32_t expert = 0; expert < options.experts; ++expert) {
    entries.push_back(ExpertEntry{ExpertKey{0, expert},
                                  FileOffset{static_cast<std::uint64_t>(expert) * expert_size},
                                  ByteCount{expert_size}});
  }
  auto make_store = [&](std::uint32_t slots) -> std::unique_ptr<Store> {
    auto source = FileSource::open(path);
    if (!source.ok()) {
      return nullptr;
    }
    StoreOptions store_options;
    store_options.slot_count = slots;
    auto created = Store::create(std::move(source).value(), entries, store_options);
    return created.ok() ? std::move(created).value() : nullptr;
  };
  auto cache_store = make_store(options.top_k);
  if (cache_store == nullptr) {
    std::cerr << "cannot create cache store\n";
    return 1;
  }
  const auto cache_a = run_round("decode_a", options, type, row_dim, out_dim, 1, decode_a,
                                 expert_bytes, expert_size, options.top_k, path,
                                 options.corrupt, cache_store.get());
  const auto cache_hit = run_round("decode_a_repeat", options, type, row_dim, out_dim, 1,
                                   decode_a_again, expert_bytes, expert_size, options.top_k,
                                   path, false, cache_store.get());
  const auto cache_b = run_round("decode_b", options, type, row_dim, out_dim, 1, decode_b,
                                 expert_bytes, expert_size, options.top_k, path, false,
                                 cache_store.get());
  auto duplicate_ids = decode_a;
  if (duplicate_ids.size() >= 2) {
    duplicate_ids[1] = duplicate_ids[0];
  }
  const auto duplicate_round = run_round("decode_duplicate", options, type, row_dim, out_dim,
                                         1, duplicate_ids, expert_bytes, expert_size,
                                         options.top_k, path, false, cache_store.get());
  std::vector<std::int32_t> noncontiguous = {0, 2, 5, 7};
  if (options.experts < 8 || options.top_k > 4) {
    noncontiguous = decode_a;
  }
  const auto noncontiguous_round = run_round("decode_noncontiguous", options, type, row_dim,
                                              out_dim, 1, noncontiguous, expert_bytes,
                                              expert_size, options.top_k, path, false,
                                              cache_store.get());
  std::uint32_t prefill_union = 0;
  for (const auto id : prefill) {
    (void)id;
  }
  std::vector<ExpertKey> prefill_keys;
  for (const auto id : prefill) {
    const ExpertKey key{0, static_cast<std::uint32_t>(id)};
    if (std::find(prefill_keys.begin(), prefill_keys.end(), key) == prefill_keys.end()) {
      prefill_keys.push_back(key);
    }
  }
  prefill_union = static_cast<std::uint32_t>(prefill_keys.size());
  auto prefill_store = make_store(prefill_union);
  if (prefill_store == nullptr) {
    std::cerr << "cannot create prefill store\n";
    return 1;
  }
  const auto prefill_round =
      run_round("prefill", options, type, row_dim, out_dim, options.prefill_tokens, prefill,
                expert_bytes, expert_size, prefill_union, path, false, prefill_store.get());

  const bool cache_ok = cache_a.acquired && cache_hit.acquired && cache_b.acquired &&
                        duplicate_round.acquired && noncontiguous_round.acquired &&
                        cache_a.comparison.identical && cache_hit.comparison.identical &&
                        cache_b.comparison.identical && duplicate_round.comparison.identical &&
                        noncontiguous_round.comparison.identical;
  const bool prefill_ok = prefill_round.acquired && prefill_round.comparison.identical &&
                          prefill_round.error.empty();
  const bool corrupt_detected = !cache_a.comparison.identical;
  const bool pass = options.corrupt ? corrupt_detected : (cache_ok && prefill_ok);
  std::cout << "{\n"
            << "  \"type\": \"" << options.type_name << "\",\n"
            << "  \"role\": \"" << options.role << "\",\n"
            << "  \"embd\": " << options.embd << ",\n"
            << "  \"ff\": " << options.ff << ",\n"
            << "  \"experts\": " << options.experts << ",\n"
            << "  \"threads\": " << options.threads << ",\n"
            << "  \"expert_bytes\": " << expert_size << ",\n"
            << "  \"decode_a\": " << (cache_a.comparison.identical ? "true" : "false") << ",\n"
            << "  \"decode_a_hit\": " << (cache_hit.comparison.identical ? "true" : "false") << ",\n"
            << "  \"decode_b\": " << (cache_b.comparison.identical ? "true" : "false") << ",\n"
            << "  \"decode_duplicate\": "
            << (duplicate_round.comparison.identical ? "true" : "false") << ",\n"
            << "  \"decode_noncontiguous\": "
            << (noncontiguous_round.comparison.identical ? "true" : "false") << ",\n"
            << "  \"prefill\": " << (prefill_round.comparison.identical ? "true" : "false") << ",\n"
            << "  \"prefill_union\": " << prefill_union << ",\n"
            << "  \"reference_fingerprint\": " << cache_a.reference_fingerprint << ",\n"
            << "  \"candidate_fingerprint\": " << cache_a.candidate_fingerprint << ",\n"
            << "  \"prefill_reference_fingerprint\": " << prefill_round.reference_fingerprint << ",\n"
            << "  \"prefill_candidate_fingerprint\": " << prefill_round.candidate_fingerprint << ",\n"
            << "  \"cache_hits\": " << cache_hit.stats.hits << ",\n"
            << "  \"cache_misses\": " << cache_hit.stats.misses << ",\n"
            << "  \"evictions\": " << cache_b.stats.evictions << ",\n"
            << "  \"corrupt\": " << (options.corrupt ? "true" : "false") << ",\n"
            << "  \"corrupt_detected\": " << (corrupt_detected ? "true" : "false") << ",\n"
            << "  \"passed\": " << (pass ? "true" : "false") << "\n"
            << "}\n";
  return pass ? 0 : 1;
}