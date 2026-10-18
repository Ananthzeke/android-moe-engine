// Expert-remap gate against the pinned llama.cpp CPU kernel.
//
// The reference keeps every expert resident in one ggml tensor. The candidate
// stores the same experts in a fixed arena, acquires only the selected set, backs a
// compact ggml tensor with that arena, and remaps the kernel's expert ids to arena
// slots. Bitwise equality is required; this gate does not claim storage speed.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
using ame::residency::StoreError;
using ame::residency::StoreOptions;
using ame::residency::StoreStats;

struct Options {
  std::uint32_t embd = 64;
  std::uint32_t ff = 128;
  std::uint32_t experts = 32;
  std::uint32_t top_k = 4;
  std::uint32_t prefill_tokens = 8;
  std::uint32_t threads = 4;
  std::uint32_t seed = 1234;
  std::string output;
  std::string workdir = ".";
  std::string backend_commit;
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
  void double_value(double value) {
    element();
    char buffer[64] = {};
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

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

struct Comparison {
  bool identical{false};
  std::uint64_t mismatched{0};
  double max_absolute_difference{0.0};
};

Comparison compare(const std::vector<float>& reference, const std::vector<float>& candidate) {
  Comparison result;
  if (reference.size() != candidate.size()) {
    result.mismatched = std::max(reference.size(), candidate.size());
    result.max_absolute_difference = -1.0;
    return result;
  }
  for (std::size_t index = 0; index < reference.size(); ++index) {
    if (std::memcmp(&reference[index], &candidate[index], sizeof(float)) != 0) {
      ++result.mismatched;
      const double difference = std::abs(static_cast<double>(reference[index]) -
                                         static_cast<double>(candidate[index]));
      if (difference > result.max_absolute_difference) {
        result.max_absolute_difference = difference;
      }
    }
  }
  result.identical = result.mismatched == 0;
  return result;
}

std::uint64_t fnv1a(const std::vector<float>& values) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  const std::size_t count = values.size() * sizeof(float);
  for (std::size_t index = 0; index < count; ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

struct BatchShape {
  std::uint32_t tokens = 0;
  std::vector<std::int32_t> ids;
};

std::vector<std::int32_t> select_ids(std::uint32_t tokens, std::uint32_t top_k,
                                     std::uint32_t experts, std::uint64_t& state) {
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
    std::sort(ids.begin() + static_cast<std::ptrdiff_t>(token) * top_k,
              ids.begin() + static_cast<std::ptrdiff_t>(token + 1) * top_k);
  }
  return ids;
}

struct ComputeOutput {
  std::vector<float> values;
  double milliseconds{0.0};
  bool ok{false};
  std::string error;
};

ComputeOutput run_kernel(ggml_context* context, ggml_tensor* weights, ggml_tensor* input,
                         ggml_tensor* ids, std::uint32_t threads) {
  ComputeOutput output;
  ggml_tensor* result = ggml_mul_mat_id(context, weights, input, ids);
  ggml_cgraph* graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, result);
  const Clock::time_point begin = Clock::now();
  const ggml_status status = ggml_graph_compute_with_ctx(context, graph, threads);
  const Clock::time_point end = Clock::now();
  output.milliseconds = seconds_between(begin, end) * 1000.0;
  if (status != GGML_STATUS_SUCCESS) {
    output.error = "ggml_graph_compute_with_ctx failed";
    return output;
  }
  const std::size_t count = ggml_nbytes(result) / sizeof(float);
  output.values.resize(count);
  std::memcpy(output.values.data(), result->data, ggml_nbytes(result));
  output.ok = true;
  return output;
}

struct RoundResult {
  std::string name;
  std::uint32_t tokens{0};
  std::uint32_t union_size{0};
  bool acquired{false};
  Comparison comparison;
  double reference_milliseconds{0.0};
  double candidate_milliseconds{0.0};
  StoreStats stats{};
  std::string error;
};

bool write_expert_file(const std::string& path, const Options& options, std::uint64_t& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return false;
  }
  std::vector<float> expert(static_cast<std::size_t>(options.embd) * options.ff);
  for (std::uint32_t index = 0; index < options.experts; ++index) {
    for (std::uint32_t element = 0; element < expert.size(); ++element) {
      expert[element] = deterministic_weight(options.seed, index, element);
    }
    stream.write(reinterpret_cast<const char*>(expert.data()),
                 static_cast<std::streamsize>(expert.size() * sizeof(float)));
  }
  stream.close();
  bytes = static_cast<std::uint64_t>(options.experts) * expert.size() * sizeof(float);
  return stream.good();
}

RoundResult run_round(const std::string& name, const Options& options, std::uint32_t tokens,
                      std::span<const std::int32_t> ids, Store& store,
                      const std::vector<float>& expert_bytes,
                      const ame::Cancellation& stop) {
  RoundResult round;
  round.name = name;
  round.tokens = tokens;
  const std::size_t top_k = static_cast<std::size_t>(options.top_k) * tokens;
  std::vector<ExpertKey> keys;
  for (std::size_t index = 0; index < top_k; ++index) {
    const ExpertKey key{0, static_cast<std::uint32_t>(ids[index])};
    bool duplicate = false;
    for (const auto& existing : keys) {
      if (existing == key) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      keys.push_back(key);
    }
  }
  round.union_size = static_cast<std::uint32_t>(keys.size());
  auto lease = store.acquire(keys, stop);
  if (!lease.ok()) {
    round.error = "store acquire failed";
    return round;
  }
  round.acquired = true;

  const std::uint32_t experts = options.experts;
  const std::uint32_t slots = store.slot_count();
  const std::size_t expert_elements = static_cast<std::size_t>(options.embd) * options.ff;

  ggml_init_params reference_params{};
  reference_params.mem_size = 64U * 1024U * 1024U;
  reference_params.mem_buffer = nullptr;
  reference_params.no_alloc = false;
  ggml_context* reference_context = ggml_init(reference_params);
  ggml_init_params stage_params{};
  stage_params.mem_size = 1U * 1024U * 1024U;
  stage_params.mem_buffer = nullptr;
  stage_params.no_alloc = true;
  ggml_context* stage_context = ggml_init(stage_params);
  ggml_init_params candidate_params{};
  candidate_params.mem_size = 64U * 1024U * 1024U;
  candidate_params.mem_buffer = nullptr;
  candidate_params.no_alloc = false;
  ggml_context* candidate_context = ggml_init(candidate_params);
  if (reference_context == nullptr || stage_context == nullptr || candidate_context == nullptr) {
    round.error = "ggml context creation failed";
    ggml_free(reference_context);
    ggml_free(stage_context);
    ggml_free(candidate_context);
    return round;
  }

  ggml_tensor* full_weights = ggml_new_tensor_3d(reference_context, GGML_TYPE_F32,
                                                 options.embd, options.ff, experts);
  std::memcpy(full_weights->data, expert_bytes.data(), expert_bytes.size() * sizeof(float));
  ggml_tensor* reference_input =
      ggml_new_tensor_3d(reference_context, GGML_TYPE_F32, options.embd, 1, tokens);
  for (std::size_t index = 0; index < static_cast<std::size_t>(options.embd) * tokens; ++index) {
    static_cast<float*>(reference_input->data)[index] =
        static_cast<float>(static_cast<std::int64_t>(index % 13U) - 6) / 7.0F;
  }
  ggml_tensor* reference_ids =
      ggml_new_tensor_2d(reference_context, GGML_TYPE_I32, options.top_k, tokens);
  std::memcpy(reference_ids->data, ids.data(), ids.size() * sizeof(std::int32_t));
  const ComputeOutput reference =
      run_kernel(reference_context, full_weights, reference_input, reference_ids, options.threads);

  ggml_tensor* stage_weights =
      ggml_new_tensor_3d(stage_context, GGML_TYPE_F32, options.embd, options.ff, slots);
  stage_weights->data = store.arena().data();
  ggml_tensor* candidate_input =
      ggml_new_tensor_3d(candidate_context, GGML_TYPE_F32, options.embd, 1, tokens);
  std::memcpy(candidate_input->data, reference_input->data,
              static_cast<std::size_t>(options.embd) * tokens * sizeof(float));
  ggml_tensor* staged_ids =
      ggml_new_tensor_2d(candidate_context, GGML_TYPE_I32, options.top_k, tokens);
  auto* staged = static_cast<std::int32_t*>(staged_ids->data);
  bool remap_ok = true;
  for (std::size_t index = 0; index < top_k; ++index) {
    const ExpertKey key{0, static_cast<std::uint32_t>(ids[index])};
    const auto slot = lease.value().slot_index(key);
    if (!slot.has_value()) {
      remap_ok = false;
      break;
    }
    staged[index] = static_cast<std::int32_t>(*slot);
  }
  ComputeOutput candidate;
  if (remap_ok && reference.ok) {
    candidate = run_kernel(candidate_context, stage_weights, candidate_input, staged_ids,
                           options.threads);
  } else {
    candidate.error = remap_ok ? reference.error : "slot remap failed";
  }
  round.reference_milliseconds = reference.milliseconds;
  round.candidate_milliseconds = candidate.milliseconds;
  round.comparison = compare(reference.values, candidate.values);
  if (!reference.ok || !candidate.ok) {
    round.error = !reference.ok ? reference.error : candidate.error;
  }
  round.stats = store.stats();

  ggml_free(reference_context);
  ggml_free(stage_context);
  ggml_free(candidate_context);
  (void)expert_elements;
  return round;
}

}  // namespace

int main(int argc, char* argv[]) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << "usage: moe_kernel_gate [--embd N] [--ff N] [--experts N] [--top-k N]\n"
                   "       [--prefill-tokens N] [--threads N] [--output PATH]\n"
                   "       [--workdir DIR] [--backend-commit SHA]\n";
      return 0;
    }
    if (argument == "--output" || argument == "--workdir" || argument == "--backend-commit" ||
        argument == "--embd" || argument == "--ff" || argument == "--experts" ||
        argument == "--top-k" || argument == "--prefill-tokens" || argument == "--threads" ||
        argument == "--seed") {
      if (index + 1 >= argc) {
        std::cerr << "Missing value for " << argument << '\n';
        return 2;
      }
      const std::string value = argv[++index];
      if (argument == "--output") {
        options.output = value;
      } else if (argument == "--workdir") {
        options.workdir = value;
      } else if (argument == "--backend-commit") {
        options.backend_commit = value;
      } else {
        std::uint32_t parsed = 0;
        if (!parse_uint(value, parsed)) {
          std::cerr << "Invalid number for " << argument << '\n';
          return 2;
        }
        if (argument == "--embd") {
          options.embd = parsed;
        } else if (argument == "--ff") {
          options.ff = parsed;
        } else if (argument == "--experts") {
          options.experts = parsed;
        } else if (argument == "--top-k") {
          options.top_k = parsed;
        } else if (argument == "--prefill-tokens") {
          options.prefill_tokens = parsed;
        } else if (argument == "--threads") {
          options.threads = parsed;
        } else {
          options.seed = parsed;
        }
      }
    } else {
      std::cerr << "Unsupported argument: " << argument << '\n';
      return 2;
    }
  }
  if (options.top_k == 0 || options.top_k > options.experts || options.embd == 0 ||
      options.ff == 0 || options.experts == 0 || options.prefill_tokens == 0) {
    std::cerr << "Invalid dimensions\n";
    return 2;
  }

  const std::string path = options.workdir + "/experts.bin";
  std::uint64_t file_bytes = 0;
  if (!write_expert_file(path, options, file_bytes)) {
    std::cerr << "Cannot write expert fixture\n";
    return 1;
  }
  auto source = FileSource::open(path);
  if (!source.ok()) {
    std::cerr << "Cannot open expert fixture\n";
    return 1;
  }
  std::vector<ExpertEntry> entries;
  const std::uint64_t expert_bytes = static_cast<std::uint64_t>(options.embd) * options.ff *
                                     sizeof(float);
  for (std::uint32_t expert = 0; expert < options.experts; ++expert) {
    entries.push_back(ExpertEntry{ExpertKey{0, expert}, FileOffset{expert * expert_bytes},
                                  ByteCount{expert_bytes}});
  }

  std::vector<float> expert_data(static_cast<std::size_t>(file_bytes / sizeof(float)));
  {
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(expert_data.data()),
                static_cast<std::streamsize>(file_bytes));
    if (!stream) {
      std::cerr << "Cannot read expert fixture\n";
      return 1;
    }
  }

  std::uint64_t random_state = options.seed;
  const std::vector<std::int32_t> decode_ids =
      select_ids(1, options.top_k, options.experts, random_state);
  const std::vector<std::int32_t> prefill_ids =
      select_ids(options.prefill_tokens, options.top_k, options.experts, random_state);

  std::vector<ExpertKey> union_keys;
  auto add_unique = [&union_keys](std::uint32_t expert) {
    const ExpertKey key{0, expert};
    for (const auto& existing : union_keys) {
      if (existing == key) {
        return;
      }
    }
    union_keys.push_back(key);
  };
  for (const auto id : prefill_ids) {
    add_unique(static_cast<std::uint32_t>(id));
  }
  const std::uint32_t union_size = static_cast<std::uint32_t>(union_keys.size());

  StoreOptions store_options;
  store_options.slot_count = union_size;
  auto store_result = Store::create(std::move(source).value(), entries, store_options);
  if (!store_result.ok()) {
    std::cerr << "Cannot create store\n";
    return 1;
  }
  Store& store = *store_result.value();
  ame::Cancellation stop;

  const RoundResult decode = run_round("decode", options, 1, decode_ids, store, expert_data,
                                       stop);
  const RoundResult prefill = run_round("prefill", options, options.prefill_tokens, prefill_ids,
                                        store, expert_data, stop);

  bool refusal_expected = false;
  std::string refusal_error;
  {
    auto refusal_source = FileSource::open(path);
    if (!refusal_source.ok()) {
      std::cerr << "Cannot reopen expert fixture\n";
      return 1;
    }
    StoreOptions refusal_options;
    refusal_options.slot_count = union_size - 1;
    auto refusal_store =
        Store::create(std::move(refusal_source).value(), entries, refusal_options);
    if (refusal_store.ok()) {
      const auto refused = refusal_store.value()->acquire(union_keys, stop);
      refusal_expected = !refused.ok() && refused.error() == StoreError::ResourceLimit;
      if (!refused.ok()) {
        refusal_error = "resource_limit";
      }
    }
  }

  JsonWriter json;
  json.begin_object();
  json.key("schema_version");
  json.unsigned_value(1);
  json.key("gate");
  json.string_value("expert_id_remap_against_llama_cpp_cpu_mul_mat_id");
  json.key("backend");
  json.begin_object();
  json.key("name");
  json.string_value("llama.cpp");
  json.key("commit");
  json.string_value(options.backend_commit);
  json.key("op");
  json.string_value("GGML_OP_MUL_MAT_ID / ggml_compute_forward_mul_mat_id");
  json.end_object();
  json.key("dimensions");
  json.begin_object();
  json.key("n_embd");
  json.unsigned_value(options.embd);
  json.key("n_ff");
  json.unsigned_value(options.ff);
  json.key("n_expert");
  json.unsigned_value(options.experts);
  json.key("n_expert_used");
  json.unsigned_value(options.top_k);
  json.key("prefill_tokens");
  json.unsigned_value(options.prefill_tokens);
  json.key("threads");
  json.unsigned_value(options.threads);
  json.end_object();
  json.key("memory");
  json.begin_object();
  json.key("resident_reference_bytes");
  json.unsigned_value(file_bytes);
  json.key("arena_bytes");
  json.unsigned_value(store.arena().size());
  json.key("slot_bytes");
  json.unsigned_value(store.slot_bytes());
  json.key("slots");
  json.unsigned_value(store.slot_count());
  json.key("prefill_union");
  json.unsigned_value(union_size);
  json.end_object();
  json.key("rounds");
  json.begin_array();
  for (const RoundResult* round : {&decode, &prefill}) {
    json.begin_object();
    json.key("name");
    json.string_value(round->name);
    json.key("tokens");
    json.unsigned_value(round->tokens);
    json.key("union");
    json.unsigned_value(round->union_size);
    json.key("identical");
    json.bool_value(round->comparison.identical && round->error.empty());
    json.key("mismatched_floats");
    json.unsigned_value(round->comparison.mismatched);
    json.key("max_absolute_difference");
    json.double_value(round->comparison.max_absolute_difference);
    json.key("reference_ms");
    json.double_value(round->reference_milliseconds);
    json.key("candidate_ms");
    json.double_value(round->candidate_milliseconds);
    if (!round->error.empty()) {
      json.key("error");
      json.string_value(round->error);
    }
    json.key("store");
    json.begin_object();
    json.key("hits");
    json.unsigned_value(round->stats.hits);
    json.key("misses");
    json.unsigned_value(round->stats.misses);
    json.key("evictions");
    json.unsigned_value(round->stats.evictions);
    json.key("read_calls");
    json.unsigned_value(round->stats.read_calls);
    json.key("read_bytes");
    json.unsigned_value(round->stats.read_bytes);
    json.key("read_seconds");
    json.double_value(round->stats.read_seconds);
    json.key("pinned_slots");
    json.unsigned_value(round->stats.pinned_slots);
    json.key("resident_slots");
    json.unsigned_value(round->stats.resident_slots);
    json.end_object();
    json.end_object();
  }
  json.end_array();
  json.key("refusal");
  json.begin_object();
  json.key("expected_resource_limit");
  json.bool_value(refusal_expected);
  json.key("error");
  json.string_value(refusal_error);
  json.end_object();
  const bool pass = decode.acquired && prefill.acquired && decode.error.empty() &&
                    prefill.error.empty() && decode.comparison.identical &&
                    prefill.comparison.identical && refusal_expected;
  json.key("status");
  json.string_value(pass ? "pass" : "fail");
  json.key("notes");
  json.begin_array();
  json.string_value("The reference keeps the complete expert table resident.");
  json.string_value("The candidate arena backs the staged ggml tensor directly.");
  json.string_value("Bitwise float equality is required; no tolerance is applied.");
  json.end_array();
  json.end_object();

  const std::string contents = json.text() + "\n";
  if (options.output.empty()) {
    std::cout << contents;
  } else {
    std::ofstream stream(options.output, std::ios::binary | std::ios::trunc);
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!stream.good()) {
      std::cerr << "Cannot write output\n";
      return 1;
    }
  }
  return pass ? 0 : 1;
}