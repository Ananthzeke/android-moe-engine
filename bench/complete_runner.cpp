#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "llama.h"

#ifdef AME_STREAMING
#include "artifact.hpp"
#include "moe_stream.hpp"
#endif

namespace {
struct Options {
  std::string model;
  std::string manifest;
  std::string sha;
  std::string output;
  std::string tokens_output;
  std::string refusal_output;
  std::uint32_t context = 256;
  std::uint32_t threads = 1;
  std::uint32_t batch = 8;
  std::uint32_t ubatch = 8;
  std::uint32_t predict = 2;
  std::uint32_t slots = 8;
  std::vector<std::int32_t> prompt_ids;
};

bool parse_ids(std::string_view text, std::vector<std::int32_t>& ids) {
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::string_view part = text.substr(begin, end == std::string_view::npos
                                                           ? text.size() - begin
                                                           : end - begin);
    std::int32_t value = 0;
    const auto result = std::from_chars(part.data(), part.data() + part.size(), value);
    if (result.ec != std::errc{} || result.ptr != part.data() + part.size() || value < 0) {
      return false;
    }
    ids.push_back(value);
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  return !ids.empty();
}

bool write_logits(const std::string& path, const std::vector<std::vector<float>>& logits) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return false;
  }
  const std::uint64_t count = logits.size();
  stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
  for (const auto& row : logits) {
    const std::uint64_t width = row.size();
    stream.write(reinterpret_cast<const char*>(&width), sizeof(width));
    stream.write(reinterpret_cast<const char*>(row.data()),
                 static_cast<std::streamsize>(row.size() * sizeof(float)));
  }
  return stream.good();
}

bool write_tokens(const std::string& path, const std::vector<std::int32_t>& tokens) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return false;
  }
  const std::uint64_t count = tokens.size();
  stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
  stream.write(reinterpret_cast<const char*>(tokens.data()),
               static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)));
  return stream.good();
}
}  // namespace

int main(int argc, char* argv[]) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    auto next = [&](std::string& value) {
      if (index + 1 >= argc) return false;
      value = argv[++index];
      return true;
    };
    std::string text;
    if (arg == "--model") {
      if (!next(options.model)) return 2;
    } else if (arg == "--manifest") {
      if (!next(options.manifest)) return 2;
    } else if (arg == "--sha") {
      if (!next(options.sha)) return 2;
    } else if (arg == "--output") {
      if (!next(options.output)) return 2;
    } else if (arg == "--tokens-output") {
      if (!next(options.tokens_output)) return 2;
    } else if (arg == "--refusal-output") {
      if (!next(options.refusal_output)) return 2;
    } else if (arg == "--prompt-ids") {
      if (!next(text) || !parse_ids(text, options.prompt_ids)) return 2;
    } else if (arg == "--context" || arg == "--threads" || arg == "--batch" ||
               arg == "--ubatch" || arg == "--predict" || arg == "--slots") {
      if (!next(text)) return 2;
      std::uint32_t value = 0;
      const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
      if (result.ec != std::errc{}) return 2;
      if (arg == "--context") options.context = value;
      else if (arg == "--threads") options.threads = value;
      else if (arg == "--batch") options.batch = value;
      else if (arg == "--ubatch") options.ubatch = value;
      else if (arg == "--slots") options.slots = value;
      else options.predict = value;
    } else {
      std::cerr << "unsupported argument: " << arg << '\n';
      return 2;
    }
  }
  if (options.model.empty() || options.prompt_ids.empty() || options.output.empty()) {
    std::cerr << "usage: complete_runner --model PATH --prompt-ids N,N --output PATH\n";
    return 2;
  }

  llama_backend_init();
  std::string model_path = options.model;
#ifdef AME_STREAMING
  if (options.manifest.empty() || options.sha.size() != 64) {
    std::cerr << "streaming runner requires --manifest and a 64-hex --sha\n";
    llama_backend_free();
    return 2;
  }
  std::error_code size_error;
  const auto bytes = std::filesystem::file_size(options.model, size_error);
  if (size_error) {
    llama_backend_free();
    return 1;
  }
  std::string error;
  const ame::backend::ExpectedIdentity expected{options.sha, bytes};
  auto artifact = ame::backend::Artifact::open(options.model, expected, error);
  if (!artifact.ok()) {
    std::cerr << "artifact: " << error << '\n';
    llama_backend_free();
    return 1;
  }
  if (!ame::backend::install_moe_stream(*artifact.value(), options.manifest, options.slots,
                                        128ULL * 1024ULL * 1024ULL)) {
    std::cerr << "install: " << ame::backend::moe_stream_stats().error << '\n';
    llama_backend_free();
    return 1;
  }
  model_path = artifact.value()->proc_path();
#endif

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
  if (model == nullptr) {
    std::cerr << "load failed\n";
    if (!options.refusal_output.empty()) {
      std::ofstream refusal(options.refusal_output, std::ios::binary | std::ios::trunc);
      refusal << "{\"status\": \"refused\", \"stage\": \"model_load\", "
                 "\"reason\": \"missing_or_invalid_metadata\"}\n";
    }
#ifdef AME_STREAMING
    ame::backend::uninstall_moe_stream();
#endif
    llama_backend_free();
    return 1;
  }
#ifdef AME_STREAMING
  {
    std::string contract_error;
    if (!ame::backend::bind_model_contract(model, contract_error)) {
      std::cerr << "contract: " << contract_error << '\n';
      llama_model_free(model);
      ame::backend::uninstall_moe_stream();
      llama_backend_free();
      return 1;
    }
  }
#endif
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const std::int32_t n_vocab = llama_vocab_n_tokens(vocab);
  for (const auto id : options.prompt_ids) {
    if (id >= n_vocab) {
      std::cerr << "prompt token outside vocab\n";
      llama_model_free(model);
#ifdef AME_STREAMING
      ame::backend::uninstall_moe_stream();
#endif
      llama_backend_free();
      return 2;
    }
  }
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = options.context;
  context_params.n_batch = options.batch;
  context_params.n_ubatch = options.ubatch;
  context_params.n_threads = static_cast<int32_t>(options.threads);
  context_params.n_threads_batch = static_cast<int32_t>(options.threads);
  llama_context* context = llama_init_from_model(model, context_params);
  if (context == nullptr) {
    std::cerr << "context failed\n";
    llama_model_free(model);
#ifdef AME_STREAMING
    ame::backend::uninstall_moe_stream();
#endif
    llama_backend_free();
    return 1;
  }

  std::vector<std::vector<float>> logits;
  std::vector<std::int32_t> generated;
  llama_batch batch = llama_batch_init(static_cast<std::int32_t>(options.batch), 0, 1);
  auto decode_batch = [&](const std::vector<std::int32_t>& ids, std::int32_t position,
                          bool all_logits) -> bool {
    batch.n_tokens = 0;
    for (std::size_t index = 0; index < ids.size(); ++index) {
      batch.token[batch.n_tokens] = ids[index];
      batch.pos[batch.n_tokens] = position + static_cast<std::int32_t>(index);
      batch.n_seq_id[batch.n_tokens] = 1;
      batch.seq_id[batch.n_tokens][0] = 0;
      batch.logits[batch.n_tokens] = (all_logits || index + 1 == ids.size()) ? 1 : 0;
      ++batch.n_tokens;
    }
    if (llama_decode(context, batch) != 0) {
      return false;
    }
    if (all_logits) {
      for (std::int32_t index = 0; index < batch.n_tokens; ++index) {
        const float* row = llama_get_logits_ith(context, index);
        if (row == nullptr) {
          return false;
        }
        logits.emplace_back(row, row + n_vocab);
      }
    } else {
      const float* row = llama_get_logits_ith(context, batch.n_tokens - 1);
      if (row == nullptr) {
        return false;
      }
      logits.emplace_back(row, row + n_vocab);
    }
    return true;
  };

  if (!decode_batch(options.prompt_ids, 0, true)) {
    std::cerr << "prefill failed\n";
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
#ifdef AME_STREAMING
    ame::backend::uninstall_moe_stream();
#endif
    llama_backend_free();
    return 1;
  }
  std::int32_t position = static_cast<std::int32_t>(options.prompt_ids.size());
  for (std::uint32_t step = 0; step < options.predict; ++step) {
    const auto& row = logits.back();
    std::int32_t best = 0;
    for (std::int32_t id = 1; id < n_vocab; ++id) {
      if (row[static_cast<std::size_t>(id)] > row[static_cast<std::size_t>(best)]) {
        best = id;
      }
    }
    generated.push_back(best);
    if (step + 1 == options.predict) {
      break;
    }
    if (!decode_batch({best}, position++, false)) {
      std::cerr << "decode failed\n";
      llama_batch_free(batch);
      llama_free(context);
      llama_model_free(model);
#ifdef AME_STREAMING
      ame::backend::uninstall_moe_stream();
#endif
      llama_backend_free();
      return 1;
    }
  }

  if (!write_logits(options.output, logits)) {
    std::cerr << "cannot write logits\n";
    return 1;
  }
  if (!options.tokens_output.empty() && !write_tokens(options.tokens_output, generated)) {
    std::cerr << "cannot write tokens\n";
    return 1;
  }
  std::cout << "{\"prompt_tokens\": " << options.prompt_ids.size()
            << ", \"logit_rows\": " << logits.size()
            << ", \"generated\": [";
  for (std::size_t index = 0; index < generated.size(); ++index) {
    if (index != 0) std::cout << ", ";
    std::cout << generated[index];
  }
  std::cout << "]";
#ifdef AME_STREAMING
  const auto stats = ame::backend::moe_stream_stats();
  std::cout << ", \"streaming\": {\"slots\": " << stats.slots
            << ", \"hits\": " << stats.hits << ", \"misses\": " << stats.misses
            << ", \"evictions\": " << stats.evictions
            << ", \"reads\": " << stats.reads << ", \"read_bytes\": " << stats.read_bytes
            << ", \"materialize_calls\": " << stats.materialize_calls
            << ", \"max_union\": " << stats.max_union
            << ", \"arena_bytes\": " << stats.arena_bytes
            << ", \"contract_routed\": " << stats.contract_routed
            << ", \"contract_shared\": " << stats.contract_shared
            << ", \"failed\": " << (stats.failed ? "true" : "false") << "}";
#endif
  std::cout << "}\n";

  llama_batch_free(batch);
  llama_free(context);
  llama_model_free(model);
#ifdef AME_STREAMING
  ame::backend::uninstall_moe_stream();
#endif
  llama_backend_free();
  return 0;
}