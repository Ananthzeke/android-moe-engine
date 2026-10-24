// A minimal llama.cpp baseline harness for project-owned evidence.
//
// It keeps model execution and tokenization in the pinned backend and records the
// exact token IDs, timing boundaries, and process-observation memory samples that
// the project validation contract requires. It does not stream or bound experts.

#include <csignal>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "artifact.hpp"
#include "llama.h"
#include "moe_stream.hpp"
#include "prefill_plan.hpp"
#include "store.hpp"

namespace {

std::atomic<bool> g_cancelled{false};

void handle_signal(int) {
  g_cancelled.store(true, std::memory_order_relaxed);
  ame::backend::request_moe_stream_cancel();
}

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
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

struct StatusSample {
  std::optional<std::uint64_t> rss_kib;
  std::optional<std::uint64_t> hwm_kib;
  std::optional<std::uint64_t> threads;
};

std::optional<std::uint64_t> status_field(const char* path, const char* key) {
  std::ifstream stream(path);
  if (!stream) {
    return std::nullopt;
  }
  std::string line;
  const std::string prefix = std::string(key) + ":";
  while (std::getline(stream, line)) {
    if (line.rfind(prefix, 0) != 0) {
      continue;
    }
    std::string_view value(line);
    value.remove_prefix(prefix.size());
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
      value.remove_prefix(1);
    }
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec == std::errc{}) {
      return parsed;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

StatusSample sample_status() {
  StatusSample sample;
  sample.rss_kib = status_field("/proc/self/status", "VmRSS");
  sample.hwm_kib = status_field("/proc/self/status", "VmHWM");
  sample.threads = status_field("/proc/self/status", "Threads");
  return sample;
}

std::optional<std::uint64_t> mem_available_kib() {
  std::ifstream stream("/proc/meminfo");
  if (!stream) {
    return std::nullopt;
  }
  std::string line;
  while (std::getline(stream, line)) {
    if (line.rfind("MemAvailable:", 0) != 0) {
      continue;
    }
    std::string_view value(line);
    value.remove_prefix(std::string_view("MemAvailable:").size());
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
      value.remove_prefix(1);
    }
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec == std::errc{}) {
      return parsed;
    }
    return std::nullopt;
  }
  return std::nullopt;
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
  void double_value(double value) {
    element();
    if (!std::isfinite(value)) {
      out_ += "null";
      return;
    }
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

void write_optional_unsigned(JsonWriter& json, const std::optional<std::uint64_t>& value) {
  if (value.has_value()) {
    json.unsigned_value(*value);
  } else {
    json.null_value();
  }
}

void write_optional_string(JsonWriter& json, const std::optional<std::string>& value) {
  if (value.has_value()) {
    json.string_value(*value);
  } else {
    json.null_value();
  }
}

struct Options {
  std::string model;
  std::string prompt;
  std::string prompt_file;
  std::string output;
  std::string artifact_sha;
  std::string binary_sha;
  std::string backend_commit;
  std::string run_id;
  std::string tag = "host";
  std::uint32_t max_new_tokens = 32;
  std::uint32_t cancel_after_ms = 0;
  std::uint32_t context = 512;
  std::uint32_t threads = 4;
  std::uint32_t batch_size = 0;
  std::uint32_t seed = 1234;
  double temperature = 0.0;
  std::uint32_t top_k = 40;
  double top_p = 0.95;
  bool mmap = true;
  bool chat = false;
  bool include_prompt_text = true;
  bool stream = false;
  bool cancel_before_decode = false;
  std::string logits_output;
  std::string manifest;
  std::uint32_t moe_slots = 64;
  std::uint32_t moe_arena_mib = 512;
  ame::backend::StreamOptions stream_options;
};

struct ParseResult {
  bool ok = true;
  std::string error;
};

bool parse_unsigned(std::string_view text, std::uint64_t& value) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_double(std::string_view text, double& value) {
  std::string copy(text);
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(copy.c_str(), &end);
  if (errno != 0 || end == copy.c_str() || *end != '\0') {
    return false;
  }
  value = parsed;
  return true;
}

ParseResult parse_options(int argc, char* argv[], Options& options) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--cancel-before-decode") {
      options.cancel_before_decode = true;
      continue;
    }
    if (argument == "--chat") {
      options.chat = true;
      continue;
    }
    if (argument == "--no-mmap") {
      options.mmap = false;
      continue;
    }
    if (argument == "--mmap") {
      options.mmap = true;
      continue;
    }
    if (argument == "--no-prompt-text") {
      options.include_prompt_text = false;
      continue;
    }
    if (argument == "--stream") {
      options.stream = true;
      continue;
    }
    if (argument == "--readers" && index + 1 < argc) {
      const std::string_view text = argv[++index];
      unsigned number = 0;
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), number);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || number < 1 ||
          number > 4) {
        return {false, "--readers must be 1 to 4"};
      }
      options.stream_options.readers = number;
      continue;
    }
    if (argument == "--overlap" && index + 1 < argc) {
      const std::string_view text = argv[++index];
      if (text != "topk" && text != "off") {
        return {false, "--overlap must be topk or off"};
      }
      options.stream_options.overlap = text == "topk";
      continue;
    }
    if (argument == "--fadvise-random") {
      options.stream_options.fadvise_random = true;
      continue;
    }
    if (argument == "--fadvise-dontneed") {
      options.stream_options.fadvise_dontneed = true;
      continue;
    }
    if ((argument == "--cache" || argument == "--eviction" || argument == "--layer-slots" ||
         argument == "--lfu-halve") &&
        index + 1 < argc) {
      const std::string_view text = argv[++index];
      if (argument == "--cache" && (text == "legacy" || text == "layered")) {
        options.stream_options.layout = text == "layered" ? ame::backend::CacheLayout::Layered
                                                          : ame::backend::CacheLayout::Legacy;
      } else if (argument == "--eviction" && (text == "lru" || text == "lfu")) {
        options.stream_options.eviction =
            text == "lfu" ? ame::backend::Eviction::Lfu : ame::backend::Eviction::Lru;
      } else if (argument == "--layer-slots" || argument == "--lfu-halve") {
        std::uint32_t number = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
          return {false, "Invalid number for " + std::string(argument)};
        }
        (argument == "--layer-slots" ? options.stream_options.layer_slots
                                     : options.stream_options.lfu_halve_every) = number;
      } else {
        return {false, "Invalid value for " + std::string(argument)};
      }
      continue;
    }
    if (argument == "--help") {
      return {false, "help"};
    }
    const bool needs_value = argument == "--model" || argument == "--prompt" ||
                             argument == "--prompt-file" || argument == "--output" ||
                             argument == "--artifact-sha" || argument == "--binary-sha" ||
                             argument == "--backend-commit" || argument == "--run-id" ||
                             argument == "--tag" || argument == "--max-new-tokens" ||
                             argument == "--context" || argument == "--threads" ||
                             argument == "--batch-size" || argument == "--seed" ||
                             argument == "--temperature" || argument == "--top-k" ||
                             argument == "--top-p" || argument == "--manifest" ||
                             argument == "--moe-slots" || argument == "--moe-arena-mib" ||
                             argument == "--cancel-after-ms" || argument == "--logits-output";
    if (!needs_value || index + 1 >= argc) {
      return {false, "Unsupported or incomplete argument: " + std::string(argument)};
    }
    const std::string value = argv[++index];
    if (argument == "--model") {
      options.model = value;
    } else if (argument == "--prompt") {
      options.prompt = value;
    } else if (argument == "--prompt-file") {
      options.prompt_file = value;
    } else if (argument == "--output") {
      options.output = value;
    } else if (argument == "--artifact-sha") {
      options.artifact_sha = value;
    } else if (argument == "--binary-sha") {
      options.binary_sha = value;
    } else if (argument == "--backend-commit") {
      options.backend_commit = value;
    } else if (argument == "--manifest") {
      options.manifest = value;
    } else if (argument == "--run-id") {
      options.run_id = value;
    } else if (argument == "--tag") {
      options.tag = value;
    } else if (argument == "--logits-output") {
      options.logits_output = value;
    } else {
      std::uint64_t number = 0;
      if (argument == "--temperature" || argument == "--top-p") {
        double parsed = 0.0;
        if (!parse_double(value, parsed)) {
          return {false, "Invalid numeric value for " + std::string(argument)};
        }
        if (argument == "--temperature") {
          options.temperature = parsed;
        } else {
          options.top_p = parsed;
        }
      } else {
        if (!parse_unsigned(value, number) || number > UINT32_MAX) {
          return {false, "Invalid numeric value for " + std::string(argument)};
        }
        const auto narrowed = static_cast<std::uint32_t>(number);
        if (argument == "--max-new-tokens") {
          options.max_new_tokens = narrowed;
        } else if (argument == "--context") {
          options.context = narrowed;
        } else if (argument == "--threads") {
          options.threads = narrowed;
        } else if (argument == "--batch-size") {
          options.batch_size = narrowed;
        } else if (argument == "--seed") {
          options.seed = narrowed;
        } else if (argument == "--moe-slots") {
          options.moe_slots = narrowed;
        } else if (argument == "--moe-arena-mib") {
          options.moe_arena_mib = narrowed;
        } else if (argument == "--cancel-after-ms") {
          options.cancel_after_ms = narrowed;
        } else {
          options.top_k = narrowed;
        }
      }
    }
  }
  if (options.model.empty()) {
    return {false, "--model is required"};
  }
  if (options.prompt.empty() == options.prompt_file.empty()) {
    return {false, "Specify exactly one of --prompt or --prompt-file"};
  }
  if (options.max_new_tokens == 0 || options.context < 8) {
    return {false, "Invalid token limits"};
  }
  if (options.threads == 0) {
    return {false, "--threads must be positive"};
  }
  if (options.stream && options.manifest.empty()) {
    return {false, "--stream requires --manifest"};
  }
  if (!options.prompt_file.empty()) {
    std::ifstream stream(options.prompt_file, std::ios::binary);
    if (!stream) {
      return {false, "Cannot read prompt file"};
    }
    options.prompt.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
  }
  return {};
}

struct Sample {
  std::optional<std::uint64_t> rss_kib;
  std::optional<std::uint64_t> hwm_kib;
  std::optional<std::uint64_t> threads;
  std::optional<std::uint64_t> mem_available_kib;
};

Sample take_sample() {
  const StatusSample status = sample_status();
  return Sample{status.rss_kib, status.hwm_kib, status.threads, mem_available_kib()};
}

bool write_text(const std::string& path, const std::string& contents, std::string& error) {
  if (path.empty()) {
    std::cout << contents;
    return std::cout.good();
  }
  const std::string temporary = path + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      error = "cannot open temporary output";
      return false;
    }
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.flush();
    if (!stream.good()) {
      error = "cannot write temporary output";
      std::filesystem::remove(temporary);
      return false;
    }
  }
  std::error_code code;
  std::filesystem::rename(temporary, path, code);
  if (code) {
    error = code.message();
    std::filesystem::remove(temporary);
    return false;
  }
  return true;
}

bool emit_status(const Options& options, const char* status, const char* reason,
                  const std::string& detail) {
  JsonWriter writer;
  writer.begin_object();
  writer.key("schema_version");
  writer.unsigned_value(1);
  writer.key("status");
  writer.string_value(status);
  writer.key("observed_at");
  writer.string_value(utc_now());
  writer.key("run");
  writer.begin_object();
  writer.key("id");
  writer.string_value(options.run_id);
  writer.key("tag");
  writer.string_value(options.tag);
  writer.end_object();
  writer.key("model");
  writer.begin_object();
  writer.key("path");
  writer.string_value(options.model);
  writer.key("artifact_sha256");
  write_optional_string(writer, options.artifact_sha.empty()
                                    ? std::nullopt
                                    : std::optional<std::string>(options.artifact_sha));
  writer.end_object();
  writer.key("finish_reason");
  writer.string_value(reason);
  writer.key("error");
  writer.string_value(detail);
  writer.end_object();
  std::string error;
  return write_text(options.output, writer.text() + "\n", error);
}


std::optional<std::string> read_chat_prompt(const llama_model* model, const std::string& prompt,
                                            std::string& error) {
  const char* chat_template = llama_model_chat_template(model, nullptr);
  if (chat_template == nullptr) {
    error = "model has no chat template";
    return std::nullopt;
  }
  const llama_chat_message message{"user", prompt.c_str()};
  std::vector<char> buffer(prompt.size() * 2 + 256);
  for (int attempt = 0; attempt < 2; ++attempt) {
    const int32_t needed = llama_chat_apply_template(chat_template, &message, 1, true,
                                                     buffer.data(),
                                                     static_cast<int32_t>(buffer.size()));
    if (needed < 0) {
      error = "chat template application failed";
      return std::nullopt;
    }
    if (static_cast<std::size_t>(needed) <= buffer.size()) {
      return std::string(buffer.data(), static_cast<std::size_t>(needed));
    }
    buffer.resize(static_cast<std::size_t>(needed) + 1);
  }
  error = "chat template buffer growth failed";
  return std::nullopt;
}

}  // namespace

int main(int argc, char* argv[]) {
  const Clock::time_point program_start = Clock::now();
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  Options options;
  const ParseResult parsed = parse_options(argc, argv, options);
  if (!parsed.ok) {
    if (parsed.error == "help") {
      std::cout << "usage: llama_baseline --model PATH (--prompt TEXT | --prompt-file PATH)\n"
                   "       [--chat] [--max-new-tokens N] [--context N] [--threads N]\n"
                   "       [--batch-size N] [--seed N] [--temperature F] [--top-k N]\n"
                   "       [--top-p F] [--no-mmap] [--artifact-sha HEX] [--output PATH]\n"
                   "       [--backend-commit SHA] [--binary-sha HEX] [--run-id ID]\n"
                   "       [--tag TAG] [--no-prompt-text] [--stream --manifest PATH]\n"
                   "       [--moe-slots N] [--moe-arena-mib N] [--cache legacy|layered]\n"
                   "       [--eviction lru|lfu] [--layer-slots N] [--lfu-halve N]\n"
                   "       [--fadvise-random] [--fadvise-dontneed] [--readers 1-4] [--overlap topk|off]\n"
                   "       [--cancel-after-ms N]\n";
      return 0;
    }
    std::cerr << parsed.error << '\n';
    return 2;
  }

  if (options.cancel_after_ms != 0) {
    std::thread([milliseconds = options.cancel_after_ms] {
      std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
      g_cancelled.store(true, std::memory_order_relaxed);
      ame::backend::request_moe_stream_cancel();
    }).detach();
  }

  JsonWriter json;
  json.begin_object();
  json.key("schema_version");
  json.unsigned_value(1);
  json.key("observed_at");
  json.string_value(utc_now());
  json.key("run");
  json.begin_object();
  json.key("id");
  json.string_value(options.run_id);
  json.key("tag");
  json.string_value(options.tag);
  json.key("backend");
  json.begin_object();
  json.key("name");
  json.string_value("llama.cpp");
  json.key("version");
  json.string_value(llama_version());
  json.key("commit");
  json.string_value(options.backend_commit);
  json.key("binary_sha256");
  write_optional_string(json, options.binary_sha.empty()
                                   ? std::nullopt
                                   : std::optional<std::string>(options.binary_sha));
  json.end_object();
  json.end_object();

  json.key("model");
  json.begin_object();
  json.key("path");
  json.string_value(options.model);
  json.key("artifact_sha256");
  write_optional_string(json, options.artifact_sha.empty()
                                   ? std::nullopt
                                   : std::optional<std::string>(options.artifact_sha));
  std::error_code size_error;
  const auto file_size = std::filesystem::file_size(options.model, size_error);
  json.key("file_bytes");
  if (!size_error) {
    json.unsigned_value(file_size);
  } else {
    json.null_value();
  }
  json.end_object();

  llama_backend_init();

  std::unique_ptr<ame::backend::Artifact> artifact;
  std::string model_load_path = options.model;
  if (options.stream) {
    bool digest_ok = options.artifact_sha.size() == 64;
    for (const char character : options.artifact_sha) {
      if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) {
        digest_ok = false;
      }
    }
    std::error_code artifact_size_error;
    const auto artifact_bytes = std::filesystem::file_size(options.model, artifact_size_error);
    if (!digest_ok || artifact_size_error) {
      std::cerr << "streaming requires --artifact-sha with a lowercase 64-hex digest\n";
      (void)emit_status(options, "refused", "artifact_identity_missing",
                        "streaming requires the enrolled artifact SHA-256");
      llama_backend_free();
      return 0;
    }
    std::string artifact_error;
    const ame::backend::ExpectedIdentity expected{options.artifact_sha, artifact_bytes};
    auto opened = ame::backend::Artifact::open(options.model, expected, artifact_error);
    if (!opened.ok()) {
      std::cerr << "artifact verification failed: " << artifact_error << '\n';
      (void)emit_status(options, "failed", "artifact_identity", artifact_error);
      llama_backend_free();
      return 1;
    }
    artifact = std::move(opened).value();
    model_load_path = artifact->proc_path();
    if (const char* replacement = std::getenv("AME_TEST_REPLACE_MODEL_AFTER_ARTIFACT");
        replacement != nullptr && replacement[0] != '\0') {
      std::error_code replace_error;
      std::filesystem::rename(replacement, options.model, replace_error);
      if (replace_error) {
        std::cerr << "test replacement failed\n";
        (void)emit_status(options, "failed", "test_replacement", "cannot replace model path");
        llama_backend_free();
        return 1;
      }
    }
    const std::uint64_t arena_bytes =
        static_cast<std::uint64_t>(options.moe_arena_mib) * 1024ULL * 1024ULL;
    if (!ame::backend::install_moe_stream(*artifact, options.manifest, options.moe_slots,
                                          arena_bytes, options.stream_options)) {
      const auto stats = ame::backend::moe_stream_stats();
      std::cerr << "moe stream install failed: " << stats.error << '\n';
      (void)emit_status(options, "failed", "manifest_layout", stats.error);
      llama_backend_free();
      return 1;
    }
  }

  const Clock::time_point load_start = Clock::now();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  model_params.load_mode =
      options.mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
  model_params.lazy_mode =
      options.mmap ? LLAMA_LAZY_MODE_AUTO : LLAMA_LAZY_MODE_OFF;
  model_params.check_tensors = false;
  model_params.use_extra_bufts = !options.stream;

  llama_model* model = llama_model_load_from_file(model_load_path.c_str(), model_params);
  if (model == nullptr) {
    std::cerr << "model load failed\n";
    (void)emit_status(options, "error", "error", "model load failed");
    llama_backend_free();
    return 1;
  }
  if (options.stream) {
    std::string contract_error;
    if (!ame::backend::bind_model_contract(model, contract_error)) {
      std::cerr << "contract: " << contract_error << '\n';
      (void)emit_status(options, "failed", "contract", contract_error);
      llama_model_free(model);
      ame::backend::uninstall_moe_stream();
      llama_backend_free();
      return 1;
    }
  }
  const Clock::time_point model_loaded = Clock::now();

  std::uint32_t expert_count = 0;
  std::uint32_t expert_used = 0;
  std::uint32_t prefill_unit = options.batch_size == 0 ? options.context : options.batch_size;
  if (options.stream) {
    std::string plan_error;
    if (!ame::backend::plan_model_prefill(
            options.batch_size == 0 ? options.context : options.batch_size,
            ame::residency::kMaxLeaseExperts, prefill_unit, expert_count, expert_used,
            plan_error)) {
      std::cerr << "prefill plan: " << plan_error << '\n';
      (void)emit_status(options, "refused", "contract_prefill_refused", plan_error);
      llama_model_free(model);
      ame::backend::uninstall_moe_stream();
      llama_backend_free();
      return 0;
    }
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = options.context;
  context_params.n_batch = options.batch_size == 0 ? options.context : options.batch_size;
  context_params.n_ubatch = prefill_unit;
  context_params.n_threads = static_cast<int32_t>(options.threads);
  context_params.n_threads_batch = static_cast<int32_t>(options.threads);
  context_params.no_perf = true;
  context_params.moe_cache_size = 0;
  context_params.abort_callback = [](void*) -> bool {
    return g_cancelled.load() || ame::backend::moe_stream_failed() ||
           ame::backend::moe_stream_cancelled();
  };
  context_params.abort_callback_data = nullptr;

  if (options.stream && options.stream_options.overlap) {
    context_params.cb_eval = ame::backend::moe_eval_callback;
    context_params.cb_eval_user_data = nullptr;
  }
  llama_context* context = llama_init_from_model(model, context_params);
  if (context == nullptr) {
    std::cerr << "context creation failed\n";
    (void)emit_status(options, "error", "error", "context creation failed");
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  const Clock::time_point context_ready = Clock::now();
  if (options.stream && llama_n_ubatch(context) > prefill_unit) {
    std::cerr << "backend rounded n_ubatch above the prefill bound\n";
    (void)emit_status(options, "refused", "prefill_rounding",
                      "backend effective microbatch exceeds the conservative union bound");
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
  }
  const Sample after_load = take_sample();

  char description[256] = {};
  llama_model_desc(model, description, sizeof(description));
  const llama_vocab* vocab = llama_model_get_vocab(model);

  std::string prompt = options.prompt;
  bool chat_applied = false;
  if (options.chat) {
    std::string chat_error;
    const auto formatted = read_chat_prompt(model, prompt, chat_error);
    if (!formatted.has_value()) {
      std::cerr << chat_error << '\n';
      (void)emit_status(options, "error", "error", chat_error);
      llama_free(context);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    prompt = *formatted;
    chat_applied = true;
  }

  json.key("model_info");
  json.begin_object();
  json.key("description");
  json.string_value(description);
  json.key("parameters");
  json.unsigned_value(llama_model_n_params(model));
  json.key("tensor_bytes");
  json.unsigned_value(llama_model_size(model));
  json.key("n_ctx_train");
  json.unsigned_value(static_cast<std::uint64_t>(llama_model_n_ctx_train(model)));
  json.key("n_embd");
  json.unsigned_value(static_cast<std::uint64_t>(llama_model_n_embd(model)));
  json.key("n_layer");
  json.unsigned_value(static_cast<std::uint64_t>(llama_model_n_layer(model)));
  json.end_object();

  json.key("options");
  json.begin_object();
  json.key("context_tokens");
  json.unsigned_value(options.context);
  json.key("max_new_tokens");
  json.unsigned_value(options.max_new_tokens);
  json.key("threads");
  json.unsigned_value(options.threads);
  json.key("batch_size");
  json.unsigned_value(context_params.n_batch);
  json.key("seed");
  json.unsigned_value(options.seed);
  json.key("temperature");
  json.double_value(options.temperature);
  json.key("top_k");
  json.unsigned_value(options.top_k);
  json.key("top_p");
  json.double_value(options.top_p);
  json.key("mmap");
  json.bool_value(options.mmap);
  json.key("chat_template");
  json.bool_value(chat_applied);
  json.key("gpu_layers");
  json.unsigned_value(0);
  json.key("expert_count");
  if (options.stream) {
    json.unsigned_value(expert_count);
  } else {
    json.null_value();
  }
  json.key("expert_used_count");
  if (options.stream) {
    json.unsigned_value(expert_used);
  } else {
    json.null_value();
  }
  json.key("prefill_unit_tokens");
  json.unsigned_value(prefill_unit);
  json.key("effective_ubatch_tokens");
  json.unsigned_value(static_cast<std::uint64_t>(llama_n_ubatch(context)));
  json.end_object();

  json.key("prompt");
  json.begin_object();
  if (options.include_prompt_text) {
    json.key("text");
    json.string_value(prompt);
  } else {
    json.key("text");
    json.null_value();
  }
  json.end_object();

  const Clock::time_point generate_start = Clock::now();
  const Clock::time_point tokenize_start = Clock::now();
  const int32_t size_needed = llama_tokenize(vocab, prompt.c_str(),
                                             static_cast<int32_t>(prompt.size()), nullptr, 0, true,
                                             true);
  if (size_needed == INT32_MIN) {
    std::cerr << "tokenization overflow\n";
    (void)emit_status(options, "error", "error", "tokenization overflow");
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  const std::size_t token_capacity =
      size_needed < 0 ? static_cast<std::size_t>(-size_needed) : static_cast<std::size_t>(size_needed);
  std::vector<llama_token> prompt_tokens(token_capacity + 1);
  const int32_t prompt_count =
      llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                     prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()), true, true);
  const Clock::time_point tokenize_end = Clock::now();
  if (prompt_count <= 0) {
    std::cerr << "tokenization failed\n";
    (void)emit_status(options, "error", "error", "tokenization failed");
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  prompt_tokens.resize(static_cast<std::size_t>(prompt_count));

  json.key("prompt_token_count");
  json.unsigned_value(static_cast<std::uint64_t>(prompt_tokens.size()));
  json.key("prompt_token_ids");
  json.begin_array();
  for (const llama_token token : prompt_tokens) {
    json.signed_value(token);
  }
  json.end_array();

  const std::uint64_t prompt_and_output =
      static_cast<std::uint64_t>(prompt_count) + options.max_new_tokens;
  if (prompt_and_output > options.context) {
    (void)emit_status(options, "refused", "context_limit",
                      "prompt plus requested output exceeds context");
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
  }

  llama_batch batch = llama_batch_init(static_cast<int32_t>(context_params.n_batch), 0, 1);
  std::vector<std::vector<float>> logit_rows;
  const char* trace_phase = "prefill";
  auto decode_chunk = [&](const std::vector<llama_token>& tokens, int32_t position) -> bool {
    if (tokens.empty()) {
      return true;
    }
    if (options.stream) {
      ame::backend::label_moe_graph(trace_phase, static_cast<std::uint32_t>(tokens.size()));
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
      batch.token[index] = tokens[index];
      batch.pos[index] = position + static_cast<int32_t>(index);
      batch.n_seq_id[index] = 1;
      batch.seq_id[index][0] = 0;
      batch.logits[index] = index + 1 == tokens.size() ? 1 : 0;
    }
    if (llama_decode(context, batch) != 0) {
      return false;
    }
    if (!options.logits_output.empty()) {
      const float* row = llama_get_logits_ith(context, batch.n_tokens - 1);
      if (row == nullptr) {
        return false;
      }
      logit_rows.emplace_back(row, row + llama_vocab_n_tokens(vocab));
    }
    return true;
  };

  const Clock::time_point prefill_start = Clock::now();
  bool prefill_ok = true;
  for (std::size_t offset = 0; offset < prompt_tokens.size();) {
    const std::size_t remaining = prompt_tokens.size() - offset;
    const std::size_t chunk =
        std::min<std::size_t>(remaining, static_cast<std::size_t>(prefill_unit));
    std::vector<llama_token> piece(prompt_tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                                   prompt_tokens.begin() +
                                       static_cast<std::ptrdiff_t>(offset + chunk));
    if (!decode_chunk(piece, static_cast<int32_t>(offset))) {
      prefill_ok = false;
      break;
    }
    offset += chunk;
  }
  const Clock::time_point prefill_end = Clock::now();
  if (!prefill_ok) {
    const bool stream_failed = options.stream && ame::backend::moe_stream_failed();
    const bool stream_cancelled = options.stream && ame::backend::moe_stream_cancelled();
    const bool cancelled = g_cancelled.load() || stream_cancelled;
    const std::string error = stream_failed ? ame::backend::moe_stream_error()
                                            : (cancelled ? "cancelled" : "prefill decode failed");
    (void)emit_status(options, stream_failed ? "failed" : (cancelled ? "cancelled" : "error"),
                      stream_failed ? "materialization_failure"
                                    : (cancelled ? "cancelled" : "error"),
                      error);
    std::cerr << error << '\n';
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return stream_failed ? 1 : (cancelled ? 0 : 1);
  }
  const Sample after_prefill = take_sample();
  if (options.cancel_before_decode) {
    g_cancelled.store(true, std::memory_order_relaxed);
  }

  llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
  llama_sampler* sampler = llama_sampler_chain_init(sampler_params);
  if (options.temperature <= 0.0) {
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  } else {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(static_cast<int32_t>(options.top_k)));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(options.top_p), 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(options.temperature)));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed));
  }

  std::vector<llama_token> generated;
  std::string output_text;
  std::string finish_reason = "limit";
  bool first_token_seen = false;
  Clock::time_point first_token_time{};
  Clock::time_point last_token_time{};
  double decode_seconds = 0.0;
  bool decode_failed = false;
  trace_phase = "decode";

  while (generated.size() < options.max_new_tokens) {
    if (options.stream && ame::backend::moe_stream_failed()) {
      decode_failed = true;
      finish_reason = "materialization_failure";
      break;
    }
    if (g_cancelled.load() || (options.stream && ame::backend::moe_stream_cancelled())) {
      finish_reason = "cancelled";
      break;
    }
    const llama_token token = llama_sampler_sample(sampler, context, -1);
    llama_sampler_accept(sampler, token);
    if (llama_vocab_is_eog(vocab, token)) {
      finish_reason = "eos";
      break;
    }
    generated.push_back(token);
    char piece[256] = {};
    int32_t piece_length = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
    if (piece_length < 0) {
      std::vector<char> larger(static_cast<std::size_t>(-piece_length));
      piece_length = llama_token_to_piece(vocab, token, larger.data(),
                                          static_cast<int32_t>(larger.size()), 0, false);
      if (piece_length >= 0) {
        output_text.append(larger.data(), static_cast<std::size_t>(piece_length));
      }
    } else {
      output_text.append(piece, static_cast<std::size_t>(piece_length));
    }
    const Clock::time_point now = Clock::now();
    if (!first_token_seen) {
      first_token_seen = true;
      first_token_time = now;
    }
    last_token_time = now;
    if (generated.size() == options.max_new_tokens) {
      finish_reason = "limit";
      break;
    }
    std::vector<llama_token> next{token};
    if (!decode_chunk(next, static_cast<int32_t>(prompt_count + generated.size() - 1))) {
      decode_failed = true;
      finish_reason = (options.stream && ame::backend::moe_stream_failed())
                          ? "materialization_failure"
                          : (g_cancelled.load() ? "cancelled" : "error");
      break;
    }
  }
  if (first_token_seen) {
    decode_seconds = elapsed_seconds(first_token_time, last_token_time);
  }
  const Clock::time_point generate_end = Clock::now();
  const Sample end_sample = take_sample();
  ame::backend::MoeStreamStats stream_stats{};
  if (options.stream) {
    stream_stats = ame::backend::moe_stream_stats();
  }

  if (!options.logits_output.empty()) {
    std::ofstream stream(options.logits_output, std::ios::binary | std::ios::trunc);
    const std::uint64_t rows = logit_rows.size();
    const std::uint64_t width = logit_rows.empty() ? 0 : logit_rows.front().size();
    stream.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    for (const auto& row : logit_rows) {
      stream.write(reinterpret_cast<const char*>(&width), sizeof(width));
      stream.write(reinterpret_cast<const char*>(row.data()),
                   static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    if (!stream.good()) {
      std::cerr << "cannot write logits\n";
      (void)emit_status(options, "error", "logits_output", "cannot write logits");
      llama_batch_free(batch);
      llama_free(context);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
  }

  json.key("output");
  json.begin_object();
  json.key("token_ids");
  json.begin_array();
  for (const llama_token token : generated) {
    json.signed_value(token);
  }
  json.end_array();
  json.key("text");
  json.string_value(output_text);
  json.key("tokens");
  json.unsigned_value(static_cast<std::uint64_t>(generated.size()));
  json.key("finish_reason");
  json.string_value(finish_reason);
  json.end_object();

  json.key("timing");
  json.begin_object();
  json.key("load_seconds");
  json.double_value(elapsed_seconds(load_start, context_ready));
  json.key("model_load_seconds");
  json.double_value(elapsed_seconds(load_start, model_loaded));
  json.key("context_init_seconds");
  json.double_value(elapsed_seconds(model_loaded, context_ready));
  json.key("tokenize_seconds");
  json.double_value(elapsed_seconds(tokenize_start, tokenize_end));
  json.key("prefill_seconds");
  json.double_value(elapsed_seconds(prefill_start, prefill_end));
  json.key("request_ttft_seconds");
  if (first_token_seen) {
    json.double_value(elapsed_seconds(generate_start, first_token_time));
  } else {
    json.null_value();
  }
  json.key("decode_seconds");
  json.double_value(decode_seconds);
  json.key("decode_tokens_per_second");
  if (generated.size() >= 2 && decode_seconds > 0.0) {
    json.double_value(static_cast<double>(generated.size() - 1) / decode_seconds);
  } else {
    json.null_value();
  }
  json.key("generate_seconds");
  json.double_value(elapsed_seconds(generate_start, generate_end));
  json.key("wall_seconds");
  json.double_value(elapsed_seconds(program_start, generate_end));
  json.end_object();

  json.key("memory");
  json.begin_object();
  json.key("rss_kib_after_load");
  write_optional_unsigned(json, after_load.rss_kib);
  json.key("rss_kib_after_prefill");
  write_optional_unsigned(json, after_prefill.rss_kib);
  json.key("rss_kib_end");
  write_optional_unsigned(json, end_sample.rss_kib);
  json.key("vm_hwm_kib_end");
  write_optional_unsigned(json, end_sample.hwm_kib);
  json.key("threads_end");
  write_optional_unsigned(json, end_sample.threads);
  json.key("mem_available_kib_after_load");
  write_optional_unsigned(json, after_load.mem_available_kib);
  json.key("mem_available_kib_after_prefill");
  write_optional_unsigned(json, after_prefill.mem_available_kib);
  json.key("mem_available_kib_end");
  write_optional_unsigned(json, end_sample.mem_available_kib);
  json.key("project_owned_bytes");
  json.null_value();
  json.key("scope");
  json.string_value(
      "process RSS/VmHWM samples and system MemAvailable; backend allocation coverage not "
      "instrumented");
  json.end_object();

  json.key("cancellation");
  json.begin_object();
  json.key("requested");
  json.bool_value(g_cancelled.load());
  json.key("after_ms");
  json.unsigned_value(options.cancel_after_ms);
  json.end_object();

  const bool stream_failed = options.stream && ame::backend::moe_stream_failed();
  const bool stream_cancelled = options.stream && ame::backend::moe_stream_cancelled();
  const bool cancelled = g_cancelled.load() || stream_cancelled;
  json.key("status");
  if (stream_failed) {
    json.string_value("failed");
  } else if (cancelled) {
    json.string_value("cancelled");
  } else if (decode_failed) {
    json.string_value("error");
  } else {
    json.string_value("ok");
  }
  if (stream_failed || (decode_failed && !cancelled)) {
    json.key("error");
    json.string_value(stream_failed ? ame::backend::moe_stream_error() : "decode failed");
  }
  json.key("finish_reason");
  json.string_value(finish_reason);
  json.key("streaming");
  json.begin_object();
  json.key("installed");
  json.bool_value(options.stream);
  json.key("manifest");
  if (options.manifest.empty()) {
    json.null_value();
  } else {
    json.string_value(options.manifest);
  }
  json.key("slots");
  json.unsigned_value(stream_stats.slots);
  json.key("cache_layout");
  json.string_value(stream_stats.layered ? "layered" : "legacy");
  json.key("eviction");
  json.string_value(options.stream_options.eviction == ame::backend::Eviction::Lfu ? "lfu" : "lru");
  json.key("layer_slots");
  json.unsigned_value(stream_stats.layer_slots);
  json.key("readers");
  json.unsigned_value(options.stream_options.readers);
  json.key("overlap");
  json.string_value(options.stream_options.overlap ? "topk" : "off");
  json.key("prefetches_started");
  json.unsigned_value(stream_stats.prefetches_started);
  json.key("prefetches_consumed");
  json.unsigned_value(stream_stats.prefetches_consumed);
  json.key("prefetch_fallbacks");
  json.unsigned_value(stream_stats.prefetch_fallbacks);
  json.key("read_wait_seconds");
  json.double_value(stream_stats.read_wait_seconds);
  json.key("fadvise_random");
  json.bool_value(options.stream_options.fadvise_random);
  json.key("fadvise_dontneed");
  json.bool_value(options.stream_options.fadvise_dontneed);
  json.key("arena_bytes");
  json.unsigned_value(stream_stats.arena_bytes);
  json.key("cache_metadata_bytes");
  json.unsigned_value(stream_stats.cache_metadata_bytes);
  json.key("pread_calls");
  json.unsigned_value(stream_stats.pread_calls);
  json.key("pread_bytes");
  json.unsigned_value(stream_stats.pread_bytes);
  json.key("materialize_calls");
  json.unsigned_value(stream_stats.materialize_calls);
  json.key("tensor_switches");
  json.unsigned_value(stream_stats.tensor_switches);
  json.key("reads");
  json.unsigned_value(stream_stats.reads);
  json.key("read_bytes");
  json.unsigned_value(stream_stats.read_bytes);
  json.key("read_seconds");
  json.double_value(stream_stats.read_seconds);
  json.key("hits");
  json.unsigned_value(stream_stats.hits);
  json.key("misses");
  json.unsigned_value(stream_stats.misses);
  json.key("evictions");
  json.unsigned_value(stream_stats.evictions);
  json.key("max_union");
  json.unsigned_value(stream_stats.max_union);
  json.key("manifest_version");
  json.unsigned_value(stream_stats.manifest_version);
  json.key("manifest_identity_checked");
  json.bool_value(stream_stats.manifest_identity_checked);
  json.key("manifest_layout_hash");
  json.string_value(stream_stats.manifest_layout_hash);
  json.key("declared_artifact_sha256");
  json.string_value(stream_stats.declared_artifact_sha256);
  json.key("verified_artifact_sha256");
  json.string_value(stream_stats.verified_artifact_sha256);
  json.key("artifact_verified");
  json.bool_value(stream_stats.artifact_verified);
  json.key("artifact_hash_seconds");
  json.double_value(stream_stats.artifact_hash_seconds);
  json.key("failed");
  json.bool_value(stream_stats.failed);
  json.key("cancelled");
  json.bool_value(stream_stats.cancelled);
  json.end_object();

  json.key("unavailable");
  json.begin_array();
  const auto note_unavailable = [&json](std::string_view field, std::string_view reason) {
    json.begin_object();
    json.key("field");
    json.string_value(field);
    json.key("reason");
    json.string_value(reason);
    json.end_object();
  };
  if (options.artifact_sha.empty()) {
    note_unavailable("model.artifact_sha256", "not_supplied");
  }
  if (options.binary_sha.empty()) {
    note_unavailable("run.binary_sha256", "not_supplied");
  }
  if (options.backend_commit.empty()) {
    note_unavailable("run.backend.commit", "not_supplied");
  }
  if (!options.include_prompt_text) {
    note_unavailable("prompt.text", "withheld");
    note_unavailable("prompt.sha256", "not_computed_by_harness");
  }
  note_unavailable("memory.project_owned_bytes", "backend_allocation_coverage_not_instrumented");
  json.end_array();
  json.end_object();

  std::string write_error;
  const bool output_written = write_text(options.output, json.text() + "\n", write_error);
  if (!output_written) {
    std::cerr << "output write failed: " << write_error << '\n';
  }

  if (options.stream) {
    ame::backend::uninstall_moe_stream();
  }
  llama_sampler_free(sampler);
  llama_batch_free(batch);
  llama_free(context);
  llama_model_free(model);
  llama_backend_free();

  if (stream_failed || (decode_failed && !cancelled)) {
    return 1;
  }
  // Inference may have finished, but without the result file no evidence was delivered.
  return output_written ? 0 : 4;
}