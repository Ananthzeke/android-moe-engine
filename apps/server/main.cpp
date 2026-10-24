#define CPPHTTPLIB_IMPLEMENTATION
#include "chat_template.hpp"
#include "ggml-cpu.h"
#include "httplib.h"
#include "json.hpp"
#include "llama.h"
#include "request_metrics.hpp"
#include "request_scope.hpp"
#ifdef AME_STREAMING
#include "artifact.hpp"
#include "moe_stream.hpp"
#include "prefill_plan.hpp"
#endif

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Options {
  std::string model;
  std::string model_id = "local-model";
  std::string manifest;
  std::string artifact_sha;
  std::string host = "127.0.0.1";
  int port = 8080;
  int context = 2048;
  int threads = 4;
  int slots = 64;
  std::uint64_t arena_mib = 256;
  bool stream = false;
  std::string request_log;
  ame::backend::StreamOptions stream_options;
  bool reader_cpus_auto = false;  // pin readers to the cores that are not the big cores
  bool threadpool_big = false;    // one ggml threadpool pinned to the big cores
  int poll = 50;                  // threadpool polling level, 0 (none) to 100
  std::string system_root = "/";
};

std::atomic<bool> g_ready{false};
std::atomic<bool> g_failed{false};
std::string g_error;
llama_model* g_model = nullptr;
ame::server::ChatTemplate* g_chat_template = nullptr;
Options g_options;
ggml_threadpool* g_threadpool = nullptr;
std::atomic<std::uint64_t> g_counter{0};

class AdmissionQueue {
 public:
  enum class Result { Active, Full, WaitTimeout };
  Result acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (outstanding_ >= 4) {
      ++rejected_;
      return Result::Full;
    }
    ++outstanding_;
    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
    while (active_) {
      if (cv_.wait_until(lock, wait_deadline) == std::cv_status::timeout) {
        --outstanding_;
        ++queue_timeouts_;
        return Result::WaitTimeout;
      }
    }
    active_ = true;
    ++admitted_;
    return Result::Active;
  }
  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) active_ = false;
    if (outstanding_ > 0) --outstanding_;
    cv_.notify_one();
  }
  json snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    return {{"outstanding", outstanding_},
            {"active", active_},
            {"admitted", admitted_},
            {"rejected", rejected_},
            {"queue_timeouts", queue_timeouts_},
            {"wait_limit_seconds", 300},
            {"execution_limit_seconds", 300}};
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int outstanding_ = 0;
  bool active_ = false;
  std::uint64_t admitted_ = 0;
  std::uint64_t rejected_ = 0;
  std::uint64_t queue_timeouts_ = 0;
};
AdmissionQueue g_queue;
std::mutex g_snapshot_mutex;
json g_stream_snapshot = json::object();
json g_last_request = json::object();
json g_startup_record = json::object();
std::unique_ptr<ame::server::SystemSampler> g_sampler;
std::mutex g_request_log_mutex;
std::ofstream g_request_log;

void publish_stream_snapshot() {
#ifdef AME_STREAMING
  if (!g_options.stream) return;
  const auto stats = ame::backend::moe_stream_stats();
  json snapshot = {{"slots", stats.slots},
                   {"hits", stats.hits},
                   {"misses", stats.misses},
                   {"evictions", stats.evictions},
                   {"reads", stats.reads},
                   {"read_bytes", stats.read_bytes},
                   {"materialize_calls", stats.materialize_calls},
                   {"arena_bytes", stats.arena_bytes},
                   {"cache_layout", stats.layered ? "layered" : "legacy"},
                   {"layer_slots", stats.layer_slots},
                   {"cache_metadata_bytes", stats.cache_metadata_bytes},
                   {"resident_slots", stats.resident_slots},
                   {"pread_calls", stats.pread_calls},
                   {"read_wait_seconds", stats.read_wait_seconds},
                   {"prefetches_started", stats.prefetches_started},
                   {"prefetches_consumed", stats.prefetches_consumed},
                   {"prefetch_fallbacks", stats.prefetch_fallbacks},
                   {"contract_routed", stats.contract_routed},
                   {"contract_shared", stats.contract_shared},
                   {"failed", stats.failed},
                   {"cancelled", stats.cancelled},
                   {"error", stats.error.substr(0, 256)},
                   {"artifact_hash_seconds", stats.artifact_hash_seconds}};
  std::lock_guard<std::mutex> lock(g_snapshot_mutex);
  g_stream_snapshot = std::move(snapshot);
#endif
}

std::shared_ptr<ame::server::RequestScope> make_request_scope() {
  return std::make_shared<ame::server::RequestScope>(ame::server::RequestScope::Hooks{
      [] { ame::backend::release_moe_stream_leases(); }, [] { publish_stream_snapshot(); },
      [] { g_queue.release(); }});
}

std::string json_error(const std::string& message,
                       const std::string& code = "invalid_request_error") {
  json body;
  body["error"] = {{"message", message}, {"type", "invalid_request_error"}, {"code", code}};
  return body.dump();
}

void set_error(httplib::Response& res, int status, const std::string& message,
               const std::string& code = "invalid_request_error") {
  res.status = status;
  res.set_content(json_error(message, code), "application/json");
}

std::string id_for() {
  return "chatcmpl-" + std::to_string(static_cast<unsigned long>(::getpid())) + "-" +
         std::to_string(g_counter.fetch_add(1));
}

struct Request {
  std::vector<llama_token> tokens;
  int max_tokens = 128;
  double temperature = 0.0;
  double top_p = 1.0;
  std::uint32_t seed = 0;
  bool stream = false;
  bool include_usage = false;
  bool enable_thinking = false;
  bool debug = false;
  std::vector<std::string> stops;
};

std::optional<Request> parse_request(const json& body, httplib::Response& res) {
  Request request;
  if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array() ||
      body["messages"].empty()) {
    set_error(res, 400, "messages must be a nonempty array");
    return std::nullopt;
  }
  const std::string model = body.value("model", g_options.model_id);
  if (model != g_options.model_id) {
    set_error(res, 404, "unknown model: " + model, "model_not_found");
    return std::nullopt;
  }
  std::vector<common_chat_msg> messages;
  for (const auto& message : body["messages"]) {
    if (!message.is_object() || !message.contains("role") || !message.contains("content") ||
        !message["role"].is_string() || !message["content"].is_string()) {
      set_error(res, 400, "only text messages with role and content are supported");
      return std::nullopt;
    }
    const std::string role = message["role"];
    if (role != "system" && role != "user" && role != "assistant") {
      set_error(res, 400, "unsupported message role: " + role);
      return std::nullopt;
    }
    common_chat_msg owned;
    owned.role = role;
    owned.content = message["content"];
    messages.push_back(std::move(owned));
  }
  if (body.contains("enable_thinking")) {
    if (!body["enable_thinking"].is_boolean()) {
      set_error(res, 400, "enable_thinking must be a boolean");
      return std::nullopt;
    }
    request.enable_thinking = body["enable_thinking"].get<bool>();
  }
  if (body.contains("ame_debug")) {
    if (!body["ame_debug"].is_boolean()) {
      set_error(res, 400, "ame_debug must be a boolean");
      return std::nullopt;
    }
    request.debug = body["ame_debug"].get<bool>();
  }
  if (body.contains("chat_template_kwargs")) {
    set_error(res, 400, "arbitrary template kwargs are unsupported; use enable_thinking");
    return std::nullopt;
  }
  if (body.contains("tools") || body.contains("tool_choice") || body.contains("functions") ||
      body.contains("response_format") || body.contains("audio") || body.contains("logprobs")) {
    set_error(res, 400, "tools, structured formats, audio, and logprobs are not supported");
    return std::nullopt;
  }
  if (body.contains("n") && body["n"].get<int>() != 1) {
    set_error(res, 400, "only n=1 is supported");
    return std::nullopt;
  }
  if (body.contains("max_tokens")) {
    if (!body["max_tokens"].is_number_integer() || body["max_tokens"].get<int>() <= 0) {
      set_error(res, 400, "max_tokens must be a positive integer");
      return std::nullopt;
    }
    request.max_tokens = body["max_tokens"].get<int>();
  } else if (body.contains("max_completion_tokens")) {
    request.max_tokens = body["max_completion_tokens"].get<int>();
  }
  if (body.contains("temperature")) {
    if (!body["temperature"].is_number()) {
      set_error(res, 400, "temperature must be numeric");
      return std::nullopt;
    }
    request.temperature = body["temperature"].get<double>();
    if (request.temperature < 0.0 || request.temperature > 2.0) {
      set_error(res, 400, "temperature must be in [0, 2]");
      return std::nullopt;
    }
  }
  if (body.contains("top_p")) {
    request.top_p = body["top_p"].get<double>();
    if (request.top_p <= 0.0 || request.top_p > 1.0) {
      set_error(res, 400, "top_p must be in (0, 1]");
      return std::nullopt;
    }
  }
  request.stream = body.value("stream", false);
  if (body.contains("stream_options") && body["stream_options"].is_object()) {
    request.include_usage = body["stream_options"].value("include_usage", false);
  }
  if (body.contains("seed") && body["seed"].is_number_integer()) {
    request.seed = body["seed"].get<std::uint32_t>();
  }
  if (body.contains("stop")) {
    if (body["stop"].is_string()) {
      request.stops.push_back(body["stop"]);
    } else if (body["stop"].is_array()) {
      for (const auto& stop : body["stop"]) {
        if (!stop.is_string()) {
          set_error(res, 400, "stop entries must be strings");
          return std::nullopt;
        }
        request.stops.push_back(stop);
      }
    } else {
      set_error(res, 400, "stop must be a string or string array");
      return std::nullopt;
    }
  }
  try {
    const auto prompt = g_chat_template->render(messages, request.enable_thinking);
    request.tokens = ame::server::tokenize_chat(llama_model_get_vocab(g_model), prompt);
  } catch (const std::exception& error) {
    set_error(res, 400, std::string("chat preparation failed: ") + error.what());
    return std::nullopt;
  }
  if (request.tokens.empty()) {
    set_error(res, 400, "prompt tokenization produced no tokens");
    return std::nullopt;
  }
  if (static_cast<int64_t>(request.tokens.size()) + request.max_tokens + 8 > g_options.context) {
    set_error(res, 400, "context length exceeded", "context_length_exceeded");
    return std::nullopt;
  }
  return request;
}

struct Generation {
  std::string id;
  std::string text;
  std::string finish = "length";
  int prompt_tokens = 0;
  int completion_tokens = 0;
  bool cancelled = false;
  bool failed = false;
  bool ended_with_eog = false;
  std::string error;
  std::vector<llama_token> token_ids;  // every sampled token, including a terminal EOG
  std::vector<double> step_seconds;    // llama_decode time after each emitted token
  std::optional<double> ttft_seconds;  // generate() start to first sampled token
  std::vector<ame::server::PhaseRecord> phases;
};

bool emit_piece(Generation& generation, std::string& pending, const std::string& piece,
                const std::vector<std::string>& stops,
                std::function<bool(const std::string&)> emit) {
  pending += piece;
  for (const auto& stop : stops) {
    const auto position = pending.find(stop);
    if (position != std::string::npos) {
      const std::string visible = pending.substr(0, position);
      if (!visible.empty() && !emit(visible)) {
        generation.cancelled = true;
        return false;
      }
      generation.text += visible;
      generation.finish = "stop";
      return false;
    }
  }
  // Emit complete UTF-8 prefix only; retain an incomplete trailing sequence.
  std::size_t valid = pending.size();
  for (std::size_t back = 1; back <= std::min<std::size_t>(3, pending.size()); ++back) {
    const unsigned char byte = static_cast<unsigned char>(pending[pending.size() - back]);
    if ((byte & 0xC0) == 0x80) continue;
    std::size_t need = 1;
    if ((byte & 0xE0) == 0xC0)
      need = 2;
    else if ((byte & 0xF0) == 0xE0)
      need = 3;
    else if ((byte & 0xF8) == 0xF0)
      need = 4;
    if (byte >= 0x80 && pending.size() - back < need) {
      valid = pending.size() - back;
    }
    break;
  }
  if (valid == 0) return true;
  const std::string visible = pending.substr(0, valid);
  pending.erase(0, valid);
  if (!emit(visible)) {
    generation.cancelled = true;
    return false;
  }
  generation.text += visible;
  return true;
}

std::string backend_error_suffix() {
#ifdef AME_STREAMING
  if (g_options.stream) {
    const std::string error = ame::backend::moe_stream_error();
    if (!error.empty()) return ": " + error.substr(0, 256);
  }
#endif
  return "";
}

void generate_impl(const Request& request, bool stream,
                   const std::function<void(const std::string&)>& emit,
                   std::atomic<bool>& cancelled, std::chrono::steady_clock::time_point deadline,
                   Generation& generation, ame::server::PhaseRecorder& recorder) {
  const auto generate_begin = std::chrono::steady_clock::now();
  std::optional<ame::server::PhaseScope> phase;
  phase.emplace(recorder, "context_create");
  auto deadline_expired = [&] { return std::chrono::steady_clock::now() > deadline; };
  generation.prompt_tokens = static_cast<int>(request.tokens.size());
  llama_context_params params = llama_context_default_params();
  params.n_ctx = g_options.context;
  int batch = std::min(g_options.context, 2048);
#ifdef AME_STREAMING
  std::uint32_t unit = 0;
  std::uint32_t experts = 0;
  std::uint32_t selected = 0;
  std::string plan_error;
  if (g_options.stream) {
    if (!ame::backend::plan_model_prefill(static_cast<std::uint32_t>(batch), 256, unit, experts,
                                          selected, plan_error)) {
      generation.failed = true;
      generation.error = plan_error;
      return;
    }
    batch = static_cast<int>(unit);
  }
#endif
  params.n_batch = batch;
  params.n_ubatch = batch;
  params.n_threads = g_options.threads;
  params.n_threads_batch = g_options.threads;
#ifdef AME_STREAMING
  if (g_options.stream && g_options.stream_options.overlap) {
    params.cb_eval = ame::backend::moe_eval_callback;
    params.cb_eval_user_data = nullptr;
  }
#endif
  llama_context* context = llama_init_from_model(g_model, params);
  if (context != nullptr && g_threadpool != nullptr) {
    llama_attach_threadpool(context, g_threadpool, g_threadpool);
  }
  if (context == nullptr) {
    generation.failed = true;
    generation.error = "context creation failed";
    return;
  }
  llama_batch batch_tokens = llama_batch_init(batch, 0, 1);
  phase.reset();
  phase.emplace(recorder, "prefill");
  const char* trace_phase = "prefill";
  auto decode = [&](const std::vector<llama_token>& tokens, int position, bool logits) {
#ifdef AME_STREAMING
    if (g_options.stream) {
      ame::backend::label_moe_graph(trace_phase, static_cast<std::uint32_t>(tokens.size()));
    }
#endif
    batch_tokens.n_tokens = static_cast<int32_t>(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
      batch_tokens.token[index] = tokens[index];
      batch_tokens.pos[index] = position + static_cast<int32_t>(index);
      batch_tokens.n_seq_id[index] = 1;
      batch_tokens.seq_id[index][0] = 0;
      batch_tokens.logits[index] = (logits && index + 1 == tokens.size()) ? 1 : 0;
    }
    return llama_decode(context, batch_tokens) == 0;
  };
  for (std::size_t offset = 0; offset < request.tokens.size();) {
    if (deadline_expired()) {
      generation.failed = true;
      generation.error = "request deadline exceeded";
      llama_batch_free(batch_tokens);
      llama_free(context);
      return;
    }
    const std::size_t chunk =
        std::min<std::size_t>(static_cast<std::size_t>(batch), request.tokens.size() - offset);
    std::vector<llama_token> piece(
        request.tokens.begin() + static_cast<std::ptrdiff_t>(offset),
        request.tokens.begin() + static_cast<std::ptrdiff_t>(offset + chunk));
    if (!decode(piece, static_cast<int>(offset), true)) {
      generation.failed = true;
      generation.error = "prefill failed" + backend_error_suffix();
      llama_batch_free(batch_tokens);
      llama_free(context);
      return;
    }
    offset += chunk;
  }
  trace_phase = "decode";
  phase.reset();
  phase.emplace(recorder, "decode");
  llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
  llama_sampler* sampler = llama_sampler_chain_init(sampler_params);
  if (request.temperature <= 0.0) {
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  } else {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler,
                            llama_sampler_init_top_p(static_cast<float>(request.top_p), 1));
    llama_sampler_chain_add(sampler,
                            llama_sampler_init_temp(static_cast<float>(request.temperature)));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(request.seed));
  }
  const llama_vocab* vocab = llama_model_get_vocab(g_model);
  std::string pending;
  int position = static_cast<int>(request.tokens.size());
  bool stopped_early = false;
  for (int step = 0; step < request.max_tokens; ++step) {
    if (deadline_expired()) {
      generation.failed = true;
      generation.error = "request deadline exceeded";
      break;
    }
    if (cancelled.load(std::memory_order_relaxed)) {
      generation.cancelled = true;
      generation.finish = "cancelled";
      break;
    }
    const llama_token token = llama_sampler_sample(sampler, context, -1);
    llama_sampler_accept(sampler, token);
    generation.token_ids.push_back(token);
    if (!generation.ttft_seconds) {
      generation.ttft_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - generate_begin).count();
    }
    if (llama_vocab_is_eog(vocab, token)) {
      generation.finish = "stop";
      generation.ended_with_eog = true;
      break;
    }
    char piece[256] = {};
    int32_t length = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
    std::string bytes;
    if (length < 0) {
      std::vector<char> larger(static_cast<std::size_t>(-length));
      length = llama_token_to_piece(vocab, token, larger.data(),
                                    static_cast<int32_t>(larger.size()), 0, false);
      if (length > 0) bytes.assign(larger.data(), static_cast<std::size_t>(length));
    } else if (length > 0) {
      bytes.assign(piece, static_cast<std::size_t>(length));
    }
    if (!emit_piece(generation, pending, bytes, request.stops, [&](const std::string& visible) {
          if (stream) {
            emit(visible);
            return !cancelled.load(std::memory_order_relaxed);
          }
          return true;
        })) {
      stopped_early = true;
      break;
    }
    ++generation.completion_tokens;
    if (step + 1 == request.max_tokens) break;
    std::vector<llama_token> next{token};
    const auto step_begin = std::chrono::steady_clock::now();
    const bool stepped = decode(next, position++, true);
    generation.step_seconds.push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - step_begin).count());
    if (!stepped) {
      generation.failed = true;
      generation.error = "decode failed" + backend_error_suffix();
      break;
    }
  }
  if (!pending.empty() && !generation.failed && !stopped_early && generation.finish != "stop") {
    if (stream) emit(pending);
    generation.text += pending;
  }
  phase.reset();
  phase.emplace(recorder, "context_teardown");
  llama_sampler_free(sampler);
  llama_batch_free(batch_tokens);
  llama_free(context);
}

ame::server::PhaseRecorder::ExpertSource expert_source() {
#ifdef AME_STREAMING
  if (g_options.stream) {
    return [] {
      const auto stats = ame::backend::moe_stream_stats();
      return ame::server::ExpertCounters{
          stats.hits,        stats.misses,      stats.reads,        stats.read_bytes,
          stats.pread_calls, stats.pread_bytes, stats.read_seconds, stats.read_wait_seconds};
    };
  }
#endif
  return nullptr;
}

std::optional<Generation> generate(const Request& request, bool stream,
                                   const std::function<void(const std::string&)>& emit,
                                   std::atomic<bool>& cancelled,
                                   std::chrono::steady_clock::time_point deadline) {
  Generation generation;
  generation.id = id_for();
  ame::server::PhaseRecorder recorder(*g_sampler, expert_source());
  generate_impl(request, stream, emit, cancelled, deadline, generation, recorder);
  generation.phases = recorder.take();
  return generation;
}

std::string utc_now() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char buffer[32] = {};
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return buffer;
}

// A counter is its value, or null with the reason recorded under "unavailable".
json counter_json(const ame::server::Counter& counter, const std::string& field,
                  json& unavailable) {
  if (counter.value.has_value()) return *counter.value;
  unavailable.push_back({{"field", field}, {"reason", counter.reason}});
  return nullptr;
}

json frequencies_json(const std::vector<ame::server::CoreFrequency>& cores,
                      const std::string& field, json& unavailable) {
  json result = json::array();
  for (const auto& core : cores) {
    result.push_back(
        {{"cpu", core.cpu},
         {"khz", counter_json(core.khz, field + ".cpu" + std::to_string(core.cpu), unavailable)}});
  }
  return result;
}

json phase_json(const ame::server::PhaseRecord& phase, const std::string& prefix,
                json& unavailable) {
  const std::string base = prefix + phase.name;
  json result = {
      {"name", phase.name},
      {"seconds", phase.seconds},
      {"proc_delta",
       {{"rchar", counter_json(phase.proc_delta.rchar, base + ".proc_delta.rchar", unavailable)},
        {"read_bytes",
         counter_json(phase.proc_delta.read_bytes, base + ".proc_delta.read_bytes", unavailable)},
        {"majflt", counter_json(phase.proc_delta.majflt, base + ".proc_delta.majflt", unavailable)},
        {"pswpin",
         counter_json(phase.proc_delta.pswpin, base + ".proc_delta.pswpin", unavailable)}}},
      {"big_core_khz_start", frequencies_json(phase.khz_start, base + ".khz_start", unavailable)},
      {"big_core_khz_end", frequencies_json(phase.khz_end, base + ".khz_end", unavailable)}};
  if (phase.expert_delta.has_value()) {
    const auto& e = *phase.expert_delta;
    result["expert_delta"] = {{"hits", e.hits},
                              {"misses", e.misses},
                              {"expert_reads", e.expert_reads},
                              {"expert_bytes", e.expert_bytes},
                              {"pread_calls", e.pread_calls},
                              {"pread_bytes", e.pread_bytes},
                              {"read_seconds", e.read_seconds},
                              {"read_wait_seconds", e.read_wait_seconds}};
  } else {
    result["expert_delta"] = nullptr;
  }
  return result;
}

json big_cores_json() {
  json cores = json::array();
  for (const int cpu : g_sampler->big_cores()) cores.push_back(cpu);
  return {{"cpus", cores},
          {"homogeneous", g_sampler->homogeneous()},
          {"discovery_reason", g_sampler->discovery_reason().empty()
                                   ? json(nullptr)
                                   : json(g_sampler->discovery_reason())}};
}

json request_record(const Generation& generation, double queue_wait_seconds, double total_seconds,
                    const std::vector<llama_token>& prompt_tokens) {
  json unavailable = json::array();
  json phases = json::array();
  for (const auto& phase : generation.phases) {
    phases.push_back(phase_json(phase, "phases.", unavailable));
  }
  std::string status = "completed";
  if (generation.failed) {
    status = generation.error.rfind("request deadline", 0) == 0 ? "deadline_exceeded" : "failed";
  } else if (generation.cancelled) {
    status = "cancelled";
  }
  json record = {
      {"schema_version", 1},
      {"type", "request"},
      {"request_id", generation.id},
      {"finished_at", utc_now()},
      {"status", status},
      {"error", generation.error.empty() ? json(nullptr) : json(generation.error.substr(0, 256))},
      {"finish_reason", generation.finish},
      {"prompt_tokens", generation.prompt_tokens},
      {"completion_tokens", generation.completion_tokens},
      {"prompt_token_ids", prompt_tokens},
      {"token_ids", generation.token_ids},
      {"ended_with_eog", generation.ended_with_eog},
      {"queue_wait_seconds", queue_wait_seconds},
      {"total_seconds", total_seconds},
      {"ttft_seconds", generation.ttft_seconds ? json(*generation.ttft_seconds) : json(nullptr)},
      {"decode_step_seconds", generation.step_seconds},
      {"phases", phases},
      {"big_cores", big_cores_json()}};
  record["unavailable"] = unavailable;
  return record;
}

// Health carries a bounded copy: long token lists stay in the request log.
json bounded_for_health(json record) {
  constexpr std::size_t kLimit = 256;
  for (const char* key : {"prompt_token_ids", "token_ids", "decode_step_seconds"}) {
    if (record.contains(key) && record[key].is_array() && record[key].size() > kLimit) {
      record[key].erase(record[key].begin() + static_cast<std::ptrdiff_t>(kLimit),
                        record[key].end());
      record[std::string(key) + "_truncated"] = true;
    }
  }
  return record;
}

void write_request_log(const json& record) {
  std::lock_guard<std::mutex> lock(g_request_log_mutex);
  if (!g_request_log.is_open()) return;
  g_request_log << record.dump() << '\n';
  g_request_log.flush();
  if (!g_request_log) {
    std::cerr << "request log write failed; further records are not written\n";
    g_request_log.close();
  }
}

// Store the finished request as the health snapshot's last request and append it to the
// log. Call it before the request scope ends, while admission is still exclusive.
json record_request(const Generation& generation, const std::vector<llama_token>& prompt_tokens,
                    std::chrono::steady_clock::time_point handler_begin, double queue_wait) {
  const double total = std::chrono::duration<double>(Clock::now() - handler_begin).count();
  json record = request_record(generation, queue_wait, total, prompt_tokens);
  {
    std::lock_guard<std::mutex> lock(g_snapshot_mutex);
    g_last_request = bounded_for_health(record);
  }
  write_request_log(record);
  return record;
}

json message_json(const Request& request, const Generation& generation) {
  json response;
  response["id"] = generation.id;
  response["object"] = "chat.completion";
  response["created"] = std::time(nullptr);
  response["model"] = g_options.model_id;
  response["choices"] =
      json::array({{{"index", 0},
                    {"message", {{"role", "assistant"}, {"content", generation.text}}},
                    {"finish_reason", generation.finish}}});
  response["usage"] = {{"prompt_tokens", generation.prompt_tokens},
                       {"completion_tokens", generation.completion_tokens},
                       {"total_tokens", generation.prompt_tokens + generation.completion_tokens}};
  (void)request;
  return response;
}
}  // namespace

class LoadingListener {
 public:
  LoadingListener(httplib::Server& server, const std::string& host, int port) : server_(server) {
    if (!server_.bind_to_port(host, port)) throw std::runtime_error("loading listener bind failed");
    try {
      thread_ = std::thread([this] { server_.listen_after_bind(); });
    } catch (...) {
      server_.stop();
      throw;
    }
  }
  ~LoadingListener() { stop(); }
  void stop() {
    server_.stop();
    if (thread_.joinable()) thread_.join();
  }

 private:
  httplib::Server& server_;
  std::thread thread_;
};

struct BackendLifetime {
  BackendLifetime() { llama_backend_init(); }
  ~BackendLifetime() { llama_backend_free(); }
};

#ifdef AME_STREAMING
struct StreamLifetime {
  ~StreamLifetime() { ame::backend::uninstall_moe_stream(); }
};
#endif

int run_server(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    auto next = [&](std::string& value) {
      if (index + 1 >= argc) return false;
      value = argv[++index];
      return true;
    };
    std::string value;
    if (arg == "--model") {
      if (!next(options.model)) return 2;
    } else if (arg == "--model-id") {
      if (!next(options.model_id)) return 2;
    } else if (arg == "--manifest") {
      if (!next(options.manifest)) return 2;
    } else if (arg == "--artifact-sha") {
      if (!next(options.artifact_sha)) return 2;
    } else if (arg == "--host") {
      if (!next(options.host)) return 2;
    } else if (arg == "--port") {
      if (!next(value)) return 2;
      options.port = std::stoi(value);
    } else if (arg == "--context") {
      if (!next(value)) return 2;
      options.context = std::stoi(value);
    } else if (arg == "--threads") {
      if (!next(value)) return 2;
      options.threads = std::stoi(value);
    } else if (arg == "--slots") {
      if (!next(value)) return 2;
      options.slots = std::stoi(value);
    } else if (arg == "--arena-mib") {
      if (!next(value)) return 2;
      options.arena_mib = std::stoull(value);
    } else if (arg == "--cache") {
      if (!next(value) || (value != "legacy" && value != "layered")) return 2;
      options.stream_options.layout = value == "layered" ? ame::backend::CacheLayout::Layered
                                                         : ame::backend::CacheLayout::Legacy;
    } else if (arg == "--eviction") {
      if (!next(value) || (value != "lru" && value != "lfu")) return 2;
      options.stream_options.eviction =
          value == "lfu" ? ame::backend::Eviction::Lfu : ame::backend::Eviction::Lru;
    } else if (arg == "--layer-slots") {
      if (!next(value)) return 2;
      options.stream_options.layer_slots = static_cast<std::uint32_t>(std::stoul(value));
    } else if (arg == "--lfu-halve") {
      if (!next(value)) return 2;
      options.stream_options.lfu_halve_every = static_cast<std::uint32_t>(std::stoul(value));
    } else if (arg == "--readers") {
      if (!next(value)) return 2;
      options.stream_options.readers = static_cast<unsigned>(std::stoul(value));
      if (options.stream_options.readers < 1 || options.stream_options.readers > 4) return 2;
    } else if (arg == "--overlap") {
      if (!next(value) || (value != "topk" && value != "off")) return 2;
      options.stream_options.overlap = value == "topk";
    } else if (arg == "--threadpool") {
      if (!next(value) || (value != "big" && value != "default")) return 2;
      options.threadpool_big = value == "big";
    } else if (arg == "--poll") {
      if (!next(value)) return 2;
      options.poll = std::stoi(value);
      if (options.poll < 0 || options.poll > 100) return 2;
    } else if (arg == "--reader-cpus") {
      if (!next(value) || (value != "auto" && value != "none")) return 2;
      options.reader_cpus_auto = value == "auto";
    } else if (arg == "--fadvise-random") {
      options.stream_options.fadvise_random = true;
    } else if (arg == "--fadvise-dontneed") {
      options.stream_options.fadvise_dontneed = true;
    } else if (arg == "--request-log") {
      if (!next(options.request_log)) return 2;
    } else if (arg == "--stream") {
      options.stream = true;
    } else {
      std::cerr << "unknown argument " << arg << '\n';
      return 2;
    }
  }
  g_options = options;
  if (options.model.empty()) {
    std::cerr << "--model required\n";
    return 2;
  }
  if (options.context <= 0 || options.threads <= 0 || options.slots <= 0 || options.port <= 0 ||
      options.port > 65535 || options.arena_mib == 0 ||
      options.arena_mib > UINT64_MAX / (1024ULL * 1024ULL)) {
    std::cerr << "invalid server resource options\n";
    return 2;
  }
  if (options.stream && (options.manifest.empty() || options.artifact_sha.size() != 64)) {
    std::cerr << "streaming requires --manifest and --artifact-sha\n";
    return 2;
  }
  const auto process_begin = Clock::now();
  g_sampler = std::make_unique<ame::server::SystemSampler>(options.system_root);
  if (!options.request_log.empty()) {
    g_request_log.open(options.request_log, std::ios::app);
    if (!g_request_log) {
      std::cerr << "cannot open request log " << options.request_log << '\n';
      return 2;
    }
  }
  if (options.reader_cpus_auto && !g_sampler->homogeneous() && !g_sampler->big_cores().empty()) {
    // Readers copy page-cache data and wait on storage; keep them off the compute cores.
    for (int cpu = 0; cpu < 256; ++cpu) {
      const auto& big = g_sampler->big_cores();
      if (std::find(big.begin(), big.end(), cpu) == big.end() &&
          std::filesystem::exists("/sys/devices/system/cpu/cpu" + std::to_string(cpu))) {
        options.stream_options.reader_cpus.push_back(cpu);
      }
    }
  }
  struct ThreadpoolLifetime {
    ~ThreadpoolLifetime() {
      if (g_threadpool != nullptr) ggml_threadpool_free(g_threadpool);
      g_threadpool = nullptr;
    }
  } threadpool_lifetime;
  if (options.threadpool_big && !g_sampler->homogeneous() && !g_sampler->big_cores().empty()) {
    // Compute threads on the big cores only, polling at the requested level.
    ggml_threadpool_params pool_params = ggml_threadpool_params_default(options.threads);
    for (const int cpu : g_sampler->big_cores()) {
      if (cpu >= 0 && cpu < GGML_MAX_N_THREADS) pool_params.cpumask[cpu] = true;
    }
    pool_params.strict_cpu = true;
    pool_params.poll = static_cast<uint32_t>(options.poll);
    g_threadpool = ggml_threadpool_new(&pool_params);
    if (g_threadpool == nullptr) {
      std::cerr << "threadpool creation failed\n";
      return 1;
    }
  }
  ame::server::PhaseRecorder startup(*g_sampler, nullptr);
  BackendLifetime backend_lifetime;
  httplib::Server loading_server;
  loading_server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"status", "loading"}, {"model", g_options.model_id}}.dump(),
                    "application/json");
  });
  LoadingListener loading_listener(loading_server, options.host, options.port);
  std::string load_path = options.model;
  std::unique_ptr<ame::backend::Artifact> artifact;
#ifdef AME_STREAMING
  StreamLifetime stream_lifetime;
#endif
  if (options.stream) {
    std::error_code size_error;
    const auto bytes = std::filesystem::file_size(options.model, size_error);
    if (size_error) return 1;
    std::string error;
    // Opening hashes the whole artifact; the stage's I/O deltas show that read.
    startup.begin("artifact_open");
    auto opened = ame::backend::Artifact::open(options.model, {options.artifact_sha, bytes}, error);
    startup.end();
    if (!opened.ok()) {
      std::cerr << "artifact: " << error << '\n';
      return 1;
    }
    artifact = std::move(opened).value();
    std::cerr << "artifact_hash_seconds=" << artifact->hash_seconds()
              << " artifact_bytes=" << artifact->size() << "\n";
    startup.begin("manifest_install");
    const bool installed = ame::backend::install_moe_stream(
        *artifact, options.manifest, static_cast<std::uint32_t>(options.slots),
        options.arena_mib * 1024ULL * 1024ULL, options.stream_options);
    startup.end();
    if (!installed) {
      std::cerr << "install: " << ame::backend::moe_stream_stats().error << '\n';
      return 1;
    }
    load_path = artifact->proc_path();
  }
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  model_params.use_extra_bufts = !options.stream;
  startup.begin("model_load");
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model_owner(
      llama_model_load_from_file(load_path.c_str(), model_params), &llama_model_free);
  startup.end();
  g_model = model_owner.get();
  if (g_model == nullptr) {
    std::cerr << "model load failed\n";
    return 1;
  }
  if (options.stream) {
    std::string error;
    startup.begin("contract_bind");
    const bool bound = ame::backend::bind_model_contract(g_model, error);
    startup.end();
    if (!bound) {
      std::cerr << "contract: " << error << '\n';
      return 1;
    }
  }
  startup.begin("template_init");
  ame::server::ChatTemplate chat_template(g_model);
  startup.end();
  g_chat_template = &chat_template;
  {
    json unavailable = json::array();
    json stages = json::array();
    for (const auto& stage : startup.take())
      stages.push_back(phase_json(stage, "stages.", unavailable));
    json record = {{"schema_version", 1},
                   {"type", "startup"},
                   {"finished_at", utc_now()},
                   {"first_ready_seconds",
                    std::chrono::duration<double>(Clock::now() - process_begin).count()},
                   {"stages", stages},
                   {"big_cores", big_cores_json()},
                   {"threadpool", g_threadpool != nullptr ? "big_cores_pinned" : "default"},
                   {"poll", options.poll},
                   {"overlap", options.stream_options.overlap ? "topk" : "off"},
                   {"readers", options.stream_options.readers},
                   {"unavailable", unavailable}};
#ifdef AME_STREAMING
    if (options.stream) {
      const auto stats = ame::backend::moe_stream_stats();
      record["artifact_hash_seconds"] = stats.artifact_hash_seconds;
      record["manifest_validation_seconds"] = stats.manifest_validation_seconds;
    }
#endif
    {
      std::lock_guard<std::mutex> lock(g_snapshot_mutex);
      g_startup_record = record;
    }
    write_request_log(record);
    std::cerr << "startup_record=" << record.dump() << '\n';
  }
  g_ready = true;
  loading_listener.stop();
  httplib::Server server;
  server.set_payload_max_length(2 * 1024 * 1024);
  server.set_read_timeout(300);
  server.set_write_timeout(10);
  server.new_task_queue = [] { return new httplib::ThreadPool(12); };
  server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    json body = {{"status", g_failed ? "failed" : "ready"}, {"model", g_options.model_id}};
    if (!g_error.empty()) body["error"] = g_error;
    {
      std::lock_guard<std::mutex> lock(g_snapshot_mutex);
      if (!g_stream_snapshot.empty()) body["streaming"] = g_stream_snapshot;
      if (!g_last_request.empty()) body["last_request"] = g_last_request;
      if (!g_startup_record.empty()) body["startup"] = g_startup_record;
    }
    body["queue"] = g_queue.snapshot();
    res.set_content(body.dump(), "application/json");
  });
  server.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
    json body = {
        {"object", "list"},
        {"data",
         json::array({{{"id", g_options.model_id}, {"object", "model"}, {"owned_by", "local"}}})}};
    res.set_content(body.dump(), "application/json");
  });
  server.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
      body = json::parse(req.body);
    } catch (const std::exception&) {
      set_error(res, 400, "malformed JSON");
      return;
    }
    auto request = parse_request(body, res);
    if (!request.has_value()) return;
    const auto handler_begin = std::chrono::steady_clock::now();
    const auto admission = g_queue.acquire();
    const double queue_wait =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - handler_begin).count();
    if (admission == AdmissionQueue::Result::Full) {
      res.set_header("Retry-After", "1");
      set_error(res, 429, "four requests are already outstanding", "server_busy");
      return;
    }
    if (admission == AdmissionQueue::Result::WaitTimeout) {
      set_error(res, 504, "request waited 300 seconds for a generation slot", "queue_timeout");
      return;
    }
    auto execution_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
    auto admission_guard = make_request_scope();
    if (g_options.stream) ame::backend::begin_moe_stream_request();
    if (!request->stream) {
      std::atomic<bool> cancelled{false};
      auto generation =
          generate(*request, false, [](const std::string&) {}, cancelled, execution_deadline);
      json record;
      if (generation.has_value()) {
        record = record_request(*generation, request->tokens, handler_begin, queue_wait);
      }
      admission_guard->finish();
      if (!generation.has_value() || generation->failed) {
        set_error(res, 500, generation.has_value() ? generation->error : "generation failed",
                  "server_error");
        return;
      }
      json response = message_json(*request, *generation);
      if (request->debug) response["ame"] = std::move(record);
      res.set_content(response.dump(), "application/json");
      return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    const std::string id = "chatcmpl-" + std::to_string(static_cast<unsigned long>(::getpid())) +
                           "-" + std::to_string(g_counter.fetch_add(1));
    const std::time_t created = std::time(nullptr);
    Request copied = *request;
    res.set_chunked_content_provider(
        "text/event-stream",
        [request = std::move(copied), id, created, execution_deadline, admission_guard,
         handler_begin, queue_wait](std::size_t, httplib::DataSink& sink) {
          auto cancelled = std::make_shared<std::atomic<bool>>(false);
          auto send = [&](const json& payload) {
            const std::string data = "data: " + payload.dump() + "\n\n";
            return sink.write(data.data(), data.size());
          };
          json first = {{"id", id},
                        {"object", "chat.completion.chunk"},
                        {"created", created},
                        {"model", g_options.model_id},
                        {"choices", json::array({{{"index", 0},
                                                  {"delta", {{"role", "assistant"}}},
                                                  {"finish_reason", nullptr}}})}};
          if (!send(first)) {
            admission_guard->finish();
            return false;
          }
          auto generation = generate(
              request, true,
              [&](const std::string& piece) {
                if (!sink.is_writable()) {
                  cancelled->store(true);
                  return;
                }
                json chunk = {{"id", id},
                              {"object", "chat.completion.chunk"},
                              {"created", created},
                              {"model", g_options.model_id},
                              {"choices", json::array({{{"index", 0},
                                                        {"delta", {{"content", piece}}},
                                                        {"finish_reason", nullptr}}})}};
                if (!send(chunk)) cancelled->store(true);
              },
              *cancelled, execution_deadline);
          if (generation.has_value()) {
            if (cancelled->load() && !generation->failed) generation->cancelled = true;
            (void)record_request(*generation, request.tokens, handler_begin, queue_wait);
          }
          if (!generation.has_value() || generation->failed || cancelled->load()) {
            json error = {{"error",
                           {{"message", generation.has_value() ? generation->error : "cancelled"},
                            {"type", "server_error"},
                            {"code", "stream_failed"}}}};
            send(error);
            admission_guard->finish();
            return false;
          }
          json last = {{"id", id},
                       {"object", "chat.completion.chunk"},
                       {"created", created},
                       {"model", g_options.model_id},
                       {"choices", json::array({{{"index", 0},
                                                 {"delta", json::object()},
                                                 {"finish_reason", generation->finish}}})}};
          if (!send(last)) {
            admission_guard->finish();
            return false;
          }
          if (request.include_usage) {
            json usage = {
                {"id", id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", g_options.model_id},
                {"choices", json::array()},
                {"usage",
                 {{"prompt_tokens", generation->prompt_tokens},
                  {"completion_tokens", generation->completion_tokens},
                  {"total_tokens", generation->prompt_tokens + generation->completion_tokens}}}};
            if (!send(usage)) {
              admission_guard->finish();
              return false;
            }
          }
          const std::string done = "data: [DONE]\n\n";
          sink.write(done.data(), done.size());
          sink.done();
          admission_guard->finish();
          return true;
        });
  });
  std::cerr << "server ready on http://" << options.host << ':' << options.port
            << " model=" << options.model_id << '\n';
  if (!server.listen(options.host.c_str(), options.port)) {
    std::cerr << "listen failed\n";
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    return run_server(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "server failed: " << error.what() << '\n';
    return 1;
  }
}