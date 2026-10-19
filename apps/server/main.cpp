#define CPPHTTPLIB_IMPLEMENTATION
#include "chat_template.hpp"
#include "httplib.h"
#include "json.hpp"
#include "llama.h"
#ifdef AME_STREAMING
#include "artifact.hpp"
#include "moe_stream.hpp"
#include "prefill_plan.hpp"
#endif

#include <unistd.h>

#include <atomic>
#include <chrono>
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
};

std::atomic<bool> g_ready{false};
std::atomic<bool> g_failed{false};
std::string g_error;
llama_model* g_model = nullptr;
ame::server::ChatTemplate* g_chat_template = nullptr;
Options g_options;
std::mutex g_generation;
std::atomic<std::uint64_t> g_counter{0};
std::atomic<bool> g_client_cancelled{false};

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
  std::string error;
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

std::optional<Generation> generate(const Request& request, bool stream,
                                   const std::function<void(const std::string&)>& emit) {
  Generation generation;
  generation.id = id_for();
  g_client_cancelled.store(false, std::memory_order_relaxed);
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
      return generation;
    }
    batch = static_cast<int>(unit);
  }
#endif
  params.n_batch = batch;
  params.n_ubatch = batch;
  params.n_threads = g_options.threads;
  params.n_threads_batch = g_options.threads;
  llama_context* context = llama_init_from_model(g_model, params);
  if (context == nullptr) {
    generation.failed = true;
    generation.error = "context creation failed";
    return generation;
  }
  llama_batch batch_tokens = llama_batch_init(batch, 0, 1);
  auto decode = [&](const std::vector<llama_token>& tokens, int position, bool logits) {
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
    const std::size_t chunk =
        std::min<std::size_t>(static_cast<std::size_t>(batch), request.tokens.size() - offset);
    std::vector<llama_token> piece(
        request.tokens.begin() + static_cast<std::ptrdiff_t>(offset),
        request.tokens.begin() + static_cast<std::ptrdiff_t>(offset + chunk));
    if (!decode(piece, static_cast<int>(offset), true)) {
      generation.failed = true;
      generation.error = "prefill failed";
      llama_batch_free(batch_tokens);
      llama_free(context);
      return generation;
    }
    offset += chunk;
  }
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
    if (g_client_cancelled.load(std::memory_order_relaxed)) {
      generation.cancelled = true;
      generation.finish = "cancelled";
      break;
    }
    const llama_token token = llama_sampler_sample(sampler, context, -1);
    llama_sampler_accept(sampler, token);
    if (llama_vocab_is_eog(vocab, token)) {
      generation.finish = "stop";
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
            return !g_client_cancelled.load(std::memory_order_relaxed);
          }
          return true;
        })) {
      stopped_early = true;
      break;
    }
    ++generation.completion_tokens;
    if (step + 1 == request.max_tokens) break;
    std::vector<llama_token> next{token};
    if (!decode(next, position++, true)) {
      generation.failed = true;
      generation.error = "decode failed";
      break;
    }
  }
  if (!pending.empty() && !generation.failed && !stopped_early && generation.finish != "stop") {
    if (stream) emit(pending);
    generation.text += pending;
  }
  llama_sampler_free(sampler);
  llama_batch_free(batch_tokens);
  llama_free(context);
  return generation;
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
    auto opened = ame::backend::Artifact::open(options.model, {options.artifact_sha, bytes}, error);
    if (!opened.ok()) {
      std::cerr << "artifact: " << error << '\n';
      return 1;
    }
    artifact = std::move(opened).value();
    if (!ame::backend::install_moe_stream(*artifact, options.manifest,
                                          static_cast<std::uint32_t>(options.slots),
                                          options.arena_mib * 1024ULL * 1024ULL)) {
      std::cerr << "install: " << ame::backend::moe_stream_stats().error << '\n';
      return 1;
    }
    load_path = artifact->proc_path();
  }
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  model_params.use_extra_bufts = !options.stream;
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model_owner(
      llama_model_load_from_file(load_path.c_str(), model_params), &llama_model_free);
  g_model = model_owner.get();
  if (g_model == nullptr) {
    std::cerr << "model load failed\n";
    return 1;
  }
  if (options.stream) {
    std::string error;
    if (!ame::backend::bind_model_contract(g_model, error)) {
      std::cerr << "contract: " << error << '\n';
      return 1;
    }
  }
  ame::server::ChatTemplate chat_template(g_model);
  g_chat_template = &chat_template;
  g_ready = true;
  loading_listener.stop();
  httplib::Server server;
  server.set_payload_max_length(2 * 1024 * 1024);
  server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    json body = {{"status", g_failed ? "failed" : "ready"}, {"model", g_options.model_id}};
    if (!g_error.empty()) body["error"] = g_error;
#ifdef AME_STREAMING
    if (g_options.stream) {
      const auto stats = ame::backend::moe_stream_stats();
      body["streaming"] = {{"slots", stats.slots},
                           {"hits", stats.hits},
                           {"misses", stats.misses},
                           {"evictions", stats.evictions},
                           {"reads", stats.reads},
                           {"read_bytes", stats.read_bytes},
                           {"materialize_calls", stats.materialize_calls},
                           {"arena_bytes", stats.arena_bytes},
                           {"contract_routed", stats.contract_routed},
                           {"contract_shared", stats.contract_shared},
                           {"failed", stats.failed},
                           {"cancelled", stats.cancelled}};
    }
#endif
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
    if (!g_generation.try_lock()) {
      set_error(res, 429, "a generation is already active", "server_busy");
      return;
    }
    if (!request->stream) {
      auto generation = generate(*request, false, [](const std::string&) {});
      g_generation.unlock();
      if (!generation.has_value() || generation->failed) {
        set_error(res, 500, generation.has_value() ? generation->error : "generation failed",
                  "server_error");
        return;
      }
      res.set_content(message_json(*request, *generation).dump(), "application/json");
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
        [request = std::move(copied), id, created](std::size_t, httplib::DataSink& sink) {
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
            g_generation.unlock();
            return false;
          }
          bool cancelled = false;
          auto generation = generate(request, true, [&](const std::string& piece) {
            if (!sink.is_writable()) {
              cancelled = true;
              g_client_cancelled.store(true);
              return;
            }
            json chunk = {{"id", id},
                          {"object", "chat.completion.chunk"},
                          {"created", created},
                          {"model", g_options.model_id},
                          {"choices", json::array({{{"index", 0},
                                                    {"delta", {{"content", piece}}},
                                                    {"finish_reason", nullptr}}})}};
            if (!send(chunk)) cancelled = true;
          });
          if (!generation.has_value() || generation->failed || cancelled) {
            json error = {{"error",
                           {{"message", generation.has_value() ? generation->error : "cancelled"},
                            {"type", "server_error"},
                            {"code", "stream_failed"}}}};
            send(error);
            g_generation.unlock();
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
            g_generation.unlock();
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
              g_generation.unlock();
              return false;
            }
          }
          const std::string done = "data: [DONE]\n\n";
          sink.write(done.data(), done.size());
          sink.done();
          g_generation.unlock();
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
