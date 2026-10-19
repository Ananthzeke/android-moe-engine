#pragma once

#include <climits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat.h"

namespace ame::server {

class ChatTemplate {
 public:
  explicit ChatTemplate(const llama_model* model, const std::string& source = "") {
    const char* metadata = model == nullptr ? nullptr : llama_model_chat_template(model, nullptr);
    if (source.empty() && (metadata == nullptr || metadata[0] == '\0')) {
      throw std::invalid_argument("model has no chat template");
    }
    templates_ = common_chat_templates_init(model, source);
    if (!templates_) throw std::runtime_error("chat template initialization failed");
  }

  common_chat_prompt render(const std::vector<common_chat_msg>& messages,
                            bool enable_thinking = false) {
    if (messages.empty()) throw std::invalid_argument("messages must not be empty");
    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking = enable_thinking;
    const std::lock_guard<std::mutex> lock(mutex_);
    return common_chat_templates_apply(templates_.get(), inputs).prompt;
  }

 private:
  common_chat_templates_ptr templates_;
  std::mutex mutex_;
};

inline std::vector<llama_token> tokenize_chat(const llama_vocab* vocab,
                                              const common_chat_prompt& prompt) {
  std::vector<llama_token> tokens;
  if (llama_vocab_get_add_bos(vocab)) {
    const auto bos = llama_vocab_bos(vocab);
    if (bos == LLAMA_TOKEN_NULL) throw std::runtime_error("missing BOS token");
    tokens.push_back(bos);
  }
  for (const auto& part : prompt.parts) {
    if (part.text.empty()) continue;
    if (part.text.size() > static_cast<std::size_t>(INT32_MAX)) {
      throw std::invalid_argument("prompt part exceeds tokenizer limit");
    }
    auto next = common_tokenize(vocab, part.text, false, !part.is_input);
    if (next.size() > static_cast<std::size_t>(INT32_MAX) - tokens.size()) {
      throw std::invalid_argument("prompt exceeds tokenizer limit");
    }
    tokens.insert(tokens.end(), next.begin(), next.end());
  }
  if (llama_vocab_get_add_eos(vocab)) {
    const auto eos = llama_vocab_eos(vocab);
    if (eos == LLAMA_TOKEN_NULL) throw std::runtime_error("missing EOS token");
    tokens.push_back(eos);
  }
  return tokens;
}

}  // namespace ame::server
