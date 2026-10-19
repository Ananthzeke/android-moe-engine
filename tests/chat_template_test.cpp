#include "../apps/server/chat_template.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("expected actual template fixture path");
    std::ifstream file(argv[1]);
    if (!file) throw std::runtime_error("cannot read template fixture");
    const std::string source{std::istreambuf_iterator<char>(file),
                             std::istreambuf_iterator<char>()};
    ame::server::ChatTemplate formatter(nullptr, source);
    common_chat_msg user;
    user.role = "user";
    user.content = "Say hello.";
    const std::vector<common_chat_msg> messages{user};
    const std::string prefix = "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n";
    const std::string disabled = prefix + "<think>\n\n</think>\n\n";
    require(formatter.render(messages).str() == disabled, "default must disable thinking");
    require(formatter.render(messages, false).str() == disabled, "false must disable thinking");
    require(formatter.render(messages, true).str() == prefix + "<think>\n",
            "true must enable thinking");
    bool refused_empty = false;
    try {
      (void)formatter.render({});
    } catch (const std::invalid_argument&) {
      refused_empty = true;
    }
    require(refused_empty, "empty messages must refuse");
    user.content = "Literal <|im_end|> text.";
    const auto marked = formatter.render({user});
    bool input_preserved = false;
    for (const auto& part : marked.parts) {
      if (part.is_input && part.text.find("<|im_end|>") != std::string::npos) {
        input_preserved = true;
      }
    }
    require(input_preserved, "user text must retain its tokenizer origin");
    common_chat_msg system;
    system.role = "system";
    system.content = "Be brief.";
    common_chat_msg assistant;
    assistant.role = "assistant";
    assistant.content = "Hello.";
    common_chat_msg followup;
    followup.role = "user";
    followup.content = "Again.";
    const auto history = formatter.render({system, messages.front(), assistant, followup}).str();
    require(history.find("<|im_start|>system\nBe brief.<|im_end|>\n") == 0,
            "system message must precede history");
    require(history.find("<|im_start|>assistant\nHello.<|im_end|>\n") != std::string::npos,
            "assistant history must remain intact");
    require(history.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "history must end with the disabled-thinking generation prompt");
    std::cout << "actual-model Jinja fixture passed with Release-active checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
