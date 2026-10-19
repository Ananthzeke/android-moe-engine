#include "compare.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string_view>
#include <vector>

namespace ame::compare {
namespace {

constexpr std::size_t kFileLimit = 4U * 1024U * 1024U;
constexpr int kDepthLimit = 64;

struct Json {
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind{Kind::Null};
  bool boolean{false};
  double number{0.0};
  std::string text;
  std::vector<Json> array;
  std::map<std::string, Json> object;

  [[nodiscard]] const Json* find(std::string_view key) const {
    if (kind != Kind::Object) {
      return nullptr;
    }
    const auto found = object.find(std::string(key));
    return found == object.end() ? nullptr : &found->second;
  }
};

class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  bool parse(Json& value) {
    skip();
    if (!parse_value(value, 0)) {
      return false;
    }
    skip();
    if (position_ != text_.size()) {
      error_ = "trailing characters";
      return false;
    }
    return true;
  }

  [[nodiscard]] const std::string& error() const { return error_; }

 private:
  void skip() {
    while (position_ < text_.size()) {
      const char character = text_[position_];
      if (character == ' ' || character == '\n' || character == '\r' || character == '\t') {
        ++position_;
      } else {
        break;
      }
    }
  }

  bool expect(char character) {
    skip();
    if (position_ >= text_.size() || text_[position_] != character) {
      error_ = std::string("expected '") + character + "'";
      return false;
    }
    ++position_;
    return true;
  }

  static int hex_value(char character) {
    if (character >= '0' && character <= '9') {
      return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
      return character - 'a' + 10;
    }
    if (character >= 'A' && character <= 'F') {
      return character - 'A' + 10;
    }
    return -1;
  }

  static void append_utf8(std::string& output, std::uint32_t code) {
    if (code <= 0x7F) {
      output += static_cast<char>(code);
    } else if (code <= 0x7FF) {
      output += static_cast<char>(0xC0 | (code >> 6));
      output += static_cast<char>(0x80 | (code & 0x3F));
    } else if (code <= 0xFFFF) {
      output += static_cast<char>(0xE0 | (code >> 12));
      output += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      output += static_cast<char>(0x80 | (code & 0x3F));
    } else {
      output += static_cast<char>(0xF0 | (code >> 18));
      output += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
      output += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      output += static_cast<char>(0x80 | (code & 0x3F));
    }
  }

  bool parse_hex4(std::uint32_t& value) {
    if (position_ + 4 > text_.size()) {
      error_ = "short unicode escape";
      return false;
    }
    value = 0;
    for (int index = 0; index < 4; ++index) {
      const int digit = hex_value(text_[position_ + static_cast<std::size_t>(index)]);
      if (digit < 0) {
        error_ = "invalid unicode escape";
        return false;
      }
      value = (value << 4U) | static_cast<std::uint32_t>(digit);
    }
    position_ += 4;
    return true;
  }

  bool parse_string(std::string& output) {
    if (!expect('"')) {
      return false;
    }
    output.clear();
    while (position_ < text_.size()) {
      const char character = text_[position_++];
      if (character == '"') {
        return true;
      }
      if (static_cast<unsigned char>(character) < 0x20) {
        error_ = "control character in string";
        return false;
      }
      if (character != '\\') {
        output += character;
        continue;
      }
      if (position_ >= text_.size()) {
        error_ = "unterminated escape";
        return false;
      }
      const char escape = text_[position_++];
      switch (escape) {
        case '"':
          output += '"';
          break;
        case '\\':
          output += '\\';
          break;
        case '/':
          output += '/';
          break;
        case 'b':
          output += '\b';
          break;
        case 'f':
          output += '\f';
          break;
        case 'n':
          output += '\n';
          break;
        case 'r':
          output += '\r';
          break;
        case 't':
          output += '\t';
          break;
        case 'u': {
          std::uint32_t code = 0;
          if (!parse_hex4(code)) {
            return false;
          }
          if (code >= 0xD800 && code <= 0xDBFF && position_ + 6 <= text_.size() &&
              text_[position_] == '\\' && text_[position_ + 1] == 'u') {
            const std::size_t saved = position_;
            position_ += 2;
            std::uint32_t low = 0;
            if (!parse_hex4(low)) {
              return false;
            }
            if (low >= 0xDC00 && low <= 0xDFFF) {
              code = 0x10000 + ((code - 0xD800) << 10U) + (low - 0xDC00);
            } else {
              position_ = saved;
            }
          }
          append_utf8(output, code);
          break;
        }
        default:
          error_ = "unknown escape";
          return false;
      }
    }
    error_ = "unterminated string";
    return false;
  }

  bool parse_number(double& value) {
    const std::size_t begin = position_;
    if (position_ < text_.size() && text_[position_] == '-') {
      ++position_;
    }
    while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
      ++position_;
    }
    if (position_ < text_.size() && text_[position_] == '.') {
      ++position_;
      while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
        ++position_;
      }
    }
    if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
      ++position_;
      if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) {
        ++position_;
      }
      while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
        ++position_;
      }
    }
    if (position_ == begin) {
      error_ = "invalid number";
      return false;
    }
    const auto result = std::from_chars(text_.data() + begin, text_.data() + position_, value,
                                        std::chars_format::general);
    if (result.ec != std::errc{} || result.ptr != text_.data() + position_) {
      error_ = "invalid number";
      return false;
    }
    return true;
  }

  bool parse_value(Json& value, int depth) {
    if (depth > kDepthLimit) {
      error_ = "nesting too deep";
      return false;
    }
    skip();
    if (position_ >= text_.size()) {
      error_ = "unexpected end";
      return false;
    }
    const char character = text_[position_];
    if (character == '{') {
      ++position_;
      value.kind = Json::Kind::Object;
      skip();
      if (position_ < text_.size() && text_[position_] == '}') {
        ++position_;
        return true;
      }
      while (true) {
        std::string key;
        if (!parse_string(key)) {
          return false;
        }
        if (!expect(':')) {
          return false;
        }
        Json member;
        if (!parse_value(member, depth + 1)) {
          return false;
        }
        if (!value.object.emplace(std::move(key), std::move(member)).second) {
          error_ = "duplicate object key";
          return false;
        }
        skip();
        if (position_ < text_.size() && text_[position_] == ',') {
          ++position_;
          continue;
        }
        return expect('}');
      }
    }
    if (character == '[') {
      ++position_;
      value.kind = Json::Kind::Array;
      skip();
      if (position_ < text_.size() && text_[position_] == ']') {
        ++position_;
        return true;
      }
      while (true) {
        Json member;
        if (!parse_value(member, depth + 1)) {
          return false;
        }
        value.array.push_back(std::move(member));
        skip();
        if (position_ < text_.size() && text_[position_] == ',') {
          ++position_;
          continue;
        }
        return expect(']');
      }
    }
    if (character == '"') {
      value.kind = Json::Kind::String;
      return parse_string(value.text);
    }
    if (text_.substr(position_, 4) == "true") {
      position_ += 4;
      value.kind = Json::Kind::Bool;
      value.boolean = true;
      return true;
    }
    if (text_.substr(position_, 5) == "false") {
      position_ += 5;
      value.kind = Json::Kind::Bool;
      value.boolean = false;
      return true;
    }
    if (text_.substr(position_, 4) == "null") {
      position_ += 4;
      value.kind = Json::Kind::Null;
      return true;
    }
    value.kind = Json::Kind::Number;
    return parse_number(value.number);
  }

  std::string_view text_;
  std::size_t position_ = 0;
  std::string error_;
};

std::optional<double> number_field(const Json& object, std::string_view key) {
  const Json* field = object.find(key);
  if (field == nullptr || field->kind != Json::Kind::Number) {
    return std::nullopt;
  }
  return field->number;
}

const Json* object_field(const Json& object, std::string_view key) {
  const Json* field = object.find(key);
  return field != nullptr && field->kind == Json::Kind::Object ? field : nullptr;
}

enum class ArrayStatus { Missing, Valid, Invalid };

ArrayStatus read_ids(const Json& root, const Json* value, std::vector<std::int32_t>& output) {
  (void)root;
  if (value == nullptr || value->kind != Json::Kind::Array) {
    return ArrayStatus::Missing;
  }
  output.clear();
  output.reserve(value->array.size());
  for (const auto& item : value->array) {
    if (item.kind != Json::Kind::Number || !std::isfinite(item.number) ||
        std::floor(item.number) != item.number ||
        item.number < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
        item.number > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
      output.clear();
      return ArrayStatus::Invalid;
    }
    output.push_back(static_cast<std::int32_t>(item.number));
  }
  return ArrayStatus::Valid;
}

bool json_equal(const Json* left, const Json* right) {
  if (left == nullptr || right == nullptr || left->kind != right->kind) {
    return false;
  }
  switch (left->kind) {
    case Json::Kind::Null:
      return true;
    case Json::Kind::Bool:
      return left->boolean == right->boolean;
    case Json::Kind::Number:
      return left->number == right->number;
    case Json::Kind::String:
      return left->text == right->text;
    case Json::Kind::Array:
    case Json::Kind::Object:
      return false;
  }
  return false;
}

const Json* required_string(const Json& root, std::string_view key) {
  const Json* value = root.find(key);
  return value != nullptr && value->kind == Json::Kind::String ? value : nullptr;
}

const Json* required_number(const Json& root, std::string_view key) {
  const Json* value = root.find(key);
  return value != nullptr && value->kind == Json::Kind::Number ? value : nullptr;
}

std::string escape(std::string_view text) {
  std::string output;
  for (const char character : text) {
    switch (character) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        output += character;
        break;
    }
  }
  return output;
}

void write_optional(std::string& output, const std::optional<double>& value) {
  if (!value.has_value()) {
    output += "null";
    return;
  }
  char buffer[64] = {};
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), *value);
  output.append(buffer, result.ptr);
}

void write_metrics(std::string& output, const Metrics& metrics) {
  output += "{\"ttft_seconds\":";
  write_optional(output, metrics.ttft_seconds);
  output += ",\"decode_tokens_per_second\":";
  write_optional(output, metrics.decode_tokens_per_second);
  output += ",\"peak_rss_kib\":";
  if (metrics.peak_rss_kib.has_value()) {
    output += std::to_string(*metrics.peak_rss_kib);
  } else {
    output += "null";
  }
  output += "}";
}

Metrics metrics_from(const Json& root) {
  Metrics metrics;
  if (const Json* timing = object_field(root, "timing"); timing != nullptr) {
    metrics.ttft_seconds = number_field(*timing, "request_ttft_seconds");
    metrics.decode_tokens_per_second = number_field(*timing, "decode_tokens_per_second");
  }
  if (const Json* memory = object_field(root, "memory"); memory != nullptr) {
    const auto value = number_field(*memory, "vm_hwm_kib_end");
    if (value.has_value()) {
      metrics.peak_rss_kib = static_cast<std::uint64_t>(*value);
    }
  }
  return metrics;
}

bool read_file(const std::string& path, std::string& contents) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return false;
  }
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  if (size < 0 || static_cast<std::uint64_t>(size) > kFileLimit) {
    return false;
  }
  stream.seekg(0, std::ios::beg);
  contents.resize(static_cast<std::size_t>(size));
  stream.read(contents.data(), static_cast<std::streamsize>(contents.size()));
  return stream.good() || stream.eof();
}

}  // namespace

Result<Comparison, CompareError> compare_files(const std::string& left_path,
                                               const std::string& right_path) {
  std::string left_text;
  std::string right_text;
  if (!read_file(left_path, left_text) || !read_file(right_path, right_text)) {
    return CompareError::IoError;
  }
  Json left;
  Json right;
  Parser left_parser(left_text);
  Parser right_parser(right_text);
  if (!left_parser.parse(left) || !right_parser.parse(right)) {
    return CompareError::InvalidJson;
  }
  if (left.kind != Json::Kind::Object || right.kind != Json::Kind::Object) {
    return CompareError::NotAnObject;
  }

  const auto schema_ok = [](const Json& root) {
    const Json* schema = required_number(root, "schema_version");
    return schema != nullptr && schema->number == 1.0;
  };
  if (!schema_ok(left) || !schema_ok(right)) {
    return CompareError::InvalidSchema;
  }

  const auto status_ok = [](const Json& root) {
    return required_string(root, "status") != nullptr;
  };
  const auto model_ok = [](const Json& root) {
    const Json* model = object_field(root, "model");
    return model != nullptr && required_string(*model, "artifact_sha256") != nullptr &&
           required_number(*model, "file_bytes") != nullptr;
  };
  const auto options_ok = [](const Json& root) { return object_field(root, "options") != nullptr; };
  const auto output_ok = [](const Json& root) {
    const Json* output = object_field(root, "output");
    return output != nullptr && output->find("token_ids") != nullptr &&
           required_string(*output, "finish_reason") != nullptr;
  };
  if (!status_ok(left) || !status_ok(right) || !model_ok(left) || !model_ok(right) ||
      !options_ok(left) || !options_ok(right) || !output_ok(left) || !output_ok(right)) {
    return CompareError::InvalidMeasurement;
  }

  Comparison comparison;
  comparison.a = metrics_from(left);
  comparison.b = metrics_from(right);

  comparison.same_status =
      required_string(left, "status")->text == required_string(right, "status")->text;
  if (!comparison.same_status) {
    comparison.differences.push_back("status");
  }

  const Json* left_model = object_field(left, "model");
  const Json* right_model = object_field(right, "model");
  const auto left_sha = required_string(*left_model, "artifact_sha256")->text;
  const auto right_sha = required_string(*right_model, "artifact_sha256")->text;
  comparison.same_artifact = left_sha == right_sha;
  if (!comparison.same_artifact) {
    comparison.differences.push_back("model.artifact_sha256");
  }
  const double left_file_bytes = required_number(*left_model, "file_bytes")->number;
  const double right_file_bytes = required_number(*right_model, "file_bytes")->number;
  comparison.same_file_bytes = left_file_bytes == right_file_bytes;
  if (!comparison.same_file_bytes) {
    comparison.differences.push_back("model.file_bytes");
  }

  std::vector<std::int32_t> left_prompt;
  std::vector<std::int32_t> right_prompt;
  const auto left_prompt_status = read_ids(left, left.find("prompt_token_ids"), left_prompt);
  const auto right_prompt_status = read_ids(right, right.find("prompt_token_ids"), right_prompt);
  if (left_prompt_status != ArrayStatus::Valid || right_prompt_status != ArrayStatus::Valid) {
    return CompareError::InvalidMeasurement;
  }
  comparison.same_prompt_tokens = left_prompt_status == ArrayStatus::Valid &&
                                  right_prompt_status == ArrayStatus::Valid &&
                                  left_prompt == right_prompt;
  if (!comparison.same_prompt_tokens) {
    comparison.differences.push_back("prompt_token_ids");
  }

  const Json* left_output = object_field(left, "output");
  const Json* right_output = object_field(right, "output");
  std::vector<std::int32_t> left_tokens;
  std::vector<std::int32_t> right_tokens;
  const auto left_token_status = read_ids(left, left_output->find("token_ids"), left_tokens);
  const auto right_token_status = read_ids(right, right_output->find("token_ids"), right_tokens);
  if (left_token_status != ArrayStatus::Valid || right_token_status != ArrayStatus::Valid) {
    return CompareError::InvalidMeasurement;
  }
  comparison.same_output_tokens = left_token_status == ArrayStatus::Valid &&
                                  right_token_status == ArrayStatus::Valid &&
                                  left_tokens == right_tokens;
  if (!comparison.same_output_tokens) {
    comparison.differences.push_back("output.token_ids");
  }
  comparison.same_finish_reason = required_string(*left_output, "finish_reason")->text ==
                                  required_string(*right_output, "finish_reason")->text;
  if (!comparison.same_finish_reason) {
    comparison.differences.push_back("output.finish_reason");
  }

  const Json* left_options = object_field(left, "options");
  const Json* right_options = object_field(right, "options");
  static constexpr std::string_view kSettings[] = {
      "context_tokens", "max_new_tokens", "threads", "batch_size",    "seed",      "temperature",
      "top_k",          "top_p",          "mmap",    "chat_template", "gpu_layers"};
  comparison.same_settings = true;
  for (const auto key : kSettings) {
    if (!json_equal(left_options->find(key), right_options->find(key))) {
      comparison.same_settings = false;
      comparison.differences.push_back("options." + std::string(key));
    }
  }

  return comparison;
}

std::string to_json(const Comparison& comparison) {
  std::string output = "{\n  \"schema_version\": 1,\n";
  output +=
      std::string("  \"same_status\": ") + (comparison.same_status ? "true" : "false") + ",\n";
  output +=
      std::string("  \"same_artifact\": ") + (comparison.same_artifact ? "true" : "false") + ",\n";
  output += std::string("  \"same_file_bytes\": ") +
            (comparison.same_file_bytes ? "true" : "false") + ",\n";
  output +=
      std::string("  \"same_settings\": ") + (comparison.same_settings ? "true" : "false") + ",\n";
  output += std::string("  \"same_prompt_tokens\": ") +
            (comparison.same_prompt_tokens ? "true" : "false") + ",\n";
  output += std::string("  \"same_output_tokens\": ") +
            (comparison.same_output_tokens ? "true" : "false") + ",\n";
  output += std::string("  \"same_finish_reason\": ") +
            (comparison.same_finish_reason ? "true" : "false") + ",\n";
  output += "  \"metrics\": {\n    \"left\": ";
  write_metrics(output, comparison.a);
  output += ",\n    \"right\": ";
  write_metrics(output, comparison.b);
  const bool compatible = comparison.same_status && comparison.same_artifact &&
                          comparison.same_file_bytes && comparison.same_settings &&
                          comparison.same_prompt_tokens && comparison.same_output_tokens &&
                          comparison.same_finish_reason;
  output += std::string("  \"compatible\": ") + (compatible ? "true" : "false") + ",\n";
  output += "  \"differences\": [";
  for (std::size_t index = 0; index < comparison.differences.size(); ++index) {
    if (index != 0) {
      output += ", ";
    }
    output += "\"" + escape(comparison.differences[index]) + "\"";
  }
  output += "]\n}\n";
  return output;
}

}  // namespace ame::compare