#include "compare.hpp"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {
int failures = 0;
void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Compare test failure: " << message << '\n';
    ++failures;
  }
}
void write_file(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream << contents;
}

const std::string kBase = R"({
  "schema_version": 1,
  "status": "ok",
  "model": {"artifact_sha256": "aa", "file_bytes": 123},
  "options": {"context_tokens": 512, "max_new_tokens": 8, "threads": 4,
              "batch_size": 64, "seed": 1234, "temperature": 0,
              "top_k": 40, "top_p": 0.95, "mmap": true,
              "chat_template": false, "gpu_layers": 0},
  "prompt_token_ids": [1, 2],
  "output": {"token_ids": [3, 4], "finish_reason": "limit"},
  "timing": {"request_ttft_seconds": 1.0, "decode_tokens_per_second": 2.0},
  "memory": {"vm_hwm_kib_end": 100}
})";
}  // namespace

int main() {
  using ame::compare::CompareError;
  const auto directory = std::filesystem::temp_directory_path() /
                         ("ame_compare_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  write_file(directory / "a.json", kBase);
  write_file(directory / "b.json", kBase);

  auto same =
      ame::compare::compare_files((directory / "a.json").string(), (directory / "b.json").string());
  expect(same.ok(), "identical files parse");
  if (same.ok()) {
    const auto& value = same.value();
    expect(value.same_status && value.same_artifact && value.same_file_bytes &&
               value.same_settings && value.same_prompt_tokens && value.same_output_tokens &&
               value.same_finish_reason,
           "identical files compare equal");
    expect(value.differences.empty(), "no differences");
    expect(value.a.ttft_seconds == 1.0 && value.b.peak_rss_kib == 100, "metrics parsed");
    const std::string json = ame::compare::to_json(value);
    expect(json.find("\"compatible\": true") != std::string::npos, "compatible json");
  }

  std::string different = kBase;
  different.replace(different.find("[3, 4]"), 6, "[3, 5]");
  write_file(directory / "c.json", different);
  auto changed =
      ame::compare::compare_files((directory / "a.json").string(), (directory / "c.json").string());
  expect(changed.ok(), "changed token file parses");
  if (changed.ok()) {
    expect(!changed.value().same_output_tokens, "changed tokens detected");
    expect(!changed.value().differences.empty(), "difference listed");
  }

  std::string changed_settings = kBase;
  changed_settings.replace(changed_settings.find("\"context_tokens\": 512"), 20,
                           "\"context_tokens\": 1024");
  write_file(directory / "d.json", changed_settings);
  auto settings =
      ame::compare::compare_files((directory / "a.json").string(), (directory / "d.json").string());
  expect(settings.ok() && !settings.value().same_settings, "different settings detected");

  std::string fractional = kBase;
  fractional.replace(fractional.find("[3, 4]"), 6, "[3, 4.5]");
  write_file(directory / "float.json", fractional);
  auto fractional_result = ame::compare::compare_files((directory / "a.json").string(),
                                                       (directory / "float.json").string());
  expect(!fractional_result.ok() && fractional_result.error() == CompareError::InvalidMeasurement,
         "fractional token id rejected");

  std::string huge = kBase;
  huge.replace(huge.find("[3, 4]"), 6, "[3, 1e20]");
  write_file(directory / "huge.json", huge);
  auto huge_result = ame::compare::compare_files((directory / "a.json").string(),
                                                 (directory / "huge.json").string());
  expect(!huge_result.ok() && huge_result.error() == CompareError::InvalidMeasurement,
         "out-of-range token id rejected");

  std::string duplicate = kBase;
  duplicate.replace(duplicate.find("\"status\": \"ok\""), 13,
                    "\"status\": \"ok\", \"status\": \"ok\"");
  write_file(directory / "duplicate.json", duplicate);
  auto duplicate_result = ame::compare::compare_files((directory / "a.json").string(),
                                                      (directory / "duplicate.json").string());
  expect(!duplicate_result.ok() && duplicate_result.error() == CompareError::InvalidJson,
         "duplicate key rejected");

  std::string schema = kBase;
  schema.replace(schema.find("\"schema_version\": 1"), 18, "\"schema_version\": 2");
  write_file(directory / "schema.json", schema);
  auto schema_result = ame::compare::compare_files((directory / "a.json").string(),
                                                   (directory / "schema.json").string());
  expect(!schema_result.ok() && schema_result.error() == CompareError::InvalidSchema,
         "unsupported schema rejected");

  std::string missing_options = kBase;
  const auto options_begin = missing_options.find("  \"options\":");
  const auto options_end = missing_options.find("  \"prompt_token_ids\":");
  expect(options_begin != std::string::npos && options_end != std::string::npos,
         "fixture options markers");
  missing_options.erase(options_begin, options_end - options_begin);
  write_file(directory / "missing.json", missing_options);
  auto missing_result = ame::compare::compare_files((directory / "a.json").string(),
                                                    (directory / "missing.json").string());
  expect(!missing_result.ok() && missing_result.error() == CompareError::InvalidMeasurement,
         "missing required field rejected");

  write_file(directory / "bad.json", "{not json");
  auto invalid = ame::compare::compare_files((directory / "a.json").string(),
                                             (directory / "bad.json").string());
  expect(!invalid.ok() && invalid.error() == CompareError::InvalidJson, "invalid json rejected");

  const auto missing_file = ame::compare::compare_files((directory / "a.json").string(),
                                                        (directory / "missing-file.json").string());
  expect(!missing_file.ok() && missing_file.error() == CompareError::IoError,
         "missing file rejected");

  std::filesystem::remove_all(directory);
  return failures == 0 ? 0 : 1;
}
