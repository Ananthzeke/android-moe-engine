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
}  // namespace

int main() {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("ame_compare_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const std::string base = R"({"status":"ok","model":{"artifact_sha256":"aa"},)"
                           R"("prompt_token_ids":[1,2],"output":{"token_ids":[3,4],)"
                           R"("finish_reason":"limit"},"timing":{"request_ttft_seconds":1.0,)"
                           R"("decode_tokens_per_second":2.0},"memory":{"vm_hwm_kib_end":100}})";
  write_file(directory / "a.json", base);
  write_file(directory / "b.json", base);
  auto same =
      ame::compare::compare_files((directory / "a.json").string(), (directory / "b.json").string());
  expect(same.ok(), "identical files parse");
  if (same.ok()) {
    const auto& value = same.value();
    expect(value.same_status && value.same_artifact && value.same_prompt_tokens &&
               value.same_output_tokens && value.same_finish_reason,
           "identical files compare equal");
    expect(value.differences.empty(), "no differences");
    expect(value.a.ttft_seconds == 1.0 && value.b.peak_rss_kib == 100, "metrics parsed");
    const std::string json = ame::compare::to_json(value);
    expect(json.find("\"same_output_tokens\": true") != std::string::npos, "json output");
  }

  std::string different = base;
  different.replace(different.find("[3,4]"), 5, "[3,5]");
  write_file(directory / "c.json", different);
  auto changed =
      ame::compare::compare_files((directory / "a.json").string(), (directory / "c.json").string());
  expect(changed.ok(), "changed file parses");
  if (changed.ok()) {
    expect(!changed.value().same_output_tokens, "changed tokens detected");
    expect(!changed.value().differences.empty(), "difference listed");
  }

  write_file(directory / "bad.json", "{not json");
  auto invalid = ame::compare::compare_files((directory / "a.json").string(),
                                             (directory / "bad.json").string());
  expect(!invalid.ok() && invalid.error() == ame::compare::CompareError::InvalidJson,
         "invalid json rejected");

  auto missing = ame::compare::compare_files((directory / "a.json").string(),
                                             (directory / "missing.json").string());
  expect(!missing.ok() && missing.error() == ame::compare::CompareError::IoError,
         "missing file rejected");

  std::filesystem::remove_all(directory);
  return failures == 0 ? 0 : 1;
}