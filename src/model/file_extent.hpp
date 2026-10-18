#pragma once

#include <cstdint>
#include <variant>

namespace ame::model {
struct FileOffset {
  std::uint64_t value;
};
struct ByteCount {
  std::uint64_t value;
};
enum class ExtentError { OffsetBeyondFile, LengthBeyondFile };
class CheckedFileExtent {
 public:
  static std::variant<CheckedFileExtent, ExtentError> within(FileOffset offset, ByteCount length,
                                                             ByteCount file_size) noexcept;
  [[nodiscard]] FileOffset offset() const noexcept { return offset_; }
  [[nodiscard]] ByteCount length() const noexcept { return length_; }

 private:
  CheckedFileExtent(FileOffset offset, ByteCount length) noexcept;
  FileOffset offset_;
  ByteCount length_;
};
}  // namespace ame::model
