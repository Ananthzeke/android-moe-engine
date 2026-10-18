#include "file_extent.hpp"

namespace ame::model {
CheckedFileExtent::CheckedFileExtent(FileOffset offset, ByteCount length) noexcept
    : offset_(offset), length_(length) {}
std::variant<CheckedFileExtent, ExtentError> CheckedFileExtent::within(
    FileOffset offset, ByteCount length, ByteCount file_size) noexcept {
  if (offset.value > file_size.value) return ExtentError::OffsetBeyondFile;
  if (length.value > file_size.value - offset.value) return ExtentError::LengthBeyondFile;
  return CheckedFileExtent(offset, length);
}
}  // namespace ame::model
