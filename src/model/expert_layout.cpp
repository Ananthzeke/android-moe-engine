#include "expert_layout.hpp"

#include <limits>

namespace ame::model {
namespace {

bool aligned(std::uint64_t value) { return value % kTensorAlignment == 0; }

}  // namespace

Result<ExpertTable, LayoutError> ExpertTable::create(FileOffset base, ByteCount expert_stride,
                                                     ByteCount payload_bytes,
                                                     std::uint32_t expert_count,
                                                     std::uint32_t layer, ByteCount file_size) {
  if (expert_count == 0) {
    return LayoutError::ZeroCount;
  }
  if (expert_stride.value == 0) {
    return LayoutError::ZeroStride;
  }
  if (payload_bytes.value == 0) {
    return LayoutError::ZeroPayload;
  }
  if (payload_bytes.value > expert_stride.value) {
    return LayoutError::PayloadExceedsStride;
  }
  if (!aligned(base.value) || !aligned(expert_stride.value) || !aligned(payload_bytes.value)) {
    return LayoutError::Misaligned;
  }
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t last_index = static_cast<std::uint64_t>(expert_count) - 1;
  if (expert_stride.value != 0 && last_index > (max - base.value) / expert_stride.value) {
    return LayoutError::OffsetOverflow;
  }
  const std::uint64_t last_offset = base.value + last_index * expert_stride.value;
  const auto checked = CheckedFileExtent::within(FileOffset{last_offset}, payload_bytes, file_size);
  if (std::holds_alternative<ExtentError>(checked)) {
    return LayoutError::BeyondFile;
  }
  return ExpertTable(base, expert_stride, payload_bytes, expert_count, layer);
}

Result<FileExtent, LayoutError> ExpertTable::extent(ExpertKey key) const {
  if (key.layer != layer_ || key.expert >= expert_count_) {
    return LayoutError::KeyOutsideTable;
  }
  if (key.expert == 0) {
    return FileExtent{base_, payload_bytes_};
  }
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t index = static_cast<std::uint64_t>(key.expert);
  if (expert_stride_.value != 0 && index > (max - base_.value) / expert_stride_.value) {
    return LayoutError::OffsetOverflow;
  }
  return FileExtent{FileOffset{base_.value + index * expert_stride_.value}, payload_bytes_};
}

}  // namespace ame::model
