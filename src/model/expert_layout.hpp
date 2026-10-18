#pragma once

#include <cstdint>

#include "file_extent.hpp"
#include "result.hpp"

namespace ame::model {

inline constexpr std::uint64_t kTensorAlignment = 32;

struct ExpertKey {
  std::uint32_t layer{0};
  std::uint32_t expert{0};
};

[[nodiscard]] constexpr bool operator==(const ExpertKey& left, const ExpertKey& right) noexcept {
  return left.layer == right.layer && left.expert == right.expert;
}
[[nodiscard]] constexpr bool operator<(const ExpertKey& left, const ExpertKey& right) noexcept {
  return left.layer < right.layer || (left.layer == right.layer && left.expert < right.expert);
}

struct FileExtent {
  FileOffset offset;
  ByteCount length;
};

enum class LayoutError {
  ZeroCount,
  ZeroStride,
  ZeroPayload,
  PayloadExceedsStride,
  Misaligned,
  OffsetOverflow,
  LengthOverflow,
  BeyondFile,
  KeyOutsideTable,
};

// One contiguous table of equally sized expert payloads with a possibly larger
// stride. The factory validates alignment, arithmetic, and file bounds. It does not
// prove that the bytes are a valid quantized tensor; that is the model index's job.
class ExpertTable {
 public:
  static Result<ExpertTable, LayoutError> create(FileOffset base, ByteCount expert_stride,
                                                 ByteCount payload_bytes,
                                                 std::uint32_t expert_count, std::uint32_t layer,
                                                 ByteCount file_size);

  [[nodiscard]] std::uint32_t layer() const noexcept { return layer_; }
  [[nodiscard]] std::uint32_t expert_count() const noexcept { return expert_count_; }
  [[nodiscard]] ByteCount payload_bytes() const noexcept { return payload_bytes_; }
  [[nodiscard]] ByteCount expert_stride() const noexcept { return expert_stride_; }
  [[nodiscard]] FileOffset base() const noexcept { return base_; }

  [[nodiscard]] Result<FileExtent, LayoutError> extent(ExpertKey key) const;

 private:
  ExpertTable(FileOffset base, ByteCount expert_stride, ByteCount payload_bytes,
              std::uint32_t expert_count, std::uint32_t layer)
      : base_(base),
        expert_stride_(expert_stride),
        payload_bytes_(payload_bytes),
        expert_count_(expert_count),
        layer_(layer) {}

  FileOffset base_;
  ByteCount expert_stride_;
  ByteCount payload_bytes_;
  std::uint32_t expert_count_;
  std::uint32_t layer_;
};

}  // namespace ame::model
