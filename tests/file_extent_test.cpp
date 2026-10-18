#include "file_extent.hpp"

#include <iostream>
#include <limits>
#include <type_traits>
#include <variant>

using ame::model::ByteCount;
using ame::model::CheckedFileExtent;
using ame::model::ExtentError;
using ame::model::FileOffset;
static_assert(!std::is_default_constructible_v<CheckedFileExtent>);
static_assert(!std::is_constructible_v<CheckedFileExtent, FileOffset, ByteCount>);
static_assert(!std::is_convertible_v<FileOffset, ByteCount>);

int main() {
  int failures = 0;
  const auto check = [&failures](std::uint64_t offset, std::uint64_t length, std::uint64_t size,
                                 bool accepted, ExtentError error = ExtentError::LengthBeyondFile) {
    const auto result = CheckedFileExtent::within({offset}, {length}, {size});
    const auto* extent = std::get_if<CheckedFileExtent>(&result);
    const bool correct = accepted ? extent != nullptr && extent->offset().value == offset &&
                                        extent->length().value == length
                                  : std::holds_alternative<ExtentError>(result) &&
                                        std::get<ExtentError>(result) == error;
    if (!correct) {
      std::cerr << "Extent failed: offset=" << offset << " length=" << length << " size=" << size
                << '\n';
      ++failures;
    }
  };
  check(9, 1, 10, true);
  check(10, 1, 10, false);
  check(11, 1, 10, false, ExtentError::OffsetBeyondFile);
  check(0, 0, 0, true);
  check(0, 1, 0, false);
  check(10, 0, 10, true);
  check(11, 0, 10, false, ExtentError::OffsetBeyondFile);
  constexpr std::uint64_t large = std::uint64_t{1} << 32;
  check(large - 1, 1, large, true);
  check(large, 1, large, false);
  check(large + 1, 1, large + 2, true);
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  check(0, max, max, true);
  check(max, 0, max, true);
  check(max - 1, 1, max, true);
  check(max, 1, max, false);
  check(max - 1, 2, max, false);
  check(1, max, max, false);
  check(max, max, 0, false, ExtentError::OffsetBeyondFile);
  return failures == 0 ? 0 : 1;
}
