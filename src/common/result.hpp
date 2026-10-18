#pragma once

#include <optional>
#include <utility>
#include <variant>

namespace ame {

// Project-local value-or-error type. Both alternatives are explicit; there is no
// implicit conversion from a value to success or from a missing value to success.
template <typename T, typename E>
class Result {
 public:
  Result(T value) : state_(std::move(value)) {}
  Result(E error) : state_(std::move(error)) {}

  [[nodiscard]] bool ok() const noexcept { return std::holds_alternative<T>(state_); }
  [[nodiscard]] const T& value() const& { return std::get<T>(state_); }
  [[nodiscard]] T&& value() && { return std::get<T>(std::move(state_)); }
  [[nodiscard]] E error() const { return std::get<E>(state_); }

 private:
  std::variant<T, E> state_;
};

template <typename E>
class Result<void, E> {
 public:
  Result() : error_(std::nullopt) {}
  Result(E error) : error_(error) {}

  [[nodiscard]] bool ok() const noexcept { return !error_.has_value(); }
  [[nodiscard]] E error() const { return *error_; }

 private:
  std::optional<E> error_;
};

}  // namespace ame
