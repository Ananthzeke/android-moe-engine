#pragma once

#include <atomic>

namespace ame {

// Project-owned cancellation flag. The pinned Android NDK's libc++ does not expose
// std::stop_token on this target, so the internal boundary uses a small atomic
// token instead. Cross-thread request is allowed; the inference owner drains work.
class Cancellation {
 public:
  void request() noexcept { requested_.store(true, std::memory_order_relaxed); }
  void reset() noexcept { requested_.store(false, std::memory_order_relaxed); }
  [[nodiscard]] bool requested() const noexcept {
    return requested_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<bool> requested_{false};
};

}  // namespace ame