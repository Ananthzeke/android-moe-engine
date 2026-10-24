#pragma once

#include <functional>
#include <utility>

namespace ame::server {

// One admission-owned request lifetime. The backend work has already drained and the
// request context is destroyed before finish() runs. finish() then releases expert
// leases, publishes the terminal snapshot while admission is still exclusive, and
// releases admission last, so the next request cannot enter while stats are read.
class RequestScope {
 public:
  struct Hooks {
    std::function<void()> release_leases;
    std::function<void()> publish_snapshot;
    std::function<void()> release_admission;
  };

  explicit RequestScope(Hooks hooks) : hooks_(std::move(hooks)) {}
  RequestScope(const RequestScope&) = delete;
  RequestScope& operator=(const RequestScope&) = delete;
  ~RequestScope() { finish(); }

  void finish() noexcept {
    if (!active_) return;
    active_ = false;
    try {
      hooks_.release_leases();
    } catch (...) {
    }
    try {
      hooks_.publish_snapshot();
    } catch (...) {
    }
    hooks_.release_admission();
  }

 private:
  Hooks hooks_;
  bool active_ = true;
};

}  // namespace ame::server
