#include "request_scope.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
int failures = 0;
void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Request scope test failure: " << message << '\n';
    ++failures;
  }
}

struct Recorder {
  std::vector<std::string> events;
  bool throw_on_release_leases{false};
  bool throw_on_publish{false};

  ame::server::RequestScope::Hooks hooks() {
    return {[this] {
              events.push_back("leases");
              if (throw_on_release_leases) throw std::runtime_error("lease");
            },
            [this] {
              events.push_back("publish");
              if (throw_on_publish) throw std::runtime_error("publish");
            },
            [this] { events.push_back("admission"); }};
  }
};

const std::vector<std::string> kOrder = {"leases", "publish", "admission"};
}  // namespace

int main() {
  {
    Recorder recorder;
    ame::server::RequestScope scope(recorder.hooks());
    expect(recorder.events.empty(), "nothing runs before the request ends");
    scope.finish();
    expect(recorder.events == kOrder, "leases, then snapshot, then admission last");
    scope.finish();
    expect(recorder.events == kOrder, "finish is idempotent");
  }
  {
    // Early return, such as a failed SSE write: only the destructor runs.
    Recorder recorder;
    { ame::server::RequestScope scope(recorder.hooks()); }
    expect(recorder.events == kOrder, "destructor performs the full cleanup order");
  }
  {
    // Shared ownership by a response provider: cleanup waits for the last owner.
    Recorder recorder;
    auto scope = std::make_shared<ame::server::RequestScope>(recorder.hooks());
    auto copy = scope;
    scope.reset();
    expect(recorder.events.empty(), "a remaining owner keeps admission held");
    copy.reset();
    expect(recorder.events == kOrder, "last owner releases in order");
  }
  {
    Recorder recorder;
    recorder.throw_on_release_leases = true;
    recorder.throw_on_publish = true;
    ame::server::RequestScope scope(recorder.hooks());
    scope.finish();
    expect(recorder.events == kOrder, "admission is released even if earlier steps throw");
  }
  return failures == 0 ? 0 : 1;
}
