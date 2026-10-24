#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "cancellation.hpp"
#include "file_source.hpp"
#include "result.hpp"

namespace ame::residency {

// One read into caller-owned memory. The destination stays valid until read_all returns.
struct ReadJob {
  std::uint64_t offset{0};
  std::span<std::byte> destination;
  bool drop_cache{false};  // posix_fadvise(DONTNEED) on the extent after it is copied
};

// A bounded set of reader threads over one FileSource. With one thread (the default) jobs run
// on the calling thread, exactly as before. With more, every job is submitted, the caller waits
// on a condition variable, and read_all returns only after every job has finished or been
// skipped: on the first error or on cancellation it stops queued jobs, lets in-flight reads end,
// and joins them all, so no worker touches a destination after the call returns.
class ReaderPool {
 public:
  // threads is clamped to 1..kMaxReaders. cpus, when non-empty, is the CPU set each reader
  // thread is restricted to (best effort; an unsupported platform ignores it).
  ReaderPool(const FileSource& source, unsigned threads, std::vector<int> cpus = {});
  ~ReaderPool();
  ReaderPool(const ReaderPool&) = delete;
  ReaderPool& operator=(const ReaderPool&) = delete;

  static constexpr unsigned kMaxReaders = 4;
  [[nodiscard]] unsigned threads() const noexcept { return threads_; }

  // Runs all jobs. Returns the first error observed (ReadFailed, UnexpectedEnd, or
  // Cancelled), or success when every job completed.
  [[nodiscard]] Result<void, FileError> read_all(std::span<const ReadJob> jobs,
                                                 const Cancellation& stop);

 private:
  struct Batch;
  void worker();

  const FileSource& source_;
  unsigned threads_;
  std::vector<int> cpus_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::deque<std::pair<Batch*, std::size_t>> queue_;
  bool shutting_down_{false};
};

}  // namespace ame::residency
