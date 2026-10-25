#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "cancellation.hpp"
#include "file_source.hpp"
#include "result.hpp"

namespace ame::residency {

// One read into caller-owned memory. The destination stays valid until the batch has ended.
struct ReadJob {
  std::uint64_t offset{0};
  std::span<std::byte> destination;
  bool drop_cache{false};  // posix_fadvise(DONTNEED) on the extent after it is copied
  std::uint8_t group{0};   // completion group, 0 to kMaxGroups - 1
};

// A bounded set of reader threads over one FileSource. Reads belong to a batch; a batch ends
// only when every read has finished or been skipped. On the first error or on cancellation the
// batch stops queued reads and lets in-flight reads end at the next chunk, so once a batch has
// ended no worker touches a destination.
//
// read_all runs a batch and waits for all of it. With one thread and no background mode it runs
// on the calling thread, as before. submit starts a batch and returns at once; the caller waits
// for one completion group at a time (wait_group) or for the whole batch (wait_all) and must
// call wait_all (or end the batch by waiting for failure) before releasing any destination.
class ReaderPool {
 public:
  static constexpr unsigned kMaxReaders = 4;
  static constexpr unsigned kMaxGroups = 4;

  struct Batch {
    std::vector<ReadJob> jobs;
    const Cancellation* stop{nullptr};
    std::atomic<bool> abort{false};
    std::mutex mutex;
    std::condition_variable done;
    std::size_t finished{0};
    std::array<std::size_t, kMaxGroups> group_remaining{};
    std::array<bool, kMaxGroups> group_failed{};  // a read of the group failed or was skipped
    bool failed{false};
    FileError first_error{FileError::ReadFailed};
  };
  using BatchHandle = std::shared_ptr<Batch>;

  // threads is clamped to 1..kMaxReaders. cpus, when non-empty, is the CPU set each reader
  // thread is restricted to (best effort). background keeps reader threads even for one
  // reader, which submit needs.
  ReaderPool(const FileSource& source, unsigned threads, std::vector<int> cpus = {},
             bool background = false);
  ~ReaderPool();
  ReaderPool(const ReaderPool&) = delete;
  ReaderPool& operator=(const ReaderPool&) = delete;

  [[nodiscard]] unsigned threads() const noexcept { return threads_; }
  [[nodiscard]] bool background() const noexcept { return !workers_.empty(); }

  [[nodiscard]] Result<void, FileError> read_all(std::span<const ReadJob> jobs,
                                                 const Cancellation& stop);

  // Requires background(). The jobs are copied; their destinations are not.
  [[nodiscard]] BatchHandle submit(std::span<const ReadJob> jobs, const Cancellation& stop);
  // Waits until every job in `group` is done, or the batch has failed. Success means every read
  // of the group completed; a failure elsewhere does not fail a group that already completed.
  // After an error return, call wait_all before releasing any destination.
  [[nodiscard]] static Result<void, FileError> wait_group(const BatchHandle& batch, unsigned group);
  // Waits until the whole batch has ended.
  [[nodiscard]] static Result<void, FileError> wait_all(const BatchHandle& batch);

 private:
  void worker();

  const FileSource& source_;
  unsigned threads_;
  std::vector<int> cpus_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::deque<std::pair<BatchHandle, std::size_t>> queue_;
  bool shutting_down_{false};
};

}  // namespace ame::residency
