#include "reader_pool.hpp"

#include <sched.h>

#include <algorithm>
#include <tuple>

namespace ame::residency {

struct ReaderPool::Batch {
  std::span<const ReadJob> jobs;
  const Cancellation* stop{nullptr};
  std::atomic<bool> abort{false};
  std::mutex mutex;
  std::condition_variable done;
  std::size_t finished{0};  // jobs completed, failed, or skipped
  bool failed{false};
  FileError first_error{FileError::ReadFailed};
};

ReaderPool::ReaderPool(const FileSource& source, unsigned threads, std::vector<int> cpus)
    : source_(source), threads_(std::clamp(threads, 1U, kMaxReaders)), cpus_(std::move(cpus)) {
  if (threads_ > 1) {
    workers_.reserve(threads_);
    for (unsigned index = 0; index < threads_; ++index) {
      workers_.emplace_back([this] { worker(); });
    }
  }
}

ReaderPool::~ReaderPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutting_down_ = true;
  }
  work_ready_.notify_all();
  for (auto& worker : workers_) {
    worker.join();
  }
}

void ReaderPool::worker() {
#if defined(__linux__)
  if (!cpus_.empty()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : cpus_) {
      if (cpu >= 0 && cpu < CPU_SETSIZE) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
        CPU_SET(cpu, &set);
#pragma GCC diagnostic pop
      }
    }
    (void)::sched_setaffinity(0, sizeof(set), &set);
  }
#endif
  while (true) {
    Batch* batch = nullptr;
    std::size_t index = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_ready_.wait(lock, [this] { return shutting_down_ || !queue_.empty(); });
      if (queue_.empty()) {
        return;  // shutting down with nothing left
      }
      std::tie(batch, index) = queue_.front();
      queue_.pop_front();
    }
    FileError error = FileError::ReadFailed;
    bool ok = true;
    if (batch->abort.load(std::memory_order_acquire)) {
      ok = true;  // skipped after an earlier failure; not an error of its own
    } else {
      const ReadJob& job = batch->jobs[index];
      const auto result =
          source_.read_exact(job.offset, job.destination, *batch->stop, &batch->abort);
      if (!result.ok()) {
        ok = false;
        error = result.error();
      } else if (job.drop_cache) {
        source_.advise_dontneed(job.offset, job.destination.size());
      }
    }
    {
      std::lock_guard<std::mutex> lock(batch->mutex);
      if (!ok && !batch->failed) {
        batch->failed = true;
        batch->first_error = error;
        batch->abort.store(true, std::memory_order_release);
      }
      ++batch->finished;
      if (batch->finished == batch->jobs.size()) {
        batch->done.notify_all();
      }
    }
  }
}

Result<void, FileError> ReaderPool::read_all(std::span<const ReadJob> jobs,
                                             const Cancellation& stop) {
  if (jobs.empty()) {
    return {};
  }
  if (threads_ == 1) {
    for (const ReadJob& job : jobs) {
      const auto result = source_.read_exact(job.offset, job.destination, stop);
      if (!result.ok()) {
        return result.error();
      }
      if (job.drop_cache) {
        source_.advise_dontneed(job.offset, job.destination.size());
      }
    }
    return {};
  }
  Batch batch;
  batch.jobs = jobs;
  batch.stop = &stop;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t index = 0; index < jobs.size(); ++index) {
      queue_.emplace_back(&batch, index);
    }
  }
  work_ready_.notify_all();
  std::unique_lock<std::mutex> lock(batch.mutex);
  batch.done.wait(lock, [&] { return batch.finished == jobs.size(); });
  if (batch.failed) {
    return batch.first_error;
  }
  return {};
}

}  // namespace ame::residency
