#include "reader_pool.hpp"

#include <sched.h>

#include <algorithm>
#include <tuple>

namespace ame::residency {

ReaderPool::ReaderPool(const FileSource& source, unsigned threads, std::vector<int> cpus,
                       bool background)
    : source_(source), threads_(std::clamp(threads, 1U, kMaxReaders)), cpus_(std::move(cpus)) {
  if (threads_ > 1 || background) {
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
    BatchHandle batch;
    std::size_t index = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_ready_.wait(lock, [this] { return shutting_down_ || !queue_.empty(); });
      if (queue_.empty()) {
        return;
      }
      std::tie(batch, index) = queue_.front();
      queue_.pop_front();
    }
    FileError error = FileError::ReadFailed;
    bool ok = true;
    const ReadJob& job = batch->jobs[index];
    bool skipped = false;
    if (batch->abort.load(std::memory_order_acquire)) {
      skipped = true;
    } else {
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
      if (!ok || skipped) {
        batch->group_failed[job.group] = true;
      }
      if (!ok && !batch->failed) {
        batch->failed = true;
        batch->first_error = error;
        batch->abort.store(true, std::memory_order_release);
      }
      --batch->group_remaining[job.group];
      ++batch->finished;
      batch->done.notify_all();
    }
  }
}

ReaderPool::BatchHandle ReaderPool::submit(std::span<const ReadJob> jobs,
                                           const Cancellation& stop) {
  auto batch = std::make_shared<Batch>();
  batch->jobs.assign(jobs.begin(), jobs.end());
  batch->stop = &stop;
  for (ReadJob& job : batch->jobs) {
    job.group = std::min<std::uint8_t>(job.group, kMaxGroups - 1);
    ++batch->group_remaining[job.group];
  }
  if (batch->jobs.empty()) {
    return batch;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t index = 0; index < batch->jobs.size(); ++index) {
      queue_.emplace_back(batch, index);
    }
  }
  work_ready_.notify_all();
  return batch;
}

Result<void, FileError> ReaderPool::wait_group(const BatchHandle& batch, unsigned group) {
  std::unique_lock<std::mutex> lock(batch->mutex);
  batch->done.wait(lock, [&] { return batch->failed || batch->group_remaining[group] == 0; });
  if (batch->group_remaining[group] == 0 && !batch->group_failed[group]) {
    return {};
  }
  return batch->failed ? batch->first_error : FileError::ReadFailed;
}

Result<void, FileError> ReaderPool::wait_all(const BatchHandle& batch) {
  std::unique_lock<std::mutex> lock(batch->mutex);
  batch->done.wait(lock, [&] { return batch->finished == batch->jobs.size(); });
  if (batch->failed) {
    return batch->first_error;
  }
  return {};
}

Result<void, FileError> ReaderPool::read_all(std::span<const ReadJob> jobs,
                                             const Cancellation& stop) {
  if (jobs.empty()) {
    return {};
  }
  if (workers_.empty()) {
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
  return wait_all(submit(jobs, stop));
}

}  // namespace ame::residency
