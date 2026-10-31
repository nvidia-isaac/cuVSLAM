
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#pragma once

#include <condition_variable>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "profiler/profiler.h"

namespace cuvslam {

template <class RESULT>
class WorkThread {
public:
  using result_type = RESULT;

  // thread_name names the worker thread, see profiler::NameCurrentThread().
  explicit WorkThread(std::string thread_name)
      : threadName_(std::move(thread_name)), mustTerminate_(false), isRunning_(true) {}

  ~WorkThread() { reset(); }

  void start() {
    isRunning_ = true;
    condHasInput_.notify_all();
  }

  void stop() { isRunning_ = false; }
  bool isRunning() const { return isRunning_; }

  template <class F>
  std::future<result_type> addTask(std::launch policy, F&& fn) {
    return doAddTask(std::packaged_task<result_type()>(std::forward<F>(fn)), policy);
  }

  void reset() {
    std::unique_lock<std::mutex> lk(mtxHasInput_);
    mustTerminate_ = true;
    condHasInput_.notify_all();
    lk.unlock();

    if (thread_.joinable()) {
      thread_.join();
    }
  }

private:
  std::future<result_type> doAddTask(std::packaged_task<result_type()> fn, std::launch policy) {
    if (policy == std::launch::async) {
      if (!thread_.joinable()) thread_ = std::thread([this] { run(); });

      std::unique_lock<std::mutex> lk(mtxHasInput_);
      input_.emplace_back(std::move(fn));
      condHasInput_.notify_all();

      return input_.back().get_future();
    } else {
      fn();
      return fn.get_future();
    }
  }

  void run() {
    profiler::NameCurrentThread(threadName_);
    std::unique_lock<std::mutex> lk(mtxHasInput_);
    while (!mustTerminate_) {
      condHasInput_.wait(lk, [this] { return (isRunning_ && !input_.empty()) || mustTerminate_; });

      auto inputs = std::move(input_);

      lk.unlock();
      for (auto& input : inputs) {
        input();
      }
      lk.lock();
    }
  }

  std::vector<std::packaged_task<result_type()>> input_;

  const std::string threadName_;
  bool mustTerminate_;
  bool isRunning_;
  std::thread thread_;
  std::mutex mtxHasInput_;
  std::condition_variable condHasInput_;
};

}  // namespace cuvslam
