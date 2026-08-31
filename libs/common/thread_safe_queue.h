
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

#include "Eigen/Core"

#include <condition_variable>
#include <mutex>
#include <queue>

namespace cuvslam {

// TODO: allocator for align-required structures
template <class T>
class ThreadSafeQueue {
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable event_;
  bool abort_ = false;

public:
  void Push(const T& value) {
    std::unique_lock<std::mutex> locker(mutex_);
    if (abort_) {
      return;
    }
    queue_.push(value);
    event_.notify_all();
  }
  void Abort() {
    std::unique_lock<std::mutex> locker(mutex_);
    abort_ = true;
    event_.notify_all();
  }
  // return false if queue aborted
  bool Wait() {
    std::unique_lock<std::mutex> locker(mutex_);
    while (true) {
      if (abort_) {
        return false;
      }
      if (!queue_.empty()) {
        return true;
      }
      event_.wait(locker);
    }
    return false;
  }
  // return false if queue aborted
  bool WaitPop(T& value) {
    std::unique_lock<std::mutex> locker(mutex_);
    while (true) {
      if (abort_) {
        return false;
      }
      if (!queue_.empty()) {
        value = queue_.front();
        return true;
      }

      event_.wait(locker);
    }
    return false;
  }
  // return true if value was popped
  bool TryPop(T& value) {
    std::unique_lock<std::mutex> locker(mutex_);
    if (!queue_.empty()) {
      value = queue_.front();
      queue_.pop();
      return true;
    }
    return false;
  }
  bool IsEmpty() {
    std::unique_lock<std::mutex> locker(mutex_);
    return queue_.empty();
  }
  // number of commands not yet taken by the consumer thread
  size_t Size() {
    std::unique_lock<std::mutex> locker(mutex_);
    return queue_.size();
  }
};

}  // namespace cuvslam
