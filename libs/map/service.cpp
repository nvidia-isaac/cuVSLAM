
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

#include "map/service.h"

#include <utility>

#include "profiler/profiler.h"

namespace cuvslam::map {

ServiceBase::ServiceBase(UnifiedMap& map, std::string thread_name, bool async)
    : map_(map), async_(async), thread_name_(std::move(thread_name)) {
  if (!async_) {
    return;
  }

  job_ = std::thread(&ServiceBase::run, this);
}

ServiceBase::~ServiceBase() { stop(); }

void ServiceBase::restart() {
  stop();

  if (!async_) {
    return;
  }

  job_ = std::thread(&ServiceBase::run, this);
}

void ServiceBase::stop() {
  if (!async_) {
    return;
  }

  {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_one();
  if (job_.joinable()) {
    job_.join();
  }
}

void ServiceBase::notify() {
  if (!async_) {
    service_task();
    return;
  }
  // when the new keyframe arrives
  {
    std::unique_lock<std::mutex> lock(mutex_);
    inputs_ready_ = true;
  }
  cv_.notify_one();
}

void ServiceBase::run() {
  profiler::NameCurrentThread(thread_name_);
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return inputs_ready_ || stop_; });

      if (stop_) {
        return;
      }

      // Clear the flag *before* releasing the lock.  Any trigger() call that
      // arrives while service_task() is running (lock not held) will set
      // inputs_ready_ = true again, so the loop will re-enter service_task()
      // for that keyframe rather than silently dropping it.
      inputs_ready_ = false;
      on_task_start();  // snapshot pending state while lock is held
    }                   // lock released before service_task()

    service_task();
  }
};

}  // namespace cuvslam::map
