
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
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "map/map.h"

namespace cuvslam::map {

class ServiceBase {
public:
  // thread_name names the async worker thread, see profiler::NameCurrentThread().
  ServiceBase(UnifiedMap& map, std::string thread_name, bool async = true);
  virtual ~ServiceBase();

  // some operation that must be done with the map;
  // may be SBA or (LC + PGO) etc
  // the user must define this function
  virtual void service_task() = 0;

  // notify the service that the new data has arrived
  // e.g. the keyframe was added to the map
  void notify();

  // waits for unfinished jobs and restarts the running thread
  void restart();

protected:
  void stop();

  // Called with mutex held, just before the lock is released for service_task().
  // Derived classes override this to snapshot any pending per-task state into
  // fields that service_task() will read without holding the lock.
  virtual void on_task_start() {}

  UnifiedMap& map_;
  const bool async_ = true;
  bool inputs_ready_ = false;
  bool stop_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;

private:
  void run();
  const std::string thread_name_;
  std::thread job_;
};

}  // namespace cuvslam::map
