
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

#include <string>

#include "profiler/profiler.h"

#if defined(__linux__)
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace cuvslam::profiler {

void NameCurrentThread(std::string_view name) {
#if defined(__linux__)
  // pthread_setname_np fails with ERANGE instead of truncating names longer than 15 characters.
  const std::string os_name(name.substr(0, 15));
  pthread_setname_np(pthread_self(), os_name.c_str());
#ifdef USE_NVTX
  nvtxNameOsThreadA(static_cast<uint32_t>(syscall(SYS_gettid)), os_name.c_str());
#endif
#elif defined(_WIN32)
  const std::string os_name(name);
  SetThreadDescription(GetCurrentThread(), std::wstring(os_name.begin(), os_name.end()).c_str());
#ifdef USE_NVTX
  nvtxNameOsThreadA(GetCurrentThreadId(), os_name.c_str());
#endif
#else
  (void)name;
#endif
}

}  // namespace cuvslam::profiler
