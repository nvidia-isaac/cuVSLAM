
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

#include <random>
#include <string>

#ifdef USE_NVTX
#ifdef _WIN32
#include <windows.h>
#define NameOsThread() \
  nvtxNameOsThreadA(GetCurrentThreadId(), ("Thread_" + std::to_string(GetCurrentThreadId())).c_str())
#endif

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "profiler/profiler.h"

#ifdef __linux__
namespace {
// Each thread names itself once: naming another thread's id from a worker emits out-of-order events
// that Nsight Systems rejects on import.
void NameOsThread() {
  thread_local bool named = false;
  if (!named) {
    const auto tid = static_cast<uint32_t>(syscall(SYS_gettid));
    nvtxNameOsThreadA(tid, ("Thread_" + std::to_string(tid)).c_str());
    named = true;
  }
}
}  // namespace
#endif

namespace cuvslam::profiler {

Enable::RangeId_t Enable::TraceEventStart(const char* message) {
  NameOsThread();
  return nvtxRangeStartA(message);
}

void Enable::TraceEventEnd(RangeId_t id) { nvtxRangeEnd(id); }

Enable::TraceEvent::TraceEvent(const char* message) {
  NameOsThread();
  nvtxRangePushA(message);
}

Enable::TraceEvent::~TraceEvent() { nvtxRangePop(); }

void Enable::Mark(const char* message) {
  NameOsThread();
  nvtxMarkA(message);
}

nvtxEventAttributes_t default_attributes(const char* message, uint32_t color = 0) {
  nvtxMessageValue_t message_;
  message_.ascii = message;
  nvtxEventAttributes_t eventAttrib = {
      NVTX_VERSION, NVTX_EVENT_ATTRIB_STRUCT_SIZE, 0,       NVTX_COLOR_ARGB, color, NVTX_PAYLOAD_UNKNOWN, 0,
      {},           NVTX_MESSAGE_TYPE_ASCII,       message_};
  return eventAttrib;
}

Enable::Domain::Domain(const char* message) {
  handle_ = nvtxDomainCreateA(message);
  std::random_device rd;   // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(0, 0xFFFFFF);
  color = distrib(gen);
}

Enable::Domain::~Domain() { nvtxDomainDestroy(handle_); }

const Enable::DomainHandle_t& Enable::Domain::get_handle() const { return handle_; }

uint32_t Enable::Domain::get_color() const { return color; }

void Enable::DomainMark(DomainHandle_t handle, const char* message, uint32_t color) {
  nvtxEventAttributes_t eventAttrib = default_attributes(message, color);
  NameOsThread();
  nvtxDomainMarkEx(handle, &eventAttrib);
}

Enable::RangeId_t Enable::DomainTraceEventStart(DomainHandle_t handle, const char* message, uint32_t color) {
  nvtxEventAttributes_t eventAttrib = default_attributes(message, color);
  NameOsThread();
  return nvtxDomainRangeStartEx(handle, &eventAttrib);
}

void Enable::DomainTraceEventEnd(DomainHandle_t handle, RangeId_t id) { nvtxDomainRangeEnd(handle, id); }

Enable::DomainTraceEvent::DomainTraceEvent(DomainHandle_t handle, const char* message) : domain_handle_(handle) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, 0xFFFFFF);
  uint32_t color = distrib(gen);
  Push(message, color);
}

Enable::DomainTraceEvent::DomainTraceEvent(DomainHandle_t handle, const char* message, uint32_t color)
    : domain_handle_(handle) {
  Push(message, color);
}

Enable::DomainTraceEvent::~DomainTraceEvent() { Pop(); }

void Enable::DomainTraceEvent::Push(const char* message, uint32_t color) {
  auto eventAttrib = default_attributes(message, color);
  NameOsThread();
  nvtxDomainRangePushEx(domain_handle_, &eventAttrib);
}

void Enable::DomainTraceEvent::Pop() {
  if (domain_handle_) {
    nvtxDomainRangePop(domain_handle_);
    domain_handle_ = nullptr;
  }
}

Enable::DomainHelper::DomainHelper(const char* domain_name) : domain_(Domain(domain_name)) {}

Enable::DomainTraceEvent Enable::DomainHelper::trace_event(const char* message) const {
  return DomainTraceEvent(domain_.get_handle(), message);
}

Enable::DomainTraceEvent Enable::DomainHelper::trace_event(const char* message, uint32_t color) const {
  return DomainTraceEvent(domain_.get_handle(), message, color);
}

void Enable::DomainHelper::mark(const char* message) const {
  DomainMark(domain_.get_handle(), message, domain_.get_color());
}

Enable::RangeId_t Enable::DomainHelper::trace_event_start(const char* message) const {
  return DomainTraceEventStart(domain_.get_handle(), message, domain_.get_color());
}

void Enable::DomainHelper::trace_event_end(RangeId_t id) const { return DomainTraceEventEnd(domain_.get_handle(), id); }
#endif

}  // namespace cuvslam::profiler
