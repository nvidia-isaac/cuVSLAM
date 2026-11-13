
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

#include <cstddef>
#include <cstdint>
#include <string_view>

#if defined(__cplusplus) && __cplusplus >= 201703L
#define PROFILER_UNUSED [[maybe_unused]]
#else
#define PROFILER_UNUSED
#pragma warning(disable : 4100)
#endif

#ifdef USE_NVTX
#include "nvtx3/nvToolsExt.h"
#endif

#define TRACE_EVENT PROFILER_UNUSED auto

namespace cuvslam::profiler {

// Names the calling thread for the OS (gdb, top, perf) and, with USE_NVTX, for Nsight Systems.
// Call once from the thread itself: Nsight Systems rejects traces where one thread names another.
// Linux truncates names to 15 characters.
void NameCurrentThread(std::string_view name);

struct Disable {
  using RangeId_t = size_t;
  using DomainHandle_t = size_t*;

  static RangeId_t TraceEventStart(const char*) { return 0; }

  static void TraceEventEnd(RangeId_t) {}

  class TraceEvent {
  public:
    explicit TraceEvent(const char*) {}
  };

  static void Mark(const char*) {}

  class Domain {
  public:
    explicit Domain(const char*) {}
    Domain() = default;
    const DomainHandle_t& get_handle() const { return temp; }
    uint32_t get_color() const { return 0; };

  private:
    size_t* temp = nullptr;
  };

  static void DomainMark(DomainHandle_t, const char*, uint32_t color = 0) { (void)color; }

  static RangeId_t DomainTraceEventStart(DomainHandle_t, const char*, uint32_t color = 0) {
    (void)color;
    return 0;
  }

  static void DomainTraceEventEnd(DomainHandle_t, RangeId_t) {}

  class DomainTraceEvent {
  public:
    DomainTraceEvent(DomainHandle_t, const char*, uint32_t) {}
    DomainTraceEvent(DomainHandle_t, const char*) {}
    DomainTraceEvent() = default;
    void Pop() {}
  };

  class DomainHelper {
  public:
    explicit DomainHelper(const char*) {}
    DomainTraceEvent trace_event(const char*) const { return {nullptr, ""}; }
    DomainTraceEvent trace_event(const char*, uint32_t) const { return {nullptr, ""}; }
    void mark(const char*) const {}
    RangeId_t trace_event_start(const char*) const { return 0; }
    void trace_event_end(RangeId_t) const {}
  };
};

#ifdef USE_NVTX
struct Enable {
  using RangeId_t = nvtxRangeId_t;
  using DomainHandle_t = nvtxDomainHandle_t;

  static RangeId_t TraceEventStart(const char* message);

  static void TraceEventEnd(RangeId_t id);

  class TraceEvent {
  public:
    explicit TraceEvent(const char* message);
    ~TraceEvent();
  };

  static void Mark(const char* message);

  class Domain {
  public:
    explicit Domain(const char* message);
    Domain() = default;
    ~Domain();
    const DomainHandle_t& get_handle() const;
    uint32_t get_color() const;

  private:
    DomainHandle_t handle_;
    uint32_t color;
  };

  static void DomainMark(DomainHandle_t handle, const char* message, uint32_t color = 0);

  static RangeId_t DomainTraceEventStart(DomainHandle_t handle, const char* message, uint32_t color = 0);

  static void DomainTraceEventEnd(DomainHandle_t handle, RangeId_t id);

  class DomainTraceEvent {
  public:
    DomainTraceEvent(DomainHandle_t domain_handle, const char* message, uint32_t color);
    DomainTraceEvent(DomainHandle_t domain_handle, const char* message);  // random color
    DomainTraceEvent() = default;
    ~DomainTraceEvent();
    void Pop();

  private:
    void Push(const char* message, uint32_t color);
    DomainHandle_t domain_handle_;
  };

  class DomainHelper {
  public:
    explicit DomainHelper(const char* domain_name);
    DomainTraceEvent trace_event(const char* message) const;
    DomainTraceEvent trace_event(const char* message, uint32_t color) const;
    void mark(const char* message) const;
    RangeId_t trace_event_start(const char* message) const;
    void trace_event_end(RangeId_t id) const;

  private:
    Domain domain_;
  };
};
#else
using Enable = Disable;
#endif

}  // namespace cuvslam::profiler
