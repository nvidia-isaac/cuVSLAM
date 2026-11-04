
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

#include <algorithm>
#include <cstdio>

#include "log/log.h"

// Root log
// Disable it to switch off all logs
#ifdef CUVSLAM_LOG_ENABLE
using LogRoot = cuvslam::log::Enable<>;
#else
using LogRoot = cuvslam::log::Disable<>;
#endif

// all Trace...() functions will be work via cuvslam::Log::message()
// #define USE_CUVSLAM_LOG

#ifdef USE_CUVSLAM_LOG
// TODO(drobustov): implement all all Trace...() HERE
// Messages:
using LogDebug = cuvslam::Log::Enable<log_root>;  // for trace

#else

namespace cuvslam::Trace {
enum class Verbosity { None = 0, Error, Warning, Message, Debug };

template <typename T>
Verbosity ToVerbosity(T val, Verbosity max_allowed = Verbosity::Debug) {
  return std::clamp(static_cast<Trace::Verbosity>(val), Trace::Verbosity::None, max_allowed);
}

void SetVerbosity(Verbosity verbosity);

Verbosity GetVerbosity();

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void PrintDecoratedMsg(const char* const prefix, const char* const strFormat, ...);

namespace detail {
// can't put static constant inside constexpr function (until C++23?)
constexpr const char* const Prefixes[] = {"", "ERROR", "WARNING", "MESSAGE", "DEBUG"};
}  // namespace detail

constexpr const char* GetPrefix(Verbosity verbosity) {
  return detail::Prefixes[static_cast<std::underlying_type_t<Verbosity> >(verbosity)];
}
}  // end namespace cuvslam::Trace

// Implementation macro, use other Trace* macros instead
#define TraceImpl(V, ...)                                             \
  do {                                                                \
    if (::cuvslam::Trace::GetVerbosity() >= (V)) {                    \
      ::cuvslam::Trace::PrintDecoratedMsg(GetPrefix(V), __VA_ARGS__); \
    }                                                                 \
  } while (0)

// TODO(vikuznetsov): delete TracePrint?
#define TracePrint(...) printf(__VA_ARGS__)
#define TraceError(...) TraceImpl(::cuvslam::Trace::Verbosity::Error, __VA_ARGS__)
#define TraceWarning(...) TraceImpl(::cuvslam::Trace::Verbosity::Warning, __VA_ARGS__)
#define TraceMessage(...) TraceImpl(::cuvslam::Trace::Verbosity::Message, __VA_ARGS__)
#if defined(NDEBUG)
#define TraceDebug(...) (void(0))
#else
#define TraceDebug(...) TraceImpl(::cuvslam::Trace::Verbosity::Debug, __VA_ARGS__)
#endif

#define TracePrintIf(cond, ...) \
  if (cond) {                   \
    TracePrint(__VA_ARGS__);    \
  }
#define TraceErrorIf(cond, ...) \
  if (cond) {                   \
    TraceError(__VA_ARGS__);    \
  }
#define TraceWarningIf(cond, ...) \
  if (cond) {                     \
    TraceWarning(__VA_ARGS__);    \
  }
#define TraceMessageIf(cond, ...) \
  if (cond) {                     \
    TraceMessage(__VA_ARGS__);    \
  }
#define TraceDebugIf(cond, ...) \
  if (cond) {                   \
    TraceDebug(__VA_ARGS__);    \
  }

#endif
