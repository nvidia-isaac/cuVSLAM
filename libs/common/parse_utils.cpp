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

#include "common/parse_utils.h"

#include <charconv>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <system_error>

namespace cuvslam::common {

namespace {

[[noreturn]] void ThrowParseError(std::string_view expected, std::string_view v) {
  std::string message = "expected ";
  message.append(expected);
  message.append(", got: ");
  message.append(v);
  throw std::runtime_error(message);
}

char ToLowerAscii(char c) {
  if (c >= 'A' && c <= 'Z') {
    return static_cast<char>(c + ('a' - 'A'));
  }
  return c;
}

bool EqualsIgnoreCaseAscii(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (ToLowerAscii(lhs[i]) != ToLowerAscii(rhs[i])) {
      return false;
    }
  }
  return true;
}

template <typename T>
T Parse(std::string_view v, std::string_view expected) {
  if (v.empty()) {
    ThrowParseError(expected, v);
  }
  T parsed{};
  const char* begin = v.data();
  const char* end = begin + v.size();
  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end) {
    ThrowParseError(expected, v);
  }
  return parsed;
}

}  // namespace

bool ParseBool(std::string_view v) {
  if (EqualsIgnoreCaseAscii(v, "true") || v == "1") return true;
  if (EqualsIgnoreCaseAscii(v, "false") || v == "0") return false;
  ThrowParseError("bool (true/false/1/0)", v);
}

float ParseFloat(std::string_view v) { return Parse<float>(v, "float"); }

int32_t ParseInt32(std::string_view v) { return Parse<int32_t>(v, "int32"); }

int64_t ParseInt64(std::string_view v) { return Parse<int64_t>(v, "int64"); }

uint32_t ParseUInt32(std::string_view v) { return Parse<uint32_t>(v, "uint32"); }

uint64_t ParseUInt64(std::string_view v) { return Parse<uint64_t>(v, "uint64"); }

}  // namespace cuvslam::common
