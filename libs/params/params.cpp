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

#include "params/params.h"

#include <cstdio>
#include <stdexcept>

namespace cuvslam::params::detail {

namespace {

std::string_view Trim(std::string_view text) {
  const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  while (!text.empty() && is_space(text.front())) {
    text.remove_prefix(1);
  }
  while (!text.empty() && is_space(text.back())) {
    text.remove_suffix(1);
  }
  return text;
}

}  // namespace

std::optional<bool> ParseOptionalBool(std::string_view text) {
  const std::string_view trimmed = Trim(text);
  if (trimmed.empty() || trimmed == "null" || trimmed == "none" || trimmed == "auto") {
    return std::nullopt;
  }
  return common::ParseBool(trimmed);
}

std::vector<int32_t> ParseInt32List(std::string_view text) {
  std::vector<int32_t> result;
  std::string_view rest = Trim(text);
  // Accept a bracketed list so a value copied out of a JSON report can be pasted back in.
  if (rest.size() >= 2 && rest.front() == '[' && rest.back() == ']') {
    rest = rest.substr(1, rest.size() - 2);
  }
  rest = Trim(rest);
  if (rest.empty()) {
    return result;
  }
  while (true) {
    const size_t comma = rest.find(',');
    const std::string_view item = Trim(rest.substr(0, comma));
    if (item.empty()) {
      throw std::runtime_error("expected comma-separated int32 list, got: " + std::string(text));
    }
    result.push_back(common::ParseInt32(item));
    if (comma == std::string_view::npos) {
      break;
    }
    rest = rest.substr(comma + 1);
  }
  return result;
}

std::string FormatFloat(float value) {
  // %.9g round-trips every float, so a dumped value reloads to the same bits.
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.9g", static_cast<double>(value));
  return std::string(buffer);
}

std::string FormatOptionalBool(const std::optional<bool>& value) {
  if (!value.has_value()) {
    return "null";
  }
  return *value ? "true" : "false";
}

std::string FormatInt32List(const std::vector<int32_t>& value) {
  std::string result;
  for (const int32_t item : value) {
    if (!result.empty()) {
      result += ',';
    }
    result += std::to_string(item);
  }
  return result;
}

void ThrowUnknownEnumValue(std::string_view text, const std::string& allowed) {
  throw std::runtime_error("expected one of {" + allowed + "}, got: " + std::string(text));
}

}  // namespace cuvslam::params::detail
