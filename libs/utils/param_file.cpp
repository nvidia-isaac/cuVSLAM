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

#include "utils/param_file.h"

#include <fstream>
#include <stdexcept>
#include <string_view>

namespace cuvslam::utils {

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

std::vector<ParamEntry> ReadParamFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open parameter file '" + path + "'");
  }

  std::vector<ParamEntry> entries;
  std::string line;
  for (size_t line_number = 1; std::getline(file, line); ++line_number) {
    std::string_view content = line;
    const size_t comment = content.find('#');
    if (comment != std::string_view::npos) {
      content = content.substr(0, comment);
    }
    content = Trim(content);
    if (content.empty()) {
      continue;
    }
    const size_t separator = content.find_first_of(":=");
    if (separator == std::string_view::npos) {
      throw std::runtime_error(path + ":" + std::to_string(line_number) + ": expected 'key: value', got '" +
                               std::string(content) + "'");
    }
    entries.push_back(ParamEntry{std::string(Trim(content.substr(0, separator))),
                                 std::string(Trim(content.substr(separator + 1))), line_number});
  }
  return entries;
}

}  // namespace cuvslam::utils
