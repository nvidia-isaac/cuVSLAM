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

#include "params/cli.h"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string_view>

namespace cuvslam::params {

std::vector<std::string> ExtractOverrides(int& argc, char** argv) {
  std::vector<std::string> overrides;
  int kept = 0;
  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (i > 0 && arg.size() > 2 && arg.substr(0, 2) == "-P") {
      overrides.emplace_back(arg.substr(2));
    } else {
      argv[kept++] = argv[i];
    }
  }
  argc = kept;
  return overrides;
}

void ApplyOverrides(Registry& registry, const std::vector<std::string>& overrides, Source source) {
  for (const std::string& entry : overrides) {
    const size_t separator = entry.find('=');
    if (separator == std::string::npos) {
      throw std::invalid_argument("-P expects key=value, got '" + entry + "'");
    }
    registry.Set(entry.substr(0, separator), entry.substr(separator + 1), source);
  }
}

void PrintReference(const Registry& registry, std::ostream& out) {
  const std::vector<ParamInfo> infos = registry.List();
  size_t width = 0;
  for (const ParamInfo& info : infos) {
    width = std::max(width, info.key.size());
  }
  for (const ParamInfo& info : infos) {
    out << "  " << std::left << std::setw(static_cast<int>(width)) << info.key << "  " << info.type << " = "
        << info.default_value << "\n      " << info.doc << "\n";
  }
}

}  // namespace cuvslam::params
