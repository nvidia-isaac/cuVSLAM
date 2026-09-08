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
#include <string>
#include <vector>

/**
 * @file param_file.h
 *
 * The `--params` file format the tools share.
 *
 * Deliberately not in libs/params: that library ships, and defining a file format there would make
 * one format part of the public API for the benefit of a couple of tools. Here it is shared by the
 * tools that want it and imposed on nobody -- a library caller passes names and values directly,
 * and a Python caller can use a real YAML parser.
 *
 * A file in this format is also a valid YAML mapping of scalars, so external tooling can read it
 * without knowing anything about cuVSLAM.
 */

namespace cuvslam::utils {

/// One entry of a parameter file, with the line it came from so errors can point at it.
struct ParamEntry {
  std::string key;
  std::string value;
  size_t line = 0;  ///< 0 when the entry did not come from a file, e.g. a -P override
};

/**
 * @brief Reads a parameter file: `key: value` or `key = value` per line, `#` starts a comment.
 *
 * Entries are returned rather than applied, because a caller typically splits them across several
 * registries -- configuration has to be final before a tracker is built, while solver parameters
 * only exist once it is.
 *
 * Blank lines and comment-only lines are skipped. Keys and values are trimmed, so alignment and
 * trailing comments are free.
 *
 * @param[in] path file to read
 * @return the entries in file order
 * @throws std::runtime_error if the file cannot be opened, or a line has no `:` or `=`; the
 *         message carries the file name and the line number
 */
std::vector<ParamEntry> ReadParamFile(const std::string& path);

}  // namespace cuvslam::utils
