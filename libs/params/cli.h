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

#include <string>
#include <vector>

#include "params/registry.h"

/**
 * @file cli.h
 *
 * Command-line plumbing for parameters, shared by the tools.
 *
 * Every tool wants the same two things -- collect `-Pkey=value` arguments, then apply them to a
 * registry -- so they live here rather than being copied into each `main()`.
 */

namespace cuvslam::params {

/**
 * @brief Removes `-Pkey=value` arguments from argv and returns them in order.
 *
 * Call before the argv parser runs. gflags has no clustered short options and would reject `-P`
 * outright, and extracting these first also makes precedence explicit: a parameter file is read
 * first and these are applied over it.
 *
 * @param[in,out] argc argument count, reduced by the number of arguments removed
 * @param[in,out] argv argument vector, compacted in place; argv[0] is never examined
 * @return the `key=value` bodies, with the `-P` stripped, in the order given
 */
std::vector<std::string> ExtractOverrides(int& argc, char** argv);

/**
 * @brief Applies `key=value` strings to @p registry.
 *
 * @throws std::invalid_argument if an entry has no `=`, or names an unknown or ambiguous
 *         parameter.
 * @throws std::runtime_error if a value does not parse or falls outside its range.
 */
void ApplyOverrides(Registry& registry, const std::vector<std::string>& overrides, Source source);

/// Prints every parameter of @p registry with its type, default and description, one per line.
void PrintReference(const Registry& registry, std::ostream& out);

}  // namespace cuvslam::params
