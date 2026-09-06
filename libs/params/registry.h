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

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "params/params.h"

namespace cuvslam::params {

/// Where a parameter's current value came from. Recorded so a run can be reproduced from its report.
enum class Source : uint8_t {
  Default,      ///< built-in default; never assigned
  File,         ///< loaded from a parameter file
  CommandLine,  ///< set by a command-line override
  Api,          ///< set programmatically
};

std::string_view ToString(Source source);

/// A parameter's identity, current state, and origin. Feeds --list-params, reports, and Python.
struct ParamInfo {
  std::string key;
  std::string_view doc;
  std::string type;
  std::string value;
  std::string default_value;
  Source source = Source::Default;

  bool is_default() const { return value == default_value; }
};

/**
 * @brief String-addressable view over a set of live settings structs.
 *
 * Registered structs keep their identity and their types: the registry writes through to the
 * caller's instances, so the hot path keeps reading plain typed fields and pays nothing for the
 * string interface.
 *
 * Keys are `<prefix>.<field>`, for example `sof.num_desired_tracks`. Every mutation records its
 * Source, which is what makes a run's configuration reconstructible after the fact.
 *
 * Not thread-safe: configure before tracking starts, or serialize access externally.
 */
class Registry {
public:
  /**
   * @brief Registers a settings struct under a key prefix.
   *
   * @p instance must outlive the registry. Requires a params::Fields<T> specialization.
   */
  template <typename T>
  void Add(std::string_view prefix, T& instance) {
    // One default-constructed instance per settings type, and the only place defaults are read
    // from, so a default is never written down a second time just to be reported.
    static const T kDefaults{};
    groups_.push_back(Group{prefix, &instance, &kDefaults, Fields<T>::kList.data(), Fields<T>::kList.size()});
  }

  /**
   * @brief Resolves a key, accepting any unambiguous suffix of a full key.
   *
   * `num_desired_tracks` resolves to `sof.num_desired_tracks` when no other parameter ends the
   * same way. Keeps command lines short without giving up namespacing.
   *
   * @throws std::invalid_argument if nothing matches, or if a suffix is ambiguous; the message
   *         lists the candidates.
   */
  std::string Resolve(std::string_view key) const;

  /**
   * @brief Assigns a parameter from its string form.
   *
   * @throws std::invalid_argument if the key is unknown or ambiguous.
   * @throws std::runtime_error if the value does not parse as the field's type.
   */
  void Set(std::string_view key, std::string_view value, Source source);

  /// Current value of a parameter in string form.
  std::string Get(std::string_view key) const;

  /**
   * @brief Applies a parameter file: `key: value` or `key = value` per line, `#` starts a comment.
   *
   * @return number of parameters assigned.
   * @throws std::runtime_error if the file cannot be read or a line is malformed, naming the file
   *         and the line number.
   */
  size_t SetFromFile(const std::string& path, Source source);

  /// Every registered parameter, in registration order. Feeds help output and reports.
  std::vector<ParamInfo> List() const;

  /// JSON object of resolved values and their origins, for embedding in a run report.
  std::string ToJson() const;

private:
  struct Group {
    std::string_view prefix;
    void* instance;
    const void* defaults;
    const FieldDesc* fields;
    size_t num_fields;
  };

  /// Locates the group and field a full key names, or nullptr for both when the key is absent.
  std::pair<const Group*, const FieldDesc*> Find(std::string_view key) const;

  Source SourceOf(const std::string& key) const;

  std::vector<Group> groups_;
  std::vector<std::pair<std::string, Source>> sources_;
};

}  // namespace cuvslam::params
