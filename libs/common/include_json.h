
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
#include <fstream>
#include <map>
#include <sstream>

#include "json/json.h"

#include "common/log.h"

namespace JsonUtils {

using JSonKeyMap = std::map<std::string, Json::Value&>;

// jsoncpp library produce exception. During parsing we need to parse a lot of user input.
// In case of mistake, it is important to provide details about what and where happened.
inline void THROW_IF_FALSE(const bool exp, const std::string& errStr = "") {
  if (!exp) {
    TraceError("%s", errStr.c_str());
    throw std::runtime_error(errStr);
  }
}

// fill values for all keys in keyMap.
inline void readKeyMap(const Json::Value& v, JSonKeyMap& keyMap) {
  for (const auto& e : keyMap) {
    const std::string& key = e.first;
    THROW_IF_FALSE(v.isMember(key), "Can't find \"" + key + "\" key");
    e.second = v[key];
  }
}

inline void readJson(const std::string& fileName, Json::Value& root) {
  std::ifstream jsonFile(fileName, std::ifstream::binary);

  THROW_IF_FALSE(jsonFile.is_open(), "Can't open " + fileName + " file");

  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  builder["strictRoot"] = true;
  builder["rejectDupKeys"] = true;

  std::string errs;
  const bool result = parseFromStream(builder, jsonFile, &root, &errs);
  THROW_IF_FALSE(result, "Syntax errors in " + fileName + " file: " + errs);
}

inline bool readJsonFromStringNoThrow(const std::string& s, Json::Value& root, std::string& errors) {
  std::stringstream jsonString(s);

  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  builder["strictRoot"] = true;
  builder["rejectDupKeys"] = true;

  return parseFromStream(builder, jsonString, &root, &errors);
}

inline void readJsonFromString(const std::string& s, Json::Value& root) {
  std::string errs;
  const bool result = readJsonFromStringNoThrow(s, root, errs);
  THROW_IF_FALSE(result, "Syntax errors while parsing JSON string: " + errs);
}

template <typename T>
inline Json::Value makeJson(T val) {
  return Json::Value(val);
}

template <>
inline Json::Value makeJson(std::uint64_t val) {
  return Json::Value(static_cast<Json::UInt64>(val));
}

template <>
inline Json::Value makeJson(std::int64_t val) {
  return Json::Value(static_cast<Json::Int64>(val));
}

}  // namespace JsonUtils
