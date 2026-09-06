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

#include "params/registry.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "common/include_json.h"

namespace cuvslam::params {

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

/// True if @p key names the field `<prefix>.<name>`.
bool KeyMatches(std::string_view key, std::string_view prefix, std::string_view name) {
  if (key.size() != prefix.size() + 1 + name.size()) {
    return false;
  }
  return key.substr(0, prefix.size()) == prefix && key[prefix.size()] == '.' && key.substr(prefix.size() + 1) == name;
}

std::string FormatDouble(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return std::string(buffer);
}

/// Human-readable form of a constraint, used in rejection messages and in help output.
std::string DescribeBounds(const Bounds& bounds) {
  const bool unbounded_above = bounds.max >= std::numeric_limits<double>::max();
  if (bounds.min == 0.0 && unbounded_above) {
    return "must be non-negative";
  }
  if (unbounded_above) {
    return "must be >= " + FormatDouble(bounds.min);
  }
  return "must be in [" + FormatDouble(bounds.min) + ", " + FormatDouble(bounds.max) + "]";
}

/// True if @p key is `<suffix>` or ends in `.<suffix>`, the abbreviation accepted on command lines.
bool IsSuffixOf(std::string_view key, std::string_view suffix) {
  if (key == suffix) {
    return true;
  }
  return key.size() > suffix.size() + 1 && key.substr(key.size() - suffix.size()) == suffix &&
         key[key.size() - suffix.size() - 1] == '.';
}

}  // namespace

std::string_view ToString(Source source) {
  switch (source) {
    case Source::Default:
      return "default";
    case Source::File:
      return "file";
    case Source::CommandLine:
      return "command-line";
    case Source::Api:
      return "api";
  }
  return "unknown";
}

std::pair<const Registry::Group*, const FieldDesc*> Registry::Find(std::string_view key) const {
  for (const Group& group : groups_) {
    for (size_t i = 0; i < group.num_fields; ++i) {
      if (KeyMatches(key, group.prefix, group.fields[i].name)) {
        return {&group, &group.fields[i]};
      }
    }
  }
  return {nullptr, nullptr};
}

std::string Registry::Resolve(std::string_view key) const {
  if (Find(key).second != nullptr) {
    return std::string(key);
  }

  std::vector<std::string> matches;
  for (const Group& group : groups_) {
    for (size_t i = 0; i < group.num_fields; ++i) {
      std::string full = std::string(group.prefix) + '.' + std::string(group.fields[i].name);
      if (IsSuffixOf(full, key)) {
        matches.push_back(std::move(full));
      }
    }
  }

  if (matches.size() == 1) {
    return matches.front();
  }
  if (matches.empty()) {
    throw std::invalid_argument("unknown parameter '" + std::string(key) + "'");
  }

  std::string candidates;
  for (const std::string& match : matches) {
    if (!candidates.empty()) {
      candidates += ", ";
    }
    candidates += match;
  }
  throw std::invalid_argument("parameter '" + std::string(key) + "' is ambiguous; candidates: " + candidates);
}

void Registry::Set(std::string_view key, std::string_view value, Source source) {
  const std::string resolved = Resolve(key);
  const auto [group, field] = Find(resolved);

  // Assignment is all-or-nothing: a rejected value leaves the field exactly as it was, so a bad
  // line in a parameter file cannot half-apply a configuration.
  const std::string previous = field->get(group->instance);
  try {
    field->set(group->instance, value);
  } catch (const std::exception& e) {
    throw std::runtime_error("parameter '" + resolved + "': " + e.what());
  }

  if (field->bounds.active && field->as_double != nullptr) {
    const double numeric = field->as_double(group->instance);
    if (numeric < field->bounds.min || numeric > field->bounds.max) {
      field->set(group->instance, previous);
      throw std::runtime_error("parameter '" + resolved + "': " + std::string(value) + " " +
                               DescribeBounds(field->bounds));
    }
  }

  const auto it = std::find_if(sources_.begin(), sources_.end(),
                               [&resolved](const auto& entry) { return entry.first == resolved; });
  if (it != sources_.end()) {
    it->second = source;
  } else {
    sources_.emplace_back(resolved, source);
  }
}

std::string Registry::Get(std::string_view key) const {
  const std::string resolved = Resolve(key);
  const auto [group, field] = Find(resolved);
  return field->get(group->instance);
}

Source Registry::SourceOf(const std::string& key) const {
  const auto it =
      std::find_if(sources_.begin(), sources_.end(), [&key](const auto& entry) { return entry.first == key; });
  return it != sources_.end() ? it->second : Source::Default;
}

size_t Registry::SetFromFile(const std::string& path, Source source) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open parameter file '" + path + "'");
  }

  size_t assigned = 0;
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
    const std::string_view key = Trim(content.substr(0, separator));
    const std::string_view value = Trim(content.substr(separator + 1));
    try {
      Set(key, value, source);
    } catch (const std::exception& e) {
      throw std::runtime_error(path + ":" + std::to_string(line_number) + ": " + e.what());
    }
    ++assigned;
  }
  return assigned;
}

std::vector<ParamInfo> Registry::List() const {
  std::vector<ParamInfo> result;
  for (const Group& group : groups_) {
    for (size_t i = 0; i < group.num_fields; ++i) {
      const FieldDesc& field = group.fields[i];
      ParamInfo info;
      info.key = std::string(group.prefix) + '.' + std::string(field.name);
      info.doc = field.doc;
      info.type = field.type_name();
      info.value = field.get(group.instance);
      info.default_value = field.get(group.defaults);
      info.source = SourceOf(info.key);
      result.push_back(std::move(info));
    }
  }
  return result;
}

std::string Registry::ToJson() const {
  Json::Value root(Json::objectValue);
  for (const ParamInfo& info : List()) {
    // Values stay strings so the dump round-trips through SetFromFile without a type table.
    Json::Value entry(Json::objectValue);
    entry["value"] = info.value;
    entry["type"] = info.type;
    entry["source"] = std::string(ToString(info.source));
    if (!info.is_default()) {
      entry["default"] = info.default_value;
    }
    root[info.key] = entry;
  }
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "  ";
  return Json::writeString(builder, root);
}

}  // namespace cuvslam::params
