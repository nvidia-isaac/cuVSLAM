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

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "common/parse_utils.h"

/**
 * @file params.h
 *
 * String-addressable description of a settings struct.
 *
 * A settings struct keeps its plain fields and documented defaults. Alongside it, a
 * specialization of Fields<T> lists one FieldDesc per tunable field. Every generic operation
 * on parameters -- set by name, read by name, load a file, dump a report, expose to Python,
 * accept a command-line override -- is then a loop over that list, so none of them needs
 * per-parameter code.
 *
 * Adding a parameter means adding the field and one line to the Fields<T> list. Nothing else
 * in the codebase has to learn its name.
 */

namespace cuvslam::params {

/// Maps an enum value to the string used in configuration files and on the command line.
template <typename E>
struct EnumEntry {
  std::string_view name;
  E value;
};

/**
 * @brief Names for an enum used as a parameter value.
 *
 * Specialize next to the enum itself, so the enum stays the single place that knows its own
 * spellings:
 * @code
 * template <>
 * struct params::EnumNames<TrackerType> {
 *   static constexpr std::array<params::EnumEntry<TrackerType>, 2> kList = {{
 *       {"lk", TrackerType::LK},
 *       {"klt", TrackerType::KLT},
 *   }};
 * };
 * @endcode
 */
template <typename E>
struct EnumNames;

namespace detail {

// Value <-> string conversion for every supported field type. Parse throws on malformed input;
// callers convert that into an error naming the offending key.
std::optional<bool> ParseOptionalBool(std::string_view text);
std::vector<int32_t> ParseInt32List(std::string_view text);
std::string FormatFloat(float value);
std::string FormatOptionalBool(const std::optional<bool>& value);
std::string FormatInt32List(const std::vector<int32_t>& value);
[[noreturn]] void ThrowUnknownEnumValue(std::string_view text, const std::string& allowed);

inline std::string FormatValue(bool value) { return value ? "true" : "false"; }
inline std::string FormatValue(int32_t value) { return std::to_string(value); }
inline std::string FormatValue(int64_t value) { return std::to_string(value); }
inline std::string FormatValue(uint32_t value) { return std::to_string(value); }
inline std::string FormatValue(uint64_t value) { return std::to_string(value); }
inline std::string FormatValue(float value) { return FormatFloat(value); }
inline std::string FormatValue(const std::optional<bool>& value) { return FormatOptionalBool(value); }
inline std::string FormatValue(const std::vector<int32_t>& value) { return FormatInt32List(value); }
inline std::string FormatValue(std::string_view value) { return std::string(value); }

template <typename E>
std::string EnumValueList() {
  std::string result;
  for (const EnumEntry<E>& entry : EnumNames<E>::kList) {
    if (!result.empty()) {
      result += '|';
    }
    result.append(entry.name);
  }
  return result;
}

// Parses the string form of a parameter into its field type. Selected on the field type alone,
// so there is no dummy argument to carry the type: the caller writes ParseValue<T>(text).
template <typename T>
T ParseValue(std::string_view text) {
  if constexpr (std::is_enum_v<T>) {
    for (const EnumEntry<T>& entry : EnumNames<T>::kList) {
      if (entry.name == text) {
        return entry.value;
      }
    }
    ThrowUnknownEnumValue(text, EnumValueList<T>());
  } else if constexpr (std::is_same_v<T, bool>) {
    return common::ParseBool(text);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return common::ParseInt32(text);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return common::ParseInt64(text);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return common::ParseUInt32(text);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return common::ParseUInt64(text);
  } else if constexpr (std::is_same_v<T, float>) {
    return common::ParseFloat(text);
  } else if constexpr (std::is_same_v<T, std::optional<bool>>) {
    return ParseOptionalBool(text);
  } else if constexpr (std::is_same_v<T, std::string_view>) {
    // Kept as-is. Registry::Set only ever passes a view into storage it owns, so the field
    // outlives the caller's string -- which is what lets a struct holding string_view rather than
    // std::string have string parameters at all.
    return text;
  } else {
    static_assert(std::is_same_v<T, std::vector<int32_t>>, "unsupported parameter field type");
    return ParseInt32List(text);
  }
}

template <typename E, typename = std::enable_if_t<std::is_enum_v<E>>>
std::string FormatValue(E value) {
  for (const EnumEntry<E>& entry : EnumNames<E>::kList) {
    if (entry.value == value) {
      return std::string(entry.name);
    }
  }
  return std::to_string(static_cast<int64_t>(value));
}

// Type name shown by --list-params and in dumped reports.
template <typename T>
std::string TypeName() {
  if constexpr (std::is_enum_v<T>) {
    return "enum{" + EnumValueList<T>() + "}";
  } else if constexpr (std::is_same_v<T, bool>) {
    return "bool";
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return "int32";
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return "int64";
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return "uint32";
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return "uint64";
  } else if constexpr (std::is_same_v<T, float>) {
    return "float";
  } else if constexpr (std::is_same_v<T, std::optional<bool>>) {
    return "bool|null";
  } else if constexpr (std::is_same_v<T, std::string_view>) {
    return "string";
  } else {
    static_assert(std::is_same_v<T, std::vector<int32_t>>, "unsupported parameter field type");
    return "int32[]";
  }
}

}  // namespace detail

/// Inclusive numeric bounds a field's value must satisfy. Unset for non-numeric fields.
struct Bounds {
  double min = 0.0;
  double max = 0.0;
  bool active = false;
};

/// Rejects negative values, the constraint most integer counts and sizes need.
constexpr Bounds NonNegative() { return Bounds{0.0, std::numeric_limits<double>::max(), true}; }

/// Rejects values outside [min, max].
constexpr Bounds InRange(double min, double max) { return Bounds{min, max, true}; }

/**
 * @brief One tunable field, addressable by name.
 *
 * Holds no data of its own: set/get take the address of the owning struct, so a single
 * constexpr descriptor list serves every instance of that struct.
 */
struct FieldDesc {
  std::string_view name;
  std::string_view doc;
  void (*set)(void* owner, std::string_view value);
  std::string (*get)(const void* owner);
  std::string (*type_name)();
  /// Reads the field as a double for bounds checking. Null when the field is not numeric.
  double (*as_double)(const void* owner);
  Bounds bounds;
};

namespace detail {

template <typename Owner, auto Member>
using MemberType = std::decay_t<decltype(std::declval<Owner&>().*Member)>;

template <typename Owner, auto Member>
void SetMember(void* owner, std::string_view value) {
  using Value = MemberType<Owner, Member>;
  static_cast<Owner*>(owner)->*Member = ParseValue<Value>(value);
}

template <typename Owner, auto Member>
std::string GetMember(const void* owner) {
  return FormatValue(static_cast<const Owner*>(owner)->*Member);
}

template <typename T>
constexpr bool kIsNumeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template <typename Owner, auto Member>
double MemberAsDouble(const void* owner) {
  return static_cast<double>(static_cast<const Owner*>(owner)->*Member);
}

/// Numeric fields get a double reader so bounds can be checked generically; others get nullptr.
template <typename Owner, auto Member>
constexpr double (*AsDoubleOrNull())(const void*) {
  if constexpr (kIsNumeric<MemberType<Owner, Member>>) {
    return &MemberAsDouble<Owner, Member>;
  } else {
    return nullptr;
  }
}

}  // namespace detail

/**
 * @brief Describes a field of Owner.
 *
 * @tparam Owner  struct that owns the field
 * @tparam Member pointer to the field, for example &sof::Settings::num_desired_tracks
 * @param  bounds optional inclusive range; a value outside it is rejected and the field keeps
 *                its previous value. Only meaningful for numeric fields.
 */
template <typename Owner, auto Member>
constexpr FieldDesc Field(std::string_view name, std::string_view doc, Bounds bounds = Bounds{}) {
  static_assert(!std::is_same_v<decltype(Member), std::nullptr_t>, "Member must be a pointer to a data member");
  return FieldDesc{name,
                   doc,
                   &detail::SetMember<Owner, Member>,
                   &detail::GetMember<Owner, Member>,
                   &detail::TypeName<detail::MemberType<Owner, Member>>,
                   detail::AsDoubleOrNull<Owner, Member>(),
                   bounds};
}

/**
 * @brief The tunable fields of a settings struct.
 *
 * Declare with CUVSLAM_PARAMS_BEGIN/END next to the struct it describes. Fields left out of the
 * list are not reachable by name -- that is the intended way to keep structural settings (camera
 * topology, buffer sizes) off the tuning surface.
 *
 * A struct nested inside another settings struct gets its own list and is registered under its
 * own prefix, so `Registry::Add("sof.feature_selection", settings.feature_selection_settings)`
 * yields keys like `sof.feature_selection.survivor_from_last`.
 */
template <typename T>
struct Fields;

}  // namespace cuvslam::params

/**
 * @brief Declares the tunable fields of a settings struct.
 *
 * Place immediately after the struct, at file scope, outside every namespace. @p type must be
 * fully qualified. Between BEGIN and END, list one CUVSLAM_PARAM per tunable field; the field name
 * is stringized from the member, so the key and the member it writes cannot disagree.
 *
 * @code
 * }  // namespace cuvslam::sof
 *
 * CUVSLAM_PARAMS_BEGIN(cuvslam::sof::Settings)
 * CUVSLAM_PARAM(box3_prefilter, "Preprocess input images with a box filter")
 * CUVSLAM_PARAM_BOUNDED(num_desired_tracks, "Number of feature tracks to maintain", NonNegative())
 * CUVSLAM_PARAMS_END()
 * @endcode
 */
// clang-format off
#define CUVSLAM_PARAMS_BEGIN(type)                \
  namespace cuvslam::params {                     \
  template <>                                     \
  struct Fields<type> {                           \
    using Owner = type;                           \
    static constexpr auto kList = std::array {

/// One tunable field. Expands with a trailing comma so entries need no separators.
#define CUVSLAM_PARAM(field, doc) \
  ::cuvslam::params::Field<Owner, &Owner::field>(#field, doc),

/// A tunable field constrained to @p bounds; a value outside it is rejected and rolled back.
#define CUVSLAM_PARAM_BOUNDED(field, doc, bounds) \
  ::cuvslam::params::Field<Owner, &Owner::field>(#field, doc, bounds),

/// Closes CUVSLAM_PARAMS_BEGIN. The array size is deduced, so there is no count to keep in sync.
#define CUVSLAM_PARAMS_END() \
    };                       \
  };                         \
  }
// clang-format on

/**
 * @brief Declares the spellings accepted for an enum used as a parameter value.
 *
 * Place immediately after the enum, at file scope, outside every namespace. Spellings are given
 * explicitly because they routinely differ from the enumerator (`lk_horizontal` for
 * `LKHorizontal`). Leaving an enumerator out makes it unreachable by name, which is how modes
 * that need more than a scalar are kept out of the tuning surface.
 *
 * @code
 * CUVSLAM_PARAM_ENUM_BEGIN(cuvslam::sof::TrackerType)
 * CUVSLAM_PARAM_ENUM_VALUE("lk", LK)
 * CUVSLAM_PARAM_ENUM_VALUE("lk_horizontal", LKHorizontal)
 * CUVSLAM_PARAM_ENUM_END()
 * @endcode
 */
// clang-format off
#define CUVSLAM_PARAM_ENUM_BEGIN(type)            \
  namespace cuvslam::params {                     \
  template <>                                     \
  struct EnumNames<type> {                         \
    using Enum = type;                            \
    static constexpr auto kList = std::array {

/// One accepted spelling for the enum opened by CUVSLAM_PARAM_ENUM_BEGIN.
#define CUVSLAM_PARAM_ENUM_VALUE(name, enumerator) \
  ::cuvslam::params::EnumEntry<Enum>{name, Enum::enumerator},

/// Closes CUVSLAM_PARAM_ENUM_BEGIN.
#define CUVSLAM_PARAM_ENUM_END() CUVSLAM_PARAMS_END()
// clang-format on
