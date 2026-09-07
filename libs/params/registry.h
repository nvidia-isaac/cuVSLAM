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
#include <deque>
#include <memory>
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
  Registry() = default;
  Registry(Registry&&) = default;
  Registry& operator=(Registry&&) = default;
  // Copying would give two registries write access to one set of settings. Spelled out rather
  // than left to the move-only member, so the error names this instead of a container internal.
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;

  /**
   * @brief Registers a settings struct under a key prefix.
   *
   * @p instance must outlive the registry, and must not be relocated while it is registered: the
   * registry holds a reference to it, not a copy, which is what lets the solvers keep reading
   * plain typed fields. Requires a params::Fields<T> specialization.
   */
  template <typename T>
  void Add(std::string_view prefix, T& instance) {
    groups_.push_back(std::make_unique<TypedGroup<T>>(prefix, instance));
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

  /// One entry of a parameter file, with the line it came from so errors can point at it.
  struct FileEntry {
    std::string key;
    std::string value;
    size_t line = 0;
  };

  /**
   * @brief Reads a parameter file without applying anything.
   *
   * For callers that split one file across several registries -- a tool typically configures
   * Odometry::Config before construction and the solver parameters after -- so the file format
   * stays defined in one place.
   *
   * @return the entries in file order.
   * @throws std::runtime_error if the file cannot be read or a line is malformed.
   */
  static std::vector<FileEntry> ReadFile(const std::string& path);

  /// True if @p key names a parameter of this registry, exactly or by unambiguous suffix.
  bool Knows(std::string_view key) const;

  /// Every registered parameter, in registration order. Feeds help output and reports.
  std::vector<ParamInfo> List() const;

private:
  /// A field's name and constraint, without binding it to any instance.
  struct FieldView {
    std::string_view name;
    std::string_view doc;
    Bounds bounds;
  };

  /**
   * @brief Type-erased access to one registered settings struct.
   *
   * The registry holds structs of many unrelated types in one container, so something has to be
   * erased. Erasing here rather than in FieldDesc keeps the descriptors and their accessors fully
   * typed, and lets the concrete group hold a `T&` -- so the requirement that a registered struct
   * outlive the registry is expressed in the type system rather than in a comment.
   */
  class Group {
  public:
    explicit Group(std::string_view prefix) : prefix_(prefix) {}
    virtual ~Group() = default;
    Group(const Group&) = delete;
    Group& operator=(const Group&) = delete;

    std::string_view prefix() const { return prefix_; }

    virtual size_t NumFields() const = 0;
    virtual FieldView Describe(size_t index) const = 0;
    virtual std::string Type(size_t index) const = 0;
    virtual void Set(size_t index, std::string_view value) = 0;
    virtual std::string Get(size_t index) const = 0;
    virtual std::string Default(size_t index) const = 0;
    /// True when the field's current value satisfies its declared bounds, or has none.
    virtual bool WithinBounds(size_t index) const = 0;

  private:
    std::string_view prefix_;
  };

  template <typename T>
  class TypedGroup final : public Group {
  public:
    TypedGroup(std::string_view prefix, T& instance) : Group(prefix), instance_(instance) {}

    size_t NumFields() const override { return Fields<T>::kList.size(); }

    FieldView Describe(size_t index) const override {
      const FieldDesc<T>& field = Fields<T>::kList[index];
      return FieldView{field.name, field.doc, field.bounds};
    }

    std::string Type(size_t index) const override { return Fields<T>::kList[index].type_name(); }

    void Set(size_t index, std::string_view value) override { Fields<T>::kList[index].set(&instance_, value); }

    std::string Get(size_t index) const override { return Fields<T>::kList[index].get(&instance_); }

    std::string Default(size_t index) const override {
      // One default-constructed instance per settings type, and the only place defaults are read
      // from, so a default is never written down a second time just to be reported.
      static const T kDefaults{};
      return Fields<T>::kList[index].get(&kDefaults);
    }

    bool WithinBounds(size_t index) const override {
      const FieldDesc<T>& field = Fields<T>::kList[index];
      if (!field.bounds.active || field.as_double == nullptr) {
        return true;
      }
      const double value = field.as_double(&instance_);
      return value >= field.bounds.min && value <= field.bounds.max;
    }

  private:
    T& instance_;
  };

  /// Locates the group and field index a full key names. The group is null when the key is absent.
  std::pair<Group*, size_t> Find(std::string_view key) const;

  Source SourceOf(const std::string& key) const;

  /// Copies @p value into storage owned by the registry and returns a view of it.
  ///
  /// String parameters land in `std::string_view` fields, which cannot own their text. Assigning
  /// a view of the caller's argument would dangle the moment it returns, so the registry keeps the
  /// text alive for as long as the settings it wrote to. A deque never relocates what it already
  /// holds, so previously handed-out views stay valid.
  std::string_view Intern(std::string_view value);

  // Move-only, because the groups are. Copying a registry would give two of them write access to
  // one set of settings, which is never what a caller means.
  std::vector<std::unique_ptr<Group>> groups_;
  std::vector<std::pair<std::string, Source>> sources_;
  std::deque<std::string> interned_;
};

}  // namespace cuvslam::params
