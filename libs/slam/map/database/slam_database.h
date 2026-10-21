
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

#include "slam/common/blob.h"
#include "slam/common/slam_common.h"

namespace cuvslam::slam {

enum class SlamDatabaseTable : uint32_t { Singletons = 0, Landmarks = 1, Descriptors = 2, SpatialCells = 3, Max = 7 };
enum class SlamDatabaseSingleton {
  SpatialIndex = 1,
  PoseGraph = 2,
  PoseGraphHypothesis = 3,
  VprMap = 4,
};
// Interface for slam database
//
// Entries:
// PoseGraph nodes and edges
// Landmarks
// Spatial index as blob
// VPR map as blob
class ISlamDatabase {
public:
  // key of tables
  using Key = uint64_t;

  // move to implementation
  // bool Open(const char* url);

  virtual ~ISlamDatabase() = default;

  // flush on the disk
  virtual bool Flush() = 0;

  // is readonly
  virtual bool IsReadOnly() const = 0;

  // Copies the database into the folder `path`, which it replaces; refuses the folders that hold this database.
  virtual bool CopyTo(const std::string& path) const = 0;

  // True when this is the database in the folder `path`.
  virtual bool IsAt(const std::string& path) const = 0;

  // statistic string
  virtual std::string Statistic() const = 0;

  // get count of recorrd in the table
  virtual size_t GetRecordsCount(SlamDatabaseTable table) const = 0;

  template <class FUNC>
  bool SetSingleton(SlamDatabaseSingleton id, size_t bytes, const FUNC& func) {
    std::function<bool(BlobWriter&)> std_func(func);
    return SetRecordV(SlamDatabaseTable::Singletons, static_cast<Key>(id), bytes, std_func);
  }

  template <class FUNC>
  bool GetSingleton(SlamDatabaseSingleton id, const FUNC& func) const {
    std::function<bool(const BlobReader&)> std_func(func);
    return GetRecordV(SlamDatabaseTable::Singletons, static_cast<Key>(id), std_func);
  }
  // Don't show error message if the singleton was not found
  template <class FUNC>
  bool TryGetSingleton(SlamDatabaseSingleton id, const FUNC& func) const {
    std::function<bool(const BlobReader&)> std_func(func);
    return GetRecordV(SlamDatabaseTable::Singletons, static_cast<Key>(id), std_func, true);
  }

  template <class FUNC>
  bool SetRecord(SlamDatabaseTable table, Key id, size_t bytes, const FUNC& func) {
    std::function<bool(BlobWriter&)> std_func(func);
    return SetRecordV(table, id, bytes, std_func);
  }

  template <class FUNC>
  bool GetRecord(SlamDatabaseTable table, Key id, const FUNC& func) const {
    std::function<bool(const BlobReader&)> std_func(func);
    return GetRecordV(table, id, std_func);
  }
  // Don't show error message if record was not found
  template <class FUNC>
  bool TryGetRecord(SlamDatabaseTable table, Key id, const FUNC& func) const {
    std::function<bool(const BlobReader&)> std_func(func);
    return GetRecordV(table, id, std_func, true);
  }

  bool DelRecord(SlamDatabaseTable table, Key id) { return DelRecordV(table, id); }

  bool TryDelRecord(SlamDatabaseTable table, Key id) { return DelRecordV(table, id, true); }

public:
  // Iterate all records in table
  // if func return true -> continue else stop processing
  // if database error occurred -> return false
  template <class FUNC>
  bool ForEach(SlamDatabaseTable table, const FUNC& func) const {
    const std::function<bool(Key id, const BlobReader&)> std_func(func);
    return ForEachV(table, std_func);
  }

protected:
  virtual bool SetRecordV(SlamDatabaseTable table, Key id, size_t bytes,
                          const std::function<bool(BlobWriter& dst)>& func) = 0;
  virtual bool GetRecordV(SlamDatabaseTable table, Key id, const std::function<bool(const BlobReader& src)>& func,
                          bool silence_if_not_found = false) const = 0;
  virtual bool DelRecordV(SlamDatabaseTable table, Key id, bool silence_if_not_found = false) = 0;

  // Iterate all records in table
  // if func return true -> continue else stop processing
  // if database error occurred -> return false
  virtual bool ForEachV(SlamDatabaseTable table,
                        const std::function<bool(Key id, const BlobReader& src)>& func) const = 0;
};

}  // namespace cuvslam::slam
