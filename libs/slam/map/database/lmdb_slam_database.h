
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

#include "slam/map/database/slam_database.h"

#include "lmdb.h"

namespace cuvslam::slam {

// Interface for slam database
class LmdbSlamDatabase : public ISlamDatabase {
protected:
  MDB_env* env = nullptr;
  MDB_dbi dbi_main = 0;
  MDB_dbi dbi_landmarks = 0;
  MDB_dbi dbi_descriptors = 0;
  MDB_dbi dbi_spatial_cells = 0;

  MDB_dbi dbi_tables_[static_cast<uint32_t>(SlamDatabaseTable::Max)];

  mutable std::map<int, int> errors_;
  bool read_only_ = true;
  std::string path_;  // absolute folder of the open database

  class WriteTxn {
  public:
    WriteTxn(LmdbSlamDatabase* db, bool ignore_MDB_NOTFOUND = false);
    ~WriteTxn();
    bool Put(MDB_dbi dbi, const std::string& key_name, const Blob& blob);
    bool Put(MDB_dbi dbi, const size_t& key, const Blob& blob);
    bool Put(MDB_dbi dbi, MDB_val* key, MDB_val* data);
    bool Del(MDB_dbi dbi, const size_t& key);
    bool Del(MDB_dbi dbi, MDB_val* key);
    bool Drop(MDB_dbi dbi, int del);
    bool Commit();
    void Abort();

  private:
    bool Begin(MDB_dbi dbi);

    LmdbSlamDatabase* db_ = nullptr;
    MDB_txn* txn_ = nullptr;
    bool ignore_MDB_NOTFOUND_ = false;
    bool failed_ = false;  // an operation failed, so Commit() aborts
  };

  class Cursor {
  public:
    ~Cursor();
    void Setup(MDB_cursor* cursor);
    bool Get(size_t& key, Blob& blob);
    bool Get(size_t& key, MDB_val& data);
    bool Get(MDB_val* key, MDB_val* data);
    void Close();

  protected:
    MDB_cursor* cursor_ = nullptr;
  };

  class ReadTxn {
  public:
    explicit ReadTxn(const LmdbSlamDatabase* db, bool ignore_MDB_NOTFOUND = false);
    ~ReadTxn();
    bool Get(MDB_dbi dbi, const std::string& key_name, Blob& blob);
    bool Get(MDB_dbi dbi, const size_t& key, Blob& blob);
    bool Get(MDB_dbi dbi, const size_t& key, MDB_val& data);
    bool Get(MDB_dbi dbi, MDB_val* key, MDB_val* data);
    bool OpenCursor(MDB_dbi dbi, Cursor& cursor);

  private:
    const LmdbSlamDatabase* db_ = nullptr;
    MDB_txn* txn_ = nullptr;
    bool ignore_MDB_NOTFOUND_ = false;
  };

  friend class WriteTxn;

public:
  LmdbSlamDatabase();
  ~LmdbSlamDatabase() override;

  enum class OpenMode {
    READ_WRITE,             // open exists or make new for write
    READ_WRITE_HARD_RESET,  // make new for write, remove directory recursively
    READ_WRITE_NEW,         // make new for write, drop all records if exists.
    READ_ONLY_EXISTS,       // open exists for read only
    READ_WRITE_EXISTS       // open exists for write; never creates the folder or the database
  };
  bool Open(const char* db_path, OpenMode open_mode = OpenMode::READ_WRITE);
  void Close();

  bool Clear();

  std::string Statistic() const override;

  // get count of record in the table
  size_t GetRecordsCount(SlamDatabaseTable table) const override;

  bool IsReadOnly() const override;

  bool CopyTo(const std::string& path) const override;

  bool IsAt(const std::string& path) const override;

  bool SetRecordV(SlamDatabaseTable table, Key id, size_t bytes, const std::function<bool(BlobWriter&)>& func) override;
  bool GetRecordV(SlamDatabaseTable table, Key id, const std::function<bool(const BlobReader&)>& func,
                  bool silence_if_not_found) const override;
  bool DelRecordV(SlamDatabaseTable table, Key id, bool silence_if_not_found) override;

  // Iterate all records in table
  // if func return true -> continue else stop processing
  bool ForEachV(SlamDatabaseTable table, const std::function<bool(Key id, const BlobReader&)>& func) const override;
  bool Flush() override;

protected:
  bool Error(int rc, MDB_dbi dbi = 0) const;
};

}  // namespace cuvslam::slam
