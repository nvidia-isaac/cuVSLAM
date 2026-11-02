
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

#include "slam/map/database/lmdb_slam_database.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>

#include "slam/common/blob_eigen.h"

namespace cuvslam::slam {

// class WriteTxn
//
LmdbSlamDatabase::WriteTxn::WriteTxn(LmdbSlamDatabase* db, bool ignore_MDB_NOTFOUND) {
  this->db_ = db;
  this->txn_ = nullptr;
  this->ignore_MDB_NOTFOUND_ = ignore_MDB_NOTFOUND;
}
LmdbSlamDatabase::WriteTxn::~WriteTxn() {
  if (txn_) {
    mdb_txn_abort(txn_);
  }
}

bool LmdbSlamDatabase::WriteTxn::Put(MDB_dbi dbi, const size_t& id, const Blob& blob) {
  MDB_val key;
  MDB_val data;

  key.mv_size = sizeof(id);
  key.mv_data = (void*)&id;

  data.mv_size = blob.size();
  data.mv_data = (void*)blob.data();

  return Put(dbi, &key, &data);
}

bool LmdbSlamDatabase::WriteTxn::Put(MDB_dbi dbi, const std::string& key_name, const Blob& blob) {
  MDB_val key;
  key.mv_size = key_name.size();
  key.mv_data = (void*)key_name.c_str();

  MDB_val data;
  data.mv_size = blob.size();
  data.mv_data = (void*)blob.data();

  return Put(dbi, &key, &data);
}

bool LmdbSlamDatabase::WriteTxn::Begin(MDB_dbi dbi) {
  if (failed_) {
    return false;
  }
  if (!txn_) {
    const int rc = mdb_txn_begin(db_->env, NULL, 0, &txn_);
    if (db_->Error(rc, dbi)) {
      txn_ = nullptr;
      failed_ = true;
      return false;
    }
  }
  return true;
}

bool LmdbSlamDatabase::WriteTxn::Put(MDB_dbi dbi, MDB_val* key, MDB_val* data) {
  if (!Begin(dbi)) {
    return false;
  }
  const int rc = mdb_put(txn_, dbi, key, data, 0);
  if (db_->Error(rc, dbi)) {
    failed_ = true;
    return false;
  }
  return true;
}

bool LmdbSlamDatabase::WriteTxn::Del(MDB_dbi dbi, const size_t& id) {
  MDB_val key;
  key.mv_size = sizeof(id);
  key.mv_data = (void*)&id;

  return Del(dbi, &key);
}
bool LmdbSlamDatabase::WriteTxn::Del(MDB_dbi dbi, MDB_val* key) {
  if (!Begin(dbi)) {
    return false;
  }
  const int rc = mdb_del(txn_, dbi, key, nullptr);
  if (ignore_MDB_NOTFOUND_ && rc == MDB_NOTFOUND) {
    return false;  // don't show MDB_NOTFOUND error; the txn stays usable
  }
  if (db_->Error(rc, dbi)) {
    failed_ = true;
    return false;
  }
  return true;
}

bool LmdbSlamDatabase::WriteTxn::Drop(MDB_dbi dbi, int del) {
  if (!Begin(dbi)) {
    return false;
  }
  const int rc = mdb_drop(txn_, dbi, del);
  if (db_->Error(rc, dbi)) {
    failed_ = true;
    return false;
  }
  return true;
}

bool LmdbSlamDatabase::WriteTxn::Commit() {
  if (failed_) {
    Abort();
    return false;
  }
  if (!txn_) {
    return true;
  }
  const int rc = mdb_txn_commit(txn_);
  txn_ = nullptr;  // LMDB frees the txn whether the commit succeeds or not
  return !db_->Error(rc);
}
void LmdbSlamDatabase::WriteTxn::Abort() {
  if (txn_) {
    mdb_txn_abort(txn_);
    txn_ = nullptr;
  }
}

// ReadTxn
//
LmdbSlamDatabase::ReadTxn::ReadTxn(const LmdbSlamDatabase* db, bool ignore_MDB_NOTFOUND) {
  this->db_ = db;
  this->ignore_MDB_NOTFOUND_ = ignore_MDB_NOTFOUND;
}
LmdbSlamDatabase::ReadTxn::~ReadTxn() {
  if (txn_) {
    mdb_txn_abort(txn_);
  }
}
bool LmdbSlamDatabase::ReadTxn::Get(MDB_dbi dbi, const std::string& key_name, Blob& blob) {
  MDB_val key;
  key.mv_size = key_name.size();
  key.mv_data = (void*)key_name.c_str();

  MDB_val data;
  if (!Get(dbi, &key, &data)) {
    return false;
  }

  blob.assign((uint8_t*)data.mv_data, (uint8_t*)data.mv_data + data.mv_size);
  return true;
}

bool LmdbSlamDatabase::ReadTxn::Get(MDB_dbi dbi, const size_t& id, Blob& blob) {
  MDB_val key;
  key.mv_size = sizeof(id);
  key.mv_data = (void*)&id;

  MDB_val data;
  if (!Get(dbi, &key, &data)) {
    return false;
  }

  blob.assign((uint8_t*)data.mv_data, (uint8_t*)data.mv_data + data.mv_size);
  return true;
}
bool LmdbSlamDatabase::ReadTxn::Get(MDB_dbi dbi, const size_t& id, MDB_val& data) {
  MDB_val key;
  key.mv_size = sizeof(id);
  key.mv_data = (void*)&id;

  if (!Get(dbi, &key, &data)) {
    return false;
  }
  return true;
}

bool LmdbSlamDatabase::ReadTxn::Get(MDB_dbi dbi, MDB_val* key, MDB_val* data) {
  int rc;
  if (!txn_) {
    rc = mdb_txn_begin(db_->env, NULL, MDB_RDONLY, &txn_);
    if (db_->Error(rc, dbi)) {
      return false;
    }
  }

  rc = mdb_get(txn_, dbi, key, data);
  if (ignore_MDB_NOTFOUND_ && rc == MDB_NOTFOUND) {
    return false;  // don't show MDB_NOTFOUND error
  }
  if (db_->Error(rc, dbi)) {
    return false;
  }

  return true;
}
bool LmdbSlamDatabase::ReadTxn::OpenCursor(MDB_dbi dbi, Cursor& cursor) {
  int rc;
  if (!txn_) {
    rc = mdb_txn_begin(db_->env, NULL, MDB_RDONLY, &txn_);
    if (db_->Error(rc, dbi)) {
      return false;
    }
  }
  // dbi_posegraph_edges
  MDB_cursor* c;
  rc = mdb_cursor_open(txn_, dbi, &c);
  if (db_->Error(rc, dbi)) {
    return false;
  }

  cursor.Setup(c);
  return true;
}

// Cursor
//
void LmdbSlamDatabase::Cursor::Setup(MDB_cursor* cursor) {
  Close();
  this->cursor_ = cursor;
}
LmdbSlamDatabase::Cursor::~Cursor() {
  if (cursor_) {
    mdb_cursor_close(cursor_);
  }
}
void LmdbSlamDatabase::Cursor::Close() {
  if (cursor_) {
    mdb_cursor_close(cursor_);
    cursor_ = nullptr;
  }
}
bool LmdbSlamDatabase::Cursor::Get(size_t& id, Blob& blob) {
  MDB_val key;
  MDB_val data;
  if (!Get(&key, &data)) {
    return false;
  }
  if (key.mv_size != sizeof(id)) {
    return false;
  }
  id = *reinterpret_cast<size_t*>(key.mv_data);
  blob.assign((uint8_t*)data.mv_data, (uint8_t*)data.mv_data + data.mv_size);
  return true;
}
bool LmdbSlamDatabase::Cursor::Get(size_t& id, MDB_val& data) {
  MDB_val key;
  if (!Get(&key, &data)) {
    return false;
  }
  if (key.mv_size != sizeof(id)) {
    return false;
  }
  id = *reinterpret_cast<size_t*>(key.mv_data);
  return true;
}
bool LmdbSlamDatabase::Cursor::Get(MDB_val* key, MDB_val* data) {
  int rc;
  if (!cursor_) {
    return false;
  }
  rc = mdb_cursor_get(cursor_, key, data, MDB_NEXT);
  return rc == 0;
}

LmdbSlamDatabase::LmdbSlamDatabase() {
  // this->Open("c:/cuVSLAM/temp/lmdb", true);
}
LmdbSlamDatabase::~LmdbSlamDatabase() { this->Close(); }

void MDB_assert_callback(MDB_env*, const char* msg) { SlamStderr("LMDB error: %s.\n", msg); }

bool LmdbSlamDatabase::Open(const char* db_path, OpenMode open_mode) {
  read_only_ = open_mode == OpenMode::READ_ONLY_EXISTS;

  if (open_mode == OpenMode::READ_WRITE_HARD_RESET) {
    // remove directory recursively
    std::error_code ec;
    const auto dir = std::filesystem::path(db_path);
    std::filesystem::remove_all(dir, ec);
    if (ec) {
      SlamStderr("LMDB database %s failed to remove_all. %s\n", db_path, ec.message().c_str());
    } else {
      SlamStdout("LMDB database %s removed.\n", db_path);
    }
  }

  if (open_mode == OpenMode::READ_WRITE_EXISTS) {
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::path(db_path) / "data.mdb", ec)) {
      SlamStderr("LMDB database %s does not exist.\n", db_path);
      return false;
    }
  }

  if (!read_only_ && open_mode != OpenMode::READ_WRITE_EXISTS) {
    // Create directory
    std::error_code ec;
    const auto dir = std::filesystem::path(db_path);
    if (!std::filesystem::create_directory(dir, ec) && ec) {
      SlamStderr("Failed to create directory %s. %s\n", db_path, ec.message().c_str());
    } else {
      SlamStdout("Directory %s is created.\n", db_path);
    }
  }

  int rc;
  rc = mdb_env_create(&env);
  if (Error(rc)) {
    return false;
  }
  rc = mdb_env_set_assert(env, &MDB_assert_callback);
  if (Error(rc)) {
    return false;
  }

  constexpr size_t gb = 1024 * 1024 * 1024;
  // On 64-bit there is no penalty for making this huge (say 1TB). Must be <2GB on 32-bit.
  const size_t max_map_size = 10 * gb;

  rc = mdb_env_set_maxdbs(env, 10);
  if (Error(rc)) {
    return false;
  }
  rc = mdb_env_set_mapsize(env, max_map_size);
  if (Error(rc)) {
    return false;
  }

  unsigned int flags = 0;
  flags |= MDB_NOSYNC;
  flags |= MDB_NOLOCK;
  if (read_only_) {
    flags |= MDB_RDONLY;
  }
  // else
  //     flags |= MDB_WRITEMAP;
  rc = mdb_env_open(env, db_path, flags, 0664);
  if (Error(rc)) {
    return false;
  }
  std::error_code ec;
  path_ = std::filesystem::absolute(db_path, ec).string();  // stays right if the working folder changes
  if (ec) {
    path_ = db_path;
  }

  MDB_txn* txn;
  rc = mdb_txn_begin(env, NULL, read_only_ ? MDB_RDONLY : 0, &txn);
  if (Error(rc)) {
    return false;
  }
  rc = mdb_dbi_open(txn, "main", MDB_CREATE, &dbi_main);
  if (Error(rc)) {
    return false;
  }
  rc = mdb_dbi_open(txn, "landmarks", MDB_CREATE | MDB_INTEGERKEY, &dbi_landmarks);
  if (Error(rc)) {
    return false;
  }
  rc = mdb_dbi_open(txn, "descriptors", MDB_CREATE | MDB_INTEGERKEY, &dbi_descriptors);
  if (Error(rc)) {
    return false;
  }

  // rc = mdb_dbi_open(txn, "posegraph_nodes", MDB_CREATE | MDB_INTEGERKEY, &dbi_posegraph_nodes);
  // if (Error(rc)) {
  //     return false;
  // }
  // rc = mdb_dbi_open(txn, "posegraph_edges", MDB_CREATE | MDB_INTEGERKEY, &dbi_posegraph_edges);
  // if (Error(rc)) {
  //     return false;
  // }

  rc = mdb_dbi_open(txn, "spatial_cells", MDB_CREATE | MDB_INTEGERKEY, &dbi_spatial_cells);
  if (Error(rc)) {
    return false;
  }

  rc = mdb_txn_commit(txn);
  if (Error(rc)) {
    return false;
  }

  if (open_mode == OpenMode::READ_WRITE_NEW) {
    // clear all tables
    this->Clear();
    SlamStdout("All existing data has been dropped from the LMDB database.\n");
  }

  dbi_tables_[static_cast<uint32_t>(SlamDatabaseTable::Singletons)] = dbi_main;
  dbi_tables_[static_cast<uint32_t>(SlamDatabaseTable::Landmarks)] = dbi_landmarks;
  dbi_tables_[static_cast<uint32_t>(SlamDatabaseTable::Descriptors)] = dbi_descriptors;
  dbi_tables_[static_cast<uint32_t>(SlamDatabaseTable::SpatialCells)] = dbi_spatial_cells;

  SlamStdout("LMDB database %s is successfully opened %s\n", db_path, read_only_ ? " in \"read only\" mode" : "");

  return true;
}

void LmdbSlamDatabase::Close() {
  if (env == nullptr) {
    return;
  }
  mdb_close(env, dbi_main);
  mdb_close(env, dbi_landmarks);
  mdb_close(env, dbi_descriptors);
  // mdb_close(env, dbi_posegraph_nodes);
  // mdb_close(env, dbi_posegraph_edges);
  mdb_close(env, dbi_spatial_cells);
  mdb_env_close(env);
  env = nullptr;

  SlamStdout("Closing LMDB database.\n");
}

bool LmdbSlamDatabase::Clear() {
  if (read_only_) {
    return false;
  }
  WriteTxn txn(this);
  txn.Drop(dbi_landmarks, 0);
  txn.Drop(dbi_descriptors, 0);
  // txn.Drop(dbi_posegraph_nodes, 0);
  // txn.Drop(dbi_posegraph_edges, 0);
  txn.Drop(dbi_spatial_cells, 0);
  txn.Drop(dbi_main, 0);
  return txn.Commit();
}

std::string LmdbSlamDatabase::Statistic() const {
  int rc;
  MDB_txn* txn;
  rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);

  std::stringstream ss;
  auto print_stat = [&](const char* name, MDB_dbi d) {
    MDB_stat stat;
    rc = mdb_stat(txn, d, &stat);
    if (rc != 0) {
      return;
    }
    ss << "  " << name << ": records=" << stat.ms_entries
       << " mem=" << stat.ms_psize * stat.ms_leaf_pages / ((float)1024 * 1024) << "Mb" << std::endl;
  };

  ss << "LmdbSlamDatabase:\n";

  MDB_stat stat;
  mdb_env_stat(env, &stat);
  ss << "ms_psize:" << stat.ms_psize << " ms_depth=" << stat.ms_depth << " ms_branch_pages=" << stat.ms_branch_pages
     << " ms_leaf_pages=" << stat.ms_leaf_pages << " ms_overflow_pages=" << stat.ms_overflow_pages
     << " ms_entries=" << stat.ms_entries << std::endl;
  MDB_envinfo info;
  mdb_env_info(env, &info);
  ss << "me_mapaddr=" << info.me_mapaddr << " me_mapsize=" << info.me_mapsize
     << " page count=" << info.me_mapsize / stat.ms_psize << " me_last_pgno=" << info.me_last_pgno
     << " me_last_txnid=" << info.me_last_txnid << " me_maxreaders=" << info.me_maxreaders
     << " me_numreaders=" << info.me_numreaders << std::endl;

  print_stat("main", dbi_main);
  print_stat("landmarks", dbi_landmarks);
  print_stat("descriptors", dbi_descriptors);
  // print_stat("posegraph_nodes", dbi_posegraph_nodes);
  // print_stat("posegraph_edges", dbi_posegraph_edges);
  print_stat("spatial_cells", dbi_spatial_cells);
  mdb_txn_abort(txn);

  if (this->errors_.size()) {
    ss << "Errors:\n";
    for (auto it : this->errors_) {
      const char* text = mdb_strerror(it.first);
      if (!text) {
        text = "<no name>";
      }
      ss << it.second << " " << text << std::endl;
    }
  }
  return ss.str();
}

size_t LmdbSlamDatabase::GetRecordsCount(SlamDatabaseTable table) const {
  uint32_t index = static_cast<uint32_t>(table);
  MDB_dbi dbi = dbi_tables_[index];

  MDB_txn* txn;
  int rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
  if (Error(rc)) {
    return 0;
  }
  MDB_stat stat;
  rc = mdb_stat(txn, dbi, &stat);
  if (Error(rc)) {
    return 0;
  }

  mdb_txn_abort(txn);
  return stat.ms_entries;
}

bool LmdbSlamDatabase::IsReadOnly() const { return this->read_only_; }

bool LmdbSlamDatabase::CopyTo(const std::string& path) const {
  if (env == nullptr) {
    return false;
  }
  std::error_code ec;
  // Without a trailing separator, so that "map/" and "map" are the same folder.
  const auto resolve = [&ec](const std::filesystem::path& folder) {
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(folder, ec);
    return resolved.has_filename() ? resolved : resolved.parent_path();
  };
  const std::filesystem::path target = resolve(path);
  const std::filesystem::path own = ec ? std::filesystem::path() : resolve(path_);
  if (ec) {
    SlamStderr("Can't resolve LMDB database paths %s and %s. %s\n", path_.c_str(), path.c_str(), ec.message().c_str());
    return false;
  }
  // Replacing the target folder must not delete this database.
  for (std::filesystem::path entry = own / "data.mdb"; !entry.empty(); entry = entry.parent_path()) {
    if (entry == target || std::filesystem::equivalent(entry, target, ec)) {
      SlamStderr("Can't copy LMDB database %s into %s, which holds it.\n", path_.c_str(), path.c_str());
      return false;
    }
    if (entry == entry.parent_path()) {
      break;
    }
  }
  ec.clear();
  std::filesystem::remove_all(target, ec);
  if (ec || !std::filesystem::create_directory(target, ec)) {
    SlamStderr("Failed to make the empty folder %s. %s\n", path.c_str(), ec.message().c_str());
    return false;
  }
  // Renamed to data.mdb only when complete, so a failed copy leaves no truncated database.
  const std::filesystem::path partial = target / "data.mdb.partial";
  const int fd = open(partial.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0664);
  if (fd < 0) {
    SlamStderr("Failed to create %s. %s\n", partial.c_str(), std::strerror(errno));
    return false;
  }
  // Page by page, so the map never has to fit in memory.
  bool copied = !Error(mdb_env_copyfd2(env, fd, 0));
  if (copied && fsync(fd) != 0) {
    SlamStderr("Failed to write %s. %s\n", partial.c_str(), std::strerror(errno));
    copied = false;
  }
  if (close(fd) != 0 && copied) {
    SlamStderr("Failed to write %s. %s\n", partial.c_str(), std::strerror(errno));
    copied = false;
  }
  if (copied) {
    std::filesystem::rename(partial, target / "data.mdb", ec);
    if (ec) {
      SlamStderr("Failed to rename %s. %s\n", partial.c_str(), ec.message().c_str());
      copied = false;
    }
  }
  if (!copied) {
    std::filesystem::remove(partial, ec);
  }
  return copied;
}

bool LmdbSlamDatabase::IsAt(const std::string& path) const {
  if (env == nullptr) {
    return false;
  }
  std::error_code ec;
  return std::filesystem::equivalent(std::filesystem::path(path_) / "data.mdb",
                                     std::filesystem::path(path) / "data.mdb", ec);
}

bool LmdbSlamDatabase::SetRecordV(SlamDatabaseTable table, Key id, size_t bytes,
                                  const std::function<bool(BlobWriter&)>& func) {
  if (read_only_) {
    return false;
  }

  Blob blob;
  blob.reserve(bytes);
  BlobWriter blob_writer(blob);
  if (!func(blob_writer)) {
    return false;
  }

  WriteTxn txn(this);
  uint32_t index = static_cast<uint32_t>(table);
  txn.Put(dbi_tables_[index], id, blob);
  return txn.Commit();
};
bool LmdbSlamDatabase::GetRecordV(SlamDatabaseTable table, Key id, const std::function<bool(const BlobReader&)>& func,
                                  bool silence_if_not_found) const {
  ReadTxn txn(this, silence_if_not_found);
  uint32_t index = static_cast<uint32_t>(table);
  MDB_val data;
  if (!txn.Get(dbi_tables_[index], id, data)) {
    return false;
  };
  BlobReader blob_reader(data.mv_data, data.mv_size);
  return func(blob_reader);
};
bool LmdbSlamDatabase::DelRecordV(SlamDatabaseTable table, Key id, bool silence_if_not_found) {
  if (read_only_) {
    return false;
  }
  WriteTxn txn(this, silence_if_not_found);
  uint32_t index = static_cast<uint32_t>(table);
  txn.Del(dbi_tables_[index], id);
  return txn.Commit();
}

bool LmdbSlamDatabase::ForEachV(SlamDatabaseTable table,
                                const std::function<bool(Key id, const BlobReader&)>& func) const {
  Blob blob;
  ReadTxn txn(this);
  Cursor cursor;
  const uint32_t index = static_cast<uint32_t>(table);
  if (!txn.OpenCursor(dbi_tables_[index], cursor)) {
    return false;
  }
  Key id;
  MDB_val data;
  while (cursor.Get(id, data)) {
    BlobReader blob_reader(data.mv_data, data.mv_size);
    const bool continue_processing = func(id, blob_reader);
    if (!continue_processing) {
      return true;  // stopped by lambda
    }
  }
  return true;
}

bool LmdbSlamDatabase::Flush() {
  if (read_only_) {
    return false;
  }

  SlamStdout("Flushing LMDB database.\n");
  int rc = mdb_env_sync(env, 1);
  if (Error(rc)) {
    return false;
  }
  return true;
}

bool LmdbSlamDatabase::Error(int rc, MDB_dbi dbi) const {
  if (rc == 0) {
    return false;  // no errors
  }
  if (this->errors_.find(rc) == this->errors_.end()) {
    std::string table_name = "unknown";
    for (uint32_t i = 0; i < static_cast<uint32_t>(SlamDatabaseTable::Max); i++) {
      if (dbi == dbi_tables_[i]) {
        switch (static_cast<SlamDatabaseTable>(i)) {
          case SlamDatabaseTable::Singletons:
            table_name = "Singletons";
            break;
          case SlamDatabaseTable::Landmarks:
            table_name = "Landmarks";
            break;
          case SlamDatabaseTable::Descriptors:
            table_name = "Descriptors";
            break;
          case SlamDatabaseTable::SpatialCells:
            table_name = "SpatialCells";
            break;
          default:
            table_name = "Unknown table";
            break;
        }
      }
    }

    this->errors_[rc] = 0;

    auto text = mdb_strerror(rc);
    if (text) {
      SlamStderr("\033[31mLMDB error: %s in %s table\033[0m\n", text, table_name.c_str());
    }

    if (rc == MDB_MAP_FULL) {
      SlamStderr("Database map is full. Statistics:\n%s\n", Statistic().c_str());
    }
  }
  this->errors_[rc]++;
  return true;  // has error
}

}  // namespace cuvslam::slam
