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

#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "camera/camera.h"
#include "common/include_gtest.h"
#include "slam/map/descriptor/st_descriptor_ops.h"
#include "slam/map/spatial_index/lsi_grid.h"

#ifdef USE_LMDB
#include "slam/map/database/lmdb_slam_database.h"
#endif

#include <unistd.h>

namespace test::slam {
using namespace cuvslam;
using namespace cuvslam::slam;

namespace {

// LSIGrid with its cells and landmarks open to the test.
class OpenGrid : public LSIGrid {
public:
  using StoredLandmark = decltype(landmarks_)::mapped_type;
  using LSIGrid::AppendCellLandmarks;
  using LSIGrid::cells_;
  using LSIGrid::dead_landmarks_;
  using LSIGrid::GetLandmark;
  using LSIGrid::landmark_quality_func_;
  using LSIGrid::landmarks_;
  using LSIGrid::LandmarkToBlob;
  using LSIGrid::LSIGrid;
  using LSIGrid::ReduceLandmarksInCell;
};

class CellLimit : public ::testing::Test {
protected:
  void SetUp() override {
    camera_ = camera::CreateCameraModel(Vector2T(640, 480), Vector2T(320, 320), Vector2T(320, 240),
                                        Distortion::Model::Pinhole, nullptr, 0);
    rig_.intrinsics[0] = camera_.get();
    rig_.num_cameras = 1;
    rig_.camera_from_rig[0].setIdentity();
    grid_ = std::make_unique<OpenGrid>(ops_, rig_, 0.25f, 1);
  }

  // A landmark in `cells`, seen `solved` times.
  OpenGrid::StoredLandmark MakeLandmark(const std::vector<HashCellId>& cells, int solved) const {
    OpenGrid::StoredLandmark landmark;
    landmark.cells = cells;
    landmark.probes_types[LP_SOLVER_OK] = solved;
    return landmark;
  }

  STDescriptorOps ops_{20, 0};
  std::unique_ptr<camera::ICameraModel> camera_;
  camera::Rig rig_;
  std::unique_ptr<OpenGrid> grid_;
};

}  // namespace

TEST_F(CellLimit, EvictionFromOneCellKeepsTheLandmarkInItsOtherCells) {
  constexpr HashCellId kFull = 10;
  constexpr HashCellId kOther = 20;
  grid_->landmarks_[1] = MakeLandmark({kFull, kOther}, 0);
  grid_->landmarks_[2] = MakeLandmark({kFull}, 5);
  grid_->cells_[kFull].landmarks_in_cell = {1, 2};
  grid_->cells_[kOther].landmarks_in_cell = {1};

  EXPECT_EQ(grid_->ReduceLandmarksInCell(kFull, grid_->landmark_quality_func_, 1), 1u);
  EXPECT_EQ(grid_->cells_[kFull].landmarks_in_cell, std::vector<LandmarkId>{2});
  EXPECT_EQ(grid_->landmarks_[1].cells, std::vector<HashCellId>{kOther});
  EXPECT_TRUE(grid_->dead_landmarks_.empty());
}

#ifdef USE_LMDB
TEST_F(CellLimit, LandmarksOnlyInTheDatabaseCompeteForTheCell) {
  constexpr HashCellId kFull = 10;
  const std::filesystem::path folder =
      std::filesystem::temp_directory_path() / ("cuvslam_lsi_grid_test_" + std::to_string(getpid()));
  const auto database = std::make_shared<LmdbSlamDatabase>();
  ASSERT_TRUE(database->Open(folder.c_str(), LmdbSlamDatabase::OpenMode::READ_WRITE_HARD_RESET));
  const OpenGrid::StoredLandmark in_database = MakeLandmark({kFull}, 5);
  ASSERT_TRUE(database->SetRecord(SlamDatabaseTable::Landmarks, 1, 0, [&](BlobWriter& writer) {
    grid_->LandmarkToBlob(in_database, writer);
    return true;
  }));
  grid_->SetDatabase(database);
  grid_->landmarks_[2] = MakeLandmark({kFull}, 0);
  grid_->cells_[kFull].landmarks_in_cell = {1, 2};

  EXPECT_EQ(grid_->ReduceLandmarksInCell(kFull, grid_->landmark_quality_func_, 1), 1u);
  EXPECT_EQ(grid_->cells_[kFull].landmarks_in_cell, std::vector<LandmarkId>{1});
  EXPECT_EQ(grid_->dead_landmarks_, std::vector<LandmarkId>{2});

  grid_->SetDatabase(nullptr);
  database->Close();
  std::filesystem::remove_all(folder);
}

// The grid backed by a scratch database, to feed it records the way a map file on disk would.
class GridOverDatabase : public CellLimit {
protected:
  void SetUp() override {
    CellLimit::SetUp();
    folder_ = std::filesystem::temp_directory_path() / ("cuvslam_lsi_grid_db_test_" + std::to_string(getpid()));
    database_ = std::make_shared<LmdbSlamDatabase>();
    ASSERT_TRUE(database_->Open(folder_.c_str(), LmdbSlamDatabase::OpenMode::READ_WRITE_HARD_RESET));
  }
  void TearDown() override {
    grid_->SetDatabase(nullptr);
    database_->Close();
    std::filesystem::remove_all(folder_);
  }

  void Put(SlamDatabaseTable table, ISlamDatabase::Key key, const Blob& record) {
    ASSERT_TRUE(database_->SetRecord(table, key, record.size(), [&](BlobWriter& writer) {
      writer.write(record.data(), record.size());
      return true;
    }));
  }

  Blob LandmarkRecord(const OpenGrid::StoredLandmark& landmark) const {
    Blob record;
    BlobWriter writer(record);
    grid_->LandmarkToBlob(landmark, writer);
    return record;
  }

  std::filesystem::path folder_;
  std::shared_ptr<LmdbSlamDatabase> database_;
};

TEST_F(GridOverDatabase, CellOnlyInTheDatabaseIsCopiedOut) {
  Blob cell;
  BlobWriter(cell).write_std(std::vector<LandmarkId>{7, 8});
  Put(SlamDatabaseTable::SpatialCells, 30, cell);

  Blob short_cell;
  BlobWriter short_writer(short_cell);
  short_writer.write(size_t{5});
  short_writer.write(LandmarkId{9});
  Put(SlamDatabaseTable::SpatialCells, 31, short_cell);

  grid_->cells_[40].landmarks_in_cell = {};
  grid_->SetDatabase(database_);

  std::vector<LandmarkId> landmarks = {1};
  grid_->AppendCellLandmarks(30, landmarks);
  EXPECT_EQ(landmarks, (std::vector<LandmarkId>{1, 7, 8}));
  grid_->AppendCellLandmarks(31, landmarks);
  grid_->AppendCellLandmarks(32, landmarks);
  grid_->AppendCellLandmarks(40, landmarks);
  EXPECT_EQ(landmarks, (std::vector<LandmarkId>{1, 7, 8}));
}

TEST_F(GridOverDatabase, CorruptLandmarkRecordIsNotLoaded) {
  Blob huge_count = LandmarkRecord(MakeLandmark({10}, 1));
  const uint32_t keyframes_size = std::numeric_limits<uint32_t>::max();
  std::memcpy(huge_count.data() + sizeof(uint32_t), &keyframes_size, sizeof(keyframes_size));
  Put(SlamDatabaseTable::Landmarks, 1, huge_count);

  Blob truncated = LandmarkRecord(MakeLandmark({10}, 1));
  truncated.resize(truncated.size() - 1);
  Put(SlamDatabaseTable::Landmarks, 2, truncated);

  Put(SlamDatabaseTable::Landmarks, 3, LandmarkRecord(MakeLandmark({10}, 1)));
  grid_->SetDatabase(database_);

  EXPECT_EQ(grid_->GetLandmark(1), nullptr);
  EXPECT_EQ(grid_->GetLandmark(2), nullptr);
  EXPECT_NE(grid_->GetLandmark(3), nullptr);
}

// The descriptor is read inside a database transaction that ends before the grid hands it out.
TEST_F(GridOverDatabase, DescriptorFromTheDatabaseOwnsItsMemory) {
  Put(SlamDatabaseTable::Landmarks, 1, LandmarkRecord(MakeLandmark({10}, 1)));
  Blob descriptor;
  BlobWriter writer(descriptor);
  writer.write(uint32_t{1});  // levels_mask
  writer.write(uint32_t{1});  // image_patches_size
  writer.write(sof::STFeaturePatch(sof::STFeaturePatch::Constant(3.f)));
  Put(SlamDatabaseTable::Descriptors, 1, descriptor);
  grid_->SetDatabase(database_);

  const FeatureDescriptor fd = grid_->GetLandmarkFeatureDescriptor(1);
  ASSERT_TRUE(fd);
  EXPECT_EQ(fd.const_memory_ptr, nullptr);
  EXPECT_EQ(static_cast<const STDescriptor*>(fd.get())->image_patches[0](0, 0), 3.f);
}
#endif  // USE_LMDB

}  // namespace test::slam
