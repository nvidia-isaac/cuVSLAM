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

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
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
  using LSIGrid::cells_;
  using LSIGrid::HashCellIdFromEye;
  using LSIGrid::landmarks_;
  using LSIGrid::LandmarkToBlob;
  using LSIGrid::LSIGrid;
};

class KeyframeMove : public ::testing::Test {
protected:
  void SetUp() override {
    camera_ = camera::CreateCameraModel(Vector2T(640, 480), Vector2T(320, 320), Vector2T(320, 240),
                                        Distortion::Model::Pinhole, nullptr, 0);
    rig_.intrinsics[0] = camera_.get();
    rig_.num_cameras = 1;
    rig_.camera_from_rig[0].setIdentity();
    grid_ = std::make_unique<OpenGrid>(ops_, rig_, 0.25f, 0);
  }

  // A landmark in `cell`, seen from `keyframe_id`.
  static OpenGrid::StoredLandmark MakeLandmark(HashCellId cell, KeyFrameId keyframe_id) {
    OpenGrid::StoredLandmark landmark;
    landmark.cells = {cell};
    landmark.keyframes.emplace_back().keyframe_id = keyframe_id;
    return landmark;
  }

  STDescriptorOps ops_{20, 0};
  std::unique_ptr<camera::ICameraModel> camera_;
  camera::Rig rig_;
  std::unique_ptr<OpenGrid> grid_;
};

}  // namespace

#ifdef USE_LMDB
TEST_F(KeyframeMove, LandmarksOnlyInTheDatabaseMoveToTheNewCells) {
  constexpr KeyFrameId kMoved = 7;
  const std::filesystem::path folder =
      std::filesystem::temp_directory_path() / ("cuvslam_lsi_grid_test_move_" + std::to_string(getpid()));
  auto database = std::make_shared<LmdbSlamDatabase>();
  ASSERT_TRUE(database->Open(folder.c_str(), LmdbSlamDatabase::OpenMode::READ_WRITE_HARD_RESET));
  Isometry3T before = Isometry3T::Identity();
  Isometry3T after = Isometry3T::Identity();
  after.translation() = Vector3T(1.f, 0.f, 0.f);  // four cells away
  std::vector<HashCellId> cells_before;
  std::vector<HashCellId> cells_after;
  grid_->HashCellIdFromEye(before.translation(), cells_before);
  grid_->HashCellIdFromEye(after.translation(), cells_after);

  // 1 and 2 are only in the database, 3 is in memory; the moved keyframe sees 1 and 3.
  const OpenGrid::StoredLandmark seen = MakeLandmark(cells_before[0], kMoved);
  const OpenGrid::StoredLandmark unseen = MakeLandmark(cells_before[0], kMoved + 1);
  for (const auto& [id, landmark] : {std::pair{1, seen}, std::pair{2, unseen}}) {
    ASSERT_TRUE(database->SetRecord(SlamDatabaseTable::Landmarks, id, 0, [&](BlobWriter& writer) {
      grid_->LandmarkToBlob(landmark, writer);
      return true;
    }));
  }
  ASSERT_TRUE(database->SetRecord(SlamDatabaseTable::SpatialCells, cells_before[0], 0, [](BlobWriter& writer) {
    writer.write_std(std::vector<LandmarkId>{1, 2, 3});
    return true;
  }));
  grid_->SetDatabase(database);
  grid_->landmarks_[3] = seen;

  grid_->OnUpdateKeyframePose(kMoved, before, after);
  grid_->OnUpdateKeyframePoseFinished();
  EXPECT_EQ(grid_->cells_[cells_after[0]].landmarks_in_cell, (std::vector<LandmarkId>{1, 3}));
  EXPECT_EQ(grid_->landmarks_.size(), 1u);  // read, not kept in memory

  grid_->SetDatabase(nullptr);
  database.reset();  // closes it
  std::filesystem::remove_all(folder);
}
#endif  // USE_LMDB

}  // namespace test::slam
