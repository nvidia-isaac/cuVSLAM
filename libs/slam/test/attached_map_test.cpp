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

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "camera/camera.h"
#include "common/include_gtest.h"
#include "slam/map/database/lmdb_slam_database.h"
#include "slam/map/descriptor/st_descriptor_ops.h"
#include "slam/slam/slam.h"

#include <unistd.h>

#ifdef USE_LMDB

namespace test::slam {
using namespace cuvslam;
using namespace cuvslam::slam;

namespace {

// Shi-Tomasi descriptor with one patch filled with `value`.
FeatureDescriptor MakeDescriptor(float value) {
  const size_t size_bytes = sizeof(STDescriptor) + sizeof(sof::STFeaturePatch);
  auto memory = std::shared_ptr<STDescriptor>(reinterpret_cast<STDescriptor*>(new char[size_bytes]),
                                              [](void* ptr) { delete[] static_cast<char*>(ptr); });
  memory->levels_mask = 1;
  memory->image_patches_size = 1;
  memory->image_patches[0].setConstant(value);
  FeatureDescriptor fd;
  fd.pull_memory = memory;
  return fd;
}

// Serializes descriptors but fails to store them, as LMDB does when its map is full.
class DescriptorRefusingDatabase : public LmdbSlamDatabase {
public:
  bool SetRecordV(SlamDatabaseTable table, Key id, size_t bytes,
                  const std::function<bool(BlobWriter&)>& func) override {
    if (table != SlamDatabaseTable::Descriptors) {
      return LmdbSlamDatabase::SetRecordV(table, id, bytes, func);
    }
    Blob blob;
    BlobWriter writer(blob);
    func(writer);
    return false;
  }
};

// A session with hand-made landmarks; AttachToNewDatabase() attaches it like Slam::Config::map_path.
class AttachedMap : public ::testing::Test {
protected:
  void SetUp() override {
    camera_ = camera::CreateCameraModel(Vector2T(640, 480), Vector2T(320, 320), Vector2T(320, 240),
                                        Distortion::Model::Pinhole, nullptr, 0);
    rig_.intrinsics[0] = camera_.get();
    rig_.num_cameras = 1;
    rig_.camera_from_rig[0].setIdentity();

    folder_ = std::filesystem::temp_directory_path() /
              ("cuvslam_attached_map_test_" + std::to_string(getpid()) + "_" +
               std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::filesystem::remove_all(folder_);
    std::filesystem::create_directories(folder_);
    mapper_ = MakeMapper();
  }

  void TearDown() override {
    mapper_.reset();
    std::filesystem::remove_all(folder_);
  }

  std::unique_ptr<LocalizerAndMapper> MakeMapper() const {
    auto mapper = std::make_unique<LocalizerAndMapper>(rig_, FeatureDescriptorType::kShiTomasi6, false);
    mapper->SetActiveCameras({0});
    return mapper;
  }

  std::string Path(const char* name) const { return (folder_ / name).string(); }

  Map& MutableMap() { return const_cast<Map&>(mapper_->GetMap()); }

  // Adds a keyframe 1 m further along Z and returns its id.
  KeyFrameId AddKeyframe() {
    Isometry3T step = Isometry3T::Identity();
    step.translation().z() = 1.f;
    VOFrameData frame_data;
    frame_data.frame_id = next_frame_;
    frame_data.timestamp_ns = next_frame_ * kFrameDeltaNs;
    ++next_frame_;
    mapper_->AddKeyframe(step, frame_data, Images());
    KeyFrameId head = InvalidKeyFrameId;
    EXPECT_TRUE(MutableMap().pose_graph_.GetHeadKeyframe(head));
    return head;
  }

  // Relates `id` to `keyframe_id`, 5 m ahead of it.
  void AddRelation(LandmarkId id, KeyFrameId keyframe_id) {
    Map& map = MutableMap();
    ASSERT_TRUE(map.pose_graph_.AddLandmarkRelation(id, keyframe_id));
    const Vector3T xyz(0.f, 0.f, 5.f);
    map.landmarks_spatial_index_->AddLandmarkRelation(id, keyframe_id, &xyz, Vector2T(0.f, 0.f),
                                                      map.pose_graph_hypothesis_);
  }

  // Adds a keyframe with a new staged landmark on it.
  LandmarkId AddKeyframeWithLandmark() {
    const KeyFrameId head = AddKeyframe();
    const float value = static_cast<float>(descriptors_.size()) + 0.5f;
    const LandmarkId id =
        MutableMap().landmarks_spatial_index_->AddLandmark(MakeDescriptor(value), next_frame_ * kFrameDeltaNs);
    EXPECT_NE(id, InvalidLandmarkId);
    AddRelation(id, head);
    descriptors_[id] = MakeDescriptor(value);
    return id;
  }

  void ActivateStagedLandmarks() {
    Map& map = MutableMap();
    // No head keyframe means lost tracking, which activates every staged landmark.
    map.landmarks_spatial_index_->MoveReadyStagedLandmarksToLSI(
        mapper_->GetCurrentPose(), {0}, map.pose_graph_hypothesis_, InvalidKeyFrameId, next_frame_ * kFrameDeltaNs);
  }

  // Adds `count` keyframes 1 m apart along Z, one landmark 5 m ahead of each, and activates them all.
  void MapLandmarks(int count) {
    for (int i = 0; i < count; ++i) {
      AddKeyframeWithLandmark();
    }
    ActivateStagedLandmarks();
  }

  void RemoveLandmark(LandmarkId id) {
    Map& map = MutableMap();
    std::function<void(LandmarkId, KeyFrameId)> remove_relation = [&](LandmarkId landmark_id, KeyFrameId keyframe_id) {
      map.pose_graph_.RemoveLandmarkRelation(landmark_id, keyframe_id);
    };
    map.landmarks_spatial_index_->RemoveLandmarks({id}, remove_relation);
    descriptors_.erase(id);
  }

  // How many ids Query() reports, counting repeats.
  static size_t QueryCount(const LocalizerAndMapper& mapper) {
    size_t count = 0;
    mapper.GetMap().GetLandmarksSpatialIndex()->Query([&](LandmarkId) {
      ++count;
      return true;
    });
    return count;
  }

  // Landmark and descriptor records in the map at `folder`.
  static std::pair<size_t, size_t> SavedRecords(const std::string& folder) {
    LmdbSlamDatabase database;
    EXPECT_TRUE(database.Open(folder.c_str(), LmdbSlamDatabase::OpenMode::READ_ONLY_EXISTS));
    return {database.GetRecordsCount(SlamDatabaseTable::Landmarks),
            database.GetRecordsCount(SlamDatabaseTable::Descriptors)};
  }

  // Landmarks in the cells of the map at `folder`.
  static std::set<LandmarkId> SavedCellLandmarks(const std::string& folder) {
    LmdbSlamDatabase database;
    EXPECT_TRUE(database.Open(folder.c_str(), LmdbSlamDatabase::OpenMode::READ_ONLY_EXISTS));
    std::set<LandmarkId> landmarks;
    database.ForEach(SlamDatabaseTable::SpatialCells, [&](ISlamDatabase::Key, const BlobReader& reader) {
      std::vector<LandmarkId> cell;
      EXPECT_TRUE(reader.read_std(cell));
      landmarks.insert(cell.begin(), cell.end());
      return true;
    });
    return landmarks;
  }

  // Expects `mapper` to read back every landmark's descriptor unchanged.
  void ExpectEveryDescriptor(const LocalizerAndMapper& mapper, const std::string& where) const {
    const Map& map = mapper.GetMap();
    int missing = 0;
    for (const auto& [id, original] : descriptors_) {
      const FeatureDescriptor fd = map.GetLandmarksSpatialIndex()->GetLandmarkFeatureDescriptor(id);
      if (!fd) {
        ++missing;
        continue;
      }
      EXPECT_EQ(map.feature_descriptor_ops_->Compare(fd, original), 0) << where << ": landmark " << id;
    }
    EXPECT_EQ(missing, 0) << where << ": " << missing << " of " << descriptors_.size()
                          << " landmarks have no descriptor";
  }

  void ExpectSavedMapHasEveryDescriptor(const std::string& folder) const {
    const auto reader = MakeMapper();
    ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(folder));
    ExpectEveryDescriptor(*reader, folder);
  }

  static constexpr int kLandmarks = 10;
  static constexpr int64_t kFrameDeltaNs = 33'000'000;

  std::unique_ptr<camera::ICameraModel> camera_;
  camera::Rig rig_;
  std::filesystem::path folder_;
  std::unique_ptr<LocalizerAndMapper> mapper_;
  std::map<LandmarkId, FeatureDescriptor> descriptors_;
  int64_t next_frame_ = 0;
};

}  // namespace

TEST_F(AttachedMap, InMemorySessionSavesEveryDescriptor) {
  MapLandmarks(kLandmarks);
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(Path("saved")));
  ExpectEveryDescriptor(*mapper_, "session after save");
  ExpectSavedMapHasEveryDescriptor(Path("saved"));
}

TEST_F(AttachedMap, AttachedSessionReadsOffloadedDescriptors) {
  ASSERT_TRUE(mapper_->AttachToNewDatabase(Path("attached")));
  MapLandmarks(kLandmarks);
  ExpectEveryDescriptor(*mapper_, "attached session");
}

TEST_F(AttachedMap, FlushedMapFileHoldsTheWholeMap) {
  ASSERT_TRUE(mapper_->AttachToNewDatabase(Path("attached")));
  MapLandmarks(kLandmarks);
  AddKeyframeWithLandmark();  // still staged at the flush
  ASSERT_TRUE(mapper_->FlushMapToDatabase());
  mapper_.reset();  // ends without the shutdown write, so the map file holds only what was flushed

  const auto reader = MakeMapper();
  ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(Path("attached")));
  EXPECT_EQ(reader->GetMap().GetPoseGraph().GetKeyframeCount(), static_cast<size_t>(kLandmarks + 1));
  ExpectEveryDescriptor(*reader, "flushed map");
  const std::set<LandmarkId> in_cells = SavedCellLandmarks(Path("attached"));
  for (const auto& [id, descriptor] : descriptors_) {
    EXPECT_EQ(in_cells.count(id), 1u) << "landmark " << id << " is in no cell";
  }
}

TEST_F(AttachedMap, StagedLandmarkStaysStagedAfterFlush) {
  ASSERT_TRUE(mapper_->AttachToNewDatabase(Path("attached")));
  MapLandmarks(kLandmarks);
  const LandmarkId staged = AddKeyframeWithLandmark();
  ASSERT_TRUE(mapper_->FlushMapToDatabase());

  // Reading it does not load the flushed copy next to the staged one.
  KeyFrameId head = InvalidKeyFrameId;
  ASSERT_TRUE(mapper_->GetMap().GetPoseGraph().GetHeadKeyframe(head));
  EXPECT_TRUE(mapper_->GetMap().GetLandmarksSpatialIndex()->GetLandmarkRelation(staged, head));
  EXPECT_EQ(QueryCount(*mapper_), static_cast<size_t>(kLandmarks + 1));

  // A relation added while it is staged stays after its activation.
  AddRelation(staged, AddKeyframe());
  ActivateStagedLandmarks();
  ASSERT_TRUE(mapper_->FlushMapToDatabase());
  mapper_.reset();
  const auto reader = MakeMapper();
  ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(Path("attached")));
  int relations = 0;
  reader->GetMap().GetLandmarksSpatialIndex()->QueryLandmarkRelations(staged, [&](KeyFrameId, const Vector2T&) {
    ++relations;
    return true;
  });
  EXPECT_EQ(relations, 2);
}

TEST_F(AttachedMap, FlushTrimsThePoseGraphToMaxMapSize) {
  // More than the 20 newest keyframes, which PoseGraph keeps from reduction.
  constexpr int kKeyframes = 25;
  mapper_->SetKeyframesLimit(22);
  ASSERT_TRUE(mapper_->AttachToNewDatabase(Path("attached")));
  MapLandmarks(kKeyframes);
  ASSERT_EQ(mapper_->GetMap().GetPoseGraph().GetKeyframeCount(), static_cast<size_t>(kKeyframes));
  ASSERT_TRUE(mapper_->FlushMapToDatabase());
  mapper_.reset();

  const auto reader = MakeMapper();
  ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(Path("attached")));
  EXPECT_EQ(reader->GetMap().GetPoseGraph().GetKeyframeCount(), 22u);
  ExpectEveryDescriptor(*reader, "flushed map");
}

TEST_F(AttachedMap, MappingGoesOnAfterFlush) {
  ASSERT_TRUE(mapper_->AttachToNewDatabase(Path("attached")));
  MapLandmarks(kLandmarks);
  ASSERT_TRUE(mapper_->FlushMapToDatabase());
  ExpectEveryDescriptor(*mapper_, "session after flush");
  MapLandmarks(kLandmarks);
  ASSERT_TRUE(mapper_->FlushMapToDatabase());
  mapper_.reset();
  ExpectSavedMapHasEveryDescriptor(Path("attached"));
}

TEST_F(AttachedMap, RemovedLandmarkTakesItsDescriptorAlong) {
  ASSERT_TRUE(mapper_->AttachToNewDatabase(Path("attached")));
  MapLandmarks(kLandmarks);
  RemoveLandmark(descriptors_.begin()->first);
  ASSERT_TRUE(mapper_->FlushMapToDatabase());
  mapper_.reset();

  const auto reader = MakeMapper();
  ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(Path("attached")));
  EXPECT_EQ(reader->GetMap().database_->GetRecordsCount(SlamDatabaseTable::Descriptors), descriptors_.size());
  ExpectEveryDescriptor(*reader, "attached map");
}

TEST_F(AttachedMap, SaveMapFailsInAnAttachedSession) {
  ASSERT_TRUE(mapper_->AttachToNewDatabase(Path("attached")));
  MapLandmarks(kLandmarks);
  EXPECT_FALSE(mapper_->AttachToNewDatabaseSaveMapAndDetach(Path("saved")));
  EXPECT_FALSE(std::filesystem::exists(Path("saved")));
  ASSERT_NE(mapper_->GetMap().database_, nullptr);
  EXPECT_FALSE(mapper_->GetMap().database_->IsReadOnly());
  ExpectEveryDescriptor(*mapper_, "session after the refused save");
}

TEST_F(AttachedMap, LandmarkRemovedFromAReadOnlyMapStaysRemoved) {
  MapLandmarks(kLandmarks);
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(Path("saved")));
  mapper_ = MakeMapper();
  ASSERT_TRUE(mapper_->AttachToExistingReadOnlyDatabase(Path("saved")));
  const LandmarkId removed = descriptors_.begin()->first;
  ASSERT_TRUE(mapper_->GetMap().GetLandmarksSpatialIndex()->GetLandmarkFeatureDescriptor(removed));
  RemoveLandmark(removed);

  // The read-only database keeps it, but the session does not read it back.
  EXPECT_EQ(QueryCount(*mapper_), descriptors_.size());
  EXPECT_FALSE(mapper_->GetMap().GetLandmarksSpatialIndex()->GetLandmarkFeatureDescriptor(removed));
  // A saved copy leaves it out, and so does the folder once the session writes it.
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(Path("copy")));
  EXPECT_EQ(SavedRecords(Path("copy")), std::make_pair(descriptors_.size(), descriptors_.size()));
  ASSERT_TRUE(mapper_->ReopenDatabaseForWriting(Path("saved")));
  ASSERT_TRUE(mapper_->FlushMapToDatabase());
  mapper_.reset();
  EXPECT_EQ(SavedRecords(Path("saved")), std::make_pair(descriptors_.size(), descriptors_.size()));
  ExpectSavedMapHasEveryDescriptor(Path("saved"));
}

TEST_F(AttachedMap, DescriptorStaysInMemoryWhenItsWriteFails) {
  const auto database = std::make_shared<DescriptorRefusingDatabase>();
  ASSERT_TRUE(database->Open(Path("full").c_str(), LmdbSlamDatabase::OpenMode::READ_WRITE_HARD_RESET));
  ASSERT_TRUE(MutableMap().AttachDatabase(database, false));
  MapLandmarks(kLandmarks);
  ExpectEveryDescriptor(*mapper_, "session whose descriptors the database refused");
}

TEST_F(AttachedMap, FlushMapNeedsAnAttachedMap) {
  MapLandmarks(kLandmarks);
  EXPECT_FALSE(mapper_->FlushMapToDatabase());
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(Path("saved")));
  const auto reader = MakeMapper();
  ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(Path("saved")));
  EXPECT_FALSE(reader->FlushMapToDatabase());
}

}  // namespace test::slam

#endif  // USE_LMDB
