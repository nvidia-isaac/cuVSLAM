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

#include <csignal>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "camera/camera.h"
#include "common/include_gtest.h"
#include "slam/async_slam/async_slam.h"
#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"
#include "slam/slam/slam.h"
#include "slam/view/map_to_view.h"
#include "sof/image_context.h"

#include <sys/resource.h>
#include <unistd.h>

#ifdef USE_LMDB

namespace test::slam {
using namespace cuvslam;
using namespace cuvslam::slam;

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr size_t kSavedLandmarks = 88;

::testing::AssertionResult Contains(const std::string& text, const std::string& part) {
  if (text.find(part) != std::string::npos) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "\"" << text << "\" has no \"" << part << "\"";
}

// Saved map: 88 landmarks seen from the first of two keyframes, on a blocky texture.
class LocalizeInMapLandmarks : public ::testing::Test {
protected:
  void SetUp() override {
    camera_ = camera::CreateCameraModel(Vector2T(kWidth, kHeight), Vector2T(320, 320), Vector2T(320, 240),
                                        Distortion::Model::Pinhole, nullptr, 0);
    rig_.intrinsics[0] = camera_.get();
    rig_.num_cameras = 1;
    rig_.camera_from_rig[0].setIdentity();

    folder_ = std::filesystem::temp_directory_path() /
              ("cuvslam_localize_in_map_test_" + std::to_string(getpid()) + "_" +
               std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::filesystem::remove_all(folder_);
    std::filesystem::create_directories(folder_);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> gray(0, 255);
    std::vector<uint8_t> blocks((kWidth / 8) * (kHeight / 8));
    for (uint8_t& block : blocks) {
      block = static_cast<uint8_t>(gray(rng));
    }
    pixels_.resize(kWidth * kHeight);
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        pixels_[y * kWidth + x] = blocks[(y / 8) * (kWidth / 8) + x / 8];
      }
    }
    image_ = MakeImage(0, 1000);
  }

  void TearDown() override { std::filesystem::remove_all(folder_); }

  std::string Folder(const std::string& name) const { return (folder_ / name).string(); }

  sof::ImageContextPtr MakeImage(FrameId frame_id, int64_t timestamp_ns) {
    auto context = std::make_shared<sof::ImageContext>(ImageShape{kWidth, kHeight}, false, false);
    ImageMeta meta{};
    meta.shape = ImageShape{kWidth, kHeight};
    meta.frame_id = frame_id;
    meta.timestamp = timestamp_ns;
    meta.camera_index = 0;
    context->set_image_meta(meta);
    ImageSource source;
    source.data = pixels_.data();
    source.type = ImageSource::U8;
    source.memory_type = ImageSource::Host;
    source.image_encoding = ImageEncoding::MONO8;
    EXPECT_TRUE(context->build_cpu_image_pyramid(source, false));
    EXPECT_TRUE(context->build_cpu_gradient_pyramid(false));
    return context;
  }

  void SaveMap(const std::string& folder) {
    LocalizerAndMapper writer(rig_, FeatureDescriptorType::kShiTomasi6, false);
    writer.SetActiveCameras({0});
    writer.SetLandmarksSpatialIndex({});
    VOFrameData frame;
    frame.frame_id = 0;
    frame.timestamp_ns = 1000;
    writer.AddKeyframe(Isometry3T::Identity(), frame, Images());

    Map& map = const_cast<Map&>(writer.GetMap());
    std::vector<Vector2T> uvs;
    for (int y = 40; y <= kHeight - 40; y += 56) {
      for (int x = 40; x <= kWidth - 40; x += 56) {
        uvs.emplace_back(static_cast<float>(x), static_cast<float>(y));
      }
    }
    std::vector<FeatureDescriptor> descriptors;
    map.feature_descriptor_ops_->CreateDescriptors(image_, uvs, descriptors);
    ASSERT_EQ(descriptors.size(), kSavedLandmarks);
    for (size_t i = 0; i < uvs.size(); ++i) {
      ASSERT_TRUE(descriptors[i]);
      const float depth = 3.f + static_cast<float>(i % 5);
      const Vector2T uv_norm((uvs[i].x() - 320) / 320, (uvs[i].y() - 240) / 320);
      const Vector3T xyz(uv_norm.x() * depth, uv_norm.y() * depth, depth);
      const LandmarkId id = map.landmarks_spatial_index_->AddLandmark(descriptors[i], 1000);
      ASSERT_TRUE(map.pose_graph_.AddLandmarkRelation(id, 0));
      map.landmarks_spatial_index_->AddLandmarkRelation(id, 0, &xyz, uv_norm, map.pose_graph_hypothesis_);
    }

    // Loop closure skips the head keyframe, so map one more.
    Isometry3T step = Isometry3T::Identity();
    step.translation().x() = 1.f;
    frame.frame_id = 1;
    frame.timestamp_ns = 1500;
    writer.AddKeyframe(step, frame, Images());
    ASSERT_TRUE(writer.AttachToNewDatabaseSaveMapAndDetach(folder));
  }

  // Landmark ids, and how many of them have a descriptor.
  static std::pair<size_t, size_t> Count(const LocalizerAndMapper& mapper) {
    size_t ids = 0;
    size_t descriptors = 0;
    const auto lsi = mapper.GetMap().GetLandmarksSpatialIndex();
    // Query() is false whenever no database is attached.
    lsi->Query([&](LandmarkId id) {
      ++ids;
      descriptors += lsi->GetLandmarkFeatureDescriptor(id) ? 1 : 0;
      return true;
    });
    return {ids, descriptors};
  }

  std::pair<size_t, size_t> CountSaved(const std::string& folder) const {
    LocalizerAndMapper reader(rig_, FeatureDescriptorType::kShiTomasi2, false);
    EXPECT_TRUE(reader.AttachToExistingReadOnlyDatabase(folder));
    return Count(reader);
  }

  // Landmarks a loop closure at the first keyframe selects.
  uint32_t SelectedByLoopClosure(LocalizerAndMapper& mapper) {
    std::unique_ptr<ILoopClosureSolver> solver(
        CreateLoopClosureSolver(LoopClosureSolverType::kTwoStepsEasy, RansacType::kPnP, false, rig_));
    LocalizerAndMapper::LoopClosureStatus status;
    mapper.DetectLoopClosure(*solver, Images{image_}, Isometry3T::Identity(), status);
    return status.selected_landmarks_count;
  }

  // Feeds one odometry result without tracks, as Slam::Track does in sync mode.
  void Track(AsyncSlam& slam, FrameId frame_id, int64_t timestamp_ns, bool keyframe) {
    odom::IVisualOdometry::VOFrameStat stat{};
    stat.keyframe = keyframe;
    slam.TrackResult(frame_id, timestamp_ns, stat, Images{MakeImage(frame_id, timestamp_ns)}, Isometry3T::Identity());
    slam.ProcessInputSynchronously();
  }

  // Localizes in `folder` with the first frame's image; returns the error, empty on success.
  std::string LocalizeAt(AsyncSlam& slam, const std::string& folder, int64_t timestamp_ns, bool attach,
                         const Isometry3T& guess = Isometry3T::Identity()) {
    std::string error = "no answer";
    slam.LocalizeInMap(
        folder, timestamp_ns, guess, Images{MakeImage(1, timestamp_ns)}, {0.5f, 0.25f, 0.5f, 0.25f, 0.1f}, attach,
        nullptr, [&](const Result<Pose>& result) { error = result.data ? "" : std::string(result.error_message); });
    return error;
  }

  // Tracks the first frame and localizes there in `folder`.
  std::string Localize(AsyncSlam& slam, const std::string& folder, bool attach = false,
                       const Isometry3T& guess = Isometry3T::Identity()) {
    Track(slam, 1, 2000, false);
    return LocalizeAt(slam, folder, 2000, attach, guess);
  }

  static bool Flush(AsyncSlam& slam) {
    bool flushed = false;
    slam.FlushMap([&](bool success) { flushed = success; });
    return flushed;
  }

  size_t SavedKeyframes(const std::string& folder) const {
    LocalizerAndMapper reader(rig_, FeatureDescriptorType::kShiTomasi2, false);
    return reader.AttachToExistingReadOnlyDatabase(folder) ? reader.GetMap().GetPoseGraph().GetKeyframeCount() : 0;
  }

  static std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  }

  static bool Save(AsyncSlam& slam, const std::string& folder) {
    bool saved = false;
    slam.CopyToDatabase(folder, [&](bool success) { saved = success; });
    return saved;
  }

  static AsyncSlamOptions SessionOptions() {
    AsyncSlamOptions options;
    options.use_gpu = false;
    options.reproduce_mode = true;
    return options;
  }

  std::unique_ptr<camera::ICameraModel> camera_;
  camera::Rig rig_;
  std::filesystem::path folder_;
  std::vector<uint8_t> pixels_;
  sof::ImageContextPtr image_;
};

}  // namespace

TEST_F(LocalizeInMapLandmarks, LocalizedSessionUsesTheSavedLandmarks) {
  SaveMap(Folder("saved"));
  AsyncSlam slam(rig_, {0}, SessionOptions());
  const auto map_view = std::make_shared<ViewManager<ViewLandmarks>>();
  map_view->init(2, 1000);
  slam.SetLandmarksView(map_view);
  ASSERT_EQ(Localize(slam, Folder("saved")), "");

  // The next keyframe tries a loop closure and publishes the Map view; the frame after it reports the loop closure.
  Track(slam, 2, 3000, true);
  Track(slam, 3, 4000, false);

  AsyncSlamLCTelemetry telemetry;
  ASSERT_TRUE(slam.GetLastTelemetry(telemetry));
  EXPECT_EQ(telemetry.timestamp_ns, 3000u);
  EXPECT_EQ(telemetry.lc_selected_landmarks_count, kSavedLandmarks);

  const std::shared_ptr<ViewLandmarks> view = map_view->acquire_latest();
  ASSERT_NE(view, nullptr);
  std::set<uint64_t> ids;
  for (const Slam::Landmark& landmark : view->landmarks) {
    ids.insert(landmark.id);
  }
  std::set<uint64_t> saved_ids;
  for (uint64_t id = 0; id < kSavedLandmarks; ++id) {
    saved_ids.insert(id);
  }
  EXPECT_EQ(ids, saved_ids);
}

TEST_F(LocalizeInMapLandmarks, MapSavedAfterLocalizingHasTheSavedLandmarks) {
  SaveMap(Folder("saved"));
  AsyncSlam slam(rig_, {0}, SessionOptions());
  ASSERT_EQ(Localize(slam, Folder("saved")), "");

  ASSERT_TRUE(Save(slam, Folder("copy1")));
  EXPECT_EQ(CountSaved(Folder("copy1")), std::make_pair(kSavedLandmarks, kSavedLandmarks));
  // The session keeps its map, so a second save has it too.
  ASSERT_TRUE(Save(slam, Folder("copy2")));
  EXPECT_EQ(CountSaved(Folder("copy2")), std::make_pair(kSavedLandmarks, kSavedLandmarks));
  // Saving over a saved map replaces it.
  ASSERT_TRUE(Save(slam, Folder("copy1")));
  EXPECT_EQ(CountSaved(Folder("copy1")), std::make_pair(kSavedLandmarks, kSavedLandmarks));
}

TEST_F(LocalizeInMapLandmarks, SaveAfterStopHasTheLoadedMap) {
  SaveMap(Folder("saved"));
  AsyncSlam slam(rig_, {0}, SessionOptions());
  ASSERT_EQ(Localize(slam, Folder("saved")), "");
  slam.Stop();  // as the launcher does before its exit-time copy
  ASSERT_TRUE(Save(slam, Folder("copy")));
  EXPECT_EQ(CountSaved(Folder("copy")), std::make_pair(kSavedLandmarks, kSavedLandmarks));
}

TEST_F(LocalizeInMapLandmarks, SaveAfterStopStillRefusesTheMapFile) {
  AsyncSlamOptions options = SessionOptions();
  options.map_path = Folder("own");
  AsyncSlam slam(rig_, {0}, options);
  Track(slam, 1, 2000, true);
  slam.Stop();
  EXPECT_FALSE(Save(slam, Folder("copy")));
  EXPECT_FALSE(std::filesystem::exists(Folder("copy")));
}

TEST_F(LocalizeInMapLandmarks, SaveRefusesTheFoldersThatHoldTheLoadedMap) {
  SaveMap(Folder("saved"));
  AsyncSlam slam(rig_, {0}, SessionOptions());
  ASSERT_EQ(Localize(slam, Folder("saved")), "");

  EXPECT_FALSE(Save(slam, Folder("saved")));
  EXPECT_FALSE(Save(slam, Folder("saved") + "/"));
  EXPECT_FALSE(Save(slam, (folder_ / "saved" / "data.mdb").string()));
  EXPECT_FALSE(Save(slam, folder_.string()));
  EXPECT_EQ(CountSaved(Folder("saved")), std::make_pair(kSavedLandmarks, kSavedLandmarks));
}

TEST_F(LocalizeInMapLandmarks, MapViewOfALoadedMapKeepsNoLandmarksInMemory) {
  SaveMap(Folder("saved"));
  LocalizerAndMapper mapper(rig_, FeatureDescriptorType::kShiTomasi2, false);
  ASSERT_TRUE(mapper.AttachToExistingReadOnlyDatabase(Folder("saved")));
  const auto lsi = mapper.GetMap().GetLandmarksSpatialIndex();
  // In memory, staged, and in the database.
  ASSERT_EQ(lsi->LandmarksCount(), kSavedLandmarks);

  ViewLandmarks view(1000);
  PublishAllLandmarksToView(mapper.GetMap(), 0, view);
  EXPECT_EQ(view.landmarks.size(), kSavedLandmarks);
  EXPECT_EQ(lsi->LandmarksCount(), kSavedLandmarks);
}

TEST_F(LocalizeInMapLandmarks, MappingContinuesOnTheReadOnlyMap) {
  SaveMap(Folder("saved"));
  // The session's mapper after LocalizeInMap: the saved map, attached read-only.
  LocalizerAndMapper mapper(rig_, FeatureDescriptorType::kShiTomasi2, false);
  ASSERT_TRUE(mapper.AttachToExistingReadOnlyDatabase(Folder("saved")));
  mapper.SetActiveCameras({0});
  mapper.RebuildSpatialIndex();
  mapper.ReduceKeyframes();

  // A new landmark continues the saved ids; its descriptor stays in memory.
  Map& map = const_cast<Map&>(mapper.GetMap());
  std::vector<FeatureDescriptor> descriptors;
  map.feature_descriptor_ops_->CreateDescriptors(image_, {Vector2T(100.f, 100.f)}, descriptors);
  ASSERT_EQ(descriptors.size(), 1u);
  const LandmarkId id = map.landmarks_spatial_index_->AddLandmark(descriptors[0], 3000);
  EXPECT_EQ(id, kSavedLandmarks);
  const Vector2T uv_norm((100.f - 320) / 320, (100.f - 240) / 320);
  const Vector3T xyz(uv_norm.x() * 4.f, uv_norm.y() * 4.f, 4.f);
  ASSERT_TRUE(map.pose_graph_.AddLandmarkRelation(id, 1));
  map.landmarks_spatial_index_->AddLandmarkRelation(id, 1, &xyz, uv_norm, map.pose_graph_hypothesis_);
  map.landmarks_spatial_index_->MoveReadyStagedLandmarksToLSI(mapper.GetCurrentPose(), {0}, map.pose_graph_hypothesis_,
                                                              InvalidKeyFrameId, 3000);

  ASSERT_NE(mapper.GetMap().database_, nullptr);
  EXPECT_TRUE(mapper.GetMap().database_->IsReadOnly());
  EXPECT_EQ(Count(mapper), std::make_pair(kSavedLandmarks + 1, kSavedLandmarks + 1));
  EXPECT_GE(SelectedByLoopClosure(mapper), kSavedLandmarks);

  // SaveMap copies the loaded map and adds the new landmark.
  ASSERT_TRUE(mapper.AttachToNewDatabaseSaveMapAndDetach(Folder("copy")));
  EXPECT_EQ(CountSaved(Folder("copy")), std::make_pair(kSavedLandmarks + 1, kSavedLandmarks + 1));
  EXPECT_EQ(CountSaved(Folder("saved")), std::make_pair(kSavedLandmarks, kSavedLandmarks));
}

TEST_F(LocalizeInMapLandmarks, AttachedSessionWritesIntoTheLoadedMap) {
  SaveMap(Folder("saved"));
  AsyncSlam slam(rig_, {0}, SessionOptions());
  ASSERT_EQ(Localize(slam, Folder("saved"), /*attach=*/true), "");
  Track(slam, 2, 3000, true);

  EXPECT_FALSE(Save(slam, Folder("copy")));  // a map written in place is flushed, not saved
  ASSERT_TRUE(Flush(slam));
  // A snapshot of the flushed folder, while the session still writes into it.
  std::filesystem::copy(Folder("saved"), Folder("flushed"), std::filesystem::copy_options::recursive);
  EXPECT_GT(SavedKeyframes(Folder("flushed")), 2u);
  EXPECT_EQ(CountSaved(Folder("flushed")), std::make_pair(kSavedLandmarks, kSavedLandmarks));
}

TEST_F(LocalizeInMapLandmarks, EndingAnAttachedSessionWritesItsMap) {
  SaveMap(Folder("saved"));
  {
    AsyncSlam slam(rig_, {0}, SessionOptions());
    ASSERT_EQ(Localize(slam, Folder("saved"), /*attach=*/true), "");
    Track(slam, 2, 3000, true);
  }
  EXPECT_GT(SavedKeyframes(Folder("saved")), 2u);
  EXPECT_EQ(CountSaved(Folder("saved")), std::make_pair(kSavedLandmarks, kSavedLandmarks));
}

TEST_F(LocalizeInMapLandmarks, EndingASessionReportsAMapFileItFailedToWrite) {
  AsyncSlamOptions options = SessionOptions();
  options.map_path = Folder("own");
  auto slam = std::make_unique<AsyncSlam>(rig_, std::vector<CameraId>{0}, options);
  Track(*slam, 1, 2000, true);

  // LMDB writes its pages past the first 8 KiB, so every write of the map file fails.
  rlimit limit{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &limit), 0);
  const rlimit old_limit = limit;
  limit.rlim_cur = 8192;
  const auto old_handler = std::signal(SIGXFSZ, SIG_IGN);
  ::testing::internal::CaptureStdout();
  const bool limited = setrlimit(RLIMIT_FSIZE, &limit) == 0;
  slam.reset();
  setrlimit(RLIMIT_FSIZE, &old_limit);
  std::signal(SIGXFSZ, old_handler);
  const std::string output = ::testing::internal::GetCapturedStdout();
  if (!limited) {
    GTEST_SKIP() << "setrlimit(RLIMIT_FSIZE) was refused";
  }
  EXPECT_TRUE(Contains(output, "Failed to write the map file"));
}

TEST_F(LocalizeInMapLandmarks, EndingASessionThatWritesItsMapReportsNoFailure) {
  AsyncSlamOptions options = SessionOptions();
  options.map_path = Folder("own");
  auto slam = std::make_unique<AsyncSlam>(rig_, std::vector<CameraId>{0}, options);
  Track(*slam, 1, 2000, true);
  ::testing::internal::CaptureStdout();
  slam.reset();
  const std::string output = ::testing::internal::GetCapturedStdout();
  EXPECT_FALSE(Contains(output, "Failed to write the map file"));
  EXPECT_GE(SavedKeyframes(Folder("own")), 1u);
}

TEST_F(LocalizeInMapLandmarks, NotAttachedSessionLeavesTheLoadedMapAlone) {
  SaveMap(Folder("saved"));
  const std::string saved = ReadFile(folder_ / "saved" / "data.mdb");
  {
    AsyncSlam slam(rig_, {0}, SessionOptions());
    ASSERT_EQ(Localize(slam, Folder("saved")), "");
    Track(slam, 2, 3000, true);
    EXPECT_FALSE(Flush(slam));
  }
  EXPECT_EQ(ReadFile(folder_ / "saved" / "data.mdb"), saved);
}

TEST_F(LocalizeInMapLandmarks, RejectedAttachLeavesTheLoadedMapAlone) {
  SaveMap(Folder("saved"));
  const std::string saved = ReadFile(folder_ / "saved" / "data.mdb");
  {
    AsyncSlam slam(rig_, {0}, SessionOptions());
    Isometry3T far_away = Isometry3T::Identity();
    far_away.translation().x() = 100.f;
    EXPECT_TRUE(Contains(Localize(slam, Folder("saved"), /*attach=*/true, far_away), "Can't localize"));
    // This one fails after the folder was opened for writing: its timestamp is older than the retention window.
    Track(slam, 2, 60'000'000'000, true);
    EXPECT_TRUE(Contains(LocalizeAt(slam, Folder("saved"), 2000, /*attach=*/true), "retention window"));
    EXPECT_FALSE(Flush(slam));  // the session kept its own map
  }
  EXPECT_EQ(ReadFile(folder_ / "saved" / "data.mdb"), saved);
}

TEST_F(LocalizeInMapLandmarks, LocalizeInMapRefusesAMapBiggerThanMaxMapSize) {
  SaveMap(Folder("saved"));  // two keyframes
  const std::string saved = ReadFile(folder_ / "saved" / "data.mdb");
  AsyncSlamOptions options = SessionOptions();
  options.max_pose_graph_nodes = 1;
  for (const bool attach : {false, true}) {
    AsyncSlam slam(rig_, {0}, options);
    EXPECT_TRUE(Contains(Localize(slam, Folder("saved"), attach), "more than max_map_size")) << "attach " << attach;
    EXPECT_FALSE(Flush(slam));
  }
  EXPECT_EQ(ReadFile(folder_ / "saved" / "data.mdb"), saved);
}

TEST_F(LocalizeInMapLandmarks, ReduceKeyframesReturnsWhenNothingCanBeMerged) {
  SaveMap(Folder("saved"));  // two keyframes, one edge at the head
  LocalizerAndMapper mapper(rig_, FeatureDescriptorType::kShiTomasi2, false);
  ASSERT_TRUE(mapper.AttachToExistingReadOnlyDatabase(Folder("saved")));
  mapper.SetKeyframesLimit(1);
  mapper.ReduceKeyframes();
  EXPECT_EQ(mapper.GetMap().GetPoseGraph().GetKeyframeCount(), 2u);
}

TEST_F(LocalizeInMapLandmarks, LocalizeInMapRefusesTheMapFileTheSessionWrites) {
  SaveMap(Folder("saved"));
  AsyncSlamOptions options = SessionOptions();
  options.map_path = Folder("own");
  AsyncSlam slam(rig_, {0}, options);
  Track(slam, 1, 2000, true);
  ASSERT_TRUE(Flush(slam));
  EXPECT_TRUE(Contains(LocalizeAt(slam, Folder("own"), 2000, /*attach=*/false), "this session is writing"));
  EXPECT_TRUE(Contains(LocalizeAt(slam, Folder("own") + "/", 2000, /*attach=*/true), "this session is writing"));
  ASSERT_EQ(LocalizeAt(slam, Folder("saved"), 2000, /*attach=*/true), "");
  Track(slam, 2, 3000, true);
  EXPECT_TRUE(Contains(LocalizeAt(slam, Folder("saved"), 3000, /*attach=*/false), "this session is writing"));
  EXPECT_TRUE(Flush(slam));  // still attached to the loaded map
}

TEST_F(LocalizeInMapLandmarks, LocalizingElsewhereFirstWritesTheSessionsMapFile) {
  SaveMap(Folder("saved"));
  AsyncSlamOptions options = SessionOptions();
  options.map_path = Folder("own");
  {
    AsyncSlam slam(rig_, {0}, options);
    Track(slam, 1, 2000, true);
    Track(slam, 2, 3000, true);
    ASSERT_EQ(LocalizeAt(slam, Folder("saved"), 3000, /*attach=*/false), "");
  }
  // The session moved to the loaded map; its own map file holds what it mapped before.
  EXPECT_GE(SavedKeyframes(Folder("own")), 2u);
}

TEST_F(LocalizeInMapLandmarks, AttachNeedsAWritableMapFile) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "root writes read-only files";
  }
  SaveMap(Folder("saved"));
  const std::filesystem::path data = folder_ / "saved" / "data.mdb";
  std::filesystem::permissions(data, std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
  {
    AsyncSlam slam(rig_, {0}, SessionOptions());
    EXPECT_NE(Localize(slam, Folder("saved"), /*attach=*/true), "");
    EXPECT_FALSE(Flush(slam));  // the session kept its own map
  }
  std::filesystem::permissions(data, std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
}

}  // namespace test::slam

#endif  // USE_LMDB
