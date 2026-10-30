
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

#include "launcher/base_launcher.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "gflags/gflags.h"

#include "common/camera_id.h"
#include "common/image_dropper.h"
#include "common/include_json.h"
#include "common/isometry.h"
#include "common/log_types.h"
#include "common/rerun.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "edex/edex.h"
#include "epipolar/camera_projection.h"
#include "launcher/launch_load_and_localize.h"
#include "launcher/visualizer.h"
#include "odometry/ground_integrator.h"
#include "odometry/increment_pose.h"
#include "odometry/multi_visual_odometry.h"
#include "pipelines/visualizer.h"
#include "profiler/profiler.h"
#include "slam/async_slam/async_slam.h"

DEFINE_bool(bobber, false, "Bobber: orient the camera by gravity vector");
// DEFINE_int32(num_desired_tracks, 400, "Number of desired active tracks in left sof selector");
DEFINE_double(max_fps, 10000, "Make a delay between the next image reading from the disk");

DEFINE_int32(max_pose_graph_nodes, 300, "SLAM: limit of the node count in the pose graph");
DEFINE_uint64(throttling_time_ms, 0, "SLAM: minimum time interval between loop closures");
DEFINE_bool(slam_reproduce_mode, true, "slam reproduce mode: synced and nonrandom");
DEFINE_string(slam_lcs, "two_steps_easy",
              "[loop_closure_solver] Type of the loop closure solver = {dummy, simple, simple_point, two_steps_easy}");
DEFINE_double(slam_cell_size, 0, "SLAM: cell size of the LSI");
DEFINE_int32(slam_max_landmarks_in_cell, 100, "SLAM: limit of landmarks in the spatial index cell");
DEFINE_double(slam_max_landmarks_distance, 100., "SLAM: limit of distance from camera to landmark");
DEFINE_string(slam_copy_to_database, "",
              "SLAM: do copy to this database finally or on frame. See 'slam_copy_on_frame'. Fails with "
              "'slam_map_path', which is written in place");
DEFINE_int32(slam_copy_on_frame, -1, "SLAM: do copy to database on this frame");
DEFINE_bool(use_gpu, true, "Use GPU");
DEFINE_int32(start_frame, 0, "Start frame index in sequence");
DEFINE_int32(end_frame, -1, "End frame index in sequence");
DEFINE_int32(slam_load_and_localize_on_frame, -1, "Load and localize in the specified frame");
DEFINE_double(slam_load_and_localize_timestamp, -1, "Load and localize timestamp in seconds");
DEFINE_bool(slam_planar_constraints, false, "SLAM: planar constraints");
DEFINE_bool(enable_ground_integrator, false, "Enable postprocessing ground aligner");
DEFINE_bool(slam_disable_pgo, false, "SLAM: disable pose graph optimization");
DEFINE_string(slam_map_path, "",
              "SLAM: if empty (default), map is kept in memory only,"
              "else, map is synced to disk (LMDB) at this path");
DEFINE_bool(print_memory_usage, false, "Print memory usage on runtime");
DEFINE_bool(disable_full_dump, false, "Disable gather 2d and 3d observations for each frame");

DEFINE_double(image_drop_rate, 0.0, "Image drop rate");
DEFINE_string(image_drop_type, "steady", "Image dropping type: steady, normal, sticky");

namespace cuvslam::launcher {

namespace {

// rss_mb - resident memory
// cached_mb - RssFile (file-backed resident memory the kernel can reclaim).
bool GetProcessMemoryUsage(int& rss_mb, int& cached_mb) {
  std::ifstream f("/proc/self/status");
  if (!f) {
    return false;
  }
  long rss_kb = -1;
  long file_kb = -1;
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 6, "VmRSS:") == 0) {
      std::istringstream iss(line.substr(6));
      iss >> rss_kb;
    } else if (line.compare(0, 8, "RssFile:") == 0) {
      std::istringstream iss(line.substr(8));
      iss >> file_kb;
    }
  }
  if (rss_kb < 0 || file_kb < 0) {
    return false;
  }
  rss_mb = static_cast<int>(rss_kb / 1024);
  cached_mb = static_cast<int>(file_kb / 1024);
  return true;
}

}  // namespace

BaseLauncher::BaseLauncher(ICameraRig& cameraRig, const odom::Settings& svo_settings)
    : cameraRig_(cameraRig), svo_settings_(svo_settings), last_frame_start_(std::chrono::steady_clock::now()) {
  if (cameraRig_.start() != ErrorCode::S_True) {
    throw std::runtime_error("Failed to start camera rig");
  }

  // Get intrinsics of cameras
  rig.num_cameras = static_cast<int32_t>(cameraRig_.getCamerasNum());
  for (int32_t i = 0; i < rig.num_cameras; i++) {
    rig.camera_from_rig[i] = cameraRig_.getExtrinsic(i).inverse();
    rig.intrinsics[i] = &(cameraRig_.getIntrinsic(i));
  }
  RERUN(pipelines::logCameraPinhole, *rig.intrinsics[0], pipelines::kCamera0ImageEntity);
}

void BaseLauncher::SetupSlam() {
  slam::LoopClosureSolverType loop_closure_solver_type;
  if (std::string("dummy") == FLAGS_slam_lcs) {
    loop_closure_solver_type = slam::LoopClosureSolverType::kDummy;
  } else if (std::string("simple") == FLAGS_slam_lcs) {
    loop_closure_solver_type = slam::LoopClosureSolverType::kSimple;
  } else if (std::string("simple_point") == FLAGS_slam_lcs) {
    loop_closure_solver_type = slam::LoopClosureSolverType::kSimplePoint;
  } else if (std::string("two_steps_easy") == FLAGS_slam_lcs) {
    loop_closure_solver_type = slam::LoopClosureSolverType::kTwoStepsEasy;
  } else {
    throw std::runtime_error(FLAGS_slam_lcs + " is unsupported loop closure type");
  }

  slam::AsyncSlamOptions options;
  options.map_path = FLAGS_slam_map_path;
  options.use_gpu = FLAGS_use_gpu;
  options.reproduce_mode = FLAGS_slam_reproduce_mode;
  options.pose_for_frame_required = true;
  options.max_pose_graph_nodes = FLAGS_max_pose_graph_nodes;
  options.pgo_options.type =
      FLAGS_slam_disable_pgo ? slam::PoseGraphOptimizerType::Dummy : slam::PoseGraphOptimizerType::Simple;
  options.loop_closure_solver_type = loop_closure_solver_type;
  options.throttling_time_ms = FLAGS_throttling_time_ms;
  options.spatial_index_options.cell_size = FLAGS_slam_cell_size;
  options.spatial_index_options.max_landmarks_in_cell = FLAGS_slam_max_landmarks_in_cell;
  options.max_landmarks_distance = FLAGS_slam_max_landmarks_distance;
  options.planar_constraints = FLAGS_slam_planar_constraints;
  async_slam_ = std::make_unique<slam::AsyncSlam>(rig, slam_cameras_, options);
}

// track3dPosition in the world frame
bool BaseLauncher::getTrackTriangulationInFrame(const FrameId& frameId, const TrackId& trackId,
                                                Vector3T& track3dPosition) const {
  assert(triangulatedTracksPerFrame_.count(frameId) == 1);

  const std::vector<TrackId>& ids = triangulatedTracksPerFrame_.at(frameId);

  if (std::find(ids.begin(), ids.end(), trackId) == ids.end()) {
    // not found the item
    return false;
  }

  assert(finalTracks3d_.count(trackId) == 1);
  track3dPosition = finalTracks3d_.at(trackId);

  return true;
}

ErrorCode BaseLauncher::launch() {
  SetupTracker(svo_settings_, FLAGS_use_gpu);
  {
    // feed forward for FLAGS_start_frame times
    if (FLAGS_start_frame != 0) {
      Sources curr_sources;
      Sources masks_sources;
      Metas curr_meta;
      DepthSources depth_sources;
      for (int i = 0; i < FLAGS_start_frame; i++) {
        if (cameraRig_.getFrame(curr_sources, curr_meta, masks_sources, depth_sources) != ErrorCode::S_True) {
          TraceError("Failed to feed forward to start_frame %d\n", FLAGS_start_frame);
          return ErrorCode::E_Failure;
        }
      }
    }
  }

  Isometry3T prev_abs_world_from_rig = Isometry3T::Identity();
  odom::GroundIntegrator ground_integrator(Isometry3T::Identity(), Isometry3T::Identity(), Isometry3T::Identity());

  std::random_device dev;
  auto drop = CreateImageDropper(ParseImageDropperType(FLAGS_image_drop_type), std::mt19937{dev()});

  while (true) {
    {  // fps throttling
      TRACE_EVENT ev_sleep = profiler_domain_.trace_event("sleep", 0x70FF70);
      const auto min_duration = std::chrono::duration<double, std::ratio<1>>(1.0 / FLAGS_max_fps);
      const auto now = std::chrono::steady_clock::now();
      const auto elapsed = now - last_frame_start_;

      if (elapsed < min_duration) {
        std::this_thread::sleep_for(min_duration - elapsed);
      }
      last_frame_start_ = std::chrono::steady_clock::now();
    }
    TRACE_EVENT ev = profiler_domain_.trace_event("frame", profiler_color_);

    {
      log::Scoped<LogFrames> frame("frames");

      if (prev_image_ptrs.size() != static_cast<size_t>(rig.num_cameras)) {
        prev_image_ptrs.assign(rig.num_cameras, nullptr);
      }
      for (CameraId cam_id = 0; cam_id < curr_image_ptrs.size(); ++cam_id) {
        if (curr_image_ptrs[cam_id] != nullptr) {
          prev_image_ptrs[cam_id] = curr_image_ptrs[cam_id];
        }
      }
      // prev_image_ptrs = curr_image_ptrs;
      curr_image_ptrs.assign(rig.num_cameras, nullptr);

      const ErrorCode err = cameraRig_.getFrame(curr_sources, curr_meta, masks_sources, depth_sources);
      if (err != ErrorCode::S_True) {
        if (err != ErrorCode::E_Bounds) {
          TraceError("getFrame fails with %d error code", err);
          return ErrorCode::E_FileNotFound;
        } else {
          // successfully reached last frame
        }
        break;
      }

      if (!image_manager_.is_initialized()) {
        // Pool size tracks the launcher-resolved depth-id list (isDepthCamera), not the rig's
        // raw advertisement, so cameras the user excluded via -fig_depth_camera_ids do not
        // pre-allocate depth slots they will never use.
        size_t num_depth_images = 0;
        for (int32_t cam_id = 0; cam_id < rig.num_cameras; ++cam_id) {
          if (isDepthCamera(static_cast<CameraId>(cam_id))) {
            ++num_depth_images;
          }
        }
        image_manager_.init(curr_meta[0].shape, (rig.num_cameras - num_depth_images) * 4, FLAGS_use_gpu,
                            num_depth_images * 4);
      }

      // Drop depth entries the launcher is not configured to use, so downstream odometry
      // (e.g. MultisensorOdometry::track) never sees depth from cameras the user excluded.
      for (CameraId cam_id = 0; cam_id < depth_sources.size(); ++cam_id) {
        if (!isDepthCamera(cam_id)) {
          depth_sources[cam_id] = {};
        }
      }

      for (CameraId camera_id = 0; camera_id < curr_sources.size(); ++camera_id) {
        if (curr_sources[camera_id].data == nullptr) {
          continue;
        }
        sof::ImageContextPtr ptr;

        if (depth_sources[camera_id].data != nullptr) {
          ptr = image_manager_.acquire_with_depth();
        } else {
          ptr = image_manager_.acquire();
        }

        if (ptr == nullptr) {
          TraceError("ImageManager::acquire returned nullptr");
          return ErrorCode::E_Pointer;
        }
        ptr->set_image_meta(curr_meta[camera_id]);
        curr_image_ptrs[camera_id] = ptr;
      }

      {
        Sources sources(rig.num_cameras);
        sof::Images image_ptrs(rig.num_cameras, nullptr);

        auto dropped_cams = drop->GetDroppedImages(FLAGS_image_drop_rate, rig.num_cameras);

        for (CameraId cam_id = 0; cam_id < curr_sources.size(); ++cam_id) {
          if (curr_sources[cam_id].data == nullptr) {
            continue;
          }
          if (dropped_cams.find(cam_id) == dropped_cams.end()) {
            sources[cam_id] = curr_sources[cam_id];
            image_ptrs[cam_id] = curr_image_ptrs[cam_id];
          }
        }

        curr_sources = sources;
        curr_image_ptrs = image_ptrs;
      }

      {
        slam_images.assign(rig.num_cameras, nullptr);
        for (CameraId cam_id : slam_cameras_) {
          if (cam_id < curr_image_ptrs.size() && curr_image_ptrs[cam_id] != nullptr) {
            slam_images[cam_id] = curr_image_ptrs[cam_id];
          }
        }
      }

      const FrameId frameId = curr_meta[0].frame_id;
      if (FLAGS_print_memory_usage && frameId % 1000 == 0) {
        int rss_mb = -1, cached_mb = -1;
        if (GetProcessMemoryUsage(rss_mb, cached_mb)) {
          TraceMessage("frame %d, resident %d mb, file_cache %d mb. Used (resident-file_cache) = %d mb", frameId,
                       rss_mb, cached_mb, rss_mb - cached_mb);
        }
      }
      const int64_t timestamp_ns = curr_meta[0].timestamp;
      // if have end_frame
      if (FLAGS_end_frame >= 0 && static_cast<int>(frameId) > FLAGS_end_frame) {
        break;
      }
      Matrix6T pose_info;
      const odom::IVisualOdometry::VOFrameStat& stat = last_vo_stat();

      RERUN(setupFrameTimeline, curr_meta);
      // log camera 0 images
      RERUN(logCameraImages, curr_meta, curr_sources, {0}, {pipelines::kCamera0ImageEntity});
      // log depth image as separate 2D view
      if (!depth_sources.empty() && depth_sources[0].data != nullptr) {
        RERUN(logDepthImage, depth_sources[0], curr_meta[0], pipelines::kCamera0DepthEntity, 1.0f, 0.0f, 5.0f);
      }
      if (gt_poses_.size() > static_cast<size_t>(frameId)) {
        RERUN(pipelines::logTrajectory, gt_poses_[frameId].inverse(), "world/trajectories/gt_trajectory",
              Color(0, 0, 255), TrajectoryType::GT);
      }

      TRACE_EVENT ev_track = profiler_domain_.trace_event("tracker->track()", profiler_color_);
      StopwatchScope sws_track_proc(sw_track_proc_);

      Isometry3T delta;  // doesn't need to be initialized
      const bool solutionFound = launch_vo(delta, pose_info, per_frame_settings_);
      if (!solutionFound) {
        TraceError("Failed to solveNextFrame in frame %d. Set pose delta to identity.", frameId);
        delta = Isometry3T::Identity();
      }

      prev_abs_world_from_rig = odom::increment_pose(prev_abs_world_from_rig, delta);
      RERUN(pipelines::logCameraPose, prev_abs_world_from_rig * rig.camera_from_rig[0].inverse(),
            pipelines::kCamera0Entity);
      Isometry3T abs_world_from_rig = prev_abs_world_from_rig;
      if (FLAGS_enable_ground_integrator) {
        ground_integrator.AddNextPose(abs_world_from_rig);
        abs_world_from_rig = ground_integrator.GetPoseOnGround();
      }
      log::Value<LogFrames>("vo_pose", abs_world_from_rig);

      if (!solutionFound) {
        if (tracking_lost_cb_) {
          tracking_lost_cb_();
        }
      }
      ev_track.Pop();
      double track_proc_duration = sws_track_proc.Stop();
      if (track_proc_duration > max_track_proc_duration_) {
        max_track_proc_duration_ = track_proc_duration;
        max_track_proc_frame_ = frameId;
      }
      assert(abs_world_from_rig.matrix().allFinite());

      if (async_slam_) {
        async_slam_->TrackResult(frameId, timestamp_ns, stat, slam_images, delta);
        abs_world_from_rig = async_slam_->GetSlamPose();
      }

      if (stat.keyframe) {
        finalKeyframes_.insert(frameId);
      }

      //
      // Store tracking result
      //
      if (!FLAGS_disable_full_dump) {
        tracks2D_[frameId] = stat.tracks2d;  // copy array
        assert(abs_world_from_rig.matrix().allFinite());
        cameraSolved_[frameId] = abs_world_from_rig;
        // update changed 3d position and add new ones
        {
          const size_t nTriangulatedTracksInFrame = stat.tracks3d.size();
          size_t i = 0;

          triangulatedTracksPerFrame_[frameId].resize(nTriangulatedTracksInFrame);

          for (const auto& t : stat.tracks3d) {
            const TrackId trackId = t.first;
            const Vector3T& position_in_rig = t.second;
            const Vector3T position_in_world = abs_world_from_rig * position_in_rig;
            finalTracks3d_[trackId] = position_in_world;
            triangulatedTracksPerFrame_[frameId][i++] = trackId;
          }
          assert(i == nTriangulatedTracksInFrame);
        }

        //
        // Calculate runtime statistics (tracksInFrame and frameResidual).
        //
        computeStatistics(abs_world_from_rig, frameId, stat.tracks3d);
      }
    }

    if (async_slam_) {
      async_slam_->ProcessInputSynchronously();
      const std::list<slam::LoopClosureStamped>& last_LCs_stamped = async_slam_->GetLastLoopClosuresStamped();
      for (const slam::LoopClosureStamped& lc_stamped : last_LCs_stamped) {
        if (loop_closure_map_.find(lc_stamped.frame_id) == loop_closure_map_.end()) {
          loop_closure_map_[lc_stamped.frame_id] = lc_stamped.pose;
        }
      }
    }

    // copy current state to database
    if (!FLAGS_slam_copy_to_database.empty()) {
      // on this frame?
      if (FLAGS_slam_copy_on_frame >= 0 && FLAGS_slam_copy_on_frame == curr_meta[0].frame_number) {
        async_slam_->CopyToDatabase(FLAGS_slam_copy_to_database);
      }
    }

    const CameraId default_camera = 0;
    if (default_camera < slam_images.size() && slam_images[default_camera] != nullptr &&
        default_camera < curr_meta.size() &&
        FLAGS_slam_load_and_localize_on_frame == curr_meta[default_camera].frame_number) {
      constexpr int64_t ns_from_second = 1000000000;  // 1.000.000.000
      const int64_t load_map_timestamp_ns =
          FLAGS_slam_load_and_localize_timestamp > 0
              ? static_cast<int64_t>(ns_from_second * FLAGS_slam_load_and_localize_timestamp)
              : slam_images[default_camera]->get_image_meta().timestamp;
      LaunchLoadMapAndLocalize(*async_slam_, load_map_timestamp_ns, slam_images, FLAGS_use_gpu);
    }
  }

  cameraRig_.stop();

  if (async_slam_) {
    async_slam_->Stop();
  }

  if (async_slam_) {
    // copy current state to database
    if (!FLAGS_slam_copy_to_database.empty()) {
      // on exit
      if (FLAGS_slam_copy_on_frame < 0) {
        async_slam_->CopyToDatabase(FLAGS_slam_copy_to_database);
      }
    }
  }

  // Post-processing
  if (async_slam_) {
    for (auto& it : cameraSolved_) {
      Isometry3T pose;
      if (!async_slam_->GetPoseForFrame(it.first, pose)) {
        TraceError("GetPoseForFrame in frame %zd\n", it.first);
      }

      it.second = pose;
    }

    // reset batch residual
    for (auto& it : tracks2D_) {
      it.second.clear();
    }
  }
  return ErrorCode::S_True;
}

void BaseLauncher::computeStatistics(const Isometry3T& /* cameraExtrinsics */, const FrameId& frameId,
                                     const Tracks3DMap& tracks3d) {
  size_t tracksInFrame = 0;
  float totalFrameResidual = 0;

  const Isometry3T& isometry = Isometry3T::Identity();  // cameraExtrinsics.inverse();  // toLocalSpace

  for (const auto& track2d : tracks2D_[frameId]) {
    const camera::ICameraModel& camera = cameraRig_.getIntrinsic(track2d.cam_id);
    const TrackId& trackId = track2d.track_id;  // trackId from 2dtracks
    const Vector2T& uv = track2d.uv;            // in pixels
    Vector2T xy;
    if (!camera.normalizePoint(uv, xy)) {
      continue;
    }
    const bool trackIsTriangulated = tracks3d.count(trackId) == 1;

    if (!trackIsTriangulated) {
      continue;  // this trackId (from current frame) has no triangulation in this frame
    }

    const Vector3T& v3 = tracks3d.at(trackId);
    Vector2T xy_reprojected;
    epipolar::Project3DPointInLocalCoordinates(isometry, v3, xy_reprojected);
    const float trackResidual = (xy - xy_reprojected).stableNorm();
    tracksInFrame++;
    totalFrameResidual += trackResidual;
  }

  assert(tracksInFrame > 0 || (tracksInFrame == 0 && totalFrameResidual == 0));
  statistic_[frameId] = {totalFrameResidual, tracksInFrame};
}

void BaseLauncher::convertTracksContainer(const Tracks2DFrameMap& mapContainer, Tracks2DVectorsMap& vecPairContainer) {
  for (const auto& pairFrameMap : mapContainer) {
    const FrameId& frameId = pairFrameMap.first;
    const Tracks2DMap& tracks = pairFrameMap.second;
    const size_t nTracks = tracks.size();

    vecPairContainer[frameId].resize(nTracks);

    size_t i = 0;

    for (const auto& pairTrackData : mapContainer.at(frameId)) {
      const TrackId& trackId = pairTrackData.first;
      const Vector2T& vec2d = pairTrackData.second;

      vecPairContainer[frameId][i] = TrackIdVector2TPair(trackId, vec2d);
      ++i;
    }

    assert(i == nTracks);
  }
}

void BaseLauncher::calcVisible2DTracksStats(float& averageVisible2DTracks, size_t& minVisible2DTracks,
                                            FrameId& minVisible2DTracksFrame) const {
  minVisible2DTracks = std::numeric_limits<size_t>::max();
  size_t totalVisible2DTracks = 0u;

  for (const auto& kv : tracks2D_) {
    const auto& frameId = kv.first;
    const auto& tracks2D = kv.second;

    const size_t nVisibleTracks = tracks2D.size();

    if (nVisibleTracks < minVisible2DTracks) {
      minVisible2DTracksFrame = frameId;
      minVisible2DTracks = nVisibleTracks;
    }

    totalVisible2DTracks += nVisibleTracks;
  }

  averageVisible2DTracks = !tracks2D_.empty() ? static_cast<float>(totalVisible2DTracks) / tracks2D_.size() : 0;
}

float BaseLauncher::calcBatchResidual() const {
  float totalResidual = 0;
  size_t totalTracks = 0;
  for (const auto& c : cameraSolved_) {
    const FrameId& frameId = c.first;

    const Isometry3T& isometry = c.second.inverse();  // toLocalSpace
    const auto& tracks2d = tracks2D_.at(frameId);

    for (const auto& track2d : tracks2d) {
      const TrackId& trackId = track2d.track_id;
      const Vector2T& uv = track2d.uv;  // in pixels
      Vector2T xy;

      const camera::ICameraModel& camera = cameraRig_.getIntrinsic(track2d.cam_id);
      if (!camera.normalizePoint(uv, xy)) {
        continue;
      }

      Vector3T v3;  // track position on the world frame
      if (!getTrackTriangulationInFrame(frameId, trackId, v3)) {
        continue;  // this trackId (from current frame) has no triangulation in this frame
      }

      Vector2T xy_reprojected;
      epipolar::Project3DPointInLocalCoordinates(isometry, v3, xy_reprojected);
      const float trackResidual = (xy - xy_reprojected).stableNorm();
      totalResidual += trackResidual;
      totalTracks++;
    }
  }

  if (totalTracks > 0) {
    totalResidual /= totalTracks;
  }

  return totalResidual;
}

// return average and max elapsed time for tracker.track() procedure
void BaseLauncher::getTimers(float& average_track_sec, float& max_track_sec, FrameId& max_track_frame) const {
  average_track_sec = static_cast<float>(sw_track_proc_.Seconds() / sw_track_proc_.Times());
  max_track_sec = static_cast<float>(max_track_proc_duration_);
  max_track_frame = max_track_proc_frame_;
}

// [OUT] averageOnlineResidual, maxFrameResidual, maxFrameResidualFrame - runtime statistics,
// which calculates when new frame arrives.
// Use calcBatchResidual method to get final (AKA batch) residual,
// which is smaller, because we refine 3d points each frame.

void BaseLauncher::calcOnlineResidual(float& averageOnlineResidual, float& maxFrameResidual,
                                      FrameId& maxFrameResidualFrame) const {
  averageOnlineResidual = 0;
  maxFrameResidual = 0;
  maxFrameResidualFrame = 0;

  float totalOnlineResidual;
  size_t totalTracks;
  calcTotalOnlineResidual(totalOnlineResidual, totalTracks);

  for (const auto& kv : statistic_) {
    const FrameId& frameId = kv.first;
    const FrameStat& fs = kv.second;

    if (fs.nTriangulatedTracks == 0) {
      continue;
    }

    const float residual = fs.totalFrameResidual / fs.nTriangulatedTracks;

    if (residual > maxFrameResidual) {
      maxFrameResidual = residual;
      maxFrameResidualFrame = frameId;
    }
  }

  if (totalTracks > 0) {
    averageOnlineResidual = totalOnlineResidual / totalTracks;
  }
}

void BaseLauncher::calcTotalOnlineResidual(float& totalOnlineResidual, size_t& totalTriangulatedTracks) const {
  totalOnlineResidual = 0;
  totalTriangulatedTracks = 0;

  for (const auto& kv : statistic_) {
    const FrameStat& fs = kv.second;

    totalOnlineResidual += fs.totalFrameResidual;
    totalTriangulatedTracks += fs.nTriangulatedTracks;
  }
}

size_t BaseLauncher::nFrames() const {
  assert(tracks2D_.size() == cameraSolved_.size());
  return cameraSolved_.size();
}

const CameraMap& BaseLauncher::cameraMap() const { return cameraSolved_; }

const CameraMap& BaseLauncher::LoopClosureMap() const { return loop_closure_map_; }

// Keep only 2dtracks visible >=visibleInNKeyFrames keyframe.

void BaseLauncher::filter2DTracksByKeyFrameVisibility(const std::unordered_map<FrameId, std::vector<Track2D>>& tracks2D,
                                                      const FrameSet& keyFrames, size_t visibleInNKeyFrames,
                                                      Tracks2DVectorsMap& track2DFiltered) {
  // per each trackID calculate number of keyFrames
  std::map<TrackId, size_t> nKeyFramesPerTrackId;

  for (const FrameId keyFrameId : keyFrames) {
    if (tracks2D.find(keyFrameId) == tracks2D.end()) {
      TraceError("Missing 2D track for frame %zu", static_cast<size_t>(keyFrameId));
      continue;
    }

    for (const auto& pair : tracks2D.at(keyFrameId)) {
      const TrackId trackId = pair.track_id;

      // create new counter for trackId or increase it
      assert(nKeyFramesPerTrackId.count(trackId) <= 1);

      if (nKeyFramesPerTrackId.count(trackId) != 0) {
        nKeyFramesPerTrackId[trackId] += 1;
      } else {
        nKeyFramesPerTrackId[trackId] = 1;
      }
    }
  }

  // do filter, storing result in map
  Tracks2DFrameMap filteredMap;

  for (const auto& p : tracks2D) {
    const FrameId& frameId = p.first;

    for (const auto& pair : p.second) {
      const TrackId& trackId = pair.track_id;

      if (nKeyFramesPerTrackId[trackId] >= visibleInNKeyFrames) {
        filteredMap[frameId][trackId] = pair.uv;
      }
    }
  }

  // convert map to array of pair
  convertTracksContainer(filteredMap, track2DFiltered);
}

// Save all data to precomputed edex file. (sequence field will be empty)
// if filter2dTracks then keep only 2dtracks visible >=2 keyframe.

bool BaseLauncher::saveResultToEdex(const std::string& outEdexName, const camera::ICameraModel& intrinsics,
                                    bool filter2dTracks, const edex::RotationStyle rs) const {
  edex::EdexFile f(rs);

  f.cameras_.resize(1);
  edex::Camera& cam = f.cameras_[0];
  cam.intrinsics.resolution = intrinsics.getResolution();
  cam.intrinsics.focal = intrinsics.getFocal();
  cam.intrinsics.principal = intrinsics.getPrincipal();
  cam.intrinsics.distortion_model = "pinhole";
  cam.intrinsics.distortion_params.clear();

  f.tracks3D_ = finalTracks3d_;  // coordinates in the world space

  assert(f.rigPositions_.size() == 0 && f.loop_closure_positions_.size() == 0);
  f.rigPositions_ = cameraSolved_;
  f.loop_closure_positions_ = loop_closure_map_;

  // fix frame range from camera motion
  if (cameraSolved_.size() != 0) {
    f.timeline_.set(cameraSolved_.begin()->first, cameraSolved_.rbegin()->first);
  } else {
    f.timeline_.set(0, 0);
  }

  f.keyFrames_ = finalKeyframes_;
  f.failedFrames_ = failedFrames_;

  if (filter2dTracks) {
    if (f.keyFrames_.size() == 0) {
      TraceError("You want to filter 2dtrack by keyframes. But no keyframes stored. Track2d wouldn't be stored.");
    }

    filter2DTracksByKeyFrameVisibility(tracks2D_, f.keyFrames_, 2, cam.tracks2D);
  } else {
    cam.tracks2D.clear();
    for (const auto& [frame_id, x] : tracks2D_) {
      auto& vec = cam.tracks2D[frame_id];

      for (const auto& track : x) {
        vec.emplace_back(track.track_id, track.uv);
      }
    }
  }

  return f.write(outEdexName);
}

[[maybe_unused]] bool BaseLauncher::saveResultToComposedRTJson(const std::string& outComposedRTJsonFileName) const {
  std::ofstream f(outComposedRTJsonFileName, std::ifstream::binary);

  if (!f.is_open()) {
    return false;
  }

  Json::Value matrices;
  Json::Value::ArrayIndex ind = 0;

  for (const auto& c : cameraSolved_) {
    const Isometry3T& isometry = c.second;

    matrices[ind]["id"] = static_cast<int>(c.first);

    for (Json::Value::ArrayIndex i = 0; i < 4; i++) {
      for (Json::Value::ArrayIndex j = 0; j < 4; j++) {
        matrices[ind]["m"][i][j] = isometry(i, j);
      }
    }

    ++ind;
  }

  std::string jsonStr = writeString(Json::StreamWriterBuilder(), matrices);
  f << jsonStr;

  return f.good();
}

void BaseLauncher::registerTrackingLostCB(const std::function<void()>& cb) { tracking_lost_cb_ = cb; }

void BaseLauncher::updateGTPoses(const Isometry3TVector& poses) {
  gt_poses_.clear();
  gt_poses_.reserve(poses.size());
  for (const auto& pose : poses) {
    gt_poses_.push_back(pose);
  }
}

}  // namespace cuvslam::launcher
