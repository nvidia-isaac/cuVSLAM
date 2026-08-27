
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

#include "camera/frustum_intersection_graph.h"
#include "camera_rig_edex/camera_rig_edex.h"
#include "common/include_gtest.h"
#include "common/rerun.h"
#include "odometry/svo_config.h"
#include "sof/image_manager.h"
#include "sof/sof_create.h"
#ifdef USE_RERUN
#include <tuple>

#include "visualizer/visualizer.hpp"
#endif

namespace test {
using namespace cuvslam;

namespace {

#ifdef USE_RERUN
// Logs the left and right frames side by side in a single 2D space, with one line per L2R match
// running from the primary observation to the secondary one (offset by the left image width).
void LogMatches(const std::string& name, const Sources& sources, const Metas& metas, const camera::Rig& rig,
                CameraId primary_id, const CameraId secondary_id, const sof::MulticamObservations& observations) {
  const size_t w = static_cast<size_t>(metas[primary_id].shape.width);
  const size_t h = static_cast<size_t>(metas[primary_id].shape.height);

  std::vector<uint8_t> composite(2 * w * h);
  const auto left = sources[primary_id].as<uint8_t>(metas[primary_id].shape);
  const auto right = sources[secondary_id].as<uint8_t>(metas[secondary_id].shape);
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      composite[y * 2 * w + x] = left(y, x);
      composite[y * 2 * w + w + x] = right(y, x);
    }
  }

  std::vector<rerun::LineStrip2D> strips;
  for (const camera::Observation& obs_l : observations[primary_id]) {
    for (const camera::Observation& obs_r : observations[secondary_id]) {
      if (obs_l.id != obs_r.id) {
        continue;
      }
      Vector2T uv_l;
      Vector2T uv_r;
      if (!rig.intrinsics[primary_id]->denormalizePoint(obs_l.xy, uv_l) ||
          !rig.intrinsics[secondary_id]->denormalizePoint(obs_r.xy, uv_r)) {
        break;
      }
      strips.push_back(rerun::LineStrip2D({{uv_l.x(), uv_l.y()}, {uv_r.x() + static_cast<float>(w), uv_r.y()}}));
      break;
    }
  }

  // Reuse the shared visualizer: it owns the spawned viewer and the blocking flush on shutdown.
  const rerun::RecordingStream& rec = visualizer::RerunVisualizer::getInstance().getRecordingStream();
  rec.log(name + "/image", rerun::Image(composite.data(), {static_cast<uint32_t>(2 * w), static_cast<uint32_t>(h)},
                                        rerun::datatypes::ColorModel::L));
  rec.log(name + "/image/matches", rerun::LineStrips2D(strips).with_colors(rerun::Color(0, 255, 0)).with_radii(0.5f));
  // The test process exits right after this, so push the data out before teardown.
  std::ignore = rec.flush_blocking();
}
#endif

// Lower bound on left-to-right matches for test_data/sof/lr_test frame 0 (640x400, 7.5 cm
// baseline, so the epipolar curves auto-detect the [0.1, 20] m depth range). Set below the
// measured count to absorb GPU/driver jitter while still catching a real L2R regression.
constexpr size_t kMinTrackedPoints = 175;

// Runs the single lr_test frame through the multicamera SOF and returns the number of
// left-to-right tracked points published for the secondary camera.
size_t TrackL2R(sof::Implementation implementation, [[maybe_unused]] const std::string& name) {
  const bool use_gpu = implementation == sof::Implementation::kGPU;

  camera_rig_edex::CameraRigEdex edex_rig(std::string(CUVSLAM_TEST_ASSETS) + "sof/lr_test/stereo.edex");
  if (const ErrorCode err = edex_rig.start(); !err) {
    ADD_FAILURE() << "CameraRigEdex::start failed: " << err.str();
    return 0;
  }

  camera::Rig rig;
  rig.num_cameras = static_cast<int32_t>(edex_rig.getCamerasNum());
  for (int32_t cam = 0; cam < rig.num_cameras; ++cam) {
    rig.intrinsics[cam] = &edex_rig.getIntrinsic(cam);
    rig.camera_from_rig[cam] = edex_rig.getExtrinsic(cam).inverse();
  }

  camera::FigSettings fig_settings;
  fig_settings.mode = camera::MulticameraMode::Performance;
  const camera::FrustumIntersectionGraph fig(rig, fig_settings);
  const CameraId primary_id = fig.primary_cameras().front();
  const CameraId secondary_id = fig.secondary_cameras(primary_id).front();

  const sof::Settings sof_settings;
  constexpr odom::KeyFrameSettings kf_settings;
  const std::unique_ptr<sof::IMultiSOF> multi_sof =
      CreateMultiSOF(implementation, rig, fig, nullptr, sof_settings, kf_settings);

  Sources sources;
  Sources masks;
  Metas metas;
  DepthSources depths;
  if (const ErrorCode err = edex_rig.getFrame(sources, metas, masks, depths); err != ErrorCode::S_True) {
    ADD_FAILURE() << "CameraRigEdex::getFrame failed: " << err.str();
    return 0;
  }

  sof::ImageManager image_manager;
  image_manager.init(metas[0].shape, sources.size(), use_gpu);
  sof::Images curr_images(sources.size(), nullptr);
  for (size_t cam = 0; cam < sources.size(); ++cam) {
    curr_images[cam] = image_manager.acquire();
    curr_images[cam]->set_image_meta(metas[cam]);
  }

  odom::TrackPerFrameSettings per_frame;
  per_frame.sof = sof_settings;
  per_frame.kf = kf_settings;

  // First frame is a keyframe: features are detected in the primary camera and the L2R scan runs.
  const sof::Images prev_images(sources.size(), nullptr);
  sof::MulticamObservations observations(rig.num_cameras);
  sof::FrameState state = sof::FrameState::None;
  multi_sof->trackNextFrame(sources, curr_images, prev_images, masks, Isometry3T::Identity(), observations, state,
                            per_frame);

  RERUN(LogMatches, name, sources, metas, rig, primary_id, secondary_id, observations);

  return observations[secondary_id].size();
}

}  // namespace

TEST(SOFL2R, TrackedPointsCPU) {
  const size_t tracked = TrackL2R(sof::Implementation::kCPU, "SOFL2R_CPU");
  std::cout << "CPU L2R tracked points: " << tracked << std::endl;
  EXPECT_GE(tracked, kMinTrackedPoints);
}

#ifdef USE_CUDA
TEST(SOFL2R, TrackedPointsGPU) {
  const size_t tracked = TrackL2R(sof::Implementation::kGPU, "SOFL2R_GPU");
  std::cout << "GPU L2R tracked points: " << tracked << std::endl;
  EXPECT_GE(tracked, kMinTrackedPoints);
}
#endif

}  // namespace test
