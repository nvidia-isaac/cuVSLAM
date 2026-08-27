
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

#include "pipelines/internal/mono/sba_mono_wrapper.h"

#include "epipolar/camera_selection.h"  // for IsDepthInFront
#include "sba/mono_sba_solver.h"

namespace cuvslam::pipelines {

static bool calcTrackResidualInFrame(const Isometry3T& inverse_camera, const storage::Vec2<float> observation,
                                     const Track& outTrack, float* p_out_residual) {
  assert(p_out_residual != NULL);

  if (!outTrack.hasLocation()) {
    return false;
  }

  const Vector3T v3 = inverse_camera * outTrack.getLocation3D();
  const float z = v3.z();

  if (!epipolar::IsDepthInFront(z)) {
    return false;
  }
  const Vector2T proj = v3.head(2) / z;

  *p_out_residual = (observation - proj).norm();
  return true;
}

static float CalcWinsorizationTreshold(const std::vector<camera::Observation>& observations,
                                       const Isometry3T& inverseCamera, const std::map<TrackId, Track>& tracks) {
  const size_t n_observations = observations.size();

  std::vector<float> residual_per_track;
  residual_per_track.resize(n_observations);

  // calculate residual per observation
  size_t n_healthy_tracks = 0;
  for (size_t i = 0; i < n_observations; ++i) {
    const camera::Observation& observation = observations[i];
    const Track& outTrack = tracks.at(observation.id);

    float residual;
    if (calcTrackResidualInFrame(inverseCamera, observation.xy, outTrack, &residual)) {
      residual_per_track[i] = residual;
      ++n_healthy_tracks;
    } else {
      residual_per_track[i] = std::numeric_limits<float>::max();
    }
  }
  if (n_healthy_tracks == 0) {
    return std::numeric_limits<float>::max();  // do not delete any points
  }
  std::sort(residual_per_track.begin(), residual_per_track.end());
  const float residual_to_have_n = residual_per_track[std::min(70, (int)n_observations - 1)];
  const float residual_to_cut_proc = residual_per_track[static_cast<int>(n_healthy_tracks * 0.95f)];  // remove 5%
  return std::max(std::max(residual_to_have_n, residual_to_cut_proc), 0.00001f);
}

void DoWinsorization(TrackingMap& map, size_t start_frame, size_t n_constained_frames,
                     const std::vector<FrameId>& in_keyframes) {
  const size_t n_frames = map.tracking_states.size();

  // front() entry corresponds to the last keyframe (frame wise)
  float rejectionThreshold;
  {
    const FrameId last_frame_id = in_keyframes[in_keyframes.size() - 1];
    const Isometry3T& inverseCamera = map.tracking_states.at(last_frame_id).world_from_rig.inverse();

    rejectionThreshold = CalcWinsorizationTreshold(map.observations[0].at(last_frame_id), inverseCamera, map.tracks);
  }

  // next we will disable triangulation of those tracks that have reprojection error beyond limit
  for (size_t i = n_constained_frames; i < n_frames; ++i) {
    const FrameId frame_id = in_keyframes[start_frame + i];

    const Isometry3T& inverseCamera = map.tracking_states.at(frame_id).world_from_rig.inverse();
    const std::vector<camera::Observation>& obs = map.observations[0].at(frame_id);
    for (size_t j = 0; j < obs.size(); ++j) {
      const camera::Observation& t = obs[j];
      Track& outTrack = map.tracks.at(t.id);

      float residual;
      if (!calcTrackResidualInFrame(inverseCamera, t.xy, outTrack, &residual) || residual > rejectionThreshold) {
        outTrack.disableTriangulation();
      }
    }
  }
}

void sparseBA(TrackingMap& map, int max_iter, bool do_winsorization) {
  const size_t n_max_frames = 10;
  size_t n_constained_frames = 3;

  const size_t n_in_keyframes = map.tracking_states.size();
  const size_t n_frames = std::min(n_in_keyframes, n_max_frames);  // number of frames in SBA

  assert(n_in_keyframes >= n_frames);
  const size_t start_frame = n_in_keyframes - n_frames;

  assert(n_frames >= 2);
  if (n_frames <= n_constained_frames) {
    --n_constained_frames;
  }

  std::vector<FrameId> in_keyframes;
  {
    in_keyframes.resize(n_in_keyframes);

    size_t i = 0;
    for (const auto& t : map.tracking_states) {
      in_keyframes[i] = t.first;
      ++i;
    }
    assert(i == n_in_keyframes);
  }

  std::map<TrackId, bool> filtered;
  {
    // gather all trackid from observation
    for (size_t i = 0; i < n_frames; ++i) {
      const FrameId frame = in_keyframes[start_frame + i];

      const std::vector<cuvslam::camera::Observation>& observations = map.observations[0].at(frame);

      for (const auto& t : observations) {
        filtered[t.id] = false;  // create new
      }
    }
    // gather all trackid from 3d positions
    for (const auto& t : map.tracks) {
      filtered[t.first] = false;  // create new
    }

    // filter all non triangulated tracks
    for (auto& t : filtered) {
      const TrackId id = t.first;

      if (!map.tracks.at(id).hasLocation()) {
        filtered.at(id) = true;
      }
    }

    // filter all tracks not visible in unconstained
    for (const auto& t : filtered) {
      const TrackId id = t.first;

      size_t i;
      for (i = n_constained_frames; i < n_frames; ++i) {
        const FrameId frame = in_keyframes[start_frame + i];

        if (FindObservation(map.observations[0].at(frame), id)) {
          break;  // found
        }
      }
      if (i == n_frames) {
        filtered.at(id) = true;
      }
    }

    // filter all tracks not visible at least in two frames
    for (auto& t : filtered) {
      const TrackId id = t.first;

      size_t n = 0;
      for (size_t i = 0; i < n_frames; ++i) {
        const FrameId frame = in_keyframes[start_frame + i];
        if (FindObservation(map.observations[0].at(frame), id)) {
          ++n;  // found
        }
      }
      if (n < 2) {
        filtered[id] = true;
      }
    }
  }

  sba::MonoSBASolver solver(n_frames, filtered.size(), max_iter, 0.02f, 0.1f);

  /* setup solver */
  // add 3d tracks
  for (const auto& t : map.tracks) {
    const TrackId id = t.first;
    const Track& track = t.second;

    if (!filtered.at(id)) {
      const Vector3T position = track.getLocation3D();

      solver.addTrack(id, position);
    }
  }

  for (size_t i = 0; i < n_frames; ++i) {
    const FrameId frame = in_keyframes[start_frame + i];
    const bool constrained = i < n_constained_frames;
    solver.addFrame(frame, constrained, map.tracking_states.at(frame).world_from_rig, filtered,
                    map.observations[0].at(frame));
  }

  /* run solver */
  int n_iterations = solver.solve();
  if (n_iterations == 0) {
    // TracePrint("SBA failed with no iterations\n");
  }

  if (solver.getFailed()) {
    // TracePrint("SBA failed with %d iterations. Residual is %f \n", n_iterations, solver.getResidual());
  }

  /* export results */
  // update cameras
  {
    const CameraMap optimized_cameras = solver.getCameras();

    for (size_t i = n_constained_frames; i < n_frames; ++i) {
      const FrameId frame = in_keyframes[start_frame + i];
      map.tracking_states.at(frame).world_from_rig = optimized_cameras.at(frame);
    }
  }

  // update tracks
  {
    const std::vector<sba::Track>& sba_tracks = solver.tracks();

    for (const sba::Track& sba_track : sba_tracks) {
      Track& out_track = map.tracks.at(sba_track.id);

      if (!sba_track.removed) {
        assert(out_track.hasLocation());
        out_track.setLocation3D(sba_track.b, TrackState::kOptimized);
      } else {
        out_track.disableTriangulation();
      }
    }
  }
  if (do_winsorization) {
    DoWinsorization(map, start_frame, n_constained_frames, in_keyframes);
  }
}

}  // namespace cuvslam::pipelines
