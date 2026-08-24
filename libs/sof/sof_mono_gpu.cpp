
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

#include "sof/internal/sof_mono_gpu.h"

#include "common/log_types.h"
#include "epipolar/fundamental_ransac.h"

namespace cuvslam::sof {

// threshold to determine that tracking have failed
const size_t FAILED_ACTIVE_TRACK_COUNT = 20;

const float RANSAC_ACCURACY_THRESHOLD = 0.002f;

MonoSOFGPU::MonoSOFGPU(CameraId cam_id, const camera::ICameraModel &intrinsics, std::unique_ptr<ISelector> selector,
                       FeaturePredictorPtr feature_predictor, [[maybe_unused]] const Settings &sof_settings)
    : MonoSOFBase(cam_id, std::move(selector), feature_predictor), intrinsics_(intrinsics) {}

void MonoSOFGPU::launch(ImageContextPtr curr_image, const ImageContextPtr &prev_image,
                        const Isometry3T &predicted_world_from_rig) {
  TRACE_EVENT ev = profiler_domain_.trace_event("launch", profiler_color_);

  {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("PredictFeatureLocations", profiler_color_);
    PredictFeatureLocations(predicted_world_from_rig);
  }
  const size_t n = tracks_.size();
  const auto max_search_radius = static_cast<float>(intrinsics_.getResolution().x());

  for (unsigned k = 0; k < n; ++k) {
    const Track &track = tracks_[k];
    const Vector2T &position = track.position();
    TrackData &data = tracks_data[k];
    TrackData &retrackdata = retrack_data[k];

    assert(!track.dead());

    if (predictedUVs_[k]) {
      Vector2T uv = *predictedUVs_[k];
      data.offset = retrackdata.offset = {uv.x() - position.x(), uv.y() - position.y()};
    }
    data.track = retrackdata.track = {position.x(), position.y()};
    data.track_status = retrackdata.track_status = false;
    data.search_radius_px = retrackdata.search_radius_px = max_search_radius;

    data.ncc_threshold = 0.8f;
    retrackdata.ncc_threshold = 0.85f;
  }

  tracks_data.copy_top_n(GPUCopyDirection::ToGPU, n, stream.get_stream());
  retrack_data.copy_top_n(GPUCopyDirection::ToGPU, n, stream.get_stream());

  const GPUGradientPyramid &prv_grad = prev_image->gpu_gradient_pyramid();
  const GPUGradientPyramid &cur_grad = curr_image->gpu_gradient_pyramid();
  const GaussianGPUImagePyramid &prv_img = prev_image->gpu_image_pyramid();
  const GaussianGPUImagePyramid &cur_img = curr_image->gpu_image_pyramid();

  feature_tracker_.track_points(prv_grad, cur_grad, prv_img, cur_img, tracks_data, n, stream.get_stream());
  feature_retracker_.track_points(prv_grad, cur_grad, prv_img, cur_img, retrack_data, n, stream.get_stream());

  tracks_data.copy_top_n(GPUCopyDirection::ToCPU, n, stream.get_stream());
  retrack_data.copy_top_n(GPUCopyDirection::ToCPU, n, stream.get_stream());
}

void MonoSOFGPU::collect() {
  cudaStreamSynchronize(stream.get_stream());

  Matrix2T info;
  const size_t n = tracks_.size();
  for (size_t k = 0; k < n; k++) {
    auto &data = tracks_data[k];
    auto &retrack = retrack_data[k];
    if (data.track_status) {
      Vector2T uv{data.track.x, data.track.y};
      info << data.info[0], data.info[1], data.info[2], data.info[3];
      tracks_.setPosition(k, uv);
      tracks_.setInfo(k, info);
    } else {
      if (retrack.track_status) {
        Vector2T uv{retrack.track.x, retrack.track.y};
        info << retrack.info[0], retrack.info[1], retrack.info[2], retrack.info[3];
        tracks_.setPosition(k, uv);
        tracks_.setInfo(k, info);
      } else {
        tracks_.kill(k);
      }
    }
  }
}

void MonoSOFGPU::track(const ImageAndSource &curr_image, const ImageContextPtr &prev_image,
                       const Isometry3T &predicted_world_from_rig, const MonoSOFFrameSettings &frame_settings,
                       const ImageSource *mask_src) {
  TRACE_EVENT ev = profiler_domain_.trace_event("track", profiler_color_);
  was_launched = false;

  const Settings &sof_settings = frame_settings.sof;

  assert(curr_image.source.type == ImageSource::U8);

  curr_image.image->build_gpu_image_pyramid(curr_image.source, sof_settings.box3_prefilter, stream.get_stream());
  curr_image.image->build_gpu_gradient_pyramid(false, stream.get_stream());
  // Sync after pyramid build to ensure GPU memory is visible to other streams.
  // Required on Blackwell (sm_121) where cross-stream memory visibility is not
  // guaranteed without explicit synchronization.
  cudaStreamSynchronize(stream.get_stream());

  curr_img_ = curr_image.image;

  if (mask_src && mask_src->data) {
    PrepareInputMask(curr_img_->get_image_meta().shape);
    curr_image.image->process_mask_gpu(*mask_src, input_mask_, stream.get_stream());
    input_mask_present_ = true;
  } else {
    input_mask_present_ = false;
  }

  // can kill tracks
  if (prev_image) {
    launch(curr_image.image, prev_image, predicted_world_from_rig);

    const ImageMeta &meta = curr_image.image->get_image_meta();
    shape_ = meta.shape;

    was_launched = true;
  }
}

const TracksVector &MonoSOFGPU::finish(FrameState &state, const MonoSOFFrameSettings &frame_settings) {
  TRACE_EVENT ev = profiler_domain_.trace_event("finish", profiler_color_);

  const Settings &sof_settings = frame_settings.sof;

  // can kill tracks
  if (was_launched) {
    collect();
    IncrementTracksAge();

    // can kill tracks
    filterByPredictionError();

    if (sof_settings.ransac_filter) {
      // can kill tracks
      ransacFilter(intrinsics_, last_keyframe_tracks_, tracks_);
    }

    // can kill tracks
    KillTracksOnBorder(shape_.width, shape_.height, sof_settings.border_top, sof_settings.border_bottom,
                       sof_settings.border_left, sof_settings.border_right, tracks_);

    // can kill tracks
    KillTracksWithinMask();

    // can kill tracks
    collapseTracks(curr_img_, tracks_);
  } else {
    //        needed for a proper image pyramid build
    cudaStreamSynchronize(stream.get_stream());
  }
  log::Value<LogFrames>("tracks2d_num_alive", tracks_.get_num_alive());

  if (tracks_.get_num_alive() <= FAILED_ACTIVE_TRACK_COUNT) {
    feature_selector_->reset_selector();
  }

  feature_selector_->set_image_width(curr_img_->get_image_meta().shape.width);
  if (SelectKeyframe(frame_settings)) {
    state = FrameState::Key;
    new_tracks_.clear();
    addFeatures(curr_img_, tracks_, new_tracks_, sof_settings);

    tracks_.add(cam_id_, new_tracks_);

    if (tracks_.get_num_alive() < FAILED_ACTIVE_TRACK_COUNT) {
      TraceError(
          "Can't select enough new tracks to reach "
          "FAILED_ACTIVE_TRACK_COUNT limit");
      feature_selector_->reset_selector();
    } else {
      feature_selector_->set_tracks(tracks_);
    }

    last_keyframe_tracks_ = tracks_;
  } else {
    state = FrameState::None;
  }

  curr_img_ = nullptr;
  was_launched = false;
  // TODO: Moving remove_dead_tracks after trackFeatures makes tracking worse
  tracks_.remove_dead_tracks();
  return tracks_;
}

void MonoSOFGPU::reset() {
  tracks_.reset();
  last_keyframe_tracks_.reset();
  alive_tracks_.clear();
  new_tracks_.clear();

  predictionTrackIds_.clear();
  predictedUVs_.clear();

  feature_selector_->reset_selector();
}

void MonoSOFGPU::addFeatures(const ImageContextPtr &image, TracksVector &existing_tracks,
                             std::vector<Vector2T> &new_tracks, const Settings &sof_settings) {
  TRACE_EVENT ev1 = profiler_domain_.trace_event("MonoSOFGPU::addFeatures", profiler_color_);
  const size_t num_alive_tracks = existing_tracks.get_num_alive();
  assert(num_alive_tracks <= static_cast<size_t>(sof_settings.num_desired_tracks) &&
         num_alive_tracks <= existing_tracks.size());
  const size_t nDesiredPointsToSelect = sof_settings.num_desired_tracks - num_alive_tracks;
  new_tracks.clear();
  new_tracks.reserve(nDesiredPointsToSelect);

  alive_tracks_.clear();
  alive_tracks_.reserve(num_alive_tracks);
  for (size_t i = 0; i < existing_tracks.size(); i++) {
    const Track &track = existing_tracks[i];
    if (!track.dead()) {
      alive_tracks_.push_back(track.position());
    }
  }

  const GPUGradientPyramid &grads = image->gpu_gradient_pyramid();

  detector_.computeGFTTAndSelectFeatures(grads, sof_settings.border_top, sof_settings.border_bottom,
                                         sof_settings.border_left, sof_settings.border_right,
                                         input_mask_present_ ? &input_mask_ : nullptr, alive_tracks_,
                                         nDesiredPointsToSelect, new_tracks, stream.get_stream());
}

void MonoSOFGPU::collapseTracks(ImageContextPtr image, TracksVector &tracks) {
  const int COLLAPSE_FACTOR = 3;

  Vector2N image_dims{image->get_image_meta().shape.width, image->get_image_meta().shape.height};

  const Vector2N trackMapDims((image_dims + Vector2N::Constant(COLLAPSE_FACTOR - 1)) / COLLAPSE_FACTOR);

  tracksMap_.resize(trackMapDims.y(), trackMapDims.x());
  tracksMap_.setZero();

  for (size_t i = 0; i < tracks.size(); ++i) {
    const Track &track = tracks[i];
    if (track.dead()) {
      continue;
    }

    size_t &p = Pixel(tracksMap_, (track.position() / float(COLLAPSE_FACTOR)).eval());
    p = (p < track.age()) ? track.age() : p;  // save max track age at this position
  }

  for (size_t i = 0; i < tracks.size(); ++i) {
    const Track &track = tracks[i];
    if (track.dead()) {
      continue;
    }

    size_t &p = Pixel(tracksMap_, (track.position() / float(COLLAPSE_FACTOR)).eval());

    if (track.age() == p) {
      p = 0;  // best (longest) track found at this location, kill others if exist
    } else {
      tracks.kill(i);
    }
  }

  assert(tracksMap_.sum() == 0);
}

void MonoSOFGPU::ransacFilter(const camera::ICameraModel &intrinsics, const TracksVector &last_keyframe_tracks,
                              TracksVector &tracks) {
#ifdef DECREASE_RANSAC_AREA
  const Vector2T furthestLocation((float)currentImage().cols() * 0.5f, (float)currentImage().rows() * 0.5f);
  float radius = furthestLocation.norm();
  const float boundaryPercentage = 0.9f;
  radius *= boundaryPercentage;
  const Vector2T lensPrincipal = intrinsics.getPrincipal();
  Vector2TPairVector sampleSequenceForRansac;
#endif
  auto lastIt = last_keyframe_tracks.cbegin();
  auto lastEnd = last_keyframe_tracks.cend();
  Vector2TPairVector sampleSequence;

  for (size_t i = 0; i < tracks.size(); ++i) {
    const Track &track = tracks[i];

    if (track.dead()) {
      continue;
    }

    for (; lastIt != lastEnd && lastIt->id() != track.id(); lastIt++)
      ;

    assert(lastIt != lastEnd);
    // @TODO: come back and revisit to make sure we do normalization only once
    Vector2T v1, v2;  // in xy coordinates
    if (!intrinsics.normalizePoint(lastIt->position(), v1) || !intrinsics.normalizePoint(track.position(), v2)) {
      tracks.kill(i);  // drop it, so the inlier loop below stays in step with sampleSequence
      continue;
    }
    sampleSequence.emplace_back(v1, v2);

#ifdef DECREASE_RANSAC_AREA
    const Vector2T trackRadius = track.position() - lensPrincipal;

    if (trackRadius.norm() > radius) {
      continue;
    }

    sampleSequenceForRansac.emplace_back(intrinsics.normalizePoint(lastIt->position()),
                                         intrinsics.normalizePoint(track.position()));
#endif
  }

  const float threshold = RANSAC_ACCURACY_THRESHOLD;

  Matrix3T essential;
  math::Ransac<epipolar::Fundamental> fr;
  fr.setOptions(math::Ransac<epipolar::Fundamental>::Epipolar, threshold, 0.002f);
#ifdef DECREASE_RANSAC_AREA
  if (!fr(essential, sampleSequenceForRansac.begin(), sampleSequenceForRansac.end()))
#else
  if (!fr(essential, sampleSequence.begin(), sampleSequence.end()))
#endif
  {
    return;
  }

  auto seqIt = sampleSequence.cbegin();

  for (size_t i = 0; i < tracks.size(); ++i) {
    const Track &track = tracks[i];
    if (track.dead()) {
      continue;
    }

    if (!fr.isInlier(essential, *seqIt++)) {
      tracks.kill(i);
    }
  }

  assert(seqIt == sampleSequence.cend());
}

}  // namespace cuvslam::sof
