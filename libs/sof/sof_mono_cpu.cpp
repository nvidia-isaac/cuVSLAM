
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

#include "sof/internal/sof_mono_cpu.h"

#include "common/log_types.h"
#include "epipolar/fundamental_ransac.h"

namespace cuvslam::sof {

// threshold to determine that tracking have failed
const size_t FAILED_ACTIVE_TRACK_COUNT = 20;

const float RANSAC_ACCURACY_THRESHOLD = 0.002f;

MonoSOFCPU::MonoSOFCPU(CameraId cam_id, const camera::ICameraModel &intrinsics, std::unique_ptr<ISelector> selector,
                       FeaturePredictorPtr feature_predictor, const Settings &sof_settings)
    : MonoSOFBase(cam_id, std::move(selector), feature_predictor),
      intrinsics_(intrinsics),
      feature_tracker_(CreateTracker(sof_settings.tracker)) {}

void MonoSOFCPU::track(const ImageAndSource &curr_image, const ImageContextPtr &prev_image,
                       const Isometry3T &worldFromRig, const MonoSOFFrameSettings &frame_settings,
                       const ImageSource *mask_src) {
  TRACE_EVENT ev = profiler_domain_.trace_event("MonotrackNextFrame()", profiler_color_);

  const Settings &sof_settings = frame_settings.sof;

  assert(curr_image.source.type == ImageSource::U8);

  bool res = curr_image.image->build_cpu_image_pyramid(curr_image.source, sof_settings.box3_prefilter);
  if (!res) {
    TraceError("Failed to build cpu image pyramid");
  }
  res = curr_image.image->build_cpu_gradient_pyramid(false);
  if (!res) {
    TraceError("Failed to build cpu gradient pyramid");
  }

  const ImageMeta &meta = curr_image.image->get_image_meta();
  const ImageShape &shape = meta.shape;
  const int w = shape.width;
  const int h = shape.height;

  if (mask_src && mask_src->data) {
    PrepareInputMask(shape);
    curr_image.image->process_mask_cpu(*mask_src, input_mask_);
    input_mask_present_ = true;
  } else {
    input_mask_present_ = false;
  }
  if (prev_image) {
    // can kill tracks
    trackFeatures(curr_image.image, prev_image, worldFromRig);
    IncrementTracksAge();

    // can kill tracks
    filterByPredictionError();
    log::Value<LogFrames>("tracks2d_killed_count", tracks_.size() - tracks_.get_num_alive());

    if (sof_settings.ransac_filter) {
      // can kill tracks
      ransacFilter(intrinsics_, last_keyframe_tracks_, tracks_);
    }

    // can kill tracks
    KillTracksOnBorder(w, h, sof_settings.border_top, sof_settings.border_bottom, sof_settings.border_left,
                       sof_settings.border_right, tracks_);
    // can kill tracks
    KillTracksWithinMask();

    // can kill tracks
    collapseTracks(curr_image.image, tracks_);
  }
  log::Value<LogFrames>("tracks2d_num_alive", tracks_.get_num_alive());

  if (tracks_.get_num_alive() <= FAILED_ACTIVE_TRACK_COUNT) {
    feature_selector_->reset_selector();
  }

  feature_selector_->set_image_width(w);
  if (SelectKeyframe(frame_settings)) {
    last_frame_state_ = FrameState::Key;
    new_tracks_.clear();
    addFeatures(curr_image.image, tracks_, new_tracks_, sof_settings);

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
    last_frame_state_ = FrameState::None;
  }
  // TODO: Moving remove_dead_tracks after trackFeatures makes tracking worse
  tracks_.remove_dead_tracks();
}

const TracksVector &MonoSOFCPU::finish(FrameState &state, [[maybe_unused]] const MonoSOFFrameSettings &frame_settings) {
  state = last_frame_state_;
  return tracks_;
}

void MonoSOFCPU::addFeatures(const ImageContextPtr &image, TracksVector &existing_tracks,
                             std::vector<Vector2T> &new_tracks, const Settings &sof_settings) {
  const size_t num_alive_tracks = existing_tracks.get_num_alive();
  assert(num_alive_tracks <= static_cast<uint32_t>(sof_settings.num_desired_tracks) &&
         num_alive_tracks <= existing_tracks.size());
  const size_t nDesiredPointsToSelect = sof_settings.num_desired_tracks - num_alive_tracks;
  new_tracks.clear();
  new_tracks.reserve(nDesiredPointsToSelect);

  detector_.computeGFTTAndSelectFeatures(image->cpu_gradient_pyramid(), sof_settings.border_top,
                                         sof_settings.border_bottom, sof_settings.border_left,
                                         sof_settings.border_right, input_mask_present_ ? &input_mask_ : nullptr,
                                         existing_tracks, nDesiredPointsToSelect, new_tracks);
}

void MonoSOFCPU::trackFeatures(const ImageContextPtr &curr_image, const ImageContextPtr &prev_image,
                               const Isometry3T &predicted_world_from_rig) {
  PredictFeatureLocations(predicted_world_from_rig);

  const size_t n = tracks_.size();

  const auto max_search_radius = static_cast<float>(intrinsics_.getResolution().x());

  for (unsigned k = 0; k < n; ++k) {
    const Track &track = tracks_[k];

    assert(!track.dead());

    Vector2T uv = tracks_[k].position();
    if (predictedUVs_[k]) {
      uv = *predictedUVs_[k];
    }

    const GradientPyramidT &prv_grad = prev_image->cpu_gradient_pyramid();
    const GradientPyramidT &cur_grad = curr_image->cpu_gradient_pyramid();
    const ImagePyramidT &prv_img = prev_image->cpu_image_pyramid();
    const ImagePyramidT &cur_img = curr_image->cpu_image_pyramid();
    const Vector2T &position = track.position();
    Matrix2T cur_info;  // ignore it

    bool res =
        feature_tracker_->trackPoint(prv_grad, cur_grad, prv_img, cur_img, position, uv, cur_info, max_search_radius);

    if (!res) {
      // There was some prediction, but tracking fails. It's probably was a wrong prediction.
      // Let's try again without. Prediction is a kind of optimization. It shouldn't help or
      // break tracking by design but speedup and fix repetitive structure uncertainties.

      // TODO: [optimization] re-track only with high dead rate
      res = feature_tracker_->trackPoint(prv_grad, cur_grad, prv_img, cur_img, position, uv, cur_info,
                                         max_search_radius, 0.85f);
    }
    if (res) {
      tracks_.setPosition(k, uv);
      tracks_.setInfo(k, cur_info);
    } else {
      // failed track: mark it for deletion and decrement active tracks counter
      tracks_.kill(k);
    }
  }
}

void MonoSOFCPU::collapseTracks(const ImageContextPtr &image, TracksVector &tracks) {
  const int COLLAPSE_FACTOR = 3;

  const Vector2N trackMapDims((ImageDims(image->cpu_image_pyramid()[0]) + Vector2N::Constant(COLLAPSE_FACTOR - 1)) /
                              COLLAPSE_FACTOR);

  tracksMap_.resize(trackMapDims.y(), trackMapDims.x());
  tracksMap_.setZero();

  for (size_t i = 0; i < tracks.size(); ++i) {
    const Track &track = tracks[i];
    if (track.dead()) {
      continue;
    }

    size_t &p = Pixel(tracksMap_, (track.position() / float(COLLAPSE_FACTOR)).eval());
    p = (p < track.age()) ? track.age() : p;  // save max track length at this position
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

void MonoSOFCPU::ransacFilter(const camera::ICameraModel &intrinsics, const TracksVector &last_keyframe_tracks,
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

void MonoSOFCPU::reset() {
  tracks_.reset();
  last_keyframe_tracks_.reset();
  new_tracks_.clear();

  predictionTrackIds_.clear();
  predictedUVs_.clear();

  feature_selector_->reset_selector();
}

}  // namespace cuvslam::sof
