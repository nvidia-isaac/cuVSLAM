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

#pragma once

#include <memory>
#include <vector>

#include "slam/common/blob.h"
#include "slam/vpr/vpr_types.h"

namespace cuvslam::slam::vpr {

/// One descriptor per pose graph node. Not thread safe; VprMap serializes access.
class IVpr {
public:
  virtual ~IVpr() = default;

  /// Backend name for logs.
  virtual const char* Name() const = 0;

  /// Store the descriptor of `image` for `node_id`, replacing any previous one.
  virtual void AddFrame(KeyFrameId node_id, const VprImage& image) = 0;

  virtual bool HasNode(KeyFrameId node_id) const = 0;

  virtual void RemoveNode(KeyFrameId node_id) = 0;

  virtual size_t Size() const = 0;

  /// Frames AddFrame() holds for the vocabulary fit before fitting early; 0 if the backend fits none.
  virtual size_t TrainingCapacity() const { return 0; }

  /// Prepare for queries, e.g. train a vocabulary; a no-op unless the map changed.
  virtual void Finalize() = 0;

  /// Closest stored descriptor to `image`. Not const: the index may be built lazily.
  virtual VprMatch Query(const VprImage& image) = 0;

  /// The `max_results` closest descriptors, best first. Default: Query()'s winner only.
  virtual std::vector<VprMatch> QueryTopK(const VprImage& image, size_t max_results) {
    std::vector<VprMatch> matches;
    if (max_results == 0) {
      return matches;
    }
    const VprMatch best = Query(image);
    if (best.found) {
      matches.push_back(best);
    }
    return matches;
  }

  /// Write the backend state.
  virtual void Serialize(BlobWriter& writer) = 0;

  /// Restore state written by Serialize(); false on a malformed or foreign blob.
  virtual bool Deserialize(const BlobReader& reader) = 0;
};

/// AnyLoc backend, nullptr unless `options.enabled`.
/// @throws std::runtime_error if the backend is not in this build.
std::unique_ptr<IVpr> CreateVpr(const VprOptions& options);

}  // namespace cuvslam::slam::vpr
