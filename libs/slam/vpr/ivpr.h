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

/// Visual place recognition backend: holds one descriptor per pose graph node and answers "which
/// node saw this frame?". It knows nothing about poses; VprMap resolves the node id it returns.
///
/// Thread safety: not thread safe. VprMap serializes all access.
class IVpr {
public:
  virtual ~IVpr() = default;

  /// Printable backend name, used in logs.
  virtual const char* Name() const = 0;

  /// Store the descriptor of `image` for pose graph node `node_id`, replacing any previous entry.
  virtual void AddFrame(KeyFrameId node_id, const VprImage& image) = 0;

  /// True when `node_id` already has a descriptor.
  virtual bool HasNode(KeyFrameId node_id) const = 0;

  /// Drop the descriptor of `node_id`, if any.
  virtual void RemoveNode(KeyFrameId node_id) = 0;

  /// Renumber every stored node id by adding `offset`.
  /// Used after loading a map so its ids cannot collide with the ids of the live pose graph.
  virtual void ShiftNodeIds(KeyFrameId offset) = 0;

  /// Number of stored descriptors.
  virtual size_t Size() const = 0;

  /// Bring the backend into a queryable state. Backends that need a vocabulary train it here.
  /// Called automatically by Query() and Serialize(); calling it twice is a no-op unless the map
  /// changed in between.
  virtual void Finalize() = 0;

  /// Find the stored descriptor closest to `image`.
  /// Not const: backends may build their search index lazily on the first query.
  virtual VprMatch Query(const VprImage& image) = 0;

  /// Find the `max_results` stored descriptors closest to `image`, best first.
  /// The best-looking place is not always the right one, so a relocalizer verifies several. The default
  /// implementation returns just the winner, which is correct but leaves it nothing to fall back on.
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

  /// Append the backend state to `blob`.
  virtual void Serialize(Blob& blob) = 0;

  /// Restore the backend state written by Serialize(). Returns false on a malformed or foreign blob.
  virtual bool Deserialize(const BlobReader& reader) = 0;
};

/// Instantiate the backend named by `options.type`.
/// Returns nullptr for VprType::kNone.
/// @throws std::runtime_error if the backend is not available in this build.
std::unique_ptr<IVpr> CreateVpr(const VprOptions& options);

}  // namespace cuvslam::slam::vpr
