
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

#include "slam/map/descriptor/st_descriptor_ops.h"

#include "sof/gaussian_coefficients.h"

#include "slam/common/blob_eigen.h"
#include "slam/map/descriptor/st_descriptor_merge_details.h"

namespace cuvslam::slam {

STDescriptorOps::STDescriptorOps(uint32_t n_shift_only_iterations, uint32_t n_full_mapping_iterations)
    : n_shift_only_iterations(n_shift_only_iterations),
      n_full_mapping_iterations(n_full_mapping_iterations),
      st_tracker(n_shift_only_iterations, n_full_mapping_iterations) {}
STDescriptorOps::~STDescriptorOps() {}

FeatureDescriptor STDescriptorOps::CreateDescriptor(const sof::ImagePyramidT& previous_image,
                                                    const sof::GradientPyramidT&, const Vector2T& UV) const {
  uint32_t levels_mask;
  std::vector<sof::STFeaturePatch> image_patches;
  bool res = st_tracker.BuildPointCache(previous_image, UV, levels_mask, image_patches);
  if (!res) {
    return FeatureDescriptorEmpty;
  }

  const size_t size_bytes = sizeof(STDescriptor) + sizeof(sof::STFeaturePatch) * image_patches.size();
  auto shared_ptr = std::shared_ptr<STDescriptor>(reinterpret_cast<STDescriptor*>(new char[size_bytes]), [](void* ptr) {
    if (ptr) {
      delete[] static_cast<char*>(ptr);
    }
  });

  STDescriptor* dst = shared_ptr.get();

  dst->levels_mask = levels_mask;
  dst->image_patches_size = static_cast<uint32_t>(image_patches.size());
  std::copy(image_patches.begin(), image_patches.end(), dst->image_patches);

  FeatureDescriptor fd;
  fd.pull_memory = shared_ptr;
  return fd;
}

bool STDescriptorOps::MapToImage(const FeatureDescriptorRef fd, const sof::ImagePyramidT& image,
                                 const sof::GradientPyramidT& image_gradient,
                                 const Vector2T* predicted_UV,  // null if no prediction
                                 Vector2T& UV, Track2dInfo* info, float search_radius_px) const {
  auto st_fd = static_cast<const STDescriptor*>(fd.get());
  if (!st_fd) {
    return false;
  }

  Vector2T previous_uv(0, 0);
  UV = previous_uv;
  if (predicted_UV) {
    UV = *predicted_UV;
    previous_uv = *predicted_UV;
  }

  float ncc;
  float current_info[4];
  return st_tracker.TrackPointWithCache(image_gradient, image, st_fd->levels_mask, st_fd->image_patches_size,
                                        st_fd->image_patches, previous_uv.data(), UV.data(), info ? info->ncc : ncc,
                                        info ? info->info : current_info, search_radius_px, 0.f);
}

float STDescriptorOps::Match(FeatureDescriptorRef fd0, FeatureDescriptorRef fd1) const {
  return merge_details::ncc(n_shift_only_iterations, n_full_mapping_iterations, fd0, fd1);
}

bool STDescriptorOps::ToBlob(const FeatureDescriptorRef fd, BlobWriter& blob_writer) const {
  if (!fd) {
    return false;
  }
  auto st_fd = static_cast<const STDescriptor*>(fd.get());
  if (!st_fd) {
    return false;
  }

  // TODO: add version
  int size = 0;
  size += sizeof(STDescriptor);
  size += sizeof(sof::STFeaturePatch) * st_fd->image_patches_size;

  blob_writer.reserve(size);
  blob_writer.write(st_fd, size);
  return true;
}

// The blob usually points into a database read transaction that ends when the caller returns, so the
// descriptor has to own a copy rather than point into the blob.
FeatureDescriptor STDescriptorOps::CreateFromBlob(const BlobReader& blob_reader) const {
  STDescriptor header;
  if (!blob_reader.read(header)) {
    return FeatureDescriptorEmpty;
  }
  if (header.image_patches_size > blob_reader.remaining() / sizeof(sof::STFeaturePatch)) {
    return FeatureDescriptorEmpty;
  }
  const size_t patches_bytes = sizeof(sof::STFeaturePatch) * header.image_patches_size;

  auto shared_ptr = std::shared_ptr<STDescriptor>(
      reinterpret_cast<STDescriptor*>(new char[sizeof(STDescriptor) + patches_bytes]), [](void* ptr) {
        if (ptr) {
          delete[] static_cast<char*>(ptr);
        }
      });
  STDescriptor* dst = shared_ptr.get();
  dst->levels_mask = header.levels_mask;
  dst->image_patches_size = header.image_patches_size;
  if (!blob_reader.read(static_cast<void*>(dst->image_patches), patches_bytes)) {
    return FeatureDescriptorEmpty;
  }

  FeatureDescriptor fd;
  fd.pull_memory = shared_ptr;
  return fd;
}
FeatureDescriptor STDescriptorOps::Copy(const FeatureDescriptor& source) const {
  if (source.pull_memory) {
    return source;
  }
  if (source.const_memory_ptr) {
    const STDescriptor* src_ptr = reinterpret_cast<const STDescriptor*>(source.const_memory_ptr);

    const size_t size_bytes = sizeof(STDescriptor) + sizeof(sof::STFeaturePatch) * src_ptr->image_patches_size;

    auto shared_ptr =
        std::shared_ptr<STDescriptor>(reinterpret_cast<STDescriptor*>(new char[size_bytes]), [](void* ptr) {
          if (ptr) {
            delete[] static_cast<char*>(ptr);
          }
        });
    IFeatureDescriptor* pfd = shared_ptr.get();
    memcpy(pfd, source.const_memory_ptr, size_bytes);

    FeatureDescriptor fd;
    fd.pull_memory = shared_ptr;
    return fd;
  }
  return FeatureDescriptor();
}

int STDescriptorOps::Compare(const FeatureDescriptorRef fd0, const FeatureDescriptorRef fd1) const {
  auto st_fd0 = static_cast<const STDescriptor*>(fd0.get());
  auto st_fd1 = static_cast<const STDescriptor*>(fd1.get());
  if (!st_fd0 || !st_fd1) {
    return -1;
  }
  if (st_fd0->levels_mask != st_fd1->levels_mask) {
    return st_fd0->levels_mask - st_fd1->levels_mask;
  }
  if (st_fd0->image_patches_size != st_fd1->image_patches_size) {
    return st_fd0->image_patches_size - st_fd1->image_patches_size;
  }
  for (uint32_t level = 0; level < st_fd0->image_patches_size; level++) {
    if ((st_fd0->levels_mask & (1 << level)) == 0) {
      continue;
    }
    auto& v0 = st_fd0->image_patches[level];
    auto& v1 = st_fd1->image_patches[level];

    for (int i = 0; i < v0.rows(); i++) {
      for (int j = 0; j < v0.cols(); j++) {
        auto e0 = v0(i, j);
        auto e1 = v1(i, j);
        if (e0 != e1) {
          return std::signbit(e0 - e1) ? -1 : 1;
        }
      }
    }
  }
  return 0;
}

// methods for process batch of tracks
void STDescriptorOps::CreateDescriptors(const ImageContextPtr& image, const std::vector<Vector2T>& input,
                                        std::vector<FeatureDescriptor>& output) const {
  output.clear();
  output.resize(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    Vector2T uv = input[i];
    output[i] = this->CreateDescriptor(image->cpu_image_pyramid(), image->cpu_gradient_pyramid(), uv);
  }
}

void STDescriptorOps::MapsToImage(const ImageContextPtr& image, const std::vector<FeatureDescriptor>& descriptors,
                                  const std::vector<IFeatureDescriptorOps::Track2dInput>& input,
                                  std::vector<IFeatureDescriptorOps::Track2dOutput>& output) const {
  if (input.empty()) {
    return;
  }
  output.clear();
  output.resize(input.size());

  for (size_t i = 0; i < input.size(); i++) {
    auto& in = input[i];
    auto& out = output[i];

    Vector2T predicted_UV = in.predict_uv;
    Vector2T uv;
    Track2dInfo info;
    out.successed =
        this->MapToImage(descriptors[in.fd_index], image->cpu_image_pyramid(), image->cpu_gradient_pyramid(),
                         in.has_predict_uv ? &predicted_UV : nullptr, uv, &info, in.search_radius_px);
    out.uv = uv;
    out.ncc = info.ncc;
    // out.info = info.info;
    memcpy(out.info, info.info, sizeof(out.info));
  }
}

}  // namespace cuvslam::slam
