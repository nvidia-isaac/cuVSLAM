
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

#include "utils/image_loader.h"

#include <cstring>

#include "cnpy.h"
#include "gflags/gflags.h"

#include "common/tga.h"
#include "utils/image_io.h"
#include "utils/image_transform.h"

// TODO: rename to cache_as_jpeg or smth
DEFINE_bool(cache_uncompressed, false, "Cache uncompressed images");
DEFINE_double(depth_scale_factor, 1000, "Depth scale factor");

namespace {
using namespace cuvslam;

bool LoadDepthFromPng16Bit(const std::string& filename, ImageMatrix<float>& depth) {
  ImageMatrix<uint16_t> depth_16;  // read first to 16bit int, then convert to float
  if (!LoadPng(filename, depth_16)) {
    return false;
  }
  depth = depth_16.cast<float>() / FLAGS_depth_scale_factor;
  return true;
}

}  // namespace

namespace cuvslam::utils {

struct ImageLoaderT::Impl {
  ImageMatrix<uint8_t> imageMatrix_u8_;
};

ImageLoaderT::ImageLoaderT() : impl_(std::make_unique<Impl>()) {}

ImageLoaderT::~ImageLoaderT() = default;

bool ImageLoaderT::load(const std::string& filename) {
  const char* filePath = filename.c_str();
  impl_->imageMatrix_u8_ = {};

  const char* dot = std::strrchr(filePath, '.');
  if (!dot++) {
    TraceError("Can't detect image format %s", filePath);
    return false;
  }
  if (!std::strcmp(dot, "png")) {
    if (FLAGS_cache_uncompressed) {
      if (!LoadPngCached(filePath, impl_->imageMatrix_u8_)) {
        TraceError("Can't load cached PNG file %s", filePath);
        return false;
      }
    } else {
      if (!LoadPng(filePath, impl_->imageMatrix_u8_)) {
        TraceError("Can't load PNG file %s", filePath);
        return false;
      }
    }
  } else if (!std::strcmp(dot, "tga")) {
    if (!LoadTga(filePath, impl_->imageMatrix_u8_)) {
      TraceError("Can't load TGA file %s", filePath);
      return false;
    }
  } else if (!std::strcmp(dot, "jpg") || !std::strcmp(dot, "jpeg")) {
    if (!LoadJpeg(filePath, impl_->imageMatrix_u8_)) {
      TraceError("Can't load JPEG file %s", filePath);
      return false;
    }
  } else {
    TraceError("Unknown image format for %s", filePath);
    return false;
  }
  return true;
}

const ImageMatrix<uint8_t>& ImageLoaderT::getImage() const {
  assert(impl_->imageMatrix_u8_.data());
  return impl_->imageMatrix_u8_;
}

struct DepthLoader::Impl {
  ImageMatrix<float> depth_matrix;
};

DepthLoader::DepthLoader() : impl_(std::make_unique<Impl>()) {}

DepthLoader::DepthLoader(const char* filePath) : impl_(std::make_unique<Impl>()) { load(filePath); }

DepthLoader::~DepthLoader() = default;

void DepthLoader::load(const char* filePath) {
  impl_->depth_matrix = {};

  const char* dot = std::strrchr(filePath, '.');
  if (!dot++) {
    throw std::invalid_argument(std::string("Unknown image format for ") + filePath);
  } else if (std::strcmp(dot, "npy") == 0) {
    cnpy::NpyArray arr = cnpy::npy_load(filePath);
    size_t height = arr.shape[0];
    size_t width = arr.shape[1];

    auto data = arr.data<float>();

    impl_->depth_matrix.resize(height, width);

    for (size_t i = 0; i < height; i++) {
      for (size_t j = 0; j < width; j++) {
        impl_->depth_matrix(i, j) = data[i * width + j];
      }
    }
  } else if (std::strcmp(dot, "png") == 0) {
    if (!LoadDepthFromPng16Bit(filePath, impl_->depth_matrix)) {
      throw std::runtime_error(std::string("Can't load depth from PNG. ") + filePath);
    }
  } else {
    throw std::invalid_argument(std::string("Unknown image format for ") + filePath);
  }
}

const ImageMatrix<float>& DepthLoader::getImage() const {
  assert(impl_->depth_matrix.data());
  return impl_->depth_matrix;
}

}  // namespace cuvslam::utils
