
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

#include <stdint.h>
#include <vector>

#include "common/isometry.h"
#include "common/types.h"
#include "common/unaligned_types.h"

namespace cuvslam::slam {

using Blob = std::vector<uint8_t>;

class BlobWriter {
  Blob& buffer_;

public:
  BlobWriter(Blob& buffer) : buffer_(buffer) {}
  void reserve(size_t bites) { buffer_.reserve(bites); };
  template <class T>
  void write(const T& data) {
    buffer_.insert(buffer_.end(), (uint8_t*)&data, ((uint8_t*)&data) + sizeof(T));
  }
  void write(const std::string& data) { write_str(data); }
  void write(const void* data, size_t size) { buffer_.insert(buffer_.end(), (uint8_t*)data, (uint8_t*)data + size); }
  template <class T>
  void write_std(const std::vector<T>& data) {
    size_t sz = data.size();
    this->write(sz);
    if (sz != 0) {
      this->write(&data[0], sizeof(T) * sz);
    }
  }
  void write_str(const std::string& text) {
    size_t sz = text.size();
    this->write(sz);
    this->write(text.c_str(), sz);
  }

  void write_eigen(const Isometry3T& m) {
    storage::Pose<float> storage_m = m;
    this->write(storage_m);
  }
  template <typename _Scalar, int _Rows, int _Cols, int _Options>
  void write_eigen(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options>& m) {
    Eigen::Matrix<_Scalar, _Rows, _Cols, Eigen::DontAlign | Eigen::ColMajor> storage_m = m;
    this->write(storage_m);
  }
};

class BlobReader {
  const uint8_t* data_;
  size_t size_;
  mutable size_t feed_offset_ = 0;

public:
  BlobReader(const Blob& buffer) : data_(&buffer[0]), size_(buffer.size()) {}
  BlobReader(const void* data, size_t size) : data_((uint8_t*)data), size_(size) {}

  template <class T>
  bool read(T& dst) const {
    auto ptr = feed_forward(sizeof(T));
    if (!ptr) {
      return false;
    }
    dst = *(const T*)ptr;
    return true;
  }
  bool read(std::string& dst) const { return read_str(dst); }
  bool read(void* dst, size_t size) const {
    auto ptr = feed_forward(size);
    if (!ptr) {
      return false;
    }
    memcpy(dst, ptr, size);
    return true;
  }
  bool read_str(std::string& dst) const {
    size_t sz;
    if (!this->read(sz)) {
      return false;
    }
    dst.resize(sz);
    return this->read(&dst[0], sz);
  }
  template <class T>
  bool read_std(std::vector<T>& dst) const {
    size_t sz;
    if (!this->read(sz)) {
      return false;
    }
    dst.resize(sz);
    return this->read(&dst[0], sizeof(T) * sz);
  }
  bool read_eigen(Isometry3T& m) const {
    storage::Pose<float> storage_m;
    if (!this->read(storage_m)) {
      return false;
    }
    m = storage_m;
    return true;
  }
  template <typename _Scalar, int _Rows, int _Cols, int _Options>
  bool read_eigen(Eigen::Matrix<_Scalar, _Rows, _Cols, _Options>& m) const {
    Eigen::Matrix<_Scalar, _Rows, _Cols, Eigen::DontAlign | Eigen::ColMajor> storage_m;
    if (!this->read(storage_m)) {
      return false;
    }
    m = storage_m;
    return true;
  }

  // Bytes left to read. A deserializer bounds a count word it reads out of the blob against this
  // before allocating for it: a corrupt or hostile count would otherwise reserve for entries the
  // file cannot possibly contain, and throw length_error or bad_alloc where the caller expects a
  // clean false.
  size_t remaining() const { return size_ - feed_offset_; }

  const uint8_t* feed_forward(size_t size) const {
    if (size_ - feed_offset_ < size) {
      return nullptr;
    }

    auto dst = data_ + feed_offset_;
    feed_offset_ += size;
    return dst;
  }
};

}  // namespace cuvslam::slam
