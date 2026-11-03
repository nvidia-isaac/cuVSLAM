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

#include "slam/vpr/anyloc_network.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <utility>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include "common/log.h"
#include "cuda_modules/cuda_helper.h"

namespace cuvslam::slam::vpr {

namespace {

constexpr char kInputName[] = "image";
constexpr char kOutputName[] = "patch_descriptors";

const std::string kTensorRtVersion = std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
                                     std::to_string(NV_TENSORRT_PATCH);
const std::string kTensorRtLibrary = "libnvinfer.so." + std::to_string(NV_TENSORRT_MAJOR);
// Major version factor in getInferLibVersion(): TensorRT 10 changed it from 1000
constexpr int32_t kTensorRtMajorFactor = NV_TENSORRT_MAJOR < 10 ? 1000 : 10000;

/// One per process, as TensorRT expects. Remembers the last error per thread.
class TensorRtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      last_error_ = message;
      TraceError("VPR AnyLoc: TensorRT: %s\n", message);
    } else if (severity == Severity::kWARNING) {
      TraceWarning("VPR AnyLoc: TensorRT: %s\n", message);
    }
  }

  std::string TakeLastError() { return std::exchange(last_error_, std::string{}); }

private:
  static thread_local std::string last_error_;
};

thread_local std::string TensorRtLogger::last_error_;

TensorRtLogger& Logger() {
  static TensorRtLogger logger;
  return logger;
}

bool Succeeded(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  TraceError("VPR AnyLoc: %s failed: %s\n", what, cudaGetErrorString(status));
  // Clear the error so another component's cudaGetLastError() does not see it.
  cudaGetLastError();
  return false;
}

std::string ToString(const nvinfer1::Dims& dims) {
  std::string text = "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(dims.d[i]);
  }
  return text + "]";
}

const char* ToString(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return "fp32";
    case nvinfer1::DataType::kHALF:
      return "fp16";
    case nvinfer1::DataType::kINT8:
      return "int8";
    case nvinfer1::DataType::kINT32:
      return "int32";
    default:
      return "other";
  }
}

/// Check before any lookup: TensorRT logs an error for an unknown tensor name.
bool HasTensor(const nvinfer1::ICudaEngine& engine, const char* name) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    if (std::string(engine.getIOTensorName(i)) == name) {
      return true;
    }
  }
  return false;
}

/// True for a static fp32 linear tensor of this mode and rank, with batch 1.
bool IsTensor(const nvinfer1::ICudaEngine& engine, const char* name, nvinfer1::TensorIOMode mode, int rank) {
  if (!HasTensor(engine, name) || engine.getTensorIOMode(name) != mode ||
      engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
      engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  const nvinfer1::Dims dims = engine.getTensorShape(name);
  if (dims.nbDims != rank || dims.d[0] != 1) {
    return false;
  }
  for (int i = 1; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      return false;
    }
  }
  return true;
}

/// False unless `path` is a readable, non-empty regular file.
bool ReadFile(const std::string& path, std::vector<char>& bytes) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return false;
  }
  const uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size == 0) {
    return false;
  }
  bytes.resize(static_cast<size_t>(size));
  std::ifstream file(path, std::ios::binary);
  return static_cast<bool>(file.read(bytes.data(), static_cast<std::streamsize>(size)));
}

void* TensorRtSymbol(void* handle, const char* symbol) {
  void* address = dlsym(handle, symbol);
  if (!address) {
    throw std::runtime_error("VPR AnyLoc: " + kTensorRtLibrary + " has no " + symbol + ": " + dlerror());
  }
  return address;
}

}  // namespace

CreateInferRuntime LoadTensorRt() {
  void* handle = dlopen(kTensorRtLibrary.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    throw std::runtime_error("VPR AnyLoc needs TensorRT " + std::to_string(NV_TENSORRT_MAJOR) + ": " + dlerror());
  }
  const int32_t version = reinterpret_cast<int32_t (*)()>(TensorRtSymbol(handle, "getInferLibVersion"))();
  if (version / kTensorRtMajorFactor != NV_TENSORRT_MAJOR) {
    throw std::runtime_error("VPR AnyLoc needs TensorRT " + std::to_string(NV_TENSORRT_MAJOR) + ", but " +
                             kTensorRtLibrary + " reports version " + std::to_string(version));
  }
  return reinterpret_cast<CreateInferRuntime>(TensorRtSymbol(handle, "createInferRuntime_INTERNAL"));
}

std::unique_ptr<AnyLocNetwork> AnyLocNetwork::Load(const std::string& engine_path, std::string& error) {
  std::vector<char> plan;
  if (!ReadFile(engine_path, plan)) {
    error = "VPR AnyLoc: could not read the TensorRT engine \"" + engine_path + "\".";
    return nullptr;
  }

  std::unique_ptr<AnyLocNetwork> network(new AnyLocNetwork());
  TensorRtLogger& logger = Logger();
  logger.TakeLastError();
  static const CreateInferRuntime create_runtime = LoadTensorRt();
  network->runtime_.reset(static_cast<nvinfer1::IRuntime*>(create_runtime(&logger, NV_TENSORRT_VERSION)));
  if (!network->runtime_) {
    error = "VPR AnyLoc: could not create the TensorRT " + kTensorRtVersion + " runtime.";
    return nullptr;
  }
  network->engine_.reset(network->runtime_->deserializeCudaEngine(plan.data(), plan.size()));
  if (!network->engine_) {
    error = "VPR AnyLoc: \"" + engine_path + "\" is not a TensorRT engine that TensorRT " + kTensorRtVersion +
            " can run on this GPU (" + logger.TakeLastError() +
            "). An engine only loads into the TensorRT version and onto the GPU architecture that built it: rebuild "
            "it with the anyloc_engine CMake target or anyloc_engine_builder.";
    return nullptr;
  }

  const nvinfer1::ICudaEngine& engine = *network->engine_;
  if (engine.getNbIOTensors() != 2 || !IsTensor(engine, kInputName, nvinfer1::TensorIOMode::kINPUT, 4) ||
      engine.getTensorShape(kInputName).d[1] != 3 ||
      !IsTensor(engine, kOutputName, nvinfer1::TensorIOMode::kOUTPUT, 3)) {
    std::string found;
    for (int i = 0; i < engine.getNbIOTensors(); ++i) {
      const char* name = engine.getIOTensorName(i);
      found += std::string(i == 0 ? "" : ", ") +
               (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? "input " : "output ") + name + " " +
               ToString(engine.getTensorDataType(name)) + " " + ToString(engine.getTensorShape(name));
    }
    error = "VPR AnyLoc: \"" + engine_path +
            "\" is not the DINOv2 patch descriptor engine AnyLoc runs, which takes input image fp32 [1, 3, H, W] and "
            "gives output patch_descriptors fp32 [1, N, D]; it has " +
            found + ".";
    return nullptr;
  }
  const nvinfer1::Dims input_dims = engine.getTensorShape(kInputName);
  const nvinfer1::Dims output_dims = engine.getTensorShape(kOutputName);
  network->height_ = static_cast<int>(input_dims.d[2]);
  network->width_ = static_cast<int>(input_dims.d[3]);
  network->patches_ = static_cast<int>(output_dims.d[1]);
  network->dim_ = static_cast<int>(output_dims.d[2]);

  network->context_.reset(network->engine_->createExecutionContext());
  if (!network->context_) {
    error = "VPR AnyLoc: could not create an execution context for \"" + engine_path + "\" (" + logger.TakeLastError() +
            ").";
    return nullptr;
  }
  network->host_input_.resize(static_cast<size_t>(3) * network->height_ * network->width_);
  network->host_output_.resize(static_cast<size_t>(network->patches_) * network->dim_);
  network->device_input_.emplace(network->host_input_.size());
  network->device_output_.emplace(network->host_output_.size());
  if (!network->context_->setTensorAddress(kInputName, network->device_input_->ptr()) ||
      !network->context_->setTensorAddress(kOutputName, network->device_output_->ptr())) {
    error = "VPR AnyLoc: could not bind the inputs and outputs of \"" + engine_path + "\".";
    return nullptr;
  }

  // Warm up, so the first keyframe does not pay for TensorRT's lazy init.
  std::vector<float> warm_up;
  if (!network->Run(warm_up)) {
    error = "VPR AnyLoc: the first inference of \"" + engine_path + "\" failed, see the log.";
    return nullptr;
  }
  return network;
}

int AnyLocNetwork::input_width() const { return width_; }

int AnyLocNetwork::input_height() const { return height_; }

int AnyLocNetwork::patch_count() const { return patches_; }

int AnyLocNetwork::descriptor_dim() const { return dim_; }

float* AnyLocNetwork::input() { return host_input_.data(); }

bool AnyLocNetwork::Run(std::vector<float>& patches) {
  cudaStream_t stream = stream_.get_stream();
  bool ok = Succeeded(cudaMemcpyAsync(device_input_->ptr(), host_input_.data(), host_input_.size() * sizeof(float),
                                      cudaMemcpyHostToDevice, stream),
                      "copying the image to the GPU");
  if (ok && !context_->enqueueV3(stream)) {
    TraceError("VPR AnyLoc: TensorRT could not run the network\n");
    ok = false;
  }
  ok = ok && Succeeded(cudaMemcpyAsync(host_output_.data(), device_output_->ptr(), host_output_.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream),
                       "copying the descriptors from the GPU");
  // Always sync, even on failure: queued work may still read the input buffer.
  ok = Succeeded(cudaStreamSynchronize(stream), "running the network") && ok;
  if (!ok) {
    return false;
  }
  for (const float value : host_output_) {
    if (!std::isfinite(value)) {
      TraceError("VPR AnyLoc: the network produced a descriptor that is not finite\n");
      return false;
    }
  }
  patches.assign(host_output_.begin(), host_output_.end());
  return true;
}

}  // namespace cuvslam::slam::vpr
