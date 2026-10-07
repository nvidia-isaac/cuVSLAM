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
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "common/log.h"
#include "cuda_modules/cuda_helper.h"

namespace cuvslam::slam::vpr {

namespace {

constexpr char kInputName[] = "image";
constexpr char kOutputName[] = "patch_descriptors";

const std::string kTensorRtVersion = std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
                                     std::to_string(NV_TENSORRT_PATCH);

/// TensorRT keeps one logger per process and only warns when it is handed a different one, so every network shares
/// this one. It forwards errors and warnings to the cuVSLAM log, and remembers the last error of the calling thread:
/// that is how a failed deserialization learns why it failed.
class TensorRtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      try {
        last_error_ = message;
      } catch (...) {
      }
      TraceError("VPR AnyLoc: TensorRT: %s\n", message);
    } else if (severity == Severity::kWARNING) {
      TraceWarning("VPR AnyLoc: TensorRT: %s\n", message);
    }
  }

  /// The last error TensorRT reported on this thread since the previous call.
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
  // The error stays behind for the next cudaGetLastError(), which is some other component's to read.
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

/// True when `name` is one of the engine's inputs or outputs. Asking an engine about a tensor it does not have is
/// an error TensorRT logs, so every lookup goes through this first.
bool HasTensor(const nvinfer1::ICudaEngine& engine, const char* name) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    if (std::string(engine.getIOTensorName(i)) == name) {
      return true;
    }
  }
  return false;
}

/// True when the engine has `name` as a static fp32 tensor in linear layout, in the given direction and of the given
/// rank, with a leading batch dimension of 1 and positive extents elsewhere.
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

std::vector<char> ReadFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (!file.good() && !file.eof()) {
    bytes.clear();
  }
  if (bytes.empty()) {
    throw std::runtime_error("VPR AnyLoc: could not read the TensorRT engine \"" + path + "\".");
  }
  return bytes;
}

}  // namespace

struct AnyLocNetwork::Impl {
  // Declared in the order they have to be created; members are destroyed in reverse, so the execution context goes
  // before the engine and the engine before the runtime, as TensorRT requires.
  cuda::Stream stream{false, ICudaStreamProvider::StreamType::Slam};
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::optional<cuda::GPUOnlyArray<float>> device_input;
  std::optional<cuda::GPUOnlyArray<float>> device_output;
  // Pinned, so the copies to and from the GPU run at full speed and asynchronously.
  std::vector<float, cuda::HostAllocator<float>> host_input;
  std::vector<float, cuda::HostAllocator<float>> host_output;
  int width = 0;
  int height = 0;
  int patches = 0;
  int dim = 0;
};

AnyLocNetwork::AnyLocNetwork(const std::string& engine_path) : impl_(std::make_unique<Impl>()) {
  Impl& impl = *impl_;
  const std::vector<char> plan = ReadFile(engine_path);

  TensorRtLogger& logger = Logger();
  logger.TakeLastError();
  impl.runtime.reset(nvinfer1::createInferRuntime(logger));
  if (!impl.runtime) {
    throw std::runtime_error("VPR AnyLoc: could not create the TensorRT " + kTensorRtVersion + " runtime.");
  }
  impl.engine.reset(impl.runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!impl.engine) {
    throw std::runtime_error("VPR AnyLoc: \"" + engine_path + "\" is not a TensorRT engine that TensorRT " +
                             kTensorRtVersion + " can run on this GPU (" + logger.TakeLastError() +
                             "). An engine only loads into the TensorRT version and onto the GPU architecture that "
                             "built it: rebuild it with the anyloc_engine CMake target or anyloc_engine_builder.");
  }

  const nvinfer1::ICudaEngine& engine = *impl.engine;
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
    throw std::runtime_error("VPR AnyLoc: \"" + engine_path +
                             "\" is not the DINOv2 patch descriptor engine AnyLoc runs, which takes input image fp32 "
                             "[1, 3, H, W] and gives output patch_descriptors fp32 [1, N, D]; it has " +
                             found + ".");
  }
  const nvinfer1::Dims input_dims = engine.getTensorShape(kInputName);
  const nvinfer1::Dims output_dims = engine.getTensorShape(kOutputName);
  impl.height = static_cast<int>(input_dims.d[2]);
  impl.width = static_cast<int>(input_dims.d[3]);
  impl.patches = static_cast<int>(output_dims.d[1]);
  impl.dim = static_cast<int>(output_dims.d[2]);

  impl.context.reset(impl.engine->createExecutionContext());
  if (!impl.context) {
    throw std::runtime_error("VPR AnyLoc: could not create an execution context for \"" + engine_path + "\" (" +
                             logger.TakeLastError() + ").");
  }
  impl.host_input.resize(static_cast<size_t>(3) * impl.height * impl.width);
  impl.host_output.resize(static_cast<size_t>(impl.patches) * impl.dim);
  impl.device_input.emplace(impl.host_input.size());
  impl.device_output.emplace(impl.host_output.size());
  if (!impl.context->setTensorAddress(kInputName, impl.device_input->ptr()) ||
      !impl.context->setTensorAddress(kOutputName, impl.device_output->ptr())) {
    throw std::runtime_error("VPR AnyLoc: could not bind the inputs and outputs of \"" + engine_path + "\".");
  }

  // The first inference initializes what TensorRT loads lazily; better here than on the first keyframe.
  std::vector<float> warm_up;
  if (!Run(warm_up)) {
    throw std::runtime_error("VPR AnyLoc: the first inference of \"" + engine_path + "\" failed, see the log.");
  }
}

AnyLocNetwork::~AnyLocNetwork() = default;

int AnyLocNetwork::input_width() const { return impl_->width; }

int AnyLocNetwork::input_height() const { return impl_->height; }

int AnyLocNetwork::patch_count() const { return impl_->patches; }

int AnyLocNetwork::descriptor_dim() const { return impl_->dim; }

float* AnyLocNetwork::input() { return impl_->host_input.data(); }

bool AnyLocNetwork::Run(std::vector<float>& patches) {
  Impl& impl = *impl_;
  cudaStream_t stream = impl.stream.get_stream();
  bool ok = Succeeded(cudaMemcpyAsync(impl.device_input->ptr(), impl.host_input.data(),
                                      impl.host_input.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
                      "copying the image to the GPU");
  if (ok && !impl.context->enqueueV3(stream)) {
    TraceError("VPR AnyLoc: TensorRT could not run the network\n");
    ok = false;
  }
  ok = ok && Succeeded(cudaMemcpyAsync(impl.host_output.data(), impl.device_output->ptr(),
                                       impl.host_output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                       "copying the descriptors from the GPU");
  // Synchronized on every path: whatever was queued still reads the input buffer the caller writes next.
  ok = Succeeded(cudaStreamSynchronize(stream), "running the network") && ok;
  if (!ok) {
    return false;
  }
  for (const float value : impl.host_output) {
    if (!std::isfinite(value)) {
      TraceError("VPR AnyLoc: the network produced a descriptor that is not finite\n");
      return false;
    }
  }
  patches.assign(impl.host_output.begin(), impl.host_output.end());
  return true;
}

}  // namespace cuvslam::slam::vpr
