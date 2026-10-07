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

// anyloc_engine_builder, step 2 of the AnyLoc model build (tools/anyloc_model/CMakeLists.txt).
//
// Builds a TensorRT engine of the ONNX model export_dinov2_onnx.py writes, for the GPU it runs on, and only writes
// it once it reproduces AnyLoc's fp32 descriptors of the reference input the export saved:
//
//   anyloc_engine_builder --onnx=<model.onnx> --engine=<out.engine>
//                         [--reference_input=<input.npy> --reference_output=<output.npy>]
//
// The engine runs in FP16 except for the descriptor head, everything after the last matrix multiply: the bias of the
// value projection and the L2 normalization, whose sum of squares is the one place an FP16 overflow could zero a
// descriptor. The layer norms are deliberately not pinned: TensorRT already computes them in FP32 inside, and pinning
// them makes it run every transformer block it fuses around them in FP32, which doubles the size and the latency of
// the engine for no measurable gain. cuBLAS, cuBLASLt and cuDNN tactics are left out, so running the engine takes
// libnvinfer and the CUDA driver, nothing else.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cnpy.h"
#include "gflags/gflags.h"

DEFINE_string(onnx, "", "ONNX model written by export_dinov2_onnx.py");
DEFINE_string(engine, "", "Where to write the TensorRT engine");
DEFINE_string(reference_input, "", "Network input to validate the engine on, .npy, fp32 [1, 3, H, W]");
DEFINE_string(reference_output, "", "AnyLoc's fp32 descriptors of that input, .npy, fp32 [1, N, D]");
DEFINE_bool(fp16, true, "Run in FP16, keeping the descriptor head in FP32");
DEFINE_int32(workspace_mb, 1024, "Most scratch GPU memory TensorRT may use per layer while building, in MiB");
DEFINE_int32(expect_sm, 0,
             "Compute capability the engine is meant for, e.g. 89 for an RTX 4090; refuses to build on another GPU, "
             "since TensorRT builds for the GPU it runs on. 0 builds for whatever CUDA device 0 is");

namespace {

// Lowest cosine similarity accepted between the engine's descriptor of a patch and the fp32 reference, averaged over
// all patches and for the worst one. The AnyLoc score is a cosine of VLAD vectors summed from these descriptors, so
// agreement at this level leaves place recognition as it is in fp32.
constexpr double kMinMeanCosine = 0.999;
constexpr double kMinCosine = 0.99;

constexpr char kInputName[] = "image";
constexpr char kOutputName[] = "patch_descriptors";

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "TensorRT: %s\n", message);
    }
  }
};

/// The validation data the export saved: the network input, and AnyLoc's fp32 descriptors of it.
struct Reference {
  cnpy::NpyArray input;
  cnpy::NpyArray output;
};

bool Succeeded(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
  }
  return true;
}

struct DeviceBuffer {
  explicit DeviceBuffer(size_t bytes) {
    if (!Succeeded(cudaMalloc(&data, bytes), "cudaMalloc")) {
      data = nullptr;
    }
  }
  ~DeviceBuffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* data = nullptr;
};

// Keeps every layer after the last matrix multiply in FP32. Returns how many layers that is.
int KeepDescriptorHeadInFp32(nvinfer1::INetworkDefinition& network) {
  int last_matrix_multiply = -1;
  for (int i = 0; i < network.getNbLayers(); ++i) {
    if (network.getLayer(i)->getType() == nvinfer1::LayerType::kMATRIX_MULTIPLY) {
      last_matrix_multiply = i;
    }
  }
  int pinned = 0;
  for (int i = last_matrix_multiply + 1; i < network.getNbLayers(); ++i) {
    nvinfer1::ILayer* layer = network.getLayer(i);
    if (layer->getType() == nvinfer1::LayerType::kCONSTANT) {
      continue;
    }
    bool computes_floats = layer->getNbOutputs() > 0;
    for (int j = 0; j < layer->getNbOutputs(); ++j) {
      computes_floats = computes_floats && layer->getOutput(j)->getType() == nvinfer1::DataType::kFLOAT;
    }
    if (!computes_floats) {
      continue;  // shape arithmetic
    }
    layer->setPrecision(nvinfer1::DataType::kFLOAT);
    for (int j = 0; j < layer->getNbOutputs(); ++j) {
      layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
    }
    ++pinned;
  }
  return pinned;
}

std::unique_ptr<nvinfer1::IHostMemory> BuildEngine(Logger& logger) {
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return nullptr;
  }
#if NV_TENSORRT_MAJOR >= 10
  const nvinfer1::NetworkDefinitionCreationFlags flags = 0;  // explicit batch is the only mode left
#else
  const nvinfer1::NetworkDefinitionCreationFlags flags =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  if (!parser->parseFromFile(FLAGS_onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      std::fprintf(stderr, "%s: %s\n", FLAGS_onnx.c_str(), parser->getError(i)->desc());
    }
    return nullptr;
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<size_t>(FLAGS_workspace_mb) << 20);
#if NV_TENSORRT_MAJOR < 10
  // TensorRT 10 has these off already. With any of them on, an engine may pick a tactic that needs the library at
  // run time, which libcuvslam does not otherwise depend on.
  nvinfer1::TacticSources sources = config->getTacticSources();
  for (const nvinfer1::TacticSource source :
       {nvinfer1::TacticSource::kCUBLAS, nvinfer1::TacticSource::kCUBLAS_LT, nvinfer1::TacticSource::kCUDNN}) {
    sources &= ~(1U << static_cast<uint32_t>(source));
  }
  config->setTacticSources(sources);
#endif
  if (FLAGS_fp16) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    const int pinned = KeepDescriptorHeadInFp32(*network);
    std::printf("FP16, with %d of %d layers kept in FP32\n", pinned, network->getNbLayers());
  }
  return std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
}

int64_t Volume(const nvinfer1::Dims& dims) {
  int64_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    volume *= dims.d[i];
  }
  return volume;
}

std::string ShapeString(const std::vector<size_t>& shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
  }
  return text + "]";
}

bool SameShape(const nvinfer1::Dims& dims, const std::vector<size_t>& shape) {
  if (static_cast<size_t>(dims.nbDims) != shape.size()) {
    return false;
  }
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0 || static_cast<size_t>(dims.d[i]) != shape[i]) {
      return false;
    }
  }
  return true;
}

// Reads the reference arrays, before the build rather than after it, so that a wrong path costs no build.
std::optional<Reference> LoadReference() {
  try {
    Reference reference{cnpy::npy_load(FLAGS_reference_input), cnpy::npy_load(FLAGS_reference_output)};
    if (reference.input.word_size != sizeof(float) || reference.output.word_size != sizeof(float) ||
        reference.input.fortran_order || reference.output.fortran_order) {
      std::fprintf(stderr, "the reference arrays must be C ordered float32\n");
      return std::nullopt;
    }
    return reference;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "could not read the reference arrays: %s\n", error.what());
    return std::nullopt;
  }
}

// TensorRT builds for the GPU it runs on, so a GPU other than the one the engine's name promises makes an engine that
// only runs there.
bool IsExpectedGpu() {
  cudaDeviceProp properties;
  if (!Succeeded(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")) {
    return false;
  }
  const int sm = properties.major * 10 + properties.minor;
  if (FLAGS_expect_sm > 0 && sm != FLAGS_expect_sm) {
    std::fprintf(stderr, "the engine is meant for sm_%d, but CUDA device 0 is %s, sm_%d\n", FLAGS_expect_sm,
                 properties.name, sm);
    return false;
  }
  return true;
}

// Runs the engine on the reference input and compares each patch descriptor with the reference one.
bool Validate(const nvinfer1::IHostMemory& plan, Logger& logger, const Reference& reference) {
  const cnpy::NpyArray& input = reference.input;
  const cnpy::NpyArray& expected = reference.output;
  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine) {
    return false;
  }
  std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
  if (!context) {
    std::fprintf(stderr, "could not create an execution context for the engine\n");
    return false;
  }
  const nvinfer1::Dims input_dims = engine->getTensorShape(kInputName);
  const nvinfer1::Dims output_dims = engine->getTensorShape(kOutputName);
  if (!SameShape(input_dims, input.shape) || !SameShape(output_dims, expected.shape) || output_dims.nbDims != 3) {
    std::fprintf(stderr, "the engine's %s and %s do not have the shapes of the reference arrays, %s and %s\n",
                 kInputName, kOutputName, ShapeString(input.shape).c_str(), ShapeString(expected.shape).c_str());
    return false;
  }

  const size_t input_bytes = static_cast<size_t>(Volume(input_dims)) * sizeof(float);
  const size_t output_count = static_cast<size_t>(Volume(output_dims));
  DeviceBuffer device_input(input_bytes);
  DeviceBuffer device_output(output_count * sizeof(float));
  cudaStream_t stream = nullptr;
  if (device_input.data == nullptr || device_output.data == nullptr ||
      !Succeeded(cudaStreamCreate(&stream), "cudaStreamCreate")) {
    return false;
  }
  std::vector<float> actual(output_count);
  bool ok =
      context->setTensorAddress(kInputName, device_input.data) &&
      context->setTensorAddress(kOutputName, device_output.data) &&
      Succeeded(cudaMemcpyAsync(device_input.data, input.data<float>(), input_bytes, cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync") &&
      context->enqueueV3(stream) &&
      Succeeded(cudaMemcpyAsync(actual.data(), device_output.data, output_count * sizeof(float), cudaMemcpyDeviceToHost,
                                stream),
                "cudaMemcpyAsync") &&
      Succeeded(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  // What one inference costs on this GPU, for the log; the first run above included the warm up.
  constexpr int kTimedRuns = 20;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; ok && i < kTimedRuns; ++i) {
    ok = context->enqueueV3(stream);
  }
  ok = ok && Succeeded(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  const double milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / kTimedRuns;
  cudaStreamDestroy(stream);
  if (!ok) {
    std::fprintf(stderr, "running the engine failed\n");
    return false;
  }

  const int patches = output_dims.d[1];
  const int dim = output_dims.d[2];
  const float* expected_values = expected.data<float>();
  double cosine_sum = 0.;
  double cosine_min = 1.;
  double max_abs_diff = 0.;
  for (int p = 0; p < patches; ++p) {
    double dot = 0.;
    double norm_a = 0.;
    double norm_b = 0.;
    for (int k = 0; k < dim; ++k) {
      const double a = actual[static_cast<size_t>(p) * dim + k];
      const double b = expected_values[static_cast<size_t>(p) * dim + k];
      if (!std::isfinite(a)) {
        std::fprintf(stderr, "the engine's descriptor of patch %d is not finite\n", p);
        return false;
      }
      dot += a * b;
      norm_a += a * a;
      norm_b += b * b;
      max_abs_diff = std::max(max_abs_diff, std::abs(a - b));
    }
    const double cosine = dot / std::max(std::sqrt(norm_a * norm_b), 1e-12);
    cosine_sum += cosine;
    cosine_min = std::min(cosine_min, cosine);
  }
  const double cosine_mean = cosine_sum / patches;
  std::printf(
      "engine vs fp32 reference over %d patches: cosine mean %.6f, min %.6f, max abs diff %.3e; "
      "inference %.2f ms\n",
      patches, cosine_mean, cosine_min, max_abs_diff, milliseconds);
  if (!(cosine_mean >= kMinMeanCosine) || !(cosine_min >= kMinCosine)) {
    std::fprintf(stderr, "the engine does not reproduce the reference (needs cosine mean >= %.3f and min >= %.3f)\n",
                 kMinMeanCosine, kMinCosine);
    return false;
  }
  return true;
}

// Writes `plan` under a temporary name and renames it into place, so a build interrupted halfway never leaves a
// truncated engine behind that looks up to date.
bool WriteAtomically(const nvinfer1::IHostMemory& plan, const std::filesystem::path& path) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  std::error_code error;
  {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(static_cast<const char*>(plan.data()), static_cast<std::streamsize>(plan.size()));
    file.close();  // what the stream still buffers is only written, and can only fail, here
    if (file.fail()) {
      std::fprintf(stderr, "could not write %s\n", temporary.c_str());
      std::filesystem::remove(temporary, error);
      return false;
    }
  }
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::fprintf(stderr, "could not move %s to %s: %s\n", temporary.c_str(), path.c_str(), error.message().c_str());
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Builds the AnyLoc TensorRT engine from the ONNX model export_dinov2_onnx.py writes.\n"
      "  anyloc_engine_builder --onnx=<model.onnx> --engine=<out.engine> "
      "[--reference_input=<input.npy> --reference_output=<output.npy>]");
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove flags = */ true);
  if (FLAGS_onnx.empty() || FLAGS_engine.empty()) {
    std::fprintf(stderr, "--onnx and --engine are required\n");
    return 2;
  }
  if (FLAGS_reference_input.empty() != FLAGS_reference_output.empty()) {
    std::fprintf(stderr, "--reference_input and --reference_output go together\n");
    return 2;
  }

  std::optional<Reference> reference;
  if (!FLAGS_reference_input.empty()) {
    reference = LoadReference();
    if (!reference) {
      return 1;
    }
  }
  if (!IsExpectedGpu()) {
    return 1;
  }

  Logger logger;
  const std::unique_ptr<nvinfer1::IHostMemory> plan = BuildEngine(logger);
  if (!plan || plan->size() == 0) {
    std::fprintf(stderr, "building the engine from %s failed\n", FLAGS_onnx.c_str());
    return 1;
  }
  if (!reference) {
    std::printf("no reference given, the engine is not validated\n");
  } else if (!Validate(*plan, logger, *reference)) {
    std::fprintf(stderr, "not writing %s\n", FLAGS_engine.c_str());
    return 1;
  }
  if (!WriteAtomically(*plan, FLAGS_engine)) {
    return 1;
  }
  std::printf("wrote %s (%.1f MB)\n", FLAGS_engine.c_str(), static_cast<double>(plan->size()) / 1e6);
  return 0;
}
