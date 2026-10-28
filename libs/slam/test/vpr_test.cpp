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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "common/vector_3t.h"
#include "slam/async_slam/async_slam.h"
#include "slam/map/database/lmdb_slam_database.h"
#include "slam/slam/slam.h"
#include "slam/vpr/ivpr.h"
#include "slam/vpr/vpr_image.h"
#include "slam/vpr/vpr_map.h"
#include "slam/vpr/vpr_sof_image.h"
#include "slam/vpr/vpr_vocabulary.h"
#include "sof/image_context.h"

#ifdef USE_CUDA
#include "cuda_modules/cuda_helper.h"
#endif

#ifdef USE_TENSORRT
#include <NvInfer.h>

#include "cnpy.h"
#include "slam/vpr/anyloc_network.h"
#include "slam/vpr/vpr_anyloc.h"
#endif

namespace test::vpr {
using namespace cuvslam;
using namespace cuvslam::slam;
using namespace cuvslam::slam::vpr;

namespace {

/// Random `block` x `block` tiles; each seed stands in for a different place.
VprImage MakeBlockyImage(int width, int height, int block, uint32_t seed) {
  VprImage image;
  image.width = width;
  image.height = height;
  image.row.resize(static_cast<size_t>(width) * height);

  const int tiles_x = (width + block - 1) / block;
  const int tiles_y = (height + block - 1) / block;
  std::mt19937 rng(seed);
  std::vector<uint8_t> tiles(static_cast<size_t>(tiles_x) * tiles_y);
  for (uint8_t& tile : tiles) {
    tile = static_cast<uint8_t>(rng() % 256u);
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image.row[static_cast<size_t>(y) * width + x] = tiles[static_cast<size_t>(y / block) * tiles_x + x / block];
    }
  }
  return image;
}

VprImage MakeConstantImage(int width, int height, uint8_t value) {
  VprImage image;
  image.width = width;
  image.height = height;
  image.row.assign(static_cast<size_t>(width) * height, value);
  return image;
}

/// Stand-in for AnyLoc, which needs TensorRT, a GPU and VprOptions::min_training_keyframes frames.
class StandInVpr : public IVpr {
public:
  static constexpr int kGrid = 8;
  static constexpr float kThreshold = 0.5f;

  const char* Name() const override { return "StandIn"; }
  void AddFrame(KeyFrameId node_id, const VprImage& image) override { entries_[node_id] = Describe(image); }
  bool HasNode(KeyFrameId node_id) const override { return entries_.count(node_id) != 0; }
  void RemoveNode(KeyFrameId node_id) override { entries_.erase(node_id); }
  size_t Size() const override { return entries_.size(); }
  void Finalize() override {}

  VprMatch Query(const VprImage& image) override {
    const std::vector<VprMatch> best = QueryTopK(image, 1);
    return best.empty() ? VprMatch{} : best.front();
  }

  /// All entries, best first; as in AnyLoc, only the best must pass kThreshold.
  std::vector<VprMatch> QueryTopK(const VprImage& image, size_t max_results) override {
    std::vector<VprMatch> matches;
    const Descriptor query = Describe(image);
    for (const auto& [node_id, descriptor] : entries_) {
      VprMatch match;
      match.found = true;
      match.node_id = node_id;
      match.score = Score(query, descriptor);
      matches.push_back(match);
    }
    std::sort(matches.begin(), matches.end(), [](const VprMatch& a, const VprMatch& b) { return a.score > b.score; });
    if (max_results == 0 || matches.empty() || matches.front().score < kThreshold) {
      return {};
    }
    matches.resize(std::min(max_results, matches.size()));
    return matches;
  }

  void Serialize(BlobWriter& writer) override {
    writer.write(kMagic);
    writer.write(static_cast<uint64_t>(entries_.size()));
    for (const auto& [node_id, descriptor] : entries_) {
      writer.write(static_cast<uint64_t>(node_id));
      writer.write(descriptor.data(), sizeof(float) * descriptor.size());
    }
  }

  bool Deserialize(const BlobReader& reader) override {
    uint32_t magic = 0;
    uint64_t count = 0;
    if (!reader.read(magic) || magic != kMagic || !reader.read(count) ||
        count > reader.remaining() / (sizeof(uint64_t) + sizeof(Descriptor))) {
      return false;
    }
    std::map<KeyFrameId, Descriptor> entries;
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t node_id = 0;
      Descriptor descriptor{};
      if (!reader.read(node_id) || !reader.read(descriptor.data(), sizeof(float) * descriptor.size())) {
        return false;
      }
      entries[static_cast<KeyFrameId>(node_id)] = descriptor;
    }
    entries_ = std::move(entries);
    return true;
  }

private:
  using Descriptor = std::array<float, kGrid * kGrid>;
  static constexpr uint32_t kMagic = 0x4E495453;  // "STIN"

  static Descriptor Describe(const VprImage& image) {
    Descriptor cells{};
    std::array<int, kGrid * kGrid> counts{};
    for (int y = 0; y < image.height; ++y) {
      for (int x = 0; x < image.width; ++x) {
        const int cell = (y * kGrid / image.height) * kGrid + x * kGrid / image.width;
        cells[cell] += static_cast<float>(image.row[static_cast<size_t>(y) * image.width + x]);
        ++counts[cell];
      }
    }
    float mean = 0.f;
    for (size_t i = 0; i < cells.size(); ++i) {
      cells[i] = counts[i] > 0 ? cells[i] / static_cast<float>(counts[i]) : 0.f;
      mean += cells[i];
    }
    mean /= static_cast<float>(cells.size());
    float norm = 0.f;
    for (float& cell : cells) {
      cell -= mean;
      norm += cell * cell;
    }
    norm = std::sqrt(norm);
    for (float& cell : cells) {
      cell = norm > 0.f ? cell / norm : 0.f;
    }
    return cells;
  }

  static float Score(const Descriptor& a, const Descriptor& b) {
    float dot = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
      dot += a[i] * b[i];
    }
    return std::clamp(dot, 0.f, 1.f);
  }

  std::map<KeyFrameId, Descriptor> entries_;
};

std::unique_ptr<IVpr> StandInBackend() { return std::make_unique<StandInVpr>(); }

std::unique_ptr<VprMap> MakeStandInMap() { return std::make_unique<VprMap>(VprOptions{}, StandInBackend()); }

Isometry3T MakePose(float x, float y, float z) {
  Isometry3T pose = Isometry3T::Identity();
  pose.translation() = Vector3T(x, y, z);
  return pose;
}

uint8_t PixelAt(const VprImage& image, int x, int y) { return image.row[static_cast<size_t>(y) * image.width + x]; }

/// Image context filled like the host pipeline does; `image` need not outlive it.
sof::ImageContextPtr MakeCpuContext(const VprImage& image, FrameId frame_id = 0) {
  const ImageShape shape{image.width, image.height};
  auto context = std::make_shared<sof::ImageContext>(shape, false, false);
  ImageMeta meta;
  meta.shape = shape;
  meta.frame_id = frame_id;
  context->set_image_meta(meta);

  ImageSource source;
  source.data = const_cast<uint8_t*>(image.row.data());
  source.type = ImageSource::U8;
  source.memory_type = ImageSource::Host;
  source.image_encoding = ImageEncoding::MONO8;
  EXPECT_TRUE(context->build_cpu_image_pyramid(source, false));
  return context;
}

}  // namespace

TEST(VprImageMake, Mono8RoundTrip) {
  constexpr int kWidth = 7;
  constexpr int kHeight = 5;
  std::vector<uint8_t> buffer(kWidth * kHeight);
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = static_cast<uint8_t>(i * 3 + 1);
  }

  const VprImage image = MakeVprImage(buffer.data(), kWidth, kHeight, 0, VprPixelFormat::kMono8);
  ASSERT_FALSE(image.Empty());
  EXPECT_EQ(image.width, kWidth);
  EXPECT_EQ(image.height, kHeight);
  EXPECT_EQ(image.row, buffer);
}

TEST(VprImageMake, Rgb8UsesBt601Weights) {
  // (77 R + 150 G + 29 B) >> 8 on the primaries, white, black and mid gray.
  const std::vector<uint8_t> buffer = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 128, 128, 128};
  const std::vector<uint8_t> expected = {76, 149, 28, 255, 0, 128};

  const VprImage image = MakeVprImage(buffer.data(), 6, 1, 0, VprPixelFormat::kRgb8);
  ASSERT_FALSE(image.Empty());
  EXPECT_EQ(image.width, 6);
  EXPECT_EQ(image.height, 1);
  EXPECT_EQ(image.row, expected);
}

TEST(VprImageMake, FloatIsRoundedAndClamped) {
  const std::vector<float> buffer = {-5.f, 0.4f, 12.6f, 254.6f, 300.f};
  const std::vector<uint8_t> expected = {0, 0, 13, 255, 255};

  const VprImage image = MakeVprImage(buffer.data(), 5, 1, 0, VprPixelFormat::kFloat);
  ASSERT_FALSE(image.Empty());
  EXPECT_EQ(image.row, expected);
}

TEST(VprImageMake, PitchSkipsRowPadding) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 3;
  constexpr int kPitch = kWidth + 5;
  std::vector<uint8_t> buffer(kPitch * kHeight, 0xAB);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      buffer[y * kPitch + x] = static_cast<uint8_t>(10 * y + x);
    }
  }

  const VprImage image = MakeVprImage(buffer.data(), kWidth, kHeight, kPitch, VprPixelFormat::kMono8);
  ASSERT_FALSE(image.Empty());
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      EXPECT_EQ(PixelAt(image, x, y), static_cast<uint8_t>(10 * y + x)) << "at " << x << "," << y;
    }
  }
}

TEST(VprImageMake, RejectsUnreadableBuffers) {
  const std::vector<uint8_t> buffer(16, 7);

  EXPECT_TRUE(MakeVprImage(nullptr, 4, 4, 0, VprPixelFormat::kMono8).Empty());
  EXPECT_TRUE(MakeVprImage(buffer.data(), 0, 4, 0, VprPixelFormat::kMono8).Empty());
  EXPECT_TRUE(MakeVprImage(buffer.data(), 4, 0, 0, VprPixelFormat::kMono8).Empty());
  EXPECT_TRUE(MakeVprImage(buffer.data(), -4, 4, 0, VprPixelFormat::kMono8).Empty());
  // A pitch shorter than a row would read into the next row.
  EXPECT_TRUE(MakeVprImage(buffer.data(), 4, 4, 3, VprPixelFormat::kMono8).Empty());
}

TEST(VprImageResize, ProducesTheRequestedSize) {
  const VprImage image = MakeBlockyImage(64, 64, 8, 5);

  const VprImage out = Resize(image, 5, 3);
  ASSERT_FALSE(out.Empty());
  EXPECT_EQ(out.width, 5);
  EXPECT_EQ(out.height, 3);
  EXPECT_EQ(out.row.size(), 15u);

  EXPECT_TRUE(Resize(image, 0, 3).Empty());
  EXPECT_TRUE(Resize(VprImage{}, 5, 3).Empty());
}

TEST(VprImageResize, MatchingSizeIsIdentity) {
  const VprImage image = MakeBlockyImage(20, 10, 5, 6);

  const VprImage out = Resize(image, image.width, image.height);
  EXPECT_EQ(out.width, image.width);
  EXPECT_EQ(out.height, image.height);
  EXPECT_EQ(out.row, image.row);
}

TEST(VprImageResize, DownscalesByMoreThanTwo) {
  EXPECT_EQ(Resize(MakeConstantImage(64, 64, 91), 8, 8).row, std::vector<uint8_t>(64, 91));

  // A vertical edge on a block boundary must not bleed through the 8x box pass.
  VprImage split = MakeConstantImage(64, 64, 0);
  for (int y = 0; y < split.height; ++y) {
    for (int x = 32; x < split.width; ++x) {
      split.row[static_cast<size_t>(y) * split.width + x] = 255;
    }
  }

  const VprImage out = Resize(split, 8, 8);
  ASSERT_EQ(out.width, 8);
  ASSERT_EQ(out.height, 8);
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      EXPECT_EQ(PixelAt(out, x, y), x < 4 ? 0 : 255) << "at " << x << "," << y;
    }
  }
}

TEST(VprImageResize, UpscaleInterpolatesBetweenTheSourcePixels) {
  // Plain bilinear: outer pixels clamp to the ends, inner ones land at 1/4 and 3/4.
  VprImage image;
  image.width = 2;
  image.height = 1;
  image.row = {0, 200};

  const VprImage out = Resize(image, 4, 1);
  ASSERT_FALSE(out.Empty());
  EXPECT_EQ(out.row, std::vector<uint8_t>({0, 50, 150, 200}));
}

TEST(VprFactory, DisabledHasNoBackend) { EXPECT_EQ(CreateVpr(VprOptions{}), nullptr); }

TEST(VprFactory, RefusesAnUnavailableBackend) {
  // Throws either way: compiled out AnyLoc is unavailable, compiled in it has no model file.
  VprOptions options;
  options.enabled = true;
  options.model_path = "/nonexistent/anyloc.engine";
  EXPECT_THROW(CreateVpr(options), std::runtime_error);
}

TEST(VprVocabularyFitter, NeedsAnEnabledBackendWithoutAVocabularyFile) {
  VprOptions options;
  EXPECT_THROW(VocabularyFitter{options}, std::invalid_argument);
  options.enabled = true;
  options.vocabulary_path = "/nonexistent/anyloc.vocabulary";
  EXPECT_THROW(VocabularyFitter{options}, std::invalid_argument);
  // As CreateVpr(): compiled out AnyLoc is unavailable, compiled in it has no model file.
  options.vocabulary_path.clear();
  options.model_path = "/nonexistent/anyloc.engine";
  EXPECT_THROW(VocabularyFitter{options}, std::runtime_error);
}

TEST(VprVocabularyFitter, ReadsNoMissingEmptyOrHugeFile) {
  EXPECT_THROW(ReadVocabularyFile("/nonexistent/anyloc.vocabulary"), std::runtime_error);
  const std::filesystem::path file = std::filesystem::temp_directory_path() /
                                     ("cuvslam_vpr_file_" + std::to_string(std::random_device{}()) + ".vocabulary");
  std::ofstream(file, std::ios::binary | std::ios::trunc).close();
  EXPECT_THROW(ReadVocabularyFile(file.string()), std::runtime_error);
  // A wrong path, like a map file, is refused unread: sparse, far larger than any vocabulary.
  std::filesystem::resize_file(file, uintmax_t{1} << 30);
  EXPECT_THROW(ReadVocabularyFile(file.string()), std::runtime_error);
  std::filesystem::remove(file);
}

namespace {

/// Implements only Query(), to exercise IVpr's default QueryTopK().
class OneAnswerVpr : public IVpr {
public:
  explicit OneAnswerVpr(const VprMatch& answer) : answer_(answer) {}

  const char* Name() const override { return "OneAnswer"; }
  void AddFrame(KeyFrameId, const VprImage&) override {}
  bool HasNode(KeyFrameId) const override { return false; }
  void RemoveNode(KeyFrameId) override {}
  size_t Size() const override { return 0; }
  void Finalize() override {}
  VprMatch Query(const VprImage&) override {
    ++queries;
    return answer_;
  }
  void Serialize(BlobWriter&) override {}
  bool Deserialize(const BlobReader&) override { return false; }

  int queries = 0;

private:
  VprMatch answer_;
};

}  // namespace

TEST(VprBackendInterface, DefaultQueryTopKReturnsTheWinnerAlone) {
  VprMatch answer;
  answer.found = true;
  answer.node_id = 12;
  answer.score = 0.9f;
  OneAnswerVpr backend(answer);

  const std::vector<VprMatch> top = backend.QueryTopK(VprImage{}, 5);
  ASSERT_EQ(top.size(), 1u);
  EXPECT_EQ(top.front().node_id, 12u);
  EXPECT_EQ(top.front().score, 0.9f);

  // Asking for no results must not reach the backend at all.
  EXPECT_TRUE(backend.QueryTopK(VprImage{}, 0).empty());
  EXPECT_EQ(backend.queries, 1);

  OneAnswerVpr silent{VprMatch{}};
  EXPECT_TRUE(silent.QueryTopK(VprImage{}, 3).empty());
}

#ifdef USE_TENSORRT

namespace {

/// The override is not CUVSLAM_ANYLOC_ENGINE, which every build's cuvslam_vars.sh exports.
std::string AnyLocEnginePath() {
  const char* variable = std::getenv("CUVSLAM_TEST_ANYLOC_ENGINE");
#ifdef CUVSLAM_ANYLOC_ENGINE_PATH
  const std::string path = variable != nullptr ? variable : CUVSLAM_ANYLOC_ENGINE_PATH;
#else
  const std::string path = variable != nullptr ? variable : "";
#endif
  std::error_code ec;
  return !path.empty() && std::filesystem::is_regular_file(path, ec) ? path : std::string{};
}

VprOptions AnyLocOptions(const std::string& engine_path) {
  VprOptions options;
  options.enabled = true;
  options.model_path = engine_path;
  return options;
}

/// Per-process temp folder, removed at exit, so concurrent test runs never share files.
const std::filesystem::path& ScratchFolder() {
  struct Folder {
    Folder()
        : path(std::filesystem::temp_directory_path() /
               ("cuvslam_vpr_anyloc_test_" + std::to_string(std::random_device{}()))) {
      std::filesystem::create_directories(path);
    }
    ~Folder() {
      std::error_code ec;
      std::filesystem::remove_all(path, ec);
    }
    std::filesystem::path path;
  };
  static const Folder folder;
  return folder.path;
}

struct StandInSpec {
  const char* input_name = "image";
  nvinfer1::DataType input_type = nvinfer1::DataType::kFLOAT;
  bool dynamic_batch = false;
  int size = 28;  ///< side of the square input
  int tile = 7;   ///< side of a tile, one descriptor each
};

class StandInLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      std::fprintf(stderr, "TensorRT: %s\n", message);
    }
  }
};

/// Engine with AnyLoc's interface whose descriptors are raw tile pixels; returns its path.
std::string BuildStandInEngine(const StandInSpec& spec, const std::string& name) {
  // One shared builder: creating one costs seconds, building these networks milliseconds.
  static StandInLogger logger;
  static const std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return {};  // no CUDA device to build for
  }
#if NV_TENSORRT_MAJOR >= 10
  const nvinfer1::NetworkDefinitionCreationFlags flags = 0;
#else
  const nvinfer1::NetworkDefinitionCreationFlags flags =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
  const int grid = spec.size / spec.tile;
  const int batch = spec.dynamic_batch ? -1 : 1;
  nvinfer1::ITensor* image =
      network->addInput(spec.input_name, spec.input_type, nvinfer1::Dims{4, {batch, 3, spec.size, spec.size}});
  // [B, 3, size, size] -> [B, grid * grid, 3 * tile^2], one row per tile; a 0 dim copies B.
  nvinfer1::IShuffleLayer* tiles = network->addShuffle(*image);
  tiles->setReshapeDimensions(nvinfer1::Dims{6, {0, 3, grid, spec.tile, grid, spec.tile}});
  tiles->setSecondTranspose(nvinfer1::Permutation{{0, 2, 4, 1, 3, 5}});
  nvinfer1::IShuffleLayer* rows = network->addShuffle(*tiles->getOutput(0));
  rows->setReshapeDimensions(nvinfer1::Dims{3, {0, grid * grid, 3 * spec.tile * spec.tile}});
  rows->getOutput(0)->setName("patch_descriptors");
  network->markOutput(*rows->getOutput(0));

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setBuilderOptimizationLevel(0);  // nothing here is worth timing tactics for
  if (spec.input_type == nvinfer1::DataType::kHALF) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (spec.dynamic_batch) {
    // MIN == MAX would make TensorRT treat the batch as static.
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(spec.input_name, nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims{4, {1, 3, spec.size, spec.size}});
    profile->setDimensions(spec.input_name, nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims{4, {1, 3, spec.size, spec.size}});
    profile->setDimensions(spec.input_name, nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims{4, {2, 3, spec.size, spec.size}});
    config->addOptimizationProfile(profile);
  }
  const std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return {};
  }
  const std::filesystem::path path = ScratchFolder() / (name + ".engine");
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
  return file ? path.string() : std::string{};
}

/// Stand-in engine by name, built once per run; empty without a CUDA device.
const std::string& StandInEngine(const std::string& name) {
  static std::map<std::string, std::string> engines;
  auto it = engines.find(name);
  if (it == engines.end()) {
    StandInSpec spec;
    if (name == "wide") {
      spec.tile = 14;
    } else if (name == "dense") {
      spec.tile = 2;
    } else if (name == "renamed") {
      spec.input_name = "input";
    } else if (name == "dynamic") {
      spec.dynamic_batch = true;
    } else if (name == "half") {
      spec.input_type = nvinfer1::DataType::kHALF;
    }
    it = engines.emplace(name, BuildStandInEngine(spec, name)).first;
  }
  return it->second;
}

/// The stand-in engine, plus the real one if built.
std::vector<std::string> AnyLocEngines() {
  std::vector<std::string> engines = {StandInEngine("valid")};
  const std::string dinov2 = AnyLocEnginePath();
  if (!dinov2.empty()) {
    engines.push_back(dinov2);
  }
  return engines;
}

constexpr char kNoStandIns[] = "TensorRT could not build the stand-in engines, which needs a CUDA device";

/// What CreateVpr() throws for `options`, or empty.
std::string ConstructionError(const VprOptions& options) {
  try {
    CreateVpr(options);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return {};
}

/// What CreateVpr() throws for the engine at `path`, or empty.
std::string ConstructionError(const std::string& path) { return ConstructionError(AnyLocOptions(path)); }

VprOptions VocabularyOptions(const std::string& engine_path, const std::string& vocabulary_path) {
  VprOptions options = AnyLocOptions(engine_path);
  options.vocabulary_path = vocabulary_path;
  return options;
}

/// Fits `frames` blocky pictures from `seed` on with `engine_path` and writes the vocabulary to `name`.
std::string FitVocabulary(const std::string& engine_path, size_t frames, uint32_t seed, const std::string& name) {
  VocabularyFitter fitter(AnyLocOptions(engine_path));
  for (size_t i = 0; i < frames; ++i) {
    fitter.AddFrame(MakeBlockyImage(256, 256, 8, seed + static_cast<uint32_t>(i)));
  }
  const std::filesystem::path path = ScratchFolder() / name;
  fitter.Save(path.string());
  return path.string();
}

Blob ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return Blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string WriteFileBytes(const Blob& bytes, const std::string& name) {
  const std::filesystem::path path = ScratchFolder() / name;
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return path.string();
}

/// The bytes `vpr` serializes.
Blob Serialized(IVpr& vpr) {
  Blob blob;
  BlobWriter writer(blob);
  vpr.Serialize(writer);
  return blob;
}

}  // namespace

TEST(VprAnyLocFactory, RefusesToConstructWithoutAnEngine) {
  // Loading in the constructor lets the caller catch errors the SLAM worker could only log.
  EXPECT_NE(ConstructionError(""), "");
  EXPECT_NE(ConstructionError("/nonexistent/anyloc.engine"), "");
  EXPECT_NE(ConstructionError(ScratchFolder().string()), "");

  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;  // reading an engine file needs a GPU
  }
  // The likeliest mix-up: the ONNX model instead of the engine.
  const std::filesystem::path model = ScratchFolder() / "model.onnx";
  {
    std::ofstream file(model, std::ios::binary | std::ios::trunc);
    file << "an ONNX model, not a TensorRT engine";
  }
  const std::string error = ConstructionError(model.string());
  EXPECT_NE(error.find("anyloc_engine"), std::string::npos) << error;
}

TEST(VprAnyLocNetwork, RunsAnEngineWithTheExpectedInterface) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  AnyLocNetwork network(engine);
  EXPECT_EQ(network.input_width(), 28);
  EXPECT_EQ(network.input_height(), 28);
  EXPECT_EQ(network.patch_count(), 16);
  EXPECT_EQ(network.descriptor_dim(), 147);

  float* input = network.input();
  for (int i = 0; i < 3 * 28 * 28; ++i) {
    input[i] = static_cast<float>(i);
  }
  std::vector<float> patches;
  ASSERT_TRUE(network.Run(patches));
  ASSERT_EQ(patches.size(), 16u * 147u);
  // Patch 1 starts at red row 0, column 7; value 49 of patch 0 is the first green pixel.
  EXPECT_EQ(patches[147], 7.f);
  EXPECT_EQ(patches[49], 28.f * 28.f);
}

TEST(VprAnyLocNetwork, ReportsAnOutputThatIsNotFiniteAsAFailure) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  AnyLocNetwork network(StandInEngine("valid"));
  std::fill_n(network.input(), 3 * 28 * 28, 0.f);
  network.input()[5] = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> patches;
  EXPECT_FALSE(network.Run(patches));

  network.input()[5] = 0.f;
  EXPECT_TRUE(network.Run(patches));
}

TEST(VprAnyLocNetwork, RefusesAnEngineWithAnotherInterface) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  for (const char* name : {"renamed", "dynamic", "half"}) {
    SCOPED_TRACE(name);
    const std::string& engine = StandInEngine(name);
    ASSERT_FALSE(engine.empty());
    // The error names the interface the backend needs.
    const std::string error = ConstructionError(engine);
    EXPECT_NE(error.find("patch_descriptors fp32 [1, N, D]"), std::string::npos) << error;
  }
}

TEST(VprAnyLocNetwork, ReproducesTheFp32ReferenceDescriptors) {
  const std::string engine = AnyLocEnginePath();
  if (engine.empty()) {
    GTEST_SKIP() << "no AnyLoc engine: build the anyloc_engine target or set CUVSLAM_TEST_ANYLOC_ENGINE";
  }
#if defined(CUVSLAM_ANYLOC_REFERENCE_INPUT) && defined(CUVSLAM_ANYLOC_REFERENCE_OUTPUT)
  // Network input and fp32 descriptors the export saved for test_data/sof/left.png.
  const cnpy::NpyArray input = cnpy::npy_load(CUVSLAM_ANYLOC_REFERENCE_INPUT);
  const cnpy::NpyArray expected = cnpy::npy_load(CUVSLAM_ANYLOC_REFERENCE_OUTPUT);
  AnyLocNetwork network(engine);
  const size_t dim = static_cast<size_t>(network.descriptor_dim());
  if (input.num_vals != 3u * network.input_width() * network.input_height() ||
      expected.num_vals != network.patch_count() * dim) {
    GTEST_SKIP() << engine << " is not the engine this build's reference arrays belong to";
  }
  std::copy_n(input.data<float>(), input.num_vals, network.input());
  std::vector<float> patches;
  ASSERT_TRUE(network.Run(patches));

  double cosine_sum = 0.;
  double cosine_min = 1.;
  for (int p = 0; p < network.patch_count(); ++p) {
    const float* a = patches.data() + p * dim;
    const float* b = expected.data<float>() + p * dim;
    double dot = 0.;
    double norm_a = 0.;
    double norm_b = 0.;
    for (size_t k = 0; k < dim; ++k) {
      dot += static_cast<double>(a[k]) * b[k];
      norm_a += static_cast<double>(a[k]) * a[k];
      norm_b += static_cast<double>(b[k]) * b[k];
    }
    const double cosine = dot / std::sqrt(norm_a * norm_b);
    cosine_sum += cosine;
    cosine_min = std::min(cosine_min, cosine);
  }
  // FP16 vs fp32: the thresholds anyloc_engine_builder uses.
  EXPECT_GE(cosine_sum / network.patch_count(), 0.999);
  EXPECT_GE(cosine_min, 0.99);
#else
  GTEST_SKIP() << "this build has no AnyLoc reference arrays (CUVSLAM_BUILD_ANYLOC_ENGINE is off)";
#endif
}

TEST(VprAnyLocBackend, VocabularyIsNotFittedBelowTheTrainingMinimum) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  for (const std::string& engine : AnyLocEngines()) {
    SCOPED_TRACE(engine);
    std::unique_ptr<IVpr> vpr = CreateVpr(AnyLocOptions(engine));
    ASSERT_NE(vpr, nullptr);
    EXPECT_STREQ(vpr->Name(), "AnyLoc");

    const size_t min_frames = VprOptions{}.min_training_keyframes;
    std::vector<VprImage> images;
    for (size_t i = 0; i + 1 < min_frames; ++i) {
      images.push_back(MakeBlockyImage(256, 256, 8, static_cast<uint32_t>(1200 + i)));
      vpr->AddFrame(i, images.back());
    }
    ASSERT_EQ(vpr->Size(), min_frames - 1);

    // The vocabulary is frozen once fitted, so it is not fitted on too few frames.
    vpr->Finalize();
    EXPECT_FALSE(vpr->Query(images.front()).found);

    // At the minimum, a mapped frame recognizes itself.
    images.push_back(MakeBlockyImage(256, 256, 8, 1300));
    vpr->AddFrame(min_frames - 1, images.back());
    vpr->Finalize();
    const VprMatch match = vpr->Query(images.front());
    EXPECT_TRUE(match.found) << "score " << match.score;
    EXPECT_EQ(match.node_id, 0u);
  }
}

TEST(VprAnyLocBackend, RawPatchesNeverOutgrowTheTrainingSet) {
  const std::string& engine = StandInEngine("dense");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  // `exact` maps the most frames the training set holds whole, `longer` one more, `roomy` holds one more.
  const size_t patches = static_cast<size_t>(AnyLocNetwork(engine).patch_count());
  VprOptions options = AnyLocOptions(engine);
  options.max_training_descriptors = 20 * patches;
  VprOptions roomy_options = options;
  roomy_options.max_training_descriptors += patches;
  std::unique_ptr<IVpr> exact = CreateVpr(options);
  std::unique_ptr<IVpr> longer = CreateVpr(options);
  std::unique_ptr<IVpr> roomy = CreateVpr(roomy_options);
  ASSERT_NE(exact, nullptr);
  ASSERT_NE(longer, nullptr);
  ASSERT_NE(roomy, nullptr);
  const size_t full = exact->TrainingCapacity();
  ASSERT_EQ(full, 20u);
  VprImage image;
  for (size_t i = 0; i <= full; ++i) {
    image = MakeBlockyImage(256, 256, 8, static_cast<uint32_t>(3000 + i));
    if (i < full) {
      exact->AddFrame(i, image);
      roomy->AddFrame(i, image);
    }
    longer->AddFrame(i, image);
  }
  Blob exact_blob;
  BlobWriter exact_writer(exact_blob);
  exact->Serialize(exact_writer);
  Blob longer_blob;
  BlobWriter longer_writer(longer_blob);
  longer->Serialize(longer_writer);

  // Same header and centers: `longer` fit on `exact`'s frames before its last one arrived.
  const BlobReader reader(exact_blob);
  uint32_t magic = 0;
  uint32_t version = 0;
  int32_t clusters = 0;
  int32_t dim = 0;
  uint64_t center_count = 0;
  ASSERT_TRUE(reader.read(magic) && reader.read(version) && reader.read(clusters) && reader.read(dim) &&
              reader.read(center_count));
  ASSERT_GT(center_count, 0u);
  const size_t prefix = exact_blob.size() - reader.remaining() + center_count * sizeof(float);
  ASSERT_GE(longer_blob.size(), prefix);
  EXPECT_TRUE(std::equal(exact_blob.begin(), exact_blob.begin() + prefix, longer_blob.begin()));
  // `exact` fit on all of its frames, as `roomy` did: a full training set does not fit early.
  const Blob roomy_blob = Serialized(*roomy);
  ASSERT_GE(roomy_blob.size(), prefix);
  EXPECT_TRUE(std::equal(exact_blob.begin(), exact_blob.begin() + prefix, roomy_blob.begin()));

  // The frame mapped after the fit is recognized.
  const VprMatch match = longer->Query(image);
  EXPECT_TRUE(match.found) << "score " << match.score;
  EXPECT_EQ(match.node_id, full);
}

TEST(VprAnyLocBackend, SerializeDeserializeKeepsAnswers) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  for (const std::string& engine : AnyLocEngines()) {
    SCOPED_TRACE(engine);
    const size_t frames = VprOptions{}.min_training_keyframes;
    std::vector<VprImage> images;
    std::unique_ptr<IVpr> vpr = CreateVpr(AnyLocOptions(engine));
    ASSERT_NE(vpr, nullptr);
    for (size_t i = 0; i < frames; ++i) {
      images.push_back(MakeBlockyImage(256, 256, 8, static_cast<uint32_t>(1400 + i)));
      vpr->AddFrame(i, images.back());
    }

    Blob blob;
    BlobWriter writer(blob);
    vpr->Serialize(writer);
    ASSERT_FALSE(blob.empty());

    // The blob holds the cluster centers too, so no re-fit is needed.
    std::unique_ptr<IVpr> restored = CreateVpr(AnyLocOptions(engine));
    ASSERT_NE(restored, nullptr);
    ASSERT_TRUE(restored->Deserialize(BlobReader(blob)));
    EXPECT_EQ(restored->Size(), frames);

    for (size_t i = 0; i < frames; ++i) {
      EXPECT_TRUE(restored->HasNode(i));
      const VprMatch match = restored->Query(images[i]);
      EXPECT_TRUE(match.found) << "frame " << i;
      EXPECT_EQ(match.node_id, i);
      EXPECT_NEAR(match.score, vpr->Query(images[i]).score, 1e-5f);
    }

    const Blob garbage(64, 0x5A);
    std::unique_ptr<IVpr> rejecting = CreateVpr(AnyLocOptions(engine));
    ASSERT_NE(rejecting, nullptr);
    EXPECT_FALSE(rejecting->Deserialize(BlobReader(garbage)));
    EXPECT_EQ(rejecting->Size(), 0u);
  }
}

TEST(VprAnyLocBackend, MapsSavedBeforeTheVocabularyWasFittedLoad) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  // Neither has cluster centers to save; both must load.
  for (const size_t frames : {size_t{0}, VprOptions{}.min_training_keyframes - 1}) {
    SCOPED_TRACE(frames);
    std::unique_ptr<IVpr> vpr = CreateVpr(AnyLocOptions(StandInEngine("valid")));
    ASSERT_NE(vpr, nullptr);
    for (size_t i = 0; i < frames; ++i) {
      vpr->AddFrame(i, MakeBlockyImage(256, 256, 8, static_cast<uint32_t>(1600 + i)));
    }
    Blob blob;
    BlobWriter writer(blob);
    vpr->Serialize(writer);

    std::unique_ptr<IVpr> restored = CreateVpr(AnyLocOptions(StandInEngine("valid")));
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(restored->Deserialize(BlobReader(blob)));
    EXPECT_EQ(restored->Size(), frames);
    EXPECT_FALSE(restored->Query(MakeBlockyImage(256, 256, 8, 1600)).found);
  }
}

TEST(VprAnyLocBackend, DeserializeRefusesAMapOfAnotherDescriptorWidth) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  // Another width means another network, whose vocabulary is meaningless here.
  std::unique_ptr<IVpr> narrow = CreateVpr(AnyLocOptions(StandInEngine("valid")));
  ASSERT_NE(narrow, nullptr);
  for (size_t i = 0; i < VprOptions{}.min_training_keyframes; ++i) {
    narrow->AddFrame(i, MakeBlockyImage(256, 256, 8, static_cast<uint32_t>(1500 + i)));
  }
  Blob blob;
  BlobWriter writer(blob);
  narrow->Serialize(writer);

  std::unique_ptr<IVpr> wide = CreateVpr(AnyLocOptions(StandInEngine("wide")));
  ASSERT_NE(wide, nullptr);
  EXPECT_FALSE(wide->Deserialize(BlobReader(blob)));
  EXPECT_EQ(wide->Size(), 0u);
}

TEST(VprAnyLocBackend, TheTrainingMinimumComesFromTheOptions) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  VprOptions options = AnyLocOptions(StandInEngine("valid"));
  options.min_training_keyframes = 3;
  std::unique_ptr<IVpr> vpr = CreateVpr(options);
  ASSERT_NE(vpr, nullptr);
  const VprImage first = MakeBlockyImage(256, 256, 8, 1700);
  for (uint32_t i = 0; i < 3; ++i) {
    vpr->AddFrame(i, i == 0 ? first : MakeBlockyImage(256, 256, 8, 1700 + i));
    EXPECT_EQ(vpr->Query(first).found, i == 2) << "after " << i + 1 << " frames";
  }
}

TEST(VprAnyLocFactory, RefusesATrainingSetThatCannotHoldItsMinimum) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  VprOptions options = AnyLocOptions(StandInEngine("valid"));  // 16 descriptors a frame
  options.max_training_descriptors = 16 * options.min_training_keyframes - 1;
  EXPECT_THROW(CreateVpr(options), std::invalid_argument);
  ++options.max_training_descriptors;
  const std::unique_ptr<IVpr> vpr = CreateVpr(options);
  ASSERT_NE(vpr, nullptr);
  EXPECT_EQ(vpr->TrainingCapacity(), options.min_training_keyframes);
  options.min_training_keyframes = 0;
  EXPECT_THROW(CreateVpr(options), std::invalid_argument);
}

TEST(VprVocabularyFitter, TakesNoMoreFramesThanItsCapacity) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  VprOptions options = AnyLocOptions(StandInEngine("valid"));
  options.max_training_descriptors = 160;  // 10 frames of 16 descriptors
  VocabularyFitter fitter(options);
  ASSERT_EQ(fitter.Capacity(), 10u);
  for (uint32_t i = 0; i < 10; ++i) {
    fitter.AddFrame(MakeBlockyImage(256, 256, 8, 1800 + i));
  }
  EXPECT_THROW(fitter.AddFrame(MakeBlockyImage(256, 256, 8, 1810)), std::invalid_argument);
  EXPECT_EQ(fitter.Size(), 10u);
}

TEST(VprVocabularyFitter, NeedsTheTrainingMinimumToSave) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  const VprOptions options = AnyLocOptions(StandInEngine("valid"));
  VocabularyFitter fitter(options);
  for (uint32_t i = 0; i + 1 < options.min_training_keyframes; ++i) {
    fitter.AddFrame(MakeBlockyImage(256, 256, 8, 1820 + i));
  }
  const std::filesystem::path path = ScratchFolder() / "too_few.vocabulary";
  EXPECT_THROW(fitter.Save(path.string()), std::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
}

TEST(VprVocabularyFitter, WritesAVocabularyThatASessionLoads) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  VocabularyFitter fitter(AnyLocOptions(engine));
  for (uint32_t i = 0; i < 8; ++i) {
    fitter.AddFrame(MakeBlockyImage(256, 256, 8, 1830 + i));
  }
  const std::filesystem::path path = ScratchFolder() / "written.vocabulary";
  const std::filesystem::path again = ScratchFolder() / "written_again.vocabulary";
  fitter.Save(path.string());
  fitter.Save(again.string());
  EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
  const Blob bytes = ReadFileBytes(path);
  EXPECT_EQ(ReadFileBytes(again), bytes);
  EXPECT_THROW(fitter.AddFrame(MakeBlockyImage(256, 256, 8, 1840)), std::invalid_argument);

  // A session on it holds that vocabulary and no places.
  const std::unique_ptr<IVpr> vpr = CreateVpr(VocabularyOptions(engine, path.string()));
  ASSERT_NE(vpr, nullptr);
  EXPECT_EQ(vpr->TrainingCapacity(), 0u);
  EXPECT_EQ(Serialized(*vpr), bytes);
}

TEST(VprVocabularyFitter, ReportsAFileItCannotWrite) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  VocabularyFitter fitter(AnyLocOptions(engine));
  for (uint32_t i = 0; i < 8; ++i) {
    fitter.AddFrame(MakeBlockyImage(256, 256, 8, 1850 + i));
  }
  const std::string path = (ScratchFolder() / "missing" / "anyloc.vocabulary").string();
  try {
    fitter.Save(path);
    ADD_FAILURE() << "saved into a missing folder";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(path), std::string::npos) << error.what();
  }
}

TEST(VprVocabularyFitter, RefusesAFrameTheBackendCannotDescribe) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  VocabularyFitter fitter(AnyLocOptions(engine));
  EXPECT_THROW(fitter.AddFrame(VprImage{}), std::runtime_error);
  EXPECT_EQ(fitter.Size(), 0u);
}

TEST(VprAnyLocFactory, RefusesAFileThatIsNotAVocabularyOfTheEngine) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty() || StandInEngine("wide").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  const std::string good_path = FitVocabulary(engine, 8, 1860, "good.vocabulary");
  ASSERT_EQ(ConstructionError(VocabularyOptions(engine, good_path)), "");
  const Blob good = ReadFileBytes(good_path);

  // The first center starts after the magic, the version, the cluster count, the width and the center count.
  constexpr size_t kCentersOffset = 24;
  constexpr size_t kWidth = 147;
  auto with_first_center = [&good](float scale, float first) {
    Blob bytes = good;
    for (size_t i = 0; i < kWidth; ++i) {
      float value = 0.f;
      std::memcpy(&value, &bytes[kCentersOffset + i * sizeof(float)], sizeof(float));
      value = i == 0 && std::isnan(first) ? first : value * scale;
      std::memcpy(&bytes[kCentersOffset + i * sizeof(float)], &value, sizeof(float));
    }
    return bytes;
  };
  Blob trailing = good;
  trailing.push_back(0);
  std::unique_ptr<IVpr> mapper = CreateVpr(AnyLocOptions(engine));
  ASSERT_NE(mapper, nullptr);
  for (uint32_t i = 0; i < 8; ++i) {
    mapper->AddFrame(i, MakeBlockyImage(256, 256, 8, 1870 + i));
  }
  std::unique_ptr<IVpr> unfitted = CreateVpr(AnyLocOptions(engine));
  ASSERT_NE(unfitted, nullptr);
  const std::string other_width = FitVocabulary(StandInEngine("wide"), 8, 1880, "wide.vocabulary");

  const std::vector<std::pair<std::string, std::string>> cases = {
      {"missing", (ScratchFolder() / "missing.vocabulary").string()},
      {"empty", WriteFileBytes(Blob{}, "empty.vocabulary")},
      {"directory", ScratchFolder().string()},
      {"garbage", WriteFileBytes(Blob(64, 0x5A), "garbage.vocabulary")},
      {"truncated", WriteFileBytes(Blob(good.begin(), good.end() - 1), "truncated.vocabulary")},
      {"trailing byte", WriteFileBytes(trailing, "trailing.vocabulary")},
      {"mapped places", WriteFileBytes(Serialized(*mapper), "places.vocabulary")},
      {"no centers", WriteFileBytes(Serialized(*unfitted), "no_centers.vocabulary")},
      {"other width", other_width},
      {"center not of unit length", WriteFileBytes(with_first_center(2.f, 0.f), "scaled.vocabulary")},
      {"center not finite",
       WriteFileBytes(with_first_center(1.f, std::numeric_limits<float>::quiet_NaN()), "nan.vocabulary")},
  };
  for (const auto& [name, path] : cases) {
    SCOPED_TRACE(name);
    const std::string error = ConstructionError(VocabularyOptions(engine, path));
    EXPECT_NE(error.find(path), std::string::npos) << error;
  }
  // The width mismatch names both widths.
  const std::string error = ConstructionError(VocabularyOptions(engine, other_width));
  EXPECT_NE(error.find("588"), std::string::npos) << error;
  EXPECT_NE(error.find("147"), std::string::npos) << error;
}

TEST(VprAnyLocVocabulary, RecognizesPlacesFromTheFirstFrame) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  std::unique_ptr<IVpr> vpr =
      CreateVpr(VocabularyOptions(engine, FitVocabulary(engine, 8, 1900, "first_frame.vocabulary")));
  ASSERT_NE(vpr, nullptr);
  const VprImage image = MakeBlockyImage(256, 256, 8, 1910);
  vpr->AddFrame(0, image);
  const VprMatch match = vpr->Query(image);
  EXPECT_TRUE(match.found) << "score " << match.score;
  EXPECT_EQ(match.node_id, 0u);
}

TEST(VprAnyLocVocabulary, SessionsSharingAVocabularyEncodeAlike) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  const VprOptions options = VocabularyOptions(engine, FitVocabulary(engine, 8, 1915, "shared.vocabulary"));
  std::unique_ptr<IVpr> first = CreateVpr(options);
  std::unique_ptr<IVpr> second = CreateVpr(options);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  for (uint32_t i = 0; i < 3; ++i) {
    const VprImage image = MakeBlockyImage(256, 256, 8, 1916 + i);
    first->AddFrame(i, image);
    second->AddFrame(i, image);
  }
  EXPECT_EQ(Serialized(*first), Serialized(*second));
}

TEST(VprAnyLocVocabulary, DeserializeTakesOnlyPlacesOfItsVocabulary) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  const std::string vocabulary = FitVocabulary(engine, 8, 1920, "own.vocabulary");
  const VprOptions options = VocabularyOptions(engine, vocabulary);

  // Places this vocabulary encoded load.
  std::unique_ptr<IVpr> mapper = CreateVpr(options);
  ASSERT_NE(mapper, nullptr);
  std::vector<VprImage> images;
  for (uint32_t i = 0; i < 3; ++i) {
    images.push_back(MakeBlockyImage(256, 256, 8, 1930 + i));
    mapper->AddFrame(i, images.back());
  }
  const Blob own = Serialized(*mapper);
  std::unique_ptr<IVpr> session = CreateVpr(options);
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(session->Deserialize(BlobReader(own)));
  ASSERT_EQ(session->Size(), 3u);
  for (size_t i = 0; i < images.size(); ++i) {
    EXPECT_EQ(session->Query(images[i]).node_id, i) << "frame " << i;
  }

  // A session that fitted its own encoded them with another vocabulary: refused, nothing changes.
  std::unique_ptr<IVpr> online = CreateVpr(AnyLocOptions(engine));
  ASSERT_NE(online, nullptr);
  for (uint32_t i = 0; i < 8; ++i) {
    online->AddFrame(i, MakeBlockyImage(256, 256, 8, 1940 + i));
  }
  EXPECT_FALSE(session->Deserialize(BlobReader(Serialized(*online))));
  EXPECT_EQ(Serialized(*session), own);

  // A map saved before any fit has no centers: its places load and the vocabulary stays.
  std::unique_ptr<IVpr> unfitted = CreateVpr(AnyLocOptions(engine));
  ASSERT_NE(unfitted, nullptr);
  for (uint32_t i = 0; i < 3; ++i) {
    unfitted->AddFrame(i, MakeBlockyImage(256, 256, 8, 1950 + i));
  }
  std::unique_ptr<IVpr> fresh = CreateVpr(options);
  ASSERT_NE(fresh, nullptr);
  ASSERT_TRUE(fresh->Deserialize(BlobReader(Serialized(*unfitted))));
  EXPECT_EQ(fresh->Size(), 3u);
  const VprImage later = MakeBlockyImage(256, 256, 8, 1960);
  fresh->AddFrame(3, later);
  EXPECT_EQ(fresh->Query(later).node_id, 3u);
  const Blob file = ReadFileBytes(vocabulary);
  const Blob saved = Serialized(*fresh);
  const size_t centers_end = file.size() - sizeof(uint64_t);  // the file ends with a zero place count
  ASSERT_GE(saved.size(), centers_end);
  EXPECT_TRUE(std::equal(file.begin(), file.begin() + centers_end, saved.begin()));
}

#endif  // USE_TENSORRT

/// `map` as the SLAM database stores it.
Blob ToBlob(VprMap& map) {
  Blob blob;
  BlobWriter writer(blob);
  EXPECT_TRUE(map.ToBlob(writer));
  return blob;
}

TEST(VprMapStorage, OneImagePerNode) {
  VprMap map(VprOptions{}, StandInBackend());
  ASSERT_TRUE(map.Enabled());
  EXPECT_STREQ(map.BackendName(), "StandIn");

  const VprImage first = MakeBlockyImage(64, 64, 8, 1);
  const VprImage second = MakeBlockyImage(64, 64, 8, 2);
  map.OnKeyframeAdded(3, 1000);

  EXPECT_TRUE(map.AddFrame(3, 1000, first));
  EXPECT_TRUE(map.HasImage(3));
  EXPECT_EQ(map.Size(), 1u);

  EXPECT_FALSE(map.AddFrame(3, 2000, second));
  EXPECT_EQ(map.Size(), 1u);
  // The node keeps its first image.
  EXPECT_EQ(map.Query(first).node_id, 3u);

  EXPECT_FALSE(map.AddFrame(InvalidKeyFrameId, 3000, second));
  EXPECT_FALSE(map.AddFrame(4, 3000, VprImage{}));
  EXPECT_EQ(map.Size(), 1u);
}

TEST(VprMapStorage, KeyframeRemovalDropsTheEntry) {
  VprMap map(VprOptions{}, StandInBackend());
  const VprImage image = MakeBlockyImage(64, 64, 8, 1);
  map.OnKeyframeAdded(3, 1000);
  ASSERT_TRUE(map.AddFrame(3, 1000, image));

  map.OnKeyframeRemoved(3, InvalidKeyFrameId);

  EXPECT_FALSE(map.HasImage(3));
  EXPECT_EQ(map.Size(), 0u);
  EXPECT_FALSE(map.Query(image).found);
}

TEST(VprMapStorage, BlobRoundTripKeepsTheIdsAndStaysWritable) {
  constexpr size_t kNodes = 3;
  std::vector<VprImage> images;
  VprMap map(VprOptions{}, StandInBackend());
  for (size_t i = 0; i < kNodes; ++i) {
    images.push_back(MakeBlockyImage(64, 64, 8, static_cast<uint32_t>(200 + i)));
    map.OnKeyframeAdded(i, static_cast<int64_t>(1000 + i));
    ASSERT_TRUE(map.AddFrame(i, static_cast<int64_t>(1000 + i), images.back()));
  }
  const Blob blob = ToBlob(map);

  VprMap loaded(VprOptions{}, StandInBackend());
  ASSERT_TRUE(loaded.FromBlob(BlobReader(blob)));
  EXPECT_EQ(loaded.Size(), kNodes);
  for (size_t i = 0; i < kNodes; ++i) {
    EXPECT_TRUE(loaded.HasImage(i));
    const VprPlace place = loaded.Query(images[i]);
    ASSERT_TRUE(place.found) << "node " << i;
    EXPECT_EQ(place.node_id, i);
    EXPECT_NEAR(place.score, 1.f, 1e-3f);
    EXPECT_EQ(place.timestamp_ns, static_cast<int64_t>(1000 + i));
  }

  // It grows with the pose graph it was saved with.
  const VprImage live = MakeBlockyImage(64, 64, 8, 555);
  loaded.OnKeyframeAdded(kNodes, 5000);
  EXPECT_TRUE(loaded.AddFrame(kNodes, 5000, live));
  EXPECT_EQ(loaded.Size(), kNodes + 1);
  EXPECT_EQ(loaded.Query(live).node_id, kNodes);
}

TEST(VprMapStorage, FromBlobReplacesTheMap) {
  VprMap saved(VprOptions{}, StandInBackend());
  const VprImage saved_image = MakeBlockyImage(64, 64, 8, 71);
  saved.OnKeyframeAdded(1, 1000);
  ASSERT_TRUE(saved.AddFrame(1, 1000, saved_image));
  const Blob blob = ToBlob(saved);

  VprMap map(VprOptions{}, StandInBackend());
  const VprImage stale_image = MakeBlockyImage(64, 64, 8, 72);
  map.OnKeyframeAdded(7, 7000);
  ASSERT_TRUE(map.AddFrame(7, 7000, stale_image));

  ASSERT_TRUE(map.FromBlob(BlobReader(blob)));
  EXPECT_EQ(map.Size(), 1u);
  EXPECT_FALSE(map.HasImage(7));
  EXPECT_FALSE(map.Query(stale_image).found);
  EXPECT_EQ(map.Query(saved_image).node_id, 1u);
}

TEST(VprMapStorage, FromBlobRejectsTruncatedAndForeignBlobsAndKeepsTheMap) {
  VprMap saved(VprOptions{}, StandInBackend());
  saved.OnKeyframeAdded(1, 1000);
  ASSERT_TRUE(saved.AddFrame(1, 1000, MakeBlockyImage(64, 64, 8, 81)));
  const Blob blob = ToBlob(saved);

  VprMap map(VprOptions{}, StandInBackend());
  const VprImage image = MakeBlockyImage(64, 64, 8, 82);
  map.OnKeyframeAdded(5, 5000);
  ASSERT_TRUE(map.AddFrame(5, 5000, image));

  for (size_t size = 1; size < blob.size(); ++size) {
    const Blob truncated(blob.begin(), blob.begin() + static_cast<std::ptrdiff_t>(size));
    EXPECT_FALSE(map.FromBlob(BlobReader(truncated))) << size << " of " << blob.size() << " bytes";
  }
  const Blob garbage(blob.size(), 0xCD);
  EXPECT_FALSE(map.FromBlob(BlobReader(garbage)));

  // A payload the backend cannot read: the stand-in's magic follows the one entry.
  Blob foreign = blob;
  const size_t payload = 2 * sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(int64_t);
  ASSERT_GT(foreign.size(), payload);
  foreign[payload] ^= 0xFF;
  EXPECT_FALSE(map.FromBlob(BlobReader(foreign)));

  EXPECT_EQ(map.Size(), 1u);
  EXPECT_EQ(map.Query(image).node_id, 5u);
  EXPECT_TRUE(map.HasImage(5));
  EXPECT_FALSE(map.HasImage(1));
  EXPECT_EQ(map.Query(image).timestamp_ns, 5000);
}

TEST(VprMapStorage, FromBlobRejectsAnAbsurdEntryCount) {
  VprMap saved(VprOptions{}, StandInBackend());
  saved.OnKeyframeAdded(1, 1000);
  ASSERT_TRUE(saved.AddFrame(1, 1000, MakeBlockyImage(64, 64, 8, 1)));
  Blob blob = ToBlob(saved);

  // The count follows the magic and the version: fail cleanly, not with bad_alloc.
  const uint64_t absurd = std::numeric_limits<uint64_t>::max();
  std::memcpy(blob.data() + 2 * sizeof(uint32_t), &absurd, sizeof(absurd));

  VprMap loaded(VprOptions{}, StandInBackend());
  EXPECT_FALSE(loaded.FromBlob(BlobReader(blob)));
  EXPECT_EQ(loaded.Size(), 0u);
}

TEST(VprMapStorage, DisabledMapIsANoOp) {
  VprMap map(VprOptions{});
  const VprImage image = MakeBlockyImage(64, 64, 8, 1);

  EXPECT_FALSE(map.Enabled());
  EXPECT_STREQ(map.BackendName(), "None");
  EXPECT_EQ(map.Size(), 0u);

  map.OnKeyframeAdded(1, 1000);
  EXPECT_FALSE(map.AddFrame(1, 1000, image));
  EXPECT_FALSE(map.HasImage(1));
  EXPECT_FALSE(map.Query(image).found);
  map.OnKeyframeRemoved(1, InvalidKeyFrameId);

  Blob blob;
  BlobWriter writer(blob);
  EXPECT_FALSE(map.ToBlob(writer));
  EXPECT_TRUE(blob.empty());
  const Blob garbage(16, 0);
  EXPECT_FALSE(map.FromBlob(BlobReader(garbage)));
  EXPECT_EQ(map.Size(), 0u);
}

TEST(VprMapStorage, QueryTopKRanksTheCandidatesAndCapsTheCount) {
  constexpr size_t kNodes = 4;
  std::vector<VprImage> images;
  VprMap map(VprOptions{}, StandInBackend());
  for (size_t i = 0; i < kNodes; ++i) {
    images.push_back(MakeBlockyImage(64, 64, 8, static_cast<uint32_t>(400 + i)));
    map.OnKeyframeAdded(i, static_cast<int64_t>(1000 + i));
    ASSERT_TRUE(map.AddFrame(i, static_cast<int64_t>(1000 + i), images.back()));
  }

  const std::vector<VprPlace> places = map.QueryTopK(images[2], kNodes + 1);
  ASSERT_EQ(places.size(), kNodes);
  EXPECT_EQ(places.front().node_id, 2u);
  for (size_t i = 0; i < places.size(); ++i) {
    EXPECT_TRUE(places[i].found) << "position " << i;
    EXPECT_EQ(places[i].timestamp_ns, static_cast<int64_t>(1000 + places[i].node_id));
    if (i > 0) {
      EXPECT_LE(places[i].score, places[i - 1].score) << "position " << i;
    }
  }

  EXPECT_EQ(map.QueryTopK(images[2], 2).size(), 2u);
  EXPECT_TRUE(map.QueryTopK(images[2], 0).empty());
  EXPECT_TRUE(map.QueryTopK(VprImage{}, 3).empty());
}

TEST(VprMapStorage, AMergedNodeIsDroppedAndTheSurvivorKeepsItsOwnImage) {
  VprMap map(VprOptions{}, StandInBackend());
  const VprImage survivor_image = MakeBlockyImage(64, 64, 8, 61);
  const VprImage merged_image = MakeBlockyImage(64, 64, 8, 62);
  map.OnKeyframeAdded(1, 1000);
  map.OnKeyframeAdded(2, 2000);
  ASSERT_TRUE(map.AddFrame(1, 1000, survivor_image));
  ASSERT_TRUE(map.AddFrame(2, 2000, merged_image));

  map.OnKeyframeRemoved(2, 1);

  // The survivor keeps its own descriptor: the merged one is from another viewpoint.
  EXPECT_EQ(map.Size(), 1u);
  EXPECT_FALSE(map.HasImage(2));
  EXPECT_TRUE(map.HasImage(1));
  EXPECT_EQ(map.Query(survivor_image).node_id, 1u);
  EXPECT_FALSE(map.Query(merged_image).found);
}

TEST(VprSofImage, ReadsTheCpuPyramidAndRefusesAnUnfilledContext) {
  const VprImage source = MakeBlockyImage(64, 48, 8, 71);

  EXPECT_TRUE(MakeVprImageFromContext(nullptr).Empty());

  // Allocated but unfilled, like a recycled pool slot: no pyramid, and its CPU getters assert.
  auto unfilled = std::make_shared<sof::ImageContext>(ImageShape{source.width, source.height}, false, false);
  EXPECT_FALSE(unfilled->has_cpu_image());
  EXPECT_FALSE(unfilled->has_gpu_image());
  EXPECT_TRUE(MakeVprImageFromContext(unfilled).Empty());

  const VprImage image = MakeVprImageFromContext(MakeCpuContext(source));
  ASSERT_FALSE(image.Empty());
  EXPECT_EQ(image.width, source.width);
  EXPECT_EQ(image.height, source.height);
  EXPECT_EQ(image.row, source.row);
}

TEST(VprSofImage, PicksTheFirstCameraThatCarriesPixels) {
  const VprImage source = MakeBlockyImage(64, 48, 8, 72);
  const sof::ImageContextPtr filled = MakeCpuContext(source);

  EXPECT_TRUE(MakeVprImageFromImages(Images{}).Empty());
  EXPECT_TRUE(MakeVprImageFromImages(Images{nullptr, nullptr}).Empty());
  // Camera-indexed: an absent camera is a nullptr.
  EXPECT_EQ(MakeVprImageFromImages(Images{nullptr, filled}).row, source.row);
}

#ifdef USE_CUDA

TEST(VprSofImage, DownloadsTheGpuImageWhenTheFrameWasBuiltOnTheDevice) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "no CUDA device";
  }

  const VprImage source = MakeBlockyImage(64, 48, 8, 73);
  const ImageShape shape{source.width, source.height};
  auto context = std::make_shared<sof::ImageContext>(shape, true, false);
  ImageMeta meta;
  meta.shape = shape;
  context->set_image_meta(meta);

  ImageSource host_source;
  host_source.data = const_cast<uint8_t*>(source.row.data());
  host_source.type = ImageSource::U8;
  host_source.memory_type = ImageSource::Host;
  host_source.image_encoding = ImageEncoding::MONO8;
  ASSERT_TRUE(context->build_gpu_image_pyramid(host_source, false, nullptr));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // No CPU pyramid: read back from the GPU as float, then rounded to the original bytes.
  EXPECT_FALSE(context->has_cpu_image());
  ASSERT_TRUE(context->has_gpu_image());
  const VprImage image = MakeVprImageFromContext(context);
  ASSERT_FALSE(image.Empty());
  EXPECT_EQ(image.width, source.width);
  EXPECT_EQ(image.height, source.height);
  EXPECT_EQ(image.row, source.row);
}

#endif  // USE_CUDA

/// Pose graph with VPR from synthetic keyframes and blocky pictures; no landmarks.
class VprPoseGraph : public ::testing::Test {
protected:
  void SetUp() override {
    camera_ = camera::CreateCameraModel(Vector2T(640, 480), Vector2T(320, 320), Vector2T(320, 240),
                                        Distortion::Model::Pinhole, nullptr, 0);
    rig_.intrinsics[0] = camera_.get();
    rig_.num_cameras = 1;
    rig_.camera_from_rig[0].setIdentity();

    folder_ = std::filesystem::temp_directory_path() /
              ("cuvslam_vpr_test_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::filesystem::remove_all(folder_);

    mapper_ = MakeMapper();
  }

  void TearDown() override { std::filesystem::remove_all(folder_); }

  std::string Folder() const { return folder_.string(); }

  /// Mapper with the stand-in VPR backend.
  std::unique_ptr<LocalizerAndMapper> MakeMapper() {
    std::unique_ptr<LocalizerAndMapper> mapper = MakeMapper(VprOptions{});
    mapper->SetVprMap(MakeStandInMap());
    return mapper;
  }

  std::unique_ptr<LocalizerAndMapper> MakeMapper(const VprOptions& options) {
    auto mapper = std::make_unique<LocalizerAndMapper>(rig_, FeatureDescriptorType::kNone, true);
    mapper->SetActiveCameras({0});
    mapper->SetVprOptions(options);
    EXPECT_TRUE(mapper->SetPoseGraphOptimizerOptions({PoseGraphOptimizerType::Simple}));
    return mapper;
  }

  /// Appends a node `step_x` m along X, with `image` as its picture if given. Returns its timestamp.
  int64_t AppendKeyframe(int index, float step_x, int64_t frame_delta_ns = kFrameDeltaNs, VprImage image = {},
                         const Images& images = {}) {
    Isometry3T step = Isometry3T::Identity();
    step.translation().x() = step_x;

    VOFrameData frame_data;
    frame_data.frame_id = index;
    frame_data.timestamp_ns = static_cast<uint64_t>(index) * frame_delta_ns;
    frame_data.vpr_image = std::move(image);
    mapper_->AddKeyframe(step, frame_data, images);
    return static_cast<int64_t>(frame_data.timestamp_ns);
  }

  /// Maps `count` keyframes along X, one picture each.
  void BuildChain(int count, float step_x, int64_t frame_delta_ns = kFrameDeltaNs) {
    for (int i = 0; i < count; ++i) {
      images_.push_back(MakeBlockyImage(64, 64, 8, static_cast<uint32_t>(800 + i)));
      AppendKeyframe(i, step_x, frame_delta_ns, images_.back());
      ASSERT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(i) + 1) << "keyframe " << i;
    }
  }

  /// Merges a node and checks that frame poses and the VPR map both follow.
  void ExpectAMergeUpdatesBothBooks() {
    // Longer than the 20 newest keyframes PoseGraph keeps from reduction.
    constexpr int kChain = 25;
    BuildChain(kChain, 1.f);
    ASSERT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(kChain));

    Map& map = MutableMap();
    std::vector<KeyFrameId> before;
    map.pose_graph_.QueryKeyframes([&before](KeyFrameId id) { before.push_back(id); });

    const EdgeId edge_id = map.pose_graph_.GetSmallestVarianceEdgeId();
    ASSERT_NE(edge_id, InvalidEdgeId);
    ASSERT_TRUE(map.pose_graph_.ReduceSingleEdge(edge_id, map.pose_graph_hypothesis_, nullptr));

    std::vector<KeyFrameId> alive;
    map.pose_graph_.QueryKeyframes([&alive](KeyFrameId id) { alive.push_back(id); });
    ASSERT_EQ(alive.size() + 1, before.size());
    KeyFrameId merged_away = InvalidKeyFrameId;
    for (const KeyFrameId id : before) {
      if (std::find(alive.begin(), alive.end(), id) == alive.end()) {
        merged_away = id;
      }
    }
    ASSERT_NE(merged_away, InvalidKeyFrameId);

    // The VPR map dropped the merged node, and only that one.
    EXPECT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(kChain) - 1);
    EXPECT_NE(mapper_->RecognizePlace(images_[merged_away]).node_id, merged_away);
    for (const KeyFrameId id : alive) {
      const VprPlace place = mapper_->RecognizePlace(images_[id]);
      EXPECT_TRUE(place.found) << "node " << id;
      EXPECT_EQ(place.node_id, id) << "node " << id;
    }

    // The merged node's frame now follows the survivor, so shifting live nodes must move it.
    const auto frame_id = static_cast<FrameId>(merged_away);
    Isometry3T pose_before;
    ASSERT_TRUE(mapper_->CalcFramePose(frame_id, pose_before));

    const Vector3T shift(10.f, 20.f, 30.f);
    for (const KeyFrameId id : alive) {
      const Isometry3T* node_pose = map.pose_graph_hypothesis_.GetKeyframePose(id);
      ASSERT_NE(node_pose, nullptr) << "node " << id;
      Isometry3T shifted = *node_pose;
      shifted.translation() += shift;
      map.pose_graph_hypothesis_.SetKeyframePose(id, shifted);
    }

    Isometry3T pose_after;
    ASSERT_TRUE(mapper_->CalcFramePose(frame_id, pose_after));
    EXPECT_LT(((pose_after.translation() - pose_before.translation()) - shift).norm(), 1e-4f);
  }

  /// Mutable map, to add loop edges directly: ApplyLoopClosureResult needs landmarks.
  Map& MutableMap() { return const_cast<Map&>(mapper_->GetMap()); }

  static constexpr int64_t kFrameDeltaNs = 33000000;

  std::unique_ptr<camera::ICameraModel> camera_;
  camera::Rig rig_;
  std::filesystem::path folder_;
  std::unique_ptr<LocalizerAndMapper> mapper_;
  std::vector<VprImage> images_;
};

TEST_F(VprPoseGraph, RecognizedPoseIsTheNodesLivePose) {
  BuildChain(4, 1.f);

  for (size_t i = 0; i < images_.size(); ++i) {
    const VprPlace place = mapper_->RecognizePlace(images_[i]);
    ASSERT_TRUE(place.found) << "frame " << i;
    EXPECT_EQ(place.timestamp_ns, static_cast<int64_t>(i) * kFrameDeltaNs);
    const Isometry3T* node_pose = MutableMap().pose_graph_hypothesis_.GetKeyframePose(place.node_id);
    ASSERT_NE(node_pose, nullptr) << "frame " << i;
    EXPECT_LT((place.pose.translation() - node_pose->translation()).norm(), 1e-5f) << "frame " << i;
  }
}

TEST_F(VprPoseGraph, RecognizedPoseFollowsPoseGraphOptimization) {
  constexpr int kKeyframes = 8;
  constexpr float kStep = 1.f;
  BuildChain(kKeyframes, kStep);

  std::vector<VprPlace> before;
  for (const VprImage& image : images_) {
    const VprPlace place = mapper_->RecognizePlace(image);
    ASSERT_TRUE(place.found);
    before.push_back(place);
  }

  // A loop closure 0.2 m off the chain, leaving drift for the optimizer to spread.
  Map& map = MutableMap();
  const Isometry3T loop_from_head = MakePose(-(kKeyframes - 1) * kStep + 0.2f, 0.f, 0.f);
  map.pose_graph_.AddEdge(map.pose_graph_hypothesis_, before.back().node_id, before.front().node_id, loop_from_head,
                          Matrix6T::Identity());
  ASSERT_TRUE(mapper_->OptimizePoseGraph(false));

  float moved = 0.f;
  for (size_t i = 0; i < images_.size(); ++i) {
    const VprPlace place = mapper_->RecognizePlace(images_[i]);
    ASSERT_TRUE(place.found) << "frame " << i;
    // Same node; only its pose changed.
    EXPECT_EQ(place.node_id, before[i].node_id) << "frame " << i;

    const Isometry3T* node_pose = map.pose_graph_hypothesis_.GetKeyframePose(place.node_id);
    ASSERT_NE(node_pose, nullptr) << "frame " << i;
    EXPECT_LT((place.pose.translation() - node_pose->translation()).norm(), 1e-5f) << "frame " << i;

    moved = std::max(moved, (place.pose.translation() - before[i].pose.translation()).norm());
  }

  // Otherwise a backend answering with stored poses would pass trivially.
  EXPECT_GT(moved, 1e-3f) << "the optimization moved no node, so this test proved nothing";
}

namespace {

/// Records the task DetectLoopClosure() built, then fails; no PnP-solvable scene needed.
class RecordingLoopClosureSolver : public ILoopClosureSolver {
public:
  explicit RecordingLoopClosureSolver(bool wants_vpr_candidates = true) : wants_vpr_candidates_(wants_vpr_candidates) {}

  bool WantsVprCandidates() const override { return wants_vpr_candidates_; }

  bool Solve(const LoopClosureTask& task, const LSIGrid&, const IFeatureDescriptorOps*, Isometry3T&, Matrix6T&,
             std::vector<LandmarkInSolver>*, DiscardLandmarkCB*, KeyframeInSightCB*) const override {
    ++solve_calls;
    captured_candidates = task.vpr_candidates;
    return false;
  }

  mutable int solve_calls = 0;
  mutable std::vector<VprPlace> captured_candidates;

private:
  bool wants_vpr_candidates_;
};

/// At this spacing, a keyframe four back is just old enough for loop closure.
constexpr int64_t kLoopClosureSpacingNs = LocalizerAndMapper::kLoopClosureVprMinAgeNs / 4;

}  // namespace

TEST_F(VprPoseGraph, DetectLoopClosureFeedsTheSolverPoseGraphCorrectedCandidates) {
  constexpr int kKeyframes = 12;
  constexpr float kStep = 1.f;
  BuildChain(kKeyframes, kStep, kLoopClosureSpacingNs);

  // Not the first node: it anchors the optimization and never moves.
  const VprImage& query = images_[2];
  const VprPlace before = mapper_->RecognizePlace(query);
  const VprPlace first = mapper_->RecognizePlace(images_.front());
  const VprPlace head = mapper_->RecognizePlace(images_.back());
  ASSERT_TRUE(before.found && first.found && head.found);

  // Same loop closure as RecognizedPoseFollowsPoseGraphOptimization, so the nodes move.
  Map& map = MutableMap();
  const Isometry3T loop_from_head = MakePose(-(kKeyframes - 1) * kStep + 0.2f, 0.f, 0.f);
  map.pose_graph_.AddEdge(map.pose_graph_hypothesis_, head.node_id, first.node_id, loop_from_head,
                          Matrix6T::Identity());
  ASSERT_TRUE(mapper_->OptimizePoseGraph(false));

  // Oracle: RecognizePlace() resolves through the optimized pose graph.
  const VprPlace expected = mapper_->RecognizePlace(query);
  ASSERT_TRUE(expected.found);
  ASSERT_EQ(expected.node_id, before.node_id);
  ASSERT_GT((expected.pose.translation() - before.pose.translation()).norm(), 1e-3f)
      << "the optimization did not move the queried node, so this test would prove nothing";

  RecordingLoopClosureSolver solver;
  LocalizerAndMapper::LoopClosureStatus status;
  mapper_->DetectLoopClosure(solver, Images{MakeCpuContext(query)}, Isometry3T::Identity(), status);

  EXPECT_FALSE(status.success);
  ASSERT_EQ(solver.solve_calls, 1);
  ASSERT_FALSE(solver.captured_candidates.empty());
  const VprPlace& best_candidate = solver.captured_candidates.front();
  EXPECT_EQ(best_candidate.node_id, expected.node_id);
  EXPECT_LT((best_candidate.pose.translation() - expected.pose.translation()).norm(), 1e-5f)
      << "DetectLoopClosure must feed the solver pose graph corrected candidates (via RecognizePlaces), "
      << "not the pose-less answer of VprMap::QueryTopK";
}

TEST_F(VprPoseGraph, DetectLoopClosureLeavesOutTheRecentPast) {
  constexpr int kKeyframes = 12;
  BuildChain(kKeyframes, 1.f, kLoopClosureSpacingNs);

  // Query with the newest picture, which the recent past matches best.
  RecordingLoopClosureSolver solver;
  LocalizerAndMapper::LoopClosureStatus status;
  mapper_->DetectLoopClosure(solver, Images{MakeCpuContext(images_.back())}, Isometry3T::Identity(), status);

  const int64_t head_timestamp_ns = (kKeyframes - 1) * kLoopClosureSpacingNs;
  const std::vector<VprPlace>& candidates = solver.captured_candidates;
  ASSERT_FALSE(candidates.empty()) << "the chain has places old enough to close a loop to";
  EXPECT_EQ(candidates.size(), LocalizerAndMapper::kLoopClosureVprCandidates);
  for (size_t i = 0; i < candidates.size(); ++i) {
    EXPECT_GE(head_timestamp_ns - candidates[i].timestamp_ns, LocalizerAndMapper::kLoopClosureVprMinAgeNs)
        << "candidate " << i << " is node " << candidates[i].node_id;
    if (i > 0) {
      EXPECT_GE(candidates[i - 1].score, candidates[i].score) << "candidates are handed over best first";
    }
  }
}

#ifdef USE_LMDB
TEST_F(VprPoseGraph, DetectLoopClosureKeepsPlacesAnotherSessionMappedLater) {
  // A loaded place's timestamp may be ahead of ours, yet it is never the recent past.
  constexpr int kKeyframes = 4;
  BuildChain(kKeyframes, 1.f, kLoopClosureSpacingNs);
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(Folder()));

  mapper_ = MakeMapper();
  ASSERT_TRUE(mapper_->AttachToExistingReadOnlyDatabase(Folder()));
  ASSERT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(kKeyframes));
  // This session maps on, still early by the saving session's clock.
  AppendKeyframe(kKeyframes, 1.f, /*frame_delta_ns=*/1, MakeBlockyImage(64, 64, 8, 990));
  ASSERT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(kKeyframes) + 1);

  RecordingLoopClosureSolver solver;
  LocalizerAndMapper::LoopClosureStatus status;
  mapper_->DetectLoopClosure(solver, Images{MakeCpuContext(images_[1])}, Isometry3T::Identity(), status);

  ASSERT_FALSE(solver.captured_candidates.empty());
  EXPECT_EQ(solver.captured_candidates.front().node_id, 1u);
}
#endif  // USE_LMDB

TEST_F(VprPoseGraph, DetectLoopClosureRecognizesPlacesOnlyForASolverThatWantsThem) {
  BuildChain(12, 1.f, kLoopClosureSpacingNs);

  RecordingLoopClosureSolver solver(/*wants_vpr_candidates=*/false);
  LocalizerAndMapper::LoopClosureStatus status;
  mapper_->DetectLoopClosure(solver, Images{MakeCpuContext(images_.front())}, Isometry3T::Identity(), status);

  EXPECT_EQ(solver.solve_calls, 1);
  EXPECT_TRUE(solver.captured_candidates.empty());
}

TEST_F(VprPoseGraph, DetectLoopClosureHasNoCandidatesWithPlaceRecognitionOff) {
  mapper_ = MakeMapper(VprOptions{});
  for (int i = 0; i < 12; ++i) {
    AppendKeyframe(i, 1.f, kLoopClosureSpacingNs);
  }

  RecordingLoopClosureSolver solver;
  LocalizerAndMapper::LoopClosureStatus status;
  const VprImage query = MakeBlockyImage(64, 64, 8, 800);
  mapper_->DetectLoopClosure(solver, Images{MakeCpuContext(query)}, Isometry3T::Identity(), status);

  EXPECT_EQ(solver.solve_calls, 1);
  EXPECT_TRUE(solver.captured_candidates.empty());
}

#ifdef USE_TENSORRT

TEST_F(VprPoseGraph, DetectLoopClosureWithAnyLocHandsOverOldPlacesBestFirst) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  for (const std::string& engine : AnyLocEngines()) {
    SCOPED_TRACE(engine);
    mapper_ = MakeMapper(AnyLocOptions(engine));
    images_.clear();
    // Enough keyframes to fit the vocabulary, at the other AnyLoc tests' picture size.
    constexpr int kKeyframes = 12;
    for (int i = 0; i < kKeyframes; ++i) {
      images_.push_back(MakeBlockyImage(256, 256, 8, static_cast<uint32_t>(1200 + i)));
      AppendKeyframe(i, 1.f, kLoopClosureSpacingNs, images_.back());
      ASSERT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(i) + 1) << "keyframe " << i;
    }

    // An old node's picture, so the best match is old enough to close a loop to.
    RecordingLoopClosureSolver solver;
    LocalizerAndMapper::LoopClosureStatus status;
    mapper_->DetectLoopClosure(solver, Images{MakeCpuContext(images_[2])}, Isometry3T::Identity(), status);

    const std::vector<VprPlace>& candidates = solver.captured_candidates;
    ASSERT_FALSE(candidates.empty());
    EXPECT_LE(candidates.size(), LocalizerAndMapper::kLoopClosureVprCandidates);
    EXPECT_EQ(candidates.front().node_id, mapper_->RecognizePlace(images_[2]).node_id);
    const int64_t head_timestamp_ns = (kKeyframes - 1) * kLoopClosureSpacingNs;
    for (const VprPlace& candidate : candidates) {
      EXPECT_GE(head_timestamp_ns - candidate.timestamp_ns, LocalizerAndMapper::kLoopClosureVprMinAgeNs)
          << "node " << candidate.node_id;
    }
  }
}

TEST_F(VprPoseGraph, AsyncSlamMapsEveryKeyframeQueuedWhileItWasBehind) {
  if (StandInEngine("valid").empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  AsyncSlamOptions options;
  options.use_gpu = false;
  options.reproduce_mode = true;
  options.loop_closure_solver_type = LoopClosureSolverType::kDummy;
  options.vpr_options = AnyLocOptions(StandInEngine("valid"));
  AsyncSlam slam(rig_, {0}, options);

  // All keyframes queue before the worker runs, by which time it has only the newest one's pixels.
  const int keyframes = static_cast<int>(VprOptions{}.min_training_keyframes);
  Isometry3T step = Isometry3T::Identity();
  step.translation().x() = 1.f;
  for (int i = 0; i < keyframes; ++i) {
    images_.push_back(MakeBlockyImage(256, 256, 8, static_cast<uint32_t>(1400 + i)));
    odom::IVisualOdometry::VOFrameStat stat;
    stat.keyframe = true;
    slam.TrackResult(i, (i + 1) * kFrameDeltaNs, stat, Images{MakeCpuContext(images_.back(), i)}, step);
  }
  slam.ProcessInputSynchronously();

  for (int i = 0; i < keyframes; ++i) {
    std::optional<int64_t> timestamp_ns;
    slam.RecognizePlace(images_[i], [&](const Result<cuvslam::Slam::PlaceRecognition>& result) {
      if (result.data) {
        timestamp_ns = result.data->timestamp_ns;
      }
    });
    ASSERT_TRUE(timestamp_ns.has_value()) << "keyframe " << i;
    EXPECT_EQ(*timestamp_ns, (i + 1) * kFrameDeltaNs) << "keyframe " << i;
  }
}

#endif  // USE_TENSORRT

TEST_F(VprPoseGraph, AKeyframeMapsTheFrameItCarries) {
  EXPECT_TRUE(mapper_->IsVprEnabled());
  // Neither a copied frame nor pixels: the node stays out of the map.
  AppendKeyframe(0, 1.f);
  EXPECT_EQ(mapper_->GetVprMapSize(), 0u);

  const VprImage copied = MakeBlockyImage(64, 64, 8, 901);
  const int64_t copied_ns = AppendKeyframe(1, 1.f, kFrameDeltaNs, copied);
  EXPECT_EQ(mapper_->GetVprMapSize(), 1u);
  EXPECT_EQ(mapper_->RecognizePlace(copied).timestamp_ns, copied_ns);

  // The copy wins over the pixels, which may belong to a later frame by now.
  const VprImage later = MakeBlockyImage(64, 64, 8, 902);
  const VprImage own = MakeBlockyImage(64, 64, 8, 903);
  const int64_t own_ns = AppendKeyframe(2, 1.f, kFrameDeltaNs, own, Images{MakeCpuContext(later)});
  EXPECT_EQ(mapper_->GetVprMapSize(), 2u);
  EXPECT_EQ(mapper_->RecognizePlace(own).timestamp_ns, own_ns);
  EXPECT_NE(mapper_->RecognizePlace(later).timestamp_ns, own_ns);

  // Without a copy, the pixels are mapped.
  const VprImage pixels = MakeBlockyImage(64, 64, 8, 904);
  const int64_t pixels_ns = AppendKeyframe(3, 1.f, kFrameDeltaNs, {}, Images{MakeCpuContext(pixels)});
  EXPECT_EQ(mapper_->GetVprMapSize(), 3u);
  EXPECT_EQ(mapper_->RecognizePlace(pixels).timestamp_ns, pixels_ns);
}

TEST_F(VprPoseGraph, RecognizePlacesRanksSeveralNodesAgainstTheLivePoseGraph) {
  BuildChain(5, 1.f);

  const std::vector<VprPlace> places = mapper_->RecognizePlaces(images_[3], 3);
  ASSERT_EQ(places.size(), 3u);
  EXPECT_EQ(places.front().node_id, mapper_->RecognizePlace(images_[3]).node_id);
  for (size_t i = 0; i < places.size(); ++i) {
    EXPECT_TRUE(places[i].found) << "position " << i;
    const Isometry3T* node_pose = MutableMap().pose_graph_hypothesis_.GetKeyframePose(places[i].node_id);
    ASSERT_NE(node_pose, nullptr) << "position " << i;
    EXPECT_LT((places[i].pose.translation() - node_pose->translation()).norm(), 1e-5f) << "position " << i;
    if (i > 0) {
      EXPECT_LE(places[i].score, places[i - 1].score) << "position " << i;
    }
  }

  EXPECT_TRUE(mapper_->RecognizePlaces(images_[3], 0).empty());
}

#ifdef USE_LMDB
TEST_F(VprPoseGraph, TheSavedSlamMapCarriesThePlaceRecognitionMap) {
  constexpr int kKeyframes = 8;
  constexpr float kStep = 1.f;
  BuildChain(kKeyframes, kStep);

  // Optimize after mapping, so the saved graph's poses differ from the mapped ones.
  Map& map = MutableMap();
  map.pose_graph_.AddEdge(map.pose_graph_hypothesis_, kKeyframes - 1, 0,
                          MakePose(-(kKeyframes - 1) * kStep + 0.2f, 0.f, 0.f), Matrix6T::Identity());
  ASSERT_TRUE(mapper_->OptimizePoseGraph(false));
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(Folder()));
  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(folder_)) {
    files.push_back(entry.path().filename().string());
  }
  EXPECT_EQ(files, std::vector<std::string>{"data.mdb"});

  // One LMDB database: the reader gets the pose graph and the VPR map, with the same node ids.
  std::unique_ptr<LocalizerAndMapper> reader = MakeMapper();
  ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(Folder()));
  EXPECT_EQ(reader->GetVprMapSize(), static_cast<size_t>(kKeyframes));
  for (size_t i = 0; i < images_.size(); ++i) {
    const VprPlace place = reader->RecognizePlace(images_[i]);
    ASSERT_TRUE(place.found) << "frame " << i;
    EXPECT_EQ(place.node_id, i) << "frame " << i;
    EXPECT_EQ(place.timestamp_ns, static_cast<int64_t>(i) * kFrameDeltaNs);
    const Isometry3T* optimized = map.pose_graph_hypothesis_.GetKeyframePose(i);
    ASSERT_NE(optimized, nullptr) << "frame " << i;
    EXPECT_LT((place.pose.translation() - optimized->translation()).norm(), 1e-4f) << "frame " << i;
  }

  // It keeps mapping on the loaded graph.
  std::swap(mapper_, reader);
  AppendKeyframe(kKeyframes, kStep, kFrameDeltaNs, MakeBlockyImage(64, 64, 8, 990));
  EXPECT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(kKeyframes) + 1);
}

TEST_F(VprPoseGraph, AMapSavedWithoutPlaceRecognitionLoadsWithAnEmptyOne) {
  mapper_ = MakeMapper(VprOptions{});
  for (int i = 0; i < 3; ++i) {
    AppendKeyframe(i, 1.f);
  }
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(Folder()));

  std::unique_ptr<LocalizerAndMapper> reader = MakeMapper();
  ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(Folder()));
  EXPECT_TRUE(reader->IsVprEnabled());
  EXPECT_EQ(reader->GetVprMapSize(), 0u);
}

TEST_F(VprPoseGraph, FlushMapWritesThePlaceRecognitionMapToTheMapFile) {
  // Like Slam::Config::map_path, then Slam::FlushMap.
  std::filesystem::create_directories(folder_);
  const std::string map_file = (folder_ / "map").string();
  ASSERT_TRUE(mapper_->AttachToNewDatabase(map_file));
  BuildChain(3, 1.f);
  ASSERT_TRUE(mapper_->FlushMapToDatabase());
  mapper_.reset();  // ends without detaching

  std::unique_ptr<LocalizerAndMapper> reader = MakeMapper();
  ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(map_file));
  EXPECT_EQ(reader->GetVprMapSize(), 3u);
  EXPECT_EQ(reader->RecognizePlace(images_[2]).node_id, 2u);
}

TEST_F(VprPoseGraph, ASessionWithoutPlaceRecognitionKeepsTheLoadedIndex) {
  std::filesystem::create_directories(folder_);
  const std::string saved = (folder_ / "saved").string();
  const std::string copy = (folder_ / "copy").string();
  BuildChain(3, 1.f);
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(saved));

  std::unique_ptr<LocalizerAndMapper> session = MakeMapper(VprOptions{});
  ASSERT_TRUE(session->AttachToExistingReadOnlyDatabase(saved));
  ASSERT_FALSE(session->IsVprEnabled());
  ASSERT_TRUE(session->AttachToNewDatabaseSaveMapAndDetach(copy));  // SaveMap
  ASSERT_TRUE(session->ReopenDatabaseForWriting(saved));            // LocalizeInMap with attach
  ASSERT_TRUE(session->FlushMapToDatabase());
  session.reset();

  for (const std::string& folder : {saved, copy}) {
    std::unique_ptr<LocalizerAndMapper> reader = MakeMapper();
    ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(folder));
    EXPECT_EQ(reader->GetVprMapSize(), 3u) << folder;
  }
}

TEST_F(VprPoseGraph, AnUnreadablePlaceRecognitionMapIsKeptAsItIs) {
  std::filesystem::create_directories(folder_);
  const std::string saved = (folder_ / "saved").string();
  const std::string copy = (folder_ / "copy").string();
  BuildChain(3, 1.f);
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(saved));
  // As from another engine or format.
  const std::string other = "another VPR map";
  {
    LmdbSlamDatabase database;
    ASSERT_TRUE(database.Open(saved.c_str(), LmdbSlamDatabase::OpenMode::READ_WRITE_EXISTS));
    ASSERT_TRUE(database.SetSingleton(SlamDatabaseSingleton::VprMap, 0, [&](BlobWriter& writer) {
      writer.write(other.data(), other.size());
      return true;
    }));
  }

  std::unique_ptr<LocalizerAndMapper> session = MakeMapper();
  ASSERT_TRUE(session->AttachToExistingReadOnlyDatabase(saved));
  ASSERT_TRUE(session->IsVprEnabled());
  EXPECT_EQ(session->GetVprMapSize(), 0u);
  ASSERT_TRUE(session->AttachToNewDatabaseSaveMapAndDetach(copy));  // SaveMap
  ASSERT_TRUE(session->ReopenDatabaseForWriting(saved));            // LocalizeInMap with attach
  ASSERT_TRUE(session->FlushMapToDatabase());
  session.reset();

  for (const std::string& folder : {saved, copy}) {
    LmdbSlamDatabase database;
    ASSERT_TRUE(database.Open(folder.c_str(), LmdbSlamDatabase::OpenMode::READ_ONLY_EXISTS));
    std::string kept;
    ASSERT_TRUE(database.GetSingleton(SlamDatabaseSingleton::VprMap, [&](const BlobReader& reader) {
      kept.resize(reader.remaining());
      return reader.read(kept.data(), kept.size());
    })) << folder;
    EXPECT_EQ(kept, other) << folder;
  }
}

#ifdef USE_TENSORRT
TEST_F(VprPoseGraph, AMapOfAnotherVocabularyIsKeptAsItIs) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  std::filesystem::create_directories(folder_);
  const std::string saved = (folder_ / "saved").string();
  const std::string copy = (folder_ / "copy").string();
  auto vpr_record = [](const std::string& folder) {
    LmdbSlamDatabase database;
    EXPECT_TRUE(database.Open(folder.c_str(), LmdbSlamDatabase::OpenMode::READ_ONLY_EXISTS)) << folder;
    std::string record;
    EXPECT_TRUE(database.GetSingleton(SlamDatabaseSingleton::VprMap, [&](const BlobReader& reader) {
      record.resize(reader.remaining());
      return reader.read(record.data(), record.size());
    })) << folder;
    return record;
  };
  // Fits its own vocabulary when saving.
  mapper_ = MakeMapper(AnyLocOptions(engine));
  BuildChain(static_cast<int>(VprOptions{}.min_training_keyframes), 1.f);
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(saved));
  const std::string recorded = vpr_record(saved);

  std::unique_ptr<LocalizerAndMapper> session =
      MakeMapper(VocabularyOptions(engine, FitVocabulary(engine, 8, 2000, "session.vocabulary")));
  ASSERT_TRUE(session->AttachToExistingReadOnlyDatabase(saved));
  EXPECT_EQ(session->GetVprMapSize(), 0u);
  EXPECT_FALSE(session->RecognizePlace(images_.front()).found);
  ASSERT_TRUE(session->AttachToNewDatabaseSaveMapAndDetach(copy));  // SaveMap
  ASSERT_TRUE(session->ReopenDatabaseForWriting(saved));            // LocalizeInMap with attach
  ASSERT_TRUE(session->FlushMapToDatabase());
  session.reset();

  for (const std::string& folder : {saved, copy}) {
    EXPECT_EQ(vpr_record(folder), recorded) << folder;
  }
}

TEST_F(VprPoseGraph, AMapOfTheSameVocabularyIsRecognized) {
  const std::string& engine = StandInEngine("valid");
  if (engine.empty()) {
    GTEST_SKIP() << kNoStandIns;
  }
  const VprOptions options = VocabularyOptions(engine, FitVocabulary(engine, 8, 2010, "same.vocabulary"));
  mapper_ = MakeMapper(options);
  BuildChain(3, 1.f);
  ASSERT_TRUE(mapper_->AttachToNewDatabaseSaveMapAndDetach(Folder()));

  std::unique_ptr<LocalizerAndMapper> reader = MakeMapper(options);
  ASSERT_TRUE(reader->AttachToExistingReadOnlyDatabase(Folder()));
  EXPECT_EQ(reader->GetVprMapSize(), 3u);
  for (size_t i = 0; i < images_.size(); ++i) {
    const VprPlace place = reader->RecognizePlace(images_[i]);
    EXPECT_TRUE(place.found) << "frame " << i;
    EXPECT_EQ(place.node_id, i) << "frame " << i;
  }
}
#endif  // USE_TENSORRT
#endif  // USE_LMDB

TEST_F(VprPoseGraph, TurningVprOffLeavesEveryEntryPointInert) {
  BuildChain(2, 1.f);
  ASSERT_TRUE(mapper_->IsVprEnabled());
  ASSERT_EQ(mapper_->GetVprMapSize(), 2u);

  mapper_->SetVprOptions(VprOptions{});

  EXPECT_FALSE(mapper_->IsVprEnabled());
  EXPECT_EQ(mapper_->GetVprMapSize(), 0u);
  AppendKeyframe(2, 1.f, kFrameDeltaNs, images_.front());
  EXPECT_EQ(mapper_->GetVprMapSize(), 0u);
  EXPECT_FALSE(mapper_->RecognizePlace(images_.front()).found);
  EXPECT_TRUE(mapper_->RecognizePlaces(images_.front(), 3).empty());
  EXPECT_TRUE(mapper_->RecognizeLoopClosurePlaces(images_.front()).empty());
}

TEST_F(VprPoseGraph, AMergeFollowsBothBooksWhenPlaceRecognitionWasInstalledLast) {
  // AsyncSlam's order; the later call used to replace PoseGraph's only removal callback.
  mapper_ = MakeMapper(VprOptions{});
  mapper_->SetKeepTrackPoses(true);
  mapper_->SetVprMap(MakeStandInMap());

  ExpectAMergeUpdatesBothBooks();
}

TEST_F(VprPoseGraph, AMergeFollowsBothBooksWhenFramePosesWereInstalledLast) {
  // The other order: SetUp installed VPR, and this re-registers the callback.
  mapper_->SetKeepTrackPoses(true);

  ExpectAMergeUpdatesBothBooks();
}

}  // namespace test::vpr
