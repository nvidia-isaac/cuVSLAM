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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "common/vector_3t.h"
#include "slam/slam/slam.h"
#include "slam/vpr/ivpr.h"
#include "slam/vpr/vpr_image.h"
#include "slam/vpr/vpr_map.h"
#include "slam/vpr/vpr_simple.h"
#include "slam/vpr/vpr_sof_image.h"
#include "sof/image_context.h"

#ifdef USE_CUDA
#include "cuda_modules/cuda_helper.h"
#endif

#include "slam/vpr/vpr_bow.h"
#include "slam/vpr/vpr_orb.h"

#ifdef USE_DBOW2
#include "slam/vpr/vpr_dbow2.h"
#endif

#ifdef USE_ONNXRUNTIME
#include "slam/vpr/vpr_anyloc.h"
#endif

namespace test::vpr {
using namespace cuvslam;
using namespace cuvslam::slam;
using namespace cuvslam::slam::vpr;

namespace {

constexpr int kDownscale = 8;

/// Image whose pixels are constant over `block` x `block` tiles, so downscaling by `block` returns
/// the tile values themselves. Two seeds give two tile patterns whose correlation is close to zero,
/// which is what makes them stand in for two different places.
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

VprOptions SimpleOptions() {
  VprOptions options;
  options.type = VprType::kSimple;
  options.downscale = kDownscale;
  return options;
}

Isometry3T MakePose(float x, float y, float z) {
  Isometry3T pose = Isometry3T::Identity();
  pose.translation() = Vector3T(x, y, z);
  return pose;
}

uint8_t PixelAt(const VprImage& image, int x, int y) { return image.row[static_cast<size_t>(y) * image.width + x]; }

/// Tracking image context filled the way the host pipeline fills one. The pyramid copies the
/// pixels, so `image` does not have to outlive the context.
sof::ImageContextPtr MakeCpuContext(const VprImage& image) {
  const ImageShape shape{image.width, image.height};
  auto context = std::make_shared<sof::ImageContext>(shape, false, false);
  ImageMeta meta;
  meta.shape = shape;
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
  // A pitch shorter than one row would make the copy read the next row's pixels.
  EXPECT_TRUE(MakeVprImage(buffer.data(), 4, 4, 3, VprPixelFormat::kMono8).Empty());
}

TEST(VprImageDownscale, FactorOneIsIdentity) {
  const VprImage image = MakeBlockyImage(16, 12, 4, 11);

  const VprImage out = Downscale(image, 1);
  EXPECT_EQ(out.width, image.width);
  EXPECT_EQ(out.height, image.height);
  EXPECT_EQ(out.row, image.row);
  EXPECT_TRUE(Downscale(VprImage{}, 4).Empty());
}

TEST(VprImageDownscale, ConstantImageKeepsItsValue) {
  const VprImage out = Downscale(MakeConstantImage(64, 64, 173), 8);
  ASSERT_FALSE(out.Empty());
  EXPECT_EQ(out.width, 8);
  EXPECT_EQ(out.height, 8);
  for (const uint8_t v : out.row) {
    EXPECT_EQ(v, 173);
  }
}

TEST(VprImageDownscale, GradientBecomesBlockAverages) {
  // 10 * x averaged over a 2x2 block starting at 2 * bx is exactly 20 * bx + 5.
  VprImage image;
  image.width = 16;
  image.height = 8;
  image.row.resize(static_cast<size_t>(image.width) * image.height);
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      image.row[static_cast<size_t>(y) * image.width + x] = static_cast<uint8_t>(10 * x);
    }
  }

  const VprImage out = Downscale(image, 2);
  ASSERT_EQ(out.width, 8);
  ASSERT_EQ(out.height, 4);
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      EXPECT_EQ(PixelAt(out, x, y), static_cast<uint8_t>(20 * x + 5)) << "at " << x << "," << y;
    }
  }
}

TEST(VprImageDownscale, NonDivisibleSizeCoversTheWholeInput) {
  // 5 does not divide by 2, so the last output column and row average a 3 pixel wide block. Every
  // output pixel is checked because that is where a block that runs past the last row would show up.
  VprImage image;
  image.width = 5;
  image.height = 5;
  image.row.resize(25);
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      image.row[static_cast<size_t>(y) * image.width + x] = static_cast<uint8_t>(10 * (x + y));
    }
  }

  const VprImage out = Downscale(image, 2);
  ASSERT_EQ(out.width, 2);
  ASSERT_EQ(out.height, 2);
  EXPECT_EQ(PixelAt(out, 0, 0), 10);
  EXPECT_EQ(PixelAt(out, 1, 0), 35);
  EXPECT_EQ(PixelAt(out, 0, 1), 35);
  EXPECT_EQ(PixelAt(out, 1, 1), 60);
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

  // A vertical edge on the block boundary survives the 8x box pass without bleeding across it.
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
  // 2x1 -> 4x1 is the plain bilinear path, with no box pre-pass in front of it: the two outer
  // destination centers fall outside the source centers and clamp to the end pixels, the two inner
  // ones sit a quarter and three quarters of the way between them.
  VprImage image;
  image.width = 2;
  image.height = 1;
  image.row = {0, 200};

  const VprImage out = Resize(image, 4, 1);
  ASSERT_FALSE(out.Empty());
  EXPECT_EQ(out.row, std::vector<uint8_t>({0, 50, 150, 200}));
}

TEST(VprImageNormalize, ConstantImageIsRejected) {
  std::vector<float> out(3, 42.f);

  EXPECT_FALSE(NormalizePixels(MakeConstantImage(8, 8, 17), out));
  ASSERT_EQ(out.size(), 64u);
  for (const float v : out) {
    EXPECT_EQ(v, 0.f);
  }

  EXPECT_FALSE(NormalizePixels(VprImage{}, out));
  EXPECT_TRUE(out.empty());
}

TEST(VprImageNormalize, ZeroMeanUnitVariance) {
  const VprImage image = MakeBlockyImage(32, 16, 4, 9);

  std::vector<float> out;
  ASSERT_TRUE(NormalizePixels(image, out));
  ASSERT_EQ(out.size(), image.row.size());

  double mean = 0;
  for (const float v : out) {
    mean += v;
  }
  mean /= static_cast<double>(out.size());

  double variance = 0;
  for (const float v : out) {
    variance += (v - mean) * (v - mean);
  }
  variance /= static_cast<double>(out.size());

  EXPECT_NEAR(mean, 0.0, 1e-4);
  EXPECT_NEAR(variance, 1.0, 1e-4);
}

class VprSimpleBackend : public ::testing::Test {
protected:
  void SetUp() override {
    vpr_ = CreateVpr(SimpleOptions());
    ASSERT_NE(vpr_, nullptr);
    for (size_t i = 0; i < kNodeIds.size(); ++i) {
      images_.push_back(MakeBlockyImage(64, 64, 8, static_cast<uint32_t>(100 + i)));
      vpr_->AddFrame(kNodeIds[i], images_.back());
    }
  }

  static constexpr std::array<KeyFrameId, 3> kNodeIds = {5, 17, 42};

  std::unique_ptr<IVpr> vpr_;
  std::vector<VprImage> images_;
};

TEST_F(VprSimpleBackend, RecognizesEveryMappedImage) {
  EXPECT_STREQ(vpr_->Name(), "Simple");
  ASSERT_EQ(vpr_->Size(), kNodeIds.size());

  for (size_t i = 0; i < kNodeIds.size(); ++i) {
    EXPECT_TRUE(vpr_->HasNode(kNodeIds[i]));
    const VprMatch match = vpr_->Query(images_[i]);
    EXPECT_TRUE(match.found);
    EXPECT_EQ(match.node_id, kNodeIds[i]);
    EXPECT_NEAR(match.score, 1.f, 1e-3f);
  }
  EXPECT_FALSE(vpr_->HasNode(7));
}

TEST_F(VprSimpleBackend, UnrelatedImageScoresLower) {
  const VprImage unrelated = MakeBlockyImage(64, 64, 8, 777);

  const VprMatch match = vpr_->Query(unrelated);
  EXPECT_TRUE(!match.found || match.score < 0.8f) << "score " << match.score;
}

TEST_F(VprSimpleBackend, RemoveNodeDropsIt) {
  vpr_->RemoveNode(kNodeIds[1]);

  EXPECT_FALSE(vpr_->HasNode(kNodeIds[1]));
  EXPECT_EQ(vpr_->Size(), kNodeIds.size() - 1);
  EXPECT_NE(vpr_->Query(images_[1]).node_id, kNodeIds[1]);
  // The surviving entries are still addressable after the swap-with-last removal.
  EXPECT_EQ(vpr_->Query(images_[2]).node_id, kNodeIds[2]);

  vpr_->RemoveNode(kNodeIds[1]);
  EXPECT_EQ(vpr_->Size(), kNodeIds.size() - 1);
}

TEST_F(VprSimpleBackend, ShiftNodeIdsRenumbers) {
  constexpr KeyFrameId kOffset = 1000;
  vpr_->ShiftNodeIds(kOffset);

  EXPECT_EQ(vpr_->Size(), kNodeIds.size());
  for (size_t i = 0; i < kNodeIds.size(); ++i) {
    EXPECT_FALSE(vpr_->HasNode(kNodeIds[i]));
    EXPECT_TRUE(vpr_->HasNode(kNodeIds[i] + kOffset));
    EXPECT_EQ(vpr_->Query(images_[i]).node_id, kNodeIds[i] + kOffset);
  }
}

TEST_F(VprSimpleBackend, SerializeDeserializeKeepsAnswers) {
  Blob blob;
  vpr_->Finalize();
  vpr_->Serialize(blob);
  ASSERT_FALSE(blob.empty());

  std::unique_ptr<IVpr> restored = CreateVpr(SimpleOptions());
  ASSERT_NE(restored, nullptr);
  const BlobReader reader(blob);
  ASSERT_TRUE(restored->Deserialize(reader));

  EXPECT_EQ(restored->Size(), vpr_->Size());
  for (size_t i = 0; i < kNodeIds.size(); ++i) {
    EXPECT_TRUE(restored->HasNode(kNodeIds[i]));
    const VprMatch match = restored->Query(images_[i]);
    EXPECT_TRUE(match.found);
    EXPECT_EQ(match.node_id, kNodeIds[i]);
    EXPECT_NEAR(match.score, vpr_->Query(images_[i]).score, 1e-5f);
  }
}

TEST_F(VprSimpleBackend, DeserializeRejectsForeignBlob) {
  const Blob garbage(64, 0x5A);

  std::unique_ptr<IVpr> restored = CreateVpr(SimpleOptions());
  ASSERT_NE(restored, nullptr);
  EXPECT_FALSE(restored->Deserialize(BlobReader(garbage)));
}

TEST_F(VprSimpleBackend, QueryTopKRanksEveryEntryAndCapsTheCount) {
  const std::vector<VprMatch> top = vpr_->QueryTopK(images_[1], kNodeIds.size() + 2);
  ASSERT_EQ(top.size(), kNodeIds.size());
  EXPECT_EQ(top.front().node_id, kNodeIds[1]);
  for (size_t i = 0; i < top.size(); ++i) {
    EXPECT_TRUE(top[i].found) << "position " << i;
    if (i > 0) {
      EXPECT_LE(top[i].score, top[i - 1].score) << "position " << i;
    }
  }

  EXPECT_EQ(vpr_->QueryTopK(images_[1], 2).size(), 2u);
  EXPECT_TRUE(vpr_->QueryTopK(images_[1], 0).empty());
}

TEST(VprSimpleThreshold, OnlyTheWinnerIsHeldToTheScoreThreshold) {
  // The runners-up exist for a relocalizer to verify geometrically, and a candidate it can reject
  // cheaply beats no candidate at all, so the threshold only decides whether there is an answer.
  VprOptions options = SimpleOptions();
  options.score_threshold = 0.9f;
  std::unique_ptr<IVpr> vpr = CreateVpr(options);
  ASSERT_NE(vpr, nullptr);

  std::vector<VprImage> images;
  for (uint32_t i = 0; i < 3; ++i) {
    images.push_back(MakeBlockyImage(64, 64, 8, 1100 + i));
    vpr->AddFrame(i, images.back());
  }

  const std::vector<VprMatch> top = vpr->QueryTopK(images[0], 3);
  ASSERT_EQ(top.size(), 3u);
  EXPECT_EQ(top.front().node_id, 0u);
  EXPECT_NEAR(top.front().score, 1.f, 1e-3f);
  EXPECT_LT(top[1].score, options.score_threshold) << "a runner-up the threshold would have dropped";

  // When the winner itself misses the threshold there is no answer, runners-up included.
  EXPECT_TRUE(vpr->QueryTopK(MakeBlockyImage(64, 64, 8, 777), 3).empty());
}

TEST(VprSimpleRatioTest, AnAmbiguousPlaceIsRefusedAndADistinctiveOneIsNot) {
  // VprOptions::ratio_threshold is not reachable from Slam::Config, so this is the only thing that
  // exercises it.
  const VprImage place = MakeBlockyImage(64, 64, 8, 21);
  const VprImage elsewhere = MakeBlockyImage(64, 64, 8, 22);
  constexpr KeyFrameId kNode = 1;
  constexpr KeyFrameId kFarAway = kNode + VprSimple::kNeighbourGuard + 1;

  VprOptions guarded = SimpleOptions();
  guarded.ratio_threshold = 1.5f;

  // The same place mapped twice, far apart along the trajectory: both entries answer the query
  // equally well and naming either of them is a coin flip, so the query is refused.
  std::unique_ptr<IVpr> ambiguous = CreateVpr(guarded);
  ambiguous->AddFrame(kNode, place);
  ambiguous->AddFrame(kFarAway, place);
  EXPECT_FALSE(ambiguous->Query(place).found);

  // The very same map answers with the ratio test off, so what refused above was the ratio test
  // and not the score threshold.
  std::unique_ptr<IVpr> unguarded = CreateVpr(SimpleOptions());
  unguarded->AddFrame(kNode, place);
  unguarded->AddFrame(kFarAway, place);
  EXPECT_TRUE(unguarded->Query(place).found);

  // Two entries that are neighbors in the trajectory are one place seen twice, not two places, so
  // the guard band keeps them from disqualifying each other.
  std::unique_ptr<IVpr> neighbors = CreateVpr(guarded);
  neighbors->AddFrame(kNode, place);
  neighbors->AddFrame(kNode + VprSimple::kNeighbourGuard, place);
  EXPECT_TRUE(neighbors->Query(place).found);

  // And a place nothing else in the map looks like passes.
  std::unique_ptr<IVpr> distinctive = CreateVpr(guarded);
  distinctive->AddFrame(kNode, place);
  distinctive->AddFrame(kFarAway, elsewhere);
  const VprMatch match = distinctive->Query(place);
  EXPECT_TRUE(match.found);
  EXPECT_EQ(match.node_id, kNode);
}

TEST(VprSimpleMapSize, ReMapsAKnownNodeButStopsAtMaxEntries) {
  const VprImage first = MakeBlockyImage(64, 64, 8, 31);
  const VprImage second = MakeBlockyImage(64, 64, 8, 32);
  const VprImage third = MakeBlockyImage(64, 64, 8, 33);

  VprOptions options = SimpleOptions();
  options.max_entries = 2;
  std::unique_ptr<IVpr> vpr = CreateVpr(options);
  ASSERT_NE(vpr, nullptr);

  vpr->AddFrame(1, first);
  vpr->AddFrame(2, second);
  ASSERT_EQ(vpr->Size(), 2u);

  // The cap turns new nodes away...
  vpr->AddFrame(3, third);
  EXPECT_EQ(vpr->Size(), 2u);
  EXPECT_FALSE(vpr->HasNode(3));

  // ...and leaves a node already in the map free to be re-mapped, which is what the cap is for:
  // bounding the map, not freezing it.
  vpr->AddFrame(1, third);
  EXPECT_EQ(vpr->Size(), 2u);
  EXPECT_TRUE(vpr->HasNode(1));
  const VprMatch match = vpr->Query(third);
  EXPECT_TRUE(match.found);
  EXPECT_EQ(match.node_id, 1u);
  EXPECT_NEAR(match.score, 1.f, 1e-3f);

  // An image with no pixels is not a descriptor, so it does not make a node either.
  std::unique_ptr<IVpr> unbounded = CreateVpr(SimpleOptions());
  unbounded->AddFrame(9, VprImage{});
  EXPECT_FALSE(unbounded->HasNode(9));
  EXPECT_EQ(unbounded->Size(), 0u);
  EXPECT_TRUE(unbounded->QueryTopK(first, 3).empty());
}

TEST(VprSimpleThumbnail, AFrameOfAnotherSizeIsResizedOntoTheMapsThumbnail) {
  // The thumbnail geometry is fixed by the first frame the map ever saw. MakeBlockyImage lays the
  // same 8x8 tile pattern down at both sizes, so these are one place photographed at two
  // resolutions, and the 96x96 one downscales to a 12x12 thumbnail where the map holds 8x8 ones.
  const VprImage mapped = MakeBlockyImage(64, 64, 8, 41);
  const VprImage larger = MakeBlockyImage(96, 96, 12, 41);
  const VprImage elsewhere = MakeBlockyImage(96, 96, 12, 42);

  std::unique_ptr<IVpr> vpr = CreateVpr(SimpleOptions());
  ASSERT_NE(vpr, nullptr);
  vpr->AddFrame(1, mapped);

  const VprMatch match = vpr->Query(larger);
  EXPECT_TRUE(match.found) << "score " << match.score;
  EXPECT_EQ(match.node_id, 1u);
  EXPECT_FALSE(vpr->Query(elsewhere).found);
}

TEST_F(VprSimpleBackend, DeserializeRejectsTruncatedAndOversizedBlobs) {
  Blob blob;
  vpr_->Finalize();
  vpr_->Serialize(blob);
  ASSERT_GT(blob.size(), 32u);

  // Every prefix of a valid blob is short of something the reader wants, and has to come back short
  // rather than read past its end.
  for (size_t length = 1; length + 1 < blob.size(); length += 7) {
    const Blob truncated(blob.begin(), blob.begin() + static_cast<Blob::difference_type>(length));
    std::unique_ptr<IVpr> restored = CreateVpr(SimpleOptions());
    ASSERT_NE(restored, nullptr);
    EXPECT_FALSE(restored->Deserialize(BlobReader(truncated))) << length << " of " << blob.size() << " bytes";
  }

  // The entry count is the sixth word of the header, see VprSimple::Serialize. A blob that declares
  // more entries than it could possibly hold is refused before it is reserved for, so a corrupt map
  // is a clean false rather than a length_error out of reserve().
  constexpr size_t kCountOffset = 2 * sizeof(uint32_t) + 3 * sizeof(int32_t);
  Blob oversized = blob;
  const uint64_t absurd = std::numeric_limits<uint64_t>::max();
  std::memcpy(oversized.data() + kCountOffset, &absurd, sizeof(absurd));

  std::unique_ptr<IVpr> restored = CreateVpr(SimpleOptions());
  ASSERT_NE(restored, nullptr);
  EXPECT_FALSE(restored->Deserialize(BlobReader(oversized)));
  EXPECT_EQ(restored->Size(), 0u);
}

TEST(VprSimpleFactory, NoneHasNoBackend) {
  VprOptions options;
  options.type = VprType::kNone;
  EXPECT_EQ(CreateVpr(options), nullptr);
  EXPECT_STREQ(ToString(VprType::kNone), "None");
  EXPECT_STREQ(ToString(VprType::kSimple), "Simple");
}

TEST(VprTypeNames, EveryBackendHasAPrintableName) {
  EXPECT_STREQ(ToString(VprType::kNone), "None");
  EXPECT_STREQ(ToString(VprType::kSimple), "Simple");
  EXPECT_STREQ(ToString(VprType::kDBoW2), "DBoW2");
  EXPECT_STREQ(ToString(VprType::kAnyLoc), "AnyLoc");
  // A value outside the enum reaches the fallback instead of running off the end of the switch.
  EXPECT_STREQ(ToString(static_cast<VprType>(200)), "Unknown");
}

TEST(VprFactory, RefusesUnknownAndUnavailableBackends) {
  VprOptions options;
  options.type = static_cast<VprType>(200);
  EXPECT_THROW(CreateVpr(options), std::runtime_error);

  // The two optional backends are a build configuration error when they were compiled out, and
  // available otherwise. This build has DBoW2, so what runs here is the other branch of the same
  // decision; a USE_DBOW2=OFF build runs the throw.
  options.type = VprType::kDBoW2;
#ifdef USE_DBOW2
  EXPECT_NE(CreateVpr(options), nullptr);
#else
  EXPECT_THROW(CreateVpr(options), std::runtime_error);
#endif

  // AnyLoc throws either way here: compiled out it is unavailable, compiled in it refuses to
  // construct without a model file, which it checks on this thread rather than on the SLAM worker.
  options.type = VprType::kAnyLoc;
  options.model_path = "/nonexistent/dinov2.onnx";
  EXPECT_THROW(CreateVpr(options), std::runtime_error);
}

namespace {

/// Backend that implements nothing but Query(), so its QueryTopK() is IVpr's default. Every backend
/// in the tree overrides that default; a new one does not have to.
class OneAnswerVpr : public IVpr {
public:
  explicit OneAnswerVpr(const VprMatch& answer) : answer_(answer) {}

  const char* Name() const override { return "OneAnswer"; }
  void AddFrame(KeyFrameId, const VprImage&) override {}
  bool HasNode(KeyFrameId) const override { return false; }
  void RemoveNode(KeyFrameId) override {}
  void ShiftNodeIds(KeyFrameId) override {}
  size_t Size() const override { return 0; }
  void Finalize() override {}
  VprMatch Query(const VprImage&) override {
    ++queries;
    return answer_;
  }
  void Serialize(Blob&) override {}
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

#ifdef USE_DBOW2

namespace {

VprOptions DBoW2Options() {
  VprOptions options;
  options.type = VprType::kDBoW2;
  return options;
}

/// 8x8 tiles over 256x256 put a corner at every tile boundary, which is what ORB detects. That is
/// close to the 500 features per frame the backend keeps, so a handful of these images already give
/// the vocabulary tree the tens of thousands of descriptors it branches on. A 64x64 image, which is
/// what the Simple backend tests use, yields too few features to reach past the first tree level.
VprImage MakeCornerRichImage(uint32_t seed) { return MakeBlockyImage(256, 256, 8, seed); }

/// DBoW2 seeds the k-means that trains its vocabulary from the global rand() state, so what a
/// training run does - down to whether it walks into the upstream empty cluster case SafeFORB
/// guards, see vpr_dbow2.cpp - depends on whatever else in the process drew from rand() first.
/// These tests pin the seed so they mean the same thing run alone and run after the rest of the
/// suite. 3 is a value the unguarded backend died on; without the guard about half of them do.
constexpr unsigned kRngSeed = 3;

}  // namespace

TEST(VprDBoW2Backend, RecognizesEveryMappedImage) {
  std::srand(kRngSeed);
  constexpr size_t kFrames = 12;
  std::unique_ptr<IVpr> vpr = CreateVpr(DBoW2Options());
  ASSERT_NE(vpr, nullptr);
  EXPECT_STREQ(vpr->Name(), "DBoW2");

  std::vector<VprImage> images;
  for (size_t i = 0; i < kFrames; ++i) {
    images.push_back(MakeCornerRichImage(static_cast<uint32_t>(300 + i)));
    vpr->AddFrame(i, images.back());
  }
  ASSERT_EQ(vpr->Size(), kFrames);
  vpr->Finalize();

  for (size_t i = 0; i < kFrames; ++i) {
    const VprMatch match = vpr->Query(images[i]);
    EXPECT_TRUE(match.found) << "frame " << i;
    EXPECT_EQ(match.node_id, i);
    EXPECT_GT(match.score, 0.5f) << "frame " << i;
  }
}

TEST(VprDBoW2Backend, TooFewFramesNeverMatch) {
  std::srand(kRngSeed);
  std::unique_ptr<IVpr> vpr = CreateVpr(DBoW2Options());
  ASSERT_NE(vpr, nullptr);

  std::vector<VprImage> images;
  for (size_t i = 0; i + 1 < VprDBoW2::kMinTrainingFrames; ++i) {
    images.push_back(MakeCornerRichImage(static_cast<uint32_t>(400 + i)));
    vpr->AddFrame(i, images.back());
  }
  ASSERT_EQ(vpr->Size(), VprDBoW2::kMinTrainingFrames - 1);

  // TF-IDF weights collapse on a handful of frames, so the backend refuses to answer rather than
  // answer on a score scale the caller's threshold does not mean anything on.
  vpr->Finalize();
  EXPECT_FALSE(vpr->Query(images.front()).found);
}

TEST(VprDBoW2Backend, RetrainingAfterEveryAddedFrameSurvives) {
  // Regression test for a crash in DBoW2's k-means, see SafeFORB in vpr_dbow2.cpp: a cluster that
  // loses its last descriptor while the centers move makes upstream release the center and then
  // dereference it as a null pointer. One training run rarely hits it; a map that is queried while
  // it grows retrains on every frame and hits it within a few dozen frames. Before the fix this
  // test died with SIGSEGV around the sixteenth frame.
  std::srand(kRngSeed);
  constexpr size_t kFrames = 20;
  std::unique_ptr<IVpr> vpr = CreateVpr(DBoW2Options());
  ASSERT_NE(vpr, nullptr);

  std::vector<VprImage> images;
  for (size_t i = 0; i < kFrames; ++i) {
    images.push_back(MakeCornerRichImage(static_cast<uint32_t>(500 + i)));
    vpr->AddFrame(i, images.back());

    const VprMatch match = vpr->Query(images.back());
    if (vpr->Size() < VprDBoW2::kMinTrainingFrames) {
      EXPECT_FALSE(match.found) << "frame " << i;
      continue;
    }
    // The frame just added is in the map, so it has to win against every earlier one.
    EXPECT_TRUE(match.found) << "frame " << i;
    EXPECT_EQ(match.node_id, i);
  }
  EXPECT_EQ(vpr->Size(), kFrames);
}

TEST(VprDBoW2Backend, DeserializedIndexKeepsTakingFrames) {
  // VprMap refuses this through its public API, but the backend has to stand on its own: adding to
  // a deserialized index retrains the vocabulary over imported and live descriptors together, which
  // is the path the k-means crash was first seen on.
  std::srand(kRngSeed);
  constexpr size_t kImported = 12;
  constexpr size_t kLive = 8;
  constexpr KeyFrameId kImportedBase = VprMap::kImportedNodeIdBase;

  std::vector<VprImage> images;
  std::unique_ptr<IVpr> source = CreateVpr(DBoW2Options());
  ASSERT_NE(source, nullptr);
  for (size_t i = 0; i < kImported + kLive; ++i) {
    images.push_back(MakeCornerRichImage(static_cast<uint32_t>(600 + i)));
    if (i < kImported) {
      source->AddFrame(i, images.back());
    }
  }
  Blob blob;
  source->Finalize();
  source->Serialize(blob);
  ASSERT_FALSE(blob.empty());

  std::unique_ptr<IVpr> restored = CreateVpr(DBoW2Options());
  ASSERT_NE(restored, nullptr);
  ASSERT_TRUE(restored->Deserialize(BlobReader(blob)));
  restored->ShiftNodeIds(kImportedBase);
  ASSERT_EQ(restored->Size(), kImported);
  EXPECT_EQ(restored->Query(images[0]).node_id, kImportedBase);

  for (size_t i = 0; i < kLive; ++i) {
    restored->AddFrame(i, images[kImported + i]);
    const VprMatch match = restored->Query(images[kImported + i]);
    EXPECT_TRUE(match.found) << "live frame " << i;
    EXPECT_EQ(match.node_id, i);
  }
  EXPECT_EQ(restored->Size(), kImported + kLive);

  // The imported half is still addressable under its shifted ids after the retraining.
  for (size_t i = 0; i < kImported; ++i) {
    const VprMatch match = restored->Query(images[i]);
    EXPECT_TRUE(match.found) << "imported frame " << i;
    EXPECT_EQ(match.node_id, kImportedBase + i);
  }
}

TEST(VprDBoW2Backend, RemoveNodeDropsIt) {
  std::srand(kRngSeed);
  constexpr size_t kFrames = 10;
  std::unique_ptr<IVpr> vpr = CreateVpr(DBoW2Options());
  ASSERT_NE(vpr, nullptr);

  std::vector<VprImage> images;
  for (size_t i = 0; i < kFrames; ++i) {
    images.push_back(MakeCornerRichImage(static_cast<uint32_t>(700 + i)));
    vpr->AddFrame(i, images.back());
  }

  vpr->RemoveNode(3);

  EXPECT_FALSE(vpr->HasNode(3));
  EXPECT_EQ(vpr->Size(), kFrames - 1);
  EXPECT_NE(vpr->Query(images[3]).node_id, 3u);
  // The entry that was swapped into the hole is still addressable under its own id.
  EXPECT_EQ(vpr->Query(images[kFrames - 1]).node_id, kFrames - 1);
}

TEST(VprDBoW2Backend, DeserializeRejectsForeignBlob) {
  const Blob garbage(64, 0x5A);

  std::unique_ptr<IVpr> vpr = CreateVpr(DBoW2Options());
  ASSERT_NE(vpr, nullptr);
  EXPECT_FALSE(vpr->Deserialize(BlobReader(garbage)));
  EXPECT_EQ(vpr->Size(), 0u);
}

TEST(VprDBoW2Factory, NameAndType) {
  EXPECT_STREQ(ToString(VprType::kDBoW2), "DBoW2");
  const std::unique_ptr<IVpr> vpr = CreateVpr(DBoW2Options());
  ASSERT_NE(vpr, nullptr);
  EXPECT_STREQ(vpr->Name(), "DBoW2");
}

TEST(VprDBoW2Backend, QueryTopKRanksEveryCandidateAndCapsTheCount) {
  std::srand(kRngSeed);
  constexpr size_t kFrames = 12;
  std::unique_ptr<IVpr> vpr = CreateVpr(DBoW2Options());
  ASSERT_NE(vpr, nullptr);

  std::vector<VprImage> images;
  for (size_t i = 0; i < kFrames; ++i) {
    images.push_back(MakeCornerRichImage(static_cast<uint32_t>(800 + i)));
    vpr->AddFrame(i, images.back());
  }
  vpr->Finalize();

  const std::vector<VprMatch> top = vpr->QueryTopK(images[4], 3);
  ASSERT_FALSE(top.empty());
  EXPECT_LE(top.size(), 3u);
  EXPECT_EQ(top.front().node_id, 4u);
  for (size_t i = 0; i < top.size(); ++i) {
    EXPECT_TRUE(top[i].found) << "position " << i;
    if (i > 0) {
      EXPECT_LE(top[i].score, top[i - 1].score) << "position " << i;
    }
  }

  EXPECT_EQ(vpr->QueryTopK(images[4], 1).size(), 1u);
  EXPECT_TRUE(vpr->QueryTopK(images[4], 0).empty());
}

TEST(VprDBoW2Backend, RatioTestRefusesAPlaceThatIsInTheMapTwice) {
  std::srand(kRngSeed);
  constexpr size_t kFrames = 12;
  constexpr KeyFrameId kFarAway = 1000;

  std::vector<VprImage> images;
  for (size_t i = 0; i < kFrames; ++i) {
    images.push_back(MakeCornerRichImage(static_cast<uint32_t>(900 + i)));
  }

  VprOptions guarded = DBoW2Options();
  guarded.ratio_threshold = 1.5f;
  std::unique_ptr<IVpr> vpr = CreateVpr(guarded);
  ASSERT_NE(vpr, nullptr);
  for (size_t i = 0; i < kFrames; ++i) {
    vpr->AddFrame(i, images[i]);
  }
  // The same place a second time, far enough along the trajectory to be a different place as far as
  // the neighbor guard is concerned. Two equally good answers is no answer.
  vpr->AddFrame(kFarAway, images[0]);
  EXPECT_FALSE(vpr->Query(images[0]).found);
  // A place that is only in the map once still answers, so the refusal above is the ratio test.
  EXPECT_TRUE(vpr->Query(images[5]).found);

  // And the same map without the ratio test answers for the repeated place too.
  std::unique_ptr<IVpr> unguarded = CreateVpr(DBoW2Options());
  ASSERT_NE(unguarded, nullptr);
  for (size_t i = 0; i < kFrames; ++i) {
    unguarded->AddFrame(i, images[i]);
  }
  unguarded->AddFrame(kFarAway, images[0]);
  EXPECT_TRUE(unguarded->Query(images[0]).found);
}

TEST(VprDBoW2Backend, DeserializeRejectsTruncatedBlobs) {
  std::srand(kRngSeed);
  constexpr size_t kFrames = 10;
  std::unique_ptr<IVpr> source = CreateVpr(DBoW2Options());
  ASSERT_NE(source, nullptr);
  for (size_t i = 0; i < kFrames; ++i) {
    source->AddFrame(i, MakeCornerRichImage(static_cast<uint32_t>(1000 + i)));
  }
  Blob blob;
  source->Finalize();
  source->Serialize(blob);
  ASSERT_GT(blob.size(), 64u);

  // A descriptor matrix that runs off the end of the blob leaves the index empty rather than
  // reading past it; Finalize() then rebuilds from whatever was read, which is nothing.
  for (const size_t length : {size_t{1}, size_t{8}, size_t{17}, blob.size() / 2, blob.size() - 1}) {
    const Blob truncated(blob.begin(), blob.begin() + static_cast<Blob::difference_type>(length));
    std::unique_ptr<IVpr> restored = CreateVpr(DBoW2Options());
    ASSERT_NE(restored, nullptr);
    EXPECT_FALSE(restored->Deserialize(BlobReader(truncated))) << length << " of " << blob.size() << " bytes";
  }
}

#endif  // USE_DBOW2

#ifdef USE_ONNXRUNTIME

namespace {

/// The DINOv2 ONNX model AnyLoc needs, or empty when this machine has none. The file is a 69 MB
/// artifact produced by cuvslam_export_dinov2 and is not in the repository, so the tests that need
/// it skip instead of failing; point CUVSLAM_ANYLOC_MODEL at one to run them.
std::string AnyLocModelPath() {
  const char* path = std::getenv("CUVSLAM_ANYLOC_MODEL");
  if (path == nullptr) {
    return {};
  }
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec) ? std::string(path) : std::string{};
}

VprOptions AnyLocOptions(const std::string& model_path) {
  VprOptions options;
  options.type = VprType::kAnyLoc;
  options.model_path = model_path;
  return options;
}

}  // namespace

TEST(VprAnyLocFactory, RefusesToConstructWithoutAModelFile) {
  // The path is checked while constructing, on the caller's thread, where a throw is an error the
  // caller can catch. The model itself is only read on the first frame, which happens on the SLAM
  // worker where a throw would be std::terminate.
  EXPECT_THROW(CreateVpr(AnyLocOptions("")), std::runtime_error);
  EXPECT_THROW(CreateVpr(AnyLocOptions("/nonexistent/dinov2.onnx")), std::runtime_error);

  const std::filesystem::path folder = std::filesystem::temp_directory_path() / "cuvslam_vpr_anyloc_not_a_file";
  std::filesystem::create_directories(folder);
  EXPECT_THROW(CreateVpr(AnyLocOptions(folder.string())), std::runtime_error);
  std::filesystem::remove_all(folder);
}

TEST(VprAnyLocBackend, VocabularyIsNotFittedBelowTheTrainingMinimum) {
  const std::string model = AnyLocModelPath();
  if (model.empty()) {
    GTEST_SKIP() << "set CUVSLAM_ANYLOC_MODEL to a DINOv2 ONNX model exported by cuvslam_export_dinov2";
  }

  std::unique_ptr<IVpr> vpr = CreateVpr(AnyLocOptions(model));
  ASSERT_NE(vpr, nullptr);
  EXPECT_STREQ(vpr->Name(), "AnyLoc");

  std::vector<VprImage> images;
  for (size_t i = 0; i + 1 < VprAnyLoc::kMinTrainingFrames; ++i) {
    images.push_back(MakeBlockyImage(256, 256, 8, static_cast<uint32_t>(1200 + i)));
    vpr->AddFrame(i, images.back());
  }
  ASSERT_EQ(vpr->Size(), VprAnyLoc::kMinTrainingFrames - 1);

  // The cluster centers are fitted once and then frozen, because re-fitting them would change the
  // meaning of every descriptor already stored. Fitting them on one or two frames would pin the
  // whole session's vocabulary to those, so below the minimum the backend answers nothing at all.
  vpr->Finalize();
  EXPECT_FALSE(vpr->Query(images.front()).found);

  // One more frame reaches the minimum, and then a mapped frame recognizes itself.
  images.push_back(MakeBlockyImage(256, 256, 8, 1300));
  vpr->AddFrame(VprAnyLoc::kMinTrainingFrames - 1, images.back());
  vpr->Finalize();
  const VprMatch match = vpr->Query(images.front());
  EXPECT_TRUE(match.found) << "score " << match.score;
  EXPECT_EQ(match.node_id, 0u);
}

TEST(VprAnyLocBackend, SerializeDeserializeKeepsAnswers) {
  const std::string model = AnyLocModelPath();
  if (model.empty()) {
    GTEST_SKIP() << "set CUVSLAM_ANYLOC_MODEL to a DINOv2 ONNX model exported by cuvslam_export_dinov2";
  }

  constexpr size_t kFrames = VprAnyLoc::kMinTrainingFrames;
  std::vector<VprImage> images;
  std::unique_ptr<IVpr> vpr = CreateVpr(AnyLocOptions(model));
  ASSERT_NE(vpr, nullptr);
  for (size_t i = 0; i < kFrames; ++i) {
    images.push_back(MakeBlockyImage(256, 256, 8, static_cast<uint32_t>(1400 + i)));
    vpr->AddFrame(i, images.back());
  }

  Blob blob;
  vpr->Serialize(blob);
  ASSERT_FALSE(blob.empty());

  // The blob carries the fitted cluster centers as well as the pooled descriptors, so the restored
  // index is queryable without the model having to be loaded or the vocabulary re-fitted.
  std::unique_ptr<IVpr> restored = CreateVpr(AnyLocOptions(model));
  ASSERT_NE(restored, nullptr);
  ASSERT_TRUE(restored->Deserialize(BlobReader(blob)));
  EXPECT_EQ(restored->Size(), kFrames);

  for (size_t i = 0; i < kFrames; ++i) {
    EXPECT_TRUE(restored->HasNode(i));
    const VprMatch match = restored->Query(images[i]);
    EXPECT_TRUE(match.found) << "frame " << i;
    EXPECT_EQ(match.node_id, i);
    EXPECT_NEAR(match.score, vpr->Query(images[i]).score, 1e-5f);
  }

  const Blob garbage(64, 0x5A);
  std::unique_ptr<IVpr> rejecting = CreateVpr(AnyLocOptions(model));
  ASSERT_NE(rejecting, nullptr);
  EXPECT_FALSE(rejecting->Deserialize(BlobReader(garbage)));
  EXPECT_EQ(rejecting->Size(), 0u);
}

#endif  // USE_ONNXRUNTIME

class VprMapStorage : public ::testing::Test {
protected:
  void SetUp() override {
    folder_ = std::filesystem::temp_directory_path() /
              ("cuvslam_vpr_test_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::filesystem::remove_all(folder_);
  }

  void TearDown() override { std::filesystem::remove_all(folder_); }

  std::string Folder() const { return folder_.string(); }

  std::filesystem::path folder_;
};

TEST_F(VprMapStorage, OneImagePerNode) {
  VprMap map(SimpleOptions());
  ASSERT_TRUE(map.Enabled());
  EXPECT_STREQ(map.BackendName(), "Simple");

  const VprImage first = MakeBlockyImage(64, 64, 8, 1);
  const VprImage second = MakeBlockyImage(64, 64, 8, 2);
  map.OnKeyframeAdded(3, 1000);

  EXPECT_TRUE(map.AddFrame(3, 1000, first));
  EXPECT_TRUE(map.HasImage(3));
  EXPECT_EQ(map.Size(), 1u);

  EXPECT_FALSE(map.AddFrame(3, 2000, second));
  EXPECT_EQ(map.Size(), 1u);
  // The node kept the image it was mapped with, not the one the second call offered.
  EXPECT_EQ(map.Query(first).node_id, 3u);

  EXPECT_FALSE(map.AddFrame(InvalidKeyFrameId, 3000, second));
  EXPECT_FALSE(map.AddFrame(4, 3000, VprImage{}));
  EXPECT_EQ(map.Size(), 1u);
}

TEST_F(VprMapStorage, KeyframeRemovalDropsTheEntry) {
  VprMap map(SimpleOptions());
  const VprImage image = MakeBlockyImage(64, 64, 8, 1);
  map.OnKeyframeAdded(3, 1000);
  ASSERT_TRUE(map.AddFrame(3, 1000, image));

  map.OnKeyframeRemoved(3, InvalidKeyFrameId);

  EXPECT_FALSE(map.HasImage(3));
  EXPECT_EQ(map.Size(), 0u);
  EXPECT_FALSE(map.Query(image).found);
}

TEST_F(VprMapStorage, SaveLoadRoundTrip) {
  constexpr size_t kNodes = 3;
  std::vector<VprImage> images;
  VprMap map(SimpleOptions());
  for (size_t i = 0; i < kNodes; ++i) {
    images.push_back(MakeBlockyImage(64, 64, 8, static_cast<uint32_t>(200 + i)));
    map.OnKeyframeAdded(i, static_cast<int64_t>(1000 + i));
    ASSERT_TRUE(map.AddFrame(i, static_cast<int64_t>(1000 + i), images.back()));
    map.UpdateNodePose(i, MakePose(static_cast<float>(i), 2.f * i, -1.f));
  }
  ASSERT_TRUE(map.Save(Folder()));

  VprMap loaded(SimpleOptions());
  ASSERT_TRUE(loaded.Load(Folder()));
  EXPECT_EQ(loaded.Size(), kNodes);

  for (size_t i = 0; i < kNodes; ++i) {
    const VprPlace place = loaded.Query(images[i]);
    ASSERT_TRUE(place.found) << "node " << i;
    EXPECT_EQ(place.node_id, i + VprMap::kImportedNodeIdBase);
    EXPECT_TRUE(place.imported);
    EXPECT_NEAR(place.score, 1.f, 1e-3f);
    EXPECT_EQ(place.timestamp_ns, static_cast<int64_t>(1000 + i));
    EXPECT_NEAR(place.pose.translation().x(), static_cast<float>(i), 1e-5f);
    EXPECT_NEAR(place.pose.translation().y(), 2.f * i, 1e-5f);
    EXPECT_NEAR(place.pose.translation().z(), -1.f, 1e-5f);
  }

  // An imported node keeps the pose the file carries: this session has no pose graph entry for it.
  loaded.UpdateNodePose(VprMap::kImportedNodeIdBase, MakePose(99.f, 99.f, 99.f));
  EXPECT_NEAR(loaded.Query(images[0]).pose.translation().x(), 0.f, 1e-5f);
}

TEST_F(VprMapStorage, LoadRejectsAnotherBackendsMap) {
  VprMap map(SimpleOptions());
  const VprImage image = MakeBlockyImage(64, 64, 8, 1);
  map.OnKeyframeAdded(1, 1000);
  ASSERT_TRUE(map.AddFrame(1, 1000, image));
  ASSERT_TRUE(map.Save(Folder()));

  // Patching the backend id in the saved file is the only way to get a foreign map here: the other
  // backends are compiled out of this build. It is the third uint32 of the header, see VprMap::Save.
  const std::filesystem::path path = folder_ / VprMap::kFileName;
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  ASSERT_TRUE(file.is_open());
  file.seekp(2 * sizeof(uint32_t));
  const auto foreign = static_cast<uint32_t>(VprType::kAnyLoc);
  ASSERT_TRUE(file.write(reinterpret_cast<const char*>(&foreign), sizeof(foreign)));
  file.close();

  VprMap loaded(SimpleOptions());
  EXPECT_FALSE(loaded.Load(Folder()));
  EXPECT_EQ(loaded.Size(), 0u);
  EXPECT_FALSE(loaded.Query(image).found);
}

TEST_F(VprMapStorage, LoadedMapIsReadOnly) {
  VprMap map(SimpleOptions());
  const VprImage image = MakeBlockyImage(64, 64, 8, 1);
  map.OnKeyframeAdded(1, 1000);
  ASSERT_TRUE(map.AddFrame(1, 1000, image));
  ASSERT_TRUE(map.Save(Folder()));

  VprMap loaded(SimpleOptions());
  ASSERT_TRUE(loaded.Load(Folder()));

  // The import renumbered the ids the backend holds and that is not reversible, so a loaded map can
  // neither be written back out nor take a second map on top of itself.
  EXPECT_FALSE(loaded.Save((folder_ / "again").string()));
  EXPECT_FALSE(loaded.Load(Folder()));
  EXPECT_EQ(loaded.Size(), 1u);
  EXPECT_TRUE(loaded.Query(image).found);
}

TEST_F(VprMapStorage, LoadedMapRefusesToGrow) {
  constexpr size_t kNodes = 3;
  std::vector<VprImage> images;
  VprMap map(SimpleOptions());
  EXPECT_FALSE(map.ReadOnly());
  for (size_t i = 0; i < kNodes; ++i) {
    images.push_back(MakeBlockyImage(64, 64, 8, static_cast<uint32_t>(300 + i)));
    map.OnKeyframeAdded(i, static_cast<int64_t>(1000 + i));
    ASSERT_TRUE(map.AddFrame(i, static_cast<int64_t>(1000 + i), images.back()));
  }
  ASSERT_TRUE(map.Save(Folder()));

  VprMap loaded(SimpleOptions());
  EXPECT_FALSE(loaded.ReadOnly());
  ASSERT_TRUE(loaded.Load(Folder()));
  EXPECT_TRUE(loaded.ReadOnly());

  // The session that loaded the map goes on tracking and goes on offering its own keyframes. Those
  // are of places it is looking at right now, so letting them in would make almost every query
  // resolve to a frame from a few seconds ago instead of to the map it was asked to localize in.
  const VprImage live = MakeBlockyImage(64, 64, 8, 909);
  loaded.OnKeyframeAdded(1, 9000);
  EXPECT_FALSE(loaded.AddFrame(1, 9000, live));
  EXPECT_FALSE(loaded.HasImage(1));
  EXPECT_EQ(loaded.Size(), kNodes);

  // Turning the frame away left the imported entries answering as before.
  const VprPlace place = loaded.Query(images[0]);
  EXPECT_TRUE(place.found);
  EXPECT_EQ(place.node_id, VprMap::kImportedNodeIdBase);
  EXPECT_TRUE(place.imported);
}

TEST_F(VprMapStorage, LoadRejectsMissingAndTruncatedFiles) {
  VprMap map(SimpleOptions());
  EXPECT_FALSE(map.Load(Folder()));

  std::filesystem::create_directories(folder_);
  {
    std::ofstream file(folder_ / VprMap::kFileName, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    const std::vector<uint8_t> garbage(9, 0xCD);
    file.write(reinterpret_cast<const char*>(garbage.data()), static_cast<std::streamsize>(garbage.size()));
  }

  EXPECT_FALSE(map.Load(Folder()));
  EXPECT_EQ(map.Size(), 0u);
}

TEST_F(VprMapStorage, DisabledMapIsANoOp) {
  VprOptions options;
  options.type = VprType::kNone;
  VprMap map(options);
  const VprImage image = MakeBlockyImage(64, 64, 8, 1);

  EXPECT_FALSE(map.Enabled());
  EXPECT_STREQ(map.BackendName(), "None");
  EXPECT_EQ(map.Size(), 0u);

  map.OnKeyframeAdded(1, 1000);
  EXPECT_FALSE(map.AddFrame(1, 1000, image));
  EXPECT_FALSE(map.HasImage(1));
  EXPECT_FALSE(map.Query(image).found);
  map.UpdateNodePose(1, MakePose(1.f, 2.f, 3.f));
  map.OnKeyframeRemoved(1, InvalidKeyFrameId);
  EXPECT_FALSE(map.Save(Folder()));
  EXPECT_FALSE(map.Load(Folder()));
  EXPECT_EQ(map.Size(), 0u);
}

TEST_F(VprMapStorage, QueryTopKRanksTheCandidatesAndCapsTheCount) {
  constexpr size_t kNodes = 4;
  std::vector<VprImage> images;
  VprMap map(SimpleOptions());
  for (size_t i = 0; i < kNodes; ++i) {
    images.push_back(MakeBlockyImage(64, 64, 8, static_cast<uint32_t>(400 + i)));
    map.OnKeyframeAdded(i, static_cast<int64_t>(1000 + i));
    ASSERT_TRUE(map.AddFrame(i, static_cast<int64_t>(1000 + i), images.back()));
    map.UpdateNodePose(i, MakePose(static_cast<float>(i), 0.f, 0.f));
  }

  const std::vector<VprPlace> places = map.QueryTopK(images[2], kNodes + 1);
  ASSERT_EQ(places.size(), kNodes);
  EXPECT_EQ(places.front().node_id, 2u);
  EXPECT_NEAR(places.front().pose.translation().x(), 2.f, 1e-5f);
  for (size_t i = 0; i < places.size(); ++i) {
    EXPECT_TRUE(places[i].found) << "position " << i;
    EXPECT_FALSE(places[i].imported) << "position " << i;
    EXPECT_EQ(places[i].timestamp_ns, static_cast<int64_t>(1000 + places[i].node_id));
    if (i > 0) {
      EXPECT_LE(places[i].score, places[i - 1].score) << "position " << i;
    }
  }

  EXPECT_EQ(map.QueryTopK(images[2], 2).size(), 2u);
  EXPECT_TRUE(map.QueryTopK(images[2], 0).empty());
  EXPECT_TRUE(map.QueryTopK(VprImage{}, 3).empty());
}

TEST_F(VprMapStorage, AMergedNodeIsDroppedAndTheSurvivorKeepsItsOwnImage) {
  VprMap map(SimpleOptions());
  const VprImage survivor_image = MakeBlockyImage(64, 64, 8, 61);
  const VprImage merged_image = MakeBlockyImage(64, 64, 8, 62);
  map.OnKeyframeAdded(1, 1000);
  map.OnKeyframeAdded(2, 2000);
  ASSERT_TRUE(map.AddFrame(1, 1000, survivor_image));
  ASSERT_TRUE(map.AddFrame(2, 2000, merged_image));

  map.OnKeyframeRemoved(2, 1);

  // A descriptor is only addressable through a node that still exists, so the merged node's goes.
  // The survivor does not inherit it: the two frames were taken from different viewpoints, and
  // re-keying the descriptor would need the pixels again.
  EXPECT_EQ(map.Size(), 1u);
  EXPECT_FALSE(map.HasImage(2));
  EXPECT_TRUE(map.HasImage(1));
  EXPECT_EQ(map.Query(survivor_image).node_id, 1u);
  EXPECT_FALSE(map.Query(merged_image).found);
}

TEST_F(VprMapStorage, LoadWithPoseGraphKeepsTheIdsAndStaysWritable) {
  constexpr size_t kNodes = 3;
  std::vector<VprImage> images;
  VprMap map(SimpleOptions());
  for (size_t i = 0; i < kNodes; ++i) {
    images.push_back(MakeBlockyImage(64, 64, 8, static_cast<uint32_t>(500 + i)));
    map.OnKeyframeAdded(i, static_cast<int64_t>(1000 + i));
    ASSERT_TRUE(map.AddFrame(i, static_cast<int64_t>(1000 + i), images.back()));
    map.UpdateNodePose(i, MakePose(static_cast<float>(i), 0.f, 0.f));
  }
  ASSERT_TRUE(map.Save(Folder()));

  // The session that loads this way IS the session that wrote it: it restored the same pose graph
  // from the same folder, so the ids in the file are its own and nothing has to be renumbered.
  VprMap loaded(SimpleOptions());
  ASSERT_TRUE(loaded.Load(Folder(), VprMap::LoadMode::kWithPoseGraph));
  EXPECT_FALSE(loaded.ReadOnly());
  EXPECT_EQ(loaded.Size(), kNodes);

  for (size_t i = 0; i < kNodes; ++i) {
    const VprPlace place = loaded.Query(images[i]);
    ASSERT_TRUE(place.found) << "node " << i;
    EXPECT_EQ(place.node_id, i);
    EXPECT_FALSE(place.imported) << "node " << i;
    EXPECT_EQ(place.timestamp_ns, static_cast<int64_t>(1000 + i));
  }

  // Not imported, so the caller's pose graph is what resolves these nodes from now on.
  loaded.UpdateNodePose(0, MakePose(7.f, 0.f, 0.f));
  EXPECT_NEAR(loaded.Query(images[0]).pose.translation().x(), 7.f, 1e-5f);

  // And the map is still a map: it takes new nodes and can be written back out.
  const VprImage live = MakeBlockyImage(64, 64, 8, 555);
  loaded.OnKeyframeAdded(kNodes, 5000);
  EXPECT_TRUE(loaded.AddFrame(kNodes, 5000, live));
  EXPECT_EQ(loaded.Size(), kNodes + 1);
  EXPECT_EQ(loaded.Query(live).node_id, kNodes);
  EXPECT_TRUE(loaded.Save((folder_ / "again").string()));
}

TEST_F(VprMapStorage, LoadRejectsAnAbsurdEntryCount) {
  VprMap map(SimpleOptions());
  const VprImage image = MakeBlockyImage(64, 64, 8, 1);
  map.OnKeyframeAdded(1, 1000);
  ASSERT_TRUE(map.AddFrame(1, 1000, image));
  ASSERT_TRUE(map.Save(Folder()));

  // The entry count is the fourth word of the header, see VprMap::Save. A count larger than the
  // file could possibly hold has to be refused before it is reserved for, so a corrupt map comes
  // back as a clean false instead of a bad_alloc.
  const std::filesystem::path path = folder_ / VprMap::kFileName;
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  ASSERT_TRUE(file.is_open());
  file.seekp(3 * sizeof(uint32_t));
  const uint64_t absurd = std::numeric_limits<uint64_t>::max();
  ASSERT_TRUE(file.write(reinterpret_cast<const char*>(&absurd), sizeof(absurd)));
  file.close();

  VprMap loaded(SimpleOptions());
  EXPECT_FALSE(loaded.Load(Folder()));
  EXPECT_EQ(loaded.Size(), 0u);
  EXPECT_FALSE(loaded.Query(image).found);
}

TEST(VprSofImage, ReadsTheCpuPyramidAndRefusesAnUnfilledContext) {
  const VprImage source = MakeBlockyImage(64, 48, 8, 71);

  EXPECT_TRUE(MakeVprImageFromContext(nullptr).Empty());

  // A context that has been allocated but not filled - the state a slot the image pool recycled is
  // in - reports neither pyramid, and reading its CPU getters would trip their assert.
  auto unfilled = std::make_shared<sof::ImageContext>(ImageShape{source.width, source.height}, false, false);
  EXPECT_FALSE(unfilled->has_cpu_image());
  EXPECT_FALSE(unfilled->has_gpu_image());
  EXPECT_TRUE(MakeVprImageFromContext(unfilled).Empty());

  // Level 0 of the CPU pyramid is the frame as the tracker received it, pixel for pixel.
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
  // Camera indexed and sparse: a camera absent from this frame is a nullptr, not a gap in the list.
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

  // The CPU pyramid was never built, so this has to come down from the device - as float luminance,
  // which MakeVprImage rounds back to the bytes that went up.
  EXPECT_FALSE(context->has_cpu_image());
  ASSERT_TRUE(context->has_gpu_image());
  const VprImage image = MakeVprImageFromContext(context);
  ASSERT_FALSE(image.Empty());
  EXPECT_EQ(image.width, source.width);
  EXPECT_EQ(image.height, source.height);
  EXPECT_EQ(image.row, source.row);
}

#endif  // USE_CUDA

/// A pose graph with place recognition on it, built without any images or landmarks: keyframes are
/// added from synthetic odometry deltas and each is given a distinct blocky picture by hand.
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

    mapper_ = MakeMapper(SimpleOptions());
  }

  void TearDown() override { std::filesystem::remove_all(folder_); }

  std::string Folder() const { return folder_.string(); }

  std::unique_ptr<LocalizerAndMapper> MakeMapper(const VprOptions& options) {
    auto mapper = std::make_unique<LocalizerAndMapper>(rig_, FeatureDescriptorType::kNone, true);
    mapper->SetActiveCameras({0});
    mapper->SetVprOptions(options);
    EXPECT_TRUE(mapper->SetPoseGraphOptimizerOptions({PoseGraphOptimizerType::Simple}));
    return mapper;
  }

  /// Append one pose graph node `step_x` meters along X from the last, with no picture attached.
  /// Returns its timestamp.
  int64_t AppendKeyframe(int index, float step_x) {
    Isometry3T step = Isometry3T::Identity();
    step.translation().x() = step_x;

    VOFrameData frame_data;
    frame_data.frame_id = index;
    frame_data.timestamp_ns = static_cast<uint64_t>(index) * kFrameDeltaNs;
    mapper_->AddKeyframe(step, frame_data, Images());
    return static_cast<int64_t>(frame_data.timestamp_ns);
  }

  /// Walk `count` steps along X, mapping one picture per keyframe.
  void BuildChain(int count, float step_x) {
    for (int i = 0; i < count; ++i) {
      const int64_t timestamp_ns = AppendKeyframe(i, step_x);
      images_.push_back(MakeBlockyImage(64, 64, 8, static_cast<uint32_t>(800 + i)));
      ASSERT_TRUE(mapper_->AddVprFrame(images_.back(), timestamp_ns)) << "keyframe " << i;
    }
  }

  /// Merge one pose graph node into a neighbor and check that both books hanging off PoseGraph's
  /// single removal callback followed it: the frame pose bookkeeping and the recognition map.
  void ExpectAMergeUpdatesBothBooks() {
    // PoseGraph guards its 20 newest keyframes against reduction, so the chain has to be longer.
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

    // The recognition map dropped the node that disappeared, and only that one: a descriptor keyed
    // by a node that no longer exists can never be resolved to a pose again.
    EXPECT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(kChain) - 1);
    EXPECT_NE(mapper_->RecognizePlace(images_[merged_away]).node_id, merged_away);
    for (const KeyFrameId id : alive) {
      const VprPlace place = mapper_->RecognizePlace(images_[id]);
      EXPECT_TRUE(place.found) << "node " << id;
      EXPECT_EQ(place.node_id, id) << "node " << id;
    }

    // And the frame pose bookkeeping did too: the frame that created the merged node now resolves
    // through the node it was merged into, so shifting every surviving node shifts it as well.
    // Without the merge recorded it would keep resolving through the merged node, whose pose the
    // hypothesis still holds untouched, and would not move at all.
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

  /// The pose graph, which LocalizerAndMapper only hands out as a const view. A loop closure
  /// normally enters through ApplyLoopClosureResult, which needs the landmarks this map has none of.
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
    EXPECT_FALSE(place.imported);
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

  // A loop closure that lands 0.2 m short of where the chain says the last keyframe is, so the
  // optimizer has drift to spread back over the nodes.
  Map& map = MutableMap();
  const Isometry3T loop_from_head = MakePose(-(kKeyframes - 1) * kStep + 0.2f, 0.f, 0.f);
  map.pose_graph_.AddEdge(map.pose_graph_hypothesis_, before.back().node_id, before.front().node_id, loop_from_head,
                          Matrix6T::Identity());
  ASSERT_TRUE(mapper_->OptimizePoseGraph(false));

  float moved = 0.f;
  for (size_t i = 0; i < images_.size(); ++i) {
    const VprPlace place = mapper_->RecognizePlace(images_[i]);
    ASSERT_TRUE(place.found) << "frame " << i;
    // The same picture still names the same node; only where that node sits has changed.
    EXPECT_EQ(place.node_id, before[i].node_id) << "frame " << i;

    const Isometry3T* node_pose = map.pose_graph_hypothesis_.GetKeyframePose(place.node_id);
    ASSERT_NE(node_pose, nullptr) << "frame " << i;
    EXPECT_LT((place.pose.translation() - node_pose->translation()).norm(), 1e-5f) << "frame " << i;

    moved = std::max(moved, (place.pose.translation() - before[i].pose.translation()).norm());
  }

  // Without this the checks above hold trivially for a backend that answers from the pose stored
  // beside the descriptor, because the stored pose and the optimized one would still be the same.
  EXPECT_GT(moved, 1e-3f) << "the optimization moved no node, so this test proved nothing";
}

TEST_F(VprPoseGraph, TheHeadNodeNeedsAPictureUntilItHasOne) {
  EXPECT_TRUE(mapper_->IsVprEnabled());
  // Nothing has been mapped yet, so there is no node for a picture to belong to. The frame is
  // dropped rather than queued: the caller offers one on every frame anyway.
  EXPECT_FALSE(mapper_->VprHeadNeedsImage());
  EXPECT_FALSE(mapper_->AddVprFrame(MakeBlockyImage(64, 64, 8, 900), 0));
  EXPECT_EQ(mapper_->GetVprMapSize(), 0u);

  const int64_t timestamp_ns = AppendKeyframe(0, 1.f);
  EXPECT_TRUE(mapper_->VprHeadNeedsImage());

  const VprImage first = MakeBlockyImage(64, 64, 8, 901);
  EXPECT_TRUE(mapper_->AddVprFrame(first, timestamp_ns));
  EXPECT_EQ(mapper_->GetVprMapSize(), 1u);
  // One picture per node: everything offered after the first lands on a node that already has one,
  // which is what lets a caller hand over every single frame.
  EXPECT_FALSE(mapper_->VprHeadNeedsImage());
  EXPECT_FALSE(mapper_->AddVprFrame(MakeBlockyImage(64, 64, 8, 902), timestamp_ns));
  EXPECT_EQ(mapper_->GetVprMapSize(), 1u);

  // A new node needs one again, and gets its own.
  const int64_t next_ns = AppendKeyframe(1, 1.f);
  EXPECT_TRUE(mapper_->VprHeadNeedsImage());
  const VprImage second = MakeBlockyImage(64, 64, 8, 903);
  EXPECT_TRUE(mapper_->AddVprFrame(second, next_ns));
  EXPECT_EQ(mapper_->GetVprMapSize(), 2u);
  EXPECT_EQ(mapper_->RecognizePlace(second).timestamp_ns, next_ns);
}

TEST_F(VprPoseGraph, RecognizePlacesRanksSeveralNodesAgainstTheLivePoseGraph) {
  BuildChain(5, 1.f);

  const std::vector<VprPlace> places = mapper_->RecognizePlaces(images_[3], 3);
  ASSERT_EQ(places.size(), 3u);
  EXPECT_EQ(places.front().node_id, mapper_->RecognizePlace(images_[3]).node_id);
  for (size_t i = 0; i < places.size(); ++i) {
    EXPECT_TRUE(places[i].found) << "position " << i;
    EXPECT_FALSE(places[i].imported) << "position " << i;
    const Isometry3T* node_pose = MutableMap().pose_graph_hypothesis_.GetKeyframePose(places[i].node_id);
    ASSERT_NE(node_pose, nullptr) << "position " << i;
    EXPECT_LT((places[i].pose.translation() - node_pose->translation()).norm(), 1e-5f) << "position " << i;
    if (i > 0) {
      EXPECT_LE(places[i].score, places[i - 1].score) << "position " << i;
    }
  }

  EXPECT_TRUE(mapper_->RecognizePlaces(images_[3], 0).empty());
}

TEST_F(VprPoseGraph, SaveVprMapWritesThePosesTheGraphHoldsNow) {
  constexpr int kKeyframes = 8;
  constexpr float kStep = 1.f;
  BuildChain(kKeyframes, kStep);

  // Move the graph after the frames were mapped, so a map that wrote the poses it happened to store
  // at mapping time would carry different numbers from the ones the graph holds now.
  Map& map = MutableMap();
  map.pose_graph_.AddEdge(map.pose_graph_hypothesis_, kKeyframes - 1, 0,
                          MakePose(-(kKeyframes - 1) * kStep + 0.2f, 0.f, 0.f), Matrix6T::Identity());
  ASSERT_TRUE(mapper_->OptimizePoseGraph(false));
  ASSERT_TRUE(mapper_->SaveVprMap(Folder()));

  // A session that has no pose graph for these nodes reads them back with the poses the writing
  // session ended with, which is exactly the situation a saved map is for.
  std::unique_ptr<LocalizerAndMapper> reader = MakeMapper(SimpleOptions());
  ASSERT_TRUE(reader->LoadVprMap(Folder()));
  EXPECT_EQ(reader->GetVprMapSize(), static_cast<size_t>(kKeyframes));

  for (size_t i = 0; i < images_.size(); ++i) {
    const VprPlace place = reader->RecognizePlace(images_[i]);
    ASSERT_TRUE(place.found) << "frame " << i;
    EXPECT_TRUE(place.imported) << "frame " << i;
    EXPECT_EQ(place.node_id, i + VprMap::kImportedNodeIdBase);
    EXPECT_EQ(place.timestamp_ns, static_cast<int64_t>(i) * kFrameDeltaNs);

    const Isometry3T* optimized = map.pose_graph_hypothesis_.GetKeyframePose(i);
    ASSERT_NE(optimized, nullptr) << "frame " << i;
    EXPECT_LT((place.pose.translation() - optimized->translation()).norm(), 1e-4f) << "frame " << i;
  }
}

TEST_F(VprPoseGraph, LoadVprMapWithThePoseGraphKeepsIdsAndResolvesThroughTheLiveGraph) {
  constexpr int kKeyframes = 4;
  BuildChain(kKeyframes, 1.f);
  ASSERT_TRUE(mapper_->SaveVprMap(Folder()));

  // A relocalizer restores the pose graph from the same folder first, so the node ids in the file
  // are its own. Here that graph is rebuilt with a different step, which is what tells the two
  // possible sources of a reported pose apart: the file says the nodes are 1 m apart, this graph
  // says 5 m.
  mapper_ = MakeMapper(SimpleOptions());
  for (int i = 0; i < kKeyframes; ++i) {
    AppendKeyframe(i, 5.f);
  }
  ASSERT_TRUE(mapper_->LoadVprMap(Folder(), VprMap::LoadMode::kWithPoseGraph));
  EXPECT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(kKeyframes));

  for (size_t i = 0; i < images_.size(); ++i) {
    const VprPlace place = mapper_->RecognizePlace(images_[i]);
    ASSERT_TRUE(place.found) << "frame " << i;
    EXPECT_EQ(place.node_id, i) << "frame " << i;
    EXPECT_FALSE(place.imported) << "frame " << i;
    const Isometry3T* live = MutableMap().pose_graph_hypothesis_.GetKeyframePose(i);
    ASSERT_NE(live, nullptr) << "frame " << i;
    EXPECT_LT((place.pose.translation() - live->translation()).norm(), 1e-5f) << "frame " << i;
    EXPECT_NEAR(place.pose.translation().x(), 5.f * static_cast<float>(i + 1), 1e-4f) << "frame " << i;
  }

  // And unlike a standalone import it stays writable, because nothing about it was renumbered.
  const int64_t timestamp_ns = AppendKeyframe(kKeyframes, 5.f);
  EXPECT_TRUE(mapper_->VprHeadNeedsImage());
  EXPECT_TRUE(mapper_->AddVprFrame(MakeBlockyImage(64, 64, 8, 990), timestamp_ns));
  EXPECT_EQ(mapper_->GetVprMapSize(), static_cast<size_t>(kKeyframes) + 1);
}

TEST_F(VprPoseGraph, TurningVprOffLeavesEveryEntryPointInert) {
  BuildChain(2, 1.f);
  ASSERT_TRUE(mapper_->IsVprEnabled());
  ASSERT_EQ(mapper_->GetVprMapSize(), 2u);

  mapper_->SetVprOptions(VprOptions{});

  EXPECT_FALSE(mapper_->IsVprEnabled());
  EXPECT_EQ(mapper_->GetVprMapSize(), 0u);
  EXPECT_FALSE(mapper_->VprHeadNeedsImage());
  EXPECT_FALSE(mapper_->AddVprFrame(images_.front(), 0));
  EXPECT_FALSE(mapper_->RecognizePlace(images_.front()).found);
  EXPECT_TRUE(mapper_->RecognizePlaces(images_.front(), 3).empty());
  EXPECT_FALSE(mapper_->SaveVprMap(Folder()));
  EXPECT_FALSE(mapper_->LoadVprMap(Folder()));
}

TEST_F(VprPoseGraph, AMergeFollowsBothBooksWhenPlaceRecognitionWasInstalledLast) {
  // PoseGraph holds exactly one removal callback and two features need it, so whichever of these
  // two calls ran second used to silently disable the other. This is the order AsyncSlam makes them.
  mapper_ = MakeMapper(VprOptions{});
  mapper_->SetKeepTrackPoses(true);
  mapper_->SetVprOptions(SimpleOptions());

  ExpectAMergeUpdatesBothBooks();
}

TEST_F(VprPoseGraph, AMergeFollowsBothBooksWhenFramePosesWereInstalledLast) {
  // The same, the other way round: SetUp installed place recognition, and turning frame poses on
  // re-registers the callback.
  mapper_->SetKeepTrackPoses(true);

  ExpectAMergeUpdatesBothBooks();
}

// ---------------------------------------------------------------------------------------------
// In-tree ORB and the binary bag of words built on it
// ---------------------------------------------------------------------------------------------

namespace {

/// An image with enough corner structure for FAST to fire on, unlike the flat tiles the thumbnail
/// tests use: a grid of bright squares on a dark field, jittered by `seed` so two seeds are two
/// different places. 256x256 is the smallest size that leaves room for the 22 pixel patch border on
/// several pyramid levels.
VprImage MakeCorneredImage(uint32_t seed) {
  constexpr int kSize = 256;
  VprImage image;
  image.width = kSize;
  image.height = kSize;
  image.row.assign(static_cast<size_t>(kSize) * kSize, 30);

  std::mt19937 rng(seed);
  for (int block_y = 0; block_y < 12; ++block_y) {
    for (int block_x = 0; block_x < 12; ++block_x) {
      const int x0 = 20 + block_x * 18 + static_cast<int>(rng() % 5u);
      const int y0 = 20 + block_y * 18 + static_cast<int>(rng() % 5u);
      const auto value = static_cast<uint8_t>(140 + rng() % 110u);
      for (int y = y0; y < y0 + 7 && y < kSize; ++y) {
        for (int x = x0; x < x0 + 7 && x < kSize; ++x) {
          image.row[static_cast<size_t>(y) * kSize + x] = value;
        }
      }
    }
  }
  return image;
}

}  // namespace

TEST(VprOrbExtractor, FindsRepeatableCornersAndDescribesThemIdentically) {
  const OrbExtractor orb;
  const VprImage image = MakeCorneredImage(1);

  const std::vector<OrbKeypoint> first = orb.Extract(image);
  ASSERT_FALSE(first.empty());
  EXPECT_LE(static_cast<int>(first.size()), OrbExtractor::kMaxFeatures);

  // The extractor has no state, so the same image has to give the same keypoints and the same bits.
  const std::vector<OrbKeypoint> again = orb.Extract(image);
  ASSERT_EQ(again.size(), first.size());
  for (size_t i = 0; i < first.size(); ++i) {
    EXPECT_FLOAT_EQ(again[i].x, first[i].x);
    EXPECT_FLOAT_EQ(again[i].y, first[i].y);
    EXPECT_EQ(again[i].descriptor, first[i].descriptor);
  }

  // Two different places must not produce the same descriptors.
  const std::vector<OrbKeypoint> other = orb.Extract(MakeCorneredImage(2));
  ASSERT_FALSE(other.empty());
  int identical = 0;
  for (const OrbKeypoint& a : first) {
    for (const OrbKeypoint& b : other) {
      if (a.descriptor == b.descriptor) {
        ++identical;
      }
    }
  }
  EXPECT_EQ(identical, 0);
}

TEST(VprOrbExtractor, RefusesAnImageTooSmallForThePatch) {
  const OrbExtractor orb;
  EXPECT_TRUE(orb.Extract(VprImage{}).empty());
  EXPECT_TRUE(orb.Extract(MakeBlockyImage(20, 20, 4, 3)).empty());
}

TEST(VprBowVocabulary, FitsDeterministicWordsAndQuantizesToTheNearest) {
  std::vector<BowVocabulary::Descriptor> descriptors;
  std::mt19937 rng(7);
  for (int cluster = 0; cluster < 4; ++cluster) {
    BowVocabulary::Descriptor seed{};
    for (uint8_t& byte : seed) {
      byte = static_cast<uint8_t>(rng() % 256u);
    }
    for (int i = 0; i < 25; ++i) {
      BowVocabulary::Descriptor noisy = seed;
      noisy[static_cast<size_t>(rng() % noisy.size())] ^= static_cast<uint8_t>(1u << (rng() % 8u));
      descriptors.push_back(noisy);
    }
  }

  BowVocabulary first;
  first.Train(descriptors, 4);
  ASSERT_TRUE(first.Ready());
  EXPECT_EQ(first.WordCount(), 4);

  // Deterministic: the seeding uses a fixed generator, so a second fit gives the same words.
  BowVocabulary second;
  second.Train(descriptors, 4);
  EXPECT_EQ(second.Centers(), first.Centers());

  // Every descriptor quantizes to the center it is closest to, by construction of Assign().
  for (const BowVocabulary::Descriptor& descriptor : descriptors) {
    const int word = first.Assign(descriptor);
    const int distance = BowVocabulary::Hamming(descriptor, first.Centers()[static_cast<size_t>(word)]);
    for (const BowVocabulary::Descriptor& center : first.Centers()) {
      EXPECT_LE(distance, BowVocabulary::Hamming(descriptor, center));
    }
  }

  EXPECT_EQ(BowVocabulary::Hamming(descriptors[0], descriptors[0]), 0);
}

TEST(VprBowVocabulary, AsksForNoMoreWordsThanThereAreDescriptors) {
  BowVocabulary vocabulary;
  vocabulary.Train({}, 16);
  EXPECT_FALSE(vocabulary.Ready());

  const std::vector<BowVocabulary::Descriptor> three(3);
  vocabulary.Train(three, 16);
  EXPECT_EQ(vocabulary.WordCount(), 3);
}

class VprBowBackend : public ::testing::Test {
protected:
  void SetUp() override {
    VprOptions options;
    options.type = VprType::kBow;
    vpr_ = CreateVpr(options);
    ASSERT_NE(vpr_, nullptr);

    // One more frame than the vocabulary needs, so the fit fires and the last frame is encoded
    // against the frozen words rather than being part of the fit.
    for (size_t i = 0; i < VprBow::kMinTrainingFrames + 1; ++i) {
      images_.push_back(MakeCorneredImage(static_cast<uint32_t>(100 + i)));
      vpr_->AddFrame(i, images_.back());
    }
  }

  std::unique_ptr<IVpr> vpr_;
  std::vector<VprImage> images_;
};

TEST_F(VprBowBackend, RecognizesEveryMappedImage) {
  EXPECT_STREQ(vpr_->Name(), "Bow");
  ASSERT_EQ(vpr_->Size(), images_.size());

  for (size_t i = 0; i < images_.size(); ++i) {
    const VprMatch match = vpr_->Query(images_[i]);
    EXPECT_TRUE(match.found) << "frame " << i;
    EXPECT_EQ(match.node_id, i);
    EXPECT_NEAR(match.score, 1.f, 1e-3f) << "frame " << i;
  }
}

TEST_F(VprBowBackend, QueryTopKRanksTheWinnerFirstAndCapsTheCount) {
  const std::vector<VprMatch> matches = vpr_->QueryTopK(images_[3], 4);
  ASSERT_FALSE(matches.empty());
  EXPECT_LE(matches.size(), 4u);
  EXPECT_EQ(matches.front().node_id, 3u);
  for (size_t i = 1; i < matches.size(); ++i) {
    EXPECT_LE(matches[i].score, matches[i - 1].score);
  }
  EXPECT_TRUE(vpr_->QueryTopK(images_[0], 0).empty());
}

TEST_F(VprBowBackend, RemoveNodeDropsItAndShiftNodeIdsRenumbers) {
  vpr_->RemoveNode(2);
  EXPECT_FALSE(vpr_->HasNode(2));
  EXPECT_EQ(vpr_->Size(), images_.size() - 1);
  EXPECT_NE(vpr_->Query(images_[2]).node_id, 2u);

  vpr_->ShiftNodeIds(1000);
  EXPECT_TRUE(vpr_->HasNode(1003));
  EXPECT_EQ(vpr_->Query(images_[3]).node_id, 1003u);
}

TEST_F(VprBowBackend, SerializeDeserializeKeepsAnswers) {
  Blob blob;
  vpr_->Serialize(blob);
  ASSERT_FALSE(blob.empty());

  VprOptions options;
  options.type = VprType::kBow;
  const std::unique_ptr<IVpr> restored = CreateVpr(options);
  const BlobReader reader(blob);
  ASSERT_TRUE(restored->Deserialize(reader));
  EXPECT_EQ(restored->Size(), vpr_->Size());

  for (size_t i = 0; i < images_.size(); ++i) {
    const VprMatch match = restored->Query(images_[i]);
    EXPECT_TRUE(match.found) << "frame " << i;
    EXPECT_EQ(match.node_id, i);
  }
}

TEST_F(VprBowBackend, DeserializeRejectsForeignAndTruncatedBlobs) {
  Blob blob;
  vpr_->Serialize(blob);
  ASSERT_GT(blob.size(), 8u);

  VprOptions options;
  options.type = VprType::kBow;

  Blob foreign = blob;
  foreign[0] ^= 0xFFu;
  EXPECT_FALSE(CreateVpr(options)->Deserialize(BlobReader(foreign)));

  for (size_t cut = 1; cut < blob.size(); cut += std::max<size_t>(1, blob.size() / 16)) {
    const Blob truncated(blob.begin(), blob.begin() + static_cast<long>(cut));
    EXPECT_FALSE(CreateVpr(options)->Deserialize(BlobReader(truncated))) << "cut at " << cut;
  }
}

TEST(VprBowFactory, TooFewFramesNeverMatch) {
  VprOptions options;
  options.type = VprType::kBow;
  const std::unique_ptr<IVpr> vpr = CreateVpr(options);
  ASSERT_NE(vpr, nullptr);

  std::vector<VprImage> images;
  for (size_t i = 0; i + 1 < VprBow::kMinTrainingFrames; ++i) {
    images.push_back(MakeCorneredImage(static_cast<uint32_t>(400 + i)));
    vpr->AddFrame(i, images.back());
  }

  // The words and the inverse document frequencies are fitted once and frozen, so fitting them to a
  // handful of frames would pin the whole session to noise. Answering nothing is the honest outcome.
  EXPECT_FALSE(vpr->Query(images.front()).found);
}

}  // namespace test::vpr
