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

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "slam/common/blob.h"
#include "slam/map/descriptor/st_descriptor_ops.h"
#include "slam/map/pose_graph/pose_graph_hypothesis.h"

#ifdef USE_LMDB
#include "slam/map/database/lmdb_slam_database.h"
#endif

#include <unistd.h>

namespace test::slam {
using namespace cuvslam;
using namespace cuvslam::slam;

namespace {

constexpr size_t kHugeCount = std::numeric_limits<size_t>::max() / 2;

// A descriptor blob as STDescriptorOps::ToBlob writes it, with `declared_patches` in the header and
// `stored_patches` patches after it.
Blob MakeDescriptorBlob(uint32_t declared_patches, uint32_t stored_patches) {
  Blob blob;
  BlobWriter writer(blob);
  writer.write(uint32_t{0b101});  // levels_mask
  writer.write(declared_patches);
  for (uint32_t k = 0; k < stored_patches; ++k) {
    sof::STFeaturePatch patch;
    patch.setConstant(static_cast<float>(k + 1));
    writer.write(patch);
  }
  return blob;
}

}  // namespace

TEST(BlobReader, EmptyBlobReadsNothing) {
  const Blob empty;
  const BlobReader reader(empty);
  uint32_t value = 0;
  EXPECT_FALSE(reader.read(value));
  EXPECT_EQ(reader.remaining(), 0u);
}

TEST(BlobReader, ReadsValuesAtUnalignedOffsets) {
  Blob blob;
  BlobWriter writer(blob);
  writer.write(uint8_t{7});
  writer.write(uint64_t{0x0123456789abcdef});
  writer.write(1.5);

  const BlobReader reader(blob);
  uint8_t head = 0;
  uint64_t value = 0;
  double real = 0;
  ASSERT_TRUE(reader.read(head));
  ASSERT_TRUE(reader.read(value));
  ASSERT_TRUE(reader.read(real));
  EXPECT_EQ(value, 0x0123456789abcdefu);
  EXPECT_EQ(real, 1.5);
}

// A count bigger than the bytes behind it would otherwise allocate gigabytes before the read fails.
TEST(BlobReader, RejectsCountsLargerThanTheBlob) {
  Blob blob;
  BlobWriter writer(blob);
  writer.write(kHugeCount);
  writer.write(uint64_t{1});

  std::vector<uint64_t> values;
  EXPECT_FALSE(BlobReader(blob).read_std(values));
  EXPECT_TRUE(values.empty());

  std::string text;
  EXPECT_FALSE(BlobReader(blob).read_str(text));
  EXPECT_TRUE(text.empty());
}

TEST(BlobReader, RoundTripsVectorsAndStrings) {
  const std::vector<uint64_t> values = {1, 2, 3};
  const std::string text = "cell";
  Blob blob;
  BlobWriter writer(blob);
  writer.write_std(values);
  writer.write_str(text);
  writer.write_std(std::vector<uint64_t>{});

  const BlobReader reader(blob);
  std::vector<uint64_t> read_values;
  std::string read_text;
  std::vector<uint64_t> read_empty = {9};
  ASSERT_TRUE(reader.read_std(read_values));
  ASSERT_TRUE(reader.read_str(read_text));
  ASSERT_TRUE(reader.read_std(read_empty));
  EXPECT_EQ(read_values, values);
  EXPECT_EQ(read_text, text);
  EXPECT_TRUE(read_empty.empty());
}

// Map descriptors are read inside a database transaction that ends when the read callback returns.
TEST(STDescriptorFromBlob, OwnsItsMemory) {
  const STDescriptorOps ops{20, 0};
  FeatureDescriptor fd;
  {
    const Blob blob = MakeDescriptorBlob(2, 2);
    fd = ops.CreateFromBlob(BlobReader(blob));
  }
  ASSERT_TRUE(fd);
  EXPECT_EQ(fd.const_memory_ptr, nullptr);
  ASSERT_NE(fd.pull_memory, nullptr);

  const auto* descriptor = static_cast<const STDescriptor*>(fd.get());
  EXPECT_EQ(descriptor->levels_mask, 0b101u);
  ASSERT_EQ(descriptor->image_patches_size, 2u);
  EXPECT_EQ(descriptor->image_patches[0](4, 4), 1.f);
  EXPECT_EQ(descriptor->image_patches[1](8, 8), 2.f);
}

TEST(STDescriptorFromBlob, RejectsPatchCountBeyondTheBlob) {
  const STDescriptorOps ops{20, 0};
  EXPECT_FALSE(ops.CreateFromBlob(BlobReader(MakeDescriptorBlob(3, 2))));
  EXPECT_FALSE(ops.CreateFromBlob(BlobReader(MakeDescriptorBlob(std::numeric_limits<uint32_t>::max(), 1))));
}

#ifdef USE_LMDB
// A truncated hypothesis used to spin through its declared count, filling the map from failed reads.
TEST(PoseGraphHypothesisFromDatabase, RejectsTruncatedRecord) {
  const std::filesystem::path folder =
      std::filesystem::temp_directory_path() / ("cuvslam_map_blob_test_" + std::to_string(getpid()));
  LmdbSlamDatabase database;
  ASSERT_TRUE(database.Open(folder.c_str(), LmdbSlamDatabase::OpenMode::READ_WRITE_HARD_RESET));

  PoseGraphHypothesis saved;
  saved.SetKeyframePose(1, Isometry3T::Identity());
  saved.SetKeyframePose(2, Isometry3T::Identity());
  ASSERT_TRUE(saved.PutToDatabase(&database));

  Blob record;
  ASSERT_TRUE(database.GetSingleton(SlamDatabaseSingleton::PoseGraphHypothesis, [&](const BlobReader& reader) {
    record.resize(reader.remaining());
    return reader.read(record.data(), record.size());
  }));

  PoseGraphHypothesis loaded;
  ASSERT_TRUE(loaded.GetFromDatabase(&database));
  EXPECT_NE(loaded.GetKeyframePose(2), nullptr);

  // claim far more vertices than the record holds, and cut the last one short
  std::string format;
  ASSERT_TRUE(BlobReader(record).read_str(format));
  const size_t count_offset = sizeof(uint64_t) + format.size();
  const uint64_t huge = kHugeCount;
  std::memcpy(record.data() + count_offset, &huge, sizeof(huge));
  record.resize(record.size() - 8);
  ASSERT_TRUE(database.SetSingleton(SlamDatabaseSingleton::PoseGraphHypothesis, record.size(), [&](BlobWriter& writer) {
    writer.write(record.data(), record.size());
    return true;
  }));

  PoseGraphHypothesis corrupt;
  EXPECT_FALSE(corrupt.GetFromDatabase(&database));
  EXPECT_EQ(corrupt.GetKeyframePose(1), nullptr);

  database.Close();
  std::filesystem::remove_all(folder);
}
#endif  // USE_LMDB

}  // namespace test::slam
