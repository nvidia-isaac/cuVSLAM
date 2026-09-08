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

#include "utils/param_file.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "common/include_gtest.h"

namespace cuvslam::utils {
namespace {

class ParamFileTest : public ::testing::Test {
protected:
  void TearDown() override {
    if (!path_.empty()) {
      std::remove(path_.c_str());
    }
  }

  /// Writes @p contents to a temporary file and returns its path.
  const std::string& Write(const std::string& contents, const std::string& name) {
    path_ = ::testing::TempDir() + "/" + name;
    std::ofstream file(path_);
    file << contents;
    return path_;
  }

  std::string path_;
};

TEST_F(ParamFileTest, ReadsBothSeparatorsAndTrimsAround) {
  const std::vector<ParamEntry> entries =
      ReadParamFile(Write("sof.num_desired_tracks: 300\n"
                          "  sof.tracker  =  klt  \n",
                          "separators.txt"));

  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].key, "sof.num_desired_tracks");
  EXPECT_EQ(entries[0].value, "300");
  EXPECT_EQ(entries[1].key, "sof.tracker");
  EXPECT_EQ(entries[1].value, "klt");
}

TEST_F(ParamFileTest, SkipsCommentsAndBlankLines) {
  const std::vector<ParamEntry> entries =
      ReadParamFile(Write("# a heading\n"
                          "\n"
                          "sof.num_desired_tracks: 300   # trailing comment\n"
                          "   \n"
                          "   # indented comment\n",
                          "comments.txt"));

  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].value, "300") << "a trailing comment must not become part of the value";
}

TEST_F(ParamFileTest, ReportsTheLineAnEntryCameFrom) {
  // Callers point their error messages at the offending line, so the numbering has to count the
  // lines it skips.
  const std::vector<ParamEntry> entries =
      ReadParamFile(Write("# comment\n"
                          "\n"
                          "first: 1\n"
                          "second: 2\n",
                          "lines.txt"));

  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].line, 3u);
  EXPECT_EQ(entries[1].line, 4u);
}

TEST_F(ParamFileTest, KeepsAValueContainingSeparators) {
  // Only the first separator splits, so a path or a list survives intact.
  const std::vector<ParamEntry> entries =
      ReadParamFile(Write("slam.map_cache_path: /tmp/a=b:c\n", "value_with_separators.txt"));

  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].key, "slam.map_cache_path");
  EXPECT_EQ(entries[0].value, "/tmp/a=b:c");
}

TEST_F(ParamFileTest, EmptyFileYieldsNothing) {
  EXPECT_TRUE(ReadParamFile(Write("", "empty.txt")).empty());
  EXPECT_TRUE(ReadParamFile(Write("# just a comment\n", "only_comment.txt")).empty());
}

TEST_F(ParamFileTest, MalformedLineThrowsWithFileAndLineNumber) {
  const std::string& path = Write("good: 1\nno separator here\n", "malformed.txt");
  try {
    ReadParamFile(path);
    FAIL() << "expected a parse failure";
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find(":2:"), std::string::npos) << message;
    EXPECT_NE(message.find("no separator here"), std::string::npos) << message;
  }
}

TEST_F(ParamFileTest, MissingFileThrowsNamingIt) {
  try {
    ReadParamFile("/nonexistent/params.txt");
    FAIL() << "expected a failure";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("/nonexistent/params.txt"), std::string::npos);
  }
}

}  // namespace
}  // namespace cuvslam::utils
