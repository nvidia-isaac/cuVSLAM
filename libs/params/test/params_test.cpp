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

#include "params/registry.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "common/include_json.h"

// Stand-in settings structs. They exercise the mechanism itself -- every supported field type,
// every failure mode -- without tying these tests to any real struct's field list.
namespace {

enum class Mode { Fast, Slow };

struct Widget {
  int32_t count = 7;
  int64_t big = 9;
  float scale = 0.5f;
  bool enabled = false;
  std::optional<bool> forced;
  std::vector<int32_t> ids;
  Mode mode = Mode::Fast;
};

struct Gadget {
  int32_t count = 100;
};

struct Bounded {
  int32_t count = 5;
  float ratio = 0.5f;
};

}  // namespace

CUVSLAM_PARAM_ENUM_BEGIN(Mode)
CUVSLAM_PARAM_ENUM_VALUE("fast", Fast)
CUVSLAM_PARAM_ENUM_VALUE("slow", Slow)
CUVSLAM_PARAM_ENUM_END()

CUVSLAM_PARAMS_BEGIN(Widget)
CUVSLAM_PARAM(count, "how many")
CUVSLAM_PARAM(big, "a wide one")
CUVSLAM_PARAM(scale, "a ratio")
CUVSLAM_PARAM(enabled, "on or off")
CUVSLAM_PARAM(forced, "tri-state")
CUVSLAM_PARAM(ids, "a list")
CUVSLAM_PARAM(mode, "which mode")
CUVSLAM_PARAMS_END()

CUVSLAM_PARAMS_BEGIN(Gadget)
CUVSLAM_PARAM(count, "how many")
CUVSLAM_PARAMS_END()

CUVSLAM_PARAMS_BEGIN(Bounded)
CUVSLAM_PARAM_BOUNDED(count, "how many", NonNegative())
CUVSLAM_PARAM_BOUNDED(ratio, "a fraction", InRange(0.0, 1.0))
CUVSLAM_PARAMS_END()

namespace cuvslam::params {
namespace {

class ParamsTest : public ::testing::Test {
protected:
  void SetUp() override { registry.Add("widget", widget); }

  Widget widget;
  Registry registry;
};

TEST_F(ParamsTest, WritesThroughToTheRegisteredInstance) {
  registry.Set("widget.count", "42", Source::Api);
  EXPECT_EQ(widget.count, 42);

  registry.Set("widget.big", "-9000000000", Source::Api);
  EXPECT_EQ(widget.big, -9000000000LL);

  registry.Set("widget.enabled", "true", Source::Api);
  EXPECT_TRUE(widget.enabled);
}

TEST_F(ParamsTest, ReadsBackEverySupportedType) {
  registry.Set("widget.scale", "0.25", Source::Api);
  registry.Set("widget.ids", "3,1,4", Source::Api);
  registry.Set("widget.mode", "slow", Source::Api);

  EXPECT_EQ(registry.Get("widget.scale"), "0.25");
  EXPECT_EQ(registry.Get("widget.ids"), "3,1,4");
  EXPECT_EQ(registry.Get("widget.mode"), "slow");
  EXPECT_EQ(widget.ids, std::vector<int32_t>({3, 1, 4}));
  EXPECT_EQ(widget.mode, Mode::Slow);
}

TEST_F(ParamsTest, FloatRoundTripsExactly) {
  registry.Set("widget.scale", "0.1", Source::Api);
  const std::string dumped = registry.Get("widget.scale");
  const float first = widget.scale;
  widget.scale = 0.f;
  registry.Set("widget.scale", dumped, Source::Api);
  EXPECT_EQ(widget.scale, first);
}

TEST_F(ParamsTest, OptionalBoolCarriesThreeStates) {
  EXPECT_EQ(registry.Get("widget.forced"), "null");

  registry.Set("widget.forced", "true", Source::Api);
  ASSERT_TRUE(widget.forced.has_value());
  EXPECT_TRUE(*widget.forced);

  registry.Set("widget.forced", "null", Source::Api);
  EXPECT_FALSE(widget.forced.has_value());
}

TEST_F(ParamsTest, BracketedListIsAcceptedSoDumpedValuesPasteBack) {
  registry.Set("widget.ids", "[5, 6]", Source::Api);
  EXPECT_EQ(widget.ids, std::vector<int32_t>({5, 6}));
}

TEST_F(ParamsTest, UnknownKeyThrows) {
  EXPECT_THROW(registry.Set("widget.nope", "1", Source::Api), std::invalid_argument);
  EXPECT_THROW(registry.Set("nope.count", "1", Source::Api), std::invalid_argument);
}

TEST_F(ParamsTest, UnparseableValueThrowsAndNamesTheKey) {
  try {
    registry.Set("widget.count", "not-a-number", Source::Api);
    FAIL() << "expected a parse failure";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("widget.count"), std::string::npos);
  }
  EXPECT_EQ(widget.count, 7) << "a rejected value must not be partially applied";
}

TEST_F(ParamsTest, UnknownEnumValueListsTheValidOnes) {
  try {
    registry.Set("widget.mode", "sideways", Source::Api);
    FAIL() << "expected a parse failure";
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("fast"), std::string::npos);
    EXPECT_NE(message.find("slow"), std::string::npos);
  }
}

TEST_F(ParamsTest, UnambiguousSuffixResolves) {
  EXPECT_EQ(registry.Resolve("count"), "widget.count");
  registry.Set("scale", "0.75", Source::CommandLine);
  EXPECT_EQ(widget.scale, 0.75f);
}

TEST_F(ParamsTest, AmbiguousSuffixThrowsAndListsCandidates) {
  Gadget gadget;
  registry.Add("gadget", gadget);
  try {
    registry.Resolve("count");
    FAIL() << "expected an ambiguity error";
  } catch (const std::invalid_argument& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("widget.count"), std::string::npos);
    EXPECT_NE(message.find("gadget.count"), std::string::npos);
  }
  // A fully qualified key stays usable while the abbreviation is ambiguous.
  registry.Set("gadget.count", "3", Source::Api);
  EXPECT_EQ(gadget.count, 3);
  EXPECT_EQ(widget.count, 7);
}

TEST_F(ParamsTest, ListReportsDefaultsSeparatelyFromCurrentValues) {
  registry.Set("widget.count", "42", Source::CommandLine);

  const std::vector<ParamInfo> infos = registry.List();
  ASSERT_EQ(infos.size(), Fields<Widget>::kList.size());

  const auto count =
      std::find_if(infos.begin(), infos.end(), [](const ParamInfo& i) { return i.key == "widget.count"; });
  ASSERT_NE(count, infos.end());
  EXPECT_EQ(count->value, "42");
  EXPECT_EQ(count->default_value, "7");
  EXPECT_EQ(count->type, "int32");
  EXPECT_EQ(count->doc, "how many");
  EXPECT_EQ(count->source, Source::CommandLine);
  EXPECT_FALSE(count->is_default());

  const auto big = std::find_if(infos.begin(), infos.end(), [](const ParamInfo& i) { return i.key == "widget.big"; });
  ASSERT_NE(big, infos.end());
  EXPECT_EQ(big->source, Source::Default);
  EXPECT_TRUE(big->is_default());
}

TEST_F(ParamsTest, EnumTypeNameAdvertisesItsValues) {
  const std::vector<ParamInfo> infos = registry.List();
  const auto mode = std::find_if(infos.begin(), infos.end(), [](const ParamInfo& i) { return i.key == "widget.mode"; });
  ASSERT_NE(mode, infos.end());
  EXPECT_EQ(mode->type, "enum{fast|slow}");
}

TEST_F(ParamsTest, LastSourceWins) {
  registry.Set("widget.count", "1", Source::File);
  registry.Set("widget.count", "2", Source::CommandLine);

  const std::vector<ParamInfo> infos = registry.List();
  const auto count =
      std::find_if(infos.begin(), infos.end(), [](const ParamInfo& i) { return i.key == "widget.count"; });
  ASSERT_NE(count, infos.end());
  EXPECT_EQ(count->source, Source::CommandLine);
  EXPECT_EQ(count->value, "2");
}

TEST_F(ParamsTest, JsonDumpRecordsValueAndProvenance) {
  registry.Set("widget.count", "42", Source::File);

  Json::Value root;
  JsonUtils::readJsonFromString(registry.ToJson(), root);

  ASSERT_TRUE(root.isMember("widget.count"));
  EXPECT_EQ(root["widget.count"]["value"].asString(), "42");
  EXPECT_EQ(root["widget.count"]["source"].asString(), "file");
  EXPECT_EQ(root["widget.count"]["default"].asString(), "7");

  ASSERT_TRUE(root.isMember("widget.big"));
  EXPECT_EQ(root["widget.big"]["source"].asString(), "default");
  EXPECT_FALSE(root["widget.big"].isMember("default")) << "defaults are only spelled out when overridden";
}

class BoundsTest : public ::testing::Test {
protected:
  void SetUp() override { registry.Add("bounded", bounded); }

  Bounded bounded;
  Registry registry;
};

TEST_F(BoundsTest, AcceptsValuesInsideTheBounds) {
  registry.Set("bounded.count", "0", Source::Api);
  EXPECT_EQ(bounded.count, 0);

  registry.Set("bounded.ratio", "1.0", Source::Api);
  EXPECT_EQ(bounded.ratio, 1.0f);
}

TEST_F(BoundsTest, RejectsNegativeCountAndLeavesTheFieldUntouched) {
  try {
    registry.Set("bounded.count", "-1", Source::Api);
    FAIL() << "expected a bounds failure";
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("bounded.count"), std::string::npos);
    EXPECT_NE(message.find("non-negative"), std::string::npos);
  }
  EXPECT_EQ(bounded.count, 5) << "a rejected value must be rolled back";
}

TEST_F(BoundsTest, RejectsValueAboveTheRangeAndReportsTheRange) {
  try {
    registry.Set("bounded.ratio", "1.5", Source::Api);
    FAIL() << "expected a bounds failure";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("[0, 1]"), std::string::npos);
  }
  EXPECT_EQ(bounded.ratio, 0.5f);
}

TEST_F(BoundsTest, RejectionDoesNotRecordASource) {
  EXPECT_THROW(registry.Set("bounded.count", "-1", Source::File), std::runtime_error);

  const std::vector<ParamInfo> infos = registry.List();
  const auto count =
      std::find_if(infos.begin(), infos.end(), [](const ParamInfo& i) { return i.key == "bounded.count"; });
  ASSERT_NE(count, infos.end());
  EXPECT_EQ(count->source, Source::Default);
  EXPECT_TRUE(count->is_default());
}

class ParamsFileTest : public ParamsTest {
protected:
  void TearDown() override {
    if (!path.empty()) {
      std::remove(path.c_str());
    }
  }

  void WriteFile(const std::string& contents, const std::string& name) {
    path = ::testing::TempDir() + "/" + name;
    std::ofstream file(path);
    file << contents;
  }

  std::string path;
};

TEST_F(ParamsFileTest, LoadsKeyValueLinesAndIgnoresCommentsAndBlanks) {
  WriteFile(
      "# a comment\n"
      "\n"
      "widget.count: 11\n"
      "widget.scale = 1.5   # trailing comment\n"
      "  widget.mode:slow\n",
      "params_basic.txt");

  EXPECT_EQ(registry.SetFromFile(path, Source::File), 3u);
  EXPECT_EQ(widget.count, 11);
  EXPECT_EQ(widget.scale, 1.5f);
  EXPECT_EQ(widget.mode, Mode::Slow);
}

TEST_F(ParamsFileTest, MalformedLineThrowsWithFileAndLineNumber) {
  WriteFile("widget.count: 1\nthis line has no separator\n", "params_malformed.txt");

  try {
    registry.SetFromFile(path, Source::File);
    FAIL() << "expected a parse failure";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(":2:"), std::string::npos);
  }
}

TEST_F(ParamsFileTest, UnknownKeyInFileThrowsWithLineNumber) {
  WriteFile("widget.count: 1\nwidget.typo: 2\n", "params_unknown.txt");

  try {
    registry.SetFromFile(path, Source::File);
    FAIL() << "expected an unknown-key failure";
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find(":2:"), std::string::npos);
    EXPECT_NE(message.find("widget.typo"), std::string::npos);
  }
}

TEST_F(ParamsFileTest, MissingFileThrows) {
  EXPECT_THROW(registry.SetFromFile("/nonexistent/params.txt", Source::File), std::runtime_error);
}

TEST_F(ParamsFileTest, DumpedJsonValuesReloadThroughAFile) {
  registry.Set("widget.count", "23", Source::Api);
  registry.Set("widget.scale", "0.3", Source::Api);
  registry.Set("widget.ids", "1,2,3", Source::Api);
  registry.Set("widget.forced", "false", Source::Api);
  registry.Set("widget.mode", "slow", Source::Api);
  const Widget expected = widget;

  Json::Value root;
  JsonUtils::readJsonFromString(registry.ToJson(), root);
  std::string reloaded;
  for (const std::string& key : root.getMemberNames()) {
    reloaded += key + ": " + root[key]["value"].asString() + "\n";
  }

  Widget fresh;
  Registry other;
  other.Add("widget", fresh);
  WriteFile(reloaded, "params_roundtrip.txt");
  other.SetFromFile(path, Source::File);

  EXPECT_EQ(fresh.count, expected.count);
  EXPECT_EQ(fresh.scale, expected.scale);
  EXPECT_EQ(fresh.ids, expected.ids);
  EXPECT_EQ(fresh.forced, expected.forced);
  EXPECT_EQ(fresh.mode, expected.mode);
}

}  // namespace
}  // namespace cuvslam::params
