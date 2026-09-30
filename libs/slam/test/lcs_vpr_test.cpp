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

#include "slam/slam/loop_closure_solver/lcs_vpr.h"

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "camera/camera.h"
#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/vector_2t.h"
#include "slam/slam/slam.h"

namespace test::lcs_vpr {
using namespace cuvslam;
using namespace cuvslam::slam;

namespace {

Isometry3T PoseAtX(float x) {
  Isometry3T pose = Isometry3T::Identity();
  pose.translation().x() = x;
  return pose;
}

vpr::VprPlace MakeCandidate(KeyFrameId node_id, float x) {
  vpr::VprPlace place;
  place.found = true;
  place.node_id = node_id;
  place.pose = PoseAtX(x);
  return place;
}

/// Stands in for the geometric verification. `verify_` maps the guess of an attempt to the pose it finds for the
/// current frame, or to nothing when the attempt fails. Every attempt is recorded, and reports one landmark, one
/// discarded landmark and one keyframe in sight, numbered after the attempt, so a test can tell whose reports got out.
class FakeVerifier : public ILoopClosureSolver {
public:
  using Verify = std::function<std::optional<Isometry3T>(const Isometry3T& guess)>;

  struct Attempt {
    Isometry3T guess;
    KeyFrameId pose_graph_head;
    size_t landmarks_on_entry;
  };

  explicit FakeVerifier(Verify verify) : verify_(std::move(verify)) {}

  bool Solve(const LoopClosureTask& task, const LSIGrid&, const IFeatureDescriptorOps*, Isometry3T& pose,
             Matrix6T& pose_covariance, std::vector<LandmarkInSolver>* landmarks,
             DiscardLandmarkCB* discard_landmark_cb, KeyframeInSightCB* keyframe_in_sight_cb) const override {
    const auto attempt = static_cast<LandmarkId>(attempts.size());
    attempts.push_back({task.guess_world_from_rig, task.pose_graph_head, landmarks ? landmarks->size() : 0});
    if (landmarks) {
      landmarks->push_back(LandmarkInSolver{100 + attempt, Vector2T(0.f, 0.f)});
    }
    if (discard_landmark_cb) {
      (*discard_landmark_cb)(200 + attempt, LP_TRACKING_FAILED);
    }
    if (keyframe_in_sight_cb) {
      (*keyframe_in_sight_cb)(300 + attempt);
    }
    const std::optional<Isometry3T> found = verify_(task.guess_world_from_rig);
    if (!found) {
      return false;
    }
    pose = *found;
    pose_covariance = Matrix6T::Identity();
    return true;
  }

  mutable std::vector<Attempt> attempts;

private:
  Verify verify_;
};

/// Verifies only a guess at `x`, and finds the current frame half a meter further along.
FakeVerifier::Verify VerifyOnlyAt(float x) {
  return [x](const Isometry3T& guess) -> std::optional<Isometry3T> {
    if (guess.translation().x() != x) {
      return std::nullopt;
    }
    return PoseAtX(x + 0.5f);
  };
}

/// A rig and an empty LocalizerAndMapper, just enough to hand LoopClosureSolverVpr::Solve() a real LSIGrid and
/// IFeatureDescriptorOps. A fake verifier never reads them.
class LcsVprTest : public ::testing::Test {
protected:
  void SetUp() override {
    camera_ = camera::CreateCameraModel(Vector2T(640, 480), Vector2T(320, 320), Vector2T(320, 240),
                                        Distortion::Model::Pinhole, nullptr, 0);
    rig_.intrinsics[0] = camera_.get();
    rig_.num_cameras = 1;
    rig_.camera_from_rig[0].setIdentity();

    mapper_ = std::make_unique<LocalizerAndMapper>(rig_, FeatureDescriptorType::kNone, true);
    mapper_->SetActiveCameras({0});
  }

  /// A solver whose verifier is `verify`; `verifier_` observes it.
  std::unique_ptr<LoopClosureSolverVpr> MakeSolver(FakeVerifier::Verify verify) {
    auto verifier = std::make_unique<FakeVerifier>(std::move(verify));
    verifier_ = verifier.get();
    return std::make_unique<LoopClosureSolverVpr>(std::move(verifier));
  }

  /// A task whose current pose estimate is at `guess_x`, away from every candidate unless a test puts one there.
  LoopClosureTask MakeTask(std::vector<vpr::VprPlace> candidates, float guess_x = 4.f) const {
    LoopClosureTask task;
    task.pose_graph_hypothesis = mapper_->GetMap().pose_graph_hypothesis_.MakeCopy();
    task.guess_world_from_rig = PoseAtX(guess_x);
    task.pose_graph_head = 42;
    task.vpr_candidates = std::move(candidates);
    return task;
  }

  bool Solve(const ILoopClosureSolver& solver, const LoopClosureTask& task) {
    ILoopClosureSolver::DiscardLandmarkCB discard_cb = [this](LandmarkId id, LandmarkProbe) {
      discarded_.push_back(id);
    };
    ILoopClosureSolver::KeyframeInSightCB in_sight_cb = [this](KeyFrameId id) { in_sight_.push_back(id); };
    const LSIGrid& lsi = *mapper_->GetMap().landmarks_spatial_index_;
    return solver.Solve(task, lsi, mapper_->GetMap().feature_descriptor_ops_.get(), pose_, pose_covariance_,
                        &landmarks_, &discard_cb, &in_sight_cb);
  }

  std::unique_ptr<camera::ICameraModel> camera_;
  camera::Rig rig_;
  std::unique_ptr<LocalizerAndMapper> mapper_;
  FakeVerifier* verifier_ = nullptr;

  Isometry3T pose_ = Isometry3T::Identity();
  Matrix6T pose_covariance_ = Matrix6T::Zero();
  std::vector<LandmarkInSolver> landmarks_;
  std::vector<LandmarkId> discarded_;
  std::vector<KeyFrameId> in_sight_;
};

}  // namespace

TEST_F(LcsVprTest, WantsVprCandidates) {
  EXPECT_TRUE(MakeSolver(VerifyOnlyAt(0.f))->WantsVprCandidates());

  const std::unique_ptr<ILoopClosureSolver> solver(
      CreateLoopClosureSolver(LoopClosureSolverType::kVpr, RansacType::kPnP, /*randomized=*/false, rig_));
  ASSERT_NE(solver, nullptr);
  EXPECT_TRUE(solver->WantsVprCandidates());
}

TEST_F(LcsVprTest, WithoutCandidatesNothingIsVerified) {
  const auto solver = MakeSolver(VerifyOnlyAt(4.f));

  // Not even the current pose estimate, which this verifier would accept: VPR is the only source of candidates.
  EXPECT_FALSE(Solve(*solver, MakeTask({})));
  EXPECT_TRUE(verifier_->attempts.empty());
}

TEST_F(LcsVprTest, VerifiesEachCandidateBestFirstAtItsOwnPose) {
  const auto solver = MakeSolver(VerifyOnlyAt(1000.f));  // no candidate is there, so every one of them gets a try

  EXPECT_FALSE(Solve(*solver, MakeTask({MakeCandidate(3, 1.f), MakeCandidate(7, 5.f), MakeCandidate(9, 9.f)})));

  ASSERT_EQ(verifier_->attempts.size(), 3u);
  const float expected_x[] = {1.f, 5.f, 9.f};
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(verifier_->attempts[i].guess.translation().x(), expected_x[i]) << "attempt " << i;
    // Only the guess moves; the rest of the task is the caller's.
    EXPECT_EQ(verifier_->attempts[i].pose_graph_head, 42u) << "attempt " << i;
  }
}

TEST_F(LcsVprTest, StopsAtTheFirstCandidateThatVerifies) {
  const auto solver = MakeSolver(VerifyOnlyAt(5.f));

  ASSERT_TRUE(Solve(*solver, MakeTask({MakeCandidate(3, 1.f), MakeCandidate(7, 5.f), MakeCandidate(9, 9.f)})));

  // The two candidates, then the estimate's own verification (which fails here); never the third candidate.
  ASSERT_EQ(verifier_->attempts.size(), 3u) << "the candidate after the one that verified must not be tried";
  EXPECT_EQ(verifier_->attempts[2].guess.translation().x(), 4.f);
  EXPECT_EQ(pose_.translation().x(), 5.5f) << "the result is the verifier's pose for the current frame";
  EXPECT_EQ(pose_covariance_, Matrix6T::Identity());
}

TEST_F(LcsVprTest, OnlyTheAttemptThatVerifiesReportsWhatItSaw) {
  const auto solver = MakeSolver(VerifyOnlyAt(5.f));

  ASSERT_TRUE(Solve(*solver, MakeTask({MakeCandidate(3, 1.f), MakeCandidate(9, 9.f), MakeCandidate(7, 5.f)})));

  // Every attempt starts from an empty landmark list, whatever the one before it left behind. The fourth is the
  // estimate's own verification, which fails here.
  ASSERT_EQ(verifier_->attempts.size(), 4u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(verifier_->attempts[i].landmarks_on_entry, 0u) << "attempt " << i;
  }
  // The third attempt (numbered 2) verified; the two rejected ones looked at places the camera is not at.
  ASSERT_EQ(landmarks_.size(), 1u);
  EXPECT_EQ(landmarks_.front().id, 102u);
  EXPECT_EQ(discarded_, std::vector<LandmarkId>{202});
  EXPECT_EQ(in_sight_, std::vector<KeyFrameId>{302});
}

TEST_F(LcsVprTest, WhenNoCandidateVerifiesNothingIsReported) {
  const auto solver = MakeSolver(VerifyOnlyAt(1000.f));
  // Left over from an earlier attempt: the caller reuses the vector, so Solve() has to clear it.
  landmarks_ = {LandmarkInSolver{1, Vector2T(0.f, 0.f)}};

  EXPECT_FALSE(Solve(*solver, MakeTask({MakeCandidate(3, 1.f), MakeCandidate(7, 5.f)})));

  EXPECT_EQ(verifier_->attempts.size(), 2u);
  EXPECT_TRUE(landmarks_.empty());
  EXPECT_TRUE(discarded_.empty());
  EXPECT_TRUE(in_sight_.empty());
}

TEST_F(LcsVprTest, ALookAlikePlaceTooFarFromThePoseEstimateIsRejected) {
  // Both candidates verify, each finding the current frame half a meter past itself. The better looking one is a
  // hundred meters from the pose estimate, which no drift explains; the second one is where the robot is.
  const auto solver =
      MakeSolver([](const Isometry3T& guess) { return std::optional(PoseAtX(guess.translation().x() + 0.5f)); });

  ASSERT_TRUE(Solve(*solver, MakeTask({MakeCandidate(3, 104.f), MakeCandidate(7, 4.f)}, /*guess_x=*/4.f)));

  // The far candidate fails the correction limit before the estimate is ever verified; the near one agrees with it.
  EXPECT_EQ(verifier_->attempts.size(), 3u);
  EXPECT_EQ(pose_.translation().x(), 4.5f);
  // Nothing the rejected attempt saw gets out, however well it verified.
  ASSERT_EQ(landmarks_.size(), 1u);
  EXPECT_EQ(landmarks_.front().id, 101u);
  EXPECT_EQ(discarded_, std::vector<LandmarkId>{201});
  EXPECT_EQ(in_sight_, std::vector<KeyFrameId>{301});
}

TEST_F(LcsVprTest, TheCorrectionLimitCoversTranslationAndRotation) {
  const float limit_m = LoopClosureSolverVpr::kMaxCorrectionM;
  const float limit_rad = LoopClosureSolverVpr::kMaxCorrectionDeg * 3.14159265f / 180.f;
  const auto moved_by = [](float x, float yaw_rad) {
    return [x, yaw_rad](const Isometry3T&) {
      Isometry3T pose = PoseAtX(x);
      pose.linear() = Eigen::AngleAxis<float>(yaw_rad, Vector3T::UnitY()).toRotationMatrix();
      return std::optional(pose);
    };
  };
  const LoopClosureTask task = MakeTask({MakeCandidate(3, 50.f)}, /*guess_x=*/0.f);

  EXPECT_TRUE(Solve(*MakeSolver(moved_by(limit_m * 0.99f, 0.f)), task));
  EXPECT_FALSE(Solve(*MakeSolver(moved_by(limit_m * 1.01f, 0.f)), task));
  EXPECT_TRUE(Solve(*MakeSolver(moved_by(0.f, limit_rad * 0.99f)), task));
  EXPECT_FALSE(Solve(*MakeSolver(moved_by(0.f, limit_rad * 1.01f)), task));
}

TEST_F(LcsVprTest, ACandidateTheEstimatesOwnVerificationDisagreesWithIsRejected) {
  // Started from a candidate, this verifier settles on the candidate's own node; started from the estimate at x=0 it
  // finds the camera 5 cm from there. A candidate 8 m away passes the correction limit, but not this check.
  const auto solver = MakeSolver([](const Isometry3T& guess) {
    const float x = guess.translation().x();
    return std::optional(PoseAtX(x == 0.f ? 0.05f : x + 0.5f));
  });

  ASSERT_TRUE(Solve(*solver, MakeTask({MakeCandidate(3, 8.f), MakeCandidate(7, 0.3f)}, /*guess_x=*/0.f)));

  EXPECT_EQ(pose_.translation().x(), 0.8f) << "the second candidate agrees with the estimate within the limit";
  // The estimate is verified once, right after the first candidate that verifies, and reports nothing itself.
  ASSERT_EQ(verifier_->attempts.size(), 3u);
  EXPECT_EQ(verifier_->attempts[1].guess.translation().x(), 0.f);
  ASSERT_EQ(landmarks_.size(), 1u);
  EXPECT_EQ(landmarks_.front().id, 102u);
  EXPECT_EQ(discarded_, std::vector<LandmarkId>{202});
  EXPECT_EQ(in_sight_, std::vector<KeyFrameId>{302});
}

TEST_F(LcsVprTest, WhenTheEstimateItselfDoesNotVerifyTheCandidateStands) {
  // The loop this mode is for: drift has carried the estimate too far for a search around it to find the place.
  const auto solver = MakeSolver(VerifyOnlyAt(8.f));

  ASSERT_TRUE(Solve(*solver, MakeTask({MakeCandidate(3, 8.f)}, /*guess_x=*/0.f)));

  EXPECT_EQ(pose_.translation().x(), 8.5f);
  ASSERT_EQ(verifier_->attempts.size(), 2u);
  EXPECT_EQ(verifier_->attempts[1].guess.translation().x(), 0.f);
}

TEST_F(LcsVprTest, TheDisagreementLimitCoversTranslationAndRotation) {
  const float limit_m = LoopClosureSolverVpr::kMaxDisagreementM;
  const float limit_rad = LoopClosureSolverVpr::kMaxDisagreementDeg * 3.14159265f / 180.f;
  // Verification finds x=0 from the estimate at x=0, and `from_candidate` from the candidate.
  const auto verifier = [](const Isometry3T& from_candidate) {
    return [from_candidate](const Isometry3T& guess) {
      return std::optional(guess.translation().x() == 0.f ? PoseAtX(0.f) : from_candidate);
    };
  };
  const auto yawed = [](float yaw_rad) {
    Isometry3T pose = PoseAtX(0.f);
    pose.linear() = Eigen::AngleAxis<float>(yaw_rad, Vector3T::UnitY()).toRotationMatrix();
    return pose;
  };
  const LoopClosureTask task = MakeTask({MakeCandidate(3, 5.f)}, /*guess_x=*/0.f);

  EXPECT_TRUE(Solve(*MakeSolver(verifier(PoseAtX(limit_m * 0.99f))), task));
  EXPECT_FALSE(Solve(*MakeSolver(verifier(PoseAtX(limit_m * 1.01f))), task));
  EXPECT_TRUE(Solve(*MakeSolver(verifier(yawed(limit_rad * 0.99f))), task));
  EXPECT_FALSE(Solve(*MakeSolver(verifier(yawed(limit_rad * 1.01f))), task));
}

TEST_F(LcsVprTest, TheDefaultVerifierRejectsCandidatesOnAMapWithoutLandmarks) {
  // The real TwoStepsEasy verification: with no landmark anywhere, no candidate can verify.
  const std::unique_ptr<ILoopClosureSolver> solver(
      CreateLoopClosureSolverVpr(rig_, RansacType::kPnP, /*randomized=*/false));
  landmarks_ = {LandmarkInSolver{1, Vector2T(0.f, 0.f)}};

  EXPECT_FALSE(Solve(*solver, MakeTask({MakeCandidate(3, 1.f), MakeCandidate(7, 5.f)})));
  EXPECT_TRUE(landmarks_.empty());
}

}  // namespace test::lcs_vpr
