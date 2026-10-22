# Visual place recognition

This module answers one question: **which pose graph node saw this frame?** The answer is a node, never a metric
pose of its own; `Slam::LocalizeInMap` is what turns that hint into a verified pose. What the feature is for is in the
[root README](../../../README.md#visual-place-recognition), and how a caller drives it in
[doc/place_recognition.md](../../../doc/place_recognition.md).

## Shape of the module

```text
vpr_types.h     VprImage, VprOptions, VprMatch — no dependencies beyond slam_common.h
ivpr.h/.cpp     the IVpr backend interface and the factory that picks a backend from VprOptions
vpr_image.*     grayscale conversion and bilinear resize
vpr_anyloc.*    the AnyLoc backend, built when USE_TENSORRT is ON (needs the engine models/anyloc builds)
anyloc_network.* the TensorRT engine AnyLoc runs, on the GPU
vpr_map.*       VprMap: owns a backend and the node bookkeeping; the only type the rest of SLAM touches
vpr_sof_image.* bridge from the tracker's sof::ImageContext to a VprImage
```

## Design decisions

**A match resolves to a node, not to a pose.** `IVpr` stores nothing but descriptors keyed by `KeyFrameId`, and
`LocalizerAndMapper::RecognizePlace` resolves the node through the pose graph at query time. A place recognized
after a loop closure therefore reports the *corrected* position, not the drifted one that was current when the
frame was mapped. `VprMap` stores no poses at all.

**Exactly one frame per node**, the keyframe's own. This bounds the map to the size of the pose graph, which
matters because AnyLoc searches exhaustively.

**The place recognition map is part of the SLAM map.** It is a singleton of the SLAM database next to the spatial
index, `SlamDatabaseSingleton::VprMap`, written by `SaveMap`, by `FlushMap` and when `Slam` is destroyed, and read
back with the pose graph whose node ids it uses. Without a database it lives in memory, like the rest of the map.

**The default score threshold, 0.68, favors precision.** On KITTI-07 (ViT-S/14 block 9 value facet at 322x322, a
64 node map of frames 0..299, 68 queries scored against ground truth) every query within 2 m of the node it matched
scored 0.703 or more, and no query more than 90 m from everything in the map passed 0.655; 0.68 sits in that gap.
Re-measured on the TensorRT engines: 0.784 and 0.643 with FP16, 0.759 to 0.776 and 0.658 to 0.662 over two FP32
runs. FP16 moves a query's score by a median 0.009, about what rebuilding the map does, so the threshold holds for
both. It costs recall on a genuine second pass, where a correct match scores 0.62 to 0.82 and only the better half
clears 0.68, but a relocalizing caller acts on a match, so a wrong one costs more than a missing one.

**An ambiguous place can be dropped instead of reported.** With `VprOptions::ratio_threshold` set (the public API
leaves it at 0, off), AnyLoc's winner has to beat the best match from a different part of the trajectory, more than
`kNeighbourGuard` nodes away in creation order.
Between two candidates that look equally much like the query, reporting either is a coin flip, and a wrong pose
costs a relocalizer more than a missing one.

**The frame is copied when the keyframe is queued.** `AsyncSlam::TrackResult` copies it into
`VOFrameData::vpr_image` on the caller's thread, and `LocalizerAndMapper::AddKeyframe` maps it. The worker may run
after the image pool recycled the frame's context, so the copy cannot wait for it.

Nothing retains an `sof::ImageContextPtr` for this. The tracker's image pool holds four contexts per camera and
hands out a slot only when its use count drops to one, so keeping a reference for even one extra frame can starve
`Odometry::Track`. Pixels are copied into a `VprImage` and the context is released immediately.

## Evaluating a backend

`cuvslam_vpr_reporter` scores a backend on two traversals of one route. `DATASETS.md` lists the public datasets
that can drive it and the ones that cannot, and why.

### Why AnyLoc is the only backend

Four backends were measured while this module was built: grayscale thumbnails compared by normalized cross
correlation, an in-tree ORB bag of words, DBoW2, and AnyLoc. Every one of them recognizes a sequence replayed against
a map of itself, at 96 to 100%. What decides is what happens across two *separate* recordings of one route, which is
the case a robot is actually in. Measured on CODa-00 against CODa-05, 2500 frames, success radius 10 m:

| Backend | Self-pair (KITTI-07) | Cross-session (CODa-00 → CODa-05) |
|---------|---------------------|------------------------------------|
| Thumbnails | 96% recognized | 49% recognized, 18% wrong place |
| In-tree ORB bag of words | 99% | 8% recognized, 2% wrong place |
| DBoW2 | 99% | 10% recognized, 1% wrong place |
| AnyLoc | 100% | 95% recognized, 1% wrong place |

For both bags of words the score distributions of correct and wrong matches **overlap almost completely** across
sessions — medians 0.279 against 0.267 for the in-tree one, 0.128 against 0.123 for DBoW2 — so no threshold
separates them and a calibration can only choose where to give up. AnyLoc's DINOv2 features are the only ones here
that survive the viewpoint, lighting and traffic differences between two drives, so it is the one backend this
module keeps. Its default threshold chooses precision: a robot told confidently that it is somewhere it is not is
worse off than one told nothing.

One lesson holds for any backend: a threshold calibrated on a self-pair does not transfer. The in-tree bag of
words' first default was set that way and produced 75% wrong answers on the cross-session pair.

## Adding a backend

Implement `IVpr` and construct it in `CreateVpr`. The tests of `VprMap` and of
the pose graph bookkeeping run a stand-in backend instead, through `VprMap(options, backend)` and
`LocalizerAndMapper::SetVprMap`, so they need neither TensorRT nor a GPU. A backend that needs a
vocabulary builds it in `Finalize()`, which `VprMap` calls before the first query and before serializing. AnyLoc
also fits it while mapping, as soon as the raw patches it holds would outgrow its training set, stalling the SLAM
thread for about 5 s on an i9-10940X; once it has a vocabulary, it pools each frame as it is mapped.

## Closing loops

`Slam::Config::loop_closure_mode = LoopClosureMode::VPR` makes place recognition the only source of loop closure
candidates. `LocalizerAndMapper::DetectLoopClosure` asks `RecognizeLoopClosurePlaces` for up to
`kLoopClosureVprCandidates` (5) places, and `LoopClosureSolverVpr` runs the LSI mode's own verification
(`LoopClosureSolverTwoStepsEasy`) re-centered on each candidate's pose in turn, keeping the first that passes. Each
of the decisions below was forced by a measurement on KITTI 00-10:

**The recent past is left out.** `AddKeyframe` puts the current frame into the map just before `DetectLoopClosure`
queries it, so the best match is the newest node itself, at exactly the pose the LSI search starts from, and the
runners-up are its neighbors: with the in-tree bag of words, on KITTI-07 the best match is within 10 s of the query
for 99% of frames. Without a
filter the mode verified the LSI search's own hypothesis plus near duplicates, and reproduced the LSI mode's
trajectory bit for bit. `RecognizeLoopClosurePlaces` ranks the whole map and drops live nodes mapped less than
`kLoopClosureVprMinAgeNs` (20 s) before the newest one. It is a time window rather than a node count because a
10-node guard (about 5 s on KITTI) let a look-alike stretch of the same pass through: a road 145 m back verified on
KITTI-02.

**A loop closure has to be plausible, not just verifiable.** Verification around a candidate's pose checks that the
current frame fits the landmarks there, not that the robot can be there. On KITTI-01, a highway with no loop at all,
the in-tree bag of words and verification together closed 129 false loops and the translation error went from 1.5% to 38%. `LoopClosureSolverVpr`
therefore rejects a result that would move the pose estimate by more than `kMaxCorrectionM` (10 m) or
`kMaxCorrectionDeg` (10 degrees). The LSI search never corrected by more than 7.4 m and 2 degrees on KITTI 00-10
(median 0.04 m), because it only searches around the estimate. A 5 m bound rejected KITTI-09's genuine loop
closure, a 10 m correction; a 20 m bound let KITTI-02's false ones through. A loop whose drift exceeds the bound is
not closed. Both bounds, and the 20 s window, were chosen on the same eleven sequences the comparison below is
measured on, and KITTI-09's loop closure sits right at the translation bound, so they are fitted to KITTI rather than
validated on it; a bound that grows with the distance travelled since the candidate was mapped is the obvious next
step. `AnyLoc` proposes the right place far more often (at 87% of KITTI-00's revisits it has one among its five
candidates), and the bound still rejected two to four verified loop closures per long sequence, 14 to 47 m away.

**Two verifications have to agree.** Started from a candidate's pose, verification can settle on the candidate's own
node, a few meters from the camera, instead of on the camera. With `AnyLoc` on KITTI-00 one loop closure moved the
estimate 8.2 m onto a node that ground truth put 8.3 m from the camera, four frames after another to the same place
had moved it 3.5 cm; the 10 m bound lets that through. So the first time a candidate verifies, `LoopClosureSolverVpr`
also runs the verification from the estimate itself, and when that succeeds as well, the two results have to agree
within `kMaxDisagreementM` (1 m) and `kMaxDisagreementDeg` (2 degrees). When it fails, drift has carried the
estimate too far for a search around it, which is the loop this mode is for, and the candidate's result stands.

**Only the accepted attempt reports.** The landmark probes a loop closure attempt reports feed
`LSIGrid::MakeLandmarkQualityFunc`, which decides what a full cell drops. A rejected candidate is usually a place the
camera is not at, so what its attempt failed to find says nothing about the landmarks there, and it is not reported.

With `AnyLoc` on its FP16 TensorRT engine, over three runs of each mode on KITTI 00-10 with `max_map_size` 0, the
mean translation error is 0.759% against 0.747% for the LSI mode and the mean absolute trajectory error 1.92 m
against 1.98 m, as it was when DINOv2 ran in fp32 on ONNX Runtime (0.756% and 1.93 m). Neither mode is deterministic
from run to run (KITTI-02's ATE ranged from 2.3 to 5.2 m in the LSI mode alone, and from 2.4 to 5.3 m with
`AnyLoc`), and most sequences are within that noise, KITTI-01 among them with no loop closed. KITTI-09's final loop
is closed in every run (ATE 1.72 m against 2.45 m, at a higher segment error, 0.92% against 0.80%), where the LSI
search closed it in one run of three; KITTI-00 and KITTI-08 do slightly worse (0.80% against 0.79%, and 1.01%
against 0.99%), within the overlap of their run-to-run ranges. Two costs come with it:

- **Speed.** A place recognition query and up to five verifications per keyframe, on the SLAM thread. `AnyLoc` runs
  DINOv2 twice per keyframe, once to map the frame and once to query it, but on TensorRT that is now the small part:
  alone on an RTX 4090, KITTI-00 tracked at 232 frames per second with the LSI mode, 167 with `AnyLoc` filling the
  map and no queries, and 66 in this mode. The rest is the query and the verifications, which a backend with no
  network pays as well: the in-tree bag of words tracked KITTI-00 at 30 frames per second in this mode. With six runs sharing the machine, this
  mode ran at 22.6 frames per second against 63.3 for the LSI one; on ONNX Runtime's CPU build it ran at 2.7.
- **Vocabulary.** `AnyLoc` fits its vocabulary once, to the first 8 keyframes; fitting it to the first 100 instead
  changed nothing on KITTI-00 and 02.

### Still open

- **A bounded map forgets the places it would close loops to.** When the pose graph exceeds
  `Slam::Config::max_map_size`, merged-away nodes take their pictures with them, while their landmarks move to the
  surviving node. Long sequences therefore keep fewer places to recognize than landmarks to verify against; set
  `max_map_size` to 0 when comparing the two modes.
- **Each candidate is verified from its node's pose only.** `LocalizeInMap` probes a small grid around the same kind of
  candidate (`Localizer::UseRecognizedPlaceProbes`), because a node is a keyframe spacing away from the camera; doing
  that here would cost several verifications per candidate, on top of a mode that is already the slow one.
- **When both verifications agree, the candidate's result is kept.** The estimate's own result, which is what the
  LSI mode would have used, may be the better pose to close the loop with; that is untested, and a candidate for
  KITTI-00's small deficit.
- **`AnyLoc` describes every keyframe twice.** The query is the frame `AddKeyframe` has just mapped, so querying by
  the newest node's stored descriptor would save one DINOv2 run and one VLAD encoding per keyframe, a few
  milliseconds now that the network runs on TensorRT.
- **`LoopClosureSolverTwoStepsEasy`'s "first to second beats current to second" check compares rotation matrix norms**,
  which are the same for every rotation, so rounding noise decides the rotation half of it. It predates this mode,
  which leans on it more than the default one does.

## Relocalizing a kidnapped robot

Recognition on its own does not relocalize anything. A match is the pose of some other keyframe, accurate only to
the keyframe spacing (measured on these datasets: 0.77 m median on KITTI-07, 1.5 m on CODa), with no orientation
guarantee and no way to tell a true match from a wrong one — across sessions on CODa even AnyLoc answers with a
wrong place for 1 to 2.5% of the queries. Geometric verification is what turns a hint into a pose, and it needs the
landmarks, which means it needs the SLAM map. That split is wired up as:

1. **The place recognition map lives inside the SLAM map.** `Slam::SaveMap` writes it into the same LMDB database
   as the landmarks and the pose graph, and `Localizer::OpenDatabase` reads all of them back together, so its node
   ids are the ids of the pose graph restored beside it and a match resolves through that graph. Because
   `LocalizeInMap` swaps the whole `LocalizerAndMapper` in on success, the session then owns both.
2. **`LocalizeInMap` seeds itself.** Its `guess_pose` is optional; without one it asks the loaded map for up to
   `LocalizationSettings::vpr_candidates` places (default 5) that look like the images and runs the existing probe
   search seeded at each in turn, keeping the first whose PnP verification passes.
3. **`RecognizePlaceByFrame` stays the cheap hint** (on its TensorRT engine AnyLoc answers in 3 ms on an RTX 4090,
   against 38 ms with DINOv2 on the CPU, over a 300 frame map of KITTI-07), because its job is *detecting* the
   kidnapping, not resolving it.

### Still open

- **The seeded probe grid is a guess, not a measurement.** `Localizer::UseRecognizedPlaceProbes` narrows the grid
  around a recognized place to one translation step and four yaw steps — about 135 probes against the 8800 a blind
  guess walks — because a recognized frame is a frame spacing away and already pointing the right way. That the
  narrowed radius actually covers the residual error has not been measured on a real map; it is the mapped frame
  spacing, and a map whose nodes are further apart than one grid step will want a wider one.
- **After `LocalizeInMap` the session maps into the loaded map.** Its new keyframes join `RecognizePlaceByFrame`'s
  search, the localization keyframe first, so a query can match a frame of this session; and the loop closure filter
  for the recent past judges loaded places by their saved timestamps. `LocalizeInMap` refuses a map with more nodes
  than the session's `max_map_size`.
- **Nothing detects the kidnapping.** The two-stage protocol — run recognition continuously, and when its answer
  disagrees with the current SLAM pose for several consecutive frames, or when there is no SLAM pose yet, call
  `LocalizeInMap` without a guess and continue tracking in the map it swaps in — is left to the caller.
