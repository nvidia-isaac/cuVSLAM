# Visual place recognition

This module answers one question: **which pose graph node saw this frame?** The answer is a node, never a metric
pose of its own; `Slam::LocalizeInMap` is what turns that hint into a verified pose. What the feature is for, and
how a caller drives it, is in the [root README](../../../README.md#visual-place-recognition).

## Shape of the module

```text
vpr_types.h     VprType, VprImage, VprOptions, VprMatch — no dependencies beyond slam_common.h
ivpr.h/.cpp     the IVpr backend interface and the factory that picks a backend from VprOptions
vpr_image.*     grayscale conversion, box downscale, bilinear resize, pixel normalization
vpr_simple.*    Simple backend, always built
vpr_orb.*       in-tree ORB: FAST-9, Harris ranking, rotated BRIEF, standard library only
vpr_bow.*       Bow backend over vpr_orb, always built
vpr_dbow2.*     DBoW2 backend, built when USE_DBOW2 is ON (needs OpenCV)
vpr_anyloc.*    AnyLoc backend, built when USE_ONNXRUNTIME is ON (needs a model file)
vpr_map.*       VprMap: owns a backend and the node bookkeeping; the only type the rest of SLAM touches
vpr_sof_image.* bridge from the tracker's sof::ImageContext to a VprImage
```

## Design decisions

**A match resolves to a node, not to a pose.** `IVpr` stores nothing but descriptors keyed by `KeyFrameId`, and
`LocalizerAndMapper::RecognizePlace` resolves the node through the pose graph at query time. A place recognized
after a loop closure therefore reports the *corrected* position, not the drifted one that was current when the
frame was mapped. The poses `VprMap` stores are only the snapshot `Save()` writes, which is what a session that
loads the map falls back on, having no pose graph for those nodes.

**Exactly one frame per node.** `VprMap::AddFrame` is a no-op when the node already has one, which is what lets
`Slam::AddFrameToVprMap` be called on every single frame: whichever frame happens to follow a new keyframe becomes
that node's picture, and the rest cost a `HasImage` lookup. It also bounds the map to the size of the pose graph,
which matters because every backend here searches exhaustively.

**A map loaded through `Slam::Config::vpr_map_path` is read only.** That session is localizing against somebody
else's route, while its own keyframes are of places it is looking at right now. Mixing both into one search loses
almost every query to a frame from a few seconds ago, which is a correct nearest neighbor and a useless answer to
"where am I"; the vocabulary backends would on top of that retrain on every added frame. Ids of a loaded map are
shifted by `VprMap::kImportedNodeIdBase` so they cannot collide with the ids the live pose graph hands out.

**An ambiguous place is dropped, not reported.** All three backends run the same ratio test: the winner has to beat
the best match from a different part of the trajectory, more than `kNeighbourGuard` nodes away in creation order.
Between two candidates that look equally much like the query, reporting either is a coin flip, and a wrong pose
costs a relocalizer more than a missing one.

**Two paths feed the map, and they cover each other.**

- `LocalizerAndMapper::AddKeyframe` calls `VprMap::OnKeyframeAdded` and attaches pixels when the frame's image
  context still holds them. This is the path that works in asynchronous mode, where the caller has no way to know a
  keyframe was just created.
- `Slam::AddFrameToVprMap` converts a caller's frame on the caller's thread and binds it to the newest node. This
  is the path that works when SLAM has fallen behind odometry and the worker found the image context recycled.

Neither path ever retains an `sof::ImageContextPtr`. The tracker's image pool holds four contexts per camera and
hands out a slot only when its use count drops to one, so keeping a reference for even one extra frame can starve
`Odometry::Track`. Pixels are copied into a `VprImage` and the context is released immediately.

## Evaluating a backend

`cuvslam_vpr_reporter` scores a backend on two traversals of one route. `DATASETS.md` lists the public datasets
that can drive it and the ones that cannot, and why.

### What the backends actually do, measured

The interesting result is not that every backend recognizes a sequence replayed against a map of itself — they all
do, at 96 to 100%. It is what happens across two *separate* recordings of one route, which is the case a robot is
actually in. Measured on CODa-00 against CODa-05, 2500 frames, success radius 10 m:

| Backend | Self-pair (KITTI-07) | Cross-session (CODa-00 → CODa-05) |
|---------|---------------------|------------------------------------|
| `Simple` | 96% recognized | 49% recognized, 18% wrong place |
| `Bow` | 99% | 8% recognized, 2% wrong place |
| `DBoW2` | 99% | 10% recognized, 1% wrong place |
| `AnyLoc` | 100% | 95% recognized, 1% wrong place |

For both bag-of-words backends the score distributions of correct and wrong matches **overlap almost completely**
across sessions — `Bow` medians 0.279 against 0.267, `DBoW2` 0.128 against 0.123 — so no threshold separates them
and the calibration can only choose where to give up. Both defaults choose precision: a robot told confidently
that it is somewhere it is not is worse off than one told nothing. AnyLoc's DINOv2 features are the only ones here
that survive the viewpoint, lighting and traffic differences between two drives.

Two consequences worth stating plainly. A backend calibrated on a self-pair does not transfer: the first `Bow`
default was set that way and produced 75% wrong answers on the cross-session pair. And the fix for the bag-of-words
backends is not a better threshold — it is retrieval that uses more than the top-1 score, which is what the
covisibility clustering listed under the open items below would provide.

## Adding a backend

Implement `IVpr`, add the type to `VprType` and `Slam::VprMode` (they are kept numerically identical and
`static_cast` between each other in `cuvslam2.cpp`), and construct it in `CreateVpr`. A backend that needs a
vocabulary builds it in `Finalize()`, which `VprMap` calls before the first query and before serializing; the
interface is written that way so a backend never has to train anything while a frame is being mapped.

## Closing loops

`Slam::Config::loop_closure_mode = LoopClosureMode::Vpr` makes place recognition the only source of loop closure
candidates. `LocalizerAndMapper::DetectLoopClosure` asks `RecognizeLoopClosurePlaces` for up to
`kLoopClosureVprCandidates` (5) places, and `LoopClosureSolverVpr` runs the default mode's own verification
(`LoopClosureSolverTwoStepsEasy`) re-centered on each candidate's pose in turn, keeping the first that passes. Each
of the decisions below was forced by a measurement on KITTI 00-10:

**The recent past is left out.** `AddKeyframe` puts the current frame into the map just before `DetectLoopClosure`
queries it, so the best match is the newest node itself, at exactly the pose the default search starts from, and the
runners-up are its neighbors: with `Bow`, on KITTI-07 the best match is within 10 s of the query for 99% of frames. Without a
filter the mode verified the default search's own hypothesis plus near duplicates, and reproduced the default mode's
trajectory bit for bit. `RecognizeLoopClosurePlaces` ranks the whole map and drops live nodes mapped less than
`kLoopClosureVprMinAgeNs` (20 s) before the newest one. It is a time window rather than a node count because a
10-node guard (about 5 s on KITTI) let a look-alike stretch of the same pass through: a road 145 m back verified on
KITTI-02.

**A loop closure has to be plausible, not just verifiable.** Verification around a candidate's pose checks that the
current frame fits the landmarks there, not that the robot can be there. On KITTI-01, a highway with no loop at all,
`Bow` and verification together closed 129 false loops and the translation error went from 1.5% to 38%. `LoopClosureSolverVpr`
therefore rejects a result that would move the pose estimate by more than `kMaxCorrectionM` (10 m) or
`kMaxCorrectionDeg` (10 degrees). The default search never corrected by more than 7.4 m and 2 degrees on KITTI 00-10
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

With `AnyLoc`, over three runs of each mode on KITTI 00-10 with `max_map_size` 0, the mean translation error is
0.756% against 0.745% for the default mode and the mean absolute trajectory error 1.93 m against 1.95 m. Neither mode
is deterministic from run to run (KITTI-02's translation error ranged from 0.77% to 0.83% in the default mode alone),
and eight of the eleven sequences are within that noise, KITTI-01 among them with no loop closed. KITTI-09's final
loop, which the default search misses, is closed (ATE 1.70 m against 2.77 m, at a higher segment error, 0.91%
against 0.82%); KITTI-00 does slightly worse (0.82% against 0.79%, ATE 1.73 m against 1.63 m); KITTI-02's ATE varies
too much between runs in both modes to call (2.4 to 5.3 m with `AnyLoc`, 2.6 to 3.5 m without). Two costs come
with it:

- **Speed.** A place recognition query and up to five verifications per keyframe, on the SLAM thread. `AnyLoc` runs
  DINOv2 on the CPU through ONNX Runtime, twice per keyframe, once to map the frame and once to query it: tracking ran
  at 2.7 frames per second against 79 with the default mode, eight runs sharing 28 cores.
- **Backend choice.** `DBoW2` trains its vocabulary from the map and retrains it whenever the map changed since the
  last query, which in this mode is every keyframe, so its cost grows with the square of the map. `AnyLoc` fits its
  vocabulary once, to the first 8 keyframes; fitting it to the first 100 instead changed nothing on KITTI-00 and 02.
  `Bow` fits its vocabulary once, to the first 50 frames.

### Still open

- **A bounded map forgets the places it would close loops to.** When the pose graph exceeds
  `Slam::Config::max_map_size`, merged-away nodes take their pictures with them, while their landmarks move to the
  surviving node. Long sequences therefore keep fewer places to recognize than landmarks to verify against; set
  `max_map_size` to 0 when comparing the two modes.
- **Each candidate is verified from its node's pose only.** `LocalizeInMap` probes a small grid around the same kind of
  candidate (`Localizer::UseRecognizedPlaceProbes`), because a node is a keyframe spacing away from the camera; doing
  that here would cost several verifications per candidate, on top of a mode that is already the slow one.
- **When both verifications agree, the candidate's result is kept.** The estimate's own result, which is what the
  default mode would have used, may be the better pose to close the loop with; that is untested, and a candidate for
  KITTI-00's small deficit.
- **`AnyLoc` describes every keyframe twice.** The query is the frame `AddKeyframe` has just mapped, so querying by
  the newest node's stored descriptor would halve the DINOv2 cost of this mode.
- **`LoopClosureSolverTwoStepsEasy`'s "first to second beats current to second" check compares rotation matrix norms**,
  which are the same for every rotation, so rounding noise decides the rotation half of it. It predates this mode,
  which leans on it more than the default one does.

## Relocalizing a kidnapped robot

Recognition on its own does not relocalize anything. A match is the pose of some other keyframe, accurate only to
the keyframe spacing (measured on these datasets: 0.77 m median on KITTI-07, 1.5 m on CODa), with no orientation
guarantee and no way to tell a true match from a wrong one — the Simple backend answers with a wrong place for
about one query in ten on CODa. Geometric verification is what turns a hint into a pose, and it needs the
landmarks, which means it needs the SLAM map. That split is wired up as:

1. **The place recognition map lives inside the SLAM map.** `Slam::SaveMap` writes `vpr_map.bin` next to the LMDB
   database, and `Localizer::OpenDatabase` reads it back with `VprMap::LoadMode::kWithPoseGraph`, so its node ids
   are the ids of the pose graph just restored beside it and a match resolves through that graph. Because
   `LocalizeInMap` swaps the whole `LocalizerAndMapper` in on success, the session then owns both.
2. **`LocalizeInMap` seeds itself.** Its `guess_pose` is optional; without one it asks the loaded map for up to
   `LocalizationSettings::vpr_candidates` places (default 5) that look like the images and runs the existing probe
   search seeded at each in turn, keeping the first whose PnP verification passes.
3. **`RecognizePlaceByFrame` stays the cheap hint** (measured here: 0.9 ms Simple, 12 ms DBoW2, 46 ms AnyLoc),
   because its job is *detecting* the kidnapping, not resolving it.

### Still open

- **The seeded probe grid is a guess, not a measurement.** `Localizer::UseRecognizedPlaceProbes` narrows the grid
  around a recognized place to one translation step and four yaw steps — about 135 probes against the 8800 a blind
  guess walks — because a recognized frame is a frame spacing away and already pointing the right way. That the
  narrowed radius actually covers the residual error has not been measured on a real map; it is the mapped frame
  spacing, and a map whose nodes are further apart than one grid step will want a wider one.
- **`Slam::Config::vpr_map_path` still loads recognition without landmarks.** Nothing can verify a match on that
  path: the map is read only, its ids are renumbered, and the only pose it can report is the snapshot stored with
  the matched frame. It answers "which frame of the old session does this look like" for a caller that wants the
  hint and no map; whether that earns its keep now that `LocalizeInMap` seeds itself is worth revisiting.
- **Nothing detects the kidnapping.** The two-stage protocol — run recognition continuously, and when its answer
  disagrees with the current SLAM pose for several consecutive frames, or when there is no SLAM pose yet, call
  `LocalizeInMap` without a guess and continue tracking in the map it swaps in — is left to the caller.
