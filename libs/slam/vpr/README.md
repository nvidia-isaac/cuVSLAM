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
