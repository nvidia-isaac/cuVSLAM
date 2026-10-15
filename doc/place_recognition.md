# cuVSLAM: Visual Place Recognition

Visual place recognition answers "where am I?" from a single frame and no pose. The
[root README](../README.md#visual-place-recognition) gives the overview; this page shows how to drive it, how it
closes loops and how to evaluate it. Relocalizing in a saved map with `LocalizeInMap` is covered in
[load_map.md](load_map.md), and how the module is built, with the measurements behind it, in
[libs/slam/vpr/README.md](../libs/slam/vpr/README.md).

## The engine

Place recognition runs AnyLoc, which describes a frame with DINOv2 features computed on the GPU. DINOv2 runs as a
TensorRT engine that a `USE_TENSORRT` build makes for one of two GPUs, an RTX 4090 or a Jetson Orin Nano
(`CUVSLAM_ANYLOC_TARGET`), on that GPU and with its TensorRT version:

```bash
cmake -S . -B build -DUSE_TENSORRT=ON
cmake --build build
source build/cuvslam_vars.sh   # exports CUVSLAM_ANYLOC_ENGINE, the engine's path
```

`Slam::Config::vpr_model_path` (Python: `vpr_model_path`) names the engine, and AnyLoc needs `use_gpu`. An engine
only loads in the TensorRT version and on the GPU architecture that built it; see
[models/anyloc](../models/anyloc/README.md).

## Mapping and recognizing places

```python
import os

import cuvslam as vslam

# The AnyLoc TensorRT engine the USE_TENSORRT build made; build/cuvslam_vars.sh exports its path.
engine = os.environ['CUVSLAM_ANYLOC_ENGINE']

def on_place(place, error):
    if error:
        print(error)   # no answer at all, for example use_vpr is off
    elif place.found:
        print(place.node_id, place.pose, place.score)

# Mapping run: offer every frame. The map keeps one frame per pose graph node, so the calls that
# land on an already mapped node cost almost nothing.
slam_cfg = vslam.Slam.Config(use_vpr=True, vpr_model_path=engine, sync_mode=True)
tracker = vslam.Tracker(rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
for timestamp, images in sequence:
    tracker.track(timestamp, images)
    tracker.slam.add_frame_to_vpr_map(images, timestamp)
tracker.slam.save_map('/path/to/map', lambda saved: print('map saved' if saved else 'map not saved'))

# Later run, starting from an unknown pose: localize in the saved map without a guess pose. On success the
# session continues in that map, place recognition included.
tracker = vslam.Tracker(rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
settings = vslam.Slam.LocalizationSettings(horizontal_search_radius=8., vertical_search_radius=2.,
                                           horizontal_step=0.5, vertical_step=0.2, angular_step_rads=0.03)
localized = []
for timestamp, images in sequence:
    tracker.track(timestamp, images)
    if not localized:
        tracker.slam.localize_in_map('/path/to/map', timestamp, None, images, settings, lambda: None,
                                     lambda pose, error: pose and localized.append(pose))
    else:
        tracker.slam.recognize_place_by_frame(images, timestamp, on_place)
```

The place recognition map is part of the SLAM map: `save_map` writes both into one LMDB database, and `localize_in_map`
reads both from it. Without `map_path` it is kept in memory, like the rest of the map; with it, it is written to that
database by `flush_map` and when SLAM is destroyed. `localize_in_map` replaces that database with the loaded one, which
is written the same way when localized with `attach=True`. After `localize_in_map` the session keeps mapping into the
loaded map, and its new frames are recognized too.

`save_map`, `localize_in_map` and `recognize_place_by_frame` are asynchronous: they queue the work for the SLAM
thread and their callbacks run when it finishes, possibly on that thread. `sync_mode=True` above, which
`Tracker.Mode`'s offline modes require anyway, runs the work before the call returns, so the callback has already
fired by the time it does.

In C++ the same calls are `Slam::AddFrameToVprMap`, `Slam::SaveMap`, `Slam::LocalizeInMap` and
`Slam::RecognizePlaceByFrame`, with `Slam::Config::use_vpr = true` and the engine in `Slam::Config::vpr_model_path`.

## Closing loops with it

By default (`LoopClosureMode::LSI`) loop closure searches the landmark spatial index around the current pose estimate,
so it closes a loop only when drift has not carried the estimate away from the landmarks of the place it is back at.
`Slam::Config::loop_closure_mode = LoopClosureMode::VPR` (Python: `Slam.LoopClosureMode.VPR`) searches instead at
the places AnyLoc finds most similar to the current frame, and verifies each against the landmarks
there the same way. Appearance alone is not trusted: the last 20 seconds of the trajectory are never a candidate,
since they always look most like the current frame, and a loop closure is rejected when it would move the pose
estimate by more than 10 m or 10 degrees, or when verification started from the estimate itself succeeds and
disagrees with it, since a look-alike stretch of road can pass the geometric check.

It queries place recognition and verifies up to five candidates on every keyframe, so it is slower than the LSI
search: on KITTI-00 with an RTX 4090, 52 frames per second with `AnyLoc` against 232. To compare the two on a KITTI
sequence, with the pose graph unbounded so that no place is merged out of the place recognition map:

```bash
cuvslam_tracker --edex <kitti>/00 --use_slam true --max_map_size 0 --loop_closure_mode lsi ...
cuvslam_tracker --edex <kitti>/00 --use_slam true --max_map_size 0 --loop_closure_mode vpr \
    --use_vpr true --vpr_model_path <build>/bin/anyloc_vits14_l9_value_322_fp16_trt<version>_sm<gpu>.engine ...
```

`cuvslam_reporter` takes the same flags, and both record `num_loop_closures` in their stats.

## Evaluating it

`cuvslam_vpr_reporter` (in `tools/python_tools`) scores place recognition on two recordings of the same route: it
maps the first, asks that map where each frame of the second was taken, and reports the recognition and false positive
rates against ground truth. It draws the query trajectory green where the place was recognized correctly, red where it was not
recognized at all, and amber where the map answered with the wrong place. See `tools/python_tools/README.md`.
