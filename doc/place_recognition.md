# cuVSLAM: Visual Place Recognition

Visual place recognition answers "where am I?" from a single frame and no pose. The
[root README](https://github.com/nvidia-isaac/cuVSLAM/blob/main/README.md#visual-place-recognition) gives the overview; this page shows how to drive it, how it
closes loops and how to evaluate it. Relocalizing in a saved map with `LocalizeInMap` is covered in
[load_map.md](load_map.md), and how the module is built, with the measurements behind it, in
[libs/slam/vpr/README.md](https://github.com/nvidia-isaac/cuVSLAM/blob/main/libs/slam/vpr/README.md).

## The engine

`Slam::VprSettings::model_path` (Python: `Slam.VprSettings.model_path`) names the AnyLoc TensorRT engine, built as
[models/anyloc](https://github.com/nvidia-isaac/cuVSLAM/blob/main/models/anyloc/README.md) describes. With the wheel,
build it once per machine with `cuvslam.build_anyloc_engine(path)`.

## Mapping and recognizing places

```python
import os

import cuvslam as vslam

engine = os.environ['CUVSLAM_ANYLOC_ENGINE']

def print_place(request):
    place = request.result()
    if place is None:
        print(request.error_message)   # no mapped place matched, or no answer at all, for example use_vpr is off
    else:
        print(place.node_id, place.pose, place.score)

# Mapping run: tracking maps each keyframe's frame, one per pose graph node.
slam_cfg = vslam.Slam.Config(use_vpr=True, vpr_settings=vslam.Slam.VprSettings(model_path=engine), sync_mode=True)
tracker = vslam.Tracker(rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
for timestamp, images in sequence:
    tracker.track(timestamp, images)
print('map saved' if tracker.slam.save_map('/path/to/map').result() else 'map not saved')

# Later run, starting from an unknown pose: localize in the saved map without a guess pose. On success the
# session continues in that map, place recognition included.
tracker = vslam.Tracker(rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
settings = vslam.Slam.LocalizationSettings(horizontal_search_radius=8., vertical_search_radius=2.,
                                           horizontal_step=0.5, vertical_step=0.2, angular_step_rads=0.03)
localized = None
for timestamp, images in sequence:
    tracker.track(timestamp, images)
    if localized is None:
        localized = tracker.slam.localize_in_map('/path/to/map', timestamp, None, images, settings).result()
    else:
        print_place(tracker.slam.recognize_place_by_frame(images, timestamp))
```

The place recognition map is part of the SLAM map: `save_map` writes both into one LMDB database, and `localize_in_map`
reads both from it. Without `map_path` it is kept in memory, like the rest of the map; with it, it is written to that
database by `flush_map` and when SLAM is destroyed. `localize_in_map` replaces that database with the loaded one, which
is written the same way when localized with `attach=True`. After `localize_in_map` the session keeps mapping into the
loaded map, and its new frames are recognized too.

`save_map`, `localize_in_map` and `recognize_place_by_frame` are asynchronous: they queue the work for the SLAM
thread and return a request, which is done when that thread finishes the work. `request.result(timeout=None)` waits
for it, and `request.wait(timeout)` and `request.done()` tell whether it is done. `sync_mode=True` above, which
`Tracker.Mode`'s offline modes require anyway, runs the work before the call returns, so the request is already done
by the time it does.

In C++ the same calls are `Slam::SaveMap`, `Slam::LocalizeInMap` and
`Slam::RecognizePlaceByFrame`, with `Slam::Config::use_vpr = true` and the engine in `Slam::VprSettings::model_path`.
They take callbacks, which run on the SLAM thread, instead of returning a request.

## One vocabulary per operating domain

Each session fits AnyLoc's vocabulary to its own first keyframes, so maps of separate sessions encode places
differently. To share one vocabulary across the sessions of an operating domain, fit it once to images spread over
recordings of that domain, with `cuvslam_vpr_vocabulary` or `cuvslam.VprVocabularyBuilder`, and give every session the
file:

```python
vpr_settings = vslam.Slam.VprSettings(model_path=engine, vocabulary_path='kitti.vocabulary')
slam_cfg = vslam.Slam.Config(use_vpr=True, vpr_settings=vpr_settings)
```

Such a session recognizes places from its first keyframe. Without the file, `vocabulary_min_keyframes` and
`vocabulary_max_keyframes` set what a session fits its own vocabulary to.

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
search: on KITTI-00 with an RTX 4090, 66 frames per second with `AnyLoc` against 232. To compare the two on a KITTI
sequence, with the pose graph unbounded so that no place is merged out of the place recognition map:

```bash
cuvslam_tracker --edex <kitti>/00 --use_slam true --max_map_size 0 --loop_closure_mode lsi ...
cuvslam_tracker --edex <kitti>/00 --use_slam true --max_map_size 0 --loop_closure_mode vpr \
    --use_vpr true --vpr_model_path <build>/bin/anyloc_vits14_l9_value_322_fp16_trt<version>.engine ...
```

`cuvslam_reporter` takes the same flags, and both record `num_loop_closures` in their stats.

## Evaluating it

`cuvslam_vpr_reporter` (in `tools/python_tools`) scores place recognition on two recordings of the same route: it
maps the first, asks that map where each frame of the second was taken, and reports the recognition and false positive
rates against ground truth. It draws the query trajectory green where the place was recognized correctly, red where it was not
recognized at all, and amber where the map answered with the wrong place. See `tools/python_tools/README.md`.
