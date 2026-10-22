# cuVSLAM: Load Map

The only mode we currently support is **LocalizeInMap**. In this mode, we first localize against the map; if that
succeeds,
the current map is replaced by the loaded one. We do **not** support unlocalized loading (load now, localize later).

Users who call **LocalizeInMap** must provide:

- Image(s)
- Timestamp (`timestamp_ns` on the call, aligned with the images)
- Prior 3D pose (`guess_pose`), or none with `Config::use_vpr`: place recognition then proposes the poses to verify
  (see [place_recognition.md](place_recognition.md))

## API

```cpp
/**
 * Process tracking results from `Odometry::Track`.
 * This should be called after each successful tracking.
 * @param[in] state Odometry state containing all tracking data
 * @param[in] gt_pose Optional ground truth pose. Should be provided if `gt_align_mode` is enabled,
 *                    otherwise should be nullptr.
 * @see `Odometry::Track`
 * @throws std::invalid_argument if `gt_pose` is passed incorrectly
 */
void Track(const Odometry::State& state, const Pose* gt_pose = nullptr);

/**
 * Get the current rig pose estimated by SLAM.
 * Call this after Track() when the current pose is needed.
 */
Pose GetPose() const;
```

```cpp
/**
 * Save SLAM database (map) to folder asynchronously.
 * This folder will be created if it does not exist.
 * Contents of the folder will be overwritten.
 * Fails while the session writes its map in place (`Config::map_path`, or `LocalizeInMap` with `attach`):
 * FlushMap() saves it there. Also fails for the folder of a map that `LocalizeInMap` loaded and the folders that
 * contain it.
 * The saved map includes the place recognition map when `Config::use_vpr` is set.
 *
 * @param[in] folder_name Folder name where the SLAM database (map) will be saved
 * @param[in] callback    Callback function to be called when save is complete;
 *                        may be called in a separate thread
 */
void SaveMap(const std::string_view& folder_name,
             std::function<void(bool success)> callback) const;
```

```cpp
/**
 * Write all unsaved data of the map to its map file asynchronously: `Config::map_path`, or the folder that
 * `LocalizeInMap` attached. The file then holds the whole map, and mapping goes on as before. Fails without one.
 *
 * @param[in] callback Callback function to be called when flush is complete;
 *                     may be called in a separate thread
 */
void FlushMap(std::function<void(bool success)> callback) const;
```

```cpp
/// Localization callbacks; may be invoked in a separate thread
using LocalizeStartCB = std::function<void()>;
using LocalizeFinishCB = std::function<void(const Result<Pose>& result)>;

/**
 * Localize in the existing database (map).
 *
 * If `Config.sync_mode` is false, the request is queued for the background SLAM thread;
 * callbacks run when localization finishes (possibly on a thread other than the caller's).
 * If `Config.sync_mode` is true, localization runs before this call returns.
 *
 * Finds the rig pose in the saved map. If successful, replaces the current map with the saved one.
 * Without `guess_pose`, it needs `Config::use_vpr` and a map saved with it.
 * Fails if the map has more nodes than `Config::max_map_size` (0: no limit), and for the folder of the map file
 * this session writes; FlushMap() saves that one.
 *
 * @param[in] folder_name  Folder containing the saved SLAM map (database)
 * @param[in] timestamp_ns Time in nanoseconds for the localized pose; if all image timestamps match,
 *                         using `images[0].timestamp_ns` is typical
 * @param[in] guess_pose   Initial guess for rig pose at the images' timestamp, or nullptr to use place recognition
 * @param[in] images       Observed images from a multicamera rig (1 = mono, 2 = stereo, etc.)
 * @param[in] settings     Localization search grid settings
 * @param[in] start_cb     Called when localization starts; may run on another thread
 * @param[in] finish_cb    Called when localization completes; may run on another thread
 * @param[in] attach       On success, the session writes its map into `folder_name` from then on, as with
 *                         `Config::map_path`: FlushMap() saves it there, SaveMap() fails, and destroying Slam
 *                         writes it there. The folder holds a whole map only after that. Nothing else may open
 *                         the folder while it is attached.
 */
void LocalizeInMap(std::string_view folder_name, int64_t timestamp_ns, const Pose* guess_pose, const ImageSet& images,
                   const LocalizationSettings& settings, LocalizeStartCB start_cb, LocalizeFinishCB finish_cb,
                   bool attach = false);
```

## QA

**Loading of maps — what do we want to support, and how should it work?**

Current functionality is minimal, solid, bug-free, and well-tested:

* No external GlobalLocalizer (no GPS, no `slam.SetPose`).
* No tracking-lost recovery support.
* No user-provided tracking-lost detection.
* Sensor fusion is limited to visual SBA only.

There is a single cuVSLAM system that runs until the first tracking loss. Users can save and load maps without
interrupting real-time processing.

**Scenarios**

1. The user creates a `cuVSLAM::Slam` instance.
2. The user feeds images to `Odometry::Track`, passes each successful odometry state to `Slam::Track`, and obtains the
   current SLAM pose with `Slam::GetPose`. This is a low-latency, real-time path (e.g. ~100 fps); the background thread
   runs LC search and map building.
3. At some point the user calls `SaveMap` (possibly from another thread). The map is written to disk by the background
   path; `Track` is not blocked by that work.
4. At some point the user calls `LocalizeInMap` with image(s), timestamp, and prior. The background thread matches the
   images to the on-disk map. On success it drops the current map and replaces it with the loaded one. After that, the
   first `GetPose` following the next `Track` reports a pose in the new map.

**Camera moves during startup, then a map load is requested — what behavior do we support?**

During the load, the user keeps calling `Track` for every frame and reading the current pose with `GetPose`. After the
map is applied, the first `GetPose` following the next `Track` returns the real robot pose in the loaded map.

**Example: car at 80 km/h, cuVSLAM starts, then 3 minutes / ~4 km later a map load is requested — what is expected?**

The user supplies a localization image + timestamp + prior for the “current” moment (e.g. timestamp = 3 min). Even if
loading takes time (e.g. 10 s), poses over that interval are reconciled to the localized pose at that timestamp. History
before that timestamp is dropped.

**Ten minutes later another map load — what is expected?**

Same pattern: timestamp = 13 min (for example), current image and prior. As in step 3, previous history is replaced when
localization succeeds.

**What if the map has nothing around the location?**

`LocalizeInMap` fails (`finish_cb` with failure); cuVSLAM keeps the current map.

**What if the map is wrong (e.g. from a different session)?**

There is no automatic repair. If the user needs a full reset (for example, tracking was lost a minute ago), they must
reinitialize cuVSLAM.

**What if the map is huge and load takes minutes?**

After the load, odometry since the localization timestamp is applied at the localized pose. cuVSLAM has
`retention_time`—history older than that window cannot be used; localization with too-old timestamps fails.

In “connected to database” setups, loading is often a quick reconnect and the map is ready immediately.

## Load map under the hood

I'm a Robot: [load_map_mermaid.md](load_map_mermaid.md).

![Load map under the hood](load_map.png)

**`Slam::Track()`**

- Accumulate odometry between keyframes.
- Append new keyframe to the end of the tail.

**`Slam::BackgroundThread::ProcessMessage`**

- Update state in the past if localize or find-FC runs.
- Recalculate poses from the updated segment to the latest.

## Use cases

1. The user calls localize with a timestamp **≥** the last pose in the tail. Odometry inputs with older timestamps are
   ignored
   until a newer timestamp arrives.
2. The user calls localize with a timestamp **outside** the retention window. Localization returns failure.
3. The user calls localize with a fresh timestamp **before** the last pose in the tail. The localized pose is injected
   at
   the correct place in the tail (same idea as FindLC).

---

Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
