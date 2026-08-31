# cuVSLAM Design Concepts

This document captures architectural decisions and design principles for cuVSLAM.
Follow these when making code changes or designing new features.

---

## 1. Per-frame internal overrides are stateless — no setters

**Rule:** A low-level parameter that must vary for a single frame is passed explicitly to
`Odometry::Track()` through `cuvslam::internal::Internals`. Do not mutate a long-lived
object through a setter to change one frame.

`Internals` is an unstable expert/development interface declared in
`libs/cuvslam/cuvslam2_internal.h`. It is not part of the stable user-facing API. Normal
applications should omit it and use the built-in defaults.

**Why:** Setters create implicit shared state between frames. A parameter change made
during one `Track()` call can silently bleed into the next frame if the setter mutates an
object that is reused across calls. This makes behavior hard to reason about, test, and
reproduce.

**How it works in cuVSLAM:**

`Odometry::Track()` accepts an optional pointer to `Internals`. All fields have concrete
defaults. `BuildTrackFrameSettings()` converts the selected values to
`odom::TrackPerFrameSettings`, which is threaded down the call stack without modifying
stored settings.

```cpp
// Expert/development use: override feature count for one frame only.
cuvslam::internal::Internals internals;
internals.num_desired_tracks = 200;
odometry.Track(images, {}, {}, &internals);

// The next call uses built-in defaults.
odometry.Track(images);
```

Construction-time settings stored by `Odometry::Impl` are not changed by the per-frame
override.

**What to avoid:**

```cpp
// BAD — setter mutates shared state and can bleed across frames.
odometry.SetNumDesiredTracks(200);
odometry.Track(images);
```

**Where to add a new per-frame internal parameter:**

1. Confirm that the parameter is for expert/development tuning rather than a normal user
   feature. Stable user-facing behavior belongs in `Odometry::Config` or another public API.
2. Add the field and its default to `cuvslam::internal::Internals` in
   `libs/cuvslam/cuvslam2_internal.h`.
3. Map it into `odom::TrackPerFrameSettings` in `BuildTrackFrameSettings()` in
   `libs/cuvslam/cuvslam2.cpp`.
4. Add it to the appropriate `TrackPerFrameSettings` sub-struct (`sof`, `kf`, `pnp`,
   `icp`, and so on), then thread it through the call chain without storing it.
5. Update the Python binding and YAML loader only when the parameter must be available to
   the corresponding development tools.

---

## 2. Resolve optional inputs at the API boundary

**Rule:** Optional input is appropriate at an API boundary where a caller may genuinely
omit a value. Internal APIs below that boundary should receive concrete settings whenever
possible.

**Why:** Optionals at every layer of the call stack force every internal function to check
`has_value()` before use. This is noise. Once the public API has resolved an optional to a
concrete value (using a default), the rest of the system should not need to know the value
was ever absent.

**How it works in cuVSLAM:**

`Odometry::Track()` accepts a nullable `Internals` pointer. A null pointer selects
`Internals{}`. `BuildTrackFrameSettings()` converts the result to a concrete
`TrackPerFrameSettings`; lower layers do not need to know whether the caller supplied an
override.

`Internals::kf_override_frame_selection` is an intentional tri-state exception: unset uses
automatic keyframe selection, `true` forces a keyframe, and `false` forces a non-keyframe.
Resolve such values at the first layer that has enough context, rather than propagating
optionality farther down the call stack.

```text
Internals* (null or expert/development overrides)
    └─► Internals{} when null
        └─► BuildTrackFrameSettings() produces TrackPerFrameSettings
            └─► IVisualOdometry::track(TrackPerFrameSettings&)   // no optional
                    └─► IMultiSOF::trackNextFrame(TrackPerFrameSettings&)  // no optional
                            └─► IMonoSOF::track(Settings&)  // no optional
```

**What to avoid:**

```cpp
// BAD — optional leaks into internal API
void trackNextFrame(..., std::optional<Settings> sof_settings = std::nullopt);

// BAD — internal function must check presence
if (sof_settings.has_value()) { ... }
```

**Corollary — no default arguments on internal functions:**

Internal functions should not have `= {}` default arguments. That is just a hidden optional.
Every call site should pass the struct explicitly, making the data flow visible in the code.

```cpp
// BAD — hides that data is being passed; caller can silently get wrong defaults
void track(const Settings& sof_settings = {});

// GOOD — caller always states what settings it is using
void track(const Settings& sof_settings);
```

---

## 3. Bundle related parameters into a struct rather than growing argument lists

**Rule:** When a group of parameters is always used together or represents a coherent
configuration unit, wrap them in a named struct. Do not add individual parameters to
function signatures.

**Why:** Long argument lists are fragile (easy to reorder), hard to extend, and obscure
what a function actually needs. A named struct documents intent, can be forwarded as a
single argument through multiple layers, and makes adding new fields backwards-compatible
at the struct level.

**How it works in cuVSLAM:**

- `sof::Settings` — all feature tracking parameters.
- `odom::KeyFrameSettings` — keyframe selection thresholds.
- `odom::TrackPerFrameSettings` — bundles the above two for passing through the VO layer.
- `sba::Settings`, `pnp::PNPSettings`, etc. — each subsystem owns its config struct.

When a new per-frame parameter category is needed (e.g. ICP overrides), add a new
sub-struct to `TrackPerFrameSettings` rather than adding individual fields or new function
parameters:

```cpp
struct TrackPerFrameSettings {
  sof::Settings sof;
  KeyFrameSettings kf;
  // Add new categories here, not as additional function parameters
};
```

---

## 4. Three-thread execution model: main, SBA, SLAM

**Rule:** cuVSLAM tracking runs on three logical threads with distinct
real-time guarantees. Code that lives on one thread must not assume the
behaviour of another.

| Thread | Owns | Real-time? | What it costs to skip work |
|---|---|---|---|
| **Main** | Feature selection, LK tracking, keyframe decision, triangulation | Yes — must finish every frame, or tracking is lost | A skipped frame can cost tracking during fast motion (see below). |
| **SBA** (background) | Sparse bundle adjustment — fine-tuning of 3D landmark positions | No | Free. Skipping SBA on a frame causes no measurable accuracy loss; it is purely an optimization pass. |
| **SLAM** (background) | Long-term map, loop-closure search | No — operates "in the past" with a command queue | Free at the cost of latency. The SLAM thread can be hundreds of frames (seconds) behind the main thread; that is the design, not a bug. |

**Why this matters:**

The main thread uses Lucas-Kanade, which assumes a small motion vector between
adjacent frames. During fast motion one missed frame can leave no matchable
patches, and then features die, tracking is lost and the system relocalizes
from the origin; during slow motion the same gap is usually survivable. Linux
is not a real-time OS, so cuVSLAM is typically the first system component to
fail when motion planners and DNNs contend for GPU/CPU. This is what `Odometry::Config::async_sba` and
`Slam::Config::sync_mode` exist to control. For reproducible debugging, force
SBA and SLAM onto the main thread (`async_sba = false`, `sync_mode = true`).
For production deployment, leave them on background threads and accept that
loop-closure corrections arrive late.

**Loop closures are retroactive.** When the SLAM thread finds a loop closure
"three minutes ago," it writes the corrected pose straight into the shared
`Tail` (`slam::Tail::UpdatePoseBySLAM`, mutex-guarded) — it does not queue a
correction for the main thread to apply. The correction is composed as a delta
onto that pose and every retained pose after it, so it moves the recent
trajectory, not just the current frame. An update older than the `Tail`
retention window is rejected and logged rather than applied. The resulting
shift is however large the accumulated drift was: usually small, but nothing
guarantees that. This is correct behaviour — the shift removes drift, it does
not introduce error.

**What to avoid:**

- Do not move main-thread work onto the SBA or SLAM thread "because they have
  spare cycles." They do not — they are working in the past.
- Do not add a synchronous wait from the main thread into the SLAM thread.
  It removes the asynchronous decoupling the design depends on: the main
  thread's per-frame deadline then has to cover SLAM's runtime too, so
  whenever SLAM overruns its budget the main thread misses its deadline, and
  that can lose tracking on real robots.

---

## 5. SLAM is a benchmark; odometry is the product

**Rule:** Treat odometry as the externally consumed surface and SLAM as an
internal benchmark feature. Reviewer expectations differ accordingly.

**Why:** Production users almost never consume cuVSLAM's SLAM output
directly. Practical robotics applications need access to the map, semantic
labelling, custom serialization, and integration with planners — so users
plug in their own mapping stack and use cuVSLAM only as a visual odometry
source. cuVSLAM's SLAM implementation exists primarily to pass academic loop-
closure benchmarks (KITTI, EuRoC, …) with a result competitive with
state-of-the-art.

**How to apply this:**

- An odometry change that affects accuracy is a high-bar review. Run the
  reporter on all datasets, both VO and VIO modes, before requesting review.
- A SLAM-only change that does not touch odometry is a lower-bar review —
  benchmark numbers must not regress, but production impact is bounded.
- When in doubt about scope, ask: "would a user with their own SLAM stack
  notice this?" If yes, treat it as odometry-class.

---

## 6. Construction-time config vs internal runtime tuning

Settings fall into three categories:

| Category | Example | Where it lives | Stability |
|---|---|---|---|
| **Construction-time configuration** | GPU on/off, odometry mode, data export | `Odometry::Config`, passed to the constructor | Public API |
| **Per-frame internal tuning** | Feature count, border sizes, keyframe threshold | `internal::Internals`, passed to `Track()`, never stored | Unstable expert/development API |
| **Persistent internal tuning** | SBA window and solver parameters | `internal::InternalParameter`, passed to `ApplyPersistentInternalParameters()` | Internal use only |

If a normal user must choose a value at startup, it belongs in `Config`. If a low-level
development tool must vary a solver value per frame, it may belong in `Internals`. Values
that intentionally persist after tracker construction use
`ApplyPersistentInternalParameters()`.

Do not expose a user-facing feature through `Internals` merely because it is convenient.
Design a stable public API for that feature instead.

---

## 7. Memory backends are pluggable through the image manager

**Rule:** New memory backends (different allocator kinds, different devices)
plug in through `sof/image_manager.{h,cpp}` and `sof/image_context.{h,cpp}`,
not through scattered allocation sites.

**Why:** `ImageManager` is a mutex-guarded pool of `ImageContext` objects.
`init()` fixes the image shape, the pool size and the `use_gpu` choice; call
sites then take what they need with `acquire()` / `acquire_with_depth()` and
hold an `ImageContextPtr`. The allocation itself is done by `ImageContext` and
the `cuda::` image and pyramid types it owns, not by the manager. Because call
sites neither allocate image memory nor branch on where it came from, the pool
plus the context is the extension point: another CUDA allocator kind (pinned
host memory, unified memory) means changing how a context is built and pooled,
not a global refactor.

A genuinely non-CUDA accelerator is a bigger job and should not be read into
the rule above. `ImageContext` exposes a CUDA-typed interface — `gpu_image()`,
`build_gpu_image_pyramid(..., cudaStream_t)`, `cuda::GaussianGPUImagePyramid`
— and the SOF, descriptor and RGB-D paths consume those types directly. Such a
backend needs a backend-neutral interface and changes in every consumer of it,
on top of adapting context construction and pooling.

**What to avoid:**

- Do not introduce a parallel image pool or allocation path in another
  library.
- Do not branch on `cudaPointerGetAttributes(...)` at call sites; the
  `ImageContext` you acquired already knows which backend it is on.
