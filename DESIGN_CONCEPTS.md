# cuVSLAM Design Concepts

This document captures architectural decisions and design principles for cuVSLAM.
Follow these when making code changes or designing new features.

---

## 1. Per-frame values are hints; everything else is configuration

**Rule:** A value that genuinely differs from one frame to the next is passed to
`Odometry::Track()` as a `TrackHints` field and never stored. A value that holds for a run is
configuration: `Odometry::Config` when it is needed at construction, otherwise a named parameter
set through `SetParameter()`. Do not add a setter to change one frame's behaviour.

**Why:** A setter used for per-frame variation creates implicit shared state between frames. The
change silently persists into the next frame, which makes behaviour hard to reason about, test and
reproduce. The distinction is not "setters are bad" -- `SetParameter()` is a setter -- it is that
the lifetime of a value must match the way it is supplied.

In practice almost nothing is per-frame. When this was measured, exactly one value in the whole
solver configuration was ever varied between frames by any caller in the repository: the keyframe
decision. Everything else was set once and left alone, which is why `TrackHints` has one field and
the rest is configuration.

**How it works in cuVSLAM:**

`Odometry::Impl` holds one `odom::TrackPerFrameSettings`, seeded at construction from the settings
the components were built with, and a `params::Registry` bound to it. `Track()` copies that struct
per frame and applies any hints to the copy, so a parameter change takes effect on the next frame
and nothing becomes shared mutable state mid-frame.

```cpp
// A value that holds for the run: set it once, by name.
odometry.SetParameter("sof.num_desired_tracks", "200");

// A value that differs for this frame only.
cuvslam::Odometry::TrackHints hints;
hints.override_keyframe = true;
odometry.Track(images, {}, {}, &hints);

// The next call keeps the parameter and drops the hint.
odometry.Track(images);
```

**What to avoid:**

```cpp
// BAD - a setter used to change a single frame; the change bleeds into the next one.
odometry.SetNumDesiredTracks(200);
odometry.Track(images);

// BAD - a run-long value smuggled in as a per-frame hint, so every call has to repeat it.
hints.num_desired_tracks = 200;
```

**Where to add a new parameter:**

1. Confirm it is expert or development tuning rather than a normal user feature. Stable
   user-facing behaviour belongs in `Odometry::Config` or another public API.
2. Add the field, with its default, to the settings struct that consumes it (`sof::Settings`,
   `pnp::PNPSettings`, and so on).
3. Describe it next to that struct with `CUVSLAM_PARAM`; see `libs/params/README.md`. That one
   line is the whole cost -- the file loader, `-P` override, report dump, help text and bounds
   check all follow from it, and nothing else in the codebase has to learn the name.
4. If the struct is new, register it under a key prefix in
   `Odometry::Impl::RegisterSolverParameters()`, and only for the modes that read it.

**Where to add a new per-frame hint:** add a field to `Odometry::TrackHints` and apply it to the
per-frame copy in `Odometry::Track()`. Do this only when the value truly differs frame to frame; a
value that holds for a run belongs in step 3 above.

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

`Odometry::Track()` accepts a nullable `TrackHints` pointer. A null pointer means "no hints", and
the per-frame copy of the stored settings is used unchanged; lower layers receive a concrete
`TrackPerFrameSettings` and never learn whether the caller supplied anything.

`TrackHints::override_keyframe` is an intentional tri-state exception: unset leaves the decision to
the tracker, `true` forces a keyframe, and `false` prevents one. Resolve such values at the first
layer with enough context, rather than propagating optionality farther down the call stack.

```text
TrackHints* (null, or hints for this frame)
    └─► ignored when null
        └─► a copy of the stored TrackPerFrameSettings, with any hints applied
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

## 4. Construction-time config vs internal runtime tuning

Settings fall into three categories:

| Category | Example | Where it lives | Stability |
|---|---|---|---|
| **Construction-time configuration** | GPU on/off, odometry mode, data export | `Odometry::Config`, passed to the constructor | Public API |
| **Internal tuning** | Feature count, border sizes, SBA window, solver thresholds | Named parameters, set with `SetParameter()` or `LoadParameters()` | Unstable, internal use only |
| **Per-frame hints** | The keyframe decision for one frame | `TrackHints`, passed to `Track()`, never stored | Unstable, internal use only |

If a normal user must choose a value at startup, it belongs in `Config`. If a value is tuning that
holds for a run, it belongs in the parameter registry. Only a value that genuinely differs between
frames belongs in `TrackHints`.

Do not expose a user-facing feature as an internal parameter merely because it is convenient.
Design a stable public API for that feature instead.
