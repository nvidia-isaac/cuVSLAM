# cuVSLAM Known Limitations

This page catalogues inherent limitations of the cuVSLAM tracking engine and
the workarounds available through its API or configuration.  Understanding
these upfront helps avoid spending time debugging issues that are environmental
or scene-driven rather than a configuration mistake.

---

## Feature detector: corners only, no line features

cuVSLAM uses **Good Features To Track (GFTT)** — a Harris-corner-based detector
— followed by Lucas-Kanade or KLT optical-flow tracking (`TrackerType::LK` /
`TrackerType::KLT` in `sof.h`).  Feature matching uses normalized cross-correlation
(NCC) over small image patches.

**Consequences:**

- Environments dominated by **line features** (corridors, office walls, structured
  environments with edges but few corners) yield very few GFTT detections.
  Tracking degrades or fails even though a human eye would consider the scene
  well-textured.
- **Repetitive patterns** (brick walls, floor tiles, wire fences) produce many
  candidate corners but cause incorrect NCC matches, leading to scale drift or
  jumps.
- **Low-texture or featureless surfaces** (smooth floors, white walls, sky) give
  no detections at all — the camera will lose tracking if these fill the frame.

**Workaround:** Ensure cameras are angled to keep richly-textured, non-repetitive
scene content visible.  If only line-dominated scenes are available, mask out
line-dominated regions with border parameters (`border_top/bottom/left/right`) or
per-frame masks to limit feature selection to the corners that do exist.

---

## Dynamic objects

cuVSLAM has **no built-in dynamic object detector**.  Moving objects (people,
vehicles, other robots) generate features that violate the static-world
assumption.  SBA provides some outlier rejection through multi-view geometry,
but it is not designed to handle large dynamic foreground regions — especially
when the dynamic object fills a significant portion of the frame.

**Workaround — per-frame mask API:**

Pass a `uint8` binary mask alongside each image in the `track()` call.  Pixels
set to `0` are excluded from feature detection in that frame.  Pixels set to
non-zero are active.

Python example:
```python
# Segment the dynamic object (e.g. using a detector) and build a mask
mask = (segmentation != dynamic_class_id).astype(np.uint8) * 255  # 0 = masked

pose, _ = tracker.track(
    timestamp,
    images_gpu,
    masks=[mask_left, mask_right],   # one per camera
)
```

A worked example with car segmentation masks is in
`examples/kitti/track_kitti_masks.py`.

Notes:
- Masks apply to **feature detection only** — they do not retroactively reject
  tracks that were initiated before the mask was applied.
- For slowly-moving or intermittent dynamic objects, update the mask every frame
  rather than applying it once.
- GPU masks are supported (`process_mask_gpu()`); prefer this path for real-time
  use.

---

## Reflective and specular surfaces

Reflective materials — **shrink-wrap packaging, polished floors, glass, metal
panels** — cause two distinct failure modes:

1. **Inconsistent feature appearance**: a corner on a reflective surface looks
   different from slightly different viewpoints (the specular highlight moves).
   NCC matching fails or produces incorrect correspondences.
2. **Spurious features**: bright specular spots are picked up as corners by GFTT
   and then immediately lost when the highlight shifts, producing a steady stream
   of short-lived, inaccurate tracks that pollute the SBA window.

There is no special reflective-surface handling in the engine.

**Workaround:** Mask out known reflective regions with border parameters or
per-frame masks.  If the environment is permanently reflective (e.g., a warehouse
floor), use static border cropping to keep the floor out of the feature region
entirely.  Adjusting camera exposure to reduce highlight saturation also helps.

---

## IR projector patterns in grayscale images

RealSense and similar active-stereo sensors emit an **IR dot or speckle pattern**
to aid depth estimation.  When using the IR (grayscale) image streams with the IR
projector on, this pattern is visible in the images as a dense field of bright
dots.

cuVSLAM has **no structured-light pattern removal**.  The dot pattern is treated
as texture, and GFTT will detect the dot grid as corners.  Because the pattern
is projected from a fixed emitter and does not move with the scene, the detected
"features" are not stable across viewpoints and cause tracking instability.

**Workaround:**

- **Preferred:** Switch to **RGB image streams** instead of IR grayscale.  The
  RGB sensor does not receive the IR pattern (it is filtered out in the optical
  path), so the images are clean.  cuVSLAM converts RGB to grayscale internally
  via `cast_rgb2gs_cpu()`.
- **Alternative:** Disable the IR projector if depth is not needed downstream.
  On RealSense this can be done via `rs2::pipeline` controls or `realsense-viewer`.
- **Alternative:** Enable `use_denoising` in the config — this attenuates
  high-frequency noise including mild dot patterns, but is not a substitute for
  a clean image stream.

---

## Simple motion model — poor fit for agile or unconstrained motions

cuVSLAM uses a **constant-velocity history model** (`PosePredictionModel` in
`pose_prediction.h`): it extrapolates the next pose by fitting motion from the
most recent pose history.  This is a reasonable prior for ground vehicles and
slow hand-held cameras, but breaks down for:

- **Drones and aerial platforms**: fast, non-planar accelerations, hovering (near-zero
  velocity with large vibration), aggressive manoeuvres.  The constant-velocity
  prior diverges quickly, and without a good initial-pose estimate, feature search
  windows miss the true location.
- **High-vibration platforms**: mechanical vibration aliases into perceived camera
  motion.
- **Sudden stops or direction reversals**: the predictor overshoots, placing the
  search window in the wrong region.

VIO mode (`OdometryMode.Inertial` / `tracking_mode: 1`) mitigates this by using
IMU preintegration for pose prediction — this is the recommended mode for drones
and any platform with non-trivial dynamics.

For **ground vehicles** specifically, `GroundIntegrator` can project the 3D
odometry estimate onto a ground plane (Y-up world frame, XZ plane), removing
spurious roll/pitch drift that accumulates when the vehicle is physically
constrained to move on flat ground.  Enable via the `use_ground_constraint`
parameter.

The maximum tolerated frame-to-frame pose change is bounded by
`max_frame_delta_s` — if a frame is dropped or the platform moves too fast, the
search window will not cover the true displacement and tracking will fail.

---

## Grayscale-only internal processing

All image data is converted to grayscale before feature detection regardless of
the input format.  RGB images are converted via `cast_rgb2gs_cpu()` (standard
luminance weighting).

**Consequence:** color information is discarded.  In scenes where objects are
distinguishable only by color (e.g., a uniformly-lit scene with two side-by-side
color patches but no luminance contrast), cuVSLAM cannot distinguish them.

**Tip:** For RGB cameras, if one channel is noticeably noisier (common in Bayer
sensors at the blue channel under artificial lighting), you can pre-select the
cleanest channel and pass it as a mono image rather than letting the automatic
conversion average across channels.
