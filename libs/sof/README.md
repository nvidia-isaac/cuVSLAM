# Sparse Optical Flow (SOF)

Feature selection and Lucas-Kanade (LK) tracking — the front end of cuVSLAM's
odometry. This README is for maintainers who need to change anything in this
library. If you only call SOF from above, you do not need to read this.

## Implementation invariants

The LK tracker here is not a textbook Lucas-Kanade. The settings and behaviors
below are all load-bearing — they were tuned together across automotive, drone,
warehouse, indoor, and street datasets. Changing any one of them in isolation
will improve one ODD and break another.

| Invariant | Value / behavior |
|---|---|
| Patch size | 7 × 7 |
| Image pyramid | Integer (`image_pyramid_u8`) |
| Gradient pyramid | Integer (`gradient_pyramid`) |
| Per-level scoring | Normalized cross-correlation (NCC) on every level |
| Level fallback | If a level fails to converge, the tracker drops to the next level rather than killing the feature outright |
| Patch padding | Zero-padded at the borders |
| Patch normalization | Mean subtracted from the patch before matching |
| Convergence check | Custom — see `lk_tracker.cpp` |

The integer pyramid + integer gradient predate the float path: they were
introduced for the original Jetson Nano port and stayed because they hold up on
all current targets. Do not convert to float "for clarity" — you will get a
different tracker.

The NCC + per-level fallback + mean subtraction combination is the part most
often missing from third-party LK implementations. If you are integrating an
external LK kernel as an alternative backend, it must replicate every invariant
in the table above to match accuracy — the 7 × 7 patch, both integer pyramids,
the zero-padded borders, and the custom convergence check just as much as NCC,
per-level fallback, and mean subtraction.

## What lives where

- `lk_tracker.{h,cpp}` — the LK iteration itself.
- `klt_tracker.{h,cpp}` — KLT wrapper used by the multi-camera path.
- `image_pyramid_u8.{h,cpp}`, `gradient_pyramid.{h,cpp}` — the integer pyramids.
- `gftt.{h,cpp}` — Shi-Tomasi (good features to track) selection.
- `selector_mono.{h,cpp}`, `selector_stereo.{h,cpp}` — pick which features to
  track this frame.
- `image_manager.{h,cpp}` — a mutex-guarded pool of `ImageContext` objects
  handed out by `acquire()` / `acquire_with_depth()`. The buffers and pyramids
  are allocated and owned by `ImageContext` itself; the manager only fixes the
  shape, the pool size and the `use_gpu` choice at `init()`. Together with
  `image_context.{h,cpp}` this is the point of extension when adding a new
  memory backend — see [DESIGN_CONCEPTS.md](../../DESIGN_CONCEPTS.md).
- `sof_mono_cpu.cpp` / `sof_mono_gpu.cpp` and
  `sof_multicamera_cpu.cpp` / `sof_multicamera_gpu.cpp` — CPU and GPU paths.

Both CPU and GPU paths are kept numerically equivalent on purpose: regressions
are diagnosed by toggling the backend and comparing. Equivalence is not exact
bit-for-bit and no test asserts it — `sof_l2r_test.cpp` checks a minimum
tracked-point count per backend rather than comparing the two, with the bound
set below the measured count to absorb GPU/driver jitter.

## Touching the LK tuning

If you genuinely need to change any entry in that table — a constant (patch
size, level count, convergence threshold, NCC threshold, gradient threshold, …)
or a behavior (level fallback, patch padding, mean subtraction, the convergence
check):

1. Run the `tools/cuvslam_app` reporter both with and without the change, on
   every dataset enabled in `scripts/run_eval.sh` (KITTI and EuRoC today —
   the other entries are commented out). See
   [DEVELOPMENT.md — Accuracy regression workflow](../../DEVELOPMENT.md#accuracy-regression-workflow-reporter).
2. Compare the resulting PDFs page by page. A constant that improves one
   dataset by 0.5% but breaks another by 5% is a regression, not a win.
3. Apply the drift-interpretation rule: < 2 % drift = the number is
   trustworthy; 2 % – 20 % = marginal, sanity-check the trajectory plot before
   drawing any conclusion from it; > 20 % = the trajectory is broken and the
   number is meaningless.
