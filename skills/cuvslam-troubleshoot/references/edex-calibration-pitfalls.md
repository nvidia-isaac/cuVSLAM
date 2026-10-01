# EDEX Calibration Pitfalls

Common mistakes in EDEX (`stereo.edex`) configuration that cause pose jumps or wrong-scale tracking, even when images look fine.

---

## 1. `"fisheye"` with zero distortion params ≠ `"pinhole"`

Setting `distortion_model: "fisheye"` with `distortion_params: [0, 0, 0, 0]` does **not** produce a standard perspective camera. cuVSLAM's fisheye model with zero coefficients computes the equidistant projection:

```
r_projected = f * atan(x/z)     ← equidistant fisheye (zero-param fisheye)
r_projected = f * (x/z)         ← perspective / pinhole
```

The divergence grows with angle from the optical axis and produces systematically wrong 3D structure, causing **pose jumps throughout the sequence** even when the images and all other parameters are correct.

**Symptom:** Large pose jumps (>3× mean frame displacement), wrong trajectory scale.

**Fix:** Use `distortion_model: "pinhole"` with `distortion_params: []` for any perspective/undistorted camera:

```json
"intrinsics": {
  "distortion_model": "pinhole",
  "distortion_params": [],
  "focal": [320, 320],
  "principal": [320, 240],
  "size": [640, 480]
}
```

Note: `pinhole` requires exactly 0 params, `fisheye` requires exactly 4.

---

## 2. Wrong baseline axis causes wrong scale

The `transform` field for each camera in an EDEX encodes `rig_from_camera` as a 3×4 matrix `[R | t]`. The translation column `t` is the camera's origin in the rig frame. Getting the **axis** wrong (e.g., Y instead of X for a horizontal stereo pair) causes cuVSLAM to look for disparity in the wrong direction:

- cuVSLAM expects disparity along the baseline axis for depth triangulation
- If the declared baseline axis has near-zero disparity in the actual images, depth estimates blow up
- Result: scale error of 10–100×, many pose jumps

**Quick diagnostic — measure actual parallax direction** with a center-patch cross-correlation:

```python
import numpy as np
from PIL import Image
from scipy.signal import correlate2d

left  = np.array(Image.open("left.png").convert("L"), dtype=float)
right = np.array(Image.open("right.png").convert("L"), dtype=float)

h, w = left.shape
p_l = left [h//2-50:h//2+50, w//2-50:w//2+50]
p_r = right[h//2-50:h//2+50, w//2-50:w//2+50]

corr = correlate2d(p_l - p_l.mean(), p_r - p_r.mean(), mode="full")
peak = np.unravel_index(corr.argmax(), corr.shape)
dy = peak[0] - corr.shape[0] // 2
dx = peak[1] - corr.shape[1] // 2
print(f"dx={dx}px  dy={dy}px → baseline axis: {'X (horizontal)' if abs(dx) > abs(dy) else 'Y (vertical)'}")
```

**Fix:** Ensure the translation axis in `rig_from_camera` matches the actual camera displacement direction. For a standard horizontal stereo pair with 0.25 m right-camera offset:

```json
✓  "transform": [[1,0,0,0.25],[0,1,0,0],[0,0,1,0]]   ← X-axis (correct)
✗  "transform": [[1,0,0,0],[0,1,0,0.25],[0,0,1,0]]   ← Y-axis (wrong for horizontal stereo)
```

---

## 3. Quantify jump severity from an output EDEX

Use this script to check jump count and scale ratio against ground truth after any tracker run:

```python
import json, numpy as np

data = json.load(open("stereo_out.edex"))
rp   = data[1]["rig_positions"]
keys = sorted(rp, key=int)
pos  = [rp[k]["translation"] for k in keys]
d    = [np.linalg.norm(np.array(pos[i+1]) - np.array(pos[i])) for i in range(len(pos)-1)]
mean = np.mean(d)
jumps = [(int(keys[i]), d[i]) for i in range(len(d)) if d[i] > 3 * mean]
print(f"mean={mean:.4f}m  jumps(>3×mean): {len(jumps)}")
# compare mean against GT mean displacement — should be ~1.0× for correct scale
```

Note: the output EDEX always stores only camera 0 intrinsics regardless of how many cameras were used — having one camera entry in the output is normal, not a sign that mono mode was used.
