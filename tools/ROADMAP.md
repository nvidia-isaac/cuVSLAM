# Development Pipeline Roadmap

## Legacy Pipeline (Obsolete)

The old development pipeline consisted of the following components:

* `tools/tracker` — C++ tool with access to internal private classes
* `tools/reporter` — C++ and Python tool for generating PDF and HTML reports
* `libs/launcher` — C++ library
* GFlags library

The workflow for new feature development was as follows: internal parameters were first
exposed through GFlags, and then the reporter CLI was run on different datasets with varying
GFlags parameter values. Once sufficient intuition was built and the appropriate parameter
values (or auto-tuning algorithms) were identified to satisfy all metrics across all test
datasets, the parameters were hidden from the public interface.

The reporter and tracker also provided options for:

* Real-time performance benchmarking (FPS simulation)
* Shuttle/repeat mode for stability testing
* Black-frame oscillation for IMU testing
* And more

The tracker also served as the primary tool for C++ debugging.


## New Design (Transition In Progress)

### Removals

1. `tools/tracker`
2. `tools/reporter`
3. `libs/launcher`
4. GFlags

### Replacements

**`tools/cuvslam_api_launcher`** — A C++ tool built on top of the public API. It accepts
EDEX image sequences and exposes repeat, shuttle, blackout, performance-throttling, and
development-only internal configuration needed by the reporter's C++ backend. Its primary
purposes are:

1. Real-time performance benchmarking (FPS simulation)
2. C++ debugging
3. C++ API unit testing

**`tools/python_tools`** — A set of pure Python packages for tracking a single sequence,
a dataset, or multiple datasets. It uses only the public `pycuvslam` Python API. All rich
benchmarking functionality lives here, including dataset reading and format conversion
(PNG, MP4, etc.), ground truth evaluation, PDF/HTML report generation, CI validation, and
metric export to public challenge formats. Additional modes include shuttle mode,
black-frame oscillation, and more.

Sub-tools:

1. `prepare_kitti` — `python_tools/cuvslam_tools/dataset_preparation/kitti`
2. `prepare_euroc` — `python_tools/cuvslam_tools/dataset_preparation/euroc`
3. `prepare_tartan` — `python_tools/cuvslam_tools/dataset_preparation/tartan`
4. `prepare_tum` — `python_tools/cuvslam_tools/dataset_preparation/tum`
5. `cuvslam_tracker` — `python_tools/cuvslam_tools/tracker`
6. `cuvslam_reporter` — `python_tools/cuvslam_tools/reporter`
7. `cuvslam_validator` — `python_tools/cuvslam_tools/datasets_validator`
8. `rosbag_extract_edex`, `rosbag_extract_images`, `rosbag_extract_urdf`, `rosbag_extract_videos` —
   `python_tools/cuvslam_tools/bag2edex`
9. `undistort_edex_images` — `python_tools/cuvslam_tools/undistort`


## Use Cases

### 1. Debugging a sequence that fails in python_tools/cuvslam_tools/reporter

- Run the single sequence in `python_tools/cuvslam_tools/tracker`.
- If the issue persists, use `cuvslam::debug_dump_directory` to dump all shuttle/MP4/etc.
  data to a simple EDEX format for `cuvslam_api_launcher`.
- Run `tools/cuvslam_api_launcher` over the exported folder to obtain a clean C++ debugging
  environment.

### 2. Developing a new feature and evaluating a parameter's effect on datasets

- Expose the development setting as a `cuvslam_api_launcher` gflag.
- Pass it through with `cuvslam_reporter --tracker_backend api_launcher ... -- --expert_<setting>=<value>`.
- Keep the Python backend for normal public-API evaluation; unstable internal parameters
  are intentionally unavailable there.

### 3. Measuring GPU load on a target device (e.g., Jetson Orin Nano at 60 fps)

- Record your data using ROS, the ZED SDK, or another capture method.
- Convert the recording to a simple edex + TGA sequence (use `debug_dump_directory` in any
  environment).
- Run `cuvslam_api_launcher` in async mode with FPS simulation enabled.
