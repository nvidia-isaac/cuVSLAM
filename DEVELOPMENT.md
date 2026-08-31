# Development

## Development workflow

### Accuracy regression workflow (reporter)

The reporter is the primary tool for evaluating any change that may affect
tracking accuracy. The development loop is:

1. Implement the change behind a build flag, CLI flag, or branch.
2. Run `tools/cuvslam_app` (Python reporter) — or the legacy C++ reporter —
   once with the flag off and once with the flag on, over the datasets enabled
   in `scripts/run_eval.sh`. Only KITTI and EuRoC are enabled today; TartanAir,
   M3ED-Spot, TUM-RGBD, AR-table and ICL-NUIM are listed but commented out, so
   a genuinely full-dataset run means uncommenting them and provisioning the
   data first. Each run produces a PDF with one page per sequence (trajectory +
   ground truth + green dots for loop closures + per-sequence metrics).
3. Open the two PDFs side-by-side and compare visually.

Visual inspection matters because good metrics do not always mean good
tracking, and a single bad sequence can be averaged out of a summary number.
Use the **drift-interpretation rule** when reading per-sequence numbers:

| Drift | What it means |
|---|---|
| < 2 % | Numeric value is trustworthy — comparisons like 1.5 % vs 1.8 % are real. |
| 2 % – 20 % | Marginal — sanity-check the trajectory plot before drawing conclusions. |
| > 20 % | The trajectory is broken or random. Do not compare 10 % vs 20 %; both are wrong. |

A regression is anything that improves one dataset and breaks another. For
multi-ODD changes (e.g. LK or feature-selector tuning), the side-by-side PDFs
are the only reliable signal — never rely on the aggregate.

### Performance benchmarking (manual)

There is **no automated performance CI**. Frame-rate and latency are checked
manually, roughly once a month, by running the tracker under NVIDIA Nsight
Systems with NVTX labels enabled:

```bash
# Configure with NVTX
cmake -S . -B build -DUSE_NVTX=ON
cmake --build build --parallel $(nproc)

# Run the tracker under Nsight with "Collect NVTX Trace" enabled
# (see libs/profiler/README.md for the NVTX API the code uses).
```

The cuVSLAM code base has NVTX ranges around every main pipeline stage
(feature selection, LK tracking, triangulation, SBA, SLAM message handling).
In the Nsight timeline you will see them as named, colored bands.

Reference loads on a healthy system (stereo, 60 FPS, VGA, blocking mode):
- GPU: ≈ 5 %
- CPU: ≈ 5 %
- Pure-CPU mode on a Jetson Orin: < 1 core.

Treat these as order-of-magnitude expectations, not a calibrated benchmark:
the host GPU/CPU model, the Jetson power mode and the utilization-measurement
method they were taken with are not recorded, so they are a smell test rather
than a number to reproduce. Compare against your own baseline captured on the
same hardware, power mode, tracker mode and sequence. A result several times
higher — not a few percent — is what points at misconfiguration before it
points at tuning.

### Tracker execution modes

`tools/cuvslam_api_launcher` has two feed rates, intended for different
purposes (`tools/tracker` has no equivalent flag):

| Mode | Selector | Use it for |
|---|---|---|
| Free-running (default) | no flag | Maximum-quality reference trajectory. Frames are fed as fast as they can be read. |
| FPS simulation | `--max_fps <hz>` | Real-time benchmarking. Use this to measure whether the system keeps up at a target hardware FPS; SBA and SLAM run as fast as they can and may skip work. |

`--max_fps` only throttles how quickly frames are fed to the tracker — it
changes no threading setting. Blocking execution is a library config, not a
tool mode: set `Odometry::Config::async_sba = false` and
`Slam::Config::sync_mode = true` to pull SBA and SLAM onto the main thread.
Neither of those is the default, and no tool flag sets them.

**IO warm-up trick.** The first run of a sequence pays the disk-read cost
(TGA files page in). The second run reads from the OS page cache, so what you
measure is tracking with cached input — read and decode are still in there,
just not the disk. Always run twice when benchmarking — discard the first.

### CI cadence

Two levels, with different cadences:

| Level | Cadence | What runs |
|---|---|---|
| Unit tests (ctest in `libs/*/test/`, `python3 -m unittest` in `python/test/`) | Per commit | Fast — minutes. |
| Reporter integration tests | Nightly | Every dataset enabled in `scripts/run_eval.sh` (KITTI and EuRoC today), full PDF output. Catches end-to-end regressions that unit tests miss. |

Before opening a non-trivial MR, run the reporter locally on at least KITTI
and EuRoC. The full nightly run will catch what you missed, but it is faster
to find regressions before you push.

## Code Style

cuVSLAM uses [Google C++ Code Style](https://google.github.io/styleguide/cppguide.html)
with two exceptions (compared to `clang-format` preset):

1. Line width: 120
2. No space before `public/private/protected` access specifiers

## Pre-commit hooks

Install pre-commit framework to manage git hooks:

```bash
pipx install pre-commit
```
(`apt install pre-commit` version can be too old)

Update `.git/hooks/pre-commit` in the repo root folder (must be done after each `git clone`):

```bash
pre-commit install
```
Git hooks will reformat C++ code using `.clang-format`, fix minor format issues, add copyright headers to source files.

### Troubleshooting

1. Pre-commit gives error on Ubuntu 22.04:

```
AssertionError: BUG: expected environment for python to be healthy() immediately after install, please open an issue describing your environment
```

[To fix this](https://stackoverflow.com/a/73698579/23690993), add this line to your .bashrc file:
`export SETUPTOOLS_USE_DISTUTILS=stdlib`
Refresh the configuration file by restarting a terminal window or running `source ~/.bashrc`.

### To skip pre-commit checks run

`git commit --no-verify`.

### Run reformat in CLion

https://www.jetbrains.com/help/clion/clangformat-as-alternative-formatter.html

### Run reformat in VSCode

Install `The C/C++ extension for Visual Studio Code`
https://code.visualstudio.com/docs/cpp/cpp-ide#_code-formatting

## Sandbox/offline external sources

On a machine with internet access, run this from the repository root:

```bash
./fetch_external_sources.sh
```

The script runs a CMake configure step and copies downloaded `FetchContent` sources to `ext_src/`.
Copy `ext_src/` to the sandbox/offline machine, then configure with:

```bash
cmake -S . -B build -C cmake/use_offline_externals.cmake
```

Then build normally:

```bash
cmake --build build
```
