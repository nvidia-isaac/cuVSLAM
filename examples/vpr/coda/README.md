# Tutorial: Visual Global Localization on the CODa Dataset

This tutorial maps a route of the CODa dataset once, then finds the camera in that map from single frames of a later
run, without an initial pose, using [visual place recognition](../../../README.md#visual-global-localization). It takes
three steps:

1. Fit a place recognition vocabulary to the images of sequence 0
2. Map sequence 0 with that vocabulary and save the map
3. Every 5 seconds of sequence 5, which retraces the route, localize its frame in the saved map without a guess pose

## Set Up

Install PyCuVSLAM and the example dependencies as the [examples README](../../README.md#prerequisites) describes;
place recognition also needs TensorRT ([root README](../../../README.md#install-from-wheels)). Then build the AnyLoc
engine for your GPU once:

```bash
cd examples/vpr/coda
python3 -c "import cuvslam; cuvslam.build_anyloc_engine('anyloc.engine')"
```

The scripts read and write `anyloc.engine`, `vpr.vocabulary` and `map/` in this folder.

## Get the Data

1. Register at the [Texas Dataverse record](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV),
   accept the dataset license, and download `0.zip` and `5.zip` (87 GB) into `~/datasets/coda/raw`
2. From the repository root, convert them into rectified stereo images in `~/datasets/coda/0` and
   `~/datasets/coda/5` (61 GB); the converter needs only Python 3.10 or later:
   ```bash
   PYTHONPATH=tools/python_tools python3 -m cuvslam_tools.dataset_preparation.coda.prepare \
       --raw-dir ~/datasets/coda/raw --output-dir ~/datasets --sequences 0 5
   ```

The scripts read the sequences from `~/datasets/coda`; set `CODA_DIR` in `vpr_utils.py` if they are elsewhere.

## Step 1: Build a Vocabulary

A vocabulary fitted to images of your operating domain makes every map of it, and every localization in it, describe
places the same way. Fit it to the left images of sequence 0:

```bash
python3 build_vocabulary.py
```

Fitting holds every image in memory: the 8213 images take 19 GiB, and 7.5 minutes on an RTX 4090 desktop.

## Step 2: Map the Route

Map sequence 0 with place recognition and that vocabulary, in about 7 minutes:

```bash
python3 make_map.py
```

The map is saved to `map/`, replacing an earlier one.

## Step 3: Recognize and Localize without a Guess Pose

Every 5 seconds of sequence 5, `localize.py` asks the saved map where the frame was taken, in two ways, and prints
both:

- Place recognition (`recognize_place_by_frame`) names the mapped frame that looks most like it, by its time in the
  mapping run, and how alike they look. It searches a map in memory, which a session gets from its first successful
  localization, so it starts with the try after that one.
- Localization (`localize_in_map`) also verifies the place against the map's landmarks and prints the rig position in
  the map, in meters, or why it failed. Each try runs in a new session that tracks nothing, so it starts like a robot
  switched on, or picked up and put down, somewhere in the mapped area.

```bash
python3 localize.py
```

It runs for about 6 minutes and prints lines such as:

```
   40.0 s: recognized the place mapped at 45.2 s (score 0.76); localized at -31.06, 1.59, -1.88
   45.0 s: recognized the place mapped at 52.3 s (score 0.80); localized at -34.23, 1.99, -4.70
   50.0 s: recognized the place mapped at 57.3 s (score 0.79); not localized: Can't localize in map using provided image and guess
   55.0 s: recognized the place mapped at 61.0 s (score 0.76); localized at -37.76, 4.01, -13.14
   60.0 s: recognized the place mapped at 67.4 s (score 0.82); localized at -39.10, 4.84, -17.68
```

At the end it counts the tries that succeeded; without ground truth, it can't tell a right place from a wrong one:

```
localized 38 of 145 tries, recognized 131 of 144
```
