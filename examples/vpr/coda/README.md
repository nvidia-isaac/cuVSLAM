# Tutorial: Visual Global Localization on the CODa Dataset

This tutorial maps a route of the CODa dataset once, then finds the camera in that map from single frames of a later
run, without a guess pose, using [visual place recognition](../../../README.md#visual-global-localization). It takes
three steps:

1. Fit a place recognition vocabulary to the images of sequence 0
2. Map sequence 0 with that vocabulary and save the map
3. Every 5 seconds of sequence 5, which retraces the route, localize its frame in the saved map without a guess pose

## Get the Data

1. Register at the [Texas Dataverse record](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV),
   accept the dataset license, and download `0.zip` and `5.zip` (87 GB) into `~/datasets/coda/raw`
2. From the repository root, convert them into rectified stereo images and ground truth poses in `~/datasets/coda/0`
   and `~/datasets/coda/5` (61 GB); the converter needs only Python 3.10 or later:
   ```bash
   PYTHONPATH=tools/python_tools python3 -m cuvslam_tools.dataset_preparation.coda.prepare \
       --raw-dir ~/datasets/coda/raw --output-dir ~/datasets --sequences 0 5
   ```

The scripts read the sequences from `~/datasets/coda`; set `CODA_DIR` in `vpr_utils.py` if they are elsewhere.

## Set Up

Install PyCuVSLAM and the example dependencies as the [examples README](../../README.md#prerequisites) describes;
place recognition also needs TensorRT ([root README](../../../README.md#install-from-wheels)), and step 1 needs 19 GiB
of RAM. Then build the AnyLoc engine for your GPU once:

```bash
cd examples/vpr/coda
python3 -c "import cuvslam; cuvslam.build_anyloc_engine('anyloc.engine')"
```

The scripts read and write `anyloc.engine`, `vpr.vocabulary` and `map/` in this folder.

## Step 1: Build a Vocabulary

A vocabulary fitted to images of your operating domain makes every map of it, and every localization in it, describe
places the same way. Fit it to the left images of sequence 0:

```bash
python3 build_vocabulary.py
```

Fitting holds every image in memory and takes 7.5 minutes on an RTX 4090 desktop.

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

Ground truth (`gt_global.txt`) checks both. A recognized place is right when it was mapped within 10 m of where the
frame was taken, as `cuvslam_vpr_reporter` counts it. A localization next to a right place shows how far it is from
where ground truth puts the frame: the offset from that place comes from ground truth and the place's pose from the
map, so the map's own drift doesn't count.

```bash
python3 localize.py
```

It runs for about 6 minutes and prints lines such as:

```
   40.0 s: recognized the place mapped at 45.2 s, 0.8 m away (score 0.76); localized at -30.19, 1.74, -2.31 (0.17 m off)
   45.0 s: recognized the place mapped at 52.3 s, 1.7 m away (score 0.80); localized at -34.00, 2.29, -4.66 (0.15 m off)
   50.0 s: recognized the place mapped at 57.3 s, 1.9 m away (score 0.79); not localized: Can't localize in map using provided image and guess
   55.0 s: recognized the place mapped at 61.0 s, 0.1 m away (score 0.76); localized at -37.76, 4.01, -13.14 (0.17 m off)
   60.0 s: recognized the place mapped at 67.4 s, 1.1 m away (score 0.82); localized at -39.09, 4.84, -17.68 (0.20 m off)
```

It ends with how many tries succeeded and how they compare with ground truth:

```
Summary: recognized 131 of 144 tries (91.0 %), localized 38 of 145 (26.2 %)
Ground truth: 129 of 131 recognized places within 10 m (98.5 %); 37 localizations checked: median 0.36 m, max 2.90 m off
```

Localization verifies each recognized place against the map's landmarks and rejects it when too few match, with
`Can't localize in map using provided image and guess` even without a guess; a robot tries again on a later frame.
