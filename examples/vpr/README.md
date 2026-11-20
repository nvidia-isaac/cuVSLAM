# Tutorial: Visual Global Localization on a RealSense Camera

This tutorial maps an area once, then finds the camera in that map from a single frame, without an initial pose, using
[visual place recognition](../../README.md#visual-global-localization). It takes three steps:

1. Record images of the area and fit a place recognition vocabulary to them
2. Map the area with that vocabulary and save the map
3. Every 5 seconds, localize the current frame in the saved map without a guess pose

Each step runs on a live RealSense stereo camera, or on recordings of the CODa dataset if you
[have no camera](#without-a-camera-coda).

## Set Up

From the repository root, build and start the RealSense [Docker](../../docker/README.md) container:

```bash
docker build -f docker/Dockerfile.realsense-cu12 -t pycuvslam:realsense-cu12 .
./docker/run_docker.sh
```

In the container, go to `examples/vpr` and build the AnyLoc engine for your GPU once:

```bash
cd examples/vpr
python3 -c "import cuvslam; cuvslam.build_anyloc_engine('anyloc.engine')"
```

The scripts read and write `anyloc.engine`, `vpr.vocabulary`, `images/` and `map/` in this folder.

## Without a Camera: CODa

Sequence 0 of the CODa dataset stands in for the camera in steps 1 and 2, and sequence 5, which retraces its route, in
step 3. Their frames are replayed at the recorded 10 fps, as a camera delivers them.

1. Register at the [Texas Dataverse record](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV),
   accept the dataset license, and download `0.zip` and `5.zip` (87 GB) into `~/datasets/coda/raw`
2. On the host, from the repository root, convert them into rectified stereo images in `~/datasets/coda/0` and
   `~/datasets/coda/5` (61 GB); the converter needs only Python 3.10 or later:
   ```bash
   PYTHONPATH=tools/python_tools python3 -m cuvslam_tools.dataset_preparation.coda.prepare \
       --raw-dir ~/datasets/coda/raw --output-dir ~/datasets --sequences 0 5
   ```

`docker/run_docker.sh` mounts `~/datasets` into the container at the same absolute path, which it prints as
`Datasets directory`; the commands below call it `<datasets>`.

## Step 1: Build a Vocabulary

A vocabulary fitted to images of your operating domain makes every map of it, and every localization in it, describe
places the same way. With the camera, walk through the area while `record_images.py` saves 3 left images per second to
`images/`, press Ctrl+C, then fit the vocabulary to all of them:

```bash
python3 record_images.py
python3 build_vocabulary.py images
```

With CODa, fit it to the left images of sequence 0:

```bash
python3 build_vocabulary.py <datasets>/coda/0/00
```

Fitting holds every image in memory: the 8213 images of CODa sequence 0 take 19 GiB, and 7 minutes on an RTX 4090
desktop.

## Step 2: Map the Area

Map the area with place recognition and that vocabulary. With the camera, walk through the area, then press Ctrl+C;
CODa sequence 0 maps in about 14 minutes:

```bash
python3 make_map.py                     # RealSense
python3 make_map.py <datasets>/coda/0   # CODa
```

The map is saved to `map/`, replacing an earlier one.

## Step 3: Recognize and Localize without a Guess Pose

Every 5 seconds, `localize.py` asks the saved map where the current frame was taken, in two ways, and prints both:

- Place recognition (`recognize_place_by_frame`) names the mapped frame that looks most like it, by its time in the
  mapping run, and how alike they look. It searches a map in memory, which a session gets from its first successful
  localization, so it starts with the try after that one.
- Localization (`localize_in_map`) also verifies the place against the map's landmarks and prints the rig position in
  the map, in meters, or why it failed. Each try runs in a new session that tracks nothing, so it starts like a robot
  switched on, or picked up and put down, somewhere in the mapped area.

```bash
python3 localize.py                     # RealSense
python3 localize.py <datasets>/coda/5   # CODa
```

On CODa sequence 5 it runs for about 12 minutes and prints lines such as:

```
   40.0 s: recognized the place mapped at 45.2 s (score 0.76); localized at -30.52, 1.55, -2.04
   45.0 s: recognized the place mapped at 52.3 s (score 0.80); localized at -33.99, 2.28, -4.64
   50.0 s: recognized the place mapped at 57.3 s (score 0.79); not localized: Can't localize in map using provided image and guess
   55.0 s: recognized the place mapped at 61.0 s (score 0.76); localized at -37.76, 4.01, -13.14
   60.0 s: recognized the place mapped at 67.4 s (score 0.82); localized at -39.12, 4.74, -17.74
```
