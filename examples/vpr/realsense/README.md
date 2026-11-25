# Tutorial: Visual Global Localization on a RealSense Camera

This tutorial maps an area once with a live RealSense stereo camera, then finds the camera in that map from a single
frame, without an initial pose, using [visual place recognition](../../../README.md#visual-global-localization). It
takes three steps:

1. Record images of the area and fit a place recognition vocabulary to them
2. Map the area with that vocabulary and save the map
3. Every 5 seconds, localize the current frame in the saved map without a guess pose

## Set Up

From the repository root, build and start the RealSense [Docker](../../../docker/README.md) container:

```bash
docker build -f docker/Dockerfile.realsense-cu12 -t pycuvslam:realsense-cu12 .
./docker/run_docker.sh
```

In the container, go to `examples/vpr/realsense` and build the AnyLoc engine for your GPU once:

```bash
cd examples/vpr/realsense
python3 -c "import cuvslam; cuvslam.build_anyloc_engine('anyloc.engine')"
```

The scripts read and write `anyloc.engine`, `vpr.vocabulary`, `images/` and `map/` in this folder.

## Step 1: Build a Vocabulary

A vocabulary fitted to images of your operating domain makes every map of it, and every localization in it, describe
places the same way. Walk through the area while `record_images.py` saves 3 left images per second to `images/`, press
Ctrl+C, then fit the vocabulary to all of them:

```bash
python3 record_images.py
python3 build_vocabulary.py
```

Fitting holds every image in memory, about 2.4 MB per image.

## Step 2: Map the Area

Walk through the area again to map it with place recognition and that vocabulary, then press Ctrl+C:

```bash
python3 make_map.py
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
python3 localize.py
```

Each try prints one line: its time, the recognized place, then the position or why localization failed. Press Ctrl+C
to stop; it then counts the tries that localized and those that were recognized.
