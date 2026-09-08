#!/bin/bash
set -euo pipefail

CUVSLAM_BIN=~/cuvslam/build/release

CUVSLAM_MAP=/tmp/cuvslam_map
CUVSLAM_DATASETS=/home/$USER/datasets

source $CUVSLAM_BIN/cuvslam_vars.sh
CUVSLAM_LAUNCHER=$CUVSLAM_BIN/bin/cuvslam_api_launcher
KITTI_EDEX=$CUVSLAM_DATASETS/kitti/06/stereo.edex

# save map
$CUVSLAM_LAUNCHER -edex="$KITTI_EDEX" --cfg_enable_slam --cfg_enable_export \
  -Podometry.rectified_stereo_camera=true \
  -output_map=$CUVSLAM_MAP

for i in $(seq 1 100); do
  slow=$((RANDOM % 5001))
  frame=$((RANDOM % 501))

  echo "run ${i}/100: slam_simulate_slow_map_load=${slow} slam_load_and_localize_on_frame=${frame}"

  $CUVSLAM_LAUNCHER \
    -max_fps=100 \
    --cfg_enable_slam --cfg_enable_export \
    -Podometry.rectified_stereo_camera=true \
    -edex="$KITTI_EDEX" \
    -Pslam.sync_mode=false \
    -slam_input_database=$CUVSLAM_MAP \
    -slam_localize_image=$CUVSLAM_DATASETS/kitti/06/image_0/000270.png \
    -slam_load_and_localize_timestamp=27 -slam_localize_guess_translation="[0.41739893, -4.497604, 290.2034]" \
    -Pslam.max_map_size=10000 \
    -slam_simulate_slow_map_load="${slow}" \
    -slam_load_and_localize_on_frame="${frame}"
done

echo "load_map_stress_test: all 100 runs finished successfully"
