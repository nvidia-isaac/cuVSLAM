#!/usr/bin/env bash
# Download KITTI's raw-data calibrations and the Cvišić et al. (ECMR 2021) recalibrations.
#
# Usage: download_kitti_recalibrated.sh OUT_DIR [--checkerboards]
#
#   --checkerboards  Also download the 2011_09_26 checkerboard drive (calibrate_09_26.py).

set -euo pipefail

out_dir=""
checkerboards=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkerboards) checkerboards=1; shift ;;
        -*) echo "error: unknown option '$1'" >&2; exit 2 ;;
        *)  out_dir="$1"; shift ;;
    esac
done
if [[ -z "${out_dir}" ]]; then
    echo "error: OUT_DIR is required" >&2
    exit 2
fi

kitti="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
authors="https://bitbucket.org/unizg-fer-lamor/kittical/raw/master"
urls=(
    "${kitti}/2011_09_26_calib.zip"
    "${kitti}/2011_09_30_calib.zip"
    "${kitti}/2011_10_03_calib.zip"
    "${authors}/lamor_00_02.yaml"
    "${authors}/lamor_04_12.yaml"
)
if [[ "${checkerboards}" -eq 1 ]]; then
    urls+=("${kitti}/2011_09_26_drive_0119/2011_09_26_drive_0119_extract.zip")
fi

mkdir -p "${out_dir}"
for url in "${urls[@]}"; do
    dest="${out_dir}/$(basename -- "${url}")"
    if [[ -s "${dest}" ]]; then
        echo "using existing ${dest}"
        continue
    fi
    echo "downloading $(basename -- "${url}") …"
    curl -fL --retry 5 --retry-delay 5 -C - -o "${dest}.download" "${url}"
    mv -f -- "${dest}.download" "${dest}"
done
