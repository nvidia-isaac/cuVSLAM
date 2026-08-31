#!/usr/bin/env bash
# Download TUM RGB-D freiburg3 sequence archives.
#
# Usage: download_tum.sh [OPTIONS] [OUT_DIR]
#
#   OUT_DIR        Directory to save archives.
#                  Defaults to <repo_root>/datasets/tum/raw
#   --force        Re-download archives even when they already exist.
#   --archive NAME Download only NAME. May be repeated. By default all 15
#                  evaluated sequence archives are downloaded.
#
# The dataset pages serve /rgbd/dataset/freiburg3/<name>.tgz, which redirects to
# a webshare host, so redirects must be followed. TUM publishes no per-file
# checksums; the converter records a sha256 for each archive it consumes in
# dataset_metadata.json instead.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# tum_download is the ordered list of sequence archives to fetch. Keep it in
# sync with ALL_SEQS in convert_tum.py.

readonly base_url="https://cvg.cit.tum.de/rgbd/dataset/freiburg3"

readonly -a tum_download=(
    "rgbd_dataset_freiburg3_sitting_halfsphere.tgz"
    "rgbd_dataset_freiburg3_nostructure_texture_far.tgz"
    "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop.tgz"
    "rgbd_dataset_freiburg3_teddy.tgz"
    "rgbd_dataset_freiburg3_structure_texture_far.tgz"
    "rgbd_dataset_freiburg3_walking_halfsphere.tgz"
    "rgbd_dataset_freiburg3_structure_notexture_far.tgz"
    "rgbd_dataset_freiburg3_sitting_xyz.tgz"
    "rgbd_dataset_freiburg3_cabinet.tgz"
    "rgbd_dataset_freiburg3_nostructure_texture_near_withloop.tgz"
    "rgbd_dataset_freiburg3_nostructure_notexture_far.tgz"
    "rgbd_dataset_freiburg3_sitting_xyz_validation.tgz"
    "rgbd_dataset_freiburg3_structure_texture_near.tgz"
    "rgbd_dataset_freiburg3_long_office_household.tgz"
    "rgbd_dataset_freiburg3_large_cabinet_validation.tgz"
)
# ──────────────────────────────────────────────────────────────────────────────

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
out_dir="${repo_root}/datasets/tum/raw"
force=0
requested_archives=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        --archive)
            [[ $# -lt 2 ]] && { echo "error: --archive requires a value" >&2; exit 2; }
            requested_archives+=("$2"); shift 2 ;;
        -*) echo "error: unknown option '$1'" >&2; exit 2 ;;
        *)  out_dir="$1"; shift ;;
    esac
done

download_names=("${tum_download[@]}")
if [[ ${#requested_archives[@]} -gt 0 ]]; then
    download_names=("${requested_archives[@]}")
fi

is_known_archive() {
    local candidate="$1" known
    for known in "${tum_download[@]}"; do
        [[ "${known}" == "${candidate}" ]] && return 0
    done
    return 1
}

download_archive() {
    local name="$1"
    local dest="${out_dir}/${name}"
    local partial="${dest}.download"
    local url="${base_url}/${name}"

    if [[ -s "${dest}" && "${force}" -eq 0 ]]; then
        echo "using existing ${dest}"
        return
    fi

    mkdir -p "${out_dir}"
    [[ "${force}" -eq 1 ]] && rm -f -- "${dest}" "${partial}"

    echo "downloading ${name} …"
    curl -fL --retry 5 --retry-delay 5 -C - -o "${partial}" "${url}"

    # curl -f only catches HTTP >=400; a 200 with an HTML error/landing page
    # would slip through. The archives are gzip (.tgz), so verify the magic
    # bytes (1f 8b) before accepting the download.
    local magic
    magic="$(head -c 2 -- "${partial}" | od -An -tx1 | tr -d ' ')"
    if [[ "${magic}" != "1f8b" ]]; then
        echo "error: ${name} is not a gzip archive (magic '${magic}'); the server likely returned an error page" >&2
        rm -f -- "${partial}"
        exit 1
    fi

    mv -f -- "${partial}" "${dest}"
}

declare -A seen_archives=()
for name in "${download_names[@]}"; do
    if ! is_known_archive "${name}"; then
        echo "error: '${name}' is not a known TUM archive; see tum_download in $(basename -- "${BASH_SOURCE[0]}")" >&2
        exit 2
    fi
    if [[ -n "${seen_archives[${name}]:-}" ]]; then
        continue
    fi
    seen_archives["${name}"]=1
    download_archive "${name}"
done

echo "done — files saved to ${out_dir}"
