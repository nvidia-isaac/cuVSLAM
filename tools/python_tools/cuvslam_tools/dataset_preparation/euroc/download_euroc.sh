#!/usr/bin/env bash
# Download EuRoC MAV dataset archives from the ETH Research Collection and
# verify them against published MD5 checksums.
#
# Usage: download_euroc.sh [OPTIONS] [OUT_DIR]
#
#   OUT_DIR        Directory to save archives.
#                  Defaults to <repo_root>/datasets/euroc/raw
#   --force        Re-download archives even when they already exist.
#   --archive NAME Download only NAME. May be repeated. By default all three
#                  sequence bundles are downloaded.
#
# Files are fetched from the DSpace REST content endpoint
# (<base_url>/<content_id>/content), which serves valid zip data and supports
# resumable (range) downloads. Do not use the /bitstreams/<id>/download UI
# route — it can return an HTML error page with a 200/500 status.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# Edit these tables when the dataset record changes.
#
# base_url       DSpace bitstream content endpoint.
# euroc_archives name -> "<content_id> <md5>" for every known archive in the
#                ETH Research Collection record (DOI 10.3929/ethz-b-000690084).
# euroc_download ordered list of archive names to actually download.

readonly base_url="https://www.research-collection.ethz.ch/server/api/core/bitstreams"
readonly user_agent="cuVSLAM-dataset-preparation/1.0"

readonly -A euroc_archives=(
    [machine_hall.zip]="7b2419c1-62b5-4714-b7f8-485e5fe3e5fe 363f5c2502b469cdd97ef85997714806"
    [vicon_room1.zip]="02ecda9a-298f-498b-970c-b7c44334d880 5ce06b405827e453a82523d3ca9c2fd0"
    [vicon_room2.zip]="ea12bc01-3677-4b4c-853d-87c7870b8c44 c6347f4e0476aaa9a43a919c163c49c5"
    [calibration_datasets.zip]="5732e864-10f1-49e7-befb-669ee29ff770 be0b53d38df53cfab699a795f31dbc7b"
)

readonly -a euroc_download=(
    "machine_hall.zip"
    "vicon_room1.zip"
    "vicon_room2.zip"
)
# ──────────────────────────────────────────────────────────────────────────────

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
out_dir="${repo_root}/datasets/euroc/raw"
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

download_names=("${euroc_download[@]}")
if [[ ${#requested_archives[@]} -gt 0 ]]; then
    download_names=("${requested_archives[@]}")
fi

download_archive() {
    local name="$1" id="$2" md5="$3"
    local dest="${out_dir}/${name}"
    local partial="${dest}.download"
    local url="${base_url}/${id}/content"

    if [[ -s "${dest}" && "${force}" -eq 0 ]]; then
        local existing_md5
        existing_md5="$(md5sum -- "${dest}" | awk '{print $1}')"
        if [[ "${existing_md5}" == "${md5}" ]]; then
            echo "using verified ${dest}"
            return
        fi
        echo "error: md5 mismatch for existing ${dest}; re-run with --force" >&2
        exit 1
    fi

    mkdir -p "${out_dir}"
    [[ "${force}" -eq 1 ]] && rm -f -- "${dest}" "${partial}"

    echo "downloading ${name} …"
    # ETH rejects curl's default user agent with HTTP 403.
    curl -fL --retry 5 --retry-delay 5 -A "${user_agent}" -C - -o "${partial}" "${url}"

    echo "verifying md5 for ${name} …"
    local actual
    actual="$(md5sum -- "${partial}" | awk '{print $1}')"
    if [[ "${actual}" != "${md5}" ]]; then
        echo "error: md5 mismatch for ${name} (expected ${md5}, got ${actual})" >&2
        rm -f -- "${partial}"
        exit 1
    fi

    mv -f -- "${partial}" "${dest}"
}

declare -A seen_archives=()
for name in "${download_names[@]}"; do
    entry="${euroc_archives[${name}]:-}"
    if [[ -z "${entry}" ]]; then
        echo "error: no config entry for '${name}' in euroc_archives" >&2
        exit 2
    fi
    if [[ -n "${seen_archives[${name}]:-}" ]]; then
        continue
    fi
    seen_archives["${name}"]=1
    read -r id md5 <<< "${entry}"
    download_archive "${name}" "${id}" "${md5}"
done

echo "done — files saved to ${out_dir}"
