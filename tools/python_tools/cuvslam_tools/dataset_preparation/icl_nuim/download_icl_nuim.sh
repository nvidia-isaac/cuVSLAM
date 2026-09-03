#!/usr/bin/env bash
# Download ICL-NUIM TUM-compatible sequence archives and their pose files.
#
# Usage: download_icl_nuim.sh [OPTIONS] [OUT_DIR]
#
#   OUT_DIR        Directory to save archives.
#                  Defaults to <repo_root>/datasets/icl_nuim/raw
#   --force        Re-download files even when they already exist.
#   --archive NAME Download only NAME (a "<sequence>.tar.gz"). May be repeated.
#                  By default all eight published trajectories are downloaded.
#
# Each archive carries its rgb/ and depth/ PNGs, its associations.txt, and its
# .gt.freiburg pose file, so the archive is the only download needed. The pose
# file the dataset page offers separately is byte-identical to the bundled one.
#
# The dataset pages link these over http; https serves the same files and is
# used here. ICL-NUIM publishes no checksums, so the converter records a sha256
# for every archive it consumes in dataset_metadata.json instead.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# icl_download is the list of sequence archives to fetch. Keep it in sync with
# ALL_SEQS in convert_icl_nuim.py.

readonly archive_base_url="https://www.doc.ic.ac.uk/~ahanda"

readonly -a icl_download=(
    "living_room_traj0_frei_png.tar.gz"
    "living_room_traj1_frei_png.tar.gz"
    "living_room_traj2_frei_png.tar.gz"
    "living_room_traj3_frei_png.tar.gz"
    "traj0_frei_png.tar.gz"
    "traj1_frei_png.tar.gz"
    "traj2_frei_png.tar.gz"
    "traj3_frei_png.tar.gz"
)
# ──────────────────────────────────────────────────────────────────────────────

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
out_dir="${repo_root}/datasets/icl_nuim/raw"
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

download_names=("${icl_download[@]}")
if [[ ${#requested_archives[@]} -gt 0 ]]; then
    download_names=("${requested_archives[@]}")
fi

is_known_archive() {
    local candidate="$1" known
    for known in "${icl_download[@]}"; do
        [[ "${known}" == "${candidate}" ]] && return 0
    done
    return 1
}

download_archive() {
    local name="$1"
    local dest="${out_dir}/${name}"
    local partial="${dest}.download"

    if [[ -s "${dest}" && "${force}" -eq 0 ]]; then
        echo "using existing ${dest}"
        return
    fi

    mkdir -p "${out_dir}"
    [[ "${force}" -eq 1 ]] && rm -f -- "${dest}" "${partial}"

    echo "downloading ${name} …"
    curl -fL --retry 5 --retry-delay 5 -C - -o "${partial}" "${archive_base_url}/${name}"

    # curl -f only catches HTTP >=400; a 200 with an HTML error/landing page
    # would slip through. The archives are gzip, so verify the magic bytes
    # (1f 8b) before accepting the download.
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
        echo "error: '${name}' is not a known ICL-NUIM archive; see icl_download in $(basename -- "${BASH_SOURCE[0]}")" >&2
        exit 2
    fi
    if [[ -n "${seen_archives[${name}]:-}" ]]; then
        continue
    fi
    seen_archives["${name}"]=1
    download_archive "${name}"
done

echo "done — files saved to ${out_dir}"
