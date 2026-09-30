#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

# Download the TartanAir V1 Hard zips the evaluation uses: left and right images
# and left depth for each environment in zipfiles.txt.
#
# Usage: download_tartanair_v1.sh [OPTIONS] [OUT_DIR]
#
#   OUT_DIR      Directory to save zips. Defaults to <repo_root>/datasets/tartanair_v1/raw
#   --force      Re-download zips even when they already exist.
#   --env NAME   Download only environment NAME. May be repeated.
#
# The zips are the public TartanAir V1 release, served anonymously from CMU
# AirLab's object store, the same source castacks/tartanair_tools downloads from.
# zipfiles.txt pins each zip's size and content MD5, so a download is only
# accepted if it is byte-identical to the one this evaluation was built on. The
# MD5 equals the store's ETag for every zip except five the store keeps as
# segmented objects (x-static-large-object: the abandonedfactory image zips and
# all three neighborhood zips), whose ETag is not a content hash; those were
# pinned from downloads whose zip CRCs all verified.

set -euo pipefail

readonly base_url="https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanair"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
manifest="${script_dir}/zipfiles.txt"
out_dir="${repo_root}/datasets/tartanair_v1/raw"
force=0
requested_envs=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        --env)
            [[ $# -lt 2 ]] && { echo "error: --env requires a value" >&2; exit 2; }
            requested_envs+=("$2"); shift 2 ;;
        -*) echo "error: unknown option '$1'" >&2; exit 2 ;;
        *)  out_dir="$1"; shift ;;
    esac
done

declare -A known_envs=()
while read -r key _ _; do
    known_envs["${key%%/*}"]=1
done < "${manifest}"
for env in "${requested_envs[@]}"; do
    if [[ -z "${known_envs[${env}]:-}" ]]; then
        echo "error: '${env}' is not an environment in $(basename -- "${manifest}")" >&2
        exit 2
    fi
done

is_requested() {
    local env="$1" requested
    [[ ${#requested_envs[@]} -eq 0 ]] && return 0
    for requested in "${requested_envs[@]}"; do
        [[ "${requested}" == "${env}" ]] && return 0
    done
    return 1
}

# <env>/Hard/<kind>.zip is saved as <env>_Hard_<kind>.zip.
download_zip() {
    local key="$1" md5="$2" size="$3"
    local dest="${out_dir}/${key//\//_}"
    local partial="${dest}.partial"

    # An existing zip is hashed too: it may predate a pin change, or have been
    # placed or damaged by hand, and a size match cannot tell.
    if [[ -f "${dest}" && "${force}" -eq 0 ]]; then
        if [[ "$(stat -c %s -- "${dest}")" == "${size}" \
              && "$(md5sum -- "${dest}" | cut -d' ' -f1)" == "${md5}" ]]; then
            echo "using existing ${dest}"
            return
        fi
        echo "discarding ${dest}: it does not match zipfiles.txt"
        rm -f -- "${dest}"
    fi

    mkdir -p "${out_dir}"
    [[ "${force}" -eq 1 ]] && rm -f -- "${dest}" "${partial}"

    echo "downloading ${key} ($((size / 1000000)) MB) …"
    # HTTP/2 full downloads of the segmented objects have returned bytes past the
    # object's end; HTTP/1.1 has not. stdin is the manifest the loop is reading.
    curl -fL --http1.1 --retry 5 --retry-delay 5 -C - -o "${partial}" "${base_url}/${key}" < /dev/null

    local actual_size actual_md5
    actual_size="$(stat -c %s -- "${partial}")"
    if [[ "${actual_size}" != "${size}" ]]; then
        echo "error: ${key} is ${actual_size} bytes, zipfiles.txt pins ${size}" >&2
        rm -f -- "${partial}"
        exit 1
    fi
    actual_md5="$(md5sum -- "${partial}" | cut -d' ' -f1)"
    if [[ "${actual_md5}" != "${md5}" ]]; then
        echo "error: ${key} has MD5 ${actual_md5}, zipfiles.txt pins ${md5}; the upstream file changed" >&2
        rm -f -- "${partial}"
        exit 1
    fi

    mv -f -- "${partial}" "${dest}"
}

while read -r key md5 size; do
    [[ -z "${key}" ]] && continue
    is_requested "${key%%/*}" && download_zip "${key}" "${md5}" "${size}"
done < "${manifest}"

echo "done — files saved to ${out_dir}"
