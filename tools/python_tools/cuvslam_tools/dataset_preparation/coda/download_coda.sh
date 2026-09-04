#!/usr/bin/env bash
# Check for the CODa sequence archives.
#
# Usage: download_coda.sh [OPTIONS] [OUT_DIR]
#
#   OUT_DIR        Directory holding the CODa sequence archives.
#                  Defaults to <repo_root>/datasets/coda/raw
#   --force        Accepted for parity with the other download scripts; has no effect.
#   --archive NAME Require NAME specifically. May be repeated. By default any one
#                  of the 23 sequence archives is enough.
#
# Nothing is downloaded here. CODa is distributed under a license every user has
# to accept on the Texas Dataverse, so the archives are fetched by hand and this
# script only reports whether they are in place.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# coda_archives is the full set of sequence archives. Keep it in sync with
# ALL_SEQS in convert_coda.py.

readonly dataset_url="https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV"

readonly -a coda_archives=(
    "0.zip"  "1.zip"  "2.zip"  "3.zip"  "4.zip"  "5.zip"  "6.zip"  "7.zip"
    "8.zip"  "9.zip"  "10.zip" "11.zip" "12.zip" "13.zip" "14.zip" "15.zip"
    "16.zip" "17.zip" "18.zip" "19.zip" "20.zip" "21.zip" "22.zip"
)
# ──────────────────────────────────────────────────────────────────────────────

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
out_dir="${repo_root}/datasets/coda/raw"
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

if [[ "${force}" -eq 1 ]]; then
    echo "note: --force has no effect — CODa archives are never downloaded"
fi

is_known_archive() {
    local candidate="$1" known
    for known in "${coda_archives[@]}"; do
        [[ "${known}" == "${candidate}" ]] && return 0
    done
    return 1
}

registration_walkthrough() {
    cat >&2 <<EOF

CODa requires accepting the dataset license, so the archives are supplied by hand.

  1. Register and accept the dataset license at:
       ${dataset_url}
  2. Download the per-sequence archives (${coda_archives[0]} … ${coda_archives[-1]}) you want.
  3. Place them in: ${out_dir}
  4. Re-run this command.
EOF
}

# A truncated download or a saved error page reads as a zip by name only, so
# check the archive magic ("PK") before the converter opens it.
check_archive() {
    local name="$1"
    local path="${out_dir}/${name}"
    local magic

    # -s above only says the entry is non-empty, so it still passes for an
    # unreadable file or a directory that happens to be named <n>.zip. Capture
    # the read failure here; set -e would otherwise abort the run with nothing
    # but head's own message.
    if ! magic="$(head -c 2 -- "${path}" | od -An -c | tr -d ' ')"; then
        echo "error: cannot read ${path} to check that it is a zip archive" >&2
        exit 1
    fi
    if [[ "${magic}" != "PK" ]]; then
        echo "error: ${path} is not a zip archive; re-download it from the Texas Dataverse" >&2
        exit 1
    fi
}

if [[ ${#requested_archives[@]} -gt 0 ]]; then
    missing=()
    for name in "${requested_archives[@]}"; do
        if ! is_known_archive "${name}"; then
            echo "error: '${name}' is not a known CODa archive; see coda_archives in $(basename -- "${BASH_SOURCE[0]}")" >&2
            exit 2
        fi
        if [[ -s "${out_dir}/${name}" ]]; then
            check_archive "${name}"
        else
            missing+=("${name}")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: missing CODa archive(s) in ${out_dir}: ${missing[*]}" >&2
        registration_walkthrough
        exit 1
    fi
    echo "Found ${#requested_archives[@]} requested CODa archive(s) in ${out_dir}."
    exit 0
fi

found=()
for name in "${coda_archives[@]}"; do
    if [[ -s "${out_dir}/${name}" ]]; then
        check_archive "${name}"
        found+=("${name}")
    fi
done

if [[ ${#found[@]} -eq 0 ]]; then
    echo "ERROR: no CODa sequence archives found in ${out_dir}." >&2
    registration_walkthrough
    exit 1
fi

echo "Found ${#found[@]} CODa sequence archive(s) in ${out_dir}: ${found[*]}"
