#!/bin/sh
# Runs `apt-get update`, retrying with a growing delay when it fails. APT's
# Acquire::Retries covers only connection failures; an index caught mid mirror
# sync ("Hash Sum mismatch", "File has unexpected size") fails at once.
set -eu

attempts=${APT_UPDATE_ATTEMPTS:-4}
attempt=1
while ! apt-get update; do
  if [ "$attempt" -ge "$attempts" ]; then
    echo "apt-get update failed after $attempts attempts." >&2
    exit 100
  fi
  delay=$((attempt * 30))
  echo "apt-get update failed; retrying in ${delay}s (attempt $((attempt + 1)) of $attempts)." >&2
  sleep "$delay"
  attempt=$((attempt + 1))
done
