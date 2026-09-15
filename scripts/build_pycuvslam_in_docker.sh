#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/build_pycuvslam_in_docker.sh <build_output_dir>"
  echo "  Run after build_cuvslam_in_docker.sh. Expects build/ in the output dir."
  echo "  Builds one CUDA-specific backend plus the common and default metadata wheels."
  echo ""
  echo "Environment variables (optional):"
  echo "  CUDA_VERSION  CUDA version used for build (default: 12.6.3)"
  exit 1
fi

OUTPUT_DIR=$(realpath "$1")
CUDA_VERSION="${CUDA_VERSION:-12.6.3}"

if [ ! -d "$OUTPUT_DIR/build" ]; then
  echo "Error: $OUTPUT_DIR/build not found."
  echo "Run './scripts/build_cuvslam_in_docker.sh Release $OUTPUT_DIR' first."
  exit 1
fi

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

docker run --runtime=nvidia --gpus all --rm $TTY_FLAG \
  --user "$(id -u):$(id -g)" --group-add video -e HOME=/tmp \
  -v "$(pwd):/cuvslam:ro" \
  -v "$OUTPUT_DIR:/output" \
  -e CUDA_VERSION="$CUDA_VERSION" \
  cuvslam:local bash -c '
    set -euo pipefail
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

    case "$CUDA_MAJOR" in
      12|13) ;;
      *)
        echo "Error: unsupported CUDA major version $CUDA_MAJOR" >&2
        exit 1
        ;;
    esac

    shopt -s nullglob
    existing_wheels=(/output/wheel/*.whl)
    shopt -u nullglob
    if [ "${#existing_wheels[@]}" -ne 0 ]; then
      echo "Error: /output/wheel must not contain stale wheels." >&2
      printf "  %s\n" "${existing_wheels[@]}" >&2
      exit 1
    fi
    staged_output=$(mktemp -d /output/.wheel-staging.XXXXXX)
    trap "rm -rf \"$staged_output\"" EXIT

    cp -r /cuvslam /tmp/cuvslam_src
    PACKAGE_TOOL=/tmp/cuvslam_src/python/packaging/build_packages.py
    python3 "$PACKAGE_TOOL" render-backend \
      --source-root /tmp/cuvslam_src \
      --cuda-major "$CUDA_MAJOR" \
      --output /tmp/cuvslam_src/python/pyproject.toml

    CUVSLAM_BUILD_DIR=/output/build pip wheel --no-deps \
      -w /tmp/backend-raw /tmp/cuvslam_src/python/

    if [ "$CUDA_MAJOR" = "12" ]; then
      EXCLUDES=(
        --exclude libcudart.so.12
        --exclude libcusolver.so.11
        --exclude libcublas.so.12
        --exclude libcublasLt.so.12
        --exclude libcusparse.so.12
        --exclude libnvJitLink.so.12
      )
    else
      EXCLUDES=(
        --exclude libcudart.so.13
        --exclude libcusolver.so.12
        --exclude libcublas.so.13
        --exclude libcublasLt.so.13
        --exclude libcusparse.so.12
        --exclude libnvJitLink.so.13
      )
    fi

    shopt -s nullglob
    raw_wheels=(/tmp/backend-raw/*.whl)
    shopt -u nullglob
    if [ "${#raw_wheels[@]}" -ne 1 ]; then
      echo "Error: expected one raw backend wheel, found ${#raw_wheels[@]}." >&2
      exit 1
    fi

    auditwheel repair "${raw_wheels[0]}" -w /tmp/backend-repaired "${EXCLUDES[@]}"

    shopt -s nullglob
    repaired_wheels=(/tmp/backend-repaired/*.whl)
    shopt -u nullglob
    if [ "${#repaired_wheels[@]}" -ne 1 ]; then
      echo "Error: expected one repaired backend wheel, found ${#repaired_wheels[@]}." >&2
      exit 1
    fi

    python3 -m wheel unpack --dest /tmp/backend-unpacked "${repaired_wheels[0]}"
    shopt -s nullglob
    unpacked_dirs=(/tmp/backend-unpacked/*)
    shopt -u nullglob
    if [ "${#unpacked_dirs[@]}" -ne 1 ]; then
      echo "Error: expected one unpacked wheel directory, found ${#unpacked_dirs[@]}." >&2
      exit 1
    fi

    libcuvslam="${unpacked_dirs[0]}/cuvslam/cu${CUDA_MAJOR}/libcuvslam.so"
    if [ ! -f "$libcuvslam" ]; then
      echo "Error: backend wheel does not contain $libcuvslam." >&2
      exit 1
    fi
    package_rpath=$(python3 "$PACKAGE_TOOL" backend-rpath \
      --source-root /tmp/cuvslam_src --cuda-major "$CUDA_MAJOR")
    existing_rpath=$(patchelf --print-rpath "$libcuvslam")
    relative_rpath=""
    IFS=: read -ra existing_entries <<< "$existing_rpath"
    for entry in "${existing_entries[@]}"; do
      case "$entry" in
        \$ORIGIN*)
          if [ -n "$relative_rpath" ]; then
            relative_rpath="$relative_rpath:$entry"
          else
            relative_rpath="$entry"
          fi
          ;;
      esac
    done
    if [ -n "$relative_rpath" ]; then
      package_rpath="$relative_rpath:$package_rpath"
    fi
    patchelf --set-rpath "$package_rpath" "$libcuvslam"

    python3 -m wheel pack --dest-dir /tmp/backend-final "${unpacked_dirs[0]}"
    shopt -s nullglob
    backend_wheels=(/tmp/backend-final/*.whl)
    shopt -u nullglob
    if [ "${#backend_wheels[@]}" -ne 1 ]; then
      echo "Error: expected one final backend wheel, found ${#backend_wheels[@]}." >&2
      exit 1
    fi

    python3 "$PACKAGE_TOOL" stage-common \
      --source-root /tmp/cuvslam_src \
      --backend-wheel "${backend_wheels[0]}" \
      --stage-dir /tmp/cuvslam-common
    python3 "$PACKAGE_TOOL" stage-meta \
      --source-root /tmp/cuvslam_src \
      --stage-dir /tmp/cuvslam-meta

    pip wheel --no-deps -w /tmp/common-wheel /tmp/cuvslam-common
    pip wheel --no-deps -w /tmp/meta-wheel /tmp/cuvslam-meta

    cp "${backend_wheels[0]}" /tmp/common-wheel/*.whl /tmp/meta-wheel/*.whl "$staged_output/"
    python3 "$PACKAGE_TOOL" verify-ownership "$staged_output"/*.whl

    if [ "$CUDA_MAJOR" = "12" ]; then
      expected_backend=cuvslam_cu12
    else
      expected_backend=cuvslam_cu13
    fi
    shopt -s nullglob
    published_backends=("$staged_output"/${expected_backend}-*.whl)
    shopt -u nullglob
    if [ "${#published_backends[@]}" -ne 1 ]; then
      echo "Error: final output is missing the ${expected_backend} wheel." >&2
      exit 1
    fi

    if [ -d /output/wheel ]; then
      rmdir /output/wheel
    fi
    mv "$staged_output" /output/wheel
    trap - EXIT
  '
