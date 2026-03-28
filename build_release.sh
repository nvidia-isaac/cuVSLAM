#!/bin/sh

# Description:
# This script builds and tests the cuVSLAM library in release or selected build type.
# It can perform different operations independently or combined:
#   1. Run module tests
#   2. Run API tests
#   3. Build cuvslam library and python bindings
#   4. Build documentation
#
# Parameters:
#   --modules_test     Run module tests
#   --api_test         Run API tests
#   --build_lib        Build cuvslam library and python bindings
#   --build_docs       Build documentation
#   --build_type=TYPE  Set CMake build type (Debug|Release[default]|RelWithDebInfo|MinSizeRel)
#   --cuda_arch=ARCH   Set CUDA architecture target (e.g. 87 for Orin Nano, default: all)
#   --jobs=N           Set number of parallel jobs (default: 8)
#
# Environment variables (optional):
#   CUVSLAM_SRC_DIR    Source directory (default: /cuvslam/src)
#   CUVSLAM_DST_DIR    Build directory (default: /cuvslam/build)
#
# Note: Boolean flags (--modules_test, --api_test, --build_lib, --build_docs)
#       don't require =true, they are just flags.
# Note: If no flags are set, make all will be performed.
# Note: Don't mix different build types in the same build directory,
#       use different build directories via CUVSLAM_DST_DIR variable.
# Note: Test commands will also build necessary targets.

set -e  # Exit immediately if a command exits with a non-zero status

# Default values
MODULETESTS=false
APITESTS=false
LIBBUILD=false
BUILDDOCS=false
BUILD_TYPE="Release"
CUDA_ARCH=""
USE_RERUN=OFF

SRC=/cuvslam/src
DST=/cuvslam/build

MAKE_JOBS=$(nproc 2>/dev/null || echo 8)

if [ ${CUVSLAM_SRC_DIR+x} ]; then
  echo "Using environment variable CUVSLAM_SRC_DIR='$CUVSLAM_SRC_DIR'."
  SRC=$CUVSLAM_SRC_DIR
fi
if [ ${CUVSLAM_DST_DIR+x} ]; then
  echo "Using environment variable CUVSLAM_DST_DIR='$CUVSLAM_DST_DIR'."
  DST=$CUVSLAM_DST_DIR
fi

# Parse named arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --modules_test)
      MODULETESTS=true
      ;;
    --api_test)
      APITESTS=true
      ;;
    --build_lib)
      LIBBUILD=true
      ;;
    --build_docs)
      BUILDDOCS=true
      ;;
    --jobs=*)
      MAKE_JOBS="${1#*=}"
      ;;
    --cuda_arch=*)
      CUDA_ARCH="${1#*=}"
      ;;
    --build_type=*)
      BUILD_TYPE="${1#*=}"
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
  shift
done

# Function to check if argument is true
is_true() {
  [ "$1" = "true" ]
}

# Check if directories exist
if [ ! -d "$SRC" ]; then
  echo "Error: Source directory '$SRC' does not exist."
  exit 1
fi

set -v # echo each command

mkdir -p $DST
cd $DST
CMAKE_ARGS="-DUSE_RERUN=$USE_RERUN -DCMAKE_BUILD_TYPE=$BUILD_TYPE"
if [ -n "$CUDA_ARCH" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
fi
cmake $CMAKE_ARGS -S $SRC -B $DST
# Build all CMake targets regardless of the flags
make -j${MAKE_JOBS} -C $DST

# Step 1: Run module tests
if is_true "$MODULETESTS"; then
  echo "Module tests executed."
  GTEST_FILTER=-*SpeedUp*:*Speedup* ctest --output-on-failure || exit 1
else
  echo "Module tests skipped."
fi

# Step 2: Run Python API tests
if is_true "$APITESTS"; then
  echo "Python API tests executed."
  CUVSLAM_BUILD_DIR=$DST pip install --user -e $SRC/python
  python3 -m unittest discover -v -s $SRC/python/test --locals || exit 1
else
  echo "Python API tests skipped."
fi

# Step 3: Build cuvslam
if is_true "$LIBBUILD"; then
  echo "libcuvslam build executed."
else
  echo "libcuvslam build skipped."
fi

# Step 4: Build documentation
if is_true "$BUILDDOCS"; then
  echo "Documentation build executed."
  make -j${MAKE_JOBS} -C $DST doc
  CUVSLAM_BUILD_DIR=$DST pip install --user -e $SRC/python
  "$(dirname "$0")/build_docs.sh" --build_dir=$DST --src_dir=$SRC/python/docs
else
  echo "Documentation build skipped."
fi

echo "Script execution completed."
