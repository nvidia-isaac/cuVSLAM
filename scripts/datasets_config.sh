S3_DATASETS_BUCKET="${S3_DATASETS_BUCKET:?Set repository variable S3_DATASETS_BUCKET to the dataset tarball location, e.g. s3://your-bucket/datasets/vslam}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:?Set repository variable AWS_DEFAULT_REGION, e.g. us-west-2}"

# Dataset names, evaluation records, and the archive layout live in
# tools/python_tools/cuvslam_tools/dataset_registry.py. That module is standard
# library only and imports converter code lazily, so it runs here with
# PYTHONPATH alone, before anything is installed in the image.
CUVSLAM_REPO_ROOT="${CUVSLAM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"

dataset_registry() {
  PYTHONPATH="$CUVSLAM_REPO_ROOT/tools/python_tools${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m cuvslam_tools.dataset_registry "$@"
}

s3_tarball_uri() {
  local name="$1"
  local _s3_path="${S3_DATASETS_BUCKET#s3://}"
  local bucket="${_s3_path%%/*}"
  local prefix=""
  case "$_s3_path" in
    */*) prefix="${_s3_path#*/}" ;;
  esac
  while [ "${prefix%/}" != "$prefix" ]; do prefix="${prefix%/}"; done
  if [ -n "$prefix" ]; then
    echo "s3://${bucket}/${prefix}/${name}.tar"
  else
    echo "s3://${bucket}/${name}.tar"
  fi
}
