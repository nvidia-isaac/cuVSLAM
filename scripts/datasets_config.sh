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

s3_dataset_bucket() {
  local _s3_path="${S3_DATASETS_BUCKET#s3://}"
  echo "${_s3_path%%/*}"
}

# Object key for one dataset tarball. S3_DATASETS_BUCKET may name a bucket root,
# in which case there is no prefix at all, and it may carry a trailing slash,
# which must not survive into the key: S3 treats "prefix//name.tar" as a
# different object than "prefix/name.tar".
s3_dataset_key() {
  local name="$1"
  local _s3_path="${S3_DATASETS_BUCKET#s3://}"
  local prefix=""
  case "$_s3_path" in
    */*) prefix="${_s3_path#*/}" ;;
  esac
  while [ "${prefix%/}" != "$prefix" ]; do prefix="${prefix%/}"; done
  if [ -n "$prefix" ]; then
    echo "${prefix}/${name}.tar"
  else
    echo "${name}.tar"
  fi
}

s3_tarball_uri() {
  echo "s3://$(s3_dataset_bucket)/$(s3_dataset_key "$1")"
}
