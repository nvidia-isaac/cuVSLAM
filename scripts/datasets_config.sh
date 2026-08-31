S3_DATASETS_BUCKET="${S3_DATASETS_BUCKET:?Set repository variable S3_DATASETS_BUCKET to the dataset tarball location, e.g. s3://your-bucket/datasets/vslam}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:?Set repository variable AWS_DEFAULT_REGION, e.g. us-west-2}"

PROVISIONABLE_DATASETS=(kitti euroc tum tartan)

EVAL_DATASET_NAMES=(
  kitti
  euroc
)

is_provisionable_dataset() {
  local name="$1" d
  for d in "${PROVISIONABLE_DATASETS[@]}"; do
    [ "$d" = "$name" ] && return 0
  done
  return 1
}

dataset_upload_subdir() {
  is_provisionable_dataset "$1" || {
    echo "Error: unknown dataset '$1' (expected: ${PROVISIONABLE_DATASETS[*]})" >&2
    return 1
  }
  case "$1" in
    kitti) echo "" ;;
    *)     echo "$1" ;;
  esac
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
