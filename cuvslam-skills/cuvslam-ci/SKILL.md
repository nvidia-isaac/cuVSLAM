---
name: cuvslam-ci
description: Use when working on cuVSLAM CI/CD - the GitHub Actions nightly and PR-verify pipelines, dataset provisioning and staging, the evaluation run and KPI reporting, the build/test/lint matrix, branch rulesets, or repository secrets and variables. Covers adding a dataset, changing dataset format or packing, and controlling which datasets run in PR versus nightly.
---

# cuVSLAM CI/CD

CI/CD runs build, unit test, lint, dataset evaluation, dataset provisioning, and
nightly releases on GitHub Actions with self-hosted GPU and Jetson runners. This
file is the task entry point. Read [reference.md](reference.md) for the
architecture, the secrets and variables, and the load-bearing constraints before
changing anything.

## Component map

Workflows (`.github/workflows/`):

- `pr-verify.yml` - lint, then build + unit test on x86, Orin, and Thor; eval on the x86 job (fork-gated); posts a KPI table to the PR comment. Jetson benchmarks do not run on PRs.
- `nightly.yml` - scheduled/manual build + test matrix; eval on the four x86 configs and CUDA micro-benchmarks on Orin and Thor; writes per-config reports and versioned Actions artifacts. Scheduled runs never create a Release. A manual dispatch from a matching `release/vX.Y.Z` branch promotes the same distributable bytes to a protected draft GitHub Release.
- `provision-datasets.yml` - manual `workflow_dispatch` on the default branch; downloads, converts, and uploads a dataset tarball to S3. The only writer of dataset storage.

Branch protection is not in this repository. The default-branch ruleset is configured in the GitHub
UI; read what is actually enforced with `gh api repos/<owner>/<repo>/rules/branches/main`.

CI scripts (`scripts/`):

- `Dockerfile.ci` - the shared `cuvslam-ci:local` image (git, python3, pre-commit, GPG-verified AWS CLI, jq).
- `datasets_config.sh` - S3 location plus the `dataset_registry` shell shim; dataset names and evaluation records live in `tools/python_tools/cuvslam_tools/dataset_registry.py`.
- `provision_dataset.sh` - runs `dataset_registry prepare`, tars the directory `prepare()` returned (uncompressed `.tar`), uploads to S3.
- `stage_eval_datasets.sh` - validates the registry, downloads each eval dataset's `<id>.tar` from S3, streams it into the local cache, and checks the staged reporter config against the registry.
- `check_eval_prerequisites.sh` - verifies credentials/cache and `RUNNER_STORAGE_ROOT`.
- `benchmark_cuvslam_in_docker.sh` - runs the active `cuda_modules_test` speed benchmarks in the product container and captures runner metadata, raw output, and GoogleTest XML.
- `cuvslam_benchmark_report.py` - validates benchmark XML properties and renders per-Jetson JSON and Markdown reports.
- `eval_cuvslam_in_docker.sh` - host wrapper: mounts datasets and KPI history, starts the eval container.
- `run_eval.sh` - in container: reads evaluation records from the registry, runs `cuvslam_app.py`, then collects
  machine-readable KPI JSON.
- `cuvslam_kpi_report.py` - owns KPI collection, rolling diffs, cross-config aggregation, soft drift data, and all
  KPI Markdown rendering. `collect` writes raw and report JSON; `render` and `aggregate` produce publication Markdown.
- `kpi_baseline_ranges.json` - committed static drift ranges.
- `package_cpp_dist.sh` - creates and validates the curated, versioned C++ SDK archive used by Actions and Releases.
- `Dockerfile` / `build_cuvslam_in_docker.sh` - product build image and wrapper; preserve Git/LFS metadata used by
  `get_version()`.

Dataset tooling: `tools/python_tools/cuvslam_tools/dataset_preparation/<name>/` (`prepare.py`, plus `download_<name>.sh` where `curl` resume and checksum behaviour is needed), `tools/cuvslam_app/` (eval runner and `edex_reader.py`).

## Task: add a dataset

The dataset ID is the only name: it is the preparation module, the `<id>.tar` object, the staged directory, and the `/sequences` mount.

1. Add `tools/python_tools/cuvslam_tools/dataset_preparation/<id>/prepare.py` (plus a downloader) exposing `prepare()` and `main()`. It must accept `--raw-dir`, `--output-dir`, and `--force-download`, and **return the directory to archive** — that directory becomes the tar root and the staged dataset root. Each generated reporter config must set `"dataset_folder": "<id>/"`.
2. Add a `DatasetSpec` to `DATASETS` in `tools/python_tools/cuvslam_tools/dataset_registry.py` with the ID and the preparation module, and no `evals` yet. Run `python3 -m cuvslam_tools.dataset_registry validate`.
3. Add the ID to the `dataset` choice input in `provision-datasets.yml`. A registry test asserts every choice resolves.
4. Run Provision dataset (`workflow_dispatch`) on the default branch. It writes `<S3_DATASETS_BUCKET>/<id>.tar`.
5. Enable evaluation by adding one or more `EvalSpec` records to that `DatasetSpec`: the reporter config filename, the `cuvslam_app` flags, and suite membership. Nothing else needs editing; staging and eval both read the registry.
6. Add expected KPI ranges to `scripts/kpi_baseline_ranges.json`. The key prefix is derived from the config filename (first hyphen-delimited token, upper-cased), so name configs with underscores inside the prefix and a hyphen only as its terminator: `tartan_flaky-vo_slam.cfg` gives `TARTAN_FLAKY`, whereas `tartan-flaky-vo_slam.cfg` would collide with `TARTAN`.

## Task: change dataset format or packing

A dataset moves through four stages. Change the one that owns the format, and keep packing and extraction in sync.

- Conversion (raw to stored layout, e.g. images vs mp4): `tools/python_tools/cuvslam_tools/dataset_preparation/<name>/prepare.py` and the converter it calls.
- Tarball packing: `scripts/provision_dataset.sh` creates an uncompressed `.tar` (`tar -cf`); `s3_tarball_uri` names it `<name>.tar`.
- Extraction: `scripts/stage_eval_datasets.sh` runs `tar -xf`.
- In-archive layout consumed at eval: `tools/cuvslam_app/cuvslam_app.py` and `tools/cuvslam_app/edex_reader.py` (already reads per-folder `<folder>.tar` archives).

Do not reintroduce gzip: provisioning uses uncompressed `.tar` to cap memory on the provisioning runner. If packing changes, change extraction in the same MR.

## Task: control the PR vs nightly matrix

- Nightly configs: `nightly.yml` `strategy.matrix.include`. Eval runs on entries flagged `eval: true` (currently the four x86 configs). Every eval-enabled config needs the `RUNNER_STORAGE_ROOT` mount and configured repo secrets/variables; the `cuvslam-ci:local` image supplies the AWS CLI.
- Jetson CUDA micro-benchmarks run only on nightly entries flagged `benchmark: true` (currently Orin and Thor). The normal C++ test invocation continues to exclude `*SpeedUp*` and `*Speedup*`; the dedicated benchmark wrapper runs the positive filter and excludes `DISABLED_` tests.
- PR config: `pr-verify.yml` runs eval only on `build-test-x86` (fork-gated). `EVAL_CONFIG` is the static slug label for the PR table.
- Active dataset set: `run_eval.sh` reads every record from `dataset_registry eval-records`, so PR and nightly run the same set. There is no per-pipeline selection today. To differ, filter the records by suite in the registry and have each workflow pass the selector; `EvalSpec.suites` already carries the membership.

## Task: preserve nightly version provenance

`VERSION` controls the package filename, while `get_version()` is generated independently as
`MAJOR.MINOR.PATCH+<short-git-sha>[-modified]`. The `-modified` suffix means the build container's
tracked worktree differs from `HEAD`.

1. Keep `git-lfs` installed and configured system-wide in `scripts/Dockerfile`. Nightly pulls LFS objects on the host,
   and Git without the LFS clean filter misidentifies the materialized files as source modifications.
2. Do not edit tracked files before the C++ build. Pass runner-specific settings such as the Ubuntu Ports mirror
   through Docker build arguments instead.
3. Keep `CUVSLAM_REQUIRE_CLEAN_SOURCE=1` on nightly C++ builds. It checks the source using the same image and Git/LFS
   configuration that generate the version header.
4. Pass the expected package version and checked-out full Git SHA to `verify_pycuvslam_wheel_in_docker.sh`. The
   verifier must reject `-modified` and a mismatched embedded revision before artifacts are uploaded.
5. When changing checkout, LFS, Docker build, or version logic, test both an LFS-materialized clean checkout and an
   intentional tracked edit.

## Task: build a draft release

1. Create or update a branch named `release/vMAJOR.MINOR[.PATCH][-SUFFIX]`.
2. Manually dispatch `Nightly Build & Test` from that branch. Release dispatches always build, even without commits in the last 24 hours.
3. After every matrix job and evaluation succeeds, the workflow validates the branch version against `VERSION`, derives the tag from the branch (`release/v17.0` -> `v17.0`), and creates a draft Release containing the already-built C++ archives, wheels, documentation, and permanent evaluation bundle.
4. Review the draft and publish it manually. A draft/published Release or Git tag with the same version is never overwritten.
5. To rebuild an unpublished release after fixes, explicitly delete the old draft and dispatch the updated release branch again.

Scheduled nightlies publish only versioned 30-day Actions artifacts. They never create or update a GitHub Release.

## Hard rules

Detail in [reference.md](reference.md). The load-bearing ones:

- Dataset and eval steps stay fork-gated (`if: ... head.repo == github.repository`); never run fork code on dataset runners.
- Eval uses the read-only `AWS_S3_RO_*` secrets; only `provision-datasets.yml` uses the read-write `AWS_S3_*` pair.
- KPI history directories and eval artifact names carry the `platform-cuda-ubuntu` slug so matrix configs never overwrite each other.
- Jetson benchmark artifacts carry the same `platform-cuda-ubuntu` slug; Orin and Thor results are reported independently and are never averaged together.
- Nightly distributables must report `VERSION+<short-checked-out-sha>` without `-modified`.
- CODEOWNERS and CI workflow changes go in their own `[infra]` MR (enforced by the `isolated-ruleset-change` pre-commit hook). The hook's protected list is the authority on which paths qualify.
