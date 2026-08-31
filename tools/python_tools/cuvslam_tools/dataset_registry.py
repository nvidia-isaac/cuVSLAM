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

"""Single source of truth for the evaluation datasets and their reporter runs.

One dataset ID is the only name written down. The preparation module, the private
tarball, the staged directory, and the ``/sequences`` mount all derive from it,
and ``prepare()`` returns the directory that provisioning archives, so the tar
root is always the dataset root.

An evaluation is identified by its reporter config filename. The KPI prefix is
derived from that name rather than declared, matching what
``scripts/cuvslam_kpi_report.py`` computes from the reporter's output directory:
the first hyphen-delimited token, upper-cased. Keep underscores inside a prefix
and use a hyphen only to terminate it, so ``tartan_flaky-vo_slam.cfg`` yields
``TARTAN_FLAKY`` while ``tartan-flaky-vo_slam.cfg`` would collide with
``TARTAN``.

This module is standard library only and imports preparation code lazily, so
``python3 -m cuvslam_tools.dataset_registry`` works with ``PYTHONPATH`` alone
inside the CI image, before anything is installed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

SUITES = ("smoke", "full")
GATING_POLICIES = ("hard", "soft", "informational")

# Values cuvslam_app accepts for --odometry_mode. cuvslam_kpi_report.py maps each
# to a distinct KPI type, and anything unrecognized there falls back to STEREO,
# so an unknown mode here would silently mislabel a KPI key.
ODOMETRY_MODES = ("multicamera", "mono", "inertial", "rgbd")

_DATASET_ID = re.compile(r"^[a-z0-9_]+$")
_KPI_PREFIX = re.compile(r"^[A-Z0-9_]+$")


class RegistryError(ValueError):
    """Raised when the registry, or a staged dataset, violates the contract."""


@dataclass(frozen=True)
class EvalSpec:
    """One reporter run: which config, which flags, which suites, how it gates."""

    config: str
    args: tuple[str, ...]
    suites: frozenset[str]
    gating: str = "soft"

    @property
    def kpi_prefix(self) -> str:
        """Dataset prefix the KPI collector will derive from this config name."""
        return Path(self.config).stem.split("-")[0].upper()

    @property
    def odometry_modes(self) -> tuple[str, ...]:
        """Every --odometry_mode value in the argument tuple."""
        return tuple(a.split("=", 1)[1] for a in self.args if a.startswith("--odometry_mode="))

    @property
    def odometry_mode(self) -> str:
        """The --odometry_mode value, which decides the KPI type.

        Validation requires exactly one, so a repeated flag cannot make this
        disagree with the last-wins value cuvslam_app would actually use.
        """
        modes = self.odometry_modes
        return modes[0] if len(modes) == 1 else ""


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset: how to prepare it, and the reporter runs it feeds."""

    dataset_id: str
    prepare_module: str
    evals: tuple[EvalSpec, ...] = field(default_factory=tuple)

    @property
    def archive_name(self) -> str:
        """Private tarball basename."""
        return f"{self.dataset_id}.tar"

    @property
    def mount_name(self) -> str:
        """Staged directory and /sequences mount name.

        Equal to the dataset ID by contract, and must match the "dataset_folder"
        value the converter writes into each reporter config.
        """
        return self.dataset_id

    @property
    def is_eval_enabled(self) -> bool:
        return bool(self.evals)

    def load_prepare(self):
        """Import the preparation module and return its ``prepare`` callable.

        Deferred so listing or validating the registry never pulls in a
        converter's numerical dependencies.
        """
        module = importlib.import_module(self.prepare_module)
        prepare = getattr(module, "prepare", None)
        if not callable(prepare):
            raise RegistryError(f"{self.prepare_module} does not expose a callable prepare()")
        return prepare


def _stereo_args(*extra: str) -> tuple[str, ...]:
    return ("--odometry_mode=multicamera", *extra, "--async_sba=false", "--multicam_mode=moderate", "--use_segments")


DATASETS: dict[str, DatasetSpec] = {
    "kitti": DatasetSpec(
        dataset_id="kitti",
        prepare_module="cuvslam_tools.dataset_preparation.kitti.prepare",
        evals=(
            EvalSpec(
                config="kitti-vio_slam_gt.cfg",
                args=_stereo_args("--rectified_stereo_camera=true"),
                suites=frozenset(SUITES),
            ),
        ),
    ),
    "euroc": DatasetSpec(
        dataset_id="euroc",
        prepare_module="cuvslam_tools.dataset_preparation.euroc.prepare",
        evals=(
            EvalSpec(
                config="euroc-vio_slam.cfg",
                args=(
                    "--odometry_mode=inertial",
                    "--rectified_stereo_camera=false",
                    "--async_sba=false",
                    "--multicam_mode=moderate",
                    "--use_segments",
                ),
                suites=frozenset(SUITES),
            ),
        ),
    ),
    # Provisionable but not yet evaluated: no reporter configs are produced and no
    # validated tarball exists. Adding an EvalSpec is what enables a dataset.
    "tum": DatasetSpec(
        dataset_id="tum",
        prepare_module="cuvslam_tools.dataset_preparation.tum.prepare",
    ),
    "tartan": DatasetSpec(
        dataset_id="tartan",
        prepare_module="cuvslam_tools.dataset_preparation.tartan.prepare",
    ),
}


def dataset(dataset_id: str) -> DatasetSpec:
    """Return one dataset spec, or raise naming the known IDs."""
    try:
        return DATASETS[dataset_id]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise RegistryError(f"unknown dataset '{dataset_id}' (known: {known})") from None


def provisionable_datasets() -> list[DatasetSpec]:
    """All datasets, in declaration order, which is also evaluation order."""
    return list(DATASETS.values())


def eval_datasets(suite: str | None = None) -> list[DatasetSpec]:
    """Datasets with at least one reporter run, optionally filtered by suite."""
    return [spec for spec in DATASETS.values() if eval_records(spec, suite)]


def eval_records(spec: DatasetSpec, suite: str | None = None) -> list[EvalSpec]:
    if suite is None:
        return list(spec.evals)
    if suite not in SUITES:
        raise RegistryError(f"unknown suite '{suite}' (known: {', '.join(SUITES)})")
    return [record for record in spec.evals if suite in record.suites]


def reporter_config_path(spec: DatasetSpec, record: EvalSpec) -> str:
    """Reporter config path relative to CUVSLAM_DATASETS."""
    return f"{spec.mount_name}/{record.config}"


def _validate_eval(spec: DatasetSpec, record: EvalSpec) -> None:
    where = f"{spec.dataset_id} eval '{record.config}'"
    if not record.config or Path(record.config).name != record.config:
        raise RegistryError(f"{where}: config must be a bare filename at the dataset root")
    if not record.config.endswith(".cfg"):
        raise RegistryError(f"{where}: config must end with .cfg")
    if not _KPI_PREFIX.match(record.kpi_prefix):
        raise RegistryError(
            f"{where}: derived KPI prefix '{record.kpi_prefix}' is not upper-case alphanumeric; "
            "keep underscores inside the prefix and use a hyphen only to terminate it"
        )
    if not record.args or not all(isinstance(a, str) and a.startswith("--") for a in record.args):
        raise RegistryError(f"{where}: args must be a non-empty tuple of --flag strings")
    modes = record.odometry_modes
    if len(modes) != 1:
        raise RegistryError(
            f"{where}: expected exactly one --odometry_mode flag, found {len(modes)}. "
            "cuvslam_app takes the last value, so a repeat would run a mode the registry did not check"
        )
    if modes[0] not in ODOMETRY_MODES:
        raise RegistryError(
            f"{where}: --odometry_mode must be one of {', '.join(ODOMETRY_MODES)}; "
            "an unrecognized mode silently becomes a STEREO KPI type"
        )
    if not record.suites:
        raise RegistryError(f"{where}: must belong to at least one suite")
    unknown = sorted(record.suites - set(SUITES))
    if unknown:
        raise RegistryError(f"{where}: unknown suite(s) {', '.join(unknown)}")
    if record.gating not in GATING_POLICIES:
        raise RegistryError(f"{where}: gating must be one of {', '.join(GATING_POLICIES)}")


def _validate_kpi_identities() -> None:
    """Reject two records that would produce the same KPI keys.

    Always spans the whole registry: a collision is a property of a pair, so
    checking one dataset in isolation cannot see it.
    """
    seen: dict[tuple[str, str], str] = {}
    for spec in DATASETS.values():
        for record in spec.evals:
            key = (record.kpi_prefix, record.odometry_mode)
            origin = f"{spec.dataset_id}/{record.config}"
            if key in seen:
                raise RegistryError(
                    f"{origin} derives the same KPI identity as '{seen[key]}': prefix {key[0]} "
                    f"with mode {key[1]}. One would silently overwrite the other in the KPI report."
                )
            seen[key] = origin


def validate(dataset_ids: Iterable[str] | None = None) -> None:
    """Check the registry and raise on the first fault.

    ``dataset_ids`` narrows the per-dataset checks; KPI identities are compared
    across every dataset regardless.
    """
    for key, spec in DATASETS.items():
        if key != spec.dataset_id:
            raise RegistryError(f"registry key '{key}' does not match dataset_id '{spec.dataset_id}'")

    specs = [dataset(name) for name in dataset_ids] if dataset_ids else provisionable_datasets()
    for spec in specs:
        if not _DATASET_ID.match(spec.dataset_id):
            raise RegistryError(f"dataset id '{spec.dataset_id}' must be lower-case alphanumeric or underscore")
        # find_spec raises rather than returning None when a parent package is
        # missing, so both outcomes mean the same thing here.
        try:
            found = importlib.util.find_spec(spec.prepare_module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            raise RegistryError(f"{spec.dataset_id}: preparation module not found: {spec.prepare_module}")
        for record in spec.evals:
            _validate_eval(spec, record)

    _validate_kpi_identities()


def verify_staged(spec: DatasetSpec, root: Path) -> None:
    """Check that a staged dataset matches the registry's mount name.

    Each reporter config declares the directory it expects under
    CUVSLAM_DATASETS. If that disagrees with the dataset ID, the reporter looks
    for sequences in a directory staging never created.
    """
    expected = f"{spec.mount_name}/"
    for record in spec.evals:
        config_path = root / record.config
        if not config_path.is_file():
            raise RegistryError(f"{spec.dataset_id}: staged config not found: {config_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{spec.dataset_id}: {config_path} is not valid JSON: {exc}") from exc
        if not isinstance(config, dict):
            raise RegistryError(
                f"{spec.dataset_id}: {config_path} must hold a JSON object, got {type(config).__name__}"
            )
        declared = config.get("dataset_folder")
        if declared != expected:
            raise RegistryError(
                f"{spec.dataset_id}: {record.config} declares dataset_folder {declared!r}, "
                f"expected {expected!r}. Regenerate the dataset or rename it in the registry."
            )


def _print_records(suite: str | None) -> None:
    for spec in eval_datasets(suite):
        for record in eval_records(spec, suite):
            print(
                "\t".join(
                    (
                        spec.dataset_id,
                        record.kpi_prefix,
                        reporter_config_path(spec, record),
                        " ".join(record.args),
                    )
                )
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cuvslam_tools.dataset_registry",
        description="Inspect and validate the cuVSLAM evaluation dataset registry.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Check the registry contract.")
    validate_parser.add_argument("--dataset", action="append", default=None, help="Limit to one dataset; repeatable.")

    list_parser = subparsers.add_parser("list", help="Print dataset IDs, one per line.")
    list_parser.add_argument("--eval", action="store_true", help="Only datasets with reporter runs.")
    list_parser.add_argument("--suite", default=None, choices=SUITES, help="Filter reporter runs by suite.")

    records_parser = subparsers.add_parser(
        "eval-records", help="Print id, KPI prefix, config path, and flags as tab-separated fields."
    )
    records_parser.add_argument("--suite", default=None, choices=SUITES)

    module_parser = subparsers.add_parser("prepare-module", help="Print a dataset's preparation module path.")
    module_parser.add_argument("dataset")

    prepare_parser = subparsers.add_parser("prepare", help="Run a dataset's prepare() and report the archive root.")
    prepare_parser.add_argument("dataset")
    prepare_parser.add_argument("--raw-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--force-download", action="store_true")
    prepare_parser.add_argument(
        "--root-file",
        type=Path,
        required=True,
        help="File to write the archive root to, keeping stdout free for converter logs.",
    )

    staged_parser = subparsers.add_parser("verify-staged", help="Check a staged dataset against the registry.")
    staged_parser.add_argument("dataset")
    staged_parser.add_argument("--root", type=Path, required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            validate(args.dataset)
            scope = ", ".join(args.dataset) if args.dataset else f"{len(DATASETS)} datasets"
            print(f"dataset registry OK ({scope})")
        elif args.command == "list":
            specs = eval_datasets(args.suite) if (args.eval or args.suite) else provisionable_datasets()
            for spec in specs:
                print(spec.dataset_id)
        elif args.command == "eval-records":
            _print_records(args.suite)
        elif args.command == "prepare-module":
            print(dataset(args.dataset).prepare_module)
        elif args.command == "prepare":
            spec = dataset(args.dataset)
            validate([spec.dataset_id])
            root = spec.load_prepare()(
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                force_download=args.force_download,
            )
            args.root_file.write_text(f"{Path(root).resolve()}\n", encoding="utf-8")
        elif args.command == "verify-staged":
            verify_staged(dataset(args.dataset), args.root)
            print(f"staged {args.dataset} matches the registry")
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
