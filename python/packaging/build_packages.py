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

"""Stage and validate the public cuVSLAM Python distributions."""

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set
import zipfile

_PUBLIC_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:(?:a|b|rc)[0-9]+|\.post[0-9]+|\.dev[0-9]+)?$")


def _load_config(source_root: Path) -> Mapping[str, object]:
    config_path = source_root / "python" / "packaging" / "packages.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_version(source_root: Path) -> str:
    version = (source_root / "VERSION").read_text(encoding="utf-8").strip()
    if not _PUBLIC_VERSION.fullmatch(version):
        raise ValueError(
            "VERSION must be a public MAJOR.MINOR.PATCH PEP 440 version without "
            f"a local '+...' label, got {version!r}"
        )
    return version


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_array(values: Iterable[str]) -> str:
    items = list(values)
    if not items:
        return "[]"
    return "[\n" + "".join(f"    {_toml_string(item)},\n" for item in items) + "]"


def _project_metadata(
    config: Mapping[str, object],
    name: str,
    version: str,
    dependencies: Iterable[str],
) -> str:
    project = config["project"]
    assert isinstance(project, dict)
    authors = project["authors"]
    assert isinstance(authors, list)
    author_entries = ", ".join(
        f"{{name = {_toml_string(str(author['name']))}}}" for author in authors
    )
    classifiers = project["classifiers"]
    urls = project["urls"]
    assert isinstance(classifiers, list)
    assert isinstance(urls, dict)

    lines = [
        "[project]",
        f"name = {_toml_string(name)}",
        f"version = {_toml_string(version)}",
        f"description = {_toml_string(str(project['description']))}",
        'readme = {file = "README.md", content-type = "text/markdown"}',
        f"authors = [{author_entries}]",
        f"license = {_toml_string(str(project['license-expression']))}",
        'license-files = ["LICENSE", "THIRD-PARTY.txt"]',
        f"requires-python = {_toml_string(str(project['requires-python']))}",
        f"classifiers = {_toml_array(str(item) for item in classifiers)}",
        f"dependencies = {_toml_array(dependencies)}",
        "",
        "[project.urls]",
    ]
    lines.extend(f"{key} = {_toml_string(str(value))}" for key, value in sorted(urls.items()))
    return "\n".join(lines) + "\n"


def render_backend_pyproject(source_root: Path, cuda_major: str) -> str:
    config = _load_config(source_root)
    version = _load_version(source_root)
    backends = config["backends"]
    common = config["common"]
    assert isinstance(backends, dict)
    assert isinstance(common, dict)
    if cuda_major not in backends:
        raise ValueError(f"Unsupported CUDA major {cuda_major!r}")
    backend = backends[cuda_major]
    assert isinstance(backend, dict)

    dependencies = [
        f"{common['name']}=={version}",
        str(backend["runtime-dependency"]),
    ]
    metadata = _project_metadata(config, str(backend["name"]), version, dependencies)
    build = f"""
[build-system]
requires = ["scikit-build-core>=1.0.3,<2", "nanobind>=2.10.2,<3"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
minimum-version = "build-system.requires"
cmake.version = "CMakeLists.txt"
cmake.args = ["-G", "Unix Makefiles", "-DCMAKE_MAKE_PROGRAM=make"]
build-dir = "build/{{wheel_tag}}"
wheel.py-api = "cp312"
cmake.source-dir = "."
wheel.packages = []
wheel.install-dir = "cuvslam"

[tool.scikit-build.cmake.define]
CUDAToolkit_ROOT = {{env = "CUDA_HOME", default = "/usr/local/cuda"}}
CUVSLAM_PYTHON_BACKEND = "cu{cuda_major}"
"""
    return "[build-system]\n" + build.split("[build-system]\n", 1)[1] + "\n" + metadata


def _pure_pyproject(
    config: Mapping[str, object],
    name: str,
    version: str,
    dependencies: Iterable[str],
    has_package: bool,
) -> str:
    build = """[build-system]
requires = ["setuptools>=77,<85", "wheel>=0.45,<1"]
build-backend = "setuptools.build_meta"

"""
    metadata = _project_metadata(config, name, version, dependencies)
    if has_package:
        setuptools = """
[tool.setuptools]
package-dir = {"" = "src"}
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
cuvslam = ["*.pyi", "py.typed"]
"""
    else:
        setuptools = """
[tool.setuptools]
packages = []
"""
    return build + metadata + setuptools


def _prepare_empty_stage(stage_dir: Path) -> None:
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError(f"Staging directory is not empty: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)


def _copy_release_documents(source_root: Path, destination: Path) -> None:
    for relative_path in ("LICENSE", "THIRD-PARTY.txt"):
        shutil.copy2(source_root / relative_path, destination / relative_path)
    shutil.copy2(source_root / "python" / "README.md", destination / "README.md")


def render_backend(source_root: Path, cuda_major: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _copy_release_documents(source_root, output.parent)
    output.write_text(render_backend_pyproject(source_root, cuda_major), encoding="utf-8")


def _read_generated_stub(backend_wheel: Path) -> str:
    with zipfile.ZipFile(backend_wheel) as wheel:
        matches = [
            name
            for name in wheel.namelist()
            if re.fullmatch(r"cuvslam/cu(?:12|13)/pycuvslam\.pyi", name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one generated pycuvslam.pyi in {backend_wheel}, found {matches}"
            )
        return wheel.read(matches[0]).decode("utf-8")


def stage_common(source_root: Path, backend_wheel: Path, stage_dir: Path) -> None:
    _prepare_empty_stage(stage_dir)
    config = _load_config(source_root)
    version = _load_version(source_root)
    common = config["common"]
    assert isinstance(common, dict)

    package_dir = stage_dir / "src" / "cuvslam"
    package_dir.mkdir(parents=True)
    for filename in ("__init__.py", "_backend.py", "utils.py"):
        shutil.copy2(source_root / "python" / filename, package_dir / filename)
    (package_dir / "py.typed").write_text("", encoding="utf-8")
    public_stub = _read_generated_stub(backend_wheel)
    public_stub += "\nfrom . import utils as utils\n\n__version__: str\n"
    (package_dir / "__init__.pyi").write_text(public_stub, encoding="utf-8")

    dependencies = common["dependencies"]
    assert isinstance(dependencies, list)
    pyproject = _pure_pyproject(
        config,
        str(common["name"]),
        version,
        (str(item) for item in dependencies),
        has_package=True,
    )
    (stage_dir / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    _copy_release_documents(source_root, stage_dir)


def stage_meta(source_root: Path, stage_dir: Path) -> None:
    _prepare_empty_stage(stage_dir)
    config = _load_config(source_root)
    version = _load_version(source_root)
    default = config["default"]
    backends = config["backends"]
    assert isinstance(default, dict)
    assert isinstance(backends, dict)
    default_backend = backends[str(default["backend"])]
    assert isinstance(default_backend, dict)

    dependency = f"{default_backend['name']}=={version}"
    pyproject = _pure_pyproject(
        config,
        str(default["name"]),
        version,
        [dependency],
        has_package=False,
    )
    (stage_dir / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    _copy_release_documents(source_root, stage_dir)


def backend_rpath(source_root: Path, cuda_major: str) -> str:
    config = _load_config(source_root)
    backends = config["backends"]
    assert isinstance(backends, dict)
    backend = backends[cuda_major]
    assert isinstance(backend, dict)
    rpaths = backend["site-packages-rpaths"]
    assert isinstance(rpaths, list)
    return ":".join(str(path) for path in rpaths)


def _wheel_distribution(wheel: zipfile.ZipFile) -> str:
    metadata_files = [
        name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_files) != 1:
        raise ValueError(f"Expected one METADATA file, found {metadata_files}")
    metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))
    name = metadata.get("Name")
    if not name:
        raise ValueError(f"METADATA has no Name field: {metadata_files[0]}")
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_metadata(wheel_path: Path):
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_files = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"Expected one METADATA file in {wheel_path}, found {metadata_files}"
            )
        metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"METADATA in {wheel_path} must contain Name and Version")
    return re.sub(r"[-_.]+", "-", name).lower(), version


def _payload_path_is_allowed(distribution: str, path: str) -> bool:
    if distribution == "cuvslam":
        return False
    if distribution == "cuvslam-common":
        return path in {
            "cuvslam/__init__.py",
            "cuvslam/__init__.pyi",
            "cuvslam/_backend.py",
            "cuvslam/py.typed",
            "cuvslam/utils.py",
        }
    if distribution in {"cuvslam-cu12", "cuvslam-cu13"}:
        cuda_major = distribution.removeprefix("cuvslam-cu")
        return path.startswith(
            f"cuvslam/cu{cuda_major}/"
        ) or path.startswith(f"cuvslam_cu{cuda_major}.libs/")
    return False


def verify_ownership(wheel_paths: Sequence[Path]) -> None:
    paths_by_distribution: Dict[str, Set[str]] = {}
    invalid_paths: List[str] = []
    for wheel_path in wheel_paths:
        with zipfile.ZipFile(wheel_path) as wheel:
            distribution = _wheel_distribution(wheel)
            owned = {
                name
                for name in wheel.namelist()
                if not name.endswith("/") and ".dist-info/" not in name
            }
        invalid_paths.extend(
            f"{distribution}: {path}"
            for path in sorted(owned)
            if not _payload_path_is_allowed(distribution, path)
        )
        paths_by_distribution.setdefault(distribution, set()).update(owned)

    if invalid_paths:
        raise ValueError(
            "Wheel payload paths violate distribution ownership:\n"
            + "\n".join(invalid_paths)
        )

    collisions: List[str] = []
    distributions = sorted(paths_by_distribution)
    for index, left in enumerate(distributions):
        for right in distributions[index + 1 :]:
            overlap = sorted(paths_by_distribution[left] & paths_by_distribution[right])
            collisions.extend(f"{left} and {right}: {path}" for path in overlap)
    if collisions:
        raise ValueError("Wheel file ownership collisions:\n" + "\n".join(collisions))


def release_manifest(
    wheel_paths: Sequence[Path],
    output: Path,
    default_size_limit: int,
    size_limits: Mapping[str, int],
    expected_version: str = "",
    expected_counts: Optional[Mapping[str, int]] = None,
) -> None:
    entries = []
    oversized = []
    actual_counts: Dict[str, int] = {}
    for wheel_path in sorted(wheel_paths):
        distribution, version = _wheel_metadata(wheel_path)
        actual_counts[distribution] = actual_counts.get(distribution, 0) + 1
        if expected_version and version != expected_version:
            raise ValueError(
                f"{wheel_path.name}: version {version} does not match {expected_version}"
            )
        if "+" in version:
            raise ValueError(
                f"{wheel_path.name}: public PyPI version must not contain a local label"
            )
        size = wheel_path.stat().st_size
        size_limit = size_limits.get(distribution, default_size_limit)
        if size > size_limit:
            oversized.append(
                f"{wheel_path.name}: {size} bytes exceeds {distribution} limit "
                f"of {size_limit} bytes"
            )
        digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        entries.append(
            {
                "distribution": distribution,
                "filename": wheel_path.name,
                "sha256": digest,
                "size": size,
                "size_limit": size_limit,
                "version": version,
            }
        )
    if oversized:
        raise ValueError(
            "PyPI file-size gate failed. Reduce the wheel or configure only an "
            "approved per-project exemption:\n" + "\n".join(oversized)
        )
    if expected_counts and actual_counts != dict(expected_counts):
        raise ValueError(
            f"Wheel distribution counts are {actual_counts}, expected {dict(expected_counts)}"
        )
    verify_ownership(wheel_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"artifacts": entries}, indent=2) + "\n", encoding="utf-8")


def _size_limit(value: str):
    distribution, separator, limit = value.partition("=")
    if not separator or not distribution or not limit.isdigit():
        raise argparse.ArgumentTypeError("size limit must be PROJECT=BYTES")
    return re.sub(r"[-_.]+", "-", distribution).lower(), int(limit)


def _distribution_count(value: str):
    distribution, separator, count = value.partition("=")
    if not separator or not distribution or not count.isdigit():
        raise argparse.ArgumentTypeError("distribution count must be PROJECT=COUNT")
    return re.sub(r"[-_.]+", "-", distribution).lower(), int(count)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render-backend")
    render.add_argument("--source-root", type=Path, required=True)
    render.add_argument("--cuda-major", choices=("12", "13"), required=True)
    render.add_argument("--output", type=Path, required=True)

    common = subparsers.add_parser("stage-common")
    common.add_argument("--source-root", type=Path, required=True)
    common.add_argument("--backend-wheel", type=Path, required=True)
    common.add_argument("--stage-dir", type=Path, required=True)

    meta = subparsers.add_parser("stage-meta")
    meta.add_argument("--source-root", type=Path, required=True)
    meta.add_argument("--stage-dir", type=Path, required=True)

    rpath = subparsers.add_parser("backend-rpath")
    rpath.add_argument("--source-root", type=Path, required=True)
    rpath.add_argument("--cuda-major", choices=("12", "13"), required=True)

    ownership = subparsers.add_parser("verify-ownership")
    ownership.add_argument("wheels", nargs="+", type=Path)

    manifest = subparsers.add_parser("release-manifest")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--default-size-limit", type=int, default=100 * 1024 * 1024)
    manifest.add_argument("--expected-version", default="")
    manifest.add_argument(
        "--expected-distribution",
        action="append",
        default=[],
        type=_distribution_count,
        help="Required wheel count as PROJECT=COUNT",
    )
    manifest.add_argument(
        "--size-limit",
        action="append",
        default=[],
        type=_size_limit,
        help="Approved per-project override as PROJECT=BYTES",
    )
    manifest.add_argument("wheels", nargs="+", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "render-backend":
        render_backend(args.source_root, args.cuda_major, args.output)
    elif args.command == "stage-common":
        stage_common(args.source_root, args.backend_wheel, args.stage_dir)
    elif args.command == "stage-meta":
        stage_meta(args.source_root, args.stage_dir)
    elif args.command == "backend-rpath":
        print(backend_rpath(args.source_root, args.cuda_major))
    elif args.command == "verify-ownership":
        verify_ownership(args.wheels)
    elif args.command == "release-manifest":
        release_manifest(
            args.wheels,
            args.output,
            args.default_size_limit,
            dict(args.size_limit),
            args.expected_version,
            dict(args.expected_distribution),
        )


if __name__ == "__main__":
    main()
