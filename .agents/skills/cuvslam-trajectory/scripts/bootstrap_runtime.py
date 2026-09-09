#!/usr/bin/env python3

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

"""Create a replay venv with a matching released cuVSLAM wheel and cuvslam-tools."""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def _cuda_major() -> int:
    candidates = [Path("/usr/local/cuda/version.json"), Path("/usr/local/cuda/version.txt")]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'"version"\s*:\s*"?(\d+)|CUDA Version\s+(\d+)', text)
        if match:
            return int(next(group for group in match.groups() if group is not None))

    # Runtime-only CUDA installations often omit nvcc and version files. The
    # canonical symlink still resolves to a versioned directory in those images.
    cuda_root = Path("/usr/local/cuda")
    directory_names = []
    if cuda_root.exists():
        directory_names.extend((cuda_root.name, cuda_root.resolve().name))
    directory_names.extend(path.name for path in Path("/usr/local").glob("cuda-*"))
    for name in directory_names:
        match = re.fullmatch(r"cuda-(\d+)(?:\.\d+)?", name)
        if match:
            return int(match.group(1))
    runtime_library = ctypes.util.find_library("cudart") or ""
    match = re.search(r"libcudart\.so\.(\d+)", runtime_library)
    if match:
        return int(match.group(1))
    try:
        output = _run(["nvcc", "--version"], capture=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        output = ""
    match = re.search(r"release\s+(\d+)", output)
    if not match:
        raise RuntimeError(
            "CUDA runtime or Toolkit was not found; install CUDA 12 or 13 before cuVSLAM"
        )
    return int(match.group(1))


def _glibc_tag() -> str:
    _, version = platform.libc_ver()
    match = re.match(r"(\d+)\.(\d+)", version)
    if not match:
        raise RuntimeError(f"could not determine glibc version from {version!r}")
    return f"manylinux_{match.group(1)}_{match.group(2)}"


def _python_tag() -> str:
    version = sys.version_info
    if version[:2] == (3, 10):
        return "cp310"
    if version >= (3, 12):
        return "cp312-abi3"
    raise RuntimeError(
        f"released wheels do not support Python {version.major}.{version.minor}; "
        "use Python 3.10 on Ubuntu 22.04 or Python 3.12+ on Ubuntu 24.04"
    )


def _architecture() -> str:
    machine = platform.machine().lower()
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    machine = aliases.get(machine, machine)
    if machine not in {"x86_64", "aarch64"}:
        raise RuntimeError(f"no released cuVSLAM wheel is known for architecture {machine}")
    return machine


def _repo_version(repo: Path) -> str:
    version_path = repo / "VERSION"
    if not version_path.is_file():
        raise RuntimeError(f"not a cuVSLAM repository (missing VERSION): {repo}")
    return version_path.read_text(encoding="utf-8").strip()


def _runtime_spec(repo: Path, cuda_major: int | None = None) -> dict[str, Any]:
    version = _repo_version(repo)
    return {
        "version": version,
        "tag": f"v{version}",
        "cuda_tag": f"cu{cuda_major or _cuda_major()}",
        "python_tag": _python_tag(),
        "platform_tag": _glibc_tag(),
        "architecture": _architecture(),
    }


def _release_asset(spec: dict[str, Any]) -> tuple[str, str]:
    api_url = f"https://api.github.com/repos/nvidia-isaac/cuVSLAM/releases/tags/{spec['tag']}"
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "cuvslam-trajectory-skill"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            release = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read cuVSLAM release metadata: {exc}") from exc

    host_platform = re.fullmatch(r"manylinux_(\d+)_(\d+)", spec["platform_tag"])
    if host_platform is None:
        raise RuntimeError(f"invalid host platform tag: {spec['platform_tag']}")
    host_glibc = tuple(int(value) for value in host_platform.groups())
    matches = []
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        wheel_platform = re.search(r"manylinux_(\d+)_(\d+)_", name)
        if wheel_platform is None:
            continue
        wheel_glibc = tuple(int(value) for value in wheel_platform.groups())
        required = (
            name.endswith(f"_{spec['architecture']}.whl"),
            f"+{spec['cuda_tag']}-" in name,
            f"-{spec['python_tag']}-" in name,
            wheel_glibc <= host_glibc,
        )
        if name.startswith(f"cuvslam-{spec['version']}") and all(required):
            matches.append((wheel_glibc, name, asset["browser_download_url"]))
    if matches:
        newest_compatible = max(match[0] for match in matches)
        matches = [match for match in matches if match[0] == newest_compatible]
    if len(matches) != 1:
        available = ", ".join(
            asset.get("name", "")
            for asset in release.get("assets", [])
            if asset.get("name", "").endswith(".whl")
        )
        raise RuntimeError(
            "expected exactly one matching wheel for "
            f"{spec['python_tag']}/{spec['cuda_tag']}/{spec['platform_tag']}/{spec['architecture']}; "
            f"found {len(matches)}. Available wheels: {available or 'none'}"
        )
    _, name, url = matches[0]
    return name, url


def _healthy(python: Path, expected_version: str) -> bool:
    if not python.is_file():
        return False
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import cuvslam, cuvslam_tools; "
                "version = cuvslam.get_version(); "
                f"assert version[0].split('+', 1)[0] == {expected_version!r}, version; "
                "print(version)"
            ),
        ],
        text=True,
        capture_output=True,
    )
    if probe.returncode == 0:
        print(probe.stdout.strip())
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="cuVSLAM repository root")
    parser.add_argument("--venv", type=Path, help="venv path; defaults to <repo>/.venv-trajectory")
    parser.add_argument("--wheel", type=Path, help="use this local wheel instead of a release asset")
    parser.add_argument("--cuda-major", type=int, choices=[12, 13], help="override detected CUDA major")
    parser.add_argument("--check", action="store_true", help="only check whether the venv is ready")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected runtime without changing files",
    )
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    venv = (args.venv or repo / ".venv-trajectory").expanduser().resolve()
    python = venv / "bin" / "python"
    try:
        expected_version = _repo_version(repo)
    except RuntimeError as exc:
        parser.exit(1, f"Runtime selection failed: {exc}\n")
    if _healthy(python, expected_version):
        print(f"Replay runtime is ready: {python}")
        return 0
    if args.check:
        parser.exit(1, f"Replay runtime is not ready: {python}\n")

    try:
        spec = _runtime_spec(repo, args.cuda_major)
    except RuntimeError as exc:
        parser.exit(1, f"Runtime selection failed: {exc}\n")
    plan = {"repo": str(repo), "venv": str(venv), **spec}
    if args.wheel:
        wheel = args.wheel.expanduser().resolve()
        if not wheel.is_file():
            parser.error(f"wheel does not exist: {wheel}")
        wheel_source = str(wheel)
        plan["wheel"] = wheel_source
    elif args.dry_run:
        plan["wheel"] = "matching GitHub release asset (resolved during installation)"
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    else:
        try:
            wheel_name, wheel_source = _release_asset(spec)
        except RuntimeError as exc:
            parser.exit(1, f"Wheel selection failed: {exc}\n")
        plan["wheel"] = wheel_name

    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    try:
        # Re-running venv is safe and repairs missing launchers after an interrupted
        # creation. Pip then repairs partial package installations in place.
        _run([sys.executable, "-m", "venv", str(venv)])
        _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(python), "-m", "pip", "install", "-e", str(repo / "tools" / "python_tools")])
        _run([str(python), "-m", "pip", "install", "--force-reinstall", wheel_source])
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Runtime installation failed: {exc}\n")
    if not _healthy(python, expected_version):
        parser.exit(1, f"Installed runtime failed its import check: {python}\n")
    print(json.dumps(plan, indent=2, sort_keys=True))
    print(f"Replay runtime is ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
