#!/usr/bin/env python3
"""Emit machine-readable agent-readiness validation evidence for cuVSLAM."""

from __future__ import annotations

import argparse
import compileall
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRODUCT = "cuVSLAM"
SCHEMA_VERSION = "1.0"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_path = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    checks = _static_checks(repo_root)
    if args.mode == "gpu":
        checks.extend(_gpu_checks(repo_root))

    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "product": PRODUCT,
        "mode": args.mode,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo": {
            "commit": _git_commit(repo_root),
            "root": str(repo_root),
        },
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path} status={status} checks={len(checks)}")
    return 0 if status == "pass" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cuVSLAM agent-readiness smoke validation.")
    parser.add_argument(
        "--mode",
        choices=("static", "gpu"),
        default="static",
        help="static checks repo readiness without CUDA; gpu also imports cuvslam and runs Python API tests.",
    )
    parser.add_argument(
        "--output",
        default="reports/validation_result.json",
        help="Path for the validation_result.json artifact.",
    )
    return parser


def _static_checks(repo_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required_files = [
        "README.md",
        "AGENTS.md",
        "llms.txt",
        "agent-readiness.yaml",
        ".env_example",
        ".agent/validation_result.schema.json",
        ".codex/skills/cuvslam-agent-readiness/SKILL.md",
        ".github/workflows/agent-readiness.yml",
    ]
    for relpath in required_files:
        path = repo_root / relpath
        checks.append(
            _check(
                f"required_file:{relpath}",
                path.exists(),
                f"found {relpath}" if path.exists() else f"missing {relpath}",
            )
        )

    test_files = sorted((repo_root / "tests").glob("test_*.py")) if (repo_root / "tests").exists() else []
    checks.append(_check("root_tests", bool(test_files), f"found {len(test_files)} root test file(s)"))
    checks.append(_check("python_syntax", _compile_python(repo_root), "compiled python, scripts, and tests"))
    return checks


def _gpu_checks(repo_root: Path) -> list[dict[str, Any]]:
    import_result = _run([sys.executable, "-c", "import cuvslam; print(cuvslam.__file__)"], repo_root, 60)
    test_result = _run(
        [sys.executable, "-m", "unittest", "discover", "-s", "python/test", "-p", "test_*.py", "-v"],
        repo_root,
        600,
    )
    return [
        _check("cuvslam_import", import_result.returncode == 0, _command_detail(import_result)),
        _check("python_api_tests", test_result.returncode == 0, _command_detail(test_result)),
    ]


def _compile_python(repo_root: Path) -> bool:
    if "PYTHONPYCACHEPREFIX" not in os.environ:
        sys.pycache_prefix = str(repo_root / ".cache" / "pycache")
    paths = [repo_root / "python", repo_root / "scripts", repo_root / "tests"]
    return all(compileall.compile_dir(str(path), quiet=1) for path in paths if path.exists())


def _run(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    output = " ".join((result.stdout or result.stderr or "").split())
    return f"exit_code={result.returncode}" + (f" output={output[:300]}" if output else "")


def _git_commit(repo_root: Path) -> str | None:
    result = _run(["git", "rev-parse", "HEAD"], repo_root, 30)
    return result.stdout.strip() if result.returncode == 0 else None


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


if __name__ == "__main__":
    raise SystemExit(main())
