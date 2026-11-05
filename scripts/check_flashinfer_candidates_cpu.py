#!/usr/bin/env python3
"""Validate FlashInfer candidates without importing any device runtime.

Runs only focused synthetic CPU suites. Offline Metal compiler tests may run;
no compiler result establishes native execution or numerical equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    "tests/test_qwen4_qsa_selective_cpu.py",
    "tests/test_arithmetic_class_cpu.py",
    "tests/test_flashinfer_gdn_replay_cpu.py",
    "tests/test_qwen38_tensorfold_coupled_laws_host.py",
    "tests/test_flash_next_policy_moe_conflicts.py",
    "tests/test_flash_next_qsa_selector_policy_cpu.py",
)
SOURCES = (
    "scripts/check_flashinfer_candidates_cpu.py",
    "pyproject.toml",
    "src/mlx2/runtime/arithmetic_class.py",
    "src/mlx2/runtime/models/qwen4_qsa_selective.py",
    "src/mlx2/runtime/models/qwen4_qsa_selective.metal",
    "src/mlx2/runtime/models/qwen4_qsa_stage1.py",
    "src/mlx2/runtime/models/qwen4_qsa_gvr_reference.py",
    "src/mlx2/runtime/lane/installer.py",
    "src/mlx2/runtime/lane/matmul.py",
    "src/mlx2/runtime/lane/simd.py",
    "src/mlx2/runtime/lane/tiled_prefill.py",
    "src/mlx2/adapters/flash_next_policy.py",
    "src/mlx2/adapters/flash_next.py",
    "src/mlx2/adapters/qwen38_tensorfold_source.py",
    "src/mlx2/adapters/qwen38_replay_fold.py",
    "src/mlx2/experimental/flashinfer_gdn_replay.py",
    "src/mlx2/experimental/flashinfer_gdn_replay.metal",
    "provenance/flashinfer-selective-qsa-20261104.json",
    "provenance/flashinfer-arithmetic-class-20261104.json",
    "provenance/flashinfer-gdn-replay-20261104.json",
    "scripts/validate_qwen38_tensorfold_coupled_laws.py",
    "tests/mlx_blocker.py",
) + SUITES

IMPORT_GUARD = '''\
import importlib.abc
import sys

class DeviceRuntimeForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"mlx", "mlx_lm", "torch", "coremltools"}:
            raise RuntimeError("Device runtime import forbidden by CPU candidate gate: " + fullname)
        return None

sys.meta_path.insert(0, DeviceRuntimeForbidden())
'''


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE
    ).strip()


def validate(evidence: Path, *, timeout: int = 300) -> dict:
    evidence = evidence.expanduser().resolve()
    if evidence.exists() and (not evidence.is_dir() or any(evidence.iterdir())):
        raise ValueError(f"evidence destination must be empty: {evidence}")
    missing = [name for name in SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise ValueError(f"candidate source is incomplete: {', '.join(missing)}")
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
    }
    started = datetime.now(UTC).isoformat()
    revision = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", *SOURCES))
    command = [
        sys.executable, "-m", "pytest", "--noconftest", "-p", "no:cacheprovider",
        "-o", "addopts=", "-q", f"--junitxml={evidence / 'pytest.xml'}", *SUITES,
    ]
    with tempfile.TemporaryDirectory(prefix="mlx2-cpu-import-guard-") as temporary:
        guard = Path(temporary)
        (guard / "sitecustomize.py").write_text(IMPORT_GUARD)
        env = dict(os.environ)
        env.update(
            PYTHONPATH=os.pathsep.join((str(guard), str(ROOT / "src"))),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            MLX2_PREFLIGHT_CPU_ONLY="1",
        )
        timed_out = False
        try:
            run = subprocess.run(
                command, cwd=ROOT, env=env, text=True, capture_output=True,
                timeout=timeout, check=False,
            )
            stdout, stderr, code = run.stdout, run.stderr, run.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
            code = 124
    (evidence / "stdout.txt").write_text(stdout)
    (evidence / "stderr.txt").write_text(stderr)
    final_manifest = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
    }
    stable = manifest == final_manifest and git("rev-parse", "HEAD") == revision
    junit = evidence / "pytest.xml"
    test_counts = None
    junit_error = None
    if junit.is_file():
        try:
            tree = ET.parse(junit)
            suites = [tree.getroot()] if tree.getroot().tag == "testsuite" else list(tree.getroot())
            test_counts = {
                field: sum(int(suite.get(field, "0")) for suite in suites)
                for field in ("tests", "failures", "errors", "skipped")
            }
        except (ET.ParseError, ValueError) as error:
            junit_error = str(error)
    receipt = {
        "schema": "mlx2.flashinfer-cpu-candidates.v1",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "checkpoint": "none; synthetic CPU arrays only",
        "source_revision": revision,
        "working_tree_changes": dirty,
        "source_sha256": manifest,
        "source_stable_during_validation": stable,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pytest")},
        "command": command,
        "test_counts": test_counts,
        "junit_error": junit_error,
        "exit_code": code,
        "timed_out": timed_out,
        "passed": code == 0 and stable and bool(test_counts) and junit_error is None,
        "device_runtime_imports": "mlx, mlx_lm, torch and coremltools blocked",
        "gpu_used": False,
        "ane_used": False,
        "native_execution": False,
        "qualification": "none",
        "performance_claim": None,
        "limits": [
            "Synthetic CPU correctness and offline compiler evidence only.",
            "No model checkpoint, native route engagement or native numerical parity tested.",
        ],
    }
    target = evidence / "receipt.json"
    temporary = evidence / ".receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(target)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    result = validate(args.evidence_dir, timeout=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
