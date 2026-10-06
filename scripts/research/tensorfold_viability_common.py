"""Shared, import-safe helpers for the TensorFold viability microbenches.

Nothing in this module imports MLX or selects a device.  GPU execution remains
behind each driver's explicit ``--run-gpu`` gate and the repository's external
GPU ownership wrapper.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

LOCK_RECEIPTS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)


def require_matching_gpu_receipts(paths=LOCK_RECEIPTS) -> dict:
    """Return one owner receipt or refuse an unowned Metal measurement."""

    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise RuntimeError(
            "GPU benchmark requires matching ownership receipts from "
            f"scripts/run_with_gpu_locks.py; missing {missing}"
        )
    receipts = [json.loads(Path(path).read_text()) for path in paths]
    if receipts[0] != receipts[1] or not receipts[0].get("lease_id"):
        raise RuntimeError("GPU ownership receipts do not name one matching lease")
    return receipts[0]


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def distribution_snapshot(name: str, expected: str) -> dict:
    """Describe an installed distribution without importing its package."""

    try:
        observed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        observed = None
    return {
        "distribution": name,
        "expected": expected,
        "observed": observed,
        "present": observed is not None,
        "exact_match": observed == expected,
    }


def require_distribution(name: str, expected: str) -> dict:
    """Refuse a measurement if its runtime distribution has drifted."""

    snapshot = distribution_snapshot(name, expected)
    if not snapshot["exact_match"]:
        raise RuntimeError(
            f"{name} distribution drift: expected {expected!r}, "
            f"observed {snapshot['observed']!r}"
        )
    return snapshot


def environment_snapshot(*, mlx_version: str) -> dict:
    """Return import-safe interpreter and MLX build identity."""

    return {
        "python": platform.python_version(),
        "mlx": distribution_snapshot("mlx", mlx_version),
        "inspection_method": "distribution metadata; MLX package not imported",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_checkout_snapshot(
    root: Path,
    *,
    expected_revision: str,
    files: dict[str, str],
    require_match: bool = False,
) -> dict:
    """Verify exact Git and file-byte identity for an external source checkout.

    The returned receipt is useful in ``--describe`` output.  A future GPU run
    passes ``require_match=True`` so a descendant revision, local edit, or
    byte-identical file copied outside the pinned commit is still refused.
    """

    root = root.resolve()
    snapshot = {
        "path": str(root),
        "expected_revision": expected_revision,
        "observed_revision": None,
        "exact_revision": False,
        "files": {},
        "ready": False,
    }
    if not (root / ".git").exists():
        snapshot["error"] = "missing_git_checkout"
    else:
        try:
            revision = git_revision(root)
            snapshot["observed_revision"] = revision
            snapshot["exact_revision"] = revision == expected_revision
            for relative, expected_sha256 in files.items():
                path = root / relative
                status = subprocess.run(
                    ["git", "status", "--porcelain", "--", relative],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                observed_sha256 = sha256_file(path) if path.is_file() else None
                blob = subprocess.run(
                    ["git", "show", f"{expected_revision}:{relative}"],
                    cwd=root,
                    check=False,
                    capture_output=True,
                )
                revision_sha256 = (
                    hashlib.sha256(blob.stdout).hexdigest()
                    if blob.returncode == 0
                    else None
                )
                snapshot["files"][relative] = {
                    "expected_sha256": expected_sha256,
                    "observed_sha256": observed_sha256,
                    "revision_sha256": revision_sha256,
                    "clean": not status,
                    "exact_match": (
                        observed_sha256 == expected_sha256
                        and revision_sha256 == expected_sha256
                        and not status
                    ),
                }
            snapshot["ready"] = snapshot["exact_revision"] and all(
                item["exact_match"] for item in snapshot["files"].values()
            )
        except (OSError, subprocess.SubprocessError) as exc:
            snapshot["error"] = f"{type(exc).__name__}: {exc}"
    if require_match and not snapshot["ready"]:
        raise RuntimeError(
            "external source checkout does not match the pinned benchmark input: "
            + json.dumps(snapshot, sort_keys=True)
        )
    return snapshot


def timed_arms(
    evaluate: Callable[[object], None],
    arms: dict[str, Callable[[], object]],
    *,
    warmups: int,
    rounds: int,
) -> dict[str, dict]:
    """Counterbalanced, synchronized timings for same-process microbench arms."""

    if warmups < 0 or rounds < 1:
        raise ValueError("warmups must be nonnegative and rounds positive")
    names = tuple(arms)
    for _ in range(warmups):
        for name in names:
            evaluate(arms[name]())
    samples = {name: [] for name in names}
    for repeat in range(rounds):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            started = time.perf_counter_ns()
            evaluate(arms[name]())
            samples[name].append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        name: {
            "median_ms": statistics.median(values),
            "minimum_ms": min(values),
            "maximum_ms": max(values),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }
