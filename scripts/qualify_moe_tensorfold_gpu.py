#!/usr/bin/env python3
"""GPU exactness gate for the opt-in ``moe-gate-up-tensorfold-v1`` candidate.

For every weight format it builds two ``SwitchGLU`` blocks from the same seed.
The reference block is never installed, so it is ordinary SwitchGLU and not
the candidate with ``set_enabled(False)``.  The candidate block gets
``moe_tensorfold.install``.  Each assignment case must be bitwise equal to the
reference and must move the mechanism counters by exactly the expected
amounts.  At the end the candidate is uninstalled and has to match the
reference again with no counter movement.

Cases cover unsorted decode rows, sorted rows, the doubled-row sorted tail
window that the candidate pads itself (``n <= 32768 < 2n``, ``2n % 64 != 0``),
and the ``n > 32768``, ``n % 64 != 0`` tail that ``_gather_sort`` pads.
Formats are dense bf16/f16/f32, affine q4/g32, q4/g64 and q8/g64, plus
mxfp4 and nvfp4 when this runtime supports them.

Before touching the GPU, the harness requires identical owner receipts in
``/Users/Shared/mlxuag/gpu.lock/owner.json`` and ``/tmp/gpu.lock/owner.json``,
bound to this process by the holder pid (an ancestor) or ``--lease-id``.  It
re-checks them before every case.  It binds the git revision, the bound
source files (which must match HEAD), the MLX build, and the device.

Timing and memory are reported as observations under whatever load the host
had.  They are not a controlled performance run.  A pass here is a
kernel-level GPU exactness gate for one block.  It does not qualify any model
or serving route, and it does not select the mechanism anywhere.

Run under the campaign lock wrapper, for example::

    zsh qualification/runs/tensorfold-coalescing-20260930/gpuq.sh moe-gate-up \\
        ~/Desktop/mlx2/.venv/bin/python \\
        scripts/qualify_moe_tensorfold_gpu.py \\
        --output qualification/runs/tensorfold-coalescing-20260930/moe-gate-up-gpu.json

``--plan`` writes the case matrix and the static identities without locks and
without using the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx
import numpy as np
from mlx import nn

from mlx2.runtime.models import activations as _activations
from mlx2.runtime.models import moe_tensorfold, switch_layers
from mlx2.runtime.models.switch_layers import SwitchGLU

SCHEMA = "mlx2.moe-gate-up-tensorfold-gpu-qualification.v1"
RUN_DIR = ROOT / "qualification/runs/tensorfold-coalescing-20260930"
LOCK_RECEIPTS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)
BOUND_SOURCES = (
    "src/mlx2/runtime/models/moe_tensorfold.py",
    "src/mlx2/runtime/models/switch_layers.py",
    "src/mlx2/runtime/models/activations.py",
    "scripts/qualify_moe_tensorfold_gpu.py",
)
BOUND_MODULES = {
    "src/mlx2/runtime/models/moe_tensorfold.py": moe_tensorfold,
    "src/mlx2/runtime/models/switch_layers.py": switch_layers,
    "src/mlx2/runtime/models/activations.py": _activations,
}
SCOPE_NOTE = (
    "Kernel-level exactness gate for one SwitchGLU block. It does not qualify "
    "any model or serving route and does not select the mechanism; timing and "
    "memory are uncontrolled observations, not a performance qualification."
)
_BITS = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}
_ITEMSIZE = {"bfloat16": 2, "float16": 2, "float32": 4}


class LockError(RuntimeError):
    """The GPU lock receipts are missing, disagree, or do not bind this process."""


class LeaseChanged(RuntimeError):
    """The GPU lock receipts changed while the harness was running."""


class SwapAbort(RuntimeError):
    """Swap-outs rose past the abort threshold during the run."""


@dataclass(frozen=True)
class Format:
    label: str
    dtype: str
    quant: dict | None = None
    required: bool = True
    dims_offset: int = 0  # nvfp4 dense-tail: input dims not a multiple of 32

    @property
    def mode(self) -> str | None:
        return None if self.quant is None else self.quant.get("mode", "affine")


FORMATS = (
    Format("dense-bf16", "bfloat16"),
    Format("dense-f16", "float16"),
    Format("dense-f32", "float32"),
    Format("affine-q4-g32", "bfloat16", {"group_size": 32, "bits": 4, "mode": "affine"}),
    Format("affine-q4-g64", "bfloat16", {"group_size": 64, "bits": 4, "mode": "affine"}),
    Format("affine-q8-g64", "bfloat16", {"group_size": 64, "bits": 8, "mode": "affine"}),
    # Optional: required only when the runtime supports the mode.
    Format("mxfp4-g32", "bfloat16", {"group_size": 32, "bits": 4, "mode": "mxfp4"}, required=False),
    Format("nvfp4-g16", "bfloat16", {"group_size": 16, "bits": 4, "mode": "nvfp4"}, required=False),
    Format("nvfp4-g16-dense-tail", "bfloat16", {"group_size": 16, "bits": 4, "mode": "nvfp4"},
           required=False, dims_offset=-16),
)


@dataclass
class Case:
    name: str
    tokens: int
    top_k: int
    kind: str
    rows: int = 0
    sorted: bool = False
    rows_seen: int = 0
    tensorfold_pad: bool = False
    expected_stats: dict = field(default_factory=dict)


# ----------------------------------------------------------------- case plan

def _sorted_rows(rows: int, limit: int, tail_bug: bool) -> int:
    """Rows the coalesced group receives after ``_gather_sort``'s own padding."""
    if tail_bug and rows > limit and rows % 64:
        return rows + 64 - rows % 64
    return rows


def expected_case(case: Case, *, limit: int | None = None, tail_bug: bool | None = None,
                  sort_min: int | None = None) -> Case:
    """Fill in routing, padding, and exact counter deltas for one forward."""
    limit = switch_layers._SORTED_GATHER_TAIL_ROWS if limit is None else limit
    tail_bug = switch_layers._SORTED_GATHER_TAIL_BUG if tail_bug is None else tail_bug
    sort_min = switch_layers._GATHER_SORT_MIN_ASSIGNMENTS if sort_min is None else sort_min
    case.rows = case.tokens * case.top_k
    case.sorted = case.rows >= sort_min
    case.rows_seen = _sorted_rows(case.rows, limit, tail_bug) if case.sorted else case.rows
    case.tensorfold_pad = bool(case.sorted and tail_bug and 2 * case.rows_seen > limit
                               and (2 * case.rows_seen) % 64)
    stats = {"calls": 1, "assignments": case.rows_seen,
             "sorted_calls" if case.sorted else "unsorted_calls": 1}
    if case.tensorfold_pad:
        stats["tail_padded_calls"] = 1
    case.expected_stats = stats
    return case


def _doubled_tail_tokens(top_k: int, limit: int) -> int:
    tokens = limit // (2 * top_k) + 1
    while not (limit < 2 * tokens * top_k and tokens * top_k <= limit
               and (2 * tokens * top_k) % 64):
        tokens += 1
        if tokens * top_k > limit:
            raise ValueError(f"no doubled-tail token count for top_k={top_k}")
    return tokens


def _stock_tail_tokens(top_k: int, limit: int) -> int:
    tokens = limit // top_k + 1
    while (tokens * top_k) % 64 == 0:
        tokens += 1
    return tokens


def build_cases(top_k: int, *, prefill_tokens: int = 512, limit: int | None = None,
                sort_min: int | None = None) -> list[Case]:
    limit = switch_layers._SORTED_GATHER_TAIL_ROWS if limit is None else limit
    sort_min = switch_layers._GATHER_SORT_MIN_ASSIGNMENTS if sort_min is None else sort_min
    unsorted = [t for t in (1, 2) if t * top_k < sort_min]
    if not unsorted:
        raise ValueError(f"top_k={top_k} leaves no unsorted case below {sort_min} assignments")
    first_sorted = -(-sort_min // top_k)
    cases = [Case(f"unsorted-t{t}", t, top_k, "unsorted") for t in unsorted]
    cases += [
        Case(f"sorted-t{first_sorted}", first_sorted, top_k, "sorted"),
        Case(f"sorted-t{prefill_tokens}", prefill_tokens, top_k, "sorted"),
    ]
    doubled = _doubled_tail_tokens(top_k, limit)
    stock = _stock_tail_tokens(top_k, limit)
    cases += [
        Case(f"doubled-tail-t{doubled}", doubled, top_k, "doubled_tail"),
        Case(f"stock-tail-t{stock}", stock, top_k, "stock_tail"),
    ]
    return [expected_case(case, limit=limit, sort_min=sort_min) for case in cases]


# ------------------------------------------------------------ lock / identity

def process_ancestors(pid: int | None = None) -> list[int]:
    """This process and its ancestors, nearest first."""
    pid = os.getpid() if pid is None else pid
    chain = []
    while pid > 1 and pid not in chain:
        chain.append(pid)
        out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)],
                             capture_output=True, text=True, check=False).stdout.strip()
        if not out.isdigit():
            break
        pid = int(out)
    return chain


def require_locks(paths=None, *, lease_id: str | None = None,
                  ancestors: list[int] | None = None) -> dict:
    """Both owner receipts must exist, be identical, and bind this process."""
    paths = LOCK_RECEIPTS if paths is None else paths
    owners = []
    for path in paths:
        path = Path(path)
        if not path.parent.is_dir() or not path.is_file():
            raise LockError(f"missing GPU lock receipt: {path}")
        try:
            owner = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise LockError(f"unreadable GPU lock receipt {path}: {exc}") from exc
        if not isinstance(owner, dict):
            raise LockError(f"GPU lock receipt is not an object: {path}")
        owners.append(owner)
    if len(owners) < 2 or any(owner != owners[0] for owner in owners[1:]):
        raise LockError(f"GPU lock receipts disagree: {owners}")
    owner = owners[0]
    lease = owner.get("lease_id")
    if not isinstance(lease, str) or not lease:
        raise LockError("GPU lock receipt has no lease_id")
    if lease_id is not None and lease != lease_id:
        raise LockError(f"GPU lease {lease!r} is not the expected {lease_id!r}")
    holder = owner.get("pid")
    ancestors = process_ancestors() if ancestors is None else ancestors
    holder_is_ancestor = type(holder) is int and holder in ancestors
    if not holder_is_ancestor and lease_id is None:
        raise LockError(
            f"GPU lock holder pid {holder!r} is not an ancestor of this process "
            f"{ancestors} and no --lease-id was given")
    return {
        "paths": [str(path) for path in paths],
        "owner": owner,
        "holder_pid_is_ancestor": holder_is_ancestor,
        "expected_lease_id": lease_id,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def source_identity() -> dict:
    files = {}
    for rel in BOUND_SOURCES:
        path = ROOT / rel
        worktree_blob = _git("hash-object", rel)
        head_blob = _git("rev-parse", f"HEAD:{rel}")
        files[rel] = {
            "sha256": _sha256(path),
            "git_blob": worktree_blob,
            "head_blob": head_blob or None,
            "matches_head": bool(head_blob) and worktree_blob == head_blob,
        }
    imported = {}
    for rel, module in BOUND_MODULES.items():
        loaded = Path(module.__file__).resolve()
        imported[rel] = {"file": str(loaded), "from_this_tree": loaded == (ROOT / rel).resolve()}
    tree = hashlib.sha256()
    for path in sorted((ROOT / "src" / "mlx2").rglob("*.py")):
        tree.update(str(path.relative_to(ROOT)).encode())
        tree.update(path.read_bytes())
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "tracked_changes": [line for line in _git("status", "--porcelain",
                                                   "--untracked-files=no").splitlines() if line],
        "files": files,
        "imported_modules": imported,
        "src_tree_sha256": tree.hexdigest(),
        "law_id": moe_tensorfold.LAW_ID,
        "all_bound_match_head": all(entry["matches_head"] for entry in files.values()),
        "all_modules_from_this_tree": all(entry["from_this_tree"] for entry in imported.values()),
    }


def mlx_identity() -> dict:
    package = Path(next(iter(__import__("mlx").__path__)))
    binaries = {}
    for path in sorted(package.rglob("*")):
        if path.suffix in {".dylib", ".metallib", ".so"}:
            binaries[str(path.relative_to(package))] = {"sha256": _sha256(path),
                                                        "bytes": path.stat().st_size}
    try:
        dist = metadata.distribution("mlx")
        dist_version, direct_url = dist.version, dist.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        dist_version, direct_url = None, None
    return {
        "version": getattr(mx, "__version__", None),
        "distribution_version": dist_version,
        "direct_url": json.loads(direct_url) if direct_url else None,
        "package_dir": str(package),
        "binaries": binaries,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "unset"),
    }


def _sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                          check=False).stdout.strip()


def host_identity(*, gpu: bool) -> dict:
    host = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "hw_memsize": _sysctl("hw.memsize"),
        "default_device": str(mx.default_device()),
    }
    if gpu:
        host["metal_available"] = bool(mx.metal.is_available())
        host["device_info"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                               for k, v in mx.device_info().items()}
    return host


def swapouts() -> int | None:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return None


def thermal() -> str:
    return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True,
                          check=False).stdout.strip()


# ------------------------------------------------------------------ building

class _Block(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.layers = [block]

    def __call__(self, x, indices):
        return self.layers[0](x, indices)


def build_block(fmt: Format, *, dims: int, hidden: int, experts: int, seed: int) -> _Block:
    """Seeded SwitchGLU; two calls with the same arguments give identical weights."""
    mx.random.seed(seed)
    block = SwitchGLU(dims + fmt.dims_offset, hidden, experts, bias=False)
    dtype = getattr(mx, fmt.dtype)
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(block, name)
        proj.update({k: v.astype(dtype) for k, v in proj.parameters().items()})
        if fmt.quant is not None:
            setattr(block, name, proj.to_quantized(**fmt.quant))
    model = _Block(block)
    model.eval()
    mx.eval(model.parameters())
    return model


def probe_format(fmt: Format, *, dims: int) -> dict:
    """Run a tiny gather through the same projection path on this device."""
    if fmt.quant is None:
        return {"supported": True}
    try:
        tiny = build_block(fmt, dims=128, hidden=64, experts=2, seed=1)
        out = tiny(mx.ones((1, tiny.layers[0].gate_proj.input_dims), getattr(mx, fmt.dtype)),
                   mx.zeros((1, 1), mx.uint32))
        mx.eval(out)
        if (dims + fmt.dims_offset) % fmt.quant["group_size"]:
            raise ValueError(f"input dims {dims + fmt.dims_offset} not divisible by group size")
    except Exception as exc:  # noqa: BLE001 - recorded as unsupported
        return {"supported": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {"supported": True}


def _param_bytes(fmt: Format, rows: int, cols: int, experts: int) -> int:
    if fmt.quant is None:
        return experts * rows * cols * _ITEMSIZE[fmt.dtype]
    scale_bytes = 4 if fmt.mode == "affine" else 1
    return experts * rows * cols * fmt.quant["bits"] // 8 + experts * rows * (
        cols // fmt.quant["group_size"]) * scale_bytes


def estimate_bytes(fmt: Format, case: Case | None, *, dims: int, hidden: int, experts: int) -> int:
    """Rough upper estimate: two blocks, the install transient, and activations."""
    k = dims + fmt.dims_offset
    proj = _param_bytes(fmt, hidden, k, experts)
    down = _param_bytes(fmt, k, hidden, experts)
    total = 2 * (2 * proj + down) + 2 * proj
    if fmt.quant is not None and fmt.mode == "nvfp4" and k % 32:
        total += 2 * experts * hidden * k * 2 * 2  # both halves dequantized per call
    if case is not None:
        act = _ITEMSIZE[fmt.dtype]
        rows = case.rows_seen + 64
        total += 3 * rows * (2 * k + 6 * hidden) * act
    return total


# ---------------------------------------------------------------- comparison

def bitwise_compare(actual, expected) -> dict:
    result = {"shape": list(expected.shape), "dtype": str(expected.dtype).removeprefix("mlx.core."),
              "shape_dtype_match": actual.shape == expected.shape and actual.dtype == expected.dtype}
    if not result["shape_dtype_match"]:
        result.update(bitwise_equal=False, actual_shape=list(actual.shape),
                      actual_dtype=str(actual.dtype))
        return result
    bits = _BITS[expected.itemsize]
    diff = mx.view(actual, bits) != mx.view(expected, bits)
    mismatches = mx.sum(diff)
    delta = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(mx.where(mx.isnan(delta), mx.array(math.inf, mx.float32), delta))
    finite = mx.all(mx.isfinite(actual)) & mx.all(mx.isfinite(expected))
    mx.eval(mismatches, max_abs, finite)
    result.update(bitwise_equal=int(mismatches.item()) == 0,
                  mismatched_elements=int(mismatches.item()),
                  max_abs=float(max_abs.item()), finite=bool(finite.item()))
    return result


def _stats_delta(before: dict, after: dict) -> dict:
    keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(keys)
            if after.get(k, 0) - before.get(k, 0)}


def _inputs(case: Case, dims: int, dtype: str, experts: int, seed: int):
    rng = np.random.default_rng(seed)
    routing = np.argsort(rng.random((case.tokens, experts)), axis=1)[:, :case.top_k]
    x = mx.array(rng.standard_normal((case.tokens, dims)).astype(np.float32)).astype(
        getattr(mx, dtype))
    indices = mx.array(routing.astype(np.uint32))
    mx.eval(x, indices)
    return x, indices


def _arm(model, x, indices):
    return lambda: model(x, indices)


def _time_arms(arms: dict, *, warmups: int, repeats: int, checkpoint) -> dict:
    """Interleaved timing; the arm order alternates every round."""
    names = list(arms)
    for round_ in range(warmups):
        for name in (names if round_ % 2 == 0 else names[::-1]):
            mx.eval(arms[name]())
    samples = {name: [] for name in names}
    for round_ in range(repeats):
        checkpoint()
        for name in (names if round_ % 2 == 0 else names[::-1]):
            start = time.perf_counter_ns()
            mx.eval(arms[name]())
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    peaks = {}
    for name in names:
        mx.eval(arms[name]())
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        mx.eval(arms[name]())
        peaks[name] = mx.get_peak_memory() - base
    summary = {}
    for name in names:
        values = samples[name]
        summary[name] = {
            "median_ms": statistics.median(values) if values else None,
            "min_ms": min(values) if values else None,
            "max_ms": max(values) if values else None,
            "samples_ms": values,
            "peak_over_active_bytes": peaks[name],
        }
    if samples[names[0]] and samples[names[1]]:
        summary["candidate_over_reference_median"] = (
            summary[names[1]]["median_ms"] / summary[names[0]]["median_ms"])
    return summary


# ------------------------------------------------------------------- running

def run_format(fmt: Format, cases: list[Case], *, dims: int, hidden: int, experts: int,
               seed: int, warmups: int, repeats: int, budget_bytes: int,
               checkpoint=lambda: None) -> dict:
    """Reference vs installed candidate for every case, then the uninstall round trip."""
    report = {"format": asdict(fmt), "input_dims": dims + fmt.dims_offset, "cases": []}
    probe = probe_format(fmt, dims=dims)
    report["probe"] = probe
    if not probe["supported"]:
        report["status"] = "unsupported" if not fmt.required else "fail"
        return report
    if estimate_bytes(fmt, None, dims=dims, hidden=hidden, experts=experts) > budget_bytes:
        report["status"] = "skipped_memory"
        return report
    checkpoint()
    moe_tensorfold.ENABLED[0] = True  # the reference is uninstalled, not disabled
    mx.clear_cache()
    active0 = mx.get_active_memory()
    reference = build_block(fmt, dims=dims, hidden=hidden, experts=experts, seed=seed)
    active_ref = mx.get_active_memory()
    candidate = build_block(fmt, dims=dims, hidden=hidden, experts=experts, seed=seed)
    active_pair = mx.get_active_memory()
    ref_params = dict(_flatten(reference.parameters()))
    cand_params = dict(_flatten(candidate.parameters()))
    pair_identical = ref_params.keys() == cand_params.keys() and all(
        bitwise_compare(cand_params[k], ref_params[k])["bitwise_equal"] for k in ref_params)
    del ref_params, cand_params
    mx.reset_peak_memory()
    receipt = moe_tensorfold.install(candidate)
    install_peak = mx.get_peak_memory()
    mx.clear_cache()
    active_installed = mx.get_active_memory()
    group = candidate.layers[0].__dict__.get(moe_tensorfold._ATTR)
    report["install"] = {
        "receipt": receipt,
        "reference_installed": moe_tensorfold._ATTR in reference.layers[0].__dict__,
        "group_format": getattr(group, "format", None),
        "pair_identical_before_install": pair_identical,
    }
    report["memory"] = {
        "reference_block_bytes": active_ref - active0,
        "candidate_block_bytes": active_pair - active_ref,
        "install_active_delta_bytes": active_installed - active_pair,
        "install_peak_over_active_bytes": install_peak - active_pair,
    }
    install_ok = (receipt["installed"] == 1 and receipt["selected"] is True
                  and receipt["qualified"] is False and receipt["refused"] == {}
                  and list(receipt["covered"]) == [getattr(group, "format", None)]
                  and not report["install"]["reference_installed"] and pair_identical)
    report["install"]["ok"] = install_ok

    held = []
    for index, case in enumerate(cases):
        entry = {"case": asdict(case)}
        report["cases"].append(entry)
        estimate = estimate_bytes(fmt, case, dims=dims, hidden=hidden, experts=experts)
        entry["estimated_bytes"] = estimate
        if estimate > budget_bytes:
            entry["status"] = "skipped_memory"
            continue
        checkpoint()
        x, indices = _inputs(case, dims + fmt.dims_offset, fmt.dtype, experts, seed + 1 + index)
        before = moe_tensorfold.stats()
        expected = reference(x, indices)
        mx.eval(expected)
        again = reference(x, indices)
        mx.eval(again)
        reference_stats = _stats_delta(before, moe_tensorfold.stats())
        before = moe_tensorfold.stats()
        actual = candidate(x, indices)
        mx.eval(actual)
        candidate_stats = _stats_delta(before, moe_tensorfold.stats())
        entry["reference_deterministic"] = bitwise_compare(again, expected)["bitwise_equal"]
        entry["parity"] = bitwise_compare(actual, expected)
        entry["reference_stats_delta"] = reference_stats
        entry["candidate_stats_delta"] = candidate_stats
        entry["counters_match"] = candidate_stats == case.expected_stats
        entry["mechanism_engaged"] = candidate_stats.get("calls", 0) > 0 and candidate_stats.get(
            "assignments", 0) > 0 and (not case.tensorfold_pad
                                       or candidate_stats.get("tail_padded_calls", 0) > 0)
        del again
        before = moe_tensorfold.stats()
        entry["timing"] = _time_arms(
            {"reference": _arm(reference, x, indices), "candidate": _arm(candidate, x, indices)},
            warmups=warmups, repeats=repeats, checkpoint=checkpoint)
        timed = _stats_delta(before, moe_tensorfold.stats())
        timed_calls = warmups + repeats + 2  # plus the two peak-memory calls
        entry["timed_candidate_calls"] = timed.get("calls", 0)
        entry["timed_calls_ok"] = timed.get("calls", 0) == timed_calls
        entry["timed_fallbacks"] = {k: timed[k] for k in ("stale_dropped", "fallback_training")
                                    if k in timed}
        entry["status"] = "pass" if (
            entry["parity"]["bitwise_equal"] and entry["counters_match"]
            and entry["mechanism_engaged"] and not reference_stats
            and entry["timed_calls_ok"] and not entry["timed_fallbacks"]) else "fail"
        held.append((case.name, x, indices, expected))

    checkpoint()
    restored = moe_tensorfold.uninstall(candidate)
    post = {"restored": restored,
            "attribute_removed": moe_tensorfold._ATTR not in candidate.layers[0].__dict__,
            "cases": []}
    for name, x, indices, expected in held:
        before = moe_tensorfold.stats()
        actual = candidate(x, indices)
        mx.eval(actual)
        post["cases"].append({"case": name,
                              "parity": bitwise_compare(actual, expected),
                              "stats_delta": _stats_delta(before, moe_tensorfold.stats())})
    post["ok"] = (restored == 1 and post["attribute_removed"] and all(
        item["parity"]["bitwise_equal"] and not item["stats_delta"] for item in post["cases"]))
    report["post_uninstall"] = post
    del held, reference, candidate
    mx.clear_cache()

    statuses = [entry["status"] for entry in report["cases"]]
    if not install_ok or not post["ok"] or "fail" in statuses:
        report["status"] = "fail"
    elif "skipped_memory" in statuses:
        report["status"] = "incomplete"
    else:
        report["status"] = "pass"
    return report


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _flatten(value, f"{prefix}{key}.")
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            yield from _flatten(value, f"{prefix}{index}.")
    else:
        yield prefix.rstrip("."), tree


def verdict(formats: list[dict], *, aborted: str | None = None) -> str:
    """``pass`` only when every required and every supported optional format passed."""
    if aborted:
        return "aborted"
    statuses = [entry["status"] for entry in formats]
    if "fail" in statuses or "error" in statuses:
        return "fail"
    if any(status in {"incomplete", "skipped_memory"} for status in statuses):
        return "incomplete"
    required = [entry for entry in formats if entry["format"]["required"]]
    if not required or any(entry["status"] != "pass" for entry in required):
        return "incomplete"
    return "pass"


# ---------------------------------------------------------------------- main

def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _selected_formats(names: str | None) -> list[Format]:
    if not names:
        return list(FORMATS)
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    known = {fmt.label: fmt for fmt in FORMATS}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown formats {unknown}; known {sorted(known)}")
    return [known[name] for name in wanted]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true",
                        help="write the case matrix and static identities; no locks, no GPU")
    parser.add_argument("--lease-id", help="expected lease_id when the holder is not an ancestor")
    parser.add_argument("--formats", help="comma-separated subset of format labels")
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--memory-budget-gib", type=float, default=24.0)
    parser.add_argument("--cache-limit-gib", type=float, default=4.0)
    parser.add_argument("--swap-abort-pages", type=int, default=25000)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="record, instead of refusing, bound sources that differ from HEAD")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.plan:
        mx.set_default_device(mx.cpu)
    formats = _selected_formats(args.formats)
    if args.experts < args.top_k:
        raise SystemExit("--experts must be at least --top-k")
    cases = build_cases(args.top_k, prefill_tokens=args.prefill_tokens)
    budget = int(args.memory_budget_gib * (1 << 30))
    report = {
        "schema": SCHEMA,
        "law_id": moe_tensorfold.LAW_ID,
        "scope": SCOPE_NOTE,
        "model_serving_qualified": False,
        "route_selected": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "controls": {
            "dims": args.dims, "hidden": args.hidden, "experts": args.experts,
            "top_k": args.top_k, "seed": args.seed, "warmups": args.warmups,
            "repeats": args.repeats, "memory_budget_bytes": budget,
            "cache_limit_bytes": int(args.cache_limit_gib * (1 << 30)),
            "swap_abort_pages": args.swap_abort_pages,
            "reference_arm": "separate never-installed SwitchGLU from the same seed",
            "timing_order": "interleaved, arm order alternating each round",
            "sorted_gather_tail_rows": switch_layers._SORTED_GATHER_TAIL_ROWS,
            "sorted_gather_tail_bug": switch_layers._SORTED_GATHER_TAIL_BUG,
            "gather_sort_min_assignments": switch_layers._GATHER_SORT_MIN_ASSIGNMENTS,
        },
        "cases": [asdict(case) for case in cases],
        "source": source_identity(),
        "mlx": mlx_identity(),
    }
    report["plan"] = [{"format": fmt.label, "required": fmt.required,
                       "estimated_block_pair_bytes": estimate_bytes(
                           fmt, None, dims=args.dims, hidden=args.hidden, experts=args.experts),
                       "cases_over_budget": [case.name for case in cases if estimate_bytes(
                           fmt, case, dims=args.dims, hidden=args.hidden,
                           experts=args.experts) > budget]}
                      for fmt in formats]
    if args.plan:
        report["host"] = host_identity(gpu=False)
        report["mode"] = "plan"
        report["verdict"] = "not_run"
        _write(args.output, report)
        print(json.dumps({"mode": "plan", "output": str(args.output),
                          "formats": [fmt.label for fmt in formats],
                          "cases": [case.name for case in cases]}, indent=2))
        return 0

    source = report["source"]
    if not source["all_modules_from_this_tree"]:
        raise SystemExit(f"bound modules imported from another tree: {source['imported_modules']}")
    if not source["all_bound_match_head"] and not args.allow_dirty:
        raise SystemExit("bound sources differ from HEAD; commit them or pass --allow-dirty")
    ancestors = process_ancestors()
    try:
        locks = require_locks(lease_id=args.lease_id, ancestors=ancestors)
    except LockError as exc:
        raise SystemExit(f"refusing to use the GPU: {exc}") from exc
    report["locks"] = locks
    if not mx.metal.is_available():
        raise SystemExit("Metal GPU is unavailable")
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(report["controls"]["cache_limit_bytes"])
    report["mode"] = "gpu"
    report["host"] = host_identity(gpu=True)
    report["thermal_start"] = thermal()
    swap_start = swapouts()
    report["swapouts_start"] = swap_start

    def checkpoint():
        current = require_locks(lease_id=args.lease_id, ancestors=ancestors)
        if current["owner"] != locks["owner"]:
            raise LeaseChanged(f"GPU lock owner changed to {current['owner']}")
        now = swapouts()
        if swap_start is not None and now is not None and now - swap_start > args.swap_abort_pages:
            raise SwapAbort(f"swapouts rose by {now - swap_start} pages")

    report["formats"] = []
    aborted = None
    for fmt in formats:
        try:
            result = run_format(fmt, cases, dims=args.dims, hidden=args.hidden,
                                experts=args.experts, seed=args.seed, warmups=args.warmups,
                                repeats=args.repeats, budget_bytes=budget, checkpoint=checkpoint)
        except (LockError, LeaseChanged, SwapAbort) as exc:
            aborted = f"{type(exc).__name__}: {exc}"
            report["formats"].append({"format": asdict(fmt), "status": "aborted",
                                      "reason": aborted})
            break
        except Exception as exc:  # noqa: BLE001 - recorded; the gate fails closed
            result = {"format": asdict(fmt), "status": "error",
                      "reason": f"{type(exc).__name__}: {exc}"}
            mx.clear_cache()
        report["formats"].append(result)
        report["verdict"] = "running"
        _write(args.output, report)
    report["swapouts_end"] = swapouts()
    report["thermal_end"] = thermal()
    report["mechanism_stats_total"] = moe_tensorfold.stats()
    report["verdict"] = verdict(report["formats"], aborted=aborted)
    report["gpu_exactness_gate"] = report["verdict"]
    _write(args.output, report)
    summary = {"verdict": report["verdict"], "output": str(args.output),
               "formats": {entry["format"]["label"]: entry["status"]
                           for entry in report["formats"]}}
    print(json.dumps(summary, indent=2))
    return {"pass": 0, "fail": 1}.get(report["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
