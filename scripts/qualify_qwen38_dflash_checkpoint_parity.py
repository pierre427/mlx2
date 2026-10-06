#!/usr/bin/env python3
"""Compare every committed TensorFold recurrent/KV checkpoint to serial replay."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / "mlx-models" / "Qwen3.8-27B-MLX-4bit"
DEFAULT_CALIBRATION = (
    ROOT
    / "qualification/runs/dflash-varlen-context-ladder-20261005/prompt-calibration.json"
)
DEFAULT_TENSORFOLD = (
    Path.home() / ".codex/worktrees/tensorfold-upstream-parity-20260928"
)
GPU_OWNERS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)
PRODUCTION_TENSORFOLD_REVISION = "1a5f38e12afbb560d8fc61c88ccb5900f7d5d170"
TENSORFOLD_SOURCE_VALIDATOR = (
    ROOT / "src/mlx2/adapters/qwen38_tensorfold_source.py"
)
_FULL_GIT_OID = re.compile(r"[0-9a-f]{40}")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qwen38_tensorfold_component_oracle import (
    FUSED_GDN_ARMS,
    PROJECTION_LAWS,
    ProjectionCapture,
    component_report,
    explicit_execution_policy,
    explicit_projection_policy,
    fused_gdn_engagement_report,
)


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=path, text=True).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_tensorfold_revision_on_disk() -> str:
    """Read the production pin as a literal without importing runtime code."""

    tree = ast.parse(TENSORFOLD_SOURCE_VALIDATOR.read_text())
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_REVISION"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if values != [PRODUCTION_TENSORFOLD_REVISION]:
        raise RuntimeError(
            "production TensorFold EXPECTED_REVISION changed on disk; review the "
            "qualification bind against the new production pin"
        )
    return values[0]


def qualification_candidate_spec(
    revision: str | None,
    tree: str | None,
    parent: str | None,
) -> dict[str, str] | None:
    """Require a complete, full-OID candidate identity or no override at all."""

    values = {"revision": revision, "tree": tree, "parent_revision": parent}
    present = {name: value is not None for name, value in values.items()}
    if any(present.values()) and not all(present.values()):
        missing = sorted(name for name, supplied in present.items() if not supplied)
        raise ValueError(
            "qualification TensorFold candidate requires revision, tree, and parent; "
            f"missing {', '.join(missing)}"
        )
    if not any(present.values()):
        return None
    invalid = [
        name
        for name, value in values.items()
        if not isinstance(value, str) or _FULL_GIT_OID.fullmatch(value) is None
    ]
    if invalid:
        raise ValueError(
            "qualification TensorFold candidate identities must be full lowercase "
            f"Git OIDs: {', '.join(sorted(invalid))}"
        )
    return {name: value for name, value in values.items() if value is not None}


def bind_tensorfold_source(
    root: Path,
    candidate: dict[str, str] | None,
    *,
    source_module=None,
) -> dict:
    """Validate and process-locally bind one production or qualification source.

    Candidate binding is deliberately confined to this qualification process.
    The production source file must still contain the reviewed production pin,
    and both the candidate commit graph and imported source tree must be exact.
    """

    production_revision = _production_tensorfold_revision_on_disk()
    if "mlx2.runtime.qwen38_tensorfold" in sys.modules:
        raise RuntimeError("TensorFold runtime imported before qualification source bind")
    if source_module is None:
        from mlx2.adapters import qwen38_tensorfold_source as source_module

    if source_module.EXPECTED_REVISION != production_revision:
        raise RuntimeError(
            "process TensorFold EXPECTED_REVISION was altered before the explicit bind"
        )

    root = root.resolve()
    revision = git(root, "rev-parse", "HEAD")
    tree = git(root, "rev-parse", "HEAD^{tree}")
    ancestry = git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(ancestry) != 2 or ancestry[0] != revision:
        raise RuntimeError(
            "qualification TensorFold candidate must have exactly one parent"
        )
    parent_revision = ancestry[1]
    tracked_diff = git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src/tensorfold",
    )
    if tracked_diff:
        raise RuntimeError(
            "qualification TensorFold imported source differs from HEAD:\n"
            + tracked_diff
        )

    if candidate is None:
        if revision != production_revision:
            raise RuntimeError(
                "non-production TensorFold source requires an explicit exact "
                "qualification candidate bind"
            )
        mode = "production"
        candidate_identity = None
    else:
        actual = {
            "revision": revision,
            "tree": tree,
            "parent_revision": parent_revision,
        }
        if actual != candidate:
            raise RuntimeError(
                f"qualification TensorFold identity {actual} differs from {candidate}"
            )
        if revision == production_revision:
            raise RuntimeError(
                "qualification candidate bind must not restate the production revision"
            )
        source_module.EXPECTED_REVISION = revision
        mode = "qualification_candidate"
        candidate_identity = actual

    try:
        identity = source_module.validate_source(root)
    except BaseException:
        source_module.EXPECTED_REVISION = production_revision
        raise
    if identity.get("revision") != revision or identity.get("tracked_diff"):
        source_module.EXPECTED_REVISION = production_revision
        raise RuntimeError("TensorFold validator returned an inconsistent identity")

    return {
        "schema": "mlx2.qualification-tensorfold-bind.v2",
        "mode": mode,
        "qualification_only": candidate is not None,
        "production_revision_on_disk": production_revision,
        "production_validator_sha256": sha256(TENSORFOLD_SOURCE_VALIDATOR),
        "candidate": candidate_identity,
        "active": {
            "root": str(root),
            "revision": revision,
            "tree": tree,
            "parent_revision": parent_revision,
            "tracked_diff": tracked_diff,
        },
    }


def require_gpu_owners(*, environ=None, paths=GPU_OWNERS) -> dict:
    environment = os.environ if environ is None else environ
    lease = environment.get("GPUQ_LEASE")
    session = environment.get("GPUQ_SESSION")
    if not lease:
        raise RuntimeError("GPUQ_LEASE is required")
    if not session:
        raise RuntimeError("GPUQ_SESSION is required")
    owners = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(
                f"missing GPU owner receipt {path}; use run_with_gpu_locks.py"
            )
        owner = json.loads(path.read_text())
        if not isinstance(owner, dict) or not owner.get("lease_id"):
            raise RuntimeError(f"invalid GPU owner receipt: {path}")
        owners.append(owner)
    if owners[0] != owners[1]:
        raise RuntimeError("GPU owner receipts disagree")
    for owner in owners:
        if owner.get("lease_id") != lease:
            raise RuntimeError("GPU owner receipt does not match GPUQ_LEASE")
        if owner.get("session") != session:
            raise RuntimeError("GPU owner receipt does not match GPUQ_SESSION")
    return owners[0]


def digest_key(record: dict) -> str | None:
    return record.get("sha256") or record.get("state_only_sha256")


def first_layer_where(layers: list[dict], predicate) -> int | None:
    """Return the first model-layer index matching ``predicate``."""

    for record in layers:
        if predicate(record):
            return int(record["index"])
    return None


def layer_divergence_summary(layers: list[dict]) -> dict:
    """Summarize exact and tolerance divergence without changing its gate."""

    return {
        "first_exact_divergent_layer": first_layer_where(
            layers,
            lambda row: bool(row.get("structural_errors"))
            or int(row.get("exact_array_count", 0))
            != int(row.get("array_count", 0)),
        ),
        "first_tolerance_failed_layer": first_layer_where(
            layers, lambda row: not bool(row.get("passed"))
        ),
        "failed_layers": [
            int(row["index"]) for row in layers if not bool(row.get("passed"))
        ],
    }


def _array_comparison(mx, actual, serial, *, atol: float, rtol: float) -> dict:
    """Compare one logical state array without copying its storage to the host."""

    if tuple(actual.shape) != tuple(serial.shape) or actual.dtype != serial.dtype:
        return {
            "equal": False,
            "close": False,
            "actual_shape": list(actual.shape),
            "serial_shape": list(serial.shape),
            "actual_dtype": str(actual.dtype),
            "serial_dtype": str(serial.dtype),
            "max_abs": None,
            "max_rel": None,
        }
    if actual.size == 0:
        return {
            "equal": True,
            "close": True,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "max_abs": 0.0,
            "max_rel": 0.0,
        }
    left = actual.astype(mx.float32)
    right = serial.astype(mx.float32)
    delta = mx.abs(left - right)
    max_abs = float(mx.max(delta).item())
    max_rel = float(mx.max(delta / mx.maximum(mx.abs(right), 1e-6)).item())
    relative_l2 = float(
        mx.sqrt(mx.sum(delta * delta) / mx.maximum(mx.sum(right * right), 1e-12)).item()
    )
    normalized_max = float(max_abs / max(float(mx.max(mx.abs(right)).item()), 1e-6))
    equal = bool(mx.all(actual == serial).item())
    close = bool(mx.all(delta <= atol + rtol * mx.abs(right)).item())
    return {
        "equal": equal,
        "close": close,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "relative_l2": relative_l2,
        "normalized_max": normalized_max,
    }


def require_explicit_gdn_reference_policy(policy: dict) -> dict[str, object]:
    """Refuse an implicit GDN reference and label the selected serial arm."""

    enabled = policy.get("fused_gdn")
    if type(enabled) is not bool:
        raise ValueError(
            "checkpoint parity requires explicit boolean fused_gdn: false for "
            "the ordinary diagnostic arm or true for default-on qualification"
        )
    return {
        "kind": (
            "selected_fused_gdn_with_selected_lane_projections"
            if enabled
            else "ordinary_unfused_gdn_with_selected_lane_projections"
        ),
        "policy_field": "fused_gdn",
        "policy_value": enabled,
        "ordinary_diagnostic": not enabled,
        "claim": (
            "explicit serial GDN reference under the applied lane projection policy; "
            "not stock-projection parity or TensorFold qualification by itself"
        ),
    }


def _transaction_commit_oracle(mx, cache, record, path, *, start: int) -> dict:
    """Validate exact KV/conv publication and tolerance-bound recurrent state."""

    index_array = mx.array([int(row) for row in path], dtype=mx.int32)
    layers = []
    for index, (item, entry) in enumerate(zip(cache, record)):
        if entry[0] == "kv":
            _, window_keys, window_values = entry
            expected = (
                mx.take(window_keys, index_array, axis=2),
                mx.take(window_values, index_array, axis=2),
            )
            actual = (
                item.keys[..., start:start + len(path), :],
                item.values[..., start:start + len(path), :],
            )
            comparison = _state_comparison(
                mx, actual, expected, atol=0.0, rtol=0.0
            )
            offset_equal = int(item.offset) == start + len(path)
            layers.append(
                {
                    "index": index,
                    "kind": "kv",
                    "offset_equal": offset_equal,
                    **comparison,
                    "passed": offset_equal and comparison["passed"],
                }
            )
            continue

        if entry[0] != "gdn":
            layers.append(
                {
                    "index": index,
                    "kind": "unknown",
                    "passed": False,
                    "reason": f"unsupported record tag {entry[0]!r}",
                }
            )
            continue
        _, n_keep, state0, conv0, row_offset, shared = entry
        _q, keys, values, gates, betas, qkv = shared
        state = state0.astype(mx.float32)
        for local_row in path:
            row = int(row_offset) + int(local_row)
            key = keys[:, row].astype(mx.float32)
            repeats = int(state.shape[1]) // int(key.shape[1])
            key = mx.repeat(key, repeats, axis=1)
            gate = gates[:, row].astype(mx.float32)[:, :, None, None]
            beta = betas[:, row].astype(mx.float32)[:, :, None]
            value = values[:, row].astype(mx.float32)
            state = state * gate
            remembered = mx.sum(state * key[:, :, None, :], axis=-1)
            delta = (value - remembered) * beta
            state = state + delta[:, :, :, None] * key[:, :, None, :]

        selected = mx.take(
            qkv,
            mx.array([int(row_offset) + int(row) for row in path], dtype=mx.int32),
            axis=1,
        )
        conv = mx.concatenate((conv0, selected), axis=1)[:, -int(n_keep):]
        state_check = _array_comparison(
            mx, item.cache[1], state, atol=2e-6, rtol=2e-5
        )
        conv_check = _array_comparison(
            mx, item.cache[0], conv, atol=0.0, rtol=0.0
        )
        layers.append(
            {
                "index": index,
                "kind": "gdn",
                "state": state_check,
                "conv": conv_check,
                "passed": state_check["close"] and conv_check["equal"],
            }
        )
    return {
        "comparison_contract": {
            "kv": "exact",
            "conv": "exact",
            "recurrent": "tolerance",
            "recurrent_atol": 2e-6,
            "recurrent_rtol": 2e-5,
        },
        "layers": layers,
        "passed": len(layers) == len(cache) and all(row["passed"] for row in layers),
        "failed_layers": [row["index"] for row in layers if not row["passed"]],
    }


def _teacher_forced_continuation(
    mx,
    adapter,
    owner,
    actual_cache,
    serial_cache,
    token,
    *,
    steps=8,
    state_atol=0.01,
    state_rtol=0.01,
):
    """Probe two checkpoints with one serial-chosen teacher stream."""

    actual = owner._copy_reference_cache(actual_cache)
    serial = owner._copy_reference_cache(serial_cache)
    capture_layers = tuple(range(len(adapter.model.layers)))
    rows = []
    current = int(token)
    for step in range(int(steps)):
        starts = [
            int(item.offset) if hasattr(item, "offset") else 0 for item in actual
        ]
        ids = mx.array([[current]], dtype=mx.uint32)
        actual_logits, actual_taps = adapter.model.forward_with_taps(
            ids, actual, capture_layers
        )
        serial_logits, serial_taps = adapter.model.forward_with_taps(
            ids, serial, capture_layers
        )
        mx.eval(actual_logits, actual_taps, serial_logits, serial_taps)
        hidden_width = int(actual_taps.shape[-1]) // len(capture_layers)
        hidden_layers = []
        for index in capture_layers:
            start = index * hidden_width
            comparison = _array_comparison(
                mx,
                actual_taps[..., start:start + hidden_width],
                serial_taps[..., start:start + hidden_width],
                atol=state_atol,
                rtol=state_rtol,
            )
            hidden_layers.append({"index": index, **comparison})
        hidden_summary = layer_divergence_summary(
            [
                {
                    "index": row["index"],
                    "array_count": 1,
                    "exact_array_count": int(bool(row["equal"])),
                    "structural_errors": [],
                    "passed": bool(row["close"]),
                }
                for row in hidden_layers
            ]
        )
        hidden_summary.update(
            max_abs=max((row["max_abs"] or 0.0 for row in hidden_layers), default=0.0),
            max_rel=max((row["max_rel"] or 0.0 for row in hidden_layers), default=0.0),
        )
        cache_comparison = _cache_comparison(
            mx,
            actual,
            serial,
            starts=starts,
            atol=state_atol,
            rtol=state_rtol,
        )
        cache_summary = layer_divergence_summary(cache_comparison["layers"])
        comparison = _array_comparison(
            mx, actual_logits, serial_logits, atol=0.0, rtol=0.0
        )
        actual_row = actual_logits[0, -1].astype(mx.float32)
        serial_row = serial_logits[0, -1].astype(mx.float32)
        actual_logp = actual_row - mx.logsumexp(actual_row)
        serial_logp = serial_row - mx.logsumexp(serial_row)
        actual_p = mx.exp(actual_logp)
        serial_p = mx.exp(serial_logp)
        tv = float((0.5 * mx.sum(mx.abs(actual_p - serial_p))).item())
        kl = float(mx.sum(serial_p * (serial_logp - actual_logp)).item())
        actual_top = int(mx.argmax(actual_row).item())
        serial_top = int(mx.argmax(serial_row).item())
        row = {
            "step": step,
            "teacher_token": current,
            "actual_argmax": actual_top,
            "serial_argmax": serial_top,
            "argmax_equal": actual_top == serial_top,
            "total_variation": tv,
            "serial_to_actual_kl": max(0.0, kl),
            "logits": comparison,
            "hidden": hidden_summary,
            "cache": {
                **cache_summary,
                "max_abs": cache_comparison["max_abs"],
                "max_rel": cache_comparison["max_rel"],
            },
        }
        row["passed"] = (
            row["argmax_equal"]
            and tv <= 0.01
            and row["serial_to_actual_kl"] <= 0.001
            and comparison["relative_l2"] <= 0.02
            and comparison["normalized_max"] <= 0.02
        )
        rows.append(row)
        current = serial_top
    return {
        "steps": rows,
        "thresholds": {
            "total_variation_max": 0.01,
            "serial_to_actual_kl_max": 0.001,
            "logit_relative_l2_max": 0.02,
            "logit_normalized_max": 0.02,
            "argmax_equal": True,
        },
        "passed": len(rows) == int(steps) and all(row["passed"] for row in rows),
    }


def _state_comparison(mx, actual, serial, *, atol: float, rtol: float) -> dict:
    """Compare nested cache state while retaining compact numeric diagnostics."""

    arrays: list[dict] = []
    structural: list[str] = []

    def walk(left, right, path: str) -> None:
        if isinstance(left, mx.array) and isinstance(right, mx.array):
            record = _array_comparison(mx, left, right, atol=atol, rtol=rtol)
            record["path"] = path
            arrays.append(record)
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                structural.append(
                    f"{path}: sequence mismatch {type(left).__name__}/{len(left)} "
                    f"!= {type(right).__name__}/{len(right)}"
                )
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                walk(left_item, right_item, f"{path}[{index}]")
            return
        if left is None or right is None:
            if left is not right:
                structural.append(f"{path}: one side is None")
            return
        if left != right:
            structural.append(f"{path}: scalar mismatch {left!r} != {right!r}")

    walk(actual, serial, "state")
    return {
        "structural_errors": structural,
        "array_count": len(arrays),
        "exact_array_count": sum(bool(row["equal"]) for row in arrays),
        "close_array_count": sum(bool(row["close"]) for row in arrays),
        "max_abs": max((row["max_abs"] or 0.0 for row in arrays), default=0.0),
        "max_rel": max((row["max_rel"] or 0.0 for row in arrays), default=0.0),
        "failures": [row for row in arrays if not row["close"]],
        "passed": not structural and all(row["close"] for row in arrays),
    }


def _cache_comparison(mx, actual, serial, *, starts, atol: float, rtol: float) -> dict:
    """Compare recurrent state and newly committed logical KV rows.

    Reference KV clones intentionally own exact-length buffers while production
    caches retain append capacity.  Comparing raw ``state`` would therefore
    reject equivalent logical caches.  The historical prefix was equal at the
    prior checkpoint, so this induction compares each newly committed slice.
    """

    layers = []
    for index, (left, right, start) in enumerate(zip(actual, serial, starts)):
        if type(left) is not type(right):
            comparison = {
                "passed": False,
                "structural_errors": [
                    f"cache class mismatch {type(left).__name__} != {type(right).__name__}"
                ],
                "array_count": 0,
                "exact_array_count": 0,
                "close_array_count": 0,
                "max_abs": 0.0,
                "max_rel": 0.0,
                "failures": [],
            }
            kind = "unknown"
        elif hasattr(left, "keys") and hasattr(left, "values"):
            kind = "kv"
            left_offset = int(left.offset)
            right_offset = int(right.offset)
            if left_offset != right_offset or left_offset < int(start):
                comparison = {
                    "passed": False,
                    "structural_errors": [
                        f"offset mismatch {left_offset} != {right_offset} from {start}"
                    ],
                    "array_count": 0,
                    "exact_array_count": 0,
                    "close_array_count": 0,
                    "max_abs": 0.0,
                    "max_rel": 0.0,
                    "failures": [],
                }
            else:
                comparison = _state_comparison(
                    mx,
                    (
                        left.keys[..., int(start):left_offset, :],
                        left.values[..., int(start):left_offset, :],
                        left_offset,
                    ),
                    (
                        right.keys[..., int(start):right_offset, :],
                        right.values[..., int(start):right_offset, :],
                        right_offset,
                    ),
                    atol=atol,
                    rtol=rtol,
                )
        else:
            kind = "recurrent"
            comparison = _state_comparison(
                mx,
                (left.state, left.meta_state),
                (right.state, right.meta_state),
                atol=atol,
                rtol=rtol,
            )
        layers.append({"index": index, "kind": kind, **comparison})
    return {
        "layers": layers,
        "passed": all(row["passed"] for row in layers),
        "max_abs": max((row["max_abs"] for row in layers), default=0.0),
        "max_rel": max((row["max_rel"] for row in layers), default=0.0),
        "array_count": sum(row["array_count"] for row in layers),
        "exact_array_count": sum(row["exact_array_count"] for row in layers),
        "close_array_count": sum(row["close_array_count"] for row in layers),
        "failed_layers": [row["index"] for row in layers if not row["passed"]],
    }


def _serial_decode_path(mx, adapter, cache, layers, inputs, path):
    """Replay accepted draft inputs with ordinary one-token decode geometry."""

    logits = taps = None
    for index in path:
        token_ids = mx.array([[int(inputs[index])]], dtype=mx.uint32)
        logits, taps = adapter.model.forward_with_taps(token_ids, cache, layers)
        mx.eval(logits, taps)
    if logits is None:
        raise RuntimeError("serial decode path is empty")
    return logits, taps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, default=DEFAULT_CALIBRATION
    )
    parser.add_argument(
        "--tensorfold-source", type=Path, default=DEFAULT_TENSORFOLD
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--state-atol", type=float, default=0.01)
    parser.add_argument("--state-rtol", type=float, default=0.01)
    parser.add_argument(
        "--projection-law",
        choices=PROJECTION_LAWS,
        default="crossover",
        help="q4 projection arm: selected crossover or one common lane law",
    )
    parser.add_argument(
        "--fused-gdn",
        choices=FUSED_GDN_ARMS,
        default="on",
        help="ordinary one-token GDN arm (TensorFold retains its tree law)",
    )
    parser.add_argument(
        "--component-oracle",
        action="store_true",
        help="capture tree-root vs ordinary projection inputs and outputs",
    )
    parser.add_argument(
        "--qualification-tensorfold-revision",
        help="full candidate commit for a process-local qualification-only bind",
    )
    parser.add_argument(
        "--qualification-tensorfold-tree",
        help="exact candidate HEAD tree for the qualification-only bind",
    )
    parser.add_argument(
        "--qualification-tensorfold-parent",
        help="exact single parent commit for the qualification-only bind",
    )
    args = parser.parse_args()
    model = args.model.resolve()
    policy_path = args.policy.resolve()
    calibration_path = args.calibration.resolve()
    tensorfold = args.tensorfold_source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing output: {output}")
    qualification_candidate = qualification_candidate_spec(
        args.qualification_tensorfold_revision,
        args.qualification_tensorfold_tree,
        args.qualification_tensorfold_parent,
    )
    policy = explicit_execution_policy(
        json.loads(policy_path.read_text()), args.fused_gdn
    )
    serial_reference = require_explicit_gdn_reference_policy(policy)

    source_commit = git(ROOT, "rev-parse", "HEAD")
    if source_commit != args.expected_source:
        raise RuntimeError("source revision differs from --expected-source")
    production_diff = git(ROOT, "status", "--porcelain", "--", "src", "scripts")
    if production_diff:
        raise RuntimeError("checkpoint parity requires clean src/ and scripts/")
    tensorfold_diff = git(
        tensorfold,
        "status",
        "--porcelain",
        "--",
        "src/tensorfold/kernels/qwen/dense/v1",
    )
    if tensorfold_diff:
        raise RuntimeError("checkpoint parity requires clean imported TensorFold sources")

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MLX2_TENSORFOLD_SOURCE": str(tensorfold),
            "MLX_ENABLE_TF32": "0",
        }
    )
    gpu_owner = require_gpu_owners()
    tensorfold_binding = bind_tensorfold_source(
        tensorfold,
        qualification_candidate,
    )

    import mlx.core as mx
    from paired_direct_ab import state_digest

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.lane import apply_policy
    from mlx2.runtime.lane.policy import detect, resolve
    from mlx2.runtime.sample_utils import LaneRNG

    adapter_class = resolve_adapter(str(model), mtp=False, qualification_mode=True)
    adapter = adapter_class(str(model), execution_policy=policy)

    artifact_config = json.loads((model / "config.json").read_text())
    lane_policy = explicit_projection_policy(
        resolve(
            detect(adapter.model, artifact_config),
            family=adapter.descriptor.family,
            adapter=adapter.lane_policy_defaults(),
            mode="auto",
        ),
        args.projection_law,
    )
    offered = getattr(adapter, "lane_projection_groups", None)
    declared = offered() if callable(offered) else ()
    lane_receipt = apply_policy(adapter.model, lane_policy, declared=declared)
    if not lane_receipt or not lane_receipt.get("covered"):
        raise RuntimeError("selected lane policy covered no projections")

    calibration = json.loads(calibration_path.read_text())
    prompt = calibration["prompts"][str(args.prompt_tokens)]["prompt"]
    tokens = adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    if len(tokens) != args.prompt_tokens:
        raise RuntimeError(f"calibrated prompt produced {len(tokens)} tokens")

    batch = adapter.create_external_batch(
        completion_batch_size=1,
        prefill_step_size=8192,
        stop_tokens=[],
    )
    batch.insert(
        [tokens],
        max_tokens=[args.max_tokens],
        sampling_configs=[{"sampling_temp": 0.0}],
        lane_rngs=[LaneRNG(20261005)],
    )
    original_commit = batch._commit
    original_target_tree_forward = batch._target_tree_forward
    shadow_cache = None
    rounds: list[dict] = []
    failures: list[dict] = []
    continuation_categories: dict[str, dict] = {}
    planted_deepest = None
    component_oracle = None
    capture_stack = contextlib.ExitStack()
    projection_capture = (
        capture_stack.enter_context(ProjectionCapture(adapter.model))
        if args.component_oracle
        else None
    )

    def audited_target_tree_forward(self, current_lane, inputs, parents):
        nonlocal shadow_cache, planted_deepest, component_oracle
        if shadow_cache is None:
            # This hook is the last boundary before TensorFold stages its full
            # speculative window into the live KV cache.  Cloning at commit
            # time would copy those unaccepted rows and double-count replay.
            shadow_cache = self._copy_reference_cache(current_lane.cache)
        planted_cache = None
        serial_cache = None
        if planted_deepest is None and len(inputs) == 16:
            planted_cache = self._copy_reference_cache(current_lane.cache)
            serial_cache = self._copy_reference_cache(current_lane.cache)
        if projection_capture is not None and component_oracle is None and len(inputs) == 16:
            from mlx2.runtime.models.qwen38_fused_gdn import stats as fused_gdn_stats
            from mlx2.runtime.qwen38_tensorfold import forward

            tree_cache = self._copy_reference_cache(current_lane.cache)
            ordinary_cache = self._copy_reference_cache(current_lane.cache)
            with projection_capture.route("tensorfold"):
                tree_logits, _tree_features, tree_transaction = forward(
                    adapter.model,
                    inputs,
                    parents,
                    tree_cache,
                    self.layers,
                    os.environ["MLX2_TENSORFOLD_SOURCE"],
                    cached=True,
                )
            try:
                root_ids = mx.array([[int(inputs[0])]], dtype=mx.uint32)
                fused_gdn_before = fused_gdn_stats(adapter.model)
                with projection_capture.route("ordinary"):
                    ordinary_logits, ordinary_taps = adapter.model.forward_with_taps(
                        root_ids, ordinary_cache, self.layers
                    )
                report = component_report(
                    mx,
                    projection_capture,
                    _array_comparison,
                    atol=args.state_atol,
                    rtol=args.state_rtol,
                )
                mx.eval(tree_logits, ordinary_logits, ordinary_taps)
                fused_gdn_after = fused_gdn_stats(adapter.model)
                report["root_logits"] = _array_comparison(
                    mx,
                    tree_logits[:, :1],
                    ordinary_logits,
                    atol=args.state_atol,
                    rtol=args.state_rtol,
                )
                report["root_argmax_equal"] = int(
                    mx.argmax(tree_logits[0, 0]).item()
                ) == int(mx.argmax(ordinary_logits[0, 0]).item())
                report["tree_rows"] = len(inputs)
                report["ordinary_rows"] = 1
                report["ordinary_fused_gdn"] = fused_gdn_engagement_report(
                    fused_gdn_before,
                    fused_gdn_after,
                    args.fused_gdn,
                )
                component_oracle = report
            finally:
                tree_transaction.abort()
        result = original_target_tree_forward(current_lane, inputs, parents)
        if planted_cache is not None:
            from mlx2.runtime.qwen38_tensorfold import forward

            logits, _features, transaction = forward(
                adapter.model,
                inputs,
                parents,
                planted_cache,
                self.layers,
                os.environ["MLX2_TENSORFOLD_SOURCE"],
                cached=True,
            )
            depths = []
            for row, parent in enumerate(parents):
                depths.append(0 if int(parent) < 0 else depths[int(parent)] + 1)
            children = {int(parent) for parent in parents if int(parent) >= 0}
            leaves = [row for row in range(len(parents)) if row not in children]
            deepest = max(leaves, key=lambda row: (depths[row], -row))
            path = []
            row = deepest
            while row >= 0:
                path.append(row)
                row = int(parents[row])
            path.reverse()
            transaction.commit_paths([path])
            oracle = _transaction_commit_oracle(
                mx,
                planted_cache,
                transaction.record,
                path,
                start=int(transaction.start),
            )
            serial_logits, serial_taps = _serial_decode_path(
                mx,
                adapter,
                serial_cache,
                self.layers,
                inputs,
                path,
            )
            mx.eval(logits, serial_logits, serial_taps)
            probe_token = int(mx.argmax(logits[0, path[-1]]).item())
            continuation = _teacher_forced_continuation(
                mx,
                adapter,
                self,
                planted_cache,
                serial_cache,
                probe_token,
                state_atol=args.state_atol,
                state_rtol=args.state_rtol,
            )
            structure_passed = len(path) - 1 == max(depths) and oracle["passed"]
            planted_deepest = {
                "parents": [int(parent) for parent in parents],
                "path": path,
                "draft_depth": len(path) - 1,
                "maximum_draft_depth": max(depths),
                "transaction_commit_oracle": oracle,
                "continuation": continuation,
                "structure_passed": structure_passed,
                "semantic_continuation_passed": continuation["passed"],
                "passed": structure_passed and continuation["passed"],
            }
        return result

    def audited_commit(self, cohort, decisions, features, *, blocks, transaction):
        if len(cohort) != 1:
            raise RuntimeError("checkpoint parity harness is B1-only")
        current_lane = cohort[0]
        if shadow_cache is None:
            raise RuntimeError("target-forward checkpoint hook did not run")
        starts = [
            int(transaction.start) if hasattr(item, "offset") else 0
            for item in current_lane.cache
        ]
        selected_path = list(
            decisions[0].commit_rows
            if decisions[0].commit_rows is not None
            else range(min(decisions[0].accepted + 1, len(decisions[0].emitted)))
        )
        block = blocks[0]
        if block is None:
            raise RuntimeError("tree checkpoint audit requires a proposal block")
        cache_inputs = [current_lane.anchor, *list(block.tokens)]
        if any(index < 0 or index >= len(cache_inputs) for index in selected_path):
            raise RuntimeError("tree commit path is outside the target input rows")
        committed_cache_inputs = [cache_inputs[index] for index in selected_path]
        transaction_record = transaction.record
        old_history = len(current_lane.history)
        original_commit(
            cohort,
            decisions,
            features,
            blocks=blocks,
            transaction=transaction,
        )
        emitted_outputs = list(current_lane.history[old_history:])
        for token in committed_cache_inputs:
            replay_ids = mx.array([[int(token)]], dtype=mx.uint32)
            logits, taps = adapter.model.forward_with_taps(
                replay_ids, shadow_cache, self.layers
            )
            mx.eval(logits, taps)

        actual = state_digest(current_lane.cache)
        expected = state_digest(shadow_cache)
        logical = _cache_comparison(
            mx,
            current_lane.cache,
            shadow_cache,
            starts=starts,
            atol=args.state_atol,
            rtol=args.state_rtol,
        )
        commit_oracle = _transaction_commit_oracle(
            mx,
            current_lane.cache,
            transaction_record,
            selected_path,
            start=int(transaction.start),
        )

        expected_offset = int(transaction.start) + len(committed_cache_inputs)
        kv_offsets = [
            {
                "index": index,
                "actual": int(item.offset),
                "expected": expected_offset,
                "equal": int(item.offset) == expected_offset,
            }
            for index, item in enumerate(current_lane.cache)
            if hasattr(item, "keys") and hasattr(item, "values")
        ]
        recurrent_finite = []
        for index, item in enumerate(current_lane.cache):
            if hasattr(item, "keys") and hasattr(item, "values"):
                continue
            finite = _state_comparison(
                mx,
                (item.state, item.meta_state),
                (item.state, item.meta_state),
                atol=0.0,
                rtol=0.0,
            )
            recurrent_finite.append({"index": index, "finite": finite["passed"]})
        checkpoint = {
            "transaction_start": int(transaction.start),
            "committed_rows": len(committed_cache_inputs),
            "expected_offset": expected_offset,
            "kv_offsets": kv_offsets,
            "all_kv_offsets_equal": all(row["equal"] for row in kv_offsets),
            "recurrent_finite": recurrent_finite,
            "all_recurrent_finite": all(row["finite"] for row in recurrent_finite),
        }

        actual_probe = self._copy_reference_cache(current_lane.cache)
        serial_probe = self._copy_reference_cache(shadow_cache)
        probe_ids = mx.array([[int(current_lane.anchor)]], dtype=mx.uint32)
        actual_logits, actual_taps = adapter.model.forward_with_taps(
            probe_ids, actual_probe, self.layers
        )
        serial_logits, serial_taps = adapter.model.forward_with_taps(
            probe_ids, serial_probe, self.layers
        )
        mx.eval(actual_logits, actual_taps, serial_logits, serial_taps)
        continuation = _array_comparison(
            mx,
            actual_logits,
            serial_logits,
            atol=args.state_atol,
            rtol=args.state_rtol,
        )
        continuation["actual_argmax"] = int(mx.argmax(actual_logits[0, -1]).item())
        continuation["serial_argmax"] = int(mx.argmax(serial_logits[0, -1]).item())
        continuation["argmax_equal"] = (
            continuation["actual_argmax"] == continuation["serial_argmax"]
        )

        proposal_count = int(block.lengths[0])
        accepted = min(decisions[0].accepted, len(decisions[0].emitted) - 1)
        parents = [] if block is None else self._target_tree_parents(block)
        depths = []
        for row, parent in enumerate(parents):
            depths.append(0 if parent < 0 else depths[int(parent)] + 1)
        children = {int(parent) for parent in parents if int(parent) >= 0}
        leaves = [row for row in range(len(parents)) if row not in children]
        max_depth = max((depths[row] for row in leaves), default=0)
        commit_rows = list(
            decisions[0].commit_rows or range(len(committed_cache_inputs))
        )
        deepest_leaf = bool(
            proposal_count == 15
            and commit_rows
            and commit_rows[-1] in leaves
            and depths[commit_rows[-1]] == max_depth
        )
        category = None
        if proposal_count == 15 and accepted == 0:
            category = "zero_accept"
        elif proposal_count == 15 and accepted > 0 and not deepest_leaf:
            category = "strict_partial"
        elif 0 < proposal_count < 15:
            category = "terminal_shortened"
        category_continuation = None
        if category is not None and category not in continuation_categories:
            category_continuation = _teacher_forced_continuation(
                mx,
                adapter,
                self,
                current_lane.cache,
                shadow_cache,
                current_lane.anchor,
                state_atol=args.state_atol,
                state_rtol=args.state_rtol,
            )
            continuation_categories[category] = category_continuation
        record = {
            "round": len(rounds),
            "span": proposal_count + 1,
            "accepted_drafts": accepted,
            "committed_cache_inputs": committed_cache_inputs,
            "emitted_outputs": emitted_outputs,
            "actual_cache": actual,
            "serial_cache": expected,
            "logical_state": logical,
            "logical_state_divergence": layer_divergence_summary(logical["layers"]),
            "transaction_commit_oracle": commit_oracle,
            "checkpoint": checkpoint,
            "commit_rows": commit_rows,
            "tree_max_depth": max_depth,
            "reached_deepest_leaf": deepest_leaf,
            "continuation_category": category,
            "category_continuation": category_continuation,
            "continuation": continuation,
            # TensorFold and the explicitly selected/labelled serial GDN replay use
            # different fused/reduction geometry.  Raw/numeric state deltas
            # remain diagnostic; the qualification gate is the authoritative
            # accepted boundary, finite recurrent publication and identical
            # next greedy decision from every checkpoint.
            "equal": checkpoint["all_kv_offsets_equal"]
            and checkpoint["all_recurrent_finite"]
            and commit_oracle["passed"]
            and continuation["argmax_equal"],
        }
        rounds.append(record)
        if not record["equal"]:
            failures.append(record)

    batch._target_tree_forward = types.MethodType(audited_target_tree_forward, batch)
    batch._commit = types.MethodType(audited_commit, batch)
    output_tokens: list[int] = []
    final_receipt = None
    idle = 0
    try:
        while batch.lanes:
            _progress, responses = batch.next()
            idle = 0 if responses else idle + 1
            if idle > 4096:
                raise RuntimeError("generator made no response progress")
            for response in responses:
                output_tokens.append(int(response.token))
                final_receipt = response.speculative_receipt
    finally:
        try:
            batch.close()
        finally:
            capture_stack.close()

    patterns = {(row["span"], row["accepted_drafts"]) for row in rounds}
    has_partial = any(span == 16 and 0 < accepted < 15 for span, accepted in patterns)
    has_zero = (16, 0) in patterns
    has_deepest_structure = bool(
        planted_deepest and planted_deepest["structure_passed"]
    )
    has_deepest_continuation = bool(
        planted_deepest and planted_deepest["semantic_continuation_passed"]
    )
    all_complete = all(
        row["actual_cache"].get("status") == "complete"
        and row["serial_cache"].get("status") == "complete"
        for row in rounds
    )
    continuation_coverage = all(
        continuation_categories.get(name, {}).get("passed")
        for name in ("zero_accept", "strict_partial", "terminal_shortened")
    )
    component_oracle_valid = (
        not args.component_oracle
        or (
            component_oracle is not None
            and component_oracle.get("complete") is True
            and (component_oracle.get("ordinary_fused_gdn") or {}).get("passed")
            is True
        )
    )
    passed = bool(
        rounds
        and not failures
        and all_complete
        and has_partial
        and has_zero
        and has_deepest_structure
        and has_deepest_continuation
        and continuation_coverage
        and component_oracle_valid
    )
    result = {
        "schema": "mlx2.tensorfold-recurrent-checkpoint-parity.v2",
        "source_commit": source_commit,
        "source_production_diff": production_diff,
        "gpu_owner": gpu_owner,
        "model": str(model),
        "model_config_sha256": sha256(model / "config.json"),
        "policy": str(policy_path),
        "policy_sha256": sha256(policy_path),
        "serial_reference": serial_reference,
        "tensorfold_commit": git(tensorfold, "rev-parse", "HEAD"),
        "tensorfold_binding": tensorfold_binding,
        "tensorfold_source_diff": tensorfold_diff,
        "tensorfold_source_sha256": {
            str(path.relative_to(tensorfold)): sha256(path)
            for path in sorted(
                (tensorfold / "src/tensorfold/kernels/qwen/dense/v1").glob("*.py")
            )
        },
        "mlx_version": mx.__version__,
        "runner_sha256": sha256(Path(__file__)),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "lane_policy": lane_receipt,
        "oracle_arm": {
            "projection_law": args.projection_law,
            "q4_min_rows": lane_policy["min_rows"]["q4"],
            "fused_gdn": args.fused_gdn,
            "component_capture": bool(args.component_oracle),
        },
        "component_oracle_helper_sha256": sha256(
            ROOT / "scripts/qwen38_tensorfold_component_oracle.py"
        ),
        "component_oracle": component_oracle,
        "prompt_tokens": len(tokens),
        "completion_tokens": len(output_tokens),
        "output_token_sha256": hashlib.sha256(
            json.dumps(output_tokens, separators=(",", ":")).encode()
        ).hexdigest(),
        "rounds": rounds,
        "planted_deepest_branch": planted_deepest,
        "continuation_categories": continuation_categories,
        "requirements": {
            "all_rounds_checkpoint_semantics": not failures,
            "complete_state_and_metadata": all_complete,
            "strict_partial_full_tree": has_partial,
            "zero_accept_full_tree": has_zero,
            "deepest_leaf_full_tree": has_deepest_structure,
            "deepest_leaf_semantic_continuation": has_deepest_continuation,
            "eight_step_semantic_continuation_coverage": continuation_coverage,
            "component_oracle_identity_and_engagement": component_oracle_valid,
        },
        "state_tolerance": {"atol": args.state_atol, "rtol": args.state_rtol},
        "final_speculation_receipt": final_receipt,
        "passed": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "rounds": len(rounds),
                "failures": len(failures),
                "requirements": result["requirements"],
                "patterns": sorted([list(pattern) for pattern in patterns]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    mx.clear_cache()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
