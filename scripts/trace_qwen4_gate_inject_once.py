#!/usr/bin/env python3
"""Run one warm-isolated Flash-Next raw-gate-plus-inject evaluation.

This helper is designed for an external Instruments ``Metal System Trace`` or
LLDB dispatch breakpoints.  It prepares and warms the exact BF16 T=1 layer
geometry, writes a ready receipt, then waits.  After a profiler attaches,
touching ``--go-file`` builds and evaluates the selected target exactly once.
The process remains alive until ``--release-file`` appears so the profiler can
flush while GPU ownership is still held.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from capture_qwen4_gate_inject import (
    SCHEMA,
    UPSTREAM_REVISION,
    _git_head,
    _sha256,
    require_gpu_ownership,
)
from profile_prefill_components import (
    build_layer,
    dtype_audit,
    load_args,
    pin_production_environment,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _wait_for(path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "~/mlx-models/"
            "Qwen3.8-Flash-Next-MLX-4bit-MTP"
        ),
    )
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--go-file", type=Path, required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="trace the one-dispatch candidate instead of the eager reference",
    )
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()

    for path in (args.ready_file, args.go_file, args.release_file, args.receipt):
        if path.exists():
            raise RuntimeError(f"refusing stale/overwrite path: {path}")
    if not args.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")

    env_profile = pin_production_environment(args.model)
    ownership = require_gpu_ownership()

    import mlx.core as mx

    from mlx2.runtime.models import qwen4_gate_inject

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("Metal System Trace requires the MLX GPU device")

    model_args, quant, config = load_args(args.model)
    text = config.get("text_config", config)
    dtype = getattr(mx, str(text.get("dtype") or "bfloat16"))
    layer = build_layer(model_args, "linear_attention", quant, dtype)
    param_dtype_mb, param_mb = dtype_audit(mx, layer)

    batch = tokens = 1
    width = model_args.hc_count * model_args.hidden_size
    gr = layer.attn_hyper_connection

    # Compile/warm the exact target chain on independent arrays.
    warm_residual = mx.random.normal((batch, tokens, width)).astype(dtype)
    warm_branch = mx.random.normal(
        (batch, tokens, model_args.hidden_size)
    ).astype(dtype)
    warm_raw = gr.block_inject_weight(gr.hc_norm(warm_residual))
    mx.eval(warm_residual, warm_branch, warm_raw)
    if args.candidate:
        qwen4_gate_inject.set_fused_gate_inject_enabled(True)
        warm_output = qwen4_gate_inject.try_qwen4_gate_inject(
            warm_residual, warm_branch, warm_raw
        )
        if warm_output is None:
            raise RuntimeError("candidate declined the exact warm geometry")
    else:
        warm_output = qwen4_gate_inject.eager_qwen4_gate_inject(
            warm_residual, warm_branch, warm_raw
        )
    mx.eval(warm_output)
    mx.synchronize()

    # Materialize fresh target inputs.  No target graph exists before GO.
    residual = mx.random.normal((batch, tokens, width)).astype(dtype)
    branch = mx.random.normal((batch, tokens, model_args.hidden_size)).astype(dtype)
    raw_gate = gr.block_inject_weight(gr.hc_norm(residual))
    mx.eval(residual, branch, raw_gate)
    mx.synchronize()

    base = {
        "schema": f"{SCHEMA}.instruments-once.v1",
        "upstream_revision": UPSTREAM_REVISION,
        "git_revision": _git_head(),
        "script_sha256": _sha256(Path(__file__)),
        "model_config": str(args.model / "config.json"),
        "model_config_sha256": _sha256(args.model / "config.json"),
        "weights_loaded": False,
        "pid": os.getpid(),
        "mlx_version": mx.__version__,
        "device": mx.metal.device_info(),
        "host_lock": ownership,
        "production_environment": env_profile,
        "geometry": {
            "batch": batch,
            "tokens": tokens,
            "hidden_size": model_args.hidden_size,
            "hc_count": model_args.hc_count,
            "stream_width": width,
            "hc_lowrank": model_args.hc_lowrank,
        },
        "dtype": str(dtype),
        "target": "candidate" if args.candidate else "eager_reference",
        "parameter_mb": param_mb,
        "parameter_mb_by_dtype": param_dtype_mb,
        "claim_boundary": (
            "one externally traced execution; not a fused implementation, "
            "model qualification, selection, or stable performance result"
        ),
    }
    _write_json(
        args.ready_file,
        {**base, "state": "ready", "ready_at_ns": time.time_ns()},
    )
    _wait_for(args.go_file, args.timeout_s)

    started_ns = time.time_ns()
    started = time.perf_counter_ns()
    if args.candidate:
        output = qwen4_gate_inject.try_qwen4_gate_inject(
            residual, branch, raw_gate
        )
        if output is None:
            raise RuntimeError("candidate declined the exact target geometry")
        inject = None
        mx.eval(output)
    else:
        inject = 2 * mx.sigmoid(raw_gate / model_args.hc_count)
        output = residual + (branch[..., None, :] * inject[..., None]).reshape(
            residual.shape
        )
        mx.eval(output, inject)
    mx.synchronize()
    elapsed_ns = time.perf_counter_ns() - started
    finished_ns = time.time_ns()

    report = {
        **base,
        "state": "evaluated",
        "target_started_at_ns": started_ns,
        "target_finished_at_ns": finished_ns,
        "wall_elapsed_ns": elapsed_ns,
        "observed_shapes": {
            "residual": list(residual.shape),
            "branch": list(branch.shape),
            "raw_gate": list(raw_gate.shape),
            "inject": None if inject is None else list(inject.shape),
            "output": list(output.shape),
        },
        "observed_dtypes": {
            "residual": str(residual.dtype),
            "branch": str(branch.dtype),
            "raw_gate": str(raw_gate.dtype),
            "inject": None if inject is None else str(inject.dtype),
            "output": str(output.dtype),
        },
        "candidate_status": (
            qwen4_gate_inject.qwen4_gate_inject_stats()
            if args.candidate
            else None
        ),
    }
    _write_json(args.receipt, report)
    _wait_for(args.release_file, args.timeout_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
