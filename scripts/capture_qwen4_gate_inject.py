#!/usr/bin/env python3
"""Capture the Flash-Next T=1 hyper-connection gate/inject dispatches.

This is the observation gate for the llama.cpp ``03a667a`` follow-up.  It
contains no candidate fusion.  The harness constructs the shipped Qwen4-Exp
``DecoderLayer`` class at the artifact's exact geometry, dtype and
quantization, warms it, then records one lazy Metal evaluation.  A second
capture isolates the graph that matters for the proposed optimization:

``block_inject_weight(normed) / hc_count -> sigmoid -> * 2 -> broadcast ->
 residual + branch * inject``.

The layer uses random weights because dispatch geometry is determined by
shape, dtype and quantization.  It is therefore a kernel/command observation,
not a model-output or performance qualification.  Run only through the live
CPG lease, the shared host lock and ``scripts/run_with_gpu_fcntl.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from profile_prefill_components import (
    build_layer,
    causal_mask,
    dtype_audit,
    load_args,
    pin_production_environment,
)

SCHEMA = "mlx2.qwen4-gate-inject-dispatch-capture.v1"
UPSTREAM_REVISION = "03a667aa304f2a8e02a9a02b2e3fb45d64bcae7f"
HOST_LOCK = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_receipt(path: Path) -> dict:
    """Hash a capture bundle without assuming ``.gputrace`` is a file."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {
            "path": str(path),
            "exists": True,
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    files = sorted(item for item in path.rglob("*") if item.is_file())
    manifest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest = _sha256(item)
        manifest.update(f"{relative}\0{size}\0{digest}\n".encode())
        total_bytes += size
    return {
        "path": str(path),
        "exists": True,
        "kind": "directory",
        "files": len(files),
        "bytes": total_bytes,
        "manifest_sha256": manifest.hexdigest(),
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_gpu_ownership() -> dict:
    """Require CPG's host lock plus the inherited /tmp advisory lock."""
    if not HOST_LOCK.is_file():
        raise RuntimeError(f"missing CPG host GPU lock receipt: {HOST_LOCK}")
    if os.environ.get("MLX2_GPU_FCNTL_LOCKED") != "1":
        raise RuntimeError(
            "missing /tmp/gpu.lock ownership; run through "
            "scripts/run_with_gpu_fcntl.py"
        )
    receipt = json.loads(HOST_LOCK.read_text())
    if receipt.get("cpg_used") is False:
        raise RuntimeError("host GPU lock explicitly says CPG was not used")
    return receipt


def build_plan(model: Path, *, kind: str, capture_dir: Path) -> dict:
    args, quant, config = load_args(model)
    text = config.get("text_config", config)
    return {
        "schema": SCHEMA,
        "upstream_revision": UPSTREAM_REVISION,
        "model_config": str(model / "config.json"),
        "model_config_sha256": _sha256(model / "config.json"),
        "weights_loaded": False,
        "method": (
            "one shipped DecoderLayer class with artifact geometry, dtype and "
            "quantization; random weights"
        ),
        "kind": kind,
        "batch": 1,
        "tokens": 1,
        "geometry": {
            "hidden_size": args.hidden_size,
            "hc_count": args.hc_count,
            "stream_width": args.hidden_size * args.hc_count,
            "hc_lowrank": args.hc_lowrank,
        },
        "quantization": {
            "group_size": quant.get("group_size", 64),
            "bits": quant.get("bits", 4),
            "mode": quant.get("mode", "affine"),
        },
        "dtype": str(text.get("dtype") or "bfloat16"),
        "captures": {
            "whole_layer": str(capture_dir / f"qwen4-{kind}-t1-layer.gputrace"),
            "gate_inject": str(capture_dir / "qwen4-t1-raw-gate-inject.gputrace"),
        },
        "graphs": {
            "whole_layer": str(capture_dir / f"qwen4-{kind}-t1-layer.dot"),
            "gate_inject": str(capture_dir / "qwen4-t1-raw-gate-inject.dot"),
        },
        "claim_boundary": (
            "dispatch observation only; not a fused implementation, model "
            "qualification, selection, or performance result"
        ),
    }


def _capture(mx, path: Path, build_output):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite capture: {path}")
    # Flush an encoder that MLX may have opened while materializing inputs.
    # Build the lazy graph only after capture starts: on this MLX revision,
    # constructing it first can encode the dispatch before ``start_capture``.
    mx.synchronize()
    mx.metal.start_capture(str(path))
    try:
        output = build_output()
        mx.eval(output)
        # ``mx.eval`` may return before Metal commits/completes the command
        # buffer.  Keep the synchronization inside the capture boundary or
        # Xcode can replay a structurally valid but command-empty bundle.
        mx.synchronize()
    finally:
        mx.metal.stop_capture()
    return output


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
    parser.add_argument(
        "--kind",
        choices=("linear_attention", "full_attention"),
        default="linear_attention",
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help=(
            "capture only the exact post-projection gate/inject seam; useful "
            "for the implementation go/no-go before the full-layer trace"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()

    env_profile = pin_production_environment(args.model)
    plan = build_plan(args.model, kind=args.kind, capture_dir=args.capture_dir)
    plan["production_environment"] = env_profile
    plan["gate_only"] = args.gate_only
    plan["git_revision"] = _git_head()
    plan["script_sha256"] = _sha256(Path(__file__))
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return 0
    if not args.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")
    ownership = require_gpu_ownership()

    import mlx.core as mx

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("Metal capture requires the MLX GPU device")

    model_args, quant, config = load_args(args.model)
    text = config.get("text_config", config)
    dtype = getattr(mx, str(text.get("dtype") or "bfloat16"))
    layer = build_layer(model_args, args.kind, quant, dtype)
    param_dtype_mb, param_mb = dtype_audit(mx, layer)

    batch = 1
    tokens = 1
    width = model_args.hc_count * model_args.hidden_size
    residual = mx.random.normal((batch, tokens, width)).astype(dtype)
    branch = mx.random.normal((batch, tokens, model_args.hidden_size)).astype(dtype)
    input_ids = mx.zeros((batch, tokens), dtype=mx.uint32)
    mask = None if args.kind == "linear_attention" else causal_mask(mx, tokens)
    mx.eval(residual, branch)

    if not args.gate_only:
        # Warm the complete layer before capture so compilation does not pollute
        # the one-evaluation command stream.
        mx.eval(layer(residual, input_ids, mask=mask, cache=None))
        mx.clear_cache()
        whole = _capture(
            mx,
            Path(plan["captures"]["whole_layer"]),
            lambda: layer(residual, input_ids, mask=mask, cache=None),
        )
        mx.export_to_dot(plan["graphs"]["whole_layer"], whole)

    # Materialize one projection output, then warm the exact post-projection
    # chain outside the capture boundary.  Do not reuse these arrays for the
    # capture: MLX can retain an evaluated lazy node even after ``clear_cache``
    # and would then record only an empty commit/wait command buffer.
    gr = layer.attn_hyper_connection
    normed = gr.hc_norm(residual)
    raw_gate = gr.block_inject_weight(normed)
    mx.eval(raw_gate, residual, branch)
    warm_inject = 2 * mx.sigmoid(raw_gate / model_args.hc_count)
    warm_output = residual + (
        branch[..., None, :] * warm_inject[..., None]
    ).reshape(
        residual.shape
    )
    mx.eval(warm_output)
    mx.synchronize()

    # Fresh, independently materialized inputs keep compilation out of the
    # trace without allowing the warm output to satisfy the captured graph.
    # The raw gate still comes from the shipped projection at exact artifact
    # geometry; only the checkpoint weight values are synthetic.
    residual = mx.random.normal((batch, tokens, width)).astype(dtype)
    branch = mx.random.normal((batch, tokens, model_args.hidden_size)).astype(dtype)
    normed = gr.hc_norm(residual)
    raw_gate = gr.block_inject_weight(normed)
    mx.eval(raw_gate, residual, branch)
    mx.synchronize()

    # This makes the command stream answer the exact question from 03a667a:
    # whether scale -> sigmoid -> scale -> broadcast multiply -> residual add
    # remains more than one dispatch after MLX's own lazy graph fusion.  The
    # whole-layer capture above retains the projection and normalization.
    def build_gate_inject():
        captured_inject = 2 * mx.sigmoid(raw_gate / model_args.hc_count)
        captured_output = residual + (
            branch[..., None, :] * captured_inject[..., None]
        ).reshape(residual.shape)
        return captured_output, captured_inject

    gate_inject, inject = _capture(
        mx, Path(plan["captures"]["gate_inject"]), build_gate_inject
    )

    # Export after capture: graph rendering is diagnostic metadata and must
    # never be allowed to materialize or otherwise satisfy the captured node.
    graph_inject = 2 * mx.sigmoid(raw_gate / model_args.hc_count)
    graph_output = residual + (
        branch[..., None, :] * graph_inject[..., None]
    ).reshape(residual.shape)
    mx.export_to_dot(
        plan["graphs"]["gate_inject"],
        graph_output,
        raw_gate=raw_gate,
        inject=graph_inject,
    )

    report = dict(plan)
    report.update(
        {
            "mlx_version": mx.__version__,
            "device": mx.metal.device_info(),
            "host_lock": ownership,
            "parameter_mb": param_mb,
            "parameter_mb_by_dtype": param_dtype_mb,
            "observed_shapes": {
                "residual": list(residual.shape),
                "branch": list(branch.shape),
                "raw_gate": list(raw_gate.shape),
                "inject": list(inject.shape),
                "output": list(gate_inject.shape),
            },
            "observed_dtypes": {
                "residual": str(residual.dtype),
                "branch": str(branch.dtype),
                "raw_gate": str(raw_gate.dtype),
                "inject": str(inject.dtype),
                "output": str(gate_inject.dtype),
            },
            "capture_files": {
                name: _bundle_receipt(Path(path))
                for name, path in plan["captures"].items()
            },
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
