#!/usr/bin/env python3
"""Bounded model-level qualification for the Qwen4 gate/inject candidate.

One model load exercises ordinary B1 decode, B4 T=1 decode, and B1 T=3
self-MTP verification independently.  Every arm starts from a fresh cache
prefilled through the ordinary path.  The numerical pass compares the bytes
after every trunk layer, logits, top token, and persisted cache state.  A
separate unsynchronized-layer pass records target-forward wall latency.

This is a candidate qualification probe, not a serving selection or a stable
performance benchmark.  It requires the CPG GPU lease plus both host locks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from capture_qwen4_gate_inject import require_gpu_ownership


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _array_payload(mx, value) -> bytes:
    mx.eval(value)
    raw = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
    prefix = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
    ).encode()
    return prefix + b"\0" + np.asarray(raw).tobytes()


def _array_digest(mx, value) -> str:
    return hashlib.sha256(_array_payload(mx, value)).hexdigest()


def _update_digest(mx, digest, value) -> None:
    array_type = type(mx.array([0]))
    if isinstance(value, array_type):
        digest.update(b"mlx\0")
        digest.update(_array_payload(mx, value))
    elif isinstance(value, np.ndarray):
        digest.update(b"numpy\0")
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes())
    elif value is None:
        digest.update(b"none\0")
    elif isinstance(value, bool):
        digest.update(b"bool\0" + str(value).encode())
    elif isinstance(value, int):
        digest.update(b"int\0" + str(value).encode())
    elif isinstance(value, float):
        digest.update(b"float\0" + value.hex().encode())
    elif isinstance(value, str):
        digest.update(b"str\0" + value.encode())
    elif isinstance(value, bytes):
        digest.update(b"bytes\0" + value)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(str(len(value)).encode() + b"\0")
        for item in value:
            _update_digest(mx, digest, item)
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=str):
            _update_digest(mx, digest, key)
            _update_digest(mx, digest, value[key])
    else:
        raise TypeError(f"unsupported cache state leaf: {type(value)!r}")


def _cache_digest(mx, cache) -> str:
    digest = hashlib.sha256()
    for index, entry in enumerate(cache):
        digest.update(f"layer:{index}:{type(entry).__name__}\0".encode())
        _update_digest(mx, digest, entry.state)
        _update_digest(mx, digest, entry.meta_state)
    return digest.hexdigest()


def _compare(mx, left, right) -> dict:
    same = mx.array_equal(left, right)
    maximum = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    mx.eval(same, maximum)
    return {
        "exact": bool(same.item()),
        "max_abs": float(maximum.item()),
        "reference_sha256": _array_digest(mx, left),
        "candidate_sha256": _array_digest(mx, right),
    }


def _start_speculation(cache) -> None:
    started = []
    try:
        for entry in cache:
            entry.start_speculation()
            started.append(entry)
    except BaseException:
        for entry in reversed(started):
            entry.stop_speculation()
        raise


def _stop_speculation(cache) -> None:
    first = None
    for entry in cache:
        try:
            entry.stop_speculation()
        except Exception as exc:  # noqa: BLE001 - close every cache first
            if first is None:
                first = exc
    if first is not None:
        raise first


def _forward(mx, model, token_rows, cache, mode):
    tokens = mx.array(token_rows, dtype=mx.uint32)
    if mode == "self_mtp_verify":
        hidden, _ = model.mtp_backbone(tokens, cache=cache)
        return model.logits(hidden)
    return model(tokens, cache=cache)


def _prefill(mx, model, rows):
    from mlx2.runtime.models import qwen4_gate_inject as gate_inject

    gate_inject.set_fused_gate_inject_enabled(False)
    cache = model.make_cache()
    logits = model(mx.array(rows, dtype=mx.uint32), cache=cache)
    mx.eval(logits)
    return cache


def _run_numerical_arm(
    mx,
    model,
    rows,
    target,
    mode,
    *,
    candidate,
    reference_layers=None,
):
    from mlx2.runtime.models import qwen4_gate_inject as gate_inject
    from mlx2.runtime.models.qwen4_exp import DecoderLayer

    cache = _prefill(mx, model, rows)
    seed_digest = _cache_digest(mx, cache)
    if mode == "self_mtp_verify":
        _start_speculation(cache)
    gate_inject.set_fused_gate_inject_enabled(candidate)
    gate_inject.reset_qwen4_gate_inject_stats()

    layer_ids = {id(layer): index for index, layer in enumerate(model.layers)}
    captured = {}
    original_call = DecoderLayer.__call__

    def traced_call(layer, *args, **kwargs):
        output = original_call(layer, *args, **kwargs)
        index = layer_ids.get(id(layer))
        if index is not None:
            mx.eval(output)
            if reference_layers is None:
                captured[index] = output
            else:
                captured[index] = _compare(mx, reference_layers[index], output)
        return output

    DecoderLayer.__call__ = traced_call
    try:
        started = time.perf_counter_ns()
        logits = _forward(mx, model, target, cache, mode)
        mx.eval(logits)
        traced_ms = (time.perf_counter_ns() - started) / 1e6
        state_digest = _cache_digest(mx, cache)
        status = gate_inject.qwen4_gate_inject_stats()
        top_tokens = [
            int(value)
            for value in np.asarray(mx.argmax(logits[:, -1], axis=-1)).tolist()
        ]
    finally:
        DecoderLayer.__call__ = original_call
        gate_inject.set_fused_gate_inject_enabled(False)
        if mode == "self_mtp_verify":
            _stop_speculation(cache)

    return {
        "cache": cache,
        "seed_cache_sha256": seed_digest,
        "cache_state_sha256": state_digest,
        "layers": captured,
        "logits": logits,
        "logits_sha256": _array_digest(mx, logits),
        "top_tokens": top_tokens,
        "traced_wall_ms": traced_ms,
        "status": status,
    }


def _run_latency_arm(mx, model, rows, target, mode, *, candidate):
    from mlx2.runtime.models import qwen4_gate_inject as gate_inject

    cache = _prefill(mx, model, rows)
    if mode == "self_mtp_verify":
        _start_speculation(cache)
    gate_inject.set_fused_gate_inject_enabled(candidate)
    gate_inject.reset_qwen4_gate_inject_stats()
    try:
        mx.clear_cache()
        started = time.perf_counter_ns()
        logits = _forward(mx, model, target, cache, mode)
        mx.eval(logits)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        status = gate_inject.qwen4_gate_inject_stats()
    finally:
        gate_inject.set_fused_gate_inject_enabled(False)
        if mode == "self_mtp_verify":
            _stop_speculation(cache)
    return {"wall_ms": elapsed_ms, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lease", required=True)
    parser.add_argument("--latency-repeats", type=int, default=2)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("refusing Metal execution without --i-own-the-gpu")
    if args.latency_repeats < 1:
        parser.error("--latency-repeats must be positive")
    locks = require_gpu_ownership()
    if locks.get("cpg_task") != args.cpg_lease:
        raise RuntimeError("CPG GPU lease task does not match --cpg-lease")

    import mlx.core as mx

    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.runtime.models import qwen4_gate_inject as gate_inject

    mx.set_default_device(mx.gpu)
    started_at = time.time()
    adapter = FlashNextAdapter(str(args.model))
    try:
        model = adapter.model
        mx.eval(model.parameters())
        prompt = [
            int(token)
            for token in adapter.tokenizer.encode(
                "Exact state makes speculative decoding trustworthy.",
                add_special_tokens=False,
            )
        ]
        if len(prompt) < 6:
            prompt = (prompt * 6)[:6]
        else:
            prompt = prompt[:6]
        scenarios = (
            ("ordinary_b1", 1, 1),
            ("batched_decode", 4, 1),
            ("self_mtp_verify", 1, 3),
        )
        results = {}
        for mode, batch, width in scenarios:
            rows = [prompt[:] for _ in range(batch)]
            target = [
                [prompt[(row + step) % len(prompt)] for step in range(width)]
                for row in range(batch)
            ]
            reference = _run_numerical_arm(
                mx, model, rows, target, mode, candidate=False
            )
            candidate = _run_numerical_arm(
                mx,
                model,
                rows,
                target,
                mode,
                candidate=True,
                reference_layers=reference["layers"],
            )
            layer_results = candidate["layers"]
            logits = _compare(mx, reference["logits"], candidate["logits"])
            latency = {"reference_ms": [], "candidate_ms": []}
            latency_status = []
            for repetition in range(args.latency_repeats):
                order = (False, True) if repetition % 2 == 0 else (True, False)
                for arm in order:
                    observed = _run_latency_arm(
                        mx, model, rows, target, mode, candidate=arm
                    )
                    key = "candidate_ms" if arm else "reference_ms"
                    latency[key].append(observed["wall_ms"])
                    latency_status.append(
                        {"candidate": arm, "status": observed["status"]}
                    )
            ref_median = statistics.median(latency["reference_ms"])
            cand_median = statistics.median(latency["candidate_ms"])
            results[mode] = {
                "geometry": {"batch": batch, "tokens": width},
                "expected_candidate_calls": 2 * len(model.layers),
                "seed_cache_exact": (
                    reference["seed_cache_sha256"]
                    == candidate["seed_cache_sha256"]
                ),
                "seed_cache_sha256": reference["seed_cache_sha256"],
                "layer_count": len(layer_results),
                "layers_exact": all(
                    row["exact"] for row in layer_results.values()
                ),
                "first_layer_divergence": next(
                    (
                        index
                        for index, row in sorted(layer_results.items())
                        if not row["exact"]
                    ),
                    None,
                ),
                "layer_results": layer_results,
                "logits": logits,
                "top_tokens_equal": (
                    reference["top_tokens"] == candidate["top_tokens"]
                ),
                "reference_top_tokens": reference["top_tokens"],
                "candidate_top_tokens": candidate["top_tokens"],
                "cache_state_exact": (
                    reference["cache_state_sha256"]
                    == candidate["cache_state_sha256"]
                ),
                "reference_cache_state_sha256": reference[
                    "cache_state_sha256"
                ],
                "candidate_cache_state_sha256": candidate[
                    "cache_state_sha256"
                ],
                "candidate_status": candidate["status"],
                "latency": {
                    **latency,
                    "reference_median_ms": ref_median,
                    "candidate_median_ms": cand_median,
                    "candidate_delta_percent": 100 * (cand_median / ref_median - 1),
                    "status": latency_status,
                    "stable_performance_claim": False,
                },
            }
            del reference, candidate
            mx.clear_cache()

        exact = all(
            row["seed_cache_exact"]
            and row["layers_exact"]
            and row["logits"]["exact"]
            and row["top_tokens_equal"]
            and row["cache_state_exact"]
            for row in results.values()
        )
        engaged = all(
            row["candidate_status"]["counts"].get(f"calls:{mode}", 0)
            == row["expected_candidate_calls"]
            and row["candidate_status"]["counts"].get("declines", 0) == 0
            for mode, row in results.items()
        )
        report = {
            "schema": "mlx2.qwen4-gate-inject-model-qualification.v1",
            "implemented": True,
            "model_validation_passed": exact and engaged,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "observed_used_boundary": (
                "Set only by a separate physical Metal dispatch observation; "
                "Python launch counters do not establish GPU execution."
            ),
            "git_head": _git_head(),
            "upstream_revision": gate_inject.qwen4_gate_inject_stats()[
                "upstream_revision"
            ],
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "model": str(args.model.resolve()),
            "model_identity": adapter.identity,
            "device": mx.device_info(),
            "locks": locks,
            "results": results,
            "started_at": started_at,
            "finished_at": time.time(),
            "limitations": [
                "Bounded single-process model qualification, not a serving load test.",
                "Latency is an interleaved target-forward measurement with per-layer tracing disabled; it is not a publishable performance result.",
                "Physical candidate engagement is established separately with LLDB dispatch breakpoints.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "model_validation_passed": report["model_validation_passed"],
                    "engaged_by_launch_receipt": engaged,
                }
            ),
            flush=True,
        )
        return 0 if report["model_validation_passed"] else 1
    finally:
        gate_inject.set_fused_gate_inject_enabled(False)
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
