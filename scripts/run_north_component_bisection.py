#!/usr/bin/env python3
"""Owned, diagnostic-only B4-versus-four-B1 North component bisection.

This script never changes the North adapter or its ordinary serving route.  It
loads one pinned local North q4 artifact, captures deterministic component
inputs once, and gives independent copies of those inputs to the host-safe
``north_component_bisection`` contract.  The result is component evidence,
not model qualification, route selection, or a performance claim.

Native execution requires both project GPU owner receipts plus explicit
``GPUQ_LEASE``/``GPUQ_SESSION`` identity.  ``--describe`` is host-only and
does not import MLX or inspect an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.north_component_bisection import (
    STAGES,
    NorthComponentBisection,
    Observation,
    tree_digest,
)

SCHEMA = "mlx2.north-native-component-bisection.v1"
PINNED_ARTIFACT_FINGERPRINT = (
    "6c88a4d3a5387abe2d97ac3d5eb70e484fb33615bb0c0f2ff8e7ef22b442f2e4"
)
LOCK_RECEIPTS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Durably replace one receipt; never expose a partially written JSON file."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _receipt_identity(receipt: dict[str, Any]) -> tuple[Any, ...]:
    session = receipt.get("session", receipt.get("session_id"))
    return (receipt.get("lease_id"), session)


def prove_gpu_ownership(
    paths: tuple[Path, Path] = LOCK_RECEIPTS,
    environ: dict[str, str] | os._Environ[str] = os.environ,
) -> dict[str, Any]:
    """Require two matching owner receipts bound to the calling GPUQ lease."""

    lease = environ.get("GPUQ_LEASE")
    session = environ.get("GPUQ_SESSION")
    if not lease or not session:
        raise RuntimeError("GPUQ_LEASE and GPUQ_SESSION are required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing paired GPU owner receipts: {missing}")
    raw_receipts = [path.read_bytes() for path in paths]
    if raw_receipts[0] != raw_receipts[1]:
        raise RuntimeError("paired GPU owner receipts are not byte-identical")
    receipts = [json.loads(raw) for raw in raw_receipts]
    owner_lease, owner_session = _receipt_identity(receipts[0])
    if owner_lease != lease or owner_session != session:
        raise RuntimeError("GPUQ identity does not own both GPU lock receipts")
    return {
        "lease_id": owner_lease,
        "session": owner_session,
        "label": receipts[0].get("label"),
        "cpg_used": receipts[0].get("cpg_used"),
        "receipt_sha256": [hashlib.sha256(raw).hexdigest() for raw in raw_receipts],
        "paths": [str(path) for path in paths],
    }


def validate_quantization(config: dict[str, Any]) -> dict[str, Any]:
    """Accept only the pinned North q4/q8 geometry this diagnostic understands."""

    quant = config.get("quantization", config.get("quantization_config"))
    if not isinstance(quant, dict):
        raise TypeError("North artifact has no quantization map")
    if {key: quant.get(key) for key in ("bits", "group_size", "mode")} != {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }:
        raise ValueError("North global quantization must be affine q4 group-64")
    expected = {f"model.layers.{index}.mlp.gate" for index in range(1, 49)}
    overrides = {key for key, value in quant.items() if isinstance(value, dict)}
    if overrides != expected:
        raise ValueError("North q8 router override set does not match layers 1..48")
    required = {"bits": 8, "group_size": 64, "mode": "affine"}
    if any(
        {key: quant[path].get(key) for key in required} != required for path in expected
    ):
        raise ValueError("North router overrides must be affine q8 group-64")
    return {
        "global": "affine-q4-group64",
        "router": "affine-q8-group64",
        "router_overrides": len(overrides),
        "tied_head": config.get("tie_word_embeddings") is None,
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=ROOT, check=False, capture_output=True, text=True
    )
    return (result.stdout + result.stderr).strip()


def _bits(mx, value) -> np.ndarray:
    mx.eval(value)
    if value.dtype == mx.bfloat16 or value.dtype == mx.float16:
        value = value.view(mx.uint16)
    elif value.dtype == mx.float32:
        value = value.view(mx.uint32)
    return np.asarray(value)


def _evidence(stage: str, arm: str, payload: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        "batched_rows": 4 if arm == "batched" else 0,
        "rowwise_calls": 0 if arm == "batched" else 4,
    }
    if stage == "q4_projection":
        evidence.update(bits=4, group_size=64)
    elif stage == "q8_router":
        evidence.update(bits=8, group_size=64)
    elif stage == "expert_gather_frozen":
        evidence.update(
            routes_frozen=True,
            route_ids_digest=tree_digest(payload["routes"]),
            route_weights_digest=tree_digest(payload["route_weights"]),
        )
    elif stage == "sdpa":
        evidence.update(
            cache_inputs_frozen=True,
            preappend_cache_digest=tree_digest(
                {"keys": payload["cache_keys"], "values": payload["cache_values"]}
            ),
        )
    elif stage == "tied_head":
        evidence.update(tied=True, bits=4, group_size=64)
    return evidence


def _module_geometry(module, *, bits: int, group_size: int, label: str) -> None:
    observed = _observed_module_geometry(module)
    got = (observed["bits"], observed["group_size"])
    if got != (bits, group_size) or observed["mode"] != "affine":
        raise ValueError(f"unsupported {label} geometry: {got!r}")


def _observed_module_geometry(module) -> dict[str, Any]:
    return {
        "type": type(module).__name__,
        "bits": getattr(module, "bits", None),
        "group_size": getattr(module, "group_size", None),
        "mode": getattr(module, "mode", "affine"),
    }


def _captured_inputs(*, hidden: int, context: int, seed: int) -> dict[str, dict]:
    """Create each common payload once; the contract clones it per arm."""

    rng = np.random.default_rng(seed)
    hidden_x = (rng.standard_normal((4, 1, hidden)) * 0.25).astype(np.float32)
    return {
        "q4_projection": {"x": hidden_x.copy()},
        "q8_router": {"x": hidden_x.copy()},
        "expert_gather_frozen": {"x": hidden_x.copy()},
        "sdpa": {
            "queries": (rng.standard_normal((4, 32, 1, 128)) * 0.25).astype(np.float32),
            "cache_keys": (rng.standard_normal((4, 4, context, 128)) * 0.25).astype(
                np.float32
            ),
            "cache_values": (rng.standard_normal((4, 4, context, 128)) * 0.25).astype(
                np.float32
            ),
            "new_keys": (rng.standard_normal((4, 4, 1, 128)) * 0.25).astype(np.float32),
            "new_values": (rng.standard_normal((4, 4, 1, 128)) * 0.25).astype(
                np.float32
            ),
        },
        "tied_head": {"x": hidden_x.copy()},
    }


def _run_loaded_native(args, artifact, owner, adapter, mx) -> dict[str, Any]:
    """Execute the component probes against an already loaded adapter."""

    from mlx2.runtime.models.base import scaled_dot_product_attention
    from mlx2.runtime.models.cache import BatchKVCache

    if adapter.identity["fingerprint"] != artifact["identity"]["fingerprint"]:
        raise RuntimeError("artifact changed between host inspection and model load")
    model = adapter.model
    layer0 = model.model.layers[0]
    layer1 = model.model.layers[1]
    q4 = layer0.self_attn.q_proj
    router = layer1.mlp.gate
    switch = layer1.mlp.switch_mlp
    head = model.model.embed_tokens
    _module_geometry(q4, bits=4, group_size=64, label="q4 projection")
    _module_geometry(router, bits=8, group_size=64, label="q8 router")
    _module_geometry(head, bits=4, group_size=64, label="tied head")
    for name in ("gate_proj", "up_proj", "down_proj"):
        _module_geometry(
            getattr(switch, name), bits=4, group_size=64, label=f"expert {name}"
        )
    if not model.args.tie_word_embeddings or hasattr(model, "lm_head"):
        raise ValueError("North diagnostic requires the tied embedding head")

    payloads = _captured_inputs(
        hidden=model.args.hidden_size, context=args.sdpa_context, seed=args.seed
    )
    # Freeze the routes once from the q8 router.  Neither arm is allowed to
    # recalculate them, so the expert stage isolates gather width semantics.
    route_x = mx.array(payloads["expert_gather_frozen"]["x"]).astype(mx.bfloat16)
    scores = mx.sigmoid(router(route_x).astype(mx.float32))
    routes = mx.argpartition(-scores, kth=model.args.num_experts_per_tok - 1, axis=-1)[
        ..., : model.args.num_experts_per_tok
    ]
    weights = mx.take_along_axis(scores, routes, axis=-1).astype(mx.bfloat16)
    mx.eval(routes, weights)
    payloads["expert_gather_frozen"]["routes"] = np.asarray(routes)
    payloads["expert_gather_frozen"]["route_weights_bits"] = np.asarray(
        weights.view(mx.uint16)
    )
    # Digest the actual frozen values, not a float32 reconstruction.
    payloads["expert_gather_frozen"]["route_weights"] = payloads[
        "expert_gather_frozen"
    ]["route_weights_bits"]

    oracle = NorthComponentBisection()

    def projection(module, stage, arm):
        def invoke(payload):
            x = mx.array(payload["x"]).astype(mx.bfloat16)
            if arm == "batched":
                output = module(x)
            else:
                output = mx.concatenate(
                    [module(x[i : i + 1]) for i in range(4)], axis=0
                )
            return Observation(
                _bits(mx, output), evidence=_evidence(stage, arm, payload)
            )

        return invoke

    oracle.observe(
        "q4_projection",
        payloads["q4_projection"],
        projection(q4, "q4_projection", "batched"),
        projection(q4, "q4_projection", "rowwise"),
    )
    oracle.observe(
        "q8_router",
        payloads["q8_router"],
        projection(router, "q8_router", "batched"),
        projection(router, "q8_router", "rowwise"),
    )

    def expert(arm):
        def invoke(payload):
            x = mx.array(payload["x"]).astype(mx.bfloat16)
            indices = mx.array(payload["routes"])
            route_weights = mx.array(payload["route_weights_bits"]).view(mx.bfloat16)
            if arm == "batched":
                gathered = switch(x, indices)
                output = mx.sum(gathered * route_weights[..., None], axis=-2)
            else:
                rows = []
                for index in range(4):
                    gathered = switch(x[index : index + 1], indices[index : index + 1])
                    rows.append(
                        mx.sum(
                            gathered * route_weights[index : index + 1, ..., None],
                            axis=-2,
                        )
                    )
                output = mx.concatenate(rows, axis=0)
            return Observation(
                _bits(mx, output),
                evidence=_evidence("expert_gather_frozen", arm, payload),
            )

        return invoke

    expert_before = switch.sort_status()["counts"]
    oracle.observe(
        "expert_gather_frozen",
        payloads["expert_gather_frozen"],
        expert("batched"),
        expert("rowwise"),
    )
    expert_after = switch.sort_status()["counts"]
    expert_sort_delta = {
        key: expert_after.get(key, 0) - expert_before.get(key, 0)
        for key in set(expert_before) | set(expert_after)
    }
    if (
        expert_sort_delta.get("auto_sorted") != 1
        or expert_sort_delta.get("auto_unsorted") != 4
    ):
        raise RuntimeError(
            f"expert gather did not execute one B4 and four B1 routes: {expert_sort_delta}"
        )

    def sdpa(arm):
        def invoke(payload):
            q = mx.array(payload["queries"]).astype(mx.bfloat16)
            old_k = mx.array(payload["cache_keys"]).astype(mx.bfloat16)
            old_v = mx.array(payload["cache_values"]).astype(mx.bfloat16)
            new_k = mx.array(payload["new_keys"]).astype(mx.bfloat16)
            new_v = mx.array(payload["new_values"]).astype(mx.bfloat16)
            if arm == "batched":
                cache = BatchKVCache([0, 0, 0, 0], attention_backend="sdpa")
                cache.update_and_fetch(old_k, old_v)
                keys, values = cache.update_and_fetch(new_k, new_v)
                output = scaled_dot_product_attention(
                    q, keys, values, cache=cache, scale=128**-0.5, mask=None
                )
                state = {
                    "keys": _bits(mx, keys),
                    "values": _bits(mx, values),
                    "offset": np.asarray(cache.offset),
                    "left_padding": np.asarray(cache.left_padding),
                }
            else:
                outputs, keys_rows, values_rows, offsets, padding = [], [], [], [], []
                for index in range(4):
                    cache = BatchKVCache([0], attention_backend="sdpa")
                    cache.update_and_fetch(
                        old_k[index : index + 1], old_v[index : index + 1]
                    )
                    keys, values = cache.update_and_fetch(
                        new_k[index : index + 1], new_v[index : index + 1]
                    )
                    outputs.append(
                        scaled_dot_product_attention(
                            q[index : index + 1],
                            keys,
                            values,
                            cache=cache,
                            scale=128**-0.5,
                            mask=None,
                        )
                    )
                    keys_rows.append(keys)
                    values_rows.append(values)
                    offsets.append(cache.offset)
                    padding.append(cache.left_padding)
                output = mx.concatenate(outputs, axis=0)
                state = {
                    "keys": _bits(mx, mx.concatenate(keys_rows, axis=0)),
                    "values": _bits(mx, mx.concatenate(values_rows, axis=0)),
                    "offset": np.asarray(mx.concatenate(offsets)),
                    "left_padding": np.asarray(mx.concatenate(padding)),
                }
            return Observation(
                _bits(mx, output),
                state=state,
                evidence=_evidence("sdpa", arm, payload),
            )

        return invoke

    oracle.observe("sdpa", payloads["sdpa"], sdpa("batched"), sdpa("rowwise"))
    oracle.observe(
        "tied_head",
        payloads["tied_head"],
        projection(head.as_linear, "tied_head", "batched"),
        projection(head.as_linear, "tied_head", "rowwise"),
    )
    component = oracle.receipt()
    return {
        "schema": SCHEMA,
        "status": "complete" if component["complete"] else "incomplete",
        "semantics": "diagnostic only; not qualification, selection, or performance evidence",
        "ordinary_route_changed": False,
        "default_changed": False,
        "qualification": False,
        "selected": False,
        "artifact": artifact["identity"],
        "geometry": validate_quantization(artifact["config"]),
        "ownership": owner,
        "controls": {
            "rows": 4,
            "rowwise_calls": 4,
            "seed": args.seed,
            "sdpa_context": args.sdpa_context,
            "component_paths": {
                "q4_projection": "model.layers.0.self_attn.q_proj",
                "q8_router": "model.layers.1.mlp.gate",
                "expert_gather_frozen": "model.layers.1.mlp.switch_mlp",
                "sdpa": "BatchKVCache + runtime.models.base.scaled_dot_product_attention",
                "tied_head": "model.embed_tokens.as_linear",
            },
            "inputs": "one deterministic payload captured per stage and independently cloned by contract",
            "expert_sort_counter_delta": expert_sort_delta,
            "observed_module_geometry": {
                "q4_projection": _observed_module_geometry(q4),
                "q8_router": _observed_module_geometry(router),
                "expert_gate": _observed_module_geometry(switch.gate_proj),
                "expert_up": _observed_module_geometry(switch.up_proj),
                "expert_down": _observed_module_geometry(switch.down_proj),
                "tied_head": _observed_module_geometry(head),
            },
        },
        "component_bisection": component,
        "source": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        },
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_native(args, artifact: dict[str, Any], owner: dict[str, Any]) -> dict[str, Any]:
    """Load the real artifact and execute the five bounded component probes."""

    import mlx.core as mx

    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("Metal GPU is unavailable")
    adapter = NorthMiniCodeAdapter(str(args.model))
    try:
        return _run_loaded_native(args, artifact, owner, adapter, mx)
    finally:
        adapter.close()


def describe() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "native": False,
        "required_stages": list(STAGES),
        "rows": {"batched": 4, "independent_rowwise_calls": 4},
        "ownership": "paired GPU owner receipts + GPUQ_LEASE + GPUQ_SESSION",
        "artifact": "North Mini Code 1.0 affine q4 group-64 with q8 group-64 routers",
        "pinned_artifact_fingerprint": PINNED_ARTIFACT_FINGERPRINT,
        "production_route_changed": False,
        "qualification": False,
        "selected": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run-native", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--sdpa-context", type=int, default=127)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.describe:
        print(json.dumps(describe(), indent=2, sort_keys=True))
        return 0
    if not args.i_own_the_gpu:
        raise SystemExit("refusing native execution without --i-own-the-gpu")
    if args.model is None or args.out is None:
        raise SystemExit("--run-native requires --model and --out")
    if args.sdpa_context < 1 or args.sdpa_context > 4095:
        raise SystemExit("--sdpa-context must be in 1..4095")
    owner = prove_gpu_ownership()
    from mlx2.adapters.north_mini_code import inspect_artifact

    artifact = inspect_artifact(args.model)
    if artifact["identity"]["fingerprint"] != PINNED_ARTIFACT_FINGERPRINT:
        raise SystemExit(
            "artifact fingerprint does not match this bounded North diagnostic"
        )
    geometry = validate_quantization(artifact["config"])
    running = {
        "schema": SCHEMA,
        "status": "loading",
        "semantics": "diagnostic only; not qualification, selection, or performance evidence",
        "artifact": artifact["identity"],
        "geometry": geometry,
        "ownership": owner,
    }
    atomic_json(args.out, running)
    try:
        result = run_native(args, artifact, owner)
    except BaseException as error:
        running.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(args.out, running)
        raise
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["component_bisection"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
