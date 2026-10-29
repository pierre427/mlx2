"""Parent-owned probes for Nanbeige four-row reuse and physical RoPE/append.

Both entry functions require a live desktop campaign request and perform no
GPU work on import. No CLI can bypass the caller's ownership contract.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path


def _owned(request):
    from mlx2.desktop_campaign_worker import check_ownership

    return check_ownership(request)


def _binding(adapter):
    from mlx2.adapters.nanbeige43_physical import source_identity

    root = Path(__file__).resolve().parents[1]
    artifact = Path(adapter.identity["path"])
    return {
        "checkpoint": artifact.name + "@" + adapter.identity["fingerprint"],
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_sha256": source_identity(),
        "qualification": False,
    }


def _thermal():
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        subprocess.check_output(
            ["/usr/bin/swift", str(root / "scripts/thermal_probe.swift")], text=True
        )
    )


def run_four_row_probe(adapter, request):
    import mlx.core as mx

    from mlx2.adapters.nanbeige43_physical import FOUR_ROW_LAW, q8_four_row
    from mlx2.runtime.lane.matmul import lane_matmul, prepare

    owner = _owned(request)
    if mx.default_device() != mx.gpu:
        raise RuntimeError("four-row probe requires owned GPU")
    layer = adapter.model.layers[0]
    attention, mlp = layer.self_attn, layer.mlp
    modules = [
        ("qkv_or_q", attention.qkv if hasattr(attention, "qkv") else attention.q_proj),
        ("gateup_or_up", mlp.gate_up if hasattr(mlp, "gate_up") else mlp.up_proj),
        ("down", mlp.down_proj),
        ("head", adapter.model.lm_head),
    ]
    cells = []
    thermal_before = _thermal()
    if thermal_before.get("thermal_state") not in (0, 1):
        raise RuntimeError("four-row thermal admission refused")
    for name, module in modules:
        _owned(request)
        k, n = module.weight.shape[1] * 4, module.weight.shape[0]
        x = mx.random.normal((1, 4, k), key=mx.random.key(4276)).astype(
            module.scales.dtype
        )
        mx.eval(x)
        prepared = module.__dict__.get("_lane_prepared") or prepare(module)

        # The ordinary reference bypasses any installed lane/class wrapper.
        def native(x=x, module=module):
            return mx.quantized_matmul(
                x,
                module.weight,
                scales=module.scales,
                biases=module.biases,
                transpose=True,
                group_size=64,
                bits=8,
                mode="affine",
            )

        arms = {
            "stock": native,
            "shared_four": lambda x=x, module=module: q8_four_row(x, module),
            "lane_four": lambda x=x, prepared=prepared: lane_matmul(x, prepared),
        }
        for fn in arms.values():
            mx.eval(fn())
        samples = {key: [] for key in arms}
        order = []
        keys = list(arms)
        for rep in range(3):
            for key in keys[rep:] + keys[:rep]:
                start = time.perf_counter()
                out = arms[key]()
                mx.eval(out)
                samples[key].append(time.perf_counter() - start)
                order.append(key)
        ref = native()
        mx.eval(ref)
        diagnostics = {}
        for key, fn in arms.items():
            out = fn()
            mx.eval(out)
            diagnostics[key] = {
                "finite": bool(mx.all(mx.isfinite(out)).item()),
                "max_abs_error_vs_stock": float(
                    mx.max(
                        mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))
                    ).item()
                ),
                "argmax_equal_vs_stock": bool(
                    mx.all(mx.argmax(out, axis=-1) == mx.argmax(ref, axis=-1)).item()
                ),
            }
        cells.append(
            {
                "projection": name,
                "rows": 4,
                "k": k,
                "n": n,
                "operand_bytes": sum(
                    int(module[key].nbytes) for key in ("weight", "scales", "biases")
                ),
                "samples_seconds": samples,
                "medians_seconds": {
                    key: statistics.median(v) for key, v in samples.items()
                },
                "order": order,
                "diagnostics": diagnostics,
                "logical_weight_passes_per_kernel": {"shared_four": 1},
            }
        )
    return {
        **_binding(adapter),
        "law": FOUR_ROW_LAW,
        "owner_before": owner,
        "owner_after": _owned(request),
        "thermal_before": thermal_before,
        "thermal_after": _thermal(),
        "cells": cells,
        "scope": "Actual checkpoint weights, four actual activation rows; synthetic inputs and completion-fenced projection diagnostics only.",
        "limitations": [
            "Independent FP32 reduction law; numerical diagnostics do not establish answer quality.",
            "No global crossover4/default change, performance qualification or hardware-counter attribution.",
        ],
    }


def run_fused_rope_probe(adapter, request):
    import mlx.core as mx
    from mlx import nn

    from mlx2.adapters.nanbeige43_physical import (
        ROPE_APPEND_LAW,
        _reference_rotate,
        fused_rope_append_arrays,
    )

    owner = _owned(request)
    if mx.default_device() != mx.gpu:
        raise RuntimeError("physical RoPE probe requires owned GPU")
    rope = adapter.model.layers[0].self_attn.rope
    if (
        type(rope) is not nn.RoPE
        or rope.dims != 128
        or rope.base != 120000000
        or rope.traditional
        or rope.scale != 1
    ):
        raise ValueError("physical RoPE probe requires native Nanbeige RoPE")
    dtype = adapter.model.layers[0].input_layernorm.weight.dtype
    capacity = 1024
    rows = 4
    keys = mx.random.normal((1, 8, capacity, 128), key=mx.random.key(4276)).astype(
        dtype
    )
    values = mx.random.normal((1, 8, capacity, 128), key=mx.random.key(4277)).astype(
        dtype
    )
    # Match served packed-QKV views, including row/head strides.
    packed = mx.random.normal((1, rows, 9216), key=mx.random.key(4278)).astype(dtype)
    q, k, v = mx.split(packed, [7168, 8192], axis=-1)
    q = q.reshape(1, rows, 56, 128).transpose(0, 2, 1, 3)
    k = k.reshape(1, rows, 8, 128).transpose(0, 2, 1, 3)
    v = v.reshape(1, rows, 8, 128).transpose(0, 2, 1, 3)
    mx.eval(q, k, v, keys, values)
    thermal_before = _thermal()
    if thermal_before.get("thermal_state") not in (0, 1):
        raise RuntimeError("physical RoPE thermal admission refused")
    cells = []
    for offset in (257, 513):
        _owned(request)
        positions = mx.array([offset, offset + 1, offset + 1, offset + 2], mx.int32)

        def native(offset=offset, positions=positions):
            qr = _reference_rotate(q, positions, rope.base)
            kr = _reference_rotate(k, positions, rope.base)
            start = mx.array([offset], mx.int32)
            return (
                qr,
                mx.slice_update(keys, kr, start, axes=(2,)),
                mx.slice_update(values, v, start, axes=(2,)),
                offset + rows,
            )

        def candidate(offset=offset, positions=positions):
            return fused_rope_append_arrays(
                q, k, v, keys, values, offset, positions, base=rope.base
            )

        arms = {"native": native, "physical_fused": candidate}
        for fn in arms.values():
            mx.eval(fn()[:3])
        samples = {key: [] for key in arms}
        order = []
        for rep in range(3):
            for key in (
                ("native", "physical_fused")
                if rep % 2 == 0
                else ("physical_fused", "native")
            ):
                start = time.perf_counter()
                out = arms[key]()
                mx.eval(out[:3])
                samples[key].append(time.perf_counter() - start)
                order.append(key)
        left, right = native(), candidate()
        mx.eval(left[:3], right[:3])
        end = offset + rows

        def exact(a, b):
            return bool(mx.all(a == b).item())

        cells.append(
            {
                "offset": offset,
                "rows": rows,
                "positions": positions.tolist(),
                "capacity": capacity,
                "input_layout": "served-packed-QKV-views",
                "q_equal": exact(left[0], right[0]),
                "keys_equal": exact(left[1], right[1]),
                "values_equal": exact(left[2], right[2]),
                "q_max_abs_error": float(
                    mx.max(
                        mx.abs(left[0].astype(mx.float32) - right[0].astype(mx.float32))
                    ).item()
                ),
                "keys_max_abs_error": float(
                    mx.max(
                        mx.abs(left[1].astype(mx.float32) - right[1].astype(mx.float32))
                    ).item()
                ),
                "prefix_unchanged": exact(
                    right[1][..., :offset, :], keys[..., :offset, :]
                )
                and exact(right[2][..., :offset, :], values[..., :offset, :]),
                "suffix_unchanged": exact(right[1][..., end:, :], keys[..., end:, :])
                and exact(right[2][..., end:, :], values[..., end:, :]),
                "offset_equal": left[3] == right[3] == end,
                "samples_seconds": samples,
                "medians_seconds": {
                    key: statistics.median(value) for key, value in samples.items()
                },
                "order": order,
                "functional_custom_kernel_invocations_per_lane": 1,
                "successor_cache_bytes": int(keys.nbytes + values.nbytes),
            }
        )
    return {
        **_binding(adapter),
        "law": ROPE_APPEND_LAW,
        "owner_before": owner,
        "owner_after": _owned(request),
        "thermal_before": thermal_before,
        "thermal_after": _thermal(),
        "cells": cells,
        "scope": "Actual D128/Q56/KV8 geometry, branched per-node positions, synthetic values; COW-safe functional successor buffers.",
        "limitations": [
            "One physical custom kernel body combines rotate/append per lane; contiguous input preparation can add launches.",
            "Full-capacity successor copies are priced in timings; no donation or lower memory traffic assumed.",
            "Trig/reduction spelling defines a separate numerical contract; no whole-model quality or speed claim.",
        ],
    }
