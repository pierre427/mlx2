"""Actual-artifact projection probe; caller owns GPU lease and paired locks.

No CLI or GPU operation on import. No packing another weight bank. Stock and
lane arithmetic differ; this is a kernel diagnostic, not a quality verdict.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path


def run_projection_profile(adapter):
    import mlx.core as mx
    from mlx import nn

    from mlx2.adapters.nanbeige43 import matches_config
    from mlx2.runtime.lane.matmul import lane_matmul, prepare

    if mx.default_device() != mx.gpu or not matches_config(adapter.config):
        raise RuntimeError("real Nanbeige projection probe requires the owned GPU")
    model = adapter.model
    root = Path(__file__).resolve().parents[1]

    def thermal():
        value = json.loads(
            subprocess.check_output(
                ["/usr/bin/swift", str(root / "scripts/thermal_probe.swift")], text=True
            )
        )
        if type(value.get("thermal_state")) is not int:
            raise RuntimeError("malformed thermal record")
        return value

    def raw(module):
        if (
            not isinstance(module, nn.QuantizedLinear)
            or module.bits != 8
            or module.group_size != 64
            or module.mode != "affine"
        ):
            raise ValueError(
                "profile requires resident native affine Q8 group64 projections"
            )
        return {key: module[key] for key in ("weight", "scales", "biases")}

    def stock(x, arrays):
        return mx.quantized_matmul(
            x,
            arrays["weight"],
            scales=arrays["scales"],
            biases=arrays["biases"],
            transpose=True,
            group_size=64,
            bits=8,
            mode="affine",
        )

    attention = model.layers[0].self_attn
    if hasattr(attention, "qkv"):
        module = attention.qkv
        qkv_modules = (module,)
        qkv_combined = raw(module)
        qkv_group = module.__dict__.get("_lane_group")
    else:
        qkv_modules = (attention.q_proj, attention.k_proj, attention.v_proj)
        groups = [item.__dict__.get("_lane_group") for item in qkv_modules]
        qkv_group = (
            groups[0]
            if groups[0] is not None and all(item is groups[0] for item in groups)
            else None
        )
        qkv_combined = qkv_group.stack if qkv_group is not None else None
    projections = [
        ("qkv", qkv_modules, qkv_combined, qkv_group),
        ("attention_output", (attention.o_proj,), raw(attention.o_proj), None),
        (
            "mlp_down",
            (model.layers[0].mlp.down_proj,),
            raw(model.layers[0].mlp.down_proj),
            None,
        ),
        ("lm_head", (model.lm_head,), raw(model.lm_head), None),
    ]
    rows = []
    for index, (name, modules, combined, group) in enumerate(projections):
        source = [raw(module) for module in modules]
        k = int(source[0]["weight"].shape[1]) * 4
        n = sum(int(item["weight"].shape[0]) for item in source)
        # Existing prepared metadata is reused; if absent only the compact
        # scale/bias pairing is materialized, never another weight table.
        prepared = (
            group.lw if group is not None else modules[0].__dict__.get("_lane_prepared")
        )
        created_metadata = False
        if prepared is None and len(modules) == 1:
            prepared = prepare(modules[0])
            created_metadata = True
        before = thermal()
        if before["thermal_state"] not in (0, 1):
            raise RuntimeError(f"{name}: thermal admission refused")
        scope = []
        for count in (1, 4, 8, 16, 32):
            x = mx.random.normal((1, count, k), key=mx.random.key(4276 + count)).astype(
                source[0]["scales"].dtype
            )
            mx.eval(x)
            arms = {
                "stock_independent": lambda x=x, source=source: tuple(
                    stock(x, item) for item in source
                ),
                "current_module_dispatch": lambda x=x, modules=modules: tuple(
                    module(x) for module in modules
                ),
            }
            if combined is not None:
                arms["stock_existing_backing"] = lambda x=x, combined=combined: stock(
                    x, combined
                )
            if prepared is not None:
                arms["lane_existing_backing"] = lambda x=x, prepared=prepared: (
                    lane_matmul(x, prepared)
                )
            for function in arms.values():
                mx.eval(function())
            samples = {key: [] for key in arms}
            keys = list(arms)
            order = []
            for rep in range(3):
                for key in keys[rep % len(keys) :] + keys[: rep % len(keys)]:
                    started = time.perf_counter()
                    output = arms[key]()
                    mx.eval(output)
                    samples[key].append(time.perf_counter() - started)
                    order.append(key)
            reference = mx.concatenate(arms["stock_independent"](), axis=-1)
            mx.eval(reference)
            numerical = {}
            for key, function in arms.items():
                output = function()
                if isinstance(output, tuple):
                    output = mx.concatenate(output, axis=-1)
                mx.eval(output)
                numerical[key] = {
                    "max_abs_error_vs_stock": float(
                        mx.max(
                            mx.abs(
                                output.astype(mx.float32) - reference.astype(mx.float32)
                            )
                        ).item()
                    ),
                    "argmax_equal_vs_stock": bool(
                        mx.all(
                            mx.argmax(output, axis=-1) == mx.argmax(reference, axis=-1)
                        ).item()
                    ),
                }
            scope.append(
                {
                    "projection": name,
                    "rows": count,
                    "k": k,
                    "n": n,
                    "samples_seconds": samples,
                    "medians_seconds": {
                        key: statistics.median(value) for key, value in samples.items()
                    },
                    "arm_order": order,
                    "arithmetic_diagnostics": numerical,
                }
            )
        after = thermal()
        rows.append(
            {
                "projection": name,
                "k": k,
                "n": n,
                "logical_weight_and_quant_metadata_bytes": sum(
                    int(array.nbytes) for item in source for array in item.values()
                ),
                "existing_grouped_backing": qkv_combined is not None
                if name == "qkv"
                else False,
                "prepared_metadata_created": created_metadata,
                "prepared_scale_bias_bytes": int(prepared.scale_bias.nbytes)
                if prepared is not None and prepared.scale_bias is not None
                else 0,
                "prepared_split_k": prepared.split_k if prepared is not None else None,
                "module_dispatch_policy": [
                    {
                        "class": type(module).__name__,
                        "lane_min_rows": getattr(module, "_lane_min_rows", None),
                        "lane_max_rows": getattr(module, "_lane_max_rows", None),
                        "stock_stacked": bool(getattr(group, "stock_stacked", False)),
                    }
                    for module in modules
                ],
                "thermal_before": before,
                "thermal_after": after,
                "cells": scope,
            }
        )
    identity = adapter.identity
    artifact = Path(identity["path"])
    paths = [
        Path(__file__),
        root / "src/mlx2/runtime/lane/matmul.py",
        root / "src/mlx2/runtime/lane/installer.py",
    ]
    return {
        "schema": "mlx2.nanbeige43-actual-projection-profile.v1",
        "checkpoint": artifact.name + "@" + identity["fingerprint"],
        "model_path": str(artifact),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "execution_numerics_contract": adapter.execution_numerics_contract(),
        "qualification": False,
        "projections": rows,
        "limitations": [
            "Resident actual-checkpoint shape microprobe; inputs are synthetic, not captured model activations.",
            "Explicit mx.eval per projection includes synchronization overhead; not production-graph attribution.",
            "Current module dispatch profiles installed module wrappers, not enclosing adapter-specific project hooks.",
            "No new grouped weight backing is built; unavailable grouped arms are omitted.",
            "Arithmetic diagnostics are not qualitative answer-quality assessments.",
            "No hardware-counter or end-to-end model speed claim.",
        ],
    }


def run_rope_append_probe(adapter):
    """D128/GQA7 functional-state probe on reserved equal-geometry buffers.

    This tests compiled successor arrays, not one physical Metal launch.
    The caller must already own the GPU and collect lease/thermal admission.
    """
    import copy
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx import nn

    from mlx2.adapters.nanbeige43 import matches_config
    from mlx2.adapters.nanbeige43_kernels import kernel_contract, rope_append
    from mlx2.runtime.models.cache import KVCache

    if mx.default_device() != mx.gpu or not matches_config(adapter.config):
        raise RuntimeError("RoPE append probe requires the owned GPU and Nanbeige")
    root = Path(__file__).resolve().parents[1]
    rope = adapter.model.layers[0].self_attn.rope
    if type(rope) is not nn.RoPE or rope.dims != 128:
        raise ValueError("RoPE append probe requires native D128 RoPE")
    dtype = adapter.model.layers[0].input_layernorm.weight.dtype
    if dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError("RoPE append probe requires floating activations")
    capacity = 1024
    keys = mx.random.normal((1, 8, capacity, 128), key=mx.random.key(4276)).astype(
        dtype
    )
    values = mx.random.normal((1, 8, capacity, 128), key=mx.random.key(4277)).astype(
        dtype
    )
    mx.eval(keys, values)

    def fresh(offset):
        cache = KVCache()
        cache.keys, cache.values = copy.deepcopy(keys), copy.deepcopy(values)
        cache.offset = offset
        mx.eval(cache.keys, cache.values)
        return cache

    def native(q, k, v, cache):
        qr = rope(q, offset=cache.offset)
        kr = rope(k, offset=cache.offset)
        cached_k, cached_v = cache.update_and_fetch(kr, v)
        return qr, cached_k, cached_v

    def run(kind, q, k, v, start):
        cache = fresh(start)
        started = time.perf_counter()
        output = (
            native(q, k, v, cache)
            if kind == "native"
            else rope_append(q, k, v, rope, cache)
        )
        if output is None:
            raise RuntimeError("admitted RoPE append geometry unexpectedly refused")
        mx.eval(output, cache.state)
        elapsed = time.perf_counter() - started
        return output, cache, elapsed

    def equal(a, b):
        return bool(mx.all(a == b).item())

    before = json.loads(
        subprocess.check_output(
            ["/usr/bin/swift", str(root / "scripts/thermal_probe.swift")], text=True
        )
    )
    if before.get("thermal_state") not in (0, 1):
        raise RuntimeError("RoPE append thermal admission refused")
    cells = []
    for start in (257, 513):
        for length in (1, 3):
            q = mx.random.normal(
                (1, 56, length, 128), key=mx.random.key(start + length)
            ).astype(dtype)
            k = mx.random.normal(
                (1, 8, length, 128), key=mx.random.key(start + length + 1)
            ).astype(dtype)
            v = mx.random.normal(
                (1, 8, length, 128), key=mx.random.key(start + length + 2)
            ).astype(dtype)
            mx.eval(q, k, v)
            # Compile/warm both forms outside retained samples.
            run("native", q, k, v, start)
            run("compiled", q, k, v, start)
            times = {"native": [], "compiled": []}
            order = []
            for rep in range(3):
                for kind in (
                    ("native", "compiled") if rep % 2 == 0 else ("compiled", "native")
                ):
                    _output, _cache, elapsed = run(kind, q, k, v, start)
                    times[kind].append(elapsed)
                    order.append(kind)
            expected, reference, _ = run("native", q, k, v, start)
            actual, candidate, _ = run("compiled", q, k, v, start)
            end = start + length
            cells.append(
                {
                    "start_offset": start,
                    "append_rows": length,
                    "end_offset": end,
                    "buffer_shape": list(keys.shape),
                    "q_shape": list(q.shape),
                    "dtype": str(dtype),
                    "output_shapes": [list(item.shape) for item in actual],
                    "q_bitwise_equal": equal(expected[0], actual[0]),
                    "visible_keys_bitwise_equal": equal(expected[1], actual[1]),
                    "visible_values_bitwise_equal": equal(expected[2], actual[2]),
                    "full_keys_bitwise_equal": equal(reference.keys, candidate.keys),
                    "full_values_bitwise_equal": equal(
                        reference.values, candidate.values
                    ),
                    "committed_prefix_unchanged": equal(
                        candidate.keys[..., :start, :], keys[..., :start, :]
                    )
                    and equal(candidate.values[..., :start, :], values[..., :start, :]),
                    "unused_suffix_unchanged": equal(
                        candidate.keys[..., end:, :], keys[..., end:, :]
                    )
                    and equal(candidate.values[..., end:, :], values[..., end:, :]),
                    "offsets_equal": reference.offset == candidate.offset == end,
                    "q_max_abs_error": float(
                        mx.max(
                            mx.abs(
                                expected[0].astype(mx.float32)
                                - actual[0].astype(mx.float32)
                            )
                        ).item()
                    ),
                    "samples_seconds": times,
                    "medians_seconds": {
                        key: statistics.median(value) for key, value in times.items()
                    },
                    "arm_order": order,
                }
            )
    q = mx.zeros((1, 56, 1, 128), dtype)
    k = mx.zeros((1, 8, 1, 128), dtype)
    refusals = []
    for name, cache, selected_rope in (
        ("missing_buffers", KVCache(), rope),
        ("capacity_exhausted", fresh(capacity), rope),
        ("foreign_cache", SimpleNamespace(offset=0), rope),
        ("wrapped_rope", fresh(257), SimpleNamespace(dims=128)),
    ):
        old_offset = getattr(cache, "offset", None)
        old_keys = getattr(cache, "keys", None)
        result = rope_append(q, k, k, selected_rope, cache)
        if (
            result is not None
            or getattr(cache, "offset", None) != old_offset
            or getattr(cache, "keys", None) is not old_keys
        ):
            raise AssertionError("RoPE append refusal mutated cache")
        refusals.append({"reason": name, "declined_before_mutation": True})
    after = json.loads(
        subprocess.check_output(
            ["/usr/bin/swift", str(root / "scripts/thermal_probe.swift")], text=True
        )
    )
    artifact = Path(adapter.identity["path"])
    source = Path(__file__)
    return {
        "schema": "mlx2.nanbeige43-rope-append-probe.v1",
        "checkpoint": artifact.name + "@" + adapter.identity["fingerprint"],
        "model_path": str(artifact),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "kernel_contract": kernel_contract({"rope_kv_append": True}),
        "rope": {
            "dims": rope.dims,
            "base": rope.base,
            "scale": rope.scale,
            "traditional": rope.traditional,
        },
        "qualification": False,
        "physical_single_launch_fusion": "unproven",
        "thermal_before": before,
        "thermal_after": after,
        "cells": cells,
        "refused_geometry_count": len(refusals),
        "refusals": refusals,
        "limitations": [
            "Synthetic Q/K/V and reserved ordinary KV buffers with actual model RoPE geometry.",
            "Native and compiled forms use the same capacity, dtype, initial contents and offset; cache preparation excluded from timing.",
            "Timing includes explicit completion fences and is not full-model serving performance.",
            "Successful functional compilation does not prove launch fusion, donation or lower memory traffic.",
        ],
    }


def run_split_k_probe(adapter):
    """Shape-specific split-K diagnostics; no runtime policy is mutated.

    The fixed 30s host budget stops new cells; an already-dispatched kernel is
    completed before return. All variants borrow resident codes and metadata.
    """
    from dataclasses import replace

    import mlx.core as mx
    from mlx import nn

    from mlx2.adapters.nanbeige43 import matches_config
    from mlx2.runtime.lane.matmul import lane_matmul, prepare

    if mx.default_device() != mx.gpu or not matches_config(adapter.config):
        raise RuntimeError("split-K probe requires the owned GPU and Nanbeige")
    root = Path(__file__).resolve().parents[1]
    layer = adapter.model.layers[0]
    attention = layer.self_attn
    if hasattr(attention, "qkv"):
        qkv = attention.qkv
        qkv_prepared = qkv.__dict__.get("_lane_prepared")
        qkv_group = None
    else:
        qkv = attention.q_proj
        qkv_group = qkv.__dict__.get("_lane_group")
        if qkv_group is None or any(
            module.__dict__.get("_lane_group") is not qkv_group
            for module in (attention.k_proj, attention.v_proj)
        ):
            raise ValueError("split-K QKV probe requires an already grouped backing")
        qkv_prepared = qkv_group.lw
    specs = [
        ("qkv", qkv, qkv_group, qkv_prepared),
        (
            "attention_output",
            attention.o_proj,
            None,
            attention.o_proj.__dict__.get("_lane_prepared"),
        ),
        (
            "mlp_down",
            layer.mlp.down_proj,
            None,
            layer.mlp.down_proj.__dict__.get("_lane_prepared"),
        ),
        (
            "lm_head",
            adapter.model.lm_head,
            None,
            adapter.model.lm_head.__dict__.get("_lane_prepared"),
        ),
    ]
    started_all = time.perf_counter()
    budget = 30.0
    before = json.loads(
        subprocess.check_output(
            ["/usr/bin/swift", str(root / "scripts/thermal_probe.swift")], text=True
        )
    )
    if before.get("thermal_state") not in (0, 1):
        raise RuntimeError("split-K thermal admission refused")
    results = []
    declined = []
    exhausted = False
    for name, module, group, prepared in specs:
        if (
            not isinstance(module, nn.QuantizedLinear)
            or module.bits != 8
            or module.group_size != 64
            or module.mode != "affine"
        ):
            raise ValueError("split-K requires native affine Q8 group64")
        if time.perf_counter() - started_all >= budget:
            exhausted = True
            break
        if prepared is None:
            prepared = prepare(module)
        if prepared.backend != "mpp":
            declined.append({"projection": name, "reason": "not_mpp"})
            continue
        arrays = (
            group.stack
            if group is not None
            else {key: module[key] for key in ("weight", "scales", "biases")}
        )
        k, n = prepared.k, prepared.n
        compatible = [split for split in (1, 2, 4, 8) if k // 64 % split == 0]
        for count in (4, 8, 16):
            if time.perf_counter() - started_all >= budget:
                exhausted = True
                break
            x = mx.random.normal((1, count, k), key=mx.random.key(4276 + count)).astype(
                arrays["scales"].dtype
            )
            mx.eval(x)

            def stock(x=x, arrays=arrays):
                return mx.quantized_matmul(
                    x,
                    arrays["weight"],
                    scales=arrays["scales"],
                    biases=arrays["biases"],
                    transpose=True,
                    group_size=64,
                    bits=8,
                    mode="affine",
                )

            arms = {"stock": stock}
            for split in compatible:
                variant = replace(prepared, split_k=split)
                # All explicit variants share exact weight/metadata objects.
                assert (
                    variant.weight is prepared.weight
                    and variant.scale_bias is prepared.scale_bias
                )
                arms[f"split_k_{split}"] = lambda variant=variant, x=x: lane_matmul(
                    x, variant
                )
            samples = {key: [] for key in arms}
            keys = list(arms)
            order = []
            for function in arms.values():
                mx.eval(function())
            for rep in range(2):
                for key in keys[rep:] + keys[:rep]:
                    if time.perf_counter() - started_all >= budget:
                        exhausted = True
                        break
                    start = time.perf_counter()
                    output = arms[key]()
                    mx.eval(output)
                    samples[key].append(time.perf_counter() - start)
                    order.append(key)
                if exhausted:
                    break
            reference = stock()
            mx.eval(reference)
            diagnostics = {}
            if not exhausted:
                for key, function in arms.items():
                    output = function()
                    mx.eval(output)
                    diagnostics[key] = {
                        "max_abs_error_vs_stock": float(
                            mx.max(
                                mx.abs(
                                    output.astype(mx.float32)
                                    - reference.astype(mx.float32)
                                )
                            ).item()
                        ),
                        "argmax_equal_vs_stock": bool(
                            mx.all(
                                mx.argmax(output, axis=-1)
                                == mx.argmax(reference, axis=-1)
                            ).item()
                        ),
                    }
            results.append(
                {
                    "projection": name,
                    "rows": count,
                    "k": k,
                    "n": n,
                    "default_prepared_split_k": prepared.split_k,
                    "compatible_split_k": compatible,
                    "existing_grouped_backing": group is not None
                    or hasattr(attention, "qkv")
                    and name == "qkv",
                    "logical_operand_bytes": sum(
                        int(array.nbytes) for array in arrays.values()
                    ),
                    "samples_seconds": samples,
                    "medians_seconds": {
                        key: statistics.median(value)
                        for key, value in samples.items()
                        if value
                    },
                    "arm_order": order,
                    "arithmetic_diagnostics": diagnostics,
                    "complete_cell": not exhausted,
                }
            )
            if exhausted:
                break
        if exhausted:
            break
    after = json.loads(
        subprocess.check_output(
            ["/usr/bin/swift", str(root / "scripts/thermal_probe.swift")], text=True
        )
    )
    artifact = Path(adapter.identity["path"])
    source = Path(__file__)
    return {
        "schema": "mlx2.nanbeige43-split-k-probe.v1",
        "checkpoint": artifact.name + "@" + adapter.identity["fingerprint"],
        "model_path": str(artifact),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "lane_source_sha256": hashlib.sha256(
            (root / "src/mlx2/runtime/lane/matmul.py").read_bytes()
        ).hexdigest(),
        "qualification": False,
        "thermal_before": before,
        "thermal_after": after,
        "elapsed_seconds": time.perf_counter() - started_all,
        "host_budget_seconds": budget,
        "budget_exhausted": exhausted,
        "complete_cell_count": sum(row["complete_cell"] for row in results),
        "declined": declined,
        "cells": results,
        "limitations": [
            "Actual checkpoint weights with synthetic inputs and explicit completion fences; no whole-model speed claim.",
            "Split-K variants define different reduction laws; diagnostics are not qualitative answer-quality scores.",
            "No universal crossover or split-K default follows from isolated shape timings.",
            "Installed kernel policies are immutable during this probe. Preparation cache keys bind array identity; mutating split-K in place after preparation is unsupported.",
            "Thirty-second host budget stops new dispatches; in-flight device work and thermal-sampling overhead must complete.",
        ],
    }
