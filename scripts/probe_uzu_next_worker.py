"""Bounded component experiments on one already-owned artifact; never qualifies it."""

import gc
import hashlib
import json
import time
from functools import partial

from bench_uzu_candidates import host_load, paired_summary
from probe_uzu_kernels import atomic, swapouts
from run_single_model_smoke_ladder import thermal_sample


def direct(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from probe_q4_gate_rows import policy_for
    from smoke_local_model import _ordinary_generate

    from mlx2.adapters.registry import inspect_model

    record = {
        "status": "running",
        "cells": [],
        "performance": [],
        "scope": "checkpoint weights/activations where named; synthetic attention caches; component timing, not model throughput",
        "selected": False,
        "qualified": False,
        "defaults_changed": False,
    }
    adapter = None

    def save():
        atomic(args.out / "direct.json", record)

    def same(a, b):
        mx.eval(a, b)
        return (
            a.shape == b.shape
            and a.dtype == b.dtype
            and bool(mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item())
        )

    def stats(a, b):
        mx.eval(a, b)
        diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
        return {
            "exact": same(a, b),
            "different_elements": int((a != b).sum().item()),
            "max_abs": float(diff.max().item()),
            "rms_diff": float(mx.sqrt(mx.mean(diff * diff)).item()),
            "reference_abs_max": float(mx.abs(a.astype(mx.float32)).max().item()),
            "finite": bool(mx.all(mx.isfinite(b)).item()),
        }

    swap0 = swapouts()

    def bench(label, baseline, candidate, *, iterations=64, **metadata):
        # Warm the same arm immediately before each timed chain, avoiding idle GPU wake bias.
        samples = []
        for rep in range(args.reps):
            before = thermal_sample()
            if before["thermal_state"] != 0 or swapouts() != swap0:
                raise RuntimeError("performance thermal/swap admission refused")
            order = (
                ("baseline", "candidate") if rep % 2 == 0 else ("candidate", "baseline")
            )
            seconds = {}
            for arm in order:
                fn = baseline if arm == "baseline" else candidate
                deadline = time.perf_counter() + 0.1
                warm_calls = 0
                while warm_calls < 16 or time.perf_counter() < deadline:
                    mx.eval(fn())
                    warm_calls += 1
                mx.synchronize()
                start = time.perf_counter()
                for _ in range(iterations):
                    mx.eval(fn())
                mx.synchronize()
                seconds[arm] = (time.perf_counter() - start) / iterations
            after = thermal_sample()
            if after["thermal_state"] != 0 or swapouts() != swap0:
                raise RuntimeError("performance thermal/swap sample refused")
            samples.append(
                {
                    "rep": rep,
                    "order": order,
                    "seconds": seconds,
                    "thermal_before": before,
                    "thermal_after": after,
                    "host_load": host_load(),
                }
            )
        cell = {
            "kind": label,
            **metadata,
            "iterations_per_arm": iterations,
            "warmup_per_arm": "at least 16 evaluated calls and 100ms immediately before timing",
            "samples": samples,
            "summary": paired_summary(
                [s["seconds"]["baseline"] for s in samples],
                [s["seconds"]["candidate"] for s in samples],
            ),
        }
        record["performance"].append(cell)
        save()
        print(label, metadata, cell["summary"], flush=True)

    try:
        mx.set_memory_limit(80 << 30)
        mx.set_cache_limit(512 << 20)
        resolution = inspect_model(args.model)
        identity = resolution.artifact.get("identity", resolution.artifact)
        record["checkpoint"] = args.model.name + "@" + identity["fingerprint"]
        adapter = resolution.adapter_type(
            str(args.model), execution_policy=policy_for(args.model, False)
        )
        model = adapter.model
        record["device"] = mx.device_info().get("device_name")
        record["mlx_version"] = mx.__version__
        record["ordinary_reference"] = _ordinary_generate(
            adapter, "Say hello in one sentence.", 16
        )
        save()
        config = json.loads((args.model / "config.json").read_text())["text_config"]
        mx.random.seed(102733)
        from uzu_next_kernels import (
            matrix_tree,
            prefix_attention,
            prefix_splits,
            uniform_prefix_attention,
        )

        from mlx2.runtime.models import qsdpa_verify_metal as attn

        # Same affine Q8 planes, partition boundaries and merge order in both arms.
        if args.only != "gdn":
            for keys in (32768, 65536):
                hk = config["num_key_value_heads"]
                hq = config["num_attention_heads"]
                d = config["head_dim"]
                kv = [
                    mx.random.normal((1, hk, keys, d)).astype(mx.bfloat16)
                    for _ in range(2)
                ]
                qk, qv = [mx.quantize(x, group_size=64, bits=8) for x in kv]
                mx.eval(qk, qv)
                del kv
                for rows in (3, 8):
                    q = mx.random.normal((1, hq, rows, d)).astype(mx.bfloat16)
                    options = {
                        "scale": d**-0.5,
                        "group_size": 64,
                        "key_bits": 8,
                        "value_bits": 8,
                    }
                    base = partial(
                        attn.gqa_quantized_verify_attention, q, qk, qv, **options
                    )
                    cand = partial(prefix_attention, q, qk, qv, **options)
                    result = stats(base(), cand())
                    groups, _ = attn.fused_groups(rows, hq // hk)
                    blocks = attn.verify_blocks(keys, 1, hk, groups)
                    record["cells"].append(
                        {
                            "kind": "attention_prefix",
                            "keys": keys,
                            "rows": rows,
                            "blocks": blocks,
                            "mask_free_splits": prefix_splits(keys, rows, blocks),
                            **result,
                        }
                    )
                    save()
                    if not result["exact"]:
                        raise AssertionError("attention split changed arithmetic")
                    bench(
                        "attention_prefix",
                        base,
                        cand,
                        keys=keys,
                        rows=rows,
                        geometry=[hq, hk, d],
                        blocks=blocks,
                    )
                    uniform = partial(uniform_prefix_attention, q, qk, qv, **options)
                    result = stats(base(), uniform())
                    record["cells"].append(
                        {
                            "kind": "attention_uniform_prefix",
                            "keys": keys,
                            "rows": rows,
                            "blocks": blocks,
                            "mask_free_splits": prefix_splits(keys, rows, blocks),
                            **result,
                        }
                    )
                    save()
                    if not result["exact"]:
                        raise AssertionError(
                            "uniform prefix specialization changed arithmetic"
                        )
                    bench(
                        "attention_uniform_prefix",
                        base,
                        uniform,
                        keys=keys,
                        rows=rows,
                        geometry=[hq, hk, d],
                        blocks=blocks,
                    )
                del qk, qv, q
                gc.collect()
                mx.clear_cache()
        if config["model_type"] == "qwen3_5_text":
            from mlx2.runtime.models import qwen38_tree_gdn as tree
            from mlx2.runtime.models.cache import make_prompt_cache
            from mlx2.runtime.models.gated_delta import compute_g
            from mlx2.runtime.models.precise_ops import gate_sigmoid

            # Capture an actual recurrent layer's normalized q/k/v and live state.
            path, gdn = next(
                (p, m)
                for p, m in model.named_modules()
                if hasattr(m, "_gated_delta_update")
            )
            cache = make_prompt_cache(model)
            ids = adapter.prompt_tokens(
                {
                    "prompt": "An engineering experiment compares numerical results and state. "
                    * 100
                }
            )
            mx.eval(
                model(mx.array(ids[:128])[None], cache=cache),
                [
                    x
                    for _, x in tree_flatten([c.state for c in cache])
                    if isinstance(x, mx.array)
                ],
            )
            capture = {}
            original = gdn._gated_delta_update

            def observed(q, k, v, a, b, state, mask, use_kernel, **kwargs):
                capture.update(
                    q=q,
                    k=k,
                    v=v,
                    g=compute_g(gdn.A_log, a, gdn.dt_bias),
                    beta=gate_sigmoid(b.astype(mx.float32)),
                    state=state,
                )
                return original(q, k, v, a, b, state, mask, use_kernel, **kwargs)

            object.__setattr__(gdn, "_gated_delta_update", observed)
            try:
                mx.eval(model(mx.array(ids[128:160])[None], cache=cache))
            finally:
                object.__delattr__(gdn, "_gated_delta_update")
            if not capture:
                raise AssertionError("ordinary GDN capture did not engage")
            capture = {k: mx.contiguous(v) for k, v in capture.items()}
            mx.eval(capture)
            record["gdn_capture"] = {
                "module": path,
                "prefix": 128,
                "rows": 32,
                "shapes": {k: list(v.shape) for k, v in capture.items()},
                "parents_scope": "chain uses captured sequential activations; branching reparents these activations as a component stress fixture",
            }
            for n in (3, 8, 16, 32):
                for shape in ("chain", "binary"):
                    parents = (
                        list(range(-1, n - 1))
                        if shape == "chain"
                        else [-1] + [(i - 1) // 2 for i in range(1, n)]
                    )
                    q, k, v, g, beta = [
                        mx.contiguous(capture[name][:, :n])
                        for name in ("q", "k", "v", "g", "beta")
                    ]
                    state = capture["state"]
                    plan = tree.TreePlan([parents])
                    kept = []
                    p = n - 1
                    while p >= 0:
                        kept.append(p)
                        p = parents[p]
                    commit = tree.CommitPlan([list(reversed(kept))], [0], 3)
                    baseline = partial(tree.tree, q, k, v, g, beta, [state], plan)
                    parent_tensor = mx.array(
                        parents + [-1] * ((16 if n <= 16 else 32) - n), mx.int32
                    )
                    mx.eval(parent_tensor)
                    candidate = partial(
                        matrix_tree,
                        q,
                        k,
                        v,
                        g,
                        beta,
                        state,
                        parents,
                        parent_tensor=parent_tensor,
                    )
                    import numpy as np

                    state_before = hashlib.sha256(
                        np.asarray(state).tobytes()
                    ).hexdigest()
                    result = stats(baseline(), candidate())
                    result["incoming_state_exact"] = (
                        state_before
                        == hashlib.sha256(np.asarray(state).tobytes()).hexdigest()
                    )
                    # Tolerance is an algebra screen only; it cannot admit this into exact serving.
                    result["algebra_screen_pass"] = result["finite"] and result[
                        "max_abs"
                    ] <= 0.02 * max(result["reference_abs_max"], 0.001)
                    record["cells"].append(
                        {"kind": "matrix_gdn", "rows": n, "shape": shape, **result}
                    )
                    save()
                    if (
                        not result["algebra_screen_pass"]
                        or not result["incoming_state_exact"]
                    ):
                        raise AssertionError(
                            "matrix GDN algebra/nonmutation gate failed"
                        )
                    # Includes the same accepted-path serial replay and gated norm in both arms.
                    z = mx.random.normal(v.shape).astype(mx.bfloat16)

                    def roundtrip(
                        fn,
                        z=z,
                        q=q,
                        k=k,
                        v=v,
                        g=g,
                        beta=beta,
                        state=state,
                        commit=commit,
                    ):
                        return (
                            gdn.norm(fn(), z),
                            tree.replay(q, k, v, g, beta, [state], commit),
                        )

                    bench(
                        "matrix_gdn_round",
                        partial(roundtrip, baseline),
                        partial(roundtrip, candidate),
                        rows=n,
                        shape=shape,
                        exact=result["exact"],
                        accepted=len(kept),
                        eligibility="exact"
                        if result["exact"]
                        else "diagnostic approximate only; no state publication",
                    )
        else:
            from mlx2.runtime import row_exact_verify as rev
            from mlx2.runtime.models import qwen3_next as qn
            from mlx2.runtime.models import qwen4_routed_decode as rd

            base_rps = rd.served_down_rows()
            candidate_rps = 4 if base_rps == 2 else 2
            record["moe_baseline_rps"] = base_rps
            record["moe_candidate_rps"] = candidate_rps

            def run(block, x, rows, rps):
                rd.set_served_down_rows(rps)
                with rev.window(rev.Window(rows)):
                    return qn._run_moe_window(block, x, rows, "row_exact")

            for layer in (0, 23, 47):
                block = model.language_model.model.layers[layer].mlp
                block.set_moe_window_consumers({"row_exact"})
                for rows in (1, 3, 8):
                    for scale in (0.5, 2.0):
                        x = (
                            mx.random.normal((1, rows, config["hidden_size"])) * scale
                        ).astype(mx.bfloat16)
                        result = stats(
                            run(block, x, rows, base_rps),
                            run(block, x, rows, candidate_rps),
                        )
                        record["cells"].append(
                            {
                                "kind": "moe_geometry",
                                "layer": layer,
                                "rows": rows,
                                "scale": scale,
                                **result,
                            }
                        )
                        save()
                        if not result["exact"]:
                            raise AssertionError("MoE down geometry changed bits")
                        # Timing scale1 uses a separate draw from the two correctness scales.
                    x = mx.random.normal((1, rows, config["hidden_size"])).astype(
                        mx.bfloat16
                    )
                    bench(
                        "moe_complete_block",
                        partial(run, block, x, rows, base_rps),
                        partial(run, block, x, rows, candidate_rps),
                        layer=layer,
                        rows=rows,
                        baseline_rps=base_rps,
                        candidate_rps=candidate_rps,
                    )
            rd.set_served_down_rows(base_rps)
        record["status"] = "passed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if adapter is not None:
            adapter.close()
        record["swapouts_delta"] = swapouts() - swap0
        save()
