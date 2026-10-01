#!/usr/bin/env python3
"""Bounded reached-prefix Qwen3 BF16 shape diagnostics; never a production route."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import signal
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRENCH = "Explain in French how a mutex prevents a race when two threads increment a counter."


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "draft", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--deadline-seconds", type=int, default=900)
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def preflight(args):
    if not 1 <= args.deadline_seconds <= 900:
        raise ValueError("deadline_seconds must be in [1,900]")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError("execution requires --i-own-the-gpu under both locks")
    return {
        "schema": "mlx2.standard-bf16-shape-diagnostic.v1",
        "will_execute": not args.dry_run,
        "qualified": False,
        "performance_claim": False,
        "production_math_changed": False,
        "model": str(args.model.resolve()),
        "draft": str(args.draft.resolve()),
        "deadline_seconds": args.deadline_seconds,
        "prompt": FRENCH,
        "max_tokens": 17,
        "num_draft": 15,
        "captured_verify_block": 2,
        "capture_layers": "all postlayer/pre-finalnorm",
        "controls": [
            "identical_precache_segmented_vs_plain",
            "identical_precache_sequential_S1",
            "rebuilt_ordinary_committed_cache",
            "B4_B15_identical_branches",
            "same_anchor_perturbed_future_suffix",
            "S1_none_vs_alltrue_mask",
            "scalar_vs_vector_RoPE",
            "same_input_projection_shape",
            "temporary_fp32_attention",
            "temporary_rowwise_linears",
        ],
        "passed": False,
    }


def source_hashes():
    paths = [
        Path(__file__),
        ROOT / "scripts/validate_xpress_metal_matrix.py",
        ROOT / "scripts/smoke_parallel_draft_metal.py",
        *sorted((ROOT / "src").rglob("*.py")),
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def delta(a, b):
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("comparison shapes differ")
    difference = a - b
    return {
        "equal": bool(np.array_equal(a, b)),
        "max_abs": float(np.max(np.abs(difference), initial=0)),
        "rms": float(np.sqrt(np.mean(difference * difference)))
        if difference.size
        else 0.0,
    }


def run(args, report):
    import os

    import mlx.core as mx
    import numpy as np
    import psutil
    from mlx import nn
    from validate_xpress_metal_matrix import (
        float32_artifact_forecast,
        guard_float32_footprint,
    )

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.models import standard_decoder as standard
    from mlx2.runtime.models.base import (
        create_attention_mask,
        scaled_dot_product_attention,
    )
    from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows

    if not mx.metal.is_available():
        raise RuntimeError("owned M3 Metal device required")
    mx.set_default_device(mx.gpu)
    device = mx.device_info()
    if "m3" not in json.dumps(device).lower():
        raise RuntimeError("owned M3 Metal device required")
    resident, largest, dtypes = float32_artifact_forecast([args.model, args.draft])
    if set(dtypes) != {"BF16"}:
        raise ValueError("this diagnostic requires original unquantized BF16 artifacts")
    # BF16 resident payload is half the FP32 forecast; fixed transient reserve
    # includes B15 logits/features and simultaneously retained prefix snapshots.
    report["memory_guard"] = guard_float32_footprint(
        resident // 2,
        largest,
        int(psutil.virtual_memory().available),
        int(device.get("max_recommended_working_set_size", 0)),
    )
    report["memory_guard"]["parameter_precision"] = (
        "original BF16; helper field float32_parameter_bytes is resident forecast only"
    )
    report["device"] = device
    report["source_sha256"] = source_hashes()
    report["attention_environment"] = {
        name: os.environ.get(name)
        for name in (
            "MLX2_FP_DECODE_KERNEL",
            "MLX2_FUSED_SDPA_MIN_L",
            "MLX2_QSDPA_DECODE_KERNEL",
        )
    }
    adapter = StandardDecoderAdapter(
        str(args.model),
        execution_policy={"draft_model": str(args.draft), "num_draft": 15},
    )
    model = adapter.model
    layers = tuple(range(len(model.layers)))
    width = model.args.hidden_size
    batch = adapter.create_external_batch(
        completion_batch_size=1, prefill_step_size=128, ready_drain="all"
    )
    try:
        prompt = adapter.prompt_tokens(
            {
                "messages": [{"role": "user", "content": FRENCH}],
                "enable_thinking": False,
            }
        )
        report["artifact_identity"] = adapter.identity
        report["draft_settings"] = adapter.draft_model.receipt_settings
        report["prompt_ids"] = prompt
        uid = batch.insert([prompt], max_tokens=[17])[0]
        lane = batch.lanes[uid]
        while lane.anchor is None:
            batch._prefill(lane)
        original = model.forward_with_taps
        frames = []

        def watched(inputs, cache, capture_layers, **kwargs):
            snapshot = None
            if not kwargs.get("body_only", False):
                rows = [
                    layer.rows[0] if hasattr(layer, "rows") else layer
                    for layer in cache
                ]
                snapshot = {
                    "inputs": np.asarray(inputs).tolist(),
                    "cache": copy.deepcopy(rows),
                    "history": list(lane.history),
                    "offsets": [int(row.offset) for row in rows],
                }
            result = original(inputs, cache, capture_layers, **kwargs)
            if snapshot is not None:
                mx.eval(result[0], result[1])
                snapshot["logits"] = np.asarray(result[0].astype(mx.float32))
                frames.append(snapshot)
            return result

        model.forward_with_taps = watched
        try:
            for _ in range(2):
                batch._round([lane])
        finally:
            model.forward_with_taps = original
        if len(frames) < 2:
            raise AssertionError(
                "second physical target verification block not observed"
            )
        reached = frames[1]
        inputs = reached["inputs"][0]
        frozen = reached["cache"]
        report["actual_capture"] = {
            "inputs": reached["inputs"],
            "history": reached["history"],
            "offsets": reached["offsets"],
            "emitted_tokens": [response.token for response in lane.ready],
            "first_block_inputs": frames[0]["inputs"],
            "first_block_offsets": frames[0]["offsets"],
        }

        def host(value):
            return np.asarray(value.astype(mx.float32))

        def cache_hashes(cache):
            return [
                hashlib.sha256(
                    json.dumps([layer.offset, str(layer.keys.dtype)]).encode()
                    + host(layer.keys[..., : layer.offset, :]).tobytes()
                    + host(layer.values[..., : layer.offset, :]).tobytes()
                ).hexdigest()
                for layer in cache
            ]

        frozen_hashes = cache_hashes(frozen)
        report["frozen_precache_sha256"] = frozen_hashes
        report["cache_hash_convention"] = (
            "committed K/V values expanded exactly from BF16 to float32; allocated unused suffix excluded"
        )

        def top(logits):
            values = np.asarray(logits, dtype=np.float32)
            indices = np.argsort(-values, kind="stable")[:5]
            return [{"token": int(i), "logit": float(values[i])} for i in indices]

        def traced(ids, cache, kind="plain", multiplicity=1):
            clones = [copy.deepcopy(cache) for _ in range(multiplicity)]
            transaction = None
            try:
                if kind == "segmented":
                    transaction = SegmentedKVRows(clones).begin(
                        lengths=[len(ids)] * multiplicity
                    )
                    active = transaction.caches
                else:
                    if multiplicity != 1:
                        raise ValueError("plain control is B1 only")
                    active = clones[0]
                logits, features = original(
                    mx.array([ids] * multiplicity), active, layers
                )
                mx.eval(logits, features)
                return host(logits), host(features).reshape(
                    multiplicity, len(ids), len(layers), width
                )
            finally:
                if transaction is not None:
                    transaction.abort()

        def sequential(ids, cache):
            active = copy.deepcopy(cache)
            logits = []
            features = []
            for token in ids:
                out, taps = original(mx.array([[token]]), active, layers)
                mx.eval(out, taps)
                logits.append(host(out))
                features.append(host(taps).reshape(1, 1, len(layers), width))
            return np.concatenate(logits, axis=1), np.concatenate(features, axis=1)

        def compare(name, observed, reference):
            out, taps = observed
            ref_out, ref_taps = reference
            n = min(out.shape[1], ref_out.shape[1])
            out = out[0, :n]
            taps = taps[0, :n]
            ref_out = ref_out[0, :n]
            ref_taps = ref_taps[0, :n]
            return {
                "name": name,
                "per_row": [
                    {
                        "row": row,
                        "logits": delta(out[row], ref_out[row]),
                        "argmax_equal": int(np.argmax(out[row]))
                        == int(np.argmax(ref_out[row])),
                        "observed_top5": top(out[row]),
                        "reference_top5": top(ref_out[row]),
                        "layers": [
                            {
                                "layer": layer,
                                **delta(taps[row, layer], ref_taps[row, layer]),
                            }
                            for layer in layers
                        ],
                    }
                    for row in range(n)
                ],
            }

        segmented = traced(inputs, frozen, "segmented")
        plain = traced(inputs, frozen)
        seq = sequential(inputs, frozen)
        report["trace_reproduces_actual_logits"] = delta(
            segmented[0], reached["logits"]
        )
        comparisons = [
            compare("segmented_full_vs_plain_full", segmented, plain),
            compare("segmented_full_vs_same_cache_sequential_S1", segmented, seq),
        ]
        rebuilt = model.make_cache()
        if prompt[:-1]:
            mx.eval(model(mx.array([prompt[:-1]]), cache=rebuilt))
        for token in reached["history"][len(prompt) - 1 :]:
            mx.eval(model(mx.array([[token]]), cache=rebuilt))
        report["precache_drift"] = [
            {
                "layer": index,
                "keys": delta(
                    host(actual.keys[..., : actual.offset, :]),
                    host(reference.keys[..., : reference.offset, :]),
                ),
                "values": delta(
                    host(actual.values[..., : actual.offset, :]),
                    host(reference.values[..., : reference.offset, :]),
                ),
            }
            for index, (actual, reference) in enumerate(
                zip(frozen, rebuilt, strict=True)
            )
        ]
        rebuilt_seq = sequential(inputs, rebuilt)
        comparisons.append(
            compare("same_cache_S1_vs_rebuilt_ordinary_S1", seq, rebuilt_seq)
        )
        comparisons.append(
            compare("actual_segmented_vs_rebuilt_ordinary_S1", segmented, rebuilt_seq)
        )
        whole = model.make_cache()
        mx.eval(model(mx.array([prompt]), cache=whole))
        for token in reached["history"][len(prompt) :]:
            mx.eval(model(mx.array([[token]]), cache=whole))
        comparisons.append(
            compare(
                "whole_prompt_prefill_vs_ordinary_prefix_anchor_prefill",
                sequential(inputs, whole),
                rebuilt_seq,
            )
        )
        for multiplicity in (4, 15):
            observed = traced(inputs, frozen, "segmented", multiplicity)
            comparison = compare(
                f"B{multiplicity}_identical_vs_B1_segmented", observed, segmented
            )
            comparison["within_batch_first3_rows"] = [
                {
                    "branch": index,
                    "logits": delta(observed[0][index, :3], observed[0][0, :3]),
                    "layers": [
                        {
                            "layer": layer,
                            **delta(
                                observed[1][index, :3, layer], observed[1][0, :3, layer]
                            ),
                        }
                        for layer in layers
                    ],
                }
                for index in range(multiplicity)
            ]
            comparisons.append(comparison)
            del observed
        perturbed = list(inputs)
        for index in range(3, len(perturbed)):
            perturbed[index] = (perturbed[index] + 7919) % model.args.vocab_size
        changed = traced(perturbed, frozen, "segmented")
        comparisons.append(
            compare(
                "perturbed_future_suffix_first3_rows",
                (changed[0][:, :3], changed[1][:, :3]),
                (segmented[0][:, :3], segmented[1][:, :3]),
            )
        )
        prefix3 = traced(inputs[:3], frozen, "segmented")
        comparisons.append(
            compare("segmented_S3_vs_S16_first3_rows", prefix3, segmented)
        )

        # Exact same immutable scalar values: isolate offset and mask dispatch.
        attention = model.layers[0].self_attn
        embedded = model.model.embed_tokens(mx.array([inputs]))
        normed = model.layers[0].input_layernorm(embedded)
        q = attention.q_norm(
            attention.q_proj(normed).reshape(
                1, len(inputs), attention.n_heads, attention.head_dim
            )
        ).transpose(0, 2, 1, 3)
        scalar = attention.rope(q, offset=frozen[0].offset)
        vector = attention.rope(q, offset=mx.array([frozen[0].offset]))
        report["scalar_vector_rope"] = delta(host(scalar), host(vector))
        none = copy.deepcopy(frozen)
        alltrue = copy.deepcopy(frozen)
        for cache in alltrue:
            cache._pld_ordinary_mask_padding = mx.array([0])
            cache._pld_ordinary_mask_calls = 0
        comparisons.append(
            compare(
                "S1_none_vs_alltrue_mask",
                traced(inputs[:1], none),
                traced(inputs[:1], alltrue),
            )
        )

        # Materialize fixed operator inputs before comparing row geometries. This
        # tests dispatch rounding separately from propagated attention/cache drift.
        mx.eval(normed)
        projection_controls = []

        def project(name, module, x):
            full = module(x)
            flat = x.reshape(-1, x.shape[-1])
            parts = [module(flat[i : i + 1]) for i in range(flat.shape[0])]
            rowwise = mx.concatenate(parts, axis=0).reshape(full.shape)
            mx.eval(full, rowwise)
            projection_controls.append(
                {
                    "name": name,
                    "full_shape": list(x.shape),
                    "same_input_rows": delta(host(full), host(rowwise)),
                }
            )
            return full

        for name in ("q_proj", "k_proj", "v_proj"):
            project("layer0." + name, getattr(attention, name), normed)
        cache = copy.deepcopy(frozen[0])
        mask = create_attention_mask(embedded, cache)
        q = attention.q_norm(
            attention.q_proj(normed).reshape(
                1, len(inputs), attention.n_heads, attention.head_dim
            )
        ).transpose(0, 2, 1, 3)
        k = attention.k_norm(
            attention.k_proj(normed).reshape(
                1, len(inputs), attention.n_kv_heads, attention.head_dim
            )
        ).transpose(0, 2, 1, 3)
        v = (
            attention.v_proj(normed)
            .reshape(1, len(inputs), attention.n_kv_heads, attention.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q, k = (
            attention.rope(q, offset=cache.offset),
            attention.rope(k, offset=cache.offset),
        )
        k, v = cache.update_and_fetch(k, v)
        attended = (
            scaled_dot_product_attention(q, k, v, cache, attention.scale, mask)
            .transpose(0, 2, 1, 3)
            .reshape(1, len(inputs), -1)
        )
        mx.eval(attended)
        o = project("layer0.o_proj", attention.o_proj, attended)
        post = model.layers[0].post_attention_layernorm(embedded + o)
        mx.eval(post)
        mlp = model.layers[0].mlp
        gate = project("layer0.gate_proj", mlp.gate_proj, post)
        up = project("layer0.up_proj", mlp.up_proj, post)
        from mlx2.runtime.models.activations import swiglu

        activation = swiglu(gate, up)
        mx.eval(activation)
        project("layer0.down_proj", mlp.down_proj, activation)
        final_hidden = mx.array(plain[1][:, :, -1], dtype=mx.bfloat16)
        final_hidden = model.model.norm(final_hidden)
        mx.eval(final_hidden)
        project(
            "lm_head",
            model.model.embed_tokens.as_linear
            if model.args.tie_word_embeddings
            else model.lm_head,
            final_hidden,
        )
        report["fixed_input_projection_controls"] = projection_controls

        # Temporary diagnostic math patches are restored even after exceptions.
        old_linear = nn.Linear.__call__
        old_embedding_linear = nn.Embedding.as_linear

        def rowwise_linear(module, x):
            flat = x.reshape(-1, x.shape[-1])
            return mx.concatenate(
                [old_linear(module, flat[i : i + 1]) for i in range(flat.shape[0])],
                axis=0,
            ).reshape(*x.shape[:-1], module.weight.shape[0])

        def rowwise_embedding_linear(module, x):
            flat = x.reshape(-1, x.shape[-1])
            return mx.concatenate(
                [
                    old_embedding_linear(module, flat[i : i + 1])
                    for i in range(flat.shape[0])
                ],
                axis=0,
            ).reshape(*x.shape[:-1], module.weight.shape[0])

        try:
            nn.Linear.__call__ = rowwise_linear
            nn.Embedding.as_linear = rowwise_embedding_linear
            rowwise = traced(inputs, frozen, "segmented")
        finally:
            nn.Linear.__call__ = old_linear
            nn.Embedding.as_linear = old_embedding_linear
        comparisons.append(
            compare("temporary_rowwise_linears_vs_same_cache_S1", rowwise, seq)
        )
        old_attention = standard.scaled_dot_product_attention

        def fp32_attention(q, k, v, cache, scale, mask, sinks=None):
            dtype = q.dtype
            if hasattr(cache, "row_views"):
                outputs = []
                for index, valid, keys, values, row_mask in cache.row_views(mask):
                    value = mx.fast.scaled_dot_product_attention(
                        q[index : index + 1, :, :valid].astype(mx.float32),
                        keys.astype(mx.float32),
                        values.astype(mx.float32),
                        scale=scale,
                        mask=row_mask,
                        sinks=sinks,
                    ).astype(dtype)
                    if valid < q.shape[2]:
                        value = mx.pad(
                            value, [(0, 0), (0, 0), (0, q.shape[2] - valid), (0, 0)]
                        )
                    outputs.append(value)
                return mx.concatenate(outputs, axis=0)
            return mx.fast.scaled_dot_product_attention(
                q.astype(mx.float32),
                k.astype(mx.float32),
                v.astype(mx.float32),
                scale=scale,
                mask=mask,
                sinks=sinks,
            ).astype(dtype)

        try:
            standard.scaled_dot_product_attention = fp32_attention
            promoted = traced(inputs, frozen, "segmented")
        finally:
            standard.scaled_dot_product_attention = old_attention
        comparisons.append(
            compare("temporary_fp32_attention_vs_same_cache_S1", promoted, seq)
        )
        report["comparisons"] = comparisons
        report["temporary_patches_restored"] = (
            nn.Linear.__call__ is old_linear
            and nn.Embedding.as_linear is old_embedding_linear
            and standard.scaled_dot_product_attention is old_attention
        )
        report["source_unchanged"] = source_hashes() == report["source_sha256"]
        report["frozen_precache_unchanged"] = frozen_hashes == cache_hashes(frozen)
        report["passed"] = (
            report["source_unchanged"]
            and report["temporary_patches_restored"]
            and report["frozen_precache_unchanged"]
        )
        report["passed_interpretation"] = (
            "diagnostic completed with restored original math; numerical deltas are evidence, not parity qualification"
        )
    finally:
        batch.close()
        adapter.close()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        report = preflight(args)
    except ValueError as error:
        p.error(str(error))
    if args.dry_run:
        print(json.dumps(report))
        return 0

    def deadline(*_):
        raise TimeoutError("BF16 diagnostic deadline exceeded")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    try:
        run(args, report)
    except Exception as error:  # noqa: BLE001 - preserve diagnostic failure evidence
        report.update(
            passed=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        signal.alarm(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "out": str(args.out),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
