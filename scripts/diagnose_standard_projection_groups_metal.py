#!/usr/bin/env python3
"""Bounded BF16 native projection-group diagnosis; no production policy changes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import signal
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = (1, 2, 4, 8, 16)
PROMPTS = (
    "Explain in French how a mutex prevents a race when two threads increment a counter.",
    "Write a Python binary search function and explain why its interval shrinks.",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--timeout-seconds", type=int, default=600)
    p.add_argument("--include-vmap", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def preflight(args):
    for name, low, high in (
        ("rows", 16, 32),
        ("queries", 3, 16),
        ("repetitions", 1, 3),
        ("timeout_seconds", 1, 900),
    ):
        if not low <= getattr(args, name) <= high:
            raise ValueError(f"{name} must be in [{low},{high}]")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError("execution requires --i-own-the-gpu under operator ownership")
    return {
        "schema": "mlx2.standard-projection-groups-diagnostic.v1",
        "will_execute": not args.dry_run,
        "model": str(args.model.expanduser().resolve()),
        "rows": args.rows,
        "groups": list(GROUPS),
        "layouts": ["sequence", "batch"] + (["vmap"] if args.include_vmap else []),
        "full_law_geometry": [15, args.queries],
        "timeout_seconds": args.timeout_seconds,
        "timing_scope": "fixed materialized resident BF16 input; synchronized projector evaluation; uncontrolled diagnostic only",
        "activation_source": "original B1/S1 target forwards over two short chat-template prompts",
        "qualified": False,
        "performance_claim": False,
        "production_math_changed": False,
        "passed": False,
    }


def source_hashes():
    files = [
        Path(__file__),
        ROOT / "scripts/validate_adaptive_metal.py",
        ROOT / "scripts/validate_xpress_metal_matrix.py",
        *sorted((ROOT / "src").rglob("*.py")),
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def grouped_projection(operation, value, group, layout):
    """Only change native input row geometry; preserve values and row order."""
    import mlx.core as mx

    if value.ndim != 3 or min(value.shape) <= 0 or group not in GROUPS:
        raise ValueError("projection requires nonempty [B,S,D] input and known group")
    flat = value.reshape(-1, value.shape[-1])
    outputs = []
    for start in range(0, flat.shape[0], group):
        rows = flat[start : start + group]
        if layout == "sequence":
            output = operation(rows[None])[0]
        elif layout == "batch":
            output = operation(rows[:, None])[:, 0]
        elif layout == "vmap":
            output = mx.vmap(lambda row: operation(row[None, None])[0, 0])(rows)
        else:
            raise ValueError("unknown projection layout")
        outputs.append(output)
    result = mx.concatenate(outputs, axis=0)
    return result.reshape(*value.shape[:2], result.shape[-1])


def delta(actual, expected):
    import numpy as np

    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if actual.shape != expected.shape:
        raise ValueError("comparison shapes differ")
    change = actual - expected
    return {
        "equal": bool(np.array_equal(actual, expected)),
        "max_absolute_error": float(np.max(np.abs(change), initial=0)),
        "rms_error": float(np.sqrt(np.mean(change * change))) if change.size else 0.0,
    }


def native_bits(value):
    """Compare stored bits, including signed zero and NaN payloads."""
    import mlx.core as mx
    import numpy as np

    return (
        str(value.dtype),
        tuple(value.shape),
        np.asarray(mx.contiguous(value).view(mx.uint8)).tobytes(),
    )


def run(args, report):
    import os

    import mlx.core as mx
    import numpy as np
    import psutil
    from mlx import nn
    from validate_adaptive_metal import artifact_identity, ordinary_reference_prefill
    from validate_xpress_metal_matrix import (
        float32_artifact_forecast,
        guard_float32_footprint,
    )

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.models import standard_decoder as standard
    from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows
    from mlx2.runtime.speculative_sampling import softmax

    if not mx.metal.is_available():
        raise RuntimeError("owned M3 Metal device required")
    mx.set_default_device(mx.gpu)
    device = mx.device_info()
    if "m3" not in json.dumps(device).lower():
        raise RuntimeError("owned M3 Metal device required")
    if os.environ.get("MLX2_FP_DECODE_KERNEL", "0") == "1":
        raise ValueError("diagnostic requires original native S1 SDPA")
    resident, largest, dtypes = float32_artifact_forecast([args.model])
    if set(dtypes) != {"BF16"}:
        raise ValueError("original unquantized BF16 target artifact required")
    report.update(
        device=device,
        source_sha256=source_hashes(),
        artifact_bytes=artifact_identity(args.model),
        memory_guard=guard_float32_footprint(
            resident // 2,
            largest,
            int(psutil.virtual_memory().available),
            int(device.get("max_recommended_working_set_size", 0)),
        ),
    )
    report["memory_guard"]["precision"] = "original BF16; no target cast"
    adapter = StandardDecoderAdapter(str(args.model))
    model = adapter.model
    try:
        model.configure_target_verify_row_exact(True)
        model.configure_target_verify_row_exact(False)
    except BaseException:
        adapter.close()
        raise
    old_linear, old_embedding = nn.Linear.__call__, nn.Embedding.as_linear
    old_project = standard._project
    projectors, recorded = {}, {}
    for index, layer in enumerate(model.layers):
        for section, names in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        ):
            for name in names:
                module = getattr(getattr(layer, section), name)
                if type(module) is not nn.Linear:
                    raise ValueError("diagnostic requires native dense projectors")
                key = f"model.layers.{index}.{section}.{name}"
                projectors[key] = module
    if model.args.tie_word_embeddings:
        projectors["model.embed_tokens.as_linear"] = model.model.embed_tokens.as_linear
        embedding_id = id(model.model.embed_tokens)
    else:
        projectors["lm_head"] = model.lm_head
        embedding_id = None
    ids = {
        id(value): name
        for name, value in projectors.items()
        if isinstance(value, nn.Linear)
    }
    output_name = "model.embed_tokens.as_linear" if embedding_id else "lm_head"
    records = {name: [] for name in projectors}
    captured = False

    def linear(module, value):
        if captured and id(module) in ids:
            records[ids[id(module)]].append(value)
        return old_linear(module, value)

    def embedding(module, value):
        if captured and id(module) == embedding_id:
            records[output_name].append(value)
        return old_embedding(module, value)

    prompts, oracle_traces = [], []
    frozen = None
    try:
        if model.args.model_type != "qwen3":
            raise ValueError("dense Qwen3 required")
        for text in PROMPTS:
            prompts.append(
                adapter.prompt_tokens(
                    {
                        "messages": [{"role": "user", "content": text}],
                        "enable_thinking": False,
                    }
                )
            )
        if any(not prompt or len(prompt) > 128 for prompt in prompts):
            raise ValueError("short nonempty chat-template prompt required")
        report["prompt_ids"] = prompts
        report["artifact_identity"] = adapter.identity
        report["native_api_source"] = {
            "Linear": "installed mlx/nn/layers/linear.py: native x @ weight.T (or addmm with bias)",
            "matmul": "installed mlx/core/__init__.pyi: broadcasting over leading batch dimensions",
            "vmap": "installed mlx/core/__init__.pyi: documented vectorized callable, optional control",
        }
        nn.Linear.__call__, nn.Embedding.as_linear = linear, embedding
        samples = [args.rows // 2, args.rows - args.rows // 2]
        for prompt_index, (prompt, count) in enumerate(
            zip(prompts, samples, strict=True)
        ):
            # Ordinary prefill remains untouched; capture starts at a true S1 emitted token.
            cache, logits = ordinary_reference_prefill(model, prompt, prefill_step=128)
            if prompt_index == 0:
                frozen = copy.deepcopy(cache)
            tokens, values, bits = [], [], []
            for _ in range(count):
                token = int(mx.argmax(logits[0, -1]).item())
                tokens.append(token)
                captured = True
                logits = model(mx.array([[token]]), cache=cache)
                captured = False
                mx.eval(logits, [items[-1] for items in records.values()])
                values.append(np.asarray(logits[0, -1].astype(mx.float32)))
                bits.append(native_bits(logits[0, -1]))
            oracle_traces.append((tokens, values, bits))
        nn.Linear.__call__, nn.Embedding.as_linear = old_linear, old_embedding
        report["capture_matches_untraced_original_S1"] = []
        for prompt, (tokens, expected, expected_bits) in zip(
            prompts, oracle_traces, strict=True
        ):
            cache, _ = ordinary_reference_prefill(model, prompt, prefill_step=128)
            actual, actual_bits = [], []
            for token in tokens:
                value = model(mx.array([[token]]), cache=cache)
                mx.eval(value)
                actual.append(np.asarray(value[0, -1].astype(mx.float32)))
                actual_bits.append(native_bits(value[0, -1]))
            report["capture_matches_untraced_original_S1"].append(
                {
                    **delta(actual, expected),
                    "bitwise_equal": actual_bits == expected_bits,
                }
            )
        if not all(
            item["bitwise_equal"]
            for item in report["capture_matches_untraced_original_S1"]
        ):
            raise AssertionError("activation capture changed original S1 logits")
        for name, values in records.items():
            if len(values) != args.rows or any(
                value.shape[:2] != (1, 1) for value in values
            ):
                raise AssertionError(
                    "fixed activation capture did not cover every native S1 projector"
                )
            recorded[name] = mx.concatenate(values, axis=1)
        mx.eval(list(recorded.values()))
        records.clear()
        exact = {
            (layout, group): True for layout in report["layouts"] for group in GROUPS
        }
        report["projector_results"] = []
        for name, operation in projectors.items():
            value = recorded[name]
            expected = grouped_projection(operation, value, 1, "sequence")
            mx.eval(expected)
            expected_host = np.asarray(expected.astype(mx.float32))
            results = []
            for layout in report["layouts"]:
                for group in GROUPS:
                    measurements = []
                    actual_host = None
                    for repetition in range(args.repetitions + 1):
                        mx.synchronize()
                        start = time.perf_counter()
                        actual = grouped_projection(operation, value, group, layout)
                        mx.eval(actual)
                        mx.synchronize()
                        elapsed = time.perf_counter() - start
                        if repetition:
                            measurements.append(elapsed)
                        actual_host = np.asarray(actual.astype(mx.float32))
                    error = delta(actual_host, expected_host)
                    error["bitwise_equal"] = native_bits(actual) == native_bits(
                        expected
                    )
                    b15 = value[:, :15].reshape(15, 1, value.shape[-1])
                    b15_actual = grouped_projection(operation, b15, group, layout)
                    mx.eval(b15_actual)
                    b15_error = delta(
                        np.asarray(b15_actual.astype(mx.float32)),
                        expected_host[:, :15].reshape(15, 1, expected_host.shape[-1]),
                    )
                    b15_error["bitwise_equal"] = native_bits(b15_actual) == native_bits(
                        expected[:, :15].reshape(15, 1, expected.shape[-1])
                    )
                    exact[layout, group] &= (
                        error["bitwise_equal"] and b15_error["bitwise_equal"]
                    )
                    results.append(
                        {
                            "layout": layout,
                            "group": group,
                            "physical_chunk_sizes": [
                                min(group, args.rows - start)
                                for start in range(0, args.rows, group)
                            ],
                            **error,
                            "b15_fixed_input": {"shape": list(b15.shape), **b15_error},
                            "median_seconds_diagnostic": float(np.median(measurements)),
                            "samples_seconds_diagnostic": measurements,
                        }
                    )
            report["projector_results"].append(
                {
                    "name": name,
                    "input_shape": list(value.shape),
                    "input_dtype": str(value.dtype),
                    "fixed_input_native_byte_sha256": hashlib.sha256(
                        native_bits(value)[2]
                    ).hexdigest(),
                    "results": results,
                }
            )
        report["all_projectors_exact"] = [
            {"layout": layout, "group": group, "equal": okay}
            for (layout, group), okay in exact.items()
        ]
        candidates = sorted(
            (
                (layout, group)
                for (layout, group), okay in exact.items()
                if okay and group > 1
            ),
            key=lambda item: (-item[1], report["layouts"].index(item[0])),
        )
        layout, group = candidates[0] if candidates else ("sequence", 1)
        report["selected_diagnostic_geometry"] = {
            "layout": layout,
            "group": group,
            "nontrivial_exact_group_found": bool(candidates),
            "production_selected": False,
        }
        # One frozen cache, original S1 teacher paths, 15 independently owned branches.
        cache = copy.deepcopy(frozen)
        _, first = ordinary_reference_prefill(model, prompts[0], prefill_step=128)
        token = int(mx.argmax(first[0, -1]).item())
        path = []
        for _ in range(args.queries):
            path.append(token)
            next_value = model(mx.array([[token]]), cache=cache)
            mx.eval(next_value)
            token = int(mx.argmax(next_value[0, -1]).item())
        paths = [
            path[:3]
            + [int((value + row) % model.args.vocab_size) for value in path[3:]]
            for row in range(15)
        ]
        reference_logits, reference_caches = [], []
        for branch in paths:
            cache = copy.deepcopy(frozen)
            outputs = []
            for token in branch:
                value = model(mx.array([[token]]), cache=cache)
                mx.eval(value)
                outputs.append(value)
            reference_logits.append(mx.concatenate(outputs, axis=1))
            reference_caches.append(cache)
        expected = mx.concatenate(reference_logits, axis=0)
        mx.eval(expected)
        model.configure_target_verify_row_exact(True)

        def selected_project(operation, value, row_exact=False):
            return (
                grouped_projection(operation, value, group, layout)
                if row_exact
                else old_project(operation, value, False)
            )

        owner = SegmentedKVRows([copy.deepcopy(frozen) for _ in paths])
        tx = owner.begin([args.queries] * len(paths))
        try:
            standard._project = selected_project
            actual, features = model.forward_with_taps(
                mx.array(paths), tx.caches, tuple(range(len(model.layers)))
            )
            mx.eval(actual, features)
            committed = tx.commit(accepted_lengths=[args.queries] * len(paths))
        except BaseException:
            if not tx.closed:
                tx.abort()
            raise
        finally:
            standard._project = old_project
        actual_host, expected_host = (
            np.asarray(value.astype(mx.float32)) for value in (actual, expected)
        )
        laws = []
        for row in range(15):
            for position in range(args.queries):
                a, b = actual_host[row, position], expected_host[row, position]
                item = {
                    "row": row,
                    "position": position,
                    **delta(a, b),
                    "greedy_equal": bool(np.argmax(a) == np.argmax(b)),
                    "temperature_0_8_law_max_absolute_error": float(
                        np.max(np.abs(softmax(a, 0.8) - softmax(b, 0.8)))
                    ),
                }
                laws.append(item)
        cache_equal = all(
            a.offset == b.offset
            and all(
                native_bits(x) == native_bits(y)
                for x, y in zip(a.keys_and_values(), b.keys_and_values(), strict=True)
            )
            for current, reference in zip(committed, reference_caches, strict=True)
            for a, b in zip(current, reference, strict=True)
        )
        report["frozen_cache_full_law_check"] = {
            "inputs": paths,
            "offset": frozen[0].offset,
            "query_geometry": [15, args.queries],
            "math_scope": "selected temporary projection grouping plus S1-prefix attention; original prefill and ordinary reference untouched",
            **delta(actual_host, expected_host),
            "bitwise_equal": native_bits(actual) == native_bits(expected),
            "all_greedy_equal": all(item["greedy_equal"] for item in laws),
            "committed_cache_bitwise_equal": cache_equal,
            "rows": laws,
            "qualified": False,
        }
        report["selected_full_math_exact"] = (
            native_bits(actual) == native_bits(expected) and cache_equal
        )
        report["optimized_candidate_exact"] = (
            bool(candidates) and report["selected_full_math_exact"]
        )
        report["source_unchanged"] = report["source_sha256"] == source_hashes()
        if not report["source_unchanged"]:
            raise AssertionError("source changed during diagnostic")
        report["passed"] = True
        report["note"] = (
            "Successful diagnosis means measurements completed; nonzero errors remain failures of exactness and never qualify an optimized route."
        )
    finally:
        captured = False
        nn.Linear.__call__, nn.Embedding.as_linear = old_linear, old_embedding
        standard._project = old_project
        model.configure_target_verify_row_exact(False)
        report["temporary_hooks_restored"] = (
            nn.Linear.__call__ is old_linear
            and nn.Embedding.as_linear is old_embedding
            and standard._project is old_project
        )
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
        raise TimeoutError("projection diagnostic deadline exceeded")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
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
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
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
