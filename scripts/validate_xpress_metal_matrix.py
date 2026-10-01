#!/usr/bin/env python3
# ruff: noqa: EXE001
"""Real-artifact XPress correctness matrix under operator-owned Metal.

No performance claim or automatic qualification. One target and head remain
resident. Draft settings change only between isolated cells; each setting has
a distinct paired-cache binding. Dry-run is tensor-import-free. Failure cells
persist diagnostics and ordinary validation continues when safe. A timeout
ends the run and writes the accumulated receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mlx2.xpress-metal-correctness-matrix.v1"
FRENCH_PROMPT = "Explain in French how a mutex prevents a race when two threads increment a counter."


def matrix():
    return [
        {
            "name": "french_cold_b1",
            "context": 128,
            "width": 1,
            "budgets": [17],
            "passes": 6,
            "french": True,
        },
        {
            "name": "french_warm_b2",
            "context": 128,
            "width": 2,
            "budgets": [17, 17],
            "passes": 6,
            "french": True,
            "warm": True,
        },
        {
            "name": "code_b1_128",
            "context": 128,
            "width": 1,
            "budgets": [17],
            "passes": 6,
            "theme": 0,
        },
        {
            "name": "math_prose_b2_128",
            "context": 128,
            "width": 2,
            "budgets": [17, 48],
            "passes": 6,
            "theme": 1,
        },
        {
            "name": "four_themes_b4_128",
            "context": 128,
            "width": 4,
            "budgets": [17, 17, 48, 17],
            "passes": 6,
            "theme": 0,
        },
        {
            "name": "math_b1_1024",
            "context": 1024,
            "width": 1,
            "budgets": [48],
            "passes": 6,
            "theme": 1,
        },
        {
            "name": "prose_multilingual_b2_1024",
            "context": 1024,
            "width": 2,
            "budgets": [48, 17],
            "passes": 6,
            "theme": 2,
        },
        {
            "name": "multilingual_b1_4096",
            "context": 4096,
            "width": 1,
            "budgets": [17],
            "passes": 6,
            "theme": 3,
        },
        {
            "name": "ragged_b4_1024",
            "context": 1024,
            "width": 4,
            "budgets": [17, 1, 17, 48],
            "passes": 6,
        },
        {
            "name": "pass1_b2_128",
            "context": 128,
            "width": 2,
            "budgets": [17, 17],
            "passes": 1,
        },
        {
            "name": "pass3_b2_1024",
            "context": 1024,
            "width": 2,
            "budgets": [17, 17],
            "passes": 3,
        },
        {
            "name": "window_wrap_apcv2",
            "context": 1024,
            "width": 2,
            "budgets": [17, 17],
            "passes": 6,
            "windows": True,
        },
        {
            "name": "later_failure_retry",
            "context": 128,
            "width": 2,
            "budgets": [48, 48],
            "passes": 6,
            "fault": True,
        },
        {
            "name": "cancellation",
            "context": 128,
            "width": 2,
            "budgets": [17, 17],
            "passes": 6,
            "cancel": True,
        },
    ]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, help="JSON object: draft_attention_windows only"
    )
    parser.add_argument("--timeout-seconds", type=int, default=480)
    parser.add_argument(
        "--compute-precision",
        choices=["artifact", "float32-diagnostic"],
        default="artifact",
    )
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def preflight(args):
    if not 1 <= args.timeout_seconds <= 900:
        raise ValueError("timeout-seconds must be between 1 and 900")
    windows = [32, 128, None]
    if args.config:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict) or set(config) != {"draft_attention_windows"}:
            raise ValueError("config must declare only draft_attention_windows")
        windows = config["draft_attention_windows"]
    if (
        not isinstance(windows, list)
        or not windows
        or any(
            x is not None and (type(x) is not int or x <= 0 or x > 4096)
            for x in windows
        )
    ):
        raise ValueError("windows must contain positive integers or null")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError(
            "Metal execution requires --i-own-the-gpu under operator ownership"
        )
    return {
        "schema": SCHEMA,
        "model": str(args.model.expanduser().resolve()),
        "draft": str(args.draft.expanduser().resolve()),
        "cells": matrix(),
        "draft_attention_windows": windows,
        "window_setting_source": "explicit_config"
        if args.config
        else "repeating_default_pattern",
        "timeout_seconds": args.timeout_seconds,
        "performance_claim": False,
        "qualified": False,
        "will_execute": not args.dry_run,
        "compute_precision_requested": args.compute_precision,
    }


def resolve_attention_windows(report, num_layers):
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("checkpoint draft layer count must be positive")
    windows = report["draft_attention_windows"]
    if report["window_setting_source"] == "explicit_config":
        if len(windows) != num_layers:
            raise ValueError(
                "explicit window config must have one entry per checkpoint draft layer"
            )
        return list(windows)
    return [windows[index % len(windows)] for index in range(num_layers)]


def source_identity():
    files = [Path(__file__), *sorted((ROOT / "src").rglob("*.py"))]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    return {
        "root": str(ROOT),
        "source_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "files_sha256": hashes,
    }


def save(args, report):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)


def host_float32(tensor):
    import mlx.core as mx
    import numpy as np

    return np.asarray(tensor.astype(mx.float32))


def top_logits(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    ids = np.argpartition(-values, min(4, len(values) - 1))[:5]
    ids = ids[np.argsort(-values[ids], kind="stable")]
    return {
        "ids": ids.tolist(),
        "logits": values[ids].tolist(),
        "top2_margin": float(values[ids[0]] - values[ids[1]]),
    }


def prediction_rows(logits):
    """Body-only prefill has taps but deliberately returns no logits."""
    if logits is None:
        return None
    return [[top_logits(values) for values in row] for row in host_float32(logits)]


def guard_float32_footprint(
    resident_bytes, largest_conversion_source_bytes, available_bytes, recommended_bytes
):
    """Conservative resident + one eager cast + 4 GiB transient allowance."""
    values = (
        resident_bytes,
        largest_conversion_source_bytes,
        available_bytes,
        recommended_bytes,
    )
    if (
        any(type(x) is not int or x < 0 for x in values)
        or not resident_bytes
        or not available_bytes
        or not recommended_bytes
    ):
        raise ValueError("float32 diagnostic requires known positive memory budgets")
    required = resident_bytes + largest_conversion_source_bytes + (4 << 30)
    if required > min(available_bytes, recommended_bytes):
        raise ValueError(
            f"float32 diagnostic unsafe footprint: needs {required} bytes; available/recommended {available_bytes}/{recommended_bytes}"
        )
    return {
        "float32_parameter_bytes": resident_bytes,
        "largest_eager_conversion_source_bytes": largest_conversion_source_bytes,
        "transient_allowance_bytes": 4 << 30,
        "required_bytes": required,
        "available_bytes": available_bytes,
        "recommended_working_set_bytes": recommended_bytes,
    }


def float32_artifact_forecast(paths):
    """Header-only prediction before original model payload allocation."""
    from mlx2.adapters.dflash2 import _read_safetensors_header

    resident = largest = 0
    dtypes = {}
    for directory in paths:
        config = json.loads((directory / "config.json").read_text())
        if config.get("quantization") or config.get("quantization_config"):
            raise ValueError("float32 diagnostic requires unquantized artifacts")
        index = directory / "model.safetensors.index.json"
        names = (
            sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if index.is_file()
            else ["model.safetensors"]
        )
        for name in names:
            _, header, _ = _read_safetensors_header(directory / name)
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                dtype = value["dtype"]
                shape = value["shape"]
                count = 1
                if dtype not in ("BF16", "F16", "F32"):
                    raise ValueError(
                        f"float32 diagnostic cannot convert source dtype {dtype}"
                    )
                for dimension in shape:
                    if type(dimension) is not int or dimension < 0:
                        raise ValueError("invalid tensor shape")
                    count *= dimension
                resident += 4 * count
                dtypes[dtype] = dtypes.get(dtype, 0) + count
                if dtype != "F32":
                    largest = max(largest, 2 * count)
    return resident, largest, dtypes


def cast_float32_diagnostic(*networks):
    """Cast/evaluate one parameter at a time, retaining shared module binding."""
    import mlx.core as mx

    def cast(value):
        if mx.issubdtype(value.dtype, mx.floating) and value.dtype != mx.float32:
            converted = value.astype(mx.float32)
            mx.eval(converted)
            return converted
        return value

    # Module.apply builds a complete replacement tree before updating it and
    # would retain every original tensor during conversion. Replace each
    # direct parameter immediately so only one source/cast pair is transient.
    for network in networks:
        for _name, module in network.named_modules():
            for key in list(module):
                value = module[key]
                if isinstance(value, mx.array):
                    module[key] = cast(value)
                    # Drop the source reference and its freed allocator block
                    # before the next conversion, matching the peak guard.
                    del value
                    mx.clear_cache()
    mx.clear_cache()


def cache_diagnostics(caches):
    """Hash committed semantics separately from append-only unused capacity.

    Recovery borrows KV buffers at their captured fill level. Columns at or
    above that level are excluded from attention and are overwritten before
    becoming live. Rotating caches have no append-only declaration: every
    retained physical position and ring coordinate remains part of semantics.
    """
    result = []
    for cache in caches:
        metadata = (
            type(cache).__name__,
            cache.offset,
            getattr(cache, "_idx", None),
            getattr(cache, "rotated", None),
            cache.meta_state,
        )
        semantic = hashlib.sha256(repr(metadata).encode())
        allocated = hashlib.sha256(repr(metadata).encode())
        append_only = {
            name: (axis, getattr(cache, level))
            for name, axis, level in getattr(cache, "_RECOVERY_APPEND_ONLY_FIELDS", ())
        }
        shapes = []
        for index, value in enumerate(cache.state):
            if not hasattr(value, "dtype"):
                semantic.update(repr(value).encode())
                allocated.update(repr(value).encode())
                shapes.append(None)
                continue
            raw = host_float32(value)
            shapes.append(list(value.shape))
            allocated.update(str((value.dtype, value.shape)).encode())
            allocated.update(raw.tobytes())
            field = ("keys", "values")[index] if index < 2 else None
            if field in append_only:
                axis, level = append_only[field]
                slices = [slice(None)] * raw.ndim
                slices[axis] = slice(0, level)
                raw = raw[tuple(slices)]
            semantic.update(str((value.dtype, raw.shape)).encode())
            semantic.update(raw.tobytes())
        result.append(
            {
                "class": type(cache).__name__,
                "offset": cache.offset,
                "semantic_sha256": semantic.hexdigest(),
                "allocated_sha256": allocated.hexdigest(),
                "physical_shapes": shapes,
            }
        )
    return result


def run(args, report):
    import mlx.core as mx
    import numpy as np

    if not mx.metal.is_available():
        raise RuntimeError("Metal backend unavailable")
    mx.set_default_device(mx.gpu)
    if mx.default_device() != mx.gpu:
        raise RuntimeError("Metal default device required")
    import mlx2.runtime.external_speculative as external
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
    from mlx2.runtime.drafters.attention_windows import (
        configure_attention_windows,
        validate_attention_windows,
    )
    from mlx2.runtime.sample_utils import LaneRNG
    from mlx2.runtime.speculative_sampling import softmax

    if not Path(external.__file__).resolve().is_relative_to((ROOT / "src").resolve()):
        raise RuntimeError("executor imported from outside source archive")
    draft_config = json.loads((Path(report["draft"]) / "config.json").read_text())
    report["draft_attention_windows"] = resolve_attention_windows(
        report, draft_config["num_hidden_layers"]
    )
    report["draft_num_hidden_layers"] = draft_config["num_hidden_layers"]
    validate_attention_windows(
        report["draft_attention_windows"], draft_config["num_hidden_layers"]
    )
    if draft_config.get("architectures") != ["Qwen3XPressModel"]:
        raise ValueError("matrix requires XPress architecture")
    report["compute_precision"] = {
        "mode": report["compute_precision_requested"],
        "original_artifact_identity_retained": True,
        "production_precision_changed": False,
    }
    if report["compute_precision_requested"] == "float32-diagnostic":
        import psutil

        resident, largest, dtypes = float32_artifact_forecast(
            [Path(report["model"]), Path(report["draft"])]
        )
        recommended = int(mx.device_info().get("max_recommended_working_set_size", 0))
        report["compute_precision"].update(
            original_parameter_dtype_counts=dtypes,
            memory_guard=guard_float32_footprint(
                resident, largest, int(psutil.virtual_memory().available), recommended
            ),
            target_parameter_dtype="float32",
            draft_parameter_dtype="float32",
            diagnostic_only=True,
        )
    started = time.perf_counter()
    adapter = StandardDecoderAdapter(
        report["model"],
        execution_policy={
            "draft_model": report["draft"],
            "num_draft": 15,
            "xpress_num_passes": 6,
        },
    )
    if adapter.descriptor.model_type != "qwen3":
        raise ValueError("matrix requires Qwen3 target")
    report.update(
        device="Metal GPU",
        artifact_identity=adapter.identity,
        load_seconds_diagnostic=time.perf_counter() - started,
    )
    model, draft = adapter.model, adapter.draft_model
    if report["compute_precision_requested"] == "float32-diagnostic":
        cast_float32_diagnostic(model, draft)
    original_config = draft.config
    report["results"] = []
    report["aggregate"] = {
        "real_rejected_blocks": 0,
        "sampled_laws": 0,
        "proposed_tokens": 0,
    }
    tokenizer = adapter.tokenizer

    def prompt(theme, length, variant):
        instructions = [
            "Write Python code that implements a stable merge sort. Explain the loop invariant and handle empty input.",
            "Solve the algebra problem 3x + 7 = 28 and then describe a general way to verify a proposed solution.",
            "Write a concise account of a researcher observing a storm from a mountain station, with concrete sensory details.",
            "Answer in French and Japanese: explain how a library lends books and summarize the rules clearly.",
        ]
        passages = []
        for index in range(600):
            if theme == 0:
                passages.append(
                    f"Case {index + variant}: values = [{index % 17}, {(index * 7) % 29}, {(index * 11) % 31}]; expected sorted output must preserve equal-element order. Define merge_{index}(left, right)."
                )
            elif theme == 1:
                passages.append(
                    f"Exercise {index + variant}: a={index % 41 + 1}, b={index % 23 + 2}, c={index % 67 + 3}; reason about ax+b=c, exact rational solutions, and substitution checks."
                )
            elif theme == 2:
                passages.append(
                    f"Observation {index + variant}: time {index % 24:02d}:{index % 60:02d}, pressure {970 + index % 50} hPa, wind {index % 45} km/h; the notebook records changes in visibility and cloud layers."
                )
            else:
                passages.append(
                    f"Question {index + variant}: bibliothèque, prêt, retour, lecture; 図書館、貸出、返却、読書。Describe example number {index} using clear multilingual sentences."
                )
        content = instructions[theme] + "\n" + "\n".join(passages)
        ids = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )
        )
        # Trim only the user-message body: preserve the opening template and
        # the last 64 IDs containing the closing template/generation prefix.
        if len(ids) < length:
            raise AssertionError("prompt corpus too short")
        return ids[: length - 64] + ids[-64:]

    def prefill_reference(ids):
        cache = model.make_cache()
        for start in range(0, len(ids) - 1, 128):
            chunk = ids[start : min(start + 128, len(ids) - 1)]
            mx.eval(model(mx.array([chunk]), cache=cache))
        logits = model(mx.array([[ids[-1]]]), cache=cache)
        mx.eval(logits, [c.state for c in cache])
        return cache, logits[0, -1]

    def ordinary(ids, count, diagnostics=None):
        cache, logits = prefill_reference(ids)
        output = []
        for index in range(count):
            if diagnostics is not None:
                diagnostics.append(top_logits(host_float32(logits)))
            token = int(mx.argmax(logits).item())
            output.append(token)
            if index + 1 < count:
                logits = model(mx.array([[token]]), cache=cache)[0, -1]
        return output, cache

    def fresh_boundary(ids, prompt_length):
        cache, _ = prefill_reference(ids[:prompt_length])
        for token in ids[prompt_length:]:
            mx.eval(model(mx.array([[token]]), cache=cache))
        return cache

    def compare_next(cache, history, anchor, prompt_length):
        branch, _, _ = snapshot_prompt_cache_descriptors(cache)
        actual = host_float32(model(mx.array([[anchor]]), cache=branch)[0, -1])
        fresh = fresh_boundary(history, prompt_length)
        expected = host_float32(model(mx.array([[anchor]]), cache=fresh)[0, -1])
        error = float(np.max(np.abs(actual - expected)))
        # BF16 block and S=1 Metal arithmetic can differ by a few ULPs.
        # Exact greedy ID is required; report the numerical difference.
        if np.argmax(actual) != np.argmax(expected) or error > 0.5:
            raise AssertionError(f"committed cache next-logit mismatch max_abs={error}")
        return {
            "max_abs": error,
            "argmax": int(np.argmax(actual)),
            "argmax_equal": True,
            "tolerance": 0.5,
        }

    def next_logit_hash(lane):
        branch, _, _ = snapshot_prompt_cache_descriptors(lane.cache)
        values = host_float32(model(mx.array([[lane.anchor]]), cache=branch)[0, -1])
        return hashlib.sha256(values.tobytes()).hexdigest()

    def lane_state(lane):
        return {
            "history": list(lane.history),
            "anchor": lane.anchor,
            "rng": lane.rng.snapshot(),
            "target": cache_diagnostics(lane.cache),
            "draft": cache_diagnostics(lane.draft_cache),
            "tail_sha256": hashlib.sha256(
                host_float32(lane.tail).tobytes()
            ).hexdigest(),
            "tail_geometry": str((lane.tail.shape, lane.tail.dtype)),
            "generated": lane.generated,
            "ready_tokens": [r.token for r in lane.ready],
            "next_logits_sha256": next_logit_hash(lane),
        }

    def semantic_lane_state(state):
        return {
            **state,
            "target": [c["semantic_sha256"] for c in state["target"]],
            "draft": [c["semantic_sha256"] for c in state["draft"]],
        }

    def settle(batch):
        for lane in list(batch.lanes.values()):
            while lane.anchor is None:
                batch._prefill(lane)

    def drain(batch):
        outputs, final = {}, {}
        for _ in range(512):
            _, responses = batch.next()
            failures = batch.take_lane_failures()
            if failures:
                raise RuntimeError(f"lane failures: {failures}")
            for response in responses:
                outputs.setdefault(response.uid, []).append(response.token)
                if response.finish_reason:
                    final[response.uid] = response
            if not batch.lanes:
                return outputs, final
        raise RuntimeError("scheduler poll limit exhausted")

    for cell_index, cell in enumerate(report["cells"]):
        result = {"cell": cell, "passed": False}
        report["results"].append(result)
        print(json.dumps({"event": "cell_start", "name": cell["name"]}), flush=True)
        started = time.perf_counter()
        batch = None
        original_forward = model.forward_with_taps
        original_verify = external.verify_proposals
        apc = None
        hit = None
        rejected = [0]
        checked_rows = [0]
        law_errors = {}
        try:
            windows = report["draft_attention_windows"] if cell.get("windows") else None
            draft.config = replace(original_config, xpress_num_passes=cell["passes"])
            for layer in draft.layers:
                layer.self_attn.is_sliding = False
                layer.self_attn.sliding_window = None
            configure_attention_windows(
                draft, validate_attention_windows(windows, len(draft.layers))
            )
            settings = dict(draft.receipt_settings)
            binding = hashlib.sha256(
                (
                    adapter.identity["fingerprint"]
                    + json.dumps(settings, sort_keys=True)
                    + report["compute_precision_requested"]
                ).encode()
            ).hexdigest()
            result.update(binding=binding, draft_settings=settings)
            result["compute_precision"] = report["compute_precision_requested"]
            width = cell["width"]
            # Different histories within cohorts, including a near-4096 pair
            # in long-context cells and small ragged differences elsewhere.
            lengths = [max(64, cell["context"] - row * 11) for row in range(width)]
            themes = [(cell.get("theme", cell_index) + row) % 4 for row in range(width)]
            ids = [
                prompt(theme, length, cell_index * 100 + row)
                for row, (theme, length) in enumerate(zip(themes, lengths))
            ]
            if cell.get("french"):
                exact = list(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": FRENCH_PROMPT}],
                        add_generation_prompt=True,
                        tokenize=True,
                        enable_thinking=False,
                    )
                )
                ids = [list(exact) for _ in range(width)]
                result["prompt_ids"] = ids
            temps = [0.0 if row % 2 == 0 else 0.8 for row in range(width)]
            if cell.get("fault"):
                temps = [0.0, 0.8]
            result.update(
                prompt_lengths=list(map(len, ids)),
                prompt_sha256=[
                    hashlib.sha256(np.asarray(p, dtype=np.int32).tobytes()).hexdigest()
                    for p in ids
                ],
                themes=themes,
                temperatures=temps,
            )
            expected = {}
            first_laws = {}
            ordinary_diagnostics = {}
            result["ordinary_first_logits"] = {}
            for row, (p, temp, budget) in enumerate(zip(ids, temps, cell["budgets"])):
                if temp == 0:
                    ordinary_diagnostics[row] = []
                    expected[row] = ordinary(p, budget, ordinary_diagnostics.get(row))[
                        0
                    ]
                else:
                    _, first = prefill_reference(p)
                    first_laws[row] = softmax(host_float32(first), temp)
                    result["ordinary_first_logits"][row] = top_logits(
                        host_float32(first)
                    )
            batch = external.ExternalDraftBatchGenerator(
                model,
                draft_model=draft,
                binding=binding,
                num_draft=15,
                completion_batch_size=width,
                prefill_step_size=128,
                stop_tokens=(),
                ready_drain="all",
            )
            insert_options = {}
            if cell.get("warm"):
                prefix_caches = []
                prefix_states = []
                for p in ids:
                    initial = external.ExternalDraftBatchGenerator(
                        model,
                        draft_model=draft,
                        binding=binding,
                        num_draft=15,
                        prefill_step_size=128,
                    )
                    initial.insert([p], max_tokens=[17])
                    settle(initial)
                    lane = initial.lanes[0]
                    frozen, state, _ = snapshot_prompt_cache_descriptors(
                        lane.cache, initial._sidecar(lane)
                    )
                    prefix_caches.append(frozen)
                    prefix_states.append(state)
                    initial.remove([0], cancelled=True)
                insert_options = {
                    "caches": prefix_caches,
                    "cache_states": prefix_states,
                    "all_tokens": [p[:-1] for p in ids],
                }
            uids = batch.insert(
                [p[-1:] for p in ids] if cell.get("warm") else ids,
                max_tokens=cell["budgets"],
                lane_rngs=[LaneRNG(1200 + cell_index * 8 + r) for r in range(width)],
                sampling_configs=[{"sampling_temp": temp} for temp in temps],
                **insert_options,
            )
            rng_rows = {id(batch.lanes[uid].rng): row for row, uid in enumerate(uids)}
            law_errors = {}
            rejected = [0]
            checked_rows = [0]

            def checked(
                tokens,
                proposals,
                targets,
                rng,
                _state=(
                    rng_rows,
                    temps,
                    first_laws,
                    law_errors,
                    checked_rows,
                    original_verify,
                    rejected,
                ),
                **kwargs,
            ):
                (
                    rng_rows,
                    temps,
                    first_laws,
                    law_errors,
                    checked_rows,
                    original_verify,
                    rejected,
                ) = _state
                row = rng_rows.get(id(rng))
                for token, q, p in zip(tokens, proposals, targets):
                    q = np.asarray(q)
                    p = np.asarray(p, dtype=np.float64)
                    if q[token] != 1 or np.count_nonzero(q) != 1:
                        raise AssertionError("proposal is not actual point mass")
                    if (
                        not np.all(np.isfinite(p))
                        or p.min() < 0
                        or abs(p.sum() - 1) > 1e-6
                    ):
                        raise AssertionError("invalid target law")
                    if temps[row] > 0:
                        checked_rows[0] += 1
                        if row not in law_errors:
                            error = float(np.abs(p - first_laws[row]).sum())
                            law_errors[row] = error
                            if error > 0.01:
                                raise AssertionError(
                                    f"sampled ordinary-law L1 mismatch {error}"
                                )
                        residual = p.copy()
                        residual[token] = 0
                        if np.max(np.abs(residual + p[token] * q - p)) > 1e-12:
                            raise AssertionError(
                                "acceptance/residual law identity mismatch"
                            )
                decision = original_verify(tokens, proposals, targets, rng, **kwargs)
                rejected[0] += int(bool(tokens) and decision.rejected)
                return decision

            external.verify_proposals = checked
            result["ordinary_logits"] = ordinary_diagnostics
            result["external_logits"] = []

            def traced(
                inputs, *values, _forward=original_forward, _result=result, **kwargs
            ):
                caches = values[0] if values else kwargs.get("cache")
                offsets = []
                for c in caches:
                    offset = c.offset
                    offsets.append(
                        offset.tolist() if hasattr(offset, "tolist") else offset
                    )
                logits, taps = _forward(inputs, *values, **kwargs)
                if (
                    not _result["cell"].get("french")
                    and len(_result["external_logits"]) >= 2
                ):
                    return logits, taps
                predictions = prediction_rows(logits)
                if predictions is None:
                    return logits, taps
                _result["external_logits"].append(
                    {
                        "query_shape": list(inputs.shape),
                        "inputs": inputs.tolist(),
                        "cache_offsets": offsets,
                        "predictions": predictions,
                    }
                )
                return logits, taps

            model.forward_with_taps = traced
            settle(batch)
            if cell.get("cancel"):
                batch.remove([uids[1]], cancelled=True)
                result["cancelled_uid"] = uids[1]
            if cell.get("fault"):
                batch._round(list(batch.lanes.values()))
                lanes = list(batch.lanes.values())
                snapshots = []
                for lane in lanes:
                    snapshots.append(lane_state(lane))

                def fail(*values, _forward=original_forward, **kwargs):
                    _forward(*values, **kwargs)
                    raise RuntimeError(
                        "injected later target failure after cache append"
                    )

                model.forward_with_taps = fail
                try:
                    batch._round(lanes)
                except RuntimeError as error:
                    if "injected later" not in str(error):
                        raise
                else:
                    raise AssertionError("fault was not triggered")
                finally:
                    model.forward_with_taps = original_forward
                result["rollback_diagnostics"] = []
                for lane, before in zip(lanes, snapshots):
                    after = lane_state(lane)
                    differences = [
                        field
                        for field, value in semantic_lane_state(before).items()
                        if semantic_lane_state(after)[field] != value
                    ]
                    result["rollback_diagnostics"].append(
                        {
                            "uid": lane.uid,
                            "before": before,
                            "after": after,
                            "semantic_changed_fields": differences,
                            "allocation_changed": {
                                plane: [
                                    i
                                    for i, (a, b) in enumerate(
                                        zip(before[plane], after[plane])
                                    )
                                    if a["allocated_sha256"] != b["allocated_sha256"]
                                ]
                                for plane in ("target", "draft")
                            },
                        }
                    )
                    if differences:
                        raise AssertionError(
                            f"committed recovery state changed after failed round: {differences}"
                        )
                result["rollback_restored_all_planes"] = True
                rng_rows.clear()
                rng_rows.update(
                    {id(batch.lanes[uid].rng): row for row, uid in enumerate(uids)}
                )
            outputs, final = drain(batch)
            result["token_ids"] = outputs
            for row, uid in enumerate(uids):
                if cell.get("cancel") and row == 1:
                    if uid in outputs:
                        raise AssertionError("cancelled lane emitted tokens")
                    continue
                if len(outputs.get(uid, [])) != cell["budgets"][row]:
                    raise AssertionError("wrong token budget")
                if temps[row] == 0 and outputs[uid] != expected[row]:
                    index = next(
                        i
                        for i, (a, e) in enumerate(zip(outputs[uid], expected[row]))
                        if a != e
                    )
                    result["first_divergence"] = {
                        "row": row,
                        "position": index,
                        "actual": outputs[uid][index],
                        "expected": expected[row][index],
                        "committed_history": ids[row] + outputs[uid][:index],
                        "ordinary_prediction": ordinary_diagnostics.get(
                            row, [None] * len(expected[row])
                        )[index],
                    }
                    result["ordinary_token_ids"] = expected
                    raise AssertionError(
                        f"greedy token mismatch row{row} position{index}"
                    )
            stats = dict(batch.scheduler_stats)
            result.update(
                stats=stats,
                sampled_first_l1=law_errors,
                real_rejected_blocks=rejected[0],
                sampled_laws=checked_rows[0],
            )
            if (
                stats["draft_fallbacks"]
                or not stats["external_rounds"]
                or not stats["proposed_tokens"]
            ):
                raise AssertionError("draft mechanism missing or fallback")
            if width > 1 and not cell.get("cancel") and stats["target_max_width"] < 2:
                raise AssertionError("batched verification did not engage")
            result["receipts"] = {
                uid: end.speculative_receipt for uid, end in final.items()
            }
            result["cache_next_logits"] = {}
            for uid, end in final.items():
                receipt = end.speculative_receipt
                if (
                    receipt["kind"] != "external_xpress"
                    or receipt["ordinary_fallback"]
                    or receipt["draft_settings"] != settings
                ):
                    raise AssertionError("incorrect setting/route receipt")
                end.cache_sidecar.validate(binding, len(end.all_tokens))
                result["cache_next_logits"][uid] = compare_next(
                    end.prompt_cache,
                    end.all_tokens,
                    end.token,
                    len(ids[uids.index(uid)]),
                )
            if cell.get("windows"):
                end = final[uids[0]]
                draft_cache, _tail = end.cache_sidecar.state
                result["window_cache"] = [
                    {"offset": c.offset, "physical_keys": c.keys.shape[-2]}
                    for c in draft_cache
                ]
                for c, w in zip(draft_cache, windows):
                    if w is not None and (c.offset <= w or c.keys.shape[-2] > w):
                        raise AssertionError(
                            "ring wrap or bounded physical storage missing"
                        )
                apc = APCv2(max_size=1, layout_name=binding)
                key = APCKey(
                    "qwen3+xpress", revision=binding, cache_layout_fingerprint=binding
                )
                apc.store(
                    key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar
                )
                hit = apc.lookup(key, end.all_tokens + [end.token])
                if not hit.hit or hit.hit_kind != "external_draft_sidecar":
                    raise AssertionError("paired APCv2 hit missing")
                other = APCKey(
                    "qwen3+xpress",
                    revision="wrong-binding",
                    cache_layout_fingerprint=binding,
                )
                if apc.lookup(other, end.all_tokens + [end.token]).hit:
                    raise AssertionError("APC revision mismatch accepted")
                external.verify_proposals = original_verify
                resumed = external.ExternalDraftBatchGenerator(
                    model,
                    draft_model=draft,
                    binding=binding,
                    num_draft=15,
                    prefill_step_size=128,
                    ready_drain="all",
                )
                resumed.insert(
                    [[end.token]],
                    max_tokens=[17],
                    caches=[hit.cache],
                    all_tokens=[end.all_tokens],
                    cache_states=[hit.sidecar],
                )
                continued, finished = drain(resumed)
                expected_continue = ordinary(end.all_tokens + [end.token], 17)[0]
                if (
                    continued[0] != expected_continue
                    or resumed.scheduler_stats["paired_cache_resumes"] != 1
                ):
                    raise AssertionError(
                        "window paired resume ordinary parity/mechanism failed"
                    )
                result["window_apcv2_resume"] = {
                    "token_ids": continued[0],
                    "stats": dict(resumed.scheduler_stats),
                    "next_logits": compare_next(
                        finished[0].prompt_cache,
                        finished[0].all_tokens,
                        finished[0].token,
                        len(ids[0]),
                    ),
                }
            for key, value in [
                ("real_rejected_blocks", rejected[0]),
                ("sampled_laws", checked_rows[0]),
                ("proposed_tokens", stats["proposed_tokens"]),
            ]:
                report["aggregate"][key] += value
            result["passed"] = True
        except TimeoutError:
            raise
        except Exception as error:  # noqa: BLE001 - per-cell diagnostic receipt
            result.update(
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        finally:
            model.forward_with_taps = original_forward
            external.verify_proposals = original_verify
            result.setdefault("real_rejected_blocks", rejected[0])
            result.setdefault("sampled_laws", checked_rows[0])
            result.setdefault("sampled_first_l1", law_errors)
            if batch is not None:
                result.setdefault("stats", dict(batch.scheduler_stats))
            if hit is not None:
                hit.cache.close()
            if apc is not None:
                apc.clear(release_memory=False)
            if batch is not None and batch.lanes:
                batch.remove(list(batch.lanes), cancelled=True)
            result["seconds_diagnostic"] = time.perf_counter() - started
            save(args, report)
            print(
                json.dumps(
                    {
                        "event": "cell_end",
                        "name": cell["name"],
                        "passed": result["passed"],
                        "error": result.get("error"),
                    }
                ),
                flush=True,
            )
            mx.clear_cache()
    if report["aggregate"]["real_rejected_blocks"] <= 0:
        raise AssertionError("no genuine rejection observed")
    if report["aggregate"]["sampled_laws"] <= 0:
        raise AssertionError("sampled target laws unobserved")
    report["draft_counters"] = dict(draft.stats)
    adapter.close()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = preflight(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    report["source_identity"] = source_identity()
    report["started_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def timeout(_sig, _frame):
        raise TimeoutError(f"matrix exceeded {args.timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        run(args, report)
        report["passed"] = all(cell["passed"] for cell in report["results"])
    except Exception as error:  # noqa: BLE001 - final diagnostic receipt
        report.update(
            passed=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        signal.alarm(0)
    save(args, report)
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
