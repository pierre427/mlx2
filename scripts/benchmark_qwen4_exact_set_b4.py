#!/usr/bin/env python3
"""Model-bound B2/B4 QSA fold benchmark, including selection and proof costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.models.qwen4_exp import QSACompactBlocks, QSASelection
from mlx2.runtime.models.qwen4_qsa_indexed import (
    compact_blocks_to_kernel_inputs,
    qwen4_qsa_indexed_private_delta_attention,
    qwen4_qsa_indexed_private_delta_exact_set_attention,
    qwen4_qsa_indexed_private_delta_exact_set_proof,
)
from mlx2.runtime.segmented_qsa_metal import segmented_shared_prefix_attention_metal


def _require_lock() -> None:
    missing = [
        str(path)
        for path in (
            Path("/tmp/gpu.lock/owner.json"),
            Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        )
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"Qwen4 B4 benchmark requires both GPU locks: {missing}")


def _load(path: Path):
    data = np.load(path)
    dtype = mx.bfloat16 if str(data["source_dtype"][0]) == "mlx.core.bfloat16" else mx.float16

    def tensor(name, target=dtype):
        return mx.array(data[name]).astype(target)

    selection = QSASelection(
        kind="explicit",
        batch=4,
        length=1,
        block_size=int(data["block_size"][0]),
        raw_block_ids=mx.array(data["selected"]),
        valid_blocks=mx.array(data["valid_blocks"]),
        q_positions=mx.array(data["q_positions"]),
        token_positions=mx.array(data["token_positions"]),
        causal_mask=None,
        left_padding=None,
        offset=int(data["q_positions"][0, 0]),
        physical_width=int(data["physical_width"][0]),
        n_blocks=int(data["n_blocks"][0]),
        scatter_chosen=True,
    )
    values = {
        "path": path,
        "dtype": dtype,
        "q": tensor("q"),
        "base_k": tensor("base_k"),
        "base_v": tensor("base_v"),
        "delta_k": tensor("delta_k"),
        "delta_v": tensor("delta_v"),
        "index_q": tensor("index_q"),
        "pooled": tensor("pooled"),
        "selection": selection,
        "scale": float(data["scale"][0]),
        "base_tokens": int(data["base_tokens"][0]),
        "layer": int(data["layer"][0]),
        "step": int(data["step"][0]),
    }
    mx.eval(*(value for value in values.values() if isinstance(value, mx.array)))
    return values


def _slice_compact(compact, rows: int):
    return QSACompactBlocks(
        block_ids=compact.block_ids[:rows],
        block_counts=compact.block_counts[:rows],
        tail_start=compact.tail_start[:rows],
        tail_stop=compact.tail_stop[:rows],
        left_padding=None,
        block_size=compact.block_size,
        physical_width=compact.physical_width,
        causal_mask=None,
    )


def _slice_selection(selection, rows: int):
    return QSASelection(
        kind="explicit",
        batch=rows,
        length=selection.length,
        block_size=selection.block_size,
        raw_block_ids=selection.raw_block_ids[:rows],
        valid_blocks=selection.valid_blocks[:rows],
        q_positions=selection.q_positions[:rows],
        token_positions=selection.token_positions,
        causal_mask=None,
        left_padding=None,
        offset=selection.offset,
        physical_width=selection.physical_width,
        n_blocks=selection.n_blocks,
        scatter_chosen=True,
    )


def _selection_replay(values, rows: int):
    query = values["index_q"][:rows]
    pooled = values["pooled"][:rows]
    valid = values["selection"].valid_blocks[:rows]
    scores = mx.einsum(
        "blhd,bnd->blnh", query.astype(mx.float32), pooled.astype(mx.float32)
    )
    scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(query.shape[-1])
    scores = mx.where(valid, scores, -mx.inf)
    n_blocks = int(scores.shape[-1])
    width = int(values["selection"].raw_block_ids.shape[-1])
    return mx.argpartition(scores, kth=n_blocks - width, axis=-1)[..., -width:]


def _b4_proof(compact, base_tokens: int):
    ids, counts, _selected, width, _qpos, _left, _total = compact_blocks_to_kernel_inputs(compact)
    base_block = base_tokens // compact.block_size
    slots = mx.arange(width, dtype=mx.uint32)[None, None, :]
    valid = (slots < counts[..., None]) & (ids < base_block)
    anchor_valid = valid[0:1]
    anchor_ids = ids[0:1]
    same_valid = mx.all(valid == anchor_valid, axis=(0, 2))
    same_ids = mx.all(~(valid | anchor_valid) | (ids == anchor_ids), axis=(0, 2))
    return mx.contiguous(same_valid & same_ids)


def _measure(function, warmups: int, repeats: int):
    for _ in range(warmups):
        result = function()
        mx.eval(*(result if isinstance(result, tuple) else (result,)))
    samples = []
    last = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        last = function()
        mx.eval(*(last if isinstance(last, tuple) else (last,)))
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }, last


def _kernel_case(values, rows: int, *, exact: bool, warmups: int, repeats: int):
    compact = _slice_compact(values["selection"].compact_blocks(), rows)
    lengths = mx.full((rows,), values["delta_k"].shape[2], dtype=mx.uint32)
    function = (
        qwen4_qsa_indexed_private_delta_exact_set_attention if exact
        else qwen4_qsa_indexed_private_delta_attention
    )
    timing, output = _measure(
        lambda: function(
            values["q"][:rows], values["base_k"], values["base_v"],
            values["delta_k"][:rows], values["delta_v"][:rows], lengths, compact,
            scale=values["scale"],
        ),
        warmups,
        repeats,
    )
    return timing, output


def _candidate_inputs(values):
    compact = values["selection"].compact_blocks()
    mx.eval(compact.block_ids, compact.block_counts, compact.tail_start, compact.tail_stop)
    ids = np.asarray(compact.block_ids)
    counts = np.asarray(compact.block_counts)
    tails = list(zip(np.asarray(compact.tail_start)[:, 0], np.asarray(compact.tail_stop)[:, 0]))
    block_size = compact.block_size
    base_tokens = values["base_tokens"]
    base_blocks = base_tokens // block_size
    base_ids = [int(x) for x in ids[0, 0, : int(counts[0, 0])] if int(x) < base_blocks]
    for row in range(1, 4):
        row_base = [int(x) for x in ids[row, 0, : int(counts[row, 0])] if int(x) < base_blocks]
        if row_base != base_ids:
            raise ValueError("captured B4 sample is not an exact immutable-base-set hit")
    base_indices = np.asarray(
        [block * block_size + token for block in base_ids for token in range(block_size)],
        dtype=np.uint32,
    )
    suffix_indices = []
    for row in range(4):
        logical = []
        for block in ids[row, 0, : int(counts[row, 0])]:
            block = int(block)
            if block >= base_blocks:
                logical.extend(block * block_size + token for token in range(block_size))
        logical.extend(range(int(tails[row][0]), int(tails[row][1])))
        logical = sorted(set(value for value in logical if base_tokens <= value < compact.physical_width))
        suffix_indices.append([value - base_tokens for value in logical])
    widths = {len(row) for row in suffix_indices}
    if len(widths) != 1:
        raise ValueError("prototype benchmark requires equal gathered suffix widths")
    suffix_index = mx.array(np.asarray(suffix_indices, dtype=np.uint32))

    def gather(source):
        index = mx.broadcast_to(
            suffix_index[:, None, :, None],
            (4, source.shape[1], suffix_index.shape[1], source.shape[3]),
        )
        return mx.take_along_axis(source, index, axis=2)

    return mx.array(base_indices), suffix_index, gather


def _errors(actual, expected):
    delta = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    relative = delta / mx.maximum(mx.abs(expected.astype(mx.float32)), 1.0e-6)
    maximum = mx.max(delta)
    maximum_relative = mx.max(relative)
    mean = mx.mean(delta)
    mx.eval(maximum, maximum_relative, mean)
    return {
        "max_abs": float(maximum.item()),
        "mean_abs": float(mean.item()),
        "max_relative": float(maximum_relative.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-sample", required=True, type=Path)
    parser.add_argument("--miss-sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    _require_lock()
    mx.set_default_device(mx.gpu)
    hit = _load(args.hit_sample)
    miss = _load(args.miss_sample)

    report = {
        "schema": "mlx2.qwen4-exact-set-b4-model-bound-ab.v1",
        "device": mx.device_info(),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "hit_sample": {"path": str(args.hit_sample.resolve()), "layer": hit["layer"], "step": hit["step"]},
        "miss_sample": {"path": str(args.miss_sample.resolve()), "layer": miss["layer"], "step": miss["step"]},
        "dtype": str(hit["dtype"]),
        "qualification": False,
        "route_selected": False,
    }

    selection = {}
    compact = {}
    proof = {}
    for rows in (2, 4):
        selection[str(rows)], selected = _measure(
            lambda rows=rows: _selection_replay(hit, rows), args.warmups, args.repeats
        )
        production = hit["selection"].raw_block_ids[:rows]
        replay_agrees = mx.all(mx.sort(selected, axis=-1) == mx.sort(production, axis=-1))
        mx.eval(replay_agrees)
        selection[str(rows)]["set_agrees_with_production"] = bool(replay_agrees.item())
        sliced = _slice_selection(hit["selection"], rows)
        compact[str(rows)], _ = _measure(
            lambda sliced=sliced: (
                sliced.compact_blocks().block_ids,
                sliced.compact_blocks().block_counts,
                sliced.compact_blocks().tail_start,
                sliced.compact_blocks().tail_stop,
            ),
            args.warmups,
            args.repeats,
        )
    compact_b2 = _slice_selection(hit["selection"], 2).compact_blocks()
    proof["b2_existing"], b2_proof = _measure(
        lambda: qwen4_qsa_indexed_private_delta_exact_set_proof(
            compact_b2, base_tokens=hit["base_tokens"]
        ),
        args.warmups,
        args.repeats,
    )
    proof["b4_candidate"], b4_proof = _measure(
        lambda: _b4_proof(hit["selection"].compact_blocks(), hit["base_tokens"]),
        args.warmups,
        args.repeats,
    )
    proof["b2_existing"]["value"] = np.asarray(b2_proof).tolist()
    proof["b4_candidate"]["value"] = np.asarray(b4_proof).tolist()
    report["selection"] = selection
    report["compaction"] = compact
    report["proof"] = proof

    b2_row, b2_row_out = _kernel_case(hit, 2, exact=False, warmups=args.warmups, repeats=args.repeats)
    b2_fold, b2_fold_out = _kernel_case(hit, 2, exact=True, warmups=args.warmups, repeats=args.repeats)
    b4_row, b4_row_out = _kernel_case(hit, 4, exact=False, warmups=args.warmups, repeats=args.repeats)
    miss_b2_row, miss_b2_row_out = _kernel_case(miss, 2, exact=False, warmups=args.warmups, repeats=args.repeats)
    miss_b2_fold, miss_b2_fold_out = _kernel_case(miss, 2, exact=True, warmups=args.warmups, repeats=args.repeats)

    base_indices, _suffix_index, gather = _candidate_inputs(hit)
    gather_timing, gathered = _measure(
        lambda: (gather(hit["delta_k"]), gather(hit["delta_v"])),
        args.warmups,
        args.repeats,
    )
    suffix_k, suffix_v = gathered
    suffix_lengths = mx.full((4,), suffix_k.shape[2], dtype=mx.uint32)
    candidate_timing, candidate_result = _measure(
        lambda: segmented_shared_prefix_attention_metal(
            hit["q"], hit["base_k"], hit["base_v"], suffix_k, suffix_v,
            suffix_lengths, scale=hit["scale"], splits=32, base_indices=base_indices,
        )[:2],
        args.warmups,
        args.repeats,
    )
    candidate_out, engaged = candidate_result
    mx.eval(engaged)

    report["kernels"] = {
        "b2_row_local_hit": b2_row,
        "b2_existing_fold_hit": b2_fold,
        "b4_row_local_hit": b4_row,
        "b4_candidate_hit": {**candidate_timing, "engaged": int(engaged.item())},
        "b4_candidate_suffix_gather": gather_timing,
        "b2_row_local_miss": miss_b2_row,
        "b2_existing_fold_miss": miss_b2_fold,
    }
    report["correctness"] = {
        "b2_fold_vs_row_local_hit": _errors(b2_fold_out, b2_row_out),
        "b4_candidate_vs_row_local_hit": _errors(candidate_out, b4_row_out),
        "b2_fold_vs_row_local_miss": _errors(miss_b2_fold_out, miss_b2_row_out),
    }

    b2_total = selection["2"]["median_ms"] + compact["2"]["median_ms"] + b2_fold["median_ms"]
    b2_row_total = selection["2"]["median_ms"] + compact["2"]["median_ms"] + b2_row["median_ms"]
    b4_total = (
        selection["4"]["median_ms"] + compact["4"]["median_ms"]
        + proof["b4_candidate"]["median_ms"] + gather_timing["median_ms"]
        + candidate_timing["median_ms"]
    )
    b4_row_total = selection["4"]["median_ms"] + compact["4"]["median_ms"] + b4_row["median_ms"]
    miss_b2_total = selection["2"]["median_ms"] + compact["2"]["median_ms"] + miss_b2_fold["median_ms"]
    miss_b2_row_total = selection["2"]["median_ms"] + compact["2"]["median_ms"] + miss_b2_row["median_ms"]
    report["composed_medians_ms"] = {
        "b2_existing_fold_hit": b2_total,
        "b2_row_local_hit": b2_row_total,
        "b4_candidate_hit": b4_total,
        "b4_row_local_hit": b4_row_total,
        "b2_existing_fold_miss": miss_b2_total,
        "b2_row_local_miss": miss_b2_row_total,
    }
    report["comparisons"] = {
        "b2_fold_hit_speedup_vs_row_local": b2_row_total / b2_total,
        "b4_candidate_hit_speedup_vs_row_local": b4_row_total / b4_total,
        "b4_candidate_per_row_speedup_vs_b2_fold": (b2_total / 2) / (b4_total / 4),
        "b2_fold_miss_speedup_vs_row_local": miss_b2_row_total / miss_b2_total,
    }
    report["passed"] = bool(
        selection["2"]["set_agrees_with_production"]
        and selection["4"]["set_agrees_with_production"]
        and all(value["max_abs"] <= 0.125 for value in report["correctness"].values())
        and int(engaged.item()) == 1
    )
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            ROOT / "src/mlx2/runtime/models/qwen4_qsa_indexed.py",
            ROOT / "src/mlx2/runtime/segmented_qsa_metal.py",
            Path(__file__).resolve(),
        )
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
