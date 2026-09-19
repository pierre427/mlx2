#!/usr/bin/env python3
"""Capture real Qwen4 QSA exact-set frequency and one model-bound sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _require_lock() -> None:
    paths = (
        Path("/tmp/gpu.lock/owner.json"),
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"real Qwen4 capture requires both GPU locks: {missing}")


def _exact_sets(selection, base_blocks: int):
    compact = selection.compact_blocks()
    import mlx.core as mx

    mx.eval(compact.block_ids, compact.block_counts, compact.tail_start, compact.tail_stop)
    ids = np.asarray(compact.block_ids)
    counts = np.asarray(compact.block_counts)
    rows = []
    full_rows = []
    for row in range(ids.shape[0]):
        count = int(counts[row, -1])
        full = tuple(int(value) for value in ids[row, -1, :count])
        full_rows.append(full)
        rows.append(tuple(value for value in full if value < base_blocks))
    equal_b2 = len(rows) >= 2 and rows[0] == rows[1]
    equal_b4 = len(rows) >= 4 and all(row == rows[0] for row in rows[1:4])
    full_equal_b2 = len(full_rows) >= 2 and full_rows[0] == full_rows[1]
    full_equal_b4 = len(full_rows) >= 4 and all(
        row == full_rows[0] for row in full_rows[1:4]
    )
    similarities = []
    anchor = set(rows[0])
    for row in rows[1:]:
        other = set(row)
        union = anchor | other
        similarities.append(1.0 if not union else len(anchor & other) / len(union))
    return {
        "compact": compact,
        "base_sets": rows,
        "full_sets": full_rows,
        "b2_base_hit": bool(equal_b2),
        "b4_base_hit": bool(equal_b4),
        "b2_full_hit": bool(full_equal_b2),
        "b4_full_hit": bool(full_equal_b4),
        "mean_anchor_jaccard": float(np.mean(similarities)) if similarities else 1.0,
        "minimum_anchor_jaccard": float(min(similarities)) if similarities else 1.0,
    }


def _sample_arrays(attention, hidden, cache, selection, base_tokens: int):
    import mlx.core as mx

    (qg, _k, _v, projected_qk) = attention._project_segmented_qsa(hidden)
    batch, length, _ = hidden.shape
    q_width = attention.num_heads * attention.head_dim
    (q, _gate) = mx.split(qg.reshape(batch, length, attention.num_heads, -1), 2, axis=-1)
    q = attention.q_norm(q).transpose(0, 2, 1, 3)
    q_pos = np.asarray(selection.q_positions)
    if not np.all(q_pos[:, -1] == q_pos[0, -1]):
        raise RuntimeError("capture requires equal decode offsets")
    q = attention.rope(q, offset=int(q_pos[0, -1]))

    index_q = projected_qk[..., : attention.indexer.n_heads * attention.indexer.head_dim]
    index_q = attention.indexer.q_layernorm(
        index_q.reshape(batch, length, attention.indexer.n_heads, attention.indexer.head_dim)
    )
    from mlx2.runtime.models.qwen4_exp import _apply_rope_positions

    index_q = _apply_rope_positions(
        index_q,
        selection.q_positions[..., None],
        attention.indexer.rotary_dim,
        attention.indexer.rope_theta,
    )
    width = int(selection.physical_width)
    keys = cache.keys[:, :, :width]
    values = cache.values[:, :, :width]
    pooled = cache._qsa_pooled_keys
    if pooled is None:
        raise RuntimeError("QSA pooled keys were not retained")
    return {
        "q": mx.contiguous(q),
        "base_k": mx.contiguous(keys[0:1, :, :base_tokens]),
        "base_v": mx.contiguous(values[0:1, :, :base_tokens]),
        "delta_k": mx.contiguous(keys[:, :, base_tokens:width]),
        "delta_v": mx.contiguous(values[:, :, base_tokens:width]),
        "index_q": mx.contiguous(index_q),
        "pooled": mx.contiguous(pooled),
        "selected": mx.contiguous(selection.raw_block_ids),
        "valid_blocks": mx.contiguous(selection.valid_blocks),
        "q_positions": mx.contiguous(selection.q_positions),
        "token_positions": mx.contiguous(selection.token_positions),
        "q_width": q_width,
        "scale": float(attention.scale),
        "block_size": int(selection.block_size),
        "physical_width": width,
        "n_blocks": int(selection.n_blocks),
    }


def _as_float32(value):
    import mlx.core as mx

    return np.asarray(value.astype(mx.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--base-tokens", type=int, default=2048)
    parser.add_argument("--suffix-tokens", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--prefill-step", type=int, default=256)
    args = parser.parse_args()
    _require_lock()
    if args.base_tokens % 4:
        raise SystemExit("base token count must align to Qwen4's four-token QSA blocks")

    import mlx.core as mx
    from mlx2.adapters.flash_next import FlashNextAdapter

    mx.set_default_device(mx.gpu)
    started = time.time()
    adapter = None
    report = {
        "schema": "mlx2.qwen4-exact-set-frequency.v1",
        "started_at": started,
        "model": str(args.model.resolve()),
        "platform": platform.platform(),
        "device": mx.device_info(),
        "workload": {
            "batch": 4,
            "base_tokens": args.base_tokens,
            "suffix_tokens": args.suffix_tokens,
            "decode_steps": args.decode_steps,
            "prefill_step": args.prefill_step,
            "shape": "one real common prefix, four equal-width divergent suffixes, greedy decode",
        },
        "passed": False,
    }
    try:
        load_started = time.perf_counter()
        adapter = FlashNextAdapter(str(args.model))
        report["load_seconds"] = time.perf_counter() - load_started
        report["artifact_identity"] = adapter.identity
        report["environment"] = adapter.environment

        attention_layers = []
        for layer_index, layer in enumerate(adapter.model.layers):
            if layer.is_linear:
                continue
            indexer = layer.self_attn.indexer
            base_class = type(indexer)

            class CapturingIndexer(base_class):
                def __call__(self, *call_args, **call_kwargs):
                    self._frequency_hidden = call_args[0]
                    result = super().__call__(*call_args, **call_kwargs)
                    self._frequency_selection = result
                    return result

            indexer.__class__ = CapturingIndexer
            attention_layers.append((layer_index, layer.self_attn))

        source = (
            "Implement an exact persistent radix tree in Python. Preserve stable ordering, "
            "transactional rollback, bounded memory, and deterministic tests. Explain each "
            "invariant before the code and include adversarial cases for shared prefixes.\n"
        )
        common = adapter.tokenizer.encode(source * 160, add_special_tokens=False)
        if len(common) < args.base_tokens:
            raise RuntimeError("common workload text did not produce enough tokens")
        common = common[: args.base_tokens]
        suffix_sources = (
            "Add deletion while preserving compressed edges and verify parent repair.\n",
            "Add concurrent readers with revision snapshots and prove isolation.\n",
            "Add prefix enumeration with a strict allocation budget and benchmarks.\n",
            "Add serialization with corruption detection and deterministic recovery.\n",
        )
        suffix_rows = []
        for text in suffix_sources:
            tokens = adapter.tokenizer.encode(text * 16, add_special_tokens=False)
            if len(tokens) < args.suffix_tokens:
                raise RuntimeError("suffix workload text did not produce enough tokens")
            suffix_rows.append(tokens[: args.suffix_tokens])
        report["workload"]["common_token_sha256"] = hashlib.sha256(
            np.asarray(common, dtype=np.int32).tobytes()
        ).hexdigest()
        report["workload"]["suffix_token_sha256"] = [
            hashlib.sha256(np.asarray(row, dtype=np.int32).tobytes()).hexdigest()
            for row in suffix_rows
        ]

        cache = adapter.model.make_cache()
        prefill_started = time.perf_counter()
        last_logits = None
        for begin in range(0, len(common), args.prefill_step):
            token_chunk = mx.array(common[begin : begin + args.prefill_step])[None]
            last_logits = adapter.model(token_chunk, cache=cache)
            mx.eval(last_logits)
            del last_logits
            mx.clear_cache()
        report["common_prefill_seconds"] = time.perf_counter() - prefill_started

        batch_cache = [item.merge([item, item, item, item]) for item in cache]
        suffix = mx.array(suffix_rows)
        suffix_started = time.perf_counter()
        logits = adapter.model(suffix, cache=batch_cache)
        mx.eval(logits)
        report["suffix_prefill_seconds"] = time.perf_counter() - suffix_started
        next_tokens = mx.argmax(logits[:, -1], axis=-1).astype(mx.int32)
        del logits, cache
        mx.clear_cache()

        events = []
        best = None
        decode_started = time.perf_counter()
        for step in range(args.decode_steps):
            logits = adapter.model(next_tokens[:, None], cache=batch_cache)
            mx.eval(logits)
            next_tokens = mx.argmax(logits[:, -1], axis=-1).astype(mx.int32)
            mx.eval(next_tokens)
            for layer_index, attention in attention_layers:
                selection = attention.indexer._frequency_selection
                if selection.kind != "explicit" or int(selection.length) != 1:
                    continue
                exact = _exact_sets(selection, args.base_tokens // selection.block_size)
                event = {
                    "step": step,
                    "layer": layer_index,
                    "q_position": int(np.asarray(selection.q_positions)[0, -1]),
                    "base_counts": [len(row) for row in exact["base_sets"]],
                    "full_counts": [len(row) for row in exact["full_sets"]],
                    "b2_base_hit": exact["b2_base_hit"],
                    "b4_base_hit": exact["b4_base_hit"],
                    "b2_full_hit": exact["b2_full_hit"],
                    "b4_full_hit": exact["b4_full_hit"],
                    "mean_anchor_jaccard": exact["mean_anchor_jaccard"],
                    "minimum_anchor_jaccard": exact["minimum_anchor_jaccard"],
                }
                events.append(event)
                rank = (int(exact["b4_base_hit"]), event["minimum_anchor_jaccard"])
                if best is None or rank > best[0]:
                    hidden = attention.indexer._frequency_hidden
                    best = (
                        rank,
                        event.copy(),
                        _sample_arrays(
                            attention,
                            hidden,
                            batch_cache[layer_index],
                            selection,
                            args.base_tokens,
                        ),
                    )
            del logits
            mx.clear_cache()
        report["decode_seconds"] = time.perf_counter() - decode_started
        report["decode_tokens"] = args.decode_steps * 4
        report["decode_tokens_per_second"] = report["decode_tokens"] / report["decode_seconds"]

        if not events or best is None:
            raise RuntimeError("no sparse decode selections were observed")
        totals = len(events)
        for field in ("b2_base_hit", "b4_base_hit", "b2_full_hit", "b4_full_hit"):
            hits = sum(int(event[field]) for event in events)
            report[field] = {"hits": hits, "opportunities": totals, "rate": hits / totals}
        report["jaccard"] = {
            "mean": float(np.mean([event["mean_anchor_jaccard"] for event in events])),
            "minimum": float(min(event["minimum_anchor_jaccard"] for event in events)),
            "p50": float(np.percentile([event["mean_anchor_jaccard"] for event in events], 50)),
            "p95": float(np.percentile([event["mean_anchor_jaccard"] for event in events], 95)),
        }
        report["per_layer"] = {}
        for layer_index, _attention in attention_layers:
            rows = [event for event in events if event["layer"] == layer_index]
            report["per_layer"][str(layer_index)] = {
                "opportunities": len(rows),
                "b2_base_hits": sum(int(row["b2_base_hit"]) for row in rows),
                "b4_base_hits": sum(int(row["b4_base_hit"]) for row in rows),
                "mean_anchor_jaccard": float(np.mean([row["mean_anchor_jaccard"] for row in rows])),
            }
        report["events"] = events
        report["sample"] = {**best[1], "path": str(args.sample.resolve())}

        sample = best[2]
        mx.eval(
            sample["q"], sample["base_k"], sample["base_v"], sample["delta_k"],
            sample["delta_v"], sample["index_q"], sample["pooled"],
            sample["selected"], sample["valid_blocks"], sample["q_positions"],
            sample["token_positions"],
        )
        args.sample.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.sample,
            q=_as_float32(sample["q"]),
            base_k=_as_float32(sample["base_k"]),
            base_v=_as_float32(sample["base_v"]),
            delta_k=_as_float32(sample["delta_k"]),
            delta_v=_as_float32(sample["delta_v"]),
            index_q=_as_float32(sample["index_q"]),
            pooled=_as_float32(sample["pooled"]),
            selected=np.asarray(sample["selected"]),
            valid_blocks=np.asarray(sample["valid_blocks"]),
            q_positions=np.asarray(sample["q_positions"]),
            token_positions=np.asarray(sample["token_positions"]),
            scale=np.asarray([sample["scale"]], dtype=np.float32),
            block_size=np.asarray([sample["block_size"]], dtype=np.int32),
            physical_width=np.asarray([sample["physical_width"]], dtype=np.int32),
            n_blocks=np.asarray([sample["n_blocks"]], dtype=np.int32),
            base_tokens=np.asarray([args.base_tokens], dtype=np.int32),
            source_dtype=np.asarray([str(sample["q"].dtype)]),
            layer=np.asarray([best[1]["layer"]], dtype=np.int32),
            step=np.asarray([best[1]["step"]], dtype=np.int32),
        )
        report["sample"]["bytes"] = args.sample.stat().st_size
        report["passed"] = True
    finally:
        if adapter is not None:
            adapter.close()
        report["finished_at"] = time.time()
        report["elapsed_seconds"] = report["finished_at"] - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
