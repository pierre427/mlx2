#!/usr/bin/env python3
"""Probe the production Muse progressive verifier against its fixed route."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

PROMPT = "Explain why a bounded dataflow scheduler should preserve causal state."


def _source_identity():
    root = Path(__file__).resolve().parents[2]
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
            ).strip()
        ),
    }


def _hash_array(value):
    if type(value).__module__.startswith("mlx"):
        import mlx.core as mx

        width = int(value.dtype.size)
        unsigned = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}
        if width not in unsigned:
            raise ValueError(f"unsupported MLX element width {width}")
        array = np.asarray(value.view(unsigned[width]))
        dtype = str(value.dtype)
        shape = list(map(int, value.shape))
    else:
        array = np.asarray(value)
        dtype = str(array.dtype)
        shape = list(array.shape)
    digest = hashlib.sha256()
    digest.update(dtype.encode())
    digest.update(json.dumps(shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _cache_digest(cache):
    digest = hashlib.sha256()
    offsets = []
    for entry in cache:
        offset = int(entry.offset)
        offsets.append(offset)
        keys = entry.keys
        values = entry.values
        if keys is None:
            digest.update(b"empty")
            continue
        temporal = getattr(entry, "_temporal_order", None)
        if callable(temporal):
            keys = temporal(keys)
            values = temporal(values)
        else:
            keys = keys[..., :offset, :]
            values = values[..., :offset, :]
        digest.update(type(entry).__qualname__.encode())
        digest.update(json.dumps(entry.meta_state, default=str).encode())
        digest.update(_hash_array(keys).encode())
        digest.update(_hash_array(values).encode())
    return {"sha256": digest.hexdigest(), "offsets": offsets}


def _response_digest(ready):
    return [
        {
            "token": int(item.token),
            "finish_reason": item.finish_reason,
            "logprobs": (
                None if item.logprobs is None else _hash_array(item.logprobs)
            ),
        }
        for item in ready
    ]


def _greedy_chain(model, prompt, count):
    import mlx.core as mx

    cache = model.make_cache()
    model(mx.array([prompt[:-1]], dtype=mx.int32), cache=cache)
    token = int(prompt[-1])
    result = []
    for _ in range(count):
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        token = int(mx.argmax(logits[0, -1]).item())
        result.append(token)
    return result


def _forced_laws(tokens, vocab):
    laws = []
    for token in tokens:
        law = np.zeros(vocab, dtype=np.float64)
        law[int(token)] = 1.0
        laws.append(law)
    return laws


def _run_arm(adapter, prompt, args, *, tile, forced, reject_at):
    import mlx.core as mx

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    draft = adapter.draft_model
    original = draft.draft_distributions
    captured = {}

    def proposal(anchors, hidden, caches, count, rngs, temps, **kwargs):
        tokens, laws = original(
            anchors, hidden, caches, count, rngs, temps, **kwargs
        )
        if forced is not None:
            row = list(forced[:count])
            if reject_at is not None:
                row[reject_at] = (row[reject_at] + 1) % adapter.model.args.vocab_size
            tokens = [row]
            laws = [_forced_laws(row, adapter.model.args.vocab_size)]
        captured["tokens"] = [int(token) for token in tokens[0]]
        return tokens, laws

    draft.draft_distributions = proposal
    batch = ExternalDraftBatchGenerator(
        adapter.model,
        draft_model=draft,
        binding=adapter.identity["fingerprint"],
        completion_batch_size=1,
        prefill_step_size=args.prefill_step,
        num_draft=args.num_draft,
        stop_tokens=[],
        pairwise_selection="host",
        progressive_verification_tile=tile,
    )
    started = time.perf_counter()
    try:
        uid = batch.insert(
            [prompt],
            max_tokens=[args.num_draft + 2],
            sampling_configs=[{"sampling_temp": 0}],
            lane_rngs=[LaneRNG(args.seed)],
        )[0]
        lane = batch.lanes[uid]
        while lane.anchor is None:
            batch._prefill(lane)
        batch._round([lane])
        mx.eval(lane.tail)
        next_logits = adapter.model(
            mx.array([[lane.anchor]], dtype=mx.int32),
            cache=copy.deepcopy(lane.cache),
        )
        mx.eval(next_logits)
        receipt = (
            None
            if not lane.ready
            else lane.ready[-1].speculative_receipt.get(
                "progressive_verification"
            )
        )
        return {
            "proposal_tokens": captured["tokens"],
            "responses": _response_digest(lane.ready),
            "accepted": int(lane.accepted),
            "proposed": int(lane.proposed),
            "history": list(map(int, lane.history)),
            "anchor": int(lane.anchor),
            "rng": lane.rng.snapshot(),
            "target_cache": _cache_digest(lane.cache),
            "draft_cache": _cache_digest(lane.draft_cache),
            "tail": _hash_array(lane.tail),
            "next_logits": _hash_array(next_logits),
            "progressive_receipt": receipt,
            "scheduler_progressive": {
                key: value
                for key, value in batch.scheduler_stats.items()
                if key.startswith("external_progressive_verify")
            },
            "elapsed_s": time.perf_counter() - started,
        }
    finally:
        draft.draft_distributions = original
        batch.close()


def _compare(fixed, progressive):
    fields = (
        "proposal_tokens",
        "responses",
        "accepted",
        "proposed",
        "history",
        "anchor",
        "rng",
        "target_cache",
        "draft_cache",
        "tail",
        "next_logits",
    )
    equal = {field: fixed[field] == progressive[field] for field in fields}
    return {"fields": equal, "strict_equal": all(equal.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=str(Path.home() / "mlx-models/Muse-Glimmer-30B-mlx-4bit"),
    )
    parser.add_argument(
        "--draft",
        default=str(Path.home() / "mlx-models/Muse-Glimmer-30B-DFlash2"),
    )
    parser.add_argument("--num-draft", type=int, default=15)
    parser.add_argument("--tiles", default="3,4")
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=919)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.i_own_the_gpu:
        raise SystemExit("real Metal execution requires --i-own-the-gpu")

    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter

    adapter = MuseGlimmerAdapter(
        args.target,
        execution_policy={
            "draft_model": args.draft,
            "num_draft": args.num_draft,
            "proposal_composition": False,
            "target_verify_row_exact": True,
        },
    )
    prompt = adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": PROMPT}]}
    )
    forced = _greedy_chain(adapter.model, prompt, args.num_draft)
    tiles = [int(value) for value in args.tiles.split(",")]
    scenarios = [("actual_dflash", None)]
    for tile in tiles:
        scenarios.extend(
            [
                (f"forced_full_tile_{tile}", None),
                (f"forced_reject_before_tile_{tile}", max(0, tile - 1)),
                (f"forced_reject_after_tile_{tile}", tile),
            ]
        )
    results = []
    fixed_actual = _run_arm(
        adapter, prompt, args, tile=None, forced=None, reject_at=None
    )
    fixed_forced = {}
    for tile in tiles:
        for name, reject_at in scenarios:
            if name == "actual_dflash" or not name.endswith(f"_{tile}"):
                continue
            key = "full" if reject_at is None else str(reject_at)
            if key not in fixed_forced:
                fixed_forced[key] = _run_arm(
                    adapter,
                    prompt,
                    args,
                    tile=None,
                    forced=forced,
                    reject_at=reject_at,
                )
            progressive = _run_arm(
                adapter,
                prompt,
                args,
                tile=tile,
                forced=forced,
                reject_at=reject_at,
            )
            results.append(
                {
                    "name": name,
                    "tile": tile,
                    "forced": True,
                    "reject_at": reject_at,
                    "fixed": fixed_forced[key],
                    "progressive": progressive,
                    "comparison": _compare(fixed_forced[key], progressive),
                }
            )
        actual = _run_arm(
            adapter, prompt, args, tile=tile, forced=None, reject_at=None
        )
        results.append(
            {
                "name": f"actual_dflash_tile_{tile}",
                "tile": tile,
                "forced": False,
                "reject_at": None,
                "fixed": fixed_actual,
                "progressive": actual,
                "comparison": _compare(fixed_actual, actual),
            }
        )
    payload = {
        "schema": "mlx2.muse-progressive-serving-probe.v1",
        "source": _source_identity(),
        "target": args.target,
        "draft": args.draft,
        "identity": adapter.identity,
        "prompt_tokens": len(prompt),
        "num_draft": args.num_draft,
        "tiles": tiles,
        "state": {
            "implemented": True,
            "qualified": False,
            "selected_by_probe": True,
            "performance_claim": False,
        },
        "results": results,
    }
    payload["verdict"] = (
        "strict_equal"
        if results and all(row["comparison"]["strict_equal"] for row in results)
        else "counterexample"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["verdict"] == "strict_equal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
