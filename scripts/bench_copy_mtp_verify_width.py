#!/usr/bin/env python3
"""Self-MTP verify-width cost curve for copy-draft sizing (rm01).

Measures, on one attached self-MTP lane at a real context length, the target
verify forward (trunk + LM head) at M = 1..--max-rows rows (B=1), and the
MTP head's single draft step.  Rows are rolled back after every sample with
the same ragged trim the verify transaction uses, so the context never grows.

The output sets the copy-draft cost model: ``verify_row_cost`` =
(t(M) - t(1)) / (M - 1) / t(1) over the region the sizer will use, and
``draft_step_cost`` = t(mtp_step) / t(1).  It also shows any tile edge where
``max_span`` should stop.

Full-model weights are far larger than the SLC, so every forward is
naturally weight-cache-cold; widths are visited in a rotated order per rep to
avoid systematic warm-up bias.

GPU:  --model <artifact> --i-own-the-gpu
CPU mechanics check (tiny random model, no GPU): --tiny --cpu
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _tiny_model():
    import mlx.core as mx
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=1 << 20,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model, 128


def _real_model(path):
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(path, mtp=True)(path)
    return adapter.model, int(getattr(adapter.tokenizer, "vocab_size", 32000))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=33)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"model": args.model or "tiny", "context": args.context,
                          "rows": list(range(1, args.max_rows + 1)), "reps": args.reps}))
        return 0
    if not args.cpu and not args.i_own_the_gpu:
        parser.error("Metal run: pass --i-own-the-gpu under the GPU lock (or --tiny --cpu)")
    if args.cpu and not args.tiny:
        parser.error("--cpu is for the tiny mechanics check only")

    import mlx.core as mx

    if args.cpu:
        mx.set_default_device(mx.cpu)
    from mlx2.runtime.hybrid_speculative import (
        _finalize_self_mtp_cache_group,
        _mtp_backbone,
        _prepare_self_mtp_cache_group,
        _trim_self_mtp_cache_group,
        attach_self_mtp_lanes,
        prepare_self_mtp_lane,
    )
    from mlx2.runtime.sample_utils import LaneRNG

    model, vocab = _tiny_model() if args.tiny else _real_model(args.model)
    prompt = mx.random.randint(0, vocab, (args.context,)).astype(mx.uint32)
    detached, _first = prepare_self_mtp_lane(
        prompt, model, uid=1, max_tokens=1 << 20, prompt_cache=None,
        mtp_state=None, lane_rng=LaneRNG(1), num_draft=2, sampling_temp=0.0,
        sampling_top_p=1.0, sampling_top_k=0, sampling_min_p=0.0,
        accept_rule="residual", logits_processors=[], prefill_step_size=2048,
        share_qsa_indices=False,
    )
    batch = attach_self_mtp_lanes(model, None, [detached])
    target = batch.caches.target
    lane = batch.lanes[0]

    def verify(rows):
        ids = mx.random.randint(0, vocab, (1, rows)).astype(mx.uint32)
        mx.eval(ids)
        start = time.perf_counter()
        _prepare_self_mtp_cache_group(target, (rows,), (0,))
        try:
            hidden, _seed = _mtp_backbone(model, ids, target)
            logits = model.logits(hidden)
        finally:
            _finalize_self_mtp_cache_group(target)
        mx.eval(logits)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        _trim_self_mtp_cache_group(target, [rows], validate=False)
        return elapsed * 1000.0

    def draft_step():
        draft = batch.caches.draft
        tokens = mx.array([[lane.cur]], mx.uint32)
        start = time.perf_counter()
        _prepare_self_mtp_cache_group(draft, (1,), (0,))
        try:
            d_logits, post = model.mtp_step(lane.seed_h, tokens, draft)
        finally:
            _finalize_self_mtp_cache_group(draft)
        mx.eval(d_logits, post)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        _trim_self_mtp_cache_group(draft, [1], validate=False)
        return elapsed * 1000.0

    widths = list(range(1, args.max_rows + 1))
    for rows in (1, 2, 4):
        verify(rows)  # warm kernels
    draft_step()
    samples = {rows: [] for rows in widths}
    drafts = []
    for rep in range(args.reps):
        shift = rep % len(widths)
        for rows in widths[shift:] + widths[:shift]:
            samples[rows].append(verify(rows))
        drafts.append(draft_step())
    curve = {rows: statistics.median(values) for rows, values in samples.items()}
    base = curve[1]
    draft_ms = statistics.median(drafts)
    per_row = {
        rows: (curve[rows] - base) / (rows - 1) / base for rows in widths if rows > 1
    }
    result = {
        "schema": "mlx2.copy-mtp-verify-width.v1",
        "model": args.model or "tiny",
        "device": "cpu" if args.cpu else "gpu",
        "context": args.context,
        "verify_ms_median": curve,
        "verify_ms_samples": samples,
        "draft_step_ms_median": draft_ms,
        "suggested_cost_model": {
            "draft_step_cost": draft_ms / base,
            "verify_row_cost_by_width": per_row,
            "verify_row_cost_to_9": per_row.get(min(9, args.max_rows)),
        },
        "mechanism_counters": {"verify_samples": sum(len(v) for v in samples.values()),
                               "draft_samples": len(drafts)},
    }
    if result["mechanism_counters"]["verify_samples"] == 0:
        raise RuntimeError("no verify sample ran")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({"verify_ms_median": curve, "draft_step_ms": draft_ms,
                      "suggested": result["suggested_cost_model"]["verify_row_cost_to_9"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
