#!/usr/bin/env python
"""Probe: MLX allocator cache growth during a long single-request self-MTP decode.

Ollama v0.34.2 saw freed KV buffers pile up in the MLX buffer pool during
speculative decode (98K ctx: >90 GB vs flat 30 GB) and fixed it with a
clear_cache every 256 tokens.  mlx2's BatchGenerator clears every
ALLOCATOR_RECLAIM_STEP_INTERVAL (=512) *steps*; on the self-MTP route a step
is one MTP cycle (1..K+1 tokens), so the effective token interval scales with
acceptance.  This probe measures it.

Samples mx.get_active_memory / get_cache_memory / get_peak_memory every
--sample-every emitted tokens and writes JSONL.

Arms (pick one per run, compare runs):
  --reclaim-steps 512      shipped behaviour (default)
  --reclaim-steps 0        disable the periodic clear (worst case / Ollama-before)
  --clear-every-tokens 256 additionally clear every N emitted tokens (Ollama fix)

GPU (real model, e.g. Qwen3.8-27B or Flash-Next, 64K+ prompt):
  PYTHONPATH=src .venv/bin/python scripts/probe_mtp_cache_memory.py --i-own-the-gpu \
      --model /path/to/artifact --context 65536 --gen 4096 --num-draft 3 \
      --reclaim-steps 512 --out /tmp/mtp_mem_512.jsonl
  ... --reclaim-steps 0 --out /tmp/mtp_mem_off.jsonl
  ... --reclaim-steps 0 --clear-every-tokens 256 --out /tmp/mtp_mem_tok256.jsonl

CPU dry run (tiny random hybrid GDN + MTP; the CPU allocator also pools
freed buffers, so the growth pattern is visible, only smaller):
  PYTHONPATH=src .venv/bin/python scripts/probe_mtp_cache_memory.py --tiny --cpu \
      --context 3000 --gen 1500 --reclaim-steps 0
"""

import argparse
import json
import sys
import time

import mlx.core as mx


def tiny_model():
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


def real_model(path, num_draft):
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(path, mtp=True)(path)
    vocab = int(getattr(adapter.tokenizer, "vocab_size", 32000))
    return adapter.model, vocab, adapter.tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--i-own-the-gpu", action="store_true",
                    help="required for any Metal run")
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--gen", type=int, default=4096)
    ap.add_argument("--num-draft", type=int, default=3)
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--sample-every", type=int, default=64)
    ap.add_argument("--reclaim-steps", type=int, default=512,
                    help="BatchGenerator periodic clear interval in steps; 0 disables")
    ap.add_argument("--clear-every-tokens", type=int, default=0,
                    help="extra mx.clear_cache every N emitted tokens (Ollama-style)")
    ap.add_argument("--cache-limit-gib", type=float, default=None,
                    help="mx.set_cache_limit before the run (alternative fix)")
    ap.add_argument("--prompt-file", default=None,
                    help="real text, tokenized with the model tokenizer and cycled to --context")
    ap.add_argument("--out", default="-")
    a = ap.parse_args()

    if a.cpu:
        mx.set_default_device(mx.cpu)
    elif not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu (or pass --cpu)")
    from mlx2.runtime import generate as G
    from mlx2.runtime.sample_utils import LaneRNG

    G.ALLOCATOR_RECLAIM_STEP_INTERVAL = a.reclaim_steps if a.reclaim_steps > 0 else (1 << 62)
    clear_calls = {"n": 0}
    real_clear = mx.clear_cache

    def counted_clear():
        clear_calls["n"] += 1
        real_clear()

    mx.clear_cache = counted_clear  # counts every clear, incl. prefill per-chunk ones
    if a.cache_limit_gib is not None:
        mx.set_cache_limit(int(a.cache_limit_gib * (1 << 30)))

    tokenizer = None
    if a.tiny:
        model, vocab = tiny_model()
    else:
        model, vocab, tokenizer = real_model(a.model, a.num_draft)
    # Deterministic, low-entropy prompt; real runs should use real text for
    # realistic acceptance (acceptance sets tokens/cycle, hence the effective
    # token interval of the step-based reclaim).
    if a.prompt_file:
        if tokenizer is None:
            ap.error("--prompt-file needs a real --model")
        with open(a.prompt_file) as handle:
            ids = list(tokenizer.encode(handle.read()))
        prompt = (ids * (a.context // max(1, len(ids)) + 1))[: a.context]
    else:
        prompt = [(i * 7919 + 13) % (vocab - 3) + 2 for i in range(a.context)]
    gen = G.BatchGenerator(
        model, completion_batch_size=1, prefill_batch_size=1,
        prefill_step_size=a.prefill_step,
        self_mtp={"num_draft": a.num_draft, "persistent": True, "rate_gate": False,
                  "prefill_step_size": a.prefill_step},
    )
    out = sys.stdout if a.out == "-" else open(a.out, "w")
    GiB = float(1 << 30)

    def sample(tag, emitted, t0):
        rec = {
            "tag": tag, "emitted": emitted, "t": round(time.perf_counter() - t0, 3),
            "active_gib": mx.get_active_memory() / GiB,
            "cache_gib": mx.get_cache_memory() / GiB,
            "peak_gib": mx.get_peak_memory() / GiB,
            "steps": int(getattr(gen, "_steps_counter", 0)),
            "clear_calls": clear_calls["n"],
            "reclaim_steps": a.reclaim_steps,
            "clear_every_tokens": a.clear_every_tokens,
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        return rec

    t0 = time.perf_counter()
    mx.reset_peak_memory()
    gen.insert([prompt], max_tokens=[a.gen], lane_rngs=[LaneRNG(1)],
               self_mtp_configs=[{"sampling_temp": 0.0}])
    emitted = 0
    next_sample = 0
    next_clear = a.clear_every_tokens or None
    first = True
    try:
        while emitted < a.gen:
            _prompts, responses = gen.next()
            if first and responses:
                sample("prefill_done", emitted, t0)
                first = False
            emitted += len(responses)
            if next_clear is not None and emitted >= next_clear:
                mx.clear_cache()
                next_clear += a.clear_every_tokens
            if emitted >= next_sample:
                sample("decode", emitted, t0)
                next_sample += a.sample_every
            if any(r.finish_reason for r in responses):
                break
        last = sample("final", emitted, t0)
        cycles = max(1, last["steps"])
        print(json.dumps({"summary": True, "emitted": emitted, "steps": last["steps"],
                          "tokens_per_step": emitted / cycles,
                          "effective_reclaim_tokens": (emitted / cycles) * a.reclaim_steps
                          if a.reclaim_steps > 0 else None,
                          "peak_gib": last["peak_gib"], "final_cache_gib": last["cache_gib"]}),
              file=sys.stderr)
    finally:
        gen.close()
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
