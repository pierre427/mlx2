#!/usr/bin/env python3
"""Bounded in-process Ling-3.2-air probes; sanity/attribution, never qualification.

One process, one resident artifact (run under the caller's GPU lease):

1. ordinary reference: greedy decode of fixed chat prompts (text + token ids);
2. router row law: fp32 router logits/top-k for each row decoded alone vs the
   same rows in one 2..16-row call (the AST verify shape);
3. SwiGLU clamp effect: last-position logits with the vendor clamps vs with
   the clamps removed (top-1 agreement, max |delta logit|) — shows the clamp
   is live, not that either arm is "right";
4. MTP depth-1 agreement: the NEXTN head's argmax vs the trunk's next-next
   greedy token over the reference continuation;
5. NAX gather vs stock sorted gather on one prompt chunk: bitwise logits and
   wall time (moe_nax_gather.set_mode in-process; receipted as a probe).

Writes ``<out>/probe.json`` atomically; refuses an existing nonempty ``<out>``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

PROMPTS = (
    "Explain in two sentences why the sky looks blue during the day.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes?",
)


def atomic_json(path: Path, value) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def greedy(model, tokens, steps, mx):
    cache = model.make_cache()
    logits = model(mx.array([tokens]), cache)
    out = []
    for _ in range(steps):
        token = int(mx.argmax(logits[0, -1]).item())
        out.append(token)
        logits = model(mx.array([[token]]), cache)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=48)
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is nonempty; refuse")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.adapters.ling32_air import Ling32AirAdapter
    from mlx2.runtime.models import bailing_hybrid as bh
    from mlx2.runtime.models import moe_nax_gather

    started = time.time()
    adapter = Ling32AirAdapter(str(args.model))
    model = adapter.model
    receipt = {"checkpoint": f"{args.model.name}@sha256:{adapter.identity['fingerprint']}",
               "load_seconds": round(time.time() - started, 1), "probes": {}}
    try:
        # 1. ordinary reference
        refs = []
        for prompt in PROMPTS:
            request = {"messages": [{"role": "user", "content": prompt}], "enable_thinking": False}
            tokens = list(adapter.prompt_tokens(request))
            t0 = time.time()
            out = greedy(model, tokens, args.decode_tokens, mx)
            refs.append({"prompt": prompt, "prompt_tokens": len(tokens), "tokens": out,
                         "text": adapter.tokenizer.decode(out),
                         "seconds": round(time.time() - t0, 2)})
        receipt["probes"]["ordinary_reference"] = refs

        # 2. router row law (layer 20 router, real hidden rows from prompt 0)
        request = {"messages": [{"role": "user", "content": PROMPTS[0]}], "enable_thinking": False}
        tokens = list(adapter.prompt_tokens(request)) + refs[0]["tokens"]
        hidden = model.model.word_embeddings(mx.array([tokens[:16]]))
        gate = model.layers[20].mlp.gate
        rows = []
        for width in (2, 4, 8, 16):
            block = hidden[:, :width]
            together_logits = block.astype(mx.float32) @ gate.gate_proj.weight.astype(mx.float32).T
            alone = mx.concatenate([
                hidden[:, i:i + 1].astype(mx.float32) @ gate.gate_proj.weight.astype(mx.float32).T
                for i in range(width)], axis=1)
            idx_t, _ = gate(block)
            idx_a = mx.concatenate([gate(hidden[:, i:i + 1])[0] for i in range(width)], axis=1)
            rows.append({
                "width": width,
                "logits_bitwise_equal": bool(mx.array_equal(together_logits, alone).item()),
                "max_abs_logit_delta": float(mx.max(mx.abs(together_logits - alone)).item()),
                "expert_sets_equal": bool(mx.array_equal(mx.sort(idx_t, axis=-1),
                                                         mx.sort(idx_a, axis=-1)).item()),
            })
        receipt["probes"]["router_row_law"] = rows

        # 3. clamp effect on last-position logits of each reference continuation
        def last_logits():
            outs = []
            for ref in refs:
                req = {"messages": [{"role": "user", "content": ref["prompt"]}], "enable_thinking": False}
                seq = list(adapter.prompt_tokens(req)) + ref["tokens"]
                outs.append(model(mx.array([seq]))[0].astype(mx.float32))
            return outs

        clamped = last_logits()
        saved = []
        for layer in list(model.layers):
            mlp = layer.mlp
            if isinstance(mlp, bh.SparseMoeBlock):
                saved.append((mlp.switch_mlp, mlp.switch_mlp.activation, mlp.shared_experts,
                              mlp.shared_experts.swiglu_limit))
                mlp.switch_mlp.activation = bh.SwiGLU()
                mlp.shared_experts.swiglu_limit = None
        try:
            unclamped = last_logits()
        finally:
            for switch, activation, shared, limit in saved:
                switch.activation = activation
                shared.swiglu_limit = limit
        effect = []
        for a, b in zip(clamped, unclamped):
            effect.append({
                "positions": int(a.shape[0]),
                "top1_agreement": float((mx.argmax(a, -1) == mx.argmax(b, -1)).astype(mx.float32).mean().item()),
                "max_abs_logit_delta": float(mx.max(mx.abs(a - b)).item()),
            })
        receipt["probes"]["swiglu_clamp_effect"] = effect

        # 4. MTP depth-1 agreement over each reference continuation
        mtp_rows = []
        for ref in refs:
            req = {"messages": [{"role": "user", "content": ref["prompt"]}], "enable_thinking": False}
            seq = list(adapter.prompt_tokens(req)) + ref["tokens"]
            arr = mx.array([seq])
            post = model.model(arr)
            greedy_next = mx.argmax(model.logits(post), -1)  # token at p+1 per position p
            mtp_logits, _ = model.mtp_step(post[:, :-1], arr[:, 1:], model.make_mtp_cache())
            draft = mx.argmax(mtp_logits, -1)  # predicts token p+2
            start = len(seq) - len(ref["tokens"]) - 1
            hits = draft[0, start:-1] == greedy_next[0, start + 1:-1]
            mtp_rows.append({"positions": int(hits.size),
                             "depth1_agreement": float(hits.astype(mx.float32).mean().item())})
        receipt["probes"]["mtp_depth1_agreement"] = mtp_rows

        # 5. NAX gather vs stock on one prompt chunk
        filler = adapter.tokenizer.encode(
            " ".join(f"Item {i}: the quick brown fox number {i * 7 % 97} jumps." for i in range(400)),
            add_special_tokens=False)[: args.prefill_tokens]
        chunk = mx.array([filler])
        nax = {"host_supported": bool(moe_nax_gather.nax_host()), "tokens": len(filler)}
        if nax["host_supported"]:
            def timed():
                mx.eval(model(chunk, model.make_cache()))  # warm
                t0 = time.time()
                value = model(chunk, model.make_cache())
                mx.eval(value)
                return value, time.time() - t0

            stock, stock_s = timed()
            old = moe_nax_gather.set_mode("gather")
            try:
                gathered, nax_s = timed()
                status = moe_nax_gather.status()
            finally:
                moe_nax_gather.set_mode(old)
            nax.update({"bitwise_equal": bool(mx.array_equal(stock, gathered).item()),
                        "stock_seconds": round(stock_s, 3), "nax_seconds": round(nax_s, 3),
                        "status": status})
        receipt["probes"]["nax_gather_prefill"] = nax
        receipt["peak_memory_gib"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        receipt["seconds"] = round(time.time() - started, 1)
        atomic_json(args.out / "probe.json", receipt)
        adapter.close()
    print(json.dumps({k: v for k, v in receipt.items() if k != "probes"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
