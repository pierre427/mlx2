"""Model-level gate for the NAX sorted MoE gather on a Flash-Next checkpoint.

Arms are ``moe_nax_gather`` modes switched in process: ``off`` (today's
path), ``gather`` (segmented NAX expert gathers) and ``fused`` (gather +
SwiGLU epilogue + row map).  One model per process.

1. Bit identity: an ~8K-token prompt prefilled at 512 / 2048 / 8192-token
   chunks per arm; the trunk hidden of every position and the last-row
   logits compared bitwise with ``off``.  Mechanism counters per arm (an
   arm whose counter is 0 is refused).
2. Prefill throughput: the same prefill per chunk size, arms alternated
   (ABBA-style rotation) after one warm-up each; median tok/s.
3. Greedy generation: a short chat prompt, N greedy tokens per arm.
4. Decode untouched: NAX counters do not move during decode steps, and the
   decode tokens / last decode logits match ``off``.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/ab_moe_nax_gather.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --out ab.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    return int(next(l for l in out.splitlines() if l.startswith("Swapouts")).split(":")[1].strip(" ."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", nargs="+", default=["off", "gather", "fused"])
    ap.add_argument("--chunks", nargs="+", type=int, default=[512, 2048, 8192])
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    import mlx.core as mx

    from mlx2.runtime.models import moe_nax_gather as nax

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    swap0 = swapouts()
    tok = adapter.tokenizer
    text = "\n\n".join((ROOT / p).read_text() for p in
                       ("docs/SERVING.md", "docs/PROVENANCE.md", "docs/QUALIFICATION.md"))
    ids = tok.encode(text, add_special_tokens=False)[: a.prompt_tokens]
    assert len(ids) == a.prompt_tokens, len(ids)
    prompt = ids
    short_ids = tok.encode((ROOT / "docs" / "QUALIFICATION.md").read_text(), add_special_tokens=False)[:900]
    chat = list(adapter.prompt_tokens({"messages": [{"role": "user", "content":
                "Summarise this excerpt in five bullet points.\n\n" + tok.decode(short_ids)}]}))

    out = {"model": a.model, "mlx": mx.__version__, "arms": a.arms, "chunks": a.chunks,
           "prompt_tokens": len(prompt), "chat_tokens": len(chat)}

    def lm(x, cache):
        return model.language_model.model(mx.array([x], mx.uint32), cache)

    def prefill(tokens, chunk, keep_hidden=False):
        cache = model.make_cache()
        hs = []
        pos = 0
        mx.synchronize()
        t0 = time.perf_counter()
        while pos < len(tokens):
            end = min(len(tokens), pos + chunk)
            h = lm(tokens[pos:end], cache)
            logits = model.logits(h[:, -1:, :])
            mx.eval(logits, h) if keep_hidden else mx.eval(logits)
            if keep_hidden:
                hs.append(h)
            pos = end
        mx.synchronize()
        dt = time.perf_counter() - t0
        return dt, logits, hs, cache

    def bits(xs):
        hsh = hashlib.sha256()
        for x in xs:
            hsh.update(memoryview(x.view(mx.uint16) if x.dtype == mx.bfloat16 else x))
        return hsh.hexdigest()

    # 1. bit identity + counters
    ident = {}
    for chunk in a.chunks:
        ref = None
        for arm in a.arms:
            nax.set_mode(arm)
            nax.status(reset=True)
            _, logits, hs, cache = prefill(prompt, chunk, keep_hidden=True)
            st = nax.status()
            del cache
            h_hash = bits(hs)
            l_hash = bits([logits])
            row = {"hidden_sha256": h_hash, "logits_sha256": l_hash,
                   "calls": st["calls"], "fallbacks": st["fallbacks"],
                   "not_candidates": st["not_candidates"]}
            if ref is None:
                ref = (arm, hs, logits)
            else:
                row["hidden_identical"] = all(bool(mx.array_equal(x, y).item()) for x, y in zip(hs, ref[1]))
                row["logits_identical"] = bool(mx.array_equal(logits, ref[2]).item())
                row["logits_max_abs_diff"] = float(mx.max(mx.abs(logits.astype(mx.float32)
                                                                 - ref[2].astype(mx.float32))).item())
                if arm != "off" and sum(st["calls"].values()) == 0:
                    row["refused"] = "mechanism counter is 0"
            ident[f"{chunk}:{arm}"] = row
            print("IDENT", chunk, arm, json.dumps({k: v for k, v in row.items() if "sha" not in k}), flush=True)
            del hs
            mx.clear_cache()
        del ref
        mx.clear_cache()
        if swapouts() - swap0 > 20000:
            raise SystemExit("aborting: swap")
    out["identity"] = ident

    # 3+4. greedy generation and decode untouched
    gen = {}
    for arm in a.arms:
        nax.set_mode(arm)
        nax.status(reset=True)
        _, logits, _, cache = prefill(chat, 2048)
        after_prefill = nax.status()
        nxt = int(mx.argmax(logits[:, -1, :], -1).item())
        toks = []
        mx.synchronize()
        t0 = time.perf_counter()
        last = None
        for _ in range(a.gen):
            toks.append(nxt)
            last = model(mx.array([[nxt]], mx.uint32), cache=cache)[:, -1, :]
            nxt = int(mx.argmax(last, -1).item())
        mx.synchronize()
        dec_s = time.perf_counter() - t0
        after_decode = nax.status()
        gen[arm] = {
            "tokens": toks,
            "prefill_calls": after_prefill["calls"],
            "decode_calls_delta": {k: after_decode["calls"][k] - after_prefill["calls"][k]
                                   for k in after_decode["calls"]},
            "decode_fallbacks_delta": sum(after_decode["fallbacks"].values())
            - sum(after_prefill["fallbacks"].values()),
            "decode_tok_s": round(a.gen / dec_s, 2),
            "last_decode_logits_sha256": bits([last]),
        }
        del cache
        mx.clear_cache()
    for arm in a.arms[1:]:
        gen[arm]["tokens_identical"] = gen[arm]["tokens"] == gen[a.arms[0]]["tokens"]
        gen[arm]["last_decode_logits_identical"] = (
            gen[arm]["last_decode_logits_sha256"] == gen[a.arms[0]]["last_decode_logits_sha256"])
    out["generation"] = gen
    print("GEN", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tokens"} for k, v in gen.items()}),
          flush=True)
    print("GEN text off:", repr(tok.decode(gen[a.arms[0]]["tokens"])[:300]), flush=True)

    # 2. prefill throughput
    if not a.skip_perf:
        perf = {}
        for chunk in a.chunks:
            ts = {arm: [] for arm in a.arms}
            for arm in a.arms:
                nax.set_mode(arm)
                prefill(prompt, chunk)
                mx.clear_cache()
            for rep in range(a.reps):
                order = a.arms if rep % 2 == 0 else a.arms[::-1]
                for arm in order:
                    nax.set_mode(arm)
                    dt, _, _, cache = prefill(prompt, chunk)
                    del cache
                    mx.clear_cache()
                    ts[arm].append(len(prompt) / dt)
            med = {arm: round(statistics.median(v), 1) for arm, v in ts.items()}
            perf[str(chunk)] = {
                "tok_s_median": med,
                "tok_s_all": {arm: [round(t, 1) for t in v] for arm, v in ts.items()},
                "ratio_vs_off": {arm: round(med[arm] / med[a.arms[0]], 4) for arm in a.arms},
            }
            print("PERF", chunk, json.dumps(perf[str(chunk)]), flush=True)
            if swapouts() - swap0 > 20000:
                raise SystemExit("aborting: swap")
        out["perf"] = perf
    nax.set_mode("off")
    out["verified_kernels"] = nax.status()["verified"]
    out["swapouts_delta_pages"] = swapouts() - swap0
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
