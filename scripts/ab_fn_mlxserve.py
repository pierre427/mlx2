"""In-process decode A/B for the ddalcu/mlx-serve ideas on Flash-Next (2026-09-25).

One process, one model load, arms rotated per rep (Latin-square order,
reversed on odd reps), one warm-up pass discarded.  Derived from
bench_decode_levers.py / bench_prefill_inproc.py (triage-20260925).

Arms (all on the served Flash-Next execution config, greedy):
  ord      ordinary route (no self-MTP)
  mtp      native self-MTP, copy drafts off (the Flash-Next default)
  copy     + self_mtp_copy_draft {"enabled": true}  (the Qwen3.8-27B default)
  mlxs     + copy drafts under the mlx-serve #523 policy (match >= 8 tokens,
           7-token spans, 14 when the match runs back >= 32, start wide)
  plestub  native self-MTP with the PLE n-gram lookup of decode/verify rows
           replaced by a cached constant (B1 only): upper bound of what moving
           the PLE gather off the host (#539) can buy per round

Cells: corpus (copy: return/edit code already in the prompt; prose: new text)
x width (B1: prompts in turn; B4: all four at once).  Per cell: decode tok/s
(token-weighted), rounds, tokens per round, copy counters, fused-GDN counters,
tokens for the greedy-identity check.  After the timed reps, a teacher-forced
check scores every B1 divergence from the ``mtp`` arm under an ordinary
forward: the diverging token must be (near-)argmax there.

  GPUQ_SESSION=... gpuq.sh fn-ab PYTHONPATH=src .venv/bin/python \\
      scripts/ab_fn_mlxserve.py --i-own-the-gpu \\
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --out ab.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import hashlib

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ab_copy_mtp import CODE, PROSE  # noqa: E402

_FILE = (ROOT / "src/mlx2/runtime/copy_draft.py").read_text().splitlines()
_BLOCK = "\n".join(_FILE[178:300])
COPY = [
    CODE[0],
    CODE[1],
    f"Here is part of a Python module:\n\n```python\n{_BLOCK}\n```\n\nReturn this code "
    "unchanged except rename the variable `length` to `live_length` everywhere. "
    "Output only the code.",
    f"Here is part of a Python module:\n\n```python\n{_BLOCK}\n```\n\nReturn this code "
    "with a one-line comment added above every `def`. Output only the code.",
]
CORPORA = {"copy": COPY, "prose": PROSE}

MLXSERVE_POLICY = {
    "enabled": True, "max_span": 7, "min_match": 8, "strong_match": 32,
    "strong_max_span": 14, "initial_span": 7,
}
ARM_COPY = {
    "copy": {"enabled": True},
    "mlxs": MLXSERVE_POLICY,
    # Decomposition of the mlx-serve policy:
    "copy7": {"enabled": True, "max_span": 7},  # fused-verify width cap only
    "wide7": {"enabled": True, "max_span": 7, "initial_span": 7},  # + start wide
    "mlxs7": {"enabled": True, "max_span": 7, "min_match": 8, "initial_span": 7},  # + match gate, no strong span
    # Fused GDN verify bound (MAX_VERIFY_STEPS) under the Flash-Next default
    # copy policy: v8 = old bound, v17 = verifier maximum; s16 also lets strong
    # matches copy 16 (verify width 17) once those widths stay fused.
    "v8": MLXSERVE_POLICY,
    "v17": MLXSERVE_POLICY,
    "s16v17": dict(MLXSERVE_POLICY, strong_max_span=16),
}
# Admitted fused GDN verify width per arm; arms not listed use the default 8.
ARM_VERIFY_CAP = {"v17": 17, "s16v17": 17}


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return 0


class SwapGuard:
    """Abort the process when vm_stat Swapouts rise past the phase's limit.

    Load and warm-up (first touch of weights, PLE rows and kernels) get a
    looser limit -- Flash-Next load alone swaps ~1 GB on this 128 GB host --
    then ``strict`` re-baselines for the timed reps.
    """

    def __init__(self, limit_mb):
        self.base = swapouts()
        self.limit = limit_mb * 64
        self.events = []
        self.stop = threading.Event()
        print(f"SWAPGUARD armed base={self.base} limit={limit_mb}MB", flush=True)
        threading.Thread(target=self._run, daemon=True).start()

    def strict(self, limit_mb):
        now = swapouts()
        self.events.append({"phase_end_swapouts_delta_pages": now - self.base})
        self.base, self.limit = now, limit_mb * 64
        print(f"SWAPGUARD strict base={now} limit={limit_mb}MB", flush=True)

    def _run(self):
        while not self.stop.is_set():
            now = swapouts()
            if now - self.base > self.limit:
                print(f"SWAPGUARD abort: swapouts {self.base}->{now}", flush=True)
                os._exit(3)
            self.stop.wait(5)


def fused_counters(model):
    totals = {}
    for _, module in model.named_modules():
        for name in ("fused_gdn_decode_calls", "fused_gdn_decode_fallbacks",
                     "fused_gdn_verify_calls", "fused_gdn_verify_fallbacks"):
            totals[name] = totals.get(name, 0) + int(getattr(module, name, 0) or 0)
        for reason, n in (getattr(module, "fused_gdn_verify_fallback_reasons", None) or {}).items():
            key = f"verify_fallback:{reason}"
            totals[key] = totals.get(key, 0) + int(n)
    return totals


class PleStub:
    """Replace NGramEmbedding lookups of <= 16-row slabs with a cached constant."""

    def __init__(self):
        from mlx2.runtime.models import qwen4_exp as Q

        self.cls = Q.NGramEmbedding
        self.real = self.cls.__call__
        self.consts = {}
        self.stubbed = 0
        self.active = False
        stub = self

        def call(module, input_ids, cache=None, mask=None):
            if stub.active and input_ids.shape[1] <= 16:
                key = (id(module), tuple(input_ids.shape))
                if key not in stub.consts:
                    value = stub.real(module, input_ids, cache, mask)
                    mx.eval(value)
                    stub.consts[key] = value
                stub.stubbed += 1
                return stub.consts[key]
            return stub.real(module, input_ids, cache, mask)

        self.cls.__call__ = call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["ord", "mtp", "copy", "mlxs", "plestub"])
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--warmup-tokens", type=int, default=96)
    ap.add_argument("--cache-limit-gib", type=int, default=4)
    ap.add_argument("--swap-limit-mb", type=int, default=256)
    ap.add_argument("--warm-swap-limit-mb", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=1.5,
                    help="sampled runs only; the Qwen3.8 instruct profile")
    ap.add_argument("--corpora", nargs="+", default=list(CORPORA))
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if a.temperature > 0 and "ord" in a.arms:
        ap.error("the ordinary arm is greedy-only in this harness")
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.sample_utils import LaneRNG

    t_load = time.perf_counter()
    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(a.cache_limit_gib << 30)
    load_s = time.perf_counter() - t_load
    print(f"LOADED {load_s:.0f}s active={mx.get_active_memory()/2**30:.1f}GiB", flush=True)
    time.sleep(30)
    guard = SwapGuard(a.warm_swap_limit_mb)
    stub = PleStub()
    config = adapter.execution_config(max_lanes=max(a.widths), prefill_step=adapter.prefill_step_default())

    def encode(text):
        return list(adapter.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            tokenize=True, enable_thinking=False))

    prompts = {name: [encode(t) for t in CORPORA[name]] for name in a.corpora}
    print("PROMPT_TOKENS", {k: [len(p) for p in v] for k, v in prompts.items()}, flush=True)

    def run_batch(arm, batch, max_tokens):
        mtp = arm != "ord"
        kwargs = {}
        if arm in ARM_COPY:
            kwargs["copy_draft"] = ARM_COPY[arm]
        stub.active = arm == "plestub"
        from mlx2.runtime.models import qwen4_fused_gdn_verify as FV

        FV.set_verify_max_steps(ARM_VERIFY_CAP.get(arm, FV.DEFAULT_VERIFY_STEPS))
        gen = BatchGenerator(
            model, completion_batch_size=len(batch), prefill_batch_size=1,
            prefill_step_size=adapter.prefill_step_default(),
            self_mtp=config if mtp else None, **kwargs)
        insert = {"max_tokens": [max_tokens] * len(batch)}
        if mtp:
            lane_config = {"sampling_temp": a.temperature}
            if a.temperature > 0:
                from mlx2.runtime.sample_utils import make_logits_processors

                lane_config.update(top_p=a.top_p, top_k=a.top_k)
                insert["logits_processors"] = [
                    make_logits_processors(
                        presence_penalty=a.presence_penalty or None,
                        presence_context_size=0, penalty_generation_start=len(p))
                    for p in batch]
            insert.update(lane_rngs=[LaneRNG(1 + i) for i in range(len(batch))],
                          self_mtp_configs=[dict(lane_config)] * len(batch))
        uids = gen.insert(batch, **insert)
        tokens, first, last, receipts, done = {}, {}, {}, {}, set()
        t_all = None
        emitted_after_all = 0
        try:
            while len(done) < len(batch):
                _, responses = gen.next()
                now = time.perf_counter()
                for r in responses:
                    tokens.setdefault(r.uid, []).append(int(r.token))
                    first.setdefault(r.uid, now)
                    last[r.uid] = now
                    if getattr(r, "mtp_receipt", None) is not None:
                        receipts[r.uid] = r.mtp_receipt
                    if r.finish_reason:
                        done.add(r.uid)
                if t_all is None and len(first) == len(batch):
                    t_all = now
                elif t_all is not None:
                    emitted_after_all += len(responses)
            t_end = time.perf_counter()
            stats = dict(gen.scheduler_stats)
        finally:
            gen.close()
            stub.active = False
        if len(batch) == 1:
            u = uids[0]
            n, secs = len(tokens[u]) - 1, last[u] - first[u]
        else:
            n, secs = emitted_after_all, t_end - t_all
        rounds = sum(int(r.get("stats", {}).get("cycles", 0)) for r in receipts.values())
        emitted = sum(int(r.get("stats", {}).get("total_emitted", 0)) for r in receipts.values())
        copy_stats = {k: v for k, v in stats.items() if k.startswith("self_mtp_copy_")}
        return {"tokens": [tokens[u] for u in uids], "n": n, "secs": secs,
                "rounds": rounds, "receipt_emitted": emitted, "copy": copy_stats,
                "copy_receipts": [r.get("copy_draft") for r in receipts.values() if r.get("copy_draft")]}

    def run_cell(arm, corpus, width, max_tokens):
        before = fused_counters(model)
        stubbed0 = stub.stubbed
        texts = prompts[corpus]
        parts = ([run_batch(arm, [p], max_tokens) for p in texts] if width == 1
                 else [run_batch(arm, texts[:width], max_tokens)])
        after = fused_counters(model)
        mx.clear_cache()
        n = sum(p["n"] for p in parts)
        secs = sum(p["secs"] for p in parts)
        rounds = sum(p["rounds"] for p in parts)
        emitted = sum(p["receipt_emitted"] for p in parts)
        copy = {}
        for p in parts:
            for k, v in p["copy"].items():
                copy[k] = copy.get(k, 0) + v
        return {
            "arm": arm, "corpus": corpus, "width": width, "tok_s": n / secs,
            "tokens_decoded": n, "rounds": rounds,
            "tokens_per_round": emitted / rounds if rounds else None,
            "ms_per_round": 1e3 * secs / rounds if rounds and width == 1 else None,
            "copy": copy,
            "fused_gdn_delta": {k: after.get(k, 0) - before.get(k, 0) for k in after
                                if after.get(k, 0) != before.get(k, 0)},
            "ple_stubbed": stub.stubbed - stubbed0,
            "tokens": [t for p in parts for t in p["tokens"]],
            "copy_receipts": [c for p in parts for c in p["copy_receipts"]],
        }

    def cells_for(arm):
        widths = [1] if arm == "plestub" else a.widths
        return [(c, w) for c in a.corpora for w in widths]

    # Warm-up: every arm, every cell shape, short; discarded.
    for arm in a.arms:
        for corpus, width in cells_for(arm):
            run_cell(arm, corpus, width, a.warmup_tokens)
    print("WARM", flush=True)
    guard.strict(a.swap_limit_mb)

    results = []
    k = len(a.arms)
    for rep in range(a.reps):
        order = a.arms[rep % k:] + a.arms[:rep % k]
        if rep % 2:
            order = order[::-1]
        for arm in order:
            for corpus, width in cells_for(arm):
                cell = run_cell(arm, corpus, width, a.max_tokens)
                cell["rep"] = rep
                results.append(cell)
                print(json.dumps({x: y for x, y in cell.items() if x not in ("tokens", "copy_receipts")}), flush=True)

    # Greedy identity at B1 against the mtp arm (rep 0) and teacher-forced check.
    ref = {c["corpus"]: c["tokens"] for c in results if c["arm"] == "mtp" and c["width"] == 1 and c["rep"] == 0}
    ordinary = {c["corpus"]: c["tokens"] for c in results if c["arm"] == "ord" and c["width"] == 1 and c["rep"] == 0}
    divergences = []
    for c in (results if a.temperature == 0 else []):
        if c["width"] != 1 or c["arm"] in ("mtp", "plestub") or c["corpus"] not in ref:
            continue
        for i, (x, y) in enumerate(zip(ref[c["corpus"]], c["tokens"])):
            j = next((t for t, (p, q) in enumerate(zip(x, y)) if p != q), None)
            if j is not None:
                divergences.append({"arm": c["arm"], "rep": c["rep"], "corpus": c["corpus"],
                                    "prompt": i, "at": j, "ref_token": x[j], "arm_token": y[j],
                                    "prefix": x[:j]})
    guard.strict(a.swap_limit_mb)
    checked = {}
    for d in divergences:
        key = (d["corpus"], d["prompt"], d["at"], d["ref_token"], d["arm_token"])
        if key not in checked:
            ids = prompts[d["corpus"]][d["prompt"]] + d["prefix"]
            cache = make_prompt_cache(model)
            x = mx.array(ids, mx.uint32)[None]
            step = 4096
            for s in range(0, x.shape[1] - 1, step):
                model(x[:, s:min(s + step, x.shape[1] - 1)], cache=cache)
                mx.eval([v for _, v in tree_flatten([getattr(c, "state", None) for c in cache])
                         if isinstance(v, mx.array)])
            logits = model(x[:, -1:], cache=cache)[0, -1].astype(mx.float32)
            lp = logits - mx.logsumexp(logits)
            top = int(mx.argmax(lp).item())
            checked[key] = {"lp_ref": lp[d["ref_token"]].item(), "lp_arm": lp[d["arm_token"]].item(),
                            "argmax": top, "lp_max": lp[top].item()}
            del cache
            mx.clear_cache()
        d.update(checked[key])
        d["near_tie"] = d["lp_max"] - d["lp_arm"] <= 0.25
        del d["prefix"]

    summary = {}
    for arm in a.arms:
        for corpus, width in cells_for(arm):
            rows = [c for c in results if (c["arm"], c["corpus"], c["width"]) == (arm, corpus, width)]
            tps = [c["tok_s"] for c in rows]
            summary[f"{arm}/{corpus}/B{width}"] = {
                "tok_s_median": statistics.median(tps), "tok_s_min": min(tps), "tok_s_max": max(tps),
                "tokens_per_round_median": statistics.median([c["tokens_per_round"] for c in rows]) if rows[0]["tokens_per_round"] else None,
                "ms_per_round_median": statistics.median([c["ms_per_round"] for c in rows]) if rows[0]["ms_per_round"] else None,
                "copy_rounds": sum(c["copy"].get("self_mtp_copy_rounds", 0) for c in rows),
                "copy_accepted": sum(c["copy"].get("self_mtp_copy_accepted_tokens", 0) for c in rows),
                "copy_proposed": sum(c["copy"].get("self_mtp_copy_proposed_tokens", 0) for c in rows),
            }
    out = {
        "schema": "mlx2.fn-mlxserve-ab.v1", "model": a.model, "mlx": mx.__version__,
        "load_s": load_s, "config": config, "arms": a.arms, "reps": a.reps,
        "max_tokens": a.max_tokens, "mlxserve_policy": MLXSERVE_POLICY,
        "arm_policies": {k: v for k, v in ARM_COPY.items() if k in a.arms},
        "arm_verify_cap": {k: ARM_VERIFY_CAP.get(k, 8) for k in a.arms},
        "sampling": {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
                     "presence_penalty": a.presence_penalty} if a.temperature > 0 else "greedy",
        "copy_block_sha256": hashlib.sha256(_BLOCK.encode()).hexdigest(),
        "prompt_tokens": {k: [len(p) for p in v] for k, v in prompts.items()},
        "ordinary_tokens_b1": ordinary, "summary": summary,
        "divergences": divergences, "cells": results,
        "peak_gib": mx.get_peak_memory() / 2**30,
        "swap": guard.events,
        "gate": {"divergences": len(divergences),
                 "non_near_tie": sum(1 for d in divergences if not d["near_tie"])},
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"summary": summary, "gate": out["gate"]}, indent=1))


if __name__ == "__main__":
    main()
