#!/usr/bin/env python3
"""In-model check of the MX lane matmul on one MXFP8/MXFP4 checkpoint.

Loads the artifact through the registry adapter (as serving does) and:

1. ordinary reference: greedy chat decode with the lane off, text retained;
2. installs the lane on every MX projection (explicit ``min_rows_by_format``,
   since MX has no builtin threshold) and records coverage;
3. at each context (real text from the repository docs): 16 one-token stock
   steps are the reference; a 16-row stock forward, a 16-row lane forward and
   16 one-token lane steps are compared to it (top-1 agreement, max
   |log-prob difference}).  Lane passes when it is no worse than stock's own
   multi-row kernel.  Whether the 16-row lane rows equal the one-token lane
   rows bit for bit is recorded, not required: attention and GDN choose their
   own kernels by row count, so that is a whole-route property;
4. forward cost at 1/4/8/16 rows, stock versus lane, alternating.

Writes one JSON receipt naming the checkpoint fingerprint and mlx2 commit.
Not a qualification: one repetition, one artifact, forward level only.
usage: probe_mx_lane_in_model.py <artifact> <out.json> [ctx,ctx] [format]
"""

from __future__ import annotations

import copy
import json
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lane_model_sweep import corpus_text, encode, forward_fn, logprobs, timed_forward  # noqa: E402

PROMPT = "Explain in three sentences why a recurrent state must be rolled back when a speculative token is rejected."


def source_revision() -> dict:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return {"commit": head, "tracked_changes": bool(dirty)}


def chat_ids(adapter) -> list[int]:
    tok = adapter.tokenizer
    template = getattr(tok, "apply_chat_template", None) or tok._tokenizer.apply_chat_template
    ids = template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                   enable_thinking=False, tokenize=True)
    return [int(i) for i in getattr(ids, "input_ids", ids)]


def greedy(call, make_cache, prompt, steps):
    cache = make_cache()
    y = call(mx.array([prompt], dtype=mx.uint32), cache)
    out = []
    for _ in range(steps):
        token = int(mx.argmax(y[0, -1]).item())
        out.append(token)
        y = call(mx.array([[token]], dtype=mx.uint32), cache)
    return out


def main():
    art, out = sys.argv[1], Path(sys.argv[2])
    ctxs = [int(c) for c in (sys.argv[3] if len(sys.argv) > 3 else "2048,8192").split(",")]
    fmt = sys.argv[4] if len(sys.argv) > 4 else "mxfp8"
    rec = {"schema": "mlx2.mx-lane-in-model.v1", "artifact": art, "format": fmt,
           "source_revision": source_revision(), "contexts": {}, "errors": []}
    try:
        from mlx2.adapters.registry import inspect_model, resolve_adapter
        from mlx2.runtime import lane

        resolution = inspect_model(art)
        rec["checkpoint"] = f"{Path(art).name}@{resolution.artifact_fingerprint}"
        rec["family"] = resolution.descriptor.family
        t0 = time.perf_counter()
        adapter = resolve_adapter(art, qualification_mode=True)(art)
        model = adapter.model
        rec["load_seconds"] = round(time.perf_counter() - t0, 1)
        rec["lane_at_load"] = sum(bool(lane.installed(m)) for _, m in model.named_modules())
        mx.set_cache_limit(8 << 30)
        call, make_cache = forward_fn(model)
        # 1. ordinary reference, lane absent
        prompt = chat_ids(adapter)
        reference = greedy(call, make_cache, prompt, 48)
        rec["ordinary_reference"] = {"prompt": PROMPT, "tokens": reference,
                                     "text": adapter.tokenizer.decode(reference)}
        print(json.dumps({"reference": rec["ordinary_reference"]["text"]}), flush=True)
        # 2. lane on every MX projection, at every row count
        rec["install"] = lane.install(model, min_rows=1, max_rows=lane.MAX_ROWS,
                                      min_rows_by_format={fmt: 1})
        lane.set_enabled(True)
        lane_tokens = greedy(call, make_cache, prompt, 48)
        lane.set_enabled(False)
        rec["lane_reference_tokens_equal"] = lane_tokens == reference
        rec["lane_greedy"] = {"tokens": lane_tokens, "text": adapter.tokenizer.decode(lane_tokens)}
        first = next((i for i, (a, b) in enumerate(zip(reference, lane_tokens)) if a != b), None)
        if first is not None:
            # Only the first divergent position, from the shared prefix (later
            # rows describe consequence, not cause).
            window = {}
            for arm, on in (("ordinary", False), ("lane", True)):
                lane.set_enabled(on)
                cache = make_cache()
                y = call(mx.array([prompt + reference[:first]], dtype=mx.uint32), cache)
                lp = logprobs(y[0, -1])
                top = mx.argsort(-lp)[:5].tolist()
                window[arm] = [{"token": int(t), "text": adapter.tokenizer.decode([int(t)]),
                                "logprob": round(float(lp[t].item()), 5)} for t in top]
                del cache
            lane.set_enabled(False)
            sel = {arm: window[arm][0] for arm in window}
            margins = {arm: round(window[arm][0]["logprob"] - window[arm][1]["logprob"], 5) for arm in window}
            rec["first_divergence"] = {"index": first, "window_top5": window, "selected": sel,
                                       "selected_vs_alternate_margin_nats": margins,
                                       "same_unordered_top2": {w["token"] for w in window["ordinary"][:2]}
                                       == {w["token"] for w in window["lane"][:2]}}
            print(json.dumps({"first_divergence": rec["first_divergence"]}), flush=True)
        text_ids = encode(adapter, corpus_text())
    except Exception as exc:  # noqa: BLE001 - one receipt, always
        rec["errors"].append({"stage": "load", "error": repr(exc)[:800], "trace": traceback.format_exc()[-2000:]})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({"load_error": repr(exc)[:400]}), flush=True)
        raise SystemExit(1)
    for ctx in ctxs:
        row = {"tokens": ctx}
        try:
            lane.set_enabled(False)
            cache = make_cache()
            for s in range(0, ctx, 2048):
                y = call(mx.array([text_ids[s:min(s + 2048, ctx)]], dtype=mx.uint32), cache)
                mx.eval(y, [c.state for c in cache if hasattr(c, "state")])
            verify = text_ids[ctx:ctx + 16]
            base = copy.deepcopy(cache)
            arms = {}
            for name, on, serial in (("stock_serial", False, True), ("stock_16row", False, False),
                                     ("lane_16row", True, False), ("lane_serial", True, True)):
                lane.set_enabled(on)
                arm_cache = copy.deepcopy(base)
                if serial:
                    lp = mx.stack([logprobs(timed_forward(call, arm_cache, [t])[1][0, -1]) for t in verify])
                else:
                    lp = logprobs(timed_forward(call, arm_cache, verify)[1][0])
                mx.eval(lp)
                arms[name] = lp
                del arm_cache
            lane.set_enabled(False)
            ref = arms["stock_serial"]
            row["quality"] = {
                name: {"top1_agree_of_16": int(mx.sum(mx.argmax(lp, -1) == mx.argmax(ref, -1)).item()),
                       "max_abs_logprob_diff": round(float(mx.max(mx.abs(lp - ref)).item()), 4)}
                for name, lp in arms.items() if name != "stock_serial"}
            row["lane_16row_equals_lane_serial"] = bool(mx.array_equal(arms["lane_16row"], arms["lane_serial"]).item())
            row["stock_16row_equals_stock_serial"] = bool(mx.array_equal(arms["stock_16row"], ref).item())
            lane_err = row["quality"]["lane_16row"]["max_abs_logprob_diff"]
            stock_err = row["quality"]["stock_16row"]["max_abs_logprob_diff"]
            row["lane_no_worse_than_stock_multirow"] = (
                row["quality"]["lane_16row"]["top1_agree_of_16"] >= row["quality"]["stock_16row"]["top1_agree_of_16"]
                and lane_err <= 2 * stock_err + 1e-3)
            del base, arms
            cost = {}
            for rows in (1, 4, 8, 16):
                samples = {"stock": [], "lane": []}
                for r in range(4):
                    for arm in (("stock", "lane") if r % 2 == 0 else ("lane", "stock")):
                        lane.set_enabled(arm == "lane")
                        ms, _ = timed_forward(call, cache, verify[:rows])
                        if r:
                            samples[arm].append(ms)
                cost[rows] = {k: round(statistics.median(v), 2) for k, v in samples.items()}
            lane.set_enabled(False)
            row["forward_ms"] = cost
            del cache
            mx.clear_cache()
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)[:800]
            row["trace"] = traceback.format_exc()[-2000:]
            lane.set_enabled(False)
            mx.clear_cache()
        rec["contexts"][ctx] = row
        print(json.dumps({"ctx": ctx, **{k: row.get(k) for k in (
            "quality", "lane_16row_equals_lane_serial", "lane_no_worse_than_stock_multirow", "forward_ms", "error")}}),
            flush=True)
    rec["lane_stats"] = {k: v for k, v in lane.stats().items() if v}
    rec["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    failed = rec["errors"] or any(r.get("error") or not r.get("lane_no_worse_than_stock_multirow")
                                  for r in rec["contexts"].values())
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
