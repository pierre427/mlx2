"""Contended prefill slice floor on served Flash-Next: TTFT, neighbour gaps, token identity.

Extends ``measure_fn_prefill_contention.py`` (packed-row lane, 2026-10-01).
Drives ``mlx2.serving.ServingEngine`` in-process as ``mlx2.server`` builds it
for the adapter's native-MTP route with the given execution policy (default:
the served ``num_draft`` 2, adaptive depth off, MTP->ordinary handoff at
``--handoff-width``; ``--handoff-width 0`` turns the handoff off).  Per run,
``D`` decode lanes answer real chat prompts; once each has emitted
``--start-after`` tokens a long prompt (a fixed window of real repository
prose per (size, D), the same in every arm) arrives.

Arms change only the decode-fairness interleave, in the same process,
interleaved per rep (ABBA): ``f<rows>`` sets the contended ``slice_floor``
(``f0`` = today), ``f<rows>s<ms>`` also sets ``stall_target_ms``.  The arm is
applied through ``BatchGenerator._fairness`` (every scheduler read), so every
generator the engine builds sees it.

APCv2 is cleared (``engine.apc.clear()``, documented safe on a live server)
after every run once all its requests have finished, so each arm prefills
the same prompts from zero while keeping the served prefill plan (interior
checkpoints at ``auto`` split a prompt that would otherwise fit one
``prefill_step``; ``skip_writing_prefix_cache`` would remove that plan and
with it the interleave, so it is not used).

Reported per arm: the prompt's TTFT, the decode lanes' delivery gaps inside
the prefill window (p50/p95/max), lane tokens/s in the window, the prompt's
prefill chunk histogram, and token identity of the prompt's answer and every
neighbour's answer against the ``f0`` arm of the same rep (and ``f0`` rep 0
vs rep 1 as the run-to-run control).

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/measure_fn_slice_floor.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --prompt-tokens 8192 --arms f0 f256 f512 f1024 --out floor-8k.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ROUNDS = []  # (perf_counter, {uid: tokens delivered})
TOKENS = {}  # uid -> [token ids]
CHUNKS = {}  # uid -> [prefill widths, in order]
ARM = {"slice_floor": 0, "stall_target_ms": 500.0, "pad_policy": None}
GENERATORS = set()


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def parse_arm(name):
    m = re.fullmatch(r"f(\d+)(?:s(\d+))?(?:@(floor|adaptive|always))?", name)
    if not m:
        raise SystemExit(f"bad arm {name!r}; expected f<rows>[s<ms>][@pad-policy]")
    return {"slice_floor": int(m.group(1)),
            "stall_target_ms": float(m.group(2)) if m.group(2) else 500.0,
            "pad_policy": m.group(3) or "floor"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--decode-lanes", nargs="+", type=int, default=[1, 3, 6])
    ap.add_argument("--max-lanes", type=int, default=16)
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--prompt-max-tokens", type=int, default=16)
    ap.add_argument("--lane-tokens", type=int, default=0,
                    help="neighbour max_tokens (0: sized to outlast the window)")
    ap.add_argument("--start-after", type=int, default=40)
    ap.add_argument("--long-lanes", action="store_true",
                    help="ask neighbours for long answers so none ends inside a long prefill window")
    ap.add_argument("--arms", nargs="+", default=["f0", "f256", "f512", "f1024"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--handoff-width", type=int, default=3)
    ap.add_argument("--corpus-offset", type=int, default=0)
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    arms = {name: parse_arm(name) for name in a.arms}
    if "f0" not in arms:
        raise SystemExit("the f0 (today) arm is the identity reference; include it")
    lane_tokens = a.lane_tokens or (400 if a.prompt_tokens <= 8192 else 1600)

    from mlx2 import server as S
    from mlx2 import serving
    from mlx2.adapters.registry import inspect_model

    policy_in = {"num_draft": 2, "adaptive_mtp_depth": False,
                 "mtp_ordinary_handoff": ({"enabled": True, "max_mtp_width": a.handoff_width}
                                          if a.handoff_width else False)}
    parser = S.build_parser()
    args = parser.parse_args(["--model", a.model, "--max-lanes", str(a.max_lanes), "--native-mtp",
                              "--max-context", str(a.max_context),
                              "--max-inflight", str(2 * a.max_lanes + 2)])
    resolution = inspect_model(args.model)
    selection = S.resolve_route_selection(args, policy_in, resolution)
    policy = S.resolve_execution_policy_defaults(policy_in, selection, resolution, approximate_kv=False,
                                                 max_lanes=args.max_lanes)
    engine = serving.ServingEngine(args.model, **S.serving_engine_kwargs(
        args, policy, native_mtp=selection.native_mtp, route_selection_source=selection.source,
        approximate_kv=S.approximate_kv_mode(args, selection.native_mtp),
        max_request_bytes=S.request_body_limit(args.max_context, args.max_request_bytes)))
    assert engine.ready.wait(1800), engine.error
    if engine.error:
        raise RuntimeError(engine.error)

    import mlx.core as mx
    from bench_fn_batch_verify import TASKS

    from mlx2.runtime import generate as G

    orig_next = G.BatchGenerator.next
    orig_fairness = G.BatchGenerator._fairness

    def hooked(self, *args_, **kwargs):
        GENERATORS.add(id(self))
        out = orig_next(self, *args_, **kwargs)
        responses = out[1] if isinstance(out, tuple) else out
        if responses:
            got = {}
            for r in responses:
                got[r.uid] = got.get(r.uid, 0) + 1
                TOKENS.setdefault(r.uid, []).append(int(r.token))
            ROUNDS.append((time.perf_counter(), got))
        return out

    def fairness(self):
        p = orig_fairness(self)
        p.slice_floor = ARM["slice_floor"]
        p.stall_target_ms = ARM["stall_target_ms"]
        if p.slice_floor:
            p.counters.setdefault("slice_floor_lifts", 0)
        return p

    orig_record = G.BatchGenerator._record_prefill_chunk

    def record(self, uid, width):
        CHUNKS.setdefault(int(uid), []).append(int(width))
        return orig_record(self, uid, width)

    G.BatchGenerator._record_prefill_chunk = record
    G.BatchGenerator.next = hooked
    G.BatchGenerator._fairness = fairness
    print(f"engine ready route={selection.route} native_mtp={selection.native_mtp} policy={policy}",
          flush=True)

    def swapouts():
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
        return int(line.split(":")[1].strip().rstrip("."))

    swap0 = swapouts()
    tok = engine.adapter.tokenizer
    corpus = "\n\n".join((ROOT / "docs" / name).read_text() for name in (
        "SERVING.md", "PROVENANCE.md", "ECOSYSTEM-PROBES-2026-09-30.md"))
    corpus_ids = tok.encode(corpus, add_special_tokens=False)
    doc = (ROOT / "docs" / "QUALIFICATION.md").read_text()[:6000]

    def prompt_for(d):
        # One fixed window per D, identical across arms and reps.
        index = a.decode_lanes.index(d)
        start = a.corpus_offset + index * (a.prompt_tokens + 101)
        if start + a.prompt_tokens > len(corpus_ids):
            raise SystemExit("corpus exhausted")
        text = tok.decode(corpus_ids[start:start + a.prompt_tokens - 64])
        return {"messages": [{"role": "user", "content":
                              f"Summarise the following excerpt in three sentences.\n\n{text}"}],
                "max_tokens": a.prompt_max_tokens, "temperature": 0}

    def lane_request(i):
        task = TASKS[i % len(TASKS)]
        if a.long_lanes:
            task += " Answer at length and in depth, in at least 2,500 words."
        if i % 2:
            task = f"Read this document:\n\n{doc}\n\nThen: {task}"
        return {"messages": [{"role": "user", "content": task}], "max_tokens": lane_tokens,
                "temperature": 0}

    class Lane:
        def __init__(self, request):
            self.t_submit = time.perf_counter()
            self.job = engine.submit(dict(request))
            self.finish = None
            self.error = None
            self.uid = None
            self.thread = threading.Thread(target=self._pump, daemon=True)
            self.thread.start()

        def _pump(self):
            while True:
                event = self.job.events.get(timeout=3600)
                if self.uid is None and getattr(self.job, "uid", None) is not None:
                    self.uid = self.job.uid
                if "error" in event:
                    self.error = event
                    return
                if "finish_reason" in event:
                    self.finish = event
                    return

        def wait(self):
            self.thread.join()
            if self.error:
                raise RuntimeError(self.error)
            return self

        def stamps(self):
            if self.uid is None and getattr(self.job, "uid", None) is not None:
                self.uid = self.job.uid
            return [t for t, got in list(ROUNDS) if self.uid in got]

        def tokens(self):
            self.stamps()
            return list(TOKENS.get(self.uid, []))

    def scheduler_counters():
        status = engine.status() if hasattr(engine, "status") else {}
        sched = status.get("scheduler") or {}
        return {k: v for k, v in sched.items()
                if "handoff" in k or "fairness" in k or "prefill" in k}

    def chunk_receipt(lane):
        widths = CHUNKS.get(lane.uid, [])
        hist = {}
        for w in widths:
            hist[str(w)] = hist.get(str(w), 0) + 1
        return {"widths": hist, "slices": len(widths), "rows": sum(widths)}

    def clear_apc():
        time.sleep(0.5)  # let finished requests publish before clearing
        apc = getattr(engine, "apc", None)
        return apc.clear() if apc is not None else None

    def alone(d):
        del ROUNDS[:]
        TOKENS.clear()
        CHUNKS.clear()
        b = Lane(prompt_for(d)).wait()
        clear_apc()
        return {"ttft_s": round(b.stamps()[0] - b.t_submit, 3), "tokens": b.tokens(),
                "chunks": chunk_receipt(b)}

    def contended(d):
        del ROUNDS[:]
        TOKENS.clear()
        CHUNKS.clear()
        before = scheduler_counters()
        lanes = [Lane(lane_request(i)) for i in range(d)]
        deadline = time.perf_counter() + 900
        while min(len(x.stamps()) for x in lanes) < 3 or min(len(x.tokens()) for x in lanes) < a.start_after:
            if time.perf_counter() > deadline:
                raise SystemExit("decode lanes never started")
            time.sleep(0.01)
        t_sub = time.perf_counter()
        b = Lane(prompt_for(d))
        b.wait()
        t_first = b.stamps()[0]
        for x in lanes:
            x.wait()
        cleared = clear_apc()
        gaps_in, gaps_before, tokens_in = [], [], 0
        finished_early = 0
        for x in lanes:
            st = x.stamps()
            if st and st[-1] < t_first:
                finished_early += 1
            gaps_in += [(q - p) * 1e3 for p, q in zip(st, st[1:]) if t_sub <= q <= t_first + 0.05]
            pre = [s for s in st if s < t_sub][-30:]
            gaps_before += [(q - p) * 1e3 for p, q in zip(pre, pre[1:])]
            tokens_in += sum(got.get(x.uid, 0) for t, got in ROUNDS if t_sub <= t <= t_first)
        after = scheduler_counters()
        window = t_first - t_sub
        return {
            "decode_lanes": d, "ttft_s": round(window, 3),
            "gap_in_window_ms": {"n": len(gaps_in), "p50": pct(gaps_in, .5), "p95": pct(gaps_in, .95),
                                 "max": max(gaps_in) if gaps_in else None},
            "gap_before_ms": {"p50": pct(gaps_before, .5), "p95": pct(gaps_before, .95)},
            "lane_tok_s_in_window": tokens_in / max(window, 1e-9),
            "lanes_finished_before_first_token": finished_early,
            "apc_cleared": cleared,
            "prompt_tokens_out": b.tokens(),
            "prompt_chunks": chunk_receipt(b),
            "lane_tokens_out": [x.tokens() for x in lanes],
            "scheduler_delta": {k: v - before.get(k, 0) for k, v in after.items()
                                if isinstance(v, (int, float)) and v != before.get(k, 0)},
        }

    from mlx2.runtime.models import switch_layers as SL

    def set_arm(name):
        ARM.update(arms[name])
        SL._RHS_PAD_POLICY = ARM["pad_policy"]

    # Warm-up (kernels, allocator, the running-max prefill rate), discarded.
    set_arm("f0")
    alone(a.decode_lanes[0])
    contended(a.decode_lanes[0])
    mx.clear_cache()
    runs = []
    for rep in range(a.reps):
        for d in a.decode_lanes:
            order = list(arms) if rep % 2 == 0 else list(arms)[::-1]
            cell = {"rep": rep, "decode_lanes": d}
            set_arm("f0")
            cell["alone"] = alone(d)
            for name in order:
                set_arm(name)
                c = contended(d)
                cell[name] = c
                g = c["gap_in_window_ms"]
                print(f"rep{rep} D={d} {name}: ttft={c['ttft_s']}s gap p50/p95/max="
                      f"{g['p50'] and round(g['p50'])}/{g['p95'] and round(g['p95'])}/"
                      f"{g['max'] and round(g['max'])} lane tok/s={c['lane_tok_s_in_window']:.1f} "
                      f"chunks={(c['prompt_chunks'] or {}).get('widths')} early={c['lanes_finished_before_first_token']} "
                      f"sched={c['scheduler_delta']}", flush=True)
                grew = swapouts() - swap0
                if grew > a.max_swapout_pages:
                    raise SystemExit(f"aborting: Swapouts grew by {grew} pages")
                mx.clear_cache()
            set_arm("f0")
            print(f"rep{rep} D={d} alone ttft={cell['alone']['ttft_s']}s", flush=True)
            runs.append(cell)

    # Token identity against the f0 arm of the same rep, and f0 across reps.
    identity = {}
    for cell in runs:
        ref = cell["f0"]
        for name in arms:
            c = cell[name]
            key = f"rep{cell['rep']}:D{cell['decode_lanes']}:{name}"
            identity[key] = {
                "prompt_vs_f0": c["prompt_tokens_out"] == ref["prompt_tokens_out"],
                "prompt_vs_alone": c["prompt_tokens_out"] == cell["alone"]["tokens"],
                "neighbours_vs_f0": [x == y for x, y in zip(c["lane_tokens_out"], ref["lane_tokens_out"])],
            }
    for d in a.decode_lanes:
        cells = [c for c in runs if c["decode_lanes"] == d]
        if len(cells) > 1:
            r0, r1 = cells[0]["f0"], cells[1]["f0"]
            identity[f"control:D{d}:f0 rep0 vs rep1"] = {
                "prompt": r0["prompt_tokens_out"] == r1["prompt_tokens_out"],
                "neighbours": [x == y for x, y in zip(r0["lane_tokens_out"], r1["lane_tokens_out"])],
                "alone": cells[0]["alone"]["tokens"] == cells[1]["alone"]["tokens"],
            }
    print("IDENTITY", json.dumps(identity), flush=True)

    summary = {}
    for d in a.decode_lanes:
        cells = [c for c in runs if c["decode_lanes"] == d]
        row = {"alone_ttft_s": statistics.median(c["alone"]["ttft_s"] for c in cells)}
        for name in arms:
            cs = [c[name] for c in cells]
            row[name] = {
                "ttft_s": statistics.median(c["ttft_s"] for c in cs),
                "ttft_spread_s": [min(c["ttft_s"] for c in cs), max(c["ttft_s"] for c in cs)],
                "gap_p50_ms": statistics.median(c["gap_in_window_ms"]["p50"] for c in cs),
                "gap_p95_ms": statistics.median(c["gap_in_window_ms"]["p95"] for c in cs),
                "gap_max_ms": max(c["gap_in_window_ms"]["max"] for c in cs),
                "lane_tok_s": statistics.median(c["lane_tok_s_in_window"] for c in cs),
            }
        summary[str(d)] = row
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)
    json.dump({"model": a.model, "route": selection.route, "policy": policy, "arms": arms,
               "max_lanes": a.max_lanes, "prompt_tokens": a.prompt_tokens, "lane_tokens": lane_tokens,
               "runs": runs, "identity": identity, "summary": summary,
               "swapouts_delta_pages": swapouts() - swap0, "mlx": mx.__version__},
              open(a.out, "w"), indent=1, default=str)
    engine.close()


if __name__ == "__main__":
    main()
