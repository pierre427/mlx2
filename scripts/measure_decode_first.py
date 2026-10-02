"""Decode-first publication A/B on served Flash-Next (mlx-vlm #1630 port).

Arms (the live generator's ``decode_first`` policy, read per scheduler round,
so arms alternate in one process and one engine; ``MLX2_DECODE_FIRST`` is a
kill switch only and cannot select an arm):

  off    today: a round's decode tokens are returned after its prefill phase;
  order  decode tokens are returned before the round's prefill phase;
  all    ``order`` plus one prefill token budget shared across the rows that
         prefill in the same round.

Engine and route are built as ``measure_fn_slice_floor.py`` builds them (the
served native-MTP route, ``num_draft`` 2, adaptive depth off, MTP->ordinary
handoff at ``--handoff-width``).  APCv2 is cleared after every run.

Workloads:

  contended  D decode lanes answer real chat prompts; once each has emitted
             ``--start-after`` tokens, one ~8.1K-token prompt arrives
             (``mixed_ab.py``/``measure_fn_slice_floor.py`` shape);
  burst      one lead lane decodes; once it has emitted ``--start-after``
             tokens, ``--burst`` x ~2K-token prompts arrive at once
             (mlx-vlm #1630's "7 x 2K" shape).

Timing is taken where a client sees it: the receive time of each ``delta``
event on the request's event queue (the queue the SSE writer drains),
coalescing events that arrive within 3 ms into one delivery (a self-MTP round
commits several tokens per lane at once).  ``lag`` is the delay from the
decode step that produced a lane's tokens to their client receipt.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/measure_decode_first.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --out decode-first.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DECODED = {}  # uid -> [perf_counter of each decode step that emitted for it]
TOKENS = {}  # uid -> token ids returned by BatchGenerator.next
PREFILL = []  # (start, end, kind) of each self-MTP prefill call
GATE = threading.Event()  # cleared: the worker holds before its next round
GATE.set()
LIVE = {}  # the live generator's scheduler_stats (status snapshots lag)
COALESCE_S = 0.003


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summary(values):
    return {"n": len(values), "p50": pct(values, .5), "p95": pct(values, .95),
            "max": max(values) if values else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--decode-lanes", nargs="+", type=int, default=[1, 3, 6])
    ap.add_argument("--max-lanes", type=int, default=16)
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--prompt-max-tokens", type=int, default=16)
    ap.add_argument("--lane-tokens", type=int, default=400)
    ap.add_argument("--burst", type=int, default=7)
    ap.add_argument("--burst-tokens", type=int, default=2048)
    ap.add_argument("--burst-max-tokens", type=int, default=64)
    ap.add_argument("--start-after", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=["off", "order", "all"])
    ap.add_argument("--workloads", nargs="+", default=["contended", "burst"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--handoff-width", type=int, default=3)
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if a.arms[0] != "off":
        raise SystemExit("the off arm is the identity reference; list it first")
    ARM = {"name": "off"}

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

    def stamp_decode(cls):
        real = cls.next

        def next_(self, *args_, **kwargs):
            out = real(self, *args_, **kwargs)
            if out:
                now = time.perf_counter()
                for r in out:
                    DECODED.setdefault(r.uid, []).append(now)
            return out

        cls.next = next_

    stamp_decode(G.GenerationBatch)
    stamp_decode(G.MTPGenerationBatch)
    real_next = G.BatchGenerator.next

    def batch_next(self, *args_, **kwargs):
        LIVE["stats"] = self.scheduler_stats
        # The arm is the live policy (counters kept); "off" still finishes a
        # pending prefill phase first, as the kill switch does.
        self.decode_first.enabled = ARM["name"] != "off"
        self.decode_first.shared_prefill_budget = ARM["name"] == "all"
        out = real_next(self, *args_, **kwargs)
        for r in out[1]:
            TOKENS.setdefault(r.uid, []).append(int(r.token))
        return out

    G.BatchGenerator.next = batch_next
    real_loop_top = engine._expire_pending_cohorts

    def loop_top(*args_, **kwargs):
        GATE.wait()  # the worker parks at its loop top, before admission
        return real_loop_top(*args_, **kwargs)

    engine._expire_pending_cohorts = loop_top

    for name in ("_advance_mtp_prefill", "_make_mtp_batch"):
        real = getattr(G.BatchGenerator, name)

        def timed(self, *args_, _real=real, _name=name, **kwargs):
            t0 = time.perf_counter()
            try:
                return _real(self, *args_, **kwargs)
            finally:
                PREFILL.append((t0, time.perf_counter(), _name))

        setattr(G.BatchGenerator, name, timed)
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

    def window(start, size, max_tokens):
        if start + size > len(corpus_ids):
            raise SystemExit("corpus exhausted")
        text = tok.decode(corpus_ids[start:start + size - 64])
        return {"messages": [{"role": "user", "content":
                              f"Summarise the following excerpt in three sentences.\n\n{text}"}],
                "max_tokens": max_tokens, "temperature": 0}

    def long_prompt(d):
        index = a.decode_lanes.index(d)
        return window(index * (a.prompt_tokens + 101), a.prompt_tokens, a.prompt_max_tokens)

    def burst_prompts():
        base = len(a.decode_lanes) * (a.prompt_tokens + 101)
        return [window(base + i * (a.burst_tokens + 37), a.burst_tokens, a.burst_max_tokens)
                for i in range(a.burst)]

    def lane_request(i):
        task = TASKS[i % len(TASKS)]
        if i % 2:
            task = f"Read this document:\n\n{doc}\n\nThen: {task}"
        return {"messages": [{"role": "user", "content": task}], "max_tokens": a.lane_tokens,
                "temperature": 0}

    class Lane:
        def __init__(self, request):
            self.t_submit = time.perf_counter()
            self.job = engine.submit(dict(request))
            self.recv = []  # receive time per delta event
            self.finish = None
            self.error = None
            self.thread = threading.Thread(target=self._pump, daemon=True)
            self.thread.start()

        def _pump(self):
            while True:
                event = self.job.events.get(timeout=3600)
                if "delta" in event:
                    self.recv.append(time.perf_counter())
                if "error" in event:
                    self.error = event
                    return
                if "finish_reason" in event:
                    self.finish = event
                    return

        @property
        def uid(self):
            return getattr(self.job, "uid", None)

        def wait(self):
            self.thread.join()
            if self.error:
                raise RuntimeError(self.error)
            return self

        def deliveries(self):
            out = []
            for t in list(self.recv):
                if not out or t - out[-1] > COALESCE_S:
                    out.append(t)
            return out

        def tokens(self):
            return list(TOKENS.get(self.uid, []))

        def lags(self, lo, hi):
            stamps = DECODED.get(self.uid, [])
            out = []
            for t in self.deliveries():
                if not lo <= t <= hi:
                    continue
                prior = [s for s in stamps if s <= t]
                if prior:
                    out.append((t - prior[-1]) * 1e3)
            return out

    def scheduler_counters():
        sched = dict(LIVE.get("stats") or {})
        return {k: v for k, v in sched.items() if isinstance(v, (int, float))
                and ("decode_first" in k or "fairness" in k or "prefill" in k or "handoff" in k)}

    def clear_apc():
        time.sleep(0.5)
        apc = getattr(engine, "apc", None)
        return apc.clear() if apc is not None else None

    def gaps(lane, lo, hi):
        st = lane.deliveries()
        return [(q - p) * 1e3 for p, q in zip(st, st[1:]) if lo <= q <= hi]

    def wait_started(lanes):
        deadline = time.perf_counter() + 900
        while min(len(x.recv) for x in lanes) < a.start_after:
            if time.perf_counter() > deadline:
                raise SystemExit("decode lanes never started")
            time.sleep(0.005)

    def trace(lanes, t0, t1):
        rel = lambda t: round((t - t0) * 1e3, 1)
        return {"deliveries": [[rel(t) for t in x.deliveries() if t0 - 1 <= t <= t1 + 1] for x in lanes],
                "decode_steps": [[rel(t) for t in sorted(set(DECODED.get(x.uid, []))) if t0 - 1 <= t <= t1 + 1]
                                 for x in lanes],
                "prefill_calls": [[rel(a_), rel(b_), k] for a_, b_, k in PREFILL if t0 - 1 <= a_ <= t1 + 1]}

    def contended(d):
        DECODED.clear()
        del PREFILL[:]
        TOKENS.clear()
        before = scheduler_counters()
        lanes = [Lane(lane_request(i)) for i in range(d)]
        wait_started(lanes)
        t_sub = time.perf_counter()
        b = Lane(long_prompt(d)).wait()
        t_first = b.deliveries()[0]
        for x in lanes:
            x.wait()
        clear_apc()
        g, lag, n_in = [], [], 0
        for x in lanes:
            g += gaps(x, t_sub, t_first + 0.05)
            lag += x.lags(t_sub, t_first)
            n_in += sum(1 for t in x.recv if t_sub <= t <= t_first)
        after = scheduler_counters()
        window_s = t_first - t_sub
        return {"ttft_s": round(window_s, 3), "gap_ms": summary(g), "lag_ms": summary(lag),
                "trace": trace(lanes, t_sub, t_first),
                "lane_tok_s": n_in / max(window_s, 1e-9),
                "prompt_tokens_out": b.tokens(), "lane_tokens_out": [x.tokens() for x in lanes],
                "lanes_finished_early": sum(1 for x in lanes if x.recv and x.recv[-1] < t_first),
                "scheduler_delta": {k: v - before.get(k, 0) for k, v in after.items()
                                    if v != before.get(k, 0)}}

    def burst():
        DECODED.clear()
        del PREFILL[:]
        TOKENS.clear()
        before = scheduler_counters()
        lead = Lane(lane_request(0))
        wait_started([lead])
        t_sub = time.perf_counter()
        peers = [Lane(r) for r in burst_prompts()]
        for p in peers:
            p.wait()
        t_end = max(p.deliveries()[-1] for p in peers)
        lead.wait()
        clear_apc()
        ttfts = [p.deliveries()[0] - t_sub for p in peers]
        lead_gaps = gaps(lead, t_sub, t_end)
        peer_gaps = []
        for p in peers:
            peer_gaps += gaps(p, t_sub, t_end)
        all_gaps = lead_gaps + peer_gaps
        tokens = sum(len(p.tokens()) for p in peers) + sum(
            1 for t in DECODED.get(lead.uid, []) if t_sub <= t <= t_end)
        after = scheduler_counters()
        return {"ttft_s": {"p50": pct(ttfts, .5), "max": max(ttfts), "all": ttfts},
                "lead_gap_ms": summary(lead_gaps), "peer_gap_ms": summary(peer_gaps),
                "gap_ms": summary(all_gaps),
                "lag_ms": summary(lead.lags(t_sub, t_end) + sum((p.lags(t_sub, t_end) for p in peers), [])),
                "trace": trace([lead] + peers, t_sub, t_end),
                "window_s": t_end - t_sub, "out_tok_s": tokens / max(t_end - t_sub, 1e-9),
                "lead_tokens_out": lead.tokens(), "peer_tokens_out": [p.tokens() for p in peers],
                "scheduler_delta": {k: v - before.get(k, 0) for k, v in after.items()
                                    if v != before.get(k, 0)}}

    def together():
        """Every request admitted in one worker iteration: no mid-stream
        arrival, so arrival alignment cannot differ between arms and token
        identity isolates the scheduling change itself."""
        DECODED.clear()
        TOKENS.clear()
        del PREFILL[:]
        before = scheduler_counters()
        GATE.clear()
        time.sleep(0.3)  # the idle worker reaches its loop top and parks
        lanes = [Lane(lane_request(i)) for i in range(3)]
        b = Lane(long_prompt(a.decode_lanes[0]))
        peers = [Lane(r) for r in burst_prompts()[:3]]
        time.sleep(0.3)
        t0 = time.perf_counter()
        GATE.set()
        for x in lanes + [b] + peers:
            x.wait()
        t_end = max(x.deliveries()[-1] for x in lanes + [b] + peers)
        clear_apc()
        g = []
        for x in lanes + [b] + peers:
            g += gaps(x, t0, t_end)
        after = scheduler_counters()
        return {"ttft_s": b.deliveries()[0] - t0, "gap_ms": summary(g),
                "lag_ms": summary(sum((x.lags(t0, t_end) for x in lanes + [b] + peers), [])),
                "out_tok_s": sum(len(x.tokens()) for x in lanes + [b] + peers) / max(t_end - t0, 1e-9),
                "lead_tokens_out": b.tokens(),
                "peer_tokens_out": [x.tokens() for x in lanes + peers],
                "scheduler_delta": {k: v - before.get(k, 0) for k, v in after.items()
                                    if v != before.get(k, 0)}}

    def set_arm(name):
        ARM["name"] = name

    def check_swap():
        grew = swapouts() - swap0
        if grew > a.max_swapout_pages:
            raise SystemExit(f"aborting: Swapouts grew by {grew} pages")

    # Warm-up, discarded: every arm once on the smallest contended cell and a burst.
    for name in a.arms:
        set_arm(name)
        if "contended" in a.workloads:
            contended(a.decode_lanes[0])
        if "burst" in a.workloads:
            burst()
        if "together" in a.workloads:
            together()
        mx.clear_cache()
    check_swap()
    print("warm-up done", flush=True)

    runs = []
    for rep in range(a.reps):
        order = list(a.arms) if rep % 2 == 0 else list(a.arms)[::-1]
        cells = ([("contended", d) for d in a.decode_lanes] if "contended" in a.workloads else []) + (
            [("burst", a.burst)] if "burst" in a.workloads else []) + (
            [("together", 7)] if "together" in a.workloads else [])
        for kind, d in cells:
            cell = {"rep": rep, "kind": kind, "d": d, "order": order}
            for name in order:
                set_arm(name)
                c = (contended(d) if kind == "contended" else burst() if kind == "burst"
                     else together())
                cell[name] = c
                g = c["gap_ms"]
                t = c["ttft_s"]["max"] if kind == "burst" else c["ttft_s"]
                tps = c.get("lane_tok_s") or c.get("out_tok_s")
                print(f"rep{rep} {kind} D={d} {name}: ttft={t:.3f}s gap p50/p95/max="
                      f"{g['p50'] and round(g['p50'])}/{g['p95'] and round(g['p95'])}/"
                      f"{g['max'] and round(g['max'])} lag p50/max={c['lag_ms']['p50'] and round(c['lag_ms']['p50'])}/"
                      f"{c['lag_ms']['max'] and round(c['lag_ms']['max'])} tok/s={tps:.1f} "
                      f"sched={ {k: v for k, v in c['scheduler_delta'].items() if 'decode_first' in k} }",
                      flush=True)
                check_swap()
                mx.clear_cache()
            runs.append(cell)
    set_arm("off")

    identity = {}
    for cell in runs:
        ref = cell["off"]
        for name in a.arms:
            c = cell[name]
            key = f"rep{cell['rep']}:{cell['kind']}{cell['d']}:{name}"
            if cell["kind"] == "contended":
                identity[key] = {"prompt": c["prompt_tokens_out"] == ref["prompt_tokens_out"],
                                 "lanes": [x == y for x, y in zip(c["lane_tokens_out"], ref["lane_tokens_out"])]}
            else:
                identity[key] = {"lead": c["lead_tokens_out"] == ref["lead_tokens_out"],
                                 "peers": [x == y for x, y in zip(c["peer_tokens_out"], ref["peer_tokens_out"])]}
    for kind, d in {(c["kind"], c["d"]) for c in runs}:
        cs = [c for c in runs if c["kind"] == kind and c["d"] == d]
        if len(cs) > 1:
            r0, r1 = cs[0]["off"], cs[1]["off"]
            outs = ("prompt_tokens_out", "lane_tokens_out") if kind == "contended" else (
                "lead_tokens_out", "peer_tokens_out")
            identity[f"control:{kind}{d}:off rep0 vs rep1"] = all(r0[k] == r1[k] for k in outs)
    print("IDENTITY", json.dumps(identity), flush=True)

    table = {}
    for kind, d in sorted({(c["kind"], c["d"]) for c in runs}):
        cs = [c for c in runs if c["kind"] == kind and c["d"] == d]
        row = {}
        for name in a.arms:
            xs = [c[name] for c in cs]
            row[name] = {
                "ttft_s": statistics.median(x["ttft_s"]["max"] if kind == "burst" else x["ttft_s"]
                                            for x in xs),
                "gap_p50_ms": statistics.median(x["gap_ms"]["p50"] for x in xs),
                "gap_p95_ms": statistics.median(x["gap_ms"]["p95"] for x in xs),
                "gap_max_ms": max(x["gap_ms"]["max"] for x in xs),
                "lag_p50_ms": statistics.median(x["lag_ms"]["p50"] for x in xs),
                "lag_max_ms": max(x["lag_ms"]["max"] for x in xs),
                "tok_s": statistics.median((x.get("lane_tok_s") or x.get("out_tok_s")) for x in xs),
            }
            if kind == "burst":
                row[name]["ttft_p50_s"] = statistics.median(x["ttft_s"]["p50"] for x in xs)
                row[name]["lead_gap_max_ms"] = max(x["lead_gap_ms"]["max"] for x in xs)
        table[f"{kind}:{d}"] = row
    print("SUMMARY", json.dumps(table, indent=1), flush=True)
    json.dump({"model": a.model, "route": selection.route, "policy": policy, "arms": a.arms,
               "args": vars(a), "runs": runs, "identity": identity, "summary": table,
               "swapouts_delta_pages": swapouts() - swap0, "mlx": mx.__version__},
              open(a.out, "w"), indent=1, default=str)
    engine.close()


if __name__ == "__main__":
    main()
