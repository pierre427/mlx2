"""Prefill beside decode on the served Flash-Next route: TTFT and neighbour gaps.

Drives ``mlx2.serving.ServingEngine`` in-process exactly as ``mlx2.server``
builds it for the adapter's default route (native MTP, the default
MTP->ordinary handoff at width 4, the default decode-fairness interleave).
Per run, ``D`` decode lanes answer real chat prompts (every other one
prefixed with a repository document); once each has emitted ``--start-after``
tokens a fresh long prompt (a distinct window of ``docs/SERVING.md`` /
``docs/ECOSYSTEM-PROBES-2026-09-30.md`` / ``docs/PROVENANCE.md``, real prose, never a cached prefix)
arrives.  Reported: the new prompt's TTFT, the same-size fresh prompt's TTFT
alone (no neighbours, interleaved), the decode lanes' round gaps inside the
prefill window and before it, and the route each lane ran (MTP or handed off).

Round timing comes from wrapping ``BatchGenerator.next``: every call that
returns tokens for a uid is one delivery round for that uid.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/measure_fn_prefill_contention.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --out contention.json
"""

from __future__ import annotations

import argparse
import json
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


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--decode-lanes", nargs="+", type=int, default=[1, 3, 6])
    ap.add_argument("--max-lanes", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--lane-tokens", type=int, default=500)
    ap.add_argument("--start-after", type=int, default=40)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2 import server as S
    from mlx2 import serving
    from mlx2.adapters.registry import inspect_model

    parser = S.build_parser()
    args = parser.parse_args(["--model", a.model, "--max-lanes", str(a.max_lanes),
                              "--max-context", "32768", "--max-inflight", str(2 * a.max_lanes + 2)])
    resolution = inspect_model(args.model)
    selection = S.resolve_route_selection(args, None, resolution)
    policy = S.resolve_execution_policy_defaults(None, selection, resolution, approximate_kv=False,
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

    def hooked(self, *args_, **kwargs):
        out = orig_next(self, *args_, **kwargs)
        responses = out[1] if isinstance(out, tuple) else out
        if responses:
            got = {}
            for r in responses:
                got[r.uid] = got.get(r.uid, 0) + 1
            ROUNDS.append((time.perf_counter(), got))
        return out

    G.BatchGenerator.next = hooked
    print(f"engine ready route={selection.route} native_mtp={selection.native_mtp}", flush=True)

    def swapouts():
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
        return int(line.split(":")[1].strip().rstrip("."))

    swap0 = swapouts()
    tok = engine.adapter.tokenizer
    corpus = "\n\n".join((ROOT / "docs" / name).read_text() for name in (
        "SERVING.md", "ECOSYSTEM-PROBES-2026-09-30.md", "PROVENANCE.md"))
    corpus_ids = tok.encode(corpus, add_special_tokens=False)
    doc = (ROOT / "docs" / "QUALIFICATION.md").read_text()[:6000]
    cursor = {"at": 0}

    def fresh_prompt():
        start = cursor["at"]
        cursor["at"] += a.prompt_tokens + 101
        if cursor["at"] > len(corpus_ids):
            raise SystemExit("corpus exhausted")
        text = tok.decode(corpus_ids[start:start + a.prompt_tokens - 64])
        return {"messages": [{"role": "user", "content":
                              f"Summarise the following excerpt in three sentences.\n\n{text}"}],
                "max_tokens": 8, "temperature": 0}

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
            self.finish = None
            self.error = None
            self.uid = None
            self.thread = threading.Thread(target=self._pump, daemon=True)
            self.thread.start()

        def _pump(self):
            while True:
                event = self.job.events.get(timeout=1800)
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
            uid = self.uid
            return [t for t, got in list(ROUNDS) if uid in got]

        def tokens(self):
            self.stamps()
            uid = self.uid
            return sum(got.get(uid, 0) for _, got in list(ROUNDS))

    def scheduler_counters():
        status = engine.status() if hasattr(engine, "status") else {}
        sched = status.get("scheduler") or {}
        return {k: v for k, v in sched.items() if "handoff" in k or "prefill_rounds" in k}

    def alone():
        del ROUNDS[:]
        b = Lane(fresh_prompt())
        b.wait()
        return round(b.stamps()[0] - b.t_submit, 3)

    def contended(d):
        del ROUNDS[:]
        before = scheduler_counters()
        lanes = [Lane(lane_request(i)) for i in range(d)]
        deadline = time.perf_counter() + 600
        while min(len(x.stamps()) for x in lanes) < 3 or min(x.tokens() for x in lanes) < a.start_after:
            if time.perf_counter() > deadline:
                raise SystemExit("decode lanes never started")
            time.sleep(0.01)
        t_sub = time.perf_counter()
        b = Lane(fresh_prompt())
        b.wait()
        t_first = b.stamps()[0]
        for x in lanes:
            x.wait()
        gaps_in, gaps_before, tokens_in, tokens_before, span_before = [], [], 0, 0, 0.0
        for x in lanes:
            st = x.stamps()
            gaps_in += [(q - p) * 1e3 for p, q in zip(st, st[1:]) if t_sub <= q <= t_first + 0.05]
            pre = [s for s in st if s < t_sub]
            pre = pre[-30:]
            gaps_before += [(q - p) * 1e3 for p, q in zip(pre, pre[1:])]
            uid = x.uid
            tokens_in += sum(got.get(uid, 0) for t, got in ROUNDS if t_sub <= t <= t_first)
        after = scheduler_counters()
        window = t_first - t_sub
        return {
            "decode_lanes": d, "ttft_s": round(window, 3),
            "gap_in_window_ms": {"n": len(gaps_in), "p50": pct(gaps_in, .5), "p95": pct(gaps_in, .95),
                                 "max": max(gaps_in) if gaps_in else None},
            "gap_before_ms": {"p50": pct(gaps_before, .5), "p95": pct(gaps_before, .95)},
            "lane_tokens_in_window": tokens_in,
            "lane_tok_s_in_window": tokens_in / max(window, 1e-9),
            "scheduler_delta": {k: v - before.get(k, 0) for k, v in after.items()
                                if isinstance(v, (int, float)) and v != before.get(k, 0)},
        }

    alone()  # warm-up (kernels, allocator), discarded
    contended(1)
    rows = []
    for rep in range(a.reps):
        for d in a.decode_lanes:
            pair = {}
            for arm in (("alone", "contended") if rep % 2 == 0 else ("contended", "alone")):
                if arm == "alone":
                    pair["alone_ttft_s"] = alone()
                else:
                    pair["contended"] = contended(d)
                grew = swapouts() - swap0
                if grew > a.max_swapout_pages:
                    raise SystemExit(f"aborting: Swapouts grew by {grew} pages")
                mx.clear_cache()
            row = {"rep": rep, **pair}
            rows.append(row)
            c = pair["contended"]
            print(f"rep{rep} D={d} alone={pair['alone_ttft_s']}s contended={c['ttft_s']}s "
                  f"gap in-window p50/p95/max={c['gap_in_window_ms']['p50']}/{c['gap_in_window_ms']['p95']}/"
                  f"{c['gap_in_window_ms']['max']} before p50/p95={c['gap_before_ms']['p50']}/"
                  f"{c['gap_before_ms']['p95']} lane tok/s in window={c['lane_tok_s_in_window']:.1f} "
                  f"sched={c['scheduler_delta']}", flush=True)
    summary = {}
    for d in a.decode_lanes:
        mine = [r for r in rows if r["contended"]["decode_lanes"] == d]
        summary[str(d)] = {
            "alone_ttft_s_median": statistics.median(r["alone_ttft_s"] for r in mine),
            "contended_ttft_s_median": statistics.median(r["contended"]["ttft_s"] for r in mine),
            "gap_in_window_p50_median": statistics.median(r["contended"]["gap_in_window_ms"]["p50"] for r in mine),
            "gap_in_window_p95_median": statistics.median(r["contended"]["gap_in_window_ms"]["p95"] for r in mine),
            "gap_in_window_max_max": max(r["contended"]["gap_in_window_ms"]["max"] for r in mine),
            "gap_before_p50_median": statistics.median(r["contended"]["gap_before_ms"]["p50"] for r in mine),
            "lane_tok_s_in_window_median": statistics.median(r["contended"]["lane_tok_s_in_window"] for r in mine),
        }
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)
    json.dump({"model": a.model, "route": selection.route, "native_mtp": selection.native_mtp,
               "max_lanes": a.max_lanes, "prompt_tokens": a.prompt_tokens, "rows": rows,
               "summary": summary, "swapouts_delta_pages": swapouts() - swap0,
               "mlx": mx.__version__}, open(a.out, "w"), indent=1)
    engine.close()


if __name__ == "__main__":
    main()
