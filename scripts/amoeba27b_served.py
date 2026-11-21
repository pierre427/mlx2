#!/usr/bin/env python3
"""Amoeba 27B served A/B: arms x widths on the 20x20 sanity set, a near-tie probe, GSM8K accuracy.

Run inside an owned GPU window (cpg_job lease + run_with_gpu_locks, :8282 paused). One development
server at a time on a non-production port (never 8282), from this tree (PYTHONPATH=src), every arm
with the identical command apart from --execution-policy (amoeba_tiles), ordinary decode
(--ordinary), --max-lanes = the largest width. Each server's whole process group is stopped in
``finally``. Refuses a non-empty evidence directory.

Per arm and width W in --widths:
  quality   scripts/amoeba_sanity_lanes.py with SANITY_WIDTH=W and ceil(N/W) rounds (N = --requests,
            400), graded counts; per-request TTFT and decode tok/s from the server's mlx2 timings.
  near tie  --probe-prompts fixed greedy prompts, sent W at a time with top_logprobs=5 (B1 sent one at
            a time, the reference); each prompt's tokens at width W are compared with its own B1
            run, the first divergence localized, and the pair classified with mlx2.near_tie at that
            position plus the functional oracle (expected answer present, or non-empty prose).
Per arm: GSM8K (scripts/spec_acceptance_gsm8k.py, base-url mode, 5-shot completions, greedy),
peak process footprint, MLX/tile receipts from /v1/status and the first response.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba27b_lib as A  # noqa: E402
import spec_acceptance_gsm8k as GH  # noqa: E402

PROBES = (
    [(f"What is the capital of {c}? Answer in one word.", k.split()[0]) for c, k in A.COUNTRIES[:10]]
    + [(f"What is {a} + {b}? Reply with the number only.", str(a + b)) for a, b in
       ((317, 548), (902, 77), (455, 389), (120, 699), (731, 264), (808, 145))]
    + [(f"Translate the English word '{e}' into French. One word.", f) for e, f in A.FRENCH[:8]]
    + [(t, None) for t in ("Write two sentences about a lighthouse keeper.", "Explain what a hash table is in one sentence.",
                           "Describe the smell of rain in one sentence.", "Give one tip for learning to juggle.",
                           "Summarize the water cycle in two sentences.", "Name three uses of copper and say why.",
                           "Write a haiku about a train at night.", "What does a cartographer do? One sentence.")]
)


def post(base, body, timeout=600):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def wait_health(base, deadline_s, proc):
    end = time.time() + deadline_s
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}")
        try:
            if get(base, "/health").get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server did not become healthy")


def stop(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def probe_body(text):
    return {"messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": 48,
            "logprobs": True, "top_logprobs": 5, "enable_thinking": False, "reasoning_effort": "none"}


def tokens_of(resp):
    lp = (resp["choices"][0].get("logprobs") or {}).get("content") or []
    return [(e["token"], {t["token"]: t["logprob"] for t in e.get("top_logprobs", [])}) for e in lp]


def oracle(text, expect):
    if expect is None:
        return len(text.split()) >= 3 and not re.search(r"(.)\1{9,}", text)
    return _ascii_dashes(expect.lower()) in _ascii_dashes(text.lower())


def _ascii_dashes(text):
    """Unicode dash punctuation (category Pd: U+2010..U+2015, U+FE58, U+FF0D, ...) and the minus sign U+2212 ->
    ASCII hyphen (AGENTS.md identifier-oracle rule; models typeset codes). No other character changes."""
    return "".join("-" if (unicodedata.category(ch) == "Pd" or ch == "\u2212") else ch for ch in text)


def no_evidence(ta, tb):
    """True when neither response carries a logprob token: nothing was compared, so the pair is not 'identical'."""
    return not ta and not tb


def near_tie_probe(base, width, ref):
    from mlx2.near_tie import near_tie_equivalent
    out = []
    bodies = [probe_body(p) for p, _ in PROBES]
    with ThreadPoolExecutor(max_workers=width) as ex:
        resps = list(ex.map(lambda b: post(base, b), bodies)) if width > 1 else [post(base, b) for b in bodies]
    if ref is None:
        return resps, None
    return resps, classify_pairs(ref, resps)


def classify_pairs(ref, resps):
    """First-divergence near-tie classification of two greedy runs of PROBES (mlx2.near_tie + oracle)."""
    from mlx2.near_tie import near_tie_equivalent
    out = []
    vocab = {}

    def ids(d):
        return {vocab.setdefault(k, len(vocab)): v for k, v in d.items()}
    rows = {"identical": 0, "near_tie_equivalent": 0, "unclassified": 0, "failed": 0, "no_evidence": 0}
    for (prompt, expect), a, b in zip(PROBES, ref, resps):
        ta, tb = tokens_of(a), tokens_of(b)
        sa, sb = [t for t, _ in ta], [t for t, _ in tb]
        if no_evidence(ta, tb):
            rows["no_evidence"] += 1
            out.append({"prompt": prompt, "verdict": "no_evidence", "why": "no logprob entries in either response",
                        "ref_text": a["choices"][0]["message"]["content"],
                        "text": b["choices"][0]["message"]["content"]})
            continue
        if sa == sb:
            rows["identical"] += 1
            continue
        k = next((i for i in range(min(len(sa), len(sb))) if sa[i] != sb[i]), min(len(sa), len(sb)))
        rec = {"prompt": prompt, "first_divergence": k, "ref_text": a["choices"][0]["message"]["content"],
               "text": b["choices"][0]["message"]["content"]}
        if k >= min(len(ta), len(tb)):
            rec["verdict"] = "unclassified"
            rec["why"] = "length-only difference (one arm stopped)"
        else:
            la, lb = ids(ta[k][1]), ids(tb[k][1])
            sel_a, sel_b = vocab.setdefault(sa[k], len(vocab)), vocab.setdefault(sb[k], len(vocab))
            func = oracle(rec["ref_text"], expect) and oracle(rec["text"], expect)
            eq = near_tie_equivalent(sel_a, la, sel_b, lb)
            rec["window"] = {"ref": ta[k][1], "arm": tb[k][1]}
            rec["functional_oracle_both"] = func
            rec["verdict"] = "near_tie_equivalent" if (eq and func) else ("failed" if not func else "unclassified")
        rows[rec["verdict"]] += 1
        out.append(rec)
    return {"counts": rows, "divergent": out, "prompts": len(PROBES)}


def perf_of(records):
    ttft = [r["mlx2"].get("ttft_seconds") for r in records if r.get("mlx2") and r["mlx2"].get("ttft_seconds") is not None]
    rates = []
    for r in records:
        m, u = r.get("mlx2") or {}, r.get("usage") or {}
        n = u.get("completion_tokens") or 0
        if m.get("elapsed_seconds") and m.get("ttft_seconds") is not None and n > 1:
            dt = m["elapsed_seconds"] - m["ttft_seconds"]
            if dt > 0:
                rates.append((n - 1) / dt)
    return {"ttft_p50_s": statistics.median(ttft) if ttft else None,
            "decode_tok_s_per_request_p50": statistics.median(rates) if rates else None,
            "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in records)}


def gsm8k_chat(base, n, workers=4, max_tokens=512):
    """GSM8K test[:n] as chat, thinking off, greedy: the raw 5-shot completion prompt is void on
    the 27B (it emits <|im_end|> first on about half the prompts, a model behaviour). Scored with
    spec_acceptance_gsm8k's answer extraction."""
    test = GH.load_split(GH.DATA / "test.jsonl")[:n]

    def one(item):
        body = {"messages": [{"role": "user", "content": item["question"] + "\nGive the final numeric answer after ####."}],
                "temperature": 0, "max_tokens": max_tokens, "enable_thinking": False, "reasoning_effort": "none"}
        d = post(base, body, timeout=1800)
        text = d["choices"][0]["message"].get("content") or ""
        ans = GH.extract_answer(text)
        gold = GH.reference_answer(item["answer"])
        return {"pass": ans is not None and GH.normalize(str(ans)) == GH.normalize(str(gold)), "invalid": ans is None,
                "tokens": (d.get("usage") or {}).get("completion_tokens"), "finish": d["choices"][0].get("finish_reason")}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, test))
    toks = [r["tokens"] or 0 for r in rows]
    return {"mode": "chat, thinking off, greedy, max_tokens %d, %d workers" % (max_tokens, workers), "questions": len(rows),
            "accuracy": sum(r["pass"] for r in rows) / len(rows), "invalid_rate": sum(r["invalid"] for r in rows) / len(rows),
            "length_finishes": sum(r["finish"] == "length" for r in rows), "completion_tokens": sum(toks),
            "completion_tokens_median": statistics.median(toks), "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name=manifest_dir|none[,layout=L][,router=ROUTER_DIR][,ast=DRAFTER] (router: per-request open tiles, "
                         "client side; ast: --ast --ast-drafter DRAFTER instead of --ordinary)")
    ap.add_argument("--model", default=str(A.MODEL_27B))
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8391)
    ap.add_argument("--widths", default="1,2,3,4")
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--gsm8k-n", type=int, default=100)
    ap.add_argument("--gsm8k-mode", default="chat-nothink", choices=["chat-nothink", "completion-5shot"],
                    help="completion-5shot is void on the 27B (<|im_end|> first); kept for other checkpoints")
    ap.add_argument("--speed-only-arms", default="", help="arms run at --speed-widths only, with no GSM8K")
    ap.add_argument("--speed-widths", default="1")
    ap.add_argument("--sampling", default="", help="comma list of SANITY_SAMPLING modes per width (e.g. "
                    "greedy,t0.7:7001,t0.7:8001,t0.7:9001: Pierre 2026-11-15, 1 greedy + 3 at T=0.7); empty = tasks' own")
    ap.add_argument("--sampling-widths", default="", help="widths that also run the non-greedy --sampling modes "
                    "(Pierre 2026-11-15: greedy at B1 and B4, the three T=0.7 runs at B4 only -> '4'); empty = all")
    ap.add_argument("--cache-gib", type=int, default=4, help="APCv2 cache cap (the host default is ~48 GiB)")
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--server-cap-gb", type=float, default=80)
    ap.add_argument("--mlx-cache-gib", type=int, default=6, help="MLX buffer-cache limit on every dev server (MLX2_CACHE_LIMIT_GIB)")
    args = ap.parse_args()
    if not 8391 <= args.port <= 8395:
        sys.exit("development servers use ports 8391-8395")
    out = A.fresh_dir(args.out)
    log = A.make_log(out)
    rev = A.git_rev()
    widths = [int(w) for w in args.widths.split(",") if w]
    speed_only = set(filter(None, args.speed_only_arms.split(",")))
    base = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MLX2_CACHE_LIMIT_GIB": str(args.mlx_cache_gib)}
    summary = {"schema": "mlx2.amoeba27b-served.v1", "mlx2_source_revision": rev, "model": args.model,
               "port": args.port, "requests_per_width": args.requests, "arms": {}}
    from mlx2.adapters.qwen38_27b import inspect_artifact
    fp = inspect_artifact(args.model)["identity"]["fingerprint"]
    summary["checkpoint"] = f"{Path(args.model).name}@{fp[:8]}"
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        man, *opts = rest.split(",")
        opts = dict(o.split("=", 1) for o in opts)
        arm_dir = out / name
        arm_dir.mkdir()
        ws = [int(w) for w in args.speed_widths.split(",")] if name in speed_only else widths
        route = ["--ast", "--ast-drafter", opts["ast"]] if opts.get("ast") else ["--ordinary"]
        cmd = [sys.executable, "-m", "mlx2.server", "--model", args.model, "--port", str(args.port), *route,
               "--max-lanes", str(max(ws or [4])), "--cache-bytes", str(args.cache_gib << 30), "--max-context", str(args.max_context)]
        if man != "none":
            pol = arm_dir / "execution_policy.json"
            pol.write_text(json.dumps({"amoeba_tiles": str(Path(man) / "manifest.json"),
                                       **({"amoeba_tiles_layout": opts["layout"]} if opts.get("layout") else {})}))
            cmd += ["--execution-policy", str(pol)]
        rec = {"manifest": man, "options": opts, "server_cmd": cmd, "widths": {}, "mlx_cache_limit_gib": args.mlx_cache_gib,
               "apcv2_cache_gib": args.cache_gib}
        slog = open(arm_dir / "server.log", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
        peak = {"gb": 0.0}
        stop_ev = threading.Event()

        def sample():
            while not stop_ev.is_set():
                fpg = A.footprint_gb(proc.pid)
                if fpg:
                    peak["gb"] = max(peak["gb"], fpg)
                    if fpg > args.server_cap_gb:
                        log(f"[{name}] server footprint {fpg:.1f} GB > {args.server_cap_gb} GB: stopping the server")
                        peak["aborted"] = True
                        stop(proc)
                        return
                stop_ev.wait(10)
        threading.Thread(target=sample, daemon=True).start()
        try:
            t0 = time.time()
            wait_health(base, 1200, proc)
            rec["load_s"] = round(time.time() - t0, 1)
            st = get(base, "/v1/status")
            rec["status"] = {k: st.get(k) for k in ("model", "qualification", "qualified", "settings", "profile")}
            rec["execution"] = (st.get("execution") or {}).get("amoeba_tiles")
            first = post(base, probe_body("Say hello in one word."))
            rec["first_response_mlx2"] = {k: (first.get("mlx2") or {}).get(k) for k in ("route_receipt", "amoeba_tiles", "qualification")}
            if man != "none":
                bad = post_expect_error(base, {**probe_body("hi"), "mlx2_amoeba_tiles": ["core", "no_such_tile"]})
                rec["fail_closed_unknown_tile"] = bad
            else:
                rec["fail_closed_tiles_on_full"] = post_expect_error(base, {**probe_body("hi"), "mlx2_amoeba_tiles": ["core"]})
            ref = None
            modes = [m for m in args.sampling.split(",") if m] or [""]
            sw = {int(x) for x in args.sampling_widths.split(",") if x}
            for w, mode in [(w, m) for w in ws for m in modes
                            if not sw or w in sw or m in ("", "greedy")]:
                rounds = math.ceil(args.requests / w)
                senv = {**env, "SANITY_WIDTH": str(w), **({"SANITY_ROUTER": opts["router"]} if opts.get("router") else {}),
                        **({"SANITY_SAMPLING": mode} if mode else {})}
                wk = f"B{w}" + (f":{mode}" if mode else "")
                t1 = time.time()
                res = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_sanity_lanes.py"), base,
                                      str(arm_dir / f"sanity_{wk.replace(':', '_')}.json"), str(rounds)],
                                     cwd=ROOT, env=senv, capture_output=True, text=True, timeout=4 * 3600)
                wall = time.time() - t1
                (arm_dir / f"sanity_{wk.replace(':', '_')}.log").write_text(res.stdout + res.stderr)
                row = {"rc": res.returncode, "rounds": rounds, "wall_s": round(wall, 1)}
                p = arm_dir / f"sanity_{wk.replace(':', '_')}.json"
                if p.exists():
                    s = json.loads(p.read_text())
                    recs = s["records"]
                    row.update({"requests": s["requests"], "graded_correct": s["graded_correct"],
                                "http_errors": s["http_errors"], "issue_totals": s["issue_totals"],
                                "observed_widths": s["observed_widths"],
                                "by_task": {k: [v["correct"], v["n"]] for k, v in s["by_task"].items()},
                                **perf_of(recs)})
                    row["aggregate_decode_tok_s"] = row["completion_tokens"] / wall if wall else None
                if mode in ("", modes[0]):                         # greedy probes once per width
                    resps, nt = near_tie_probe(base, w, ref)
                    if w == 1 or ref is None:
                        ref = resps
                    if w == 1:
                        (arm_dir / "probe_B1.json").write_text(json.dumps(resps))
                    row["near_tie"] = nt
                else:
                    nt = None
                row["sampling"] = mode or "task_defaults"
                rec["widths"][wk] = row
                log(f"[{name} {wk}] {row.get('graded_correct')}/{row.get('requests')} ttft_p50={row.get('ttft_p50_s')} "
                    f"decode_p50={row.get('decode_tok_s_per_request_p50')} agg={row.get('aggregate_decode_tok_s')} "
                    f"near_tie={nt['counts'] if nt else 'reference'}")
                A.write_json(out / "served.json", {**summary, "arms": {**summary["arms"], name: rec}})
            if name not in speed_only and args.gsm8k_n and args.gsm8k_mode == "chat-nothink":
                rec["gsm8k"] = gsm8k_chat(base, args.gsm8k_n)
                log(f"[{name}] gsm8k (chat, no think) {rec['gsm8k']}")
            elif name not in speed_only and args.gsm8k_n:
                g = subprocess.run([sys.executable, str(ROOT / "scripts/spec_acceptance_gsm8k.py"), "--output",
                                    str(arm_dir / "gsm8k"), "--base-url", base, "--checkpoint", f"{summary['checkpoint']}:{name}",
                                    "--num-questions", str(args.gsm8k_n)],
                                   cwd=ROOT, env=env, capture_output=True, text=True, timeout=6 * 3600)
                (arm_dir / "gsm8k.log").write_text(g.stdout + g.stderr)
                rec["gsm8k_rc"] = g.returncode
                for f in sorted((arm_dir / "gsm8k").glob("*.json")) if (arm_dir / "gsm8k").exists() else []:
                    try:
                        d = json.loads(f.read_text())
                    except Exception:
                        continue
                    if isinstance(d, dict) and "accuracy" in json.dumps(d)[:20000]:
                        rec["gsm8k_file"] = str(f)
                        rec["gsm8k"] = {k: d.get(k) for k in ("accuracy", "invalid_rate", "decode_tok_s", "n", "arms") if k in d}
                log(f"[{name}] gsm8k rc={g.returncode} {rec.get('gsm8k')}")
            st = get(base, "/v1/status")
            rec["execution_after"] = (st.get("execution") or {}).get("amoeba_tiles")
        except Exception as e:
            rec["error"] = repr(e)
            log(f"[{name}] error {e!r}")
        finally:
            stop(proc)
            stop_ev.set()
            slog.close()
        rec["peak_footprint_gb"] = round(peak["gb"], 1)
        rec["server_aborted_on_footprint"] = bool(peak.get("aborted"))
        summary["arms"][name] = rec
        A.write_json(out / "served.json", summary)
    # cross-arm: each tiled arm's B1 greedy probes against the full model's (2026-11-15: label part_all's
    # mismatches as order-induced near ties vs real divergences)
    full_name = next((s_.split("=", 1)[0] for s_ in args.arm if s_.split("=", 1)[1].split(",")[0] == "none"), None)
    fp1 = out / str(full_name) / "probe_B1.json" if full_name else None
    if fp1 is not None and fp1.exists():
        fref = json.loads(fp1.read_text())
        summary["cross_arm_vs_full"] = {}
        for spec in args.arm:
            nm = spec.split("=", 1)[0]
            p1 = out / nm / "probe_B1.json"
            if nm != full_name and p1.exists():
                summary["cross_arm_vs_full"][nm] = classify_pairs(fref, json.loads(p1.read_text()))
                log(f"[cross {nm} vs {full_name} B1] {summary['cross_arm_vs_full'][nm]['counts']}")
        A.write_json(out / "served.json", summary)
    log(json.dumps({k: {w: (v.get("graded_correct"), v.get("requests")) for w, v in a.get("widths", {}).items()}
                    for k, a in summary["arms"].items()}))
    bad = failed_arms(summary)
    if bad:
        sys.exit(f"arms failed: {bad}")


def failed_arms(summary):
    """{arm: why} for arms that did not run to the end: an exception (server never healthy, ...), a server stopped
    by the footprint cap, or a sanity pass that left no report. A low graded score is a result, not a failure."""
    bad = {}
    for name, rec in summary["arms"].items():
        if rec.get("error"):
            bad[name] = rec["error"]
        elif rec.get("server_aborted_on_footprint"):
            bad[name] = "server stopped on the footprint cap"
        else:
            lost = [w for w, row in rec.get("widths", {}).items() if row.get("rc") and "requests" not in row]
            if lost:
                bad[name] = f"sanity pass left no report at {lost}"
    return bad


def post_expect_error(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return {"status": r.status, "refused": False}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "refused": True, "body": e.read().decode(errors="replace")[:300]}


if __name__ == "__main__":
    main()
