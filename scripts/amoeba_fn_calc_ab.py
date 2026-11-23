#!/usr/bin/env python3
"""Calculator det tile on Flash-Next: served on/off A/B on a dev port (experimental, unqualified).

One server (``--det-tile calculator=MANIFEST``, the native-MTP production route) serves both arms:
every item is sent twice, with ``"det_tiles": {"calculator": true|false}``, pair order alternating.
Every request sets ``skip_writing_prefix_cache`` so no arm reuses the other's prompt state.

Sets (``--sets``):
  arith    add:4:4 addk:3:4 mul:3:2 mul:4:2 mul:3:3 (scripts/amoeba_bridge_lab.py items), plain chat,
           thinking off, greedy, n per task; width ``--width``.
  gsm8k    GSM8K test questions (``--gsm-indices``) with the harness default (chat 5-shot, 2048
           tokens, scripts/spec_acceptance_gsm8k.py); width ``--width``.
  nomath   no-arithmetic English (prose, number-heavy text, copy/echo lines, code), B1: any plan is a
           false fire; on/off text differences are drift (a difference without a plan is a defect).
  timing   B1 subset (arith 8 per task + 20 GSM8K), sequential, for tok/s and TTFT.
  plain    a second server without --det-tile replays the nomath set and the arith timing subset at
           B1; calc-off outputs must equal it token for token (off is bit-exact).

Reports accuracy per arm with paired-bootstrap CIs of the difference, fire rates, false fires, drift,
tok/s and TTFT. Receipts name the checkpoint (``<dir>@<fingerprint8>``) and the mlx2 commit.

Other models (``--model``; default Flash-Next): ``--execution-policy FILE`` is passed to both servers,
``--tasks`` keeps a subset of the arithmetic tasks (same items per task as the full set), and
``--limit-gb`` is the server footprint kill limit. Hy4-flash keeps its DFly2 default route with a
route-neutral policy (``{"engram_table_format": "iq4_nl"}``) plus ``--det-tile``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import http.client
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODEL = Path.home() / "mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"
ARITH_TASKS = ("add:4:4", "addk:3:4", "mul:3:2", "mul:4:2", "mul:3:3")
SELECTED_TASKS = ARITH_TASKS
EXECUTION_POLICY = None     # --execution-policy FILE, passed to every server

PROSE = ["Write three sentences about a lighthouse keeper.", "Explain in two sentences what a compiler does.",
         "Describe a rainy morning in a small harbour town.", "Give two tips for keeping houseplants alive.",
         "Summarise the plot of a heist film you invent, in three sentences.",
         "Write a short note thanking a neighbour for help moving.", "What makes a good cup of tea? Two sentences.",
         "Describe the sound of a forest at night.", "Explain why bridges have expansion joints.",
         "Write a four-line poem about autumn leaves.", "Suggest a name for a bakery and say why.",
         "Explain what a hash table is to a beginner.", "Describe your ideal reading nook.",
         "Write a polite reminder about a library book.", "What is the difference between weather and climate?",
         "Tell a short story about a lost umbrella.", "Explain how a sailboat moves against the wind.",
         "Write a product blurb for a canvas backpack.", "Describe a market stall selling spices.",
         "Give advice to someone starting to learn the violin.", "Explain what photosynthesis does, briefly.",
         "Write an opening line for a mystery novel.", "Describe a train journey through mountains.",
         "Why do cats knead blankets? Answer briefly.", "Write a haiku about snowfall.",
         "Explain what version control is for.", "Describe the taste of a ripe peach.",
         "Write a short welcome message for a hiking club.", "Explain the rules of tic-tac-toe.",
         "Describe a busy kitchen during dinner service.", "What should go in a first-aid kit? List briefly.",
         "Write a two-sentence horror story.", "Explain what a museum curator does.",
         "Describe the first warm day of spring.", "Give three ideas for a rainy-day activity.",
         "Explain why the sky looks blue, simply.", "Write a limerick about a forgetful owl.",
         "Describe an old bookshop.", "Explain what an API is in one paragraph.",
         "Write a short apology for missing a meeting."]
NUMBERS = ["Repeat exactly: Call 555-1234 on 12/25/2024 at 10:30.",
           "Repeat exactly: Order #4471-889 shipped; version 3.11.2 is current.",
           "Copy this line: The price is $5.99 and the score was 3-2.",
           "Write one sentence about the year 1999 and the number 42.",
           "Repeat exactly: Room 214, floor 3, building 7B.", "Copy this line: Flight BA 2490 departs at 06:45.",
           "Repeat exactly: The match ended 21-19 after extra time.", "Copy this line: Use a 16:9 screen at 1920x1080.",
           "Repeat exactly: Pages 120-145 cover the 1848 revolutions.", "Copy this line: Zip code 02139, phone (617) 555-0101.",
           "Write a sentence mentioning the years 1066 and 1215.", "Repeat exactly: Python 3.12.4 and Node 20.11.1 are installed.",
           "Copy this line: Serial number 8842-1177-0093.", "Repeat exactly: Born 3/14/1879, died 4/18/1955.",
           "Write one sentence using the numbers 7, 12 and 300.", "Copy this line: Mix 2 cups flour with 1 cup water.",
           "Repeat exactly: The ratio was 3:1 in favour.", "Copy this line: Latitude 51.5074, longitude -0.1278.",
           "Repeat exactly: Ticket 00451 for seat 12A.", "Write a sentence about a 5k run finished in 24 minutes.",
           "Copy this line: The 2024-2025 season starts in August.", "Repeat exactly: ISBN 978-0-14-044913-6.",
           "Copy this line: Apartment 4/12 on 33rd Street.", "Repeat exactly: Doors open 7-9 pm on Fridays.",
           "Write one sentence that mentions 1, 2 and 3 without doing any arithmetic.",
           "Copy this line: Model X-200, firmware 1.0.17.", "Repeat exactly: Chapter 11, verse 4.",
           "Copy this line: Temperature range 18-24 degrees.", "Repeat exactly: Invoice 2024/0187 is due 30/06.",
           "Copy this line: Track 9 of 14, length 3:47."]
ECHO = ["Repeat exactly: 2 + 2 = 5 is a famous line from Orwell.", "Copy this line verbatim: 7 * 6 = 41 (typo kept).",
        "Repeat exactly, keeping the error: 10 - 3 = 8.", "Copy this line: the sign read 3 + 4 = ?",
        "Repeat exactly: Is 9 + 10 = 21? No.", "Copy verbatim: x + 1 = 3 so x = 2.",
        "Repeat exactly: a + b = c is the template.", "Copy this line: 100 / 4 = twenty five (spelled out).",
        "Repeat exactly the slogan: 1 + 1 = 3 when we work together.", "Copy this line: 12 * 12 = 144 is on the poster.",
        "Repeat exactly: Score 3 + 2 = 5 points total.", "Copy verbatim: 8 - 8 = 0 was the riddle's answer.",
        "Repeat exactly: '5 + 5 = 10' is printed on the mug.", "Copy this line: 6 / 3 = 2 appears twice.",
        "Repeat exactly: the student wrote 15 + 7 = 23.", "Copy verbatim: 3 * 3 = 9, 4 * 4 = 16.",
        "Repeat exactly: version 2 + patch 1 = release 3.", "Copy this line: 40 - 15 = 25 years later.",
        "Repeat exactly, errors included: 11 + 11 = 2211.", "Copy verbatim: 9 * 9 = 81 and that's that."]
CODE = ["Write a Python line that sets total = 3 + 4 without evaluating it in a comment.",
        "Show a JavaScript snippet: let y = x + 4; console.log(y);",
        "Write a SQL query selecting rows where price = 10.",
        "Write a shell command that echoes 2+2 as text, not the result.",
        "Write a C for loop from i = 0 to 9.", "Show a Python dict with keys a=1 and b=2.",
        "Write a regex that matches dates like 2024-01-31.", "Write an HTML input with value=\"5\".",
        "Write a CSS rule setting width: 100px and margin: 0 auto.", "Write a Python f-string printing a + b literally.",
        "Write a YAML key replicas: 3 with a comment.", "Show a git command to view the last 5 commits.",
        "Write a Makefile variable CFLAGS = -O2 -Wall.", "Write a Rust let binding: let n: u32 = 7;",
        "Write a Go constant const Max = 100.", "Write a Python assert that 1 + 1 == 2, as code.",
        "Write an Excel formula that sums A1 to A10.", "Write JSON with a field count equal to 12.",
        "Write a LaTeX equation for E = mc^2.", "Write a TOML entry timeout = 30."]


def nomath_items():
    out = [("prose", t) for t in PROSE] + [("numbers", t) for t in NUMBERS]
    out += [("echo", t) for t in ECHO] + [("code", t) for t in CODE]
    return out


def make_items(spec, n, seed):
    """Task items as scripts/amoeba_bridge_lab.py v2 makes them (1cec2d559):
    add:A:B an A-digit plus a B-digit number, mul:A:B an A-digit times a B-digit number, addk:A:K K
    A-digit numbers summed; distinct term sets; "What is ...? Answer with only the number."."""
    import numpy as np
    op, x, y = spec.split(":")
    x, y = int(x), int(y)
    rng = np.random.default_rng(seed)

    def num(d):
        return int(rng.integers(10 ** (d - 1), 10 ** d))
    seen, out = set(), []
    while len(out) < n:
        terms = [num(x) for _ in range(y)] if op == "addk" else [num(x), num(y)]
        if tuple(sorted(terms)) in seen:
            continue
        seen.add(tuple(sorted(terms)))
        ans = int(np.prod(terms, dtype=object)) if op == "mul" else sum(terms)
        sym = " × " if op == "mul" else " + "
        out.append({"prompt": f"What is {sym.join(str(t) for t in terms)}? Answer with only the number.",
                    "answer": ans})
    return out


def arith_items(n, seed=4511):
    items = []
    for k, spec in enumerate(ARITH_TASKS):
        if spec not in SELECTED_TASKS:
            continue                       # the seed stays the task's index in the full set
        for it in make_items(spec, n, seed + k):
            items.append({"task": spec, "prompt": it["prompt"], "expect": str(it["answer"])})
    return items


def gsm_items(indices):
    import spec_acceptance_gsm8k as G
    _, files = G.dataset_files()
    train, test = G.load_split(files["train"]), G.load_split(files["test"])
    shots = G.few_shot_turns(train, 5)
    return [{"task": "gsm8k", "index": i, "messages": shots + [{"role": "user", "content": test[i]["question"] + G.CHAT_INSTRUCTION}],
             "expect": G.reference_answer(test[i]["answer"])} for i in indices]


def last_number(text):
    text = text.split("</think>")[-1].replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return m[-1] if m else None


# ---------------------------------------------------------------------------------------------- HTTP

def post(base, body, timeout=1800):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        out = {"http_error": e.code, "body": e.read().decode()[:2000]}
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, http.client.HTTPException) as e:
        # A dropped connection (server killed, timeout) is one failed row, not the end of the set.
        out = {"http_error": "transport", "body": repr(e)[:2000]}
    out["_wall_s"] = time.time() - t0
    return out


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read().decode()


def health(base):
    try:
        return json.loads(get(base, "/health", 5)).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def request_body(item, arm):
    if "messages" in item:                                  # GSM8K: the harness payload, unchanged
        body = {"messages": item["messages"], "temperature": 0, "max_tokens": 2048, "stream": False}
    else:
        body = {"messages": [{"role": "user", "content": item["prompt"]}], "temperature": 0,
                "max_tokens": item.get("max_tokens", 48), "enable_thinking": False, "reasoning_effort": "none",
                "stream": False}
    if arm is not None:
        body["det_tiles"] = {"calculator": arm == "on"}
    body["skip_writing_prefix_cache"] = True
    return body


def row_of(item, arm, r):
    msg = ((r.get("choices") or [{}])[0].get("message") or {}) if "choices" in r else {}
    content = msg.get("content") or ""
    m = r.get("mlx2") or {}
    rr = m.get("route_receipt") or {}
    det = rr.get("amoeba_det_tiles_request") if isinstance(rr, dict) else None
    usage = r.get("usage") or {}
    ttft, el, ct = m.get("ttft_seconds"), m.get("elapsed_seconds"), usage.get("completion_tokens") or 0
    row = {"task": item["task"], "arm": arm, "content": content, "expect": item.get("expect"),
           "prompt": item.get("prompt", item.get("index")), "http_error": r.get("http_error"),
           "error_body": r.get("body"), "completion_tokens": ct, "ttft_s": ttft, "elapsed_s": el,
           "wall_s": r.get("_wall_s"), "decode_tps": (ct - 1) / (el - ttft) if ttft and el and ct > 1 and el > ttft else None,
           "route": m.get("route"), "speculation": m.get("speculation"), "det": det,
           "plans": ((det or {}).get("calculator") or {}).get("plans", 0),
           "finish_reason": ((r.get("choices") or [{}])[0] or {}).get("finish_reason")}
    if item["task"] == "gsm8k":
        import spec_acceptance_gsm8k as G
        pred = G.extract_answer(content)
        row.update(index=item["index"], predicted=pred, ok=pred is not None and pred == item["expect"])
    elif item.get("expect") is not None:
        row.update(predicted=last_number(content), ok=last_number(content) == item["expect"])
    return row


def run_pairs(base, items, width, log, arms=("on", "off")):
    """Each item under both arms; pair order alternates by index; ``width`` requests in flight."""
    work = []
    for i, it in enumerate(items):
        order = arms if i % 2 == 0 else tuple(reversed(arms))
        work += [(i, it, a) for a in order]
    rows = [None] * len(work)
    with cf.ThreadPoolExecutor(width) as ex:
        futs = {ex.submit(post, base, request_body(it, a)): k for k, (i, it, a) in enumerate(work)}
        for done, f in enumerate(cf.as_completed(futs), 1):
            k = futs[f]
            i, it, a = work[k]
            rows[k] = {**row_of(it, a, f.result()), "pair": i}
            if done % 50 == 0:
                log(f"  {done}/{len(work)}")
    return rows


# ---------------------------------------------------------------------------------------------- stats

def paired_bootstrap(on, off, n_boot=10000, seed=0):
    """Accuracy on/off and the 95% CI of (on - off) over paired items."""
    rng = random.Random(seed)
    n = len(on)
    if not n:
        return {"n": 0}
    d = [a - b for a, b in zip(on, off)]
    boots = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {"n": n, "acc_on": sum(on) / n, "acc_off": sum(off) / n, "diff": sum(d) / n,
            "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
            "on_only": sum(1 for a, b in zip(on, off) if a and not b),
            "off_only": sum(1 for a, b in zip(on, off) if b and not a)}


def pair_up(rows):
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["pair"]), {})[r["arm"]] = r
    return by


def summarize_scored(rows):
    out = {}
    by = pair_up(rows)
    for task in sorted({t for t, _ in by}):
        allp = [v for (t, _), v in sorted(by.items()) if t == task and "on" in v and "off" in v]
        # A pair with a failed request in either arm is reported, not scored.
        pairs = [p for p in allp if not p["on"]["http_error"] and not p["off"]["http_error"]]
        on = [bool(p["on"].get("ok")) for p in pairs]
        off = [bool(p["off"].get("ok")) for p in pairs]
        s = paired_bootstrap(on, off)
        s["fire_rate_on"] = sum(1 for p in pairs if p["on"]["plans"]) / len(pairs) if pairs else None
        s["plans_off"] = sum(p["off"]["plans"] for p in pairs)
        s["errors"] = sum(1 for p in allp for a in ("on", "off") if p[a]["http_error"])
        s["pairs_dropped_for_errors"] = len(allp) - len(pairs)
        s["text_differs"] = sum(1 for p in pairs if p["on"]["content"] != p["off"]["content"])
        s["differs_without_fire"] = sum(1 for p in pairs if p["on"]["content"] != p["off"]["content"]
                                        and not p["on"]["plans"])
        # At width > 1 an "on" lane changes its batch's draft path and batch-mates, so differences
        # without a fire there can be batch numerics; only the B1 sets test exactness.
        s["fail_closed_on"] = {k: sum(((p["on"]["det"] or {}).get("calculator") or {}).get(k, 0) for p in pairs)
                               for k in ("gate_unmatched", "gate_ambiguous", "row_unbound", "plan_aborted")}
        out[task] = s
    return out


def timing(rows):
    out = {}
    for arm in ("on", "off", "plain"):
        rs = [r for r in rows if r["arm"] == arm and r["decode_tps"]]
        if not rs:
            continue
        tps = sorted(r["decode_tps"] for r in rs)
        tt = sorted(r["ttft_s"] for r in rs if r["ttft_s"] is not None)
        out[arm] = {"n": len(rs), "decode_tps_median": tps[len(tps) // 2],
                    "decode_tps_mean": sum(tps) / len(tps),
                    "ttft_median_s": tt[len(tt) // 2] if tt else None}
    if "on" in out and "off" in out:
        out["decode_tps_ratio_on_off"] = out["on"]["decode_tps_median"] / out["off"]["decode_tps_median"]
    return out


# ---------------------------------------------------------------------------------------------- server

class Server:
    def __init__(self, out, port, det_manifest, log, limit_gb=88.0):
        self.out, self.port, self.log = out, port, log
        self.base = f"http://127.0.0.1:{port}"
        cmd = [sys.executable, "-m", "mlx2.server", "--model", str(MODEL), "--host", "127.0.0.1", "--port", str(port),
               "--max-context", "131072", "--max-lanes", "4", "--max-inflight", "8", "--agent-compat", "on"]
        if EXECUTION_POLICY:
            cmd += ["--execution-policy", str(EXECUTION_POLICY)]
        if det_manifest:
            cmd += ["--det-tile", f"calculator={det_manifest}"]
        self.cmd = cmd
        out.mkdir(parents=True, exist_ok=True)
        self.slog = open(out / "server.log", "w")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1"}
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=self.slog, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.peak = {"gb": 0.0, "killed": None}
        threading.Thread(target=self._watch, args=(limit_gb,), daemon=True).start()
        end = time.time() + 1800
        try:
            while not health(self.base):
                if self.proc.poll() is not None:
                    raise RuntimeError(f"server exited {self.proc.returncode} (see {out / 'server.log'})")
                if time.time() > end:
                    raise RuntimeError("server not healthy after 30 min")
                time.sleep(3)
        except BaseException:
            self.stop()          # never leave the process group holding the GPU
            raise
        log(f"server up on {port}: {' '.join(cmd[3:])}")

    def _watch(self, limit):
        import amoeba27b_lib as A
        while self.proc.poll() is None:
            fp = A.footprint_gb(self.proc.pid)
            if fp is not None:
                self.peak["gb"] = max(self.peak["gb"], round(fp, 1))
                if fp > limit:
                    self.peak["killed"] = f"footprint {fp:.1f} GB > {limit} GB"
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    return
            time.sleep(10)

    def snapshot(self, name):
        for path, fn in (("/v1/status", f"{name}-status.json"), ("/metrics", f"{name}-metrics.txt")):
            try:
                (self.out / fn).write_text(get(self.base, path))
            except Exception as e:  # noqa: BLE001
                (self.out / fn).write_text(repr(e))

    def stop(self):
        if self.slog.closed:
            return
        # The whole group, even if the leader already exited (its children may not have).
        for sig, wait in ((signal.SIGTERM, 120), (signal.SIGKILL, 30)):
            try:
                os.killpg(self.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                self.proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        self.slog.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--port", type=int, default=8391)
    ap.add_argument("--sets", default="nomath,timing,arith,gsm8k,plain")
    ap.add_argument("--n", type=int, default=100, help="items per arithmetic task")
    ap.add_argument("--gsm-indices", default="0-199")
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--model", type=Path, default=None, help="model directory (default Flash-Next)")
    ap.add_argument("--execution-policy", type=Path, default=None, help="policy JSON passed to every server")
    ap.add_argument("--tasks", default=",".join(ARITH_TASKS), help="arithmetic tasks to keep (comma list)")
    ap.add_argument("--limit-gb", type=float, default=88.0, help="server footprint kill limit")
    args = ap.parse_args()
    global MODEL, SELECTED_TASKS, EXECUTION_POLICY
    MODEL = (args.model or MODEL).expanduser()
    SELECTED_TASKS = tuple(t for t in args.tasks.split(",") if t)
    if not SELECTED_TASKS or set(SELECTED_TASKS) - set(ARITH_TASKS):
        sys.exit(f"--tasks must name tasks of {ARITH_TASKS}")
    EXECUTION_POLICY = args.execution_policy
    if not 8391 <= args.port <= 8395:
        sys.exit("development servers use ports 8391-8395")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite nonempty {out}")
    out.mkdir(parents=True)
    sets = args.sets.split(",")
    a, b = args.gsm_indices.split("-")
    gsm = gsm_items(range(int(a), int(b) + 1))
    arith = arith_items(args.n)
    nomath = [{"task": "nomath:" + k, "prompt": t, "max_tokens": 96} for k, t in nomath_items()]
    tsub = [it for k in SELECTED_TASKS for it in [x for x in arith if x["task"] == k][:8]] + gsm[:20]
    rev = subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=ROOT, text=True)
    from mlx2.runtime.amoeba_det_tiles import load_manifest
    man = load_manifest(args.manifest)
    # The adapter refuses a manifest built on another checkpoint, so a running det server proves it.
    fp = man["source"]["fingerprint"]
    logf = open(out / "ab.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    summary = {"schema": "mlx2.amoeba-fn-calc-ab.v1", "qualification": "unqualified",
               "checkpoint": f"{MODEL.name}@{str(fp).split(':')[-1][:8]}", "source_fingerprint": fp,
               "mlx2_source_revision": rev, "dirty_paths": dirty.splitlines(),
               "manifest": str(Path(args.manifest)), "manifest_sha256": man["_digest"],
               "tile": man["tiles"][0], "args": vars(args), "sets": {}}

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    srv = Server(out / "det-server", args.port, args.manifest, log, limit_gb=args.limit_gb)
    summary["det_server"] = {"cmd": srv.cmd, "footprint": srv.peak}
    rows = {}
    try:
        for name in [s for s in sets if s != "plain"]:
            t0 = time.time()
            log(f"set {name}")
            if name == "nomath":
                rows[name] = run_pairs(srv.base, nomath, 1, log)
            elif name == "timing":
                rows[name] = run_pairs(srv.base, tsub, 1, log)
            elif name == "arith":
                rows[name] = run_pairs(srv.base, arith, args.width, log)
            elif name == "gsm8k":
                rows[name] = run_pairs(srv.base, gsm, args.width, log)
            (out / f"rows-{name}.json").write_text(json.dumps(rows[name], indent=1, default=str))
            s = summarize_scored(rows[name]) if name in ("arith", "gsm8k", "timing") else {}
            if name == "nomath":
                by = pair_up(rows[name])
                pairs = [v for v in by.values() if "on" in v and "off" in v]
                s = {"n": len(pairs), "false_fires": sum(1 for p in pairs if p["on"]["plans"]),
                     "false_fire_rate": sum(1 for p in pairs if p["on"]["plans"]) / len(pairs),
                     "false_fire_prompts": [p["on"]["prompt"] for p in pairs if p["on"]["plans"]],
                     "text_differs": sum(1 for p in pairs if p["on"]["content"] != p["off"]["content"]),
                     "differs_without_fire": sum(1 for p in pairs if p["on"]["content"] != p["off"]["content"]
                                                 and not p["on"]["plans"]),
                     "plans_off": sum(p["off"]["plans"] for p in pairs),
                     "errors": sum(1 for p in pairs for a in ("on", "off") if p[a]["http_error"]),
                     "by_kind": {k: sum(1 for p in pairs if p["on"]["task"] == "nomath:" + k and p["on"]["plans"])
                                 for k in ("prose", "numbers", "echo", "code")}}
            s["timing"] = timing(rows[name])
            s["wall_s"] = time.time() - t0
            summary["sets"][name] = s
            save()
            log(f"set {name} done: {json.dumps({k: v for k, v in s.items() if k != 'false_fire_prompts'})[:900]}")
        srv.snapshot("det")
    finally:
        srv.stop()
        summary["det_server"]["footprint"] = srv.peak
        save()
    if "plain" in sets:
        log("set plain (no --det-tile): off must equal it")
        plain = Server(out / "plain-server", args.port, None, log, limit_gb=args.limit_gb)
        try:
            items = nomath + tsub[:40]
            prow = run_pairs(plain.base, items, 1, log, arms=(None,))
            for r in prow:
                r["arm"] = "plain"
            (out / "rows-plain.json").write_text(json.dumps(prow, indent=1, default=str))
            offs = {(r["task"], r["prompt"]): r for name in ("nomath", "timing")
                    for r in rows.get(name, []) if r["arm"] == "off"}
            same, diff, missing = 0, [], 0
            for r in prow:
                o = offs.get((r["task"], r["prompt"]))
                if o is None:
                    missing += 1
                elif o["content"] == r["content"]:
                    same += 1
                else:
                    diff.append({"task": r["task"], "prompt": r["prompt"], "off": o["content"], "plain": r["content"]})
            summary["sets"]["plain"] = {"n": len(prow), "off_equals_plain": same, "differs": len(diff),
                                        "missing": missing, "differences": diff[:20], "timing": timing(prow)}
            plain.snapshot("plain")
        finally:
            plain.stop()
            summary["plain_server"] = {"cmd": plain.cmd, "footprint": plain.peak}
            save()
    log("done " + str(out))


if __name__ == "__main__":
    main()
