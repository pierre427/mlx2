#!/usr/bin/env python3
"""Maxwell-V1-Flash ABBA performance ladder: current default (A) vs best option set (B).

  maxwell_abba_ladder.py --model DIR --out DIR [--cells 3000,6000,16000,32000] [--widths 1,4]

Four server segments in the order A, B, B, A, one loopback ``mlx2.server`` per
segment (same serving shape in both arms: --max-lanes 4 --max-inflight 4).
Per segment, per context cell and batch width: APCv2 cleared, then ``width``
concurrent greedy needle requests (distinct prompts, identical across
segments; thinking off; the answer must start with the code, then a ~100-word
summary so decode is long enough to time).  Each request streams with
``top_logprobs`` 2, so every cell carries its own first-divergence evidence.

Per cell the summary reports A (segments 1, 4) and B (2, 3) means of client
TTFT, per-request decode tok/s and aggregate tok/s, B/A ratios, the needle
oracle, and token status of B vs A (and A vs A, B vs B as repeatability):
exact, near_tie_equivalent (AGENTS.md first-divergence contract: shared
prefix, same top-2 set with both margins <= 0.5 nat, or the exact-tie clause,
and both continuations pass the oracle) or unclassified.  Thermal (pmset) and
swap receipts per segment.  No thermal hold: sanity/attribution numbers, not
publishable statistics.  The caller owns the GPU (gpuq-reference.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
from maxwell_overnight import memory, needle_prompt, normalize_code, revision, write_atomic  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from qualify_prefill_handoff import near_tie_passes  # noqa: E402
from run_guarded_benchmark import owns_listener  # noqa: E402

PORT = 18321
ARMS = {
    "A": {"label": "current default (ordinary route, bf16 head)", "flags": [], "policy": None, "env": {}},
    "B": {"label": "native MTP depth 2 + bf16 indexer kernel + masked long prefill + fused q/k/v",
          "flags": ["--native-mtp"], "policy": {"num_draft": 2},
          "env": {"MLX2_MAXWELL_INDEX_KERNEL": "1", "MLX2_MAXWELL_SPARSE_PREFILL": "masked",
                  "MLX2_MAXWELL_FUSED_QKV": "1"}},
}
ORDER = ("A", "B", "B", "A")
QUESTION = ("What is the access code for the north vault? Write the code alone on the first line. "
            "Then, in about 100 words, describe what the ledger records.")


def thermal():
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    return {"pmset_therm": out.strip(), "time": time.time()}


def prompts(tokenizer, cell, width):
    rows = []
    for j in range(width):
        request, code = needle_prompt(tokenizer, cell, seed=20261122 + 1000 * cell + j)
        text = request["messages"][0]["content"]
        text = text[: text.rindex("\n\n")] + "\n\n" + QUESTION
        rows.append(({"role": "user", "content": text}, code))
    return rows


def stream(base, model_name, message, max_tokens):
    body = {"model": model_name, "messages": [message], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "logprobs": True, "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    arrivals, tokens, parts, receipt, done = [], [], [], None, False
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"SSE error: {event['error']}")
            if event.get("mlx2") is not None:
                receipt = event["mlx2"]
            for choice in event.get("choices", ()):
                for item in (choice.get("logprobs") or {}).get("content", ()) or ():
                    top = [[int(t["id"]) if "id" in t else t.get("token"), float(t["logprob"])]
                           for t in item.get("top_logprobs", [])[:2]]
                    tokens.append({"id": int(item["id"]) if "id" in item else item.get("token"),
                                   "logprob": float(item["logprob"]), "top2": top})
                    arrivals.append(time.perf_counter() - started)
                parts.append((choice.get("delta") or {}).get("content") or "")
    if not done or receipt is None or not tokens:
        raise ValueError("incomplete generation stream, token trace or terminal receipt")
    decode = ((len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
              if len(arrivals) > 1 and arrivals[-1] > arrivals[0] else None)
    return {"text": "".join(parts), "tokens": tokens, "ttft_s": arrivals[0] if arrivals else None,
            "elapsed_s": time.perf_counter() - started, "decode_tok_s": decode, "receipt": receipt}


def classify(a, b):
    """First-divergence verdict for two streamed rows (AGENTS.md near-tie contract)."""
    ta, tb = a["tokens"], b["tokens"]
    if not ta or not tb or "error" in a or "error" in b:
        return {"verdict": "unclassified", "reason": "missing token trace or failed request"}
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x["id"] != y["id"]:
            break
    else:
        if len(ta) == len(tb):
            return {"verdict": "exact"}
        return {"verdict": "unclassified", "reason": "length", "index": min(len(ta), len(tb))}
    oracle = a["found"] and b["found"]
    top_a, top_b = ta[i]["top2"], tb[i]["top2"]
    if len(top_a) < 2 or len(top_b) < 2:
        return {"verdict": "unclassified", "reason": "no top-2 window", "index": i}
    ma, mb = top_a[0][1] - top_a[1][1], top_b[0][1] - top_b[1][1]
    ok = near_tie_passes({
        "shared_prefix": True, "ref_selected": x["id"], "arm_selected": y["id"],
        "ref_top2": [t[0] for t in top_a], "arm_top2": [t[0] for t in top_b],
        "ref_margin": ma, "arm_margin": mb, "ref_window": top_a, "arm_window": top_b,
    }, functional_pass=oracle)
    return {"verdict": "near_tie_equivalent" if ok else "unclassified", "index": i,
            "margins": [ma, mb], "top2": [top_a, top_b], "oracle": oracle}


class Server:
    def __init__(self, model, arm, out):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_hex(24)
        token_file = out / "admin-token"
        token_file.write_text(self.token)
        token_file.chmod(0o600)
        spec = ARMS[arm]
        cmd = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
               "--port", str(PORT), "--max-lanes", "4", "--max-inflight", "4", "--max-context", "40000",
               "--cache-bytes", str(4 << 30), "--no-cache-dir", "--admin-token-file", str(token_file),
               *spec["flags"]]
        if spec["policy"] is not None:
            (out / "policy.json").write_text(json.dumps(spec["policy"]))
            cmd += ["--execution-policy", str(out / "policy.json")]
        self.command, self.env = cmd, spec["env"]
        self.log = (out / "server.log").open("w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, process_group=0,
                                     cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **spec["env"]})
        self.base = f"http://127.0.0.1:{PORT}"

    def call(self, path, payload=None, timeout=60):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data else "GET",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.token})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body.strip().startswith(b"{") else body.decode()

    def ready(self, limit=1200):
        end = time.monotonic() + limit
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                if owns_listener(self.proc.pid, PORT):
                    status = self.call("/v1/status", timeout=10)
                    if status.get("healthy"):
                        return status
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("server readiness timed out")

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        self.log.close()
        (self.out / "admin-token").unlink(missing_ok=True)


def run_segment(model, arm, index, out, cell_prompts, widths, max_tokens, max_batch_tokens):
    seg = out / f"seg{index + 1}-{arm}"
    before = memory()
    row = {"segment": index + 1, "arm": arm, "spec": ARMS[arm], "thermal_before": thermal(), "cells": []}
    server = Server(model, arm, seg)
    row["command"] = server.command
    try:
        status = server.ready()
        row["load_status"] = {k: status.get(k) for k in ("route_receipt", "route", "qualification", "model")}
        row["checkpoint"] = (status.get("execution") or {}).get("checkpoint")
        model_name = status.get("model") or Path(model).name
        for cell, width in plan(cell_prompts, widths, max_batch_tokens):
            server.call("/v1/admin/apc/clear", {"reason": f"abba-{cell}-{width}"})
            batch = cell_prompts[cell][:width]
            results = [None] * width

            def go(j):
                msg, code = batch[j]
                try:
                    r = stream(server.base, model_name, msg, max_tokens)
                    r["found"] = code in normalize_code(r["text"])
                except Exception as error:          # recorded, never silently dropped
                    r = {"error": repr(error), "found": False, "tokens": []}
                results[j] = r

            started = time.perf_counter()
            threads = [threading.Thread(target=go, args=(j,)) for j in range(width)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - started
            generated = sum(len(r.get("tokens", [])) for r in results)
            row["cells"].append({"cell": cell, "width": width, "wall_s": round(wall, 3),
                                 "aggregate_tok_s": round(generated / wall, 2) if wall else None,
                                 "requests": results})
            print(json.dumps({"segment": index + 1, "arm": arm, "cell": cell, "width": width,
                              "found": [r["found"] for r in results],
                              "ttft": [round(r["ttft_s"], 2) if r.get("ttft_s") else None for r in results],
                              "decode": [round(r["decode_tok_s"], 1) if r.get("decode_tok_s") else None
                                         for r in results]}), flush=True)
            write_atomic(seg / "segment.json", row)
        row["final_status_metrics"] = server.call("/metrics", timeout=30)[:200000]
    finally:
        server.stop()
    after = memory()
    row.update(thermal_after=thermal(), memory=[before, after], swapouts_delta=after["swapouts"] - before["swapouts"])
    write_atomic(seg / "segment.json", row)
    return row


def plan(cells, widths, max_batch_tokens):
    """(cell, width) pairs whose total prompt tokens fit the batch budget (KV headroom)."""
    return [(c, w) for c in cells for w in widths if w == 1 or c * w <= max_batch_tokens]


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(sum(values) / len(values), 3) if values else None


def summarize(segments):
    by_arm = {"A": [s for s in segments if s["arm"] == "A"], "B": [s for s in segments if s["arm"] == "B"]}
    keys = [(c["cell"], c["width"]) for c in segments[0]["cells"]]
    out = []
    for cell, width in keys:
        def cells(arm):
            return [next(c for c in s["cells"] if (c["cell"], c["width"]) == (cell, width)) for s in by_arm[arm]]
        ca, cb = cells("A"), cells("B")
        row = {"cell": cell, "width": width}
        for arm, cs in (("A", ca), ("B", cb)):
            reqs = [r for c in cs for r in c["requests"]]
            row[arm] = {"ttft_s": mean(r.get("ttft_s") for r in reqs),
                        "decode_tok_s": mean(r.get("decode_tok_s") for r in reqs),
                        "aggregate_tok_s": mean(c["aggregate_tok_s"] for c in cs),
                        "found": all(r["found"] for r in reqs), "errors": [r["error"] for r in reqs if "error" in r]}
        for metric in ("decode_tok_s", "aggregate_tok_s", "ttft_s"):
            a, b = row["A"][metric], row["B"][metric]
            row[f"B/A_{metric}"] = round(b / a, 3) if a and b else None
        row["B_vs_A"] = [classify(x, y) for x, y in zip(ca[0]["requests"], cb[0]["requests"])]
        row["A_vs_A"] = [classify(x, y) for x, y in zip(ca[0]["requests"], ca[1]["requests"])] if len(ca) > 1 else []
        row["B_vs_B"] = [classify(x, y) for x, y in zip(cb[0]["requests"], cb[1]["requests"])] if len(cb) > 1 else []
        out.append(row)
    return out


def attribute(segments):
    """Every segment's cells against the first A segment: token verdict and decode/TTFT ratios."""
    ref = next(s for s in segments if s["arm"] == "A")
    rows = []
    for seg in segments[1:] if segments[0] is ref else segments:
        if seg is ref:
            continue
        for c in seg["cells"]:
            base = next(x for x in ref["cells"] if (x["cell"], x["width"]) == (c["cell"], c["width"]))
            ratio = {m: round(mean(r.get(m) for r in c["requests"]) / mean(r.get(m) for r in base["requests"]), 3)
                     if mean(r.get(m) for r in base["requests"]) and mean(r.get(m) for r in c["requests"]) else None
                     for m in ("decode_tok_s", "ttft_s")}
            # At width > 1 the per-stream decode mean is not throughput: an arm
            # that admits lanes one at a time decodes each stream beside more
            # others and shows lower per-stream rates at the same wall clock.
            # Compare the cell's generated tokens over its wall time instead.
            ratio["aggregate_tok_s"] = (round(c["aggregate_tok_s"] / base["aggregate_tok_s"], 3)
                                        if c.get("aggregate_tok_s") and base.get("aggregate_tok_s") else None)
            rows.append({"segment": seg["segment"], "arm": seg["arm"], "cell": c["cell"], "width": c["width"],
                         "vs_A": [classify(x, y) for x, y in zip(base["requests"], c["requests"])],
                         "ratio_vs_A": ratio})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cells", default="3000,6000,16000,32000")
    p.add_argument("--widths", default="1,4")
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--main-sha", help="origin/main commit the run's branch was merged onto (recorded)")
    p.add_argument("--arms", type=Path, help="JSON {name: {label, flags, policy, env}} replacing A/B "
                   "(attribution runs; arm A must stay the reference)")
    p.add_argument("--order", help="comma-separated segment order (default A,B,B,A)")
    p.add_argument("--max-batch-tokens", type=int, default=70000,
                   help="skip width>1 cells whose total prompt tokens exceed this (bf16: 32K x 4 swapped)")
    args = p.parse_args()
    order = tuple(args.order.split(",")) if args.order else ORDER
    if args.arms:
        ARMS.clear()
        ARMS.update(json.loads(args.arms.read_text()))
    if "A" not in order or set(order) - set(ARMS):
        raise SystemExit(f"order {order} needs reference arm A and only defined arms {sorted(ARMS)}")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing nonempty evidence directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from transformers import AutoTokenizer

    from mlx2.adapters.registry import inspect_model

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    widths = [int(w) for w in args.widths.split(",")]
    cell_prompts = {int(c): prompts(tokenizer, int(c), max(widths)) for c in args.cells.split(",")}
    fingerprint = inspect_model(args.model).artifact_fingerprint
    receipt = {"schema": "mlx2.maxwell-abba-ladder.v1", **revision(), "origin_main": args.main_sha,
               "checkpoint": f"{args.model.name}@sha256:{fingerprint}", "order": order, "arms": ARMS,
               "widths": widths, "cells": list(cell_prompts), "max_tokens": args.max_tokens,
               "max_batch_tokens": args.max_batch_tokens,
               "skipped": [(c, w) for c in cell_prompts for w in widths
                           if (c, w) not in plan(cell_prompts, widths, args.max_batch_tokens)],
               "thermal_hold": False, "publishable": False, "segments": []}
    write_atomic(out / "ladder.json", receipt)
    for i, arm in enumerate(order):
        receipt["segments"].append({"arm": arm, "dir": f"seg{i + 1}-{arm}"})
        seg = run_segment(args.model, arm, i, out, cell_prompts, widths, args.max_tokens, args.max_batch_tokens)
        receipt["segments"][-1].update({k: seg.get(k) for k in ("checkpoint", "load_status", "swapouts_delta",
                                                                "thermal_before", "thermal_after")})
        write_atomic(out / "ladder.json", receipt)
        if seg["swapouts_delta"] > 50000:
            receipt["aborted"] = f"swap growth in segment {i + 1}"
            write_atomic(out / "ladder.json", receipt)
            raise SystemExit(receipt["aborted"])
    segments = [json.loads((out / s["dir"] / "segment.json").read_text()) for s in receipt["segments"]]
    receipt["request_failures"] = [
        {"segment": seg["segment"], "cell": cell["cell"], "lane": lane, "error": row.get("error", "needle missing")}
        for seg in segments for cell in seg["cells"] for lane, row in enumerate(cell["requests"])
        if "error" in row or not row["found"]
    ]
    if order != ORDER:
        receipt["attribution"] = attribute(segments)
        write_atomic(out / "ladder.json", receipt)
        for row in receipt["attribution"]:
            print(json.dumps(row), flush=True)
        return int(bool(receipt["request_failures"]))
    receipt["summary"] = summarize(segments)
    write_atomic(out / "ladder.json", receipt)
    for row in receipt["summary"]:
        print(json.dumps({k: row[k] for k in ("cell", "width", "B/A_decode_tok_s", "B/A_aggregate_tok_s",
                                              "B/A_ttft_s")} | {"B_vs_A": [v["verdict"] for v in row["B_vs_A"]]}),
              flush=True)

    return int(bool(receipt["request_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
