#!/usr/bin/env python3
"""Gemma 4 decode A/B: served D=512 decode masks and the TK 2 kernel, three arms.

Original mlx2 script (recon 2026-11-22 item 2).  Served Gemma 4 decode hands
its head_dim-512 full-attention layers an all-true array mask, so MLX takes
its unfused path and never ``sdpa_vector_2pass`` (the first run of this A/B:
the kernel counter stayed 0).  ``MLX2_DECODE_MASK=none`` drops that mask on
an unpadded cache; ``MLX2_SDPA_D512_TK2=1`` then replaces MLX's TK 4 vector
kernel with the TK 2 one (``src/mlx2/runtime/models/sdpa_vector_d512.py``).

Arms, one loopback ``mlx2.server`` each (never :8282), ``--ordinary``,
``--max-lanes 1``, same arguments; only the environment differs:

  stock          neither variable (today's serving: array mask, unfused)
  maskless       MLX2_DECODE_MASK=none (MLX fused vector decode, TK 4)
  maskless_tk2   MLX2_DECODE_MASK=none MLX2_SDPA_D512_TK2=1

maskless_tk2 must equal maskless token for token (bitwise by construction).
maskless vs stock is a numerical change: the first divergence is classified
from top-5 windows with ``mlx2.near_tie`` (AGENTS.md near-tie rule, oracle
on both continuations); only an unclassified or high-margin row fails.

Per cell (prompt length) and arm: APCv2 cleared, then a cold and a warm
streaming greedy request (``reasoning_effort`` none, logprobs) with a buried
access code in varied repository text (never a repeated string).  Recorded:
TTFT, decode tok/s, the functional oracle on both channels, token ids, the
terminal receipt, and the kernel's dispatch counter from ``/v1/status``
(engagement).  Verdict: tk2 token ids equal stock per cell and request, the
oracle passes, the counter moved in tk2 and stayed zero in stock.

The caller owns the GPU (``cpg_job.py`` lease -> ``run_with_gpu_locks.py``).
Sanity numbers, one repetition, no thermal hold: not publishable statistics.
Refuses a nonempty output directory and port 8282.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hy4_context_ladder as H  # noqa: E402
from inkling2_context_ladder import lane_prompts, oracle_channels, stream_one  # noqa: E402

SCHEMA = "mlx2.gemma4-d512-decode-ab.v2"
ARMS = ({"name": "stock", "env": {}},
        {"name": "maskless", "env": {"MLX2_DECODE_MASK": "none"}},
        {"name": "maskless_tk2", "env": {"MLX2_DECODE_MASK": "none", "MLX2_SDPA_D512_TK2": "1"}})
MANAGED_ENV = ("MLX2_DECODE_MASK", "MLX2_SDPA_D512_TK2")
CEILING = 0.5


def write_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    tmp.replace(path)


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def counter(status) -> dict:
    section = H.find_key(status, "d512_decode_attention") or {}
    counts = section.get("counts") or {}
    return {"selected": section.get("selected"), "loaded": section.get("loaded", True),
            "decode_mask_mode": section.get("decode_mask_mode"),
            "calls": counts.get("d512_decode_calls", 0),
            "maskless": counts.get("maskless_decode_masks", 0)}


def wait_ready(base, server, timeout_s=1200):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited rc={server.returncode} before readiness")
        try:
            status = H.get_json(base, "/v1/status")
            if status.get("healthy"):
                return status
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("server readiness timed out")


def request_body(model_name, text, max_tokens):
    return {"model": model_name, "messages": [{"role": "user", "content": text}],
            "stream": True, "temperature": 0, "seed": 0, "max_tokens": max_tokens,
            "reasoning_effort": "none", "logprobs": True, "top_logprobs": 5}


def run_arm(a, arm, plan, out, model_name):
    pdir = out / arm["name"]
    pdir.mkdir()
    token = secrets.token_hex(24)
    token_file = pdir / "admin-token"
    token_file.write_text(token + "\n")
    token_file.chmod(0o600)
    cmd = [str(a.server_python), "-u", "-m", "mlx2.server", "--model", str(a.model),
           "--host", "127.0.0.1", "--port", str(a.port), "--ordinary",
           "--max-lanes", "1", "--max-inflight", "1",
           "--max-context", str(max(c["cell"] for c in plan) + 2048),
           "--admin-token-file", str(token_file), "--no-cache-dir"]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", **arm["env"]}
    for name in MANAGED_ENV:
        if name not in arm["env"]:
            env.pop(name, None)
    base = f"http://127.0.0.1:{a.port}"
    row = {"name": arm["name"], "env": arm["env"], "command": cmd, "cells": []}
    log = (pdir / "server.log").open("w")
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                              cwd=ROOT, start_new_session=True)
    try:
        status = wait_ready(base, server)
        row["checkpoint"] = H.checkpoint_label(a.model.name, status)
        row["counter_at_ready"] = counter(status)
        for cell in plan:
            H.post_json(base, "/v1/admin/apc/clear", {}, token)
            before = counter(H.get_json(base, "/v1/status"))
            entry = {"cell": cell["cell"], "prompt_sha256": cell["prompt_sha256"],
                     "code": cell["code"], "swap_mb_before": H.swap_used_mb()}
            body = request_body(status.get("model") or model_name, cell["text"], a.max_tokens)
            for shape in ("cold", "warm"):
                result = stream_one(base, body, timeout=a.request_timeout)
                receipt = result.pop("receipt")
                WINDOWS[(arm["name"], cell["cell"], shape)] = result.pop("windows", None)
                result["oracle"] = oracle_channels(result, cell["code"])
                result["cached_tokens"] = H.find_key(receipt, "cached_tokens")
                result["prompt_tokens"] = H.find_key(receipt, "prompt_tokens")
                result["route"] = H.find_key(receipt, "route")
                entry[shape] = result
            after = counter(H.get_json(base, "/v1/status"))
            entry["counter_delta"] = after["calls"] - before["calls"]
            entry["maskless_delta"] = after["maskless"] - before["maskless"]
            entry["cold_warm_tokens_equal"] = entry["cold"]["token_ids"] == entry["warm"]["token_ids"]
            row["cells"].append(entry)
            print(json.dumps({"arm": arm["name"], "cell": cell["cell"],
                              "decode_tok_s": [entry[s]["decode_tok_s"] for s in ("cold", "warm")],
                              "ttft_s": entry["cold"]["ttft_s"],
                              "oracle": [entry[s]["oracle"]["pass"] for s in ("cold", "warm")],
                              "kernel_calls": entry["counter_delta"],
                              "maskless_masks": entry["maskless_delta"]}), flush=True)
        status = H.get_json(base, "/v1/status")
        row["counter_final"] = counter(status)
        (pdir / "status-final.json").write_text(json.dumps(status, indent=1, default=str))
        row["passed_run"] = True
        (pdir / "windows.json").write_text(json.dumps(
            {f"{c}:{s}": w for (n, c, s), w in WINDOWS.items() if n == arm["name"]}))
    except Exception as error:  # the receipt says why; the next arm still runs
        row["passed_run"] = False
        row["error"] = repr(error)
    finally:
        row["server_stop"] = H.stop_group(server)
        log.close()
    return row


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


WINDOWS = {}


def classify(ref, other, ref_name, other_name, cell, shape):
    """First divergence of two greedy streams, near-tie classified (AGENTS.md)."""
    from mlx2.near_tie import near_tie_equivalent

    a, b = ref[shape]["token_ids"], other[shape]["token_ids"]
    i = first_divergence(a, b)
    if i is None:
        return {"diverged": False}
    out = {"diverged": True, "index": i}
    if i >= min(len(a), len(b)):
        out["verdict"] = "length_mismatch"
        return out
    wa = (WINDOWS.get((ref_name, cell, shape)) or [None] * (i + 1))[i]
    wb = (WINDOWS.get((other_name, cell, shape)) or [None] * (i + 1))[i]
    out.update(ref_token=a[i], other_token=b[i], ref_window=wa, other_window=wb)
    if not wa or not wb:
        out["verdict"] = "unclassified"
        return out
    x, y = a[i], b[i]
    if y not in wa or x not in wb:
        far = (y not in wa and max(wa.values()) - min(wa.values()) > CEILING) or (
            x not in wb and max(wb.values()) - min(wb.values()) > CEILING)
        out["verdict"] = "high_margin" if far else "unclassified"
        return out
    if not near_tie_equivalent(x, wa, y, wb, ceiling=CEILING):
        out["verdict"] = "high_margin"
    elif ref[shape]["oracle"]["pass"] and other[shape]["oracle"]["pass"]:
        out["verdict"] = "near_tie_equivalent"
    else:
        out["verdict"] = "near_tie_oracle_failed"
    return out


def verdict(arms):
    by = {r["name"]: r for r in arms}
    names = [x["name"] for x in ARMS]
    rows, failures = [], []
    if not all(by.get(n) and by[n].get("passed_run") for n in names):
        done = [n for n in names if by.get(n) and by[n].get("passed_run")]
        return {"passed": False, "failures": [f"arms incomplete; completed: {done}"], "cells": rows}
    if len({by[n].get("checkpoint") for n in names}) != 1 or not by["stock"].get("checkpoint"):
        failures.append("arms did not identify the same checkpoint")

    def prompts(arm):
        return [(c["cell"], c["prompt_sha256"]) for c in arm["cells"]]
    if not by["stock"]["cells"] or any(prompts(by[n]) != prompts(by["stock"]) for n in names):
        return {"passed": False, "failures": failures + ["missing or mismatched cells"], "cells": rows}
    for cells in zip(*(by[n]["cells"] for n in names)):
        c = dict(zip(names, cells))
        cell = c["stock"]["cell"]
        row = {"cell": cell, "kernel_calls": {}, "maskless_masks": {}}
        for n in names:
            e = c[n]
            if e["cold"].get("cached_tokens") != 0 or (e["warm"].get("cached_tokens") or 0) <= 0:
                failures.append(f"{cell}:{n}: cold/warm cache evidence missing or incorrect")
            if not e["cold"]["token_ids"] or e["cold"]["token_ids"] != e["warm"]["token_ids"]:
                failures.append(f"{cell}:{n}: width-one replay failed")
            for shape in ("cold", "warm"):
                if not e[shape]["oracle"]["pass"]:
                    failures.append(f"{cell}:{shape}:{n}: oracle failed")
            row["kernel_calls"][n] = e["counter_delta"]
            row["maskless_masks"][n] = e["maskless_delta"]
        if c["stock"]["counter_delta"] or c["stock"]["maskless_delta"]:
            failures.append(f"{cell}: stock arm dropped masks or dispatched the kernel")
        if c["maskless"]["maskless_delta"] <= 0 or c["maskless_tk2"]["maskless_delta"] <= 0:
            failures.append(f"{cell}: a maskless arm never dropped the decode mask")
        if c["maskless"]["counter_delta"]:
            failures.append(f"{cell}: maskless arm dispatched the TK 2 kernel")
        if c["maskless_tk2"]["counter_delta"] <= 0:
            failures.append(f"{cell}: maskless_tk2 arm never dispatched the kernel")
        for shape in ("cold", "warm"):
            exact = first_divergence(c["maskless"][shape]["token_ids"], c["maskless_tk2"][shape]["token_ids"])
            row[f"{shape}_tk2_vs_maskless_divergence"] = exact
            if exact is not None:
                failures.append(f"{cell}:{shape}: TK 2 differs from maskless at {exact} (kernel defect)")
            cls = classify(c["stock"], c["maskless"], "stock", "maskless", cell, shape)
            row[f"{shape}_maskless_vs_stock"] = cls
            if cls["diverged"] and cls.get("verdict") != "near_tie_equivalent":
                failures.append(f"{cell}:{shape}: maskless vs stock {cls.get('verdict')} at {cls['index']}")
            sd = c["stock"][shape]["decode_tok_s"]
            row[f"{shape}_decode_tok_s"] = {n: c[n][shape]["decode_tok_s"] for n in names}
            row[f"{shape}_speedup_vs_stock"] = {
                n: (c[n][shape]["decode_tok_s"] / sd) if sd and c[n][shape]["decode_tok_s"] else None
                for n in names[1:]}
        rows.append(row)
    return {"passed": not failures, "failures": failures, "cells": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8398)
    ap.add_argument("--cells", default="4096,16384,65536")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--server-python", type=Path,
                    default=ROOT / "scripts/research/memwatched_python")
    ap.add_argument("--arms", default=",".join(x["name"] for x in ARMS))
    a = ap.parse_args(argv)
    if a.port == 8282:
        raise SystemExit("port 8282 is reserved for release trees")
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise SystemExit("refusing nonempty evidence directory")
    if not port_free(a.port):
        raise SystemExit(f"port {a.port} is occupied")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(a.model))
    plan = []
    for index, cell in enumerate(int(c) for c in a.cells.split(",")):
        lane = lane_prompts(tokenizer, index, cell, 1)[0]
        plan.append({"cell": cell, **lane})
    receipt = {"schema": SCHEMA, "qualified": False, "model": str(a.model),
               "driver_revision": H.git("rev-parse", "HEAD"),
               "dirty": bool(H.git("status", "--porcelain", "--untracked-files=no")),
               "kind": "greedy, B1, --ordinary, one repetition, no thermal hold, "
                       "sanity under system contention: timings are not performance evidence",
               "cells": [{k: v for k, v in c.items() if k != "text"} for c in plan],
               "swap_mb_before": H.swap_used_mb(), "arms": []}
    path = a.out_dir / "receipt.json"
    write_atomic(path, receipt)
    wanted = a.arms.split(",")
    for arm in [x for x in ARMS if x["name"] in wanted]:
        receipt["arms"].append(run_arm(a, arm, plan, a.out_dir, a.model.name))
        receipt["swap_mb_after"] = H.swap_used_mb()
        write_atomic(path, receipt)
    receipt["checkpoint"] = next((r.get("checkpoint") for r in receipt["arms"] if r.get("checkpoint")), None)
    receipt["verdict"] = verdict(receipt["arms"])
    write_atomic(path, receipt)
    print(json.dumps(receipt["verdict"], indent=1), flush=True)
    return 0 if receipt["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
