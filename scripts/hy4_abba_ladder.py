#!/usr/bin/env python3
"""Hy4-flash ABBA performance ladder: two option sets, interleaved A,B,B,A, per context cell and width.

Pierre 2026-11-22 (17:25): A = the current default on the best Hy4 route, B = the best measured
option set. One loopback server per arm run (four runs, in ABBA order, identical server arguments
apart from the arm's execution policy), greedy, the needle prompts of ``hy4_context_ladder.py``
(byte-identical across arms). Per cell, APCv2 is cleared and every lane sends one cold request;
the record keeps client TTFT, prefill and decode tok/s, the needle oracle, metal peak, swap and a
thermal reading (``scripts/thermal_probe.swift``).

Exactness per cell and lane: A's tokens are compared with B's. Equal -> ``token_exact``. Otherwise
the first divergence is located and each arm, in a later run, sends one untimed width-one request
with ``top_logprobs`` and keeps only that position's window. The row is ``near_tie_equivalent``
when ``mlx2.near_tie.near_tie_equivalent`` accepts the two windows, the prefix is shared and both
continuations pass the oracle; anything else stays ``unclassified`` (or ``high_margin`` when both
windows exist and the rule rejects them). Same-arm run-to-run differences (A1 vs A4, B2 vs B3) are
reported separately. No thermal hold: sanity/perf A/B numbers, unqualified, not publishable.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from hy4_context_ladder import (
    DASHES,
    DEPTHS,
    build_prompt,
    checkpoint_label,
    find_key,
    get_json,
    git,
    make_code,
    post_json,
    stop_group,
    stream_one,
    swap_used_mb,
)

ORDER = ("A", "B", "B", "A")


def thermal() -> dict | None:
    try:
        return json.loads(subprocess.check_output(["swift", str(ROOT / "scripts/thermal_probe.swift")],
                                                  text=True, timeout=60))
    except Exception as error:  # noqa: BLE001 - a missing reading is recorded, never fatal
        return {"error": f"{type(error).__name__}: {error}"}


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def window_at(top: list, index: int) -> dict | None:
    """{token id: logprob} at ``index`` from a stream's per-token logprob rows (selected + top)."""
    if index >= len(top) or top[index] is None:
        return None
    row = top[index]
    out = {int(row["id"]): float(row["logprob"])}
    for alt in row.get("top_logprobs") or ():
        out[int(alt["id"])] = float(alt["logprob"])
    return out


def stream_top(base: str, body: dict, timeout: float) -> tuple[list[int], list]:
    """One untimed greedy request with top_logprobs; returns (token ids, logprob rows)."""
    import urllib.request
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    ids, rows = [], []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            event = json.loads(line[5:].strip())
            if "error" in event:
                raise RuntimeError(f"generation SSE error: {event['error']}")
            for choice in event.get("choices", ()):
                for item in (choice.get("logprobs") or {}).get("content", ()) or ():
                    ids.append(int(item["id"]))
                    rows.append(item)
    return ids, rows


def cell_plan(b1: list[int], wide: list[int], width: int) -> list[tuple[int, int]]:
    return [(c, 1) for c in b1] + ([(c, width) for c in wide] if width > 1 else [])


def prompts_for(tokenizer, cells: list[tuple[int, int]]) -> dict:
    """Deterministic needle prompts per (cell, width, lane); identical in every arm run."""
    out, offset = {}, 0
    for i, (cell, width) in enumerate(cells):
        for lane in range(width):
            depth = DEPTHS[(i + lane) % len(DEPTHS)]
            code = make_code(cell * 10 + lane + 1000 * width)
            text, est = build_prompt(tokenizer, cell, depth, code, offset)
            offset += 50_000
            out[(cell, width, lane)] = {"prompt": text, "code": code, "depth": depth, "estimate": est,
                                       "sha256": hashlib.sha256(text.encode()).hexdigest()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--policy-a", type=Path, required=True, help="execution policy JSON for arm A")
    ap.add_argument("--policy-b", type=Path, required=True, help="execution policy JSON for arm B")
    ap.add_argument("--route", choices=("ordinary", "ast"), default="ast")
    ap.add_argument("--cells", default="1024,4096,16384,32768,65536,131072", help="width-one cells")
    ap.add_argument("--wide-cells", default="1024,8192,32768", help="cells also run at --width")
    ap.add_argument("--width", type=int, default=4, help="batch width for --wide-cells (DFly2 cap 4)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--port", type=int, default=8287)
    ap.add_argument("--swap-stop-mb", type=float, default=4096.0)
    ap.add_argument("--load-timeout", type=float, default=900.0)
    ap.add_argument("--cell-timeout", type=float, default=3600.0)
    ap.add_argument("--server-arg", action="append", default=[], help="extra mlx2.server argument (both arms)")
    a = ap.parse_args(argv)
    if a.port == 8282:
        ap.error("port 8282 is reserved for release trees")
    out = a.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        ap.error(f"{out} is not empty")
    model = a.model.expanduser().resolve(strict=True)
    policies = {"A": json.loads(a.policy_a.read_text()), "B": json.loads(a.policy_b.read_text())}
    for arm, policy in policies.items():
        (out / f"policy-{arm}.json").write_text(json.dumps(policy, indent=2) + "\n")
    b1 = [int(c) for c in a.cells.split(",") if c]
    wide = [int(c) for c in a.wide_cells.split(",") if c]
    plan = cell_plan(b1, wide, a.width)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    prompts = prompts_for(tokenizer, plan)
    token_file = out / "admin-token"
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    token = secrets.token_hex(24)
    os.write(fd, (token + "\n").encode())
    os.close(fd)
    base = f"http://127.0.0.1:{a.port}"
    swap0 = swap_used_mb()
    report = {
        "schema": "mlx2.hy4-abba-ladder.v1",
        "kind": "ABBA perf A/B, greedy, one rep per arm run, no thermal hold; unqualified, not publishable",
        "source": {"commit": git("rev-parse", "HEAD"),
                   "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                   "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "model_path": str(model), "route": a.route, "policies": policies, "order": ORDER,
        "plan": [{"cell": c, "width": w} for c, w in plan], "max_tokens": a.max_tokens,
        "seed": a.seed, "reasoning_effort": a.reasoning_effort, "swap_used_mb_start": swap0,
        "thermal_start": thermal(), "started_at": datetime.now(UTC).isoformat(),
        "prompts": {f"{c}x{w}:{l}": {k: v for k, v in p.items() if k != "prompt"}
                    for (c, w, l), p in prompts.items()},
        "runs": [], "status": "running",
    }

    def save():
        tmp = out / "abba.json.partial"
        tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        os.replace(tmp, out / "abba.json")

    def body(key, top=False):
        b = {"messages": [{"role": "user", "content": prompts[key]["prompt"]}], "max_tokens": a.max_tokens,
             "seed": a.seed, "reasoning_effort": a.reasoning_effort, "stream": True, "logprobs": True,
             "temperature": 0}
        if top:
            b["top_logprobs"] = a.top_logprobs
        return b

    windows = {"A": {}, "B": {}}        # arm -> {(cell, width, lane, index): window}
    tokens = {"A": {}, "B": {}}         # arm -> {(cell, width, lane): first run's token ids}
    stopped = None
    for run_index, arm in enumerate(ORDER):
        run = {"run": run_index + 1, "arm": arm, "cells": []}
        report["runs"].append(run)
        policy_path = out / f"policy-{arm}.json"
        command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
                   "--port", str(a.port), f"--{a.route}", "--max-lanes", str(max(a.width, 1)),
                   "--max-inflight", str(max(a.width, 1)), "--max-context", str(max(b1 + wide) + 4096),
                   "--execution-policy", str(policy_path), "--admin-token-file", str(token_file),
                   "--no-cache-dir", *a.server_arg]
        run["server_command"] = command
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", a.port)) == 0:
                run["error"] = f"port {a.port} is occupied"
                stopped = run["error"]
                break
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        log = open(out / f"server-{run_index + 1}-{arm}.log", "w")
        server = None
        try:
            run["swap_used_mb_before_load"] = swap_before = swap_used_mb()
            t0 = time.monotonic()
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode} during load")
                try:
                    if get_json(base, "/health").get("status") == "ok":
                        break
                except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                    pass
                if time.monotonic() - t0 > a.load_timeout:
                    raise TimeoutError("server not healthy within the load timeout")
                time.sleep(1)
            run["load_seconds"] = round(time.monotonic() - t0, 1)
            status = get_json(base, "/v1/status")
            run["server_status_start"] = status
            run["checkpoint"] = checkpoint_label(model.name, status)
            report.setdefault("checkpoint", run["checkpoint"])
            # One untimed short request so kernel compilation does not land in the first timed cell.
            warm = {"messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 16,
                    "temperature": 0, "stream": True, "logprobs": True, "reasoning_effort": a.reasoning_effort}
            run["warmup_tokens"] = len(stream_one(base, warm, timeout=a.cell_timeout)["token_ids"])
            post_json(base, "/v1/admin/apc/clear", {"reason": "abba-warmup"}, token)
            # The swap stop measures the cells, not the ~85 GB load: rebase after load + warmup and record the
            # load's own growth (an eager Hy4 load can push swap for its first seconds).
            run_swap0 = swap_used_mb()
            run["load_swap_mb"] = round(run_swap0 - swap_before, 1)
            save()
            for cell, width in plan:
                row = {"cell": cell, "width": width, "lanes": []}
                run["cells"].append(row)
                row["apc_clear"] = post_json(base, "/v1/admin/apc/clear", {"reason": f"abba-{cell}x{width}"}, token)
                keys = [(cell, width, lane) for lane in range(width)]
                started = time.perf_counter()
                with cf.ThreadPoolExecutor(width) as pool:
                    results = list(pool.map(lambda k: stream_one(base, body(k), timeout=a.cell_timeout), keys))
                row["wall_s"] = round(time.perf_counter() - started, 3)
                completion = 0
                for key, result in zip(keys, results):
                    receipt = result.pop("receipt")
                    pt = receipt.get("prompt_tokens")
                    lane = {"lane": key[2], "token_ids": result["token_ids"], "text": result["text"],
                            "oracle_pass": prompts[key]["code"] in result["text"].translate(DASHES),
                            "prompt_tokens": pt, "cached_tokens": receipt.get("cached_tokens"),
                            "ttft_s": result["ttft_s"], "decode_tok_s": result["decode_tok_s"],
                            "prefill_tok_s": round(pt / result["ttft_s"], 1) if pt and result["ttft_s"] else None,
                            "finish_reason": result["finish_reason"], "route": find_key(receipt, "route"),
                            "speculation": find_key(receipt, "speculation")}
                    completion += len(result["token_ids"])
                    row["lanes"].append(lane)
                    tokens[arm].setdefault(key, result["token_ids"])
                row["aggregate_tok_s"] = round(completion / row["wall_s"], 2) if row["wall_s"] else None
                st = get_json(base, "/v1/status")
                row["metal_peak_gb"] = round((find_key(st, "metal_peak_bytes") or 0) / 2**30, 2)
                row["physical_footprint_gb"] = round((find_key(st, "process_physical_footprint_bytes") or 0) / 2**30, 2)
                row["swap_used_mb"] = swap_used_mb()
                row["thermal"] = thermal()
                print(json.dumps({"run": run_index + 1, "arm": arm, "cell": cell, "width": width,
                                  "ttft_s": [l["ttft_s"] and round(l["ttft_s"], 2) for l in row["lanes"]],
                                  "decode": [l["decode_tok_s"] and round(l["decode_tok_s"], 1) for l in row["lanes"]],
                                  "agg": row["aggregate_tok_s"], "oracle": [l["oracle_pass"] for l in row["lanes"]]}),
                      flush=True)
                save()
                if row["swap_used_mb"] - run_swap0 > a.swap_stop_mb:
                    stopped = f"swap grew {row['swap_used_mb'] - run_swap0:.0f} MB after {cell}x{width}"
                    break
            # Untimed divergence windows against the other arm's first run, one position each.
            other = "B" if arm == "A" else "A"
            probes = []
            for key, mine in list(tokens[arm].items()):
                theirs = tokens[other].get(key)
                if theirs is None:
                    continue
                d = first_divergence(mine, theirs)
                if d is None or (*key, d) in windows[arm]:
                    continue
                ids, rows = stream_top(base, body(key, top=True), a.cell_timeout)
                probe = {"key": list(key), "index": d, "probe_prefix_matches": ids[:d + 1] == mine[:d + 1]}
                if probe["probe_prefix_matches"]:
                    windows[arm][(*key, d)] = window_at(rows, d)
                probes.append(probe)
            run["divergence_probes"] = probes
        except Exception as error:  # noqa: BLE001 - recorded; the next arm run still goes
            run["error"] = f"{type(error).__name__}: {error}"
            print(f"[run {run_index + 1} {arm}] ERROR {run['error']}", flush=True)
        finally:
            run["server_stop"] = stop_group(server)
            log.close()
            save()
        if run.get("server_stop") == "still_running":
            stopped = "server process group did not stop"
        if stopped:
            break
    report["verdict"] = verdict(report, windows)
    report["status"] = f"stopped: {stopped}" if stopped else (
        "completed" if not any(r.get("error") for r in report["runs"]) else "completed_with_errors")
    report["thermal_end"] = thermal()
    report["swap_used_mb_end"] = swap_used_mb()
    save()
    print(json.dumps({"status": report["status"], "checkpoint": report.get("checkpoint"),
                      "summary": report["verdict"]["summary"]}, default=str), flush=True)
    return 0 if report["status"] == "completed" else 1


def verdict(report: dict, windows: dict) -> dict:
    from mlx2.near_tie import near_tie_equivalent
    runs = {(r["run"], r["arm"]): {(c["cell"], c["width"], l["lane"]): l for c in r["cells"] for l in c["lanes"]}
            for r in report["runs"]}
    arm_runs = {arm: [v for (n, a), v in sorted(runs.items()) if a == arm] for arm in ("A", "B")}
    rows, perf = [], {}
    keys = sorted(set().union(*[set(v) for v in runs.values()])) if runs else []
    for key in keys:
        la = [r[key] for r in arm_runs["A"] if key in r]
        lb = [r[key] for r in arm_runs["B"] if key in r]
        if not la or not lb:
            continue
        a0, b0 = la[0], lb[0]
        d = first_divergence(a0["token_ids"], b0["token_ids"])
        entry = {"cell": key[0], "width": key[1], "lane": key[2],
                 "same_arm_repeat_equal": {"A": all(x["token_ids"] == a0["token_ids"] for x in la),
                                           "B": all(x["token_ids"] == b0["token_ids"] for x in lb)},
                 "oracle": {"A": all(x["oracle_pass"] for x in la), "B": all(x["oracle_pass"] for x in lb)}}
        if d is None:
            entry["status"] = "token_exact"
        else:
            entry["first_divergence"] = d
            wa, wb = windows["A"].get((*key, d)), windows["B"].get((*key, d))
            entry["windows"] = {"A": wa, "B": wb}
            sa = a0["token_ids"][d] if d < len(a0["token_ids"]) else None
            sb = b0["token_ids"][d] if d < len(b0["token_ids"]) else None
            if wa is None or wb is None or sa is None or sb is None:
                entry["status"] = "unclassified"
            elif near_tie_equivalent(sa, wa, sb, wb) and entry["oracle"]["A"] and entry["oracle"]["B"]:
                entry["status"] = "near_tie_equivalent"
            else:
                entry["status"] = "high_margin"
        rows.append(entry)
        p = perf.setdefault(f"{key[0]}x{key[1]}", {"A": {}, "B": {}})
        for arm, lst in (("A", la), ("B", lb)):
            for field in ("ttft_s", "decode_tok_s", "prefill_tok_s"):
                vals = [x[field] for x in lst if x.get(field)]
                p[arm].setdefault(field, []).extend(vals)
    for cell_key, p in perf.items():
        for arm in ("A", "B"):
            p[arm] = {f: round(sum(v) / len(v), 3) if v else None for f, v in p[arm].items()}
        da, db = p["A"].get("decode_tok_s"), p["B"].get("decode_tok_s")
        ta, tb = p["A"].get("ttft_s"), p["B"].get("ttft_s")
        p["decode_B_over_A"] = round(db / da, 3) if da and db else None
        p["ttft_B_over_A"] = round(tb / ta, 3) if ta and tb else None
    aggregate = {}
    for r in report["runs"]:
        for c in r["cells"]:
            aggregate.setdefault(f"{c['cell']}x{c['width']}", {"A": [], "B": []})[r["arm"]].append(c.get("aggregate_tok_s"))
    counts = {}
    for e in rows:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return {"rows": rows, "perf_mean_of_runs": perf, "aggregate_tok_s_by_run": aggregate,
            "summary": {"statuses": counts,
                        "oracle_failures": [f"{e['cell']}x{e['width']}:{e['lane']}" for e in rows
                                            if not (e["oracle"]["A"] and e["oracle"]["B"])],
                        "failures": [f"{e['cell']}x{e['width']}:{e['lane']}" for e in rows
                                     if e["status"] in ("high_margin", "unclassified")]}}


if __name__ == "__main__":
    sys.exit(main())
