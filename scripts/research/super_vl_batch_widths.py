#!/usr/bin/env python3
"""Super VL NVFP4: served GSM8K at batch widths 1, 2 and 4.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them.  Per arm: one loopback server (``--max-lanes W``),
the first ``--questions`` GSM8K test questions (5-shot completion prompts,
greedy, the spec-acceptance harness prompts) sent ``W`` at a time from a
thread pool, so the scheduler holds up to ``W`` live sessions.  Recorded per
arm: aggregate completion tok/s (all completion tokens over the arm's wall
time), the median per-request decode tok/s, accuracy, and each completion's
text against the first arm (B1, the reference): exact-text agreement and,
where text differs, the first differing character offset.  The status
receipt is kept.  Width 3 is not offered: the adapter admits 1, 2 or 4.
Attribution only, no thermal hold; text differences at width > 1 are not
classified here (no logprob windows), so they are reported, not judged.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, str(ROOT / "src"))
import spec_acceptance_gsm8k as gsm  # noqa: E402
import super_vl_nvfp4_unit as unit_mod  # noqa: E402

rm = unit_mod.rm
LILICORR = str(unit_mod.LILICORR)
CHAIN_B = {"draft_model": LILICORR, "batch_size_route": "chain",
           "progressive_verification_tile": 4,
           "progressive_verification_fidelity": "exact",
           "proposal_composition": False}
ARMS = {
    "b1_tree": {"width": 1, "flags": []},
    "b1_chain": {"width": 1, "flags": ["--external-draft"],
                 "policy": {"draft_model": LILICORR, "batch_size_route": "chain"}},
    "b2_chain": {"width": 2, "flags": ["--external-draft"], "policy": CHAIN_B},
    "b4_chain": {"width": 4, "flags": ["--external-draft"], "policy": CHAIN_B},
    "b1_ordinary": {"width": 1, "flags": ["--ordinary"]},
}


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def run_arm(name, spec, out, args, model_id, ident, reference):
    out.mkdir(parents=True)
    width = spec["width"]
    flags = list(spec["flags"]) + ["--max-lanes", str(width)]
    if "policy" in spec:
        (out / "policy.json").write_text(json.dumps(spec["policy"], indent=1) + "\n")
        flags += ["--execution-policy", str(out / "policy.json")]
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(unit_mod.TARGET),
               "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
               "--max-context", "32768", "--cache-bytes", str(2 << 30), *flags]
    row = {"arm": name, "width": width, "command": command, "started_at": time.time()}
    base = f"http://127.0.0.1:{args.port}"
    log = (out / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                              start_new_session=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    try:
        row["load_seconds"] = rm.wait_health(base, server, 1800, lambda: None)
        _revision, files = gsm.dataset_files()
        train, test = gsm.load_split(files["train"]), gsm.load_split(files["test"])
        items = test[: args.questions]
        prefix = gsm.few_shot_prefix(train, 5)
        client = gsm.Client(base)
        # Warm (compiles, canaries) on a question outside the measured set.
        gsm.run_questions(client, model_id, prefix, test[-1:], chat=False, max_tokens=256)

        def one(index):
            rows = gsm.run_questions(client, model_id, prefix, [items[index]],
                                     chat=False, max_tokens=256)
            rows[0]["index"] = index
            return rows[0]

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=width) as pool:
            results = list(pool.map(one, range(len(items))))
        wall = time.monotonic() - started
        tokens = sum(r["completion_tokens"] for r in results)
        decode = [r["decode_tps"] for r in results if r["decode_tps"]]
        agree = []
        for r in results:
            ref = reference.get(r["index"]) if reference else None
            r["reference_offset"] = None if ref is None else first_difference(r["text"], ref)
            agree.append(ref is not None and r["text"] == ref)
        row.update({
            "wall_seconds": round(wall, 2), "completion_tokens": tokens,
            "aggregate_tok_s": round(tokens / wall, 2),
            "median_request_decode_tok_s": round(statistics.median(decode), 2) if decode else None,
            "accuracy": sum(r["correct"] for r in results) / len(results),
            "text_identical_to_reference": (sum(agree) if reference else None),
            "text_differs": [{"index": r["index"], "offset": r["reference_offset"]}
                             for r, same in zip(results, agree) if reference and not same],
            "rows": results,
        })
        row["status"] = rm.http_json(base + "/v1/status", None, timeout=30)[1]
        row["passed"] = True
        print(json.dumps({k: row[k] for k in ("arm", "width", "aggregate_tok_s",
                                              "median_request_decode_tok_s", "accuracy",
                                              "text_identical_to_reference")}), flush=True)
    except BaseException as error:  # noqa: BLE001 - recorded, then the next arm runs
        row["passed"] = False
        row["error"] = repr(error)
        print(json.dumps({"arm": name, "error": repr(error)}), flush=True)
    finally:
        row["server_stop"] = rm.stop_server(server)
        log.close()
        row["finished_at"] = time.time()
        unit_mod.atomic(out / "arm.json", row)
    return row


def unit(out, args):
    ident, adapter = unit_mod.checkpoint()
    report = {"schema": "mlx2.super-vl-batch-widths.v1", "checkpoint": ident,
              "adapter": adapter, "source": unit_mod.source(), "qualified": False,
              "evidence_kind": "served width attribution, no thermal hold", "arms": []}
    reference = None
    for name in args.arms:
        row = run_arm(name, ARMS[name], out / name, args, unit_mod.TARGET.name, ident, reference)
        if reference is None and row.get("passed"):
            reference = {r["index"]: r["text"] for r in row["rows"]}
            report["reference_arm"] = name
        report["arms"].append({k: v for k, v in row.items() if k not in ("rows", "status")})
        unit_mod.atomic(out / "report.json", report)
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    ident, _adapter = unit_mod.checkpoint()
    job_id = "super-vl-batch-widths-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=ident,
                         source_revision=unit_mod.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--arms", *args.arms, "--questions", str(args.questions),
                   "--port", str(args.port)]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--arms", nargs="+", default=["b1_tree", "b2_chain", "b4_chain"],
                        choices=list(ARMS))
    parser.add_argument("--questions", type=int, default=32)
    parser.add_argument("--port", type=int, default=8797)
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    sys.exit(main())
