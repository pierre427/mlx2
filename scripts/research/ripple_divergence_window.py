#!/usr/bin/env python3
"""Bounded first-divergence log-prob windows for a Ripple classification plan.

One loopback server per plan arm (control first), greedy, ``max_tokens`` =
first divergent position + 1, ``top_logprobs`` = 5. Each response must
reproduce the arm's original tokens through that position, or the row is
recorded as unreproduced (logprob serialization may perturb scheduling).
Only the windows at plan positions are kept. Requires the caller's CPG lease
and paired locks; every server process group is stopped in ``finally``.
"""
import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def window(entry):
    top = [(int(item["id"]), float(item["logprob"])) for item in entry.get("top_logprobs", [])]
    return {"selected": int(entry["id"]), "selected_logprob": float(entry["logprob"]), "top": top}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=18297)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from qualification_gpu_ownership import require_qualification_lease

    owner = json.loads((args.cpg_lock / "owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    from mlx2.adapters.ripple import inspect_artifact

    plan = json.loads(args.plan.read_text())
    receipt = {"schema": "mlx2.ripple-divergence-window.v1",
               "checkpoint": args.model.name + "@sha256:" + inspect_artifact(args.model)["identity"]["fingerprint"],
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
               "ownership": lease, "qualified": False, "arms": {}}
    token_file = args.output / "admin-token"
    token_file.write_text(secrets.token_hex(24))
    token_file.chmod(0o600)
    root = f"http://127.0.0.1:{args.port}"

    def call(path, payload=None, timeout=1800):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(root + path, data=data, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)

    # Control: one bounded request per cell, long enough for every arm's row.
    work = {"control": {}}
    for row in plan["rows"]:
        work.setdefault(row["arm"], {}).setdefault(row["cell"], []).append(row)
        work["control"].setdefault(row["cell"], []).append(row)
    try:
        for arm in ["control", *[name for name in work if name != "control"]]:
            out = args.output / arm
            out.mkdir()
            command = [args.python, "-u", "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
                       "--port", str(args.port), "--max-lanes", "1", "--max-inflight", "1",
                       "--max-context", "16384", "--cache-bytes", str(1 << 30), "--cache-dir", str(out / "cache"),
                       "--admin-token-file", str(token_file), *plan["servers"][arm]]
            log = (out / "server.log").open("w")
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                      env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)
            record = {"command": command, "rows": []}
            try:
                deadline = time.monotonic() + 900
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"{arm} server exited before readiness")
                    try:
                        if call("/v1/status", timeout=10).get("healthy"):
                            break
                    except (URLError, TimeoutError, OSError):
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{arm} readiness timeout")
                    time.sleep(.5)
                for cell, rows in work[arm].items():
                    for row in rows:
                        p = row["position"]
                        # Bounded window: only completion position p + 1 crosses
                        # to the host (logprobs.wants_logprobs_at).
                        result = call("/v1/chat/completions", {
                            "model": args.model.name, "temperature": 0, "stream": False, "max_tokens": p + 1,
                            "logprobs": True, "top_logprobs": 5, "logprobs_start": p + 1, "logprobs_end": p + 1,
                            "return_token_trace": True, "skip_writing_prefix_cache": True,
                            "messages": [{"role": "user", "content": plan["cells"][cell]}]})
                        wire = result.get("mlx2") or {}
                        got = [int(t) for t in (wire.get("token_trace") or {}).get("token_ids", [])]
                        content = (result["choices"][0].get("logprobs") or {}).get("content") or []
                        expected = (row["control_tokens"] if arm == "control" else row["arm_tokens"])[: p + 1]
                        record["rows"].append({
                            "row_arm": row["arm"], "cell": cell, "position": p,
                            "reproduced": got[: p + 1] == expected,
                            "first_mismatch": next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None),
                            "window_entries": len(content),
                            "window": window(content[-1]) if content else None,
                            "route": wire.get("route")})
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait(timeout=10)
                log.close()
                record["server_stopped"] = server.poll() is not None
                receipt["arms"][arm] = record
                tmp = args.output / "receipt.tmp"
                tmp.write_text(json.dumps(receipt, indent=1) + "\n")
                os.replace(tmp, args.output / "receipt.json")
            print(json.dumps({"arm": arm, "reproduced": [r["reproduced"] for r in record["rows"]]}), flush=True)
    finally:
        token_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
