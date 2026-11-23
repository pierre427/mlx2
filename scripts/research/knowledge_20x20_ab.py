#!/usr/bin/env python3
"""A/B the 20x20 domain-knowledge check across server arms (one loopback server per arm, one at a time).

Each arm is ``{"name", "args": [extra mlx2.server args], "policy": {execution policy} | null}``. Per arm: start
``mlx2.server`` on ``--port`` (never 8282), wait for /v1/status healthy, save the status, run
``knowledge_20x20.py`` into ``OUT/<arm>/``, stop the server's process group in ``finally``. Run under the caller's
CPG lease and paired GPU locks. Greedy, single rep, no thermal hold: sanity numbers, unqualified.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mlx2_deploy import port_listeners
from spec_acceptance_gsm8k import write_atomic


def status(url):
    return json.loads(urllib.request.urlopen(url + "/v1/status", timeout=10).read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--arms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8286)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    a.model = a.model.resolve()
    a.out = a.out.resolve()
    if os.path.sep in a.python:
        a.python = str(Path(a.python).resolve())
    if a.port == 8282:
        raise SystemExit("port 8282 is reserved for release trees")
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty")
    a.out.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{a.port}"
    rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"schema": "mlx2.knowledge-20x20-ab.v1", "qualification": "unqualified", "mlx2_source_revision": rev,
               "model": str(a.model), "concurrency": a.concurrency, "arms": []}
    rc_all = 0
    for arm in json.loads(a.arms.read_text()):
        out = a.out / arm["name"]
        out.mkdir()
        cmd = [a.python, "-u", "-m", "mlx2.server", "--model", str(a.model),
               "--no-cache-dir", *arm.get("args", []), "--host", "127.0.0.1", "--port", str(a.port)]
        if arm.get("policy") is not None:
            (out / "policy.json").write_text(json.dumps(arm["policy"]))
            cmd += ["--execution-policy", str(out / "policy.json")]
        log = (out / "server.log").open("w")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **arm.get("env", {})}
        server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=ROOT)
        row = {"name": arm["name"], "command": cmd}
        try:
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited rc={server.returncode} before readiness")
                try:
                    st = status(url)
                    if st.get("healthy"):
                        break
                except Exception:  # noqa: BLE001
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("server readiness timed out")
                time.sleep(1)
            if port_listeners(a.port) != {server.pid}:
                raise RuntimeError("healthy endpoint belongs to a different process")
            (out / "status_before.json").write_text(json.dumps(st, indent=1, default=str))
            t0 = time.time()
            rc = subprocess.call([a.python, str(ROOT / "scripts/research/knowledge_20x20.py"), str(out / "k20"),
                                  "--url", url, "--concurrency", str(a.concurrency), "--max-tokens", str(a.max_tokens)])
            row.update(rc=rc, wall_s=round(time.time() - t0, 1))
            (out / "status_after.json").write_text(json.dumps(status(url), indent=1, default=str))
            report_path = out / "k20/report.json"
            row["report"] = str(report_path) if report_path.is_file() else None
            if rc == 0:
                report = json.loads(report_path.read_text())
                row["request_errors"] = report["errors"]
                if report["errors"]:
                    raise RuntimeError(f"{report['errors']} knowledge requests failed")
            rc_all |= rc
        except Exception as error:  # noqa: BLE001
            row["error"] = repr(error)
            rc_all = 1
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(90)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            log.close()
        summary["arms"].append(row)
        write_atomic(a.out / "summary.json", summary)
        print(json.dumps(row), flush=True)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
