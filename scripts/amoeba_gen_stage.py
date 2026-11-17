#!/usr/bin/env python3
"""Front-of-night generation stage (Pierre 2026-11-15: 1 greedy + 3 at T=0.7 per carve prompt, full model, fast
path). Starts one mlx2 dev server for the FULL model with the given route flags (never port 8282), waits for
health, runs amoeba_onpolicy_gen.py against it, then stops the server's whole process group in finally.
Inside the caller's GPU window. Usage:
  amoeba_gen_stage.py --model M --route "--ast --ast-drafter D" --out DIR --corpus a.jsonl [--corpus b.jsonl] [gen args...]
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba27b_served as S  # noqa: E402  (wait_health, stop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--route", default="", help="server route flags (empty = the adapter's default best route)")
    ap.add_argument("--port", type=int, default=8395)
    ap.add_argument("--extra", default="--max-lanes 4 --cache-bytes 4294967296 --max-context 32768")
    a, gen_args = ap.parse_known_args()
    if not 8391 <= a.port <= 8395:
        sys.exit("development servers use ports 8391-8395 (never 8282)")
    from mlx2.adapters.registry import inspect_model
    fp = inspect_model(a.model).artifact_fingerprint
    ckpt = f"{Path(a.model).name}@sha256:{fp}"
    base = f"http://127.0.0.1:{a.port}"
    cmd = [sys.executable, "-m", "mlx2.server", "--model", a.model, "--port", str(a.port),
           *shlex.split(a.route), *shlex.split(a.extra)]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    import signal
    import socket

    def _term(signum, frame):           # a SIGTERM to this process group must still stop the server (finally)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _term)
    with socket.socket() as so:         # a server already on the port would answer /health for another model
        if so.connect_ex(("127.0.0.1", a.port)) == 0:
            sys.exit(f"port {a.port} is already in use: refusing (its /health would not be this model's)")
    out = Path(gen_args[gen_args.index("--out") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(str(out) + ".server.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        S.wait_health(base, 1800, proc)
        st = S.get(base, "/v1/status")
        route = (st.get("settings") or {}).get("route")
        print(json.dumps({"checkpoint": ckpt, "route": route, "server_cmd": cmd}), flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_onpolicy_gen.py"), "--base-url", base,
                            "--checkpoint", ckpt, *gen_args], cwd=ROOT, env=env)
        man = out / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            m.update({"server_route": route, "server_cmd": cmd})
            man.write_text(json.dumps(m, indent=1))
        sys.exit(r.returncode)
    finally:
        S.stop(proc)
        log.close()


if __name__ == "__main__":
    main()
