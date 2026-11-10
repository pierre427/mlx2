"""One-artifact P-core-share probe: shell-launched vs launchd-launched server.

Run inside the GPU lock (gpuq2). For each arm: start one loopback mlx2 server
on the artifact, warm it, run ``probe_pcore_share.py`` against its pid for
``--repeats`` steady B1 decodes, capture /v1/status, and stop the whole
process group (or ``launchctl bootout``) in ``finally``. Refuses a nonempty
evidence directory.

The launchd arm uses a temporary LaunchAgent with no ``ProcessType``, the
same shape as the production plists in ~/Library/LaunchAgents, and removes
it afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The interpreter running this probe (the venv python under gpuq2); absolute,
# so the temporary LaunchAgent arm can exec it too.
PY = os.environ.get("MLX2_PROBE_PYTHON") or sys.executable
if not os.path.isabs(PY):
    # launchd resolves no relative path against this cwd: refuse rather than
    # let the LaunchAgent arm exec the wrong (or no) interpreter.
    raise SystemExit(f"MLX2_PROBE_PYTHON must be an absolute path, got {PY!r}")
LABEL = "com.example.mlx2-pcore-probe"


def get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(base: str, proc_alive, limit_s: float = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if not proc_alive():
            raise RuntimeError("server exited before /health")
        try:
            get(base + "/health", 2)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("server never became healthy")


def server_argv(model: str, port: int) -> list[str]:
    return [PY, "-u", "-m", "mlx2.server", "--model", model, "--host", "127.0.0.1",
            "--port", str(port), "--max-context", "32768", "--max-lanes", "4",
            "--max-inflight", "8"]


def measure(base: str, pid: int, out: Path, model_id: str, repeats: int, max_tokens: int) -> None:
    # Warm: compiles kernels and fills APCv2 for the prompt.
    subprocess.run([PY, "-I", str(ROOT / "scripts/probe_pcore_share.py"), "--pid", str(pid),
                    "--url", base, "--model", model_id, "--max-tokens", "64",
                    "--label", "warm", "--out", str(out / "warm.json")], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run([PY, "-I", str(ROOT / "scripts/probe_pcore_share.py"), "--pid", str(pid),
                    "--url", base, "--model", model_id, "--max-tokens", str(max_tokens),
                    "--repeats", str(repeats), "--label", out.name,
                    "--out", str(out / "probe.json")], check=True, stdout=subprocess.DEVNULL)
    (out / "status.json").write_text(json.dumps(get(base + "/v1/status", 30), indent=1, default=str))


def arm_shell(args, out: Path) -> None:
    port = args.port
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1"}
    log = open(out / "server.log", "w")
    proc = subprocess.Popen(server_argv(args.model, port), cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        base = f"http://127.0.0.1:{port}"
        wait_health(base, lambda: proc.poll() is None)
        model_id = get(base + "/v1/models")["data"][0]["id"]
        measure(base, proc.pid, out, model_id, args.repeats, args.max_tokens)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(60)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)


def arm_launchd(args, out: Path) -> None:
    port = args.port + 1
    plist = out / f"{LABEL}.plist"
    plistlib.dump({
        "Label": LABEL,
        "ProgramArguments": server_argv(args.model, port),
        "EnvironmentVariables": {"PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
                                 "PYTHONUNBUFFERED": "1"},
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": False,
        "StandardOutPath": str(out / "server.log"),
        "StandardErrorPath": str(out / "server.log"),
    }, plist.open("wb"))
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True)
    try:
        pid = None
        for _ in range(60):
            text = subprocess.run(["launchctl", "print", f"{domain}/{LABEL}"],
                                  capture_output=True, text=True).stdout
            for line in text.splitlines():
                if line.strip().startswith("pid ="):
                    pid = int(line.split("=")[1])
            if pid:
                break
            time.sleep(1)
        if not pid:
            raise RuntimeError("launchd job has no pid")
        (out / "launchctl_print.txt").write_text(text)
        base = f"http://127.0.0.1:{port}"
        wait_health(base, lambda: subprocess.run(["kill", "-0", str(pid)]).returncode == 0)
        model_id = get(base + "/v1/models")["data"][0]["id"]
        measure(base, pid, out, model_id, args.repeats, args.max_tokens)
    finally:
        subprocess.run(["launchctl", "bootout", f"{domain}/{LABEL}"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--port", type=int, default=8391)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--arms", default="shell,launchd,shell,launchd")
    args = ap.parse_args()
    ev = Path(args.evidence)
    if ev.exists() and any(ev.iterdir()):
        sys.exit(f"refusing nonempty evidence dir {ev}")
    ev.mkdir(parents=True, exist_ok=True)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (ev / "receipt.json").write_text(json.dumps({"model": args.model, "source_revision": rev,
                                                 "arms": args.arms, "started": time.time()}))
    for i, arm in enumerate(args.arms.split(",")):
        out = ev / f"{i}-{arm}"
        out.mkdir()
        print(f"arm {out.name} start {time.strftime('%T')}", flush=True)
        (arm_shell if arm == "shell" else arm_launchd)(args, out)
        print(f"arm {out.name} done {time.strftime('%T')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
