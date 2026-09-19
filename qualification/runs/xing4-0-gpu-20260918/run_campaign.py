"""Sequential GPU qualification campaign for the Xing4.0-29B-A4B port (2026-09-18).

One server at a time on port 8297.  Every stage records its steps in
status.json; a failed stage never stops the campaign.  The launchd production service was not
loaded for this window, so it is neither quiesced nor started.
"""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("MLX2_CAMPAIGN_ROOT", "/private/tmp/mlx2-xing"))
RUN = Path(__file__).resolve().parent
MODELS = Path("~/mlx-models")
PY = "~/Desktop/mlx2/.venv/bin/python"
PORT = 8297
BASE = f"http://127.0.0.1:{PORT}"
SERVICE = "com.example.fn-uncensored-mlx-serve"
PLIST = Path.home() / "Library/LaunchAgents" / f"{SERVICE}.plist"
STATUS = RUN / "status.json"
ENV = {**os.environ, "PYTHONPATH": "src", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
PREFLIGHT = RUN / "preflight.json"

X6 = "Xing4.0-29B-A4B-mlx-6bit"
X16 = "Xing4.0-29B-A4B-mlx-bf16"
POL = RUN / "policies"


def qualifier(name, *extra):
    return ("qualifier", [PY, "scripts/qualify_serving.py", "--url", BASE, "--output", str(RUN / f"{name}-qualification.json"),
                          "--preflight-receipt", str(PREFLIGHT), "--timeout", "1800", *extra], 3600)


def probe(script, name):
    return (script, [PY, str(RUN / f"{script}.py"), BASE, str(RUN / f"{name}-{script}.json")], 2400)


def stage(name, model, args, steps, context="32768", lanes="4"):
    return {"name": name, "model": model, "steps": steps,
            "args": ["--max-context", context, "--max-lanes", lanes, "--max-inflight", "8",
                     "--cache-bytes", str(8 << 30), "--qualification-mode", *args]}

STAGES = [
    stage("x6-mtp1", X6, ["--execution-policy", str(POL / "mtp1.json")], [qualifier("x6-mtp1")]),
    stage("x6-ordinary", X6, ["--ordinary"], [qualifier("x6-ordinary")]),
    stage("x6-pld", X6, ["--prompt-lookup", "--execution-policy", str(POL / "pld.json")], [qualifier("x6-pld")]),
    stage("x16-mtp1", X16, ["--execution-policy", str(POL / "mtp1.json")], [qualifier("x16-mtp1")]),
    stage("x16-ordinary", X16, ["--ordinary"], [qualifier("x16-ordinary")]),
]

state = {"started": time.strftime("%F %T"), "phase": "starting", "stages": {}, "order": [s["name"] for s in STAGES]}
if len(sys.argv) > 1 and STATUS.exists():
    previous = json.loads(STATUS.read_text())
    state["stages"] = previous.get("stages", {})
    state["history"] = previous.get("history", []) + [{name: previous["stages"].get(name) for name in sys.argv[1:]}]


def save():
    state["updated"] = time.strftime("%F %T")
    STATUS.write_text(json.dumps(state, indent=2))


def healthy():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def stop(server):
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(90)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(server.pid, signal.SIGKILL)  # spawn workers, zombies
    except ProcessLookupError:
        pass
    time.sleep(3)


def run_stage(spec):
    entry = state["stages"][spec["name"]] = {"state": "loading", "steps": {}, "started": time.strftime("%T")}
    save()
    log = open(RUN / f"{spec['name']}-server.log", "w")
    command = [PY, "-u", "-m", "mlx2.server", "--model", str(MODELS / spec["model"]), "--host", "127.0.0.1",
               "--port", str(PORT), "--cache-dir", str(RUN / "cache" / spec["name"]), *spec["args"]]
    entry["command"] = " ".join(command[3:])
    server = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 900
        while time.time() < deadline and server.poll() is None and not healthy():
            time.sleep(3)
        if not healthy():
            entry["state"] = "failed"
            entry["reason"] = "server did not become healthy" if server.poll() is None else f"server exited {server.returncode}"
            entry["log_tail"] = (RUN / f"{spec['name']}-server.log").read_text()[-1500:]
            return
        entry["state"] = "running"
        save()
        ok = True
        for label, command, limit in spec["steps"]:
            out = open(RUN / f"{spec['name']}-{label}.log", "w")
            started = time.time()
            try:
                code = subprocess.run(command, cwd=ROOT, env=ENV, stdout=out, stderr=subprocess.STDOUT, timeout=limit).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            lines = (RUN / f"{spec['name']}-{label}.log").read_text().splitlines()
            entry["steps"][label] = {"returncode": code, "seconds": round(time.time() - started),
                                     "failed_lines": [l[:300] for l in lines if l.startswith("FAIL") or ": FAIL" in l][:12],
                                     "last": lines[-1][:300] if lines else ""}
            ok = ok and code == 0
            save()
        entry["state"] = "passed" if ok else "failed"
        entry["server_alive"] = healthy()
    finally:
        stop(server)
        entry["finished"] = time.strftime("%T")
        save()


def gpu_busy():
    """Another session's lock, or any model server / campaign process running."""
    lock = Path("/tmp/gpu.lock")
    if lock.exists() and f"pid {os.getpid()}" not in lock.read_text():
        return f"lock: {lock.read_text().strip()}"
    running = subprocess.run(
        ["pgrep", "-fl", "mlx2.server|mlx_lm.server|sanity_20x20|qualify_serving"],
        capture_output=True, text=True,
    ).stdout.strip()
    return f"processes: {running}" if running else None


def wait_for_idle_gpu():
    while (reason := gpu_busy()) is not None:
        state["phase"] = f"waiting for GPU ({reason[:120]})"
        save()
        time.sleep(30)


def main():
    only = set(sys.argv[1:])
    wait_for_idle_gpu()
    Path("/tmp/gpu.lock").write_text(f"claude mlx2 xing4-0 campaign pid {os.getpid()}\n")
    try:
        state["phase"] = "preflight"
        save()
        if not PREFLIGHT.exists():
            result = subprocess.run([PY, "scripts/qualify_serving.py", "--preflight-only", "--output", str(PREFLIGHT)],
                                    cwd=ROOT, env=ENV, capture_output=True, text=True)
            state["preflight"] = {"returncode": result.returncode, "tail": (result.stdout + result.stderr)[-400:]}
        state["phase"] = "stages"
        save()
        for spec in STAGES:
            if only and spec["name"] not in only:
                continue
            try:
                run_stage(spec)
            except Exception as error:  # noqa: BLE001 - keep the campaign going
                state["stages"].setdefault(spec["name"], {})["state"] = "failed"
                state["stages"][spec["name"]]["reason"] = repr(error)
                save()
    finally:
        lock = Path("/tmp/gpu.lock")
        if lock.exists() and f"pid {os.getpid()}" in lock.read_text():
            lock.unlink()  # never remove another session's lock
        state["phase"] = "done"
        save()


if __name__ == "__main__":
    main()
