"""Sequential GPU viability/correctness campaign for the 2026-09-18 integrations.

One server at a time on port 8297.  Every stage records its steps in
status.json; a failed stage never stops the campaign.  The production service
is quiesced for the window and always restored.
"""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

MAIN = Path("~/Desktop/mlx2")
# MLX2_CAMPAIGN_ROOT pins the served source to a clean worktree when another
# session is editing the main checkout.
ROOT = Path(os.environ.get("MLX2_CAMPAIGN_ROOT", str(MAIN)))
RUN = Path(__file__).resolve().parent
MODELS = Path("~/mlx-models")
PY = str(MAIN / ".venv/bin/python")
PORT = 8297
BASE = f"http://127.0.0.1:{PORT}"
SERVICE = "com.example.fn-uncensored-mlx-serve"
PLIST = Path.home() / "Library/LaunchAgents" / f"{SERVICE}.plist"
STATUS = RUN / "status.json"
ENV = {**os.environ, "PYTHONPATH": "src", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
PREFLIGHT = RUN / "preflight.json"

QWEN36 = "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp"
NORTH = "North-Mini-Code-1.0-mlx-4bit"
MUSE = "Muse-Glimmer-30B-mlx-4bit"
QWEN38 = "Qwen3.8-27B-oQ4e-mtp"
FLASH = "Qwen3.8-Flash-Next-MLX-4bit-MTP"
POL = ROOT / "qualification/policies"
PLD = str(MAIN / "qualification/runs/integration-gpu-20260918/policies/pld.json")


def sanity(name):
    return ("sanity", [PY, str(RUN / "sanity_20x20.py"), BASE, str(RUN / f"{name}-sanity.json")], 5400)


def qualifier(name, *extra):
    return ("qualifier", [PY, "scripts/qualify_serving.py", "--url", BASE, "--output", str(RUN / f"{name}-qualification.json"),
                          "--preflight-receipt", str(PREFLIGHT), "--timeout", "1800", *extra], 3600)


def stage(name, model, args, steps=None, context="32768", lanes="20", cache=8):
    return {"name": name, "model": model, "steps": steps or [sanity(name)],
            "args": ["--max-context", context, "--max-lanes", lanes, "--max-inflight", "40",
                     "--cache-bytes", str(cache << 30), "--qualification-mode", *args]}

STAGES = [
    stage("qwen36-ordinary", QWEN36, ["--ordinary", "--execution-policy", str(POL / "qwen36-mtp-artifact-ordinary.json")]),
    stage("qwen36-mtp2", QWEN36, ["--execution-policy", str(POL / "qwen36-mtp2.json")]),
    stage("qwen36-pld", QWEN36, ["--prompt-lookup", "--execution-policy", PLD]),
    stage("north-ordinary", NORTH, ["--ordinary"]),
    stage("north-pld", NORTH, ["--prompt-lookup", "--execution-policy", PLD]),
    stage("muse-ordinary", MUSE, ["--ordinary"]),
    stage("muse-dflash2", MUSE, ["--external-draft", "--execution-policy", str(POL / "muse-dflash2.json")]),
    stage("muse-pld", MUSE, ["--prompt-lookup", "--execution-policy", PLD]),
    stage("qwen38-ordinary", QWEN38, ["--ordinary", "--execution-policy", str(POL / "qwen38-27b-mtp2.json")]),
    stage("qwen38-mtp2", QWEN38, ["--execution-policy", str(POL / "qwen38-27b-mtp2.json")]),
    stage("flash-next-mtp2", FLASH, ["--execution-policy", str(POL / "flash-next-mtp2.json")], cache=16),
    stage("flash-next-ordinary", FLASH, ["--ordinary", "--execution-policy", str(POL / "flash-next-mtp2.json")], cache=16),
    stage("north-qualifier", NORTH, ["--ordinary"], [qualifier("north-ordinary")], lanes="4", cache=8),
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
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            server.wait(90)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(server.pid, signal.SIGKILL)  # spawn workers, zombies
    except (ProcessLookupError, PermissionError):  # macOS: EPERM on an already-reaped group
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
        deadline = time.time() + 2400
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


def main():
    only = set(sys.argv[1:])
    Path("/tmp/gpu.lock").write_text(f"claude mlx2 integration-gpu campaign pid {os.getpid()}\n")
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{SERVICE}"], capture_output=True)
    time.sleep(8)
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
        Path("/tmp/gpu.lock").unlink(missing_ok=True)
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST)], capture_output=True)
        restored = False
        for _ in range(80):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8282/v1/models", timeout=3) as r:
                    if b'"state":"ready"' in r.read():
                        restored = True
                        break
            except Exception:
                pass
            time.sleep(3)
        state["production_restored"] = restored
        state["phase"] = "done"
        save()


if __name__ == "__main__":
    main()
