"""North re-qualification on the corrected normalization (2026-09-20).

Everything North was qualified on before 2026-09-20 was measured with
nn.LayerNorm(eps=1e-5) where the reference is RMSNorm(eps=1e-6).  This driver
re-runs the ordinary-route serving qualification against the corrected body.

Two things this script is careful about, both of which have bitten this lab:

* `PYTHONPATH` is the ABSOLUTE path of this worktree's `src`.  The
  2026-09-18 campaign used `PYTHONPATH=src` with `cwd=~/Desktop/mlx2`,
  which silently imports mlx2 from the main checkout.  The server log is
  checked for this worktree's path before any result is accepted.
* the served body's norm counters are read from `/v1/status` and from the
  model itself, so a run cannot pass while quietly serving LayerNorm.

Stages are independent; each is a separate GPU slot.  Usage:

    run_requal.py qualify        # server + scripts/qualify_serving.py
    run_requal.py calibrate      # commit-direction trace/extract/grid
    run_requal.py sanity         # 20x20 serving sanity on the corrected body
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/private/tmp/mlx2-rm06b-north-norm")
RUN = ROOT / "qualification/runs/north-requal-20260920"
MODEL = Path("~/mlx-models/North-Mini-Code-1.0-mlx-4bit")
PY = "~/Desktop/mlx2/.venv/bin/python"
PORT = 8297
BASE = f"http://127.0.0.1:{PORT}"
ENV = {
    **os.environ,
    # ABSOLUTE.  A relative "src" resolves against whatever cwd the child gets.
    "PYTHONPATH": str(ROOT / "src"),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def log(msg):
    print(f"[{time.strftime('%T')}] {msg}", flush=True)


def healthy():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def start_server(name, args, wait=1200):
    log_path = RUN / f"{name}-server.log"
    command = [PY, "-u", "-m", "mlx2.server", "--model", str(MODEL), "--host", "127.0.0.1",
               "--port", str(PORT), "--cache-dir", str(RUN / "cache" / name), *args]
    log(f"server: {' '.join(command)}")
    handle = open(log_path, "w")
    server = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=handle,
                              stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + wait
    while time.time() < deadline and server.poll() is None and not healthy():
        time.sleep(3)
    if not healthy():
        tail = log_path.read_text()[-4000:]
        stop_server(server)
        raise SystemExit(f"server never became healthy\n{tail}")
    return server


def stop_server(server):
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(90)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(3)


def assert_our_code(name):
    """Refuse a result produced by mlx2 imported from another checkout.

    Two independent checks, because a clean server log proves nothing:
    resolve `mlx2.__file__` under exactly the env/cwd the server got, and scan
    the server log for the main checkout's path.
    """
    ours, theirs = str(ROOT / "src"), "~/Desktop/mlx2/src"
    probe = subprocess.run(
        [PY, "-c", "import mlx2, mlx2.thinking_calibration as t, mlx2.runtime.models.cohere2_moe as c;"
                   "print(mlx2.__file__); print(t.SCHEMA); print(c.__file__)"],
        cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=300)
    resolved = probe.stdout.strip().splitlines()
    if not resolved or not resolved[0].startswith(ours):
        raise SystemExit(f"mlx2 resolves to {resolved or probe.stderr!r}, not under {ours}")
    text = (RUN / f"{name}-server.log").read_text()
    if theirs in text:
        raise SystemExit(f"server log mentions {theirs}")
    return {"expected_src": ours, "mlx2_file": resolved[0],
            "commit_direction_schema": resolved[1] if len(resolved) > 1 else None,
            "cohere2_moe_file": resolved[2] if len(resolved) > 2 else None,
            "main_checkout_in_server_log": False}


def provenance(name):
    """Independent evidence that the served body runs the corrected norm."""
    status = get("/v1/status")
    (RUN / f"{name}-status.json").write_text(json.dumps(status, indent=1))
    record = {
        "source_path_check": assert_our_code(name),
        "runtime": status.get("runtime"),
        "artifact": status.get("artifact"),
        "profile": status.get("profile"),
        "thinking_steer": (status.get("settings") or {}).get("thinking_steer"),
        "thinking_defaults_source": (status.get("settings") or {}).get("thinking_defaults_source"),
    }
    (RUN / f"{name}-provenance.json").write_text(json.dumps(record, indent=1))
    log(f"provenance: {json.dumps(record.get('thinking_steer'))}")
    return record


def run(name, command, limit):
    out = open(RUN / f"{name}.log", "w")
    log(f"step {name}: {' '.join(str(c) for c in command)}")
    started = time.time()
    rc = subprocess.call([str(c) for c in command], cwd=ROOT, env=ENV, stdout=out,
                         stderr=subprocess.STDOUT, timeout=limit)
    log(f"step {name}: rc={rc} in {time.time() - started:.0f}s")
    return rc


# Matches the 2026-09-18 sanity-20x20 stage, which is the configuration that
# produced North's only `passed=true` qualifier record (4 lanes, 8 GiB APCv2).
# The integration campaign's 2 GiB variant failed `shared_cohort_priming` twice.
SERVER_ARGS = ["--max-context", "32768", "--max-lanes", "4", "--max-inflight", "8",
               "--cache-bytes", str(8 << 30), "--qualification-mode", "--ordinary"]


def check_preflight():
    """Fail before the GPU lock, not after it.

    `validate_preflight_receipt` compares the receipt's git revision, src and
    tests tree hashes, runtime and harness hash against the live ones. A
    receipt cut before a commit is rejected, and the first attempt at this
    stage burned a GPU slot discovering that 0.4 s after a 70 s model load.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("qs", ROOT / "scripts/qualify_serving.py")
    qs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qs)
    receipt = json.loads((RUN / "preflight.json").read_text())
    if receipt.get("identity") != qs.preflight_identity():
        raise SystemExit("preflight receipt identity is stale; regenerate it before queueing")
    log("preflight receipt identity matches HEAD")


def stage_qualify():
    check_preflight()
    server = start_server("north-ordinary", SERVER_ARGS)
    try:
        record = provenance("north-ordinary")
        rc = run("north-ordinary-qualifier",
                 [PY, "scripts/qualify_serving.py", "--url", BASE,
                  "--output", RUN / "north-ordinary-qualification.json",
                  "--preflight-receipt", RUN / "preflight.json", "--timeout", "1800"],
                 3600)
    finally:
        stop_server(server)
    return {"stage": "qualify", "rc": rc, "provenance": record}


def stage_sanity():
    server = start_server("north-sanity", SERVER_ARGS[:-1] + ["--ordinary"])
    try:
        record = provenance("north-sanity")
        rc = run("north-sanity-20x20",
                 [PY, ROOT / "qualification/runs/sanity-20x20-20260918/sanity_20x20.py",
                  BASE, RUN / "north-ordinary-sanity.json"],
                 3600)
    finally:
        stop_server(server)
    return {"stage": "sanity", "rc": rc, "provenance": record}


def stage_calibrate():
    """Direct-model commit-direction recalibration; no server involved."""
    script = ROOT / "scripts/calibrate_thinking_direction.py"
    rc = run("calib-trace", [PY, script, "trace", "--model", MODEL,
                             "--out", RUN / "calib-traces.jsonl"], 3600)
    if rc:
        return {"stage": "calibrate", "rc": rc, "step": "trace"}
    rc = run("calib-extract", [PY, script, "extract", "--model", MODEL,
                               "--traces", RUN / "calib-traces.jsonl",
                               "--out", RUN / "calib-direction.npz"], 1800)
    return {"stage": "calibrate", "rc": rc, "step": "extract"}


# The 2026-09-18 grid used max_think 2400; keep it so the arms are comparable
# to the numbers this run is voiding.  Split across GPU slots: the unsteered and
# random arms run to the cap on every run-on prompt and are the slow ones.
GRID_ARMS = {
    # L32 is what `choose_layer` picks on the corrected body (consistency
    # +0.518, against +0.511 at L28); L28 was the 2026-09-18 operating point
    # and is kept so the two campaigns are comparable.
    "a": [{"name": "off"}, {"name": "L32-a0.2", "layer": 32, "alpha": 0.2}],
    "b": [{"name": "L28-a0.2", "layer": 28, "alpha": 0.2},
          {"name": "L32-random-a0.2", "layer": 32, "alpha": 0.2, "random": True}],
}


def stage_grid(which="a"):
    script = ROOT / "scripts/calibrate_thinking_direction.py"
    rc = run(f"calib-grid-{which}",
             [PY, script, "grid", "--model", MODEL, "--max-think", "2400",
              "--vectors", RUN / "calib-direction.npz",
              "--arms", json.dumps(GRID_ARMS[which]),
              "--out", RUN / f"calib-grid-{which}.json"], 3000)
    return {"stage": f"grid-{which}", "rc": rc}


STAGES = {"qualify": stage_qualify, "sanity": stage_sanity,
          "calibrate": stage_calibrate,
          "grid-a": lambda: stage_grid("a"), "grid-b": lambda: stage_grid("b")}

if __name__ == "__main__":
    which = sys.argv[1]
    RUN.mkdir(parents=True, exist_ok=True)
    result = STAGES[which]()
    result["finished"] = time.strftime("%F %T")
    (RUN / f"stage-{which}.json").write_text(json.dumps(result, indent=1))
    log(json.dumps(result))
    raise SystemExit(0 if result.get("rc") == 0 else 1)
