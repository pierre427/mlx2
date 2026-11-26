#!/usr/bin/env python3
"""P8 (safety and concurrency) of docs/research/2026-11-26-metal-replay-feasibility-probes.md,
on the M5 only (the M1/M2 pipeline-residency case needs an M2 and Pierre's OK; until then
ICB replay is gated to Apple9 and later).

GPU, under the lease. Uses ``step_replay.replay_program``/``replay_apply``: a synthetic sealed
program (K affine dispatches in an ICB over program-owned scratch) run as an MLX primitive inside
MLX's own command stream, with its input and output declared to MLX's encoder. Every check
compares bit for bit with the same graph built from ordinary MLX ops (``t + 0.5 * K``, exact on
the integer-valued data used here).

  correctness   one replay, K in (1, 8, 64)
  interleave    replays between ordinary MLX ops at a forced small commit cadence
                (``MLX_MAX_OPS_PER_BUFFER=4``), two replays of one program per graph, several
                graphs in flight (``async_eval``); ring 4 and ring 1 (the in-flight rebind hazard)
  concurrent    the interleave graphs while another stream runs MLX matmuls continuously
  cross_stream  input produced on one stream, replayed on another, consumed on the first
  retire        a program's Python handle dropped while its replays are in flight; and an old
                program's scratch filled with a sentinel after its last completion, then replayed
  fault         (child process) an ICB write 256 MB past its buffer: what the child sees, and
                whether this process's MLX work stays correct

    MLX_MAX_OPS_PER_BUFFER=4 scripts/replay_p8_safety.py --output DIR
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path

import mlx.core as mx  # first: the extension shares its nanobind registry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlx2.runtime.native_extension import load_extension  # noqa: E402

STAGED = ROOT / "build" / "step_replay" / "stage" / ("_step_replay" + sysconfig.get_config_var("EXT_SUFFIX"))
PLAN_PROBE = ROOT / "build" / "replay_probe" / "gpu_plan_probe"
N = 4096


def reference(t, K):
    for _ in range(K):
        t = t * 1.0 + 0.5
    return t


def data(seed, n=N):
    return mx.random.randint(0, 1000, (n,), key=mx.random.key(seed)).astype(mx.float32)


def equal(a, b):
    return bool(mx.array_equal(a, b).item())


def correctness(sr):
    out = []
    for K in (1, 8, 64):
        prog = sr.replay_program(K, N, 4)
        x = data(K)
        y = sr.replay_apply(prog, x)
        out.append({"K": K, "equal": equal(y, reference(x, K)), "stats": prog.stats()})
    return out


def graph(fn_replay, seed, K):
    a = data(seed)
    b = a * 2.0 - 3.0
    z = (b + 1.0) * 1.0
    y = fn_replay(z)
    w = y * 3.0 - 7.0
    w2 = fn_replay(w)
    return w2 + 1.0


def interleave(sr, ring, trials=200, inflight=4, K=8):
    prog = sr.replay_program(K, N, ring)
    bad, batch, refs = 0, [], []
    for t in range(trials):
        batch.append(graph(lambda v: sr.replay_apply(prog, v), t, K))
        refs.append(graph(lambda v: reference(v, K), t, K))
        if len(batch) == inflight:
            mx.async_eval(batch)
            for got, want in zip(batch, refs):
                bad += not equal(got, want)
            batch, refs = [], []
    return {"ring": ring, "trials": trials, "graphs_in_flight": inflight, "replays_per_graph": 2,
            "bad": bad, "stats": prog.stats()}


def concurrent(sr, trials=200):
    stop = threading.Event()
    s2 = mx.new_stream(mx.gpu)
    done = {"iters": 0}

    def background():
        m = mx.random.normal((1024, 1024), key=mx.random.key(99))
        with mx.stream(s2):
            while not stop.is_set():
                m = (m @ m) / 1024.0
                mx.eval(m)
                done["iters"] += 1

    thread = threading.Thread(target=background, daemon=True)
    thread.start()
    try:
        result = interleave(sr, ring=4, trials=trials)
    finally:
        stop.set()
        thread.join(timeout=30)
    result["background_matmuls"] = done["iters"]
    return result


def cross_stream(sr, trials=100, K=8):
    s1, s2 = mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)
    prog = sr.replay_program(K, N, 4)
    bad = 0
    for t in range(trials):
        with mx.stream(s1):
            z = data(t) * 2.0 + 1.0
        with mx.stream(s2):
            y = sr.replay_apply(prog, z)
        with mx.stream(s1):
            got = y * 3.0
        want = reference(data(t) * 2.0 + 1.0, K) * 3.0
        bad += not equal(got, want)
    return {"trials": trials, "bad": bad, "stats": prog.stats()}


def retire(sr, K=8):
    before = sr.replay_programs_destroyed()
    prog = sr.replay_program(K, N, 8)
    xs = [data(1000 + i) for i in range(32)]
    mx.eval(xs)
    ys = [sr.replay_apply(prog, x) for x in xs]
    mx.async_eval(ys)
    del prog  # the primitives and the completion handlers keep it alive
    gc.collect()
    ok = all(equal(y, reference(x, K)) for x, y in zip(xs, ys))
    del ys
    gc.collect()
    mx.synchronize()
    time.sleep(0.2)
    after = sr.replay_programs_destroyed()
    # Sentinel: fill an old program's scratch after its last completion, then replay it.
    old = sr.replay_program(K, N, 4)
    x = data(7)
    mx.eval(sr.replay_apply(old, x))
    mx.synchronize()
    stats = old.stats()
    old.fill_scratch(float("nan"))
    new = sr.replay_program(K, N, 4)
    y_new = sr.replay_apply(new, x)
    y_old = sr.replay_apply(old, x)
    sentinel_free = not bool(mx.any(mx.isnan(y_old)).item()) and not bool(mx.any(mx.isnan(y_new)).item())
    return {"dropped_handle_outputs_ok": ok,
            "destroyed_with_work_in_flight": after["destroyed_with_work_in_flight"] - before["destroyed_with_work_in_flight"],
            "destroyed": after["destroyed"] - before["destroyed"],
            "old_program_completions_before_sentinel": stats,
            "replay_after_sentinel_equal": equal(y_old, reference(x, K)) and equal(y_new, reference(x, K)),
            "sentinel_free": sentinel_free}


def fault():
    if not PLAN_PROBE.exists():
        return {"skipped": "gpu_plan_probe not built"}
    x = data(5)
    want = reference(x, 4)
    proc = subprocess.Popen([str(PLAN_PROBE), "fault"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    checks = 0
    bad = 0
    deadline = time.time() + 60
    while proc.poll() is None and time.time() < deadline:
        bad += not equal(reference(x, 4), want)
        checks += 1
    try:
        stdout, stderr = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    after_ok = all(equal(reference(data(50 + i), 4), reference(data(50 + i), 4)) for i in range(8))
    child = None
    try:
        child = json.loads(stdout) if stdout.strip() else None
    except ValueError:
        pass
    return {"child_returncode": proc.returncode, "child": child, "child_stderr_tail": stderr[-1500:],
            "parent_checks_during_child": checks, "parent_bad_during_child": bad,
            "parent_ok_after_child": after_ok}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sr = load_extension("mlx2_step_replay._step_replay", str(STAGED))
    hooks = sr.install_hooks()
    if not hooks.get("installed") or "replay_op_probe" not in sr.CAPABILITIES:
        raise SystemExit(f"step_replay not usable: {hooks}")
    doc = {"schema": "mlx2.replay-p8.v1", "probe": "P8",
           "meta": {"source_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"],
                                                      capture_output=True, text=True).stdout.strip(),
                    "mlx": mx.__version__, "gpu": sr.gpu_identity(), "checkpoint": "none (synthetic)",
                    "macos": platform.mac_ver()[0],
                    "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                    "MLX_MAX_OPS_PER_BUFFER": os.environ.get("MLX_MAX_OPS_PER_BUFFER")},
           "m1_m2_residency": "not run: needs an M2 and Pierre's OK; ICB replay gated to Apple9+ until then"}
    for name, fn in (("correctness", lambda: correctness(sr)),
                     ("interleave_ring4", lambda: interleave(sr, 4)),
                     ("concurrent", lambda: concurrent(sr)),
                     ("cross_stream", lambda: cross_stream(sr)),
                     ("retire", lambda: retire(sr)),
                     # Last but one: ring 1 rebinds in-flight commands on purpose, and its
                     # wrong bindings may write anywhere this process owns.
                     ("interleave_ring1", lambda: interleave(sr, 1)),
                     ("fault", fault)):
        started = time.time()
        try:
            doc[name] = fn()
        except Exception as error:
            doc[name] = {"exception": f"{type(error).__name__}: {error}"}
        doc[name + "_seconds"] = round(time.time() - started, 2)
        (out / "p8.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(name, json.dumps(doc[name])[:400], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
