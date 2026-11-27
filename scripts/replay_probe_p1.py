#!/usr/bin/env python3
"""P1 (synthetic ICB ceiling) and the P4 trap tests of
docs/research/2026-11-26-metal-replay-feasibility-probes.md.

    scripts/replay_probe_p1.py build
        compile native/replay_probe/replay_probe.mm into build/replay_probe/ (CPU only)
    scripts/replay_probe_p1.py run --output DIR [--rounds 1000]
        GPU (run under the lease): the native arms 1-5 and the DAG shape (the
        binary's JSON), then arm 0 here: the same chain through MLX, one
        ``mx.fast.metal_kernel`` per dispatch under ``mx.compile``, evaluated
        once per round. GPU time for arm 0 comes from the staged step_replay
        GPU timeline (built by scripts/build_step_replay.py); one recorded round
        gives MLX's dispatch and command-buffer counts. Writes DIR/p1.json.

Kill criterion (probe doc): ICB replay saves less than T = 5% x round wall /
dispatches per round, both from P0, against arm 0 at the measured dispatch
count. P0 on Raon (quiet 1K): 29.4 ms, 3,803 dispatches, so T = 0.387 us.
Synthetic, no model: ``checkpoint`` is ``none (synthetic)``.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import platform
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "replay_probe" / "replay_probe.mm"
BINARY = ROOT / "build" / "replay_probe" / "replay_probe"
STEP_REPLAY = ROOT / "build" / "step_replay" / "stage" / ("_step_replay" + sysconfig.get_config_var("EXT_SUFFIX"))
P0_ROUND_MS, P0_DISPATCHES = 29.4, 3803
THRESHOLD_US = 0.05 * P0_ROUND_MS * 1000 / P0_DISPATCHES
NS = (64, 256, 1024, 4096)
ARM_NAMES = {0: "mlx", 1: "direct", 2: "tape", 3: "icb", 4: "icb_cpu_patch", 5: "icb_gpu_patch"}


def build():
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["clang++", "-std=c++20", "-fobjc-arc", "-O2", "-arch", "arm64", str(SOURCE),
           "-framework", "Metal", "-framework", "Foundation", "-o", str(BINARY)]
    subprocess.run(cmd, check=True)
    print(BINARY)


def git_head():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def load_step_replay():
    if not STEP_REPLAY.exists():
        return None
    loader = importlib.machinery.ExtensionFileLoader("mlx2_step_replay._step_replay", str(STEP_REPLAY))
    spec = importlib.util.spec_from_file_location("mlx2_step_replay._step_replay", str(STEP_REPLAY), loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    module.install_hooks()
    return module


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * (len(values) - 1) + 0.5))] if values else None


def stats(values):
    return {"p10": round(pct(values, 0.1), 3), "p50": round(pct(values, 0.5), 3), "p90": round(pct(values, 0.9), 3)}


def gpu_busy_us(entries, a, b):
    spans = sorted((max(s, a), min(e, b)) for _, s, e, status in entries if status == 4 and e > s)
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy * 1e6


def arm0(mx, native, n, lanes, rounds, warm):
    """The chain (lanes=1) or the DAG (lanes=8) through MLX."""
    kernel = mx.fast.metal_kernel(
        name="p1_affine", input_names=["inp", "ab"], output_names=["out"],
        source="uint i = thread_position_in_grid.x; out[i] = inp[i] * ab[0] + ab[1];")

    def chain(xs, ab):
        xs = list(xs)
        for k in range(n):
            lane = k % lanes
            xs[lane] = kernel(inputs=[xs[lane], ab], grid=(1024, 1, 1), threadgroup=(256, 1, 1),
                              output_shapes=[(1024,)], output_dtypes=[mx.float32])[0]
        return xs

    fn = mx.compile(chain)
    xs = [mx.arange(1024, dtype=mx.float32) for _ in range(lanes)]
    ab = mx.array([1.0, 0.5], mx.float32)
    mx.eval(xs, ab)
    cpu, e2e, gpu = [], [], []
    for r in range(warm + rounds):
        if native is not None:
            native.timeline_drain()
        h0 = native.host_now() if native is not None else None
        t0 = time.perf_counter()
        ys = fn(xs, ab)
        mx.async_eval(ys)
        t1 = time.perf_counter()
        mx.eval(ys)
        t2 = time.perf_counter()
        if r >= warm:
            cpu.append((t1 - t0) * 1e6)
            e2e.append((t2 - t0) * 1e6)
            if native is not None:
                h1 = native.host_now()
                deadline = time.perf_counter() + 0.05
                while native.timeline_pending() and time.perf_counter() < deadline:
                    time.sleep(0.0005)
                gpu.append(gpu_busy_us(native.timeline_drain(), h0, h1))
    got = ys[0].tolist()
    steps = len(range(0, n, lanes))
    ok = all(v == i + 0.5 * steps for i, v in enumerate(got))
    out = {"arm": 0, "mismatches": 0 if ok else 1,
           "stats": {"cpu_us": stats(cpu), "e2e_us": stats(e2e), "gpu_us": stats(gpu) if gpu else None}}
    if native is not None:
        mx.synchronize()
        native.record_begin()
        try:
            mx.eval(fn(xs, ab))
        finally:
            rec = native.record_end()
        summary = rec.summary()
        out["recorded"] = {k: summary[k] for k in ("dispatches", "barriers", "command_buffers_committed",
                                                   "encoders_opened", "unsupported")}
    return out


def verdict(doc):
    rows = {}
    for shape in ("chain", "dag"):
        for n in NS:
            arms = {a["arm"]: a for a in doc[shape][str(n)]}
            base = arms[0]["stats"]["e2e_us"]["p50"]
            rows.setdefault(shape, {})[str(n)] = {
                ARM_NAMES[a]: {
                    "e2e_p50_us": arms[a]["stats"]["e2e_us"]["p50"],
                    "cpu_p50_us": arms[a]["stats"]["cpu_us"]["p50"],
                    "gpu_p50_us": (arms[a]["stats"].get("gpu_us") or {}).get("p50"),
                    "saved_per_dispatch_vs_mlx_us": round((base - arms[a]["stats"]["e2e_us"]["p50"]) / n, 4),
                } for a in sorted(arms)}
    icb = rows["chain"]["4096"]["icb"]["saved_per_dispatch_vs_mlx_us"]
    return {"threshold_us": round(THRESHOLD_US, 4), "rows": rows,
            "icb_saved_per_dispatch_at_4096_us": icb,
            "verdict": "pass" if icb >= THRESHOLD_US else "kill",
            "kill_criterion": "ICB (arm 3) saves less than T per dispatch against arm 0 at N=4096 (chain)"}


def run(args):
    import mlx.core as mx
    native = load_step_replay()  # before MLX compiles kernels: hooks + timeline
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=False)
    proc = subprocess.run([str(BINARY), str(args.rounds)], capture_output=True, text=True, timeout=1800)
    (out_dir / "native.stdout.json").write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"bench failed rc {proc.returncode}: {proc.stderr[-2000:]}")
    doc = json.loads(proc.stdout)
    if native is not None:
        native.set_timeline(True)
    for shape, lanes in (("chain", 1), ("dag", 8)):
        for n in NS:
            doc[shape][str(n)].insert(0, arm0(mx, native, n, lanes, args.rounds, 20))
    doc["arm_names"] = ARM_NAMES
    doc["meta"] = {"source_revision": git_head(), "mlx": mx.__version__, "checkpoint": "none (synthetic)",
                   "macos": platform.mac_ver()[0],
                   "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                   "step_replay": str(STEP_REPLAY) if native is not None else None,
                   "thermal_admission": False, "performance_claim": False}
    doc["p1"] = verdict(doc)
    doc["all_outputs_ok"] = doc.get("all_outputs_ok") and all(
        a["mismatches"] == 0 for shape in ("chain", "dag") for n in NS for a in doc[shape][str(n)])
    (out_dir / "p1.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"p1": {k: v for k, v in doc["p1"].items() if k != "rows"}, "traps": doc["traps"],
                      "icb_limits": doc["icb_limits"], "all_outputs_ok": doc["all_outputs_ok"]}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--output", required=True)
    r.add_argument("--rounds", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
