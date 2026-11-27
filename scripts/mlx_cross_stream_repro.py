#!/usr/bin/env python3
"""Minimal repro: MLX 0.32.5 cross-stream Fence update placed after a command-buffer split.

Hypothesis (read from MLX cfc675e9e source, not yet run):
  eval_impl (mlx/transforms.cpp:330-359) encodes the producer with gpu::eval(arr), which may
  commit the command buffer (metal/eval.cpp:63-68, needs_commit) and then may call
  scheduler::wait_for_capacity (transforms.cpp:342, finalizes every open stream when more than
  10 command buffers are in flight). Only then does it call Fence::update for an array consumed
  on another stream. In fast-synch mode (MLX_METAL_FAST_SYNCH=1, which `import mlx2` sets)
  Fence::update (metal/fence.cpp:122-132) orders the fence_update kernel after the producer only
  with memoryBarrier, which orders within one compute encoder. After a split the kernel sits in
  a new encoder, in a new command buffer, with no MTLFence wait on the producer's encoder. That
  wait exists only on the cross-device path, through set_input_array in input_coherent. So the
  consumer stream's fence_wait can pass while the producer is still writing.

Each configuration runs in a fresh child process, because MLX latches MLX_METAL_FAST_SYNCH and
MLX_MAX_OPS_PER_BUFFER on first use. This script does not import mlx2, which would force fast
synch on. Arrays are small (n x n float32, default n=512, 1 MB). Each child has a timeout.

Variants (predicted bad count if the hypothesis holds):
  fast_ops0_split          fast synch, commit after every kernel, y on s2 consumed on the
                           default stream in the same graph                  -> bad > 0
  fast_ops0_consumer_s1    same, consumed on a second new stream s1, then compared on the
                           default stream (two fence hops)                    -> bad > 0
  fast_ops4_round_trip     the original P8 graph (s1 -> s2 -> s1, 8 affine steps), ops=4
                                                                              -> bad > 0
  fast_default_split       fast synch, MLX default cadence (fence_update shares the
                           producer's encoder)                                -> 0
  fast_ops0_eval_between   mx.eval(y) before the consumer is built (no fence) -> 0
  fast_ops0_same_stream    producer on the default stream (no fence)          -> 0
  slow_ops0_split          MLX_METAL_FAST_SYNCH=0: Fence is an MTLSharedEvent signalled at
                           command-buffer level (encodeSignalEvent)            -> 0
                           (if this one also fails, event signalling across a split is
                           unordered too, and the patch must also cover the slow path)

Usage (GPU, through the coordinator only, nothing else on the GPU):
    python3 scripts/mlx_cross_stream_repro.py --trials 100

Fast-synch children are SIGKILLed (whole process group) after 20 s; slow-synch children after
--timeout (default 60 s). Each variant's record is also echoed to stderr as it finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

# Pierre 2026-11-26: a fast-synch spin hang killed WindowServer on 2026-10-21, and its watchdog
# fires at about 40 s. Every MLX_METAL_FAST_SYNCH=1 child gets at most 20 s and is then
# SIGKILLed (its whole process group, not SIGTERM: a GPU spin may never service a polite signal).
FAST_SYNCH_KILL_S = 20.0

VARIANTS = {
    # name: (MLX_METAL_FAST_SYNCH, MLX_MAX_OPS_PER_BUFFER or None for the MLX default, graph)
    "fast_ops0_split": ("1", "0", "split"),
    "fast_ops0_consumer_s1": ("1", "0", "consumer_s1"),
    "fast_ops4_round_trip": ("1", "4", "round_trip"),
    "fast_default_split": ("1", None, "split"),
    "fast_ops0_eval_between": ("1", "0", "eval_between"),
    "fast_ops0_same_stream": ("1", "0", "same_stream"),
    "slow_ops0_split": ("0", "0", "split"),
}


def child(graph: str, trials: int, n: int) -> dict:
    import mlx.core as mx

    s1, s2 = mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)
    bad = 0
    for t in range(trials):
        if graph == "round_trip":
            def data():
                return mx.random.randint(0, 1000, (4096,), key=mx.random.key(t)).astype(mx.float32)

            def ref(v):
                for _ in range(8):
                    v = v * 1.0 + 0.5
                return v

            with mx.stream(s1):
                z = data() * 2.0 + 1.0
            with mx.stream(s2):
                y = ref(z)
            with mx.stream(s1):
                got = y * 3.0
            want = ref(data() * 2.0 + 1.0) * 3.0
            bad += not mx.array_equal(got, want).item()
            continue

        # Integer-valued inputs in 0..3 make the matmul exact in float32 (and in TF32).
        # The input changes every trial, so a recycled output buffer holding the previous
        # trial's result cannot pass for the right answer.
        x = mx.random.randint(0, 4, (n, n), key=mx.random.key(t)).astype(mx.float32)
        want = x @ x
        mx.eval(x, want)
        if graph == "same_stream":
            y = x @ x
        else:
            with mx.stream(s2):
                y = x @ x  # one producing op; with ops=0 a commit follows it at once
        if graph == "eval_between":
            mx.eval(y)
        if graph == "consumer_s1":
            with mx.stream(s1):
                y = y + 0.0
        # The comparison on the default stream is the cross-stream consumer.
        bad += not mx.array_equal(y, want).item()
    return {"trials": trials, "bad": bad, "mlx": mx.__version__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--n", type=int, default=512, help="matmul side (max 1024)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds per slow-synch child (fast-synch children: at most %.0f)" % FAST_SYNCH_KILL_S)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    trials = max(1, min(args.trials, 10000))
    n = max(16, min(args.n, 1024))

    if args.child:
        print(json.dumps(child(args.child, trials, n)))
        return

    out = {}
    for name in args.variants.split(","):
        fast, ops, graph = VARIANTS[name]
        env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_")}
        env["MLX_METAL_FAST_SYNCH"] = fast
        if ops is not None:
            env["MLX_MAX_OPS_PER_BUFFER"] = ops
        cmd = [sys.executable, os.path.abspath(__file__), "--child", graph,
               "--trials", str(trials), "--n", str(n)]
        rec = {"fast_synch": fast, "max_ops_per_buffer": ops or "default", "graph": graph}
        limit = min(args.timeout, FAST_SYNCH_KILL_S) if fast == "1" else args.timeout
        rec["kill_after_s"] = limit
        started = time.monotonic()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=limit)
            if proc.returncode == 0 and stdout.strip():
                rec.update(json.loads(stdout.strip().splitlines()[-1]))
            else:
                rec["error"] = (stderr or stdout)[-2000:]
                rec["returncode"] = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            rec["error"] = f"SIGKILLed after {limit:.0f}s (possible fast-synch GPU spin)"
            rec["killed"] = True
        rec["seconds"] = round(time.monotonic() - started, 2)
        out[name] = rec
        print(json.dumps({name: rec}), file=sys.stderr, flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
