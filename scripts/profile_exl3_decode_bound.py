#!/usr/bin/env python3
"""What bounds B1 decode for upstream EXL3 on Metal (sanity profile).

One checkpoint per process, loaded on its default policy (nax_fused,
resident verify weights). After a 512-token random prompt:

- ``stream``: achievable device bandwidth (a 2 GiB uint32 sum, median of 5).
- ``token``: B1 greedy decode, three ways:
  ``sync`` (argmax read every token), ``pipelined`` (the next token's graph
  built and ``async_eval``-ed before the previous token is read, as the
  server does), and ``split`` (graph build time vs ``mx.eval`` time per token).
  GPU busy fraction (IOAccelerator "Device Utilization %") is sampled
  during the pipelined loop.
- ``components``: every call of one decode token to a decoder layer, mixer,
  MoE block, EXL3 dense linear, EXL3 projection group and EXL3 MoE decode
  tail is captured with its evaluated inputs and replayed alone (64 calls
  back to back in one evaluation, median of 3), with the trellis bytes each
  EXL3 call reads and the resulting GB/s. Replays mutate the cache, so they
  run after the token timings.

Not thermally gated; not a published performance figure.

  profile_exl3_decode_bound.py --i-own-the-gpu --model DIR --out NEW_DIR
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx  # noqa: E402

from check_exl3_checkpoint import _artifact_name  # noqa: E402

PROMPT = 512
TOKENS = 96
CALLS = 64
UTIL = re.compile(r'"Device Utilization %"=(\d+)')


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _per_call(fn, calls=None, reps=3):
    calls = CALLS if calls is None else calls
    mx.eval([fn() for _ in range(2)])
    samples = []
    for _ in range(reps):
        mx.synchronize()
        start = time.perf_counter()
        mx.eval([fn() for _ in range(calls)])
        mx.synchronize()
        samples.append((time.perf_counter() - start) / calls)
    return float(np.median(samples))


def _stream():
    a = mx.zeros((1 << 29,), mx.uint32)  # 2 GiB
    mx.eval(a)
    samples = []
    for _ in range(5):
        mx.synchronize()
        start = time.perf_counter()
        mx.eval(mx.sum(a))
        mx.synchronize()
        samples.append(time.perf_counter() - start)
    del a
    mx.clear_cache()
    return {"bytes": 1 << 31, "GBps": (1 << 31) / float(np.median(samples)) / 1e9}


class _Sampler:
    def __init__(self, period=0.05):
        self.period, self.values, self._stop = period, [], threading.Event()

    def _run(self):
        while not self._stop.is_set():
            out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                                 capture_output=True, text=True).stdout
            hit = UTIL.search(out)
            if hit:
                self.values.append(int(hit.group(1)))
            self._stop.wait(self.period)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        v = np.array(self.values or [0])
        return {"samples": len(self.values), "mean": float(v.mean()), "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def _prefill(model, cache, rng):
    prompt = rng.integers(1000, 100000, PROMPT).tolist()
    logits = model(mx.array(prompt, mx.uint32)[None], cache=cache)
    return mx.argmax(logits[0, -1])


def _token_timings(model, rng):
    from mlx2.runtime.models.cache import make_prompt_cache

    out = {}
    cache = make_prompt_cache(model)
    token = int(_prefill(model, cache, rng).item())
    for _ in range(8):
        token = int(mx.argmax(model(mx.array([[token]], mx.uint32), cache=cache)[0, -1]).item())
    mx.synchronize()
    start = time.perf_counter()
    for _ in range(TOKENS):
        token = int(mx.argmax(model(mx.array([[token]], mx.uint32), cache=cache)[0, -1]).item())
    sync = (time.perf_counter() - start) / TOKENS
    out["sync_ms"] = 1e3 * sync

    # Pipelined: build and submit token n+1 before reading token n.
    y = mx.array([token], mx.uint32)
    nxt = mx.argmax(model(y[None], cache=cache)[0, -1:], axis=-1).astype(mx.uint32)
    mx.async_eval(nxt)
    with _Sampler() as sampler:
        mx.synchronize()
        start = time.perf_counter()
        for _ in range(TOKENS):
            cur = nxt
            nxt = mx.argmax(model(cur[None], cache=cache)[0, -1:], axis=-1).astype(mx.uint32)
            mx.async_eval(nxt)
            cur.item()
        nxt.item()
        piped = (time.perf_counter() - start) / TOKENS
    out["pipelined_ms"] = 1e3 * piped
    out["gpu_utilization_pct"] = sampler.summary()

    # Split: host graph build vs evaluation, per token.
    builds, evals = [], []
    token = int(nxt.item())
    for _ in range(32):
        mx.synchronize()
        t0 = time.perf_counter()
        logits = model(mx.array([[token]], mx.uint32), cache=cache)
        t1 = time.perf_counter()
        mx.eval(logits)
        t2 = time.perf_counter()
        token = int(mx.argmax(logits[0, -1]).item())
        builds.append(t1 - t0)
        evals.append(t2 - t1)
    out["build_ms"] = 1e3 * float(np.median(builds))
    out["eval_ms"] = 1e3 * float(np.median(evals))
    del cache
    gc.collect()
    mx.clear_cache()
    return out


def _bytes_of(obj):
    """Trellis bytes one B1 call reads, or None for non-EXL3 kinds."""
    from mlx2.runtime.models import exl3

    if isinstance(obj, exl3.Exl3DenseLinear):
        return obj.bank.packed_nbytes
    if isinstance(obj, exl3.Exl3ProjectionGroup):
        return sum(lin.bank.packed_nbytes for lin in obj.linears)
    return None


def _tail_bytes(switch, block, inds):
    per = sum(p.packed_nbytes // p.num_experts for p in (switch.gate_proj, switch.up_proj, switch.down_proj))
    shared = getattr(block, "shared_expert", None)
    extra = 0
    if shared is not None:
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(shared, name, None)
            if lin is not None and hasattr(lin, "bank"):
                extra += lin.bank.packed_nbytes
    return int(inds.size) * per + extra


def _components(model, rng):
    from mlx2.runtime.models import exl3
    from mlx2.runtime.models.cache import make_prompt_cache

    names = {id(m): n for n, m in model.named_modules()}
    calls = []
    patched = []

    def wrap(cls, attr, kind):
        orig = getattr(cls, attr)

        def hook(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            if capture[0]:
                calls.append({"kind": kind, "obj": self, "orig": orig, "args": args, "kwargs": kwargs,
                              "out": out})
            return out

        setattr(cls, attr, hook)
        patched.append((cls, attr, orig))

    # Hook the classes the loaded model actually uses (no name branching).
    layers = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
    kinds = {type(layers[0]): "layer", type(layers[0].mlp): "moe"}
    for layer in layers:
        for attr, kind in (("linear_attn", "mixer.gdn"), ("self_attn", "mixer.attn")):
            if hasattr(layer, attr):
                kinds.setdefault(type(getattr(layer, attr)), kind)
    for cls, kind in kinds.items():
        wrap(cls, "__call__", kind)
    wrap(exl3.Exl3DenseLinear, "__call__", "exl3.dense")
    wrap(exl3.Exl3ProjectionGroup, "__call__", "exl3.group")
    wrap(exl3.Exl3SwitchGLU, "decode_tail", "exl3.tail")
    capture = [False]
    try:
        cache = make_prompt_cache(model)
        token = int(_prefill(model, cache, rng).item())
        for _ in range(4):
            token = int(mx.argmax(model(mx.array([[token]], mx.uint32), cache=cache)[0, -1]).item())
        capture[0] = True
        logits = model(mx.array([[token]], mx.uint32), cache=cache)
        capture[0] = False
        mx.eval(logits, [c["args"] for c in calls], [c["kwargs"] for c in calls],
                [c["out"] for c in calls if c["out"] is not None])
        rows = []
        for c in calls:
            if c["out"] is None:  # a declined group or tail: the fallback ran and was captured
                continue
            obj, orig, args, kwargs = c["obj"], c["orig"], c["args"], c["kwargs"]
            nbytes = _bytes_of(obj)
            if c["kind"] == "exl3.tail":
                nbytes = _tail_bytes(obj, args[0], args[2])
            us = 1e6 * _per_call(lambda: orig(obj, *args, **kwargs))
            name = names.get(id(obj))
            if name is None and c["kind"] == "exl3.group":
                name = "+".join(obj.names)
            row = {"kind": c["kind"], "name": name, "us": us}
            if nbytes:
                row["bytes"] = int(nbytes)
                row["GBps"] = nbytes / (us * 1e-6) / 1e9
            rows.append(row)
        del cache, logits
    finally:
        for cls, attr, orig in patched:
            setattr(cls, attr, orig)
        calls.clear()
        gc.collect()
        mx.clear_cache()
    summary = {}
    for row in rows:
        s = summary.setdefault(row["kind"], {"calls": 0, "us": 0.0, "bytes": 0})
        s["calls"] += 1
        s["us"] += row["us"]
        s["bytes"] += row.get("bytes", 0)
    for s in summary.values():
        if s["bytes"]:
            s["GBps"] = s["bytes"] / (s["us"] * 1e-6) / 1e9
    return {"per_kind": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-components", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny counts to check the script runs")
    args = parser.parse_args()
    if args.smoke:
        global PROMPT, TOKENS, CALLS
        PROMPT, TOKENS, CALLS = 32, 4, 2
    if not args.i_own_the_gpu:
        parser.error("this runs Metal kernels; pass --i-own-the-gpu under the GPU lock")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence directory {out}")
    from mlx2.adapters.registry import inspect_model, resolve_adapter

    path = Path(args.model).expanduser().resolve()
    artifact = inspect_model(path).artifact
    receipt = {
        "schema": "mlx2.exl3-decode-bound-profile.v1",
        "checkpoint": f"{_artifact_name(path)}@{artifact['identity']['fingerprint']}",
        "source_revision": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "mlx_version": mx.__version__,
        "prompt_tokens": PROMPT, "tokens": TOKENS, "replay_calls": CALLS, "smoke": args.smoke,
        "note": "sanity profile, not thermally gated; not a published performance figure",
    }
    mx.set_cache_limit(4 << 30)
    receipt["stream"] = _stream()
    print("stream", json.dumps(receipt["stream"]), flush=True)
    adapter = resolve_adapter(path)(path)
    model = adapter.model
    receipt["execution_policy"] = getattr(adapter, "execution_policy", None)
    rng = np.random.default_rng(20261109)
    receipt["token"] = _token_timings(model, rng)
    print("token", json.dumps(receipt["token"]), flush=True)
    if not args.skip_components:
        receipt["components"] = _components(model, rng)
        print("components", json.dumps(receipt["components"]["per_kind"]), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str))
    print("wrote", out / "receipt.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
