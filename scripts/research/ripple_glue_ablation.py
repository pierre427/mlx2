#!/usr/bin/env python3
"""Attribute Ripple decode time above the weight floor (Metal, one artifact).

Pipelined decode ms/token over a ~``--context`` cache, as served and with one
glue stage at a time replaced by a cheap stand-in that keeps every data
dependency (so lazy evaluation cannot prune upstream matmuls, the flaw of an
SDPA stub that drops its inputs):

* ``norms``   every RMSNorm (block and per-head Q/K) -> ``x * w``
* ``rope``    split-half rotation -> identity (tables still built)
* ``blend``   value-residual lerp -> ``x``
* ``gate``    FP32 sigmoid gate -> ``out * g`` (g_proj matmul kept)
* ``swiglu``  clamp + FP32 SiLU -> ``gate * up``
* ``all``     every stand-in at once

Tokens under a stand-in are meaningless; only time is recorded. Also:

* weight floor by projection role (q/k/v/g/o/gate/up/down/lm_head): each
  role's modules called back to back on one row, with GB/s;
* verify attention as served on speculative transactions (8 rows, explicit
  bool masks on both layer kinds): stock SDPA vs GQA-folded, x80 calls.
Requires the caller's GPU ownership. Writes one JSON receipt.
"""
import argparse
import contextlib
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

ROLES = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")


@contextlib.contextmanager
def patched(module, **values):
    original = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)


def gate_standin_attend(self, x, q, k, v, cache, mask):
    """``Attention._attend`` with the gate glue as one multiply."""
    from mlx2.runtime.models import ricursive

    args = self.args
    b, length, _ = x.shape
    if cache is not None:
        k, v = cache.update_and_fetch(k, v)
    out = ricursive.scaled_dot_product_attention(q, k, v, cache=cache, scale=args.head_dim ** -0.5, mask=mask)
    gate = self.g_proj(x).reshape(b, length, args.num_attention_heads, args.head_dim)
    return self.o_proj((out.transpose(0, 2, 1, 3) * gate).reshape(b, length, -1))


def variants():
    from mlx2.runtime.models import ricursive

    norms = dict(_rms=lambda x, w, eps: x * w)
    rope = dict(apply_rotary=lambda x, cos, sin: x)
    blend = dict(_blend=lambda seed32, x, blend: x)
    swiglu = dict(_swiglu=lambda gate, up, clamp: gate * up)
    return {
        "served": [],
        "norms": [(ricursive, norms)],
        "rope": [(ricursive, rope)],
        "blend": [(ricursive, blend)],
        "gate": [(ricursive.Attention, dict(_attend=gate_standin_attend))],
        "swiglu": [(ricursive, swiglu)],
        "all": [(ricursive, {**norms, **rope, **blend, **swiglu}),
                (ricursive.Attention, dict(_attend=gate_standin_attend))],
    }


def decode_ms(model, cache, first, steps):
    pending = mx.argmax(model(first, cache=cache)[:, -1:], axis=-1)
    mx.async_eval(pending)
    start = time.perf_counter()
    for _ in range(steps):
        nxt = mx.argmax(model(pending, cache=cache)[:, -1:], axis=-1)
        mx.async_eval(nxt)
        pending.item()
        pending = nxt
    mx.eval(pending)
    return (time.perf_counter() - start) / steps * 1e3


def floor_by_role(model):
    groups = {role: [] for role in ROLES}
    for name, module in model.named_modules():
        role = name.rsplit(".", 1)[-1]
        if role in groups and hasattr(module, "weight"):
            groups[role].append(module)
    out = {}
    for role, modules in groups.items():
        if not modules:
            continue
        first = modules[0]
        width = (first.weight.shape[-1] * 32 // first.bits if isinstance(first, nn.QuantizedLinear)
                 else first.weight.shape[-1])
        x = mx.zeros((1, 1, width), dtype=mx.bfloat16) + 0.01
        mx.eval(x)
        nbytes = sum(v.nbytes for m in modules for _, v in tree_flatten(m.parameters()))
        run = lambda: mx.eval([m(x) for m in modules])
        run()
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            run()
            samples.append(time.perf_counter() - start)
        median = sorted(samples)[3]
        bits = sorted({getattr(m, "bits", 16) for m in modules})
        out[role] = {"modules": len(modules), "bytes": nbytes, "bits": bits,
                     "median_ms": median * 1e3, "gb_per_s": nbytes / median / 1e9}
    return out


def verify_attention(context, rows=8, layers=80):
    from mlx2.runtime.models.base import create_causal_mask
    from mlx2.runtime.models.ricursive import gqa_folded_attention

    rng = np.random.default_rng(3)

    def a(shape):
        return mx.array(rng.normal(size=shape).astype(np.float32)).astype(mx.bfloat16)

    out, scale = {}, 128 ** -0.5
    q = a((1, 80, rows, 128))
    for name, n, window in (("sliding", 1023 + rows, 1024), ("global", context + rows, None)):
        k, v = a((1, 8, n, 128)), a((1, 8, n, 128))
        mask = create_causal_mask(rows, offset=n - rows, window_size=window)  # explicit, as served
        mx.eval(q, k, v, mask)
        stock = lambda: mx.eval([mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
                                 for _ in range(layers)])
        fold = lambda: mx.eval([gqa_folded_attention(q, k, v, scale=scale, mask=mask) for _ in range(layers)])
        timings = {}
        for label, fn in (("stock_x80", stock), ("folded_x80", fold)):
            fn()
            samples = []
            for _ in range(9):
                start = time.perf_counter()
                fn()
                samples.append(time.perf_counter() - start)
            timings[label] = sorted(samples)[4] * 1e3
        out[f"verify{rows}_{name}_{n}"] = timings
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from mlx2.adapters.ripple import RippleAdapter

    adapter = RippleAdapter(str(args.model))
    model = adapter.model
    text = (Path(__file__).resolve().parents[2] / "docs" / "ripple-65b.md").read_text()
    ids = adapter.tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < args.context:
        ids = ids + ids
    cache = model.make_cache()
    for start in range(0, args.context, 512):
        mx.eval(model(mx.array([ids[start:min(args.context, start + 512)]]), cache=cache))
    first = mx.array([[ids[0]]])
    receipt = {"schema": "mlx2.ripple-glue-ablation.v1", "mlx": mx.__version__,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "checkpoint": args.model.name + "@sha256:" + adapter.identity["fingerprint"],
               "context": args.context, "decode_ms": {}}
    order = list(variants())
    # served first and last brackets thermal drift.
    for name in [*order, "served"]:
        with contextlib.ExitStack() as stack:
            for target, values in variants()[name]:
                stack.enter_context(patched(target, **values))
            key = name if name not in receipt["decode_ms"] else "served_again"
            receipt["decode_ms"][key] = decode_ms(model, cache, first, args.steps)
    served = (receipt["decode_ms"]["served"] + receipt["decode_ms"]["served_again"]) / 2
    receipt["saving_ms_upper_bound"] = {name: served - value for name, value in receipt["decode_ms"].items()
                                        if name not in ("served", "served_again")}
    receipt["floor_by_role"] = floor_by_role(model)
    receipt["kv_offset_at_end"] = cache[4].offset
    adapter.close()
    mx.clear_cache()
    receipt["verify_attention"] = verify_attention(args.context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"decode_ms": receipt["decode_ms"], "saving": receipt["saving_ms_upper_bound"]}))
    print(json.dumps({r: (round(v["median_ms"], 2), round(v["gb_per_s"])) for r, v in receipt["floor_by_role"].items()}))
    print(json.dumps(receipt["verify_attention"]))


if __name__ == "__main__":
    main()
