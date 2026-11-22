"""Inkling 2 decode launch counts on the tiny CPU model (no checkpoint, no GPU).

Builds ``tests/test_inkling2_cpu.py::_tiny_model()``, prefills 64 tokens,
then exports one one-token decode step's lazy graph with ``mx.export_to_dot``
and counts primitive nodes, excluding views (Broadcast, Transpose, Reshape,
Slice, ExpandDims, Squeeze, Flatten, Unflatten, AsStrided). A ``Compiled``
node is one fused launch.

Usage: PYTHONPATH=src python scripts/research/inkling2_launch_count.py
       [--rows N] [--option name=value ...] [--json]
Each ``--option`` is passed to ``Model.configure_kernels``; ``--all-fuse``
turns on every ``fuse_*`` switch.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

VIEWS = {"Broadcast", "Transpose", "Reshape", "Slice", "ExpandDims", "Squeeze",
         "Flatten", "Unflatten", "AsStrided", "BroadcastAxes"}


def count_graph(*outputs):
    import mlx.core as mx

    with tempfile.NamedTemporaryFile("w+", suffix=".dot") as handle:
        mx.export_to_dot(handle.name, *outputs)
        text = Path(handle.name).read_text()
    counts = collections.Counter()
    for label in re.findall(r'label\s*=\s*"([^"]*)", shape=rectangle', text):
        counts[label.split()[0] if label else "?"] += 1
    total = sum(counts.values())
    kernels = sum(v for k, v in counts.items() if k not in VIEWS)
    return {"nodes": total, "kernels": kernels, "by_primitive": dict(counts.most_common())}


def _arrays(value):
    import mlx.core as mx

    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in _arrays(v)]
    if isinstance(value, dict):
        return [a for v in value.values() for a in _arrays(v)]
    return []


def cache_arrays(cache):
    """Every array the step leaves in the cache (KV buffers, conv states):
    the next step evaluates them, so their producers are launches too."""
    out = []
    for plane in cache:
        for name in ("keys", "values", "cache", "offset_array"):
            out.extend(_arrays(getattr(plane, name, None)))
    return out


def decode_step(model, rows=1, prefill=64, seed=0, device=None, prefill_gpu=False):
    """``(logits, cache)`` of one lazy one-token decode step.

    The prefill always runs on the CPU.  ``device="gpu"`` only *builds* the
    decode graph on the GPU stream (nothing is evaluated there), so the
    count reflects GPU primitives (``prefill_gpu`` evaluates the prefill on
    the GPU instead, for BF16 experts, which the CPU GatherMM refuses): ``mx.fast`` ops are single primitives on
    the GPU but decompose into their fallbacks on the CPU."""
    import mlx.core as mx

    mx.random.seed(seed + 1)
    tokens = mx.random.randint(0, 500, (rows, prefill + 1))
    cache = model.make_cache()
    with mx.stream(mx.gpu if prefill_gpu else mx.cpu):
        mx.eval(model(tokens[:, :prefill], cache=cache), cache_arrays(cache))
    if device == "gpu":
        with mx.stream(mx.gpu):
            return model(tokens[:, prefill:], cache=cache), cache
    return model(tokens[:, prefill:], cache=cache), cache


MODULES = ("Attention", "MessageController", "AttentionResidual", "SparseMoE",
           "ShortConvolution", "DecoderLayer")


def by_module(model, rows=1):
    """Kernels per module class: each call's inputs (and the cache it may
    write) are evaluated first, so its graph holds only its own launches."""
    import mlx.core as mx

    from mlx2.runtime.models.inkling2 import language

    totals = {name: collections.Counter() for name in MODULES}
    calls = collections.Counter()
    state = {"on": False, "depth": 0}
    originals = {}

    def wrap(cls):
        original = cls.__call__

        def call(self, *args, **kwargs):
            if not state["on"] or state["depth"]:
                return original(self, *args, **kwargs)
            cache = kwargs.get("cache")
            mx.eval(_arrays(args), _arrays(kwargs), _arrays(getattr(cache, "caches", None) and
                                                            cache_arrays(cache.caches)))
            state["depth"] += 1
            try:
                out = original(self, *args, **kwargs)
            finally:
                state["depth"] -= 1
            side = cache_arrays(cache.caches) if hasattr(cache, "caches") else []
            if cls.__name__ == "ShortConvolution" and args[1:2] and args[1] is not None:
                side = _arrays(args[1].cache)
            result = count_graph(*_arrays(out), *side)
            totals[cls.__name__].update({k: v for k, v in result["by_primitive"].items()
                                         if k not in VIEWS})
            calls[cls.__name__] += 1
            return out

        originals[cls] = original
        cls.__call__ = call

    for name in MODULES[:-1]:
        wrap(getattr(language, name))
    try:
        mx.random.seed(1)
        tokens = mx.random.randint(0, 500, (rows, 65))
        cache = model.make_cache()
        mx.eval(model(tokens[:, :64], cache=cache), cache_arrays(cache))
        state["on"] = True
        mx.eval(model(tokens[:, 64:], cache=cache))
    finally:
        for cls, original in originals.items():
            cls.__call__ = original
    return {name: {"calls": calls[name], "kernels": sum(c.values()),
                   "per_call": round(sum(c.values()) / max(1, calls[name]), 1),
                   "by_primitive": dict(c.most_common())}
            for name, c in totals.items() if calls[name]}


def main():
    import mlx.core as mx

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--all-fuse", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--by-module", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="cast the tiny model to BF16")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                        help="gpu: build (never evaluate) the decode graph on the GPU stream")
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    from test_inkling2_cpu import _tiny_model

    model = _tiny_model()
    if args.bf16:
        from mlx.utils import tree_flatten

        # CPU GatherMM is FP32-only: the routed experts stay FP32.
        model.load_weights([(k, v if ".switch_mlp." in k else v.astype(mx.bfloat16))
                            for k, v in tree_flatten(model.parameters())])
        mx.eval(model.parameters())
    options = {}
    for item in args.option:
        key, value = item.split("=", 1)
        options[key] = json.loads(value)
    if args.all_fuse:
        import inspect

        params = inspect.signature(model.configure_kernels).parameters
        options.update({name: True for name in params if name.startswith("fuse_")})
    if options:
        model.configure_kernels(**options)
    out, cache = decode_step(model, rows=args.rows, device=args.device)
    result = {"rows": args.rows, "device": args.device, "bf16": args.bf16, "options": options,
              **count_graph(out, *cache_arrays(cache))}
    if args.by_module:
        result["modules"] = by_module(model, rows=args.rows)
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"options={options} rows={args.rows} nodes={result['nodes']} "
              f"kernels={result['kernels']}")
        for name, value in result["by_primitive"].items():
            print(f"  {name:28s} {value}")
        for name, info in result.get("modules", {}).items():
            top = ", ".join(f"{k} {v}" for k, v in list(info["by_primitive"].items())[:8])
            print(f"{name}: {info['calls']} calls, {info['kernels']} kernels "
                  f"({info['per_call']}/call): {top}")


if __name__ == "__main__":
    main()
