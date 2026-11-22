"""Inkling 2 long-context prefill profile: what grows with context?

One 512-token-chunk prefill of varied repo text runs to about 64K tokens.
At three depths (about 4K, 16K and 62K keys) the next four chunks are timed:

- ``wall``: plain run, wall time only.
- ``hooked``: every component synchronized and timed. The total is
  inflated, but the split is per component.
- ``flash_wall`` and ``flash_hooked``: the same with
  ``inkling2_flash_prefill`` on.

Each timed chunk also advances the prefill, so state is consistent. Hooked
components:
- global vs local ``banded_additive_mask`` and SDPA;
- global vs local attention in total, and the remaining attention work
  (projections, convs, fetch);
- MoE;
- attention residual mixes;
- loop controllers;
- the decoder layer in total, and the rest of the layer.

Usage: python scripts/research/inkling2_prefill_profile.py <model dir> <run dir>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512


def corpus_tokens(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py", "scripts/**/*.py", "tests/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
    if len(ids) < n:
        raise SystemExit(f"only {len(ids)} corpus tokens")
    return ids[:n]


class Profiler:
    """Class/function hooks that synchronize and accumulate wall time."""

    def __init__(self, language):
        self.lang = language
        self.ms = defaultdict(float)
        self.calls = defaultdict(int)
        self.active = False

    def _time(self, key, fn, *args, **kwargs):
        import mlx.core as mx

        if not self.active:
            return fn(*args, **kwargs)
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        mx.eval(out)
        mx.synchronize()
        self.ms[key] += (time.perf_counter() - t0) * 1e3
        self.calls[key] += 1
        return out

    @contextmanager
    def installed(self):
        import mlx.core as mx

        lang, prof = self.lang, self
        saved = {
            "mask": lang.banded_additive_mask,
            "sdpa": mx.fast.scaled_dot_product_attention,
            "flash": lang.flash_relative_attention,
            "attn": lang.Attention.__call__,
            "moe": lang.SparseMoE.__call__,
            "res": lang.AttentionResidual.__call__,
            "ctrl": lang.MessageController.__call__,
            "layer": lang.DecoderLayer.__call__,
        }
        state = {"kind": "local"}

        def mask(*a, **k):
            return prof._time(f"mask_{state['kind']}", saved["mask"], *a, **k)

        def sdpa(*a, **k):
            return prof._time(f"sdpa_{state['kind']}", saved["sdpa"], *a, **k)

        def flash(*a, **k):
            return prof._time(f"flash_{state['kind']}", saved["flash"], *a, **k)

        def attn(module, *a, **k):
            state["kind"] = "local" if module.is_sliding else "global"
            return prof._time(f"attention_{state['kind']}", saved["attn"], module, *a, **k)

        def moe(module, *a, **k):
            return prof._time("moe", saved["moe"], module, *a, **k)

        def res(module, *a, **k):
            return prof._time("attention_residual", saved["res"], module, *a, **k)

        def ctrl(module, *a, **k):
            return prof._time("controllers", saved["ctrl"], module, *a, **k)

        def layer(module, *a, **k):
            return prof._time("decoder_layer", saved["layer"], module, *a, **k)

        lang.banded_additive_mask = mask
        mx.fast.scaled_dot_product_attention = sdpa
        lang.flash_relative_attention = flash
        lang.Attention.__call__ = attn
        lang.SparseMoE.__call__ = moe
        lang.AttentionResidual.__call__ = res
        lang.MessageController.__call__ = ctrl
        lang.DecoderLayer.__call__ = layer
        try:
            yield self
        finally:
            lang.banded_additive_mask = saved["mask"]
            mx.fast.scaled_dot_product_attention = saved["sdpa"]
            lang.flash_relative_attention = saved["flash"]
            lang.Attention.__call__ = saved["attn"]
            lang.SparseMoE.__call__ = saved["moe"]
            lang.AttentionResidual.__call__ = saved["res"]
            lang.MessageController.__call__ = saved["ctrl"]
            lang.DecoderLayer.__call__ = saved["layer"]

    def take(self):
        ms = dict(self.ms)
        for kind in ("global", "local"):
            inner = sum(ms.get(f"{p}_{kind}", 0.0) for p in ("mask", "sdpa", "flash"))
            if f"attention_{kind}" in ms:
                ms[f"attention_{kind}_other"] = ms[f"attention_{kind}"] - inner
        parts = ("attention_global", "attention_local", "moe", "attention_residual", "controllers")
        if "decoder_layer" in ms:
            ms["decoder_layer_other"] = ms["decoder_layer"] - sum(ms.get(p, 0.0) for p in parts)
        out = {k: round(v, 1) for k, v in sorted(ms.items())}
        calls = dict(self.calls)
        self.ms.clear()
        self.calls.clear()
        return out, calls


def run(model, tokens, *, depths, set_flash, chunk=CHUNK):
    """Prefill ``tokens``; at each depth, time four consecutive chunks."""
    import mlx.core as mx

    from mlx2.runtime.models.inkling2 import language

    prof = Profiler(language)
    cache = model.make_cache()
    plan = {}
    for depth in depths:
        start = depth // chunk * chunk
        for i, label in enumerate(("wall", "hooked", "flash_wall", "flash_hooked")):
            plan[start + i * chunk] = (depth, label)
    results = defaultdict(dict)
    with prof.installed():
        for a in range(0, len(tokens), chunk):
            item = plan.get(a)
            ids = mx.array([tokens[a:a + chunk]])
            if item is None:
                mx.eval(model.model(ids, cache))
                continue
            depth, label = item
            set_flash(label.startswith("flash"))
            prof.active = label.endswith("hooked")
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(model.model(ids, cache))
            mx.synchronize()
            wall = round((time.perf_counter() - t0) * 1e3, 1)
            prof.active = False
            entry = {"keys_after_chunk": a + chunk, "wall_ms": wall}
            if label.endswith("hooked"):
                entry["components_ms"], entry["calls"] = prof.take()
            results[f"depth_{depth}"][label] = entry
            set_flash(False)
            print(label, depth, json.dumps({k: v for k, v in entry.items() if k != "calls"}),
                  flush=True)
    return dict(results)


def main():
    model_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_prefill_profile.py", "mlx2": head,
               "mlx_version": mx.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "profile (hooked timings synchronize every component; attribution only)"}
    adapter = Inkling2Adapter(str(model_dir))
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        tokens = corpus_tokens(adapter.tokenizer, 64512)

        def set_flash(on):
            model.configure_kernels(**{**model.kernel_policy, "flash_prefill": on})

        t0 = time.monotonic()
        receipt["profile"] = run(model, tokens, depths=(4096, 16384, 61440), set_flash=set_flash)
        receipt["seconds"] = round(time.monotonic() - t0, 1)
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = run_dir / "profile.json.partial"
    tmp.write_text(json.dumps(receipt, indent=1, default=str))
    os.replace(tmp, run_dir / "profile.json")


if __name__ == "__main__":
    main()
