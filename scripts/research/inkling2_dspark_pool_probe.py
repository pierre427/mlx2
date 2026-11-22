"""Inkling 2 DSpark long-prefill memory: live tensors or the MLX buffer pool?

The DSpark B1 ladder at 32K was stopped by the memory watchdog at an 84.6 GiB
footprint; ordinary B1 at the same depth stayed at about 25 GB. Ordinary
prefill ends every chunk with ``mx.clear_cache()`` (MLX2_PREFILL_CLEAR_CACHE
defaults to ``always``). The external-draft engine does so only with its
default-off ``prefill_allocator_reclaim`` candidate.

This probe repeats the engine's B1 prefill sequence in 512-token chunks.
Each chunk is ``prefill_body`` with the DSpark taps; the previous chunk's
tail then goes into the draft KV. Three arms:

- ``target``: target only, no clear.
- ``dspark``: target and draft, no clear (the ladder's route).
- ``dspark_clear``: target and draft, with ``mx.clear_cache()`` after each
  chunk.

After each chunk it records active memory, the cached pool and the peak. An
arm stops early, and is marked ``capped``, if active + cache passes the cap.

Usage: python scripts/research/inkling2_dspark_pool_probe.py <model dir> <run dir> [tokens]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
CAP_GIB = 60.0
GIB = 2**30


def corpus_tokens(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py", "scripts/**/*.py", "tests/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    raise SystemExit("corpus too short")


def run_arm(model, draft, layers, tokens, *, use_draft, clear, chunk=CHUNK, cap_gib=CAP_GIB,
            every=4):
    """The engine's B1 prefill order: taps, then the previous tail into draft KV."""
    import mlx.core as mx

    mx.clear_cache()
    mx.reset_peak_memory()
    cache = model.make_cache()
    draft_cache = draft.make_cache() if use_draft else None
    tail, samples, capped = None, [], False
    t0 = time.monotonic()
    for i, a in enumerate(range(0, len(tokens), chunk)):
        if use_draft and tail is not None:
            draft.append_context(tail, draft_cache)
        tail = model.prefill_body(mx.array([tokens[a:a + chunk]]), cache, layers)
        mx.eval(tail, [c.state for c in cache])
        if use_draft:
            mx.eval([c.state for c in draft_cache if c.offset])
        if clear:
            mx.clear_cache()
        active, pooled = mx.get_active_memory() / GIB, mx.get_cache_memory() / GIB
        last = a + chunk >= len(tokens)
        if i % every == every - 1 or last or active + pooled > cap_gib:
            samples.append({"keys": min(a + chunk, len(tokens)), "active_gib": round(active, 2),
                            "cache_gib": round(pooled, 2),
                            "peak_gib": round(mx.get_peak_memory() / GIB, 2)})
        if active + pooled > cap_gib:
            capped = True
            break
    seconds = time.monotonic() - t0
    del cache, draft_cache, tail
    mx.clear_cache()
    return {"use_draft": use_draft, "clear_per_chunk": clear, "capped": capped,
            "seconds": round(seconds, 1), "samples": samples}


def main():
    model_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_dspark_pool_probe.py", "mlx2": head,
               "mlx_version": mx.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "memory diagnosis (no timing claims)", "chunk": CHUNK, "cap_gib": CAP_GIB,
               "arms": {}}
    out = run_dir / "pool_probe.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    adapter = Inkling2Adapter(str(model_dir),
                              execution_policy={"draft_model": "embedded_dspark", "num_draft": 7})
    try:
        receipt["checkpoint"] = adapter.checkpoint
        draft = adapter.draft_model
        while hasattr(draft, "backend"):
            draft = draft.backend
        layers = tuple(draft.config.target_layer_ids)
        receipt["taps"] = list(layers)
        tokens = corpus_tokens(adapter.tokenizer, n)
        receipt["tokens"] = len(tokens)
        for name, use_draft, clear in (("target", False, False), ("dspark", True, False),
                                       ("dspark_clear", True, True)):
            receipt["arms"][name] = run_arm(adapter.model, draft, layers, tokens,
                                            use_draft=use_draft, clear=clear)
            print(name, json.dumps(receipt["arms"][name]["samples"][-1]),
                  "capped" if receipt["arms"][name]["capped"] else "", flush=True)
            save()
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()


if __name__ == "__main__":
    main()
