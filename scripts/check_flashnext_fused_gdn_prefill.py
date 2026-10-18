#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""GPU bit gate for the Flash-Next fused GDN prefill (default on) on real weights.

The Flash-Next adapter selects the omlx #3903 prefill prework and gated norm
(``MLX_QWEN4_FUSED_GDN_PREFILL=1``) as bit-identical to the eager prefill.
This check loads one artifact through the adapter and prefills each prompt in
fixed chunks with every GDN layer's prefill mode ``stock`` and then
``fused``; the last chunk's logits, every cache entry and a short greedy
continuation must be bit-identical, and the fused arm must have engaged on
every >= 64-row chunk of every GDN layer.  It also counts, per arm, conv
outputs of the exact bf16 value -6.84375 (the one input where the
``metal::exp`` and ``metal::precise::exp`` bf16 sigmoids round apart) seen by
the eager reference, so a pass can be told apart from a prompt that never
reaches it.  Per-arm prefill wall time is recorded but is not a controlled
performance measurement.

Never selects or qualifies a route.  No model load or Metal work happens
before ``--i-own-the-gpu`` and ``MLX_ENABLE_TF32=0`` are checked.

    MLX_ENABLE_TF32=0 PYTHONPATH=src python scripts/check_flashnext_fused_gdn_prefill.py \\
        --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
        --out receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("qwen4_exp.py", "qwen4_fused_gdn_prefill.py", "qwen4_fused_gdn.py", "qwen3_5.py",
           "gated_delta.py", "served_exp.py")
EDGE = -6.84375


def _prompt_texts(root: Path) -> dict:
    corpus = []
    for name in ("docs/SERVING.md", "docs/QUALIFICATION.md", "AGENTS.md", "docs/PROVENANCE.md"):
        path = root / name
        if path.is_file():
            corpus.append(path.read_text(errors="replace"))
    text = "\n\n".join(corpus)
    return {
        "short": "Explain how a hash map handles collisions, in one paragraph.",
        "medium": text[:1500],
        "long": text[:24000],
    }


def _same(mx, a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    unsigned = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[a.dtype.size]
    return bool(mx.array_equal(a.view(unsigned), b.view(unsigned)).item())


def _flat_state(cache) -> list:
    out = []
    for entry in cache:
        state = entry.state
        for item in state if isinstance(state, (list, tuple)) else [state]:
            if isinstance(item, (list, tuple)):
                out.extend(x for x in item if x is not None)
            elif item is not None:
                out.append(item)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", nargs="+", type=int, default=[512, 2048])
    ap.add_argument("--prompts", default="short,medium,long")
    ap.add_argument("--continuation", type=int, default=8)
    ap.add_argument("--src", type=Path, default=ROOT / "src",
                    help="mlx2 source root to import (A/B against another checkout)")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        ap.error("set MLX_ENABLE_TF32=0")
    src = args.src.expanduser().resolve()
    sys.path.insert(0, str(src))
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    model_path = Path(args.model).expanduser().resolve()
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    report = {
        "schema": "mlx2.flashnext-fused-gdn-prefill-check.v1",
        "model": str(model_path), "src": str(src), "git_head": head,
        "mlx_version": mx.__version__,
        "source_sha256": {
            name: hashlib.sha256((src / "mlx2/runtime/models" / name).read_bytes()).hexdigest()
            for name in SOURCES if (src / "mlx2/runtime/models" / name).is_file()
        },
        "qualified": False, "route_selected": "adapter default (unchanged by this check)",
        "passed": False,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["model_check"] = check_model(args, mx, model_path)
        report["passed"] = report["model_check"]["passed"]
    except Exception as exc:  # noqa: BLE001 - recorded in the receipt
        import traceback

        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "error": report.get("error")}))
    return 0 if report["passed"] else 1


def check_model(args, mx, model_path):
    from mlx2.adapters.flash_next import FlashNextAdapter

    import mlx2

    if not Path(mlx2.__file__).resolve().is_relative_to(Path(args.src).expanduser().resolve()):
        raise RuntimeError(f"mlx2 imported from {mlx2.__file__}, not --src")

    started = time.perf_counter()
    adapter = FlashNextAdapter(str(model_path))
    load_s = time.perf_counter() - started
    model = adapter.model
    layers = [layer.linear_attn for layer in model.language_model.model.layers
              if getattr(layer, "linear_attn", None) is not None]
    edge_hits = {"count": 0}

    # Count conv outputs of exactly EDGE on the eager arm: the GDN recurrent
    # core (qwen3_5) sees a proxy ``nn`` whose ``silu`` counts the conv SiLU
    # inputs and returns the real op unchanged.  mlx.nn itself is untouched.
    from mlx2.runtime.models import qwen3_5

    real_nn = qwen3_5.nn
    conv_dim = int(layers[0].conv_dim)

    class CountingNN:
        def __getattr__(self, name):
            return getattr(real_nn, name)

        @staticmethod
        def silu(x):
            if x.ndim == 3 and x.shape[-1] == conv_dim:
                edge_hits["count"] += int((x == EDGE).sum().item())
            return real_nn.silu(x)

    prompts = {name: list(adapter.tokenizer.encode(text))
               for name, text in _prompt_texts(ROOT).items()
               if name in args.prompts.split(",")}

    def prefill(ids, chunk, mode):
        for layer in layers:
            layer.set_fused_gdn_prefill_mode(mode)
        cache = model.make_cache()
        before = sum(int(layer.fused_gdn_prefill_calls) for layer in layers)
        logits = None
        t0 = time.perf_counter()
        for start in range(0, len(ids), chunk):
            logits = model(mx.array(ids[start:start + chunk])[None], cache=cache)
            mx.eval(logits, _flat_state(cache))
        prefill_s = time.perf_counter() - t0
        tokens = []
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        for _ in range(args.continuation):
            tokens.append(int(nxt.item()))
            step = model(nxt[None], cache=cache)
            nxt = mx.argmax(step[:, -1, :], axis=-1)
        calls = sum(int(layer.fused_gdn_prefill_calls) for layer in layers) - before
        return logits, cache, tokens, calls, prefill_s

    cases = {}
    for name, ids in prompts.items():
        for chunk in args.chunks:
            if name == "short" and chunk != args.chunks[0]:
                continue
            edge_hits["count"] = 0
            qwen3_5.nn = CountingNN()
            try:
                ref = prefill(ids, chunk, "stock")
            finally:
                qwen3_5.nn = real_nn
            hits = edge_hits["count"]
            got = prefill(ids, chunk, "fused")
            ref_state, got_state = _flat_state(ref[1]), _flat_state(got[1])
            states_equal = len(ref_state) == len(got_state) and all(
                _same(mx, a, b) for a, b in zip(ref_state, got_state))
            differing = [i for i, (a, b) in enumerate(zip(ref_state, got_state)) if not _same(mx, a, b)]
            expected = sum(1 for s in range(0, len(ids), chunk) if min(chunk, len(ids) - s) >= 64)
            case = {
                "tokens": len(ids), "chunk": chunk,
                "eager_conv_outputs_at_edge": hits,
                "logits_bit_identical": _same(mx, ref[0], got[0]),
                "cache_bit_identical": states_equal,
                "differing_cache_entries": differing[:16],
                "differing_cache_entry_count": len(differing),
                "continuation_identical": ref[2] == got[2],
                "continuation": got[2],
                "fused_prefill_calls": got[3],
                "reference_prefill_calls": ref[3],
                "expected_fused_calls": expected * len(layers),
                "stock_prefill_s": round(ref[4], 3),
                "fused_prefill_s": round(got[4], 3),
                "fallback_reasons": {k: v for layer in layers[:1]
                                     for k, v in layer.fused_gdn_prefill_fallback_reasons.items()},
            }
            case["passed"] = (case["logits_bit_identical"] and states_equal
                              and case["continuation_identical"]
                              and ref[3] == 0 and got[3] == case["expected_fused_calls"])
            cases[f"{name}/{chunk}"] = case
            print(json.dumps({"case": f"{name}/{chunk}", "passed": case["passed"],
                              "edge": hits, "differing": len(differing)}), flush=True)
            del ref, got
            mx.clear_cache()
    for layer in layers:
        layer.set_fused_gdn_prefill_mode("fused")
    return {"load_s": round(load_s, 1), "gdn_layers": len(layers), "cases": cases,
            "passed": bool(cases) and all(c["passed"] for c in cases.values())}


if __name__ == "__main__":
    raise SystemExit(main())
