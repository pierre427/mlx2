#!/usr/bin/env python3
"""Bounded LFM2.5 24B-A2B GPU smoke: one arm per process, one model resident.

Arms (``--arm``):
  unified  mlx-lm-unified ``lfm2_moe`` on the same weights (external reference)
  mlx2     the mlx2 adapter, reference ShortConv and stock expert gathers
  fused    mlx2 with ``MLX2_LFM25_FUSED_SHORTCONV=1``
  nax      mlx2 with ``MLX2_MOE_NAX_GATHER=gather``

Each arm writes ``<out>/<arm>.json`` (greedy tokens, decoded text, timing,
peak memory, swapouts) and ``<out>/<arm>.npz`` (final-position logits of the
chat prompt and of every greedy step), so arms compare offline.  Timings are
indicative only: one host, no thermal control, not a performance claim.
Run each arm under ``scripts/run_with_gpu_locks.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = Path.home() / "Desktop/mlx-uag/mlx-lm-unified"
ARM_ENV = {
    "unified": {},
    "mlx2": {"MLX2_LFM25_FUSED_SHORTCONV": "0", "MLX2_MOE_NAX_GATHER": "off"},
    "fused": {"MLX2_LFM25_FUSED_SHORTCONV": "1", "MLX2_MOE_NAX_GATHER": "off"},
    "nax": {"MLX2_LFM25_FUSED_SHORTCONV": "0", "MLX2_MOE_NAX_GATHER": "gather"},
}
CHAT = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]


def swapouts() -> int:
    text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(re.search(r"Swapouts:\s+(\d+)", text).group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--arm", choices=sorted(ARM_ENV), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--greedy-tokens", type=int, default=48)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    args = parser.parse_args()

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", **ARM_ENV[args.arm])
    if args.arm == "unified":
        sys.path.insert(0, str(UNIFIED))
    else:
        sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    import numpy as np
    from transformers import AutoTokenizer

    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(4 * 1024**3)
    swap_start = swapouts()
    started = time.perf_counter()
    diagnostics = {}
    if args.arm == "unified":
        from mlx_lm.utils import load_model

        model, _config = load_model(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        rendered = tokenizer.apply_chat_template(
            CHAT, add_generation_prompt=True, tokenize=False
        )
        prompt = list(tokenizer.encode(rendered, add_special_tokens=False))
        make_cache = model.make_cache
        decode = tokenizer.decode
        close = lambda: None  # noqa: E731
    else:
        from mlx2.adapters.registry import inspect_model

        resolution = inspect_model(args.model_path)
        adapter = resolution.adapter_type(str(args.model_path))
        model = adapter.model
        prompt = list(adapter.prompt_tokens({"messages": CHAT}))
        make_cache = model.make_cache
        decode = adapter.tokenizer.decode
        close = adapter.close
        diagnostics["fingerprint"] = adapter.identity["fingerprint"]
    load_s = time.perf_counter() - started
    mx.synchronize()

    try:
        def step(tokens, cache):
            logits = model(mx.array([tokens], dtype=mx.int32), cache=cache)
            return logits[0, -1].astype(mx.float32)

        # Greedy chat decode, logits kept per step.
        cache = make_cache()
        row = step(prompt, cache)
        rows, greedy = [row], []
        for _ in range(args.greedy_tokens):
            token = int(mx.argmax(row).item())
            greedy.append(token)
            row = step([token], cache)
            rows.append(row)
        mx.eval(rows)
        offsets = sorted({int(getattr(c, "offset", -1)) for c in cache if hasattr(c, "offset")})

        # Prefill timing on a synthetic long prompt (two timed repeats after one warm-up).
        long_prompt = (prompt * (args.prefill_tokens // len(prompt) + 1))[: args.prefill_tokens]
        prefill = []
        for _ in range(3):
            cache = make_cache()
            mx.synchronize()
            t0 = time.perf_counter()
            for start in range(0, len(long_prompt), 2048):
                out = step(long_prompt[start : start + 2048], cache)
                mx.eval(out, [c.state for c in cache])
            mx.synchronize()
            prefill.append(time.perf_counter() - t0)

        # Decode timing after the long prefill.
        token = int(mx.argmax(out).item())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.decode_tokens):
            out = step([token], cache)
            token = int(mx.argmax(out).item())
        mx.synchronize()
        decode_s = time.perf_counter() - t0

        if args.arm != "unified":
            counters = getattr(adapter, "fused_shortconv_counters", None)
            diagnostics["fused_shortconv"] = vars(counters) if counters else None
            diagnostics["execution_config"] = adapter.execution_config(
                max_lanes=1, prefill_step=2048
            )
            from mlx2.runtime.models import moe_nax_gather, switch_layers

            diagnostics["moe_nax_gather_mode"] = moe_nax_gather.MODE
            diagnostics["moe_pad_status"] = switch_layers.moe_pad_status()
            diagnostics["moe_nax_status"] = moe_nax_gather.status()

        args.out.mkdir(parents=True, exist_ok=True)
        np.savez(args.out / f"{args.arm}.npz", rows=np.array(mx.stack(rows)))
        receipt = {
            "schema": "mlx2.lfm25-24b-a2b.gpu-smoke.v2",
            "timestamp": datetime.now(UTC).isoformat(),
            "arm": args.arm,
            "env": ARM_ENV[args.arm],
            "qualification": "unqualified",
            "performance_claim": False,
            "device": mx.device_info().get("device_name"),
            "model_path": str(args.model_path),
            "load_s": round(load_s, 2),
            "prompt_tokens": len(prompt),
            "greedy_tokens": greedy,
            "greedy_text": decode(greedy),
            "greedy_finite": bool(mx.all(mx.isfinite(mx.stack(rows))).item()),
            "cache_offsets": offsets,
            "prefill_tokens": len(long_prompt),
            "prefill_s": [round(v, 3) for v in prefill],
            "prefill_tok_s_best": round(len(long_prompt) / min(prefill[1:]), 1),
            "decode_tokens": args.decode_tokens,
            "decode_tok_s": round(args.decode_tokens / decode_s, 2),
            "peak_memory_gib": round(mx.get_peak_memory() / 1024**3, 2),
            "swapouts_delta": swapouts() - swap_start,
            **diagnostics,
        }
        (args.out / f"{args.arm}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k != "greedy_tokens"}, indent=2))
    finally:
        close()


if __name__ == "__main__":
    main()
