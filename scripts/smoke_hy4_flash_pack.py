#!/usr/bin/env python3
"""Ordinary-decode smoke of a Hy4-flash artifact through the mlx2 adapter.

Original mlx2 script.  Loads the artifact with ``HYV4FlashAdapter`` (the
ordinary route, with the given execution policy), greedy-decodes a few
chat prompts with a functional oracle each, and writes one atomic receipt
``smoke.json`` naming ``<artifact>@<fingerprint>``, the mlx2 commit, the
route receipt (``adapter.diagnostics()``), swap before/after and peak
memory.  Refuses a nonempty run directory.  Run it under
``scripts/run_with_gpu_locks.py``; it never touches port 8282.

Usage::

    smoke_hy4_flash_pack.py MODEL RUN_DIR [--policy JSON] [--max-tokens N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = (
    ("capital", "What is the capital of France? Answer in one word.", r"\bParis\b"),
    ("arithmetic", "What is 17 + 25? Reply with the number only.", r"\b42\b"),
    ("prose", "Write one sentence about a lighthouse keeper.", r"[A-Za-z]{3,}.*[A-Za-z]{3,}"),
)


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    match = re.search(r"used = ([\d.]+)M", out)
    return float(match.group(1)) if match else -1.0


def git_commit() -> dict:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    return {"commit": head, "dirty": dirty}


def greedy(adapter, tokens, max_tokens, eos):
    import mlx.core as mx

    model = adapter.model
    cache = model.make_cache()
    logits = model(mx.array([tokens]), cache)
    out, first_logits = [], None
    for _ in range(max_tokens):
        last = logits[0, -1].astype(mx.float32)
        if first_logits is None:
            first_logits = last
            if not bool(mx.all(mx.isfinite(last)).item()):
                raise RuntimeError("nonfinite logits on the first decode step")
        token = int(mx.argmax(last).item())
        out.append(token)
        if token in eos:
            break
        logits = model(mx.array([[token]]), cache)
    top2 = mx.topk(first_logits, 2)
    return out, sorted(float(v) for v in top2.tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--max-tokens", type=int, default=48)
    a = parser.parse_args(argv)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    if any(a.run_dir.iterdir()):
        raise SystemExit(f"{a.run_dir} is not empty; use a fresh run directory")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter

    receipt = {"script": "scripts/smoke_hy4_flash_pack.py", "mlx2": git_commit(),
               "mlx_version": mx.__version__, "policy": json.loads(a.policy),
               "swap_used_mb_before": swap_used_mb(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "qualification": "unqualified (smoke only; not a qualification run)"}
    t0 = time.monotonic()
    adapter = HYV4FlashAdapter(str(a.model), execution_policy=receipt["policy"])
    try:
        receipt["load_seconds"] = round(time.monotonic() - t0, 1)
        receipt["checkpoint"] = f"{a.model.expanduser().resolve().name}@{adapter.identity['fingerprint']}"
        receipt["active_memory_gb_after_load"] = round(mx.get_active_memory() / 2**30, 2)
        configured = json.loads((a.model.expanduser() / "config.json").read_text())
        configured = (configured.get("text_config") or {}).get("eos_token_id", configured.get("eos_token_id"))
        eos = {int(t) for t in (configured if isinstance(configured, list) else [configured]) if t is not None}
        if adapter.tokenizer.eos_token_id is not None:
            eos.add(int(adapter.tokenizer.eos_token_id))
        receipt["eos_token_ids"] = sorted(eos)
        cases = []
        for name, prompt, oracle in PROMPTS:
            tokens = adapter.prompt_tokens({"messages": [{"role": "user", "content": prompt}]})
            t1 = time.monotonic()
            out, top2 = greedy(adapter, tokens, a.max_tokens, eos)
            seconds = time.monotonic() - t1
            text = adapter.tokenizer.decode(out, skip_special_tokens=True)
            cases.append({"name": name, "prompt": prompt, "prompt_tokens": len(tokens),
                          "output_tokens": out, "text": text, "oracle": oracle,
                          "oracle_pass": bool(re.search(oracle, text)),
                          "first_step_top2_logits": top2, "seconds": round(seconds, 2)})
            print(f"[{name}] {'PASS' if cases[-1]['oracle_pass'] else 'FAIL'} {text!r}", flush=True)
        receipt["cases"] = cases
        receipt["all_oracles_pass"] = all(c["oracle_pass"] for c in cases)
        receipt["route_receipt"] = adapter.diagnostics()
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        adapter.close()
    receipt["swap_used_mb_after"] = swap_used_mb()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = a.run_dir / "smoke.json.partial"
    tmp.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    os.replace(tmp, a.run_dir / "smoke.json")
    print(json.dumps({k: receipt[k] for k in ("checkpoint", "mlx2", "all_oracles_pass", "peak_memory_gb",
                                               "swap_used_mb_before", "swap_used_mb_after")}, default=str))
    return 0 if receipt["all_oracles_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
