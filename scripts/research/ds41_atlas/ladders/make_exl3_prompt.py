#!/usr/bin/env python3
"""EXL3 prompt test (Pierre 2026-11-09): translate-7 with every routed expert from the 3 bpw EXL3
bank in prefill and decode, original fallback for any expert the bank lacks (receipts + missing file).
Arms: exact; EXL3 with fp32 reference math; EXL3 with the fp16 Metal kernels. Tier arms carry the
default kernel bundle explicitly (the default guard skips expert policies with exl3).
"""
import json
from pathlib import Path

from make_compare_all import DEFAULTS, B90

BANK = str(Path.home() / "mlx-models/DeepSeek-V4.1-Flash-1017-Source-Expert-EXL3-3bpw")
OUT = str(Path.home() / "Desktop/mlx2/output/ds41-source-20261108/ladder-exl3-translate7")


def exl3(math, run):
    return {"deepseek_source_experts": {"mode": "original_lfu", "budget_bytes": B90,
            "exl3": {"root": BANK, "scope": "all", "pool_bytes": 45_000_000_000, "math": math,
                     "missing_path": f"{OUT}/missing-{run}.json"}}, **DEFAULTS}


arms = [("exact_90", {"deepseek_source_experts": {"mode": "original_lfu", "budget_bytes": B90}}),
        ("exl3_reference", exl3("reference", "reference")),
        ("exl3_kernel", exl3("kernel", "kernel"))]
spec = {"base": {}, "arms": [{"name": n, "policy": p} for n, p in arms], "prompts": ["translate-7"], "max_tokens": 192}
Path(__file__).with_name("exl3_translate7.json").write_text(json.dumps(spec, indent=1))
print(len(arms), "arms")
