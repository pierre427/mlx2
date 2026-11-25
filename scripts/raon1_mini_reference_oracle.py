#!/usr/bin/env python3
"""Teacher-forced logits oracle for Raon-1-mini: vendor PyTorch reference vs mlx2.

    # 1. reference (CPU torch, the vendor's modeling_raon1_mini.py read from --ref-code)
    python -I scripts/raon1_mini_reference_oracle.py torch --source ~/mlx-models/Raon-1-mini \
        --ref-code DIR --out ref.npz
    # 2. mlx2 (any artifact; --device cpu for mxfp8e, gpu for the NVFP4 pack under a lease)
    PYTHONPATH=src python -I scripts/raon1_mini_reference_oracle.py mlx2 --model ART --out mlx2.npz
    # 3. compare
    python -I scripts/raon1_mini_reference_oracle.py compare ref.npz mlx2.npz

The prompt is fixed (chat template, thinking off, effort "none"); logits are
kept in fp32 for every position.  --ref-code holds a copy of the vendor's
configuration/modeling files (reviewed; executed only as the oracle).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MESSAGES = [
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "A train leaves at 09:40 and arrives at 13:05. How long is the trip? "
                                "Then name three rivers in Korea and explain what a hash table is in two sentences."},
]
CONTINUATION = (" The trip takes 3 hours and 25 minutes. Three rivers in Korea are the Han, the Nakdong "
                "and the Geum. A hash table stores key-value pairs in an array of buckets.")


def tokens(path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    ids = tok.apply_chat_template(MESSAGES, add_generation_prompt=True, enable_thinking=False,
                                  return_dict=True)["input_ids"]
    return list(ids) + tok.encode(CONTINUATION, add_special_tokens=False)


def run_torch(args):
    import torch

    sys.path.insert(0, str(Path(args.ref_code).resolve()))
    from raon_ref.configuration_raon1_mini import Raon1MiniConfig
    from raon_ref.modeling_raon1_mini import Raon1MiniForCausalLM

    torch.set_num_threads(args.threads)
    config = Raon1MiniConfig.from_pretrained(str(args.source))
    model = Raon1MiniForCausalLM.from_pretrained(str(args.source), config=config, dtype=torch.bfloat16)
    model.eval()
    model.set_mxfp8_moe_backend("dequantize")
    model.set_reasoning_effort("none")
    ids = tokens(args.source)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float().numpy()
    np.savez(args.out, ids=np.array(ids), logits=logits)
    print(json.dumps({"tokens": len(ids), "out": str(args.out)}))


def run_mlx2(args):
    import mlx.core as mx

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    from mlx2.adapters import raon1_mini as A
    from mlx2.runtime.models import raon1_mini as R
    from mlx2.runtime.ubc_evict import load_shards_evicting

    artifact = A.inspect_artifact(args.model)
    model = R.Model(R.ModelArgs.from_dict(artifact["config"]))
    R.install_for_layout(model, artifact["layout"], artifact["config"], artifact["weight_map"].keys())
    root = Path(artifact["identity"]["path"])
    weights = model.sanitize(load_shards_evicting([root / n for n in sorted(set(artifact["weight_map"].values()))]))
    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    model.eval()
    model.configure(A.parse_policy({"mla": args.mla}))
    model.set_reasoning_effort("none")
    ids = tokens(root)
    logits = model(mx.array([ids]))[0].astype(mx.float32)
    np.savez(args.out, ids=np.array(ids), logits=np.array(logits), checkpoint=artifact["checkpoint"])
    print(json.dumps({"tokens": len(ids), "checkpoint": artifact["checkpoint"], "out": str(args.out)}))


def compare(args):
    a, b = np.load(args.a), np.load(args.b)
    if not np.array_equal(a["ids"], b["ids"]):
        raise SystemExit("token ids differ")
    la, lb = a["logits"].astype(np.float64), b["logits"].astype(np.float64)

    def logsoftmax(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    pa, pb = logsoftmax(la), logsoftmax(lb)
    kl = (np.exp(pa) * (pa - pb)).sum(-1)
    top_a, top_b = la.argmax(-1), lb.argmax(-1)
    margin = np.sort(la, -1)[:, -1] - np.sort(la, -1)[:, -2]
    flips = np.where(top_a != top_b)[0]
    print(json.dumps({
        "positions": int(la.shape[0]), "top1_agree": float((top_a == top_b).mean()),
        "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
        "max_abs_logit_diff": float(np.abs(la - lb).max()),
        "flips": [{"pos": int(i), "ref_margin": float(margin[i])} for i in flips],
        "checkpoint": str(b["checkpoint"]) if "checkpoint" in b else None}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("torch")
    t.add_argument("--source", type=Path, required=True)
    t.add_argument("--ref-code", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--threads", type=int, default=12)
    m = sub.add_parser("mlx2")
    m.add_argument("--model", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    m.add_argument("--mla", default="auto")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = parser.parse_args()
    {"torch": run_torch, "mlx2": run_mlx2, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
