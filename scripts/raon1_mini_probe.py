#!/usr/bin/env python3
"""Raon-1-mini GPU probe: greedy smoke, decode speed, chunk invariance, DFlash2 bind.

    PYTHONPATH=src python -I scripts/raon1_mini_probe.py --model ART [--draft DIR] --out DIR

Runs only inside a GPU lease (window script).  Writes probe.json atomically;
refuses a nonempty --out.  Names the checkpoint and the mlx2 revision.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PROMPTS = [
    ("What is 17 * 23? Reply with the number only.", False),
    ("Write one sentence about the city of Busan.", False),
    ("If a shirt costs $25 after a 20% discount, what was the original price?", True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mlx2.adapters import raon1_mini as A
    from mlx2.runtime.models import raon1_mini as R
    from mlx2.runtime.ubc_evict import load_shards_evicting

    tree = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    started = time.time()
    artifact = A.inspect_artifact(args.model)
    model = R.Model(R.ModelArgs.from_dict(artifact["config"]))
    storage = R.install_for_layout(model, artifact["layout"], artifact["config"], artifact["weight_map"].keys())
    root = Path(artifact["identity"]["path"])
    weights = model.sanitize(load_shards_evicting([root / n for n in sorted(set(artifact["weight_map"].values()))]))
    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    model.eval()
    model.configure(A.parse_policy({}))
    mx.eval(model.parameters())
    load_s = time.time() - started
    tok = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    report = {"checkpoint": artifact["checkpoint"], "layout": artifact["layout"], "storage": storage,
              "mlx2_revision": revision, "mlx": mx.__version__, "load_s": round(load_s, 1),
              "peak_after_load_gb": round(mx.get_peak_memory() / 1e9, 2), "generations": []}

    def generate(ids, limit, effort):
        model.set_reasoning_effort(effort)
        cache = model.make_cache()
        t0 = time.time()
        logits = model(mx.array([ids]), cache)
        nxt = mx.argmax(logits[:, -1], -1)
        mx.eval(nxt)
        ttft = time.time() - t0
        out = [int(nxt.item())]
        t1 = time.time()
        while len(out) < limit and out[-1] not in A.EOS_TOKEN_IDS:
            logits = model(nxt[:, None], cache)
            nxt = mx.argmax(logits[:, -1], -1)
            out.append(int(nxt.item()))
        decode = time.time() - t1
        return out, ttft, (len(out) - 1) / max(decode, 1e-9)

    for text, thinking in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=thinking, return_dict=True)["input_ids"]
        out, ttft, rate = generate(list(ids), args.max_tokens if not thinking else 4 * args.max_tokens,
                                   None if thinking else "none")
        report["generations"].append({"prompt": text, "thinking": thinking, "prompt_tokens": len(ids),
                                      "ttft_s": round(ttft, 3), "decode_tok_s": round(rate, 1),
                                      "tokens": len(out), "text": tok.decode(out)})
        print(json.dumps(report["generations"][-1])[:400], flush=True)
    # Chunk invariance: one 2-chunk prefill vs one pass (APCv2 checkpoint law).
    ids = tok.apply_chat_template([{"role": "user", "content": "Summarize the plot of Hamlet in five sentences."}],
                                  add_generation_prompt=True, enable_thinking=False, return_dict=True)["input_ids"]
    model.set_reasoning_effort("none")
    one = model(mx.array([ids]), model.make_cache())[:, -1].astype(mx.float32)
    cache = model.make_cache()
    half = len(ids) // 2
    model(mx.array([ids[:half]]), cache)
    two = model(mx.array([ids[half:]]), cache)[:, -1].astype(mx.float32)
    report["chunk_invariance"] = {"max_abs": float(mx.abs(one - two).max()),
                                  "same_argmax": int(mx.argmax(one)) == int(mx.argmax(two))}
    # Long-ish decode speed at ~4K context.
    filler = tok.encode(" ".join(f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(330)),
                        add_special_tokens=False)
    _, ttft, rate = generate(filler[:4096], 64, "none")
    report["context_4k"] = {"ttft_s": round(ttft, 3), "decode_tok_s": round(rate, 1)}
    if args.draft:
        from mlx2.adapters.raon1_mini_dflash2 import inspect_drafter, load_drafter

        record = inspect_drafter(args.draft, target_path=root, target_config=artifact["config"],
                                 target_fingerprint=artifact["identity"]["fingerprint"])
        draft = load_drafter(record, model)
        _, features = model.forward_with_taps(mx.array([ids]), None, record["args"].target_layer_ids)
        report["dflash2"] = {"checkpoint": record["checkpoint"], "feature_shape": list(features.shape),
                             "fc_shape": list(draft.fc.weight.shape)}
    report["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    tmp = args.out / "probe.json.tmp"
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, args.out / "probe.json")
    print(json.dumps({k: v for k, v in report.items() if k != "generations"}, indent=1))


if __name__ == "__main__":
    main()
