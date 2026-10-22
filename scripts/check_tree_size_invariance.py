#!/usr/bin/env python3
"""Bitwise tree-size invariance of the Qwen3.8 TensorFold tree verify on a real checkpoint.

The Kapitsa ``tools/moe_tree_check`` test (nace-ai/kapitsa 55b6153), ported:
for each prompt, a teacher-forced greedy stream is verified round by round
through trees of different sizes (1, 4, 8, 16 rows), shapes (chains,
path-first trees, jump rows, random trees) and per-round mixes, so the same
absolute position is verified under different anchors, tile rows and slots.
Every pair of schedules must give bitwise equal fp32 logits per position.
Each requested law (``MLX2_TREE_ATTENTION`` value) runs the same stream.

The model is loaded as the Qwen3.8-27B adapter loads its target for an
external drafter (no embedded MTP head) with the served lane policy
(``resolve(detect(...))``: the default crossover), then the served tree
forward (``lane_multi.multi_tree_forward``) runs directly; no server.  One
model per process; run under ``scripts/run_with_gpu_locks.py``.  Refuses a
nonempty ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _messages(model_path: str, name: str) -> list[dict]:
    from bench_qwen38_b1_routes import CORPUS, SHORT
    from run_spomin_20x20 import FROZEN_CORPUS_SHA256, make_body, prepare_case
    from transformers import AutoTokenizer
    import hashlib

    if name in SHORT:
        return [{"role": "user", "content": SHORT[name]}]
    if not name.startswith("spomin"):
        raise SystemExit(f"unknown prompt {name}")
    capacity = {"spomin2k": 2400, "spomin7k": 8192}[name]
    raw = CORPUS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_CORPUS_SHA256:
        raise RuntimeError("frozen Spomin corpus differs")
    corpus = json.loads(raw)
    case = next(c for c in corpus["cases"] if c["case_id"] == "software_architecture.01")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return make_body(corpus["system"], prepare_case(case, tokenizer, capacity), "full", 64)["messages"]


def _prompt_ids(model_path: str, name: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    text = tokenizer.apply_chat_template(_messages(model_path, name), add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < 8:
        raise RuntimeError(f"prompt {name} rendered to {len(ids)} tokens")
    return [int(t) for t in ids]


def _load(model_path: str):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.adapters.qwen38_27b import _load_target_weights
    from mlx2.adapters.registry import inspect_model
    from mlx2.runtime.lane import apply_policy
    from mlx2.runtime.lane.policy import detect, resolve
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

    resolution = inspect_model(model_path)
    artifact = resolution.artifact
    path = Path(model_path)
    config = dict(artifact["config"])
    config["text_config"] = dict(config.get("text_config", config))
    config["text_config"]["mtp_num_hidden_layers"] = 0
    model = Model(ModelArgs.from_dict(config))
    files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
    weights = _load_target_weights(files, external_draft=True, sanitize=model.sanitize, mtp_sidecar=None)
    quant = config.get("quantization", config.get("quantization_config"))
    if quant:
        def predicate(name, module):
            if name in quant:
                return quant[name]
            return hasattr(module, "to_quantized") and f"{name}.scales" in weights

        nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"], mode=quant.get("mode", "affine"),
                    class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    receipt = apply_policy(model, resolve(detect(model, artifact["config"]), family=None, mode="auto"))
    identity = artifact["identity"]
    return model.language_model, receipt, identity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True, help="<artifact name>@<revision or fingerprint> for the receipt")
    p.add_argument("--prompts", default="code,spomin2k,spomin7k")
    p.add_argument("--n-tok", type=int, default=32)
    p.add_argument("--laws", default="position,masked")
    p.add_argument("--schedules", default="c1,c4,c8,c16,t8,t16,j8,j16,r16:1,mix1")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing nonempty {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    from mlx2.runtime import tree_invariance as T
    from mlx2.runtime.tensorfold_qwen38 import lane_multi

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=ROOT, text=True)
    t0 = time.time()
    model, lane_receipt, identity = _load(args.model)
    summary = {
        "checkpoint": args.checkpoint,
        "artifact_fingerprint": identity.get("fingerprint"),
        "mlx2_commit": head,
        "tracked_source_dirty": bool(dirty.strip()),
        "lane_law": lane_receipt and lane_receipt.get("law_id"),
        "load_seconds": round(time.time() - t0, 1),
        "schedules": args.schedules.split(","),
        "n_tok": args.n_tok,
        "mlx": mx.__version__,
        "device": mx.metal.device_info().get("device_name"),
        "results": {},
    }
    laws = args.laws.split(",")
    for prompt in args.prompts.split(","):
        ids = _prompt_ids(args.model, prompt)
        lane_multi._TREE_ATTENTION = laws[0]
        t1 = time.time()
        stream = T.teacher_stream(model, ids, args.n_tok)
        entry = {"prompt_tokens": len(ids), "stream_seconds": round(time.time() - t1, 1), "laws": {}}
        for law in laws:
            lane_multi._TREE_ATTENTION = law
            t2 = time.time()
            report = T.run(model, ids, args.n_tok, summary["schedules"], stream=stream)
            report["law"] = lane_multi.tree_law()
            report["seconds"] = round(time.time() - t2, 1)
            entry["laws"][law] = report
            print(json.dumps({"prompt": prompt, "law": law, "positions": report["positions"],
                              "split_positions": report["split_positions"],
                              "pairs_differing": report["pairs_differing"], "pairs_total": report["pairs_total"],
                              "invariant": report["invariant"], "seconds": report["seconds"]}), flush=True)
        entry["stream_sha256"] = report["stream_sha256"]
        summary["results"][prompt] = entry
        tmp = args.out / "summary.json.tmp"
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, args.out / "summary.json")
        mx.clear_cache()
    print(json.dumps({"done": True, "out": str(args.out / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
