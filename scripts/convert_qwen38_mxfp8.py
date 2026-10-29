#!/usr/bin/env python3
"""Quantize a BF16 Qwen3.8-27B (HF layout) to MLX MXFP8 linear weights.

Every rank-2 ``.weight`` of a text or MTP linear projection (attention, GDN
in/out projections, MLP, ``mtp.fc`` and ``lm_head``) becomes MLX ``mxfp8``
(E4M3 elements, group 32, E8M0 scales, no biases) under its original key.
Norms, conv1d, A_log/dt_bias, the token embedding and the vision tower keep
their source tensors unchanged, so the mlx2 sanitizer still sees the raw HF
conv layout and folds the norm gammas exactly as for the BF16 source.

One output shard per source shard; an existing complete output shard is
skipped, so an interrupted run resumes. The receipt binds the source index
and per-shard digests. This is an evaluation artifact for the MX lane, not a
quality-selected quantization recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx

MODE, BITS, GROUP = "mxfp8", 8, 32
LINEAR = re.compile(
    r"^(?:model\.language_model\.layers\.\d+\.(?:self_attn\.[qkvo]_proj|linear_attn\.(?:in_proj_\w+|out_proj)"
    r"|mlp\.(?:gate|up|down)_proj)|lm_head|mtp\.fc|mtp\.layers\.\d+\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj))"
    r"\.weight$"
)
COPY = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
        "vocab.json", "merges.txt", "special_tokens_map.json", "preprocessor_config.json",
        "video_preprocessor_config.json", "LICENSE", "README.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src, out = args.source, args.output
    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    out.mkdir(parents=True, exist_ok=True)
    weight_map, quantized, kept, shard_receipts = {}, 0, 0, {}
    started = time.time()
    for shard in shards:
        target = out / shard
        done = out / f".{shard}.done.json"
        weights = mx.load(str(src / shard))
        names = sorted(weights)
        if done.exists() and target.exists():
            record = json.loads(done.read_text())
            for key in record["keys"]:
                weight_map[key] = shard
            quantized += record["quantized"]
            kept += record["kept"]
            shard_receipts[shard] = record["sha256"]
            continue
        result, q, k = {}, 0, 0
        for name in names:
            value = weights[name]
            if LINEAR.match(name) and value.ndim == 2 and value.shape[-1] % GROUP == 0:
                packed, scales = mx.quantize(value, group_size=GROUP, bits=BITS, mode=MODE)
                stem = name[: -len(".weight")]
                result[name], result[f"{stem}.scales"] = packed, scales
                q += 1
            else:
                result[name] = value
                k += 1
        mx.eval(result)
        partial = out / f".{shard}.partial.safetensors"
        mx.save_safetensors(str(partial), result, metadata={"format": "mlx"})
        partial.rename(target)
        digest = sha256(target)
        done.write_text(json.dumps({"keys": sorted(result), "quantized": q, "kept": k, "sha256": digest}))
        for key in result:
            weight_map[key] = shard
        quantized, kept = quantized + q, kept + k
        shard_receipts[shard] = digest
        del weights, result
        mx.clear_cache()
        print(json.dumps({"shard": shard, "quantized": q, "kept": k, "elapsed_s": round(time.time() - started)}),
              flush=True)
    total = sum((out / s).stat().st_size for s in shards)
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=1) + "\n")
    config = json.loads((src / "config.json").read_text())
    quant = {"group_size": GROUP, "bits": BITS, "mode": MODE}
    config["quantization"] = dict(quant)
    config["quantization_config"] = dict(quant)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in COPY:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    receipt = {
        "schema": "mlx2.qwen38-mxfp8-conversion.v1",
        "source": str(src),
        "source_index_sha256": sha256(src / "model.safetensors.index.json"),
        "source_config_sha256": sha256(src / "config.json"),
        "quantization": quant,
        "quantized_tensors": quantized,
        "kept_tensors": kept,
        "output_shards_sha256": shard_receipts,
        "output_total_bytes": total,
        "mlx": mx.__version__,
        "elapsed_s": round(time.time() - started),
    }
    (out / "mlx2-conversion-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("quantized_tensors", "kept_tensors", "output_total_bytes")}))


if __name__ == "__main__":
    main()
