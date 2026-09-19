"""Convert the Xing4.0-29B-A4B HF checkpoint into an mlx2 artifact.

    python scripts/convert_xing4_0.py HF_DIR OUT_DIR --bits 16
    python scripts/convert_xing4_0.py HF_DIR OUT_DIR --bits 6

The output holds the sanitized module tree (stacked experts, folded MLA
``kv_b_proj``, MTP head at ``mtp.layers.0``) so loading never repeats the
fold.  ``--bits 16`` keeps the checkpoint's bf16 weights.  Quantized builds use
affine group-64 weights; the embeddings and output heads stay at 8 bits, and
the router, its correction bias and every mHC operand stay unquantized at
their stored precision.  The tokenizer files (and the reference tokenizer code, which the
parity build and the opt-in slow fallback need) are copied unchanged; build
the fast tokenizer afterwards with ``scripts/xing4_0_tokenizer.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx2.runtime.models.xing4_0 import Model, ModelArgs  # noqa: E402

TOKENIZER_FILES = (
    "tokenizer.model",
    "tokenization_xing4_0.py",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "README.md",
    "LICENSE",
)
HEAD_BITS = 8
SHARD_BYTES = 4 << 30


def _is_head(path: str) -> bool:
    return path in ("model.embed_tokens", "lm_head") or path.endswith(
        ("embed_tokens", "shared_head.head")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(hf_dir: Path, out_dir: Path, bits: int, group_size: int) -> None:
    config = json.loads((hf_dir / "config.json").read_text())
    if config.get("model_type") != "xing4_0":
        raise SystemExit("not a Xing4.0 checkpoint")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Model(ModelArgs.from_dict(config))
    index = json.loads((hf_dir / "model.safetensors.index.json").read_text())
    weights = {}
    for name in sorted(set(index["weight_map"].values())):
        weights.update(mx.load(str(hf_dir / name)))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    del weights

    for path, array in tree_flatten(model.parameters()):
        if array.dtype not in (mx.bfloat16, mx.float32):
            raise SystemExit(f"unexpected dtype {array.dtype} at {path}")

    config.pop("auto_map", None)
    if bits != 16:
        keep = model.quant_predicate
        overrides, skipped = {}, []

        def predicate(path, module):
            if not hasattr(module, "to_quantized") or not keep(path, module):
                return False
            if module.weight.shape[-1] % group_size:
                skipped.append(path)
                return False
            if _is_head(path):
                overrides[path] = {"group_size": group_size, "bits": HEAD_BITS}
                return overrides[path]
            return True

        nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
        if skipped:
            print(f"left unquantized (width not divisible by {group_size}): {skipped}")
        config["quantization"] = {
            "group_size": group_size,
            "bits": bits,
            "mode": "affine",
            **overrides,
        }
        config["quantization_config"] = config["quantization"]

    mtp_layer = model.mtp.layers[0] if model.mtp is not None else None
    flat = dict(tree_flatten(model.parameters()))
    shards, current, size = [], {}, 0
    for key in sorted(flat):
        nbytes = flat[key].nbytes
        if current and size + nbytes > SHARD_BYTES:
            shards.append(current)
            current, size = {}, 0
        current[key] = flat[key]
        size += nbytes
    shards.append(current)
    weight_map, total, digests = {}, 0, {}
    for number, shard in enumerate(shards, 1):
        name = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        mx.eval(list(shard.values()))
        mx.save_safetensors(str(out_dir / name), shard, metadata={"format": "mlx"})
        digests[name] = _sha256(out_dir / name)
        for key, value in shard.items():
            weight_map[key] = name
            total += value.nbytes
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    config["mlx2_conversion"] = {
        "source": str(hf_dir),
        "layout": "xing4_0-sanitized-v1",
        "bits": bits,
        "head_bits": HEAD_BITS if bits != 16 else 16,
        "mtp": model.mtp is not None,
        # Whether the MTP layer's embedding / output head were proven equal to
        # the trunk's and dropped; the loader requires the tensors otherwise.
        "mtp_embedding_shared": mtp_layer is not None and "embed_tokens" not in mtp_layer,
        "mtp_head_shared": mtp_layer is not None and "head" not in mtp_layer.shared_head,
        "shard_sha256": digests,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    for name in TOKENIZER_FILES:
        if (hf_dir / name).is_file():
            shutil.copy2(hf_dir / name, out_dir / name)
    print(f"wrote {len(shards)} shards, {total / 1e9:.2f} GB, to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--bits", type=int, choices=(16, 8, 6, 5, 4), required=True)
    parser.add_argument("--group-size", type=int, default=64, choices=(32, 64, 128))
    args = parser.parse_args()
    convert(args.hf_dir.expanduser().resolve(), args.out_dir.expanduser(), args.bits, args.group_size)


if __name__ == "__main__":
    main()
