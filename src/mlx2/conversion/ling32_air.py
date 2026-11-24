"""Layer-streamed block-FP8 -> MLX affine conversion for inclusionAI/Ling-3.2-air.

Provenance: provenance/ling32-air.json.  Not a serving backend.

The source checkpoint stores every linear the vendor quantized as FP8 e4m3
with a 128x128 ``weight_scale_inv`` block grid (DeepSeek-style:
``W = fp8(W) * scale_inv[i // 128, j // 128]``).  This module dequantizes one
decoder layer at a time and re-quantizes it to MLX affine:

* routed experts (``mlp.experts.*``): 4-bit, group 64, pre-stacked into the
  ``mlp.switch_mlp.{gate,up,down}_proj`` layout that ``bailing_hybrid`` loads;
* every other FP8 linear (KDA/MLA projections, shared experts, dense MLPs,
  the MTP layer's ``g_proj``) plus ``lm_head`` and ``word_embeddings``: 8-bit,
  group 64 (the MTP ``eh_proj`` is BF16 in the source and stays BF16);
* everything the vendor kept out of FP8 (router ``mlp.gate``, ``b_proj``, the
  global layers' head-wise ``g_proj``, norms, ``A_log``/``dt_bias``, short-conv
  kernels, ``expert_bias``) keeps its source dtype.  MLA ``kv_b_proj`` is
  dequantized to BF16 and left for ``Model.sanitize`` to split into the
  absorbed ``embed_q``/``unembed_out`` pair, so the absorbed MLA law sees
  unquantized latent projections.

The MTP layer (``model.layers.42``) is converted with the same rules and kept
under its checkpoint name.  The vision tower (``model.visual``) and its
``linear_proj`` merger are copied unchanged into their own shard for the
later VL route.  The memory high-water mark is one layer's BF16 expert stack
(about 2 GiB per projection) plus its quantized output.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

SOURCE_REPO = "inclusionAI/Ling-3.2-air"
SOURCE_REVISION = "1613b1a4dcc6aa0d8f346606628cd9a92061a525"
RECIPE_ID = "ling32-air-affine-e4g64-r8g64-v1"
EXPERT_BITS, EXPERT_GROUP = 4, 64
DEFAULT_BITS, DEFAULT_GROUP = 8, 64
BLOCK = 128
COPY_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenization_bailing.py",
    "README.md",
    "preprocessor_config.json",
)
_DTYPES = {
    "BF16": ("<u2", mx.bfloat16),
    "F16": ("<f2", mx.float16),
    "F32": ("<f4", mx.float32),
    "F8_E4M3": ("u1", None),
}
_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")


class Reader:
    """Read named tensors from the source shards without loading whole files."""

    def __init__(self, path: Path):
        self.path = Path(path)
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.headers: dict[str, tuple[dict, int]] = {}
        self._streams = {}
        for shard in sorted(set(self.weight_map.values())):
            with (self.path / shard).open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                self.headers[shard] = (json.loads(stream.read(size)), 8 + size)

    def entry(self, name):
        shard = self.weight_map[name]
        return self.headers[shard][0][name]

    def raw(self, name) -> mx.array:
        shard = self.weight_map[name]
        header, payload = self.headers[shard]
        entry = header[name]
        kind = _DTYPES.get(entry["dtype"])
        if kind is None:
            raise ValueError(f"unsupported source dtype {entry['dtype']} for {name}")
        start, end = entry["data_offsets"]
        stream = self._streams.get(shard)
        if stream is None:
            stream = self._streams[shard] = (self.path / shard).open("rb")
        stream.seek(payload + start)
        data = stream.read(end - start)
        if len(data) != end - start:
            raise ValueError(f"truncated source tensor: {name}")
        array = mx.array(np.frombuffer(data, kind[0]).reshape(entry["shape"]))
        return array.view(kind[1]) if kind[1] is not None else array

    def is_fp8(self, name) -> bool:
        return self.entry(name)["dtype"] == "F8_E4M3"

    def weight(self, name) -> mx.array:
        """Dequantized tensor (BF16 for FP8 sources, source dtype otherwise)."""
        value = self.raw(name)
        if not self.is_fp8(name):
            return value
        scale_name = name + "_scale_inv"
        if scale_name not in self.weight_map:
            raise ValueError(f"FP8 tensor without block scales: {name}")
        return dequantize_block_fp8(value, self.raw(scale_name))

    def close(self):
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()


def dequantize_block_fp8(codes: mx.array, scale_inv: mx.array, block: int = BLOCK) -> mx.array:
    """``fp8(W) * scale_inv`` on a ``block x block`` grid, ragged edges cropped."""
    if codes.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError("block FP8 dequantization expects 2-D weight and scale")
    rows, cols = codes.shape
    if scale_inv.shape != (-(-rows // block), -(-cols // block)):
        raise ValueError(f"scale grid {scale_inv.shape} does not cover weight {codes.shape}")
    values = mx.from_fp8(codes, dtype=mx.float32)
    scale = mx.repeat(mx.repeat(scale_inv.astype(mx.float32), block, axis=0), block, axis=1)
    return (values * scale[:rows, :cols]).astype(mx.bfloat16)


def _quantize(prefix: str, weight: mx.array, bits: int, group: int, out: dict):
    q, scales, biases = mx.quantize(weight, group_size=group, bits=bits, mode="affine")
    out[prefix + ".weight"] = q
    out[prefix + ".scales"] = scales
    out[prefix + ".biases"] = biases


def plan_names(names):
    """Partition source names into per-layer groups, trunk, and vision."""
    layers: dict[int, list[str]] = {}
    trunk, vision = [], []
    for name in names:
        if name.endswith("_scale_inv"):
            continue
        if name.startswith("model.visual.") or name.startswith("linear_proj."):
            vision.append(name)
        elif name.startswith("model.layers."):
            layers.setdefault(int(name.split(".")[2]), []).append(name)
        else:
            trunk.append(name)
    return layers, sorted(trunk), sorted(vision)


def convert_layer(reader: Reader, index: int, names, num_experts: int):
    out: dict[str, mx.array] = {}
    overrides: dict[str, dict] = {}
    experts: dict[str, list] = {"gate": [None] * num_experts, "up": [None] * num_experts,
                                "down": [None] * num_experts}
    for name in sorted(names):
        match = _EXPERT.match(name)
        if match:
            experts[match.group(3)][int(match.group(2))] = name
            continue
        module = name[: -len(".weight")] if name.endswith(".weight") else None
        if module is not None and reader.is_fp8(name) and not module.endswith("kv_b_proj"):
            _quantize(module, reader.weight(name), DEFAULT_BITS, DEFAULT_GROUP, out)
        elif name.endswith(".mlp.gate.weight"):
            # router_dtype fp32: store the router once in fp32 (see Gate).
            out[name] = reader.weight(name).astype(mx.float32)
        else:
            out[name] = reader.weight(name)
        mx.eval(*out.values())
    if any(item is not None for column in experts.values() for item in column):
        for projection, column in experts.items():
            if any(item is None for item in column):
                raise ValueError(f"layer {index} is missing routed {projection}_proj experts")
            stacked = mx.stack([reader.weight(name) for name in column])
            module = f"model.layers.{index}.mlp.switch_mlp.{projection}_proj"
            _quantize(module, stacked, EXPERT_BITS, EXPERT_GROUP, out)
            mx.eval(out[module + ".weight"], out[module + ".scales"], out[module + ".biases"])
            del stacked
            mx.clear_cache()
            overrides[module] = {"group_size": EXPERT_GROUP, "bits": EXPERT_BITS}
    return out, overrides


def text_config(source_config: dict) -> dict:
    text = dict(source_config["text_config"])
    text.update(
        {
            "model_type": "bailing_hybrid",
            "architectures": ["BailingMoeV3ForCausalLM"],
            "num_shared_experts": text.get("num_shared_experts", 1),
            "tie_word_embeddings": bool(source_config.get("tie_word_embeddings", False)),
            "torch_dtype": "bfloat16",
            # Retained for the later VL route; ignored by the text model args.
            "vision_config": source_config.get("vision_config"),
            "mrope_section": source_config.get("mrope_section"),
            "image_token_id": source_config.get("image_token_id"),
            "video_token_id": source_config.get("video_token_id"),
            "vision_start_token_id": source_config.get("vision_start_token_id"),
            "vision_end_token_id": source_config.get("vision_end_token_id"),
        }
    )
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, dest: Path, *, log=print, layers_only=None) -> dict:
    source, dest = Path(source), Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty {dest}")
    marker = source / ".download-complete"
    if not marker.is_file() or SOURCE_REVISION not in marker.read_text():
        raise ValueError(f"source download is not complete at {SOURCE_REVISION}")
    source_config = json.loads((source / "config.json").read_text())
    if source_config.get("model_type") != "bailing_moe_v3_vl":
        raise ValueError("source is not a bailing_moe_v3_vl checkpoint")
    qc = source_config.get("quantization_config") or {}
    if qc.get("quant_method") != "fp8" or qc.get("weight_block_size") != [BLOCK, BLOCK]:
        raise ValueError("source is not block-128 FP8")
    config = text_config(source_config)
    num_experts = int(config["num_experts"])
    dest.mkdir(parents=True, exist_ok=True)
    reader = Reader(source)
    layers, trunk, vision = plan_names(reader.weight_map)
    expected = config["num_hidden_layers"] + config.get("num_nextn_predict_layers", 0)
    if sorted(layers) != list(range(expected)):
        raise ValueError(f"source layer indices do not match {expected} layers")
    weight_map: dict[str, str] = {}
    quantization = {"group_size": DEFAULT_GROUP, "bits": DEFAULT_BITS, "mode": "affine"}
    started = time.time()
    try:
        selected = sorted(layers) if layers_only is None else list(layers_only)
        for index in selected:
            out, overrides = convert_layer(reader, index, layers[index], num_experts)
            if index >= config["num_hidden_layers"]:
                # NEXTN layers load as the runtime's ``mtp.layers.<k>`` module.
                source_prefix = f"model.layers.{index}."
                target_prefix = f"mtp.layers.{index - config['num_hidden_layers']}."
                out = {target_prefix + k[len(source_prefix):]: v for k, v in out.items()}
                overrides = {target_prefix + k[len(source_prefix):]: v for k, v in overrides.items()}
            shard = f"model-layer-{index:02d}.safetensors"
            mx.save_safetensors(str(dest / shard), out, metadata={"format": "mlx"})
            weight_map.update(dict.fromkeys(out, shard))
            quantization.update(overrides)
            log(f"layer {index:02d}: {len(out)} tensors, {time.time() - started:.0f}s")
            del out
            mx.clear_cache()
        out = {}
        for name in trunk:
            if name in ("lm_head.weight", "model.word_embeddings.weight"):
                _quantize(name[: -len(".weight")], reader.weight(name), DEFAULT_BITS, DEFAULT_GROUP, out)
            else:
                out[name] = reader.weight(name)
        mx.eval(*out.values())
        mx.save_safetensors(str(dest / "model-trunk.safetensors"), out, metadata={"format": "mlx"})
        weight_map.update(dict.fromkeys(out, "model-trunk.safetensors"))
        del out
        vision_out = {name: reader.weight(name) for name in vision}
        if vision_out:
            mx.eval(*vision_out.values())
            # Not in the text weight map: the text loader is strict.
            mx.save_safetensors(str(dest / "vision.safetensors"), vision_out,
                                metadata={"format": "mlx"})
        del vision_out
    finally:
        reader.close()
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    config["mlx2_conversion"] = {
        "recipe": RECIPE_ID,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "experts": {"bits": EXPERT_BITS, "group_size": EXPERT_GROUP, "mode": "affine"},
        "default": {"bits": DEFAULT_BITS, "group_size": DEFAULT_GROUP, "mode": "affine"},
        "full_precision": "vendor modules_to_not_convert, norms, conv, A_log/dt_bias, kv_b_proj (bf16)",
        "vision_shard": "vision.safetensors" if vision else None,
    }
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    total = sum((dest / shard).stat().st_size for shard in set(weight_map.values()))
    (dest / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n"
    )
    for name in COPY_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, dest / name)
    receipt = {
        "recipe": RECIPE_ID,
        "source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION, "path": str(source)},
        "mlx": mx.__version__,
        "seconds": round(time.time() - started, 1),
        "shards": {shard: _sha256(dest / shard) for shard in sorted(set(weight_map.values()))},
        "total_bytes": total,
    }
    (dest / "mlx2-conversion-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


__all__ = ["Reader", "convert", "dequantize_block_fp8", "plan_names", "text_config"]
