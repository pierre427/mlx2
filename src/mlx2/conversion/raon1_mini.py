"""Exact KRAFTON/Raon-1-mini (BF16 + MXFP8 experts) -> mlx2 ``mxfp8e`` artifact.

Provenance: provenance/raon1-mini.json.  Not a serving backend.

The source stores every routed expert projection as E4M3 codes
(``F8_E4M3 [out, in]``) with one E8M0 exponent byte per 32 inputs
(``weight_scale_inv`` ``U8 [out, in/32]``): exactly MLX's ``mxfp8`` layout,
whose packed weight is the same bytes read as little-endian uint32.  This
conversion therefore changes no value: per MoE layer it stacks the 128
experts into ``moe.switch_mlp.{gate,up,down}_proj.{weight,scales}``
(``uint32 [128, out, in/4]`` / ``uint8 [128, out, in/32]``) and copies every
other tensor in its source dtype (BF16; A_log, dt_bias, router bias, mHC eta
and steering vectors F32).  ``mx.dequantize`` of a converted bank equals the
reference ``value * 2**(scale - 127)`` bit for bit (tests/test_raon1_mini_cpu.py).
Memory high-water: one layer's expert codes (about 1.1 GB).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx

SOURCE_REPO = "KRAFTON/Raon-1-mini"
SOURCE_REVISION = "6691a076e13a225148b1bce2abdd57a0055fae63"
RECIPE_ID = "raon1-mini-mxfp8e-v1"
COPY_FILES = ("chat_template.jinja", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "LICENSE", "README.md")
_EXPERT = re.compile(r"^model\.layers\.(\d+)\.moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$")


def pack_codes(codes: mx.array) -> mx.array:
    """E4M3 bytes [..., in] -> MLX mxfp8 packed uint32 [..., in/4] (same bytes)."""
    if codes.dtype != mx.uint8 or codes.shape[-1] % 32:
        raise ValueError("MXFP8 codes must be uint8 with a multiple-of-32 input width")
    return mx.view(codes, mx.uint32)


def convert(source: Path, dest: Path, *, log=print) -> dict:
    source, dest = Path(source).expanduser().resolve(), Path(dest).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization_config") or {}
    if config.get("model_type") != "raon1_mini" or quant.get("quant_method") != "mxfp8" \
            or quant.get("weight_block_size") != [1, 32] or quant.get("dequantize"):
        raise ValueError("source is not the KRAFTON Raon-1-mini BF16/MXFP8 release")
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    for name in shards:
        if not (source / name).is_file():
            raise FileNotFoundError(f"incomplete download: {name}")
    experts = config["num_experts"]
    dest.mkdir(parents=True, exist_ok=True)
    lazy = {}
    for name in shards:
        lazy.update(mx.load(str(source / name)))
    by_layer: dict[int, dict] = {}
    plain = {}
    for key, value in lazy.items():
        match = _EXPERT.match(key)
        if match:
            layer, expert, proj, part = match.groups()
            by_layer.setdefault(int(layer), {})[(int(expert), proj, part)] = value
        else:
            plain[key] = value
    index, out_files, started = {}, [], time.time()

    def emit(name, tensors):
        mx.eval(list(tensors.values()))
        mx.save_safetensors(str(dest / name), tensors, metadata={"format": "mlx"})
        out_files.append(name)
        index.update({k: name for k in tensors})

    for layer in sorted(by_layer):
        items = by_layer[layer]
        out = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            codes = mx.stack([items[(e, proj, "weight")] for e in range(experts)])
            scales = mx.stack([items[(e, proj, "weight_scale_inv")] for e in range(experts)])
            if codes.dtype != mx.uint8 or scales.dtype != mx.uint8:
                raise ValueError(f"layer {layer} {proj}: expected E4M3/E8M0 bytes")
            if scales.shape[-1] * 32 != codes.shape[-1]:
                raise ValueError(f"layer {layer} {proj}: scale grid is not group 32")
            prefix = f"model.layers.{layer}.moe.switch_mlp.{proj}"
            out[prefix + ".weight"] = pack_codes(codes)
            out[prefix + ".scales"] = scales
        emit(f"experts-{layer:02d}.safetensors", out)
        log(f"layer {layer}: experts packed ({time.time() - started:.0f}s)")
    plain_names = sorted(plain)
    chunk, size = {}, 0
    for key in plain_names:
        chunk[key] = plain[key]
        size += plain[key].nbytes
        if size >= 4 << 30:
            emit(f"trunk-{len([f for f in out_files if f.startswith('trunk')]):02d}.safetensors", chunk)
            chunk, size = {}, 0
    if chunk:
        emit(f"trunk-{len([f for f in out_files if f.startswith('trunk')]):02d}.safetensors", chunk)
    config = dict(config)
    config.pop("quantization_config", None)
    config.pop("auto_map", None)
    config["mlx2_conversion"] = {"recipe": RECIPE_ID, "source_repository": SOURCE_REPO,
                                 "source_revision": SOURCE_REVISION,
                                 "experts": "mxfp8 g32 (source E4M3/E8M0 bytes, unchanged)",
                                 "other": "source dtype"}
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (dest / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"format": "mlx"}, "weight_map": dict(sorted(index.items()))}, indent=1) + "\n")
    for name in COPY_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, dest / name)
    digest = hashlib.sha256()
    for name in sorted(out_files):
        digest.update(name.encode())
        digest.update(str((dest / name).stat().st_size).encode())
    receipt = {"schema": "mlx2.raon1-mini-conversion.v1", "recipe": RECIPE_ID,
               "source": {"repository": SOURCE_REPO, "revision": SOURCE_REVISION, "path": str(source),
                          "shards": shards},
               "outputs": sorted(out_files), "tensors": len(index), "layout_digest": digest.hexdigest(),
               "seconds": round(time.time() - started, 1)}
    (dest / "mlx2-conversion-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


# ---------------------------------------------------------------------------
# affine 4-bit recipe (Pierre 2026-11-25: "convert the nvfp4 to affine")
# ---------------------------------------------------------------------------

AFFINE_RECIPE_ID = "raon1-mini-affine-a4g64-h8g64-v1"
AFFINE_BITS, AFFINE_GROUP = 4, 64
HEAD_BITS, HEAD_GROUP = 8, 64
# Linears the KRAFTON NVFP4 pack quantizes, quantized here to affine 4-bit g64
# (the same 4.5 bits/weight as NVFP4 g16 with E4M3 scales, on MLX's most-tuned
# affine qmv/qmm/gather kernels).  Everything else keeps its source dtype:
# embedding, kv_b_proj (absorbed MLA), router, mHC selectors, norms, short
# convs, A_log/dt_bias, expert bias and steering vectors.
_AFFINE_DENSE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|g|f_a|f_b|b|rda_w|o|q_a|q_b)_proj"
    r"|self_attn\.kv_a_proj_with_mqa|mlp\.(gate|up|down)_proj|moe\.shared_experts\.(gate|up|down)_proj)\.weight$")


def _dequant_expert(codes: mx.array, scales: mx.array) -> mx.array:
    return mx.dequantize(pack_codes(codes), scales, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16)


def convert_affine(source: Path, dest: Path, *, log=print) -> dict:
    """KRAFTON/Raon-1-mini (BF16 + MXFP8 experts) -> affine a4g64 (+ 8-bit g64 head).

    Quantizes once from the release's own values (MXFP8 experts are decoded
    exactly to BF16 first); the NVFP4 pack is not an input, so no double
    quantization.  Memory high-water: one layer's BF16 expert stack (~2 GB).
    """
    source, dest = Path(source).expanduser().resolve(), Path(dest).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization_config") or {}
    if config.get("model_type") != "raon1_mini" or quant.get("quant_method") != "mxfp8":
        raise ValueError("source is not the KRAFTON Raon-1-mini BF16/MXFP8 release")
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    lazy = {}
    for name in shards:
        lazy.update(mx.load(str(source / name)))
    dest.mkdir(parents=True, exist_ok=True)
    experts = config["num_experts"]
    by_layer: dict[int, dict] = {}
    plain = {}
    for key, value in lazy.items():
        match = _EXPERT.match(key)
        if match:
            layer, expert, proj, part = match.groups()
            by_layer.setdefault(int(layer), {})[(int(expert), proj, part)] = value
        else:
            plain[key] = value
    index, out_files, started = {}, [], time.time()
    quantized = {}

    def emit(name, tensors):
        mx.eval(list(tensors.values()))
        mx.save_safetensors(str(dest / name), tensors, metadata={"format": "mlx"})
        out_files.append(name)
        index.update({k: name for k in tensors})

    def put(out, prefix, weight, bits, group):
        w, s, b = mx.quantize(weight, group_size=group, bits=bits)
        out[prefix + ".weight"], out[prefix + ".scales"], out[prefix + ".biases"] = w, s, b
        quantized[prefix] = (bits, group)

    for layer in sorted(by_layer):
        items, out = by_layer[layer], {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            dense = mx.stack([_dequant_expert(items[(e, proj, "weight")], items[(e, proj, "weight_scale_inv")])
                              for e in range(experts)])
            put(out, f"model.layers.{layer}.moe.switch_mlp.{proj}", dense, AFFINE_BITS, AFFINE_GROUP)
        emit(f"experts-{layer:02d}.safetensors", out)
        log(f"layer {layer}: experts a{AFFINE_BITS}g{AFFINE_GROUP} ({time.time() - started:.0f}s)")
    chunk, size, count = {}, 0, 0
    for key in sorted(plain):
        value = plain[key]
        if _AFFINE_DENSE.match(key):
            put(chunk, key[: -len(".weight")], value, AFFINE_BITS, AFFINE_GROUP)
        elif key == "lm_head.weight":
            put(chunk, "lm_head", value, HEAD_BITS, HEAD_GROUP)
        else:
            chunk[key] = value
        size += value.nbytes
        if size >= 4 << 30:
            emit(f"trunk-{count:02d}.safetensors", chunk)
            chunk, size, count = {}, 0, count + 1
    if chunk:
        emit(f"trunk-{count:02d}.safetensors", chunk)
    config = dict(config)
    config.pop("quantization_config", None)
    config.pop("auto_map", None)
    config["quantization"] = {"group_size": AFFINE_GROUP, "bits": AFFINE_BITS, "mode": "affine",
                              "lm_head": {"group_size": HEAD_GROUP, "bits": HEAD_BITS}}
    config["mlx2_conversion"] = {"recipe": AFFINE_RECIPE_ID, "source_repository": SOURCE_REPO,
                                 "source_revision": SOURCE_REVISION,
                                 "experts": "MXFP8 decoded to BF16, then affine 4-bit g64",
                                 "dense": "affine 4-bit g64 (the NVFP4 pack's quantized set)",
                                 "head": "affine 8-bit g64", "other": "source dtype"}
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (dest / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"format": "mlx"}, "weight_map": dict(sorted(index.items()))}, indent=1) + "\n")
    for name in COPY_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, dest / name)
    receipt = {"schema": "mlx2.raon1-mini-conversion.v1", "recipe": AFFINE_RECIPE_ID,
               "source": {"repository": SOURCE_REPO, "revision": SOURCE_REVISION, "path": str(source),
                          "shards": shards},
               "quantized_modules": len(quantized), "outputs": sorted(out_files), "tensors": len(index),
               "seconds": round(time.time() - started, 1)}
    (dest / "mlx2-conversion-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
