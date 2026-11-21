#!/usr/bin/env python3
"""Convert tencent/Hy4-flash bf16 to an mlx2 MLX affine-quantized artifact.

Original mlx2 script.  CPU only (``mx.cpu``); never touches the GPU.  One
source shard at a time, resumable: each finished source shard leaves one
output shard and a line in ``convert-progress.jsonl``; ``--resume`` skips
them after checking the source fingerprint.

Layout written (``config.json`` gets ``mlx2_layout: hy_v4_flash-sanitized-v1``):

* trunk tensors under ``Model.sanitize`` names (packed experts split,
  ``kv_b_proj`` folded into ``embed_q`` / ``unembed_out``, fp32 modules fp32);
* routed experts at ``--expert-bits`` (default 4), the other trunk linears at
  ``--trunk-bits`` (default 8), group size ``--group-size``; router, iHC,
  indexer pooling, sinks, MLA folds, norms, Engram projections, embeddings
  and the LM head stay unquantized;
* each Engram table, under its checkpoint name, in its own shard per
  ``--engram-formats`` entry (default ``bf16``), for the SSD row reader:
  ``bf16`` copies it byte for byte (``engram-<i>.safetensors``);
  ``fp8_e4m3_row`` (E4M3 codes + one row scale, exact vs the transformers
  round trip) and ``affine4_g32`` (MLX affine 4-bit, group 32, approximate)
  write ``engram-<i>.<format>.safetensors`` with the format declared in the
  shard metadata (``hy_v4_flash_engram.WRITABLE_FORMATS``) and list it under
  ``config["mlx2_engram_formats"]``.  The weight map names the ``bf16``
  shard when it is written, else the first format's;
* ``model.dfly_layers.*`` and the ``vit.*`` vision tower kept in the source
  dtype under checkpoint names (the loader attaches the tower unquantized).

Usage::

    python scripts/convert_hy_v4_flash_mlx.py SRC OUT [--resume]
        [--engram-formats bf16,fp8_e4m3_row,affine4_g32]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LAYOUT = "hy_v4_flash-sanitized-v1"
TABLE = "engram_layer.multi_head_embedding.embedding.weight"
EXPERT_SUFFIXES = ("mlp.experts.gate_proj", "mlp.experts.up_proj", "mlp.experts.down_proj")
TRUNK_SUFFIXES = (
    "self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa",
    "self_attn.linear_gate", "self_attn.o_proj",
    "self_attn.compressed_indexer.query_projection",
    "self_attn.compressed_indexer.head_weight_projection",
    "self_attn.compressed_indexer.token_key_projection",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj",
    "mlp.routed_expert_down_proj", "mlp.routed_expert_up_proj",
)
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "special_tokens_map.json", "LICENSE", "README.md",
              # The vision tower is kept, so its image-processor settings
              # (mean/std, pixel bounds) must travel with it.
              "processor_config.json", "preprocessor_config.json")


def plan_bits(key: str, *, expert_bits: int, trunk_bits: int):
    """Bits for a sanitized ``*.weight`` key, or None to keep it unquantized."""
    if not key.startswith("model.layers.") or not key.endswith(".weight"):
        return None
    module = key[: -len(".weight")]
    if module.endswith(EXPERT_SUFFIXES):
        return expert_bits
    if module.endswith(TRUNK_SUFFIXES):
        return trunk_bits
    return None


def source_fingerprint(src: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        digest.update((src / name).read_bytes())
    return digest.hexdigest()


def copy_table(src_file: Path, name: str, out_file: Path, chunk=64 << 20):
    """Stream one tensor's bytes into a single-tensor safetensors file."""
    from mlx2.runtime.models.hy_v4_flash_engram import safetensors_tensor

    start, dtype, shape = safetensors_tensor(src_file, name)
    width = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}[dtype]
    size = width
    for s in shape:
        size *= s
    header = json.dumps({name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, size]},
                         "__metadata__": {"format": "mlx2-hy4-engram"}}).encode()
    header += b" " * (-len(header) % 8)
    tmp = out_file.with_suffix(".partial")
    with open(src_file, "rb") as source, open(tmp, "wb") as sink:
        sink.write(struct.pack("<Q", len(header)))
        sink.write(header)
        source.seek(start)
        left = size
        while left:
            block = source.read(min(chunk, left))
            if not block:
                raise OSError(f"short read copying {name}")
            sink.write(block)
            left -= len(block)
    os.replace(tmp, out_file)


def _source_rows(mx, raw: bytes, dtype: str, count: int, dim: int):
    """Source table bytes -> an MLX array in the table's own dtype."""
    import numpy as np

    if dtype == "BF16":
        return mx.array(np.frombuffer(raw, "<u2").reshape(count, dim)).view(mx.bfloat16)
    if dtype in ("F32", "F16"):
        return mx.array(np.frombuffer(raw, "<f4" if dtype == "F32" else "<f2").reshape(count, dim))
    raise SystemExit(f"cannot repack a {dtype} Engram table")


def _pack_rows(mx, table_format: str, x):
    """One chunk of rows -> packed ``uint8 [n, row_bytes]`` (CPU)."""
    import numpy as np

    from mlx2.runtime.models.hy_v4_flash import fp8_quantize
    from mlx2.runtime.models.hy_v4_flash_engram import AFFINE4, AFFINE4_GROUP, FP8_ROW

    as_bytes = lambda a: np.array(a.view(mx.uint8)).reshape(a.shape[0], -1)
    if table_format == FP8_ROW:
        codes, scale = fp8_quantize(x)
        parts = (codes, scale)
    elif table_format == AFFINE4:
        parts = mx.quantize(x, group_size=AFFINE4_GROUP, bits=4, mode="affine")
    else:
        raise ValueError(table_format)
    mx.eval(*parts)
    return np.concatenate([as_bytes(p) for p in parts], axis=1)


def write_packed_table(src_file: Path, name: str, out_file: Path, table_format: str,
                       chunk_rows: int = 1 << 18):
    """Stream one table into ``table_format`` rows of a single-tensor shard (CPU)."""
    import mlx.core as mx

    from mlx2.runtime.models.hy_v4_flash_engram import (
        _DTYPES, AFFINE4, AFFINE4_GROUP, FORMAT_KEY, FORMAT_RELATION, packed_row_bytes,
        safetensors_tensor)

    mx.set_default_device(mx.cpu)  # the converter never touches the GPU
    start, dtype, shape = safetensors_tensor(src_file, name)
    if dtype not in ("BF16", "F32", "F16") or len(shape) != 2:
        raise SystemExit(f"cannot repack a {dtype} {list(shape)} Engram table")
    rows, dim = shape
    row_bytes = packed_row_bytes(table_format, dim, dtype)
    meta = {"format": "mlx2-hy4-engram", FORMAT_KEY: table_format, "mlx2_engram_dim": str(dim),
            "mlx2_engram_scale_dtype": dtype, "mlx2_engram_source_dtype": dtype,
            "mlx2_engram_relation": FORMAT_RELATION[table_format],
            "mlx2_engram_qualified": "false"}
    if table_format == AFFINE4:
        meta.update(mlx2_engram_group_size=str(AFFINE4_GROUP), mlx2_engram_bits="4",
                    mlx2_engram_row_layout=f"u32[{dim // 8}] codes low nibble first, "
                                           f"{dtype}[{dim // AFFINE4_GROUP}] scales, "
                                           f"{dtype}[{dim // AFFINE4_GROUP}] biases")
    else:
        meta.update(mlx2_engram_round_trip_dtype=dtype,
                    mlx2_engram_row_layout=f"e4m3[{dim}] codes, {dtype}[1] scale = amax/448")
    size = rows * row_bytes
    header = json.dumps({name: {"dtype": "U8", "shape": [rows, row_bytes], "data_offsets": [0, size]},
                         "__metadata__": meta}).encode()
    header += b" " * (-len(header) % 8)
    width = _DTYPES[dtype][0] * dim
    tmp = out_file.with_suffix(".partial")
    with open(src_file, "rb") as source, open(tmp, "wb") as sink:
        sink.write(struct.pack("<Q", len(header)))
        sink.write(header)
        for first in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - first)
            source.seek(start + first * width)
            raw = source.read(count * width)
            if len(raw) != count * width:
                raise OSError(f"short read repacking {name}")
            rows_mx = _source_rows(mx, raw, dtype, count, dim)
            if not mx.all(mx.isfinite(rows_mx)).item():
                raise SystemExit(f"{name} has nonfinite values in rows {first}..{first + count - 1}")
            packed = _pack_rows(mx, table_format, rows_mx)
            if packed.shape != (count, row_bytes):
                raise AssertionError("packed Engram rows have the wrong width")
            sink.write(packed.tobytes())
    os.replace(tmp, out_file)


def table_shard_name(layer: str, table_format: str) -> str:
    return f"engram-{layer}.safetensors" if table_format == "bf16" else \
        f"engram-{layer}.{table_format}.safetensors"


def parse_formats(text) -> list:
    from mlx2.runtime.models.hy_v4_flash_engram import WRITABLE_FORMATS

    formats = [f.strip() for f in (text.split(",") if isinstance(text, str) else text) if f.strip()]
    if not formats or len(set(formats)) != len(formats) or set(formats) - set(WRITABLE_FORMATS):
        raise SystemExit(f"--engram-formats takes distinct entries of {WRITABLE_FORMATS}")
    return formats


def convert(src: Path, out: Path, *, group_size: int, expert_bits: int, trunk_bits: int,
            resume: bool, engram_formats=("bf16",)) -> None:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx2.runtime.models.hy_v4_flash import Model, ModelArgs
    from mlx2.runtime.models.hy_v4_flash_engram import FP8_ROW

    engram_formats = parse_formats(engram_formats)
    primary = "bf16" if "bf16" in engram_formats else engram_formats[0]
    config = json.loads((src / "config.json").read_text())
    if config.get("model_type") != "hy_v4_flash" or config.get("mlx2_layout"):
        raise SystemExit("source must be an original hy_v4_flash checkpoint")
    if FP8_ROW in engram_formats and not (config.get("text_config") or {}).get("engram_fp8_embedding", True):
        raise SystemExit("fp8_e4m3_row tables carry the E4M3 round trip; this config has engram_fp8_embedding off")
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    fingerprint = source_fingerprint(src)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "convert-progress.jsonl"
    done = {}
    if any(out.iterdir()):
        if not resume:
            raise SystemExit(f"{out} is not empty; pass --resume to continue a run")
        for line in progress.read_text().splitlines() if progress.exists() else []:
            record = json.loads(line)
            if record["source_fingerprint"] != fingerprint:
                raise SystemExit("progress file belongs to a different source")
            if record.get("engram_formats", ["bf16"]) != engram_formats:
                raise SystemExit("progress file was written with other --engram-formats")
            done[record["source"]] = record
    args = ModelArgs.from_hf_config(config)
    model = Model(args)
    tables = {f"model.layers.{i}.{TABLE}" for i in args.engram_layers}
    weight_map: dict = {}
    overrides: dict = {}
    table_formats: dict = {f: {} for f in engram_formats}
    for record in done.values():
        weight_map.update(record["weight_map"])
        overrides.update(record["overrides"])
        # A record from the converter before --engram-formats has no
        # "tables": its tables are the bf16 copies in its weight map.
        recorded = record.get("tables") or {
            "bf16": {k: v for k, v in record["weight_map"].items() if k in tables}}
        for f, shards in recorded.items():
            table_formats[f].update(shards)
    for shard in sorted(set(index.values())):
        if shard in done:
            continue
        names = [k for k, v in index.items() if v == shard]
        shard_map, shard_overrides = {}, {}
        shard_tables: dict = {f: {} for f in engram_formats}
        for name in names:
            if name in tables:
                for table_format in engram_formats:
                    target = table_shard_name(name.split(".")[2], table_format)
                    if table_format == "bf16":
                        copy_table(src / shard, name, out / target)
                    else:
                        write_packed_table(src / shard, name, out / target, table_format)
                    shard_tables[table_format][name] = target
                shard_map[name] = shard_tables[primary][name]
        rest = [n for n in names if n not in tables]
        if rest:
            loaded = mx.load(str(src / shard))
            kept = ("model.dfly_layers.", "vit.")
            dfly = {k: loaded[k] for k in rest if k.startswith(kept)}
            trunk = model.sanitize({k: loaded[k] for k in rest if not k.startswith(kept)})
            output = dict(dfly)
            for key, value in trunk.items():
                bits = plan_bits(key, expert_bits=expert_bits, trunk_bits=trunk_bits)
                if bits is None or value.shape[-1] % group_size:
                    output[key] = value
                    continue
                stem = key[: -len(".weight")]
                w, scales, biases = mx.quantize(value, group_size=group_size, bits=bits, mode="affine")
                output[stem + ".weight"], output[stem + ".scales"], output[stem + ".biases"] = w, scales, biases
                if bits != expert_bits:
                    shard_overrides[stem] = {"group_size": group_size, "bits": bits}
            target = f"model-{Path(shard).stem}.safetensors"
            tmp = out / (target + ".partial.safetensors")
            mx.save_safetensors(str(tmp), output, metadata={"format": "mlx"})
            os.replace(tmp, out / target)
            shard_map.update({k: target for k in output})
            del loaded, output, trunk, dfly
        record = {"source": shard, "source_fingerprint": fingerprint,
                  "weight_map": shard_map, "overrides": shard_overrides,
                  "engram_formats": engram_formats, "tables": shard_tables}
        with open(progress, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        weight_map.update(shard_map)
        overrides.update(shard_overrides)
        for f, shards in shard_tables.items():
            table_formats[f].update(shards)
        print(f"converted {shard}: {len(shard_map)} tensors", flush=True)
    quant = {"group_size": group_size, "bits": expert_bits, "mode": "affine", **overrides}
    config = dict(config, quantization=quant, mlx2_layout=LAYOUT,
                  mlx2_source={"fingerprint": fingerprint, "path": str(src)},
                  mlx2_engram_formats=table_formats)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "mlx"}, "weight_map": weight_map}, indent=1) + "\n")
    for name in COPY_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, out / name)
    print(f"wrote {out} ({len(weight_map)} tensors)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--expert-bits", type=int, default=4)
    parser.add_argument("--trunk-bits", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--engram-formats", default="bf16",
                        help="comma list of bf16, fp8_e4m3_row, affine4_g32 (default bf16); "
                             "each format gets its own shards")
    a = parser.parse_args(argv)
    convert(a.src, a.out, group_size=a.group_size, expert_bits=a.expert_bits,
            trunk_bits=a.trunk_bits, resume=a.resume, engram_formats=a.engram_formats)


if __name__ == "__main__":
    main()
