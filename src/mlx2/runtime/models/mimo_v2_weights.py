# SPDX-License-Identifier: MIT
"""MiMo-V2 checkpoint reader: one decoder layer at a time; provenance/mimo-v2-layer-stream.json.

The MiMo-V2.6-Pro checkpoints store three tensor formats, all read here
straight from the safetensors byte ranges (no shard is loaded whole):

* FP8 E4M3 with ``<name>_scale_inv`` F32 block scales (``weight_block_size``,
  128x128): ``W = fp8 * scale_inv`` per block. The fused ``qkv_proj`` is
  stored **tensor-parallel sharded**: ``tp_size`` ranks concatenated, each
  holding its own q heads, then k, then v, and each carrying its own scale
  grid padded up to whole blocks (8 ranks x 27 = 216 scale rows for 27136
  rows, not 212). The rank layout is checked on the real checkpoint by the
  E4M3 block-amax test (every block of the rank-interleaved reading reaches
  448, a global reading does not) and agrees with mlx-vlm 0.7.10's
  ``mimo_v2`` sanitize.
* MXFP4 routed experts: U8 ``weight`` (two E2M1 codes a byte, low nibble =
  even column) and U8 E8M0 ``weight_scale`` (``2^(s-127)`` per 32 columns).
  The bytes map onto MLX's native ``mxfp4`` mode unchanged (uint32 view).
* BF16 / F32 for everything else (``o_proj`` is in ``ignored_layers``).

Dequantized weights come back in HF layout (``q_proj``/``k_proj``/``v_proj``
split, global head order) so the math module never sees storage details.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..weight_stream import SafetensorsIndex, read_safetensors_header

MXFP4_GROUP = 32


# ---------------------------------------------------------------- MLX dequant

def fp8_block_dequant(codes: mx.array, scale_inv: mx.array, block: tuple, dtype) -> mx.array:
    """MLX: ``from_fp8(codes)`` (float32) times the padded block scale grid, cast to ``dtype``."""
    br, bc = block
    r, c = codes.shape
    gr, gc = scale_inv.shape
    if gr * br < r or gc * bc < c or (gr - 1) * br >= r or (gc - 1) * bc >= c:
        raise ValueError(f"scale grid {scale_inv.shape} does not tile {codes.shape} by {block}")
    w = mx.from_fp8(codes, dtype=mx.float32)
    w = mx.pad(w, ((0, gr * br - r), (0, gc * bc - c))).reshape(gr, br, gc, bc)
    w = (w * scale_inv.astype(mx.float32)[:, None, :, None]).reshape(gr * br, gc * bc)
    return w[:r, :c].astype(dtype)


def fused_qkv_shard_layout(rows: int, scale_rows: int, heads: int, kv_heads: int,
                           head_dim: int, v_head_dim: int, tp: int, block_rows: int):
    """``(q_rows, k_rows, v_rows, blocks_per_rank)`` for a TP-sharded fused qkv, or raise."""
    if heads % tp or kv_heads % tp:
        raise ValueError(f"tp_size={tp} does not divide heads={heads}/kv_heads={kv_heads}")
    q, k, v = heads // tp * head_dim, kv_heads // tp * head_dim, kv_heads // tp * v_head_dim
    blocks = -(-(q + k + v) // block_rows)
    if (q + k + v) * tp != rows or blocks * tp != scale_rows:
        raise ValueError(f"fused qkv {rows} rows / {scale_rows} scale rows do not match tp_size={tp} "
                         f"(shard {q}+{k}+{v} rows, {blocks} blocks a rank)")
    return q, k, v, blocks


class MiMoV2Checkpoint:
    """Name -> bytes for one MiMo-V2 checkpoint directory. Reads headers only on construction."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.config = json.loads((self.path / "config.json").read_text())
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.index_metadata = index.get("metadata") or {}
        shards = sorted(set(index["weight_map"].values()))
        if any(Path(s).name != s or s.startswith("._") for s in shards):
            raise ValueError("weight index names an invalid shard")
        entries = []
        for s in shards:
            p = self.path / s
            entries.append((p, read_safetensors_header(p), p.stat().st_size))
        self.tensors = SafetensorsIndex.from_headers(entries)
        q = self.config.get("quantization_config") or {}
        self.fp8_block = tuple(q.get("weight_block_size") or (128, 128))
        if q and (q.get("quant_method") != "fp8" or q.get("fmt", "e4m3") != "e4m3"):
            raise ValueError(f"unsupported quantization_config {q.get('quant_method')}/{q.get('fmt')}")
        if int(q.get("mxfp4_block_size", MXFP4_GROUP)) != MXFP4_GROUP:
            raise ValueError("MXFP4 group size other than 32")
        self.tp_size = int(self.index_metadata.get("tp_size", 1))

    # ---- raw access
    def has(self, name: str) -> bool:
        return name in self.tensors

    def raw(self, name: str) -> tuple:
        """``(dtype, shape, uint8 bytes)`` read by byte range."""
        loc = self.tensors[name]
        data = np.fromfile(loc.path, dtype=np.uint8, count=loc.nbytes, offset=loc.begin)
        if data.size != loc.nbytes:
            raise IOError(f"short read for {name}")
        return loc.dtype, loc.shape, data

    def numpy(self, name: str) -> np.ndarray:
        """Storage values: F32/F16 as is, BF16 widened to float32, U8/F8 as uint8 codes."""
        dtype, shape, data = self.raw(name)
        if dtype == "BF16":
            return (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dtype == "F32":
            return data.view(np.float32).reshape(shape)
        if dtype == "F16":
            return data.view(np.float16).reshape(shape)
        if dtype in ("U8", "F8_E4M3", "F8_E4M3FN"):
            return data.reshape(shape)
        raise ValueError(f"{name}: unsupported dtype {dtype}")

    def rows(self, name: str, ids: np.ndarray) -> np.ndarray:
        """Selected rows of a 2-D BF16/F32 tensor (embedding lookup without reading the table)."""
        loc = self.tensors[name]
        item = {"BF16": 2, "F32": 4}[loc.dtype]
        mm = np.memmap(loc.path, dtype=np.uint8, mode="r", offset=loc.begin, shape=(loc.nbytes,))
        table = mm.reshape(loc.shape[0], loc.shape[1] * item)
        out = np.ascontiguousarray(table[np.asarray(ids).reshape(-1)])
        del mm
        if loc.dtype == "BF16":
            return (out.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return out.view(np.float32)

    def dense(self, name: str, dtype) -> mx.array:
        """A float weight in compute ``dtype``: FP8 block dequant when ``<name>_scale_inv`` exists."""
        dt = self.tensors[name].dtype
        if self.has(name + "_scale_inv"):
            if dt not in ("F8_E4M3", "F8_E4M3FN"):
                raise ValueError(f"{name}: scale_inv beside {dt}")
            return fp8_block_dequant(mx.array(self.numpy(name)), mx.array(self.numpy(name + "_scale_inv")),
                                     self.fp8_block, dtype)
        if dt not in ("BF16", "F32", "F16"):
            raise ValueError(f"{name}: {dt} weight without a scale")
        if dt == "BF16":
            _, shape, data = self.raw(name)
            return mx.array(data.view(np.uint16).reshape(shape)).view(mx.bfloat16).astype(dtype)
        return mx.array(self.numpy(name)).astype(dtype)

    def fused_qkv(self, prefix: str, heads: int, kv_heads: int, head_dim: int, v_head_dim: int,
                  dtype) -> tuple:
        """Dequantize ``<prefix>qkv_proj`` rank by rank and return global-order (q, k, v)."""
        name = prefix + "qkv_proj.weight"
        rows, tp = self.tensors[name].shape[0], self.tp_size
        if self.has(name + "_scale_inv"):
            scale = self.numpy(name + "_scale_inv")
            q, k, v, blocks = fused_qkv_shard_layout(rows, scale.shape[0], heads, kv_heads,
                                                     head_dim, v_head_dim, tp, self.fp8_block[0])
            codes = self.numpy(name)
        else:
            scale, w = None, self.dense(name, dtype)
            q, k, v, _ = fused_qkv_shard_layout(rows, tp, heads, kv_heads, head_dim, v_head_dim, tp,
                                                rows // tp)
        parts = {"q": [], "k": [], "v": []}
        shard = q + k + v
        for r in range(tp):
            if scale is None:
                dense = w[r * shard:(r + 1) * shard]
            else:
                dense = fp8_block_dequant(mx.array(codes[r * shard:(r + 1) * shard]),
                                          mx.array(scale[r * blocks:(r + 1) * blocks]),
                                          self.fp8_block, dtype)
            parts["q"].append(dense[:q])
            parts["k"].append(dense[q:q + k])
            parts["v"].append(dense[q + k:])
        return tuple(mx.concatenate(parts[n], 0) for n in "qkv")

    def mxfp4(self, name: str) -> tuple:
        """``(packed uint32 [..., cols/8], scales uint8 [..., cols/32])`` in MLX mxfp4 layout."""
        dtype, shape, data = self.raw(name + ".weight")
        sdt, sshape, sdata = self.raw(name + ".weight_scale")
        if dtype != "U8" or sdt != "U8" or shape[-1] * 2 != sshape[-1] * MXFP4_GROUP or shape[-1] % 4:
            raise ValueError(f"{name}: {dtype}{list(shape)} / {sdt}{list(sshape)} is not MXFP4 g32")
        return (mx.array(data.view(np.uint32).reshape(*shape[:-1], shape[-1] // 4)),
                mx.array(sdata.reshape(sshape)))

    def mxfp4_stack(self, names: list) -> tuple:
        """Stacked ``(packed [E, rows, cols/8], scales [E, rows, cols/32])`` without a doubled peak."""
        first = self.tensors[names[0] + ".weight"].shape
        sfirst = self.tensors[names[0] + ".weight_scale"].shape
        packed = np.empty((len(names), first[0], first[1] // 4), np.uint32)
        scales = np.empty((len(names), *sfirst), np.uint8)
        for e, n in enumerate(names):
            dtype, shape, data = self.raw(n + ".weight")
            sdt, sshape, sdata = self.raw(n + ".weight_scale")
            if (dtype, sdt) != ("U8", "U8") or tuple(shape) != tuple(first) or tuple(sshape) != tuple(sfirst):
                raise ValueError(f"{n}: expert geometry differs from {names[0]}")
            packed[e] = data.view(np.uint32).reshape(first[0], -1)
            scales[e] = sdata.reshape(sfirst)
        return mx.array(packed), mx.array(scales)
