# SPDX-License-Identifier: Apache-2.0
# Row format from vLLM ``fp8_ds_mla`` (vllm-project/vllm PRs 68365, 68387);
# original mlx2 code.  See provenance/hy-v4-flash-fp8-latent-kv.json.
"""Hy4-flash fp8 latent KV rows: approximate state, opt-in, unqualified.

One packed ``uint8`` row per cached token replaces the ``normed latent (r) ‖
roped key pe (p)`` row of the native cache:

* ``r`` bytes: the latent as E4M3 codes, in tiles of ``min(128, r)`` lanes;
* ``4 * r / tile`` bytes: one fp32 power-of-two scale per tile, the smallest
  ``2**k`` with ``amax / 2**k <= 448`` (so no value is clipped and the
  dequantize ``code * scale`` is exact in bf16/fp32; fp16 results are
  clamped to its finite range);
* ``2 * p`` bytes: the roped key pe in bf16, never 8-bit (an fp32 model
  therefore also loses RoPE-tail precision to bf16).

For Hy4-flash (r 512, p 64) that is 512 + 16 + 128 = 656 bytes against 1152
for a bf16 row (43% smaller).  Only the stored key/value rows are
quantized: queries stay in the model dtype (vllm#62678: quantizing the
absorbed query dropped GSM8K from 95% to 50%).  MLX SDPA cannot read the
codes, so attention dequantizes the rows it reads (the gathered recalled
rows, the dense <= candidate_topk history, or the sliding ring) before the
fused kernel; there is no Metal fp8-MLA kernel here.
"""

from __future__ import annotations

import mlx.core as mx

FORMAT = "fp8"
NATIVE = "native"
FORMATS = (NATIVE, FORMAT)
ROW_REVISION = "hy4-flash-fp8-ds-mla-row-v1"
TILE = 128
FP8_MAX = 448.0
# amax floor: an all-zero tile gets scale 2**-108 and codes 0.
_AMAX_FLOOR = 1e-30

# H3 research switch (A/B only, not a policy key): False restores the fp32
# unpack route for bf16 rows (bitwise the same values; scripts/ab_hy4_nonbitwise.py).
BF16_DIRECT = True

_COUNTERS = {"pack_calls": 0, "rows_packed": 0, "unpack_calls": 0, "rows_unpacked": 0,
             "bf16_direct_unpacks": 0}


def latent_fp8_stats(*, reset: bool = False) -> dict:
    """Host-side mechanism counters (no device synchronization)."""
    snapshot = dict(_COUNTERS)
    if reset:
        for name in _COUNTERS:
            _COUNTERS[name] = 0
    return snapshot


def tile_width(rank: int) -> int:
    width = min(TILE, int(rank))
    if width < 1 or rank % width:
        raise ValueError(f"fp8 latent rank {rank} is not a whole number of {width}-lane tiles")
    return width


def row_bytes(rank: int, rope: int) -> int:
    return int(rank) + 4 * (int(rank) // tile_width(rank)) + 2 * int(rope)


def native_row_bytes(rank: int, rope: int, itemsize: int = 2) -> int:
    return (int(rank) + int(rope)) * int(itemsize)


def pack_rows(keys: mx.array, rank: int) -> mx.array:
    """``[..., r+p]`` float rows -> ``[..., row_bytes]`` uint8 packed rows."""
    lead = keys.shape[:-1]
    width = tile_width(rank)
    tiles = rank // width
    latent = keys[..., :rank].astype(mx.float32).reshape(*lead, tiles, width)
    amax = mx.maximum(mx.max(mx.abs(latent), axis=-1, keepdims=True), _AMAX_FLOOR)
    scale = mx.power(mx.array(2.0, mx.float32), mx.ceil(mx.log2(amax / FP8_MAX)))
    # log2 rounding at an exact boundary can leave amax / scale just over 448.
    scale = mx.where(amax / scale > FP8_MAX, scale * 2.0, scale)
    codes = mx.to_fp8(mx.clip(latent / scale, -FP8_MAX, FP8_MAX)).reshape(*lead, rank)
    scales = mx.view(scale.reshape(*lead, tiles), mx.uint8)
    tail = mx.view(keys[..., rank:].astype(mx.bfloat16), mx.uint8)
    _COUNTERS["pack_calls"] += 1
    _COUNTERS["rows_packed"] += int(_prod(lead))
    return mx.concatenate([codes, scales, tail], axis=-1)


def unpack_rows(packed: mx.array, rank: int, dtype) -> mx.array:
    """``[..., row_bytes]`` uint8 -> ``[..., r+p]`` rows in ``dtype``."""
    if packed.dtype != mx.uint8:
        raise TypeError("fp8 latent rows must be packed uint8")
    lead = packed.shape[:-1]
    width = tile_width(rank)
    tiles = rank // width
    end = rank + 4 * tiles
    scales = mx.view(mx.contiguous(packed[..., rank:end]), mx.float32)
    tail = mx.view(mx.contiguous(packed[..., end:]), mx.bfloat16)
    if dtype == mx.bfloat16 and BF16_DIRECT:
        # H3 (2026-11-22): bf16 rows skip the fp32 temporaries (a 128K-row
        # prefill unpack wrote ~302 MB fp32 + 151 MB bf16 per layer).  Bitwise
        # the fp32 route: an E4M3 code has 3 mantissa bits and the scale is a
        # power of two inside fp32's (= bf16's) exponent range, so the code,
        # the scale and their product are exact in bf16, and the tail already
        # is bf16.  tests/test_hy4_nonbitwise_cpu.py checks every code value.
        latent = mx.from_fp8(packed[..., :rank], dtype=mx.bfloat16).reshape(*lead, tiles, width)
        latent = (latent * scales.astype(mx.bfloat16)[..., None]).reshape(*lead, rank)
        _COUNTERS["unpack_calls"] += 1
        _COUNTERS["rows_unpacked"] += int(_prod(lead))
        _COUNTERS["bf16_direct_unpacks"] += 1
        return mx.concatenate([latent, tail], axis=-1)
    latent = mx.from_fp8(packed[..., :rank], dtype=mx.float32).reshape(*lead, tiles, width)
    latent = (latent * scales[..., None]).reshape(*lead, rank)
    rows = mx.concatenate([latent, tail.astype(mx.float32)], axis=-1)
    if dtype == mx.float16:
        # E4M3 / bf16 rounding can step past fp16's largest finite value.
        rows = mx.clip(rows, -65504.0, 65504.0)
    _COUNTERS["unpack_calls"] += 1
    _COUNTERS["rows_unpacked"] += int(_prod(lead))
    return rows.astype(dtype)


def _prod(shape) -> int:
    out = 1
    for n in shape:
        out *= int(n)
    return out
