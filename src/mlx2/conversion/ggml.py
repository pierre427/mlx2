# SPDX-License-Identifier: MIT
# Portions derived from gguf-py 0.19.0 (gguf/quants.py), part of llama.cpp:
# Copyright (c) 2023 Georgi Gerganov; MIT License. The IQ2_XXS grid and sign
# tables are the ggml reference tables (ggml-common.h, The ggml authors, MIT).
"""CPU dequantization of the GGML Q8_0 and IQ2_XXS block laws.

Used only at conversion time; no accelerator is imported.  Arithmetic order
matches ggml's dequantize_row_q8_0 / dequantize_row_iq2_xxs in float32, so
the decoded values are bit-identical to the reference.  See
provenance/nemotron35-bolt.json.
"""
from __future__ import annotations

import numpy as np

Q8_0_BLOCK, Q8_0_BYTES = 32, 34
IQ2_XXS_BLOCK, IQ2_XXS_BYTES = 256, 66

_IQ2XXS_GRID_HEX = (
    b"00000200050008000a00110014002000220028002a0041004400500058006100"
    b"6400800082008a00a20001010401100115014001840198010002020222028202"
    b"010404041004210424044004420448046004810484049004a404000502050805"
    b"200546056905800591050906100640068406a406000805080808140828084108"
    b"440850085208880804094009020a140a01100410101021104010601084109010"
    b"951000110811201150115a118011241245120014081420142514491480141815"
    b"6215001616160118041810184018811800190519a019511a002002200a204420"
    b"6120802082202921482100220222012404241024402456240025412564259026"
    b"082820289428442a014004401040184021402440404048405640604081408440"
    b"9040004120416141804185410142104248425642684200440844204480449944"
    b"124524450046014804481048404845480049584961498249454a904a00500850"
    b"1150195020508050885004514251a4519152905492540a550156545600581158"
    b"195864584059085a046010604060686000615561186260620064056410651265"
    b"84654268008002800a8041808280048118814081118201840484108415844084"
    b"608400854685948509864086608602880489118a0490109024904090a1901691"
    b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
)
_KSIGNS = bytes.fromhex(
    "008182038405068788090a8b0c8d8e0f"
    "901112931495961718999a1b9c1d1e9f"
    "a02122a324a5a62728a9aa2bac2d2eaf"
    "30b1b233b43536b7b8393abb3cbdbe3f"
    "c04142c344c5c64748c9ca4bcc4d4ecf"
    "50d1d253d45556d7d8595adb5cddde5f"
    "60e1e263e46566e7e8696aeb6cedee6f"
    "f07172f374f5f67778f9fa7bfc7d7eff"
)


def _decode_grid():
    hexed = np.frombuffer(_IQ2XXS_GRID_HEX, dtype=np.uint8).reshape(-1, 2)
    nibbles = np.where(hexed > 0x40, hexed + 9, hexed) & 0x0F
    packed = (nibbles[:, 0] << 4) | nibbles[:, 1]
    codes = (packed.reshape(-1, 1) >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
    levels = np.array((0x08, 0x19, 0x2B), dtype=np.float32)
    return levels[codes.reshape(256, 8)]


IQ2XXS_GRID = _decode_grid()  # (256, 8) float32 magnitudes 8, 25, 43
KSIGNS = np.frombuffer(_KSIGNS, dtype=np.uint8)  # 7 sign bits + parity bit
_SIGN_BITS = (KSIGNS[:, None] >> np.arange(8, dtype=np.uint8)) & 1
SIGN_TABLE = np.where(_SIGN_BITS == 0, np.float32(1), np.float32(-1))  # (128, 8)


def _blocks(raw, block_bytes):
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    if raw.ndim < 2 or raw.shape[-1] != block_bytes:
        raise ValueError("GGML payload must end in whole blocks")
    return raw


def dequantize_q8_0(raw):
    """(..., n_blocks, 34) uint8 -> (..., n_blocks * 32) float32."""
    raw = _blocks(raw, Q8_0_BYTES)
    d = raw[..., :2].copy().view("<f2").astype(np.float32)
    q = raw[..., 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(*raw.shape[:-2], -1)


def dequantize_iq2_xxs(raw):
    """(..., n_blocks, 66) uint8 -> (..., n_blocks * 256) float32."""
    raw = _blocks(raw, IQ2_XXS_BYTES)
    lead = raw.shape[:-1]
    d = raw[..., :2].copy().view("<f2").astype(np.float32)  # (..., 1)
    words = raw[..., 2:].copy().view("<u4").reshape(*lead, 8, 2)
    grid_index = words[..., 0].copy().view(np.uint8).reshape(*lead, 8, 4)
    aux = words[..., 1]
    db = d * (np.float32(0.5) + (aux >> 28).astype(np.float32)) * np.float32(0.25)
    sign_index = (aux[..., None] >> np.array([0, 7, 14, 21], dtype=np.uint32)) & 0x7F
    # ggml: db * grid[j] * sign; multiply db first to keep the reference order.
    values = (db[..., None, None] * IQ2XXS_GRID[grid_index]) * SIGN_TABLE[sign_index]
    return values.reshape(*lead[:-1], -1)
