"""CPU parity: technologies.embedding_tables vs the legacy Engram/PLE copies.

Each ``_legacy_*`` function below is the pre-consolidation body, moved here
verbatim (only renamed, and module-qualified where it referenced module
globals).  Every test asserts exact equality, or the same exception type and
message, between the legacy copy and the migrated code path.
"""
from __future__ import annotations

import json
import math
import os
import struct
from typing import Dict, Optional, Sequence

import numpy as np
import pytest

from mlx2.runtime import weight_stream as _ws
from mlx2.runtime.models import hy_v4_flash_engram as _hy
from mlx2.runtime.technologies import embedding_tables as et

# ---- legacy: deepseek_v41_engram.EngramHasher (verbatim) ----
class _legacy_ds41_EngramHasher:
    def __init__(self, metadata):
        p = "deepseek41.engram."
        self.token_map = np.asarray(metadata[p + "token_map"], dtype=np.int64)
        self.rows = tuple(metadata[p + "rows"])
        self.primes = np.asarray(metadata[p + "primes"], dtype=np.int64).reshape(
            len(self.rows), 24
        )
        self.multipliers = np.asarray(
            metadata[p + "multipliers"], dtype=np.int64
        ).reshape(len(self.rows), 4)
        self.vocab = metadata[p + "compressed_vocab_size"]
        self.pad_id = metadata[p + "pad_id"]
        if (
            self.token_map.ndim != 1
            or not len(self.token_map)
            or self.vocab < 1
            or np.any(self.token_map < 0)
            or np.any(self.token_map >= self.vocab)
            or not 0 <= self.pad_id < self.vocab
            or np.any(self.primes < 2)
            or tuple(self.primes.sum(-1)) != self.rows
            or np.any(self.multipliers <= 0)
            or np.any(self.multipliers % 2 != 1)
            or np.any(self.multipliers > np.iinfo(np.int64).max // self.vocab)
        ):
            raise ValueError("invalid sealed V4.1 Engram hash layout")
        self.offsets = np.cumsum(
            np.concatenate(
                (np.zeros((len(self.rows), 1), np.int64), self.primes[:, :-1]), axis=-1
            ),
            axis=-1,
        )

    def __call__(self, ids, tail, mask=None):
        ids, tail = np.asarray(ids), np.asarray(tail)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= len(self.token_map))
        ):
            raise ValueError("Engram requires in-vocabulary integer token IDs")
        if (
            tail.shape != (3,)
            or tail.dtype.kind not in "iu"
            or np.any(tail < -1)
            or np.any(tail >= self.vocab)
        ):
            raise ValueError("invalid compressed n-gram tail")
        compressed = self.token_map[ids].copy()
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != ids.shape or mask.dtype != np.bool_:
                raise ValueError("Engram token mask must be boolean and match IDs")
            compressed[~mask] = -1
        history = np.concatenate((tail[::-1], compressed))
        positions = np.arange(len(ids)) + 3
        blocked = np.zeros(len(ids), bool)
        products = []
        for shift in range(4):
            current = history[positions - shift]
            blocked |= current == -1
            products.append(
                np.where(blocked, self.pad_id, current)[:, None]
                * self.multipliers[:, shift]
            )
        rolling, output = products[0], []
        for shift in range(1, 4):
            rolling = rolling ^ products[shift]
            output.append(
                rolling[..., None] % self.primes[:, (shift - 1) * 8 : shift * 8]
            )
        hashes = np.concatenate(output, axis=-1) + self.offsets
        return hashes.astype(np.uint32), history[-3:][::-1].astype(np.int32).copy()

# ---- legacy: deepseek_v41_engram.decode_affine4_rows (verbatim) ----
def _legacy_ds41_decode_affine4_rows(codes, scales, biases, compact=False):
    """MLX affine 4-bit/group-32 rows -> BF16-rounded FP32 (or BF16 bits).

    Codes are packed low nibble first in little-endian U32 words. Each value
    is scale*q+bias in F32 (scale*q is exact: 4-bit by BF16), rounded once
    to BF16 ties-to-even. This is the Metal mx.dequantize law that omlx
    serves with (GPU oracle on real rows of both tables, 2026-11-04). MLX's
    CPU kernel instead rounds after the product and after the sum; it
    differs on about 45% of values and is not used.
    """
    codes, scales, biases = (np.asarray(v) for v in (codes, scales, biases))
    if (codes.dtype != np.uint32 or codes.ndim != 2 or codes.shape[1] != 32
            or scales.dtype != np.uint16 or biases.dtype != np.uint16
            or scales.shape != (codes.shape[0], 8) or biases.shape != scales.shape):
        raise ValueError("Engram affine rows require U32[32] codes and BF16[8] scales/biases")
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    q = ((codes[:, :, None] >> shifts) & np.uint32(15)).reshape(codes.shape[0], 256).astype(np.float32)
    as_f32 = lambda v: (v.astype(np.uint32) << 16).view(np.float32)

    def bf16(value):
        bits = np.ascontiguousarray(value, np.float32).view(np.uint32)
        return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)

    with np.errstate(over="ignore", invalid="ignore"):
        values = bf16(q.reshape(-1, 8, 32)*as_f32(scales)[:, :, None]
                      + as_f32(biases)[:, :, None]).reshape(-1, 256)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite decoded Engram row")
    return (values.view(np.uint32) >> 16).astype(np.uint16) if compact else values

# ---- legacy: hy_v4_flash.EngramHasher (verbatim) ----
class _legacy_hy4_EngramHasher:
    """Host-side n-gram hashing of one Engram layer (``hash_ngrams``), exact int64."""

    def __init__(self, lookup_table: np.ndarray, head_vocab_sizes: Sequence[Sequence[int]],
                 *, layer_id: int, seed: int, max_ngram: int, heads_per_ngram: int, pad_id: int):
        table = np.asarray(lookup_table).astype(np.int64).reshape(-1)
        if table.size == 0 or not 0 <= pad_id < table.size:
            raise ValueError("Engram lookup table must cover the pad id")
        self.table = table
        self.pad_id = int(pad_id)
        self.max_ngram = int(max_ngram)
        self.heads = int(heads_per_ngram)
        sizes = [int(s) for order in head_vocab_sizes for s in order]
        if len(sizes) != (self.max_ngram - 1) * self.heads:
            raise ValueError("Engram head vocabulary layout mismatch")
        self.moduli = np.asarray(sizes, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.moduli)[:-1]]).astype(np.int64)
        self.rows = int(self.moduli.sum())
        compressed = int(table.max()) + 1
        half_bound = max(1, np.iinfo(np.int64).max // compressed // 2)
        rng = np.random.default_rng(int(seed) + 10007 * int(layer_id))
        drawn = rng.integers(0, half_bound, size=self.max_ngram, dtype=np.int64)
        self.multipliers = drawn * 2 + 1

    def hash_ids(self, ngram_tokens: np.ndarray) -> np.ndarray:
        """``[B, L, max_ngram]`` token ids -> per-head ids ``[B, L, heads*orders]``."""
        tokens = np.asarray(ngram_tokens, dtype=np.int64)
        size = self.table.size
        compressed = np.where(tokens < size, self.table[np.minimum(tokens, size - 1)],
                              self.table[self.pad_id])
        mixed = compressed * self.multipliers
        out = []
        for n in range(2, self.max_ngram + 1):
            mix = mixed[..., 0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, mixed[..., k])
            start = (n - 2) * self.heads
            out.append(mix[..., None] % self.moduli[start:start + self.heads])
        return np.concatenate(out, axis=-1)

    def table_rows(self, ngram_tokens: np.ndarray) -> np.ndarray:
        """Flat table row of each head's id (head offset added)."""
        return self.hash_ids(ngram_tokens) + self.offsets


# ---- legacy: hy_v4_flash.Model._ngram_tokens (verbatim; self.args -> args) ----
def _legacy_hy4_ngram_tokens(args, tokens: np.ndarray, start: int, tail: np.ndarray,
                  document_positions: Optional[np.ndarray] = None) -> np.ndarray:
    """Each position's token with the ``max_ngram - 1`` before it; pad before 0.

    ``document_positions`` (the temporal M-RoPE plane, given only when the
    lane has vision spans) replaces the slot as the reference's
    ``document_offsets`` (``build_ngram_tokens``).
    """
    a = args
    n = a.engram_max_ngram_size
    B, L = tokens.shape
    history = np.concatenate([tail[:, -(n - 1):] if tail.size else np.zeros((B, 0), np.int64),
                              tokens], axis=1)
    lead = history.shape[1] - L
    out = np.full((B, L, n), a.engram_pad_id, dtype=np.int64)
    for k in range(n):
        pos = np.arange(L) + start if document_positions is None else document_positions
        src = np.arange(L) + lead - k
        ok = (pos >= k) & (src >= 0)
        out[:, ok, k] = history[:, src[ok]]
    return out


# ---- legacy: hy_v4_flash_engram packed affine4 (verbatim) ----
def _legacy_hy4_as_f32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Little-endian scale/bias bytes ``[..., k*width]`` -> fp32 ``[..., k]``."""
    raw = np.ascontiguousarray(raw)
    if dtype == "BF16":
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    return raw.view("<f4" if dtype == "F32" else "<f2").astype(np.float32)



def _legacy_hy4_decode_affine4_row(raw: np.ndarray, dim: int, scale_dtype: str) -> np.ndarray:
    """Packed ``affine4_g32`` rows ``uint8 [n, row_bytes]`` -> fp32 ``[n, dim]``.

    Same law as ``deepseek_v41_engram.decode_affine4_rows`` (whose layout is
    fixed at 256 values in three planes): ``scale * q`` exact, plus ``bias``
    rounded once in fp32.
    """
    raw = np.asarray(raw, np.uint8)
    if raw.ndim != 2 or raw.shape[1] != _hy.packed_row_bytes(_hy.AFFINE4, dim, scale_dtype):
        raise ValueError("affine4_g32 rows have the wrong width")
    n, groups, width = raw.shape[0], dim // _hy.AFFINE4_GROUP, _hy._SCALE_BYTES[scale_dtype]
    words = np.ascontiguousarray(raw[:, : dim // 2]).view("<u4")
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    q = ((words[:, :, None] >> shifts) & np.uint32(15)).reshape(n, groups, _hy.AFFINE4_GROUP).astype(np.float32)
    at = dim // 2
    scales = _legacy_hy4_as_f32(raw[:, at: at + groups * width], scale_dtype)
    biases = _legacy_hy4_as_f32(raw[:, at + groups * width:], scale_dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        values = (q * scales[:, :, None] + biases[:, :, None]).reshape(n, dim)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite decoded Engram row")
    return values.astype(np.float32, copy=False)

# ---- legacy: hy_v4_flash_engram.safetensors_header (verbatim) ----
def _legacy_hy4_safetensors_header(path) -> dict:
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        if not 2 <= size <= 1 << 28:
            raise ValueError("implausible safetensors header")
        header = json.loads(handle.read(size))
    header["__header_bytes__"] = 8 + size
    return header

# ---- legacy: qwen4_ple_nvme helpers (verbatim) ----
def _legacy_qwen4_bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """View uint16 bfloat16 bits as float32 values."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _legacy_qwen4_f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bits with round-to-nearest-even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(32767) + (bits >> 16 & 1)
    return (rounded >> 16).astype(np.uint16)


def _legacy_qwen4_dequant_rows_numpy(row_bytes: np.ndarray, dims: int) -> np.ndarray:
    """Dequantize packed q4/g32 rows to bfloat16 bits.

    Args:
        row_bytes: uint8 array of shape ``[n, row_bytes]`` in sidecar layout.
        dims: row width in elements (must be a multiple of 32).

    Returns:
        uint16 array of shape ``[n, dims]`` holding bfloat16 bits.
    """
    n = row_bytes.shape[0]
    weight_bytes = dims // 2
    group_bytes = dims // 32 * 2
    words = np.ascontiguousarray(row_bytes[:, :weight_bytes]).view(np.uint32)
    scales = np.ascontiguousarray(
        row_bytes[:, weight_bytes : weight_bytes + group_bytes]
    ).view(np.uint16)
    biases = np.ascontiguousarray(
        row_bytes[:, weight_bytes + group_bytes : weight_bytes + 2 * group_bytes]
    ).view(np.uint16)
    shifts = np.uint32(4) * np.arange(8, dtype=np.uint32)
    q = (words[..., None] >> shifts & np.uint32(15)).astype(np.float32)
    q = q.reshape(n, dims)
    s = np.repeat(_legacy_qwen4_bf16_bits_to_f32(scales), 32, axis=1)
    b = np.repeat(_legacy_qwen4_bf16_bits_to_f32(biases), 32, axis=1)
    return _legacy_qwen4_f32_to_bf16_bits(q * s + b)


def _legacy_qwen4_read_safetensors_header(path):
    """Return (tensor header dict, data section offset) for a safetensors file."""
    import struct

    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return (header, 8 + header_len)

# ---- legacy: weight_stream._decode_header (verbatim) ----
def _legacy_ws_decode_header(first8: bytes, read_payload, size: int, path) -> Dict[str, dict]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _ws.StreamingUnavailable(f"duplicate safetensors key {key!r}: {path}")
            result[key] = value
        return result

    if len(first8) != 8:
        raise _ws.StreamingUnavailable(f"truncated safetensors header: {path}")
    length = int.from_bytes(first8, "little")
    if not 0 < length <= min(_ws._SAFETENSORS_HEADER_LIMIT, size - 8):
        raise _ws.StreamingUnavailable(f"invalid safetensors header length: {path}")
    raw = read_payload(length)
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise _ws.StreamingUnavailable(f"invalid safetensors JSON header: {path}") from error
    if not isinstance(header, dict) or "__data_offset__" in header:
        raise _ws.StreamingUnavailable(f"invalid safetensors header object: {path}")
    header.pop("__metadata__", None)
    return {"__data_offset__": 8 + length, **header}

# ---- legacy: hy_v4_flash_buckets.read_header (verbatim) ----
def _legacy_buckets_read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


# ---- legacy: qwen4_exp multiplier law (verbatim) ----
_L_MASK64 = (1 << 64) - 1
_L_SPLITMIX_GAMMA = 11400714819323198485
_L_SPLITMIX_M1 = 13787848793156543929
_L_SPLITMIX_M2 = 10723151780598845931
_L_PRIME_1 = 10007


def _legacy_qwen4_splitmix64(value: int) -> int:
    value = value + _L_SPLITMIX_GAMMA & _L_MASK64
    value = (value ^ value >> 30) * _L_SPLITMIX_M1 & _L_MASK64
    value = (value ^ value >> 27) * _L_SPLITMIX_M2 & _L_MASK64
    return (value ^ value >> 31) & _L_MASK64


def _legacy_qwen4_build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _L_PRIME_1 * ple_layer_index
    return [
        2
        * (
            _legacy_qwen4_splitmix64(base_seed + _L_SPLITMIX_GAMMA * (index + 1) & _L_MASK64)
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    ]


# ---- legacy: qwen4_exp.NGramEmbedding._ngram_ids_numpy (verbatim, dedented) ----
from mlx2.runtime.models.qwen4_exp import (ArraysCache, _PLE_VECTOR_SHIFT, _row_tail,
                                           _valid_span_end, record_verify_sync)
import mlx.core as mx


def _legacy_qwen4_ngram_ids_numpy(
    self,
    input_ids: mx.array,
    cache: Optional[ArraysCache] = None,
    mask: Optional[mx.array] = None,
    previous: Optional[np.ndarray] = None,
    record_sync: bool = True,
    host_constants: tuple | None = None,
) -> np.ndarray:
    if host_constants is not None:
        if cache is not None or mask is not None or not isinstance(input_ids, np.ndarray):
            raise ValueError("host PLE hashing requires host tokens and no mutable cache")
    else:
        if record_sync:
            record_verify_sync("qwen4.ple.ids_eval")
        mx.eval(input_ids, mask)
    tokens = np.asarray(input_ids, dtype=np.int64)
    (batch, seq_len) = tokens.shape
    if previous is not None:
        previous = np.asarray(previous, dtype=np.int64)
        if previous.shape[-1] < self.context_len:
            pad = np.full(
                (batch, self.context_len - previous.shape[-1]),
                self.eos_token_id,
                dtype=np.int64,
            )
            previous = np.concatenate([pad, previous], axis=-1)
    elif cache is not None and cache[3] is not None:
        previous = np.asarray(cache[3], dtype=np.int64)
    else:
        previous = np.full(
            (batch, self.context_len), self.eos_token_id, dtype=np.int64
        )
    if mask is not None:
        record_verify_sync("qwen4.ple.mask_asarray")
        mask_array = np.asarray(mask)
        tokens = np.where(mask_array, tokens, self.eos_token_id)
    else:
        mask_array = None
    history = np.concatenate([previous, tokens], axis=-1)
    if cache is not None:
        tail = (
            history[:, -self.context_len :]
            if mask is None
            else _row_tail(history, _valid_span_end(mask_array), self.context_len)
        )
        cache[3] = mx.array(tail, dtype=mx.int64)
    if _PLE_VECTOR_SHIFT:
        shifted = self._shift_history_vectorized(history)
    else:
        shifted = []
        for shift in range(self.ngram_size):
            out = np.full_like(history, self.eos_token_id)
            if shift == 0:
                out = history.copy()
            else:
                for b in range(batch):
                    segment_start = 0
                    for pos in range(history.shape[1]):
                        if pos - segment_start >= shift:
                            out[b, pos] = history[b, pos - shift]
                        if history[b, pos] == self.eos_token_id:
                            segment_start = pos + 1
            shifted.append(out)
    (multipliers, sizes, offsets) = (
        self._hash_constants_numpy() if host_constants is None else host_constants
    )
    blocks = []
    for ngram in range(2, self.ngram_size + 1):
        with np.errstate(over="ignore"):
            mixed = shifted[0] * multipliers[0]
            for position in range(1, ngram):
                mixed = np.bitwise_xor(
                    mixed, shifted[position] * multipliers[position]
                )
        start = (ngram - 2) * self.heads_per_ngram
        end = start + self.heads_per_ngram
        blocks.append(
            np.remainder(mixed[..., None], sizes[start:end]) + offsets[start:end]
        )
    return np.concatenate(blocks, axis=-1)[:, -seq_len:]


# ---- legacy: deepseek_v41_source.hasher_metadata (verbatim) ----
def _legacy_ds41_hasher_metadata(backend, args):
    """Official tokenizer normalization, prime layout and per-layer RNG, CPU only."""
    from tokenizers import Regex, normalizers
    sentinel = "\ue000"
    normalize = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    mapping, lookup = {}, []
    for token in range(backend.get_vocab_size()):
        text = backend.decode([token], skip_special_tokens=False)
        key = backend.id_to_token(token) if "\ufffd" in text else (normalize.normalize_str(text) or text)
        lookup.append(mapping.setdefault(key, len(mapping)))
    if len(mapping) != args.engram_compressed_vocab_size:
        raise ValueError("official Engram compressed vocabulary mismatch")
    seen, primes = set(), []
    for _ in args.engram_layer_ids:
        layer = []
        for _ in range(3):
            current = 15_999_999
            for _ in range(8):
                current += 1
                while current in seen or any(current % n == 0 for n in range(2, math.isqrt(current)+1)):
                    current += 1
                seen.add(current)
                layer.append(current)
        primes.append(layer)
    bound = max(1, (np.iinfo(np.int64).max // len(mapping)) // 2)
    multipliers = [np.random.default_rng(10007*layer).integers(0, bound, size=4, dtype=np.int64)*2+1 for layer in args.engram_layer_ids]
    prefix = "deepseek41.engram."
    return {prefix+"token_map": lookup, prefix+"rows": list(args.engram_num_embeddings), prefix+"primes": primes,
            prefix+"multipliers": np.array(multipliers).tolist(), prefix+"compressed_vocab_size": len(mapping), prefix+"pad_id": lookup[2]}



# ---- legacy: deepseek_v41_source multiplier derivation (verbatim lines) ----
def _legacy_ds41_source_multipliers(layer_ids, vocab):
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    multipliers = [np.random.default_rng(10007*layer).integers(0, bound, size=4, dtype=np.int64)*2+1 for layer in layer_ids]
    return np.array(multipliers).tolist()


# ---- legacy: qwen4_exp NGramEmbedding._ngram_ids_numpy hash block (verbatim loop) ----
def _legacy_qwen4_hash_block(shifted, multipliers, sizes, offsets, ngram_size, heads_per_ngram, seq_len):
    blocks = []
    for ngram in range(2, ngram_size + 1):
        with np.errstate(over="ignore"):
            mixed = shifted[0] * multipliers[0]
            for position in range(1, ngram):
                mixed = np.bitwise_xor(
                    mixed, shifted[position] * multipliers[position]
                )
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        blocks.append(
            np.remainder(mixed[..., None], sizes[start:end]) + offsets[start:end]
        )
    return np.concatenate(blocks, axis=-1)[:, -seq_len:]


# =========================================================== helpers

def _outcome(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as error:  # noqa: BLE001 - parity of the refusal itself
        return ("err", type(error), str(error))


def _same(a, b):
    assert a[0] == b[0], (a, b)
    if a[0] == "err":
        assert a[1:] == b[1:], (a, b)
        return
    x, y = a[1], b[1]
    if isinstance(x, tuple):
        assert len(x) == len(y)
        for u, v in zip(x, y):
            _same(("ok", u), ("ok", v))
    elif isinstance(x, np.ndarray):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.array_equal(x.view(np.uint8), y.view(np.uint8))
    else:
        assert x == y


def _primes_from(start, count):
    out, current = [], start
    while len(out) < count:
        current += 1
        if all(current % d for d in range(2, math.isqrt(current) + 1)):
            out.append(current)
    return out


def _ds41_metadata(rng, tables=2, vocab=97, raw_vocab=300):
    p = "deepseek41.engram."
    primes = np.array(_primes_from(1000 + int(rng.integers(0, 5000)), tables * 24)).reshape(tables, 24)
    bound = np.iinfo(np.int64).max // vocab
    multipliers = rng.integers(0, bound // 2, size=(tables, 4), dtype=np.int64) * 2 + 1
    return {p + "token_map": rng.integers(0, vocab, size=raw_vocab).tolist(),
            p + "rows": primes.sum(-1).tolist(), p + "primes": primes.tolist(),
            p + "multipliers": multipliers.tolist(), p + "compressed_vocab_size": vocab,
            p + "pad_id": int(rng.integers(0, vocab))}


# ================================================== DS4.1 hashing

@pytest.mark.parametrize("seed", range(6))
def test_ds41_hasher_matches_legacy(seed):
    from mlx2.runtime.models.deepseek_v41_engram import EngramHasher

    rng = np.random.default_rng(seed)
    metadata = _ds41_metadata(rng, tables=1 + seed % 3)
    new, old = EngramHasher(metadata), _legacy_ds41_EngramHasher(metadata)
    assert np.array_equal(new.offsets, old.offsets) and new.offsets.dtype == old.offsets.dtype
    for n in (0, 1, 2, 3, 4, 7, 33):
        for dtype in (np.int32, np.int64, np.uint16):
            ids = rng.integers(0, 300, size=n).astype(dtype)
            tail = rng.integers(-1, 97, size=3).astype(np.int64 if dtype is np.uint16 else dtype)
            if n % 2:
                tail[rng.integers(0, 3)] = -1  # a masked slot inside the tail
            mask = None if n % 3 == 0 else rng.random(n) < 0.7
            _same(_outcome(new, ids, tail, mask), _outcome(old, ids, tail, mask))
    # refusals keep their messages
    for ids, tail in ((np.array([300]), np.zeros(3, int)), (np.array([1]), np.array([0, 0])),
                      (np.array([1.0]), np.zeros(3, int)), (np.array([1]), np.array([0, 0, 97]))):
        _same(_outcome(new, ids, tail), _outcome(old, ids, tail))


def test_ds41_spec_from_metadata():
    metadata = _ds41_metadata(np.random.default_rng(0))
    spec = et.EngramSpec.from_metadata(metadata)
    assert (spec.max_ngram, spec.heads_per_ngram, spec.multiplier_law, spec.history_law) == \
        (4, 8, "metadata", "blocked_pad")
    assert spec.pad_id == metadata["deepseek41.engram.pad_id"]


@pytest.mark.parametrize("vocab", [1, 2, 97, 129280])
def test_ds41_source_multipliers_match_legacy(vocab):
    layers = [0, 2, 5, 41]
    law = et.EngramSpec(max_ngram=4, heads_per_ngram=8, multiplier_law="numpy_default_rng",
                        history_law="blocked_pad", seed=0)
    new = np.array([et.seeded_multipliers(law, layer=l, compressed_vocab=vocab) for l in layers]).tolist()
    assert new == _legacy_ds41_source_multipliers(layers, vocab)


# ==================================================== Hy4 hashing

class _HyArgs:
    def __init__(self, n, pad):
        self.engram_max_ngram_size = n
        self.engram_pad_id = pad
        self.engram_n_head_per_ngram = 4
        self.engram_seed = 7


@pytest.mark.parametrize("seed", range(6))
def test_hy4_hasher_matches_legacy(seed):
    from mlx2.runtime.models.hy_v4_flash import EngramHasher

    rng = np.random.default_rng(seed)
    max_ngram, heads = 2 + seed % 3, 1 + seed % 4
    table = rng.integers(0, 50 + 1000 * seed, size=211)
    sizes = [[int(p) for p in _primes_from(5000 + 97 * o, heads)] for o in range(max_ngram - 1)]
    kw = dict(layer_id=int(rng.integers(1, 40)), seed=int(rng.integers(0, 1 << 20)),
              max_ngram=max_ngram, heads_per_ngram=heads, pad_id=int(rng.integers(0, 211)))
    new, old = EngramHasher(table, sizes, **kw), _legacy_hy4_EngramHasher(table, sizes, **kw)
    for name in ("multipliers", "moduli", "offsets", "table"):
        a, b = getattr(new, name), getattr(old, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    assert new.rows == old.rows
    for shape in ((1, 1), (2, 5), (3, 0), (1, 17)):
        tokens = rng.integers(0, 260, size=shape + (max_ngram,))  # some beyond the table
        for dtype in (np.int32, np.int64):
            t = tokens.astype(dtype)
            _same(_outcome(new.hash_ids, t), _outcome(old.hash_ids, t))
            _same(_outcome(new.table_rows, t), _outcome(old.table_rows, t))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_hy4_ngram_tokens_match_legacy(n):
    from mlx2.runtime.models.hy_v4_flash import Model

    rng = np.random.default_rng(n)
    args = _HyArgs(n, pad=3)
    holder = type("M", (), {"args": args, "_engram_window_spec": Model._engram_window_spec})()
    for B, L in ((1, 1), (2, 6), (3, 1), (1, 9)):
        for start in (0, 1, n - 2, 5, 100):
            for tail_len in (0, 1, n - 1, n + 3):
                tokens = rng.integers(0, 500, size=(B, L))
                tail = rng.integers(0, 500, size=(B, tail_len))
                for doc in (None, rng.integers(0, 4, size=L)):
                    new = Model._ngram_tokens(holder, tokens, start, tail, doc)
                    old = _legacy_hy4_ngram_tokens(args, tokens, start, tail, doc)
                    assert new.dtype == old.dtype and np.array_equal(new, old)


def test_hy4_spec_from_config():
    spec = et.EngramSpec.from_config({"engram_max_ngram_size": 3, "engram_n_head_per_ngram": 16,
                                      "engram_pad_id": 9, "engram_seed": 0})
    assert (spec.max_ngram, spec.heads, spec.multiplier_law, spec.history_law, spec.pad_id) == \
        (3, 32, "numpy_default_rng", "pad_before_start", 9)


# =================================================== Qwen4 PLE hashing

@pytest.mark.parametrize("vocab", [0, 1, 3, 151936, 1 << 40])
def test_qwen4_multipliers_match_legacy(vocab):
    from mlx2.runtime.models.qwen4_exp import _build_layer_multipliers

    for ngram in (2, 3, 5):
        for layer in (0, 1, 7, 63):
            for seed in (0, 42, (1 << 64) - 5):
                assert _build_layer_multipliers(vocab, ngram, layer, seed) == \
                    _legacy_qwen4_build_layer_multipliers(vocab, ngram, layer, seed)


@pytest.mark.parametrize("ngram", [2, 3, 4])
def test_qwen4_hash_block_matches_legacy(ngram):
    from mlx2.runtime.models.qwen4_exp import _build_layer_multipliers

    rng = np.random.default_rng(ngram)
    heads = 3
    multipliers = np.asarray(_build_layer_multipliers(151936, ngram, 2, 0), dtype=np.int64)
    sizes = np.asarray(_primes_from(30000, (ngram - 1) * heads), dtype=np.int64)
    offsets = np.cumsum([0] + sizes[:-1].tolist(), dtype=np.int64)
    for B, T in ((1, 1), (2, 7), (3, 33)):
        shifted = [rng.integers(0, 151936, size=(B, T)).astype(np.int64) for _ in range(ngram)]
        for seq_len in (1, T):
            new = et.ngram_hash(np.stack(shifted, axis=-1), multipliers, sizes, heads, offsets)[:, -seq_len:]
            old = _legacy_qwen4_hash_block(shifted, multipliers, sizes, offsets, ngram, heads, seq_len)
            assert new.dtype == old.dtype and np.array_equal(new, old)


def test_qwen4_spec_from_config():
    spec = et.EngramSpec.from_config({"ngram_size": 3, "heads_per_ngram": 8, "seed": 5,
                                      "eos_token_id": [151645, 2]})
    assert (spec.max_ngram, spec.multiplier_law, spec.history_law, spec.pad_id, spec.seed) == \
        (3, "splitmix64", "eos_segment", 151645, 5)


# ===================================================== affine rows

def _random_bf16(rng, shape, special=True):
    bits = rng.integers(0x3000, 0x4000, size=shape).astype(np.uint16)  # moderate magnitudes
    bits |= (rng.integers(0, 2, size=shape).astype(np.uint16) << 15)
    if special:
        bits[rng.random(shape) < 0.002] = 0x7F80  # inf
        bits[rng.random(shape) < 0.002] = 0x7FC1  # nan
        bits[rng.random(shape) < 0.01] = 0
    return bits


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("special", [False, True])
def test_ds41_affine_matches_legacy(compact, special):
    from mlx2.runtime.models.deepseek_v41_engram import decode_affine4_rows

    rng = np.random.default_rng(int(compact) + 2 * int(special))
    for n in (0, 1, 5, 64):
        codes = rng.integers(0, 1 << 32, size=(n, 32), dtype=np.uint64).astype(np.uint32)
        scales, biases = _random_bf16(rng, (n, 8), special), _random_bf16(rng, (n, 8), special)
        _same(_outcome(decode_affine4_rows, codes, scales, biases, compact),
              _outcome(_legacy_ds41_decode_affine4_rows, codes, scales, biases, compact))
    bad = (np.zeros((1, 31), np.uint32), np.zeros((1, 8), np.uint16), np.zeros((1, 8), np.uint16))
    _same(_outcome(decode_affine4_rows, *bad), _outcome(_legacy_ds41_decode_affine4_rows, *bad))


@pytest.mark.parametrize("scale_dtype", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("special", [False, True])
def test_hy4_affine_matches_legacy(scale_dtype, special):
    rng = np.random.default_rng(len(scale_dtype) + int(special))
    for dim in (32, 96, 256):
        width = _hy.packed_row_bytes(_hy.AFFINE4, dim, scale_dtype)
        for n in (0, 1, 9):
            raw = rng.integers(0, 256, size=(n, width), dtype=np.int64).astype(np.uint8)
            if scale_dtype == "BF16" and not special:
                groups = dim // 32
                raw[:, dim // 2:] = _random_bf16(rng, (n, 2 * groups), False).view(np.uint8).reshape(n, 4 * groups)
            _same(_outcome(_hy.decode_affine4_row, raw, dim, scale_dtype),
                  _outcome(_legacy_hy4_decode_affine4_row, raw, dim, scale_dtype))
    raw = np.zeros((1, 7), np.uint8)
    _same(_outcome(_hy.decode_affine4_row, raw, 96, scale_dtype),
          _outcome(_legacy_hy4_decode_affine4_row, raw, 96, scale_dtype))


@pytest.mark.parametrize("dims", [32, 256, 1536])
def test_qwen4_ple_dequant_matches_legacy(dims):
    from mlx2.runtime.models import qwen4_ple_nvme as q

    rng = np.random.default_rng(dims)
    groups = dims // 32
    for n in (0, 1, 17):
        words = rng.integers(0, 256, size=(n, dims // 2), dtype=np.int64).astype(np.uint8)
        sb = _random_bf16(rng, (n, 2 * groups)).view(np.uint8).reshape(n, 4 * groups)
        raw = np.concatenate([words, sb], axis=1)
        with np.errstate(all="ignore"):
            old = _legacy_qwen4_dequant_rows_numpy(raw, dims)
        new = q.dequant_rows_numpy(raw, dims)
        assert new.dtype == old.dtype and np.array_equal(new, old)
    values = rng.standard_normal(1000).astype(np.float32)
    values[:4] = [np.inf, -np.inf, np.nan, 3.0e38]
    assert np.array_equal(q.f32_to_bf16_bits(values), _legacy_qwen4_f32_to_bf16_bits(values))
    bits = rng.integers(0, 1 << 16, size=1000, dtype=np.int64).astype(np.uint16)
    assert np.array_equal(q.bf16_bits_to_f32(bits).view(np.uint32),
                          _legacy_qwen4_bf16_bits_to_f32(bits).view(np.uint32))


# ====================================================== row reads

def test_pread_exact_short_read(tmp_path):
    path = tmp_path / "rows.bin"
    path.write_bytes(bytes(range(100)))
    fd = os.open(path, os.O_RDONLY)
    try:
        assert et.pread_exact(fd, 10, 5) == bytes(range(5, 15))
        with pytest.raises(OSError, match="short Engram row read"):
            et.pread_exact(fd, 10, 95)
    finally:
        os.close(fd)


def test_ds41_split_plane_chunk_reads(tmp_path):
    """Packed-264 and affine three-plane chunk readers read the same bytes as before."""
    from mlx2.runtime.models import deepseek_v41_engram as ds

    rng = np.random.default_rng(3)
    rows = 6
    codes = rng.integers(0, 1 << 32, size=(rows, 32), dtype=np.uint64).astype(np.uint32)
    scales, biases = _random_bf16(rng, (rows, 8), False), _random_bf16(rng, (rows, 8), False)
    blob = codes.tobytes() + scales.tobytes() + biases.tobytes()
    path = tmp_path / "affine.bin"
    path.write_bytes(blob)
    fd = os.open(path, os.O_RDONLY)
    try:
        bases = (0, rows * 128, rows * 128 + rows * 16)
        pick = np.array([4, 0, 5])
        got = ds._read_affine_chunk(fd, bases, pick)
        want = _legacy_ds41_decode_affine4_rows(codes[pick], scales[pick], biases[pick])
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
        with pytest.raises(OSError, match="short Engram row read"):
            ds._read_affine_chunk(fd, (0, rows * 128, len(blob)), np.array([0]))
    finally:
        os.close(fd)


# ===================================================== safetensors

def _st_cases():
    good = json.dumps({"__metadata__": {"format": "pt"},
                       "t": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}}).encode()
    dup = b'{"t": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}, "t": {}}'
    return {
        "good": struct.pack("<Q", len(good)) + good + bytes(12),
        "no_meta": struct.pack("<Q", 2) + b"{}" + bytes(4),
        "short_prefix": b"\x01\x00\x00",
        "empty": b"",
        "zero_length": struct.pack("<Q", 0),
        "one_length": struct.pack("<Q", 1) + b"{",
        "huge_length": struct.pack("<Q", 1 << 40) + b"{}",
        "past_limit": struct.pack("<Q", (1 << 28) + 1) + b"{}",
        "truncated_payload": struct.pack("<Q", 50) + b'{"a": 1}',
        "bad_json": struct.pack("<Q", 5) + b"{abc}",
        "bad_utf8": struct.pack("<Q", 4) + b'"\xff"' + b" ",
        "list_json": struct.pack("<Q", 2) + b"[]",
        "duplicate": struct.pack("<Q", len(dup)) + dup,
        "reserved": struct.pack("<Q", 24) + b'{"__data_offset__": 1}  ',
        "padded": struct.pack("<Q", len(good) + 3) + good + b"   ",
    }


@pytest.mark.parametrize("case", sorted(_st_cases()))
def test_safetensors_header_parsers_match_legacy(tmp_path, case):
    from mlx2.runtime.models import hy_v4_flash_buckets as buckets
    from mlx2.runtime.models import qwen4_ple_nvme as q

    path = tmp_path / f"{case}.safetensors"
    path.write_bytes(_st_cases()[case])
    pairs = [
        (_hy.safetensors_header, _legacy_hy4_safetensors_header),
        (q._read_safetensors_header, _legacy_qwen4_read_safetensors_header),
        (buckets.read_header, _legacy_buckets_read_header),
        (_ws.read_safetensors_header, lambda p: _legacy_ws_read(p)),
    ]
    for new, old in pairs:
        _same(_outcome(new, path), _outcome(old, path))
    # the descriptor route of weight_stream too
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        _same(_outcome(_ws._read_header_fd, fd, size, path),
              _outcome(_legacy_ws_decode_header, os.pread(fd, 8, 0),
                       lambda n: _ws._pread_all(fd, 8, n, path), size, path))
    finally:
        os.close(fd)


def _legacy_ws_read(path):
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        return _legacy_ws_decode_header(handle.read(8), handle.read, size, path)


# ===================================================== registry

def test_hash_registry_has_only_runnable_paths():
    hy = et.EngramSpec(max_ngram=3, heads_per_ngram=2, multiplier_law="numpy_default_rng",
                       history_law="pad_before_start")
    ds = et.EngramSpec(max_ngram=4, heads_per_ngram=8, multiplier_law="metadata",
                       history_law="blocked_pad")
    fn, receipt = et.HASH_REGISTRY.select(hy)
    assert receipt.selected == "reference" and receipt.refusals == {}
    fn, receipt = et.HASH_REGISTRY.select(ds)
    assert receipt.selected == "reference"
    assert et.HASH_REGISTRY.technology == "embedding_tables" and et.HASH_REGISTRY.op == "ngram_hash"
    with pytest.raises(RuntimeError, match="unknown fast path"):
        et.HASH_REGISTRY.select(ds, policy="device_limb16")
    out = fn(ds, np.ones((2, 1, 4), np.int64), np.full((1, 4), 3, np.int64),
             np.full((1, 24), 5, np.int64))
    assert np.array_equal(out, et.ngram_hash(np.ones((2, 1, 4), np.int64), np.full((1, 4), 3, np.int64),
                                             np.full((1, 24), 5, np.int64), 8))
    assert out.shape == (2, 1, 24) and set(out.reshape(-1).tolist()) == {0, 3}


# ============================================ real call sites (review)

class _Q4Stub:
    """The attributes ``NGramEmbedding._ngram_ids_numpy`` reads, no weights."""

    def __init__(self, ngram, heads, eos):
        from mlx2.runtime.models import qwen4_exp as q4

        self.ngram_size, self.context_len, self.heads_per_ngram = ngram, ngram - 1, heads
        self.eos_token_id = eos
        self.multipliers = np.asarray(q4._build_layer_multipliers(151936, ngram, 3, 11), np.int64)
        self.sizes = np.asarray(_primes_from(40000, (ngram - 1) * heads), np.int64)
        self.offsets = np.cumsum([0] + self.sizes[:-1].tolist(), dtype=np.int64)
        self._shift_history_vectorized = q4.NGramEmbedding._shift_history_vectorized.__get__(self)

    def _hash_constants_numpy(self):
        return self.multipliers, self.sizes, self.offsets


@pytest.mark.parametrize("vector_shift", [False, True])
@pytest.mark.parametrize("ngram", [2, 3, 4])
def test_qwen4_ngram_ids_numpy_call_site_matches_legacy(monkeypatch, vector_shift, ngram):
    from mlx2.runtime.models import qwen4_exp as q4

    monkeypatch.setattr(q4, "_PLE_VECTOR_SHIFT", vector_shift)
    monkeypatch.setitem(globals(), "_PLE_VECTOR_SHIFT", vector_shift)
    rng = np.random.default_rng(10 * ngram + vector_shift)
    stub = _Q4Stub(ngram, 3, eos=7)
    constants = stub._hash_constants_numpy()
    for B, T in ((1, 1), (2, 6), (3, 17)):
        tokens = rng.integers(0, 40, size=(B, T)).astype(np.int64)  # the EOS id 7 appears often
        for previous in (None, rng.integers(0, 40, size=(B, ngram - 1)), rng.integers(0, 40, size=(B, 0))):
            kw = dict(previous=previous, record_sync=False)
            new = q4.NGramEmbedding._ngram_ids_numpy(stub, tokens, host_constants=constants, **kw)
            old = _legacy_qwen4_ngram_ids_numpy(stub, tokens, host_constants=constants, **kw)
            assert new.dtype == old.dtype and np.array_equal(new, old)
        # the device-array route with a mask (live constants, no host constants)
        ids = mx.array(tokens)
        mask = mx.array(rng.random((B, T)) < 0.8)
        new = q4.NGramEmbedding._ngram_ids_numpy(stub, ids, None, mask, record_sync=False)
        old = _legacy_qwen4_ngram_ids_numpy(stub, ids, None, mask, record_sync=False)
        assert new.dtype == old.dtype and np.array_equal(new, old)


class _FakeTokenizerBackend:
    def __init__(self, texts):
        self.texts = texts

    def get_vocab_size(self):
        return len(self.texts)

    def decode(self, ids, skip_special_tokens=False):
        return self.texts[ids[0]]

    def id_to_token(self, token):
        return f"<tok{token}>"


def test_ds41_hasher_metadata_call_site_matches_legacy():
    from types import SimpleNamespace

    from mlx2.adapters.deepseek_v41_source import hasher_metadata

    texts = ["<pad>", "A", "a", " a", "B", "b�", "Café", "cafe", "  ", "\t", "x y", "X  Y", "z"]
    backend = _FakeTokenizerBackend(texts)
    matched = False
    for vocab in range(1, len(texts) + 1):
        args = SimpleNamespace(engram_compressed_vocab_size=vocab, engram_layer_ids=[1, 4],
                               engram_num_embeddings=[100, 200])
        old = _outcome(_legacy_ds41_hasher_metadata, backend, args)
        new = _outcome(hasher_metadata, backend, args)
        _same(new, old)  # the mismatch refusals too
        matched |= old[0] == "ok"
    assert matched


def test_hy4_null_pad_id_refuses_and_spec_is_built_once():
    from mlx2.runtime.models.hy_v4_flash import Model

    holder = type("M", (), {"args": _HyArgs(3, pad=None), "_engram_window_spec": Model._engram_window_spec})()
    with pytest.raises(TypeError, match="pad_id"):
        Model._ngram_tokens(holder, np.zeros((1, 2), np.int64), 0, np.zeros((1, 0), np.int64))
    holder = type("M", (), {"args": _HyArgs(3, pad=1), "_engram_window_spec": Model._engram_window_spec})()
    Model._ngram_tokens(holder, np.zeros((1, 2), np.int64), 0, np.zeros((1, 0), np.int64))
    first = holder._engram_spec_cache
    Model._ngram_tokens(holder, np.zeros((1, 2), np.int64), 5, np.zeros((1, 0), np.int64))
    assert holder._engram_spec_cache is first


def test_strict_header_without_a_limit():
    good = struct.pack("<Q", 2) + b"{}"
    header, offset = et.parse_safetensors_header(good[:8], lambda n: good[8:8 + n],
                                                 strict_error=RuntimeError, path="p",
                                                 file_size=len(good), header_limit=None)
    assert header == {} and offset == 10
    with pytest.raises(RuntimeError, match="invalid safetensors header length"):
        et.parse_safetensors_header(good[:8], lambda n: b"", strict_error=RuntimeError, path="p",
                                    file_size=9, header_limit=None)


def test_device_limb16_refusal_needs_inputs_and_mirrors_the_device_hasher():
    hy = et.EngramSpec(max_ngram=3, heads_per_ngram=2, multiplier_law="numpy_default_rng",
                       history_law="pad_before_start")
    ok = dict(tables=np.array([[0, 5, 9]]), multipliers=np.array([[3, 5, 7]]),
              moduli=np.array([[7, 11, 13, 17]]), offsets=np.array([[0, 7, 18, 31]]))
    assert et.device_limb16_refusal(hy, **ok) is None
    with pytest.raises(TypeError):
        et.device_limb16_refusal(hy)  # a caller must supply what the device hasher needs
    cases = {
        "compressed ids must fit 32 bits": dict(tables=np.array([[0, 1 << 32]])),
        "multiplier bound": dict(multipliers=np.array([[3, 4, (1 << 62)]])),
        "head moduli": dict(moduli=np.array([[7, 11, 13, (1 << 24) + 1]])),
        "int32": dict(offsets=np.array([[0, 7, 18, (1 << 31)]])),
    }
    for text, change in cases.items():
        reason = et.device_limb16_refusal(hy, **{**ok, **change})
        assert reason.startswith("geometry: ") and text in reason
    ds = et.EngramSpec(max_ngram=4, heads_per_ngram=8, multiplier_law="metadata", history_law="blocked_pad")
    assert et.device_limb16_refusal(ds, **ok).startswith("policy_off: ")


def test_qwen_ple_truncated_bias_is_not_broadcast_to_all_groups():
    from mlx2.runtime.models.qwen4_ple_nvme import dequant_rows_numpy

    # Two g32 groups need two BF16 biases. One remaining bias broadcasts
    # against both groups in the shared decoder unless the row width is checked.
    truncated = np.zeros((1, 32 + 2 * 2 + 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="packed affine4 rows"):
        dequant_rows_numpy(truncated, 64)
