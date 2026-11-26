# SPDX-License-Identifier: Apache-2.0
# Portions (bf16 helpers, PLE q4/g32 dequant law, PLE safetensors header
# reader) adapted from mlx-lm-unified under the MIT License; see
# provenance/LICENSE.unified-MIT and provenance/flashnext.json.
"""Hashed and streamed embedding tables: one reference for Engram/PLE readers.

Three families carry per-layer tables addressed by hashed n-grams or token
ids and read row by row from storage:

* DeepSeek V4.1 Engram (``models/deepseek_v41_engram.py``): primes and odd
  multipliers sealed in checkpoint metadata, a fixed 3-token compressed
  tail, a backward walk that turns everything before a masked token into the
  pad id;
* Hy4-flash Engram (``models/hy_v4_flash.py`` hashing,
  ``models/hy_v4_flash_engram.py`` rows): multipliers drawn from
  ``numpy.random.default_rng(seed + 10007 * layer_id)``, ``max_ngram`` window
  padded before position 0;
* Qwen4-exp / Flash-Next PLE n-grams (``models/qwen4_exp.py`` hashing,
  ``models/qwen4_ple_nvme.py`` rows): SplitMix64 multipliers, primes after a
  base, history reset at each EOS.

The hash law is the same in all three: map each window token to a compressed
id ``c_k``, multiply by an odd per-position multiplier ``m_k`` (int64,
wrapping), XOR the products of the first ``n`` positions for each order
``n = 2..max_ngram``, reduce modulo each head's modulus and add that head's
row offset.  :func:`ngram_hash` is that law, once.  What differs between the
families is where the constants come from and how the window is built; both
are :class:`EngramSpec` fields named for what they do.

Rows are packed affine 4-bit / group 32 in two layouts (three planes, or
one packed row) and decoded by :func:`decode_affine4_g32`, parameterised by
:class:`Affine4RowSpec` (rounding and nonfinite policy differ by caller and
are spec fields).  Safetensors headers are parsed by
:func:`parse_safetensors_header`, with each caller's historical refusal
behaviour selected by its arguments.

Everything here is host numpy; nothing touches the GPU.  The bodies are the
repository's own existing copies, unified.  Licences of what they carry:
the Hy4 hashing, window and packed-affine code (Apache-2.0, ported from
transformers PR 50903 / vLLM PR 68296, provenance/hy-v4-flash.json and
provenance/hy-v4-flash-engram-4bit.json) and the Qwen4-exp SplitMix64
multiplier law (Apache-2.0, provenance/flashnext.json) make this module
Apache-2.0; the bf16 helpers, PLE q4/g32 dequant law and PLE header reader
come from mlx-lm-unified (MIT, provenance/flashnext.json); the DS4.1 hash
and three-plane decode are original mlx2 code (MIT,
provenance/deepseek-v41-adapter.json).  See
provenance/technologies-embedding-tables-20261125.json.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .dispatch import KernelRegistry, spec_from_config

INT64_MAX = (1 << 63) - 1

# -------------------------------------------------------------------- spec

MULTIPLIER_LAWS = ("metadata", "numpy_default_rng", "splitmix64")
HISTORY_LAWS = ("blocked_pad", "pad_before_start", "eos_segment")


@dataclass(frozen=True)
class EngramSpec:
    """Geometry and laws of one family's hashed n-gram tables.

    ``max_ngram``: window length (orders ``2..max_ngram`` are hashed).
    ``heads_per_ngram``: moduli per order.
    ``multiplier_law``: ``metadata`` (sealed in the checkpoint, DS4.1),
    ``numpy_default_rng`` (Hy4: ``default_rng(seed + stride*layer)``,
    bound ``INT64_MAX // compressed // 2``), ``splitmix64`` (Qwen4 PLE:
    bound ``INT64_MAX // max(vocab, 1) // 2``).
    ``history_law``: how a window reaches before the chunk:
    ``blocked_pad`` (a ``-1``-masked token turns it and every older slot
    into ``pad_id``; DS4.1), ``pad_before_start`` (slots before absolute
    position 0 are ``pad_id``; Hy4), ``eos_segment`` (slots before the last
    EOS are the EOS id; Qwen4 PLE).
    ``tail_len``: history tokens a cache keeps (``max_ngram - 1``).
    ``pad_id``: pad token (DS4.1: a compressed id; Hy4: a raw token id).
    ``seed`` / ``layer_seed_stride``: seeded laws only.
    """

    max_ngram: int
    heads_per_ngram: int
    multiplier_law: str
    history_law: str
    pad_id: int = 0
    seed: int = 0
    layer_seed_stride: int = 10007

    def __post_init__(self):
        for name in ("max_ngram", "heads_per_ngram", "pad_id", "seed", "layer_seed_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"EngramSpec.{name} must be an integer, got {value!r}")
        if self.max_ngram < 2 or self.heads_per_ngram < 1:
            raise ValueError("Engram needs max_ngram >= 2 and at least one head")
        if self.multiplier_law not in MULTIPLIER_LAWS:
            raise ValueError(f"multiplier_law must be one of {MULTIPLIER_LAWS}")
        if self.history_law not in HISTORY_LAWS:
            raise ValueError(f"history_law must be one of {HISTORY_LAWS}")

    @property
    def orders(self) -> int:
        return self.max_ngram - 1

    @property
    def tail_len(self) -> int:
        return self.max_ngram - 1

    @property
    def heads(self) -> int:
        return self.orders * self.heads_per_ngram

    @classmethod
    def from_config(cls, config: Any, **overrides) -> "EngramSpec":
        """Hy4 (``engram_*`` keys) or Qwen4-exp PLE (``ngram_size`` keys)."""
        get = (config.get if isinstance(config, Mapping)
               else lambda k, d=None: getattr(config, k, d))
        if get("engram_max_ngram_size") is not None:
            mapping = {"max_ngram": "engram_max_ngram_size",
                       "heads_per_ngram": "engram_n_head_per_ngram",
                       "pad_id": "engram_pad_id", "seed": "engram_seed",
                       "multiplier_law": lambda c: "numpy_default_rng",
                       "history_law": lambda c: "pad_before_start"}
        elif get("ngram_size") is not None:
            def eos(c):
                value = get("eos_token_id")
                return value[0] if isinstance(value, list) else value
            mapping = {"max_ngram": "ngram_size", "heads_per_ngram": "heads_per_ngram",
                       "pad_id": eos, "seed": "seed",
                       "multiplier_law": lambda c: "splitmix64",
                       "history_law": lambda c: "eos_segment"}
        else:
            raise ValueError("config declares no Engram/PLE n-gram geometry")
        return spec_from_config(cls, config, mapping, **overrides)

    @classmethod
    def from_metadata(cls, metadata: Mapping, prefix: str = "deepseek41.engram.",
                      **overrides) -> "EngramSpec":
        """DS4.1: sealed layout (primes ``[tables, 24]``, multipliers ``[tables, 4]``)."""
        rows = tuple(metadata[prefix + "rows"])
        primes = np.asarray(metadata[prefix + "primes"], dtype=np.int64).reshape(len(rows), -1)
        multipliers = np.asarray(metadata[prefix + "multipliers"], dtype=np.int64).reshape(len(rows), -1)
        max_ngram = multipliers.shape[1]
        values = {"max_ngram": max_ngram, "heads_per_ngram": primes.shape[1] // (max_ngram - 1),
                  "multiplier_law": "metadata", "history_law": "blocked_pad",
                  "pad_id": int(metadata[prefix + "pad_id"])}
        values.update(overrides)
        return cls(**values)


# ---------------------------------------------------------- constants

_SPLITMIX_GAMMA = 11400714819323198485  # 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 13787848793156543929
_SPLITMIX_M2 = 10723151780598845931
_MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = value + _SPLITMIX_GAMMA & _MASK64
    value = (value ^ value >> 30) * _SPLITMIX_M1 & _MASK64
    value = (value ^ value >> 27) * _SPLITMIX_M2 & _MASK64
    return (value ^ value >> 31) & _MASK64


def seeded_multipliers(spec: EngramSpec, *, layer: int, compressed_vocab: int):
    """Odd per-position multipliers of one layer under ``spec.multiplier_law``.

    ``numpy_default_rng`` returns an int64 array (Hy4 ``EngramHasher``);
    ``splitmix64`` returns a list of Python ints (Qwen4
    ``_build_layer_multipliers``).  ``compressed_vocab`` is Hy4's
    ``table.max() + 1`` or Qwen4's ``vocab_size``.
    """
    count = spec.max_ngram
    if spec.multiplier_law == "numpy_default_rng":
        half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
        rng = np.random.default_rng(int(spec.seed) + spec.layer_seed_stride * int(layer))
        drawn = rng.integers(0, half_bound, size=count, dtype=np.int64)
        return drawn * 2 + 1
    if spec.multiplier_law == "splitmix64":
        base_seed = spec.seed + spec.layer_seed_stride * layer
        half_bound = max(1, INT64_MAX // max(compressed_vocab, 1) // 2)
        return [2 * (splitmix64(base_seed + _SPLITMIX_GAMMA * (index + 1) & _MASK64)
                     % half_bound) + 1 for index in range(count)]
    raise ValueError("metadata multipliers come from the checkpoint, not a seed")


def head_offsets(moduli) -> np.ndarray:
    """Flat row offset of each head: exclusive prefix sum over the last axis."""
    moduli = np.asarray(moduli, dtype=np.int64)
    zeros = np.zeros(moduli.shape[:-1] + (1,), np.int64)
    return np.cumsum(np.concatenate((zeros, moduli[..., :-1]), axis=-1), axis=-1).astype(np.int64)


# ----------------------------------------------------------- windows

def ngram_windows(spec: EngramSpec, history: np.ndarray, positions: np.ndarray,
                  valid: Optional[Callable] = None) -> np.ndarray:
    """Window ``[..., max_ngram]`` of each position: slot ``k`` is ``position - k``.

    Per ``spec.history_law``:

    * ``blocked_pad``: ``history`` is 1-D (compressed ids, ``-1`` = masked);
      once a ``-1`` is met walking back, that slot and every older one is
      ``pad_id``.
    * ``pad_before_start``: ``valid(k) -> bool[...]`` says which slots exist;
      the others are ``pad_id``.  ``history`` is ``[B, T]``.
    """
    n = spec.max_ngram
    if spec.history_law == "blocked_pad":
        blocked = np.zeros(len(positions), bool)
        out = []
        for shift in range(n):
            current = history[positions - shift]
            blocked |= current == -1
            out.append(np.where(blocked, spec.pad_id, current))
        return np.stack(out, axis=-1)
    if spec.history_law == "pad_before_start":
        B = history.shape[0]
        L = len(positions)
        out = np.full((B, L, n), spec.pad_id, dtype=np.int64)
        for k in range(n):
            src = positions - k
            ok = valid(k)
            out[:, ok, k] = history[:, src[ok]]
        return out
    raise ValueError(f"history law {spec.history_law} builds its windows in its model")


def pad_before_start_windows(spec: EngramSpec, tokens: np.ndarray, start: int, tail: np.ndarray,
                             document_positions: Optional[np.ndarray] = None) -> np.ndarray:
    """Hy4 window: each position's token with the ``max_ngram - 1`` before it; pad before 0.

    ``document_positions`` replaces the absolute slot position (the
    reference's ``document_offsets``).
    """
    n = spec.max_ngram
    B, L = tokens.shape
    history = np.concatenate([tail[:, -(n - 1):] if tail.size else np.zeros((B, 0), np.int64),
                              tokens], axis=1)
    lead = history.shape[1] - L
    pos = np.arange(L) + start if document_positions is None else document_positions
    local = np.arange(L) + lead

    def valid(k):
        return (pos >= k) & (local - k >= 0)

    return ngram_windows(spec, history, local, valid)


# -------------------------------------------------------------- hash

def ngram_hash(windows: np.ndarray, multipliers: np.ndarray, moduli: np.ndarray,
               heads_per_ngram: int, offsets: Optional[np.ndarray] = None) -> np.ndarray:
    """The shared n-gram hash law (exact int64, wrapping multiply).

    ``windows`` ``[..., K]`` compressed ids (slot 0 = current token);
    ``multipliers`` broadcastable to it; ``moduli`` ``[..., (K-1)*heads]``
    broadcastable against the hashed ``[..., 1]``.  For each order
    ``n = 2..K``: ``XOR_{k<n}(c_k * m_k) mod moduli[order heads]``.  Returns
    int64 ``[..., (K-1)*heads]``, plus ``offsets`` when given.
    """
    windows, multipliers, moduli = (np.asarray(v) for v in (windows, multipliers, moduli))
    K = windows.shape[-1]
    with np.errstate(over="ignore"):
        mixed = windows * multipliers
    out = []
    mix = mixed[..., 0]
    for n in range(2, K + 1):
        mix = np.bitwise_xor(mix, mixed[..., n - 1])
        start = (n - 2) * heads_per_ngram
        out.append(np.remainder(mix[..., None], moduli[..., start:start + heads_per_ngram]))
    hashed = np.concatenate(out, axis=-1)
    return hashed if offsets is None else hashed + offsets


def _hash_reference(spec, windows, multipliers, moduli, offsets=None):
    return ngram_hash(windows, multipliers, moduli, spec.heads_per_ngram, offsets)


HASH_REGISTRY = KernelRegistry("embedding_tables.ngram_hash", _hash_reference)

DEVICE_HASH_MAX_MODULUS = 1 << 24


def device_limb16_refusal(spec, *, tables, multipliers, moduli, offsets):
    """Why ``hy_v4_flash_device_hash.DeviceEngramHasher`` would refuse, else ``None``.

    Stacked per-layer inputs: ``tables`` ``[layers, size]`` lookup tables,
    ``multipliers`` ``[layers, max_ngram]``, ``moduli`` and ``offsets``
    ``[layers, orders*heads]``.  Mirrors the constructor's refusals in order
    (``compressed ids must fit 32 bits``, multiplier bound, modulus range,
    int32 rows).  The device hasher is not a registered fast path: it maps
    raw tokens through the tables itself, so it cannot take the reference's
    ``(windows, multipliers, moduli)`` arguments; the Hy4 model builds it
    directly under ``engram_hash: device``.
    """
    if spec.multiplier_law != "numpy_default_rng" or spec.history_law != "pad_before_start":
        return f"policy_off: device limb hash is built for the Hy4 laws only ({spec.multiplier_law}/{spec.history_law})"
    tables, multipliers = np.asarray(tables), np.asarray(multipliers)
    moduli, offsets = np.asarray(moduli), np.asarray(offsets)
    if tables.min() < 0 or tables.max() >= 1 << 32:
        return "geometry: compressed ids must fit 32 bits"
    for c_max, row in zip(tables.max(axis=1).tolist(), multipliers.tolist()):
        if any(m <= 0 or m >= 1 << 63 or int(c_max) * m >= 1 << 63 for m in row):
            return "geometry: multiplier bound exceeds the exact device range"
    if moduli.min() <= 0 or moduli.max() > DEVICE_HASH_MAX_MODULUS:
        return f"geometry: head moduli outside 1..{DEVICE_HASH_MAX_MODULUS}"
    if (offsets + moduli).max() >= 1 << 31:
        return "geometry: flat rows do not fit int32"
    return None


# ------------------------------------------------------ bf16 helpers

def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """View uint16 bfloat16 bits as float32 values."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bits with round-to-nearest-even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(32767) + (bits >> 16 & 1)
    return (rounded >> 16).astype(np.uint16)


def round_f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """float32 rounded to bfloat16 (ties to even), kept as float32."""
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def scale_bytes_to_f32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Little-endian scale/bias bytes ``[..., k*width]`` -> fp32 ``[..., k]``."""
    raw = np.ascontiguousarray(raw)
    if dtype == "BF16":
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    return raw.view("<f4" if dtype == "F32" else "<f2").astype(np.float32)


# ------------------------------------------------------ affine rows

SCALE_BYTES = {"BF16": 2, "F32": 4, "F16": 2}


@dataclass(frozen=True)
class Affine4RowSpec:
    """MLX affine 4-bit rows: ``value = scale * q + bias`` in fp32.

    ``rounding``: ``none`` (fp32 result; the model's cast rounds it, Hy4) or
    ``bf16_rne`` (round once to bf16 ties-to-even, the Metal
    ``mx.dequantize`` law; DS4.1 and Qwen4 PLE).  ``nonfinite``: ``refuse``
    (raise ``ValueError("nonfinite decoded Engram row")``) or ``propagate``
    (Qwen4 PLE).  ``output``: ``f32`` or ``bf16_bits`` (uint16).
    """

    dim: int
    group_size: int = 32
    scale_dtype: str = "BF16"
    rounding: str = "bf16_rne"
    nonfinite: str = "refuse"
    output: str = "f32"

    def __post_init__(self):
        if self.dim % self.group_size or self.dim % 8:
            raise ValueError("affine4 rows need a width divisible by the group size")
        if self.rounding not in ("none", "bf16_rne") or self.nonfinite not in ("refuse", "propagate"):
            raise ValueError("unknown affine4 rounding/nonfinite policy")
        if self.output not in ("f32", "bf16_bits") or (self.output == "bf16_bits" and self.rounding != "bf16_rne"):
            raise ValueError("bf16_bits output needs bf16_rne rounding")
        if self.scale_dtype not in SCALE_BYTES:
            raise ValueError(f"affine4 scale dtype must be one of {sorted(SCALE_BYTES)}")

    @property
    def groups(self) -> int:
        return self.dim // self.group_size

    @property
    def packed_row_bytes(self) -> int:
        return self.dim // 2 + 2 * SCALE_BYTES[self.scale_dtype] * self.groups


def decode_affine4_g32(spec: Affine4RowSpec, words: np.ndarray, scales: np.ndarray,
                       biases: np.ndarray) -> np.ndarray:
    """``words`` U32 ``[n, dim/8]`` (low nibble first), fp32 ``scales``/``biases`` ``[n, groups]``."""
    n = words.shape[0]
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    q = ((words[:, :, None] >> shifts) & np.uint32(15)).reshape(n, spec.dim).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        values = (q.reshape(n, spec.groups, spec.group_size) * scales[:, :, None]
                  + biases[:, :, None])
        if spec.rounding == "bf16_rne":
            values = round_f32_to_bf16(values)
    values = values.reshape(n, spec.dim)
    if spec.nonfinite == "refuse" and not np.isfinite(values).all():
        raise ValueError("nonfinite decoded Engram row")
    if spec.output == "bf16_bits":
        return (values.view(np.uint32) >> 16).astype(np.uint16)
    return values.astype(np.float32, copy=False)


def split_packed_affine4(spec: Affine4RowSpec, raw: np.ndarray):
    """One packed row per table row: codes, then scales, then biases."""
    if raw.ndim != 2 or raw.dtype != np.uint8 or raw.shape[1] != spec.packed_row_bytes:
        raise ValueError(f"packed affine4 rows require uint8 [n, {spec.packed_row_bytes}]")
    groups, width = spec.groups, SCALE_BYTES[spec.scale_dtype]
    words = np.ascontiguousarray(raw[:, : spec.dim // 2]).view("<u4")
    at = spec.dim // 2
    scales = scale_bytes_to_f32(raw[:, at: at + groups * width], spec.scale_dtype)
    biases = scale_bytes_to_f32(raw[:, at + groups * width: at + 2 * groups * width], spec.scale_dtype)
    return words, scales, biases


def decode_packed_affine4(spec: Affine4RowSpec, raw: np.ndarray) -> np.ndarray:
    return decode_affine4_g32(spec, *split_packed_affine4(spec, raw))


# ---------------------------------------------------------- row I/O

def pread_exact(fd: int, size: int, offset: int, short_message: str = "short Engram row read") -> bytes:
    """Read exactly ``size`` bytes at ``offset``; ``OSError(short_message)`` at EOF."""
    data = bytearray()
    while len(data) < size:
        block = os.pread(fd, size - len(data), offset + len(data))
        if not block:
            raise OSError(short_message)
        data.extend(block)
    return bytes(data)


# ------------------------------------------------------ safetensors

def parse_safetensors_header(first8: bytes, read_payload: Callable[[int], bytes], *,
                             strict_error: Optional[type] = None, path: Any = None,
                             file_size: Optional[int] = None, header_limit: Optional[int] = None,
                             length_bounds: Optional[tuple] = None,
                             bound_message: str = "implausible safetensors header"):
    """``(header, data_offset)`` of a safetensors file.  ``__metadata__`` is kept.

    Lax mode (``strict_error=None``; the Engram/PLE/bucket readers):
    ``struct.unpack`` the prefix (``struct.error`` when short), refuse a length
    outside ``length_bounds`` with ``ValueError(bound_message)``, and
    ``json.loads`` the payload bytes as returned.

    Strict mode (``weight_stream``): ``strict_error(...)`` for a short prefix,
    a length outside ``0 < length <= min(header_limit, file_size - 8)`` (a
    ``None`` bound is skipped),
    undecodable UTF-8/JSON or a duplicate key, each naming ``path``.
    """
    if strict_error is None:
        (length,) = struct.unpack("<Q", first8)
        if length_bounds is not None and not length_bounds[0] <= length <= length_bounds[1]:
            raise ValueError(bound_message)
        return json.loads(read_payload(length)), 8 + length
    if len(first8) != 8:
        raise strict_error(f"truncated safetensors header: {path}")
    length = int.from_bytes(first8, "little")
    caps = [limit for limit in (header_limit, None if file_size is None else file_size - 8)
            if limit is not None]
    if not 0 < length <= (min(caps) if caps else length):
        raise strict_error(f"invalid safetensors header length: {path}")
    raw = read_payload(length)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise strict_error(f"duplicate safetensors key {key!r}: {path}")
            result[key] = value
        return result

    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise strict_error(f"invalid safetensors JSON header: {path}") from error
    return header, 8 + length


def read_safetensors_header_file(path, **rules):
    """Open ``path`` and :func:`parse_safetensors_header` it (lax unless ``strict_error``)."""
    with open(path, "rb") as handle:
        if rules.get("strict_error") is not None and rules.get("file_size") is None:
            rules["file_size"] = os.fstat(handle.fileno()).st_size
        return parse_safetensors_header(handle.read(8), handle.read, path=path, **rules)


__all__ = [
    "Affine4RowSpec", "EngramSpec", "HASH_REGISTRY", "bf16_bits_to_f32", "device_limb16_refusal", "decode_affine4_g32",
    "decode_packed_affine4", "f32_to_bf16_bits", "head_offsets", "ngram_hash", "ngram_windows",
    "pad_before_start_windows", "parse_safetensors_header", "pread_exact",
    "read_safetensors_header_file", "round_f32_to_bf16", "scale_bytes_to_f32",
    "seeded_multipliers", "split_packed_affine4", "splitmix64",
]
