# SPDX-License-Identifier: MIT
"""Storage-only int8 codec for recurrent state held in APCv2 entries.

An approximate state operation (AGENTS.md): default off, and selectable only
in qualification mode until a qualification record carries its evidence.
Live decode state and speculative rollback records are never touched; the
codec applies when APCv2 *stores* an entry (resident tier, its interior
checkpoints, and therefore its disk spill) and is undone when an entry is
restored into a request-private cache.

Which leaves: every slot of an ``ArraysCache``-family cache (including its
recorded prefill checkpoints) that holds a float32 array of rank 4, i.e. a
recurrent state ``[B, H, Dv, Dk]``.  Convolution windows (rank 3, activation
dtype) and token-history slots (integer) are left exact.  No model name is
consulted.

Layout ``int8-row-v1``: one symmetric float32 scale per row of the last axis
(per head and value row for GDN), round-to-nearest, ``qmax = 127``.  This is
the measured "once" arm of docs/research/GDN-STATE-QUANT-2026-10-01.md: a
single round-trip at a restore point sits at the drift floor on Qwen3.6-35B
and Qwen3.8-27B.  Prior art (read, nothing copied): omlx #2644's storage-only
``int8`` codec for persisted GDN sidecars, which uses the same row axis.

An encoded leaf is a plain dict of arrays, so it rides through COW cloning
and ``save_prompt_cache``/``load_prompt_cache`` unchanged.  It names its
codec in a uint8 array; a reader that does not recognise it, or finds a
malformed payload, fails closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import mlx.core as mx

from ..contracts import Fidelity

TAG = "recurrent-state-codec-v1"
RECEIPT_SCHEMA = "mlx2.approximate-recurrent-state.v1"
CODEC_KEY = "rsc_codec"
INT8_ROW_V1 = "int8-row-v1"
CODECS = {
    INT8_ROW_V1: {
        "codec": INT8_ROW_V1,
        "leaf": "ArraysCache slot, float32, rank 4",
        "payload": "int8",
        "qmax": 127,
        "rounding": "round-to-nearest",
        "scale": "float32 per row of the last axis, max|x|/qmax",
        "nonfinite_row": "decodes to NaN",
    }
}
_QMAX = 127.0
_FIELDS = frozenset({CODEC_KEY, "q", "scale"})

# Always-on bounded host counters (no device sync).  Process-wide: a
# restored copy is decoded below APCv2, in ``_copy_prompt_cache_for_restore``.
STATS = {
    "encoded_leaves": 0,
    "decoded_leaves": 0,
    "source_bytes": 0,
    "encoded_bytes": 0,
}


def codec_revision(codec: str) -> str:
    descriptor = json.dumps(CODECS[codec], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(descriptor.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class RecurrentStateCodecPolicy:
    """Server-selected storage codec for recurrent APCv2 state.  Default off.

    ``qualified`` stays False until a qualification record carries evidence;
    serving refuses an enabled, unqualified policy outside qualification
    mode, exactly like approximate KV and int8 prefill.
    """

    enabled: bool = False
    codec: str = INT8_ROW_V1
    qualified: bool = False
    evidence: tuple = field(default=())

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("recurrent state codec enabled must be a boolean")
        if self.codec not in CODECS:
            raise ValueError(f"recurrent state codec must be one of {sorted(CODECS)}")
        if not isinstance(self.qualified, bool):
            raise ValueError("recurrent state codec qualified must be a boolean")
        if self.qualified and not self.evidence:
            raise ValueError("a qualified recurrent state codec requires evidence")

    @classmethod
    def from_value(cls, value) -> "RecurrentStateCodecPolicy":
        """``None``/``False``/``"off"`` -> disabled; a codec name -> enabled;
        or a mapping of the dataclass fields."""
        if value is None or value is False or value == "off":
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            if value not in CODECS:
                raise ValueError(
                    f"recurrent state codec must be 'off' or one of {sorted(CODECS)}, got {value!r}"
                )
            return cls(enabled=True, codec=value)
        if isinstance(value, Mapping):
            unknown = set(value) - {"enabled", "codec", "qualified", "evidence"}
            if unknown:
                raise ValueError(f"unknown recurrent state codec keys: {sorted(unknown)}")
            values = dict(value)
            if "evidence" in values:
                values["evidence"] = tuple(str(item) for item in values["evidence"])
            return cls(**values)
        raise ValueError("recurrent state codec must be 'off', a codec name, or a mapping")

    @property
    def revision(self) -> str:
        return codec_revision(self.codec)

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "operation": f"apcv2-recurrent-state-{self.codec}" if self.enabled else None,
            "codec": self.codec if self.enabled else None,
            "revision": self.revision if self.enabled else None,
            "fidelity": Fidelity.APPROXIMATE.value if self.enabled else Fidelity.EXACT.value,
            "qualified": bool(self.qualified) if self.enabled else None,
            "state": (
                None
                if not self.enabled
                else "qualified" if self.qualified else "candidate"
            ),
            "scope": "apcv2 stored entries (resident, interior checkpoints, disk); "
            "live state and rollback records stay exact",
        }

    def receipt(self, *, restored_tokens: int, restored_leaves: int = 0) -> dict:
        """Per-request route receipt; only emitted when the policy is on."""
        return {
            "schema": RECEIPT_SCHEMA,
            "operation": f"apcv2-recurrent-state-{self.codec}",
            "codec": self.codec,
            "revision": self.revision,
            "fidelity": Fidelity.APPROXIMATE.value,
            "qualified": bool(self.qualified),
            "reason": "qualified" if self.qualified else "candidate_validation",
            "observed_used": restored_leaves > 0,
            "restored_from_codec_state": restored_leaves > 0,
            "restored_leaves": int(restored_leaves),
            "restored_tokens": int(restored_tokens) if restored_leaves > 0 else 0,
        }


def apc_state_codec_fingerprint(base, policy):
    """APCv2 semantic namespace for entries stored under ``policy``.

    Disabled: ``base`` unchanged, so exact namespaces (and their persisted
    blocks) keep their identity.  Enabled: ``(base, TAG, codec@revision)``,
    so a compressed entry is never served to an exact server, in memory or
    across a restart, and vice versa.
    """
    if policy is None or not policy.enabled:
        return base
    return (base, TAG, f"{policy.codec}@{policy.revision}")


def key_codec_layer(semantic):
    """The ``codec@revision`` layer of an APCv2 semantic namespace, or None."""
    while isinstance(semantic, tuple) and len(semantic) == 3 and isinstance(semantic[1], str):
        if semantic[1] == TAG:
            return semantic[2]
        semantic = semantic[0]
    return None


# --------------------------------------------------------------- tensor codec
def is_recurrent_leaf(value) -> bool:
    return (
        isinstance(value, mx.array)
        and value.dtype == mx.float32
        and value.ndim == 4
        and value.shape[-1] > 0
    )


def is_encoded(value) -> bool:
    return isinstance(value, Mapping) and CODEC_KEY in value


def leaf_nbytes(value) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        return sum(int(item.nbytes) for item in value.values() if isinstance(item, mx.array))
    return int(value.nbytes)


def encode(state: mx.array, codec: str = INT8_ROW_V1) -> dict:
    if codec != INT8_ROW_V1:
        raise ValueError(f"unknown recurrent state codec {codec!r}")
    scale = mx.max(mx.abs(state), axis=-1, keepdims=True) / _QMAX
    # An all-zero row keeps a positive scale (and decodes to exact zeros).
    scale = mx.maximum(scale, mx.array(1e-30, dtype=mx.float32))
    q = mx.clip(mx.round(state / scale), -_QMAX, _QMAX).astype(mx.int8)
    return {
        CODEC_KEY: mx.array(list(codec.encode()), dtype=mx.uint8),
        "q": q,
        "scale": scale.astype(mx.float32),
    }


def _check_structure(record) -> None:
    if not isinstance(record, Mapping) or set(record) != _FIELDS:
        raise ValueError("malformed recurrent state codec record")
    tag, q, scale = record[CODEC_KEY], record["q"], record["scale"]
    if not all(isinstance(item, mx.array) for item in (tag, q, scale)):
        raise ValueError("recurrent state codec record holds non-array fields")
    if tag.dtype != mx.uint8 or tag.ndim != 1:
        raise ValueError("malformed recurrent state codec tag")
    if q.dtype != mx.int8 or q.ndim != 4:
        raise ValueError("recurrent state codec payload must be rank-4 int8")
    if scale.dtype != mx.float32 or tuple(scale.shape) != (*q.shape[:-1], 1):
        raise ValueError("recurrent state codec scale does not match its payload")


def record_codec(record) -> str:
    """The codec a record names (a tiny host read)."""
    _check_structure(record)
    try:
        return bytes(int(v) for v in record[CODEC_KEY].tolist()).decode()
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("unreadable recurrent state codec tag") from error


def decode(record) -> mx.array:
    """Structure-checked decode; values are checked by ``validate_prompt_cache``.

    A row whose stored scale is not finite (a nonfinite source row) decodes
    to NaN rather than to finite garbage."""
    codec = record_codec(record)
    if codec not in CODECS:
        raise ValueError(f"unknown recurrent state codec {codec!r}")
    scale = record["scale"]
    value = record["q"].astype(mx.float32) * scale
    return mx.where(mx.isfinite(scale), value, mx.array(float("nan"), dtype=mx.float32))


# --------------------------------------------------------- cache-tree walking
def _arrays_caches(prompt_cache):
    from .models.cache import ArraysCache, CacheList

    stack = list(prompt_cache)
    while stack:
        item = stack.pop()
        if isinstance(item, CacheList):
            stack.extend(item.caches)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, ArraysCache):
            yield item


def _encode_cache(cache, codec, counts):
    clone = copy.copy(cache)
    leaves = []
    for value in cache.cache:
        if is_recurrent_leaf(value):
            record = encode(value, codec)
            counts["encoded_leaves"] += 1
            counts["source_bytes"] += int(value.nbytes)
            counts["encoded_bytes"] += leaf_nbytes(record)
            leaves.append(record)
        else:
            leaves.append(value)
    clone.cache = leaves
    clone._checkpoints = [
        [
            (
                position,
                [
                    _encode_snapshot_leaf(value, codec, counts)
                    for value in snapshot
                ],
            )
            for position, snapshot in lane
        ]
        for lane in getattr(cache, "_checkpoints", [])
    ]
    return clone


def _encode_snapshot_leaf(value, codec, counts):
    if not is_recurrent_leaf(value):
        return value
    record = encode(value, codec)
    counts["encoded_leaves"] += 1
    counts["source_bytes"] += int(value.nbytes)
    counts["encoded_bytes"] += leaf_nbytes(record)
    return record


def encode_prompt_cache(prompt_cache, policy):
    """A store-private copy of ``prompt_cache`` with recurrent leaves encoded.

    The caller's cache objects and arrays are never modified (the live
    request may keep decoding from them).  Returns ``(cache, counts)``.
    """
    from .models.cache import ArraysCache, CacheList

    counts = {"encoded_leaves": 0, "source_bytes": 0, "encoded_bytes": 0}

    def convert(item):
        if isinstance(item, CacheList):
            return CacheList(*[convert(child) for child in item.caches])
        if isinstance(item, ArraysCache):
            if getattr(item, "speculating", False) or getattr(item, "_rollbacks", None):
                # A live speculation transaction is never a store source;
                # leave it to the existing COW validation to refuse.
                return item
            return _encode_cache(item, policy.codec, counts)
        return item

    converted = [convert(item) for item in prompt_cache]
    for name in counts:
        STATS[name] += counts[name]
    return converted, counts


def decode_prompt_cache(prompt_cache) -> int:
    """Decode every encoded leaf of a request-private restored copy in place.

    Slots are written through ``__setitem__`` (a COW branch records the
    first write of its recurrent plane); checkpoint lanes are rebuilt as new
    lists so a COW source is never mutated.  Returns the decoded count.
    """
    decoded = 0
    for cache in _arrays_caches(prompt_cache):
        before = decoded
        for index, value in enumerate(list(cache.cache)):
            if is_encoded(value):
                cache.cache[index] = decode(value)
                decoded += 1
        lanes = getattr(cache, "_checkpoints", None)
        if lanes and any(is_encoded(v) for lane in lanes for _p, snap in lane for v in snap):
            new_lanes = []
            for lane in lanes:
                new_lane = []
                for position, snapshot in lane:
                    leaves = []
                    for value in snapshot:
                        if is_encoded(value):
                            leaves.append(decode(value))
                            decoded += 1
                        else:
                            leaves.append(value)
                    new_lane.append((position, leaves))
                new_lanes.append(new_lane)
            cache._checkpoints = new_lanes
        # Request-private attribution, reset on every restore. A dense KV-only
        # warm hit must not inherit the process's other codec restores.
        cache._recurrent_codec_restored_leaves = decoded - before
    STATS["decoded_leaves"] += decoded
    return decoded


def restored_leaf_count(prompt_cache) -> int:
    """Local leaves decoded by this restore, independent of global counters."""
    return sum(
        int(getattr(cache, "_recurrent_codec_restored_leaves", 0))
        for cache in _arrays_caches(prompt_cache or [])
    )


def _leaves(cache):
    yield from cache.cache
    for lane in getattr(cache, "_checkpoints", []) or []:
        for _position, snapshot in lane:
            yield from snapshot


def validate_prompt_cache(prompt_cache, policy) -> int:
    """Fail closed unless ``prompt_cache`` matches ``policy`` exactly.

    Enabled: every recurrent leaf must be a well-formed record of the
    policy's codec with finite positive scales and payload in [-127, 127].
    Disabled: no encoded record may appear.  One batched device read.
    Returns the number of records checked; raises ValueError otherwise.
    """
    enabled = policy is not None and policy.enabled
    checks = []
    records = 0
    for cache in _arrays_caches(prompt_cache):
        for value in _leaves(cache):
            if is_encoded(value):
                if not enabled:
                    raise ValueError("codec-encoded recurrent state in an exact APCv2 namespace")
                if record_codec(value) != policy.codec:
                    raise ValueError("recurrent state codec does not match this server's policy")
                checks.append(mx.all(mx.isfinite(value["scale"]) & (value["scale"] > 0)))
                checks.append(mx.min(value["q"]) >= -127)
                records += 1
            elif enabled and is_recurrent_leaf(value):
                raise ValueError("exact recurrent state in a codec APCv2 namespace")
    if checks:
        flags = mx.stack(checks)
        mx.eval(flags)
        if not bool(mx.all(flags)):
            raise ValueError("corrupt recurrent state codec payload")
    return records
