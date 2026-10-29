# SPDX-License-Identifier: MIT
"""Forward-owned Source prefill geometry; no persistent arrays or backend import.

Expressions come from the ordinary Rotary and V4.1 Attention implementations.
See provenance/deepseek-v41-prefill-frame-geometry.json.
"""
from functools import lru_cache
from pathlib import Path
import hashlib

POLICY = "prefill_v1"


class FrameGeometry:
    """Opaque input-only owner, intentionally not a completion-root container.

    Each consumer's graph retains its phase/mask inputs. These are not independent
    cache writes and must not add scalar completion witnesses at every layer.
    """
    __slots__ = ("phases", "masks", "counts", "counters")

    def __init__(self, counters=None):
        self.phases, self.masks, self.counts = {}, {}, {}
        self.counters = counters

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


@lru_cache(maxsize=1)
def implementation_identity():
    paths = (__file__, str(Path(__file__).with_name("deepseek_v41.py")),
             str(Path(__file__).with_name("deepseek_v41_fused.py")))
    return hashlib.sha256((POLICY + ":" + ":".join(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths
    )).encode()).hexdigest()


def frame_geometry(weights, length, counters=None):
    if length > 1 and getattr(weights, "source_frame_reuse_policy", "reference") == "prefill_v1":
        return FrameGeometry(counters)
    return None


def _count(cache, name):
    counts = cache["counts"]
    counts[name] = counts.get(name, 0) + 1
    counters = cache.get("counters")
    if counters is not None:
        public = "source_frame_" + name.replace("phase", "rope")
        counters[public] = counters.get(public, 0) + 1


def phase_operands(cache, xp, rope, rope_key, x, offset=0, *, positions=None, inverse=False):
    """Share only CPU-described positions; array-valued positions keep fallback.

    The key is the exact constructor configuration, not the inverse-array
    address (each compatible Attention owns an independently computed vector).
    Inverse computes negative angles before cos/sin, as the ordinary law does.
    """
    if cache is None:
        return None
    if positions is None:
        position_key = ("range", offset, x.shape[-2])
    elif isinstance(positions, (tuple, list, range)) and all(type(p) is int for p in positions):
        position_key = ("explicit", tuple(positions))
    else:
        return None
    key = (rope_key, position_key, str(x.dtype), str(xp.default_device()), bool(inverse))
    phases = cache["phases"]
    if key in phases:
        _count(cache, "phase_hit")
        return phases[key]
    pos = xp.arange(offset, offset + x.shape[-2]) if positions is None else xp.array(positions)
    angles = pos.astype(xp.float32)[:, None] * rope.inv[None]
    if inverse:
        angles = -angles
    value = xp.cos(angles).astype(x.dtype), xp.sin(angles).astype(x.dtype)
    phases[key] = value
    _count(cache, "phase_build")
    return value


def apply_rotary(xp, x, rope, phase):
    """The ordinary Rotary expression, with its phase operands supplied."""
    c, s = phase
    prefix, rotary = x[..., : -rope.dims], x[..., -rope.dims :]
    a, b = rotary[..., ::2], rotary[..., 1::2]
    rotated = xp.stack((a * c - b * s, a * s + b * c), axis=-1).reshape(rotary.shape)
    return xp.concatenate((prefix, rotated), axis=-1)


def local_mask(cache, xp, start, length, local_length, window):
    key = (start, length, local_length, window, str(xp.default_device()))
    if cache is not None and key in cache["masks"]:
        _count(cache, "mask_hit")
        return cache["masks"][key]
    queries = xp.arange(start, start + length)[:, None]
    keys = xp.arange(start + length - local_length, start + length)
    value = ((queries >= keys) & (queries < keys + window))[None, None]
    if cache is not None:
        cache["masks"][key] = value
        _count(cache, "mask_build")
    return value
