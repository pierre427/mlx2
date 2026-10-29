"""Exact DeepSeek V4 local/pooling state, independent of tensor arithmetic imports.

Original implementation of the official overlapping-compression equations;
see provenance/deepseek-v4-cache.json. Pass numpy for CPU probes or mlx.core
for execution. Normalization, rotary positions and QAT remain model-owned.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cache_operations import (
    CacheOperationsContract, CachePlaneDeclaration, CheckpointPublicationCondition,
)


STATE_REVISION = "mlx2.deepseek-v4-cache.v2"


def _parse_meta_int(value, *, nullable=False):
    if nullable and value == "none":
        return None
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError("invalid cache integer metadata")
    number = int(value)
    if str(number) != value:
        raise ValueError("noncanonical cache integer metadata")
    return number


def _copy(xp, value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(_copy(xp, item) for item in value)
    # A contiguous slice can still retain the entire prefill allocation.
    # asarray(copy=True) explicitly owns storage, unlike array/contiguous.
    return (xp.asarray(value, copy=True) if xp.__name__ == "mlx.core"
            else xp.array(value, copy=True))


def _backend(state):
    for value in state:
        if isinstance(value, (tuple, list)):
            found = _backend(value)
            if found is not None:
                return found
        elif value is not None:
            if type(value).__module__.startswith("numpy"):
                import numpy
                return numpy
            import mlx.core
            return mlx.core
    return None


class _BackendOwner:
    # Generic APC descriptor cloning traverses instance fields. Store the
    # backend name, never a module object (modules cannot be COW/deep-copied).
    @property
    def xp(self):
        if self._backend_name == "numpy":
            import numpy
            return numpy
        import mlx.core
        return mlx.core

    @xp.setter
    def xp(self, module):
        if module.__name__ not in ("numpy", "mlx.core"):
            raise ValueError("unsupported cache tensor backend")
        self._backend_name = module.__name__


class DeepSeekV4PoolCache(_BackendOwner):
    """Retain incomplete and previous completed windows, plus transformed pools.

    ``compress`` consumes FP32 projected KV/gates (without APE), returns raw
    pooled vectors and their absolute starting positions. The caller applies
    norm, RoPE and any declared QAT, then calls ``append_pooled`` exactly once.
    Only complete windows are visible; the next window preserves the preceding
    window's first half when ratio=4, including across separate invocations.
    """

    def __init__(self, ratio, head_dim, *, xp):
        if ratio not in (4, 128) or head_dim <= 0:
            raise ValueError("unsupported DeepSeek V4 pool geometry")
        self.ratio, self.head_dim, self.xp = ratio, head_dim, xp
        self.offset = 0
        self.buf_kv = self.buf_gate = None
        self.previous_kv = self.previous_gate = None
        self.pooled = None

    @property
    def remainder(self):
        return self.offset % self.ratio

    @property
    def state(self):
        return (self.buf_kv, self.buf_gate, self.previous_kv,
                self.previous_gate, self.pooled)

    @state.setter
    def state(self, value):
        if len(value) != 5:
            raise ValueError("invalid DeepSeek V4 pool state")
        (self.buf_kv, self.buf_gate, self.previous_kv,
         self.previous_gate, self.pooled) = _copy(self.xp, value)

    def compress(self, kv, gate, ape, offset):
        xp, ratio, dim = self.xp, self.ratio, self.head_dim
        width = dim * (2 if ratio == 4 else 1)
        if offset != self.offset:
            raise ValueError("pool offset must continue the exact prefix")
        if (kv.ndim != 3 or gate.shape != kv.shape or kv.shape[0] != 1
                or kv.shape[-1] != width or ape.shape != (ratio, width)):
            raise ValueError("invalid B1 compressor projection geometry")
        if self.pooled_count != offset // ratio:
            raise ValueError("previous compressed outputs were not appended")
        count = kv.shape[1]
        kv, gate = kv.astype(xp.float32), gate.astype(xp.float32)
        if self.buf_kv is not None:
            kv = xp.concatenate((self.buf_kv, kv), axis=1)
            gate = xp.concatenate((self.buf_gate, gate), axis=1)
        windows = kv.shape[1] // ratio
        usable = windows * ratio
        starts = tuple(range(offset - self.remainder,
                             offset - self.remainder + usable, ratio))
        if windows:
            raw_kv = kv[:, :usable].reshape(1, windows, ratio, width)
            raw_gate = gate[:, :usable].reshape(1, windows, ratio, width)
            scores = raw_gate + ape.astype(xp.float32)
            if ratio == 4:
                prior_kv = self.previous_kv
                prior_gate = self.previous_gate
                if prior_kv is None:
                    prior_kv = xp.zeros((1, 1, ratio, dim), dtype=xp.float32)
                    prior_gate = xp.full((1, 1, ratio, dim), -float("inf"), dtype=xp.float32)
                left_kv = xp.concatenate((prior_kv, raw_kv[:, :-1, :, :dim]), axis=1)
                left_gate = xp.concatenate((prior_gate, scores[:, :-1, :, :dim]), axis=1)
                values = xp.concatenate((left_kv, raw_kv[..., dim:]), axis=2)
                scores_for_pool = xp.concatenate((left_gate, scores[..., dim:]), axis=2)
            else:
                values, scores_for_pool = raw_kv, scores
            maximum = xp.max(scores_for_pool, axis=2, keepdims=True)
            weights = xp.exp(scores_for_pool - maximum)
            pooled = xp.sum(values * (weights / xp.sum(weights, axis=2, keepdims=True)), axis=2)
            if ratio == 4:
                self.previous_kv = _copy(xp, raw_kv[:, -1:, :, :dim])
                self.previous_gate = _copy(xp, scores[:, -1:, :, :dim])
        else:
            pooled = xp.zeros((1, 0, dim), dtype=xp.float32)
        self.buf_kv = _copy(xp, kv[:, usable:])
        self.buf_gate = _copy(xp, gate[:, usable:])
        self.offset += count
        return pooled, starts

    def append_pooled(self, transformed):
        if (transformed.ndim != 3 or transformed.shape[0] != 1
                or transformed.shape[2] != self.head_dim):
            raise ValueError("invalid transformed pool geometry")
        previous = self.pooled_count
        if previous + transformed.shape[1] != self.offset // self.ratio:
            raise ValueError("transformed pool count differs from completed windows")
        if transformed.shape[1]:
            self.pooled = (_copy(self.xp, transformed) if self.pooled is None else
                           self.xp.concatenate((self.pooled, transformed), axis=1))
        return self.pooled

    @property
    def pooled_shape(self):
        return None if self.pooled is None else self.pooled.shape

    @property
    def pooled_count(self):
        shape = self.pooled_shape
        return 0 if shape is None else shape[1]

    def make_mask(self, length, offset):
        count = self.pooled_count
        return self.xp.arange(count)[None, :] < (
            (self.xp.arange(length) + offset + 1) // self.ratio)[:, None]


@dataclass(frozen=True)
class DeepSeekV4Snapshot:
    metadata: tuple
    arrays: tuple


class DeepSeekV4LayerCache(_BackendOwner):
    """B1 exact checkpoint state; speculation restores a snapshot then replays.

    Local KV uses chronological bounded storage rather than an in-place ring.
    ``update_local`` returns the entire verification/prefill band, while only
    the last window is retained. This preserves each query's causal context.
    Model/source identity belongs in ``identity`` and is checked on restore.
    """

    cache_operations_contract = CacheOperationsContract(
        family="deepseek_v4", topology="checkpointed_hybrid",
        state_revision=STATE_REVISION, cache_layout="local-overlap-pooled-b1",
        numerics="stored_dtype_fp32_pooling",
        planes=(CachePlaneDeclaration("attention_ring", ("state",),
                                      ("identity", "ratio", "window_size")),),
        exact_prefix=True, restores_at_checkpoints=True, rollback=None,
        publication_conditions=(CheckpointPublicationCondition(
            "pending", "truthy", "DeepSeek V4 local and pool updates are incomplete"),),
    )

    def __init__(self, ratio=0, *, xp, window_size=128, head_dim=512,
                 index_head_dim=128, identity=""):
        if ratio not in (0, 4, 128) or window_size <= 0 or not identity:
            raise ValueError("invalid DeepSeek V4 cache geometry or identity")
        self.xp, self.identity, self.ratio = xp, identity, ratio
        self.window_size, self.head_dim = window_size, head_dim
        self.index_head_dim, self.offset = index_head_dim, 0
        self._closed = False
        self.keys = None
        self.compressor = DeepSeekV4PoolCache(ratio, head_dim, xp=xp) if ratio else None
        self.indexer = DeepSeekV4PoolCache(4, index_head_dim, xp=xp) if ratio == 4 else None

    def update_local(self, keys):
        if self._closed:
            raise ValueError("cache row is closed")
        if keys.ndim != 4 or keys.shape[:2] != (1, 1) or keys.shape[-1] != self.head_dim:
            raise ValueError("invalid B1 local KV geometry")
        length = keys.shape[2]
        if not length:
            raise ValueError("local update must contain tokens")
        prior = None if self.keys is None else self.keys[:, :, -(self.window_size - 1):, :]
        if self.window_size == 1:
            prior = None
        band = keys if prior is None else self.xp.concatenate((prior, keys), axis=2)
        self.keys = _copy(self.xp, band[:, :, -self.window_size:, :])
        self.offset += length
        return band

    def update_and_fetch(self, keys, values):
        if values.shape != (*keys.shape[:3], 0):
            raise ValueError("DeepSeek V4 uses shared KV and zero-width values")
        band = self.update_local(keys)
        return band, self.xp.zeros((*band.shape[:3], 0), dtype=band.dtype)

    @property
    def state(self):
        return (self.keys, None if self.compressor is None else self.compressor.state,
                None if self.indexer is None else self.indexer.state)

    @state.setter
    def state(self, value):
        if len(value) != 3:
            raise ValueError("invalid DeepSeek V4 layer state")
        self.keys = _copy(self.xp, value[0])
        for pool, arrays in zip((self.compressor, self.indexer), value[1:]):
            if pool is not None:
                pool.state = arrays
            elif arrays:
                raise ValueError("unexpected pool state")

    @property
    def meta_state(self):
        return (STATE_REVISION, self.identity, str(self.ratio), str(self.window_size),
                str(self.head_dim), str(self.index_head_dim), str(self.offset),
                "none" if self.compressor is None else str(self.compressor.offset),
                "none" if self.indexer is None else str(self.indexer.offset))

    @meta_state.setter
    def meta_state(self, value):
        if tuple(value[:6]) != self.meta_state[:6] or len(value) != 9:
            raise ValueError("cache revision, identity or geometry mismatch")
        offsets = tuple(_parse_meta_int(v, nullable=True) for v in value[6:])
        if offsets[0] is None:
            raise ValueError("invalid cache offset")
        self.offset = offsets[0]
        for pool, offset in zip((self.compressor, self.indexer), offsets[1:]):
            if (pool is None) != (offset is None):
                raise ValueError("cache pool topology mismatch")
            if pool is not None:
                pool.offset = offset

    @classmethod
    def from_state(cls, state, meta_state, *, xp=None):
        xp = xp or _backend(state)
        if xp is None:
            # Generic safetensors restore has no arrays from which to infer the
            # backend for an empty cache. Its runtime backend is MLX.
            import mlx.core as xp
        if len(meta_state) != 9:
            raise ValueError("invalid cache metadata")
        _, identity, ratio, window, head_dim, index_dim, *_ = meta_state
        ratio, window, head_dim, index_dim = map(_parse_meta_int, (ratio, window, head_dim, index_dim))
        result = cls(ratio, xp=xp, window_size=window, head_dim=head_dim,
                     index_head_dim=index_dim, identity=identity)
        result.meta_state = meta_state
        result.state = state
        result.validate_state()
        return result

    def snapshot(self):
        self.validate_state()
        self.cache_operations_contract.validate_publication(self)
        return DeepSeekV4Snapshot(self.meta_state, _copy(self.xp, self.state))

    def restore(self, snapshot):
        restored = self.from_state(snapshot.arrays, snapshot.metadata, xp=self.xp)
        if restored.meta_state[:6] != self.meta_state[:6]:
            raise ValueError("snapshot revision, identity or geometry mismatch")
        restored.cache_operations_contract.validate_publication(restored)
        self.meta_state = restored.meta_state
        self.state = restored.state

    @property
    def pending(self):
        if self._closed:
            return True
        for pool in (self.compressor, self.indexer):
            if pool is None:
                continue
            count = pool.pooled_count
            if pool.offset != self.offset or count != self.offset // pool.ratio:
                return True
        return False

    @property
    def local_shape(self):
        return None if self.keys is None else self.keys.shape

    def validate_state(self):
        expected = min(self.offset, self.window_size)
        if self.local_shape is None:
            if expected:
                raise ValueError("local KV missing at nonempty offset")
        elif self.local_shape != (1, 1, expected, self.head_dim):
            raise ValueError("local KV shape differs from retained prefix")
        for pool in (self.compressor, self.indexer):
            if pool is None:
                continue
            width = pool.head_dim * (2 if pool.ratio == 4 else 1)
            for buf in (pool.buf_kv, pool.buf_gate):
                if buf is None:
                    if pool.offset:
                        raise ValueError("pool remainder buffer missing")
                elif buf.shape != (1, pool.remainder, width):
                    raise ValueError("pool remainder shape differs from offset")
            for previous in (pool.previous_kv, pool.previous_gate):
                required = pool.ratio == 4 and pool.offset >= 4
                if previous is None:
                    if required:
                        raise ValueError("previous overlap window missing")
                elif not required or previous.shape != (1, 1, 4, pool.head_dim):
                    raise ValueError("previous overlap window shape differs")
            shape = pool.pooled_shape
            if shape is not None and (len(shape) != 3 or
                    shape[0] != 1 or shape[2] != pool.head_dim):
                raise ValueError("completed pool shape differs")

    def is_trimmable(self):
        return False

    @classmethod
    def merge(cls, caches):
        if len(caches) != 1 or type(caches[0]) is not cls:
            raise ValueError("DeepSeek V4 supports only singleton B1 cache merge")
        source = caches[0]
        snapshot = source.snapshot()
        return cls.from_state(snapshot.arrays, snapshot.metadata, xp=source.xp)

    def extract(self, index):
        if index != 0:
            raise ValueError("DeepSeek V4 cache extract requires row zero")
        return type(self).merge([self])

    def filter(self, keep):
        keep = list(keep)
        if keep == []:
            self.close()
        elif keep != [0] or self._closed:
            raise ValueError("DeepSeek V4 supports only B1 row filtering")

    @property
    def left_padding(self):
        return self.xp.array([0], dtype=self.xp.int32)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if self._closed:
            raise ValueError("cache row is closed")
        for padding in (left_padding, right_padding):
            if padding is not None and list(padding) != [0]:
                raise ValueError("DeepSeek V4 B1 cache does not support padding")
        if lengths is not None and (len(lengths) != 1 or lengths[0] < 0):
            raise ValueError("DeepSeek V4 requires one nonnegative B1 length")

    def finalize(self):
        if self._closed:
            raise ValueError("cache row is closed")
        self.cache_operations_contract.validate_publication(self)

    def extend(self, other):
        raise ValueError("DeepSeek V4 does not support physical batch extension")

    def close(self):
        self.keys = None
        for pool in (self.compressor, self.indexer):
            if pool is not None:
                pool.state = (None,) * 5
                pool.offset = 0
        self.offset = 0
        self._closed = True

    def trim(self, count):
        raise NotImplementedError("DeepSeek V4 requires exact snapshot restore and replay")

    def empty(self):
        return self.offset == 0

    def size(self):
        return min(self.offset, self.window_size)

    @property
    def nbytes(self):
        def total(value):
            return sum(map(total, value)) if isinstance(value, (tuple, list)) else (0 if value is None else value.nbytes)
        return total(self.state)
