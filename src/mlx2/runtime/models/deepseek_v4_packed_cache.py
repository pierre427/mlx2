"""Opt-in physical mixed FP8 cache; ordinary FP32 cache remains the default.

Only local and completed attention pools are packed. Compressor carry and
indexer state stay FP32. Unpacking is temporary attention workspace, never
published state. See provenance/deepseek-v4-packed-kv.json.
"""

from dataclasses import replace

from .deepseek_v4_cache import DeepSeekV4LayerCache, DeepSeekV4PoolCache, _copy
from .deepseek_v4_packed_kv import LAYOUT_REVISION, PackedKV, pack_fp8_kv, unpack_fp8_kv

STATE_REVISION = "mlx2.deepseek-v4-packed-cache.v2"
CACHE_LAYOUT = "deepseek-v4-local-overlap-pooled-b1-fp8-mixed-v1"


def _arrays(packed):
    return None if packed is None else (packed.codes, packed.scales, packed.rope)


def _validate_packed_arrays(xp, arrays):
    if arrays is None:
        return
    if not isinstance(arrays, (tuple, list)) or len(arrays) != 3:
        raise ValueError("invalid packed KV state")
    codes, scales, rope = arrays
    if (
        any(value is None for value in arrays)
        or codes.ndim < 1
        or codes.dtype != xp.uint8
        or codes.shape[-1] != 448
        or scales.dtype != xp.float32
        or rope.dtype != xp.float32
        or scales.shape != (*codes.shape[:-1], 7)
        or rope.shape != (*codes.shape[:-1], 64)
    ):
        raise ValueError("invalid packed KV state geometry")


def _restore(xp, arrays):
    _validate_packed_arrays(xp, arrays)
    return None if arrays is None else PackedKV(*_copy(xp, arrays))


class DeepSeekV4PackedPoolCache(DeepSeekV4PoolCache):
    @property
    def pooled(self):
        return (
            None
            if self._packed_pool is None
            else unpack_fp8_kv(self._packed_pool, xp=self.xp)
        )

    @pooled.setter
    def pooled(self, value):
        self._packed_pool = None if value is None else pack_fp8_kv(value, xp=self.xp)

    @property
    def pooled_shape(self):
        return (
            None
            if self._packed_pool is None
            else (*self._packed_pool.codes.shape[:-1], self.head_dim)
        )

    @property
    def state(self):
        return (
            self.buf_kv,
            self.buf_gate,
            self.previous_kv,
            self.previous_gate,
            _arrays(self._packed_pool),
        )

    @state.setter
    def state(self, arrays):
        if len(arrays) != 5:
            raise ValueError("invalid packed pool state")
        self.buf_kv, self.buf_gate, self.previous_kv, self.previous_gate = _copy(
            self.xp, arrays[:4]
        )
        self._packed_pool = _restore(self.xp, arrays[4])

    @property
    def packed_pool(self):
        """Raw immutable physical planes; never materialize FP32 pool rows."""
        return self._packed_pool

    def append_pooled(self, transformed):
        self.append_pooled_packed(transformed)
        return self.pooled

    def append_pooled_packed(self, transformed):
        """Append identical stored rows, returning raw planes for direct attention."""
        if (
            transformed.ndim != 3
            or transformed.shape[0] != 1
            or transformed.shape[2] != self.head_dim
        ):
            raise ValueError("invalid transformed pool geometry")
        if self.pooled_count + transformed.shape[1] != self.offset // self.ratio:
            raise ValueError("transformed pool count differs from completed windows")
        if transformed.shape[1]:
            new = pack_fp8_kv(transformed, xp=self.xp)
            if self._packed_pool is not None:
                fields = tuple(
                    self.xp.concatenate((a, b), axis=1)
                    for a, b in zip(_arrays(self._packed_pool), _arrays(new))
                )
                new = PackedKV(*fields)
            self._packed_pool = new
        return self._packed_pool


class DeepSeekV4PackedLayerCache(DeepSeekV4LayerCache):
    cache_operations_contract = replace(
        DeepSeekV4LayerCache.cache_operations_contract,
        state_revision=STATE_REVISION,
        cache_layout=CACHE_LAYOUT,
        numerics=LAYOUT_REVISION,
    )

    def __init__(
        self,
        ratio=0,
        *,
        xp,
        window_size=128,
        head_dim=512,
        index_head_dim=128,
        identity="",
    ):
        if head_dim != 512:
            raise ValueError("packed cache requires width512")
        super().__init__(
            ratio,
            xp=xp,
            window_size=window_size,
            head_dim=head_dim,
            index_head_dim=index_head_dim,
            identity=identity,
        )
        self.compressor = (
            DeepSeekV4PackedPoolCache(ratio, head_dim, xp=xp) if ratio else None
        )

    @property
    def keys(self):
        return (
            None
            if self._packed_local is None
            else unpack_fp8_kv(self._packed_local, xp=self.xp)
        )

    @keys.setter
    def keys(self, value):
        self._packed_local = None if value is None else pack_fp8_kv(value, xp=self.xp)

    @property
    def local_shape(self):
        return (
            None
            if self._packed_local is None
            else (*self._packed_local.codes.shape[:-1], self.head_dim)
        )

    def update_local(self, keys):
        if self._closed:
            raise ValueError("cache row is closed")
        if (
            keys.ndim != 4
            or keys.shape[:2] != (1, 1)
            or keys.shape[-1] != self.head_dim
            or not keys.shape[2]
        ):
            raise ValueError("invalid B1 local KV geometry")
        previous = self._packed_local
        # Attention needs the chronological FP32 band, but persistent retention
        # copies raw planes: historical rows must never be requantized here.
        prior = None
        if previous is not None and self.window_size > 1:
            prior = unpack_fp8_kv(
                PackedKV(
                    *(a[:, :, -(self.window_size - 1) :] for a in _arrays(previous))
                ),
                xp=self.xp,
            )
        band = keys if prior is None else self.xp.concatenate((prior, keys), axis=2)
        retained = pack_fp8_kv(keys[:, :, -self.window_size :], xp=self.xp)
        old_count = self.window_size - keys.shape[2]
        if previous is not None and old_count > 0:
            # Concatenation owns a compact allocation for each retained plane;
            # pack_fp8_kv also owns its planes when no old rows are retained.
            retained = PackedKV(
                *(
                    self.xp.concatenate((a[:, :, -old_count:], b), axis=2)
                    for a, b in zip(_arrays(previous), _arrays(retained))
                )
            )
        self._packed_local = retained
        self.offset += keys.shape[2]
        return band

    @property
    def state(self):
        return (
            _arrays(self._packed_local),
            None if self.compressor is None else self.compressor.state,
            None if self.indexer is None else self.indexer.state,
        )

    def validate_storage(self):
        # Reading keys/pooled would reconstruct the entire compressed history.
        _validate_packed_arrays(self.xp, _arrays(self._packed_local))
        if self.compressor is not None:
            _validate_packed_arrays(self.xp, _arrays(self.compressor.packed_pool))
        values = []
        if self.compressor is not None:
            values.extend(self.compressor.state[:4])
        if self.indexer is not None:
            values.extend(self.indexer.state)
        if any(value is not None and value.dtype != self.xp.float32 for value in values):
            raise ValueError("DeepSeek compressor/indexer state requires physical FP32 dtype")

    @state.setter
    def state(self, arrays):
        if len(arrays) != 3:
            raise ValueError("invalid packed layer state")
        self._packed_local = _restore(self.xp, arrays[0])
        for pool, state in zip((self.compressor, self.indexer), arrays[1:]):
            if pool is not None:
                pool.state = state
            elif state:
                raise ValueError("unexpected packed pool state")

    @property
    def meta_state(self):
        return (STATE_REVISION, *super().meta_state[1:])

    @meta_state.setter
    def meta_state(self, value):
        DeepSeekV4LayerCache.meta_state.fset(self, value)
