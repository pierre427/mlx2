# SPDX-License-Identifier: MIT
"""Revision-bound CSA2 state containing original quantizer codes and scales."""

from dataclasses import replace

from .cache import register_cache_class
from .deepseek_v4_cache import _copy, _parse_meta_int
from .deepseek_v41_cache import DeepSeekV41LayerCache
from .deepseek_v41_packed_qat import (
    PackedQAT,
    concatenate,
    decode,
    gather_decode,
    validate,
    with_codec,
)

STATE_REVISION = "mlx2.deepseek-v41-native-qat-cache.v1"
CACHE_LAYOUT = "deepseek-v41-shared-csa2-engram-b1-native-qat-v1"
# Model identities carry this salt when the exact fused Metal codec decodes
# the same code/scale bytes; the state layout and revision are unchanged.
CODEC_SALT = ":packed_codec="
RETAINED_STAGING_ROWS = 512


def _planes(value):
    return (None, None) if value is None else (value.codes, value.scales)


def _restore(codes, scales, kind):
    if (codes is None) != (scales is None):
        raise ValueError("incomplete V4.1 packed code/scale pair")
    return None if codes is None else PackedQAT(codes, scales, kind)


@register_cache_class
class DeepSeekV41PackedLayerCache(DeepSeekV41LayerCache):
    cache_operations_contract = replace(
        DeepSeekV41LayerCache.cache_operations_contract,
        state_revision=STATE_REVISION,
        cache_layout=CACHE_LAYOUT,
        numerics="original-fp8-g32-e8m0-fp4-g16-e4m3-index-fp4-g32-e8m0-bf16-values-v1",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.head_dim % 32 or self.index_head_dim % 32:
            raise ValueError("native QAT cache requires group32 aligned widths")

    @property
    def codec(self):
        identity = self.identity if type(self.identity) is str else ""
        marker = identity.find(CODEC_SALT)
        if marker < 0:
            return "reference"
        return identity[marker + len(CODEC_SALT):].split(":", 1)[0]

    @property
    def keys(self):
        return None if self._packed_local is None else decode(with_codec(self._packed_local, self.codec))

    @keys.setter
    def keys(self, value):
        if value is not None:
            raise ValueError("native QAT cache requires original producer codes")
        self._packed_local = None

    @property
    def packed_pooled(self):
        return None if self._pooled is None else with_codec(self._pooled.sliced((slice(None), slice(self._pool_count))), self.codec)

    @property
    def packed_index_keys(self):
        return None if self._index_keys is None else with_codec(self._index_keys.sliced((slice(None), slice(self._pool_count))), self.codec)

    @property
    def pooled(self):
        return None if self._pooled is None else decode(self.packed_pooled)

    @property
    def index_keys(self):
        return None if self._index_keys is None else decode(self.packed_index_keys)

    def gather_index_keys(self, indices):
        """Dequantize only gathered candidates, including per-query candidate sets."""
        return gather_decode(self.packed_index_keys, indices)

    def gather_pooled(self, indices):
        return gather_decode(self.packed_pooled, indices)

    def update_local(self, keys):
        raise ValueError("native QAT cache requires original producer codes")

    def update_local_packed(self, keys):
        if self._closed:
            raise ValueError("cache row is closed")
        shape = keys.shape
        if len(shape) != 4 or shape[:2] != (1, 1) or not shape[2]:
            raise ValueError("invalid B1 native local KV geometry")
        validate(keys, shape=(1, 1, shape[2], self.head_dim), kind="fp8", xp=self.xp)
        previous = self._packed_local
        prior = None
        if previous is not None and self.window_size > 1:
            prior = previous.sliced((slice(None), slice(None), slice(-(self.window_size - 1), None)))
        band = keys if prior is None else concatenate(prior, keys, 2)
        retained = band.sliced((slice(None), slice(None), slice(-self.window_size, None)))
        self._packed_local = PackedQAT(*_copy(self.xp, _planes(retained)), "fp8")
        self.offset += shape[2]
        return decode(with_codec(band, self.codec))

    def append_pools(self, pooled, index_keys):
        raise ValueError("native QAT cache requires original producer codes")

    def append_pools_packed(self, pooled, index_keys):
        if self._closed or not self.source:
            raise ValueError("invalid native QAT source owner")
        count = pooled.shape[1]
        validate(pooled, shape=(1, count, self.head_dim), kind="compressed_fp4", xp=self.xp)
        validate(index_keys, shape=(1, count, self.index_head_dim), kind="index_fp4", xp=self.xp)
        end = self._pool_count + count
        if end != self.offset // self.ratio:
            raise ValueError("native QAT append does not complete exact frontier")
        if self._pooled is None or end > self._capacity:
            capacity = ((end + 255) // 256) * 256
            values = []
            for old, new in ((self.packed_pooled, pooled), (self.packed_index_keys, index_keys)):
                codes = self.xp.zeros((1, capacity, new.codes.shape[-1]), self.xp.uint8)
                scales = self.xp.zeros((1, capacity, new.scales.shape[-1]), self.xp.uint8)
                if old is not None:
                    codes[:, :self._pool_count] = old.codes
                    scales[:, :self._pool_count] = old.scales
                values.append(PackedQAT(codes, scales, new.kind))
            self._pooled, self._index_keys = values
            self._capacity = capacity
        for old, new in ((self._pooled, pooled), (self._index_keys, index_keys)):
            old.codes[:, self._pool_count:end] = new.codes
            old.scales[:, self._pool_count:end] = new.scales
        self._pool_count = end

    @property
    def state(self):
        return (*_planes(self._packed_local), self.buf_kv, self.buf_gate,
                *_planes(self.packed_pooled), *_planes(self.packed_index_keys), self.hash_tail)

    @state.setter
    def state(self, arrays):
        if len(arrays) != 9:
            raise ValueError("invalid native QAT cache array inventory")
        lc, ls, buf_kv, buf_gate, pc, ps, ic, iss, hash_tail = _copy(self.xp, arrays)
        local = _restore(lc, ls, "fp8")
        pooled = _restore(pc, ps, "compressed_fp4")
        index_keys = _restore(ic, iss, "index_fp4")
        # Serialized planes contain live rows, not backing capacity. Validate
        # their raw extents before the public accessors can truncate them.
        if (pooled is None) != (index_keys is None):
            raise ValueError("incomplete native QAT source plane inventory")
        count = 0
        if pooled is not None:
            if len(pooled.shape) != 3:
                raise ValueError("invalid native QAT source plane geometry")
            count = pooled.shape[1]
            validate(pooled, shape=(1, count, self.head_dim), kind="compressed_fp4", xp=self.xp)
            validate(index_keys, shape=(1, count, self.index_head_dim), kind="index_fp4", xp=self.xp)
        self._packed_local, self._pooled, self._index_keys = local, pooled, index_keys
        self.buf_kv, self.buf_gate, self.hash_tail = buf_kv, buf_gate, hash_tail
        self._pool_count = self._capacity = count

    @property
    def meta_state(self):
        revision = (STATE_REVISION if self.ast_private_profile is None else
                    STATE_REVISION + ":private:" + self.ast_private_profile)
        return (revision, *super().meta_state[1:])

    @meta_state.setter
    def meta_state(self, value):
        DeepSeekV41LayerCache.meta_state.fset(self, value)

    @classmethod
    def from_state(cls, state, meta_state, *, xp=None):
        if len(meta_state) != 10 or meta_state[0] != STATE_REVISION:
            raise ValueError("invalid native QAT cache metadata revision")
        if xp is None:
            import mlx.core as xp
        ratio, window, head, index, source, history, vocab, offset = map(_parse_meta_int, meta_state[2:])
        if source not in (0, 1) or history not in (0, 1):
            raise ValueError("invalid native QAT ownership flags")
        result = cls(ratio, xp=xp, window_size=window, head_dim=head, index_head_dim=index,
                     identity=meta_state[1], source=source, history=history, compressed_vocab=vocab)
        result.offset = offset
        result.state = state
        result.validate_state()
        if result.pending:
            raise ValueError("incomplete native QAT frontier")
        return result

    @property
    def pending(self):
        if self._closed or self._in_forward or self.ced_local_deferred:
            return True
        return self.source and any((0 if p is None else min(p.shape[1], self._pool_count)) != self.offset // self.ratio
                                   for p in (self._pooled, self._index_keys))

    def validate_storage(self):
        if self.ced_local_deferred and (self._packed_local is not None or self.source):
            raise ValueError("native QAT deferred CED local state mismatch")
        for value, shape, kind in (
            (self._packed_local, (1, 1, min(self.offset, self.window_size), self.head_dim), "fp8"),
            (self.packed_pooled, (1, self.offset // self.ratio if self.source else 0, self.head_dim), "compressed_fp4"),
            (self.packed_index_keys, (1, self.offset // self.ratio if self.source else 0, self.index_head_dim), "index_fp4"),
        ):
            if value is None:
                if shape[-2] and not (kind == "fp8" and self.ced_local_deferred):
                    raise ValueError("native QAT cache plane missing")
            else:
                validate(value, shape=shape, kind=kind, xp=self.xp)
                if kind != "fp8" and not self.source:
                    raise ValueError("unexpected native QAT source plane")
        for value in (self.buf_kv, self.buf_gate):
            rem = self.offset % self.ratio if self.source else 0
            if value is None:
                if rem:
                    raise ValueError("native QAT partial group missing")
            elif not self.source or value.dtype != self.xp.float32 or value.shape != (1, rem, self.head_dim):
                raise ValueError("invalid native QAT partial group")
        if self.history:
            if self.hash_tail is None or self.hash_tail.dtype != self.xp.int32 or self.hash_tail.shape != (3,):
                raise ValueError("invalid native QAT n-gram tail")
            values = self.hash_tail.tolist()
            if any(type(i) is not int or i < -1 or i >= self.compressed_vocab for i in values):
                raise ValueError("invalid native QAT n-gram IDs")
            if any(i != -1 for i in values[min(self.offset, 3):]):
                raise ValueError("native QAT n-gram history precedes sequence")
        elif self.hash_tail is not None:
            raise ValueError("unexpected native QAT n-gram owner")

    def validate_state(self):
        if self.ast_private_profile is not None and (
            type(self.ast_private_profile) is not str or not self.ast_private_profile
        ):
            raise ValueError("invalid native QAT private AST numerical profile")
        self.validate_storage()

    @property
    def nbytes(self):
        return sum(0 if value is None else value.nbytes for value in
                   (self._packed_local, self.buf_kv, self.buf_gate, self._pooled, self._index_keys, self.hash_tail))


def stage_retained_pool(model, packed_pooled, indices, length, start, shared):
    """Compatible packed consumer for Source all-retained attention profiles.

    Under the trusted finite all-retained chronological selection, a prefill
    chunk visits every pooled row. Decode those at most 512 original code rows
    once into forward-local FP32 scratch, bitwise equal to the reference cache's
    floating plane, so the broadcast/oMLX laws see identical operands. Any other
    call keeps the bounded packed gather path. Nothing is retained on the cache.
    """
    count = packed_pooled.shape[1]
    if (length <= 1 or count > RETAINED_STAGING_ROWS or not count
            or getattr(model, "index_policy", "reference") != "finite_all_retained"
            or shared.get("_all_retained_chronological")
            != (id(indices), count, length, start, shared.get("_index_generation"))):
        return None, packed_pooled
    counters = model.scheduled_kv_calls
    counters["packed_retained_staged_rows"] = counters.get("packed_retained_staged_rows", 0) + count
    counters["packed_retained_staging_max_rows"] = max(counters.get("packed_retained_staging_max_rows", 0), count)
    return decode(packed_pooled), None
