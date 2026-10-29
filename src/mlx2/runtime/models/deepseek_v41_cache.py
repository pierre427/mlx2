# SPDX-License-Identifier: MIT
"""Exact B1 V4.1 frontier state; shared pools live only at their source layers.

APCv2 owns copies of the local band, partial group, source KV/index pools and
compressed n-gram tail. No process-global sharing or unrecorded hash history.
"""

from ..cache_operations import (
    CacheOperationsContract,
    CachePlaneDeclaration,
    CheckpointPublicationCondition,
)
from .deepseek_v4_cache import (
    DeepSeekV4LayerCache,
    DeepSeekV4Snapshot,
    _backend,
    _copy,
    _parse_meta_int,
)

STATE_REVISION = "mlx2.deepseek-v41-cache.v1"
CACHE_LAYOUT = "deepseek-v41-shared-csa2-engram-b1-v1"


class DeepSeekV41LayerCache(DeepSeekV4LayerCache):
    cache_operations_contract = CacheOperationsContract(
        family="deepseek_v41",
        topology="checkpointed_hybrid",
        state_revision=STATE_REVISION,
        cache_layout=CACHE_LAYOUT,
        numerics="fp32-storage-bf16-fp8-fp4-values",
        # APCv2 freezes this compound frontier as one atomic primary plane;
        # state includes local KV, partial groups, source pools/index and tail.
        planes=(
            CachePlaneDeclaration(
                "attention_ring",
                ("state",),
                (
                    "identity",
                    "ratio",
                    "window_size",
                    "head_dim",
                    "index_head_dim",
                    "source",
                    "history",
                    "compressed_vocab",
                ),
            ),
        ),
        exact_prefix=True,
        restores_at_checkpoints=True,
        rollback=None,
        publication_conditions=(
            CheckpointPublicationCondition(
                "pending", "truthy", "V4.1 forward or source pool is incomplete"
            ),
            CheckpointPublicationCondition(
                "ast_private_profile", "truthy",
                "V4.1 AST numerical profile is private and lacks exact publication admission",
            ),
        ),
        segmented_publication_conditions=(
            CheckpointPublicationCondition(
                "pending", "truthy", "V4.1 forward or source pool is incomplete"
            ),
            CheckpointPublicationCondition(
                "ast_private_profile", "truthy",
                "V4.1 AST numerical profile is private and lacks exact publication admission",
            ),
        ),
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
        source=False,
        history=False,
        compressed_vocab=99092,
    ):
        super().__init__(
            0,
            xp=xp,
            window_size=window_size,
            head_dim=head_dim,
            index_head_dim=index_head_dim,
            identity=identity,
        )
        if type(ratio) is not int or ratio not in (0, 1, 2) or (source and not ratio):
            raise ValueError("invalid V4.1 source geometry")
        self.ratio, self.source, self.history = ratio, bool(source), bool(history)
        self.compressed_vocab = compressed_vocab
        self.buf_kv = self.buf_gate = self._pooled = self._index_keys = None
        self._pool_count = self._capacity = 0
        self.hash_tail = xp.full((3,), -1, dtype=xp.int32) if history else None
        self._in_forward = False
        # A different native-M arithmetic profile may be executed privately
        # while its numerical gate is unresolved. This mark survives ordinary
        # continuation and descriptor cloning; it cannot donate an APC prefix.
        self.ast_private_profile = None
        # Source position-grid provenance (host-only, not serialized): the
        # grid route stamps these; tainted or off-grid state cannot donate.
        self.source_partition_grid = None
        self.source_partition_tainted = False

    @property
    def pooled(self):
        return None if self._pooled is None else self._pooled[:, : self._pool_count]

    @property
    def index_keys(self):
        return (
            None
            if self._index_keys is None
            else self._index_keys[:, : self._pool_count]
        )

    def append_pools(self, pooled, index_keys):
        if (
            not self.source
            or pooled.shape[0] != 1
            or index_keys.shape != (1, pooled.shape[1], self.index_head_dim)
            or pooled.shape[2] != self.head_dim
        ):
            raise ValueError("invalid V4.1 source append geometry")
        count = self._pool_count + pooled.shape[1]
        if count != self.offset // self.ratio:
            raise ValueError("V4.1 source append does not complete the exact frontier")
        if self._pooled is None or count > self._capacity:
            capacity = ((count + 255) // 256) * 256
            kv = self.xp.zeros((1, capacity, self.head_dim), dtype=self.xp.float32)
            keys = self.xp.zeros(
                (1, capacity, self.index_head_dim), dtype=self.xp.float32
            )
            if self._pool_count:
                kv[:, : self._pool_count] = self.pooled
                keys[:, : self._pool_count] = self.index_keys
            self._pooled, self._index_keys, self._capacity = kv, keys, capacity
        if pooled.shape[1]:
            self._pooled[:, self._pool_count : count] = pooled
            self._index_keys[:, self._pool_count : count] = index_keys
        self._pool_count = count

    @property
    def state(self):
        return (
            self.keys,
            self.buf_kv,
            self.buf_gate,
            self.pooled,
            self.index_keys,
            self.hash_tail,
        )

    @state.setter
    def state(self, value):
        if len(value) != 6:
            raise ValueError("invalid V4.1 cache array inventory")
        values = _copy(self.xp, value)
        pooled, index_keys = values[3:5]
        # Capacity slack is internal only. Restored arrays must agree on
        # their live extent before the accessors slice at _pool_count.
        if (pooled is None) != (index_keys is None):
            raise ValueError("incomplete V4.1 source plane inventory")
        count = 0
        if pooled is not None:
            if len(pooled.shape) != 3:
                raise ValueError("invalid V4.1 source plane geometry")
            count = pooled.shape[1]
            if (pooled.shape != (1, count, self.head_dim)
                    or index_keys.shape != (1, count, self.index_head_dim)):
                raise ValueError("V4.1 source plane extent mismatch")
        (
            self.keys,
            self.buf_kv,
            self.buf_gate,
            self._pooled,
            self._index_keys,
            self.hash_tail,
        ) = values
        self._pool_count = self._capacity = count

    @property
    def meta_state(self):
        return (
            STATE_REVISION if self.ast_private_profile is None else
            STATE_REVISION + ":private:" + self.ast_private_profile,
            self.identity,
            str(self.ratio),
            str(self.window_size),
            str(self.head_dim),
            str(self.index_head_dim),
            str(int(self.source)),
            str(int(self.history)),
            str(self.compressed_vocab),
            str(self.offset),
        )

    @meta_state.setter
    def meta_state(self, value):
        if len(value) != 10 or tuple(value[:9]) != self.meta_state[:9]:
            raise ValueError("V4.1 cache revision, identity or topology mismatch")
        self.offset = _parse_meta_int(value[9])

    @classmethod
    def from_state(cls, state, meta_state, *, xp=None):
        if len(meta_state) != 10 or meta_state[0] != STATE_REVISION:
            raise ValueError("invalid V4.1 cache metadata revision")
        xp = xp or _backend(state)
        if xp is None:
            import mlx.core as xp
        ratio, window, head, index, source, history, vocab, offset = map(
            _parse_meta_int, meta_state[2:]
        )
        if source not in (0, 1) or history not in (0, 1):
            raise ValueError("invalid V4.1 cache ownership flags")
        result = cls(
            ratio,
            xp=xp,
            window_size=window,
            head_dim=head,
            index_head_dim=index,
            identity=meta_state[1],
            source=source,
            history=history,
            compressed_vocab=vocab,
        )
        result.offset = offset
        result.state = state
        result.validate_state()
        if result.pending:
            raise ValueError("incomplete V4.1 cache frontier")
        return result

    @property
    def pending(self):
        if self._closed or self._in_forward:
            return True
        if self.source:
            count = self.offset // self.ratio
            return any(
                (0 if v is None else v.shape[1]) != count
                for v in (self.pooled, self.index_keys)
            )
        return False

    def validate_storage(self):
        for v in self.state[:5]:
            if v is not None and v.dtype != self.xp.float32:
                raise ValueError("V4.1 cache values require physical FP32")
        if self.hash_tail is not None and self.hash_tail.dtype != self.xp.int32:
            raise ValueError("V4.1 n-gram state requires int32")

    def validate_state(self):
        if self.ast_private_profile is not None and (
            type(self.ast_private_profile) is not str or not self.ast_private_profile
        ):
            raise ValueError("invalid V4.1 private AST numerical profile")
        self.validate_storage()
        expected = min(self.offset, self.window_size)
        if self.keys is None:
            if expected:
                raise ValueError("V4.1 local KV missing")
        elif self.keys.shape != (1, 1, expected, self.head_dim):
            raise ValueError("V4.1 local KV frontier mismatch")
        for value, dim in (
            (self.pooled, self.head_dim),
            (self.index_keys, self.index_head_dim),
        ):
            count = self.offset // self.ratio if self.source else 0
            if value is None:
                if count:
                    raise ValueError("V4.1 source pool missing")
            elif not self.source or value.shape != (1, count, dim):
                raise ValueError("V4.1 source pool topology/frontier mismatch")
        for value in (self.buf_kv, self.buf_gate):
            rem = self.offset % self.ratio if self.source else 0
            if value is None:
                if rem:
                    raise ValueError("V4.1 partial group missing")
            elif not self.source or value.shape != (1, rem, self.head_dim):
                raise ValueError("V4.1 partial group mismatch")
        if self.history:
            if self.hash_tail is None or self.hash_tail.shape != (3,):
                raise ValueError("V4.1 n-gram tail missing")
            # Three compressed IDs newest first; -1 marks an absent position.
            values = self.hash_tail.tolist()
            if any(
                type(i) is not int or i < -1 or i >= self.compressed_vocab
                for i in values
            ):
                raise ValueError("invalid V4.1 compressed n-gram IDs")
            if any(i != -1 for i in values[min(self.offset, 3) :]):
                raise ValueError("V4.1 n-gram history precedes sequence start")
        elif self.hash_tail is not None:
            raise ValueError("unexpected V4.1 n-gram owner")

    def restore(self, snapshot):
        if self._closed:
            raise ValueError("cache row is closed")
        restored = type(self).from_state(snapshot.arrays, snapshot.metadata, xp=self.xp)
        if restored.meta_state[:9] != self.meta_state[:9]:
            raise ValueError("V4.1 snapshot identity/topology mismatch")
        self.offset = restored.offset
        self.state = restored.state
        self._in_forward = False
        # A snapshot taken from tainted grid state stays tainted on rollback.
        self.source_partition_tainted = bool(
            getattr(snapshot, "source_partition_tainted", False))

    def snapshot(self):
        self.validate_state()
        self.cache_operations_contract.validate_publication(self)
        result = DeepSeekV4Snapshot(self.meta_state, _copy(self.xp, self.state))
        if self.source_partition_tainted:
            object.__setattr__(result, "source_partition_tainted", True)
        return result

    def checkpoint_publication_refusal(self):
        """Host-only exact-state guard for APCv2 and generic checkpoints."""
        if self.pending:
            return "V4.1 forward or source pool is incomplete"
        if self.ast_private_profile is not None:
            return "V4.1 AST numerical profile is private and lacks exact publication admission"
        if self.source_partition_grid is not None:
            if self.source_partition_tainted:
                return "V4.1 Source grid state followed an off-grid forward"
            if self.offset % self.source_partition_grid:
                return "V4.1 Source grid state is not on a grid boundary"
        return None

    @property
    def nbytes(self):
        return sum(
            0 if x is None else x.nbytes
            for x in (
                self.keys,
                self.buf_kv,
                self.buf_gate,
                self._pooled,
                self._index_keys,
                self.hash_tail,
            )
        )

    def close(self):
        self.keys = self.buf_kv = self.buf_gate = self._pooled = self._index_keys = (
            self.hash_tail
        ) = None
        self._pool_count = self._capacity = 0
        self.offset = 0
        self._closed = True
