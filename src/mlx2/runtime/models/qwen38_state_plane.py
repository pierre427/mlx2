"""Qwen3.8-owned exact state movement and compact replay admission."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from ..state_plane_operations import (
    StatePlaneContract,
    StatePlaneOperationRefused,
)
from .gdn_kernel_contract import FusedGdnContract
from .qwen_gdn_contracts import QWEN38_GDN_CONTRACT

_EXACT_MOVEMENT = frozenset(
    {"checkpoint", "private_successor", "rollback", "promotion"}
)


@dataclass(frozen=True)
class CompactGdnReplayAuthorization:
    provider_revision: str
    contract_identity: str
    steps: int
    tape_bytes: int
    snapshot_bytes_avoided: int
    fidelity: str = "exact"


@dataclass(frozen=True)
class Qwen38CompactReplayPlan:
    """One exact verify transaction: checkpoint, tape, and final buffers."""

    authorization: CompactGdnReplayAuthorization
    checkpoint_conv: mx.array
    checkpoint_state: mx.array
    raw_qkv: mx.array
    replay_keys: mx.array
    replay_corrections: mx.array
    replay_decay: mx.array
    reconstruct: Callable[..., mx.array]
    threadgroup_y: int
    keep: int

    def _validate_prefix(self, accepted: int) -> int:
        if isinstance(accepted, bool) or not isinstance(accepted, int):
            raise TypeError("accepted prefix must be an integer")
        if not 1 <= accepted < self.authorization.steps:
            raise ValueError("accepted prefix is outside the compact replay tape")
        return accepted

    def accepted_prefix(self, accepted: int) -> list[mx.array]:
        accepted = self._validate_prefix(accepted)
        combined = mx.concatenate([self.checkpoint_conv, self.raw_qkv], axis=1)
        conv = mx.contiguous(combined[:, accepted : accepted + self.keep])
        state = self.reconstruct(
            self.checkpoint_state,
            self.replay_keys,
            self.replay_corrections,
            self.replay_decay,
            accepted,
            threadgroup_y=self.threadgroup_y,
        )
        return [conv, state]

    def cancel(self) -> list[mx.array]:
        """Return the untouched checkpoint; no tape row is published."""
        return [self.checkpoint_conv, self.checkpoint_state]

    def commit_full(self, conv: mx.array, state: mx.array) -> list[mx.array]:
        """Publish only the already-computed final pair after full acceptance."""
        return [conv, state]


class Qwen38StatePlaneOperations:
    """Own Qwen3.8 cache identity without teaching shared code model names."""

    provider_revision = "qwen38-state-plane-operations-v1"

    def contract(self, cache: Any) -> StatePlaneContract | None:
        from ..segmented_batch_cache import SegmentedBatchArraysCache
        from ..segmented_plain_kv import (
            SegmentedBatchKVCache,
            SegmentedBatchQuantizedKVCache,
        )
        from .cache import ArraysCache, KVCache, QuantizedKVCache

        if isinstance(cache, (ArraysCache, SegmentedBatchArraysCache)):
            layout = "qwen38-gdn-arrays-v1"
        elif type(cache) in (QuantizedKVCache, SegmentedBatchQuantizedKVCache):
            layout = "qwen38-quantized-attention-kv-v1"
        elif isinstance(cache, (KVCache, SegmentedBatchKVCache)):
            layout = "qwen38-full-attention-kv-v1"
        else:
            return None
        return StatePlaneContract(
            revision=self.provider_revision,
            layout=layout,
            fidelity="exact",
            operations=_EXACT_MOVEMENT,
        )

    def build_segmented_view(self, row_caches, *, note=None, shared_prefix=False):
        if shared_prefix:
            raise StatePlaneOperationRefused(
                "Qwen3.8 has no declared shared-prefix state representation"
            )
        from ..segmented_batch_cache import build_segmented_batch_cache_pair

        return build_segmented_batch_cache_pair(row_caches, note=note)

    def validate_segmented_view(self, view, row_caches) -> None:
        from ..segmented_batch_cache import SegmentedBatchArraysCache
        from ..segmented_plain_kv import (
            SegmentedBatchKVCache,
            SegmentedBatchQuantizedKVCache,
        )

        row_caches = tuple(row_caches)
        declared = (
            SegmentedBatchArraysCache,
            SegmentedBatchKVCache,
            SegmentedBatchQuantizedKVCache,
        )
        for name, group in (("target", view.target), ("draft", view.draft)):
            if not group or any(not isinstance(cache, declared) for cache in group):
                raise StatePlaneOperationRefused(
                    "Qwen3.8 segmented view contains an undeclared state plane"
                )
            expected_groups = [getattr(pair, name) for pair in row_caches]
            if any(len(row) != len(group) for row in expected_groups):
                raise StatePlaneOperationRefused(
                    f"segmented {name} layer width changed"
                )
            for layer, cache in enumerate(group):
                expected_rows = tuple(id(row[layer]) for row in expected_groups)
                if tuple(id(row) for row in cache.rows) != expected_rows:
                    raise StatePlaneOperationRefused(
                        f"segmented {name} view no longer owns the live rows"
                    )

    def authorize_compact_gdn_replay(
        self, cache: Any, gdn_contract: FusedGdnContract, steps: int
    ) -> CompactGdnReplayAuthorization:
        """Validate state ownership before replacing snapshots with a tape."""
        from .cache import ArraysCache

        if type(cache) is not ArraysCache:
            raise StatePlaneOperationRefused(
                "compact Qwen3.8 replay requires its exact ArraysCache owner"
            )
        state_contract = self.contract(cache)
        if (
            state_contract is None
            or state_contract.revision != self.provider_revision
            or state_contract.fidelity != "exact"
            or "rollback" not in state_contract.operations
        ):
            raise StatePlaneOperationRefused("exact recurrent rollback is undeclared")
        if gdn_contract != QWEN38_GDN_CONTRACT:
            raise StatePlaneOperationRefused("Qwen3.8 GDN contract identity drifted")
        refusal = gdn_contract.refusal()
        if refusal is not None:
            raise StatePlaneOperationRefused(refusal)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 17:
            raise StatePlaneOperationRefused("compact replay width must be 2..17")
        if not cache.speculating or len(cache.cache) != 2:
            raise StatePlaneOperationRefused(
                "compact replay requires an active two-plane speculative cache"
            )
        conv, state = cache.cache
        geometry = gdn_contract.geometry
        conv_dim = (
            2 * geometry.num_key_heads * geometry.key_head_dim
            + geometry.num_value_heads * geometry.value_head_dim
        )
        if (
            not isinstance(conv, mx.array)
            or tuple(conv.shape) != (1, geometry.conv_kernel - 1, conv_dim)
            or not isinstance(state, mx.array)
            or tuple(state.shape)
            != (
                1,
                geometry.num_value_heads,
                geometry.value_head_dim,
                geometry.key_head_dim,
            )
            or state.dtype not in (mx.float32, mx.float16)
        ):
            raise StatePlaneOperationRefused("live recurrent cache geometry is unsupported")
        activation_bytes = int(conv.itemsize)
        tape_steps = steps - 1
        tape_bytes = tape_steps * (
            geometry.num_key_heads * geometry.key_head_dim * activation_bytes
            + geometry.num_value_heads * geometry.value_head_dim * 4
            + geometry.num_value_heads * 4
        )
        snapshot_bytes = tape_steps * (int(state.nbytes) + int(conv.nbytes))
        if tape_bytes >= snapshot_bytes:
            raise StatePlaneOperationRefused("compact replay does not reduce live memory")
        return CompactGdnReplayAuthorization(
            provider_revision=self.provider_revision,
            contract_identity=gdn_contract.identity,
            steps=steps,
            tape_bytes=tape_bytes,
            snapshot_bytes_avoided=snapshot_bytes - tape_bytes,
        )

    def make_compact_gdn_replay_plan(
        self,
        authorization: CompactGdnReplayAuthorization,
        *,
        cache: Any,
        raw_qkv: mx.array,
        replay_keys: mx.array,
        replay_corrections: mx.array,
        replay_decay: mx.array,
        reconstruct: Callable[..., mx.array],
        threadgroup_y: int,
    ) -> Qwen38CompactReplayPlan:
        if authorization.provider_revision != self.provider_revision:
            raise StatePlaneOperationRefused("compact replay authorization is stale")
        if authorization.contract_identity != QWEN38_GDN_CONTRACT.identity:
            raise StatePlaneOperationRefused("compact replay authorization drifted")
        if int(raw_qkv.shape[1]) != authorization.steps:
            raise StatePlaneOperationRefused("compact replay input width drifted")
        tape_steps = authorization.steps - 1
        if any(
            int(value.shape[1]) != tape_steps
            for value in (replay_keys, replay_corrections, replay_decay)
        ):
            raise StatePlaneOperationRefused("compact replay tape width drifted")
        if not callable(reconstruct):
            raise StatePlaneOperationRefused("compact replay reconstruction is missing")
        return Qwen38CompactReplayPlan(
            authorization=authorization,
            checkpoint_conv=cache[0],
            checkpoint_state=cache[1],
            raw_qkv=raw_qkv,
            replay_keys=replay_keys,
            replay_corrections=replay_corrections,
            replay_decay=replay_decay,
            reconstruct=reconstruct,
            threadgroup_y=int(threadgroup_y),
            keep=QWEN38_GDN_CONTRACT.geometry.conv_kernel - 1,
        )


QWEN38_STATE_PLANE_OPERATIONS = Qwen38StatePlaneOperations()
