"""Qwen4 provider for the generalized state-plane movement contract."""

from __future__ import annotations

import time
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..state_plane_operations import (
    StatePlaneContract,
    StatePlaneMaterialization,
    StatePlaneBinding,
    StatePlaneOperationRefused,
    StatePlaneReservation,
    SharedPrefixMaterialization,
    SharedPrefixPlan,
)

_EXACT_MOVEMENT = frozenset(
    {"checkpoint", "private_successor", "rollback", "promotion"}
)


@dataclass(frozen=True)
class _Qwen4SharedPrefixPayload:
    row_caches: tuple[Any, ...]
    layers: tuple[int, ...]
    prefix_id: str


class Qwen4StatePlaneOperations:
    """Declare the exact QSA/recurrent state layouts used by Qwen4 promotion.

    Imports are local to avoid coupling the generalized protocol to model
    classes and to avoid a cycle with segmented cache construction.
    """

    provider_revision = "qwen4-state-plane-operations-v1"

    def bind_loaded_state(self, model, row_caches) -> StatePlaneBinding:
        """Bind cache state to this exact loaded Qwen4 instance and numerics."""
        row_caches = tuple(row_caches)
        if not row_caches:
            raise StatePlaneOperationRefused("Qwen4 binding requires cache rows")
        cache_keys = []
        for pair in row_caches:
            groups = []
            for name in ("target", "draft"):
                caches = tuple(getattr(pair, name, ()))
                if not caches:
                    raise StatePlaneOperationRefused(
                        f"Qwen4 binding requires non-empty {name} caches"
                    )
                contracts = tuple(self.contract(cache) for cache in caches)
                if any(contract is None for contract in contracts):
                    raise StatePlaneOperationRefused(
                        f"Qwen4 {name} cache lacks an exact state-plane contract"
                    )
                groups.append(
                    tuple(
                        (contract.revision, contract.layout, contract.fidelity)
                        for contract in contracts
                    )
                )
            cache_keys.append(tuple(groups))
        if any(key != cache_keys[0] for key in cache_keys[1:]):
            raise StatePlaneOperationRefused("Qwen4 cache row layouts differ")
        args = getattr(model, "args", None)
        language_model = getattr(model, "language_model", None)
        return StatePlaneBinding(
            provider_revision=self.provider_revision,
            artifact_key=(
                type(model).__module__,
                type(model).__qualname__,
                id(model),
                id(language_model),
                id(getattr(model, "mtp", None)),
            ),
            cache_key=cache_keys[0],
            numerics_key=(
                repr(args),
                repr(getattr(model, "norm_convention", None)),
                repr(getattr(model, "model_type", None)),
            ),
        )

    def plan_shared_prefix(
        self,
        row_caches,
        *,
        binding,
        prefix_id,
        remaining_tokens,
        note=None,
    ) -> SharedPrefixPlan:
        """Validate policy and geometry without changing any live cache."""
        from ..segmented_self_mtp import (
            qsa_private_delta_enabled,
            shared_qsa_suffix_admission,
        )
        from .qwen4_exp import QSAKVCache

        row_caches = tuple(row_caches)
        target_groups = tuple(tuple(pair.target) for pair in row_caches)
        qsa_layers = tuple(
            layer
            for layer, rows in enumerate(zip(*target_groups))
            if all(type(row) is QSAKVCache for row in rows)
        )
        contexts = tuple(
            int(target_groups[row][layer].offset)
            for layer in qsa_layers
            for row in (0,)
        )
        base_tokens = min(contexts, default=0)
        admitted, _reason, cutoff = shared_qsa_suffix_admission(
            base_tokens=base_tokens, remaining_tokens=int(remaining_tokens)
        )
        admitted = bool(admitted and qsa_private_delta_enabled() and qsa_layers)
        return SharedPrefixPlan(
            binding=binding,
            payload=_Qwen4SharedPrefixPayload(
                row_caches=row_caches,
                layers=qsa_layers,
                prefix_id=str(prefix_id),
            ),
            admitted=admitted,
            telemetry=(
                ("shared_qsa_policy_checks", 1),
                (
                    "shared_qsa_policy_admitted"
                    if admitted
                    else "shared_qsa_policy_declined",
                    1,
                ),
                ("shared_qsa_policy_context_tokens_cumulative", base_tokens),
                (
                    "shared_qsa_policy_remaining_tokens_cumulative",
                    int(remaining_tokens),
                ),
                ("shared_qsa_policy_cutoff_tokens_cumulative", cutoff),
            ),
        )

    def materialize_shared_prefix(self, plan) -> SharedPrefixMaterialization:
        """Build replacements without publishing them into request state."""
        from ..qsa_shared_suffix import split_attested_qsa_rows

        if not isinstance(plan, SharedPrefixPlan) or not isinstance(
            plan.payload, _Qwen4SharedPrefixPayload
        ):
            raise StatePlaneOperationRefused("invalid Qwen4 shared-prefix plan")
        payload = plan.payload
        targets = [list(pair.target) for pair in payload.row_caches]
        if plan.admitted:
            digest = hashlib.sha256(payload.prefix_id.encode()).hexdigest()[:16]
            for layer in payload.layers:
                rows = split_attested_qsa_rows(
                    [target[layer] for target in targets],
                    layout_id=f"qwen4-shared-qsa-v1:{digest}:{layer}",
                )
                for target, row in zip(targets, rows):
                    target[layer] = row
        telemetry = ()
        if plan.admitted and payload.layers:
            shared = [targets[row][payload.layers[0]] for row in range(len(targets))]
            telemetry = (
                ("shared_qsa_rows", len(shared) * len(payload.layers)),
                ("shared_qsa_base_bytes", int(shared[0].base.nbytes)),
                (
                    "shared_qsa_private_bytes",
                    sum(
                        int(targets[row][layer].private_nbytes)
                        for row in range(len(targets))
                        for layer in payload.layers
                    ),
                ),
            )
        return SharedPrefixMaterialization(
            target_rows=tuple(tuple(target) for target in targets),
            telemetry=telemetry,
        )

    @staticmethod
    def is_shared_prefix_cache(cache) -> bool:
        from ..qsa_shared_suffix import SharedSuffixQSAKVCache

        return isinstance(cache, SharedSuffixQSAKVCache)

    def materialize_private_prefix(self, cache):
        if not self.is_shared_prefix_cache(cache):
            return cache
        materialized, _receipt = cache.materialize_to_qsa()
        return materialized

    @staticmethod
    def _offset(cache: Any) -> int:
        value = getattr(cache, "offset", None)
        if isinstance(value, int):
            return value
        size = getattr(cache, "size", None)
        if callable(size):
            result = size()
            if isinstance(result, int):
                return result
        raise StatePlaneOperationRefused(
            f"{type(cache).__name__} has no host-readable scalar offset"
        )

    def contract(self, cache: Any) -> StatePlaneContract | None:
        from ..segmented_batch_cache import (
            SegmentedBatchArraysCache,
            SegmentedBatchQSAKVCache,
        )
        from ..segmented_plain_kv import SegmentedBatchQuantizedKVCache
        from ..qsa_shared_suffix import SharedSuffixQSAKVCache
        from .cache import ArraysCache, QuantizedKVCache
        from .qwen4_exp import BatchQSAKVCache, QSAKVCache, Qwen4ArraysCache

        if isinstance(
            cache,
            (
                QSAKVCache,
                BatchQSAKVCache,
                SegmentedBatchQSAKVCache,
                SharedSuffixQSAKVCache,
            ),
        ):
            layout = "qwen4-qsa-raw-key-ledger-v1"
        elif isinstance(
            cache, (ArraysCache, Qwen4ArraysCache, SegmentedBatchArraysCache)
        ):
            layout = "qwen4-recurrent-arrays-v1"
        elif type(cache) in (QuantizedKVCache, SegmentedBatchQuantizedKVCache):
            # Movement is exact with respect to the already-quantized target
            # state. The approximation was selected before this provider owns
            # the row and must not disable segmented execution.
            layout = "qwen4-quantized-kv-v1"
        else:
            return None
        return StatePlaneContract(
            revision=self.provider_revision,
            layout=layout,
            fidelity="exact",
            operations=_EXACT_MOVEMENT,
        )

    def build_segmented_view(self, row_caches, *, note=None, shared_prefix=False):
        from ..segmented_batch_cache import build_segmented_batch_cache_pair

        return build_segmented_batch_cache_pair(
            row_caches, note=note, shared_qsa_prefix=shared_prefix
        )

    def validate_segmented_view(self, view, row_caches) -> None:
        from ..segmented_batch_cache import (
            SegmentedBatchArraysCache,
            SegmentedBatchQSAKVCache,
        )
        from ..segmented_plain_kv import SegmentedBatchQuantizedKVCache

        row_caches = tuple(row_caches)
        for name, group in (("target", view.target), ("draft", view.draft)):
            if not group or any(
                not isinstance(
                    cache,
                    (
                        SegmentedBatchQSAKVCache,
                        SegmentedBatchArraysCache,
                        SegmentedBatchQuantizedKVCache,
                    ),
                )
                for cache in group
            ):
                raise StatePlaneOperationRefused(
                    "Qwen4 promotion supports only declared segmented state planes"
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

    def _reserve_qsa_rows(
        self, rows, tail, *, shared_prefix=False, timing=None
    ) -> StatePlaneReservation:
        from .qwen4_exp import BatchQSAKVCache, QSAKVCache

        started = time.perf_counter_ns() if timing is not None else 0
        rows = tuple(rows)
        starts = [self._offset(row) for row in rows]
        if len(set(starts)) != 1:
            raise StatePlaneOperationRefused(
                "suffix-only promotion requires equal starting QSA offsets"
            )
        start = starts[0]
        if start <= 0 or tail < 0:
            raise StatePlaneOperationRefused("invalid QSA base/tail geometry")
        populated = next((row for row in rows if row.keys is not None), None)
        if populated is None:
            raise StatePlaneOperationRefused("cannot promote an empty QSA cache")
        for row in rows:
            if type(row) is not QSAKVCache or row.keys is None or row.values is None:
                raise StatePlaneOperationRefused(
                    "physical promotion requires plain populated QSAKVCache rows"
                )
            if row.index_keys is None or row.index_keys.shape[1] < start:
                raise StatePlaneOperationRefused(
                    "a QSA raw-key ledger is shorter than its base offset"
                )
        validated = time.perf_counter_ns() if timing is not None else 0
        live_width = start + tail
        step = int(BatchQSAKVCache.step)
        slabs = max(1, (tail + step - 1) // step)
        kv_width = start + slabs * step
        index = rows[0].index_keys
        batch = BatchQSAKVCache([0] * len(rows))
        if shared_prefix:
            if any(row is not rows[0] for row in rows[1:]):
                raise StatePlaneOperationRefused(
                    "shared-prefix reservation requires one canonical source row"
                )
            batch.keys = mx.concatenate(
                [
                    mx.broadcast_to(
                        populated.keys[..., :start, :],
                        (
                            len(rows),
                            populated.keys.shape[1],
                            start,
                            populated.keys.shape[3],
                        ),
                    ),
                    mx.zeros(
                        (
                            len(rows),
                            populated.keys.shape[1],
                            kv_width - start,
                            populated.keys.shape[3],
                        ),
                        dtype=populated.keys.dtype,
                    ),
                ],
                axis=2,
            )
            batch.values = mx.concatenate(
                [
                    mx.broadcast_to(
                        populated.values[..., :start, :],
                        (
                            len(rows),
                            populated.values.shape[1],
                            start,
                            populated.values.shape[3],
                        ),
                    ),
                    mx.zeros(
                        (
                            len(rows),
                            populated.values.shape[1],
                            kv_width - start,
                            populated.values.shape[3],
                        ),
                        dtype=populated.values.dtype,
                    ),
                ],
                axis=2,
            )
            batch.index_keys = mx.concatenate(
                [
                    mx.broadcast_to(
                        index[:, :start], (len(rows), start, index.shape[2])
                    ),
                    mx.zeros(
                        (len(rows), live_width - start, index.shape[2]),
                        dtype=index.dtype,
                    ),
                ],
                axis=1,
            )
        else:
            batch.keys = mx.zeros(
                (len(rows), populated.keys.shape[1], kv_width, populated.keys.shape[3]),
                dtype=populated.keys.dtype,
            )
            batch.values = mx.zeros(
                (
                    len(rows),
                    populated.values.shape[1],
                    kv_width,
                    populated.values.shape[3],
                ),
                dtype=populated.values.dtype,
            )
            batch.index_keys = mx.zeros(
                (len(rows), live_width, index.shape[2]), dtype=index.dtype
            )
        allocated = time.perf_counter_ns() if timing is not None else 0
        if not shared_prefix:
            for i, row in enumerate(rows):
                batch.keys[i : i + 1, :, :start, :] = row.keys[..., :start, :]
                batch.values[i : i + 1, :, :start, :] = row.values[..., :start, :]
                batch.index_keys[i : i + 1, :start] = row.index_keys[:, :start]
        copied = time.perf_counter_ns() if timing is not None else 0
        batch.offset = mx.array(starts)
        batch.left_padding = mx.zeros((len(rows),), dtype=mx.int32)
        batch._idx = start
        if timing is not None:
            finished = time.perf_counter_ns()
            timing["validation_ns"] += validated - started
            timing["allocation_graph_ns"] += allocated - validated
            timing["copy_graph_ns"] += copied - allocated
            timing["metadata_graph_ns"] += finished - copied
            timing["reserve_total_ns"] += finished - started
        values = (batch.keys, batch.values, batch.index_keys)
        reserved = sum(int(value.nbytes) for value in values)
        return StatePlaneReservation(batch, start, reserved, values)

    def reserve_promotion_layer(
        self, cache, tail, *, rows=None, shared_prefix=False, timing=None
    ) -> StatePlaneReservation:
        from ..segmented_batch_cache import SegmentedBatchQSAKVCache
        from .qwen4_exp import QSAKVCache

        if isinstance(cache, SegmentedBatchQSAKVCache):
            source_rows = cache.rows
            if rows is not None and int(rows) != len(source_rows):
                raise StatePlaneOperationRefused("reservation row width changed")
            if shared_prefix:
                source_rows = [source_rows[0]] * len(source_rows)
            return self._reserve_qsa_rows(
                source_rows, tail, shared_prefix=shared_prefix, timing=timing
            )
        if isinstance(cache, QSAKVCache):
            if rows is None:
                rows = 1
            source_rows = [cache] * int(rows)
            return self._reserve_qsa_rows(
                source_rows, tail, shared_prefix=shared_prefix, timing=timing
            )
        contract = self.contract(cache)
        if contract is None or contract.layout != "qwen4-recurrent-arrays-v1":
            raise StatePlaneOperationRefused(
                f"{type(cache).__name__} cannot be reserved for Qwen4 promotion"
            )
        return StatePlaneReservation(None, None)

    def validate_staged_layer(self, source, candidate, base) -> None:
        from ..segmented_batch_cache import SegmentedBatchQSAKVCache

        if isinstance(source, SegmentedBatchQSAKVCache):
            if candidate is None or base is None:
                raise StatePlaneOperationRefused(
                    "staged QSA layer map differs from admitted state"
                )
            if any(self._offset(row) != base for row in source.rows):
                raise StatePlaneOperationRefused(
                    "admitted QSA offset differs from staged base"
                )
            return
        if candidate is not None or base is not None:
            raise StatePlaneOperationRefused(
                "staged recurrent layer map differs from admitted state"
            )

    @staticmethod
    def _unwrap_recurrent(cache):
        row_types = {type(row) for row in cache.rows}
        if len(row_types) != 1:
            raise StatePlaneOperationRefused(
                "segmented recurrent rows changed concrete cache type"
            )
        result = row_types.pop()(len(cache.cache))
        result._adopt_empty_fill(cache.rows)
        result.cache = list(cache.cache)
        result.left_padding = cache.left_padding
        result.lengths = cache.lengths
        result._host_left_padding = cache._host_left_padding
        result._host_lengths = cache._host_lengths
        result._checkpoints = list(cache._checkpoints)
        result._rollbacks = deque()
        result.speculating = False
        return result

    def materialize_promoted_layer(
        self, source, candidate, base, *, advance, target
    ) -> StatePlaneMaterialization:
        from ..segmented_batch_cache import SegmentedBatchQSAKVCache
        from .qwen4_exp import _qsa_merge_summaries

        if isinstance(source, SegmentedBatchQSAKVCache):
            if candidate is None or base is None:
                raise StatePlaneOperationRefused(
                    "QSA destination reservation is absent"
                )
            stop = base + advance if target else base
            if target:
                for i, row in enumerate(source.rows):
                    if self._offset(row) != stop:
                        raise StatePlaneOperationRefused(
                            "QSA target offset disagrees with committed advance"
                        )
                    candidate.keys[i : i + 1, :, base:stop, :] = row.keys[
                        ..., base:stop, :
                    ]
                    candidate.values[i : i + 1, :, base:stop, :] = row.values[
                        ..., base:stop, :
                    ]
                    candidate.index_keys[i : i + 1, base:stop] = row.index_keys[
                        :, base:stop
                    ]
                candidate._idx = stop
                candidate.index_keys = mx.contiguous(candidate.index_keys[:, :stop])
                candidate.offset = mx.array([stop] * len(source.rows))
                candidate.left_padding = mx.zeros((len(source.rows),), dtype=mx.int32)
            elif any(self._offset(row) != base for row in source.rows):
                raise StatePlaneOperationRefused(
                    "draft QSA changed across the first committed cycle"
                )
            pooled, identity = _qsa_merge_summaries(
                source.rows, [stop] * len(source.rows)
            )
            candidate._qsa_pooled_keys = pooled
            candidate._qsa_pooled_ratio = (
                None if pooled is None else int(source.rows[0]._qsa_pooled_ratio)
            )
            candidate._qsa_summary_identity = identity
            patched = 0
            if target:
                patched = sum(
                    int(row.keys[..., base:stop, :].nbytes)
                    + int(row.values[..., base:stop, :].nbytes)
                    + int(row.index_keys[:, base:stop].nbytes)
                    for row in source.rows
                )
            values = tuple(
                value
                for value in (candidate.keys, candidate.values, candidate.index_keys)
                if value is not None
            )
            return StatePlaneMaterialization(candidate, patched, 0, values)
        unwrapped = self._unwrap_recurrent(source)
        reused = sum(value is not None for value in source.cache)
        values = tuple(value for value in unwrapped.cache if value is not None)
        return StatePlaneMaterialization(unwrapped, 0, reused, values)


QWEN4_STATE_PLANE_OPERATIONS = Qwen4StatePlaneOperations()
