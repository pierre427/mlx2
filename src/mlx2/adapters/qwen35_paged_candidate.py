# SPDX-License-Identifier: MIT
# Model math adapted from existing lab sources; provenance/qwen35-paged-candidate-2026-10-04.json.
"""Default-off hybrid Qwen3.5/3.8 packed Q1 math over private native KV.

CPU-safe import. Runtime dependencies load only during an explicit invocation.
Publication/sampling belong to the caller's revision-bound hybrid state owner.
"""
from __future__ import annotations
from collections import deque
import copy
from dataclasses import dataclass
from functools import lru_cache
import os
import time
from types import SimpleNamespace
from typing import Any, Callable


@dataclass(frozen=True)
class HybridLayerMap:
    full_attention: tuple[int, ...]
    recurrent: tuple[int, ...]


def hybrid_layer_map(trunk: Any, args: Any) -> HybridLayerMap:
    """Admit only the two dense, known hybrid geometries, without tensor imports."""
    expected = {
        32: (2560, 16, 4, 32),
        64: (5120, 24, 4, 48),
    }.get(args.num_hidden_layers)
    actual = (args.hidden_size, args.num_attention_heads,
              args.num_key_value_heads, args.linear_num_value_heads)
    if (expected is None or actual != expected or args.num_experts or
            args.full_attention_interval != 4 or args.head_dim != 256 or
            args.linear_num_key_heads != 16 or args.linear_key_head_dim != 128 or
            args.linear_value_head_dim != 128 or
            len(trunk.layers) != args.num_hidden_layers or
            getattr(trunk, 'pipeline_size', 1) != 1 or
            getattr(trunk, 'pipeline_rank', 0) != 0):
        raise ValueError('unsupported dense hybrid topology')
    full = tuple(i for i in range(len(trunk.layers)) if (i + 1) % 4 == 0)
    recurrent = tuple(i for i in range(len(trunk.layers)) if i not in full)
    if any(bool(layer.is_linear) != (i in recurrent)
           for i, layer in enumerate(trunk.layers)):
        raise ValueError('hybrid layer order differs from full-attention mapping')
    return HybridLayerMap(full, recurrent)


def _copy_host_containers(value: Any, memo: dict[int, Any]) -> Any:
    """Copy host containers while sharing immutable tensor leaves explicitly."""
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        result = {}; memo[id(value)] = result
        result.update((key, _copy_host_containers(item, memo)) for key, item in value.items())
        return result
    if isinstance(value, list):
        result = []; memo[id(value)] = result
        result.extend(_copy_host_containers(item, memo) for item in value)
        return result
    if isinstance(value, tuple):
        result = tuple(_copy_host_containers(item, memo) for item in value)
        memo[id(value)] = result; return result
    if isinstance(value, deque):
        result = deque(maxlen=value.maxlen); memo[id(value)] = result
        result.extend(_copy_host_containers(item, memo) for item in value)
        return result
    return value


def clone_recurrent_caches(caches: tuple[Any, ...]) -> tuple[Any, ...]:
    """Private cache containers; MLX array inputs stay immutable shared roots."""
    if (type(caches) is not tuple or not caches or
            len({id(c) for c in caches}) != len(caches)):
        raise ValueError('distinct recurrent cache tuple required')
    result = []
    for cache in caches:
        if (not hasattr(cache, '__dict__') or len(cache.cache) != 2 or
                bool(getattr(cache, 'speculating', False)) or
                getattr(cache, 'left_padding', None) is not None or
                getattr(cache, 'lengths', None) is not None or
                any(value is None or isinstance(value, dict) for value in cache.cache)):
            raise ValueError('initialized ordinary B1 recurrent cache required')
        successor = copy.copy(cache)
        successor.__dict__ = _copy_host_containers(cache.__dict__, {})
        if successor is cache or successor.cache is cache.cache:
            raise RuntimeError('recurrent cache clone aliases its host container')
        result.append(successor)
    return tuple(result)


@dataclass(frozen=True)
class HybridPackedLane:
    token_ids: tuple[int, ...]
    layers: tuple[Any, ...]  # Full-attention order, not every decoder layer.
    recurrent_caches: tuple[Any, ...]  # Private successors, linear-layer order.


@dataclass(frozen=True)
class HybridBootstrap:
    logits: Any
    recurrent_caches: tuple[Any, ...]
    offset: int
    receipt: dict[str, Any]


@dataclass(frozen=True)
class _JoinedRecurrentBinding:
    """One private B2 compute leaf, valid only for the next exact public boundary."""
    revision: str
    lane_ids: tuple[int, int]
    owners: tuple[Any, Any]
    generations: tuple[int, int]
    offsets: tuple[int, int]
    ordinal: int
    row_leaves: tuple[tuple[Any, Any], tuple[Any, Any]]
    joined_leaves: tuple[Any, Any]


@lru_cache(maxsize=4)
def _joined_view_factory(base):
    class JoinedView(base):
        def __init__(self, rows, joined):
            self._joined_preseed = joined
            super().__init__(rows)
            self._joined_preseed = None

        def _refresh_state(self):
            # SegmentedBatchArraysCache.__init__ calls this virtual method.
            # A validated binding bypasses both mx.concatenate calls entirely.
            if self._joined_preseed is None:
                return super()._refresh_state()
            self.cache = list(self._joined_preseed)

    return JoinedView


def _runtime():
    import mlx.core as mx
    from ..runtime.models.cache import ArraysCache
    from ..runtime.models.precise_ops import gate_sigmoid
    from ..runtime.segmented_batch_cache import SegmentedBatchArraysCache
    from ..runtime.paged_attention_pack import prepare_staged_token_read
    from ..runtime import round_levers
    return SimpleNamespace(mx=mx, gate_sigmoid=gate_sigmoid,
                           batch_cache=SegmentedBatchArraysCache, array_cache_type=ArraysCache,
                           joined_batch_cache=_joined_view_factory(SegmentedBatchArraysCache),
                           prepare_read=prepare_staged_token_read, bump=round_levers.bump)


def _pack_attention_q1(attention, hidden, offsets, runtime):
    """Ordinary Q/gate split, norms, partial RoPE; preserve projection dtypes."""
    mx = runtime.mx
    batch, length, _ = hidden.shape
    if length != 1 or batch != len(offsets):
        raise ValueError('one Q1 row per lane required')
    qg, keys, values = attention.q_proj(hidden), attention.k_proj(hidden), attention.v_proj(hidden)
    queries, gate = mx.split(qg.reshape(batch, 1, attention.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(batch, 1, -1)
    queries = attention.q_norm(queries).transpose(0, 2, 1, 3)
    keys = attention.k_norm(keys.reshape(batch, 1, attention.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(batch, 1, attention.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    # Ordinary scalar-position RoPE remains authoritative for ragged offsets.
    queries = mx.concatenate([attention.rope(queries[row:row + 1], offset=offset)
                              for row, offset in enumerate(offsets)], axis=0)
    keys = mx.concatenate([attention.rope(keys[row:row + 1], offset=offset)
                           for row, offset in enumerate(offsets)], axis=0)
    queries = queries.transpose(0, 2, 1, 3).reshape(batch, attention.num_attention_heads, attention.head_dim)
    keys = keys.transpose(0, 2, 1, 3).reshape(batch, attention.num_key_value_heads, attention.head_dim)
    values = values.transpose(0, 2, 1, 3).reshape(batch, attention.num_key_value_heads, attention.head_dim)
    return queries, keys, values, gate


class Qwen35PagedCandidate:
    """One explicitly selected hybrid B2 decode over private successor state.

    No public state mutation, sampler, cache engine or serving selection here.
    The caller retains branches and this backend on every ambiguous failure.
    """
    def __init__(self, model, backend, *, q1_stripes: int = 16,
                 kv_precision: str = "same", q1_split_partition: int = 0, stock_long: bool = False,
                 stock_long_inline_metadata: bool = False,
                 q1_all_valid_recurrent: bool = False, q1_joined_recurrent: bool = False,
                 q1_direct_grouped_fence: bool = False,
                 q1_deferred: bool = False,
                 q1_deferred_writes_only: bool = False,
                 require_grouped_write: bool = True, _runtime_factory=None):
        self.model = model
        self.language_model = getattr(model, 'language_model', model)
        self.trunk = self.language_model.model
        self.args = self.language_model.args
        self.layer_map = hybrid_layer_map(self.trunk, self.args)
        if type(q1_stripes) is not int or q1_stripes not in (8, 16):
            raise ValueError('D256 hybrid Q1 requires 8 or 16 SIMD stripes')
        if type(require_grouped_write) is not bool:
            raise TypeError("require_grouped_write must be boolean")
        if kv_precision not in ("same", "float16_candidate"):
            raise ValueError("unsupported hybrid native cache precision policy")
        if type(q1_split_partition) is not int or q1_split_partition not in (0, 128, 256):
            raise ValueError("hybrid split-KV partition must be 0, 128 or 256")
        if type(stock_long) is not bool or (stock_long and q1_split_partition):
            raise ValueError('stock-long selector must be boolean and excludes old split-KV')
        self.stock_long = stock_long
        if type(stock_long_inline_metadata) is not bool or (stock_long_inline_metadata and not stock_long):
            raise ValueError('stock-long inline metadata requires explicit stock-long selection')
        self.stock_long_inline_metadata = stock_long_inline_metadata
        if type(q1_all_valid_recurrent) is not bool:
            raise TypeError('all-valid recurrent Q1 selector must be boolean')
        if type(q1_joined_recurrent) is not bool or (q1_joined_recurrent and not q1_all_valid_recurrent):
            raise ValueError('joined recurrent Q1 requires explicit all-valid selection')
        self.q1_all_valid_recurrent = q1_all_valid_recurrent
        self.q1_joined_recurrent = q1_joined_recurrent
        if type(q1_direct_grouped_fence) is not bool:
            raise TypeError('direct grouped Q1 fence selector must be boolean')
        self.q1_direct_grouped_fence = q1_direct_grouped_fence
        if type(q1_deferred) is not bool or (q1_deferred and not require_grouped_write):
            raise ValueError('deferred hybrid Q1 requires explicit grouped-write selection')
        if (type(q1_deferred_writes_only) is not bool or
                (q1_deferred_writes_only and (q1_deferred or not require_grouped_write))):
            raise ValueError('write-only hybrid Q1 deferral requires exclusive grouped-write selection')
        self.q1_deferred = q1_deferred
        self.q1_deferred_writes_only = q1_deferred_writes_only
        if backend is not None:
            backend.direct_grouped_fence = q1_direct_grouped_fence
        self._joined_bindings: tuple[_JoinedRecurrentBinding, ...] = ()
        self._joined_charge = None
        self._joined_closed = False
        self.q1_split_partition = q1_split_partition
        self.max_visible_tokens = 8192 if q1_split_partition or stock_long else 128
        self.kv_precision = kv_precision
        self.require_grouped_write = require_grouped_write
        self.q1_stripes = q1_stripes
        self.backend = backend
        self._runtime_factory = _runtime_factory or _runtime
        self._failure_roots: list[tuple[Any, ...]] = []
        self._bootstrap_failure_roots: list[tuple[Any, ...]] = []
        self.state_planes = ("kv", "gdn")
        self.native_layer_count = len(self.layer_map.full_attention)
        self.logical_layer_count = len(self.trunk.layers)
        self.bootstrap_generation = 0
        self.supports_singleton = True
        self.owns_physical_dispatch_proof = True

    def packed_lane(self, token_ids: tuple[int, ...], branch) -> HybridPackedLane:
        return HybridPackedLane(token_ids, branch.layers, branch.recurrent_caches)

    @staticmethod
    def _joined_identity(branches, ordinal):
        if len(branches) != 2:
            return None
        revisions = tuple(getattr(branch._origin, 'revision', None) for branch in branches)
        requests = tuple(branch._request for branch in branches)
        lane_ids = tuple(getattr(request, 'lane_id', None) for request in requests)
        owners = tuple(getattr(branch, '_owner', None) for branch in branches)
        generations = tuple(getattr(branch._origin, 'generation', None) for branch in branches)
        offsets = tuple(branch._origin.layers[0].offset for branch in branches)
        if (type(revisions[0]) is not str or revisions[0] != revisions[1] or
                any(getattr(request, 'revision', None) != revisions[0] for request in requests) or
                any(type(value) is not int for value in lane_ids + generations + offsets) or
                lane_ids[0] == lane_ids[1] or owners[0] is None or owners[1] is None or
                owners[0] is owners[1]):
            return None
        for branch, lane_id, generation, offset in zip(branches, lane_ids, generations, offsets):
            try:
                companions = dict(getattr(branch._origin, 'companions', ()))
            except (TypeError, ValueError):
                return None
            checkpoints = companions.get('gdn', ())
            if (type(checkpoints) is not tuple or len(checkpoints) != 1 or
                    getattr(checkpoints[0], 'revision', None) != revisions[0] or
                    getattr(checkpoints[0], 'lane_id', None) != lane_id or
                    getattr(checkpoints[0], 'generation', None) != generation or
                    getattr(checkpoints[0], 'offset', None) != offset):
                return None
        return revisions[0], lane_ids, owners, generations, offsets, ordinal

    @staticmethod
    def _binding_matches(binding, identity, rows):
        if (identity is None or binding is None or
                (binding.revision, binding.lane_ids, binding.generations,
                 binding.offsets, binding.ordinal) !=
                (identity[0], identity[1], identity[3], identity[4], identity[5]) or
                any(old is not current for old, current in zip(binding.owners, identity[2]))):
            return False
        for slot in (0, 1):
            joined = binding.joined_leaves[slot]
            if (joined.shape != (2, *rows[0][slot].shape[1:]) or
                    joined.dtype != rows[0][slot].dtype):
                return False
            for lane in (0, 1):
                row = rows[lane][slot]
                if (row is not binding.row_leaves[lane][slot] or
                        row.shape != (1, *joined.shape[1:]) or row.dtype != joined.dtype):
                    return False
        return True

    def release_joined_after_retirement(self, owners: tuple[Any, ...]) -> None:
        """Drop auxiliary roots only after every owner and native lease is retired."""
        writer = self.backend.writer
        if (type(owners) is not tuple or not owners or not all(owner.fully_retired for owner in owners) or
                writer.pending_epochs or writer.ledger.pending_count or self.backend._orphaned_reads or
                (writer.poisoned and not writer.failed_arena_torn_down)):
            raise RuntimeError('joined recurrent roots require completed owner retirement')
        if self._joined_charge is not None:
            self._joined_charge.release()
        self._joined_bindings = ()
        self._joined_charge = None
        self._joined_closed = True

    def native_dtype_preflight(self) -> str:
        dtype = getattr(self.language_model, 'compute_dtype', None)
        if dtype is None:
            embedding = self.trunk.embed_tokens
            tensor = getattr(embedding, 'scales', getattr(embedding, 'weight', None))
            dtype = getattr(tensor, 'dtype', None)
        name = str(dtype).removeprefix('mlx.core.')
        if name not in ('float16', 'bfloat16'):
            raise ValueError('hybrid native slice requires loaded float16 or bfloat16 compute')
        return name

    def _native_qkv(self, planes: tuple[Any, ...], mx):
        """Explicit candidate conversion only at native QKV/cache boundary."""
        if self.kv_precision == 'same':
            expected = getattr(mx, self.native_dtype_preflight())
            if any(plane.dtype != expected for plane in planes):
                raise ValueError('same-dtype native rows differ from loaded compute dtype')
            return planes, ()
        converted = tuple(plane.astype(mx.float16) for plane in planes)
        checks = tuple(mx.all(mx.isfinite(source)) & mx.all(mx.isfinite(target))
                       for source, target in zip(planes, converted))
        return converted, checks

    @staticmethod
    def _require_representable(checks: tuple[Any, ...]) -> None:
        if any(not bool(check.item()) for check in checks):
            raise ValueError('hybrid native QKV is nonfinite or unrepresentable in float16')

    def _check_owners(self, layers, offset, mx):
        args = self.args
        storage = self.native_dtype_preflight() if self.kv_precision == 'same' else 'float16'
        actual_storage = getattr(self.backend.writer.backend, 'storage_dtype', 'float16')
        if actual_storage != storage:
            raise ValueError('native arena storage dtype differs from hybrid precision policy')
        if type(layers) is not tuple or len(layers) != len(self.layer_map.full_attention):
            raise ValueError('one native owner per full-attention layer required')
        if len({id(owner) for owner in layers}) != len(layers):
            raise ValueError('full-attention owners alias')
        for owner in layers:
            p = owner.profile
            if (owner.writer is not self.backend.writer or owner.offset != offset or
                    p.dtype != storage or p.head_dim != 256 or
                    p.kv_heads != args.num_key_value_heads):
                raise ValueError('native hybrid owner offset/geometry mismatch')

    def forward_staged(self, lanes: tuple[HybridPackedLane, ...], branches: tuple,
                       *, permit_candidate: bool = False, reserve_scratch=None):
        if not permit_candidate:
            raise RuntimeError('hybrid native Q1 requires explicit candidate enablement')
        if self._joined_closed:
            raise RuntimeError('joined recurrent candidate was retired')
        if (type(lanes) is not tuple or type(branches) is not tuple or
                len(lanes) not in (1, 2) or len(branches) != len(lanes) or
                len({id(branch) for branch in branches}) != len(branches)):
            raise ValueError('one or two distinct private hybrid branches required')
        batch_size = len(lanes)
        stock_switch = os.environ.get('MLX2_PAGED_Q1_STOCK_REDUCTION', '0')
        if stock_switch not in ('0', '1'):
            raise ValueError('hybrid stock-reduction selector must be 0 or 1')
        stock_enabled = stock_switch == '1'
        declared_stock = getattr(self, '_serving_stock_reduction', None)
        if declared_stock is not None and (type(declared_stock) is not bool or declared_stock != stock_enabled):
            raise ValueError('live hybrid stock selector differs from admitted source-bound profile')
        singleton_switch = os.environ.get('MLX2_PAGED_Q1_STOCK_SINGLETON', '0')
        if singleton_switch not in ('0', '1') or (singleton_switch == '1' and not stock_enabled):
            raise ValueError('stock singleton requires explicit stock reduction and selector0/1')
        singleton_enabled = singleton_switch == '1'
        declared_singleton = getattr(self, '_serving_stock_singleton', None)
        if declared_singleton is not None and (type(declared_singleton) is not bool or declared_singleton != singleton_enabled):
            raise ValueError('live singleton stock selector differs from admitted profile')
        stock_reduction = stock_enabled and (batch_size == 2 or singleton_enabled)
        effective_stripes = 32 if stock_reduction else self.q1_stripes
        if (os.environ.get('MLX2_PAGED_Q1_SIMD_TILE') != '1' or
                os.environ.get('MLX2_PAGED_Q1_SIMD_STRIPES') != str(self.q1_stripes) or
                os.environ.get('MLX2_PAGED_Q1_STOCK_SDPA') == '1' or
                (self.require_grouped_write and os.environ.get('MLX2_PAGED_GROUPED_Q1_WRITE') != '1') or
                os.environ.get('MLX2_PAGED_Q1_SPLIT_KV', '0') != str(self.q1_split_partition)):
            raise ValueError('hybrid short Q1 native tile selection differs')
        if os.environ.get('MLX2_PAGED_Q1_STOCK_LONG', '0') != ('1' if self.stock_long else '0'):
            raise ValueError('hybrid stock-long flag differs from explicit candidate selector')
        inline_switch = os.environ.get('MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA', '0')
        if inline_switch not in ('0', '1') or inline_switch != ('1' if self.stock_long_inline_metadata else '0'):
            raise ValueError('hybrid stock-long inline metadata flag differs from explicit candidate selector')
        compute_dtype = self.native_dtype_preflight()
        runtime = self._runtime_factory(); mx = runtime.mx
        if self.q1_all_valid_recurrent and not isinstance(getattr(runtime, 'array_cache_type', None), type):
            raise ValueError('all-valid recurrent Q1 requires an exact ordinary ArraysCache type')
        expected_compute = getattr(mx, compute_dtype)
        offsets = []
        for lane, branch in zip(lanes, branches):
            if (type(lane) is not HybridPackedLane or type(lane.token_ids) is not tuple or
                    len(lane.token_ids) != 1 or type(lane.token_ids[0]) is not int or
                    not 0 <= lane.token_ids[0] < self.args.vocab_size or
                    getattr(branch, '_closed', True) or branch._request.proposed_rows != 1 or
                    branch._request.planes != ('kv', 'gdn') or
                    lane.layers is not branch.layers or
                    lane.recurrent_caches is not branch.recurrent_caches or
                    len(lane.recurrent_caches) != len(self.layer_map.recurrent)):
                raise ValueError('lane does not match private hybrid branch')
            offset = branch._origin.layers[0].offset
            if not 32 <= offset + 1 <= self.max_visible_tokens:
                raise ValueError('hybrid Q1 visible context exceeds bounded native profile')
            self._check_owners(lane.layers, offset, mx)
            for cache in lane.recurrent_caches:
                if ((self.q1_all_valid_recurrent and type(cache) is not runtime.array_cache_type) or
                        len(cache.cache) != 2 or bool(getattr(cache, 'speculating', False)) or
                        getattr(cache, 'lengths', None) is not None or
                        getattr(cache, 'left_padding', None) is not None or
                        (self.q1_all_valid_recurrent and
                         (getattr(cache, '_host_lengths', None) is not None or
                          getattr(cache, '_host_left_padding', None) is not None)) or
                        any(value is None or value.shape[0] != 1 for value in cache.cache)):
                    raise ValueError('private initialized B1 recurrent cache required')
            offsets.append(offset)
        if len({id(c) for lane in lanes for c in lane.recurrent_caches}) != batch_size * len(self.layer_map.recurrent):
            raise ValueError('recurrent cache containers alias across lanes')
        for ordinal, index in enumerate(self.layer_map.recurrent):
            first = lanes[0].recurrent_caches[ordinal]
            second = lanes[-1].recurrent_caches[ordinal]
            module = self.trunk.layers[index].linear_attn
            state_dtype = getattr(module, '_gdn_state_dtype', None) or mx.float32
            for slot, expected_dtype in ((0, expected_compute), (1, state_dtype)):
                left, right = first.cache[slot], second.cache[slot]
                if (left.shape[1:] != right.shape[1:] or left.dtype != right.dtype or
                        left.dtype != expected_dtype):
                    raise ValueError('recurrent lane geometry/dtype differs from model state policy')
        writer = self.backend.writer
        if getattr(self.backend, 'direct_grouped_fence', False) is not self.q1_direct_grouped_fence:
            raise ValueError('hybrid direct grouped fence selection differs')
        if writer.poisoned or writer.pending_epochs or writer.ledger.pending_count:
            raise RuntimeError('hybrid native writer is not idle')
        joined_active = self.q1_joined_recurrent and batch_size == 2
        reusable = ()
        if self.q1_joined_recurrent:
            # B1 survivors and any changed/stale cohort use the ordinary
            # constructor, including its real concatenations.
            identities = tuple(self._joined_identity(branches, ordinal)
                               for ordinal in range(len(self.layer_map.recurrent))) if joined_active else ()
            if (joined_active and len(self._joined_bindings) == len(identities) and
                    all(self._binding_matches(binding, identity,
                                              tuple(lane.recurrent_caches[ordinal] for lane in lanes))
                        for ordinal, (binding, identity) in enumerate(zip(self._joined_bindings, identities)))):
                reusable = self._joined_bindings
            else:
                self._joined_bindings = ()
            if reusable and not callable(getattr(runtime, 'joined_batch_cache', None)):
                raise ValueError('joined recurrent view constructor is unavailable')
            charge = (2 * sum(slot.nbytes for lane in lanes
                              for cache in lane.recurrent_caches for slot in cache.cache)
                      if joined_active else 0)
            if joined_active and self._joined_charge is not None and self._joined_charge.bytes < charge:
                joined_active = False
                reusable = ()
                self._joined_bindings = ()
            if joined_active and self._joined_charge is None:
                if not callable(reserve_scratch):
                    raise ValueError('joined recurrent roots require a charged reservation')
                # Two complete generations may overlap during a forward.
                reservation = reserve_scratch(charge)
                if (type(getattr(reservation, 'bytes', None)) is not int or reservation.bytes < charge or
                        not callable(getattr(reservation, 'release', None)) or
                        not callable(getattr(reservation, 'retain_failure_roots', None))):
                    self._failure_roots.append((reservation, lanes, branches))
                    raise ValueError('joined recurrent charge is insufficient')
                self._joined_charge = reservation
        arena = writer.backend
        if (getattr(arena, 'defer_staged_q1_eval', False) or getattr(arena, 'defer_staged_q1_writes', False) or
                (callable(getattr(arena, 'deferred_q1_roots', None)) and arena.deferred_q1_roots())):
            raise RuntimeError('hybrid Q1 requires idle deferred submission state')
        deferred_active = (self.q1_deferred or self.q1_deferred_writes_only) and batch_size == 2
        if deferred_active and (not callable(getattr(arena, 'begin_deferred_q1', None)) or
                not callable(getattr(arena, 'end_deferred_q1', None)) or
                not callable(getattr(arena, 'eval_submission_snapshot', None)) or
                not callable(getattr(self.backend, 'abort_deferred_q1', None)) or
                not callable(getattr(self.backend, 'retain_deferred_q1_roots', None))):
            raise ValueError('hybrid deferred Q1 lifecycle capability unavailable before native write')
        eval_before = arena.eval_submission_snapshot() if deferred_active else None
        short_q1 = all(32 <= offset + 1 <= 128 for offset in offsets)
        stock_singleton = stock_enabled and singleton_enabled and batch_size == 1 and short_q1
        if batch_size == 1 and not short_q1:
            stock_reduction = False
            effective_stripes = self.q1_stripes
        stock_long_active = self.stock_long and batch_size == 2 and max(offsets) + 1 > 1024
        if self.stock_long_inline_metadata and not stock_long_active:
            raise ValueError('inline metadata requires eligible long B2 before native write')
        long_q1 = batch_size == 2 and not short_q1 and not stock_long_active
        if stock_long_active:
            if (self.args.num_attention_heads // self.args.num_key_value_heads <= 4 or
                    os.environ.get('MLX_SDPA_BLOCKS', '0') != '0' or
                    any(getattr(getattr(owner, 'sequence', None), 'retained_start', 0) != 0
                        for lane in lanes for owner in lane.layers)):
                raise ValueError('stock-long requires GQA above four and unwindowed B2 without block overrides')
            device_info = getattr(mx, 'device_info', None)
            if not callable(device_info) or not str(device_info().get('architecture', '')).endswith('s'):
                raise ValueError('stock-long requires native architecture s block policy')
            stock_reduction = False
            effective_stripes = self.q1_stripes
        long_partial = getattr(arena, 'q1_stock_long_partial_dispatch_count', None)
        long_reduce = getattr(arena, 'q1_stock_long_reduce_dispatch_count', None)
        inline_counter = getattr(arena, 'q1_stock_long_metadata_dispatch_count', None)
        if self.stock_long_inline_metadata and (not callable(inline_counter) or
                not callable(getattr(getattr(arena, '_native', None), 'q1_stock_long_metadata_dispatch_count', None))):
            raise ValueError('native inline metadata physical-counter ABI unavailable')
        if stock_long_active and (not callable(long_partial) or not callable(long_reduce)):
            raise ValueError('native stock-long partial/reduce capability is unavailable')
        long_partial_before = long_partial() if callable(long_partial) else 0
        long_reduce_before = long_reduce() if callable(long_reduce) else 0
        inline_before = inline_counter() if callable(inline_counter) else 0
        if stock_reduction and (not short_q1 or self.q1_split_partition or
                (batch_size == 2 and abs(offsets[0] - offsets[1]) % 32) or
                any(getattr(getattr(owner, 'sequence', None), 'retained_start', 0) != 0
                    for lane in lanes for owner in lane.layers)):
            raise ValueError('stock reduction requires short full-prefix B1 or aligned B2')
        stock_counter = getattr(arena, 'q1_stock_reduction_dispatch_count', None)
        if stock_enabled and not callable(stock_counter):
            raise ValueError('native stock-reduction physical counter is unavailable')
        singleton_counter = getattr(arena, 'q1_stock_singleton_dispatch_count', None)
        if singleton_enabled and not callable(getattr(getattr(arena, '_native', None), 'q1_stock_singleton_dispatch_count', None)):
            raise ValueError('native singleton stock physical counter capability is unavailable')
        singleton_before = singleton_counter() if callable(singleton_counter) else 0
        stock_before = stock_counter() if callable(stock_counter) else 0
        partial_counter = getattr(arena, "q1_split_partial_dispatch_count", None)
        reduce_counter = getattr(arena, "q1_split_reduce_dispatch_count", None)
        if long_q1 and (not self.q1_split_partition or not callable(partial_counter) or not callable(reduce_counter)):
            raise ValueError("native long-Q1 split-KV capability is unavailable")
        partial_before = partial_counter() if callable(partial_counter) else 0
        reduce_before = reduce_counter() if callable(reduce_counter) else 0
        scratch_bytes = self.forward_scratch_bytes(tuple(offsets))
        scratch_reservation = None
        if scratch_bytes:
            if not callable(reserve_scratch):
                raise ValueError('charged long-Q1 scratch reservation required')
            scratch_reservation = reserve_scratch(scratch_bytes)
            if (getattr(scratch_reservation, 'bytes', 0) < scratch_bytes or
                    not callable(getattr(scratch_reservation, 'release', None)) or
                    not callable(getattr(scratch_reservation, 'retain_failure_roots', None))):
                self._failure_roots.append((scratch_reservation, lanes, branches))
                raise ValueError('charged long-Q1 scratch reservation is insufficient')
        tile_before = arena.q1_tile_dispatch_count()
        stripe_before = arena.q1_stripe_dispatch_count(effective_stripes)
        grouped_before = arena.grouped_q1_write_count()
        scalar_before = arena.write_dispatch_count()
        uses = []; all_layers = tuple(owner for lane in lanes for owner in lane.layers)
        expected_scalar_spans = dependency_count = cow_dependencies = 0
        direct_fence_before = getattr(self.backend, 'direct_grouped_fence_reads', 0)
        expected_direct_fences = 0
        recurrent_views = []; failed = True; hidden = logits = None; roots = (); boundary_checks = []
        deferred_started = False
        all_valid_recurrent_layers = 0
        joined_reused_slots = joined_fallback_slots = 0
        try:
            if deferred_active:
                arena.begin_deferred_q1(writes_only=self.q1_deferred_writes_only)
                deferred_started = True
            with mx.stream(arena.stream):
                hidden = self.trunk.embed_tokens(mx.array([[lane.token_ids[0]] for lane in lanes]))
                if hidden.dtype != expected_compute:
                    raise ValueError('hybrid activation dtype differs from loaded compute dtype')
                fa_index = gdn_index = 0
                stride = getattr(self.trunk, "eager_dispatch_stride", 0)
                if batch_size > getattr(self.trunk, "eager_dispatch_max_rows", 64): stride = 0
                if stride: getattr(runtime, "bump", lambda *args: None)("eager_dispatch_forwards")
                for layer_index, layer in enumerate(self.trunk.layers):
                    residual = hidden; mixed = layer.input_layernorm(hidden)
                    if layer.is_linear:
                        rows = [lane.recurrent_caches[gdn_index] for lane in lanes]
                        if reusable:
                            constructor = getattr(runtime, 'joined_batch_cache', None)
                            if not callable(constructor):
                                raise ValueError('joined recurrent view constructor is unavailable')
                            view = constructor(rows, reusable[gdn_index].joined_leaves)
                            joined_reused_slots += 2
                        else:
                            view = runtime.batch_cache(rows)
                            if self.q1_joined_recurrent:
                                joined_fallback_slots += 2
                        # This is ordinary Q1; private successor ownership does not imply speculation.
                        view.speculating = False
                        if self.q1_all_valid_recurrent:
                            if (getattr(view, 'lengths', None) is not None or
                                    getattr(view, 'left_padding', None) is not None):
                                raise ValueError('all-valid recurrent view has active row metadata')
                        else:
                            view.prepare(lengths=[1] * batch_size)
                        recurrent_views.append(view)
                        attended = layer.linear_attn(mixed, None, view)
                        if self.q1_all_valid_recurrent:
                            all_valid_recurrent_layers += 1
                        else:
                            view.finalize()
                        gdn_index += 1
                    else:
                        attention = layer.self_attn
                        queries, keys, values, gate = _pack_attention_q1(attention, mixed, offsets, runtime)
                        boundary_probe = getattr(self, "_fa_boundary_probe", None)
                        ordinary_qkv = (queries, keys, values) if callable(boundary_probe) else None
                        (queries, keys, values), checks = self._native_qkv((queries, keys, values), mx)
                        boundary_checks.extend(checks)
                        owners = tuple(lane.layers[fa_index] for lane in lanes)
                        layer_spans = (sum(len(owner.planned_spans(1, staged=True)) for owner in owners)
                                       if batch_size == 1 else 0)
                        expected_scalar_spans += layer_spans
                        tickets = self.backend.append_staged(owners, keys, values, (1,) * batch_size)
                        dependency_count += len(tickets)
                        if (self.q1_direct_grouped_fence and len(tickets) == 1 and
                                getattr(tickets[0], 'grouped_q1', False) is True):
                            expected_direct_fences += 1
                        if batch_size == 1:
                            # CopyTicket is an extra ordering dependency; CopyPrimitive
                            # does not increment the native head-local write counter.
                            cow_dependencies += len(tickets) - layer_spans
                        use = runtime.prepare_read(owners, (1,) * batch_size, query_heads=attention.num_attention_heads,
                                                   permit_candidate=True, profile_host=self.backend.profiling_enabled)
                        uses.append(use)
                        native = self.backend.read_staged(use, queries, tickets, scale=attention.scale)
                        if callable(boundary_probe):
                            boundary_probe({"layer_index": layer_index, "fa_index": fa_index,
                                "offsets": tuple(offsets), "queries": ordinary_qkv[0],
                                "keys": ordinary_qkv[1], "values": ordinary_qkv[2],
                                "native_attention": native, "hidden_dtype": str(hidden.dtype)})
                        attended = attention.o_proj(native.astype(hidden.dtype).reshape(batch_size, 1, -1) * runtime.gate_sigmoid(gate))
                        fa_index += 1
                    hidden = residual + attended
                    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
                    if stride and (layer_index == len(self.trunk.layers) - 1 or (layer_index + 1) % stride == 0):
                        mx.async_eval(hidden)
                        getattr(runtime, "bump", lambda *args: None)("eager_async_evals")
                logits = self.language_model.logits(self.trunk.norm(hidden))[:, 0, :]
                roots = tuple(value for lane in lanes for cache in lane.recurrent_caches for value in cache.cache)
                if deferred_active:
                    arena.deferred_q1_final_evals += 1
                graph_eval_started = time.perf_counter_ns() if self.backend.profiling_enabled else 0
                graph_eval_cpu_started = time.process_time_ns() if self.backend.profiling_enabled else 0
                mx.eval(logits, *roots, *boundary_checks,
                        *(arena.deferred_q1_roots() if deferred_active else ()))
                if self.backend.profiling_enabled:
                    self.backend.host_profile_ns['graph_eval'] += time.perf_counter_ns() - graph_eval_started
                    self.backend.host_profile_ns['graph_eval_process_cpu'] = (
                        self.backend.host_profile_ns.get('graph_eval_process_cpu', 0) +
                        time.process_time_ns() - graph_eval_cpu_started)
                self._require_representable(tuple(boundary_checks))
            proofs = self.backend.drain_staged(all_layers, tuple(uses))
            depth = len(self.layer_map.full_attention)
            if (len(proofs) != depth or arena.q1_tile_dispatch_count() - tile_before != (depth if short_q1 else 0) or
                    arena.q1_stripe_dispatch_count(effective_stripes) - stripe_before != (depth if short_q1 else 0)):
                raise RuntimeError('hybrid native full-attention tile dispatch proof differs')
            partial_delta = partial_counter() - partial_before if callable(partial_counter) else 0
            reduce_delta = reduce_counter() - reduce_before if callable(reduce_counter) else 0
            if (partial_delta, reduce_delta) != ((depth, depth) if long_q1 else (0, 0)):
                raise RuntimeError('hybrid long-Q1 split-KV dispatch proof differs')
            stock_long_partial_delta = long_partial() - long_partial_before if callable(long_partial) else 0
            stock_long_reduce_delta = long_reduce() - long_reduce_before if callable(long_reduce) else 0
            if (stock_long_partial_delta, stock_long_reduce_delta) != ((depth, depth) if stock_long_active else (0, 0)):
                raise RuntimeError('hybrid stock-long partial/reduce dispatch proof differs')
            inline_delta = inline_counter() - inline_before if callable(inline_counter) else 0
            if inline_delta != (depth if self.stock_long_inline_metadata else 0):
                raise RuntimeError('hybrid stock-long inline metadata dispatch proof differs')
            eval_after = arena.eval_submission_snapshot() if deferred_active else None
            eval_delta = ({key: eval_after[key] - eval_before[key]
                           for key in eval_before} if deferred_active else {})
            if deferred_active and eval_delta != {
                    'grouped_write_async_evals': 0,
                    'staged_read_async_evals': depth if self.q1_deferred_writes_only else 0,
                    'deferred_q1_write_roots': depth,
                    'deferred_q1_read_roots': depth,
                    'deferred_q1_final_evals': 1,
                    'deferred_q1_failure_flushes': 0}:
                raise RuntimeError('hybrid deferred Q1 submission/root proof differs')
            stock_delta = stock_counter() - stock_before if callable(stock_counter) else 0
            if stock_delta != (depth if stock_reduction else 0):
                raise RuntimeError('hybrid stock-reduction dispatch proof differs')
            singleton_delta = singleton_counter() - singleton_before if callable(singleton_counter) else 0
            if singleton_delta != (depth if stock_singleton else 0):
                raise RuntimeError('hybrid singleton stock32 dispatch proof differs')
            grouped_delta = arena.grouped_q1_write_count() - grouped_before
            direct_fence_delta = getattr(self.backend, 'direct_grouped_fence_reads', 0) - direct_fence_before
            if direct_fence_delta != expected_direct_fences:
                raise RuntimeError('hybrid direct grouped fence read proof differs')
            scalar_delta = arena.write_dispatch_count() - scalar_before
            if ((batch_size == 2 and self.require_grouped_write and (grouped_delta != depth or scalar_delta != 0)) or
                    (batch_size == 1 and (grouped_delta != 0 or scalar_delta != expected_scalar_spans))):
                raise RuntimeError('hybrid grouped native write dispatch proof differs')
            for index, proof in enumerate(proofs):
                for lane_index, branch in enumerate(branches):
                    branch.prove_staged_layer_read(index, lane_index, proof)
            new_bindings = ()
            if joined_active:
                produced = []
                for ordinal, view in enumerate(recurrent_views):
                    identity = self._joined_identity(branches, ordinal)
                    if identity is None:
                        break
                    revision, lane_ids, owner_refs, generations, prior_offsets, _ = identity
                    joined = tuple(view.cache)
                    rows = tuple(tuple(lane.recurrent_caches[ordinal].cache) for lane in lanes)
                    if (len(joined) != 2 or any(value is None for value in joined) or
                            any(joined[slot].shape != (2, *rows[0][slot].shape[1:]) or
                                joined[slot].dtype != rows[0][slot].dtype or
                                any(row[slot].shape != (1, *joined[slot].shape[1:]) or
                                    row[slot].dtype != joined[slot].dtype for row in rows)
                                for slot in (0, 1))):
                        break
                    produced.append(_JoinedRecurrentBinding(
                        revision, lane_ids, owner_refs, tuple(value + 1 for value in generations),
                        tuple(value + 1 for value in prior_offsets), ordinal, rows, joined))
                output_bytes = sum(value.nbytes for binding in produced
                                   for value in binding.joined_leaves)
                if (len(produced) == len(self.layer_map.recurrent) and
                        self._joined_charge.bytes >= 2 * output_bytes):
                    new_bindings = tuple(produced)
            for lane, branch, offset in zip(lanes, branches, offsets):
                branch.stage_recurrent_boundary(lane.recurrent_caches, offset=offset + 1)
            if self.q1_joined_recurrent:
                self._joined_bindings = new_bindings
            if scratch_reservation is not None: scratch_reservation.release()
            failed = False
        finally:
            if failed:
                if deferred_started:
                    try:
                        self.backend.abort_deferred_q1(all_layers, tuple(uses))
                    except BaseException:
                        # An ambiguous flush leaves roots, leases and charge pinned.
                        pass
                for use in uses:
                    if use.state == 'prepared':
                        try: use.abort_before_submit()
                        except BaseException: pass
                    elif use.state == 'submitted':
                        self.backend._orphaned_reads[use.lease.epoch] = use
                self.backend._failed = True
                writer.poisoned = True
                self._failure_roots.append((lanes, branches, tuple(uses),
                                            tuple(recurrent_views), hidden, logits, roots, tuple(boundary_checks),
                                            scratch_reservation, self._joined_bindings, self._joined_charge,
                                            arena.deferred_q1_roots() if deferred_started else ()))
                if self._joined_charge is not None:
                    try: self._joined_charge.retain_failure_roots(self._failure_roots[-1])
                    except BaseException: pass
                if scratch_reservation is not None:
                    try: scratch_reservation.retain_failure_roots(self._failure_roots[-1])
                    except BaseException: pass
            if deferred_started:
                arena.end_deferred_q1(retain_roots=
                    self.backend.retain_deferred_q1_roots(tuple(uses)))
        receipt = {'route': 'qwen35-hybrid-paged-q1-candidate', 'implemented': True,
                   'qualified': False, 'selected': False, 'observed_used': False,
                   'packed_lanes': batch_size, 'full_attention_layers': self.layer_map.full_attention,
                   'recurrent_layers': self.layer_map.recurrent, 'q1_simd_stripes': effective_stripes,
                   'q1_all_valid_recurrent_selected': self.q1_all_valid_recurrent,
                   'q1_all_valid_recurrent_layers': all_valid_recurrent_layers,
                   'q1_joined_recurrent_selected': self.q1_joined_recurrent,
                   'q1_direct_grouped_fence_selected': self.q1_direct_grouped_fence,
                   'q1_deferred_selected': self.q1_deferred,
                   'q1_deferred_writes_only_selected': self.q1_deferred_writes_only,
                   'q1_deferred_active': deferred_active,
                   'q1_deferred_eval_delta': eval_delta,
                   'direct_grouped_fence_reads': direct_fence_delta,
                   'q1_joined_reused_slots': joined_reused_slots,
                   'q1_joined_fallback_materializations': joined_fallback_slots,
                   'q1_joined_reserved_bytes': (self._joined_charge.bytes if self._joined_charge is not None else 0),
                   'native_tile_dispatches': depth if short_q1 else 0, 'native_recurrent_boundaries': batch_size,
                   'q1_split_partition': self.q1_split_partition,
                   'stock_long_selected': self.stock_long, 'stock_long_active': stock_long_active,
                   'stock_long_inline_metadata_selected': self.stock_long_inline_metadata,
                   'native_stock_long_metadata_dispatches': inline_delta,
                   'native_stock_long_partial_dispatches': stock_long_partial_delta,
                   'native_stock_long_reduce_dispatches': stock_long_reduce_delta,
                   'native_partial_numerator_rounding': (
                       compute_dtype if self.kv_precision == 'same' else 'float16') if stock_long_active else None,
                   'native_split_partial_dispatches': partial_delta,
                   'native_split_reduce_dispatches': reduce_delta,
                   'native_split_scratch_reserved_bytes': scratch_bytes,
                   'grouped_q1_writes': grouped_delta, 'scalar_native_writes': scalar_delta,
                   'expected_scalar_write_spans': expected_scalar_spans,
                   'native_write_dependency_count': dependency_count,
                   'native_cow_copy_dependencies': cow_dependencies,
                   'native_stock_reduction_dispatches': stock_delta,
                   'stock_singleton_selected': stock_singleton,
                   'native_stock_singleton_dispatches': singleton_delta,
                   'require_grouped_write': self.require_grouped_write,
                   'ordinary_projection_modules': True, 'activation_dtype': compute_dtype,
                   'native_qkv_dtype': compute_dtype if self.kv_precision == 'same' else 'float16',
                   'native_cache_dtype': compute_dtype if self.kv_precision == 'same' else 'float16',
                   'native_attention_result_cast_to': compute_dtype,
                   'kv_precision_policy': self.kv_precision,
                   'numerical_policy': ('same-' + compute_dtype if self.kv_precision == 'same' else
                                        'bf16-compute-fp16-kv-candidate' if compute_dtype == 'bfloat16' else
                                        'fp16-compute-fp16-kv-candidate'),
                   'representability_checks': len(boundary_checks),
                   'public_state_published': False, 'apcv2': 'ordinary_route_only'}
        return logits, receipt

    def forward_scratch_bytes(self, offsets: tuple[int, ...]) -> int:
        """Conservative all-FA-layer split scratch charge before graph construction."""
        if len(offsets) not in (1, 2) or any(type(o) is not int or o < 0 for o in offsets):
            raise ValueError('one or two nonnegative lane offsets required')
        if len(offsets) == 1: return 0
        maximum = max(offsets) + 1
        if maximum <= 128: return 0
        if self.stock_long and 1025 <= maximum <= self.max_visible_tokens:
            # Native scratch is FP32 [B2,QH,128 blocks,D+2]. Numerators are
            # rounded through arena BF16/FP16 before being stored as FP32.
            return self.native_layer_count * 2 * self.args.num_attention_heads * 128 * (256 + 2) * 4
        if not self.q1_split_partition or maximum > self.max_visible_tokens:
            raise ValueError('long-Q1 scratch has no bounded native partition profile')
        partitions = (maximum + self.q1_split_partition - 1) // self.q1_split_partition
        return self.native_layer_count * 2 * self.args.num_attention_heads * partitions * (256 + 2) * 4

    def release_failure_roots_after_teardown(self) -> None:
        """One-way failed arena proof, never a substitute for cancellation fencing."""
        writer = self.backend.writer
        if (writer.pending_epochs or writer.ledger.pending_count or
                self.backend._orphaned_reads or
                not getattr(writer, 'failed_arena_torn_down', False)):
            raise RuntimeError('hybrid failure roots require completed failed-arena teardown')
        self._failure_roots.clear()

    def bootstrap_staging_bytes(self, prompt_tokens: int) -> int:
        if type(prompt_tokens) is not int or not 1 <= prompt_tokens < self.max_visible_tokens:
            raise ValueError('hybrid bootstrap prompt exceeds bounded visible-token profile')
        args = self.args
        kv = len(self.layer_map.full_attention) * 2 * args.num_key_value_heads * 256 * prompt_tokens * 2
        recurrent = 0
        for index in self.layer_map.recurrent:
            module = self.trunk.layers[index].linear_attn
            dtype = getattr(module, '_gdn_state_dtype', None)
            state_bytes = 2 if str(dtype) in ('mlx.core.float16', 'float16') else 4
            conv_dim = 2 * args.linear_num_key_heads * args.linear_key_head_dim + args.linear_num_value_heads * args.linear_value_head_dim
            recurrent += args.linear_num_value_heads * args.linear_key_head_dim * args.linear_value_head_dim * state_bytes
            recurrent += (args.linear_conv_kernel_dim - 1) * conv_dim * 2
        # Ordinary KVCache grows in 256-token buffers; import packs logical tokens.
        allocated_kv = len(self.layer_map.full_attention) * 2 * args.num_key_value_heads * 256 * ((prompt_tokens + 255) // 256 * 256) * 2
        conversion_kv = kv if (self.kv_precision == "float16_candidate" and
                               self.native_dtype_preflight() == "bfloat16") else 0
        return allocated_kv + kv + conversion_kv + recurrent

    def bootstrap_ordinary(self, token_ids: tuple[int, ...], layers: tuple[Any, ...],
                           *, reserve_staging: Callable[[int], Any],
                           permit_candidate: bool = False) -> HybridBootstrap:
        """Ordinary isolated prefill; exact completed FA import, no publication.

        Reservation must expose bytes, release(), retain_failure_roots(tuple).
        The caller creates the initial joint KV/GDN checkpoint only after return.
        """
        if not permit_candidate:
            raise RuntimeError('hybrid bootstrap requires explicit candidate enablement')
        if (type(token_ids) is not tuple or not token_ids or
                any(type(t) is not int or not 0 <= t < self.args.vocab_size for t in token_ids)):
            raise ValueError('valid prompt token tuple required')
        compute_dtype = self.native_dtype_preflight()
        required = self.bootstrap_staging_bytes(len(token_ids))
        runtime = self._runtime_factory(); mx = runtime.mx
        self._check_owners(layers, 0, mx)
        reservation = reserve_staging(required)
        caches = (); logits = None; completed = False
        try:
            if (type(getattr(reservation, 'bytes', None)) is not int or reservation.bytes < required or
                    not callable(getattr(reservation, 'release', None)) or
                    not callable(getattr(reservation, 'retain_failure_roots', None))):
                raise ValueError('charged hybrid bootstrap staging reservation required')
            caches = tuple(self.language_model.make_cache())
            with mx.stream(self.backend.writer.backend.stream):
                hidden = self.trunk(mx.array([list(token_ids)]), cache=list(caches))
                if hidden.dtype != getattr(mx, compute_dtype):
                    raise ValueError('hybrid bootstrap activation dtype differs from loaded compute dtype')
                logits = self.language_model.logits(hidden[:, -1:, :])[:, 0, :]
                recurrent = tuple(caches[i] for i in self.layer_map.recurrent)
                for ordinal, cache in enumerate(recurrent):
                    module = self.trunk.layers[self.layer_map.recurrent[ordinal]].linear_attn
                    expected_state = getattr(module, '_gdn_state_dtype', None) or mx.float32
                    if (len(cache.cache) != 2 or cache.cache[0] is None or cache.cache[1] is None or
                            cache.cache[0].shape[0] != 1 or cache.cache[1].shape[0] != 1 or
                            cache.cache[0].dtype != getattr(mx, compute_dtype) or
                            cache.cache[1].dtype != expected_state):
                        raise ValueError('bootstrap recurrent leaves differ from model state dtype')
                roots = tuple(value for cache in recurrent for value in cache.cache)
                kv_roots = tuple(value for i in self.layer_map.full_attention for value in (caches[i].keys, caches[i].values))
                imports = []
                bootstrap_checks = []
                for index in self.layer_map.full_attention:
                    (keys, values), checks = self._native_qkv(caches[index].keys_and_values(), mx)
                    imports.append((keys, values)); bootstrap_checks.extend(checks)
                mx.eval(logits, *roots, *kv_roots, *bootstrap_checks)
                self._require_representable(tuple(bootstrap_checks))
                actual_kv = sum(value.nbytes for value in kv_roots)
                actual_recurrent = sum(value.nbytes for value in roots)
                packed_kv = sum(value.nbytes for i in self.layer_map.full_attention
                                for value in caches[i].keys_and_values())
                conversion_kv = sum(value.nbytes for pair in imports for value in pair) if self.kv_precision == "float16_candidate" and compute_dtype == "bfloat16" else 0
                if actual_kv + packed_kv + conversion_kv + actual_recurrent > reservation.bytes:
                    raise RuntimeError('hybrid bootstrap actual staging exceeds reservation')
                for ordinal, index in enumerate(self.layer_map.full_attention):
                    cache = caches[index]
                    if cache.offset != len(token_ids):
                        raise RuntimeError('ordinary prefill attention boundary drifted')
                    keys, values = imports[ordinal]
                    expected = (1, self.args.num_key_value_heads, len(token_ids), 256)
                    if (tuple(keys.shape) != expected or tuple(values.shape) != expected or
                            keys.dtype != getattr(mx, compute_dtype if self.kv_precision == 'same' else 'float16') or
                        values.dtype != keys.dtype):
                        raise ValueError('ordinary hybrid KV import geometry/dtype mismatch')
                    self.backend.append_completed((layers[ordinal],),
                        keys.transpose(0, 2, 1, 3).reshape(len(token_ids), self.args.num_key_value_heads, 256),
                        values.transpose(0, 2, 1, 3).reshape(len(token_ids), self.args.num_key_value_heads, 256),
                        (len(token_ids),))
            if any(owner.offset != len(token_ids) or owner._pending for owner in layers):
                raise RuntimeError('hybrid bootstrap native import is not completed')
            completed = True
            return HybridBootstrap(logits, recurrent, len(token_ids),
                {'route': 'qwen35-hybrid-ordinary-prefill-import', 'qualified': False,
                 'selected': False, 'observed_used': False, 'public_state_published': False,
                 'prompt_tokens': len(token_ids), 'full_attention_imports': len(layers),
                 'staging_reserved_bytes': reservation.bytes, 'ordinary_kv_bytes': actual_kv,
                 'recurrent_boundary_bytes': actual_recurrent, 'packed_import_bytes': packed_kv,
                 'conversion_kv_bytes': conversion_kv,
                 'staging_required_bytes': required, 'activation_dtype': compute_dtype,
                 'native_cache_dtype': compute_dtype if self.kv_precision == 'same' else 'float16',
                 'kv_precision_policy': self.kv_precision,
                 'representability_checks': len(bootstrap_checks)})
        finally:
            if completed:
                try: reservation.release()
                except BaseException:
                    self._bootstrap_failure_roots.append((reservation, caches, logits, layers, self.backend))
                    raise
            else:
                retained = (reservation, caches, logits, layers, self.backend)
                self._bootstrap_failure_roots.append(retained)
                retain = getattr(reservation, 'retain_failure_roots', None)
                if callable(retain):
                    try: retain((caches, logits, layers, self.backend))
                    except BaseException: pass
                else:
                    release = getattr(reservation, 'release', None)
                    if callable(release):
                        try: release()
                        except BaseException: pass


__all__ = ['HybridLayerMap', 'HybridPackedLane', 'HybridBootstrap',
           'hybrid_layer_map', 'clone_recurrent_caches', 'Qwen35PagedCandidate']
