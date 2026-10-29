"""Opt-in packed full-attention target-tree execution for Nanbeige.

One [B,W] target forward shares projections across lanes. KV remains owned by
independent B1 caches; only temporary attention operands are left-padded.
The implicit-attention option eliminates dense context masks but is a separate
GQA7/D128 split-K numerical law. Neither route is qualified or a default.
"""

import hashlib
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx

from ..runtime.attention_tree_executor import (
    _ABSENT,
    AttentionTreeCohortTransaction,
    AttentionTreeExecutor,
    AttentionTreeTransaction,
    _ancestors,
    _depths,
    _restore,
    _snapshot,
    attention_modules,
)
from ..runtime.models.cache import KVCache
from ..runtime.target_tree_executor import TargetTreeDispatchMetrics

PROTOCOL = "nanbeige43_packed_attention_tree_v1"
DEFAULT_TEMPORARY_BUDGET_BYTES = 2 << 30


def source_identity():
    from .nanbeige43_attention import source_identity as attention_identity
    from .nanbeige43_segmented_attention import source_identity as segmented_identity
    return hashlib.sha256((Path(__file__).read_text() + attention_identity() + segmented_identity()).encode()).hexdigest()


def estimate_temporary_memory(*, prefixes, width, layer_geometry, query_heads, implicit_attention, segmented_attention=False):
    """Conservative incremental provider workspace, not resident model/KV.

    ``layer_geometry`` contains (KV heads, K dim, V dim, K bytes, V bytes).
    Sum all layer temporaries because lazy dependencies may retain them.
    Native SDPA is conservatively allowed two full FP32 score matrices per
    layer; implicit attention counts its exact partial/statistics slabs.
    """
    if (not prefixes or any(type(n) is not int or n < 0 for n in prefixes)
            or type(width) is not int or not 1 <= width <= 32
            or type(query_heads) is not int or query_heads < 1):
        raise ValueError("invalid packed tree memory geometry")
    batch, maximum = len(prefixes), max(prefixes)
    length = maximum + width
    padded_lanes = sum(prefix < maximum for prefix in prefixes)
    kv = scratch = 0
    span = max(256, (length+127)//128)
    splits = (length+span-1)//span
    for geometry in layer_geometry:
        if len(geometry) != 5 or any(type(n) is not int or n < 1 for n in geometry):
            raise ValueError("invalid packed tree per-layer memory geometry")
        heads, kd, vd, kb, vb = geometry
        # B1 lends its existing view. B>1 concatenates the complete batch
        # and also temporarily materializes any shorter lane's left pad.
        if batch > 1 and not segmented_attention:
            kv += (batch+padded_lanes)*heads*length*(kd*kb+vd*vb)
        if implicit_attention or segmented_attention:
            scratch += batch*query_heads*width*splits*(kd+2)*4
        else:
            scratch += batch*query_heads*width*length*8
    mask = batch*width*width if implicit_attention or segmented_attention else batch*width*length
    return {"padded_kv_bytes": kv, "segmented_attention": segmented_attention, "attention_scratch_bytes": scratch,
            "mask_bytes": mask, "total_bytes": kv+scratch+mask,
            "layers": len(layer_geometry), "prefixes": list(prefixes), "width": width,
            "scope": "conservative incremental provider workspace; excludes resident KV/weights and common model activations"}


def _memory_geometry(model, caches):
    result = []
    for index, layer in enumerate(model.layers):
        attention = next((getattr(layer, name, None) for name in ("self_attn", "attention", "attn")
                          if getattr(layer, name, None) is not None), None)
        if attention is None:
            raise ValueError("packed tree needs attention geometry before allocation")
        heads, dim = int(attention.n_kv_heads), int(attention.head_dim)
        default_dtype = layer.input_layernorm.weight.dtype
        default_bytes = int(layer.input_layernorm.weight.itemsize)
        geometry = None
        for cache in caches:
            item = cache[index]
            if (item.keys is None) != (item.values is None):
                raise ValueError("packed tree KV storage is incomplete")
            if item.keys is None:
                current = (heads, dim, dim, default_bytes, default_bytes)
            else:
                k, v = item.keys, item.values
                if (k.ndim != 4 or v.ndim != 4 or k.shape[:2] != (1, heads)
                        or v.shape[:2] != (1, heads) or k.shape[2] < item.offset
                        or v.shape[2] < item.offset or k.shape[3] != dim or v.shape[3] != dim
                        or k.dtype != default_dtype or v.dtype != default_dtype):
                    raise ValueError("packed tree KV storage geometry mismatch")
                current = (heads, dim, dim, int(k.itemsize), int(v.itemsize))
            if geometry is not None and current != geometry:
                raise ValueError("packed tree lanes require matching KV geometry")
            geometry = current
        result.append(geometry)
    return result


class _Plan:
    def __init__(self, parents, prefixes, implicit):
        self.prefixes, self.implicit = list(prefixes), implicit
        self.width = len(parents[0])
        self.depths = mx.array([_depths(tree) for tree in parents], dtype=mx.int32)
        self.ancestors = mx.array([_ancestors(tree) for tree in parents], dtype=mx.bool_)
        self.positions = self.depths + mx.array(prefixes, dtype=mx.int32)[:, None]
        self._mask = None

    def mask(self):
        if self.implicit:
            return None
        if self._mask is None:
            maximum = max(self.prefixes)
            rows = []
            for index, prefix in enumerate(self.prefixes):
                head = mx.broadcast_to(mx.arange(maximum)[None, :] >= maximum-prefix, (self.width, maximum))
                rows.append(mx.concatenate([head, self.ancestors[index]], axis=1))
            self._mask = mx.stack(rows)[:, None]
        return self._mask


class _PackedRope:
    def __init__(self, rope, plan):
        self.rope, self.plan = rope, plan

    def __call__(self, x, offset=0):
        batch, heads, rows, dim = x.shape
        if (batch, rows) != self.plan.positions.shape:
            raise ValueError("packed tree RoPE geometry mismatch")
        value = x.transpose(0, 2, 1, 3).reshape(batch*rows, heads, 1, dim)
        value = self.rope(value, offset=self.plan.positions.reshape(-1))
        return value.reshape(batch, rows, heads, dim).transpose(0, 2, 1, 3)


class _LaneKV:
    """A transient batch view; updates never merge underlying cache ownership."""
    def __init__(self, owners, plan, *, shared_kv_rows=1, rope_kv_fused=False):
        self.owners, self.plan, self.offset = owners, plan, 0
        self.shared_kv_rows, self.rope_kv_fused = shared_kv_rows, rope_kv_fused

    def make_mask(self, n, return_array=False, window_size=None):
        if n != self.plan.width or window_size is not None:
            raise ValueError("packed tree requires ordinary full attention")
        return self.plan.mask()

    def _collect(self, rows):
        maximum = max(self.plan.prefixes)
        out_k, out_v = [], []
        for (k, v), prefix in zip(rows, self.plan.prefixes, strict=True):
            left = maximum-prefix
            if left:
                k = mx.pad(k, [(0, 0), (0, 0), (left, 0), (0, 0)])
                v = mx.pad(v, [(0, 0), (0, 0), (left, 0), (0, 0)])
            out_k.append(k); out_v.append(v)
        if len(out_k) == 1:
            return out_k[0], out_v[0]
        return mx.concatenate(out_k, axis=0), mx.concatenate(out_v, axis=0)

    def update_and_fetch(self, keys, values):
        rows=[]
        for index, (owner, prefix) in enumerate(zip(self.owners, self.plan.prefixes, strict=True)):
            if owner.offset != prefix:
                raise ValueError("packed tree cache changed during traversal")
            rows.append(owner.update_and_fetch(keys[index:index+1], values[index:index+1]))
        return self._collect(rows)

    def try_fused_rope_append(self, q, k, v, rope):
        if not self.rope_kv_fused:
            return None
        from mlx import nn
        base=rope.rope if isinstance(rope, _PackedRope) else None
        if (type(base) is not nn.RoPE or base.dims != 128 or base.scale != 1.0
                or base.traditional or base.base != 120000000.0):
            return None
        from .nanbeige43_physical import fused_rope_append_arrays
        results=[]
        for index,(owner,prefix) in enumerate(zip(self.owners,self.plan.prefixes,strict=True)):
            if owner.offset!=prefix:
                raise ValueError("packed tree cache changed during fused traversal")
            result=fused_rope_append_arrays(q[index:index+1],k[index:index+1],v[index:index+1],
                owner.keys,owner.values,prefix,self.plan.positions[index],base=base.base)
            if result is None:
                return None
            results.append(result)
        # All lanes admitted; borrowed descriptors are never modified by the kernel.
        rows=[]; queries=[]
        for owner,(qr,keys,values,end) in zip(self.owners,results,strict=True):
            owner.keys,owner.values,owner.offset=keys,values,end
            rows.append((keys[...,:end,:],values[...,:end,:])); queries.append(qr)
        joined=self._collect(rows)
        if isinstance(self,_ImplicitLaneKV):
            self._kv=joined
        return (queries[0] if len(queries)==1 else mx.concatenate(queries,axis=0),*joined)



class _ImplicitLaneKV(_LaneKV):
    def update_and_fetch(self, keys, values):
        self._kv = super().update_and_fetch(keys, values)
        return self._kv

    def bucketed_attention(self, queries, scale, mask, sinks=None):
        if sinks is not None or mask is not None:
            raise ValueError("implicit packed tree refuses sinks or external masks")
        from .nanbeige43_attention import attend
        return attend(queries, *self._kv, prefixes=self.plan.prefixes,
                      ancestors=self.plan.ancestors, scale=scale)


class _SegmentedLaneKV(_LaneKV):
    """Independent lane views; no joined/padded history, including B2/B4."""
    def _collect(self, rows):
        self._segments=tuple(rows)
        return None,None

    def bucketed_attention(self, queries, scale, mask, sinks=None):
        if sinks is not None or mask is not None:
            raise ValueError("segmented packed tree refuses sinks or external masks")
        from .nanbeige43_segmented_attention import attend_segments
        return attend_segments(queries,[pair[0] for pair in self._segments],
            [pair[1] for pair in self._segments],prefixes=self.plan.prefixes,
            ancestors=self.plan.ancestors,scale=scale,shared_kv_rows=self.shared_kv_rows)


@contextmanager
def _scope(model, plan):
    installed = []
    try:
        for module in attention_modules(model):
            shadowed = module.__dict__.get("rope", _ABSENT)
            installed.append((module, shadowed))
            object.__setattr__(module, "rope", _PackedRope(module.rope, plan))
        yield
    finally:
        for module, shadowed in reversed(installed):
            if shadowed is _ABSENT:
                object.__delattr__(module, "rope")
            else:
                object.__setattr__(module, "rope", shadowed)


class NanbeigePackedTreeExecutor(AttentionTreeExecutor):
    protocol = PROTOCOL

    def __init__(self, *, implicit_attention=False, segmented_attention=False, shared_kv_rows=1, rope_kv_fused=False, temporary_budget_bytes=DEFAULT_TEMPORARY_BUDGET_BYTES):
        if type(temporary_budget_bytes) is not int or not 1 <= temporary_budget_bytes <= DEFAULT_TEMPORARY_BUDGET_BYTES:
            raise ValueError("packed tree temporary budget must be positive and at most 2 GiB")
        if shared_kv_rows not in (1,4) or (shared_kv_rows != 1 and not segmented_attention):
            raise ValueError("shared KV rows4 requires segmented attention")
        self.segmented_attention = bool(segmented_attention)
        self.shared_kv_rows = shared_kv_rows
        self.rope_kv_fused = bool(rope_kv_fused)
        self.implicit_attention = bool(implicit_attention or segmented_attention)
        self.temporary_budget_bytes = temporary_budget_bytes
        self.last_memory_estimate = None
        self.source_revision_token = self._revision()
        self.source_identity = (("module", "mlx2.adapters.nanbeige43_tree"),
                                ("sha256", self.source_revision_token),
                                ("implicit_attention", str(self.implicit_attention)),
                                ("segmented_attention", str(self.segmented_attention)),
                                ("shared_kv_rows", str(self.shared_kv_rows)),
                                ("rope_kv_fused", str(self.rope_kv_fused)),
                                ("temporary_budget_bytes", str(self.temporary_budget_bytes)))
        self.dispatches = 0

    @classmethod
    def create(cls, *, implicit_attention=False, segmented_attention=False, shared_kv_rows=1, rope_kv_fused=False, temporary_budget_bytes=DEFAULT_TEMPORARY_BUDGET_BYTES):
        return cls(implicit_attention=implicit_attention, segmented_attention=segmented_attention, shared_kv_rows=shared_kv_rows, rope_kv_fused=rope_kv_fused, temporary_budget_bytes=temporary_budget_bytes)

    def _revision(self):
        physical=""
        if self.rope_kv_fused:
            from .nanbeige43_physical import source_identity as physical_identity
            physical=physical_identity()
        return hashlib.sha256(f"{source_identity()}:{physical}:{int(self.implicit_attention)}:{int(self.segmented_attention)}:{self.shared_kv_rows}:{int(self.rope_kv_fused)}:{self.temporary_budget_bytes}".encode()).hexdigest()

    def validate_revision(self):
        if self._revision() != self.source_revision_token:
            raise RuntimeError("packed tree source changed after executor creation")

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        return self.forward_many(model, [inputs], [parents], [cache], layers,
                                 cached=cached, configured_limit=1)

    def forward_many(self, model, inputs, parents, caches, layers, *, cached, configured_limit):
        self.validate_revision()
        self.validate_cohort(len(caches), configured_limit)
        if not len(inputs) == len(parents) == len(caches):
            raise ValueError("packed tree lane count mismatch")
        width = len(inputs[0])
        if not 1 <= width <= 32 or any(len(row) != width or len(tree) != width for row, tree in zip(inputs, parents, strict=True)):
            raise ValueError("packed tree needs uniform widths in 1..32")
        if getattr(model, "_target_verify_row_exact", False):
            raise ValueError("packed tree has a separate law from serial row-exact")
        if len({id(item) for cache in caches for item in cache}) != sum(map(len, caches)):
            raise ValueError("packed tree requires independent cache owners")
        # Distinct cache objects can still alias one mutable mx.array
        # descriptor after a shallow copy. __setitem__ then changes both
        # owners even though their offsets differ. Proper COW clones use
        # distinct stop_gradient descriptors and remain admissible.
        buffers = [value for cache in caches for item in cache
                   for value in (getattr(item, "keys", None), getattr(item, "values", None))
                   if value is not None]
        if len({id(value) for value in buffers}) != len(buffers):
            raise ValueError("packed tree requires independent mutable KV descriptors")
        prefixes = []
        for cache in caches:
            if len(cache) != len(model.layers) or any(type(item) is not KVCache for item in cache):
                raise ValueError("packed tree requires one plain full KVCache per layer")
            offsets = {int(item.offset) for item in cache}
            if len(offsets) != 1:
                raise ValueError("packed tree requires one committed boundary per lane")
            prefixes.append(offsets.pop())
        for tree in parents:
            if tree[0] != -1 or any(type(p) is not int or not 0 <= p < row for row, p in enumerate(tree[1:], start=1)):
                raise ValueError("packed tree parents must form a rooted topological tree")
        if self.implicit_attention:
            args = model.args
            if (args.num_attention_heads, args.num_key_value_heads, args.head_dim) != (56, 8, 128):
                raise ValueError("implicit packed tree requires Q56/KV8/D128")
        estimate = estimate_temporary_memory(prefixes=prefixes, width=width,
            layer_geometry=_memory_geometry(model, caches),
            query_heads=int(model.args.num_attention_heads), implicit_attention=self.implicit_attention, segmented_attention=self.segmented_attention)
        if self.rope_kv_fused:
            # Pure fused append returns full-capacity successors, and input
            # preparation can also materialize strided borrowed arrays. Sum
            # across every layer/lane because lazy graphs can retain both.
            successors=sum(int(value.size)*int(value.itemsize)*2
                for cache in caches for owner in cache
                for value in (owner.keys,owner.values) if value is not None)
            preparation=len(caches)*width*128*(56+8+8)*4*2*len(model.layers)
            estimate["fused_successor_and_preparation_bytes"]=successors+preparation
            estimate["total_bytes"]+=successors+preparation
        self.last_memory_estimate = estimate
        if estimate["total_bytes"] > self.temporary_budget_bytes:
            raise MemoryError(f"packed tree temporary estimate {estimate['total_bytes']} bytes exceeds "
                              f"budget {self.temporary_budget_bytes} bytes")
        plan = _Plan(parents, prefixes, self.implicit_attention)
        snapshots = [_snapshot(cache) for cache in caches]
        wrapper = _SegmentedLaneKV if self.segmented_attention else (_ImplicitLaneKV if self.implicit_attention else _LaneKV)
        layers_cache = [wrapper([cache[layer] for cache in caches], plan, shared_kv_rows=self.shared_kv_rows, rope_kv_fused=self.rope_kv_fused) for layer in range(len(model.layers))]
        try:
            with _scope(model, plan):
                logits, features = model.forward_with_taps(mx.array(inputs, dtype=mx.uint32), layers_cache, layers)
        except BaseException:
            for snapshot in snapshots:
                _restore(snapshot)
            raise
        transactions = [AttentionTreeTransaction(cache, snapshot, width, protocol=self.protocol,
                        token=self.source_revision_token, parents=tree)
                        for cache, snapshot, tree in zip(caches, snapshots, parents, strict=True)]
        transaction = AttentionTreeCohortTransaction(transactions, protocol=self.protocol, token=self.source_revision_token)
        transaction.physical_packed = len(caches) > 1
        self.dispatches += 1
        return logits, features, transaction

    def dispatch_metrics(self, width, transaction, *, cached):
        return TargetTreeDispatchMetrics(width=width, physical_packed=bool(transaction.physical_packed),
                                        executor_cached=False, cached=bool(cached), provider_protocol=self.protocol)

    def receipt(self):
        return {"provider": self.protocol, "source_identity": dict(self.source_identity),
                "source_revision_token": self.source_revision_token, "dispatches": self.dispatches,
                "qualified": False, "packed_projections": True, "kv_layout": "lane-owned-independent-segments" if self.segmented_attention else "lane-owned-left-padded-temporary",
                "segmented_attention": self.segmented_attention, "shared_kv_rows": self.shared_kv_rows, "rope_kv_fused": self.rope_kv_fused,
                "implicit_attention": self.implicit_attention,
                "temporary_budget_bytes": self.temporary_budget_bytes,
                "last_memory_estimate": self.last_memory_estimate, "tree_law": {"law_id":
                ("nanbeige43-segmented-gqa7-r"+str(self.shared_kv_rows)+"-tile16-v1") if self.segmented_attention else
                "nanbeige43-packed-" + ("implicit-gqa7-v1" if self.implicit_attention else "masked-sdpa-v1"),
                "tree_size_invariant": False}}
