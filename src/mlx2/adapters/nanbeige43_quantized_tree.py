"""Explicit Q8-KV target-tree composition for Nanbeige.

Quantization is a separate approximate-state operation. This executor keeps
already-produced packed codes/scales/biases exact across branch commit/abort;
it does not make approximate state eligible for exact APCv2 publication.
Direct composed QSDPA consumes packed history without full-history dequantize.
See provenance/nanbeige43-quantized-tree.json.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ..runtime.approximate_kv import KVQuantizationDescriptor, operation_revision
from ..runtime.attention_tree_executor import AttentionTreeExecutor, _depths
from ..runtime.models.base import _composed_qsdpa
from ..runtime.models.cache import QuantizedKVCache
from ..runtime.target_tree_executor import TargetTreeDispatchMetrics
from .nanbeige43_tree import DEFAULT_TEMPORARY_BUDGET_BYTES, _Plan, _scope

PROTOCOL = "nanbeige43_q8_attention_tree_v1"
LAYOUT = "native-q8g64-lane-owned-quantized-tree-v1"


def source_identity():
    from ..runtime.models import base, cache
    from .nanbeige43_tree import source_identity as tree_identity
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for module in (base, cache):
        digest.update(Path(module.__file__).read_bytes())
    digest.update(tree_identity().encode())
    return digest.hexdigest()


def _format(item):
    return (item.group_size, item.key_bits, item.value_bits, item.bits, item.rotate, item.normalize)


def _parts(item):
    return (() if item.keys is None else tuple(item.keys)) + (() if item.values is None else tuple(item.values))


def _check_owner(item, heads, dim, dtype):
    if type(item) is not QuantizedKVCache:
        raise ValueError("Q8 tree requires native single-lane QuantizedKVCache owners")
    if (item.group_size, item.key_bits, item.value_bits, item.bits, item.rotate, item.normalize) != (64, 8, 8, 8, False, False):
        raise ValueError("Q8 tree supports only unrotated, non-normalized affine Q8/group64")
    if type(item.offset) is not int or item.offset < 0 or dim % 64:
        raise ValueError("invalid Q8 tree owner offset/head dimension")
    if (item.keys is None) != (item.values is None):
        raise ValueError("incomplete Q8 tree owner")
    if item.keys is None:
        if item.offset:
            raise ValueError("nonempty Q8 tree offset has no storage")
        return
    lengths = set()
    for triple in (item.keys, item.values):
        if len(triple) != 3:
            raise ValueError("Q8 tree owner needs packed/scales/biases triples")
        for part, width, expected_dtype in zip(triple, (dim // 4, dim // 64, dim // 64), (mx.uint32, dtype, dtype), strict=True):
            if (part.ndim != 4 or part.shape[:2] != (1, heads) or part.shape[-1] != width
                    or part.shape[2] < item.offset or part.dtype != expected_dtype):
                raise ValueError("Q8 tree packed owner geometry mismatch")
            lengths.add(part.shape[2])
    if len(lengths) != 1:
        raise ValueError("Q8 tree packed owner capacities differ")


@dataclass(frozen=True)
class _Snapshot:
    owner: object
    keys: object
    values: object
    offset: int
    format: tuple


def _snapshot(owners):
    # These owners are append-only after offset. Preserve the original graph
    # containers; appending/compacting touches only unpublished suffix cells.
    # Proper source clones have distinct descriptors (see clone_q8_cache).
    return tuple(_Snapshot(owner, owner.keys, owner.values, int(owner.offset), _format(owner)) for owner in owners)


def _restore(snapshots):
    for state in snapshots:
        state.owner.keys, state.owner.values, state.owner.offset = state.keys, state.values, state.offset
        (state.owner.group_size, state.owner.key_bits, state.owner.value_bits,
         state.owner.bits, state.owner.rotate, state.owner.normalize) = state.format


def clone_q8_cache(owners):
    """Independent mutable descriptors; immutable packed storage remains COW."""
    result = []
    for owner in owners:
        if type(owner) is not QuantizedKVCache:
            raise ValueError("Q8 clone requires native quantized owners")
        clone = QuantizedKVCache(group_size=owner.group_size, bits=owner.bits,
                                 key_bits=owner.key_bits, value_bits=owner.value_bits,
                                 rotate=owner.rotate, normalize=owner.normalize)
        clone.offset = owner.offset
        for name in ("keys", "values"):
            value = getattr(owner, name)
            setattr(clone, name, None if value is None else tuple(mx.stop_gradient(part) for part in value))
        clone.key_scale, clone.value_scale = owner.key_scale, owner.value_scale
        result.append(clone)
    return result


class Q8TreeTransaction:
    def __init__(self, snapshots, parents, *, provider):
        self.snapshots, self.parents = tuple(snapshots), tuple(tuple(row) for row in parents)
        self.provider = provider
        self.provider_protocol, self.provider_revision_token = provider.protocol, provider.source_revision_token
        self.closed, self.executor_cached = False, False
        self.physical_packed = len(snapshots) > 1

    def _validate(self, paths):
        self.provider.validate_revision()
        if self.closed or len(paths) != len(self.snapshots):
            raise RuntimeError("Q8 tree transaction is closed or lane count changed")
        result = []
        for states, parents, path in zip(self.snapshots, self.parents, paths, strict=True):
            path = tuple(path)
            if (not path or path[0] != 0 or any(type(row) is not int or not 0 <= row < len(parents) for row in path)
                    or any(parents[row] != parent for parent, row in zip((-1, *path[:-1]), path, strict=True))):
                raise ValueError("Q8 tree commit needs a root-connected path")
            if any(state.owner.offset != state.offset + len(parents) for state in states):
                raise RuntimeError("Q8 tree owner moved before commit")
            if any(_format(state.owner) != state.format for state in states):
                raise RuntimeError("Q8 tree owner format changed before commit")
            if any(len(_parts(state.owner)) != 6 or any(part.shape[2] < state.owner.offset for part in _parts(state.owner)) for state in states):
                raise RuntimeError("Q8 tree owner storage changed before commit")
            result.append(path)
        return result

    def commit_paths(self, paths):
        paths = self._validate(paths)
        staged = []
        try:
            for states, path, parents in zip(self.snapshots, paths, self.parents, strict=True):
                lane = []
                for state in states:
                    owner, start = state.owner, state.offset
                    if path == tuple(range(len(path))):
                        lane.append((state, None, None, len(path)))
                    else:
                        rows = mx.array(path, dtype=mx.int32)
                        triples = [tuple(mx.take(part[..., start:start+len(parents), :], rows, axis=2) for part in triple)
                                   for triple in (owner.keys, owner.values)]
                        lane.append((state, triples[0], triples[1], len(path)))
                staged.append(lane)
            # Only bounded suffix staging is evaluated; never dequantize or
            # requantize any committed prefix, regardless of token alignment.
            mx.eval([part for lane in staged for _, keys, values, _ in lane
                     for triple in (keys, values) if triple is not None for part in triple])
            for lane in staged:
                for state, keys, values, keep in lane:
                    owner, start = state.owner, state.offset
                    if keys is not None:
                        for destination, selected in ((owner.keys, keys), (owner.values, values)):
                            for target, source in zip(destination, selected, strict=True):
                                target[..., start:start+keep, :] = source
                    owner.offset = start + keep
        except BaseException:
            for states in self.snapshots:
                _restore(states)
            raise
        finally:
            self.closed = True
        return [[state.owner for state in states] for states in self.snapshots]

    def commit(self, *, accepted_lengths):
        return self.commit_paths([list(range(int(length))) for length in accepted_lengths])

    def abort(self):
        if not self.closed:
            for states in self.snapshots:
                _restore(states)
            self.closed = True


class _Q8LaneView:
    bits = key_bits = value_bits = 8
    group_size = 64
    rotate = normalize = False
    key_scale = value_scale = None

    def __init__(self, owners, plan):
        self.owners, self.plan, self.offset = owners, plan, 0
        self._kv = None

    def make_mask(self, n, return_array=False, window_size=None):
        if n != self.plan.width or window_size is not None:
            raise ValueError("Q8 tree requires ordinary full attention")
        # The private attention hook owns one unpadded prefix/ancestor mask
        # per lane. No B*max(prefix) mask or padded historical KV is built.

    def update_and_fetch(self, keys, values):
        pair = []
        for lane, (owner, prefix) in enumerate(zip(self.owners, self.plan.prefixes, strict=True)):
            if owner.offset != prefix:
                raise RuntimeError("Q8 tree cache changed during traversal")
            pair.append(owner.update_and_fetch(keys[lane:lane+1], values[lane:lane+1]))
        self._kv = tuple(pair)
        # base.scaled_dot_product_attention invokes bucketed_attention before
        # inspecting operands; these placeholders never reach stock SDPA.
        return None, None

    def bucketed_attention(self, queries, scale, mask, sinks=None):
        if sinks is not None or mask is not None or self._kv is None:
            raise ValueError("Q8 tree refuses sinks, external masks or attention before append")
        self.attention_calls = getattr(self, "attention_calls", 0) + len(self.owners)
        outputs = [
            _composed_qsdpa(queries[lane:lane+1], *pair, scale,
                           self.plan.q8_masks[lane], 64, 8, 8)
            for lane, pair in enumerate(self._kv)
        ]
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)


class NanbeigeQ8TreeExecutor(AttentionTreeExecutor):
    protocol = PROTOCOL

    def __init__(self, *, adapter_fingerprint, temporary_budget_bytes=DEFAULT_TEMPORARY_BUDGET_BYTES):
        if not isinstance(adapter_fingerprint, str) or not adapter_fingerprint:
            raise ValueError("Q8 tree requires an exact target artifact fingerprint")
        if type(temporary_budget_bytes) is not int or not 1 <= temporary_budget_bytes <= DEFAULT_TEMPORARY_BUDGET_BYTES:
            raise ValueError("Q8 tree temporary budget must be positive and at most 2 GiB")
        self.operation_revision = operation_revision(adapter_fingerprint, "kv_q8", KVQuantizationDescriptor(8,8,64))
        self.temporary_budget_bytes, self.dispatches = temporary_budget_bytes, 0
        self.scheduled_attention_calls = 0
        self.last_memory_estimate = None
        self.source_revision_token = self._revision()
        self.source_identity = (("module", "mlx2.adapters.nanbeige43_quantized_tree"),
                                ("sha256", self.source_revision_token),
                                ("approximate_operation_revision", self.operation_revision))

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def preserve_fresh_target_cache(self, target, prefix_tokens):
        """Admit only operation-produced empty owners, never an APC hit.

        External scheduling otherwise discards a target without draft sidecar.
        Empty Q8 owners contain no history to mismatch with fresh draft state.
        Nonempty restored state still requires a matching sidecar or replay.
        """
        self.validate_revision()
        if prefix_tokens or not isinstance(target, (list, tuple)) or not target:
            return False
        return all(
            type(item) is QuantizedKVCache and item.offset == 0
            and item.keys is None and item.values is None
            and (item.group_size, item.key_bits, item.value_bits, item.bits,
                 item.rotate, item.normalize) == (64, 8, 8, 8, False, False)
            for item in target
        )

    def _revision(self):
        return hashlib.sha256(f"{source_identity()}:{self.operation_revision}:{self.temporary_budget_bytes}".encode()).hexdigest()

    def validate_revision(self):
        if self._revision() != self.source_revision_token:
            raise RuntimeError("Q8 tree source changed after executor creation")

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        return self.forward_many(model, [inputs], [parents], [cache], layers, cached=cached, configured_limit=1)

    def forward_many(self, model, inputs, parents, caches, layers, *, cached, configured_limit):
        self.validate_revision()
        self.validate_cohort(len(caches), configured_limit)
        if not len(inputs) == len(parents) == len(caches) or not inputs:
            raise ValueError("Q8 tree lane count mismatch")
        width = len(inputs[0])
        if not 1 <= width <= 32 or any(len(row)!=width or len(tree)!=width for row,tree in zip(inputs,parents,strict=True)):
            raise ValueError("Q8 tree needs uniform tile widths in1..32")
        if getattr(model, "_target_verify_row_exact", False):
            raise ValueError("Q8 tree has a distinct approximate numerical law")
        owners = [item for lane in caches for item in lane]
        if len({id(item) for item in owners}) != len(owners):
            raise ValueError("Q8 tree requires independent lane cache owners")
        parts = [part for owner in owners for part in _parts(owner)]
        if len({id(part) for part in parts}) != len(parts):
            raise ValueError("Q8 tree requires independent mutable packed descriptors")
        modules = [layer.self_attn for layer in model.layers]
        prefixes = []
        for lane in caches:
            if len(lane) != len(modules):
                raise ValueError("Q8 tree layer/cache count mismatch")
            offsets = set()
            for owner, attention, layer in zip(lane, modules, model.layers, strict=True):
                dtype = layer.input_layernorm.weight.dtype
                _check_owner(owner, int(attention.n_kv_heads), int(attention.head_dim), dtype)
                offsets.add(owner.offset)
            if len(offsets) != 1:
                raise ValueError("Q8 tree requires one committed boundary per lane")
            prefixes.append(offsets.pop())
        for tree in parents:
            if tree[0] != -1 or any(type(p)is not int or not 0<=p<row for row,p in enumerate(tree[1:],start=1)):
                raise ValueError("Q8 tree parents must form one rooted topological tree")
            _depths(tree)
        # Packed projections share one target traversal. Attention consumes
        # independent quantized views, so no historical KV is padded/copied.
        useful_columns = sum(prefix + width for prefix in prefixes)
        scratch = int(model.args.num_attention_heads)*width*useful_columns*8*len(modules)
        mask_bytes = width * useful_columns
        estimate = {"packed_temporary_kv_bytes": 0, "score_scratch_bytes": scratch,
                    "mask_bytes": mask_bytes, "total_bytes": scratch+mask_bytes,
                    "prefixes":prefixes,"width":width,
                    "scope": "conservative per-lane score workspace; excludes resident packed KV, cache growth and weights"}
        self.last_memory_estimate = estimate
        if estimate["total_bytes"] > self.temporary_budget_bytes:
            raise MemoryError("Q8 tree incremental workspace exceeds admitted temporary budget")
        plan = _Plan(parents, prefixes, False)
        plan.q8_masks = tuple(
            mx.concatenate([mx.broadcast_to(mx.array(True), (width, prefix)), plan.ancestors[lane]], axis=1)[None, None]
            for lane, prefix in enumerate(prefixes)
        )
        snapshots = tuple(_snapshot(lane) for lane in caches)
        views = [_Q8LaneView([lane[index] for lane in caches], plan) for index in range(len(modules))]
        try:
            with _scope(model, plan):
                logits, features = model.forward_with_taps(mx.array(inputs,dtype=mx.uint32),views,layers)
        except BaseException:
            for states in snapshots:
                _restore(states)
            raise
        transaction = Q8TreeTransaction(snapshots, parents, provider=self)
        self.scheduled_attention_calls += sum(getattr(view, "attention_calls", 0) for view in views)
        self.dispatches += 1
        return logits, features, transaction

    def dispatch_metrics(self, width, transaction, *, cached):
        return TargetTreeDispatchMetrics(width=width, physical_packed=transaction.physical_packed,
            executor_cached=False, cached=bool(cached), provider_protocol=self.protocol)

    def receipt(self):
        return {"provider":self.protocol,"source_identity":dict(self.source_identity),
                "source_revision_token":self.source_revision_token,"dispatches":self.dispatches,
                "qualified":False,"approximate":True,"approximate_operation_revision":self.operation_revision,
                "kv_layout":LAYOUT,"packed_projections":True,"full_history_dequantization":False,
                "attention_layout": "per-lane direct quantized QSDPA",
                "single_batched_attention": False, "historical_kv_concat": False,
                "historical_kv_padding": False, "scheduled_attention_calls": self.scheduled_attention_calls,
                "apcv2_publication":"refused_without_qualified_approximate_operation",
                "last_memory_estimate":self.last_memory_estimate,
                "tree_law":{"law_id":"native-q8g64-composed-qsdpa-tree-v1","tree_size_invariant":False}}
