# SPDX-License-Identifier: MIT
"""Private, physically batched Source tree with shared layer checkpoints.

The prefix is borrowed through APCv2's descriptor snapshot operation. Layer
segments remain private until a caller selects a root-connected input path.
This module never inserts a speculative node into the prefix cache.
"""
from __future__ import annotations

from contextlib import nullcontext,contextmanager
import hashlib

import mlx.core as mx
import numpy as np

PROFILE = "source_ast_native_batched_segments_v1"
FORWARD_LAW = "fully_batched_v1"
M1_FORWARD_LAW = "fully_batched_m1_v1"
PROFILES = {FORWARD_LAW: PROFILE, M1_FORWARD_LAW: "source_ast_m1_batch_axis_segments_candidate_v1"}
RESIDENT_EXPERT_LAW = "source_ast_resident_rows_v1"
HYBRID_EXPERT_LAW = "source_ast_hybrid_rows_v1"
_EXACT_PUBLICATION_SEAL = object()


def _snapshot(cache, profile=None, source_revision=None):
    from ..runtime.cow_cache import snapshot_prompt_cache_descriptors
    private = any(getattr(row,"ast_private_profile",None) for row in cache)
    if private:
        if (profile is None or source_revision is None
                or any(getattr(row,"ast_private_profile",None)!=profile
                       or getattr(row,"ast_private_revision",None)!=source_revision for row in cache)):
            raise ValueError("Private Source AST checkpoint belongs to another numerical profile")
        for row in cache:
            if row.pending:raise ValueError("Private Source AST recovery checkpoint is incomplete")
            row.validate_state()
    # This is explicit private recovery, never a publication operation. The
    # descriptor copier retains the private numerical marker on every layer.
    return snapshot_prompt_cache_descriptors(cache,require_committed=not private)[0]


class _LayerView:
    """Diagnostic view; full state is built only on explicit inspection."""
    def __init__(self, descriptor, node, profile=PROFILE, source_revision=None):
        self.descriptor, self.node, self.profile = descriptor, node, profile
        self.source_revision = source_revision

    def __getattr__(self, name):
        row = self.descriptor.materialize(self.node)
        row.ast_private_profile = self.profile
        row.ast_private_revision = self.source_revision
        return getattr(row, name)


class BatchedSourceASTTransaction:
    """Revision-bound selected-path materialization, no speculative donation."""
    def __init__(self, model, live, base, segments, revision, profile=PROFILE, private_revision=None):
        from .deepseek_v41_ast_target import PROTOCOL, PARENTS, _paths, _stamp
        self.provider_protocol = PROTOCOL
        self.model, self.live, self.base, self.segments = model, live, base, segments
        self.provider_revision_token = revision
        self.paths, self.original_stamp = _paths(PARENTS), _stamp(live)
        self.model_identity = model.identity
        self.profile = profile
        self.private_source_revision = private_revision
        self.cow_owner = getattr(live, "cow_owner", None)
        self.cow_generation = getattr(live, "cow_generation", None)
        self.closed, self.complete = False, False
        self._exact_publication_admitted = False
        self.prepared_nodes = 0

    def check_live(self):
        """Check request/plane authority without synchronizing device data."""
        from .deepseek_v41_ast_target import _stamp
        if (self.closed or self.model.identity != self.model_identity
                or getattr(self.model,"_mlx2_ds41_ast_active",None) is not self
                or _stamp(self.live) != self.original_stamp
                or getattr(self.live, "cow_owner", None) is not self.cow_owner
                or getattr(self.live, "cow_generation", None) != self.cow_generation
                or getattr(self.live, "closed", False)):
            raise ValueError("Batched Source AST frontier closed, incomplete or changed")
        if self.cow_owner is not None and not self.cow_owner.branchable(expected_generation=self.cow_generation):
            raise ValueError("Batched Source AST APCv2 prefix generation or segment changed")
        ready=getattr(self.model.weights,"_ready",None)
        if ready is not None:ready()
        return True

    def _validate(self, path):
        self.check_live()
        if not self.complete:
            raise ValueError("Batched Source AST frontier is incomplete")
        path = tuple(path)
        if path and (any(type(i) is not int for i in path)
                     or not 0 <= path[-1] < 16 or self.paths[path[-1]] != path):
            raise ValueError("Batched Source AST requires one root-connected path")
        return path

    def _materialize(self, path):
        path = self._validate(path)
        result = (_snapshot(self.base,self.profile,self.private_source_revision) if not path else
                  [segment.materialize(path[-1]) for segment in self.segments])
        self.prepared_nodes += bool(path)
        return result

    def diagnostic_cache(self, path):
        """Private inspection/continuation; APCv2 donation remains forbidden."""
        result = self._materialize(path)
        for row in result:
            row.ast_private_profile = self.profile
            row.ast_private_revision = self.private_source_revision
        return result

    def checkpoint_manifest(self, path):
        """Address private checkpoints through their actual APCv2 parent."""
        from .deepseek_v41_ast_segment_manifest import checkpoint_manifest
        return checkpoint_manifest(self,path)

    def _admit_exact_publication(self, seal, revision):
        if (seal is not _EXACT_PUBLICATION_SEAL or revision != self.provider_revision_token
                or not self.complete or self.closed
                or any(getattr(row,"ast_private_profile",None) for row in self.base)):
            raise ValueError("Batched Source AST exact publication binding differs")
        self._exact_publication_admitted = True

    def prepared_cache(self, path):
        from ..runtime.checkpoint_publication import require_checkpoint_publishable
        if not self._exact_publication_admitted:
            raise ValueError("Native batched Source state has no exact publication admission; use private diagnostics")
        result = self._materialize(path)
        require_checkpoint_publishable(result)
        return result

    def commit_paths(self, paths):
        if len(paths) != 1:
            raise ValueError("Batched Source AST owns exactly one request")
        result = self.prepared_cache(paths[0])
        try:
            mx.eval([row.state for row in result])
        except BaseException:
            self.model._mlx2_ds41_ast_poisoned = True
            self.model._mlx2_ds41_ast_failed_roots = (result, self.base, self.segments, self)
            self.abort()
            raise
        self.abort()
        return [result]

    def abort(self):
        if not self.closed:
            self.closed = True
            self.base, self.segments = [], []
            if getattr(self.model, "_mlx2_ds41_ast_active", None) is self:
                self.model._mlx2_ds41_ast_active = None


@contextmanager
def _completed_provider_window(provider_context,bank_scope):
    """Defer ONLY a proved private miss beyond the provider's final fence.

    Native windows poison their provider for any exception through their body.
    A drained cohort miss is a recoverable target outcome, not provider damage.
    All unknown/status/nonfinite/fence errors retain the normal error path.
    The caller keeps the outer bank lease until this context returns/raises.
    """
    from .deepseek_v41_ast_batched_moe import ResidentCohortMiss
    state=dict(called=False,miss=None)
    with provider_context:
        def finish(logits,features,roots):
            if state['called']:raise RuntimeError('Batched provider completion repeated')
            state['called']=True
            if bank_scope is not None:
                try:bank_scope.complete(logits,features,roots)
                except ResidentCohortMiss as error:
                    if (not bank_scope.recover_misses or not bank_scope.drained or bank_scope.poisoned
                            or not error.drained):raise
                    state['miss']=error
            else:mx.eval(logits,features,roots)
            if not bool(mx.all(mx.isfinite(logits)))or(features is not None and not bool(mx.all(mx.isfinite(features)))):
                raise ValueError('Batched Source AST produced nonfinite outputs')
        yield finish
        if not state['called']:raise RuntimeError('Batched provider completion is required')
    # The provider/completion window has now retired every epoch and completed
    # its normal final fence. No miss crosses its exception/poison boundary.
    if state['miss']is not None:raise state['miss']


def _execute_batched_tree_once(model, cache, anchor, paths3x5, capture_layers=(37, 38, 39), *,
                         cpu_reference=False, expert_law=RESIDENT_EXPERT_LAW, forward_law=FORWARD_LAW,consumer_law=None):
    from .deepseek_v41_ast_target import (
        PARENTS, PROTOCOL, SourceASTResult, source_revision_token,
    )
    if forward_law not in PROFILES:
        raise ValueError("Unsupported batched Source forward law")
    profile = PROFILES[forward_law]
    if forward_law == M1_FORWARD_LAW:
        from .deepseek_v41_ast_batched_dense_exact import (
            shared_expert_tile, head_tile, router_tile, engram_tile,
        )
        from .deepseek_v41_ast_batched_attention_exact import attention_half_tile
    else:
        from .deepseek_v41_ast_batched_dense import (
            shared_expert_tile, head_tile, router_tile, engram_tile,
        )
        from .deepseek_v41_ast_batched_attention import attention_half_tile
    c, weights = model.args, model.weights
    consumer_law=expert_law if consumer_law is None else consumer_law
    if ((expert_law==HYBRID_EXPERT_LAW and consumer_law not in (RESIDENT_EXPERT_LAW,'source_ast_m1_rows_v1'))
            or (expert_law!=HYBRID_EXPERT_LAW and consumer_law!=expert_law)):
        raise ValueError('Source AST logical/consumer expert law differs')
    if expert_law not in ("ordinary_m16_v1", "source_ast_m1_rows_v1", RESIDENT_EXPERT_LAW,HYBRID_EXPERT_LAW):
        raise ValueError("Batched Source AST requires a batched expert law")
    if (not cpu_reference and (c.hidden_size, c.num_hidden_layers,
            c.n_routed_experts, c.num_experts_per_tok) != (5120, 40, 384, 6)):
        raise ValueError("Batched Source AST requires H5120/L40/E384/top6")
    if cpu_reference and mx.default_device() != mx.cpu:
        raise ValueError("Synthetic batched Source AST requires explicit CPU")
    paths3x5 = tuple(tuple(row) for row in paths3x5)
    tokens = (anchor,) + tuple(token for row in paths3x5 for token in row)
    if (len(paths3x5) != 3 or any(len(row) != 5 for row in paths3x5)
            or any(type(token) is not int or not 0 <= token < c.vocab_size for token in tokens)):
        raise ValueError("Batched Source AST requires exact3x5 tokens")
    layers = tuple(capture_layers)
    if (tuple(sorted(set(layers))) != layers or any(type(i) is not int or
            not 0 <= i < c.num_hidden_layers for i in layers)):
        raise ValueError("Batched Source AST capture layers must be ordered unique")
    if not cpu_reference and layers != (37, 38, 39):
        raise ValueError("Source DSpark requires exact attention-input taps37/38/39")
    if (getattr(model, "_mlx2_ds41_ast_active", None) is not None
            or getattr(model, "_mlx2_ds41_ast_poisoned", False) or model._weight_executions):
        raise ValueError("Batched Source AST target is active or poisoned")
    if getattr(weights, "source_row_packet_policy", "reference") != "reference":
        raise ValueError("Batched Source AST cannot consume contiguous prefill packets")
    inputs = mx.array([tokens], mx.int32)
    live, _ = model.validate_forward(inputs[:, :1], cache, 0)
    if live[0].offset + 6 > model.context_limit:
        raise ValueError("Batched Source AST ancestry exceeds context")
    revision = source_revision_token(expert_law,forward_law)
    private_revision = source_revision_token(
        "source_ast_m1_rows_v1" if expert_law in (RESIDENT_EXPERT_LAW,HYBRID_EXPERT_LAW) else expert_law, forward_law)
    base = _snapshot(live,profile,private_revision)
    segments, shared = [None] * len(model.layers), [{} for _ in range(16)]
    transaction = BatchedSourceASTTransaction(model, live, base, segments, revision, profile,private_revision)
    model._mlx2_ds41_ast_active = transaction
    model._weight_executions += 1
    host_before = getattr(weights, "source_route_host_materializations", 0)
    roots = [[getattr(row, name) for name in
              ("keys", "buf_kv", "buf_gate", "_pooled", "_index_keys", "hash_tail")]
             for row in base]
    taps, hashes, tails = [], [], []
    scope = getattr(weights, "execution_scope_for_rows", None)
    bank = nullcontext()
    attention_ownership = nullcontext()
    try:
        if forward_law == M1_FORWARD_LAW:
            from .deepseek_v41_ast_batched_attention_exact import owned_stock_splitk
            attention_ownership = owned_stock_splitk(transaction.check_live)
        if consumer_law == RESIDENT_EXPERT_LAW:
            from .deepseek_v41_ast_batched_moe import resident_scope
            if expert_law==HYBRID_EXPERT_LAW:
                manager=model._source_ast_bank_manager
                bank=resident_scope(weights,ownercheck=manager.check_live,bank=manager.bank,
                    cancelled=manager.cancelled,recover_misses=True)
            else:bank = resident_scope(weights)
        if model.hasher is not None:
            initial = np.asarray(base[0].hash_tail)
            for node, parent in enumerate(PARENTS):
                hashed, tail = model.hasher(np.asarray([tokens[node]], np.int64),
                                            initial if parent < 0 else tails[parent])
                hashes.append(hashed)
                tails.append(tail)
        with attention_ownership, bank as bank_scope:
            with _completed_provider_window((scope(16) if scope is not None else nullcontext()),bank_scope)as finish:
                with model._weight_scope():
                    embedding = model.round(weights.embedding(inputs.astype(mx.uint32)))
                    h = mx.broadcast_to(embedding[:, :, None], (1, 16, c.hc_mult, c.hidden_size))
                    pre = mx.broadcast_to(mx.array([1.] + [0.] * (c.hc_mult-1)), (1, 16, c.hc_mult))
                    model._weight_eval(h, pre)
                for layer, block in enumerate(model.layers):
                    with model._weight_scope():
                        if layer in c.engram_layer_ids:
                            index = c.engram_layer_ids.index(layer)
                            grouped = np.concatenate([value[:, index] for value in hashes])
                            rows = mx.array(model.engram_tables[layer].read(grouped))
                            h = engram_tile(model, h, layer, rows)
                        if layer in layers:
                            taps.append(mx.mean(h, axis=-2))
                        h, x, pre, post, comb, descriptor = attention_half_tile(
                            block, h, pre, base, segments, shared)
                        segments[layer] = descriptor
                        if layer == 0 and tails:
                            descriptor.node_hash_tails = tuple(mx.array(tail, mx.int32) for tail in tails)
                        ids, scores = router_tile(block, x)
                        shared_out = shared_expert_tile(block, x)
                        if consumer_law == RESIDENT_EXPERT_LAW:
                            from .deepseek_v41_ast_batched_moe import moe_tile
                            routed = moe_tile(weights, x.reshape(16, c.hidden_size),
                                              ids.reshape(16, c.num_experts_per_tok),
                                              scores.reshape(16, c.num_experts_per_tok), block.prefix[:-1])
                        elif consumer_law == "source_ast_m1_rows_v1":
                            from .deepseek_v41_ast_moe import moe
                            routed = moe(weights, x.reshape(16, c.hidden_size),
                                         ids.reshape(16, c.num_experts_per_tok),
                                         scores.reshape(16, c.num_experts_per_tok), block.prefix[:-1])
                        else:
                            routed = weights.moe(x.reshape(16, c.hidden_size),
                                                 ids.reshape(16, c.num_experts_per_tok),
                                                 scores.reshape(16, c.num_experts_per_tok), block.prefix[:-1])
                        h = block.finish(h, routed.reshape(1, 16, c.hidden_size), shared_out, post, comb)
                        roots = [h, pre, taps, shared,
                                 [segment.evaluation_roots() for segment in segments if segment is not None]]
                        model._weight_eval(*roots)
                with model._weight_scope():
                    logits = head_tile(model, h, pre)
                    features = mx.concatenate(taps, -1) if taps else None
                    model._weight_eval(logits, features)
                finish(logits,features,roots)
        transaction.complete = True
        receipt = dict(protocol=PROTOCOL, numerical_profile=profile, expert_law=expert_law,
                       forward_law=forward_law,expert_execution_law=consumer_law, qualified=False, defaulted=False,
                       checkpoint=model.identity, source_revision_token=revision,
                       tensor_tile_rows=16, physical_target_invocations=1, serial_whole_model_calls=0,
                       layer_traversals=1, grouped_moe_calls=len(model.layers),
                       expert_projection_rows_per_layer=16, attention_rows_per_call=16,
                       attention_calls=len(model.layers), shared_expert_calls=len(model.layers),
                       head_calls=1, capture_layers=layers, parents=PARENTS,
                       selected_kv_preparation=dict(
                           policy=getattr(model, 'source_ast_kv_preparation_policy', 'reference'),
                           builds=sum(getattr(segment, '_selected_prepare_builds', 0) for segment in segments),
                           hits=sum(getattr(segment, '_selected_prepare_hits', 0) for segment in segments),
                           built_payload_bytes=sum(getattr(segment, '_selected_prepare_bytes', 0) for segment in segments),
                           measured_device_traffic=False),
                       source_route_host_materializations=getattr(weights, "source_route_host_materializations", 0)-host_before,
                       source_route_host_policy="gpu_resident_tile" if consumer_law == RESIDENT_EXPERT_LAW else "ordinary_grouped",
                       zero_host_expert_ids_claimed=consumer_law == RESIDENT_EXPERT_LAW,
                       apc_publication=False, resident_bank_decode=False,
                       resident_bank_tile=consumer_law == RESIDENT_EXPERT_LAW,
                       prefix_ownership="apcv2_descriptor_snapshot",
                       private_state_revision=private_revision,
                       layer_segments=[segment.receipt() for segment in segments],
                       materialized_node_frontiers_during_forward=0,
                       apcv2_lineage_bound=transaction.cow_owner is not None)
        frontiers = tuple(tuple(_LayerView(segment, node, profile,private_revision) for segment in segments) for node in range(16))
        return SourceASTResult(logits, features, frontiers, transaction, receipt)
    except BaseException:
        try:
            mx.eval(roots)
        except BaseException:
            model._mlx2_ds41_ast_poisoned = True
            model._mlx2_ds41_ast_failed_roots = (roots, base, segments, transaction)
        transaction.abort()
        raise
    finally:
        model._weight_executions -= 1


def execute_batched_tree(model,cache,anchor,paths3x5,capture_layers=(37,38,39),*,
                         cpu_reference=False,expert_law=RESIDENT_EXPERT_LAW,forward_law=FORWARD_LAW):
    from .deepseek_v41_ast_target import _require_reference_cache
    _require_reference_cache(model, cache)
    if expert_law!=HYBRID_EXPERT_LAW:
        return _execute_batched_tree_once(model,cache,anchor,paths3x5,capture_layers,
            cpu_reference=cpu_reference,expert_law=expert_law,forward_law=forward_law)
    if forward_law!=M1_FORWARD_LAW:raise ValueError('Hybrid Source AST requires fully_batched_m1_v1')
    manager=getattr(model,'_source_ast_bank_manager',None)
    if manager is None:raise ValueError('Hybrid Source AST requires its bound adapter bank manager')
    def attempt(consumer):
        return _execute_batched_tree_once(model,cache,anchor,paths3x5,capture_layers,
            cpu_reference=cpu_reference,expert_law=expert_law,forward_law=forward_law,consumer_law=consumer)
    return manager.execute(model,attempt)
