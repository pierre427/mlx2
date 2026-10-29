# SPDX-License-Identifier: MIT
"""Default-off Source AST: one layer traversal, 16 private ancestry rows.

Width-one attention/dense/QAT remains model-owned. Only independent routed
expert projections are grouped into M16. This numerical profile needs actual
Source verification before an AST planner may admit it as an exact backend.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

PROTOCOL = "mlx2.deepseek-v41-source-ast-target.v1"
PROFILE = "source_ast_b1_attention_m16_experts_v1"
M1_PROFILE = "source_ast_b1_attention_m1_expert_occurrences_v1"
EXPERT_LAWS = ("ordinary_m16_v1", "source_ast_m1_rows_v1")
FORWARD_LAWS = ("reference", "fully_batched_v1", "fully_batched_m1_v1")
RESIDENT_EXPERT_LAW = "source_ast_resident_rows_v1"
HYBRID_EXPERT_LAW = "source_ast_hybrid_rows_v1"
PARENTS = (-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14)


def _require_reference_cache(model, cache=()):
    if (getattr(model, "kv_storage_policy", "reference") != "reference"
            or any(hasattr(row, "_packed_local") for row in (() if cache is None else cache))):
        raise ValueError("Source AST requires reference FP32 cache state")


def _clone(row):
    from ..runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    return DeepSeekV41LayerCache.from_state(row.state, row.meta_state, xp=mx)


def _stamp(cache):
    # pooled/index_keys properties manufacture temporary visibility slices.
    # Bind the retained storage owners, not those ephemeral Python wrappers.
    return tuple((row.meta_state, row._pool_count, row._capacity,getattr(row,"ast_private_revision",None),
                  tuple(id(getattr(row, name)) for name in
                        ("keys", "buf_kv", "buf_gate", "_pooled", "_index_keys", "hash_tail")))
                 for row in cache)


def _paths(parents):
    paths = []
    for node, parent in enumerate(parents):
        paths.append((node,) if parent < 0 else paths[parent] + (node,))
    return tuple(paths)


def _canonical_rows(parents):
    """Map a topological caller order of exactly three five-node chains."""
    parents = tuple(parents)
    if (len(parents) != 16 or any(type(p) is not int for p in parents)
            or parents[0] != -1 or any(not 0 <= p < node for node, p in enumerate(parents[1:], 1))):
        raise ValueError("Source AST requires topological exact3x5 ancestry")
    children = [[] for _ in parents]
    for node, parent in enumerate(parents[1:], 1):
        children[parent].append(node)
    if len(children[0]) != 3 or any(len(row) > 1 for row in children[1:]):
        raise ValueError("Source AST ancestry requires three independent five-node chains")
    canonical = [0]
    for first in children[0]:
        chain, node = [], first
        while True:
            chain.append(node)
            if not children[node]:
                break
            node = children[node][0]
        if len(chain) != 5:
            raise ValueError("Source AST ancestry requires three independent five-node chains")
        canonical.extend(chain)
    if len(set(canonical)) != 16:
        raise ValueError("Source AST exact3x5 ancestry does not cover sixteen nodes")
    return tuple(canonical)


class ReindexedSourceASTTransaction:
    """Caller-index view of the unchanged fixed-order private transaction."""
    provider_protocol = PROTOCOL

    def __init__(self, inner, parents, canonical_to_caller):
        self.inner, self.paths = inner, _paths(parents)
        self.caller_to_canonical = tuple(canonical_to_caller.index(i) for i in range(16))

    @property
    def provider_revision_token(self):
        return self.inner.provider_revision_token

    @property
    def closed(self):
        return self.inner.closed

    def _translate(self, path):
        path = tuple(path)
        if path and (any(type(i) is not int for i in path)
                     or not 0 <= path[-1] < 16 or self.paths[path[-1]] != path):
            raise ValueError("Source AST publication requires one root-connected caller path")
        return tuple(self.caller_to_canonical[i] for i in path)

    def prepared_cache(self, path):
        return self.inner.prepared_cache(self._translate(path))

    def commit_paths(self, paths):
        if len(paths) != 1:
            raise ValueError("Source AST owns exactly one request")
        return self.inner.commit_paths([self._translate(paths[0])])

    def abort(self):
        self.inner.abort()


class SourceASTTransaction:
    provider_protocol = PROTOCOL

    def __init__(self, model, live, base, frontiers, revision):
        self.model, self.live, self.base, self.frontiers = model, live, base, frontiers
        self.original_stamp, self.paths = _stamp(live), _paths(PARENTS)
        self.model_identity = model.identity
        self.provider_revision_token = revision
        self.closed = False

    def prepared_cache(self, path):
        if (self.closed or self.model.identity != self.model_identity
                or _stamp(self.live) != self.original_stamp):
            raise ValueError("Source AST transaction closed or live frontier changed")
        path = tuple(path)
        if path and (any(type(i) is not int for i in path)
                     or not 0 <= path[-1] < 16 or self.paths[path[-1]] != path):
            raise ValueError("Source AST publication requires one root-connected path")
        source = self.base if not path else self.frontiers[path[-1]]
        return [_clone(row) for row in source]

    def commit_paths(self, paths):
        if len(paths) != 1:
            raise ValueError("Source AST owns exactly one request")
        result = self.prepared_cache(paths[0])
        try:
            mx.eval([row.state for row in result])
        except BaseException:
            self.model._mlx2_ds41_ast_poisoned = True
            self.model._mlx2_ds41_ast_failed_roots = (result, self.base, self.frontiers, self)
            self.abort()
            raise
        self.abort()
        return [result]

    def abort(self):
        if not self.closed:
            self.closed = True
            self.frontiers = self.base = []
            if getattr(self.model, "_mlx2_ds41_ast_active", None) is self:
                self.model._mlx2_ds41_ast_active = None


@dataclass(frozen=True)
class SourceASTResult:
    logits: object
    features: object
    frontiers: tuple
    transaction: SourceASTTransaction
    receipt: dict


def implementation_identity():
    root = Path(__file__).resolve().parents[3]
    names = ("src/mlx2/adapters/deepseek_v41_ast_target.py",
             "src/mlx2/runtime/models/deepseek_v41.py",
             "src/mlx2/runtime/models/deepseek_v41_cache.py",
             "src/mlx2/adapters/deepseek_v41_source.py",
             "src/mlx2/adapters/deepseek_v41_source_canonical_prefill.py",
             "src/mlx2/adapters/deepseek_v41_source_checkpoint_resources.py",
             "src/mlx2/adapters/deepseek_v41_ast_bank_manager.py",
             "src/mlx2/adapters/deepseek_v41_source_bank.py",
             "src/mlx2/adapters/deepseek_v41_source_prefill_bank.py",
             "src/mlx2/adapters/deepseek_v41_source_bank_session.py",
             "src/mlx2/adapters/deepseek_v41_mlx_weights.py",
             "src/mlx2/adapters/deepseek_v41_source_completion.py",
             "src/mlx2/adapters/deepseek_v41_source_weights.py",
             "src/mlx2/runtime/models/deepseek_v41_quant.py",
             "src/mlx2/runtime/models/deepseek_v41_source_hc.py",
             "src/mlx2/runtime/models/deepseek_v41_fused.py",
             "src/mlx2/runtime/models/served_exp.py",
             "src/mlx2/adapters/deepseek_v41_source_math.py",
             "src/mlx2/runtime/gguf/source_gate_up_fusion.py",
             "src/mlx2/runtime/gguf/source_down_fusion.py",
             "src/mlx2/runtime/gguf/source_middle_fusion.py",
             "src/mlx2/adapters/deepseek_v41_ast_moe.py",
             "src/mlx2/runtime/gguf/source_ast_m1_expert.py",
             "src/mlx2/runtime/gguf/source_ast_down_rows.py",
             "src/mlx2/adapters/deepseek_v41_ast_batched_target.py",
             "src/mlx2/adapters/deepseek_v41_ast_batched_dense.py",
             "src/mlx2/adapters/deepseek_v41_ast_batched_dense_exact.py",
             "src/mlx2/adapters/deepseek_v41_ast_batched_attention.py",
             "src/mlx2/adapters/deepseek_v41_ast_batched_attention_exact.py",
             "src/mlx2/adapters/deepseek_v41_ast_batched_index_exact.py",
             "src/mlx2/adapters/deepseek_v41_ast_kv_preparation.py",
             "src/mlx2/runtime/gguf/source_ast_batched_index_matmul.py",
             "src/mlx2/adapters/deepseek_v41_ast_segment_manifest.py",
             "src/mlx2/adapters/deepseek_v41_ast_batched_moe.py",
             "src/mlx2/runtime/gguf/source_ast_batched_moe.py",
             "src/mlx2/runtime/gguf/source_ast_m1_dense_rows.py",
             "src/mlx2/runtime/gguf/source_ast_batched_splitk.py",
             "src/mlx2/runtime/gguf/source_ast_head_axis_matmul.py",
             "src/mlx2/adapters/deepseek_v41_dspark_ast.py",
             "src/mlx2/runtime/cow_cache.py",
             "src/mlx2/runtime/cache_operations.py",
             "src/mlx2/runtime/expert_residency_hints.py",
             "src/mlx2/runtime/checkpoint_publication.py")
    return tuple((name, hashlib.sha256((root / name).read_bytes()).hexdigest()) for name in names)


def source_revision_token(expert_law,forward_law):
    closure=None
    if forward_law=="fully_batched_m1_v1":
        from ..runtime.gguf.source_ast_batched_splitk import implementation_identity as selected_identity
        from ..runtime.gguf.source_ast_batched_index_matmul import implementation_identity as index_identity
        from ..runtime.models.deepseek_v41_source_hc import installed_receipt
        # The existing installed audit pins both libmlx.dylib and
        # mlx.metallib plus stock reduction/exp headers. It performs no GPU
        # work and caches verified immutable process-build facts.
        closure=(selected_identity(),index_identity(),installed_receipt())
    return hashlib.sha256(repr((implementation_identity(),expert_law,forward_law,closure)).encode()).hexdigest()


def execute_tree(model, cache, anchor, paths3x5, capture_layers=(37, 38, 39), *, cpu_reference=False,
                 expert_law="ordinary_m16_v1", forward_law="reference"):
    """Consume anchor+3x5 proposed tokens without mutating the caller's cache.

    ``frontiers[n]`` is the complete state after consuming node n's ancestry.
    A target decision on node n predicts the next token, as in ordinary decode.
    Caller must commit only its target-accepted root-connected input path.
    """
    _require_reference_cache(model, cache)
    if forward_law not in FORWARD_LAWS:
        raise ValueError("Unsupported Source AST forward law")
    if forward_law != "reference":
        from .deepseek_v41_ast_batched_target import execute_batched_tree
        return execute_batched_tree(model, cache, anchor, paths3x5, capture_layers,
                                    cpu_reference=cpu_reference, expert_law=expert_law, forward_law=forward_law)
    c = model.args
    if expert_law not in EXPERT_LAWS:
        raise ValueError("Unsupported Source AST expert arithmetic law")
    if (not cpu_reference and (c.hidden_size, c.num_hidden_layers,
            c.n_routed_experts, c.num_experts_per_tok) != (5120, 40, 384, 6)):
        raise ValueError("Source AST requires H5120/L40/E384/top6")
    if cpu_reference and mx.default_device() != mx.cpu:
        raise ValueError("synthetic Source AST reference requires explicit CPU")
    paths3x5 = tuple(tuple(path) for path in paths3x5)
    tokens = (anchor,) + tuple(token for path in paths3x5 for token in path)
    if (len(paths3x5) != 3 or any(len(path) != 5 for path in paths3x5)
            or any(type(token) is not int or not 0 <= token < c.vocab_size for token in tokens)):
        raise ValueError("Source AST requires one anchor and exactly three five-token paths")
    layers = tuple(capture_layers)
    if (len(set(layers)) != len(layers) or tuple(sorted(layers)) != layers
            or any(type(i) is not int or not 0 <= i < c.num_hidden_layers for i in layers)):
        raise ValueError("Source AST capture layers must be ordered unique target inputs")
    if not cpu_reference and layers != (37, 38, 39):
        raise ValueError("Source DSpark requires exact attention-input taps37/38/39")
    if (getattr(model, "_mlx2_ds41_ast_active", None) is not None
            or getattr(model, "_mlx2_ds41_ast_poisoned", False)
            or model._weight_executions):
        raise ValueError("Source AST target is active or poisoned")
    weights = model.weights
    expert_counts_before = dict(getattr(weights, "scheduled_expert_calls", {}))
    host_routes_before = getattr(weights, "source_route_host_materializations", 0)
    session = getattr(weights, "source_resident_bank_session", None)
    if session is not None and (session.active or getattr(session, "replaying", False)):
        raise ValueError("Source AST M16 cannot enter an active B1 resident-bank epoch")
    if getattr(weights, "source_row_packet_policy", "reference") != "reference":
        raise ValueError("Source AST does not support contiguous prefill row packets")
    inputs = mx.array([tokens], mx.int32)
    live, _ = model.validate_forward(inputs[:, :1], cache, 0)
    if live[0].offset + 6 > model.context_limit:
        raise ValueError("Source AST ancestry exceeds target context")
    base = [_clone(row) for row in live]
    frontiers = [[None] * len(base) for _ in tokens]
    revision = source_revision_token(expert_law,forward_law)
    transaction = SourceASTTransaction(model, live, base, frontiers, revision)
    model._mlx2_ds41_ast_active = transaction
    model._weight_executions += 1
    hidden, pre, taps, shared = [], [], [[] for _ in tokens], [{} for _ in tokens]
    hashes, tails = [], []
    stream = getattr(weights, "expert_stream", None)
    if stream is not None and getattr(stream, "atlas", None) is not None:
        transaction.abort()
        model._weight_executions -= 1
        raise ValueError("Source AST requires ancestry-aware atlas context; unsupported")
    scope = getattr(weights, "execution_scope_for_rows", None)
    # Keep Python owners even when graph construction fails between evaluations.
    roots = [hidden, pre, taps, shared]
    try:
        execution = scope(16) if scope is not None else nullcontext()
        if model.hasher is not None:
            initial = np.asarray(base[0].hash_tail)
            for node, parent in enumerate(PARENTS):
                hashed, tail = model.hasher(np.asarray([tokens[node]], np.int64),
                                            initial if parent < 0 else tails[parent])
                hashes.append(hashed)
                tails.append(tail)
        with execution:
            with model._weight_scope():
                for node in range(16):
                    x = model.round(weights.embedding(inputs[:, node:node+1].astype(mx.uint32)))
                    hidden.append(mx.broadcast_to(x[:, :, None], (1, 1, c.hc_mult, c.hidden_size)))
                    pre.append(mx.broadcast_to(mx.array([1.] + [0.] * (c.hc_mult-1)), (1, 1, c.hc_mult)))
                model._weight_eval(hidden, pre)
            for layer, block in enumerate(model.layers):
                with model._weight_scope():
                    staged, routes, scores = [], [], []
                    engram_rows = None
                    if layer in c.engram_layer_ids:
                        index = c.engram_layer_ids.index(layer)
                        grouped_hashes = np.concatenate([v[:, index] for v in hashes])
                        # EngramRows owns exactly one pending lookup window.
                        # Consume the same grouped16 window once, then slice
                        # its decoded rows; individual reads cannot replace it.
                        engram_rows = mx.array(model.engram_tables[layer].read(grouped_hashes))
                    roots = [hidden, pre, taps, shared, staged, routes, scores, engram_rows]
                    for node, parent in enumerate(PARENTS):
                        local = _clone(base[layer] if parent < 0 else frontiers[parent][layer])
                        local._in_forward = True
                        frontiers[node][layer] = local
                        # Cache owners are retained separately in the failure
                        # path; the current roots include every partial node.
                        if engram_rows is not None:
                            hidden[node] = model.engram(hidden[node], layer, None,
                                                       rows=engram_rows[node:node+1])
                        if layer in layers:
                            taps[node].append(model.round(mx.mean(hidden[node], axis=-2)))
                        h, x, pre[node], post, comb = block.attention_half(
                            hidden[node], pre[node], local, frontiers[node], shared[node])
                        if layer == 0 and tails:
                            local.hash_tail = mx.array(tails[node], mx.int32)
                        local._in_forward = False
                        local.validate_state()
                        staged.append((h, x, post, comb))
                        ids, score = block.router(x)
                        routes.append(ids.reshape(1, c.num_experts_per_tok))
                        scores.append(score.reshape(1, c.num_experts_per_tok))
                    x = mx.concatenate([item[1].reshape(1, c.hidden_size) for item in staged])
                    shared_out = [block.shared_expert(item[1]) for item in staged]
                    roots.extend((x, shared_out,
                        [row.state for frontier in frontiers for row in frontier if row is not None]))
                    if expert_law == "source_ast_m1_rows_v1":
                        from .deepseek_v41_ast_moe import moe
                        routed = moe(weights, x, mx.concatenate(routes), mx.concatenate(scores), block.prefix[:-1])
                    else:
                        routed = weights.moe(x, mx.concatenate(routes), mx.concatenate(scores), block.prefix[:-1])
                    for node, (h, _, post, comb) in enumerate(staged):
                        hidden[node] = block.finish(h, routed[node:node+1][None], shared_out[node], post, comb)
                    roots = [hidden, pre, taps, shared,
                             [row.state for frontier in frontiers for row in frontier if row is not None]]
                    model._weight_eval(*roots)
            with model._weight_scope():
                logits = mx.concatenate([model.linear(model.norm(model.collapse(h, p), model.norm_weight),
                                         "output.weight", rounded=False) for h, p in zip(hidden, pre)], 1)
                features = mx.concatenate([mx.concatenate(tap, -1) for tap in taps], 1) if layers else None
                model._weight_eval(logits, features)
            # One completed private tile before any selected-path publication.
            mx.eval(logits, features, roots)
        for frontier in frontiers:
            for row in frontier:
                row.cache_operations_contract.validate_publication(row)
        finite = mx.all(mx.isfinite(logits))
        if features is not None:
            finite = finite & mx.all(mx.isfinite(features))
        if not bool(finite):
            raise ValueError("Source AST produced nonfinite private outputs")
        numerical_profile = M1_PROFILE if expert_law == "source_ast_m1_rows_v1" else PROFILE
        receipt = dict(protocol=PROTOCOL, numerical_profile=numerical_profile, expert_law=expert_law,
                       qualified=False, defaulted=False,
                       checkpoint=model.identity, source_revision_token=revision, tensor_tile_rows=16,
                       physical_target_invocations=1, serial_whole_model_calls=0,
                       layer_traversals=1, grouped_moe_calls=len(model.layers),
                       expert_projection_rows_per_layer=16, attention_rows_per_call=1,
                       attention_calls=len(model.layers)*16, capture_layers=layers,
                       source_route_host_counter_present=hasattr(weights, "source_route_host_materializations"),
                       source_route_host_materializations=getattr(weights, "source_route_host_materializations", 0)-host_routes_before,
                       source_route_host_policy="cpu_ids_once_per_grouped_layer",
                       zero_host_expert_ids_claimed=False,
                       parents=PARENTS, apc_publication=False, resident_bank_decode=False,
                       expert_law_counts={key: value-expert_counts_before.get(key, 0)
                           for key, value in getattr(weights, "scheduled_expert_calls", {}).items()
                           if key.startswith("source_ast_m1_")})
        return SourceASTResult(logits, features, tuple(tuple(row) for row in frontiers), transaction, receipt)
    except BaseException:
        try:
            mx.eval(roots, [row.state for frontier in frontiers for row in frontier if row is not None])
        except BaseException:
            model._mlx2_ds41_ast_poisoned = True
            model._mlx2_ds41_ast_failed_roots = (roots, base, frontiers, transaction)
        transaction.abort()
        raise
    finally:
        model._weight_executions -= 1


class SourceASTTargetTreeExecutor:
    """Generic provider seam; numerical AST admission stays explicitly refused."""
    protocol = PROTOCOL
    maximum_cohort_width = 1

    def __init__(self, model, *, cpu_reference=False, expert_law="ordinary_m16_v1", forward_law="reference"):
        _require_reference_cache(model)
        if (forward_law not in FORWARD_LAWS or expert_law not in EXPERT_LAWS + (RESIDENT_EXPERT_LAW,HYBRID_EXPERT_LAW)
                or (expert_law == RESIDENT_EXPERT_LAW and forward_law == "reference")
                or (expert_law == HYBRID_EXPERT_LAW and forward_law != "fully_batched_m1_v1")):
            raise ValueError("Unsupported Source AST expert arithmetic law")
        self.expert_law, self.forward_law = expert_law, forward_law
        from .deepseek_v41_ast_batched_target import PROFILES
        self.numerical_profile = (PROFILES[forward_law] if forward_law != "reference" else
                                  M1_PROFILE if expert_law == "source_ast_m1_rows_v1" else PROFILE)
        self.model, self.cpu_reference, self.launches = model, cpu_reference, 0
        self.physical_invocations=0;self.failed_private_invocations=0
        self.model_identity = model.identity
        self.ast_oracle, self.ast_input_widths = None, ()
        self._exact_publication_binding = None
        self.source_identity = implementation_identity()
        self.source_revision_token = source_revision_token(expert_law,forward_law)

    def validate_revision(self):
        if (implementation_identity() != self.source_identity
                or self.source_revision_token!=source_revision_token(self.expert_law,self.forward_law)):
            raise ValueError("Source AST implementation changed")

    def validate_cohort(self, width, configured_limit):
        if width != 1 or configured_limit < 1:
            raise ValueError("Source AST supports one request and one16-row tile")

    def forward(self, model, inputs, parents, cache, layers, *, cached=True):
        self.validate_revision()
        if model is not self.model or model.identity != self.model_identity or len(inputs) != 16:
            raise ValueError("Source AST requires the bound target and exact3x5 ancestry")
        canonical_to_caller = _canonical_rows(parents)
        if self.ast_oracle is not None and tuple(layers) != self.ast_oracle[3]:
            raise ValueError("Source AST capture layers differ from its numerical oracle")
        ordered = tuple(inputs[i] for i in canonical_to_caller)
        result = execute_tree(model, cache, ordered[0], (ordered[1:6], ordered[6:11], ordered[11:16]),
                              layers, cpu_reference=self.cpu_reference, expert_law=self.expert_law,
                              forward_law=self.forward_law)
        if self.forward_law != "reference" and self._exact_publication_binding is not None:
            from .deepseek_v41_ast_batched_target import _EXACT_PUBLICATION_SEAL
            result.transaction._admit_exact_publication(_EXACT_PUBLICATION_SEAL,self.source_revision_token)
        self.launches += 1
        self.physical_invocations+=result.receipt.get('physical_target_invocations',1)
        self.failed_private_invocations+=result.receipt.get('failed_private_target_invocations',0)
        if canonical_to_caller == tuple(range(16)):
            return result.logits, result.features, result.transaction
        try:
            transaction = ReindexedSourceASTTransaction(result.transaction, parents, canonical_to_caller)
            indices = mx.array(transaction.caller_to_canonical, mx.uint32)
            logits = mx.take(result.logits, indices, axis=1)
            features = None if result.features is None else mx.take(result.features, indices, axis=1)
            return logits, features, transaction
        except BaseException:
            result.transaction.abort()
            raise

    def forward_many(self, model, inputs, parents, caches, layers, *, cached, configured_limit):
        self.validate_cohort(len(inputs), configured_limit)
        return self.forward(model, inputs[0], parents[0], caches[0], layers, cached=cached)

    def dispatch_metrics(self, width, transaction, *, cached):
        from ..runtime.target_tree_executor import TargetTreeDispatchMetrics
        if (width != 1 or transaction.closed or transaction.provider_protocol != self.protocol
                or transaction.provider_revision_token != self.source_revision_token):
            raise ValueError("Source AST metrics require a matching live transaction")
        return TargetTreeDispatchMetrics(1, False, True, cached, self.protocol)

    @property
    def ast_tree_tile_rows(self):
        return 16 if self.ast_input_widths else 0

    def oracle_binding(self, *, artifact, capture_layers, target_binding, profile):
        return dict(protocol=PROTOCOL, source_revision_token=self.source_revision_token,
                    source_identity=dict(self.source_identity), artifact=artifact,
                    numerical_profile=self.numerical_profile, capture_layers=list(capture_layers),
                    target_binding=target_binding, profile_binding=profile.binding,
                    device=profile.device, build=profile.build)

    @property
    def oracle_checks(self):
        checks=('logits','features','prepared_state','continuation','isolation','commit_abort')
        return checks+(('hybrid_resident','hybrid_genuine_miss_recovery')if self.expert_law==HYBRID_EXPERT_LAW else())

    def with_ast_oracle(self, path, *, artifact, capture_layers, target_binding):
        self._exact_publication_binding = None
        document = json.loads(Path(path).read_text())
        checks = self.oracle_checks
        if (set(document) != {"schema", "binding", "input_widths", "checks", "receipt_sha256", "numerical_classification"}
                or document["schema"] != "mlx2.deepseek-v41-source-ast-oracle.v1"
                or document["input_widths"] != [16]
                or document["checks"] != {key: True for key in checks}
                or document["numerical_classification"] != "raw_bits_exact"
                or not isinstance(document["receipt_sha256"], str)
                or len(document["receipt_sha256"]) != 64):
            raise ValueError("Source AST exact numerical oracle is incomplete")
        self.ast_oracle = (document, str(Path(path).resolve()), artifact, tuple(capture_layers), target_binding)
        return self

    def ast_execution_backend(self, profile):
        if self.ast_oracle is None or self.cpu_reference:
            return None
        from ..runtime.ast_planner import AlternativeExecutionBackend, NumericalEquivalenceOracle
        self.validate_revision()
        document, path, artifact, layers, target_binding = self.ast_oracle
        expected = self.oracle_binding(artifact=artifact, capture_layers=layers,
                                       target_binding=target_binding, profile=profile)
        if document["binding"] != expected:
            raise ValueError("Source AST oracle source/artifact/profile binding changed")
        self.ast_input_widths = (16,)
        self._exact_publication_binding = expected
        return AlternativeExecutionBackend(
            backend="target_tree_single_session", profile_binding=profile.binding,
            device=profile.device, build=profile.build,
            source_revision_token=self.source_revision_token, input_widths=(16,),
            row_costs=profile.row_costs, launch_cost=profile.launch_cost,
            context_row_cost=profile.context_row_cost, context_windows=profile.context_windows,
            cost_authority="unmeasured_ordinary_prior", timing_boundary="target_successor",
            oracle=NumericalEquivalenceOracle(profile.binding, profile.device, profile.build,
                                              self.source_revision_token, path, True))

    def receipt(self):
        manager=getattr(self.model,'_source_ast_bank_manager',None)
        return dict(protocol=self.protocol,hybrid_bank_manager=None if manager is None else manager.receipt(), source_identity=self.source_identity,
                    source_revision_token=self.source_revision_token, launches=self.launches,logical_provider_calls=self.launches,
                    physical_target_invocations=self.physical_invocations,failed_private_target_invocations=self.failed_private_invocations,
                    qualified=False, defaulted=False, ast_numerical_admission=bool(self.ast_input_widths),
                    numerical_profile=self.numerical_profile, expert_law=self.expert_law,
                    forward_law=self.forward_law)
