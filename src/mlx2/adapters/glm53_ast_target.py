"""Exact ordinary-row AST provider for GLM-5.3's hybrid state.

Each proposal node runs ordinary B1 arithmetic from its parent's private
checkpoint. This first provider preserves every state plane and reports its
serial physical work honestly; it claims no packed-tree acceleration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx

from ..runtime.cow_cache import snapshot_prompt_cache_descriptors
from ..runtime.models.glm53_sparse_attention import Glm53MLAFrontier
from ..runtime.target_tree_executor import TargetTreeDispatchMetrics

PROTOCOL = "glm53-ordinary-row-tree-v1"
ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    "src/mlx2/adapters/glm53_ast_target.py",
    "src/mlx2/adapters/glm53_layer_tree.py",
    "src/mlx2/runtime/models/glm53_gguf.py",
    "src/mlx2/runtime/models/glm53_gguf_indexer.py",
    "src/mlx2/runtime/models/glm53_sparse_attention.py",
    "src/mlx2/runtime/gguf/packed_metal.py",
    "src/mlx2/runtime/gguf/k_metal.py",
    "src/mlx2/runtime/gguf/kblock_metal.py",
    "src/mlx2/runtime/gguf/kblock_union_shared.py",
    "src/mlx2/runtime/cow_cache.py",
    "src/mlx2/runtime/models/gated_delta.py",
    "src/mlx2/adapters/glm53_gguf.py",
    "src/mlx2/adapters/external_draft_policy.py",
    "src/mlx2/runtime/ast_tree_verify.py",
    "src/mlx2/runtime/ast_tree_tiles.py",
)


def source_identity():
    return tuple(
        (p, hashlib.sha256((ROOT / p).read_bytes()).hexdigest()) for p in SOURCE_PATHS
    )


def clone(cache):
    return snapshot_prompt_cache_descriptors(cache)[0]


def stamp(cache):
    def arrays(value):
        if isinstance(value, mx.array):
            return (id(value),)
        if isinstance(value, (tuple, list)):
            return tuple(i for v in value for i in arrays(v))
        return ()

    return tuple(
        (id(c), json.dumps(c.meta_state, sort_keys=True), arrays(c.state))
        for c in cache
    )


class Glm53TreeTransaction:
    provider_protocol = PROTOCOL

    def __init__(self, provider, live, base, frontiers, parents):
        self.provider, self.live, self.base, self.frontiers = (
            provider,
            live,
            base,
            frontiers,
        )
        self.provider_protocol = provider.protocol
        self.parents = tuple(parents)
        self.original = stamp(live)
        self.provider_revision_token = provider.source_revision_token
        self.closed = False

    def prepared_cache(self, path):
        self.provider.validate_revision()
        if self.closed or stamp(self.live) != self.original:
            raise ValueError("GLM-5.3 AST transaction closed or live frontier changed")
        path = tuple(path)
        if path and (
            path[0] != 0
            or any(type(i) is not int or not 0 <= i < len(self.parents) for i in path)
            or any(self.parents[i] != p for p, i in zip((-1, *path[:-1]), path))
        ):
            raise ValueError("GLM-5.3 AST commit requires a root-connected path")
        return self.materialize(self.base if not path else self.frontiers[path[-1]])

    def materialize(self, states):
        """Real caches for one path: clones, and Glm53MLACache for MLA frontiers.

        A frontier (glm53_tree_frontier=tails_v1) becomes clone(base) plus
        its tail rows appended one at a time, equal to the clone-and-append
        path's state bit for bit; KDA layers are descriptor clones as before.
        """
        frontier = [isinstance(s, Glm53MLAFrontier) for s in states]
        if not any(frontier):
            return clone(states)
        plain = [s for s, f in zip(states, frontier) if not f]
        plain = iter(clone(plain) if plain else ())
        calls = getattr(self.provider.model, "calls", None)
        result = []
        for state, is_frontier in zip(states, frontier):
            if not is_frontier:
                result.append(next(plain))
                continue
            result.append(state.materialize(clone=clone))
            if calls is not None:
                calls["tree_frontier_materialized"] = calls.get("tree_frontier_materialized", 0) + 1
        return result

    def commit_paths(self, paths):
        if len(paths) != 1:
            raise ValueError("GLM-5.3 AST owns one request")
        result = self.prepared_cache(paths[0])
        try:
            mx.eval([c.state for c in result])
        finally:
            self.abort()
        return [result]

    def commit(self, *, accepted_lengths):
        if len(accepted_lengths) != 1:
            raise ValueError("GLM-5.3 AST owns one request")
        return self.commit_paths([tuple(range(accepted_lengths[0]))])

    def abort(self):
        self.closed = True
        self.base, self.frontiers = [], []


class Glm53TargetTreeExecutor:
    protocol = PROTOCOL
    maximum_cohort_width = 1

    def __init__(self, model, *, implementation="serial"):
        if implementation not in ("serial", "layered"):
            raise ValueError("unknown GLM AST target implementation")
        self.implementation = implementation
        self.protocol = (
            PROTOCOL
            if implementation == "serial"
            else "glm53-layered-row-exact-tree-v1"
        )
        self.model = model
        self.model_identity = model.identity
        self.source_identity = source_identity()
        self.source_revision_token = hashlib.sha256(
            json.dumps([self.protocol, self.source_identity]).encode()
        ).hexdigest()
        self.invocations = self.nodes = 0

    def validate_revision(self):
        if (
            self.model.identity != self.model_identity
            or source_identity() != self.source_identity
        ):
            raise ValueError("GLM-5.3 AST target/source binding changed")

    def validate_cohort(self, width, configured_limit):
        if width != 1 or configured_limit < 1:
            raise ValueError("GLM-5.3 AST currently supports one live request")

    def forward(self, model, inputs, parents, cache, layers, *, cached=True):
        self.validate_revision()
        if (
            model is not self.model
            or not 1 <= len(inputs) <= 32
            or len(parents) != len(inputs)
            or parents[0] != -1
            or any(
                type(p) is not int or not 0 <= p < i
                for i, p in enumerate(parents[1:], 1)
            )
        ):
            raise ValueError(
                "GLM-5.3 AST requires a parent-ordered rooted tree of at most32 rows"
            )
        if any(
            type(t) is not int or not 0 <= t < model.args.vocab_size for t in inputs
        ):
            raise ValueError("GLM-5.3 AST token outside vocabulary")
        base = clone(cache)
        frontiers = []
        logits = []
        features = []
        before = stamp(cache)
        if self.implementation == "layered":
            from .glm53_layer_tree import forward

            out, tap, frontiers = forward(model, inputs, parents, base, layers, clone)
            logits.append(out)
            features.append(tap)
        else:
            for i, token in enumerate(inputs):
                private = clone(base if parents[i] < 0 else frontiers[parents[i]])
                out, tap = model.forward_with_taps(mx.array([[token]]), private, layers)
                mx.eval(out, tap, [c.state for c in private])
                frontiers.append(private)
                logits.append(out)
                features.append(tap)
        if stamp(cache) != before:
            raise RuntimeError("GLM-5.3 AST changed live cache during verification")
        transaction = Glm53TreeTransaction(self, cache, base, frontiers, parents)
        self.invocations += 1
        self.nodes += len(inputs)
        return (
            mx.concatenate(logits, axis=1),
            mx.concatenate(features, axis=1),
            transaction,
        )

    def forward_many(
        self, model, inputs, parents, caches, layers, *, cached, configured_limit
    ):
        self.validate_cohort(len(inputs), configured_limit)
        return self.forward(
            model, inputs[0], parents[0], caches[0], layers, cached=cached
        )

    def dispatch_metrics(self, width, transaction, *, cached):
        self.validate_cohort(width, 1)
        if (
            transaction.closed
            or transaction.provider_revision_token != self.source_revision_token
        ):
            raise ValueError("GLM-5.3 AST metrics require this open transaction")
        return TargetTreeDispatchMetrics(1, False, False, cached, self.protocol)

    def receipt(self):
        return {
            "protocol": self.protocol,
            "source_revision_token": self.source_revision_token,
            "source_identity": dict(self.source_identity),
            "implemented": True,
            "qualified": False,
            "physical_packed": False,
            "physical_target_invocations": self.invocations
            if self.implementation == "layered"
            else self.nodes,
            "implementation": self.implementation,
            "serial_model_calls": 0 if self.implementation == "layered" else self.nodes,
            "shared_layer_traversal": self.implementation == "layered",
            "batched_packed_mlp_rows": self.implementation == "layered",
            "batched_packed_kda_projections": self.implementation == "layered",
            "batched_attention_output_projection": self.implementation == "layered",
            "tree_transactions": self.invocations,
            "numerics": "ordinary B1 per node; private exact parent checkpoints",
            # Layered-tree MLA policies (glm53_layer_tree); defaults = today.
            "tree_rows_attention": getattr(self.model, "tree_rows_attention", "per_row"),
            "tree_frontier": getattr(self.model, "tree_frontier", "clone"),
            "tree_compile": getattr(self.model, "tree_compile", "off"),
            "tree_router": getattr(self.model, "tree_router", "reference"),
            "tree_moe": getattr(self.model, "tree_moe", "fold"),
            "tree_compile_counters": dict(getattr(
                self.model.__dict__.get("_tree_regions"), "counters", {})),
        }
