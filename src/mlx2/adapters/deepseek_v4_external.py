"""Adapter-owned B1 DSpark serving seam; unqualified, chain verification only.

The target's pooled/overlapping state cannot be rewound with generic KV trim.
This conservative reference owner snapshots every plane and replays the accepted
input prefix with ordinary width-one arithmetic. No packed AST claim is made.
See provenance/deepseek-v4-dspark.json for the mined proposal equations.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from ..runtime.cache_operations import CacheOperationsContract, CachePlaneDeclaration
from ..runtime.drafters.deepseek_v4_dspark import (
    SIDECAR_SHA256,
    TARGET_SHA256,
    DeepSeekV4DSparkBackbone,
    DSparkContractError,
    DSparkHistory,
)
from ..runtime.models.cache import register_cache_class


class DeepSeekV4DraftCache:
    """Per-request committed support features, never proposed draft state."""

    cache_operations_contract = CacheOperationsContract(
        family="deepseek_v4_dspark",
        topology="checkpointed_hybrid",
        state_revision="deepseek-v4-dspark-support-v2",
        cache_layout="committed-support-b1",
        numerics="target_post_ffn_hc_mean_fp32",
        planes=(CachePlaneDeclaration("recurrent_state", ("state",), ("meta_state",)),),
        exact_prefix=True,
        restores_at_checkpoints=True,
        rollback=None,
    )

    def __init__(self, identity, stages, window=128, *, target_identity="unbound"):
        if type(target_identity) is not str or not target_identity:
            raise DSparkContractError("DSpark target execution identity required")
        self.target_identity = target_identity
        self.history = DSparkHistory(identity, stages, window)

    @property
    def offset(self):
        return self.history.next_position

    @property
    def state(self):
        return (self.history.main_history, *self.history.stage_kv)

    @property
    def meta_state(self):
        h = self.history
        return (
            "deepseek-v4-dspark-support-v2",
            h.identity,
            self.target_identity,
            str(h.stages),
            str(h.window),
            str(h.next_position),
            str(h.start_position),
        )

    @classmethod
    def from_state(cls, state, meta_state):
        from ..runtime.drafters.deepseek_v4_dspark import DSparkSnapshot

        if len(meta_state) != 7 or meta_state[0] != "deepseek-v4-dspark-support-v2":
            raise DSparkContractError("invalid DSpark support cache revision")
        _, identity, target_identity, stages, window, end, start = meta_state
        positions = (stages, window, end, start)
        if any(
            type(x) is not str or not x.isascii() or not x.isdecimal() or len(x) > 12
            for x in positions
        ):
            raise DSparkContractError("invalid DSpark support metadata positions")
        stages, window, end, start = map(int, positions)
        if len(state) != stages + 1:
            raise DSparkContractError("DSpark support cache stage mismatch")
        result = cls(identity, stages, window, target_identity=target_identity)
        result.history.restore(
            DSparkSnapshot(identity, end, start, state[0], tuple(state[1:]))
        )
        return result

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.state if x is not None)

    def is_trimmable(self):
        return False


class DeepSeekV4DSparkDraftModel:
    """Loaded, hash-verified DSpark backbone exposed to the external loop."""

    receipt_kind = "external_deepseek_v4_dspark_chain"
    supports_sampling = False
    supports_logits_processors = False
    supports_ordered_alternatives = False
    qualified = False

    def __init__(
        self, backbone, *, verification_mode="sequential", confidence_threshold=0.6
    ):
        if (
            type(confidence_threshold) not in (int, float)
            or not math.isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise DSparkContractError("confidence threshold must be finite in [0,1]")
        self.confidence_threshold = float(confidence_threshold)
        if verification_mode not in ("sequential", "causal_slab"):
            raise DSparkContractError("unknown DeepSeek target verifier mode")
        self.verification_mode = verification_mode
        # The backbone constructor verifies full sidecar/target fingerprints and
        # actual metadata before allocating runtime stages. No path-only bypass.
        prefix = f"{TARGET_SHA256}:{SIDECAR_SHA256}:"
        if (
            not isinstance(backbone, DeepSeekV4DSparkBackbone)
            or not getattr(backbone, "identity", "").startswith(prefix)
            or len(backbone.identity) != len(prefix) + 64
            or any(
                c not in "0123456789abcdef" for c in backbone.identity[len(prefix) :]
            )
        ):
            raise DSparkContractError("a verified DSpark backbone is required")
        self.backbone = backbone
        c = backbone.config
        self.config = SimpleNamespace(
            target_layer_ids=c.target_layers,
            feature_width=len(c.target_layers) * c.hidden_size,
            hidden_size=c.hidden_size,
            vocab_size=c.vocab_size,
            block_size=c.block_size,
            num_hidden_layers=c.stages,
            intermediate_size=c.expert_hidden,
            num_key_value_heads=1,
            head_dim=c.head_dim,
        )
        self.proposal_rows = c.block_size
        self.maximum_proposal_length = c.block_size
        self.adaptive_confidence_features = [[]]
        self.target_identity = None
        self.execution_binding = None

    def bind(self, target_model):
        if (
            getattr(target_model, "weights", None) is not self.backbone.target_weights
            or getattr(target_model, "args", None) != self.backbone.target_config
        ):
            raise DSparkContractError(
                "DSpark wrapper target weights/config binding mismatch"
            )
        identity = getattr(target_model, "identity", None)
        if type(identity) is not str or not identity:
            raise DSparkContractError("target runtime execution identity required")
        from ..runtime.models.deepseek_v4_identity import (
            dspark_execution_binding,
            dspark_target_identity,
        )

        target_identity = dspark_target_identity(
            identity,
            getattr(target_model.weights, "compute_policy", None),
            getattr(target_model.weights, "embedding_policy", None),
            self.verification_mode,
            self.confidence_threshold,
        )
        if self.target_identity is not None and self.target_identity != target_identity:
            raise DSparkContractError(
                "a bound DSpark wrapper cannot change target policy"
            )
        self.target_identity = target_identity
        self.execution_binding = dspark_execution_binding(
            self.backbone.identity, target_identity, self.verification_mode
        )
        return self

    @property
    def receipt_settings(self):
        return {
            "binding": self.execution_binding,
            "backbone_binding": self.backbone.identity,
            "target_execution_identity": self.target_identity,
            "qualified": False,
            "proposal_distribution": "deterministic_point_mass",
            "trained_proposals": self.proposal_rows,
            "confidence_threshold": self.confidence_threshold,
            "confidence_law": "sigmoid_prenorm_markov_strict_less_prefix_including_first",
            "proposal_attention": "noncausal_support_target_draft",
            "target_feature_law": "post_ffn_uniform_hc_mean",
            "verification": (
                "ordinary_width_one_snapshot_replay"
                if self.verification_mode == "sequential"
                else "causal_kplus1_layer_major_width_one_v2"
            ),
            "target_row_invariance_established": False,
            "arithmetic_rows": 1,
            "execution_order": "token_major" if self.verification_mode == "sequential" else "layer_major",
            "packed_ast": False,
            "sampling": "greedy_only",
        }

    def extra_memory_reservation(self, cache, *, append, pending_tokens, prefill):
        # Fixed trainedB1/block5 support attention, query/HC activations,
        # packedexpert intermediates and vocabulary logits fit below64MiB.
        # Resident sidecar weights are preloaded and already removed from live
        # serving headroom; they must not be charged as per-request scratch.
        return 64 << 20

    def make_cache(self):
        if self.target_identity is None:
            raise DSparkContractError(
                "DSpark must bind target execution before cache allocation"
            )
        c = self.backbone.config
        return [
            DeepSeekV4DraftCache(
                self.backbone.identity,
                c.stages,
                c.window,
                target_identity=self.target_identity,
            )
        ]

    def batch_caches(self, rows):
        if len(rows) != 1:
            raise DSparkContractError("DSpark support currently requires B1")
        return rows[0]

    def append_context(self, hidden, cache):
        if (
            len(cache) != 1
            or cache[0].history.identity != self.backbone.identity
            or cache[0].target_identity != self.target_identity
        ):
            raise DSparkContractError("DSpark support binding mismatch")
        if (
            len(hidden.shape) != 3
            or hidden.shape[0] != 1
            or hidden.shape[-1] != self.config.feature_width
        ):
            raise DSparkContractError("DSpark target feature geometry mismatch")
        if not hidden.shape[1]:
            return
        d = self.config.hidden_size
        captures = {
            layer: hidden[:, :, i * d : (i + 1) * d]
            for i, layer in enumerate(self.config.target_layer_ids)
        }
        self.backbone.record_features(
            captures, cache[0].offset, history=cache[0].history
        )

    def draft_distributions(
        self, anchors, hidden, cache, proposal_length, rngs, temperatures, **kwargs
    ):
        from ..runtime.proposal_providers import PointMassLaw

        if len(anchors) != 1 or len(temperatures) != 1 or float(temperatures[0]) != 0:
            raise DSparkContractError(
                "DSpark serving seam currently requires greedy B1"
            )
        if kwargs:
            raise DSparkContractError(
                "DSpark processor/alternative extensions not enabled"
            )
        if (
            type(proposal_length) is not int
            or not 0 <= proposal_length <= self.proposal_rows
        ):
            raise DSparkContractError("DSpark proposal count exceeds trained capacity")
        self.append_context(hidden, cache)
        if proposal_length == 0 or cache[0].offset == 0:
            self.adaptive_confidence_features = [[]]
            return [[]], [[]]
        result = self.backbone.propose(
            int(anchors[0]),
            position=cache[0].offset,
            history=cache[0].history,
            max_proposals=min(proposal_length, self.proposal_rows),
            confidence_threshold=self.confidence_threshold,
        )
        tokens = list(result[0])
        self.adaptive_confidence_features = [list(result[1])]
        return [tokens], [[PointMassLaw(token) for token in tokens]]


class DeepSeekV4VerifyRows:
    """Exclusive ordinary-arithmetic B1 verification owner."""

    def __init__(self, model, rows, layers):
        if (
            len(rows) != 1
            or not rows[0]
            or len({id(c) for c in rows[0]}) != len(rows[0])
        ):
            raise DSparkContractError(
                "DeepSeek verification requires independent B1 cache planes"
            )
        self.model, self.rows, self.layers = model, [list(rows[0])], tuple(layers)
        if len({c.offset for c in rows[0]}) != 1 or any(
            getattr(c, "pending", False) for c in rows[0]
        ):
            raise DSparkContractError("target cache is not at a committed boundary")
        if any(
            not callable(getattr(c, "snapshot", None))
            or not callable(getattr(c, "restore", None))
            for c in rows[0]
        ):
            raise DSparkContractError("target cache lacks exact snapshot/restore")
        self._active = None
        self._poisoned = False
        self.revision = 0

    def holds(self, rows):
        return (
            len(rows) == 1
            and len(rows[0]) == len(self.rows[0])
            and all(a is b for a, b in zip(rows[0], self.rows[0]))
        )

    def begin(self, lengths, *, plain_single_token=False):
        if self._poisoned:
            raise DSparkContractError("verification owner poisoned by failed restore")
        if self._active is not None and not self._active.closed:
            raise DSparkContractError("verification owner already active")
        if len(lengths) != 1 or type(lengths[0]) is not int or lengths[0] < 1:
            raise DSparkContractError("invalid B1 verification length")
        self._active = DeepSeekV4VerifyTransaction(self, lengths[0], plain_single_token)
        return self._active


class DeepSeekV4VerifyTransaction:
    def __init__(self, owner, width, plain):
        self.owner, self.width, self.lengths = owner, width, [width]
        self.caches = owner.rows[0]
        self.plain, self.batched, self.closed = bool(plain and width == 1), False, False
        self._base_offset = self.caches[0].offset
        self._snapshots = [c.snapshot() for c in self.caches]
        self._inputs = None

    @property
    def retained_nbytes(self):
        return sum(getattr(c, "nbytes", 0) for c in self.caches)

    def _restore(self):
        for cache, state in zip(self.caches, self._snapshots):
            cache.restore(state)

    def _close(self):
        self.closed = True
        self.owner.revision += 1
        self.owner._active = None
        self._snapshots = []
        self._inputs = None

    def _forward(self, inputs, taps):
        import mlx.core as mx

        logits, features = [], []
        for column in range(inputs.shape[1]):
            token = inputs[:, column : column + 1]
            if taps:
                out, hidden = self.owner.model.forward_with_taps(
                    token, self.caches, self.owner.layers
                )
                features.append(hidden)
            else:
                out = self.owner.model(token, self.caches)
            logits.append(out)
        return mx.concatenate(logits, axis=1), mx.concatenate(
            features, axis=1
        ) if taps else None

    def forward(self, inputs, *, taps=True):
        if self.closed or self.owner._active is not self or self._inputs is not None:
            raise DSparkContractError("stale or repeated verification forward")
        if tuple(inputs.shape) != (1, self.width):
            raise DSparkContractError("verification input geometry mismatch")
        import mlx.core as mx

        self._inputs = mx.asarray(inputs, copy=True)
        try:
            return self._forward(inputs, taps)
        except BaseException:
            self.abort()
            raise

    def commit(self, accepted_lengths):
        if self.closed or self._inputs is None:
            raise DSparkContractError("no verified prefix to commit")
        if (
            len(accepted_lengths) != 1
            or type(accepted_lengths[0]) is not int
            or not 0 <= accepted_lengths[0] <= self.width
        ):
            raise DSparkContractError("accepted prefix exceeds verified input")
        try:
            if any(
                c.offset != self._base_offset + self.width
                or getattr(c, "pending", False)
                for c in self.caches
            ):
                raise DSparkContractError(
                    "target verification did not commit every state plane"
                )
            if accepted_lengths[0] != self.width:
                self._restore()
                if accepted_lengths[0]:
                    self._forward(self._inputs[:, : accepted_lengths[0]], True)
            rows = [list(self.caches)]
            self._close()
            return rows
        except BaseException:
            self.abort()
            raise

    def abort(self):
        if not self.closed:
            try:
                self._restore()
            except BaseException:
                self.owner._poisoned = True
                raise
            finally:
                # Failed restoration poisons this request; never retain a live
                # owner/transaction that could accidentally publish its state.
                self._close()


# Importing this module is the explicit adapter action that registers its cache.
register_cache_class(DeepSeekV4DraftCache)


class DeepSeekV4ExternalBatchGenerator:
    """Factory that binds DS4-owned transactions without scheduler name branches."""

    def __new__(cls, model, *, draft_model, completion_batch_size=1, **kwargs):
        from ..runtime.external_speculative import ExternalDraftBatchGenerator

        if completion_batch_size != 1:
            raise DSparkContractError(
                "DeepSeek DSpark verification currently requires B1"
            )
        # Do not masquerade an ordinary sequential chain as packed tree AST.
        blocked = (
            "ast_tree_tile_rows",
            "target_tree_executor_provider",
            "ast_target_tree_executor_provider",
            "progressive_verification_tile",
            "dynamic_singleton_tree",
            "ast_b1_tile_policy",
        )
        if any(kwargs.get(key) for key in blocked):
            raise DSparkContractError("DeepSeek packed AST executor is not implemented")
        if not isinstance(draft_model, DeepSeekV4DSparkDraftModel):
            raise DSparkContractError("DeepSeek requires its verified DSpark wrapper")

        class OwnedGenerator(ExternalDraftBatchGenerator):
            def _target_owner(self, rows):
                if self.draft.verification_mode == "causal_slab":
                    from .deepseek_v4_slab import DeepSeekV4CausalSlabVerifyRows

                    return DeepSeekV4CausalSlabVerifyRows(self.model, rows, self.layers)
                return DeepSeekV4VerifyRows(self.model, rows, self.layers)

            def _cohort_target_forward(
                self, inputs, transaction, cohort, *, site, taps=True
            ):
                if not isinstance(transaction, DeepSeekV4VerifyTransaction):
                    raise DSparkContractError(
                        "foreign DeepSeek verification transaction"
                    )
                return transaction.forward(inputs, taps=taps)

        draft_model.bind(model)
        requested_binding = kwargs.pop("binding", None)
        if requested_binding not in (None, draft_model.execution_binding):
            raise DSparkContractError(
                "external route binding omits target runtime policy"
            )
        return OwnedGenerator(
            model,
            draft_model=draft_model,
            binding=draft_model.execution_binding,
            completion_batch_size=1,
            **kwargs,
        )
