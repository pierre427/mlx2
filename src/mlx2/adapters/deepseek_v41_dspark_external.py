# SPDX-License-Identifier: MIT
"""Request-local embedded V4.1 draft state and exact linear target verification."""

import hashlib
import json
from types import SimpleNamespace

from ..runtime.cache_operations import CacheOperationsContract, CachePlaneDeclaration
from ..runtime.drafters.deepseek_v4_dspark import DSparkHistory, DSparkSnapshot
from ..runtime.models.cache import register_cache_class
from .deepseek_v4_external import (
    DeepSeekV4DSparkDraftModel,
    DeepSeekV4VerifyRows,
    DeepSeekV4VerifyTransaction,
)

CACHE_REVISION = "mlx2.v41.dspark-committed-pre-attention.v1"


class DeepSeekV41DraftCache:
    cache_operations_contract = CacheOperationsContract(
        family="deepseek_v41_dspark",
        topology="checkpointed_hybrid",
        state_revision=CACHE_REVISION,
        cache_layout="committed-support-b1",
        numerics="pre_attention_uniform_hc_mean_bf16",
        planes=(CachePlaneDeclaration("recurrent_state", ("state",), ("meta_state",)),),
        exact_prefix=True,
        restores_at_checkpoints=True,
        rollback=None,
    )

    def __init__(self, identity, target_identity, *, window=128):
        if not identity or not target_identity:
            raise ValueError("draft cache requires execution identities")
        self.target_identity = target_identity
        self.history = DSparkHistory(identity, 3, window)

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
            CACHE_REVISION,
            h.identity,
            self.target_identity,
            str(h.window),
            str(h.next_position),
            str(h.start_position),
        )

    @classmethod
    def from_state(cls, state, meta_state):
        if len(meta_state) != 6 or meta_state[0] != CACHE_REVISION or len(state) != 4:
            raise ValueError("invalid V4.1 draft snapshot revision/geometry")
        _, identity, target, window, end, start = meta_state
        if any(
            type(v) is not str or not v.isascii() or not v.isdecimal() or len(v) > 12
            for v in (window, end, start)
        ):
            raise ValueError("invalid V4.1 draft snapshot positions")
        result = cls(identity, target, window=int(window))
        result.history.restore(
            DSparkSnapshot(identity, int(end), int(start), state[0], tuple(state[1:]))
        )
        return result

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.state if x is not None)

    def is_trimmable(self):
        return False


class DeepSeekV41DSparkDraftModel(DeepSeekV4DSparkDraftModel):
    kind = "deepseek_v41_dspark"
    receipt_kind = "external_deepseek_v41_embedded_dspark_chain"

    def __init__(self, backbone, *, confidence_threshold=0.0):
        from ..runtime.drafters.deepseek_v41_dspark import DeepSeekV41DSparkBackbone

        if not isinstance(backbone, DeepSeekV41DSparkBackbone):
            raise TypeError("bound embedded V4.1 draft required")
        if (
            type(confidence_threshold) not in (int, float)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("draft confidence threshold must be in[0,1]")
        self.backbone = backbone
        self.confidence_threshold = float(confidence_threshold)
        c, t = backbone.config, backbone.target.args
        self.config = SimpleNamespace(
            target_layer_ids=c.target_layers,
            feature_width=3 * t.hidden_size,
            hidden_size=t.hidden_size,
            vocab_size=t.vocab_size,
            block_size=c.block_size,
            num_hidden_layers=3,
            intermediate_size=t.moe_intermediate_size,
            num_key_value_heads=1,
            head_dim=t.head_dim,
        )
        self.proposal_rows = self.maximum_proposal_length = c.block_size
        self.adaptive_confidence_features = [[]]
        self.target_identity = None
        self.execution_binding = None
        self.verification_mode = "sequential"

    def bind(self, target):
        if (
            target is not self.backbone.target
            or target.weights is not self.backbone.target_weights
        ):
            raise ValueError("embedded DSpark target binding changed")
        identity = target.identity
        if self.target_identity is not None and self.target_identity != identity:
            raise ValueError("draft cannot change target execution identity")
        self.target_identity = identity
        self.execution_binding = hashlib.sha256(
            json.dumps(
                [
                    CACHE_REVISION,
                    self.backbone.identity,
                    identity,
                    self.confidence_threshold,
                ]
            ).encode()
        ).hexdigest()
        return self

    def make_cache(self):
        if self.target_identity is None:
            raise ValueError("draft must bind before cache allocation")
        return [DeepSeekV41DraftCache(self.backbone.identity, self.target_identity)]

    def extra_memory_reservation(self, cache, *, append, pending_tokens, prefill):
        from .deepseek_v41_dspark_artifact import dspark_workspace_bound
        return dspark_workspace_bound(self.backbone.target.args)

    @property
    def receipt_settings(self):
        return {
            "binding": self.execution_binding,
            "backbone_binding": self.backbone.identity,
            "target_execution_identity": self.target_identity,
            "qualified": False,
            "proposal_distribution": "deterministic_point_mass",
            "trained_proposals": 5,
            "confidence_threshold": self.confidence_threshold,
            "confidence_law": "sigmoid_prenorm_markov_strict_less_retained_prefix",
            "proposal_attention": "noncausal_committed_main_and_draft",
            "target_feature_law": "pre_attention_uniform_hc_mean_layers37_38_39",
            "verification": "ordinary_width_one_snapshot_replay",
            "arithmetic_rows": 1,
            "packed_ast": False,
            "sampling": "greedy_only",
        }


def create_dspark_batch(
    model, draft_model, *, binding=None, completion_batch_size=1, **kwargs
):
    from ..runtime.external_speculative import ExternalDraftBatchGenerator

    if (
        not isinstance(draft_model, DeepSeekV41DSparkDraftModel)
        or completion_batch_size != 1
    ):
        raise ValueError("embedded DSpark requires B1 learned draft wrapper")
    blocked = (
        "ast_tree_tile_rows",
        "target_tree_executor_provider",
        "ast_target_tree_executor_provider",
        "progressive_verification_tile",
        "dynamic_singleton_tree",
        "ast_b1_tile_policy",
    )
    if any(kwargs.get(k) for k in blocked):
        raise ValueError("embedded DSpark has no packed-tree route")
    draft_model.bind(model)
    if binding not in (None, draft_model.execution_binding):
        raise ValueError("DSpark route binding omits target policy")

    class OwnedGenerator(ExternalDraftBatchGenerator):
        def insert(self, prompts, *, sampling_configs=None, **options):
            for config in sampling_configs or [{}] * len(prompts):
                temperature = config.get("sampling_temp", 0.0)
                if type(temperature) not in (int, float) or temperature != 0:
                    raise ValueError(
                        "embedded V4.1 DSpark requires greedy temperature0"
                    )
            return super().insert(prompts, sampling_configs=sampling_configs, **options)

        def _target_owner(self, rows):
            return DeepSeekV4VerifyRows(self.model, rows, self.layers)

        def _cohort_target_forward(
            self, inputs, transaction, cohort, *, site, taps=True
        ):
            if not isinstance(transaction, DeepSeekV4VerifyTransaction):
                raise TypeError("foreign DSpark verifier owner")
            return transaction.forward(inputs, taps=taps)

    return OwnedGenerator(
        model,
        draft_model=draft_model,
        binding=draft_model.execution_binding,
        completion_batch_size=1,
        **kwargs,
    )


register_cache_class(DeepSeekV41DraftCache)
