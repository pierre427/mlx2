# SPDX-License-Identifier: MIT
"""Request-local embedded V4.1 draft state and exact linear target verification."""

import copy
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

    def __init__(self, backbone, *, confidence_threshold=0.0, verify_schedule="fixed",
                 verify_costs=None, draft_cost=None, admission="off"):
        from ..runtime.drafters.deepseek_v41_dspark import DeepSeekV41DSparkBackbone
        from .deepseek_v41_dspark_schedule import validate

        if not isinstance(backbone, DeepSeekV41DSparkBackbone):
            raise TypeError("bound embedded V4.1 draft required")
        if (
            type(confidence_threshold) not in (int, float)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("draft confidence threshold must be in[0,1]")
        validate(verify_schedule, verify_costs, draft_cost)
        from .deepseek_v41_dspark_miss_gate import validate as validate_admission
        self.admission = validate_admission(admission)
        self.admission_stats = {}
        self.backbone = backbone
        self.confidence_threshold = float(confidence_threshold)
        self.verify_schedule = verify_schedule
        self._verify_costs = None if verify_costs is None else tuple(float(v) for v in verify_costs)
        self._draft_cost = draft_cost
        self.schedule_stats = {}
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
                    *([] if self.verify_schedule == "fixed" else [
                        self.verify_schedule, self.schedule_costs, self.schedule_draft_cost]),
                    *([] if self.admission == "off" else ["admission", self.admission]),
                ]
            ).encode()
        ).hexdigest()
        return self

    @property
    def schedule_costs(self):
        from .deepseek_v41_dspark_schedule import DEFAULT_COSTS
        if self._verify_costs is not None:
            return list(self._verify_costs)
        rows = getattr(self.backbone.target, "verify_rows_policy", "serial")
        return list(DEFAULT_COSTS.get(rows, DEFAULT_COSTS["serial"]))

    @property
    def schedule_draft_cost(self):
        from .deepseek_v41_dspark_schedule import DEFAULT_DRAFT_COST
        return float(DEFAULT_DRAFT_COST if self._draft_cost is None else self._draft_cost)

    def draft_distributions(self, anchors, hidden, cache, proposal_length, rngs, temperatures, **kwargs):
        ledger = self._verify_ledger(cache)
        committed = int(hidden.shape[1]) if len(getattr(hidden, "shape", ())) == 3 else 0
        self._record_verify(ledger.close(committed))
        if self.admission != "off" and proposal_length:
            result = self._admitted_draft(anchors, hidden, cache, proposal_length, rngs, temperatures, **kwargs)
        else:
            result = self._scheduled_draft(anchors, hidden, cache, proposal_length, rngs, temperatures, **kwargs)
        ledger.open(min(len(result[0][0]) if result[0] else 0, 5) + 1)
        return result

    def _verify_ledger(self, cache):
        # Request-local, like the miss gate: lives on this request's draft plane.
        from .deepseek_v41_dspark_verify_ledger import VerifyExpertLedger
        plane = cache[0]
        ledger = getattr(plane, "_mlx2_dspark_verify_ledger", None)
        if ledger is None:
            stream = getattr(getattr(self.backbone, "target_weights", None), "expert_stream", None)
            ledger = VerifyExpertLedger(getattr(stream, "stats", None),
                                        timed=getattr(self, "verify_schedule", "fixed") == "survival_miss_v1")
            plane._mlx2_dspark_verify_ledger = ledger
        return ledger

    def _record_verify(self, delta):
        """Process-lifetime expert loads per verify width (bounded integer counters)."""
        if delta is None:
            return
        stats = self.__dict__.setdefault("verify_expert_stats", {})
        width = str(delta["width"])
        stats["rounds"] = stats.get("rounds", 0) + 1
        stats["verify_rows"] = stats.get("verify_rows", 0) + delta["width"]
        for key in ("expert_acquisitions", "expert_loads", "miss_bytes"):
            stats[key] = stats.get(key, 0) + delta[key]
            per = stats.setdefault(key + "_by_width", {})
            per[width] = per.get(width, 0) + delta[key]
        per = stats.setdefault("rounds_by_width", {})
        per[width] = per.get(width, 0) + 1

    def _admission_gate(self, cache):
        # Request-local: the gate lives on this request's draft cache plane.
        from .deepseek_v41_dspark_miss_gate import MissCostGate
        plane = cache[0]
        gate = getattr(plane, "_mlx2_dspark_miss_gate", None)
        if gate is None:
            stream = getattr(self.backbone.target_weights, "expert_stream", None)
            gate = MissCostGate(lambda: dict(getattr(stream, "stats", None) or {}))
            plane._mlx2_dspark_miss_gate = gate
        return gate

    def _admitted_draft(self, anchors, hidden, cache, proposal_length, rngs, temperatures, **kwargs):
        """expert_miss_cost_v1: plain or speculative round from measured cost."""
        import time
        from ..runtime.external_speculative import DraftUnavailable
        from .deepseek_v41_dspark_miss_gate import REASON
        gate = self._admission_gate(cache)
        committed = int(hidden.shape[1]) if len(getattr(hidden, "shape", ())) == 3 else 0
        gate.observe(committed)
        decision = gate.decision()
        stats = self.admission_stats
        key = "rounds_" + decision["action"]
        if decision["handoff"]:
            stats["handoffs"] = stats.get("handoffs", 0) + 1
            stats["last_handoff"] = gate.receipt()
            error = DraftUnavailable(REASON + ": " + decision["reason"])
            error.ordinary_reason = REASON
            raise error
        stats[key] = stats.get(key, 0) + 1
        if decision["priced"]:
            stats["priced_" + ("admits" if decision["admit"] else "rejections")] = (
                stats.get("priced_" + ("admits" if decision["admit"] else "rejections"), 0) + 1)
        started = time.perf_counter()
        length = proposal_length if decision["action"] == "speculate" else 0
        result = self._scheduled_draft(anchors, hidden, cache, length, rngs, temperatures, **kwargs)
        gate.begin("speculative" if result[0] and result[0][0] else "plain", time.perf_counter() - started)
        stats["last_request"] = gate.receipt()
        return result

    def _scheduled_draft(self, anchors, hidden, cache, proposal_length, rngs, temperatures, **kwargs):
        tokens, laws = super().draft_distributions(anchors, hidden, cache, proposal_length, rngs,
                                                   temperatures, **kwargs)
        if self.verify_schedule == "fixed" or not tokens[0]:
            return tokens, laws
        from .deepseek_v41_dspark_schedule import choose
        stats = self.schedule_stats
        costs = self.schedule_costs
        if self.verify_schedule == "survival_miss_v1":
            ledger = self._verify_ledger(cache)
            if ledger.needs_plain_probe():
                stats["miss_probe_rounds"] = stats.get("miss_probe_rounds", 0) + 1
                stats["rounds"] = stats.get("rounds", 0) + 1
                stats["proposed"] = stats.get("proposed", 0) + len(tokens[0])
                stats["verify_drafts_0"] = stats.get("verify_drafts_0", 0) + 1
                return [tokens[0][:0]], [laws[0][:0]]
            from .deepseek_v41_dspark_verify_ledger import miss_priced_costs
            stream = getattr(self.backbone.target_weights, "expert_stream", None)
            if ledger.needs_speculative_probe():
                pricing = {"priced": False, "reason": "speculative_reprobe"}
                stats["miss_speculative_probe_rounds"] = stats.get("miss_speculative_probe_rounds", 0) + 1
            else:
                costs, pricing = miss_priced_costs(costs, ledger, fallback_stats=getattr(stream, "stats", None))
            key = "miss_priced_rounds" if pricing["priced"] else "miss_unpriced_rounds"
            stats[key] = stats.get(key, 0) + 1
            if pricing["priced"]:
                stats["last_miss_pricing"] = pricing
        confidence = self.adaptive_confidence_features[0][:len(tokens[0])]
        k, receipt = choose(confidence, maximum=len(tokens[0]), costs=costs,
                            draft_cost=self.schedule_draft_cost)
        stats["rounds"] = stats.get("rounds", 0) + 1
        stats["proposed"] = stats.get("proposed", 0) + len(tokens[0])
        stats["scheduled"] = stats.get("scheduled", 0) + k
        stats["expected_tokens"] = stats.get("expected_tokens", 0.0) + receipt["expected_tokens"]
        key = f"verify_drafts_{k}"
        stats[key] = stats.get(key, 0) + 1
        return [tokens[0][:k]], [laws[0][:k]]

    def make_cache(self):
        if self.target_identity is None:
            raise ValueError("draft must bind before cache allocation")
        return [DeepSeekV41DraftCache(self.backbone.identity, self.target_identity)]

    def extra_memory_reservation(self, cache, *, append, pending_tokens, prefill):
        from .deepseek_v41_dspark_artifact import dspark_workspace_bound
        return dspark_workspace_bound(self.backbone.target.args)

    @property
    def verification_receipt(self):
        target = self.backbone.target
        rows = getattr(target, "verify_rows_policy", "serial")
        commit = getattr(target, "verification_commit_policy", "replay")
        return ("lockstep_multirow_v1_" if rows == "lockstep_multirow_v1" else "ordinary_width_one_") + (
            "local_frontier" if commit == "local_frontier" else "snapshot_replay")

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
            "verification": self.verification_receipt,
            "verify_schedule": self.verify_schedule,
            "admission": self.admission,
            **({} if self.admission == "off" else {
                "admission_law": "speculate iff draft_s+verify_nonread_s+predicted_miss_bytes/read_bandwidth"
                                 " <= useful_tokens/plain_tokens_per_second (request-measured EWMAs)",
                "admission_observations": dict(self.admission_stats)}),
            **({} if self.verify_schedule == "fixed" else {
                "verify_schedule_law": "argmax_k (1+sum_j<=k prod_i<=j sigmoid(c_i))/(draft_cost+cost[k+1])"
                + ("; cost[w]+=(w-1)*measured_miss_bytes_per_extra_row/bandwidth/plain_round_s"
                   if self.verify_schedule == "survival_miss_v1" else ""),
                "verify_costs": self.schedule_costs, "draft_cost": self.schedule_draft_cost,
                "schedule_observations": dict(self.schedule_stats)}),
            # Bounded counters: target expert loads per closed verify round,
            # by verify width (deepseek_v41_dspark_verify_ledger).
            "verify_expert_observations": copy.deepcopy(getattr(self, "verify_expert_stats", {})),
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
            if getattr(self.model, "verification_commit_policy", "replay") == "local_frontier":
                from .deepseek_v41_external import DeepSeekV41VerifyRows
                return DeepSeekV41VerifyRows(self.model, rows, self.layers)
            if getattr(self.model, "verify_rows_policy", "serial") != "serial":
                raise ValueError("V4.1 multirow verification requires local_frontier commits")
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
