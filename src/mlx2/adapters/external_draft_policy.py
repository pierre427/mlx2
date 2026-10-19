"""Shared adapter plumbing for a candidate external draft/verify route.

Each adapter still owns its drafter family, default depth and profile name;
this only removes the duplicated policy parsing, identity binding and batch
construction.  Nothing here qualifies a route.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from ..contracts import Capability, StatePlane

DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION = {
    "prompt_lookup": True,
    "ngram_min": 3,
    "ngram_max": 6,
    "lookback": 256,
    "min_context_match": 4,
    "max_sources": 8,
}

COMPOSITION_DEFAULT_SKIPPED_NO_EXACT_LAW = (
    "composition default skipped: backend lacks exact proposal-law support"
)
PROGRESSIVE_VERIFICATION_TILE_AUTO = "auto"


def progressive_external_terminal_input_widths(target, *, max_lanes, num_draft):
    """Return executor-declared exact terminal target-input widths.

    These are actual ``inputs.shape[1]`` widths, including the anchor and the
    bonus-law input, not proposal tile widths.  Keeping this a separate
    capability prevents an automatic K-wide tile from silently treating a
    K-row non-final executor claim as evidence for its K+1-row terminal slab.
    """

    widths = getattr(
        target, "progressive_external_exact_terminal_input_widths", None
    )
    if widths is None:
        raise ValueError(
            "automatic progressive verification requires target-declared exact "
            "terminal input widths"
        )
    if callable(widths):
        widths = widths(max_lanes=max_lanes, num_draft=int(num_draft))
    if not isinstance(widths, (tuple, list, range)) or not widths or any(
        type(width) is not int or width < 2 for width in widths
    ):
        raise ValueError(
            "target exact terminal input widths must be nonempty integers "
            "of at least two"
        )
    return tuple(dict.fromkeys(widths))


def normalize_progressive_external_policy(policy, *, num_draft, minimum_proposals=1):
    """Validate shared external progressive-verification options."""
    policy = dict(policy)
    tile = policy.get("progressive_verification_tile")
    fidelity = policy.get("progressive_verification_fidelity")
    if tile is None and fidelity is not None:
        raise ValueError(
            "progressive_verification_fidelity requires progressive_verification_tile"
        )
    if tile is not None:
        from ..runtime.progressive_external_verify import (
            PROGRESSIVE_FIDELITIES,
            PROGRESSIVE_FIDELITY_EXACT,
        )

        if tile != PROGRESSIVE_VERIFICATION_TILE_AUTO and (
            type(tile) is not int or not 1 <= tile < num_draft
        ):
            raise ValueError(
                "progressive_verification_tile must be 'auto' or an integer "
                "from 1 below num_draft"
            )
        if policy.get("proposal_composition") not in (None, False):
            raise ValueError(
                "progressive_verification_tile cannot combine with proposal composition"
            )
        # A requested tile must not silently acquire the PLD composition default.
        policy["proposal_composition"] = False
        if policy.get("adaptive_verification") is not None:
            raise ValueError(
                "progressive_verification_tile cannot combine with adaptive verification"
            )
        pool_branches = policy.get("progressive_alternative_branches", 1)
        if policy.get("continuation_pool") is not None and (
            type(pool_branches) is not int or pool_branches <= 1
        ):
            raise ValueError(
                "progressive continuation_pool requires ordered alternative branches"
            )
        if policy.get("lilicorr_feedback") is not None:
            raise ValueError(
                "progressive_verification_tile cannot combine with LiLiCorr feedback"
            )
        if fidelity is None:
            fidelity = PROGRESSIVE_FIDELITY_EXACT
        if fidelity not in PROGRESSIVE_FIDELITIES:
            raise ValueError("unsupported progressive verification fidelity")
        policy["progressive_verification_fidelity"] = fidelity
    cap = policy.get("progressive_multilane_draft_cap")
    if cap is not None and (
        tile is None
        or type(cap) is not int
        or not min(int(minimum_proposals), int(num_draft)) <= cap <= num_draft
    ):
        raise ValueError(
            "progressive_multilane_draft_cap requires progressive verification "
            "and must be between minimum proposals and num_draft"
        )
    alternatives = policy.get("progressive_alternative_branches", 1)
    if (
        type(alternatives) is not int
        or not 1 <= alternatives <= 4
        or alternatives > 1 and tile is None
    ):
        raise ValueError(
            "progressive_alternative_branches must be 1 through 4 and requires "
            "progressive verification"
        )
    alternative_strategy = policy.get(
        "progressive_alternative_strategy", "distinct_root"
    )
    alternative_depth_policy = policy.get(
        "progressive_alternative_depth_policy", "full"
    )
    rootless_fallback = policy.get("progressive_ast_rootless_fallback", True)
    if type(rootless_fallback) is not bool:
        raise ValueError("progressive_ast_rootless_fallback must be boolean")
    cross_session_pack_rows = policy.get("progressive_cross_session_pack_rows")
    if (
        cross_session_pack_rows is not None
        and cap is not None
        and type(tile) is int
        and cap <= tile
    ):
        raise ValueError(
            "cross-session AST requires progressive_multilane_draft_cap to "
            "exceed progressive_verification_tile"
        )
    if cross_session_pack_rows is not None and (
        cross_session_pack_rows not in {4, 8}
        or alternatives <= 1
        or alternative_strategy != "confidence_tree"
    ):
        raise ValueError(
            "progressive_cross_session_pack_rows must be 4 or 8 and requires "
            "confidence_tree alternatives"
        )
    cross_session_pack_mode = policy.get("progressive_cross_session_pack_mode")
    if cross_session_pack_mode not in {None, "ordered_live_frontier"}:
        raise ValueError(
            "progressive_cross_session_pack_mode must be ordered_live_frontier"
        )
    if (
        cross_session_pack_mode == "ordered_live_frontier"
        and cross_session_pack_rows is None
    ):
        raise ValueError(
            "ordered_live_frontier requires progressive_cross_session_pack_rows"
        )
    if alternative_strategy not in {
        "distinct_root",
        "shared_prefix",
        "paired_continuations",
        "confidence_tree",
    }:
        raise ValueError("unsupported progressive alternative strategy")
    if (
        alternative_strategy
        in {"shared_prefix", "paired_continuations", "confidence_tree"}
        and alternatives <= 1
    ):
        raise ValueError(
            "shared-prefix progressive alternatives require multiple branches"
        )
    if alternative_depth_policy not in {"full", "confidence_tail"}:
        raise ValueError("unsupported progressive alternative depth policy")
    if alternative_depth_policy == "confidence_tail" and (
        alternative_strategy != "confidence_tree" or alternatives < 3
    ):
        raise ValueError(
            "confidence_tail requires confidence_tree with at least three branches"
        )
    if not rootless_fallback and (
        tile is None
        or policy.get("continuation_pool") is None
        or alternatives <= 1
        or alternative_strategy != "confidence_tree"
    ):
        raise ValueError(
            "progressive_ast_rootless_fallback=false requires a continuation_pool "
            "confidence_tree with exact progressive alternatives"
        )
    # Explicit true is the compatibility spelling of omission. Keep old route
    # identities stable; only the non-default refusal changes execution law.
    if rootless_fallback:
        policy.pop("progressive_ast_rootless_fallback", None)
    return policy


def resolve_progressive_external_tile(policy, *, num_draft, target, max_lanes):
    """Resolve automatic packing only from declared exact target geometry."""
    requested = policy.get("progressive_verification_tile")
    if type(max_lanes) is not int or max_lanes < 1:
        raise ValueError("progressive verification max_lanes must be a positive integer")
    if requested != PROGRESSIVE_VERIFICATION_TILE_AUTO:
        if (
            requested is not None
            and policy.get("progressive_verification_fidelity", "exact") == "exact"
            and max_lanes > 1
        ):
            minimum = max(
                1,
                int(
                    getattr(
                        target,
                        "progressive_external_lane_exact_min_width",
                        1,
                    )
                ),
            )
            if requested < minimum:
                raise ValueError(
                    "exact progressive verification tile is narrower than the "
                    "target's declared multi-lane exact width: "
                    f"tile={requested}, minimum={minimum}"
                )
        return requested

    if policy.get("progressive_verification_fidelity", "exact") != "exact":
        raise ValueError("automatic progressive verification requires exact fidelity")
    widths = getattr(target, "progressive_external_exact_tile_widths", None)
    maximum_lanes = getattr(target, "progressive_external_exact_max_lanes", None)
    if widths is None or maximum_lanes is None:
        raise ValueError(
            "automatic progressive verification requires target-declared exact "
            "tile widths and maximum lanes"
        )
    if callable(widths):
        widths = widths(max_lanes=max_lanes, num_draft=int(num_draft))
    if type(maximum_lanes) is not int or maximum_lanes < 1:
        raise ValueError("target exact maximum lanes must be a positive integer")
    if not 1 <= max_lanes <= maximum_lanes:
        raise ValueError(
            "automatic progressive verification cohort exceeds the target's "
            f"declared exact lane limit: lanes={max_lanes}, maximum={maximum_lanes}"
        )
    if not isinstance(widths, (tuple, list, range)) or any(
        type(width) is not int or width < 1 for width in widths
    ):
        raise ValueError("target exact tile widths must be positive integers")

    terminal_widths = progressive_external_terminal_input_widths(
        target, max_lanes=max_lanes, num_draft=num_draft
    )

    simple_primary = (
        policy.get("progressive_alternative_branches", 1) == 1
        and policy.get("continuation_pool") is None
        and policy.get("progressive_cross_session_pack_rows") is None
        and policy.get("progressive_cross_session_pack_mode") is None
    )
    # A simple primary chain may verify all K proposals and the bonus law in
    # one terminal K+1-input target launch.  Alternative/continuation routes
    # still need a suffix beyond the tile and therefore retain the K-1 bound.
    ceiling = int(num_draft) - 1
    if simple_primary and all(
        width in terminal_widths for width in range(2, int(num_draft) + 2)
    ):
        ceiling = int(num_draft)
    cap = policy.get("progressive_multilane_draft_cap")
    pool = policy.get("continuation_pool")
    planned_cohort = isinstance(pool, dict) and pool.get("ast_planner") is True
    # The planner prices actual ready rows. A future multi-request depth cap
    # must not shrink its singleton capability ceiling at server startup.
    # _propose still enforces the explicit cap when multiple requests live.
    if max_lanes > 1 and cap is not None and not planned_cohort:
        ceiling = min(ceiling, int(cap))
        if policy.get("progressive_cross_session_pack_rows") is not None:
            ceiling = min(ceiling, int(cap) - 1)
    minimum = (
        max(
            1,
            int(
                getattr(
                    target,
                    "progressive_external_lane_exact_min_width",
                    1,
                )
            ),
        )
        if max_lanes > 1
        else 1
    )
    feasible = [
        width
        for width in widths
        if minimum <= width <= ceiling
        and all(
            terminal_width in terminal_widths
            for terminal_width in range(2, min(width, int(num_draft)) + 2)
        )
    ]
    if not feasible:
        raise ValueError(
            "automatic progressive verification has no target-declared exact "
            "tile for the proposal/cohort geometry: "
            f"num_draft={num_draft}, lanes={max_lanes}, minimum={minimum}, "
            f"ceiling={ceiling}"
        )
    return max(feasible)


def progressive_external_identity(policy, binding=None):
    tile = policy.get("progressive_verification_tile")
    if tile is None:
        return ""
    from ..runtime.progressive_external_verify import (
        PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
    )

    identity = {
        "algorithm": PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
        "verification_tile": tile,
        "multilane_draft_cap": policy.get("progressive_multilane_draft_cap"),
        "alternative_branches": policy.get("progressive_alternative_branches", 1),
        "alternative_strategy": policy.get(
            "progressive_alternative_strategy", "distinct_root"
        ),
        "alternative_depth_policy": policy.get(
            "progressive_alternative_depth_policy", "full"
        ),
        "cross_session_pack_rows": policy.get(
            "progressive_cross_session_pack_rows"
        ),
        "cross_session_pack_mode": policy.get(
            "progressive_cross_session_pack_mode"
        ),
        "target_protocol": None if binding is None else binding["target_protocol"],
        "fidelity": policy.get("progressive_verification_fidelity", "exact"),
        "source_hashes": None if binding is None else binding["source_hashes"],
    }
    if policy.get("progressive_ast_rootless_fallback") is False:
        identity["ast_rootless_fallback"] = False
    return json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    )


class ExternalDraftAdapterMixin:
    EXTERNAL_DEFAULT_NUM_DRAFT = 3
    EXTERNAL_ROUTE_TAG = "external-draft-v1"
    EXTERNAL_PROFILE = "external-draft"

    def _parse_external_policy(self, execution_policy, *, family):
        self.external_policy = dict(execution_policy or {})
        self.draft_model = None
        allowed = {
            "draft_model",
            "num_draft",
            "adaptive_verification",
            "proposal_composition",
            "continuation_pool",
            "continuation_strategy",
            "lilicorr_feedback",
            "progressive_verification_tile",
            "progressive_multilane_draft_cap",
            "progressive_alternative_branches",
            "progressive_alternative_strategy",
            "progressive_alternative_depth_policy",
            "progressive_ast_rootless_fallback",
            "progressive_cross_session_pack_rows",
            "progressive_cross_session_pack_mode",
            "progressive_verification_fidelity",
            "cost_per_commit_handoff",
        } | set(getattr(self, "EXTERNAL_POLICY_EXTENSIONS", ()))
        if set(self.external_policy) - allowed or (
            self.external_policy and not self.external_policy.get("draft_model")
        ):
            raise ValueError(
                f"{family} execution policy has no qualified overrides; only an "
                "external draft_model, num_draft and adaptive_verification may be configured"
            )
        return bool(self.external_policy)

    def _check_num_draft(self, record):
        count = self.external_policy.get("num_draft", self.EXTERNAL_DEFAULT_NUM_DRAFT)
        if type(count) is not int or not 1 <= count < record["args"].block_size:
            raise ValueError(
                "num_draft must be a positive integer below the draft block size"
            )
        minimum = min(
            int(getattr(record["args"], "minimum_proposal_length", 1)), count
        )
        self.external_policy = normalize_progressive_external_policy(
            self.external_policy,
            num_draft=count,
            minimum_proposals=minimum,
        )
        adaptive = self.external_policy.get("adaptive_verification")
        if adaptive is not None:
            from ..runtime.acceptance_estimator import AdaptiveVerificationPolicy

            AdaptiveVerificationPolicy.from_value(adaptive, count)
        if self.external_policy.get("cost_per_commit_handoff") is not None:
            from ..runtime.cost_per_commit_handoff import (
                CostPerCommitHandoffSettings,
            )

            CostPerCommitHandoffSettings.from_value(
                self.external_policy["cost_per_commit_handoff"]
            )
        if self.external_policy.get("proposal_composition") is not False and (
            "proposal_composition" in self.external_policy
        ):
            from ..runtime.proposal_composition import ProposalCompositionPolicy

            composition = ProposalCompositionPolicy.from_value(
                self.external_policy["proposal_composition"]
            )
            if composition.trusted_pld and adaptive is not None:
                raise ValueError(
                    "trusted PLD cannot combine with adaptive verification"
                )
        if "continuation_pool" in self.external_policy:
            from ..runtime.proposal_providers import ContinuationPoolPolicy

            ContinuationPoolPolicy.from_value(self.external_policy["continuation_pool"])
            if self.external_policy.get("proposal_composition") is not False and (
                "proposal_composition" in self.external_policy
            ):
                raise ValueError(
                    "continuation_pool already arbitrates proposal sources"
                )
            if not (
                hasattr(record["args"], "xpress_rank")
                or hasattr(record["args"], "lilicorr_candidate_topk")
                or record["config"].get("architectures") == ["DFlash2DraftModel"]
                or getattr(record["args"], "native_mtp_head", False) is True
            ):
                raise ValueError(
                    "continuation_pool requires a compatible complete-path external head"
                )
        if "continuation_strategy" in self.external_policy:
            if self.external_policy["continuation_strategy"] != "longest_first_exact_prefix":
                raise ValueError("unsupported continuation_strategy")
            if "continuation_pool" not in self.external_policy:
                raise ValueError("continuation_strategy requires continuation_pool")
        if "lilicorr_feedback" in self.external_policy:
            from ..runtime.lilicorr_feedback import LiLiCorrFeedbackPolicy

            LiLiCorrFeedbackPolicy.from_value(self.external_policy["lilicorr_feedback"])
            if not hasattr(record["args"], "lilicorr_candidate_topk"):
                raise ValueError(
                    "lilicorr_feedback requires a compatible LiLiCoRR artifact"
                )

    def _bind_external_drafter(self, record, loader, base_descriptor):
        self._progressive_external_binding = None
        if self.external_policy.get("progressive_verification_tile") is not None:
            from ..runtime.progressive_external_verify import (
                progressive_target_protocol_binding,
            )

            self._progressive_external_binding = progressive_target_protocol_binding(
                self.model,
                self.external_policy.get("progressive_verification_fidelity", "exact"),
            )
        self.draft_model = loader(record, self.model)
        fold_identity = ""
        fold_value = self.external_policy.get("fold")
        if fold_value is not None and fold_value is not False:
            from ..runtime.drafters.fold import FOLD_NUMERICS_VERSION, configure

            policy = configure(self.draft_model, self.external_policy["fold"])
            fold_identity = json.dumps(
                {"fold": policy.as_dict(), "fold_numerics": FOLD_NUMERICS_VERSION},
                sort_keys=True, separators=(",", ":")
            )
        if (
            "proposal_composition" not in self.external_policy
            and "continuation_pool" not in self.external_policy
            # Composition verifies chains; a tree route (e.g. the pinned
            # Qwen3.8 pair's tree15 batch-size route) keeps its own drafts.
            and self.external_policy.get("batch_size_route") is None
            and self.external_policy.get("progressive_verification_tile") is None
            and not bool(getattr(self.draft_model, "requires_context_tokens", False))
            and callable(getattr(self.draft_model, "draft_distributions", None))
        ):
            from ..runtime.proposal_composition import ComposedDraftModel

            if ComposedDraftModel.supports_backend(self.draft_model):
                self.external_policy["proposal_composition"] = dict(
                    DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION
                )
            else:
                # A default must never turn a servable route into a startup
                # crash (Laguna's DFlash head publishes no exact law).  An
                # explicit request still reaches ComposedDraftModel below and
                # fails closed there.
                skipped = dict(getattr(self, "skipped_route_defaults", None) or {})
                skipped["proposal_composition"] = (
                    COMPOSITION_DEFAULT_SKIPPED_NO_EXACT_LAW
                )
                self.skipped_route_defaults = skipped
        # Capture effective head geometry before wrappers add policy receipts.
        # Artifact bytes alone do not pin refinement passes or retained context.
        effective_draft_settings = (
            self.draft_model.receipt_settings
            if "continuation_pool" in self.external_policy
            else None
        )
        target_artifact_revision = self.identity["fingerprint"]
        self._external_target_revision = target_artifact_revision
        self._external_draft_revision = record["fingerprint"]
        # Direct adapter users construct shared source critics without a serving
        # engine. Pin adapter-owned target math before creating those sources.
        describe = getattr(self, "execution_numerics_contract", None)
        if describe is not None and not callable(describe):
            raise ValueError("adapter execution_numerics_contract must be callable")
        adapter_numerics = None if describe is None else describe()
        numerical_identity = ""
        if adapter_numerics is not None:
            if not isinstance(adapter_numerics, dict) or not adapter_numerics:
                raise ValueError(
                    "adapter numerical contract must be a nonempty object or None"
                )
            numerical_identity = json.dumps(
                {
                    "schema": "mlx2.adapter-execution-numerics.v1",
                    "contract": adapter_numerics,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            self._external_target_revision = hashlib.sha256(
                (target_artifact_revision + numerical_identity).encode()
            ).hexdigest()
        if "lilicorr_feedback" in self.external_policy and not hasattr(
            self.draft_model, "lilicorr"
        ):
            raise ValueError(
                "lilicorr_feedback requires an actual resident LiLiCoRR head"
            )
        composition_identity = ""
        if self.external_policy.get("proposal_composition") is not False and (
            "proposal_composition" in self.external_policy
        ):
            from ..runtime.proposal_composition import ComposedDraftModel
            from .proposal_sources import native_mtp_source

            value = self.external_policy["proposal_composition"]
            source = (
                native_mtp_source(self.model)
                if value.get("native_mtp", False)
                else None
            )
            self.draft_model = ComposedDraftModel(
                self.draft_model, value, native_mtp_source=source
            )
            composition_identity = json.dumps(
                self.draft_model.policy.as_dict(), sort_keys=True, separators=(",", ":")
            )
        if "continuation_pool" in self.external_policy:
            from ..runtime.acceptance_estimator import AdaptiveVerificationPolicy
            from ..runtime.proposal_providers import ContinuationPoolPolicy
            from .proposal_path_sources import build_continuation_drafter

            value = self.external_policy["continuation_pool"]
            # Normalized pool math includes its fixed verification algorithm,
            # binding both APC route identity and critic source/session state.
            normalized = ContinuationPoolPolicy.from_value(value).as_dict()
            composition_identity = json.dumps(
                normalized, sort_keys=True, separators=(",", ":")
            )
            if "continuation_strategy" in self.external_policy:
                composition_identity += json.dumps(
                    {
                        "continuation_strategy": self.external_policy[
                            "continuation_strategy"
                        ]
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            adaptive = AdaptiveVerificationPolicy.from_value(
                self.external_policy.get("adaptive_verification"),
                self._external_num_draft(),
            )
            adaptive_settings = None if adaptive is None else asdict(adaptive)
            if adaptive_settings is not None:
                adaptive_settings["draft_cost"] = float(adaptive.draft_cost)
                adaptive_settings["min_gain"] = float(adaptive.min_gain)
            session_revision = hashlib.sha256(
                json.dumps(
                    {
                        "schema": "mlx2.continuation-session.v2",
                        "target_revision": self._external_target_revision,
                        "draft_revision": self._external_draft_revision,
                        "route": self.EXTERNAL_ROUTE_TAG,
                        "continuation_pool": normalized,
                        "continuation_strategy": self.external_policy.get(
                            "continuation_strategy"
                        ),
                        "draft_settings": effective_draft_settings,
                        "num_draft": self._external_num_draft(),
                        "adaptive_verification": adaptive_settings,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            self.draft_model = build_continuation_drafter(
                self.model,
                self.draft_model,
                value,
                target_revision=self._external_target_revision,
                draft_revision=self._external_draft_revision,
                tokenizer_revision=target_artifact_revision,
                session_revision=session_revision,
            )
        composition_identity += progressive_external_identity(
            self.external_policy, self._progressive_external_binding
        )
        if "draft_weight_quantization" in self.external_policy:
            composition_identity += json.dumps(
                {
                    "draft_weight_quantization": self.external_policy[
                        "draft_weight_quantization"
                    ]
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        if "lilicorr_feedback" in self.external_policy:
            from ..runtime.lilicorr_feedback import LiLiCorrFeedbackPolicy

            composition_identity += json.dumps(
                asdict(
                    LiLiCorrFeedbackPolicy.from_value(
                        self.external_policy["lilicorr_feedback"]
                    )
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
        self.profile_name = self.external_profile_name
        digest = hashlib.sha256(
            (
                self.identity["fingerprint"]
                + record["fingerprint"]
                + self.EXTERNAL_ROUTE_TAG
                + composition_identity
                # Wrapper policy must not erase the drafter cache's FOLD law.
                + fold_identity
                + numerical_identity
            ).encode()
        ).hexdigest()
        self.identity = {
            **self.identity,
            "target_fingerprint": self.identity["fingerprint"],
            "draft_fingerprint": record["fingerprint"],
            "fingerprint": digest,
            **(
                {"target_execution_fingerprint": self._external_target_revision}
                if numerical_identity else {}
            ),
        }
        self.layout += f":{self.EXTERNAL_ROUTE_TAG}:" + digest
        self.descriptor = replace(
            base_descriptor,
            capabilities=base_descriptor.capabilities | {Capability.EXTERNAL_DRAFT},
            state_planes=base_descriptor.state_planes | {StatePlane.DRAFT},
            cache_layout=self.layout,
            metadata={
                **(base_descriptor.metadata or {}),
                **(
                    {
                        "progressive_verification": {
                            "implemented": True,
                            "qualified": False,
                            "selected": True,
                            "verification_tile": self.external_policy[
                                "progressive_verification_tile"
                            ],
                            "multilane_draft_cap": self.external_policy.get(
                                "progressive_multilane_draft_cap"
                            ),
                            "cross_session_apt_pack_rows": (
                                self.external_policy.get(
                                    "progressive_cross_session_pack_rows"
                                )
                            ),
                            "cross_session_ast_pack_rows": (
                                self.external_policy.get(
                                    "progressive_cross_session_pack_rows"
                                )
                            ),
                            "cross_session_apt_pack_mode": self.external_policy.get(
                                "progressive_cross_session_pack_mode"
                            ),
                            "cross_session_ast_pack_mode": self.external_policy.get(
                                "progressive_cross_session_pack_mode"
                            ),
                            "deprecated_cross_session_apt_alias": True,
                            "algorithm": self._progressive_external_binding[
                                "algorithm"
                            ],
                            "target_protocol": self._progressive_external_binding[
                                "target_protocol"
                            ],
                            "fidelity": self._progressive_external_binding[
                                "fidelity"
                            ],
                            "source_hashes": self._progressive_external_binding[
                                "source_hashes"
                            ],
                            "performance_claim": False,
                        }
                    }
                    if self.external_policy.get("progressive_verification_tile")
                    is not None
                    else {}
                ),
            },
        )

    @classmethod
    def external_profile_name(cls, mtp):
        if mtp:
            raise ValueError("External draft is not native MTP")
        return cls.EXTERNAL_PROFILE

    def _external_num_draft(self):
        return self.external_policy.get("num_draft", self.EXTERNAL_DEFAULT_NUM_DRAFT)

    def _external_execution_config(self, *, max_lanes, prefill_step):
        """Only the external route adds keys; the ordinary config is unchanged."""
        progressive_binding = getattr(self, "_progressive_external_binding", None)
        progressive_fidelity = self.external_policy.get(
            "progressive_verification_fidelity", "exact"
        )
        progressive_tile = self.external_policy.get(
            "progressive_verification_tile"
        )
        if (
            progressive_tile is not None
            and progressive_fidelity != "exact"
            and progressive_binding is None
        ):
            raise ValueError(
                "approximate progressive verification requires a bound target "
                "protocol before execution configuration"
            )
        resolved_tile = resolve_progressive_external_tile(
            self.external_policy,
            num_draft=self._external_num_draft(),
            target=getattr(self, "model", None),
            max_lanes=max_lanes,
        )
        terminal_input_widths = (
            progressive_external_terminal_input_widths(
                getattr(self, "model", None),
                max_lanes=max_lanes,
                num_draft=self._external_num_draft(),
            )
            if progressive_tile == PROGRESSIVE_VERIFICATION_TILE_AUTO
            else None
        )
        return {
            "persistent": True,
            "num_draft": self._external_num_draft(),
            "backend": "external_draft",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
            **(
                {
                    "cost_per_commit_handoff": self.external_policy[
                        "cost_per_commit_handoff"
                    ]
                }
                if self.external_policy.get("cost_per_commit_handoff") is not None
                else {}
            ),
            **(
                {
                    "progressive_verification_tile": resolved_tile,
                    **(
                        {"progressive_verification_tile_policy": "auto"}
                        if progressive_tile == PROGRESSIVE_VERIFICATION_TILE_AUTO
                        else {}
                    ),
                    **(
                        {
                            "progressive_terminal_input_widths": (
                                terminal_input_widths
                            )
                        }
                        if terminal_input_widths is not None
                        else {}
                    ),
                    "progressive_multilane_draft_cap": self.external_policy.get(
                        "progressive_multilane_draft_cap"
                    ),
                    "progressive_alternative_branches": self.external_policy.get(
                        "progressive_alternative_branches", 1
                    ),
                    "progressive_alternative_strategy": self.external_policy.get(
                        "progressive_alternative_strategy", "distinct_root"
                    ),
                    "progressive_alternative_depth_policy": self.external_policy.get(
                        "progressive_alternative_depth_policy", "full"
                    ),
                    "progressive_ast_rootless_fallback": self.external_policy.get(
                        "progressive_ast_rootless_fallback", True
                    ),
                    "progressive_cross_session_pack_rows": self.external_policy.get(
                        "progressive_cross_session_pack_rows"
                    ),
                    "progressive_cross_session_pack_mode": self.external_policy.get(
                        "progressive_cross_session_pack_mode"
                    ),
                    "progressive_verification_fidelity": progressive_fidelity,
                    **(
                        {
                            "progressive_target_protocol": progressive_binding[
                                "target_protocol"
                            ],
                            "progressive_protocol_binding": progressive_binding,
                        }
                        if progressive_binding is not None
                        else {}
                    ),
                }
                if progressive_tile is not None
                else {}
            ),
        }

    def create_external_batch(self, **kwargs):
        """Candidate external draft/verify batch; implemented, not qualified."""
        if getattr(self, "draft_model", None) is None:
            raise ValueError("No external draft model bound")
        from ..runtime.external_speculative import ExternalDraftBatchGenerator

        self._initialize_external_feedback()
        self._external_execution_started = True
        if hasattr(self.draft_model, "last_continuation_selections"):
            kwargs.setdefault("continuation_pool", self.draft_model.policy)
            strategy = getattr(self, "continuation_verification_strategy", None)
            if callable(strategy):
                selected_strategy = strategy()
                if selected_strategy is not None:
                    kwargs.setdefault(
                        "continuation_verification_strategy", selected_strategy
                    )

        adaptive = self.external_policy.get("adaptive_verification")
        if adaptive is not None:
            kwargs.setdefault("adaptive_verification", adaptive)
        if self.external_policy.get("cost_per_commit_handoff") is not None:
            kwargs.setdefault(
                "cost_per_commit_handoff",
                self.external_policy["cost_per_commit_handoff"],
            )
        if self.external_policy.get("progressive_verification_tile") is not None:
            requested_tile = self.external_policy["progressive_verification_tile"]
            batch_lanes = kwargs.get("completion_batch_size")
            if (
                requested_tile == PROGRESSIVE_VERIFICATION_TILE_AUTO
                and batch_lanes is None
            ):
                raise ValueError(
                    "automatic progressive verification requires "
                    "completion_batch_size at batch construction"
                )
            resolved_tile = resolve_progressive_external_tile(
                self.external_policy,
                num_draft=self._external_num_draft(),
                target=self.model,
                max_lanes=1 if batch_lanes is None else batch_lanes,
            )
            if "continuation_pool" in self.external_policy:
                kwargs.setdefault("continuation_pool_ast_bridge", True)
            kwargs.setdefault(
                "progressive_verification_tile",
                resolved_tile,
            )
            if requested_tile == PROGRESSIVE_VERIFICATION_TILE_AUTO:
                kwargs.setdefault("progressive_verification_tile_policy", "auto")
                kwargs.setdefault(
                    "progressive_terminal_input_widths",
                    progressive_external_terminal_input_widths(
                        self.model,
                        max_lanes=batch_lanes,
                        num_draft=self._external_num_draft(),
                    ),
                )
            kwargs.setdefault(
                "multilane_draft_cap",
                self.external_policy.get("progressive_multilane_draft_cap"),
            )
            kwargs.setdefault(
                "progressive_alternative_branches",
                self.external_policy.get("progressive_alternative_branches", 1),
            )
            kwargs.setdefault(
                "progressive_alternative_strategy",
                self.external_policy.get(
                    "progressive_alternative_strategy", "distinct_root"
                ),
            )
            kwargs.setdefault(
                "progressive_alternative_depth_policy",
                self.external_policy.get(
                    "progressive_alternative_depth_policy", "full"
                ),
            )
            kwargs.setdefault(
                "progressive_ast_rootless_fallback",
                self.external_policy.get("progressive_ast_rootless_fallback", True),
            )
            kwargs.setdefault(
                "progressive_cross_session_pack_rows",
                self.external_policy.get("progressive_cross_session_pack_rows"),
            )
            kwargs.setdefault(
                "progressive_cross_session_pack_mode",
                self.external_policy.get("progressive_cross_session_pack_mode"),
            )
            kwargs.setdefault(
                "progressive_verification_fidelity",
                self.external_policy.get("progressive_verification_fidelity", "exact"),
            )
            kwargs.setdefault(
                "progressive_target_protocol",
                self._progressive_external_binding["target_protocol"],
            )
            kwargs.setdefault(
                "progressive_protocol_binding", self._progressive_external_binding
            )
        return ExternalDraftBatchGenerator(
            self.model,
            draft_model=self.draft_model,
            binding=self.identity["fingerprint"],
            num_draft=self._external_num_draft(),
            **kwargs,
        )

    def bind_external_serving_namespace(self, namespace):
        """Bind effective target numerical laws before any cache or learning.

        Serving passes the same composed namespace used by APCv2. Its default
        is an identity, preserving existing ordinary/chain/pool route hashes.
        """
        if (
            getattr(self, "draft_model", None) is None
            or namespace == "external-learning-v1"
        ):
            return False
        payload = json.dumps(
            namespace, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        previous = getattr(self, "_external_serving_namespace", None)
        if previous == payload:
            return False
        if (
            previous is not None
            or getattr(self, "_external_execution_started", False)
            or getattr(self, "_external_feedback_manager", None) is not None
        ):
            raise ValueError(
                "external serving numerical laws must bind before execution or feedback"
            )
        old = self.draft_model
        if hasattr(old, "proposal_pool") and (
            old.proposal_pool._pending or old.proposal_pool.feedback_revision
        ):
            raise ValueError(
                "external serving numerical laws require an unconsumed continuation pool"
            )
        target_revision = hashlib.sha256(
            (self._external_target_revision + payload).encode()
        ).hexdigest()
        replacement = old
        if hasattr(old, "proposal_pool"):
            from ..runtime.proposal_providers import ContinuationDraftModel

            session_revision = hashlib.sha256(
                (old.session.session_revision + payload).encode()
            ).hexdigest()
            session = replace(
                old.session,
                session_revision=session_revision,
                target_revision=target_revision,
            )
            records = {
                name: replace(
                    record,
                    session_revision=session_revision,
                    target_revision=target_revision,
                )
                for name, record in old.source_records.items()
            }
            replacement = ContinuationDraftModel(
                old.backend,
                old.policy.as_dict(),
                session=session,
                source_records=records,
                providers=old.providers,
            )
            replacement.proposal_pool.ranking_registry = (
                old.proposal_pool.ranking_registry
            )
        fingerprint = hashlib.sha256(
            (self.identity["fingerprint"] + payload).encode()
        ).hexdigest()
        self.draft_model = replacement
        self._external_target_revision = target_revision
        self._external_serving_namespace = payload
        self.identity = {
            **self.identity, "fingerprint": fingerprint,
            "target_execution_fingerprint": target_revision,
        }
        self.layout += ":external-numerical-laws:" + fingerprint
        self.descriptor = replace(self.descriptor, cache_layout=self.layout)
        return True

    def _initialize_external_feedback(self):
        if (
            "lilicorr_feedback" not in self.external_policy
            or getattr(self, "_external_feedback_manager", None) is not None
        ):
            return
        from ..runtime.lilicorr_feedback import LiLiCorrFeedbackManager

        drafter = self.draft_model
        while hasattr(drafter, "backend"):
            drafter = drafter.backend
        manager = LiLiCorrFeedbackManager(
            drafter,
            self.external_policy["lilicorr_feedback"],
            target_revision=self._external_target_revision,
            draft_revision=self._external_draft_revision,
            binding=self.identity["fingerprint"],
        )
        self._external_feedback_manager = manager
        drafter.feedback_manager = manager

    def _close_external_feedback(self):
        manager = getattr(self, "_external_feedback_manager", None)
        if manager is not None:
            manager.close()

    def close(self):
        try:
            self._close_external_feedback()
        finally:
            parent_close = getattr(super(), "close", None)
            if callable(parent_close):
                parent_close()


__all__ = [
    "PROGRESSIVE_VERIFICATION_TILE_AUTO",
    "ExternalDraftAdapterMixin",
    "normalize_progressive_external_policy",
    "progressive_external_identity",
    "resolve_progressive_external_tile",
]
