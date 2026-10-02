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


class ExternalDraftAdapterMixin:
    EXTERNAL_DEFAULT_NUM_DRAFT = 3
    EXTERNAL_ROUTE_TAG = "external-draft-v1"
    EXTERNAL_PROFILE = "external-draft"

    def _parse_external_policy(self, execution_policy, *, family):
        self.external_policy = dict(execution_policy or {})
        self.draft_model = None
        if set(self.external_policy) - {
            "draft_model",
            "num_draft",
            "adaptive_verification",
            "proposal_composition",
            "continuation_pool",
            "lilicorr_feedback",
        } or (self.external_policy and not self.external_policy.get("draft_model")):
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
        adaptive = self.external_policy.get("adaptive_verification")
        if adaptive is not None:
            from ..runtime.acceptance_estimator import AdaptiveVerificationPolicy

            AdaptiveVerificationPolicy.from_value(adaptive, count)
        if "proposal_composition" in self.external_policy:
            from ..runtime.proposal_composition import ProposalCompositionPolicy

            ProposalCompositionPolicy.from_value(
                self.external_policy["proposal_composition"]
            )
        if "continuation_pool" in self.external_policy:
            from ..runtime.proposal_providers import ContinuationPoolPolicy

            ContinuationPoolPolicy.from_value(self.external_policy["continuation_pool"])
            if "proposal_composition" in self.external_policy:
                raise ValueError(
                    "continuation_pool already arbitrates proposal sources"
                )
            if not (
                hasattr(record["args"], "xpress_rank")
                or hasattr(record["args"], "lilicorr_candidate_topk")
            ):
                raise ValueError(
                    "continuation_pool requires a compatible deterministic external head"
                )
        if "lilicorr_feedback" in self.external_policy:
            from ..runtime.lilicorr_feedback import LiLiCorrFeedbackPolicy

            LiLiCorrFeedbackPolicy.from_value(self.external_policy["lilicorr_feedback"])
            if not hasattr(record["args"], "lilicorr_candidate_topk"):
                raise ValueError(
                    "lilicorr_feedback requires a compatible LiLiCoRR artifact"
                )

    def _bind_external_drafter(self, record, loader, base_descriptor):
        self.draft_model = loader(record, self.model)
        # Capture effective head geometry before wrappers add policy receipts.
        # Artifact bytes alone do not pin refinement passes or retained context.
        effective_draft_settings = (
            self.draft_model.receipt_settings
            if "continuation_pool" in self.external_policy
            else None
        )
        self._external_target_revision = self.identity["fingerprint"]
        self._external_draft_revision = record["fingerprint"]
        if "lilicorr_feedback" in self.external_policy and not hasattr(
            self.draft_model, "lilicorr"
        ):
            raise ValueError(
                "lilicorr_feedback requires an actual resident LiLiCoRR head"
            )
        composition_identity = ""
        if "proposal_composition" in self.external_policy:
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
                tokenizer_revision=self._external_target_revision,
                session_revision=session_revision,
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
            ).encode()
        ).hexdigest()
        self.identity = {
            **self.identity,
            "target_fingerprint": self.identity["fingerprint"],
            "draft_fingerprint": record["fingerprint"],
            "fingerprint": digest,
        }
        self.layout += f":{self.EXTERNAL_ROUTE_TAG}:" + digest
        self.descriptor = replace(
            base_descriptor,
            capabilities=base_descriptor.capabilities | {Capability.EXTERNAL_DRAFT},
            state_planes=base_descriptor.state_planes | {StatePlane.DRAFT},
            cache_layout=self.layout,
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
        return {
            "persistent": True,
            "num_draft": self._external_num_draft(),
            "backend": "external_draft",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
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

        adaptive = self.external_policy.get("adaptive_verification")
        if adaptive is not None:
            kwargs.setdefault("adaptive_verification", adaptive)
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
        self.identity = {**self.identity, "fingerprint": fingerprint}
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
        self._close_external_feedback()
        parent_close = getattr(super(), "close", None)
        if callable(parent_close):
            parent_close()


__all__ = ["ExternalDraftAdapterMixin"]
