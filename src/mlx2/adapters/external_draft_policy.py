"""Shared adapter plumbing for a candidate external draft/verify route.

Each adapter still owns its drafter family, default depth and profile name;
this only removes the duplicated policy parsing, identity binding and batch
construction.  Nothing here qualifies a route.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

from ..contracts import Capability, StatePlane


class ExternalDraftAdapterMixin:
    EXTERNAL_DEFAULT_NUM_DRAFT = 3
    EXTERNAL_ROUTE_TAG = "external-draft-v1"
    EXTERNAL_PROFILE = "external-draft"

    def _parse_external_policy(self, execution_policy, *, family):
        self.external_policy = dict(execution_policy or {})
        self.draft_model = None
        if set(self.external_policy) - {"draft_model", "num_draft"} or (
            self.external_policy and not self.external_policy.get("draft_model")
        ):
            raise ValueError(
                f"{family} execution policy has no qualified overrides; only an "
                "external draft_model (and num_draft) may be configured"
            )
        return bool(self.external_policy)

    def _check_num_draft(self, record):
        count = self.external_policy.get("num_draft", self.EXTERNAL_DEFAULT_NUM_DRAFT)
        if type(count) is not int or not 1 <= count < record["args"].block_size:
            raise ValueError("num_draft must be a positive integer below the draft block size")

    def _bind_external_drafter(self, record, loader, base_descriptor):
        self.draft_model = loader(record, self.model)
        self.profile_name = self.external_profile_name
        digest = hashlib.sha256(
            (self.identity["fingerprint"] + record["fingerprint"] + self.EXTERNAL_ROUTE_TAG).encode()
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

        return ExternalDraftBatchGenerator(
            self.model,
            draft_model=self.draft_model,
            binding=self.identity["fingerprint"],
            num_draft=self._external_num_draft(),
            **kwargs,
        )


__all__ = ["ExternalDraftAdapterMixin"]
