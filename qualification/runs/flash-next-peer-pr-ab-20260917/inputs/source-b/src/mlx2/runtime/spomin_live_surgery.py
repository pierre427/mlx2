"""Serving-safe bridge for revision-bound Spomin Qwen4 KV surgery."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any

from .spomin_layer import (
    SpominCapabilityError,
    SpominConfig,
    SpominLayer,
    SpominPlan,
    SpominTargetState,
)
from .spomin_qwen4_surgery import Qwen4SpominSurgeryBackend


@dataclass(frozen=True)
class LiveSurgeryEpoch:
    request_id: str
    generation: int


@dataclass
class LiveSurgeryTransaction:
    manager: SpominLiveSurgeryManager
    epoch: LiveSurgeryEpoch
    state: SpominTargetState
    plan: SpominPlan
    prompt_token_ids: tuple[int, ...]
    receipt: dict[str, Any]
    retained_token_ids: tuple[int, ...] | None = None

    def apply(self, model, prompt_cache, *, request_quiescent, device_work_drained):
        return self.manager.apply(
            self,
            model,
            prompt_cache,
            request_quiescent=request_quiescent,
            device_work_drained=device_work_drained,
        )

    def close(self):
        self.manager.close(self.epoch)


class SpominLiveSurgeryManager:
    """Own request epochs and publish edits only at an explicit drained barrier."""

    def __init__(self, *, enabled=False, history_size=64):
        if history_size < 1:
            raise ValueError("live-surgery history size must be positive")
        self.enabled = bool(enabled)
        self._lock = Lock()
        self._next_generation = 0
        self._epochs = {}
        self._counts = Counter()
        self._recent = deque(maxlen=history_size)

    def _record(self, receipt):
        result = dict(receipt)
        with self._lock:
            self._record_unlocked(result)
        return result

    def _record_unlocked(self, receipt):
        status = str(receipt.get("status", "unknown"))
        reason = str(receipt.get("reason", status))
        self._counts[status] += 1
        self._counts[f"reason:{reason}"] += 1
        self._recent.append(dict(receipt))

    def decline(self, request_id, reason, *, detail=None):
        receipt = {"request_id": request_id, "status": "declined", "reason": reason}
        if detail is not None:
            receipt["detail"] = detail
        return self._record(receipt)

    def prepare(
        self,
        *,
        request_id: str,
        prompt_token_ids: Sequence[int],
        transcript,
        capacity_tokens: int,
        strategy: str,
        has_mtp_state: bool,
        has_recurrent_state: bool,
        cache_is_request_private: bool,
    ) -> LiveSurgeryTransaction | None:
        if not self.enabled:
            self.decline(request_id, "disabled")
            return None
        if transcript is None:
            self.decline(request_id, "transcript_unavailable")
            return None
        if not cache_is_request_private:
            self.decline(request_id, "cache_not_request_private")
            return None
        prompt = tuple(int(token) for token in prompt_token_ids)
        if tuple(transcript.token_ids) != prompt:
            self.decline(request_id, "transcript_prompt_mismatch")
            return None
        if has_mtp_state:
            self.decline(request_id, "mtp_state_active")
            return None
        if strategy == "lowest_importance":
            self.decline(request_id, "importance_scores_unavailable")
            return None
        state = SpominTargetState(
            revision=f"request:{request_id}:{transcript.fingerprint.digest}",
            target_tokens=len(prompt),
            transcript=transcript,
            visible_segment_ids=tuple(segment.segment_id for segment in transcript.segments),
            has_mtp_state=False,
            has_recurrent_state=has_recurrent_state,
        )
        layer = SpominLayer(SpominConfig(capacity_tokens=capacity_tokens, strategy=strategy))
        plan = layer.plan(state)
        if plan is None:
            self.decline(request_id, "below_pressure")
            return None
        if not plan.ready:
            self.decline(request_id, "target_unreachable")
            return None
        with self._lock:
            self._next_generation += 1
            epoch = LiveSurgeryEpoch(request_id, self._next_generation)
            self._epochs[request_id] = epoch.generation
        receipt = {
            "schema": "mlx2.spomin-live-surgery.v1",
            "request_id": request_id,
            "status": "prepared",
            "reason": "pressure",
            "source_tokens": len(prompt),
            "target_tokens": plan.projected_target_tokens,
            "strategy": strategy,
            "epoch": epoch.generation,
            "selected": False,
        }
        return LiveSurgeryTransaction(self, epoch, state, plan, prompt, receipt)

    def apply(
        self,
        transaction,
        model,
        prompt_cache: Sequence[object],
        *,
        request_quiescent: bool,
        device_work_drained: bool,
    ) -> Mapping[str, Any]:
        epoch = transaction.epoch
        with self._lock:
            current = self._epochs.get(epoch.request_id)
            if current != epoch.generation:
                return self._decline_transaction(transaction, "stale_epoch")
            if not request_quiescent:
                return self._decline_transaction(transaction, "request_not_quiescent")
            if not device_work_drained:
                return self._decline_transaction(transaction, "device_work_not_drained")
            del self._epochs[epoch.request_id]
            try:
                updated = SpominLayer(
                    SpominConfig(
                        capacity_tokens=max(transaction.state.target_tokens, 1),
                        strategy=transaction.plan.selection.strategy,
                    )
                ).apply(
                    transaction.state,
                    transaction.plan,
                    Qwen4SpominSurgeryBackend(model, prompt_cache),
                )
            except SpominCapabilityError as exc:
                return self._decline_transaction(transaction, "capability_refused", detail=str(exc))
            removed = set(transaction.plan.selection.segment_ids)
            transaction.retained_token_ids = tuple(
                token
                for segment in transaction.state.transcript.segments
                if segment.segment_id not in removed
                for token in segment.token_ids
            )
            transaction.state = updated
            receipt = dict(transaction.receipt, status="applied", reason="committed", selected=True)
            transaction.receipt = receipt
            self._record_unlocked(receipt)
            return receipt

    def _decline_transaction(self, transaction, reason, *, detail=None):
        receipt = dict(transaction.receipt, status="declined", reason=reason, selected=False)
        if detail is not None:
            receipt["detail"] = detail
        transaction.receipt = receipt
        self._record_unlocked(receipt)
        return receipt

    def close(self, epoch):
        with self._lock:
            if self._epochs.get(epoch.request_id) == epoch.generation:
                del self._epochs[epoch.request_id]

    def snapshot(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "active_epochs": len(self._epochs),
                "counts": dict(self._counts),
                "recent": list(self._recent),
            }
