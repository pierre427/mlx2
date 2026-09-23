"""Serving-safe bridge for revision-bound Spomin Qwen4 KV surgery."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from threading import Lock
from typing import Any

from .spomin_layer import (
    SpominBackendError,
    SpominCapabilityError,
    SpominConfig,
    SpominLayer,
    SpominPlan,
    SpominPlanError,
    SpominRevisionError,
    SpominTargetState,
)
from .cache_planes import TranscriptLedgerPlane, TranscriptLedgerSegment


_TERMINAL_RECEIPT_REASONS = frozenset(
    {
        "committed",
        "stale_epoch",
        "capability_refused",
        "revision_changed",
        "plan_invalid",
        "backend_refused",
        "closed_without_apply",
        "backend_unavailable",
    }
)


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
        receipt = self.manager.close(self.epoch)
        if receipt is not None:
            self.receipt = receipt
        return receipt


@dataclass(frozen=True)
class ServingSpominPolicy:
    """Default-off policy for the isolated ordinary-decode serving seam.

    Fixed token chunks are deliberately an approximate policy: they provide
    stable revision-bound surgery units without pretending that tokenizer
    chunks are semantic conversation turns.  A future semantic segmenter can
    be supplied behind the same boundary and independently qualified.
    """

    enabled: bool = False
    capacity_tokens: int = 0
    segment_tokens: int = 1024
    strategy: str = "oldest_contiguous"
    # Leading segments carry BOS/system attention sinks; removing them first is
    # the one edit "oldest" would otherwise always choose.
    protect_prefix_segments: int = 1

    def __post_init__(self):
        if self.enabled and self.capacity_tokens <= 0:
            raise ValueError("enabled serving Spomin requires capacity_tokens")
        if self.segment_tokens <= 0:
            raise ValueError("serving Spomin segment_tokens must be positive")
        if isinstance(self.protect_prefix_segments, bool) or self.protect_prefix_segments < 0:
            raise ValueError("serving Spomin protect_prefix_segments must be non-negative")

    @classmethod
    def from_value(cls, value):
        if value in (None, False):
            return cls()
        if value is True or not isinstance(value, Mapping):
            raise ValueError("spomin_live_surgery must be a policy mapping")
        unknown = set(value) - {
            "enabled", "capacity_tokens", "segment_tokens", "strategy",
            "protect_prefix_segments",
        }
        if unknown:
            raise ValueError(
                "unknown serving Spomin policy keys: " + ", ".join(sorted(unknown))
            )
        return cls(**value)

    def transcript(self, token_ids, *, tokenizer_identity, revision):
        tokens = tuple(int(token) for token in token_ids)
        segments = []
        for start in range(0, len(tokens), self.segment_tokens):
            stop = min(start + self.segment_tokens, len(tokens))
            segments.append(
                TranscriptLedgerSegment(
                    f"prefill:{start}:{stop}", start, stop, tokens[start:stop]
                )
            )
        return TranscriptLedgerPlane(
            tokenizer_identity=str(tokenizer_identity),
            tokenizer_version="serving-v1",
            revision=str(revision),
            segments=tuple(segments),
            compaction_strategy=self.strategy,
        )


class SpominLiveSurgeryManager:
    """Own request epochs and publish edits only at an explicit drained barrier."""

    def __init__(self, *, enabled=False, history_size=64, backend_factory=None):
        if history_size < 1:
            raise ValueError("live-surgery history size must be positive")
        self.enabled = bool(enabled)
        # ``backend_factory(model, prompt_cache)`` is adapter-owned; it returns
        # None when the adapter has no surgical backend for this state.
        self._backend_factory = backend_factory
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

    def _record_unlocked(self, receipt, *, history=True):
        status = str(receipt.get("status", "unknown"))
        reason = str(receipt.get("reason", status))
        self._counts[status] += 1
        self._counts[f"reason:{reason}"] += 1
        if history:
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
        protected_segment_ids: Sequence[str] = (),
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
        if has_recurrent_state:
            self.decline(request_id, "recurrent_state_unrepairable")
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
        plan = layer.plan(state, protected_segment_ids=tuple(protected_segment_ids))
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
                "source_transcript_digest": transcript.fingerprint.digest,
            }
            # Count the preparation so the exported "prepared" operation is
            # observable. The recent history keeps one terminal receipt per
            # transaction; serving reads its tail as the decline receipt.
            self._record_unlocked(receipt, history=False)
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
                existing = transaction.receipt
                if (
                    existing.get("epoch") == epoch.generation
                    and existing.get("reason") in _TERMINAL_RECEIPT_REASONS
                ):
                    return dict(existing)
                return self._decline_transaction(transaction, "stale_epoch")
            if not request_quiescent:
                return self._decline_transaction(transaction, "request_not_quiescent")
            if not device_work_drained:
                return self._decline_transaction(transaction, "device_work_not_drained")
            del self._epochs[epoch.request_id]
            if self._backend_factory is None:
                from .spomin_qwen4_surgery import Qwen4SpominSurgeryBackend

                backend = Qwen4SpominSurgeryBackend(model, prompt_cache)
            else:
                backend = self._backend_factory(model, prompt_cache)
            if backend is None:
                return self._decline_transaction(transaction, "backend_unavailable")
            try:
                updated = SpominLayer(
                    SpominConfig(
                        capacity_tokens=max(transaction.state.target_tokens, 1),
                        strategy=transaction.plan.selection.strategy,
                    )
                ).apply(
                    transaction.state,
                    transaction.plan,
                    backend,
                )
            except SpominCapabilityError as exc:
                return self._decline_transaction(transaction, "capability_refused", detail=str(exc))
            except SpominRevisionError as exc:
                return self._decline_transaction(transaction, "revision_changed", detail=str(exc))
            except SpominPlanError as exc:
                return self._decline_transaction(transaction, "plan_invalid", detail=str(exc))
            except SpominBackendError as exc:
                return self._decline_transaction(transaction, "backend_refused", detail=str(exc))
            removed = set(transaction.plan.selection.segment_ids)
            transaction.retained_token_ids = tuple(
                token
                for segment in transaction.state.transcript.segments
                if segment.segment_id not in removed
                for token in segment.token_ids
            )
            transaction.state = updated
            receipt = dict(transaction.receipt, status="applied", reason="committed", selected=True)
            receipt["retained_tokens"] = len(transaction.retained_token_ids)
            receipt["retained_token_digest"] = hashlib.sha256(
                b"".join(
                    int(token).to_bytes(8, "big", signed=False)
                    for token in transaction.retained_token_ids
                )
            ).hexdigest()
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
        receipt = None
        with self._lock:
            if self._epochs.get(epoch.request_id) == epoch.generation:
                del self._epochs[epoch.request_id]
                receipt = {
                    "schema": "mlx2.spomin-live-surgery.v1",
                    "request_id": epoch.request_id,
                    "status": "declined",
                    "reason": "closed_without_apply",
                    "epoch": epoch.generation,
                    "selected": False,
                }
                self._record_unlocked(receipt)
        return receipt

    def snapshot(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "active_epochs": len(self._epochs),
                "counts": dict(self._counts),
                "recent": list(self._recent),
            }
