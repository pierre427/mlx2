"""Fail-closed control plane for future approximate KV operations.

This module intentionally contains no tensor math.  Model adapters own such
operations (for example K8V4 or TurboQuant) and the runtime may call one only
after an evidence-bearing policy explicitly selects the exact operation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from ..contracts import Fidelity


class ApproximateStateError(RuntimeError):
    pass


class ApproximateStateAdapter(Protocol):
    revision: str

    def apply(self, state: Any) -> Any: ...


@dataclass(frozen=True)
class ApproximateKVPolicy:
    operation: str
    enabled: bool = False
    qualified: bool = False
    evidence: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.operation:
            raise ValueError("approximate KV operation name is required")
        if self.qualified and not self.evidence:
            raise ValueError("qualified approximate KV policy requires evidence")


class ApproximateKVController:
    """Select and publish revision-bound adapter operations with receipts."""

    def __init__(self, policy: ApproximateKVPolicy | None = None):
        self.policy = policy

    def prepare(
        self,
        *,
        request_id: str,
        state_revision: str,
        adapters: Mapping[str, Any],
        candidate: bool = False,
    ):
        """Select the adapter operation or fail closed.

        ``candidate`` is the qualification-mode escape: an enabled but not yet
        qualified policy may execute so evidence can be produced, and the
        receipt says so (``qualified: false``, ``reason:
        candidate_validation``).  It never relaxes the implementation or
        revision checks.
        """
        policy = self.policy
        base = {
            "schema": "mlx2.approximate-kv.v1",
            "request_id": str(request_id),
            "fidelity": Fidelity.APPROXIMATE.value,
            "source_revision": str(state_revision),
            "selected": False,
        }
        if policy is None or not policy.enabled:
            return None, dict(base, status="declined", reason="disabled")
        if not policy.qualified and not candidate:
            raise ApproximateStateError(
                f"approximate KV operation {policy.operation!r} is not qualified"
            )
        adapter = adapters.get(policy.operation)
        if adapter is None:
            raise ApproximateStateError(
                f"model adapter does not implement approximate KV operation {policy.operation!r}"
            )
        if str(getattr(adapter, "revision", "")) != str(state_revision):
            raise ApproximateStateError("approximate KV adapter revision mismatch")
        return adapter, dict(
            base,
            operation=policy.operation,
            evidence=list(policy.evidence),
            qualified=bool(policy.qualified),
            status="prepared",
            reason="qualified" if policy.qualified else "candidate_validation",
        )

    def apply(
        self,
        *,
        request_id: str,
        state_revision: str,
        state: Any,
        adapters: Mapping[str, Any],
        candidate: bool = False,
        stage: Callable[[Any], Any] | None = None,
    ):
        """Publish one operation over a privately staged state.

        ``stage`` replaces the default ``copy.deepcopy`` for live lane state
        that cannot or must not be deep-copied (leased COW branches, device
        arrays).  It must return a request-private object; returning the
        source object itself still fails closed.
        """
        adapter, receipt = self.prepare(
            request_id=request_id,
            state_revision=state_revision,
            adapters=adapters,
            candidate=candidate,
        )
        if adapter is None:
            return state, receipt
        live_revision = str(getattr(state, "revision", ""))
        if live_revision and live_revision != str(state_revision):
            raise ApproximateStateError("approximate KV source state revision mismatch")
        try:
            staged = copy.deepcopy(state) if stage is None else stage(state)
        except BaseException as error:
            raise ApproximateStateError(
                "approximate KV state cannot be privately staged"
            ) from error
        if staged is state:
            raise ApproximateStateError(
                "approximate KV state staging did not produce a private object"
            )
        updated = adapter.apply(staged)
        target_revision = str(getattr(updated, "revision", ""))
        if not target_revision or target_revision == str(state_revision):
            raise ApproximateStateError(
                "approximate KV operation did not advance the state revision"
            )
        return updated, dict(
            receipt,
            status="applied",
            reason="committed",
            selected=True,
            target_revision=target_revision,
        )
