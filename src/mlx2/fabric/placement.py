from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .link import LinkBudget, LinkReceipt


@dataclass(frozen=True)
class PeerView:
    nid: str
    state: Literal["alive", "suspect", "dead", "revoked", "refused"]
    models: tuple[str, ...] | set[str]
    queue_depth: int | None
    free_bytes: int | None
    device_class: str | None
    runtime_digest: str | None
    link: LinkReceipt | None
    hostname: str | None
    link_history: tuple[LinkReceipt, ...] = ()
    model_entries: tuple[dict[str, Any], ...] = ()


@dataclass
class Placement:
    request_id: str
    model: str
    kind: str
    origin_nid: str
    served_by: PeerView | None
    reason: str
    candidates: list[dict[str, Any]]
    selected: bool
    observed_used: bool = False
    outcome: str | None = None
    exactness: str = "whole_model_one_node"
    approximate: bool = False
    qualified: bool = False
    prefill_handoff: dict[str, Any] | None = None
    fallback: str | None = None
    failed_step: str | None = None

    def receipt(self) -> dict[str, Any]:
        served = None
        link = None
        if self.served_by is not None:
            served = {
                "nid": self.served_by.nid,
                "hostname": self.served_by.hostname,
                "device_class": self.served_by.device_class,
                "runtime_digest": self.served_by.runtime_digest,
            }
            if self.kind == "remote_resident" and self.served_by.link is not None:
                link = self.served_by.link.as_dict()
        receipt = {
            "schema": "mlx2.fabric.route.v1",
            "request_id": self.request_id,
            "model": self.model,
            "kind": self.kind,
            "origin_nid": self.origin_nid,
            "served_by": served,
            "reason": self.reason,
            "candidates": self.candidates,
            "link": link,
            "exactness": self.exactness,
            "approximate": self.approximate,
            "states": {
                "implemented": True,
                "qualified": self.qualified,
                "selected": self.selected,
                "observed_used": self.observed_used,
            },
        }
        if self.outcome is not None:
            receipt["outcome"] = self.outcome
        if self.prefill_handoff is not None:
            receipt["prefill_handoff"] = self.prefill_handoff
        if self.fallback is not None:
            receipt["fallback"] = self.fallback
        if self.failed_step is not None:
            receipt["failed_step"] = self.failed_step
        return receipt


def model_entry(peer: PeerView, model: str) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in peer.model_entries
            if item.get("id") == model and item.get("loaded") is True
        ),
        None,
    )


def prefill_candidate(
    *,
    requested: bool,
    model: str,
    local_model: dict[str, Any] | None,
    peers: Sequence[PeerView],
    plan_response: dict[str, Any] | None,
    link_budget: LinkBudget,
) -> tuple[dict[str, Any], PeerView | None, dict[str, Any] | None]:
    candidate: dict[str, Any] = {"option": "approximate_prefill_offload"}

    def refuse(reason: str, nid: str | None = None):
        candidate.update({"verdict": "refused", "reason": reason})
        if nid is not None:
            candidate["nid"] = nid
        return candidate, None, None

    if not requested:
        candidate.update({"verdict": "not_requested", "reason": "not_requested"})
        return candidate, None, None
    if local_model is None:
        return refuse("local_model_not_resident")
    local_cap = local_model.get("prefill_handoff")
    if not isinstance(local_cap, dict) or local_cap.get("import") is not True:
        return refuse("local_import_unavailable")
    if local_cap.get("ready") is False:
        return refuse("local_import_unavailable")
    required = (
        "operation",
        "operation_revision",
        "artifact_fingerprint",
        "layout_fingerprint",
        "tokenizer_digest",
    )
    for key in required:
        if not isinstance(local_cap.get(key), str) or not local_cap[key]:
            return refuse(f"local_{key}_missing")
    if (
        not isinstance(local_cap.get("runtime_digest"), str)
        or not local_cap["runtime_digest"]
    ):
        return refuse("local_runtime_digest_missing")
    if (
        not isinstance(local_cap.get("execution_digest"), str)
        or not local_cap["execution_digest"]
    ):
        return refuse("local_execution_digest_missing")

    exporters: list[tuple[PeerView, dict[str, Any]]] = []
    mismatch: str | None = None
    for peer in peers:
        entry = model_entry(peer, model)
        cap = entry.get("prefill_handoff") if entry is not None else None
        if (
            peer.state != "alive"
            or not isinstance(cap, dict)
            or cap.get("export") is not True
            or cap.get("ready") is False
        ):
            continue
        if not isinstance(cap.get("runtime_digest"), str) or not cap["runtime_digest"]:
            mismatch = mismatch or "peer_runtime_digest_missing"
            continue
        if not isinstance(cap.get("execution_digest"), str) or not cap["execution_digest"]:
            mismatch = mismatch or "peer_execution_digest_missing"
            continue
        different = next(
            (key for key in required if cap.get(key) != local_cap[key]), None
        )
        if different is not None:
            mismatch = mismatch or f"{different}_mismatch"
            continue
        exporters.append((peer, cap))
    if not exporters:
        return refuse(mismatch or "no_alive_export_peer")
    exporters.sort(key=lambda item: _peer_rank(item[0]))
    link_refusal: tuple[str, str] | None = None
    for peer, cap in exporters:
        admitted, reason = link_budget.admit(
            "prefill_offload", peer.link, history=peer.link_history
        )
        if not admitted:
            link_refusal = (peer.nid, reason)
            continue
        if plan_response is None:
            candidate.update(
                {"verdict": "refused", "reason": "plan_unavailable", "nid": peer.nid}
            )
            return candidate, peer, None
        plan = plan_response.get("plan")
        if not isinstance(plan, dict):
            return refuse("plan_unavailable", peer.nid)
        record = plan_response.get("record")
        if record is None:
            return refuse("no_qualification_record", peer.nid)
        if not isinstance(record, dict):
            return refuse("malformed_plan_response", peer.nid)
        if not isinstance(record.get("record_id"), str) or not record["record_id"]:
            return refuse("malformed_plan_response", peer.nid)
        maximum = record.get("max_prompt_tokens")
        if not isinstance(maximum, int) or maximum < 1:
            return refuse("malformed_plan_response", peer.nid)
        if plan.get("prompt_tokens", 0) > maximum:
            return refuse("context_exceeds_record", peer.nid)
        record_state = record.get("record_state")
        if record_state not in {"final", "provisional"}:
            return refuse("malformed_plan_response", peer.nid)
        candidate.update(
            {
                "nid": peer.nid,
                "verdict": "selected",
                "reason": reason,
                "record_id": record.get("record_id"),
                "record_state": record_state,
            }
        )
        return candidate, peer, record
    if link_refusal is not None:
        return refuse(f"link_budget_refused: {link_refusal[1]}", link_refusal[0])
    return refuse("no_qualification_record")


def _peer_rank(peer: PeerView) -> tuple[int, int, float, str]:
    return (
        peer.queue_depth if peer.queue_depth is not None else 2**63 - 1,
        -(peer.free_bytes if peer.free_bytes is not None else -1),
        peer.link.rtt_ms_median if peer.link is not None else float("inf"),
        peer.nid,
    )


def place(
    *,
    request_id: str,
    model: str,
    local_nid: str,
    local_models: set[str],
    peers: Sequence[PeerView],
    link_budget: LinkBudget,
    approximate_candidate: dict[str, Any] | None = None,
) -> Placement:
    candidates: list[dict[str, Any]] = []
    if approximate_candidate is not None:
        candidates.append(approximate_candidate)
    if model in local_models:
        local = PeerView(
            local_nid, "alive", local_models, 0, None, None, None, None, None
        )
        candidates.append(
            {
                "option": "local",
                "nid": local_nid,
                "verdict": "selected",
                "reason": "model resident locally",
            }
        )
        return Placement(
            request_id,
            model,
            "local",
            local_nid,
            local,
            "model resident locally",
            candidates,
            True,
        )

    resident: list[PeerView] = []
    for peer in peers:
        if model not in peer.models:
            continue
        if peer.state == "alive":
            resident.append(peer)
        else:
            candidates.append(
                {
                    "option": "remote_resident",
                    "nid": peer.nid,
                    "verdict": "rejected",
                    "reason": peer.state,
                }
            )
    resident.sort(key=_peer_rank)
    if resident:
        chosen = resident[0]
        _, link_reason = link_budget.admit(
            "route", chosen.link, history=chosen.link_history
        )
        for peer in resident:
            selected = peer is chosen
            candidates.append(
                {
                    "option": "remote_resident",
                    "nid": peer.nid,
                    "verdict": "selected" if selected else "not_selected",
                    "reason": link_reason
                    if selected
                    else "lower ranked resident candidate",
                }
            )
        return Placement(
            request_id,
            model,
            "remote_resident",
            local_nid,
            chosen,
            link_reason,
            candidates,
            True,
        )

    measured = sorted(
        (peer for peer in peers if peer.state == "alive" and peer.link_history),
        key=_peer_rank,
    )
    pipeline_reason = ""
    if measured:
        best = measured[0]
        _, pipeline_reason = link_budget.admit(
            "pipeline", best.link, history=best.link_history
        )
    candidates.extend(
        [
            {
                "option": "pipeline",
                "verdict": "not_implemented",
                "reason": (
                    f"not implemented; {pipeline_reason}"
                    if pipeline_reason
                    else "not implemented"
                ),
            },
            {
                "option": "streaming",
                "verdict": "not_implemented",
                "reason": "not implemented",
            },
            {
                "option": "approximate_prefill_offload",
                "verdict": "refused",
                "reason": "unqualified approximate state operation (AGENTS.md); not selectable",
            },
        ]
    )
    return Placement(
        request_id,
        model,
        "refused",
        local_nid,
        None,
        "no node has the model resident",
        candidates,
        False,
    )
