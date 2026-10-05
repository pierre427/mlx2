"""Exact-identity, exact-shape live-step prices for the default-off pack chooser.

The caller supplies the live identity and context signature. This module never
probes hardware or turns an unqualified benchmark into a selected route.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .paged_pack_scheduler import PrefillOption, ReservedRows


SCHEMA = "mlx2.paged-pack-price.v1"
PACK_STEP_UNIT = "live_scheduler_pack_step"
IDENTITY_FIELDS = ("host", "hardware", "artifact_sha256", "source_commit",
                   "source_tree_sha256", "mlx_wheel_version", "mlx_wheel_sha256",
                   "kernel_sha256")


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _identity(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(IDENTITY_FIELDS):
        raise ValueError("complete price identity is required")
    if any(not isinstance(value[k], str) or not value[k] for k in IDENTITY_FIELDS):
        raise ValueError("empty price identity field")
    if any(not _sha(value[k]) for k in ("artifact_sha256", "source_tree_sha256",
                                    "mlx_wheel_sha256", "kernel_sha256")):
        raise ValueError("price identity requires lowercase SHA-256 digests")
    if len(value["source_commit"]) != 40 or any(c not in "0123456789abcdef" for c in value["source_commit"]):
        raise ValueError("price identity requires a full source commit")
    return dict(value)


def evidence_sha256(data: dict) -> str:
    """Bind the complete raw receipt body, excluding only its digest field."""
    body = {key: value for key, value in data.items()
            if key != "complete_request_evidence_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _complete_request_proofs(case: dict, timings: list, route: str,
                             context_tokens: tuple[int, ...]) -> None:
    ordinary = case.get("ordinary_request_ms")
    native = case.get("complete_native_request_ms")
    ordinary_steps = case.get("ordinary_pack_step_ms")
    proofs = case.get("request_proofs")
    if (not isinstance(ordinary, list) or len(ordinary) != len(timings) or
            any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0
                for v in ordinary) or
            not isinstance(native, list) or len(native) != len(timings) or
            not isinstance(ordinary_steps, list) or len(ordinary_steps) != len(timings) or
            any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0
                for values in (native, ordinary_steps) for value in values) or
            any(step > total for step, total in zip(timings, native)) or
            any(step > total for step, total in zip(ordinary_steps, ordinary)) or
            not isinstance(proofs, list) or len(proofs) != len(timings)):
        raise ValueError("separate paired pack-step and complete-request timings required")
    expected_step = {"scope": PACK_STEP_UNIT,
                     "reserved": case["reserved"],
                     "prefill_rows": case["prefill_rows"],
                     "context_tokens": list(context_tokens),
                     "active_lane_count": len(case["reserved"]) + bool(case["prefill_rows"]),
                     "generator_step": True, "sampled_response": True,
                     "synchronized": True}
    reads = terminals = 0
    for index, pair in enumerate(proofs):
        if not isinstance(pair, dict):
            raise ValueError("paired request proof missing")
        plain, paged = pair.get("ordinary"), pair.get("paged")
        if not isinstance(plain, dict) or not isinstance(paged, dict):
            raise ValueError("paired request proof missing")
        required = ("request_inserted", "admitted", "cache_transaction_published",
                    "response_emitted", "sampled_output", "synchronized",
                    "request_removed", "request_state_released")
        if any(plain.get(key) is not True or paged.get(key) is not True
               for key in required):
            raise ValueError("complete-request lifecycle proof missing")
        if (plain.get("scheduler_step") != expected_step or
                paged.get("scheduler_step") != expected_step or
                plain.get("scheduler_step_ms") != ordinary_steps[index] or
                paged.get("scheduler_step_ms") != timings[index]):
            raise ValueError("observed scheduler step shape, context or timing differs")
        plain_route, paged_route = plain.get("route_receipt"), paged.get("route_receipt")
        if (not isinstance(plain_route, dict) or plain_route.get("route") != "ordinary" or
                not isinstance(paged_route, dict) or paged_route.get("route") != route or
                paged_route.get("selected") is not True or
                paged_route.get("observed_used") is not True or
                type(plain.get("ordinary_model_forward_calls")) is not int or
                plain["ordinary_model_forward_calls"] < 1 or
                type(paged.get("ordinary_model_forward_calls")) is not int or
                paged["ordinary_model_forward_calls"] != 0 or
                type(paged.get("model_layers")) is not int or
                paged["model_layers"] < 1 or
                type(paged.get("paged_read_calls")) is not int or
                paged["paged_read_calls"] < paged["model_layers"] or
                type(paged.get("terminal_successes")) is not int or
                paged["terminal_successes"] != paged["paged_read_calls"] or
                type(paged.get("pending_native_epochs")) is not int or
                paged["pending_native_epochs"] != 0 or
                type(paged.get("retained_pages")) is not int or
                paged["retained_pages"] != 0 or
                type(plain.get("output_token_id")) is not int or
                plain["output_token_id"] < 0 or
                plain["output_token_id"] != paged.get("output_token_id")):
            raise ValueError("observed native request or paired ordinary proof missing")
        reads += paged["paged_read_calls"]
        terminals += paged["terminal_successes"]
    if case.get("kernel_engagement") != {"paged_read_calls": reads,
                                          "terminal_successes": terminals}:
        raise ValueError("native engagement summary differs from raw requests")


def shape_key(reserved: tuple[ReservedRows, ...], prefill: tuple[int, PrefillOption] | None) -> str:
    """Lane IDs do not affect cost; their ordered phase/row shape does."""
    return json.dumps({"reserved": [[r.phase, r.rows] for r in reserved],
                       "prefill_rows": prefill[1].rows if prefill else 0},
                      sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class MeasuredPackPrice:
    profile_id: str
    context_tokens: tuple[int, ...]
    costs_ms: Mapping[str, float]
    measurement_scope: str
    complete_request_evidence_sha256: str
    identity: Mapping[str, str]
    _validation_token: object

    @property
    def validated(self) -> bool:
        return (self._validation_token is _LOADER_TOKEN and
                self.measurement_scope == "complete_request" and
                _sha(self.complete_request_evidence_sha256))

    def estimate_ms(self, reserved: tuple[ReservedRows, ...],
                    prefill: tuple[int, PrefillOption] | None) -> float:
        if not reserved and prefill is None:
            return 0.0
        try:
            return self.costs_ms[shape_key(reserved, prefill)]
        except KeyError as exc:
            raise ValueError("unmeasured paged pack shape") from exc


_LOADER_TOKEN = object()


@dataclass(frozen=True)
class ResearchCalibratedPrice:
    """Explicit B1 bootstrap price; never a measured serving pack profile."""

    profile_id: str
    context_tokens: tuple[int, ...]
    identity: Mapping[str, str]
    evidence_sha256: str
    cold_bound_ms: float
    q1_bound_ms: float
    _validation_token: object

    @property
    def validated(self) -> bool:
        return self._validation_token is _RESEARCH_TOKEN

    def estimate_ms(self, reserved: tuple[ReservedRows, ...],
                    prefill: tuple[int, PrefillOption] | None) -> float:
        if not reserved and prefill is None:
            return 0.0
        if not reserved and prefill is not None and prefill[1].rows == 63:
            return self.cold_bound_ms
        if (len(reserved) == 1 and reserved[0].phase == "decode" and
                reserved[0].rows == 1 and prefill is None):
            return self.q1_bound_ms
        raise ValueError("research calibration admits only cold 63-token B1 and q1")


_RESEARCH_TOKEN = object()


_RESEARCH_WARM_TOKEN = object()


@dataclass(frozen=True)
class ResearchWarmCalibratedPrice:
    """Exact APCv2-63 plus suffix-1 research admission bound for context 64."""

    profile_id: str
    context_tokens: tuple[int, ...]
    identity: Mapping[str, str]
    evidence_sha256: str
    warm_bound_ms: float
    q1_bound_ms: float
    _validation_token: object

    @property
    def validated(self) -> bool:
        return self._validation_token is _RESEARCH_WARM_TOKEN

    def estimate_ms(self, reserved: tuple[ReservedRows, ...],
                    prefill: tuple[int, PrefillOption] | None) -> float:
        if not reserved and prefill is None:
            return 0.0
        if not reserved and prefill is not None and prefill[1].rows == 1:
            return self.warm_bound_ms
        if (len(reserved) == 1 and reserved[0].phase == "decode" and
                reserved[0].rows == 1 and prefill is None):
            return self.q1_bound_ms
        raise ValueError("warm research calibration admits only suffix-1 and q1")


_GRAPH_B2_TOKEN = object()
_GRAPH_B2_SCOPE = "research_live_scheduler_b2_step"
_GRAPH_B2_ROUTE = "native_qwen3_paged_graph_research"
_GRAPH_B2_STEP = {"scope": PACK_STEP_UNIT,
                  "reserved": [["decode", 1], ["decode", 1]],
                  "prefill_rows": 0, "context_tokens": [63, 65],
                  "active_lane_count": 2, "generator_step": True,
                  "sampled_response": True, "synchronized": True}


@dataclass(frozen=True)
class ResearchGraphB2Price:
    """Validated exact B2 research price; never serving route selection."""

    profile_id: str
    context_tokens: tuple[int, ...]
    identity: Mapping[str, str]
    evidence_sha256: str
    upper_bound_ms: float
    _validation_token: object

    @property
    def validated(self) -> bool:
        return self._validation_token is _GRAPH_B2_TOKEN

    def estimate_ms(self, reserved: tuple[ReservedRows, ...],
                    prefill: tuple[int, PrefillOption] | None) -> float:
        if not reserved and prefill is None:
            return 0.0
        if (type(reserved) is tuple and len(reserved) == 2 and prefill is None and
                all(type(row) is ReservedRows and row.phase == "decode" and
                    row.rows == 1 for row in reserved)):
            return self.upper_bound_ms
        raise ValueError("research graph price admits only exact two-lane q=1 decode")


def load_research_graph_b2_price(path: str | Path, *, live_identity: dict[str, str],
                                  context_tokens: tuple[int, ...]) -> ResearchGraphB2Price:
    """Validate physical B2 research steps without changing serving price gates."""
    expected = _identity(live_identity)
    if context_tokens != (63, 65):
        raise ValueError("research graph price requires ordered contexts (63, 65)")
    data = json.loads(Path(path).read_text())
    if (data.get("schema") != SCHEMA or data.get("status") != "screened" or
            data.get("measurement_scope") != _GRAPH_B2_SCOPE or
            data.get("gpu_executed") is not True or
            data.get("context_tokens") != [63, 65] or
            data.get("scheduler_step") != _GRAPH_B2_STEP or
            data.get("qualified") is not False or
            data.get("selected") is not False or
            data.get("serving_selected") is not False or
            data.get("research_executed") is not True or
            data.get("research_price_candidate") is not True or
            data.get("price_usable") is not False or
            _identity(data.get("identity")) != expected):
        raise ValueError("exact research graph identity or scope differs")
    owner = data.get("gpuq_owner")
    if (not isinstance(owner, dict) or
            any(not isinstance(owner.get(key), str) or not owner[key]
                for key in ("session", "lease_id")) or
            type(owner.get("pid")) is not int or owner["pid"] < 1):
        raise ValueError("owned research graph GPU lease missing")
    digest = data.get("research_request_evidence_sha256")
    if not _sha(digest):
        raise ValueError("research graph evidence digest missing")
    body = {key: value for key, value in data.items()
            if key != "research_request_evidence_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("research graph evidence digest differs")
    profile = data.get("profile_id")
    cases = data.get("paired_cases")
    if not isinstance(profile, str) or not profile or not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("three paired B2 samples and a profile are required")
    costs = []
    for index, pair in enumerate(cases):
        order = ["ordinary", "paged"] if index != 1 else ["paged", "ordinary"]
        if type(pair) is not dict or pair.get("order") != order:
            raise ValueError("paired B2 arm order differs")
        proofs = {}
        for arm in ("ordinary", "paged"):
            item = pair.get(arm)
            if type(item) is not dict:
                raise ValueError("paired B2 request missing")
            total, proof = item.get("complete_request_ms"), item.get("proof")
            if (type(total) not in (int, float) or not math.isfinite(total) or total <= 0 or
                    type(proof) is not dict or proof.get("arm") != arm):
                raise ValueError("paired complete-request timing missing")
            step = proof.get("ready_pack_step_ms")
            if (type(step) not in (int, float) or not math.isfinite(step) or
                    not 0 < step <= total or proof.get("scheduler_step") != _GRAPH_B2_STEP or
                    proof.get("ordered_context_tokens") != [63, 65] or
                    proof.get("model_layers") != 28 or
                    proof.get("output_token_ids") is None or
                    any(proof.get(key) is not True for key in (
                        "request_inserted", "admitted", "cache_transaction_published",
                        "response_emitted", "sampled_output", "synchronized",
                        "request_removed", "request_state_released"))):
                raise ValueError("paired live B2 scheduler or lifecycle proof missing")
            proofs[arm] = proof
        ordinary, paged = proofs["ordinary"], proofs["paged"]
        route = paged.get("route_receipt")
        responses = paged.get("response_receipts")
        outputs = paged.get("output_token_ids")
        if (type(outputs) is not list or len(outputs) != 2 or
                any(type(row) is not list or len(row) != 2 or
                    any(type(token) is not int or token < 0 for token in row)
                    for row in outputs) or
                outputs != ordinary.get("output_token_ids") or
                type(ordinary.get("ordinary_model_forward_calls")) is not int or
                ordinary["ordinary_model_forward_calls"] < 1 or
                ordinary.get("route_receipt", {}).get("route") != "ordinary" or
                type(route) is not dict or route.get("route") != _GRAPH_B2_ROUTE or
                route.get("selected") is not False or
                route.get("observed_used") is not False or
                route.get("research_executed") is not True or
                route.get("serving_selected") is not False or
                paged.get("ordinary_model_forward_calls") != 0 or
                paged.get("response_execution_widths") != [[1, 2], [1, 2]] or
                type(responses) is not list or len(responses) != 2 or
                any(type(rows) is not list or len(rows) != 2 or
                    rows[1].get("packed_lanes") != 2 or
                    rows[1].get("native_span_counts") != [2] * 28 or
                    rows[1].get("native_read_delta") != 28 or
                    rows[1].get("terminal_success_delta") != 28
                    for rows in responses) or
                paged.get("paged_read_calls") != 84 or
                paged.get("prefill_read_calls") != 56 or
                paged.get("decode_read_calls") != 28 or
                paged.get("terminal_successes") != 84 or
                paged.get("pending_native_epochs") != 0 or
                paged.get("retained_pages") != 0):
            raise ValueError("paired physical native graph B2 proof missing")
        costs.append(float(paged["ready_pack_step_ms"]))
    return ResearchGraphB2Price(profile, (63, 65), MappingProxyType(expected),
                                digest, max(costs), _GRAPH_B2_TOKEN)


def load_research_calibration(path: str | Path, *, live_identity: dict[str, str],
                              context_tokens: tuple[int, ...]) -> ResearchCalibratedPrice:
    """Validate a raw, paired live request calibration for one explicit route.

    The receipt is evidence for a bounded bootstrap decision, not a serving
    qualification or a general pack price. Exact source identity is mandatory.
    """
    expected = _identity(live_identity)
    if context_tokens != (63,):
        raise ValueError("research calibration requires exact context (63,)")
    data = json.loads(Path(path).read_text())
    if (data.get("schema") != SCHEMA or data.get("status") != "calibrated" or
            data.get("measurement_scope") != "research_live_request_calibration" or
            data.get("gpu_executed") is not True or
            data.get("measured_route") != "native_qwen3_paged" or
            data.get("context_tokens") != [63] or
            data.get("request_shape") != {"prompt_tokens": 63, "output_tokens": 2,
                                          "decode_rows_after_prefill": 1} or
            data.get("qualified") is not False or data.get("selected") is not False or
            data.get("serving_selected") is not False or
            data.get("research_executed") is not True or
            data.get("price_usable") is not False or
            _identity(data.get("identity")) != expected):
        raise ValueError("exact research calibration identity or scope differs")
    owner = data.get("gpuq_owner")
    if (not isinstance(owner, dict) or
            any(not isinstance(owner.get(key), str) or not owner[key]
                for key in ("session", "lease_id")) or
            type(owner.get("pid")) is not int or owner["pid"] < 1):
        raise ValueError("owned GPU calibration evidence missing")
    digest = data.get("research_request_evidence_sha256")
    if not _sha(digest):
        raise ValueError("research evidence digest missing")
    body = {key: value for key, value in data.items()
            if key != "research_request_evidence_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("research evidence digest differs")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("one exact B1 calibration case required")
    case = cases[0]
    if not isinstance(case, dict):
        raise ValueError("malformed research case")
    fields = ("cold_native_request_ms", "cold_ordinary_request_ms",
              "native_q1_step_ms", "ordinary_q1_step_ms")
    if any(not isinstance(case.get(key), list) or len(case[key]) < 3 or
           any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in case[key]) for key in fields):
        raise ValueError("paired raw B1 timings missing")
    if len({len(case[key]) for key in fields}) != 1:
        raise ValueError("paired B1 timing counts differ")
    proofs = case.get("request_proofs")
    if not isinstance(proofs, list) or len(proofs) != len(case[fields[0]]):
        raise ValueError("paired live request proofs missing")
    if case.get("kernel_engagement") != {"paged_read_calls": 56 * len(proofs),
                                          "terminal_successes": 56 * len(proofs)}:
        raise ValueError("native engagement aggregate differs")
    for index, pair in enumerate(proofs):
        if not isinstance(pair, dict):
            raise ValueError("paired request proof missing")
        plain, native = pair.get("ordinary"), pair.get("paged")
        if not isinstance(plain, dict) or not isinstance(native, dict):
            raise ValueError("paired request proof missing")
        common = ("request_inserted", "admitted", "cache_transaction_published",
                  "response_emitted", "sampled_output", "synchronized",
                  "request_removed", "request_state_released")
        if any(plain.get(key) is not True or native.get(key) is not True for key in common):
            raise ValueError("live lifecycle proof missing")
        route = native.get("route_receipt")
        first = native.get("first_response_receipt")
        second = native.get("second_response_receipt")
        if (native.get("output_token_ids") != plain.get("output_token_ids") or
                not isinstance(route, dict) or route.get("research_executed") is not True or
                route.get("selected") is not False or route.get("observed_used") is not False or
                not isinstance(first, dict) or not isinstance(second, dict) or
                first.get("research_executed") is not True or
                second.get("research_executed") is not True or
                first.get("native_read_calls") != 28 or
                first.get("terminal_successes") != 28 or
                second.get("native_read_calls") != 56 or
                second.get("terminal_successes") != 56 or
                native.get("ordinary_model_forward_calls") != 0 or
                type(native.get("paged_read_calls")) is not int or
                native["paged_read_calls"] < 56 or
                native.get("terminal_successes") != native["paged_read_calls"] or
                native.get("pending_native_epochs") != 0 or
                native.get("retained_pages") != 0 or
                native.get("prefill_read_calls") != 28 or
                native.get("decode_read_calls") != 28 or
                native.get("q1_step_ms") != case["native_q1_step_ms"][index] or
                plain.get("q1_step_ms") != case["ordinary_q1_step_ms"][index] or
                type(plain.get("ordinary_model_forward_calls")) is not int or
                plain["ordinary_model_forward_calls"] < 1):
            raise ValueError("native read or paired ordinary proof missing")
    profile = data.get("profile_id")
    if not isinstance(profile, str) or not profile:
        raise ValueError("calibration profile missing")
    return ResearchCalibratedPrice(profile, (63,), MappingProxyType(expected), digest,
                                   float(max(case["cold_native_request_ms"])),
                                   float(max(case["native_q1_step_ms"])), _RESEARCH_TOKEN)


def load_research_warm_calibration(path: str | Path, *, live_identity: dict[str, str],
                                   context_tokens: tuple[int, ...]) -> ResearchWarmCalibratedPrice:
    """Validate three paired, unselected APCv2 warm requests before opt-in use."""
    expected = _identity(live_identity)
    if context_tokens != (64,):
        raise ValueError("warm research calibration requires context (64,)")
    data = json.loads(Path(path).read_text())
    shape = {"prompt_tokens": 64, "cached_tokens": 63, "suffix_rows": 1,
             "output_tokens": 2, "decode_rows_after_prefill": 1}
    if (data.get("schema") != SCHEMA or data.get("status") != "calibrated" or
            data.get("measurement_scope") != "research_live_warm_request_calibration" or
            data.get("gpu_executed") is not True or
            data.get("measured_route") != "native_qwen3_paged" or
            data.get("context_tokens") != [64] or data.get("request_shape") != shape or
            data.get("qualified") is not False or data.get("selected") is not False or
            data.get("serving_selected") is not False or
            data.get("research_executed") is not True or
            data.get("price_usable") is not False or
            type(data.get("peak_resident_bytes")) is not int or
            not 0 < data["peak_resident_bytes"] <= 24 * (1 << 30) or
            _identity(data.get("identity")) != expected):
        raise ValueError("exact warm research calibration identity or scope differs")
    owner = data.get("gpuq_owner")
    if (not isinstance(owner, dict) or
            any(not isinstance(owner.get(key), str) or not owner[key]
                for key in ("session", "lease_id")) or
            type(owner.get("pid")) is not int or owner["pid"] < 1):
        raise ValueError("owned warm calibration evidence missing")
    digest = data.get("research_warm_evidence_sha256")
    body = {key: value for key, value in data.items()
            if key != "research_warm_evidence_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    if not _sha(digest) or hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("warm raw evidence digest differs")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise ValueError("one warm research case required")
    case = cases[0]
    fields = ("warm_native_request_ms", "warm_ordinary_request_ms",
              "native_q1_step_ms", "ordinary_q1_step_ms")
    if any(not isinstance(case.get(key), list) or len(case[key]) != 3 or
           any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in case[key]) for key in fields):
        raise ValueError("three paired warm complete-request timings required")
    if any(step > total for step, total in zip(case["native_q1_step_ms"],
                                               case["warm_native_request_ms"])):
        raise ValueError("warm q1 step exceeds complete request")
    proofs = case.get("request_proofs")
    if (not isinstance(proofs, list) or len(proofs) != 3 or
            case.get("kernel_engagement") != {"paged_read_calls": 168,
                                              "terminal_successes": 168}):
        raise ValueError("three paired warm proofs and physical reads required")
    common = ("request_inserted", "admitted", "cache_transaction_published",
              "response_emitted", "sampled_output", "synchronized",
              "request_removed", "request_state_released")
    for index, pair in enumerate(proofs):
        if not isinstance(pair, dict):
            raise ValueError("warm pair proof missing")
        ordinary, native = pair.get("ordinary"), pair.get("paged")
        if not isinstance(ordinary, dict) or not isinstance(native, dict):
            raise ValueError("warm arm proof missing")
        if any(ordinary.get(key) is not True or native.get(key) is not True
               for key in common):
            raise ValueError("warm complete-request lifecycle missing")
        route = native.get("route_receipt")
        first, second = native.get("first_response_receipt"), native.get("second_response_receipt")
        tokens = native.get("output_token_ids")
        if (type(tokens) is not list or len(tokens) != 2 or
                any(type(token) is not int or token < 0 for token in tokens) or
                tokens != ordinary.get("output_token_ids") or
                native.get("apcv2_cached_tokens") != 63 or
                ordinary.get("apcv2_cached_tokens") != 63 or
                native.get("apcv2_stores_delta") != 0 or
                ordinary.get("apcv2_stores_delta") != 0 or
                not isinstance(route, dict) or route.get("route") != "native_qwen3_paged" or
                route.get("research_executed") is not True or
                route.get("selected") is not False or route.get("observed_used") is not False or
                route.get("apcv2_restored_tokens") != 63 or
                not isinstance(first, dict) or not isinstance(second, dict) or
                first.get("research_executed") is not True or
                second.get("research_executed") is not True or
                first.get("selected") is not False or
                second.get("selected") is not False or
                first.get("observed_used") is not False or
                second.get("observed_used") is not False or
                first.get("native_read_calls") != 28 or first.get("terminal_successes") != 28 or
                second.get("native_read_calls") != 56 or second.get("terminal_successes") != 56 or
                native.get("ordinary_model_forward_calls") != 0 or
                native.get("paged_read_calls") != 56 or
                native.get("terminal_successes") != 56 or
                native.get("pending_native_epochs") != 0 or
                native.get("retained_pages") != 0 or
                native.get("owner_fully_retired") is not True or
                native.get("writer_poisoned") is not False or
                native.get("q1_step_ms") != case["native_q1_step_ms"][index] or
                ordinary.get("q1_step_ms") != case["ordinary_q1_step_ms"][index] or
                type(ordinary.get("ordinary_model_forward_calls")) is not int or
                ordinary["ordinary_model_forward_calls"] < 1):
            raise ValueError("warm response, state, or physical terminal proof missing")
    profile = data.get("profile_id")
    if not isinstance(profile, str) or not profile:
        raise ValueError("warm calibration profile missing")
    return ResearchWarmCalibratedPrice(profile, (64,), MappingProxyType(expected), digest,
                                       float(max(case["warm_native_request_ms"])),
                                       float(max(case["native_q1_step_ms"])),
                                       _RESEARCH_WARM_TOKEN)


def load_price(path: str | Path, *, live_identity: dict[str, str],
               context_tokens: tuple[int, ...]) -> MeasuredPackPrice:
    """Refuse stale, incomplete, or semantically mismatched price receipts."""
    expected = _identity(live_identity)
    if (type(context_tokens) is not tuple or not context_tokens or
            any(type(n) is not int or n < 0 for n in context_tokens)):
        raise ValueError("exact live context lengths are required")
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA or data.get("status") != "measured":
        raise ValueError("measured price receipt required")
    if data.get("measurement_scope") != "complete_request":
        raise ValueError("serving price requires complete-request timing, not a kernel or forward probe")
    if data.get("price_unit") != PACK_STEP_UNIT:
        raise ValueError("live scheduler pack-step price unit required")
    if not _sha(data.get("complete_request_evidence_sha256")):
        raise ValueError("complete-request evidence digest missing")
    if data.get("gpu_executed") is not True:
        raise ValueError("complete-request GPU execution evidence missing")
    owner = data.get("gpuq_owner")
    if (not isinstance(owner, dict) or not isinstance(owner.get("session"), str) or
            not owner["session"] or not isinstance(owner.get("lease_id"), str) or
            not owner["lease_id"] or type(owner.get("pid")) is not int or
            owner["pid"] < 1):
        raise ValueError("owned GPU lease evidence missing")
    route = data.get("measured_route")
    if not isinstance(route, str) or not route:
        raise ValueError("measured native route identity missing")
    if _identity(data.get("identity")) != expected:
        raise ValueError("price identity differs from live identity")
    if data.get("context_tokens") != list(context_tokens):
        raise ValueError("price context lengths differ from live lengths")
    if not isinstance(data.get("profile_id"), str) or not data["profile_id"]:
        raise ValueError("price profile ID missing")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("price cases missing")
    costs: dict[str, float] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("reserved"), list):
            raise ValueError("malformed price case")
        try:
            rows = tuple(ReservedRows(i, phase, count, 0, 1.0)
                         for i, (phase, count) in enumerate(case["reserved"]))
            prefill_rows = case["prefill_rows"]
            prefill = (len(rows), PrefillOption(prefill_rows, 0)) if prefill_rows else None
            key = shape_key(rows, prefill)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("malformed price shape") from exc
        timings = case.get("pack_step_ms")
        engagement = case.get("kernel_engagement")
        if not rows and not prefill_rows:
            raise ValueError("zero-work scheduler baseline is implicit")
        if "forward_ms" in case:
            raise ValueError("complete-forward timing cannot price a live scheduler step")
        if (not isinstance(timings, list) or len(timings) < 3 or
                any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in timings)):
            raise ValueError("at least three positive raw scheduler-step timings required")
        if (not isinstance(engagement, dict) or
                type(engagement.get("paged_read_calls")) is not int or
                engagement["paged_read_calls"] < 1 or
                type(engagement.get("terminal_successes")) is not int or
                engagement["terminal_successes"] < 1):
            raise ValueError("paged kernel engagement evidence required")
        _complete_request_proofs(case, timings, route, context_tokens)
        if key in costs:
            raise ValueError("duplicate price shape")
        # Conservative observed upper bound; no extrapolation or invented rate.
        costs[key] = float(max(timings))
    if evidence_sha256(data) != data["complete_request_evidence_sha256"]:
        raise ValueError("complete-request evidence digest differs from raw receipt")
    return MeasuredPackPrice(data["profile_id"], context_tokens,
                             MappingProxyType(costs), "complete_request",
                             data["complete_request_evidence_sha256"],
                             MappingProxyType(expected), _LOADER_TOKEN)


__all__ = ["IDENTITY_FIELDS", "PACK_STEP_UNIT", "MeasuredPackPrice", "ResearchCalibratedPrice",
           "ResearchWarmCalibratedPrice", "load_research_warm_calibration",
           "ResearchGraphB2Price", "SCHEMA", "evidence_sha256", "load_price",
           "load_research_calibration", "load_research_graph_b2_price",
           "shape_key"]
