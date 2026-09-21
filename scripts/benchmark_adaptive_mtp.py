#!/usr/bin/env python3
"""Three-arm production qualification for adaptive depth or MTP handoff.

The script owns each server process group from startup through confirmed exit.
It deliberately does not acquire the lab GPU lease; the coordinator wraps the
whole command after taking that lease.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import signal
import statistics
import subprocess
import sys
import time
from typing import Any
import urllib.error
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
FEATURE_SMOKE = (
    ROOT / "qualification/runs/quality-campaign-20260919/feature_smoke.py"
)
SCHEMA = "mlx2.adaptive-mtp-benchmark.v2"
DEFAULT_THROUGHPUT_TOLERANCE = 0.08
DEFAULT_MAX_PROBE_FRACTION = 0.075
DEFAULT_MIN_BUCKET_ROUNDS = 64
# GPU probes found bf16/quantized batch-shaped reductions can flip a greedy
# near-tie through 0.5 nats even for ordinary-vs-ordinary runs.  Equality is
# therefore safe; only a reference gap strictly above this ceiling fails.
#
# This ceiling classifies a divergence as "high margin"; it is NOT on its own a
# verdict.  Measured 2026-09-20 across four models, the statistic it thresholds
# is bf16-quantised: near the ceiling the observed grid is 0.375 / 0.500 /
# 0.625, so the two sides of the comparison differ by one representable step,
# and Xing's worst margin lands exactly on 0.500.  A high-margin count is
# therefore only meaningful against a control measured the same way -- see
# ``_high_margin_excess``.
DEFAULT_ORDINARY_MARGIN_FAILURE_THRESHOLD_NATS = 0.5
# Batch-width divergence is a property of batched decode, not of the handoff:
# at B16 the ordinary arm -- no MTP, no handoff -- reproduced its own width-one
# reference on 1/16 prompts (Flash-Next), 2/16 (Qwen3.6) and 0/16 (Xing,
# Qwen3.8).  The batched test is consequently a differential screen against the
# arm the handoff replaces (fixed native MTP at the same width), one-sided
# because only an excess matters.  It is a screen for gross regression, not the
# correctness gate: with eight prompts per width the smallest detectable excess
# is large, and it is recorded per run as ``minimum_detectable_excess``.  The
# correctness gate is the width-one exactness test in ``correctness()``, which
# runs in the only regime where token identity is attainable.
DEFAULT_DIFFERENTIAL_ALPHA = 0.05

PROMPTS = (
    "Explain how a compiler lowers a program, using numbered sections.",
    "Explain how a database transaction commits, using numbered sections.",
    "Explain how a CPU cache handles a miss, using numbered sections.",
    "Explain how a network router forwards a packet, using numbered sections.",
    "Explain how a filesystem journals a write, using numbered sections.",
    "Explain how a garbage collector reclaims objects, using numbered sections.",
    "Explain how TLS establishes a session, using numbered sections.",
    "Explain how a scheduler time-slices work, using numbered sections.",
)
MIXED_PROMPTS = (
    "Write a concise explanation of speculative decoding and its exactness rule.",
    "List six differences between latency and throughput in inference serving.",
    "Explain prefix caching to a systems engineer with one concrete example.",
    "Describe continuous batching and why request widths change over time.",
    "Give a short, deterministic overview of an append-only transaction log.",
    "Explain an exponentially weighted moving average with a numerical example.",
    "Describe backpressure in a bounded work queue and its failure modes.",
    "Explain why a production feature needs both selection and observed-use evidence.",
)
FEATURE_SMOKE_PROMPTS = (
    "Reply with exactly CAMPAIGN_READY.",
    "What is 37 plus 58? Reply with the number only.",
    "Name the capital of Canada in one word.",
    "Translate 'water' to French. One word.",
)


@dataclass(frozen=True)
class ModelPreset:
    path: str
    depth: int
    max_context: int = 262_144
    cache_gib: int = 8


PRESETS = {
    "qwen38": ModelPreset(
        "~/mlx-models/Qwen3.8-27B-oQ4e-mtp", 2
    ),
    "flash-next": ModelPreset(
        "~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP",
        2,
        cache_gib=16,
    ),
    "xing": ModelPreset(
        "~/mlx-models/Xing4.0-29B-A4B-mlx-6bit", 1
    ),
    "qwen36": ModelPreset(
        "~/mlx-models/"
        "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp",
        2,
    ),
}
CAPABILITIES = (
    "apc",
    "grammar",
    "reasoning",
    "streaming",
    "text",
    "thinking-deferral",
    "tools",
)


def _json_get(url: str, path: str, timeout: float = 30.0) -> dict[str, Any]:
    with urlopen(url + path, timeout=timeout) as response:
        return json.load(response)


def _health_ready(url: str) -> bool:
    try:
        status = _json_get(url, "/v1/status", timeout=2.0)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return bool(status.get("healthy"))


def _assert_arm_route(arm: str, status: dict) -> None:
    """Fail closed when an arm is not running the route it claims to measure."""

    expected = "ordinary" if arm == "ordinary" else "native_mtp"
    settings = status.get("settings") or {}
    actual = settings.get("route")
    if actual != expected:
        raise RuntimeError(
            f"arm {arm!r} resolved route {actual!r}, expected {expected!r}: "
            "the measurement would not be of the route it is labelled with"
        )


def _wait_ready(
    process: subprocess.Popen,
    url: str,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with {process.returncode}")
        if _health_ready(url) and _serves_a_request(url):
            return _json_get(url, "/v1/status")
        time.sleep(0.25)
    raise TimeoutError(f"server did not become healthy within {timeout:g}s")


def _serves_a_request(url: str) -> bool:
    """Readiness is "answers the operation we are about to measure".

    ``/v1/status`` reports ready seconds into a 27B load, well before the
    engine will admit work, so the first measured request dies on a 503.  The
    probe and the measurement must be the same operation.
    """

    try:
        model = (_json_get(url, "/v1/models", timeout=5.0)["data"] or [{}])[0].get("id")
    except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError):
        return False
    if not model:
        return False
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "ready?"}],
            "max_tokens": 1,
            "temperature": 0,
            "enable_thinking": False,
        }
    ).encode()
    request = Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.status == 200
    except urllib.error.HTTPError as error:
        # 503 AdmissionClosed / 429 are "not ready yet", not a failure.
        if error.code in (429, 503):
            return False
        raise
    except (OSError, urllib.error.URLError):
        return False


def stop_process_group(
    process: subprocess.Popen,
    url: str,
    *,
    grace_seconds: float = 15.0,
) -> dict[str, Any]:
    """Terminate the whole group, escalate if needed, and prove it is gone."""
    escalated = False
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            escalated = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=grace_seconds)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _health_ready(url):
        time.sleep(0.1)
    if process.poll() is None or _health_ready(url):
        raise RuntimeError("server process group or health endpoint survived shutdown")
    return {"returncode": process.returncode, "escalated_to_sigkill": escalated}


def _stream_request(url: str, prompt: str, timeout: float, max_tokens: int) -> dict:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 260919,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": True,
        "top_logprobs": 2,
    }
    started = time.monotonic()
    first_token_at = None
    output_parts: list[str] = []
    tokens: list[dict[str, Any]] = []
    terminal: dict[str, Any] = {}
    with urlopen(
        Request(
            url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=timeout,
    ) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            choices = event.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                logs = (choice.get("logprobs") or {}).get("content") or []
                if content:
                    output_parts.append(content)
                for item in logs:
                    tokens.append(
                        {
                            "id": item.get("id"),
                            "token": str(item.get("token", "")),
                            "logprob": item.get("logprob"),
                            "top_logprobs": [
                                {
                                    "id": candidate.get("id"),
                                    "token": str(candidate.get("token", "")),
                                    "logprob": candidate.get("logprob"),
                                }
                                for candidate in item.get("top_logprobs", [])
                            ],
                        }
                    )
                if first_token_at is None and (content or logs):
                    first_token_at = time.monotonic()
            if "usage" in event:
                terminal = event
    finished = time.monotonic()
    if not terminal.get("usage") or first_token_at is None:
        raise RuntimeError("stream completed without token timing or terminal usage")
    completion_tokens = int(terminal["usage"]["completion_tokens"])
    decode_seconds = max(finished - first_token_at, 1e-12)
    return {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "output": "".join(output_parts),
        "output_sha256": hashlib.sha256("".join(output_parts).encode()).hexdigest(),
        "tokens": tokens,
        "usage": terminal["usage"],
        "ttft_seconds": first_token_at - started,
        "wall_seconds": finished - started,
        "decode_tokens_per_second": max(0, completion_tokens - 1) / decode_seconds,
        "receipt": terminal.get("mlx2") or {},
    }


def _observed_widths(requests: list[dict]) -> list[int]:
    widths = set()
    for row in requests:
        receipt = row["receipt"]
        mtp = receipt.get("mtp") or {}
        values = mtp.get("observed_compute_widths")
        if values is None:
            value = receipt.get("ordinary_compute_width")
            values = [] if value is None else [value]
        widths.update(int(value) for value in values)
    return sorted(widths)


def _mtp_acceptance(requests: list[dict]) -> dict[str, float | int | None]:
    proposed = accepted = 0
    for row in requests:
        stats = ((row["receipt"].get("mtp") or {}).get("stats") or {})
        proposed += int(stats.get("draft_proposed", 0))
        accepted += int(stats.get("draft_accepted", 0))
    return {
        "accepted": accepted,
        "proposed": proposed,
        "fraction": accepted / proposed if proposed else None,
    }


def _adaptive_traces(requests: list[dict]) -> list[dict]:
    traces = []
    seen = set()
    for row in requests:
        adaptive = (row["receipt"].get("mtp") or {}).get("adaptive_depth") or {}
        trace = adaptive.get("trace") or []
        digest = hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if trace and digest not in seen:
            traces.append(
                {
                    "request": row["prompt_sha256"],
                    "current": adaptive.get("current"),
                    "counters": adaptive.get("counters") or {},
                    "cost_model": adaptive.get("cost_model") or {},
                    "trace": trace,
                }
            )
            seen.add(digest)
    return traces


def _run_width(
    url: str,
    width: int,
    timeout: float,
    max_tokens: int,
) -> dict:
    prompts = [MIXED_PROMPTS[index % len(MIXED_PROMPTS)] for index in range(width)]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=width) as pool:
        requests = list(
            pool.map(
                lambda prompt: _stream_request(url, prompt, timeout, max_tokens),
                prompts,
            )
        )
    elapsed = time.monotonic() - started
    completion_tokens = sum(row["usage"]["completion_tokens"] for row in requests)
    return {
        "width": width,
        "elapsed_seconds": elapsed,
        "aggregate_tokens_per_second": completion_tokens / elapsed,
        "observed_compute_widths": _observed_widths(requests),
        "mtp_acceptance": _mtp_acceptance(requests),
        "requests": requests,
        "adaptive_traces": _adaptive_traces(requests),
    }


def _run_sequential(url: str, timeout: float, max_tokens: int) -> dict:
    requests = [
        _stream_request(url, prompt, timeout, max_tokens) for prompt in PROMPTS
    ]
    return {
        "width": 1,
        "count": len(requests),
        "median_decode_tokens_per_second": statistics.median(
            row["decode_tokens_per_second"] for row in requests
        ),
        "median_ttft_seconds": statistics.median(
            row["ttft_seconds"] for row in requests
        ),
        "aggregate_tokens_per_second": sum(
            row["usage"]["completion_tokens"] for row in requests
        )
        / sum(row["wall_seconds"] for row in requests),
        "observed_compute_widths": _observed_widths(requests),
        "mtp_acceptance": _mtp_acceptance(requests),
        "requests": requests,
        "adaptive_traces": _adaptive_traces(requests),
    }


def _run_handoff_reference(url: str, timeout: float, max_tokens: int) -> dict:
    """Collect two independent width-one references for every batched prompt."""
    passes = [
        [
            _stream_request(url, prompt, timeout, max_tokens)
            for prompt in MIXED_PROMPTS
        ]
        for _ in range(2)
    ]
    return {
        "width": 1,
        "passes": passes,
        "observed_compute_widths": [
            _observed_widths(requests) for requests in passes
        ],
    }


def first_token_difference(left: list[Any], right: list[Any]) -> dict | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": index, "reference": a, "candidate": b}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "reference": left[index] if index < len(left) else None,
            "candidate": right[index] if index < len(right) else None,
        }
    return None


def _token_identities(request: dict) -> list[Any]:
    return [
        token.get("id") if token.get("id") is not None else token.get("token")
        for token in request["tokens"]
    ]


def _reference_margin(request: dict, index: int) -> float | None:
    if not 0 <= index < len(request["tokens"]):
        return None
    alternatives = request["tokens"][index].get("top_logprobs") or []
    values = sorted(
        (
            float(candidate["logprob"])
            for candidate in alternatives
            if isinstance(candidate.get("logprob"), (int, float))
            and not isinstance(candidate.get("logprob"), bool)
        ),
        reverse=True,
    )
    return values[0] - values[1] if len(values) >= 2 else None


def _reference_top_two(request: dict, index: int) -> list[Any] | None:
    if not 0 <= index < len(request["tokens"]):
        return None
    alternatives = request["tokens"][index].get("top_logprobs") or []
    ranked = sorted(
        (
            candidate
            for candidate in alternatives
            if isinstance(candidate.get("logprob"), (int, float))
            and not isinstance(candidate.get("logprob"), bool)
        ),
        key=lambda candidate: float(candidate["logprob"]),
        reverse=True,
    )
    if len(ranked) < 2:
        return None
    identities = [
        candidate.get("id")
        if candidate.get("id") is not None
        else candidate.get("token")
        for candidate in ranked[:2]
    ]
    return sorted(identities, key=lambda value: (type(value).__name__, str(value)))


def _stable_width_one_reference(
    references: list[dict], candidate: dict, index: int | None
) -> dict[str, Any]:
    widths = [_observed_widths([reference]) for reference in references]
    evidence = {
        "sample_count": len(references),
        "widths": widths,
        "stable": False,
        "reason": None,
        "top_two": None,
        "margin_samples_nats": [],
        "margin_bound_nats": None,
        "margin_spread_nats": None,
    }
    if len(references) < 2:
        evidence["reason"] = "insufficient_samples"
        return evidence
    prompt = candidate.get("prompt_sha256")
    if not prompt or any(reference.get("prompt_sha256") != prompt for reference in references):
        evidence["reason"] = "prompt_identity_mismatch"
        return evidence
    if any(width != [1] for width in widths):
        evidence["reason"] = "not_width_one"
        return evidence
    tokens = [_token_identities(reference) for reference in references]
    if any(row != tokens[0] for row in tokens[1:]):
        evidence["reason"] = "token_identity_mismatch"
        return evidence
    if index is None:
        evidence["stable"] = True
        return evidence
    top_two = [_reference_top_two(reference, index) for reference in references]
    if any(pair is None for pair in top_two):
        evidence["reason"] = "missing_top_two"
        return evidence
    if any(pair != top_two[0] for pair in top_two[1:]):
        evidence["reason"] = "top_two_identity_mismatch"
        return evidence
    margins = [_reference_margin(reference, index) for reference in references]
    if any(margin is None for margin in margins):
        evidence["reason"] = "missing_margin"
        return evidence
    margins = [float(margin) for margin in margins]
    evidence.update(
        stable=True,
        top_two=top_two[0],
        margin_samples_nats=margins,
        margin_bound_nats=max(margins),
        margin_spread_nats=max(margins) - min(margins),
    )
    return evidence


def _comparison_rows(reference: list[dict], candidate: list[dict]) -> list[dict]:
    rows = []
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        difference = first_token_difference(
            _token_identities(expected), _token_identities(actual)
        )
        if difference is not None:
            difference["reference_top2_margin_nats"] = _reference_margin(
                expected, difference["index"]
            )
        rows.append(
            {
                "prompt_index": index,
                "token_exact": difference is None,
                "text_exact": expected["output"] == actual["output"],
                "first_differing_token": difference,
                "reference_sha256": expected["output_sha256"],
                "candidate_sha256": actual["output_sha256"],
            }
        )
    return rows


def _fisher_one_sided(candidate_hits, candidate_n, control_hits, control_n):
    """P(candidate excess this large or larger | same underlying rate).

    One-sided because only a candidate that is *worse* than the arm it replaces
    is a defect; a candidate that diverges less is not evidence of a problem.
    """
    total_hits = candidate_hits + control_hits
    total = candidate_n + control_n
    if total == 0 or total_hits == 0 or total_hits == total:
        return 1.0

    def probability(hits):
        return (
            math.comb(candidate_n, hits)
            * math.comb(control_n, total_hits - hits)
            / math.comb(total, total_hits)
        )

    upper = min(candidate_n, total_hits)
    lower = max(0, total_hits - control_n)
    return sum(
        probability(hits)
        for hits in range(candidate_hits, upper + 1)
        if lower <= hits <= upper
    )


def _minimum_detectable_excess(candidate_n, control_n, alpha):
    """The smallest candidate count that would fail against a clean control.

    Recorded so the screen's weakness is visible in the report rather than
    implied by a passing verdict.
    """
    for hits in range(0, candidate_n + 1):
        if _fisher_one_sided(hits, candidate_n, 0, control_n) < alpha:
            return hits
    return None


def _by_prompt(rows):
    """Collapse rows to one observation per prompt.

    A batch runs each prompt on several lanes, so its rows share a prompt, a
    reference and therefore the near-tie that decides whether a divergence is
    "high margin".  They are not independent draws, and Fisher's exact test
    assumes they are: counting rows turns eight prompts into twenty-four
    observations and reports a confidence the measurement does not have.
    Measured 2026-09-21 on Flash-Next, where three runs each produced an
    identical 3/24 -- the giveaway, since independent 12.5% events do not
    repeat exactly -- which was two distinct prompts, one of them shared with
    the control.  By prompt it is 2/8 against 1/8.
    """
    hits = {}
    for row in rows:
        key = row.get("prompt_sha256") or row.get("reference_sha256")
        hits[key] = hits.get(key, False) or row["high_margin_divergence"]
    return sum(hits.values()), len(hits)


def _high_margin_excess(candidate_rows, control_rows, *, alpha):
    """Differential verdict: is the candidate worse than the arm it replaces?"""
    candidate_hits, candidate_n = _by_prompt(candidate_rows)
    control_hits, control_n = _by_prompt(control_rows)
    p_value = _fisher_one_sided(
        candidate_hits, candidate_n, control_hits, control_n
    )
    return {
        "candidate_high_margin_divergences": candidate_hits,
        "candidate_comparisons": candidate_n,
        "control_arm": "fixed",
        "control_high_margin_divergences": control_hits,
        "control_comparisons": control_n,
        # Raw row counts kept beside the deduplicated ones: the gap between
        # them is how much the batch inflates an apparent sample.
        "candidate_rows": len(candidate_rows),
        "control_rows": len(control_rows),
        "candidate_high_margin_rows": sum(
            row["high_margin_divergence"] for row in candidate_rows
        ),
        "control_high_margin_rows": sum(
            row["high_margin_divergence"] for row in control_rows
        ),
        "alpha": alpha,
        "p_value": p_value,
        "minimum_detectable_excess": _minimum_detectable_excess(
            candidate_n, control_n, alpha
        ),
        # Fail on a significant EXCESS -- a control that fires as often as the
        # candidate means the environment is producing the divergence -- and
        # fail closed when there are candidate rows but no control to judge
        # them against, because an unjudgeable screen is not a passing one.
        "passed": (
            (control_n > 0 or candidate_n == 0)
            and not (candidate_n > 0 and control_n > 0 and p_value < alpha)
        ),
        "reason": (
            None
            if candidate_n and control_n
            else "no_control_comparisons_fail_closed"
            if candidate_n
            else "no_candidate_comparisons"
        ),
    }


def _handoff_comparison(
    references: list[dict],
    candidate: dict,
    *,
    ordinary_margin_threshold: float,
    observed_widths: list[int],
    boundary_required: bool = True,
) -> dict:
    """Compare one batched request against its width-one references.

    ``boundary_required`` is False for the control arm, which has no handoff and
    therefore no boundary to validate.  Everything else -- reference stability,
    the divergence index, the margin classification -- is computed identically
    for both arms, because a control measured differently is not a control.
    """
    reference = references[0] if references else {
        "output": "",
        "output_sha256": "",
        "tokens": [],
    }
    row = _comparison_rows([reference], [candidate])[0]
    mtp = (candidate.get("receipt") or {}).get("mtp") or {}
    handoff = mtp.get("mtp_ordinary_handoff") or {}
    boundary = handoff.get("committed_tokens_before_handoff")
    valid_boundary = (
        isinstance(boundary, int)
        and not isinstance(boundary, bool)
        and 0 <= boundary <= len(candidate.get("tokens") or [])
    )
    difference = row["first_differing_token"]
    stability = _stable_width_one_reference(
        references,
        candidate,
        None if difference is None else difference["index"],
    )
    phase = None
    margin_safe = False
    high_margin = False
    if difference is not None:
        if valid_boundary and difference["index"] < boundary:
            phase = "pre_handoff"
        else:
            phase = "post_handoff" if valid_boundary else "unknown_boundary"
        margin = stability["margin_bound_nats"]
        difference["reference_top2_margin_nats"] = margin
        margin_safe = bool(
            margin is not None and margin <= ordinary_margin_threshold
        )
        high_margin = bool(
            margin is not None and margin > ordinary_margin_threshold
        )
    widths = sorted({int(width) for width in observed_widths if int(width) > 0})
    observed_width = max(widths) if widths else None
    strict_identity = observed_width == 1
    if observed_width is None:
        correctness_rule = "unknown_width_fail_closed"
        rule_passed = False
    elif not stability["stable"]:
        correctness_rule = "unstable_width_one_reference_fail_closed"
        rule_passed = False
    elif strict_identity:
        # Width one is the reproducible regime, so identity is required here and
        # a failure is a real defect rather than batched nondeterminism.
        correctness_rule = "strict_token_identity"
        rule_passed = difference is None
    else:
        # Above width one, identity is unattainable for EVERY arm, so this row
        # carries no verdict of its own; it contributes a classified divergence
        # to the differential screen in ``_high_margin_excess``.
        correctness_rule = "differential_high_margin_screen"
        rule_passed = True
    row.update(
        {
            "committed_tokens_before_handoff": boundary,
            "valid_handoff_boundary": valid_boundary,
            "divergence_phase": phase,
            "pre_handoff_margin_safe": margin_safe,
            "high_margin_divergence": high_margin,
            "post_handoff_continuation_exact": (
                difference is None if strict_identity and valid_boundary else None
            ),
            "observed_compute_widths": widths,
            "observed_width": observed_width,
            "width": observed_width,
            "reference_sample_count": stability["sample_count"],
            "reference_widths": stability["widths"],
            "reference_stable": stability["stable"],
            "reference_stability_reason": stability["reason"],
            "reference_top_two": stability["top_two"],
            "reference_top2_margin_samples_nats": stability[
                "margin_samples_nats"
            ],
            "reference_top2_margin_bound_nats": stability[
                "margin_bound_nats"
            ],
            "reference_top2_margin_spread_nats": stability[
                "margin_spread_nats"
            ],
            "correctness_rule": correctness_rule,
            "ordinary_margin_failure_threshold_nats": ordinary_margin_threshold,
            "margin_at_threshold_passes": True,
            "prompt_sha256": candidate.get("prompt_sha256"),
            "boundary_required": boundary_required,
            "passed": bool(
                (valid_boundary or not boundary_required)
                and observed_width is not None
                and rule_passed
            ),
        }
    )
    return row


def _adaptive_b1_fixed_depth(request: dict, fixed_depth: int) -> bool:
    mtp = (request.get("receipt") or {}).get("mtp") or {}
    adaptive = mtp.get("adaptive_depth") or {}
    width_one = [
        row for row in adaptive.get("trace") or [] if int(row.get("width", 0)) == 1
    ]
    return (
        int(mtp.get("num_draft", -1)) == fixed_depth
        and bool(width_one)
        and all(int(row.get("selected_depth", -1)) == fixed_depth for row in width_one)
    )


def correctness(
    arms: dict[str, dict],
    *,
    candidate_arm: str = "adaptive",
    fixed_depth: int,
    ordinary_margin_threshold: float,
) -> dict:
    ordinary = arms["ordinary"]["sequential"]["requests"]
    fixed = arms["fixed"]["sequential"]["requests"]
    candidate = arms[candidate_arm]["sequential"]["requests"]

    candidate_rows = _comparison_rows(fixed, candidate)
    depth_key = f"{candidate_arm}_used_fixed_depth_at_b1"
    for row, request in zip(candidate_rows, candidate):
        if candidate_arm == "adaptive":
            used_fixed_depth = _adaptive_b1_fixed_depth(request, fixed_depth)
        else:
            mtp = (request.get("receipt") or {}).get("mtp") or {}
            used_fixed_depth = int(mtp.get("num_draft", -1)) == fixed_depth
        row[depth_key] = used_fixed_depth
    candidate_matches = [
        row["token_exact"] and row[depth_key]
        for row in candidate_rows
    ]

    versus_ordinary = {}
    unsafe = []
    for name, requests in (("fixed", fixed), (candidate_arm, candidate)):
        rows = _comparison_rows(ordinary, requests)
        for row in rows:
            difference = row["first_differing_token"]
            margin = (
                None
                if difference is None
                else difference["reference_top2_margin_nats"]
            )
            row["ordinary_margin_exceeds_threshold"] = bool(
                margin is not None and margin > ordinary_margin_threshold
            )
            if row["ordinary_margin_exceeds_threshold"]:
                unsafe.append({"arm": name, **row})
        versus_ordinary[name] = {
            "exact_match_fraction": sum(row["token_exact"] for row in rows)
            / len(rows),
            "comparisons": rows,
        }

    candidate_result = {
        "exact_match_fraction": sum(candidate_matches) / len(candidate_matches),
        "comparisons": candidate_rows,
    }
    result = {
        "ordinary_margin_failure_threshold_nats": ordinary_margin_threshold,
        "qualification_arm": candidate_arm,
        "candidate_vs_fixed": candidate_result,
        "versus_ordinary": versus_ordinary,
        "unsafe_ordinary_divergences": unsafe,
        "passed": all(candidate_matches) and not unsafe,
    }
    result[f"{candidate_arm}_vs_fixed"] = candidate_result
    return result


def _policy(arm: str, depth: int) -> dict:
    policy: dict[str, Any] = {"num_draft": depth}
    if arm in {"fixed", "adaptive"}:
        # Keep fixed/adaptive controls isolated from the adapter handoff default.
        policy["mtp_ordinary_handoff"] = False
    if arm == "adaptive":
        policy["adaptive_mtp_depth"] = {"enabled": True}
    if arm == "handoff":
        policy["adaptive_mtp_depth"] = False
    return policy


def commands_for_arm(args, arm: str, run_dir: Path, preset: ModelPreset) -> dict:
    policy_path = run_dir / f"{arm}-policy.json"
    log_path = run_dir / f"{arm}-server.log"
    cache_dir = run_dir / f"{arm}-cache"
    server = [
        args.python,
        "-m",
        "mlx2.server",
        "--model",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-context",
        str(args.max_context or preset.max_context),
        "--max-lanes",
        str(max(args.widths)),
        "--max-inflight",
        str(max(args.widths) * 2),
        "--cache-bytes",
        str((args.cache_gib or preset.cache_gib) << 30),
        "--cache-dir",
        str(cache_dir),
        "--execution-policy",
        str(policy_path),
        "--qualification-mode",
        "--admin-token-file",
        str(run_dir / "admin-token"),
    ]
    if arm == "ordinary":
        server.append("--ordinary")
    else:
        # Name the route explicitly. Adapter defaults decide the route when no
        # flag is given (Qwen3.6 defaults to ordinary), which would silently
        # measure ordinary decode in an arm labelled native MTP.
        server.append("--native-mtp")
        if arm == "adaptive":
            # Exercise the CLI selection while recording the identical
            # structured selection in the policy identity.
            server.append("--adaptive-mtp-depth")
    server.extend(args.server_arg)
    smoke = [
        args.python,
        str(FEATURE_SMOKE),
        "--url",
        f"http://127.0.0.1:{args.port}",
        "--output",
        str(run_dir / f"{arm}-feature-smoke.json"),
        "--raw-dir",
        str(run_dir / f"{arm}-feature-smoke-raw"),
        "--model",
        args.model,
        "--model-id",
        Path(args.model_path).name,
        "--route",
        "adaptive-mtp-depth" if arm == "adaptive" else "mtp-ordinary-handoff",
        "--group",
        "core",
        "--speculative",
        "--fly",
        "--ordinary-baseline",
        str(run_dir / "ordinary-feature-baseline.json"),
        "--check-timeout",
        str(args.request_timeout),
        "--admin-token-file",
        str(run_dir / "admin-token"),
    ]
    for capability in CAPABILITIES:
        smoke.extend(("--capability", capability))
    policy = _policy(arm, args.depth)
    if arm == "handoff":
        policy["mtp_ordinary_handoff"] = {
            "enabled": True,
            "max_mtp_width": args.mtp_ordinary_handoff_max_width,
        }
    return {
        "policy_path": str(policy_path),
        "policy": policy,
        "server": server,
        "server_log": str(log_path),
        "feature_smoke": smoke if arm in {"adaptive", "handoff"} else None,
    }


def _run_arm(args, arm: str, commands: dict, run_dir: Path) -> dict:
    Path(commands["policy_path"]).write_text(
        json.dumps(commands["policy"], indent=2) + "\n"
    )
    cache_dir = run_dir / f"{arm}-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    log_path = Path(commands["server_log"])
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            commands["server"],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    lifecycle = {"pid": process.pid}
    try:
        initial = _wait_ready(process, url, args.startup_timeout)
        _assert_arm_route(arm, initial)
        # Concurrent load comes first so the following eight B1 requests test
        # recovery of the same retained controller, not a fresh controller.
        concurrent = [
            _run_width(url, width, args.request_timeout, args.max_tokens)
            for width in args.widths
        ]
        sequential = _run_sequential(url, args.request_timeout, args.max_tokens)
        handoff_reference = None
        if arm == "ordinary":
            handoff_reference = _run_handoff_reference(
                url, args.request_timeout, args.max_tokens
            )
            baseline = {
                prompt: _stream_request(url, prompt, args.request_timeout, 64)[
                    "output"
                ]
                for prompt in FEATURE_SMOKE_PROMPTS
            }
            (run_dir / "ordinary-feature-baseline.json").write_text(
                json.dumps(baseline, indent=2, ensure_ascii=False) + "\n"
            )
        feature_smoke = None
        if arm in {"adaptive", "handoff"}:
            smoke_log = run_dir / f"{arm}-feature-smoke.log"
            with smoke_log.open("wb") as output:
                completed = subprocess.run(
                    commands["feature_smoke"],
                    cwd=ROOT,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            feature_smoke = {
                "command": commands["feature_smoke"],
                "returncode": completed.returncode,
                "output": str(run_dir / f"{arm}-feature-smoke.json"),
                "log": str(smoke_log),
            }
        final = _json_get(url, "/v1/status")
        if not final.get("healthy") or final.get("inflight"):
            raise RuntimeError("server was not healthy and idle after workload")
        return {
            "commands": commands,
            "initial_status": initial,
            "concurrent": concurrent,
            "sequential": sequential,
            "handoff_reference": handoff_reference,
            "final_status": final,
            "feature_smoke": feature_smoke,
            "lifecycle": lifecycle,
        }
    finally:
        lifecycle["shutdown"] = stop_process_group(process, url)


def _text_chart(report: dict) -> str:
    candidate_arm = report.get("qualification_arm", "adaptive")
    lines = [
        "Native-MTP policy qualification",
        "",
        "arm       B1 decode tok/s  B1 TTFT ms  "
        + "  ".join(f"B{width} aggregate" for width in report["widths"]),
    ]
    for name in report.get("arm_order", ("ordinary", "fixed", candidate_arm)):
        arm = report["arms"].get(name, {})
        seq = arm.get("sequential", {})
        concurrent = {row["width"]: row for row in arm.get("concurrent", [])}
        fields = [
            f"{name:<10}",
            f"{seq.get('median_decode_tokens_per_second', float('nan')):>15.2f}",
            f"{1000 * seq.get('median_ttft_seconds', float('nan')):>10.1f}",
        ]
        fields.extend(
            f"{concurrent.get(width, {}).get('aggregate_tokens_per_second', float('nan')):>13.2f}"
            for width in report["widths"]
        )
        lines.append(" ".join(fields))
    correctness_report = report.get("correctness") or {}
    qualification = report.get("adaptive_qualification") or {}
    handoff = qualification.get("handoff") or {}
    handoff_exact = handoff.get("exact_match_fraction")
    handoff_exact_text = (
        f"{handoff_exact:.3f}" if handoff_exact is not None else "n/a"
    )
    handoff_rules = handoff.get("correctness_rules") or {}
    lines.extend(
        (
            "",
            f"{candidate_arm} vs fixed B1 exact-match fraction: "
            f"{correctness_report.get('candidate_vs_fixed', {}).get('exact_match_fraction', float('nan')):.3f}",
            f"fixed / {candidate_arm} exact vs ordinary: "
            f"{correctness_report.get('versus_ordinary', {}).get('fixed', {}).get('exact_match_fraction', float('nan')):.3f} / "
            f"{correctness_report.get('versus_ordinary', {}).get(candidate_arm, {}).get('exact_match_fraction', float('nan')):.3f}",
            "ordinary divergences above margin threshold: "
            f"{len(correctness_report.get('unsafe_ordinary_divergences', []))}",
            "enabled benchmark features: "
            f"{', '.join(qualification.get('enabled_features', [])) or 'none'}",
            "missing feature evidence: "
            f"{', '.join(qualification.get('missing_feature_evidence', [])) or 'none'}",
            "adaptive decrease-under-concurrency / recovery-alone: "
            f"{report.get('adaptive_observation', {}).get('decreases', 0)} / "
            f"{report.get('adaptive_observation', {}).get('recoveries', 0)}",
            "handoff exact fraction / unsafe divergences: "
            f"{handoff_exact_text} / "
            f"{len(handoff.get('unsafe_divergences', []))}",
            "handoff identity / stable-margin / unknown-reference comparisons: "
            f"{handoff_rules.get('strict_token_identity', 0)} / "
            f"{handoff_rules.get('stable_width_one_top2_margin', 0)} / "
            f"{handoff_rules.get('unknown_width_fail_closed', 0) + handoff_rules.get('unstable_width_one_reference_fail_closed', 0)}",
            "policy qualification: "
            f"{'PASS' if report.get('adaptive_qualification', {}).get('passed') else 'FAIL'}",
        )
    )
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(PRESETS), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8296)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--max-tokens", type=int, default=256)
    # The qualifier's long-context checks prime a ~131k-token shared prefix.
    # A preset sized for throughput arms (8 GiB) cannot hold it, so
    # shared_warm_requests fails on capacity rather than on the feature under
    # test. Qualification runs pass 16 to match the historical qualified profile.
    parser.add_argument("--cache-gib", type=int, default=None,
                        help="override the preset APC cache size (GiB)")
    # Qualification runs pin this to 131072: APC disk-tier restore fails at the
    # full 262144 cap (pre-existing defect, measured 2026-09-20, see
    # wiki lessons/apc-disk-restore-fails-at-full-context), so the qualifier's
    # `context` check cannot pass there for reasons unrelated to this feature.
    parser.add_argument("--max-context", type=int, default=None,
                        help="override the preset max context")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--ordinary-margin-threshold",
        type=float,
        default=DEFAULT_ORDINARY_MARGIN_FAILURE_THRESHOLD_NATS,
        help=(
            "Fail ordinary-reference divergences only when the top-2 margin "
            "is strictly greater than this value in nats (equality passes)."
        ),
    )
    parser.add_argument(
        "--throughput-tolerance",
        type=float,
        default=DEFAULT_THROUGHPUT_TOLERANCE,
        help="Maximum fractional candidate throughput loss versus fixed.",
    )
    parser.add_argument(
        "--max-probe-fraction",
        type=float,
        default=DEFAULT_MAX_PROBE_FRACTION,
        help="Maximum exploration fraction in an eligible concurrent bucket.",
    )
    parser.add_argument(
        "--min-bucket-rounds",
        type=int,
        default=DEFAULT_MIN_BUCKET_ROUNDS,
        help="Concurrent bucket rounds required before complete sampling is gated.",
    )
    parser.add_argument(
        "--mtp-ordinary-handoff-max-width",
        type=int,
        help=(
            "Run the fixed-depth handoff qualification arm and retain native "
            "MTP through this physical width."
        ),
    )
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    preset = PRESETS[args.model]
    args.model_path = args.model_path or preset.path
    args.depth = args.depth or preset.depth
    args.text_output = args.text_output or args.output.with_suffix(".txt")
    if args.depth < 1 or args.max_tokens < 1:
        parser.error("depth and max-tokens must be positive")
    if (
        not args.widths
        or len(args.widths) != len(set(args.widths))
        or any(width < 2 for width in args.widths)
    ):
        parser.error("widths must be unique integers >= 2")
    if not 1024 <= args.port <= 65535:
        parser.error("port must be in 1024..65535")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    if (
        not math.isfinite(args.ordinary_margin_threshold)
        or args.ordinary_margin_threshold < 0
    ):
        parser.error("ordinary-margin-threshold must be finite and nonnegative")
    if not 0 <= args.throughput_tolerance < 1:
        parser.error("throughput-tolerance must be in [0, 1)")
    if not 0 < args.max_probe_fraction <= 1:
        parser.error("max-probe-fraction must be in (0, 1]")
    if args.min_bucket_rounds < 1:
        parser.error("min-bucket-rounds must be positive")
    if (
        args.mtp_ordinary_handoff_max_width is not None
        and args.mtp_ordinary_handoff_max_width < 1
    ):
        parser.error("mtp-ordinary-handoff-max-width must be positive")
    return args


def adaptive_qualification_evidence(
    report: dict,
    *,
    throughput_tolerance: float = DEFAULT_THROUGHPUT_TOLERANCE,
    max_probe_fraction: float = DEFAULT_MAX_PROBE_FRACTION,
    min_bucket_rounds: int = DEFAULT_MIN_BUCKET_ROUNDS,
    differential_alpha: float = DEFAULT_DIFFERENTIAL_ALPHA,
) -> dict[str, Any]:
    """Evaluate the selected MTP policy without conflating its features."""
    fixed = report["arms"]["fixed"]
    candidate_arm = report.get("qualification_arm", "adaptive")
    if candidate_arm not in {"adaptive", "handoff"}:
        raise ValueError(f"unsupported qualification arm {candidate_arm!r}")
    candidate = report["arms"][candidate_arm]
    throughput = []
    fixed_b1 = float(fixed["sequential"]["median_decode_tokens_per_second"])
    candidate_b1 = float(candidate["sequential"]["median_decode_tokens_per_second"])
    throughput.append(
        {
            "width": 1,
            "fixed_tokens_per_second": fixed_b1,
            "candidate_tokens_per_second": candidate_b1,
            "passed": candidate_b1 >= fixed_b1 * (1.0 - throughput_tolerance),
        }
    )
    fixed_widths = {row["width"]: row for row in fixed["concurrent"]}
    for row in candidate["concurrent"]:
        width = int(row["width"])
        fixed_rate = float(fixed_widths[width]["aggregate_tokens_per_second"])
        candidate_rate = float(row["aggregate_tokens_per_second"])
        throughput.append(
            {
                "width": width,
                "fixed_tokens_per_second": fixed_rate,
                "candidate_tokens_per_second": candidate_rate,
                "passed": candidate_rate
                >= fixed_rate * (1.0 - throughput_tolerance),
            }
        )

    status = candidate["final_status"]
    scheduler = status.get("scheduler") or {}
    cost_model = scheduler.get("adaptive_mtp_cost_model") or {}
    settings = status.get("settings") or {}
    adaptive_policy = settings.get("adaptive_mtp_depth") or {}
    adaptive_selected = bool(adaptive_policy.get("enabled"))
    max_depth = int(report["qualified_depth"])
    min_samples = int(adaptive_policy.get("min_samples_per_depth", 3))
    bucket_rows = []
    eligible = 0
    for bucket, state in sorted((cost_model.get("buckets") or {}).items()):
        rounds = int(state.get("rounds", 0))
        if bucket == "1" or rounds < min_bucket_rounds:
            continue
        eligible += 1
        samples = state.get("samples") or {}
        missing = [
            depth
            for depth in range(max_depth + 1)
            if int(samples.get(str(depth), 0)) < min_samples
        ]
        probe_fraction = float(state.get("probe_fraction", 0.0))
        bucket_rows.append(
            {
                "bucket": bucket,
                "rounds": rounds,
                "minimum_samples_per_depth": min_samples,
                "missing_depths": missing,
                "probe_fraction": probe_fraction,
                "passed": not missing and probe_fraction <= max_probe_fraction,
            }
        )
    decreases = int(scheduler.get("adaptive_mtp_depth_decreases_concurrent", 0))
    recoveries = int(scheduler.get("adaptive_mtp_depth_recoveries_alone", 0))
    recovery = {
        "decreases": decreases,
        "recoveries": recoveries,
        "required": decreases > 0,
        "passed": decreases == 0 or recoveries > 0,
    }
    smoke = candidate.get("feature_smoke") or {}
    smoke_passed = smoke.get("returncode") == 0
    handoff_selected = bool(
        (settings.get("mtp_ordinary_handoff") or {}).get("enabled")
    )
    handoff_events = int(scheduler.get("mtp_ordinary_handoff_events", 0))
    handoff_lanes = int(scheduler.get("mtp_ordinary_handoff_lanes", 0))
    ordinary_margin_threshold = float(
        report["correctness"]["ordinary_margin_failure_threshold_nats"]
    )
    handoff_comparisons = []
    control_comparisons = []
    if handoff_selected:
        reference_by_prompt: dict[str, list[dict]] = {}
        reference_passes = (
            report["arms"]["ordinary"].get("handoff_reference") or {}
        ).get("passes") or []
        for reference_pass in reference_passes:
            for reference in reference_pass:
                prompt = reference.get("prompt_sha256")
                if prompt:
                    reference_by_prompt.setdefault(prompt, []).append(reference)

        def _compare(arm_requests, width, *, boundary_required):
            return [
                dict(
                    _handoff_comparison(
                        reference_by_prompt.get(
                            request.get("prompt_sha256"), []
                        ),
                        request,
                        ordinary_margin_threshold=ordinary_margin_threshold,
                        observed_widths=_observed_widths([request]),
                        boundary_required=boundary_required,
                    ),
                    workload_width=int(width),
                )
                for request in arm_requests
            ]

        for candidate_width in candidate["concurrent"]:
            engaged_requests = [
                request
                for request in (candidate_width.get("requests") or [])
                if (
                    (
                        ((request.get("receipt") or {}).get("mtp") or {}).get(
                            "mtp_ordinary_handoff"
                        )
                        or {}
                    ).get("engaged")
                )
            ]
            handoff_comparisons.extend(
                _compare(
                    engaged_requests,
                    candidate_width["width"],
                    boundary_required=True,
                )
            )
        # The control: the arm the handoff replaces, at the same widths, against
        # the same references, classified by the same rule.  Without it an
        # absolute count cannot separate the lever from the route it runs on.
        control_widths = {
            int(bucket["width"]) for bucket in candidate["concurrent"]
        }
        for control_width in report["arms"]["fixed"]["concurrent"]:
            if int(control_width["width"]) in control_widths:
                control_comparisons.extend(
                    _compare(
                        control_width.get("requests") or [],
                        control_width["width"],
                        boundary_required=False,
                    )
                )
    unsafe_handoff = [
        row for row in handoff_comparisons if not row["passed"]
    ]
    excess = _high_margin_excess(
        handoff_comparisons, control_comparisons, alpha=differential_alpha
    )
    rule_names = (
        "strict_token_identity",
        "differential_high_margin_screen",
        "unknown_width_fail_closed",
        "unstable_width_one_reference_fail_closed",
    )
    handoff = {
        "selected": handoff_selected,
        "events": handoff_events,
        "lanes": handoff_lanes,
        "ordinary_margin_failure_threshold_nats": ordinary_margin_threshold,
        "differential_alpha": differential_alpha,
        "comparison_count": len(handoff_comparisons),
        "control_comparison_count": len(control_comparisons),
        "correctness_rules": {
            rule: sum(
                row["correctness_rule"] == rule for row in handoff_comparisons
            )
            for rule in rule_names
        },
        "control_correctness_rules": {
            rule: sum(
                row["correctness_rule"] == rule for row in control_comparisons
            )
            for rule in rule_names
        },
        "exact_match_fraction": (
            sum(row["token_exact"] for row in handoff_comparisons)
            / len(handoff_comparisons)
            if handoff_comparisons
            else None
        ),
        # Recorded beside the candidate's, because a candidate fraction near
        # zero is only alarming if the control's is not.
        "control_exact_match_fraction": (
            sum(row["token_exact"] for row in control_comparisons)
            / len(control_comparisons)
            if control_comparisons
            else None
        ),
        "high_margin_excess": excess,
        "comparisons": handoff_comparisons,
        "control_comparisons": control_comparisons,
        "unsafe_divergences": unsafe_handoff,
        "passed": (
            not handoff_selected
            or (
                handoff_events > 0
                and bool(handoff_comparisons)
                and not unsafe_handoff
                and excess["passed"]
            )
        ),
    }
    correctness_passed = bool(report.get("correctness", {}).get("passed"))
    throughput_passed = all(row["passed"] for row in throughput)
    common_passed = correctness_passed and throughput_passed and smoke_passed
    adaptive_evidence_passed = (
        not adaptive_selected
        or (
            eligible > 0
            and all(row["passed"] for row in bucket_rows)
            and recovery["passed"]
        )
    )
    adaptive_feature_passed = common_passed and adaptive_evidence_passed
    handoff_feature_passed = common_passed and handoff["passed"]
    features = {
        "adaptive_mtp_depth": {
            "selected": adaptive_selected,
            "passed": (
                adaptive_feature_passed if adaptive_selected else True
            ),
        },
        "mtp_ordinary_handoff": {
            "selected": handoff_selected,
            "passed": handoff_feature_passed if handoff_selected else True,
        },
    }
    missing_feature_evidence = [
        name
        for name in ("adaptive_mtp_depth", "mtp_ordinary_handoff")
        if features[name]["selected"] and not features[name]["passed"]
    ]
    enabled_features = [
        name for name, feature in features.items() if feature["selected"]
    ]
    passed = bool(enabled_features) and not missing_feature_evidence
    return {
        "passed": passed,
        "qualification_arm": candidate_arm,
        "enabled_features": enabled_features,
        "features": features,
        "missing_feature_evidence": missing_feature_evidence,
        "throughput_tolerance": throughput_tolerance,
        "differential_alpha": differential_alpha,
        "throughput": throughput,
        "max_probe_fraction": max_probe_fraction,
        "min_bucket_rounds": min_bucket_rounds,
        "eligible_concurrent_buckets": eligible,
        "buckets": bucket_rows,
        "recovery": recovery,
        "feature_smoke_passed": smoke_passed,
        "handoff": handoff,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    preset = PRESETS[args.model]
    run_dir = args.output.parent / f"{args.output.stem}-artifacts"
    qualification_arm = (
        "handoff"
        if args.mtp_ordinary_handoff_max_width is not None
        else "adaptive"
    )
    arm_order = ("ordinary", "fixed", qualification_arm)
    command_plan = {
        arm: commands_for_arm(args, arm, run_dir, preset)
        for arm in arm_order
    }
    for arm in arm_order:
        print(f"[{arm}] {shlex.join(command_plan[arm]['server'])}")
        if command_plan[arm]["feature_smoke"]:
            print(
                f"[{arm} feature-smoke] "
                f"{shlex.join(command_plan[arm]['feature_smoke'])}"
            )
    if args.dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    token_path = run_dir / "admin-token"
    token_path.write_text(secrets.token_urlsafe(32) + "\n")
    token_path.chmod(0o600)
    report = {
        "schema": SCHEMA,
        "benchmark_harness": {
            "name": "scripts/benchmark_adaptive_mtp.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "model": args.model,
        "model_path": args.model_path,
        "qualified_depth": args.depth,
        "max_tokens": args.max_tokens,
        "widths": args.widths,
        "qualification_arm": qualification_arm,
        "arm_order": list(arm_order),
        "prompts": {
            "sequential": list(PROMPTS),
            "mixed": list(MIXED_PROMPTS),
            "handoff_reference": list(MIXED_PROMPTS),
            "handoff_reference_passes": 2,
        },
        "arms": {},
    }
    for arm in arm_order:
        report["arms"][arm] = _run_arm(
            args, arm, command_plan[arm], run_dir
        )
    report["correctness"] = correctness(
        report["arms"],
        candidate_arm=qualification_arm,
        fixed_depth=args.depth,
        ordinary_margin_threshold=args.ordinary_margin_threshold,
    )
    scheduler = report["arms"][qualification_arm]["final_status"].get("scheduler") or {}
    report["adaptive_observation"] = {
        "decreases": int(
            scheduler.get("adaptive_mtp_depth_decreases_concurrent", 0)
        ),
        "recoveries": int(
            scheduler.get("adaptive_mtp_depth_recoveries_alone", 0)
        ),
    }
    report["adaptive_qualification"] = adaptive_qualification_evidence(
        report,
        throughput_tolerance=args.throughput_tolerance,
        max_probe_fraction=args.max_probe_fraction,
        min_bucket_rounds=args.min_bucket_rounds,
    )
    report["passed"] = report["adaptive_qualification"]["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.text_output.write_text(_text_chart(report))
    print(_text_chart(report), end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
