#!/usr/bin/env python3
"""Served A/B on Qwen3.8 27B: ordinary, self-MTP, and DFlash2.

GPU-only (starts real ``mlx2.server`` processes, one model per process).
Refuses to run without ``--i-own-the-gpu``; ``--dry-run`` prints the plan.

Arms (same target artifact, same server flags except the route):

* ``ord``  ``--ordinary``
* ``ordv`` ``--ordinary`` with target-side varlen dense MLP packing
* ``mtp2`` native self-MTP with an explicit ``num_draft: 2`` policy
* ``mtp3`` current adapter-default native self-MTP route (num_draft 3)
* ``dkN``  ``--external-draft`` with the pinned DFlash2 policy
  ``policy-kN.json`` (block size N+1)
* ``dkNv`` the same external route composed with target-side varlen packing

Per arm process: load, a discarded warm-up (every workload, both
temperatures, B1 and B4, so first-use Metal shape compiles land outside the
timed cells), then every cell: workload x width x temperature.  Arms run in
the order given; a campaign alternates the order between reps (ABBA over
processes) so drift lands on both sides.  APCv2 writes and the in-process
host prompt cache are disabled; deterministic nonces keep warm-up and timed
prompt geometry identical without silently warming tokenization.

Swap guard: ``vm_stat`` pageouts/swapouts are sampled every 2 s.  A rise of
1.5 GiB from the pre-launch baseline aborts during load/warm-up; once the
timed cells start, a rise of 256 MiB aborts.

Per request it records TTFT, decode tok/s ((tokens-1)/(elapsed-ttft)), the
route receipt and the full output text (for the greedy equality gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "qualification/runs/sp-dflash2-27b-20260925"
MODEL = str(Path.home() / "mlx-models/Qwen3.8-27B-oQ4e-mtp")

CODE = [
    "Write a Python module implementing an LRU cache class with get, put and "
    "delete, O(1) operations, type hints, docstrings and five pytest tests. "
    "Output only the code.",
    "Implement Dijkstra's shortest path in TypeScript over an adjacency list "
    "with a binary-heap priority queue, plus a small usage example. Output "
    "only the code.",
    "Write a Rust function that parses a CSV line with quoted fields and "
    "escaped quotes into a Vec<String>, with unit tests. Output only the code.",
    "Write a bash script that rotates log files in a directory: gzip files "
    "older than 1 day, delete archives older than 30 days, and print a "
    "summary. Output only the code.",
]
PROSE = [
    "Write a 500-word essay on why cities build public libraries.",
    "Explain, for a general audience, how vaccines train the immune system.",
    "Write a short story about a lighthouse keeper who finds a message in a "
    "bottle written in her own handwriting.",
    "Describe the history of the printing press and its effect on Europe.",
]
CHAT = [
    [
        {"role": "user", "content": "I'm planning a 3-day trip to Kyoto in November."},
        {"role": "assistant", "content": "Great choice: autumn leaves peak in mid to late November. What are your interests?"},
        {"role": "user", "content": "Temples, food, and a bit of hiking. Give me a day-by-day plan."},
    ],
    [
        {"role": "user", "content": "My Python script says 'list index out of range' in a loop."},
        {"role": "assistant", "content": "Can you share the loop?"},
        {"role": "user", "content": "for i in range(len(xs)+1): print(xs[i])\nWhy does it fail and how do I fix it? Also explain enumerate."},
    ],
    [
        {"role": "user", "content": "What's the difference between a Roth IRA and a traditional IRA, in general terms?"},
        {"role": "assistant", "content": "The main difference is when you pay tax. Want a comparison table?"},
        {"role": "user", "content": "Yes, a table, then a short explanation of who each tends to suit."},
    ],
    [
        {"role": "user", "content": "Help me write SQL: tables orders(id, customer_id, total, created_at) and customers(id, name)."},
        {"role": "assistant", "content": "Sure. What should the query return?"},
        {"role": "user", "content": "Top 5 customers by total spend in 2025, with their order count. Then explain the query."},
    ],
]
REPETITIVE = [
    "Write exactly 80 Markdown checklist lines using this template: "
    "- [ ] Review item N: owner=ops status=pending. Replace only N with the "
    "line number. Output only the checklist.",
    "Write exactly 80 CSV data rows after the header id,region,status. Use "
    "successive integer ids, alternate region between east and west, and keep "
    "status=pending. Output only CSV.",
    "Write exactly 60 SQL INSERT statements for audit_events(id, actor, action). "
    "Use successive ids, actor 'worker', and action 'heartbeat'. Output only SQL.",
]
WORKLOADS = {
    "code": CODE,
    "prose": PROSE,
    "chat": CHAT,
    "repetitive": REPETITIVE,
}
DEFAULT_WORKLOADS = ("code", "prose", "chat")
POLICIES = ROOT / "qualification/policies"
INGRESS_COALESCE_MS = 250
INGRESS_MIN_PROMPT_TOKENS = 1
INGRESS_TARGET_LANES = 4
MAX_B1_PROMPTS = min(len(WORKLOADS[name]) for name in DEFAULT_WORKLOADS)

# Qwen3.8's model-card non-thinking profile.  ``temperature`` is supplied by
# the campaign cell so an explicit override remains visible and truthful.
# Presence penalty 1.5 installs a stateful logits processor and therefore makes
# DFlash's batched pairwise proposal selector ineligible.  Greedy cells use a
# separately named neutral-penalty control so they can prove pairwise use.
NON_THINKING_SAMPLED = {
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.0,
}
CONTROLLED_GREEDY = {
    "temperature": 0.0,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def sampling_controls(temperature):
    """Return the complete sampling law used by one campaign cell."""
    value = float(temperature)
    if value < 0:
        raise ValueError("temperature must be non-negative")
    if value == 0:
        return dict(CONTROLLED_GREEDY)
    return {"temperature": value, **NON_THINKING_SAMPLED}


def validate_campaign_shape(widths, b1_prompts, temperatures):
    """Keep the named comparison exactly B1 serial controls versus B4."""
    if list(widths) != [1, 4]:
        raise ValueError("widths must be exactly 1,4 for the B1/B4 campaign")
    if type(b1_prompts) is not int or b1_prompts != MAX_B1_PROMPTS:
        raise ValueError(
            f"b1-prompts must be exactly {MAX_B1_PROMPTS} so B1 and B4 use "
            "the same prompt mix"
        )
    values = list(temperatures)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("temperatures must be finite, non-negative values")
    if len(values) != len(set(values)):
        raise ValueError("temperatures must be unique")


def workload_batch(workload, width, b1_prompts):
    """Return the identical prompt set used by warm-up and timed cells."""
    items = WORKLOADS[workload]
    return items[:b1_prompts] if width == 1 else items[:width]


def request_nonce(nonce_salt, rep, workload, width, temperature, index):
    """Bind warm-up and timed requests to identical, cache-cold prompt shapes."""
    cell = hashlib.sha256(
        f"{nonce_salt}:{rep}:{workload}:{width}:{temperature}".encode()
    ).hexdigest()[:16]
    return hashlib.sha256(f"{cell}:{index}".encode()).hexdigest()[:16]


def arm_route_args(arm, external_policy=None):
    if arm == "ord":
        return ["--ordinary"]
    if arm == "ordv":
        return ["--ordinary", "--execution-policy", str(RUN / "policy-varlen.json")]
    if arm == "mtp2":
        return [
            "--execution-policy",
            str(POLICIES / "qwen38-27b-mtp2.json"),
        ]
    if arm == "mtp3":
        return []
    match = re.fullmatch(r"dk(\d)(v?)", arm)
    if match:
        depth = int(match.group(1))
        if depth not in {3, 4, 5, 6, 7}:
            raise ValueError(
                f"unknown DFlash arm {arm}: policies exist only for dk3 through dk7"
            )
        if match.group(2) and depth not in {4, 7}:
            raise ValueError(
                f"unknown varlen arm {arm}: only dk4v and dk7v have policies"
            )
        if match.group(2):
            policy = POLICIES / (
                f"qwen38-27b-dflash2-k{depth}-varlen-ingress.json"
            )
        else:
            policy = (
                Path(external_policy)
                if external_policy
                else RUN / f"policy-k{depth}.json"
            )
        return ["--external-draft", "--execution-policy", str(policy)]
    raise ValueError(f"unknown arm {arm}")


def parse_arms(value, external_policy=None):
    """Return a nonempty, unique, validated campaign arm sequence."""
    arms = [item.strip() for item in value.split(",")]
    if not arms or any(not arm for arm in arms):
        raise ValueError("arms must be a nonempty comma-separated list")
    if len(arms) != len(set(arms)):
        raise ValueError("arms must be unique within a campaign repetition")
    for arm in arms:
        arm_route_args(arm, external_policy)
    return arms


def server_command(args, arm, port):
    """Build a cache-cold server command for one measured arm."""
    return [
        sys.executable,
        "-m",
        "mlx2.server",
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-context",
        str(args.max_context),
        "--max-lanes",
        str(args.max_lanes),
        "--max-inflight",
        str(args.max_inflight),
        "--host-prompt-cache-entries",
        "0",
        "--host-prompt-cache-tokens",
        "0",
        "--qualification-mode",
        "--lane-matmul",
        args.lane_matmul,
        *arm_route_args(arm, args.external_policy),
    ]


def arm_contract(arm):
    """Expected status/receipt identity for an arm, before any timed work."""
    if arm in {"ord", "ordv"}:
        return {
            "route": "ordinary",
            "speculation": "ordinary",
            "mtp": False,
            "num_draft": None,
            "pairwise_selection": None,
        }
    if arm in {"mtp2", "mtp3"}:
        return {
            "route": "native_mtp",
            "speculation": "self_mtp",
            "mtp": True,
            "num_draft": int(arm.removeprefix("mtp")),
            "pairwise_selection": None,
        }
    match = re.fullmatch(r"dk(\d)(v?)", arm)
    if match:
        depth = int(match.group(1))
        if depth not in {3, 4, 5, 6, 7}:
            raise ValueError(
                f"unknown DFlash arm {arm}: policies exist only for dk3 through dk7"
            )
        if match.group(2) and depth not in {4, 7}:
            raise ValueError(
                f"unknown varlen arm {arm}: only dk4v and dk7v have policies"
            )
        return {
            "route": "external_draft",
            "speculation": "external_draft",
            "mtp": False,
            "num_draft": depth,
            "pairwise_selection": "batched",
        }
    raise ValueError(f"unknown arm {arm}")


def route_snapshot(status):
    """Copy route identity from the nested serving settings namespace."""
    settings = status.get("settings")
    if not isinstance(settings, dict):
        raise RuntimeError("refusing: status has no serving settings")
    return {
        "route": settings.get("route"),
        "speculation": settings.get("speculation"),
        "mtp": settings.get("mtp"),
        "execution_policy": settings.get("execution_policy"),
        "apcv2_reuse": settings.get("apcv2_reuse"),
        "host_prompt_cache_entries": settings.get(
            "host_prompt_cache_entries"
        ),
        "host_prompt_cache_tokens": settings.get("host_prompt_cache_tokens"),
        "route_selection_source": settings.get("route_selection_source"),
        "profile": status.get("profile"),
        "artifact": status.get("artifact"),
        "runtime": status.get("runtime"),
    }


def validate_startup_route(arm, status):
    """Fail before warm-up if the server did not select the named arm."""
    expected = arm_contract(arm)
    route = route_snapshot(status)
    for name in ("host_prompt_cache_entries", "host_prompt_cache_tokens"):
        if route[name] != 0:
            raise RuntimeError(
                f"refusing arm {arm}: status {name}={route[name]!r}, expected 0"
            )
    for name in ("route", "speculation", "mtp"):
        if route[name] != expected[name]:
            raise RuntimeError(
                f"refusing arm {arm}: status {name}={route[name]!r}, "
                f"expected {expected[name]!r}"
            )
    policy = route["execution_policy"]
    if not isinstance(policy, dict):
        raise RuntimeError(f"refusing arm {arm}: status has no execution policy")
    if expected["num_draft"] is not None and (
        policy.get("num_draft") != expected["num_draft"]
    ):
        raise RuntimeError(
            f"refusing arm {arm}: status num_draft={policy.get('num_draft')!r}, "
            f"expected {expected['num_draft']}"
        )
    if expected["speculation"] == "external_draft" and (
        policy.get("backend") != "external_draft"
    ):
        raise RuntimeError(
            f"refusing arm {arm}: status is not the external-draft backend"
        )
    if expected["pairwise_selection"] is not None and (
        policy.get("pairwise_selection") != expected["pairwise_selection"]
    ):
        raise RuntimeError(
            f"refusing arm {arm}: status pairwise_selection="
            f"{policy.get('pairwise_selection')!r}, expected "
            f"{expected['pairwise_selection']!r}"
        )
    ingress = policy.get("ingress_cohort") or {}
    varlen_requested = arm.startswith("dk") and arm.endswith("v")
    varlen_selected = (
        (policy.get("external_varlen_prefill") or {}).get("enabled") is True
    )
    ingress_selected = ingress.get("enabled") is True
    if varlen_requested and not varlen_selected:
        raise RuntimeError(
            f"refusing arm {arm}: external varlen prefill was not selected"
        )
    if varlen_requested and not ingress_selected:
        raise RuntimeError(
            f"refusing arm {arm}: ingress cohort policy was not selected"
        )
    if varlen_requested and (
        ingress.get("mechanism") != "external_varlen_prefill"
        or ingress.get("target_lanes") != INGRESS_TARGET_LANES
        or ingress.get("maximum_wait_ms") != INGRESS_COALESCE_MS
        or ingress.get("minimum_prompt_tokens")
        != INGRESS_MIN_PROMPT_TOKENS
    ):
        raise RuntimeError(
            f"refusing arm {arm}: invalid ingress cohort contract {ingress}"
        )
    if varlen_requested and route["apcv2_reuse"] != {
        "enabled": False,
        "reason": "external_varlen_prefill_not_batch_invariant",
    }:
        raise RuntimeError(
            f"refusing arm {arm}: APCv2 reuse was not disabled fail-closed"
        )
    if not varlen_requested and (varlen_selected or ingress_selected):
        raise RuntimeError(
            f"refusing arm {arm}: unrequested varlen ingress policy was selected"
        )
    return route


def validate_served_model(status, model):
    """Require the status model identity to equal the requested basename."""
    expected = Path(model).name
    served = status.get("model")
    if served != expected:
        raise RuntimeError(
            f"refusing: server reports model {served!r}, expected {expected!r}"
        )
    return served


def validate_request_receipt(arm, receipt, expected_sampling):
    """Validate per-request route/depth and return effective controls."""
    expected = arm_contract(arm)
    speculation = receipt.get("speculation") or {}
    mtp = receipt.get("mtp")
    if receipt.get("route") != expected["route"]:
        raise RuntimeError(
            f"refusing arm {arm}: request route={receipt.get('route')!r}, "
            f"expected {expected['route']!r}"
        )
    if expected["speculation"] == "external_draft":
        if speculation.get("kind") != "external_dflash2":
            raise RuntimeError(
                f"refusing arm {arm}: receipt is not DFlash2: {speculation}"
            )
        if speculation.get("execution") != "external_draft_verify" or (
            speculation.get("ordinary_fallback") is not False
        ):
            raise RuntimeError(
                f"refusing arm {arm}: DFlash2 request did not stay on its route: "
                f"{speculation}"
            )
        if mtp:
            raise RuntimeError(f"refusing arm {arm}: native MTP receipt on DFlash2")
        if arm.endswith("v") and receipt.get("apcv2_reuse") != {
            "enabled": False,
            "reason": "external_varlen_prefill_not_batch_invariant",
        }:
            raise RuntimeError(
                f"refusing arm {arm}: request did not disclose disabled APCv2 reuse"
            )
    elif expected["mtp"]:
        if not isinstance(mtp, dict):
            raise RuntimeError(f"refusing arm {arm}: request is not native self-MTP")
        if mtp.get("num_draft") != expected["num_draft"]:
            raise RuntimeError(
                f"refusing arm {arm}: receipt num_draft={mtp.get('num_draft')!r}, "
                f"expected {expected['num_draft']}"
            )
        if speculation:
            raise RuntimeError(
                f"refusing arm {arm}: external speculation receipt on native MTP"
            )
    elif mtp or speculation:
        raise RuntimeError(
            f"refusing arm {arm}: speculative receipt on the ordinary arm"
        )

    controls = receipt.get("request_controls")
    effective = controls.get("effective_sampling") if isinstance(controls, dict) else None
    if not isinstance(effective, dict):
        raise RuntimeError(f"refusing arm {arm}: receipt lacks effective sampling")
    if controls.get("skip_writing_prefix_cache") is not True:
        raise RuntimeError(
            f"refusing arm {arm}: request did not suppress APCv2 writes"
        )
    if receipt.get("cached_tokens") != 0:
        raise RuntimeError(
            f"refusing arm {arm}: benchmark request reused APCv2 state"
        )
    mismatches = {
        name: {"expected": value, "observed": effective.get(name)}
        for name, value in expected_sampling.items()
        if effective.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            f"refusing arm {arm}: effective sampling mismatch: {mismatches}"
        )
    return controls


def pairwise_control_eligibility(effective_sampling):
    """Conservatively identify controls supported by batched pair selection."""
    blockers = []
    if effective_sampling.get("presence_penalty", 0.0) != 0.0:
        blockers.append("presence_penalty")
    if effective_sampling.get("frequency_penalty", 0.0) != 0.0:
        blockers.append("frequency_penalty")
    if effective_sampling.get("repetition_penalty", 1.0) != 1.0:
        blockers.append("repetition_penalty")
    if effective_sampling.get("logit_bias"):
        blockers.append("logit_bias")
    return {"eligible": not blockers, "blockers": blockers}


def _strict_counter(status, section, name, *, required):
    values = status.get(section)
    if not isinstance(values, dict) or name not in values:
        if required:
            raise RuntimeError(
                f"refusing: {section}.{name} counter is missing"
            )
        return 0
    value = values[name]
    if type(value) is not int or value < 0:
        raise RuntimeError(
            f"refusing: {section}.{name} must be a non-negative integer"
        )
    return value


def pairwise_cell_observation(arm, width, rows, before, after):
    """Bind pairwise claims to effective controls and a per-cell counter delta."""
    contract = arm_contract(arm)
    selected = contract["pairwise_selection"] == "batched"
    controls = [
        pairwise_control_eligibility(row["request_controls"]["effective_sampling"])
        for row in rows
    ]
    blockers = sorted({reason for item in controls for reason in item["blockers"]})
    controls_eligible = bool(controls) and all(item["eligible"] for item in controls)
    concurrent = int(width) > 1 and len(rows) > 1
    expected_engagement = selected and controls_eligible and bool(rows)
    name = "external_pairwise_selection_groups"
    initial = _strict_counter(
        before, "scheduler", name, required=selected
    )
    final = _strict_counter(
        after, "scheduler", name, required=selected
    )
    if final < initial:
        raise RuntimeError(f"refusing arm {arm}: pairwise counter moved backwards")
    delta = final - initial
    if not expected_engagement and delta:
        raise RuntimeError(
            f"refusing arm {arm}: cell advanced pairwise counter outside "
            f"eligible engagement ({blockers})"
        )
    if expected_engagement and delta < 1:
        raise RuntimeError(
            f"refusing arm {arm}: eligible batched pairwise cell did not engage"
        )
    return {
        "selected": selected,
        "mode": contract["pairwise_selection"],
        "controls_eligible": controls_eligible,
        "concurrent": concurrent,
        "expected_engagement": expected_engagement,
        "ineligibility_reasons": blockers,
        "groups_before": initial,
        "groups_after": final,
        "groups_delta": delta,
        "observed_used": delta > 0,
    }


def ingress_timed_observation(arm, before, after):
    """Require target-width ingress engagement during measured cells only."""
    policy = (
        ((after.get("settings") or {}).get("execution_policy") or {})
        .get("ingress_cohort")
        or {}
    )
    selected = policy.get("enabled") is True
    requested = arm.startswith("dk") and arm.endswith("v")
    name = "ingress_cohort_target_reached"

    required = requested or selected
    initial = _strict_counter(
        before, "counts", name, required=required
    )
    final = _strict_counter(
        after, "counts", name, required=required
    )
    if final < initial:
        raise RuntimeError(f"refusing arm {arm}: ingress counter moved backwards")
    delta = final - initial
    if requested and not selected:
        raise RuntimeError(
            f"refusing arm {arm}: ingress cohort policy was not selected"
        )
    if not requested and selected:
        raise RuntimeError(
            f"refusing arm {arm}: unrequested ingress cohort policy was selected"
        )
    if selected and delta < 1:
        raise RuntimeError(
            f"refusing arm {arm}: timed cells did not execute a physical "
            "target-width ingress cohort"
        )
    return {
        "requested": requested,
        "selected": selected,
        "target_lanes": policy.get("target_lanes") if selected else None,
        "target_reached_before": initial,
        "target_reached_after": final,
        "target_reached_delta": delta,
        "observed_used": delta > 0,
    }


def ingress_cell_observation(arm, width, rows):
    """Bind every named B1/B4 result to its own physical ingress receipt."""
    requested = arm.startswith("dk") and arm.endswith("v")
    receipts = [row.get("ingress_cohort") for row in rows]
    if not requested:
        if any(receipt is not None for receipt in receipts):
            raise RuntimeError(
                f"refusing arm {arm}: unrequested ingress receipt in timed cell"
            )
        return {
            "requested": False,
            "target_expected": False,
            "observed_used": False,
        }
    if not receipts or not all(isinstance(receipt, dict) for receipt in receipts):
        raise RuntimeError(
            f"refusing arm {arm}: timed cell lacks ingress receipts"
        )
    for receipt in receipts:
        if (
            receipt.get("selected") is not True
            or receipt.get("mechanism") != "external_varlen_prefill"
            or receipt.get("target_lanes") != INGRESS_TARGET_LANES
            or receipt.get("maximum_wait_ms") != INGRESS_COALESCE_MS
        ):
            raise RuntimeError(
                f"refusing arm {arm}: timed cell ingress identity mismatch"
            )
    target_expected = int(width) == INGRESS_TARGET_LANES
    if target_expected:
        if not all(
            receipt.get("target_reached") is True
            and receipt.get("member_observed_used") is True
            and receipt.get("observed_used") is True
            and receipt.get("execution_width") == INGRESS_TARGET_LANES
            and receipt.get("admission_width") == INGRESS_TARGET_LANES
            and receipt.get("formation_target_reached") is True
            and receipt.get("expired") is False
            for receipt in receipts
        ):
            raise RuntimeError(
                f"refusing arm {arm}: B4 timed cell did not physically execute "
                "its four-lane ingress cohort"
            )
    elif not all(
        receipt.get("admission_width") == 1
        and receipt.get("execution_width") == 0
        and receipt.get("formation_target_reached") is False
        and receipt.get("target_reached") is False
        and receipt.get("member_observed_used") is False
        and receipt.get("observed_used") is False
        and receipt.get("expired") is True
        for receipt in receipts
    ):
        raise RuntimeError(
            f"refusing arm {arm}: B1 timed control ingress receipt is not "
            "an expired width-one formation"
        )
    return {
        "requested": True,
        "target_expected": target_expected,
        "target_reached": all(
            receipt.get("target_reached") is True for receipt in receipts
        ),
        "observed_used": any(
            receipt.get("observed_used") is True for receipt in receipts
        ),
        "execution_widths": sorted(
            {int(receipt.get("execution_width", 0)) for receipt in receipts}
        ),
    }


def varlen_timed_observation(arm, before, after, *, phase="timed"):
    """Bind varlen claims to paired phase-local mechanism deltas."""
    if phase not in {"timed", "warm-up"}:
        raise ValueError("phase must be timed or warm-up")
    requested = arm.endswith("v")
    initial = (before.get("execution") or {}).get("varlen_dense_mlp") or {}
    final = (after.get("execution") or {}).get("varlen_dense_mlp") or {}
    initial_selected = initial.get("selected") is True
    selected = final.get("selected") is True
    if requested and not (initial_selected and selected):
        raise RuntimeError(
            f"refusing arm {arm}: target varlen policy was not selected"
        )
    if not requested and (initial_selected or selected):
        raise RuntimeError(
            f"refusing arm {arm}: unrequested target varlen policy was selected"
        )

    deltas = {}
    for name in ("mlp_compaction_calls", "padding_token_rows"):
        start = _strict_counter(
            initial,
            "counters",
            name,
            required=(requested or initial_selected) and phase != "warm-up",
        )
        end = _strict_counter(
            final, "counters", name, required=requested or selected
        )
        if end < start:
            raise RuntimeError(
                f"refusing arm {arm}: varlen {name} counter moved backwards"
            )
        deltas[name] = end - start
    observed = min(deltas.values()) > 0
    if requested and not observed:
        raise RuntimeError(
            f"refusing arm {arm}: {phase} cells did not engage varlen "
            "compaction with padding"
        )
    return {
        "requested": requested,
        "selected": selected,
        "counter_deltas": deltas,
        "observed_used": observed,
    }


def varlen_cell_observation(arm, width, before, after, *, phase="timed"):
    """Bind each B1/B4 cell to its own target-varlen counter deltas."""
    if phase not in {"timed", "warm-up"}:
        raise ValueError("phase must be timed or warm-up")
    requested = arm.endswith("v")
    initial = (before.get("execution") or {}).get("varlen_dense_mlp") or {}
    final = (after.get("execution") or {}).get("varlen_dense_mlp") or {}
    initial_selected = initial.get("selected") is True
    selected = final.get("selected") is True
    if requested and not (initial_selected and selected):
        raise RuntimeError(
            f"refusing arm {arm}: target varlen policy was not selected"
        )
    if not requested and (initial_selected or selected):
        raise RuntimeError(
            f"refusing arm {arm}: unrequested target varlen policy was selected"
        )

    target_expected = requested and int(width) == INGRESS_TARGET_LANES
    deltas = {}
    for name in ("mlp_compaction_calls", "padding_token_rows"):
        start = _strict_counter(
            initial,
            "counters",
            name,
            required=requested and phase == "timed",
        )
        end = _strict_counter(
            final,
            "counters",
            name,
            required=requested and (phase == "timed" or target_expected),
        )
        if end < start:
            raise RuntimeError(
                f"refusing arm {arm}: varlen {name} counter moved backwards"
            )
        deltas[name] = end - start

    observed = min(deltas.values()) > 0
    if target_expected and not observed:
        raise RuntimeError(
            f"refusing arm {arm}: {phase} B4 cell did not engage varlen "
            "compaction with padding"
        )
    if not target_expected and any(deltas.values()):
        raise RuntimeError(
            f"refusing arm {arm}: {phase} B1/control cell unexpectedly "
            "advanced target-varlen counters"
        )
    return {
        "requested": requested,
        "selected": selected,
        "target_expected": target_expected,
        "counter_deltas": deltas,
        "observed_used": observed,
    }


def warmup_observation(arm, before, after, cells):
    """Prove warm-up exercised the same ingress/varlen shapes as timed work."""
    ingress = [
        ingress_cell_observation(arm, width, rows)
        for width, rows, _pairwise, _varlen in cells
    ]
    return {
        "ingress_cells": ingress,
        "pairwise_cells": [
            pairwise for _width, _rows, pairwise, _varlen in cells
        ],
        "varlen_cells": [
            varlen for _width, _rows, _pairwise, varlen in cells
        ],
        "varlen": varlen_timed_observation(
            arm, before, after, phase="warm-up"
        ),
    }


def _post(url, body, timeout):
    request = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _status(url):
    with urlopen(url + "/v1/status", timeout=30) as response:
        return json.load(response)


class SwapGuard:
    """Sample vm_stat swapouts; flag a breach instead of killing from a thread."""

    PAGE = 16384

    def __init__(self):
        self.base = self.read()
        self.limit = int(1.5 * (1 << 30))
        self.breach = None
        self.peak = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    @classmethod
    def read(cls):
        text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        match = re.search(r"Swapouts:\s+(\d+)", text)
        return int(match.group(1)) * cls.PAGE if match else 0

    def arm_timed(self):
        self.base = self.read()
        self.limit = 256 << 20

    def _loop(self):
        while not self._stop.wait(2.0):
            rise = self.read() - self.base
            self.peak = max(self.peak, rise)
            if rise >= self.limit and self.breach is None:
                self.breach = rise

    def check(self):
        if self.breach is not None:
            raise RuntimeError(f"swap guard: swapouts rose {self.breach >> 20} MiB")

    def close(self):
        self._stop.set()


def _wait_ready(url, process, timeout, guard):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        guard.check()
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        try:
            status = _status(url)
            if status.get("healthy") and status.get("ready", True):
                return status
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("server did not become ready")


def _messages(item, nonce):
    if isinstance(item, str):
        return [{"role": "user", "content": f"[{nonce}] {item}"}]
    messages = [dict(message) for message in item]
    messages[0]["content"] = f"[{nonce}] {messages[0]['content']}"
    return messages


def _request(url, item, *, arm, max_tokens, temperature, timeout, nonce, seed):
    expected_sampling = sampling_controls(temperature)
    body = {"messages": _messages(item, nonce), "max_tokens": max_tokens,
            "enable_thinking": False, "skip_writing_prefix_cache": True,
            **expected_sampling}
    if expected_sampling["temperature"] > 0:
        body["seed"] = seed
    started = time.perf_counter()
    result = _post(url, body, timeout)
    wall = time.perf_counter() - started
    receipt = result["mlx2"]
    speculation = receipt.get("speculation") or {}
    mtp = receipt.get("mtp")
    request_controls = validate_request_receipt(
        arm, receipt, expected_sampling
    )
    tokens = int(receipt["completion_tokens"])
    ttft = float(receipt["ttft_seconds"])
    decode_s = float(receipt["elapsed_seconds"]) - ttft
    content = result["choices"][0]["message"].get("content") or ""
    reasoning = result["choices"][0]["message"].get("reasoning_content") or ""
    return {
        "completion_tokens": tokens,
        "prompt_tokens": receipt.get("prompt_tokens"),
        "cached_tokens": receipt.get("cached_tokens"),
        "ttft_s": ttft,
        "decode_tok_s": (tokens - 1) / decode_s if decode_s > 0 and tokens > 1 else None,
        "wall_s": wall,
        "finish_reason": result["choices"][0].get("finish_reason"),
        "output_sha256": hashlib.sha256((reasoning + "\x00" + content).encode()).hexdigest(),
        "output": content,
        "reasoning": reasoning,
        "speculation": speculation or None,
        "mtp": mtp,
        "request_controls": request_controls,
        "ingress_cohort": receipt.get("ingress_cohort"),
        "route": receipt.get("route"),
        "profile": receipt.get("profile"),
    }


def run_cell(url, arm, workload, width, temperature, args, seed_base, rep):
    batch = workload_batch(workload, width, args.b1_prompts)
    # Identical across arms (so greedy outputs are comparable), distinct per
    # request and cell. Every request also suppresses APCv2 writes and refuses
    # a nonzero cached-token receipt, so common template/BOS prefixes cannot
    # make the ordinary/MTP controls warmer than packed-varlen's forced-cold
    # route. APCv2 disk persistence is off between server processes.
    def one(pair):
        index, item = pair
        nonce = request_nonce(
            args.nonce_salt, rep, workload, width, temperature, index
        )
        return _request(url, item, arm=arm, max_tokens=args.max_tokens,
                        temperature=temperature, timeout=args.timeout,
                        nonce=nonce, seed=seed_base + index)

    started = time.perf_counter()
    if width > 1:
        with ThreadPoolExecutor(max_workers=width) as pool:
            rows = list(pool.map(one, enumerate(batch)))
    else:
        rows = [one(pair) for pair in enumerate(batch)]
    wall = time.perf_counter() - started
    for index, row in enumerate(rows):
        row["prompt_index"] = index
    return {"rows": rows, "cell_wall_s": wall,
            "aggregate_tok_s": sum(r["completion_tokens"] for r in rows) / wall}


def run_arm(args, arm, rep, *, port=None):
    port = args.port if port is None else int(port)
    url = f"http://127.0.0.1:{port}"
    command = server_command(args, arm, port)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    out = Path(args.out)
    log_path = out.with_suffix(f".{arm}.r{rep}.server.log")
    import socket

    with socket.socket() as probe:
        # Another session's server on this port would answer our polls.
        probe.bind(("127.0.0.1", port))
    guard = SwapGuard()
    log = open(log_path, "w")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    cells = []
    final_status = {}
    ingress_timed = None
    varlen_timed = None
    warmup_evidence = None
    try:
        status = _wait_ready(url, process, args.startup_timeout, guard)
        validate_served_model(status, args.model)
        route = validate_startup_route(arm, status)
        lane = status.get("lane_matmul") or {}
        if args.require_simd_lane and (
            lane.get("backend") != "simd" or not lane.get("installed")
        ):
            raise RuntimeError(
                "refusing: SIMD lane matmul is not installed: " + json.dumps(lane)
            )
        # Discarded warm-up: every workload, both temperatures, B1 and B4.
        warmup_baseline = _status(url)
        warmup_cells = []
        for temperature in args.temperatures:
            for workload in args.workloads:
                for width in args.widths:
                    guard.check()
                    items = workload_batch(
                        workload, width, args.b1_prompts
                    )
                    status_before = _status(url)
                    with ThreadPoolExecutor(max_workers=width) as pool:
                        rows = list(pool.map(lambda pair: _request(
                            url, pair[1], arm=arm,
                            max_tokens=args.warmup_tokens,
                            temperature=temperature, timeout=args.timeout,
                            nonce=request_nonce(
                                args.nonce_salt, rep, workload, width,
                                temperature, pair[0]
                            ), seed=7), enumerate(items)))
                    status_after = _status(url)
                    pairwise = pairwise_cell_observation(
                        arm, width, rows, status_before, status_after
                    )
                    varlen = varlen_cell_observation(
                        arm,
                        width,
                        status_before,
                        status_after,
                        phase="warm-up",
                    )
                    warmup_cells.append((width, rows, pairwise, varlen))
        guard.check()
        timed_baseline = _status(url)
        warmup_evidence = warmup_observation(
            arm, warmup_baseline, timed_baseline, warmup_cells
        )
        guard.arm_timed()
        seed_base = 1000
        for temperature in args.temperatures:
            for workload in args.workloads:
                for width in args.widths:
                    status_before = _status(url)
                    cell = run_cell(url, arm, workload, width, temperature, args, seed_base, rep)
                    guard.check()
                    status_after = _status(url)
                    cell["pairwise_selection"] = pairwise_cell_observation(
                        arm,
                        width,
                        cell["rows"],
                        status_before,
                        status_after,
                    )
                    cell["ingress_observation"] = ingress_cell_observation(
                        arm, width, cell["rows"]
                    )
                    cell["varlen_observation"] = varlen_cell_observation(
                        arm, width, status_before, status_after
                    )
                    cell.update({"arm": arm, "rep": rep, "workload": workload, "width": width,
                                 "temperature": temperature,
                                 "metal_peak_bytes": status_after.get("metal_peak_bytes"),
                                 "route_status": route})
                    cells.append(cell)
                    rates = [r["decode_tok_s"] for r in cell["rows"] if r["decode_tok_s"]]
                    print(json.dumps({"arm": arm, "rep": rep, "workload": workload,
                                      "width": width, "temperature": temperature,
                                      "decode_tok_s": [round(x, 2) for x in rates],
                                      "aggregate_tok_s": round(cell["aggregate_tok_s"], 2),
                                      "swap_peak_mib": guard.peak >> 20}), flush=True)
        final_status = _status(url)
        final_lane = final_status.get("lane_matmul") or {}
        lane_counts = final_lane.get("counts") or {}
        if args.require_simd_lane and not any(
            int(value) > 0
            for key, value in lane_counts.items()
            if "lane" in key or "simd" in key
        ):
            raise RuntimeError(
                "refusing: SIMD lane matmul installed but no lane calls observed: "
                + json.dumps(final_lane)
            )
        varlen_timed = varlen_timed_observation(
            arm, timed_baseline, final_status
        )
        ingress_timed = ingress_timed_observation(
            arm, timed_baseline, final_status
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
        guard.close()
    return cells, {"arm": arm, "rep": rep, "swap_peak_mib": guard.peak >> 20,
                   "scheduler": final_status.get("scheduler") if cells else None,
                   "execution": final_status.get("execution") if cells else None,
                   "ingress_cohort": {
                       key: value
                       for key, value in (final_status.get("counts") or {}).items()
                       if key.startswith("ingress_cohort_")
                   } if cells else None,
                   "ingress_timed": ingress_timed if cells else None,
                   "warmup_evidence": warmup_evidence if cells else None,
                   "varlen_timed": varlen_timed if cells else None,
                   "lane_matmul": final_status.get("lane_matmul") if cells else None,
                   "metal_peak_bytes": final_status.get("metal_peak_bytes") if cells else None}


def source_identity(expected):
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("refusing: source worktree is dirty")
    if expected and revision != expected:
        raise RuntimeError(
            f"refusing: source revision {revision} does not match {expected}"
        )
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms", required=True,
        help=("comma list: ord,ordv,mtp2,mtp3,dk3,dk4,dk5,dk6,dk7,"
              "dk4v,dk7v"),
    )
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--external-policy",
        help="override the pinned policy path for dkN compatibility arms",
    )
    parser.add_argument("--port", type=int, default=18731)
    parser.add_argument(
        "--lane-matmul",
        choices=("auto", "off", "crossover", "exact"),
        default="auto",
    )
    parser.add_argument("--require-simd-lane", action="store_true")
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--max-lanes", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--warmup-tokens", type=int, default=48)
    parser.add_argument("--b1-prompts", type=int, default=MAX_B1_PROMPTS)
    parser.add_argument("--workloads", default="code,prose,chat")
    parser.add_argument("--widths", default="1,4")
    parser.add_argument("--temperatures", default="0,0.7")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--nonce-salt", default="sp-dflash2")
    parser.add_argument("--expected-source")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    args.widths = [int(v) for v in args.widths.split(",")]
    args.temperatures = [float(v) for v in args.temperatures.split(",")]
    args.workloads = [value for value in args.workloads.split(",") if value]
    unknown_workloads = set(args.workloads) - set(WORKLOADS)
    if not args.workloads or unknown_workloads:
        parser.error(f"unknown workloads: {sorted(unknown_workloads)}")
    custom_shape = (
        args.workloads != list(DEFAULT_WORKLOADS)
        or args.widths != [1, 4]
        or args.b1_prompts != MAX_B1_PROMPTS
        or args.max_lanes != 4
        or args.max_inflight != 8
        or args.external_policy is not None
        or args.lane_matmul != "auto"
    )
    if custom_shape:
        if (
            not args.widths
            or any(width < 1 for width in args.widths)
            or not 1 <= args.b1_prompts <= min(
                len(WORKLOADS[name]) for name in args.workloads
            )
            or not args.temperatures
            or any(
                not math.isfinite(value) or value < 0
                for value in args.temperatures
            )
            or len(args.temperatures) != len(set(args.temperatures))
        ):
            parser.error("invalid custom campaign shape")
    else:
        try:
            validate_campaign_shape(
                args.widths, args.b1_prompts, args.temperatures
            )
        except ValueError as error:
            parser.error(str(error))
    try:
        arms = parse_arms(args.arms, args.external_policy)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"arms": arms, "rep": args.rep, "widths": args.widths,
                          "temperatures": args.temperatures,
                          "b1_prompts": args.b1_prompts,
                          "sampling_controls": {
                              str(value): sampling_controls(value)
                              for value in args.temperatures
                          },
                          "arm_contracts": {
                              arm: arm_contract(arm) for arm in arms
                          },
                          "max_tokens": args.max_tokens}))
        return 0
    if not args.i_own_the_gpu:
        raise SystemExit("refusing: pass --i-own-the-gpu inside the GPU lock")
    revision = source_identity(args.expected_source)
    out = Path(args.out)
    with open(out, "a") as stream:
        stream.write(json.dumps({"campaign": {
            "source_revision": revision,
            "arms": arms,
            "rep": args.rep,
            "widths": args.widths,
            "temperatures": args.temperatures,
            "b1_prompts": args.b1_prompts,
            "nonce_salt": args.nonce_salt,
            "host_prompt_cache": {"entries": 0, "tokens": 0},
            "max_tokens": args.max_tokens,
        }}) + "\n")
        stream.flush()
        for offset, arm in enumerate(arms):
            # macOS may retain a just-closed listening port briefly. Distinct
            # checked ports keep arm transitions deterministic and prevent a
            # later process from ever attaching to the previous endpoint.
            cells, summary = run_arm(
                args, arm, args.rep, port=args.port + offset
            )
            for cell in cells:
                stream.write(json.dumps(cell) + "\n")
            stream.write(json.dumps({"summary": summary}) + "\n")
            stream.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
