#!/usr/bin/env python3
"""Exercise an exclusively owned candidate server and write its route evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import re
from pathlib import Path
import platform
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


QUALIFICATION_HARNESS_SCHEMA = "mlx2.qualification-harness.v1"
APPROVED_ADAPTIVE_BENCHMARK_SHA256 = (
    "d7d2f51195008db855a942632ba0ab87cfa709ecf6d27996c824cc5aeacabd6f"
)
PREFLIGHT_SCHEMA = "mlx2.qualification-preflight.v2"
PREFLIGHT_DELTA_SCHEMA = "mlx2.qualification-preflight-delta.v1"
CROSS_HOST_PREFLIGHT_SCHEMA = "mlx2.qualification-preflight.v3"
PREFLIGHT_TENSORFOLD_OWNED_SOURCE = (
    Path.home() / ".codex/worktrees/tensorfold-upstream-parity-20260928"
)
PREFLIGHT_TENSORFOLD_OWNED_MLX_LM_SOURCE = (
    Path.home() / "Desktop/mlx-uag/mlx-lm-unified"
)
QUALIFICATION_COVERAGE = {
    "response_format_json_object": False,
    "strict_json_schema": True,
    "explicit_grammar": False,
    "physical_n2": False,
    "overload_429_retry_after": False,
    "latency_ttft_itl_percentiles": False,
    "tenant_jain_fairness": False,
    "progress_events": False,
}
LONG_CONTEXT_HEADROOM = 256
LONG_CONTEXT_COMPLETION_TOKENS = 64
CANCELLATION_PROBE_MAX_TOKENS = 8192
REASONING_PROBE_TOKEN_BUDGETS = (512, 1024, 2048)
# Keep enough room for a final answer even when a model ignores the request for
# short reasoning.  This is sent only when the adapter declares state-aware
# thinking deferral; adapters without a single-token close marker still use
# their unguarded, naturally-closing reasoning path.
REASONING_PROBE_THINKING_BUDGET = 128
# No literal reasoning markers: where "</think>" is a special token it would be
# encoded as a control token inside the system turn (Xing4.0 then never closes
# its own reasoning under greedy decoding).
REASONING_PROBE_SYSTEM = (
    "Use one short sentence of private reasoning. Do not discuss formatting or "
    "repeat the instructions. Then end your reasoning and give the requested "
    "final answer."
)
LONG_CONTEXT_CACHE_TOLERANCE = 50
# Extra completion budget for servers whose model thinks by default
# (`/v1/status.thinking_default`).  The checks' own budgets are sized for the
# answer; a thinking model spends tokens reasoning first, and North's reasoning
# on an open-ended prompt has been observed past 800 tokens.
THINKING_BUDGET_TOKENS = 2048
SERVER_MAX_TOKENS = 2_097_152  # API ceiling; the allowance never crosses it
LONG_CONTEXT_INSTRUCTION = (
    "\nStart with exactly LONG_READY then write a detailed numbered guide to "
    "compiler optimization. Keep writing until the output limit."
)

LONG_CONTEXT_CHECKS = (
    "context",
    "shared_cohort_priming",
    "shared_warm_requests",
    "context_bound",
)


# Distinct batch-check subjects; one per lane of the qualifier's width-4 load.
BATCH_SUBJECTS = ("a compiler", "a refrigerator", "a jet engine", "photosynthesis")


def validate_context_matrix_delegation(path, context_cap):
    """Bind deferred long-context coverage to a runnable thermal matrix.

    Deferral is deliberately fail closed: a caller must supply the generated
    matrix manifest, and its context suite must reach the candidate server's
    near-limit domain with thermal admission and repeated measurements.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "mlx2.qualification-matrix.v1":
        raise AssertionError("long-context delegation requires a qualification matrix manifest")
    thermal = manifest.get("thermal", {})
    if (int(thermal.get("consecutive_samples", 0)) < 2
            or float(thermal.get("max_wait_seconds", 0)) <= 0):
        raise AssertionError("long-context delegation requires thermal admission control")
    eligible = []
    for model in manifest.get("models", []):
        context = model.get("context", {})
        contexts = model.get("contexts", [])
        tokens = [int(row.get("tokens", 0)) for row in contexts]
        generated = bool(contexts)
        for row in contexts:
            prompt_path = Path(str(row.get("prompt_path", "")))
            if not prompt_path.is_absolute():
                prompt_path = path.parent / prompt_path
            generated = (
                generated and prompt_path.is_file()
                and row.get("prompt_sha256")
                == hashlib.sha256(prompt_path.read_bytes()).hexdigest()
            )
        if (tokens and max(tokens) > near_limit_prompt_floor(context_cap)
                and int(context.get("runs_per_cell", 0)) >= 3
                and int(context.get("max_tokens", 0)) >= LONG_CONTEXT_COMPLETION_TOKENS
                and generated and model.get("arms")):
            eligible.append({"model": model.get("name"), "contexts": tokens,
                             "runs_per_cell": context["runs_per_cell"]})
    if not eligible:
        raise AssertionError(
            "long-context delegation manifest has no generated, repeated near-limit context suite"
        )
    return {
        "mode": "thermally_controlled_context_matrix",
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "delegated_checks": list(LONG_CONTEXT_CHECKS),
        "eligible_models": eligible,
    }


def thinking_disabled(body):
    """The request turns reasoning off in any of the accepted spellings."""
    return (
        body.get("enable_thinking") is False
        or body.get("think") is False
        or str(body.get("reasoning_effort", "")).lower() == "none"
    )


def with_thinking_budget(body, thinking_default, allowance=THINKING_BUDGET_TOKENS):
    """Add the reasoning allowance to a request that will actually think.

    Requests that disable thinking keep their exact budget (the near-limit
    context checks depend on it), and so does every request to a server that
    does not think by default: its reasoning probes already escalate their own
    budgets.
    """
    if not thinking_default or thinking_disabled(body) or "max_tokens" not in body:
        return body
    budget = int(body["max_tokens"])
    return {**body, "max_tokens": max(budget, min(budget + int(allowance), SERVER_MAX_TOKENS))}


def near_limit_prompt_floor(context_cap):
    """Lowest accepted rendered prompt size for the near-limit domain."""
    context_cap = int(context_cap)
    if context_cap <= LONG_CONTEXT_HEADROOM:
        raise ValueError("context cap must exceed near-limit headroom")
    return context_cap - LONG_CONTEXT_HEADROOM


def cancellation_probe_budget(context_cap):
    """Long cancellation budget that remains legal for the served context."""
    budget = min(
        CANCELLATION_PROBE_MAX_TOKENS,
        int(context_cap) - LONG_CONTEXT_HEADROOM,
    )
    if budget < LONG_CONTEXT_COMPLETION_TOKENS:
        raise ValueError("context cap is too small for the cancellation probe")
    return budget


# Exactly one token each, with a leading space, in every supported tokenizer
# (Qwen3.x/Qwen4, Muse-Glimmer, North/Cohere): verified by
# tests/test_qualify_serving_receipts.py when the artifacts are present.
LONG_CONTEXT_WORDS = (
    "data the system value table record signal network window object format "
    "memory number simple paper river stone garden market winter summer letter "
    "music color train bridge forest island doctor teacher kitchen engine planet "
    "silver orange yellow mountain village morning evening history science future "
    "picture water light house story world money power place point group"
).split()


def long_context_filler(tokens):
    """``tokens`` one-token words in a fixed pseudo-random order.

    A single repeated token is degenerate input: North-Mini-Code collapses to
    an immediate end-of-turn on ~30K copies of ``data`` (GPU, 2026-09-18), which
    says nothing about serving.  The order is a fixed LCG so every arm and every
    repeat sees the same bytes and warm-prefix checks still hold.
    """
    state, words = 20260918, []
    for _ in range(int(tokens)):
        state = (state * 1103515245 + 12345) % (1 << 31)
        words.append(LONG_CONTEXT_WORDS[(state >> 8) % len(LONG_CONTEXT_WORDS)])
    if words:
        # The one word with no leading space; ``data`` is a single token bare.
        words[0] = LONG_CONTEXT_WORDS[0]
    return " ".join(words)


def long_context_prompt(context_cap):
    """Build the shared near-limit probe from one-token ASCII words."""
    return long_context_filler(near_limit_prompt_floor(context_cap)) + LONG_CONTEXT_INSTRUCTION


def long_context_answer_passes(text):
    """The reply shows the model read the instruction at the far end.

    Obedient models open with the marker.  A model that prefaces its answer
    (North restates the request first) cannot reach the marker inside the
    64-token budget; naming the requested topic is the same evidence that the
    tail of the near-limit prompt was attended.  So is the marker anywhere in
    the reply: it occurs only in that instruction.  North's first answer token
    on this prompt is a near tie (Arabic, "Here" and "data" within 0.13 nats),
    and a warm B2 cohort answers with an Arabic rendering of the instruction
    that quotes LONG_READY but names no English topic (GPU, 2026-09-23).
    """
    text = text.strip()
    return "LONG_READY" in text or "compiler" in text.lower()


REPLY_EVIDENCE_TEXT_CHARS = 2000


def _truncate_evidence_text(text, limit=REPLY_EVIDENCE_TEXT_CHARS):
    if not isinstance(text, str) or len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def reply_evidence(response, limit=REPLY_EVIDENCE_TEXT_CHARS):
    """Diagnosable evidence for one chat reply: text, usage and receipt.

    A check that records only the ``mlx2`` receipt cannot say why an answer
    check failed once the reply text is gone.  Refusal entries pass through
    unchanged.  Never raises: a malformed reply is recorded as it came.
    """
    if not isinstance(response, dict) or "refused" in response:
        return response
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        choice, message = {}, {}
    evidence = {
        "content": _truncate_evidence_text(
            (message.get("content") or "").strip(), limit
        ),
    }
    reasoning = message.get("reasoning_content")
    if reasoning:
        evidence["reasoning_content"] = _truncate_evidence_text(reasoning, limit)
    evidence["finish_reason"] = choice.get("finish_reason")
    evidence["usage"] = response.get("usage")
    evidence["mlx2"] = response.get("mlx2")
    return evidence


def near_limit_usage_passes(
    usage, context_cap, completion_budget=LONG_CONTEXT_COMPLETION_TOKENS
):
    """Require a full bounded completion from a genuinely near-limit prompt."""
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return (
        type(prompt_tokens) is int
        and type(completion_tokens) is int
        and prompt_tokens > near_limit_prompt_floor(context_cap)
        and completion_tokens == int(completion_budget)
        and prompt_tokens + completion_tokens <= int(context_cap)
    )


def http_refusal(error):
    """Status and error body of a refused request, for check evidence."""
    try:
        body = json.loads(error.read() or b"{}")
    except (OSError, ValueError):
        body = {}
    return {"status": error.code, "error": body.get("error", body)}


def shared_qsa_completion_budget(settings, context_cap):
    """Choose a cohort budget eligible for the selected shared-QSA policy.

    The runtime's automatic crossover admits at most one output token per
    1024 cached base tokens, capped by the configured maximum. Qualification
    must exercise that selected domain instead of demanding engagement from a
    request the policy is required to decline.
    """
    environment = settings.get("environment", {})
    mode = str(environment.get("MLX_LM_SHARED_QSA_SUFFIX", "")).strip().lower()
    if not settings.get("mtp") or mode != "auto":
        return LONG_CONTEXT_COMPLETION_TOKENS
    maximum = int(environment.get("MLX_LM_SHARED_QSA_SUFFIX_MAX_REMAINING", "64"))
    base_tokens = near_limit_prompt_floor(context_cap)
    minimum = int(environment.get("MLX_LM_SHARED_QSA_SUFFIX_MIN_CONTEXT", "16380"))
    if base_tokens < minimum:
        return LONG_CONTEXT_COMPLETION_TOKENS
    cutoff = min(maximum, (base_tokens + 1023) // 1024)
    if cutoff < 1:
        raise AssertionError("shared-QSA auto policy has no positive output budget")
    return min(LONG_CONTEXT_COMPLETION_TOKENS, cutoff)


def reasoning_response_passes(response):
    """Require a real reasoning channel followed by a distinct final answer."""
    try:
        choice = response["choices"][0]
        message = choice["message"]
        reasoning = message.get("reasoning_content")
        answer = message.get("content")
    except (KeyError, IndexError, TypeError):
        return False
    return (
        isinstance(reasoning, str)
        and bool(reasoning.strip())
        and isinstance(answer, str)
        and bool(answer.strip())
        and re.search(r"(?<!\d)221(?!\d)", answer) is not None
    )


def run_reasoning_probe(post, *, thinking_budget=None):
    """Retry only when the output cap interrupts reasoning before final content."""
    attempts = []
    for max_tokens in REASONING_PROBE_TOKEN_BUDGETS:
        request = {
            "messages": [
                {"role": "system", "content": REASONING_PROBE_SYSTEM},
                {
                    "role": "user",
                    "content": "Calculate 13 times 17. Reply with the integer.",
                },
            ],
            "enable_thinking": True,
            "max_tokens": max_tokens,
        }
        if thinking_budget is not None:
            request["thinking_budget"] = int(thinking_budget)
        response = post(request)
        attempts.append(response)
        if reasoning_response_passes(response):
            break
        choices = response.get("choices") if isinstance(response, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {})
        cap_interrupted_transition = (
            choice.get("finish_reason") == "length"
            and isinstance(message.get("reasoning_content"), str)
            and bool(message["reasoning_content"].strip())
            and not str(message.get("content") or "").strip()
        )
        if not cap_interrupted_transition:
            break
    return attempts[-1], attempts


def run_structured_thinking_probe(post, status):
    """Thinking-enabled chat with ``json_object``: the grammar defers past </think>.

    Passes iff the server answers 200, ``content`` parses as a JSON object and
    the receipt shows ``deferred: true``.  A 400 "requires thinking to be
    disabled" is ``skipped_unsupported``: it passes only when the adapter does
    not declare a thinking-close marker (``/v1/status`` structured_output
    capability; an older server without that block passes with a note).
    """
    declared = (status.get("structured_output") or {}).get("thinking_deferral")
    attempts = []
    for max_tokens in REASONING_PROBE_TOKEN_BUDGETS:
        try:
            request = {
                "messages": [
                    {"role": "system", "content": REASONING_PROBE_SYSTEM},
                    {
                        "role": "user",
                        "content": "Calculate 13 times 17. Return a JSON object "
                        "with the key answer set to the integer.",
                    },
                ],
                "enable_thinking": True,
                "response_format": {"type": "json_object"},
                "max_tokens": max_tokens,
            }
            if declared is True:
                request["thinking_budget"] = REASONING_PROBE_THINKING_BUDGET
            response = post(request)
        except HTTPError as error:
            message = error.read().decode(errors="replace")
            unsupported = error.code == 400 and "requires thinking to be disabled" in message
            return unsupported and declared is not True, {
                "outcome": "skipped_unsupported" if unsupported else "http_error",
                "http_status": error.code,
                "adapter_declares_marker": declared,
                "note": (
                    "status does not report structured_output capability; 400 accepted"
                    if unsupported and declared is None else None
                ),
                "error": message,
            }
        attempts.append(response)
        choice = response["choices"][0]
        message = choice.get("message", {})
        if not (
            choice.get("finish_reason") == "length"
            and not str(message.get("content") or "").strip()
        ):
            break
    response = attempts[-1]
    receipt = response["mlx2"]["request_controls"].get("structured_output") or {}
    try:
        value = json.loads(response["choices"][0]["message"]["content"])
    except (TypeError, ValueError, json.JSONDecodeError):
        value = None
    return isinstance(value, dict) and receipt.get("deferred") is True, {
        "outcome": "deferred" if receipt.get("deferred") is True else "not_deferred",
        "engine": receipt.get("engine"),
        "deferred_tokens": receipt.get("deferred_tokens"),
        "adapter_declares_marker": declared,
        "attempts": attempts,
    }


TOOLS_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
TOOLS_PROBE_REQUEST = {
    "messages": [{
        "role": "user",
        "content": "Use the weather tool to get the weather in Toronto.",
    }],
    "tools": TOOLS_PROBE_TOOLS,
    "max_tokens": 128,
}


class ToolsProbe:
    """Outcome of the ``tools`` check.

    ``status`` is "passed", "failed" or ``known_model_behaviour``;
    ``call_response`` is the response whose tool call the round trip
    continues from (the enforced re-ask under a known behaviour).
    """

    def __init__(self, status, evidence, request, call_response, known=None):
        self.status = status
        self.evidence = evidence
        self.request = request
        self.call_response = call_response
        self.known = known


def run_tools_probe(post, artifact):
    """Run the ``tools`` check (tool_choice auto).

    On failure, and only when this artifact declares a known model behaviour
    for ``tools`` (mlx2.known_model_behaviour), re-ask with an enforced tool
    choice and accept the result as known behaviour only if the pair matches
    the declared failure shape exactly.  Anything else fails as before.
    """
    from mlx2.known_model_behaviour import (
        KNOWN_MODEL_BEHAVIOUR_STATUS,
        TOOLS_FORCED_CHOICE,
        known_model_behaviour_for,
        match_shape,
        tools_check_passes,
    )

    request = json.loads(json.dumps(TOOLS_PROBE_REQUEST))
    response = post(request)
    if tools_check_passes(response):
        return ToolsProbe("passed", response, request, response)
    entry = known_model_behaviour_for("tools", artifact)
    if entry is None:
        return ToolsProbe("failed", response, request, response)
    forced = post({**json.loads(json.dumps(request)), "tool_choice": TOOLS_FORCED_CHOICE})
    evidence = {"auto": response, "forced": forced}
    matched, reasons = match_shape(entry.shape, evidence)
    if not matched:
        evidence["known_model_behaviour_rejected"] = {"id": entry.id, "reasons": reasons}
        return ToolsProbe("failed", evidence, request, response)
    return ToolsProbe(
        KNOWN_MODEL_BEHAVIOUR_STATUS, evidence, request, forced,
        known=entry.record(artifact),
    )


def record_tools_probe(report, probe):
    """Record the ``tools`` check; raise (stop the run) unless it passed or
    matched its declared known model behaviour.  A known behaviour is never
    recorded as passed; the receipt names the exception it relied on."""
    from mlx2.known_model_behaviour import KNOWN_MODEL_BEHAVIOUR_STATUS

    if probe.status == KNOWN_MODEL_BEHAVIOUR_STATUS:
        report["checks"]["tools"] = {
            "passed": False,
            "status": KNOWN_MODEL_BEHAVIOUR_STATUS,
            "known_model_behaviour": probe.known,
            "evidence": probe.evidence,
        }
        report.setdefault("known_model_behaviour", {})["tools"] = probe.known
        print(f"tools: KNOWN MODEL BEHAVIOUR ({probe.known['id']})", flush=True)
        return
    passed = probe.status == "passed"
    report["checks"]["tools"] = {"passed": passed, "evidence": probe.evidence}
    print(f"tools: {'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        raise AssertionError("tools")


def selected_capability_scope(status):
    """Trust only the selected route's capabilities, cross-checked at status."""
    selected = status.get("selected_capabilities")
    implemented = status.get("implemented_capabilities")
    active = status.get("capabilities")
    if any(not isinstance(value, list) or any(type(name) is not str for name in value)
           for value in (selected, implemented, active)):
        return set(), False
    selected_set, implemented_set, active_set = map(set, (selected, implemented, active))
    valid = (len(selected_set) == len(selected)
             and selected_set <= implemented_set
             and selected_set == active_set)
    return selected_set, valid


def read_adapter_qualification(path, status):
    """Embed a source-bound media companion only for the same live route."""
    from mlx2.qualification import validate_adapter_qualification

    report = json.loads(Path(path).read_text())
    validate_adapter_qualification(
        report, runtime=status.get("runtime"), artifact=status.get("artifact"),
        settings=status.get("settings"),
    )
    return report


def source_text_cases(companion):
    """Use only a validated Smol companion's direct-source text witnesses."""
    if not isinstance(companion, dict) or "text_source" not in companion:
        return None
    if companion.get("model_type") != "smolvlm":
        raise ValueError("source-text HTTP semantics require the approved Smol producer")
    text_source = companion["text_source"]
    if not isinstance(text_source, dict) or not isinstance(text_source.get("cases"), dict):
        raise ValueError("source-text cases are missing")
    return text_source["cases"]


def source_token_response_matches(response, case, *, prompt_text):
    """Check HTTP token output against a source-bound greedy witness."""
    if not isinstance(response, dict) or not isinstance(case, dict):
        return False
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return False
    choice = choices[0]
    if not isinstance(choice, dict):
        return False
    entries = (choice.get("logprobs") or {}).get("content")
    expected = case.get("generated_token_ids")
    receipt = response.get("mlx2") or {}
    controls = receipt.get("request_controls") or {}
    usage = response.get("usage") or {}
    budget = case.get("max_tokens")
    return (case.get("prompt") == prompt_text
            and type(budget) is int and budget in (16, 64)
            and isinstance(expected, list) and len(expected) == budget
            and all(type(token) is int and token >= 0 for token in expected)
            and isinstance(entries, list)
            and [entry.get("id") if isinstance(entry, dict) else None
                 for entry in entries] == expected
            and usage.get("prompt_tokens") == case.get("prompt_tokens")
            and usage.get("completion_tokens") == len(expected)
            and choice.get("finish_reason") == "length"
            and isinstance(choice.get("message", {}).get("content"), str)
            and bool(choice["message"]["content"])
            and receipt.get("route") == "ordinary"
            and receipt.get("qualification") == "candidate"
            and receipt.get("cache") == "apcv2"
            and controls.get("max_tokens") == budget
            and controls.get("min_tokens") == case.get("min_tokens", 0)
            and controls.get("sampling") == case.get("sampling"))


def interior_stop_witness(text):
    """Choose a unique noninitial substring from a verified source output."""
    if not isinstance(text, str) or len(text) < 12:
        return None
    middle = len(text) // 2
    for width in (10, 8, 6, 4):
        for start in range(middle, max(1, len(text) - width)):
            fragment = text[start:start + width]
            if fragment and text.count(fragment) == 1 and text[:start].strip():
                return fragment, text[:start]
    return None


def qualification_harness_identity(path=None):
    source = Path(__file__) if path is None else Path(path)
    return {
        "schema": QUALIFICATION_HARNESS_SCHEMA,
        "name": "scripts/qualify_serving.py",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


# Files whose content a preflight's pass depends on besides the runtime
# (src/), the harness and the pytest configuration, which the binding covers:
# everything under tests/, scripts/ (tests assert on scripts/fixtures/*.json
# through the scripts that load them, and glob scripts/*.py), provenance/
# (records and the NOTICE files a qualifier hashes), native/ and
# qualification/ (plans, policies, top-level records, committed artifacts a
# test round-trips, corpora, experiments, history, receipts), plus the
# experiment companions and docs/PROVENANCE.md that tests read.  Campaign
# evidence under qualification/runs/ is left out by design: a campaign
# writes there and must not invalidate the receipt; the few tests that
# assert on frozen run evidence are outside the binding.
PREFLIGHT_TREE_PATTERNS = (
    "tests/**/*", "scripts/**/*", "provenance/**/*", "native/**/*", "qualification/**/*",
    "docs/experiments/**/*", "docs/PROVENANCE.md",
)
PREFLIGHT_TREE_EXCLUDED_PREFIXES = ("qualification/runs/",)
# Identity fields a preflight receipt is bound to.  The git revision is
# recorded but not bound: a commit that changes no bound file (docs, campaign
# evidence) leaves every earlier test result valid, and a commit that changes
# tests or scripts is caught file by file through the tree.
PREFLIGHT_BINDING_KEYS = ("runtime", "qualification_harness", "pytest_config_sha256")


def preflight_tree(root=None):
    """Per-file SHA-256 of PREFLIGHT_TREE_PATTERNS (relative paths)."""
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    tree = {}
    for pattern in PREFLIGHT_TREE_PATTERNS:
        for path in sorted(root.glob(pattern)):
            relative = str(path.relative_to(root))
            if (path.is_file() and "__pycache__" not in path.parts
                    and not relative.startswith(PREFLIGHT_TREE_EXCLUDED_PREFIXES)):
                tree[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return tree


def preflight_tree_sha256(tree):
    """Aggregate digest of a per-file tree (the identity's test_source_sha256)."""
    return hashlib.sha256(json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preflight_binding(identity):
    return {key: identity.get(key) for key in PREFLIGHT_BINDING_KEYS}


def preflight_tree_changes(before, after):
    """``{"added", "modified", "removed"}`` relative paths between two trees."""
    return {
        "added": sorted(set(after) - set(before)),
        "modified": sorted(p for p in set(before) & set(after) if before[p] != after[p]),
        "removed": sorted(set(before) - set(after)),
    }


def _is_test_module(path):
    return path.startswith("tests/") and Path(path).name.startswith("test_") and path.endswith(".py")


def preflight_impacted_tests(changes, tree, root=None):
    """Test modules a change can reach: the transitive reverse closure of a
    lexical name scan (over-selecting).

    Every touched path (added, modified or removed) is a search name: the
    module stem for a ``.py`` file (tests load scripts by path and helpers by
    import), the file name otherwise.  Every ``.py`` file in the tree -- a
    test module, a test helper or a script -- whose text names it is a
    consumer: a consumer that is a test module is selected, and every
    consumer is a search name in turn, so helper->helper->test and
    script->script->test chains are followed to a fixed point.  A module
    that walks a bound directory (glob, rglob, iterdir, walk, listdir or
    scandir, with the directory's name on the call's line or the two before
    it) reads files under it without naming one, so it is a consumer of each
    touched file there whose name matches a literal glob pattern it uses (or
    of every file, for a pattern-less or non-literal walk).
    Changed or added test modules also run themselves.

    Fails closed (the caller needs the full preflight) when a touched file
    that is not a test module reaches no test module through any chain --
    the scan is lexical, so a load it cannot see must not pass with no test
    run -- and when a ``conftest.py`` is touched, since that reaches every
    test.  Both the writer and the validator call this, so a delta that
    skipped either is refused.  ``src/`` is not here: it is bound through
    the runtime identity, so a change there needs the full suite."""
    import fnmatch
    import re

    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    touched = changes["added"] + changes["modified"] + changes["removed"]
    conftests = sorted(p for p in touched if Path(p).name == "conftest.py")
    if conftests:
        raise AssertionError(
            f"preflight delta: {', '.join(conftests)} changed and reaches every test: "
            "rerun the full preflight"
        )
    modules = sorted(p for p in tree if p.endswith(".py"))
    tests = {p for p in modules if _is_test_module(p)}
    texts = {p: (root / p).read_text(errors="replace") for p in modules}
    words = {p: set(re.findall(r"\w+", text)) for p, text in texts.items()}
    # Directory walkers, per top-level directory named (case-insensitively, so
    # a SCRIPTS constant counts) on the walk call's line or the two before
    # it: the basename pattern of a literal glob/rglob argument (a "*.py"
    # rglob reads only .py files), or None for a non-literal pattern or a
    # pattern-less walk, which reads everything.  ast.walk is a tree walk,
    # not a directory walk.
    walk = re.compile(r"(?<!ast\.)\b(r?glob|iterdir|walk|listdir|scandir)\s*\(\s*(?:(['\"])([^'\"]*)\2)?")
    tops = {p.split("/", 1)[0] for p in tree}
    top_words = {top: re.compile(r"(?<![\w])" + re.escape(top) + r"(?![\w])", re.I) for top in tops}
    walkers = {}
    for p in modules:
        lines = texts[p].splitlines()
        for index, line in enumerate(lines):
            for call, _, literal in walk.findall(line):
                window = "\n".join(lines[max(0, index - 2):index + 1])
                pattern = Path(literal).name if call in ("glob", "rglob") and literal else None
                for top, word in top_words.items():
                    if word.search(window):
                        walkers.setdefault(p, {}).setdefault(top, set()).add(pattern)

    def directory_consumers(path):
        top, name = path.split("/", 1)[0], Path(path).name
        return {p for p, by_top in walkers.items()
                if p != path and top in by_top
                and any(pattern is None or fnmatch.fnmatch(name, pattern) for pattern in by_top[top])}

    consumers_of = {}

    def consumers(path):
        if path not in consumers_of:
            name = Path(path).stem if path.endswith(".py") else Path(path).name
            if re.fullmatch(r"\w+", name):
                hits = {p for p in modules if p != path and name in words[p]}
            else:
                pattern = re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])")
                hits = {p for p in modules if p != path and pattern.search(texts[p])}
            consumers_of[path] = hits
        return consumers_of[path]

    selected = {p for p in changes["added"] + changes["modified"] if p in tests}
    unreached = []
    for start in touched:
        reached, frontier, seen = set(), [start], set()
        while frontier:
            path = frontier.pop()
            if path in seen:
                continue
            seen.add(path)
            hits = consumers(path)
            if path == start:
                hits = hits | directory_consumers(path)
            reached.update(hits & tests)
            frontier.extend(sorted(hits - seen))
        if not reached and not _is_test_module(start):
            unreached.append(start)
        selected.update(reached)
    if unreached:
        raise AssertionError(
            "preflight delta cannot reach any test module from "
            f"{', '.join(sorted(unreached))}: rerun the full preflight"
        )
    return sorted(selected)


INTERPRETER_IDENTITY_KEYS = ("executable", "implementation", "version")


def interpreter_identity():
    """The interpreter this harness runs under, as invoked.

    A receipt binds it (integration review 2026-10-09): the arguments after
    a command's head say what pytest was asked, not that Python ran it, so
    the validators compare the head and this identity against the trusted
    local values, never against the receipt alone."""
    return {
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "version": "%d.%d.%d" % sys.version_info[:3],
    }


def _canonical_executable(executable):
    """A path with symlinks resolved; malformed unless it names an existing
    file.  A venv's python, python3 and the binary they point at are one
    interpreter; a path that does not exist cannot have run anything."""
    if not isinstance(executable, str) or not executable:
        return _MALFORMED_COMMAND
    try:
        return os.path.realpath(executable, strict=True)
    except (OSError, RuntimeError):
        return _MALFORMED_COMMAND


def _well_formed_interpreter(identity):
    return (isinstance(identity, dict) and set(identity) == set(INTERPRETER_IDENTITY_KEYS)
            and all(isinstance(identity[key], str) and identity[key]
                    for key in INTERPRETER_IDENTITY_KEYS))


def _same_local_interpreter(identity, trusted):
    """``identity`` names the trusted local interpreter: same implementation
    and version, executables canonical-equal (symlinks resolved on both
    sides).  Only for interpreters on this host; a remote one is compared
    exactly, since its paths cannot be resolved here."""
    if (not _well_formed_interpreter(identity)
            or identity["implementation"] != trusted["implementation"]
            or identity["version"] != trusted["version"]):
        return False
    canonical = _canonical_executable(identity["executable"])
    return canonical is not _MALFORMED_COMMAND and canonical == _canonical_executable(
        trusted["executable"])


def _ran_under(command, trusted):
    """A commanded lane's head is the trusted local interpreter; an absent
    lane has nothing to prove."""
    if command is None:
        return True
    if _command_args(command) is _MALFORMED_COMMAND:
        return False
    canonical = _canonical_executable(command[0])
    return canonical is not _MALFORMED_COMMAND and canonical == _canonical_executable(
        trusted["executable"])


def _command_head(command):
    return command[0] if isinstance(command, list) and command else command


def preflight_identity(runtime_identity_fn=None):
    """Identity that lets a live qualifier trust an earlier full-suite run."""
    root = Path(__file__).resolve().parents[1]
    if runtime_identity_fn is None:
        from mlx2.serving import runtime_identity as runtime_identity_fn
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    if revision.returncode:
        raise RuntimeError(f"could not identify git revision: {revision.stderr.strip()}")
    return {
        "git": {"revision": revision.stdout.strip()},
        "runtime": runtime_identity_fn(),
        "qualification_harness": qualification_harness_identity(),
        "interpreter": interpreter_identity(),
        # The whole preflight tree, not only tests/**/*.py: the cross-host (v3)
        # receipt binds the identity alone.  The key keeps its name for that schema.
        "test_source_sha256": preflight_tree_sha256(preflight_tree(root)),
        "pytest_config_sha256": hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest(),
    }


FULL_SUITE_PYTEST_ARGS = ("-m", "pytest")
# These tests deliberately reject a real MLX import.  tests/conftest.py imports
# MLX to put ordinary tensor tests on the CPU, so the guards need a fresh
# interpreter without that conftest.  Every listed module is still required:
# the ordinary run excludes exactly these files and the guard run includes
# exactly these files.  A new guard must be added here before it can qualify.
PREFLIGHT_IMPORT_GUARD_MODULES = (
    "tests/test_agnes_hy_v3_adapters.py",
    "tests/test_continuation_pool_metal_cli.py",
    "tests/test_deepseek_v4_candidate_cpu.py",
    "tests/test_deepseek_v4_vision_layout_cpu.py",
    "tests/test_diffusion_gemma_cpu.py",
    "tests/test_flash_next_vlm_cpu.py",
    "tests/test_glm52_dsa_candidate_cpu.py",
    "tests/test_granite_hils_cpu.py",
    "tests/test_hybrid_packed_geometry_gate_cpu.py",
    "tests/test_hybrid_packed_prefill_finite_cpu.py",
    "tests/test_laguna_s21_adapter.py",
    "tests/test_lfm25_dspark_compat_cpu.py",
    "tests/test_lfm25_dspark_eval_contract_cpu.py",
    "tests/test_lfm25_dspark_stats_cpu.py",
    "tests/test_lfm25_fused_shortconv_cpu.py",
    "tests/test_lfm25_hybrid_persistence_cpu.py",
    "tests/test_lfm25_vl_cpu.py",
    "tests/test_llada_denoising_cpu.py",
    "tests/test_long_affine_diagnostics_cpu.py",
    "tests/test_long_affine_plan_cpu.py",
    "tests/test_long_b2_cap20_http_smoke_cpu.py",
    "tests/test_mechanism_intake_merge_cpu.py",
    "tests/test_multimodal_prefill_contract_cpu.py",
    "tests/test_multimodal_registry_gate_cpu.py",
    "tests/test_muse_glimmer_vision_candidate_cpu.py",
    "tests/test_n20_attention_controlled_perf_cpu.py",
    "tests/test_native_mtp_head_probe_cpu.py",
    "tests/test_nemotron3_diarization_candidate_cpu.py",
    "tests/test_packed_gdn_evaluation_waves_cpu.py",
    "tests/test_packed_n20_admission_cpu.py",
    "tests/test_packed_n20_arena_admission_cpu.py",
    "tests/test_packed_prefill_composition_cpu.py",
    "tests/test_packed_prefill_http_receipt_cpu.py",
    "tests/test_packed_prefill_long_contract_cpu.py",
    "tests/test_packed_prefill_serving_cpu.py",
    "tests/test_paged_source_inventory_cpu.py",
    "tests/test_paired_direct_state_gate_cpu.py",
    "tests/test_parallel_serving_metal_cli.py",
    "tests/test_phi4mm_candidate_cpu.py",
    "tests/test_pinned_vlm_candidates_cpu.py",
    "tests/test_qsa_output_gate_vectors_cpu.py",
    "tests/test_qwen25_rope_cpu.py",
    "tests/test_qwen25_vision_grouped_cpu.py",
    "tests/test_qwen35_122b_cpu.py",
    "tests/test_qwen35_4b_adapter.py",
    "tests/test_qwen35_vlm_candidate_cpu.py",
    "tests/test_ragged_pld_evidence_cpu.py",
    "tests/test_ragged_pld_identity_cpu.py",
    "tests/test_ragged_pld_north_identity_cpu.py",
    "tests/test_ragged_pld_survivor_continuation_cpu.py",
    "tests/test_segmented_moe_native_identity_cpu.py",
    "tests/test_smolvlm2_apcv2_cpu.py",
    "tests/test_spomin400_control_wave_cpu.py",
    "tests/test_spomin400_http_contract_cpu.py",
    "tests/test_spomin400_init_order_cpu.py",
    "tests/test_spomin400_owned_clock_cpu.py",
    "tests/test_spomin400_phased_cpu.py",
    "tests/test_spomin400_sampling_contract_cpu.py",
    "tests/test_standard_bf16_metal_cli.py",
    "tests/test_standard_decoder_cpu.py",
    "tests/test_standard_projection_groups_metal_cli.py",
    "tests/test_varlen_n20_prompt_lookup_http_gate_cpu.py",
    "tests/test_vision_feature_reuse_cpu.py",
    "tests/test_xpress_metal_matrix_cli.py",
)


def preflight_test_commands(*, pytest_args=None):
    root = Path(__file__).resolve().parents[1]
    guards = PREFLIGHT_IMPORT_GUARD_MODULES
    if len(guards) != len(set(guards)) or not guards or any(
        not (root / module).is_file() for module in guards
    ):
        raise AssertionError("preflight import-guard manifest is missing or invalid")
    ordinary = [sys.executable, *FULL_SUITE_PYTEST_ARGS,
                *(f"--ignore={module}" for module in guards), *(pytest_args or [])]
    # One interpreter per guard module: module-level import poisons differ
    # (some refuse only mlx, some every mlx2 import) and break each other when
    # they share a process.  The runner lives in this pinned harness.
    guarded = [sys.executable, str(Path(__file__).resolve().relative_to(root)),
               "--run-import-guards", *guards]
    return ordinary, guarded


def run_import_guards(modules) -> int:
    """Run each import-guard module alone (fresh interpreter, no conftest)."""
    root = Path(__file__).resolve().parents[1]
    failed = []
    for module in modules:
        if module not in PREFLIGHT_IMPORT_GUARD_MODULES:
            print(f"refusing unlisted import-guard module {module}", flush=True)
            return 2
        print(f"=== import guard: {module}", flush=True)
        completed = subprocess.run(
            [sys.executable, *FULL_SUITE_PYTEST_ARGS, "-q", "--noconftest", module],
            cwd=root,
        )
        if completed.returncode != 0:
            failed.append(module)
    print(f"import guards: {len(modules) - len(failed)}/{len(modules)} modules passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


def write_preflight_receipt(path, *, pytest_args=None, run=subprocess.run,
                            identity_fn=preflight_identity, tree_fn=preflight_tree):
    command, guard_command = preflight_test_commands(pytest_args=pytest_args)
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
           "PYTEST_ADDOPTS": "", "PYTEST_PLUGINS": ""}
    external_sources = (
        ("MLX2_TEST_TENSORFOLD_OWNED_SOURCE", PREFLIGHT_TENSORFOLD_OWNED_SOURCE),
        (
            "MLX2_TEST_TENSORFOLD_OWNED_MLX_LM_SOURCE",
            PREFLIGHT_TENSORFOLD_OWNED_MLX_LM_SOURCE,
        ),
    )
    for name, source in external_sources:
        if source.exists():
            env.setdefault(name, str(source.resolve()))
    root = Path(__file__).resolve().parents[1]
    # Bind the source that was present before execution, not only whatever
    # happens to be on disk after a long suite completes.
    identity_before = json.loads(json.dumps(identity_fn()))
    tree_before = dict(tree_fn())
    completed = run(command, capture_output=True, text=True, env=env, cwd=root)
    guarded = run(guard_command, capture_output=True, text=True, env=env, cwd=root)
    output = completed.stdout + completed.stderr
    guard_output = guarded.stdout + guarded.stderr
    identity_after = identity_fn()
    tree_after = tree_fn()
    source_stable = identity_before == identity_after and tree_before == tree_after
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "passed": source_stable and _zero_returncode(completed.returncode) and _zero_returncode(guarded.returncode),
        "timestamp": time.time(),
        "identity": identity_after,
        "tree": tree_after,
        "source_stable": source_stable,
        "identity_before": identity_before,
        "tree_before": tree_before,
        "test_command": command,
        "returncode": completed.returncode,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output_tail": output[-16000:],
        "guard_test_command": guard_command,
        "guard_returncode": guarded.returncode,
        "guard_output_sha256": hashlib.sha256(guard_output.encode()).hexdigest(),
        "guard_output_tail": guard_output[-16000:],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not receipt["passed"]:
        raise AssertionError(f"full unit suite/import guards failed or source changed during testing; see {path}")
    return receipt


def _active_runtime_matches_build(active_runtime, build_runtime):
    """Full-suite proof stays build-wide; independently rebind the active route."""
    from mlx2.route_identity import recompute_runtime_identity
    try:
        return recompute_runtime_identity(active_runtime, build=build_runtime) == active_runtime
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _validate_full_preflight(path, active_runtime, expected):
    """A passing, unscoped full-suite receipt bound to ``expected``'s binding.

    Returns the receipt.  The tree is checked by the caller (a full receipt
    must match the current tree; a delta's base matches its own tree)."""
    path = Path(path)
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != PREFLIGHT_SCHEMA or receipt.get("passed") is not True:
        raise AssertionError(f"preflight receipt is absent or failed: {path}")
    if preflight_binding(receipt.get("identity") or {}) != preflight_binding(expected):
        raise AssertionError(
            "preflight receipt does not match the current runtime source, harness or "
            "pytest configuration: rerun the full preflight"
        )
    # The interpreter is compared against this process, not ``expected``: a
    # receipt may not vouch for the Python that wrote it.
    interpreter = interpreter_identity()
    if not _same_local_interpreter(receipt["identity"].get("interpreter"), interpreter):
        raise AssertionError(
            "preflight receipt was not written under this interpreter "
            f"({_describe_interpreter(interpreter)}): rerun the full preflight"
        )
    if not _active_runtime_matches_build(active_runtime, receipt["identity"].get("runtime")):
        raise AssertionError("preflight receipt runtime does not match active server")
    command = receipt.get("test_command")
    guard_command = receipt.get("guard_test_command")
    ordinary_args, guarded_args = preflight_test_commands()
    # --preflight-pytest-arg exists for scoped historical/control receipts.
    # Any extra argument can scope the run (--collect-only, -k, a test path,
    # --lf, --deselect, --ignore) and still exit 0 with passed=True, so only
    # the unscoped default command stands in for the unit_tests check.
    if (_command_args(command) != list(ordinary_args[1:])
            or _command_args(guard_command) != list(guarded_args[1:])
            or not _zero_returncode(receipt.get("returncode"))
            or not _zero_returncode(receipt.get("guard_returncode"))):
        raise AssertionError(
            "preflight receipt did not run the full unit suite: "
            f"{command!r} / {guard_command!r}: {path}"
        )
    # The arguments above say what pytest was asked; the head says who ran
    # it.  `true` with the same arguments exits 0 and runs no test.
    if not _ran_under(command, interpreter) or not _ran_under(guard_command, interpreter):
        raise AssertionError(
            "preflight receipt commands were not run by this interpreter "
            f"({interpreter['executable']}): {_command_head(command)!r} / "
            f"{_command_head(guard_command)!r}: {path}"
        )
    return receipt


def _describe_interpreter(identity):
    return f"{identity['implementation']} {identity['version']} at {identity['executable']}"


def _delta_test_commands(impacted):
    """Commands a delta runs: impacted ordinary modules in one pytest process,
    impacted import-guard modules through the pinned guard runner."""
    guards = set(PREFLIGHT_IMPORT_GUARD_MODULES)
    ordinary = [p for p in impacted if p not in guards]
    guarded = [p for p in impacted if p in guards]
    root = Path(__file__).resolve().parents[1]
    return (
        [sys.executable, *FULL_SUITE_PYTEST_ARGS, *ordinary] if ordinary else None,
        [sys.executable, str(Path(__file__).resolve().relative_to(root)),
         "--run-import-guards", *guarded] if guarded else None,
    )


def validate_preflight_receipt(path, active_runtime, *, identity_fn=preflight_identity,
                               tree_fn=preflight_tree):
    """Trust a full-suite receipt for the current tree, or a delta over one.

    A full receipt must match the current binding (runtime source, harness,
    pytest configuration) and the current tests/scripts tree.  A delta
    receipt names a full base with the same binding and covers exactly the
    tree changes since that base with the impacted tests it reran; both are
    recomputed here, so a delta that skipped an impacted module is refused."""
    path = Path(path)
    receipt = json.loads(path.read_text())
    expected = identity_fn()
    if receipt.get("schema") == PREFLIGHT_DELTA_SCHEMA:
        return _validate_preflight_delta(path, receipt, active_runtime, expected, tree_fn)
    full = _validate_full_preflight(path, active_runtime, expected)
    if "tree" in full:
        changes = preflight_tree_changes(full["tree"], tree_fn())
        if any(changes.values()):
            count = sum(len(v) for v in changes.values())
            raise AssertionError(
                f"preflight receipt predates {count} test/script change(s); write a "
                "delta with --preflight-delta --base-preflight on this receipt"
            )
    elif full["identity"].get("test_source_sha256") != expected.get("test_source_sha256"):
        # Receipts written before the per-file tree carry one aggregate digest.
        raise AssertionError("preflight receipt does not match the current test sources")
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "identity": expected, "test_command": full["test_command"],
            "guard_test_command": full["guard_test_command"], "passed": True}


def _validate_preflight_delta(path, receipt, active_runtime, expected, tree_fn):
    if receipt.get("passed") is not True:
        raise AssertionError(f"preflight delta receipt failed: {path}")
    base_ref = receipt.get("base") or {}
    base_path = Path(base_ref.get("path", ""))
    if not base_path.is_file() or hashlib.sha256(base_path.read_bytes()).hexdigest() != base_ref.get("sha256"):
        raise AssertionError("preflight delta base receipt is missing or changed")
    base = _validate_full_preflight(base_path, active_runtime, expected)
    if "tree" not in base:
        raise AssertionError("preflight delta base has no per-file tree; rerun the full preflight")
    if preflight_binding(receipt.get("identity") or {}) != preflight_binding(expected):
        raise AssertionError("preflight delta receipt does not match the current binding")
    interpreter = interpreter_identity()
    if not _same_local_interpreter(receipt["identity"].get("interpreter"), interpreter):
        raise AssertionError(
            "preflight delta receipt was not written under this interpreter "
            f"({_describe_interpreter(interpreter)}): rerun the full preflight"
        )
    current = tree_fn()
    if receipt.get("tree") != current:
        raise AssertionError("preflight delta receipt does not match the current tests/scripts")
    changes = preflight_tree_changes(base["tree"], current)
    if receipt.get("changes") != changes:
        raise AssertionError("preflight delta receipt does not cover the tree changes since its base")
    impacted = preflight_impacted_tests(changes, current)
    if receipt.get("impacted") != impacted:
        raise AssertionError("preflight delta receipt did not rerun every impacted test module")
    ordinary, guarded = _delta_test_commands(impacted)
    if (_command_args(receipt.get("test_command")) != _command_args(ordinary)
            or _command_args(receipt.get("guard_test_command")) != _command_args(guarded)
            or not _lane_passed(ordinary, receipt.get("returncode"))
            or not _lane_passed(guarded, receipt.get("guard_returncode"))):
        raise AssertionError("preflight delta receipt commands or results do not match its impacted tests")
    if (not _ran_under(receipt.get("test_command"), interpreter)
            or not _ran_under(receipt.get("guard_test_command"), interpreter)):
        raise AssertionError(
            "preflight delta receipt commands were not run by this interpreter "
            f"({interpreter['executable']}): {_command_head(receipt.get('test_command'))!r} / "
            f"{_command_head(receipt.get('guard_test_command'))!r}: {path}"
        )
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "identity": expected, "test_command": base["test_command"],
            "guard_test_command": base["guard_test_command"], "passed": True,
            "delta": {"base": base_ref, "changes": changes, "impacted": impacted}}


_MALFORMED_COMMAND = object()


def _command_args(command):
    """A command's arguments after its interpreter; None for an absent lane.

    Anything but a non-empty list headed by a non-empty string is malformed
    and equals nothing, so it never matches an expected command."""
    if command is None:
        return None
    if (not isinstance(command, list) or not command
            or not isinstance(command[0], str) or not command[0]):
        return _MALFORMED_COMMAND
    return list(command[1:])


def _zero_returncode(code):
    """Exactly int 0: JSON false and None compare equal to or pass as 0."""
    return type(code) is int and code == 0


def _lane_passed(command, code):
    """A lane with a command must show returncode 0; a lane without one must
    show None.  A missing code is no evidence the run happened."""
    return _zero_returncode(code) if command is not None else code is None


def write_preflight_delta(path, base_path, *, run=subprocess.run, identity_fn=preflight_identity,
                          tree_fn=preflight_tree):
    """Rerun only the tests a tests/scripts change can reach, over a full base.

    Refuses (full preflight needed) when the binding moved -- any change under
    src/, the mlx build, the harness or the pytest configuration -- when a
    conftest.py changed, or when a touched file reaches no test module."""
    base_path = Path(base_path).resolve()
    expected = identity_fn()
    base = _validate_full_preflight(base_path, expected.get("runtime"), expected)
    if "tree" not in base:
        raise AssertionError("base preflight has no per-file tree; rerun the full preflight")
    current = tree_fn()
    changes = preflight_tree_changes(base["tree"], current)
    impacted = preflight_impacted_tests(changes, current)  # refuses conftest/unreachable
    ordinary, guarded = _delta_test_commands(impacted)
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
           "PYTEST_ADDOPTS": "", "PYTEST_PLUGINS": ""}
    root = Path(__file__).resolve().parents[1]
    outputs, codes = [], []
    for command in (ordinary, guarded):
        if command is None:
            outputs.append("")
            codes.append(None)
            continue
        completed = run(command, capture_output=True, text=True, env=env, cwd=root)
        outputs.append(completed.stdout + completed.stderr)
        codes.append(completed.returncode)
    receipt = {
        "schema": PREFLIGHT_DELTA_SCHEMA,
        "passed": _lane_passed(ordinary, codes[0]) and _lane_passed(guarded, codes[1]),
        "timestamp": time.time(),
        "identity": expected,
        "base": {"path": str(base_path),
                 "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest()},
        "tree": current,
        "changes": changes,
        "impacted": impacted,
        "test_command": ordinary,
        "returncode": codes[0],
        "output_tail": outputs[0][-16000:],
        "guard_test_command": guarded,
        "guard_returncode": codes[1],
        "guard_output_tail": outputs[1][-16000:],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not receipt["passed"]:
        raise AssertionError(f"impacted tests failed; see {path}")
    return receipt


def _proof_sha256(proof):
    return hashlib.sha256(json.dumps(proof, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()


def _checked_node_results(results, expected, *, approved_skips=None):
    """Compare observed calls to an independently collected exact node set."""
    if not isinstance(results, list) or not expected:
        raise AssertionError("cross-host preflight has no structured test results")
    approved_skips = approved_skips or {}
    seen = {}
    for result in results:
        if not isinstance(result, dict) or set(result) - {"nodeid", "status", "reason"}:
            raise AssertionError("cross-host preflight has malformed test results")
        node = result.get("nodeid")
        if not isinstance(node, str) or not node.startswith("tests/") or node in seen:
            raise AssertionError("cross-host preflight has duplicate or invalid test node")
        status = result.get("status")
        if status == "passed" and "reason" not in result:
            pass
        elif (status == "skipped" and isinstance(result.get("reason"), str)
              and result["reason"] == approved_skips.get(node)):
            pass
        else:
            raise AssertionError("cross-host preflight has failed or unapproved test outcome")
        seen[node] = status
    if set(seen) != set(expected):
        raise AssertionError("cross-host preflight test coverage is incomplete or unexpected")
    return seen


def validate_cross_host_preflight_receipt(
    path, *, active_runtime, active_artifact, active_settings, active_host_id,
    expected_source_nodes, required_m3_nodes, target_command,
    artifact_dependent_nodes=(), approved_source_skips=None,
    identity_fn=preflight_identity, source_interpreter=None,
):
    """Validate opt-in M5 source plus M3 target proof; v2 remains unchanged.

    The caller must supply an independently collected, reviewed node inventory,
    the target command and the source host's interpreter identity
    (``source_interpreter``: executable as invoked, implementation, version,
    as ``interpreter_identity()`` prints it there).  The source proof's
    identity and command heads are compared with it exactly, since this host
    cannot resolve paths on the other one; the target proof's interpreter
    and command head must be this process's interpreter.  This pure
    validator is not wired to route selection: no model gains ``unit_tests``
    from a self-declared receipt.
    """
    raw = Path(path).read_bytes()
    bundle = json.loads(raw)
    if (not isinstance(bundle, dict)
            or bundle.get("schema") != CROSS_HOST_PREFLIGHT_SCHEMA
            or bundle.get("passed") is not True):
        raise AssertionError("cross-host preflight receipt is absent or failed")
    source, target = bundle.get("source"), bundle.get("target")
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise AssertionError("cross-host preflight needs source and target proofs")
    if (bundle.get("source_proof_sha256") != _proof_sha256(source)
            or bundle.get("target_proof_sha256") != _proof_sha256(target)):
        raise AssertionError("cross-host preflight proof digest mismatch")
    if not _well_formed_interpreter(source_interpreter):
        raise AssertionError(
            "cross-host preflight needs a trusted source interpreter identity "
            "(executable, implementation, version) supplied by the caller"
        )
    expected_identity = identity_fn()
    interpreter = interpreter_identity()
    source_identity, target_identity = source.get("identity"), target.get("identity")
    if (not isinstance(expected_identity, dict)
            or set(expected_identity) != {"git", "runtime", "qualification_harness",
                                          "test_source_sha256", "pytest_config_sha256",
                                          "interpreter"}
            or not isinstance(source_identity, dict)
            or not isinstance(target_identity, dict)
            or set(target_identity) != set(expected_identity)
            or any(target_identity[key] != value for key, value in expected_identity.items()
                   if key != "interpreter")
            or not _same_local_interpreter(target_identity.get("interpreter"), interpreter)
            or set(source_identity) != set(expected_identity)
            or any(source_identity[key] != value for key, value in expected_identity.items()
                   if key not in ("runtime", "interpreter"))
            or source_identity.get("interpreter") != source_interpreter
            or not isinstance(source_identity.get("runtime"), dict)
            or set(source_identity["runtime"]) != set(expected_identity["runtime"])
            or source_identity["runtime"].get("source_sha256") != expected_identity["runtime"].get("source_sha256")
            or not _active_runtime_matches_build(active_runtime, target_identity.get("runtime"))):
        raise AssertionError("cross-host preflight source or target identity mismatch")
    from mlx2.qualification import APPROVED_QUALIFICATION_HARNESS
    if expected_identity["qualification_harness"] != APPROVED_QUALIFICATION_HARNESS:
        raise AssertionError("cross-host preflight producer is not approved")
    if (not isinstance(source.get("host_id"), str) or not source["host_id"]
            or not isinstance(target.get("host_id"), str) or not target["host_id"]
            or not isinstance(active_host_id, str) or not active_host_id
            or target["host_id"] != active_host_id
            or source["host_id"] == target["host_id"]):
        raise AssertionError("cross-host preflight host identity mismatch")
    if (target.get("artifact") != active_artifact
            or target.get("settings") != active_settings):
        raise AssertionError("cross-host preflight target route mismatch")
    ordinary, guarded = preflight_test_commands()
    commands = {"ordinary": ordinary[1:], "guard": guarded[1:]}
    if (not isinstance(expected_source_nodes, dict)
            or set(expected_source_nodes) != set(commands)
            or any(not isinstance(nodes, (set, frozenset)) or not nodes
                   for nodes in expected_source_nodes.values())
            or set(expected_source_nodes["ordinary"]) & set(expected_source_nodes["guard"])):
        raise AssertionError("cross-host preflight needs an exact source inventory")
    suites = source.get("suites")
    if not isinstance(suites, dict) or set(suites) != set(commands):
        raise AssertionError("cross-host preflight source partition is incomplete")
    approved_source_skips = approved_source_skips or {}
    if (not isinstance(approved_source_skips, dict)
            or not set(approved_source_skips) <= set().union(*expected_source_nodes.values())):
        raise AssertionError("cross-host preflight skip policy is invalid")
    observed = {}
    for name, command in commands.items():
        suite = suites[name]
        if (not isinstance(suite, dict) or type(suite.get("returncode")) is not int
                or suite["returncode"] != 0
                or not isinstance(suite.get("command"), list)
                or not suite["command"] or not isinstance(suite["command"][0], str)
                or not suite["command"][0]
                or suite["command"][1:] != command):
            raise AssertionError("cross-host preflight source command failed or was scoped")
        if suite["command"][0] != source_interpreter["executable"]:
            raise AssertionError(
                "cross-host preflight source command was not run by the trusted "
                f"source interpreter ({source_interpreter['executable']}): "
                f"{suite['command'][0]!r}"
            )
        observed.update(_checked_node_results(
            suite.get("results"), expected_source_nodes[name],
            approved_skips=approved_source_skips,
        ))
    artifact_nodes = set(artifact_dependent_nodes)
    if not artifact_nodes or not artifact_nodes <= set(observed) or any(
        observed[node] != "passed" for node in artifact_nodes
    ):
        raise AssertionError("cross-host preflight waived an artifact-dependent test")
    artifacts = source.get("artifact_evidence")
    if (not isinstance(artifacts, dict) or set(artifacts) != artifact_nodes
            or any(not isinstance(value, str) or
                   re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in artifacts.values())):
        raise AssertionError("cross-host preflight artifact evidence is incomplete")
    required = set(required_m3_nodes)
    if (not required or not required <= set(observed)
            or not isinstance(target_command, (list, tuple)) or not target_command
            or not isinstance(target_command[0], str) or not target_command[0]):
        raise AssertionError("cross-host preflight M3 coverage policy is missing")
    if not _ran_under(list(target_command), interpreter):
        raise AssertionError(
            "cross-host preflight target command was not run by this interpreter "
            f"({interpreter['executable']}): {target_command[0]!r}"
        )
    if (type(target.get("returncode")) is not int or target["returncode"] != 0
            or target.get("command") != list(target_command)
            or not isinstance(target.get("command"), list)):
        raise AssertionError("cross-host preflight target command failed or was scoped")
    _checked_node_results(target.get("results"), required)
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "schema": CROSS_HOST_PREFLIGHT_SCHEMA, "passed": True,
            "source_host": source["host_id"], "target_host": target["host_id"],
            "source_nodes": len(observed), "target_nodes": len(required),
            "runtime": active_runtime, "artifact": active_artifact,
            "settings": active_settings}


# The concurrent warm pair must finish in less than this fraction of the same
# pair run one after the other.  No noise allowance above 1.0: across the 25
# committed receipts (qualification/, 2026-09-18..20) the worst ratio was
# 0.81 (Xing x6 PLD) and the rest 0.61-0.77, and the sequential reference is
# the pair's first serve, so noise already favours the concurrent pair.  A
# pair that is merely serialized (ratio ~1.0) fails.
MIXED_WARM_MAX_RATIO = 1.0


# A route that keeps the per-lane driver (prompt lookup over hybrid caches:
# every receipt reports width 1) cannot beat its sequential pair; its evidence
# is that the two lanes overlapped at no more than this timing-noise margin.
MIXED_WARM_PER_LANE_MAX_RATIO = 1.10


# Share of each lane's streamed tokens that must arrive while the other lane
# is also still producing.  A serialized pair scores 0, a lane given one
# early token and then stalled until the other finished scores about 1 of
# its 64, and lanes a per-lane driver interleaves score near 1.  A quarter
# sits far from both ends, and still admits a lane that joined as late as
# three quarters of the way through the other lane's decode.
MIXED_WARM_MIN_OVERLAP_SHARE = 0.25


def mixed_warm_timing_passes(
    concurrent_seconds, sequential_seconds, receipts=(), lane_token_seconds=None
):
    """Judge the concurrent warm pair against its own sequential warm-up.

    The former ``max(30.0, sequential)`` floor let any pair under 30 s pass
    regardless of the reference, and every non-Xing model finishes its
    sequential pair in 1-21 s, so the comparison never applied to them.

    Batching routes must beat the sequential pair.  A per-lane route, whose
    receipts all report width 1, shows concurrency the way the batch check
    accepts it: both lanes ran for most of the concurrent window, their
    tokens arrived interleaved (see ``mixed_warm_lanes_overlapped``), and
    the pair was not slower than sequential beyond noise.
    """
    widths = [width for receipt in receipts for width in observed_compute_widths(receipt)]
    if widths and max(widths) == 1:
        ran_most_of_window = all(
            float(receipt.get("elapsed_seconds") or 0.0) > 0.5 * concurrent_seconds
            for receipt in receipts
        )
        return (
            ran_most_of_window
            and mixed_warm_lanes_overlapped(lane_token_seconds)
            and concurrent_seconds <= sequential_seconds * MIXED_WARM_PER_LANE_MAX_RATIO
        )
    return concurrent_seconds < sequential_seconds * MIXED_WARM_MAX_RATIO


def mixed_warm_lane_overlap_shares(lane_token_seconds):
    """Each lane's share of its tokens that arrived while every other produced.

    ``lane_token_seconds`` holds, per lane, the client arrival instant of each
    streamed token delta on the pair's shared clock.  Another lane is
    producing from its first delta to its last, so a lane's share counts its
    deltas inside every other lane's span.  Returns None without at least
    two lanes of finite instants.
    """
    if lane_token_seconds is None or len(lane_token_seconds) < 2:
        return None
    lanes = []
    for arrivals in lane_token_seconds:
        if not arrivals or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in arrivals
        ):
            return None
        lanes.append([float(value) for value in arrivals])
    shares = []
    for index, arrivals in enumerate(lanes):
        spans = [
            (min(other), max(other))
            for position, other in enumerate(lanes)
            if position != index
        ]
        inside = sum(
            all(first <= instant <= last for first, last in spans)
            for instant in arrivals
        )
        shares.append(inside / len(arrivals))
    return shares


def mixed_warm_lanes_overlapped(lane_token_seconds):
    """Whether every lane got a material share of its tokens during the others.

    Receipt times (``ttft_seconds`` to the first token, ``elapsed_seconds``
    to the end) cannot show overlap: they imply continuous service from
    first token to finish, so a lane given one early token, then stalled
    while the other lane ran to completion, then run alone looked
    overlapped.  Only the arrival instants of the tokens themselves show
    whether both lanes were served over the same stretch of time.
    """
    shares = mixed_warm_lane_overlap_shares(lane_token_seconds)
    return shares is not None and all(
        share >= MIXED_WARM_MIN_OVERLAP_SHARE for share in shares
    )


def read_streamed_chat(lines, clock):
    """Rebuild a streamed chat response and stamp each token delta's arrival.

    ``lines`` yields the SSE wire lines as they arrive and ``clock`` reads
    the shared clock, so each stamp is the client arrival instant of one
    delta: one engine emission, a token or, on a speculative route, several.
    The finish chunk carries the usage and the ``mlx2`` receipt of the
    non-streamed response.  A stream that ends without it raises.
    """
    arrivals, parts, final, last = [], [], None, None
    for raw in lines:
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data: {"):
            continue
        instant = clock()
        last = chunk = json.loads(line[6:])
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                arrivals.append(instant)
            parts.append(delta.get("content") or "")
        if "mlx2" in chunk:
            final = chunk
    if final is None:
        raise AssertionError(f"stream ended without its receipt chunk: {last!r}")
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "".join(parts)},
                "finish_reason": (final.get("choices") or [{}])[0].get("finish_reason"),
            }
        ],
        "usage": final.get("usage"),
        "mlx2": final["mlx2"],
    }, arrivals


def normalize_chat_content(value):
    """Normalize only outer whitespace, identically for stream and non-stream."""
    return (value or "").strip()


def observed_compute_widths(receipt):
    """Return actual execution widths across ordinary and speculative routes."""
    mtp = receipt.get("mtp") or {}
    widths = mtp.get("observed_compute_widths") or []
    if widths:
        return [int(width) for width in widths]
    speculation = receipt.get("speculation") or {}
    if speculation.get("target_width") is not None:
        return [int(speculation["target_width"])]
    ordinary = receipt.get("ordinary_compute_width")
    return [] if ordinary is None else [int(ordinary)]


def route_mechanism_checks(initial, final):
    """Return aggregate checks that bind route receipts to observed mechanisms."""
    def value(status, path):
        current = status
        for name in path.split("."):
            if not isinstance(current, dict) or name not in current:
                return None
            current = current[name]
        return current if type(current) is int and current >= 0 else None

    def delta(path):
        before, after = value(initial, path), value(final, path)
        return None if before is None or after is None else after - before

    def row(passed, paths):
        return {
            "passed": bool(passed),
            "evidence": {
                path: {
                    "before": value(initial, path),
                    "after": value(final, path),
                    "delta": delta(path),
                }
                for path in paths
            },
        }

    checks = {
        "apcv2_reuse": row(
            delta("apcv2.hits") is not None and delta("apcv2.hits") > 0,
            ["apcv2.hits"],
        ),
        "apcv2_stores": row(
            delta("apcv2.stores") is not None and delta("apcv2.stores") > 0,
            ["apcv2.stores"],
        ),
    }
    if not initial.get("settings", {}).get("mtp"):
        return checks
    segmented = "execution.segmented_mtp."
    checks.update({
        "mtp_segmented_attention": row(
            delta(segmented + "segmented_attention_calls") is not None
            and delta(segmented + "segmented_attention_calls") > 0,
            [segmented + "segmented_attention_calls"],
        ),
        "mtp_transaction_branches": row(
            delta(segmented + "transaction_branches") is not None
            and delta(segmented + "transaction_branches") > 0,
            [segmented + "transaction_branches"],
        ),
        "mtp_committed_cycles": row(
            delta(segmented + "committed_cycles") is not None
            and delta(segmented + "committed_cycles") > 0,
            [segmented + "committed_cycles"],
        ),
        "mtp_true_batching": row(
            delta(segmented + "true_batched_engaged") is not None
            and delta(segmented + "true_batched_engaged") > 0
            and delta(segmented + "batched_target_forwards") is not None
            and delta(segmented + "batched_target_forwards") > 0,
            [segmented + "true_batched_engaged", segmented + "batched_target_forwards"],
        ),
        "mtp_zero_full_prefix_materializations": row(
            value(final, segmented + "full_prefix_materializations") == 0,
            [segmented + "full_prefix_materializations"],
        ),
        "mtp_zero_physical_b2": row(
            value(final, segmented + "physical_b2_formations") == 0,
            [segmented + "physical_b2_formations"],
        ),
        "mtp_zero_failures": row(
            value(final, segmented + "failures") == 0,
            [segmented + "failures"],
        ),
    })
    return checks


def wait_for_quiescence(
    get_status,
    *,
    timeout_seconds=5.0,
    poll_interval_seconds=0.05,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    """Wait for request cleanup before sampling final lease invariants."""
    timeout_seconds = float(timeout_seconds)
    poll_interval_seconds = float(poll_interval_seconds)
    if (
        not math.isfinite(timeout_seconds)
        or not math.isfinite(poll_interval_seconds)
        or timeout_seconds <= 0
        or poll_interval_seconds < 0
    ):
        raise ValueError("quiescence timing must be positive and bounded")
    started = monotonic()
    samples = []
    while True:
        status = get_status()
        cow = status.get("apcv2", {}).get("cow", {})
        def counter(mapping, name):
            value = mapping.get(name)
            return value if type(value) is int and value >= 0 else None
        sample = {
            "elapsed_seconds": monotonic() - started,
            "inflight": counter(status, "inflight"),
            "queue_depth": counter(status, "queue_depth"),
            "active_cow_leases": counter(cow, "active_leases"),
        }
        samples.append(sample)
        settled = all(sample[name] == 0 for name in (
            "inflight", "queue_depth", "active_cow_leases"
        ))
        if settled:
            return status, {
                "passed": True,
                "timed_out": False,
                "timeout_seconds": timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
                "attempts": len(samples),
                "elapsed_seconds": sample["elapsed_seconds"],
                "samples": samples,
            }
        remaining = timeout_seconds - sample["elapsed_seconds"]
        if remaining <= 0:
            return status, {
                "passed": False,
                "timed_out": True,
                "timeout_seconds": timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
                "attempts": len(samples),
                "elapsed_seconds": sample["elapsed_seconds"],
                "samples": samples,
            }
        sleep(min(poll_interval_seconds, remaining))


# The prefill-scheduling forcing load (see run_prefill_scheduling_forcing).
# The hold lane keeps the admission gate shut while the load is queued; it
# holds EOS for this many tokens and is cancelled long before it gets there.
SRPT_FORCING_HOLD_TOKENS = 4096
SRPT_FORCING_COMPLETION_TOKENS = 4
# Tokens past one prefill slice, so the long prompt stays a multi-slice
# (incremental) prefill after its chat template and any APC hit.
SRPT_FORCING_SLICE_MARGIN = 512
# Hold lane + an atomic cohort of two + long + short, admitted in one pass.
# The ordinary route prefills two prompts per admission (prefill_batch_size
# 2), so it reorders only with a second short prompt queued: one more lane.
SRPT_FORCING_MIN_LANES = 4
SRPT_FORCING_ORDINARY_MIN_LANES = 5
SRPT_FORCING_COHORT_ATTEMPTS = 3
SRPT_FORCING_SETUP_SECONDS = 60.0
# The worker republishes its scheduler counters into /v1/status at most once
# a second; a stale snapshot can hide a reorder but never invent one.
SRPT_FORCING_COUNTER_SETTLE_SECONDS = 10.0


def prefill_scheduling_counters(status):
    """The scheduler's own SRPT counters from a ``/v1/status`` snapshot."""
    scheduler = status.get("scheduler") or {}
    return {
        key: int(scheduler.get(f"prefill_scheduling_{key}", 0) or 0)
        for key in ("bypasses", "bypass_forced", "one_slice_clamps")
    }


def prefill_scheduling_reorders(before, after):
    """Reorders between two counter readings: overtakes plus capped service.

    The same quantity ``feature_observations`` reports, as a delta, so the
    forcing check credits only reorders its own load produced.
    """
    return sum(
        max(0, after[key] - before[key]) for key in ("bypasses", "bypass_forced")
    )


def srpt_forcing_long_tokens(settings):
    """Prompt words for the long request: one prefill slice plus a margin.

    ``_next_mtp`` applies SRPT only while a candidate still needs more than
    one slice (``prefill_step + 1`` residual tokens); a prompt that fits one
    slice is admitted together with the short one and nothing is reordered.
    Capped below the route's context so the request is always accepted.
    """
    step = int(settings.get("prefill_step") or 0)
    context = int(settings.get("max_context") or 0)
    cap = context - LONG_CONTEXT_HEADROOM - SRPT_FORCING_COMPLETION_TOKENS
    return max(1, min(step + SRPT_FORCING_SLICE_MARGIN, cap))


def run_prefill_scheduling_forcing(
    *,
    get_status,
    post,
    settings,
    nonce,
    long_text_for=None,
    setup_seconds=SRPT_FORCING_SETUP_SECONDS,
    settle_seconds=SRPT_FORCING_COUNTER_SETTLE_SECONDS,
    poll_seconds=0.02,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    """Force one SRPT reorder and return ``(passed, evidence)``.

    Natural qualification traffic never queues a short prompt behind an
    older multi-slice one, so ``feature_prefill_scheduling`` could not pass
    on any route that selects the policy.  This load makes the reorder a
    consequence of the server's admission rules, not of timing:

    1. A *hold* request streams with EOS held, so one lane is active.
    2. An atomic ``batch_cohort`` of two tiny requests is published.  The
       worker never attaches a declared cohort beside live work; it holds
       the cohort and stops admitting anything queued behind it.
    3. The *long* (multi-slice) prompt is published, then, each only after
       the server reports the previous one queued, the *short* ones: FIFO
       order long -> short (-> second short when the lanes allow it).
    4. Closing the hold stream cancels its single lane.  On the next worker
       pass the cohort attaches and, in the same admission loop, the long
       and short prompts behind it.  Ungrouped prefill waits while the
       cohort owns the batch, so when it drains the scheduler chooses among
       prompts that are all fully queued: SRPT serves a short one first and
       counts the older long one as overtaken.  (The ordinary route admits
       two prompts per prefill round, hence its second short prompt.)

    Every step is confirmed through ``/v1/status`` (``queue_depth``,
    ``inflight``) before the next one, and the verdict is the scheduler's own
    counter delta.  A cohort that misses its staging deadline is retried
    before anything was released; nothing after the release is retried.
    """
    evidence = {
        "schema": "mlx2.prefill-scheduling-forcing.v1",
        "policy": settings.get("prefill_scheduling"),
        "max_lanes": settings.get("max_lanes"),
        "max_inflight": settings.get("max_inflight"),
        "prefill_step": settings.get("prefill_step"),
        "steps": [],
    }

    def step(name, **detail):
        evidence["steps"].append({"step": name, **detail})

    def fail(reason):
        evidence["failure"] = reason
        return False, evidence

    lanes = int(settings.get("max_lanes") or 0)
    native_mtp = settings.get("mtp") is True
    needed = SRPT_FORCING_MIN_LANES if native_mtp else SRPT_FORCING_ORDINARY_MIN_LANES
    evidence["route"] = "native_mtp" if native_mtp else "ordinary"
    if lanes < needed:
        return fail(
            f"the forcing load needs --max-lanes >= {needed} on this route "
            "(the released lane must admit the cohort and every queued prompt "
            "in one admission pass)"
        )
    # Every lane past the hold lane and the cohort takes a queued prompt.
    shorts = min(2, lanes - 3)
    evidence["short_prompts"] = shorts
    inflight = 3 + 1 + shorts
    max_inflight = settings.get("max_inflight")
    if max_inflight is not None and int(max_inflight) < inflight:
        return fail(f"the forcing load needs --max-inflight >= {inflight}")

    def wait_for(predicate, what):
        deadline = monotonic() + setup_seconds
        while True:
            status = get_status()
            if predicate(status):
                return status
            if monotonic() >= deadline:
                raise TimeoutError(
                    f"{what}: inflight={status.get('inflight')} "
                    f"queue_depth={status.get('queue_depth')}"
                )
            sleep(poll_seconds)

    def no_thinking(text, max_tokens, **kw):
        return {
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
            "think": False,
            **kw,
        }

    def call(body):
        try:
            response = post(body)
        except HTTPError as error:
            return {"refused": http_refusal(error), "finished_at": monotonic()}
        except Exception as error:  # noqa: BLE001 - becomes check evidence
            return {"error": f"{type(error).__name__}: {error}",
                    "finished_at": monotonic()}
        return {"response": response, "finished_at": monotonic()}

    release = {}

    def summary(result):
        # Completion instants are relative to the release, for the reader;
        # the verdict never depends on them.
        finished = result["finished_at"] - release.get("at", result["finished_at"])
        if "response" not in result:
            return {**{k: v for k, v in result.items() if k != "finished_at"},
                    "finished_after_release_seconds": finished}
        response = result["response"]
        receipt = response.get("mlx2") or {}
        return {
            "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
            "usage": response.get("usage"),
            "cached_tokens": receipt.get("cached_tokens"),
            "finished_after_release_seconds": finished,
        }

    try:
        idle = wait_for(
            lambda s: s.get("inflight") == 0 and s.get("queue_depth") == 0,
            "server did not become idle before the forcing load",
        )
    except TimeoutError as error:
        return fail(str(error))
    before = prefill_scheduling_counters(idle)
    evidence["counters_before"] = before

    hold_tokens = max(
        16,
        min(
            SRPT_FORCING_HOLD_TOKENS,
            int(settings.get("max_context") or 0) - LONG_CONTEXT_HEADROOM,
        ),
    )
    try:
        hold = post(
            no_thinking(
                f"Count upward from 1, one number per line. ({nonce} hold)",
                hold_tokens,
                min_tokens=hold_tokens,
            ),
            stream=True,
        )
    except OSError as error:  # HTTPError included: a refused hold request
        return fail(f"hold request failed: {type(error).__name__}: {error}")
    pool = ThreadPoolExecutor(max_workers=8)
    futures = {}
    try:
        # A token on the hold stream proves its lane is in the worker's
        # active set, which is what keeps a held cohort from attaching.
        while True:
            raw = hold.readline()
            if not raw:
                return fail("hold stream closed before its first token")
            line = raw.decode() if isinstance(raw, bytes) else raw
            if line.startswith("data: [DONE]"):
                return fail("hold stream finished before its first token")
            if not line.startswith("data: {"):
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                return fail(f"hold stream failed: {chunk['error']}")
            if any(
                (choice.get("delta") or {}).get("content")
                or (choice.get("delta") or {}).get("reasoning_content")
                for choice in chunk.get("choices") or ()
            ):
                break
        held = get_status()
        base = int(held.get("queue_depth") or 0)
        step("hold_active", inflight=held.get("inflight"), queue_depth=base)

        cohort_held = False
        for attempt in range(1, SRPT_FORCING_COHORT_ATTEMPTS + 1):
            cohort = {"id": f"qualification-srpt-gate-{nonce}-{attempt}", "size": 2}
            members = [
                pool.submit(call, no_thinking(
                    f"Reply with exactly GATE_{index} ({nonce})",
                    SRPT_FORCING_COMPLETION_TOKENS,
                    batch_cohort=cohort,
                ))
                for index in range(2)
            ]
            deadline = monotonic() + setup_seconds
            while True:
                status = get_status()
                if int(status.get("queue_depth") or 0) >= base + 2:
                    cohort_held = True
                    break
                if any(member.done() for member in members) or monotonic() >= deadline:
                    break
                sleep(poll_seconds)
            if cohort_held:
                futures["cohort"] = members
                step("cohort_held", attempt=attempt,
                     queue_depth=status.get("queue_depth"))
                break
            outcomes = [summary(member.result()) for member in members]
            step("cohort_not_published", attempt=attempt, outcomes=outcomes)
        if not cohort_held:
            return fail("the gating cohort was never published")

        long_words = srpt_forcing_long_tokens(settings)
        evidence["long_prompt_words"] = long_words
        long_text = (long_text_for or long_context_filler)(long_words)
        # The nonce leads so no earlier prompt (the near-limit context check
        # uses the same filler) can serve the long prompt from APC.
        futures["long"] = pool.submit(call, no_thinking(
            f"nonce {nonce} {long_text} Reply with exactly SRPT_LONG",
            SRPT_FORCING_COMPLETION_TOKENS,
        ))
        status = wait_for(
            lambda s: int(s.get("queue_depth") or 0) >= base + 3,
            "long request was not queued behind the held cohort",
        )
        step("long_queued", queue_depth=status.get("queue_depth"))
        futures["shorts"] = []
        for index in range(shorts):
            futures["shorts"].append(pool.submit(call, no_thinking(
                f"Reply with exactly SRPT_SHORT_{index} ({nonce})",
                SRPT_FORCING_COMPLETION_TOKENS,
            )))
            queued = base + 4 + index
            status = wait_for(
                lambda s, queued=queued: int(s.get("queue_depth") or 0) >= queued,
                "short request was not queued behind the long one",
            )
        gate = {
            "queue_depth": status.get("queue_depth"),
            "inflight": status.get("inflight"),
            "expected_queue_depth": base + 3 + shorts,
        }
        evidence["gate_before_release"] = gate
        step("shorts_queued", **gate)
        if int(status.get("queue_depth") or 0) != base + 3 + shorts:
            return fail("the admission gate opened before the release")
    except (TimeoutError, OSError, ValueError) as error:
        # A stalled, refused or malformed setup step is a failed check with
        # its evidence, never an exception that discards the run.
        return fail(f"{type(error).__name__}: {error}")
    finally:
        # Release (or abandon) the hold lane: its cancellation is the one
        # event that opens the gate.
        hold.close()
        release["at"] = monotonic()
        step("hold_released")
        results = {}
        for name in ("cohort", "long", "shorts"):
            future = futures.get(name)
            if future is None:
                continue
            if isinstance(future, list):
                results[name] = [summary(member.result()) for member in future]
            else:
                results[name] = summary(future.result())
        pool.shutdown(wait=True)
        evidence["results"] = results

    settle_deadline = monotonic() + settle_seconds
    while True:
        after = prefill_scheduling_counters(get_status())
        reorders = prefill_scheduling_reorders(before, after)
        if reorders > 0 or monotonic() >= settle_deadline:
            break
        sleep(min(0.25, settle_seconds))
    evidence["counters_after"] = after
    evidence["reorders"] = reorders
    results = evidence["results"]
    completed = all(
        "usage" in result
        for result in [*results.get("cohort", []), results.get("long", {}),
                       *results.get("shorts", [])]
    ) and len(results.get("shorts", [])) == shorts
    long_result = results.get("long", {})
    if "usage" in long_result:
        prompt_tokens = int((long_result.get("usage") or {}).get("prompt_tokens") or 0)
        residual = prompt_tokens - int(long_result.get("cached_tokens") or 0)
        evidence["long_residual_tokens"] = residual
        evidence["long_multi_slice"] = residual > int(settings.get("prefill_step") or 0) + 1
    if not completed:
        return fail("a forcing-load request did not complete")
    if reorders <= 0:
        return fail("the scheduler did not reorder a queued short prompt")
    return True, evidence


def sp_qmm_routed_observation(initial, final):
    """Count only routed calls made during this qualification run."""
    if initial is None:
        return 0
    selected = (initial.get("settings") or {}).get("sp_qmm")
    if not selected or selected != (final.get("settings") or {}).get("sp_qmm"):
        return 0
    before = initial.get("sp_qmm") or {}
    after = final.get("sp_qmm") or {}
    if before.get("enabled") is not True or after.get("enabled") is not True:
        return 0
    modules = before.get("modules")
    if type(modules) is not int or modules <= 0 or after.get("modules") != modules:
        return 0
    start, end = before.get("routed_calls"), after.get("routed_calls")
    if (type(start) is not int or type(end) is not int
            or start < 0 or end <= start):
        return 0
    return end - start


def qsdpa_verify_observation(initial, final):
    """Count only selected verify-kernel calls made during this run."""
    if initial is None:
        return 0
    selected = (initial.get("settings") or {}).get("qsdpa_verify_kernel") or {}
    if selected.get("enabled") is not True or selected != (
        (final.get("settings") or {}).get("qsdpa_verify_kernel") or {}
    ):
        return 0
    before = (initial.get("qsdpa_verify") or {}).get("counts") or {}
    after = (final.get("qsdpa_verify") or {}).get("counts") or {}
    start, end = before.get("verify_kernel_calls"), after.get("verify_kernel_calls")
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        return 0
    if not ((initial.get("settings") or {}).get("approximate_kv") or {}).get(
        "compose_mtp", False
    ):
        return end - start
    # L=1 can be an ordinary fallback. For self-MTP prove that a multirow
    # target verification, rather than just an unrelated decode, engaged.
    before_rows = (initial.get("qsdpa_verify") or {}).get("by_rows") or {}
    after_rows = (final.get("qsdpa_verify") or {}).get("by_rows") or {}
    multirow_delta = 0
    for rows in range(2, 9):
        key = f"verify_kernel_L{rows}"
        old, new = before_rows.get(key, 0), after_rows.get(key, 0)
        if type(old) is not int or type(new) is not int or new < old:
            return 0
        multirow_delta += new - old
    return min(end - start, multirow_delta)


def feature_observations(final, kv_fidelity=None, adaptive_benchmark=None, initial=None):
    execution = final.get("execution", {})
    initial_execution = (initial or {}).get("execution", {})

    def prefill_counter_delta(mechanism, key):
        if initial is None:
            return 0
        before = initial_execution.get(mechanism, {}).get("counters", {})
        after = execution.get(mechanism, {}).get("counters", {})
        start, end = before.get(key, 0), after.get(key, 0)
        if type(start) is not int or type(end) is not int or end <= start:
            return 0
        return end - start

    def prefill_delta(mechanism, keys):
        return sum(prefill_counter_delta(mechanism, key) for key in keys)

    def section_counter_delta(section, key):
        if initial is None:
            return 0
        before = ((initial or {}).get(section) or {}).get(key, 0)
        after = (final.get(section) or {}).get(key, 0)
        if type(before) is not int or type(after) is not int or after <= before:
            return 0
        return after - before

    scheduler = final.get("scheduler", {})
    initial_scheduler = (initial or {}).get("scheduler") or {}

    def run_delta(after, before, *path):
        """A counter's growth during this run (its final value without an
        initial snapshot), so engagement at load or in an earlier run
        cannot pass for this run's evidence."""
        def read(root):
            value = root
            for key in path:
                if not isinstance(value, dict):
                    return 0
                value = value.get(key, 0)
            return value if type(value) is int and value >= 0 else 0
        return max(0, read(after) - read(before))

    def count_delta(*path):
        """``run_delta`` of a status counter addressed from the snapshot root."""
        return run_delta(final, initial or {}, *path)

    # Progressive verification is stateful across target launches, so its
    # engagement gate is intentionally stronger than a single counter bump.
    # Bind both snapshots to the same selected tile and require a successfully
    # published round that crossed a nonfinal tile boundary.  The round's
    # counters are restored on failure by ExternalDraftBatchGenerator.
    initial_policy = ((initial or {}).get("settings") or {}).get(
        "execution_policy"
    ) or {}
    final_policy = (final.get("settings") or {}).get("execution_policy") or {}
    initial_tile = initial_policy.get("progressive_verification_tile")
    final_tile = final_policy.get("progressive_verification_tile")
    scheduler_tile_before = initial_scheduler.get(
        "external_progressive_verify_tile"
    )
    scheduler_tile_after = scheduler.get("external_progressive_verify_tile")
    progressive_rounds = run_delta(
        scheduler, initial_scheduler, "external_progressive_verify_rounds"
    )
    progressive_launches = run_delta(
        scheduler, initial_scheduler, "external_progressive_verify_launches"
    )
    progressive_rows = run_delta(
        scheduler, initial_scheduler, "external_progressive_verify_target_rows"
    )
    progressive_full_tiles = run_delta(
        scheduler, initial_scheduler, "external_progressive_verify_full_tiles"
    )
    progressive_observed = 0
    if (
        type(initial_tile) is int
        and initial_tile > 0
        and final_tile == initial_tile
        and scheduler_tile_before == initial_tile
        and scheduler_tile_after == initial_tile
    ):
        progressive_observed = min(
            progressive_rounds,
            progressive_full_tiles,
            max(0, progressive_launches - progressive_rounds),
            progressive_rows,
        )
    initial_multilane_cap = initial_policy.get(
        "progressive_multilane_draft_cap"
    )
    final_multilane_cap = final_policy.get(
        "progressive_multilane_draft_cap"
    )
    scheduler_multilane_cap_before = initial_scheduler.get(
        "external_multilane_draft_cap"
    )
    scheduler_multilane_cap_after = scheduler.get(
        "external_multilane_draft_cap"
    )
    multilane_cap_rounds = run_delta(
        scheduler, initial_scheduler, "external_multilane_draft_cap_rounds"
    )
    multilane_cap_lanes = run_delta(
        scheduler, initial_scheduler, "external_multilane_draft_cap_lanes"
    )
    multilane_cap_observed = 0
    if (
        type(initial_multilane_cap) is int
        and initial_multilane_cap > 0
        and final_multilane_cap == initial_multilane_cap
        and scheduler_multilane_cap_before == initial_multilane_cap
        and scheduler_multilane_cap_after == initial_multilane_cap
    ):
        multilane_cap_observed = min(
            multilane_cap_rounds,
            max(0, multilane_cap_lanes - multilane_cap_rounds),
        )
    fly_receipt_relaxed = 0
    initial_receipts = (initial or {}).get("recent_receipts") or ()
    for receipt in final.get("recent_receipts", ()):
        if receipt in initial_receipts:
            continue  # served before this run
        for key in ("mtp", "speculation"):
            verification = receipt.get(key) or {}
            if verification.get("verification") == "fly":
                fly_receipt_relaxed = max(
                    fly_receipt_relaxed,
                    int(verification.get("relaxed_accepts", 0)),
                )
    settings = final.get("settings", {})
    ple_tables = execution.get("ple_tables", [])
    initial_ple_tables = initial_execution.get("ple_tables") or []
    ple_compile = execution.get("ple_compile", {})
    ple_compile_counts = ple_compile.get("counts", {})
    fused_gdn = execution.get("fused_gdn", {})
    moe = execution.get("moe", {})
    # A fallback demotes its signature to the eager chain for the life of
    # the process, so the health guard stays lifetime; only hits are the run's.
    compiled_ple_healthy = (
        ple_compile.get("enabled") is True
        and ple_compile_counts.get("builds", 0) > 0
        and all(ple_compile_counts.get(name, 0) == 0
                for name in ("fallbacks", "overflow", "skips"))
    )
    latched = latched_before_run(initial)

    def latch_observation(feature):
        """A latch with no run counter counts only if it was clear initially."""
        if feature in latched:
            return 0
        value = final
        for key in LATCHED_FEATURE_GATES[feature]:
            value = value.get(key) if isinstance(value, dict) else None
        return int(bool(value))
    # External-draft transactions publish their counters through the scheduler.
    # Native self-MTP publishes the same logical evidence through its segmented
    # execution counters instead.  A zero/partial accepted draft is a real
    # rollback: commit keeps only the accepted prefix (and target token), while
    # the unaccepted speculative suffix is discarded.  transaction_rejections
    # is narrower; it records an entire delivery/branch rejection and is not
    # expected during healthy nonterminal serving.
    # Both are this run's growth, like every other engagement counter.
    native_segmented = settings.get("mtp") is True
    if native_segmented:
        def segmented_delta(key):
            return run_delta(execution, initial_execution, "segmented_mtp", key)

        segmented_transactions = segmented_delta("transaction_branches")
        segmented_rollbacks = (
            segmented_delta("accepted_zero") + segmented_delta("accepted_partial")
        )
    else:
        segmented_transactions = run_delta(
            scheduler, initial_scheduler, "segmented_transactions"
        )
        segmented_rollbacks = run_delta(scheduler, initial_scheduler, "segmented_rollbacks")
    apcv2 = final.get("apcv2", {})
    idle_disk = apcv2.get("idle_disk", {})
    rescan = apcv2.get("persistence", {}).get("rescan", {})
    host_available = final.get("host_memory_available_bytes")
    benchmark_evidence = (
        (adaptive_benchmark or {}).get("adaptive_qualification", {})
    )
    benchmark_features = benchmark_evidence.get("features") or {}
    adaptive_feature = benchmark_features.get("adaptive_mtp_depth") or {}
    adaptive_observed = int(
        bool(
            adaptive_feature.get("selected")
            and adaptive_feature.get("passed")
        )
    )
    benchmark_handoff = benchmark_evidence.get("handoff", {})
    handoff_feature = benchmark_features.get("mtp_ordinary_handoff") or {}
    handoff_observed = int(
        bool(
            handoff_feature.get("selected")
            and handoff_feature.get("passed")
            and benchmark_handoff.get("selected")
            and benchmark_handoff.get("passed")
            and benchmark_handoff.get("events", 0) > 0
            and benchmark_handoff.get("comparison_count", 0) > 0
            and not benchmark_handoff.get("unsafe_divergences")
        )
    )
    return {
        "indexed_fused_merge": latch_observation("indexed_fused_merge"),
        "indexed_output_gate": latch_observation("indexed_output_gate"),
        "shared_qsa": run_delta(
            execution, initial_execution, "segmented_mtp", "shared_qsa_batched_selections"
        ),
        "async_promotion": run_delta(
            execution, initial_execution, "segmented_mtp", "async_qsa_promotion_engaged"
        ),
        "indexed_qsa": run_delta(execution, initial_execution, "indexed_qsa", "counts", "engaged"),
        "private_delta": run_delta(
            execution, initial_execution, "segmented_mtp", "private_delta_attention_calls"
        ),
        "known_tail_prefetch": run_delta(
            execution, initial_execution, "round_levers", "ple_tail_prefetch_tables"
        ),
        # Tables are fixed at load, so the same index is the same table.
        "file_backed_ple": sum(
            run_delta(
                table,
                initial_ple_tables[index] if index < len(initial_ple_tables) else {},
                "lookups",
            )
            for index, table in enumerate(ple_tables)
        ),
        "compiled_ple": (
            run_delta(execution, initial_execution, "ple_compile", "counts", "hits")
            if compiled_ple_healthy
            else 0
        ),
        "pooled_qsa": run_delta(
            execution, initial_execution, "round_levers", "qsa_pooled_key_cache_hits"
        ),
        "scatter_qsa": run_delta(
            execution, initial_execution, "round_levers", "qsa_scatter_chosen_calls"
        ),
        "fused_gdn_decode": fused_gdn_decode_observation(
            fused_gdn,
            None if initial is None else initial_execution.get("fused_gdn") or {},
        ),
        "fused_gdn_verify": run_delta(execution, initial_execution, "fused_gdn", "verify_calls"),
        "fused_gdn_replay_rollback": run_delta(
            execution, initial_execution, "fused_gdn", "replay_rollback_calls"
        ),
        "eager_dispatch": run_delta(
            execution, initial_execution, "round_levers", "eager_async_evals"
        ),
        "fused_moe": (
            sum(
                run_delta(execution, initial_execution, "moe", "dispatches", mode)
                for mode in moe.get("dispatches", {})
            )
            if moe.get("fused_gate_up_layers", 0) > 0
            else 0
        ),
        # A fallback during this run voids its external evidence; one from
        # an earlier run does not.
        "external_draft": (
            run_delta(scheduler, initial_scheduler, "external_rounds")
            if run_delta(scheduler, initial_scheduler, "draft_fallbacks") == 0
            else 0
        ),
        "proposal_distribution": run_delta(scheduler, initial_scheduler, "proposed_tokens"),
        "paired_draft_cache": run_delta(
            scheduler, initial_scheduler, "paired_cache_resumes"
        ),
        "segmented_transaction": segmented_transactions,
        "segmented_rollback": segmented_rollbacks,
        "prompt_lookup": run_delta(scheduler, initial_scheduler, "pld_retrieval_cycles"),
        "prompt_lookup_proposals": run_delta(scheduler, initial_scheduler, "pld_proposed"),
        "prompt_lookup_rollback": run_delta(scheduler, initial_scheduler, "pld_rollbacks"),
        "prompt_lookup_batched_verify": run_delta(
            scheduler, initial_scheduler, "pld_batched_rounds"
        ),
        "prompt_lookup_rotating_replay": run_delta(
            scheduler, initial_scheduler, "pld_rotating_replay_rounds"
        ),
        # The bound policy benchmark proves only the features it selected.
        # Adaptive depth still needs bounded exploration, complete sampling,
        # and conditional recovery; handoff uses its independent fixed-depth
        # event/correctness/throughput gate.
        "adaptive_mtp_depth": adaptive_observed,
        "mtp_ordinary_handoff": handoff_observed,
        "self_mtp_copy_draft": run_delta(scheduler, initial_scheduler, "self_mtp_copy_rounds"),
        "fly_verification": max(
            run_delta(scheduler, initial_scheduler, "fly_relaxed_accepts"),
            fly_receipt_relaxed,
        ),
        "spomin_surgery": count_delta("spomin_live_surgery", "counts", "applied"),
        "approximate_kv": count_delta("approximate_kv", "applied"),
        # Int8 NAX prefill: engaged GEMMs.  required_feature_checks demands
        # feature_int8_prefill whenever the policy is enabled, so the
        # observation key must exist or the qualifier raises KeyError.
        "int8_prefill": count_delta("int8_prefill", "counts", "engaged_calls"),
        "prefill_projection": prefill_delta(
            "tensorfold_prefill", ("projection_calls", "grouped_calls", "swiglu_calls")
        ),
        "prefill_scan": prefill_delta("gdn_prefill_scan", ("calls",)),
        "varlen_dense_mlp": min(
            prefill_counter_delta("varlen_dense_mlp", "mlp_compaction_calls"),
            prefill_counter_delta("varlen_dense_mlp", "padding_token_rows"),
        ),
        "varlen_sparse_moe": min(
            prefill_counter_delta("varlen_sparse_moe", "moe_compaction_calls"),
            prefill_counter_delta("varlen_sparse_moe", "padding_token_rows"),
        ),
        "external_varlen_prefill": min(
            section_counter_delta(
                "scheduler", "external_batched_prefill_rounds"
            ),
            section_counter_delta(
                "scheduler", "external_batched_prefill_lanes"
            ),
        ),
        "ingress_cohort": section_counter_delta(
            "counts", "ingress_cohort_target_reached"
        ),
        "sp_qmm": sp_qmm_routed_observation(initial, final),
        "qsdpa_verify_kernel": qsdpa_verify_observation(initial, final),
        "verify_bitexact": (
            count_delta("verify_bitexact", "dispatches")
            if (final.get("verify_bitexact") or {}).get("active") is True
            else 0
        ),
        "approximate_kv_mtp": count_delta("approximate_kv", "mtp_lanes"),
        # Offline measurement (scripts/measure_kv_quant_fidelity.py) judged by
        # runtime/kv_quant_fidelity.py; 1 only when that verdict passed.
        "approximate_kv_fidelity": int(bool((kv_fidelity or {}).get("passed"))),
        "apc_interior_checkpoints": min(
            count_delta("counts", "apc_interior_checkpoints_captured"),
            count_delta("counts", "apc_interior_checkpoints_published"),
        ),
        # Persistence qualification is deliberately restart-bound: one run
        # must have written the snapshot, and this process must have rescanned
        # and restored it. A fresh one-shot process cannot self-certify this.
        "apc_persistence": min(
            idle_disk.get("persisted_writes", 0),
            rescan.get("registered", 0),
            idle_disk.get("restores", 0),
        ),
        "apc_sessions": min(
            count_delta("apcv2", "idle_disk", key)
            for key in ("parks", "resumes", "prefetch_restores_ok", "prefetch_hits")
        ),
        # Each observation below counts the mechanism engaging, never the
        # policy being selected.  A hybrid rolling checkpoint must have been
        # published and later resumed from; the KV route's only publication
        # is a cancelled prefill's partial cache, which is its evidence.
        "apc_rolling_checkpoints": max(
            min(
                count_delta("counts", "apc_rolling_checkpoints_published"),
                count_delta("apcv2", "lifetime", "rolling_hits"),
            ),
            count_delta("counts", "apc_rolling_checkpoints_cancel_published"),
        ),
        # A junction is useful only when a later diverging request hit it.
        "apc_junction_checkpoints": min(
            count_delta("counts", "apc_junction_checkpoints_published"),
            count_delta("apcv2", "lifetime", "junction_hits"),
        ),
        "apc_inflight_prefix_wait": min(
            count_delta("counts", "apc_inflight_checkpoints_published"),
            count_delta("counts", "apc_inflight_prefix_hits"),
        ),
        # A bypass is an SRPT reorder (an older prompt was overtaken); a
        # forced bypass is bypass-capped service.  Slice clamps are neither.
        "prefill_scheduling": (
            count_delta("scheduler", "prefill_scheduling_bypasses")
            + count_delta("scheduler", "prefill_scheduling_bypass_forced")
        ),
        # The status snapshot carries a host reading only when the policy is
        # on and the Mach probe answered; psutil fallback leaves it absent.
        "host_memory_signals": int(
            type(host_available) is int and host_available > 0
        ),
        # A streamed model that never paged an expert in was fully resident.
        # Serving-phase counter: load-time probe page-ins are counted apart.
        "moe_expert_streaming": count_delta("counts", "stream_page_ins_total"),
        "dense_weight_streaming": count_delta("counts", "dense_stream_page_ins_total"),
        # Preemption proves nothing unless the parked lane was replayed.
        "memory_preemption": min(
            count_delta("counts", "memory_preemptions"),
            count_delta("counts", "preempted_replays"),
        ),
        "tool_grammar_auto": count_delta(
            "counts", "constrained_tool_grammar_auto_engagements"
        ),
        "tool_grammar_streaming": count_delta("counts", "constrained_tool_grammar_streams"),
        "external_pairwise_selection": run_delta(
            scheduler, initial_scheduler, "external_pairwise_selection_groups"
        ),
        # The default Qwen3.8 tree route: lanes served by tree rounds and
        # rounds the TensorFold target executed, both during this run.
        "external_tree": min(
            run_delta(scheduler, initial_scheduler, "external_tree_rounds"),
            run_delta(scheduler, initial_scheduler, "external_tensorfold_target_rounds"),
        ),
        "progressive_verification": progressive_observed,
        "progressive_multilane_draft_cap": multilane_cap_observed,
        "fused_gdn_dynamic_accept": run_delta(
            execution, initial_execution, "fused_gdn", "replay_dynamic_rollback_calls"
        ),
        **default_on_observations(final, initial),
    }


def default_on_observations(final, initial=None):
    """Engagement of the default-on Flash-Next / 27B kernels during this run.

    Each value is a counter delta from the initial status (the final value
    when there is none), so a mechanism that only ran at load cannot pass.
    """
    from mlx2.qualification import (
        SLICE_FLOOR_LIFTS,
        moe_nax_gather_engagement,
        qsa_nax_prefill_engagement,
    )

    execution = final.get("execution") or {}
    initial_execution = (initial or {}).get("execution") or {}
    environment = (final.get("settings") or {}).get("environment") or {}

    def delta(*path):
        def read(root):
            value = root
            for key in path:
                if not isinstance(value, dict):
                    return 0
                value = value.get(key, 0)
            return value if type(value) is int and value >= 0 else 0
        return max(0, read(execution) - read(initial_execution))

    scheduler = final.get("scheduler") or {}
    initial_scheduler = (initial or {}).get("scheduler") or {}

    def scheduler_delta(name):
        def read(root):
            value = root.get(name, 0)
            return value if type(value) is int and value >= 0 else 0
        return max(0, read(scheduler) - read(initial_scheduler))

    hc = execution.get("hc_decode") or {}
    hc_decode = (
        delta("hc_decode", "calls")
        if hc.get("enabled") is True and not hc.get("broken") and not hc.get("errors")
        else 0
    )
    rows = execution.get("attn_fused_rows") or {}
    rows_counts = ("attn_fused_rows", "counts")
    attn_rows = (
        min(
            delta(*rows_counts, "projection_grouped"),
            delta(*rows_counts, "prep_rows"),
            delta(*rows_counts, "sdpa_1pass_rows") + delta(*rows_counts, "sdpa_2pass_rows"),
        )
        if rows.get("enabled") is True and not rows.get("kernel_failures")
        else 0
    )
    window = ("moe", "moe_window")
    topk_mode = environment.get("MLX_QWEN4_MOE_TOPK_FOLD")
    topk = (
        delta(*window, "topk_calls", topk_mode)
        if topk_mode in {"launch", "fold"}
        else delta(*window, "topk_calls", "launch") + delta(*window, "topk_calls", "fold")
    )
    routed_mode = environment.get("MLX_QWEN4_MOE_ROUTED_DECODE") or ""
    routed = ("moe", "routed_decode")
    routed_parts = [delta(*routed, "calls")]
    if routed_mode.startswith("gate_up_down"):
        routed_parts.append(delta(*routed, "down_calls"))
    if routed_mode.endswith("_shared"):
        routed_parts.append(delta(*routed, "shared_fold_calls"))
    # Qwen3.6 reports the same mechanisms under decode_wins.moe
    # (runtime/models/qwen36_35b.py qwen36_decode_wins_stats): ``calls`` and
    # ``shared_calls`` also count multi-row window calls, so the one-token
    # share is the difference; gate_up / two_launch run the parent path,
    # counted in ``routed_gate_up_calls``.
    q36 = ("decode_wins", "moe")
    q36_window = delta(*q36, "window_calls")
    if routed_mode.startswith("gate_up_down"):
        q36_parts = [max(0, delta(*q36, "calls") - q36_window)]
        if routed_mode.endswith("_shared"):
            q36_parts.append(max(0, delta(*q36, "shared_calls") - q36_window))
        q36_routed = min(q36_parts)
    else:
        q36_routed = delta(*q36, "routed_gate_up_calls")
    q36_topk = delta(*q36, "topk_launch_calls") if topk_mode in {None, "launch"} else 0
    scores = ((execution.get("tensorfold_longctx") or {}).get("qsa_fused_scores") or {})
    qwen38 = execution.get("fused_gdn") or {}
    lane_after = ((final.get("lane_matmul") or {}).get("counts") or {}).get("lane_calls", 0)
    lane_before = (((initial or {}).get("lane_matmul") or {}).get("counts") or {}).get(
        "lane_calls", 0
    )
    return {
        # Flash-Next fused GDN prefill prework: multi-token prefill forwards.
        "fused_gdn_prefill": delta("fused_gdn", "prefill_calls"),
        # Flash-Next sorted-order MoE weighted sum: sorted prefill tails.
        "moe_weighted_sum": delta("moe", "weighted_sum", "calls"),
        # Covered projections that ran the lane kernel (status["lane_matmul"]).
        "lane_matmul": (
            max(0, lane_after - lane_before)
            if type(lane_after) is int and type(lane_before) is int
            else 0
        ),
        # Base decode-time fairness: contended prefill slices it accounted.
        "decode_fairness": scheduler_delta("decode_fairness_prefill_chunks"),
        "hc_decode": hc_decode,
        "fused_gdn_batch_decode": delta("fused_gdn", "batch_decode", "calls"),
        "fused_gdn_batch_verify": delta("fused_gdn", "batch_verify", "calls"),
        "attn_fused_rows": attn_rows,
        "attn_fused_rows_qsa_mask": (
            delta(*rows_counts, "mask_rows") if attn_rows else 0
        ),
        "attn_fused_rows_index_q": (
            delta(*rows_counts, "index_q_rows") if attn_rows else 0
        ),
        # Flash-Next shape (moe.moe_window / moe.routed_decode) or the
        # Qwen3.6 shape (decode_wins.moe), whichever this adapter reports.
        "moe_topk_fold": max(topk, q36_topk),
        "moe_routed_decode": max(min(routed_parts), q36_routed),
        "qsa_fused_scores": (
            delta("tensorfold_longctx", "qsa_fused_scores", "counts", "engaged")
            if scores.get("enabled") is True
            else 0
        ),
        # Qwen3.8-27B reports every admitted fused GDN form, including the
        # owned tree kernel that serves TensorFold tree verify (tree_calls).
        # A route passes this feature only from calls observed after the
        # initial snapshot.  prefill_calls also books the fused prefill
        # chunks of the separate fused_gdn_prefill switch; only the bounded
        # catch-up blocks left after subtracting them are this switch's.
        "qwen38_fused_gdn": (
            delta("fused_gdn", "decode_calls")
            + delta("fused_gdn", "batch_decode_calls")
            + delta("fused_gdn", "verify_calls")
            + max(0, delta("fused_gdn", "prefill_calls")
                  - delta("fused_gdn", "prefill_chunk_calls"))
            + delta("fused_gdn", "tree_calls")
            if qwen38.get("enabled") is True
            else 0
        ),
        # Qwen3.8-27B fused GDN prefill chunks (independent of the decode
        # switch; "prefill_calls" above also counts bounded catch-up blocks).
        "qwen38_fused_gdn_prefill": (
            delta("fused_gdn", "prefill_chunk_calls")
            if qwen38.get("prefill_enabled") is True
            else 0
        ),
        # NAX segmented MoE gather (prefill); "fused" needs gate/up and down.
        "moe_nax_gather": moe_nax_gather_engagement(execution, initial_execution),
        # Qwen3.6 decode slices: one-row fused GDN decode (its own
        # diagnostics key), batched decode / verify / batched verify and
        # the MoE row window (decode_wins).
        "qwen36_fused_gdn_decode": delta("fused_gdn_decode", "fused_calls"),
        "qwen36_fused_gdn_batch_decode": delta("decode_wins", "gdn", "batch_decode", "calls"),
        "qwen36_fused_gdn_verify": delta("decode_wins", "gdn", "verify", "calls"),
        "qwen36_fused_gdn_batch_verify": delta("decode_wins", "gdn", "batch_verify", "calls"),
        "qwen36_moe_window": q36_window,
        # Sorted-MoE pad policy: one recorded choice per sorted expert gather
        # the adaptive (or always) rule decided (switch_layers.moe_pad_status).
        "moe_rhs_pad": sum(
            delta("moe_pad", "choices", reason)
            for reason in ((execution.get("moe_pad") or {}).get("choices") or {})
        ),
        # Prefill forwards through the slice-invariant lane.
        "invariant_prefill": delta("invariant_prefill", "counts", "forwards"),
        # fp16 GDN recurrent-state launches (runtime/models/gdn_state.py).
        "gdn_state_fp16": (
            delta("gdn_state", "counters", "kernel_launches")
            + delta("gdn_state", "counters", "ops_calls")
            if (execution.get("gdn_state") or {}).get("state_dtype") == "float16"
            else 0
        ),
        # MLX gated_delta_update recurrences (MLX_GDN_CORE).
        "gdn_core": (
            delta("gdn_core", "calls")
            if (execution.get("gdn_core") or {}).get("enabled") is True
            else 0
        ),
        # A load-time weight transform with no per-call path: the head's
        # scales and biases are stored fp32, so every later head call returns
        # fp32 logits.  Observed as the installed receipt; the run's decode
        # check proves the head ran.
        "fp32_head_logits": int(
            (execution.get("fp32_head_logits") or {}).get("enabled") is True
            and type((execution.get("fp32_head_logits") or {}).get("extra_resident_bytes")) is int
            and execution["fp32_head_logits"]["extra_resident_bytes"] > 0
        ),
        # Scheduler counters (status["scheduler"]).
        "decode_first": scheduler_delta("decode_first_published_rounds"),
        "decode_fairness_slice_floor": scheduler_delta(SLICE_FLOOR_LIFTS),
        # NAX block-sparse QSA prefill (Flash-Next, M5): kernel calls.
        "qsa_nax_prefill": qsa_nax_prefill_engagement(execution, initial_execution),
    }


# Declines that mean the single-token decode the kernel exists for was refused
# on geometry it should have admitted.  "batch of N rows" is a legitimate B>1
# refusal and does not count.
FUSED_GDN_DECODE_GEOMETRY_REFUSALS = (
    "rollback geometry not describable",
    "masked decode",
    "padded rollback geometry",
)


def fused_gdn_decode_observation(fused_gdn, initial_fused_gdn=None):
    """Fused B1/T1 GDN decode calls, or 0 when any decode was refused on geometry.

    ``fused_calls > 0`` alone passed a route whose decode was refused on 70%
    of its B=1 layer-calls (10,224 fused against 24,012 "rollback geometry not
    describable"): an all-True GDN mask hid a fully valid slab.  The mechanism
    is engaged only if it ran and no single-token decode was declined for
    geometry.  With the initial snapshot's ``fused_gdn`` both are run deltas:
    calls and refusals before the run neither pass it nor void it.
    """
    before = initial_fused_gdn or {}
    reasons = fused_gdn.get("decode_fallback_reasons") or {}
    reasons_before = before.get("decode_fallback_reasons") or {}

    def grown(after, start):
        after = after if type(after) is int and after >= 0 else 0
        start = start if type(start) is int and start >= 0 else 0
        return max(0, after - start)

    if any(
        grown(reasons.get(reason, 0), reasons_before.get(reason, 0))
        for reason in FUSED_GDN_DECODE_GEOMETRY_REFUSALS
    ):
        return 0
    return grown(fused_gdn.get("fused_calls", 0), before.get("fused_calls", 0))


# Feature gates whose only status evidence is a latched boolean with no run
# counter (path from the snapshot root).  A latch the initial snapshot already
# shows set cannot be attributed to this run, so the gate fails closed.
LATCHED_FEATURE_GATES = {
    "indexed_fused_merge": ("execution", "indexed_qsa", "fused_merge", "engaged"),
    "indexed_output_gate": ("execution", "indexed_qsa", "fused_merge", "gate_engaged"),
}


def latched_before_run(initial):
    """``{feature: reason}`` for latched gates already set in ``initial``."""
    if initial is None:
        return {}
    reasons = {}
    for feature, path in LATCHED_FEATURE_GATES.items():
        value = initial
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value:
            reasons[feature] = (
                f"{'.'.join(path)} was already set in the initial status and has "
                "no run counter, so this run's engagement cannot be attributed; "
                "qualify on a freshly started server"
            )
    return reasons


def unobservable_features(features):
    """Return required features this harness has no observation for.

    Checked before any request is sent: a selected mechanism the harness
    cannot observe can never be qualified, so the run must stop before it
    spends GPU time rather than fail on a missing key at the very end.
    """
    observable = set(feature_observations({}))
    return sorted(set(features) - observable)


def _adaptive_benchmark_evaluator():
    """Load the sibling benchmark both as a script and as a package module."""
    import importlib.util

    source = Path(__file__).with_name("benchmark_adaptive_mtp.py")
    spec = importlib.util.spec_from_file_location(
        "_mlx2_benchmark_adaptive_mtp", source
    )
    if spec is None or spec.loader is None:
        raise ValueError("adaptive benchmark evaluator could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module.adaptive_qualification_evidence


def validate_adaptive_benchmark(path, initial):
    """Bind external MTP policy evidence to this exact candidate settings set."""
    if path is None:
        return None
    path = Path(path)
    raw = path.read_bytes()
    benchmark = json.loads(raw)
    if benchmark.get("schema") != "mlx2.adaptive-mtp-benchmark.v2":
        raise ValueError("adaptive benchmark has an unsupported schema")
    source = Path(__file__).with_name("benchmark_adaptive_mtp.py")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha256 != APPROVED_ADAPTIVE_BENCHMARK_SHA256:
        raise ValueError("local adaptive benchmark harness is not approved")
    harness = benchmark.get("benchmark_harness") or {}
    if (
        harness.get("name") != "scripts/benchmark_adaptive_mtp.py"
        or harness.get("sha256") != source_sha256
    ):
        raise ValueError("adaptive benchmark harness is not the approved source")
    adaptive_qualification_evidence = _adaptive_benchmark_evaluator()
    recorded = benchmark.get("adaptive_qualification") or {}
    throughput_tolerance = float(recorded.get("throughput_tolerance", 1.0))
    max_probe_fraction = float(recorded.get("max_probe_fraction", 1.0))
    min_bucket_rounds = int(recorded.get("min_bucket_rounds", 0))
    handoff_margin_threshold = float(
        (recorded.get("handoff") or {}).get(
            "ordinary_margin_failure_threshold_nats", 1.0
        )
    )
    # The batched handoff screen fails on a significant EXCESS over the control
    # arm, so a SMALLER alpha is the permissive direction: it takes a larger
    # excess to trip.  The bound is therefore a floor, unlike the others here.
    differential_alpha = float(recorded.get("differential_alpha", 0.0))
    if (
        not all(math.isfinite(value) for value in (
            throughput_tolerance, max_probe_fraction,
            handoff_margin_threshold, differential_alpha,
        ))
        or not 0.0 <= throughput_tolerance <= 1.0
        or not 0.0 <= max_probe_fraction <= 1.0
        or min_bucket_rounds < 64
        or not 0.0 <= handoff_margin_threshold <= 0.5
        or not 0.05 <= differential_alpha <= 1.0
    ):
        raise ValueError("adaptive benchmark used weaker qualification limits")
    recomputed = adaptive_qualification_evidence(
        benchmark,
        throughput_tolerance=throughput_tolerance,
        max_probe_fraction=max_probe_fraction,
        min_bucket_rounds=min_bucket_rounds,
        differential_alpha=differential_alpha,
    )
    if recomputed != recorded:
        raise ValueError("adaptive benchmark qualification evidence was modified")
    if not benchmark.get("passed") or not recomputed.get("passed"):
        raise ValueError("adaptive benchmark did not pass its qualification gate")
    qualification_arm = benchmark.get("qualification_arm", "adaptive")
    if qualification_arm not in {"adaptive", "handoff"}:
        raise ValueError("adaptive benchmark has an unsupported qualification arm")
    candidate_status = (
        benchmark.get("arms", {}).get(qualification_arm, {}).get("final_status")
        or {}
    )
    for key in ("runtime", "artifact", "settings"):
        if candidate_status.get(key) != initial.get(key):
            raise ValueError(
                f"benchmark {qualification_arm} arm {key} does not match "
                "the candidate server"
            )
    return benchmark, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema": benchmark["schema"],
        "adaptive_qualification": benchmark["adaptive_qualification"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8285")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--quiescence-timeout", type=float, default=5.0)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run the full unit suite, write a bound receipt to --output, and exit")
    parser.add_argument(
        "--preflight-pytest-arg",
        action="append",
        default=[],
        help=("Additional pytest argument recorded in a preflight-only receipt. "
              "Use only for an explicitly scoped historical/control source; "
              "--preflight-receipt refuses such a receipt as unit_tests evidence."),
    )
    parser.add_argument("--preflight-receipt", type=Path,
                        help=("Use a passing preflight receipt (full, or a delta over a full "
                              "base) instead of rerunning pytest"))
    parser.add_argument("--preflight-delta", action="store_true",
                        help=("Rerun only the test modules a tests/scripts change since "
                              "--base-preflight can reach; write a delta receipt to --output"))
    parser.add_argument("--base-preflight", type=Path,
                        help="Full preflight receipt a --preflight-delta builds on")
    parser.add_argument("--adapter-qualification", type=Path,
                        help="Source-bound media companion for this exact live route")
    parser.add_argument(
        "--adaptive-benchmark",
        type=Path,
        help=(
            "Passing three-arm MTP policy benchmark bound to the candidate's "
            "runtime, artifact and settings"
        ),
    )
    parser.add_argument(
        "--defer-long-context-to-matrix", type=Path, metavar="MANIFEST",
        help=("Defer only near-limit and shared-cohort context checks to the "
              "generated thermally controlled matrix manifest"),
    )
    parser.add_argument(
        "--kv-fidelity-report", type=Path,
        help=("Measured KV-quantization fidelity report "
              "(scripts/measure_kv_quant_fidelity.py) for an approximate-KV route"),
    )
    parser.add_argument("--require-feature", action="append", default=[], choices=["shared_qsa", "async_promotion", "indexed_qsa", "private_delta", "known_tail_prefetch", "indexed_fused_merge", "indexed_output_gate", "file_backed_ple", "compiled_ple", "pooled_qsa", "scatter_qsa", "fused_gdn_decode", "fused_gdn_verify", "fused_gdn_replay_rollback", "eager_dispatch", "fused_moe", "external_draft", "proposal_distribution", "paired_draft_cache", "segmented_transaction", "segmented_rollback", "prompt_lookup", "prompt_lookup_proposals", "prompt_lookup_rollback", "prompt_lookup_rotating_replay", "prompt_lookup_batched_verify", "adaptive_mtp_depth", "mtp_ordinary_handoff", "fly_verification", "self_mtp_copy_draft", "spomin_surgery", "approximate_kv", "approximate_kv_mtp", "approximate_kv_fidelity", "apc_interior_checkpoints", "apc_persistence", "apc_sessions"], help="Fail this candidate unless its mechanism actually executed")
    args = parser.parse_args()

    if args.preflight_only:
        write_preflight_receipt(args.output, pytest_args=args.preflight_pytest_arg)
        print(f"Preflight evidence: {args.output}", flush=True)
        return
    if args.preflight_delta:
        if args.base_preflight is None:
            parser.error("--preflight-delta needs --base-preflight")
        delta = write_preflight_delta(args.output, args.base_preflight)
        print(f"Preflight delta evidence: {args.output} "
              f"({len(delta['impacted'])} impacted test modules)", flush=True)
        return

    def get(path):
        with urlopen(args.url + path, timeout=30) as response:
            return json.load(response)

    thinking = {"default": False, "allowance": THINKING_BUDGET_TOKENS}  # set from /v1/status below
    # Checks rely on greedy decoding unless they say otherwise.  Models now
    # carry vendor sampling defaults (which may include presence or
    # repetition penalties that change greedy output), so every request pins
    # temperature 0 and neutral penalties explicitly; a check that exercises
    # sampling overrides them.  ``vendor_defaults=True`` sends the body as is.
    GREEDY_PINS = {
        "temperature": 0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }

    def post(body, stream=False, vendor_defaults=False):
        body = {"max_tokens": 64, **body} if vendor_defaults else {
            **GREEDY_PINS, "max_tokens": 64, **body
        }
        body = with_thinking_budget(body, thinking["default"], thinking["allowance"])
        if stream:
            body["stream"] = True
        response = urlopen(
            Request(
                args.url + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=args.timeout,
        )
        if stream:
            return response
        with response:
            return json.load(response)

    def post_or_refusal(body):
        # An atomic cohort refused under memory pressure answers 429 by
        # design.  That is evidence for a failed check; raising discarded
        # every other check of the run.
        try:
            return post(body)
        except HTTPError as error:
            return {"refused": http_refusal(error)}

    def post_json(path, body):
        with urlopen(
            Request(
                args.url + path,
                method="POST",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=args.timeout,
        ) as response:
            return json.load(response)

    def prompt(text, **kw):
        return {"messages": [{"role": "user", "content": text}], **kw}

    def content(response):
        return normalize_chat_content(response["choices"][0]["message"]["content"])

    initial = get("/v1/status")
    # Explicit raises, not assert: under python -O an assert is skipped, a
    # failed check was recorded and the run went on to write passed: true.
    if not (initial["healthy"] and initial["inflight"] == 0):
        raise AssertionError("server must start healthy and idle")
    selected_capabilities, scope_valid = selected_capability_scope(initial)
    adaptive_benchmark = validate_adaptive_benchmark(
        args.adaptive_benchmark, initial
    )
    thinking["default"] = bool(initial.get("thinking_default"))
    # A model may declare a larger allowance for its reasoning (Xing4.0 does);
    # never less than the harness default.
    thinking["allowance"] = max(
        THINKING_BUDGET_TOKENS, int(initial.get("thinking_allowance_tokens") or 0)
    )
    report = {
        "schema": "mlx2.serving-qualification.v1",
        "timestamp": time.time(),
        "runtime": initial["runtime"],
        "qualification_harness": qualification_harness_identity(),
        "artifact": initial["artifact"],
        "settings": initial["settings"],
        "coverage": QUALIFICATION_COVERAGE,
        "checks": {},
        "responses": {},
        "passed": False,
        "thinking_budget": {"thinking_default": thinking["default"],
                            "extra_completion_tokens": thinking["allowance"] if thinking["default"] else 0},
    }
    if args.adapter_qualification:
        report["adapter_qualification"] = read_adapter_qualification(
            args.adapter_qualification, initial
        )
    source_cases = source_text_cases(report.get("adapter_qualification"))
    if adaptive_benchmark is not None:
        report["adaptive_benchmark"] = adaptive_benchmark[1]
    context_delegation = (
        validate_context_matrix_delegation(
            args.defer_long_context_to_matrix, initial["max_context"]
        )
        if args.defer_long_context_to_matrix else None
    )
    report["coverage_scope"] = {
        "core_serving": "inline",
        "long_context": context_delegation or {"mode": "inline", "checks": list(LONG_CONTEXT_CHECKS)},
    }
    trusted_preflight = (validate_preflight_receipt(args.preflight_receipt, initial["runtime"])
                         if args.preflight_receipt else None)
    if trusted_preflight:
        report["preflight_receipt"] = trusted_preflight

    def check(name, condition, evidence):
        report["checks"][name] = {"passed": bool(condition), "evidence": evidence}
        print(f"{name}: {'PASS' if condition else 'FAIL'}", flush=True)
        if not condition:
            raise AssertionError(name)

    from mlx2.qualification import (
        contention_gated_not_observed,
        host_gate_record,
        host_gated_not_observed,
        required_feature_checks,
        selected_not_observed_features,
    )

    required_features = {
        name.removeprefix("feature_")
        for name in required_feature_checks(initial["settings"])
    } | set(args.require_feature)
    # Selected mechanisms this route cannot engage (or that are optional for
    # now) are recorded as "selected, not observed", never as passed checks.
    report["selected_not_observed"] = {
        name: {"status": "selected, not observed", "reason": reason}
        for name, reason in selected_not_observed_features(initial["settings"]).items()
        if name.removeprefix("feature_") not in required_features
    }
    try:
        unobservable = unobservable_features(required_features)
        check(
            "feature_observability",
            not unobservable,
            {"required": sorted(required_features), "unobservable": unobservable},
        )
        check(
            "capability_scope",
            scope_valid,
            {"selected": sorted(selected_capabilities),
             "implemented": initial.get("implemented_capabilities")},
        )
        if trusted_preflight:
            check("unit_tests", True, trusted_preflight)
        else:
            tests = subprocess.run(
                [sys.executable, "-m", "pytest"], capture_output=True, text=True
            )
            check("unit_tests", tests.returncode == 0, tests.stdout + tests.stderr)
        hermes_text = "Reply with exactly HERMES_READY"
        hermes_request = prompt(hermes_text, options={"num_ctx": 262144},
                                reasoning_effort="none", think=False)
        if source_cases is not None:
            hermes_request.update(max_tokens=16, logprobs=True)
        hermes = post(hermes_request)
        hermes_output_ok = (
            source_token_response_matches(
                hermes, source_cases["hermes_client"], prompt_text=hermes_text
            ) if source_cases is not None else content(hermes) == "HERMES_READY"
        )
        check("hermes_client", hermes_output_ok and
              hermes["mlx2"]["request_controls"]["thinking"] is False, hermes)
        probability_request = prompt("Write a short sentence about compilers.", max_tokens=8)
        probability_reference = post(probability_request)
        probability_response = post({**probability_request, "logprobs": True,
                                     "top_logprobs": 3, "response_format": {"type": "text"}})
        entries = probability_response["choices"][0].get("logprobs", {}).get("content", [])
        def valid_probability_entry(entry):
            alternatives = entry.get("top_logprobs", [])
            return (isinstance(entry.get("id"), int) and entry["id"] >= 0
                    and isinstance(entry.get("token"), str)
                    and math.isfinite(entry["logprob"]) and entry["logprob"] <= 1e-6
                    and len(alternatives) == 3
                    and all(isinstance(item.get("id"), int) and item["id"] >= 0
                            and isinstance(item.get("token"), str)
                            and math.isfinite(item["logprob"]) and item["logprob"] <= 1e-6
                            for item in alternatives)
                    and [item["logprob"] for item in alternatives]
                        == sorted((item["logprob"] for item in alternatives), reverse=True)
                    and any(item["id"] == entry["id"] and item["logprob"] == entry["logprob"]
                            for item in alternatives))
        with post({**probability_request, "logprobs": True, "top_logprobs": 3}, stream=True) as response:
            probability_wire = response.read().decode()
        probability_chunks = [json.loads(line[6:]) for line in probability_wire.splitlines()
                              if line.startswith("data: ") and line != "data: [DONE]"]
        streamed_entries = [entry for chunk in probability_chunks for choice in chunk.get("choices", [])
                            for entry in choice.get("logprobs", {}).get("content", [])]
        streamed_usage = next(chunk["usage"] for chunk in reversed(probability_chunks) if "usage" in chunk)
        check("logprobs", content(probability_response) == content(probability_reference)
              and len(entries) == probability_response["usage"]["completion_tokens"] > 0
              and len(streamed_entries) == streamed_usage["completion_tokens"] > 0
              and all(valid_probability_entry(entry) for entry in entries + streamed_entries)
              and probability_response["mlx2"]["request_controls"]["logprob_semantics"].startswith("execution_target:")
              and probability_wire.rstrip().endswith("data: [DONE]"),
              {"reference": probability_reference, "response": probability_response,
               "stream_entries": streamed_entries, "stream_usage": streamed_usage})
        sampling_request = prompt("Name three fruits.", temperature=0.7, top_p=0.9, top_k=24,
            min_p=0.05, repetition_penalty=1.05, presence_penalty=0.1,
            frequency_penalty=0.1, seed=237)
        sampling_a, sampling_b = post(sampling_request), post(sampling_request)
        report["responses"]["sampling_controls"] = [sampling_a, sampling_b]
        check("sampling_controls", content(sampling_a) == content(sampling_b) and bool(content(sampling_a))
            and sampling_b["mlx2"]["request_controls"]["sampling"].get("min_p") == 0.05,
            [sampling_a["mlx2"], sampling_b["mlx2"]])
        # Vendor sampling defaults: a request that sets no sampling field gets
        # the adapter's declared profile, recorded with its sources.
        declared = initial.get("sampling_defaults")
        defaults_request = prompt("Name three fruits.", max_tokens=32, seed=311)
        defaults_a = post(defaults_request, vendor_defaults=True)
        defaults_b = post(defaults_request, vendor_defaults=True)
        defaults_record = defaults_a["mlx2"]["request_controls"].get("sampling_defaults") or {}
        defaults_effective = defaults_a["mlx2"]["request_controls"].get("effective_sampling") or {}
        declared_values = (
            (declared or {}).get("profiles", {}).get(defaults_record.get("profile"), {}).get("values", {})
        )
        check(
            "sampling_defaults",
            defaults_record.get("schema") == "mlx2.sampling-defaults.v1"
            and defaults_record.get("explicit") == []
            and (
                declared is None
                or (
                    defaults_record.get("applied") == declared_values
                    and all(defaults_effective.get(k) == v for k, v in declared_values.items())
                    and set(defaults_record.get("sources", {})) == set(declared_values)
                )
            )
            and content(defaults_a) == content(defaults_b)
            and bool(content(defaults_a)),
            {"declared": declared, "a": defaults_a["mlx2"], "b": defaults_b["mlx2"]},
        )
        progressive_tile = (
            initial["settings"].get("execution_policy") or {}
        ).get("progressive_verification_tile")
        if type(progressive_tile) is int:
            # This prompt produced a stable, tile-spanning Muse/DFlash chain
            # in the integrated M3 parity run.  Its request receipt is an
            # immediate product-level assertion; the final run-local counter
            # delta below remains the qualification authority.
            num_draft = int(
                (initial["settings"].get("execution_policy") or {}).get(
                    "num_draft", progressive_tile + 1
                )
            )
            progressive_probe = post(
                prompt(
                    "Explain why a bounded dataflow scheduler should preserve causal state.",
                    max_tokens=num_draft + 2,
                    seed=919,
                    temperature=0,
                )
            )
            report["responses"]["progressive_verification"] = progressive_probe
            progressive_receipt = (
                progressive_probe.get("mlx2", {}).get("speculation", {}).get(
                    "progressive_verification"
                ) or {}
            )
            check(
                "progressive_verification_probe",
                bool(content(progressive_probe))
                and progressive_receipt.get("selected") is True
                and progressive_receipt.get("observed_used") is True
                and progressive_receipt.get("verification_tile") == progressive_tile
                and progressive_receipt.get("full_nonfinal_tiles", 0) > 0
                and progressive_receipt.get("state_promotion")
                == "request_private_then_atomic_lane_publish",
                progressive_probe.get("mlx2"),
            )
        cold_text_prompt = "Reply with exactly MLX2_READY"
        request = prompt(cold_text_prompt)
        if source_cases is not None:
            request.update(max_tokens=16, logprobs=True)
        cold, warm = post(request), post(request)
        report["responses"].update(cold=cold, warm=warm)
        cold_output_ok = (
            source_token_response_matches(
                cold, source_cases["cold_text"], prompt_text=cold_text_prompt
            ) if source_cases is not None else content(cold) == "MLX2_READY"
        )
        check("cold_text", cold_output_ok, cold["mlx2"])
        check(
            "warm_prefix",
            content(warm) == content(cold)
            and warm["mlx2"]["cached_tokens"] > 0
            and (source_cases is None or source_token_response_matches(
                warm, source_cases["cold_text"], prompt_text=cold_text_prompt
            )),
            warm["mlx2"],
        )
        if initial["settings"].get("disk_cache") is True:
            session_id = f"qualification-{int(time.time())}"
            session_request = prompt(
                "Reply with exactly APC_SESSION_READY",
                session_id=session_id,
                max_tokens=32,
            )
            seeded = post(session_request)
            parked = post_json(
                f"/v1/apc/sessions/{session_id}/park", {"ttl_seconds": 60}
            )
            # A park that arrives on an HTTP thread is completed by the
            # generation worker (MLX streams are thread-local); resuming
            # before the spill lands would cancel it and nothing would park.
            park_deadline = time.monotonic() + 10.0
            while parked.get("state") != "disk" and time.monotonic() < park_deadline:
                time.sleep(0.05)
                parked = get(f"/v1/apc/sessions/{session_id}")
            resumed = post_json(
                f"/v1/apc/sessions/{session_id}/resume", {}
            )
            deadline = time.monotonic() + 5.0
            state = get(f"/v1/apc/sessions/{session_id}")
            while state.get("state") != "resident" and time.monotonic() < deadline:
                time.sleep(0.05)
                state = get(f"/v1/apc/sessions/{session_id}")
            restored = post(session_request)
            report["responses"]["apc_session_lifecycle"] = {
                "seeded": seeded,
                "parked": parked,
                "resumed": resumed,
                "state": state,
                "restored": restored,
            }
        if "grammar" in initial.get("capabilities", []):
            structured_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"const": "yes"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }
            # Greedy decoding makes the admissible-token mask exact (the
            # receipt's tail_mass_bound must be 0); sampled structured output
            # is numerically bounded and reports the bound it incurred.
            structured = post(
                prompt(
                    "Return a JSON object whose answer is yes.",
                    response_format=structured_format,
                    temperature=0,
                )
            )
            try:
                structured_value = json.loads(content(structured))
            except (TypeError, ValueError, json.JSONDecodeError):
                structured_value = None
            structured_receipt = structured["mlx2"]["request_controls"]["structured_output"]
            check(
                "structured_output",
                structured_value == {"answer": "yes"}
                and structured_receipt.get("kind") == "json_schema"
                and structured_receipt.get("enforced") is True
                and structured_receipt.get("tail_mass_bound") == 0.0,
                # ``engine`` is recorded, not a pass condition.
                {"engine": structured_receipt.get("engine"), "response": structured},
            )
            sampled = post(
                prompt(
                    "Return a JSON object whose answer is yes.",
                    response_format=structured_format,
                    temperature=0.7,
                    seed=7,
                )
            )
            try:
                sampled_value = json.loads(content(sampled))
            except (TypeError, ValueError, json.JSONDecodeError):
                sampled_value = None
            sampled_receipt = sampled["mlx2"]["request_controls"]["structured_output"]
            check(
                "structured_output_sampled",
                sampled_value == {"answer": "yes"}
                and sampled_receipt.get("enforced") is True
                and 0.0 <= float(sampled_receipt.get("tail_mass_bound", 1.0)) <= 0.25,
                {"engine": sampled_receipt.get("engine"), "response": sampled},
            )
            thinking_passed, thinking_evidence = run_structured_thinking_probe(post, initial)
            check("structured_output_thinking", thinking_passed, thinking_evidence)
        with post(request, stream=True) as response:
            wire = response.read().decode()
        chunks = [
            json.loads(line[6:])
            for line in wire.splitlines()
            if line.startswith("data: {")
        ]
        text = "".join(
            c["choices"][0].get("delta", {}).get("content", "") for c in chunks
        )
        check(
            "stream",
            normalize_chat_content(text)
            == (content(cold) if source_cases is not None else "MLX2_READY")
            and wire.endswith("data: [DONE]\n\n")
            and "mlx2" in chunks[-1],
            chunks,
        )
        if "tools" in selected_capabilities:
            # A declared known model behaviour (artifact-scoped, exact failure
            # shape) is recorded as such, never as a pass, and the run goes on.
            tools_probe = run_tools_probe(post, initial["artifact"])
            record_tools_probe(report, tools_probe)
            tools = TOOLS_PROBE_TOOLS
            tool_request = tools_probe.request
            choice = tools_probe.call_response["choices"][0]
            calls = choice["message"].get("tool_calls", [])
            followup = post(
                {
                    "tools": tools,
                    "messages": tool_request["messages"]
                    + [
                        choice["message"],
                        {
                            "role": "tool",
                            "tool_call_id": calls[0]["id"],
                            "name": "weather",
                            "content": '{"temperature_c": 21, "condition": "sunny"}',
                        },
                    ],
                }
            )
            check("tool_roundtrip", "21" in content(followup), followup)
        if "reasoning" in selected_capabilities:
            reasoning_budget = (
                REASONING_PROBE_THINKING_BUDGET
                if (initial.get("structured_output") or {}).get("thinking_deferral") is True
                else None
            )
            thought, thought_attempts = run_reasoning_probe(
                post, thinking_budget=reasoning_budget
            )
            check(
                "reasoning",
                reasoning_response_passes(thought),
                {"attempts": thought_attempts},
            )
        # Unrelated to reasoning, so thinking is off: a thinking-default model
        # (Laguna) otherwise quotes MLX2_READY in its reasoning, the stop
        # matches there, and the answer is empty (the pass-3 triage's
        # "checks consume only reasoning" harness class).
        stop_witness = (interior_stop_witness(content(cold))
                        if source_cases is not None else None)
        stop_text = stop_witness[0] if stop_witness else "_READY"
        stopped = post(prompt(
            cold_text_prompt, stop=stop_text,
            reasoning_effort="none", think=False,
        ))
        check(
            "stop",
            (stop_witness is not None if source_cases is not None else True)
            and content(stopped) == (stop_witness[1] if stop_witness else "MLX2")
            and stopped["choices"][0]["finish_reason"] == "stop",
            stopped,
        )
        seeded = prompt(
            "Give three short facts about stars.",
            temperature=0.7,
            seed=123,
            max_tokens=64,
        )
        first, second = post(seeded), post(seeded)
        check(
            "seed_repeat",
            content(first) == content(second),
            {"first": first, "second": second},
        )
        if "continuous_batch" in selected_capabilities:
            # Four concurrent requests cover row lifetimes and observed compute width.
            # Fixed 160-token lanes measure batching, not reasoning: thinking is
            # turned off so a thinking-default model fills the same budget.
            # Four distinct subjects: APCv2 serves identical prompts one after
            # another (same-prefix wait), so four copies of one prompt reached
            # only width 3 and could never exceed a width-3 MTP handoff
            # (qualify-f4cdb698-uncensored, 2026-10-02).
            batch_requests = [
                prompt(
                    f"Explain how {subject} works in detail.", max_tokens=160,
                    reasoning_effort="none", think=False,
                )
                for subject in BATCH_SUBJECTS
            ]
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=len(batch_requests)) as pool:
                batch = list(pool.map(post, batch_requests))
            elapsed = time.monotonic() - start
            report["responses"]["batch"] = batch
            widths = sorted(
                {
                    width
                    for r in batch
                    for width in observed_compute_widths(r["mlx2"])
                }
            )
            # The prompt-lookup route reports its verify width in the speculation
            # receipt: shared (>= 2) where batched verification engages, 1 on
            # models that keep the per-lane driver, whose concurrency evidence is
            # then wall-time overlap of the four lanes.
            lane_seconds = sum(r["mlx2"]["elapsed_seconds"] for r in batch)
            prompt_lookup_route = initial["settings"].get("speculation") == "prompt_lookup"
            if prompt_lookup_route:
                widths = sorted(set(widths) | {
                    int((r["mlx2"].get("speculation") or {}).get("target_width") or 0) for r in batch
                })
            check(
                "batch",
                all(r["usage"]["completion_tokens"] == 160 and content(r) for r in batch)
                and (
                    max(widths, default=0) >= 2
                    or (prompt_lookup_route and elapsed < 0.6 * lane_seconds)
                ),
                {
                    "observed_widths": widths,
                    "wall_seconds": elapsed,
                    "lane_seconds": lane_seconds,
                    "prompt_lookup_route": prompt_lookup_route,
                    "aggregate_tokens_per_second": 640 / elapsed,
                    "request_receipts": [r["mlx2"] for r in batch],
                },
            )
            # Both pairs must do identical work for their wall times to compare:
            # thinking off (a thinking-default model otherwise gets the reasoning
            # allowance and answered 906 and 1,490 tokens on North) and EOS held
            # until the 64-token budget, as the batch check above does.
            mixed_prompts = [
                prompt(
                    "Explain how a compiler works, in numbered sections.",
                    max_tokens=64, min_tokens=64,
                    reasoning_effort="none", think=False,
                ),
                prompt(
                    "Explain how a database transaction works, in numbered sections.",
                    max_tokens=64, min_tokens=64,
                    reasoning_effort="none", think=False,
                ),
            ]
            # The sequential warm-up is also the timing reference: two warm
            # requests together must finish faster than the same pair one after
            # the other.  A fixed wall clock failed models that simply answer at
            # length (Xing4.0: ~2.2K tokens each, 45-48 s concurrent).
            # Both pairs stream, so their client work is identical too, and each
            # token delta of the concurrent pair is stamped on the pair's shared
            # clock: a per-lane route shows its overlap by interleaved tokens.
            def post_streamed(item, clock_start):
                with post(item, stream=True) as response:
                    return read_streamed_chat(
                        response, lambda: time.monotonic() - clock_start
                    )

            sequential_start = time.monotonic()
            for item in mixed_prompts:
                post_streamed(item, sequential_start)
            sequential = time.monotonic() - sequential_start
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=2) as pool:
                streamed = list(
                    pool.map(lambda item: post_streamed(item, start), mixed_prompts)
                )
            concurrent = time.monotonic() - start
            mixed = [response for response, _ in streamed]
            lane_token_seconds = [arrivals for _, arrivals in streamed]
            check(
                "mixed_warm",
                all(content(r) and r["mlx2"]["cached_tokens"] > 0
                    and r["usage"]["completion_tokens"] == 64 for r in mixed)
                and mixed_warm_timing_passes(
                    concurrent,
                    sequential,
                    [r["mlx2"] for r in mixed],
                    lane_token_seconds=lane_token_seconds,
                ),
                {
                    "concurrent_seconds": concurrent,
                    "sequential_seconds": sequential,
                    "lane_token_seconds": lane_token_seconds,
                    "lane_overlap_shares": mixed_warm_lane_overlap_shares(
                        lane_token_seconds
                    ),
                    "receipts": [r["mlx2"] for r in mixed],
                },
            )
        if context_delegation:
            check("long_context_delegation", True, context_delegation)
        # Keep the near-limit cohort alive long enough to join before decoding
        # finishes. A three-token sentinel can complete as B1 before its sibling
        # is prepared and therefore cannot qualify shared-prefix batching.
        # Reserve enough room for the output budget and model-specific chat
        # template overhead.  Muse's direct recipient suffix makes a fixed
        # 128-token reserve too small even though every repeated ``data ``
        # fragment is one token.
        n = near_limit_prompt_floor(initial["max_context"])
        def long_context_request(
            context_cap, completion_budget=LONG_CONTEXT_COMPLETION_TOKENS
        ):
            # These checks measure near-limit context serving inside a bounded
            # output budget. A model that thinks by default (North) would spend
            # the budget reasoning and return empty content.  min_tokens holds
            # EOS until the budget: near_limit_usage_passes requires the full
            # completion, and whether a model ends its answer early is not what
            # these checks measure (North, once its layer 0 was rotated, ended
            # a warm B2 answer at 39 of 64 tokens).
            source_near_case = (
                source_cases.get("near_context") if source_cases is not None
                and context_cap == initial["max_context"] else None
            )
            return prompt(
                source_near_case["prompt"] if source_near_case is not None
                else long_context_prompt(context_cap),
                max_tokens=completion_budget,
                min_tokens=completion_budget,
                **({"logprobs": True} if source_near_case is not None else {}),
                reasoning_effort="none", think=False,
            )
        if not context_delegation:
            if "continuous_batch" in selected_capabilities:
                # Context capacity and concurrency are separate domains. The near-cap
                # B1 check below reaches the full cap. Exercise B2 shared/private
                # state first, up to 131K where two complete copies plus reserve fit
                # this deployment profile. Running B2 first prevents the deliberate
                # near-cap B1 allocation from perturbing cohort admission.
                shared_context = min(initial["max_context"], 131072)
                shared_completion_budget = shared_qsa_completion_budget(
                    initial["settings"], shared_context
                )
                shared_request = long_context_request(
                    shared_context, shared_completion_budget
                )
                # The cohort check below asserts a warm hit for both members, so
                # the shared prompt must be primed first regardless of whether the
                # cohort context equals the served cap: an atomic cohort admits
                # both members together, so neither can warm the other.
                primed = post(shared_request)
                check("shared_cohort_priming", long_context_answer_passes(content(primed))
                      and primed["usage"]["completion_tokens"]
                          == shared_completion_budget, reply_evidence(primed))
                report["shared_cohort_domain"] = {
                    "context_limit": shared_context, "requested_width": 2,
                    "completion_budget": shared_completion_budget,
                    "primed_separately": True,
                }
                # Exact same immutable P-1 checkpoint admits a shared-prefix cohort.
                shared_cohort = {"id": "qualification-shared-prefix-b2", "size": 2}
                shared_pair = [
                    {**shared_request, "batch_cohort": shared_cohort}
                    for _ in range(2)
                ]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    shared = list(pool.map(post_or_refusal, shared_pair))
                check(
                    "shared_warm_requests",
                    all("refused" not in r
                        and long_context_answer_passes(content(r))
                        and near_limit_usage_passes(
                            r["usage"], shared_context, shared_completion_budget
                        )
                        and r["mlx2"]["cached_tokens"]
                            > near_limit_prompt_floor(shared_context) - LONG_CONTEXT_CACHE_TOLERANCE
                        and r["mlx2"]["request_controls"].get("batch_cohort")
                            == shared_cohort
                        for r in shared),
                    [reply_evidence(r) for r in shared],
                )
            long_request = long_context_request(initial["max_context"])
            long = post(long_request)
            repeated = post(long_request)
            check(
                "context",
                long_context_answer_passes(content(long))
                and long_context_answer_passes(content(repeated))
                and (source_cases is None or (
                    source_token_response_matches(
                        long, source_cases["near_context"],
                        prompt_text=source_cases["near_context"]["prompt"]
                    )
                    and source_token_response_matches(
                        repeated, source_cases["near_context"],
                        prompt_text=source_cases["near_context"]["prompt"]
                    )
                ))
                and near_limit_usage_passes(long["usage"], initial["max_context"])
                and repeated["mlx2"]["cached_tokens"] > n - LONG_CONTEXT_CACHE_TOLERANCE,
                {"cold": reply_evidence(long), "warm": reply_evidence(repeated)},
            )
            try:
                post(prompt(long_context_filler(initial["max_context"]), max_tokens=16))
                rejected = False
            except HTTPError as error:
                rejected = error.code == 400
            check(
                "context_bound",
                rejected,
                "prompt plus output beyond limit rejected with400",
            )
        before = get("/v1/status")["counts"].get("cancelled", 0)
        response = post(
            prompt(
                "Write a very long guide to compiler optimization.",
                max_tokens=cancellation_probe_budget(initial["max_context"]),
                reasoning_effort="none",
                think=False,
            ),
            stream=True,
        )
        response.readline()
        response.close()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = get("/v1/status")
            if state["inflight"] == 0:
                break
            time.sleep(0.2)
        check(
            "cancel",
            state["inflight"] == 0 and state["counts"].get("cancelled", 0) > before,
            state["counts"],
        )
        recovered = post(request)
        check(
            "recovery",
            content(recovered) == (content(cold) if source_cases is not None else "MLX2_READY")
            and (source_cases is None or source_token_response_matches(
                recovered, source_cases["cold_text"], prompt_text=cold_text_prompt
            ))
            and get("/health")["status"] == "ok",
            recovered,
        )
        spomin_policy = initial["settings"].get("spomin_live_surgery") or {}
        if spomin_policy.get("enabled"):
            # A nonce keeps the first run cold.  The instruction sits in the
            # protected recent segment.  The repeat must warm-hit the *exact*
            # pre-surgery boundary (more cached tokens than the compacted
            # state even holds) and be compacted again: the prefix cache only
            # ever sees exact state, and repeated long prompts skip prefill.
            import uuid

            filler = long_context_filler(int(spomin_policy["capacity_tokens"] * 0.8)) + " "
            spomin_request = prompt(
                f"nonce {uuid.uuid4().hex} " + filler + "Reply with exactly SPOMIN_OK",
                reasoning_effort="none", think=False, max_tokens=16,
            )
            spomin_runs = [post(spomin_request) for _ in range(2)]
            spomin_receipts = [run["mlx2"].get("spomin_live_surgery") or {} for run in spomin_runs]
            check(
                "spomin_surgery",
                all(receipt.get("status") == "applied"
                    and receipt.get("retained_tokens", 0) < receipt.get("source_tokens", 0)
                    for receipt in spomin_receipts)
                and spomin_runs[0]["mlx2"]["cached_tokens"] < 256
                and spomin_runs[1]["mlx2"]["cached_tokens"]
                    > spomin_receipts[1].get("retained_tokens", 1 << 60)
                and all("SPOMIN_OK" in content(run) for run in spomin_runs),
                {"receipts": spomin_receipts,
                 "content": [content(run) for run in spomin_runs],
                 "cached_tokens": [run["mlx2"]["cached_tokens"] for run in spomin_runs]},
            )
        forced = None
        if initial["settings"].get("prefill_scheduling"):
            # Only a route that selects SRPT runs the forcing load; every
            # other route keeps feature_prefill_scheduling unrequired.  The
            # load runs before the final status is sampled; its verdict is
            # checked with the feature checks below, so a failure there does
            # not hide the quiescence, lease and mechanism checks.
            import uuid

            forced, forcing_evidence = run_prefill_scheduling_forcing(
                get_status=lambda: get("/v1/status"),
                post=post,
                settings=initial["settings"],
                nonce=uuid.uuid4().hex,
            )
            report["prefill_scheduling_forcing"] = forcing_evidence
        final, quiescence = wait_for_quiescence(
            lambda: get("/v1/status"),
            timeout_seconds=args.quiescence_timeout,
        )
        report["quiescence"] = quiescence
        report["final_status"] = final
        check("quiescence", quiescence["passed"], quiescence)
        for name, result in route_mechanism_checks(initial, final).items():
            check(name, result["passed"], result["evidence"])
        if initial["settings"]["mtp"]:
            segmented = final["execution"]["segmented_mtp"]
            check(
                "mtp_execution",
                segmented["batched_target_forwards"] > 0
                and segmented["true_batched_engaged"] > 0
                and segmented["full_prefix_materializations"] == 0
                and segmented["failures"] == 0,
                final["execution"],
            )
        execution = final.get("execution", {})
        kv_fidelity = None
        approximate_settings = initial["settings"].get("approximate_kv") or {}
        if args.kv_fidelity_report is not None:
            from mlx2.runtime.kv_quant_fidelity import evaluate_fidelity_report

            kv_fidelity = evaluate_fidelity_report(
                json.loads(args.kv_fidelity_report.read_text()),
                operation=approximate_settings.get("operation"),
                adapter_fingerprint=initial["artifact"],
            )
            report["kv_fidelity"] = kv_fidelity
        # Lease and runtime checks come before the required-feature checks: a
        # feature whose evidence this run cannot produce (the adaptive MTP
        # benchmark in a smoke run) fails the run there, and must not hide
        # these two from its evidence.
        check(
            "cache_leases", final["apcv2"]["cow"]["active_leases"] == 0, final["apcv2"]
        )
        from mlx2.route_identity import recompute_runtime_identity
        from mlx2.serving import runtime_identity

        check(
            "runtime_stable",
            initial["runtime"] == final["runtime"] == recompute_runtime_identity(
                initial["runtime"], build=runtime_identity()
            ),
            initial["runtime"],
        )
        observed = feature_observations(
            final,
            kv_fidelity=kv_fidelity,
            adaptive_benchmark=(
                None if adaptive_benchmark is None else adaptive_benchmark[0]
            ),
            initial=initial,
        )
        report["feature_observations"] = observed
        latched = latched_before_run(initial)
        if forced is not None:
            check(
                "prefill_scheduling_forced_reorder",
                forced,
                report["prefill_scheduling_forcing"],
            )
        # Host-gated mechanisms (MLX2_MOE_NAX_GATHER: M5 only, canaried; the
        # NAX QSA prefill kernel: M5 only, probed) that
        # this host could not run are "selected, not observed", neither a
        # pass nor a failure.
        host_gated = host_gated_not_observed(
            final.get("execution") or {}, initial.get("execution") or {}
        )
        for feature in sorted(required_features & set(host_gated)):
            report["selected_not_observed"]["feature_" + feature] = {
                "status": "selected, not observed",
                "reason": host_gated[feature],
                "host_gated": True,
                # Re-evaluated against the serving host at load.
                "host_gate": host_gate_record(feature, final.get("execution") or {}),
            }
            print(f"feature_{feature}: SELECTED, NOT OBSERVED "
                  f"({host_gated[feature]})", flush=True)
        # Contention-gated mechanisms (the decode-fairness slice floor) this
        # run's load never reached: constructed, nothing to lift.
        contention_gated = contention_gated_not_observed(final, initial)
        for feature in sorted((required_features & set(contention_gated)) - set(host_gated)):
            report["selected_not_observed"]["feature_" + feature] = {
                "status": "selected, not observed",
                "reason": contention_gated[feature],
                "contention_gated": True,
            }
            print(f"feature_{feature}: SELECTED, NOT OBSERVED "
                  f"({contention_gated[feature]})", flush=True)
        for feature in sorted(required_features - set(host_gated) - set(contention_gated)):
            if feature in {"adaptive_mtp_depth", "mtp_ordinary_handoff"}:
                evidence = report.get("adaptive_benchmark")
            elif feature == "sp_qmm":
                evidence = {
                    "before": initial.get("sp_qmm"),
                    "after": final.get("sp_qmm"),
                    "routed_delta": observed["sp_qmm"],
                }
            elif feature == "qsdpa_verify_kernel":
                evidence = {
                    "before": initial.get("qsdpa_verify"),
                    "after": final.get("qsdpa_verify"),
                    "verify_delta": observed["qsdpa_verify_kernel"],
                }
            elif feature == "prefill_scheduling":
                evidence = {
                    "counters": prefill_scheduling_counters(final),
                    "forcing_load": report.get("prefill_scheduling_forcing"),
                }
            elif feature == "progressive_verification":
                keys = (
                    "external_progressive_verify_tile",
                    "external_progressive_verify_rounds",
                    "external_progressive_verify_launches",
                    "external_progressive_verify_target_rows",
                    "external_progressive_verify_rejections",
                    "external_progressive_verify_full_tiles",
                )
                evidence = {
                    "selected_tile": (
                        initial["settings"].get("execution_policy") or {}
                    ).get("progressive_verification_tile"),
                    "before": {
                        key: (initial.get("scheduler") or {}).get(key)
                        for key in keys
                    },
                    "after": {
                        key: (final.get("scheduler") or {}).get(key)
                        for key in keys
                    },
                    "observation": observed.get(feature, 0),
                    "probe": report["checks"].get(
                        "progressive_verification_probe"
                    ),
                }
            elif feature == "progressive_multilane_draft_cap":
                keys = (
                    "external_multilane_draft_cap",
                    "external_multilane_draft_cap_rounds",
                    "external_multilane_draft_cap_lanes",
                )
                evidence = {
                    "selected_cap": (
                        initial["settings"].get("execution_policy") or {}
                    ).get("progressive_multilane_draft_cap"),
                    "before": {
                        key: (initial.get("scheduler") or {}).get(key)
                        for key in keys
                    },
                    "after": {
                        key: (final.get("scheduler") or {}).get(key)
                        for key in keys
                    },
                    "observation": observed.get(feature, 0),
                }
            elif feature in {"decode_first", "decode_fairness_slice_floor"}:
                prefix = "decode_first_" if feature == "decode_first" else "decode_fairness_"
                evidence = {
                    "before": {k: v for k, v in (initial.get("scheduler") or {}).items()
                               if k.startswith(prefix)},
                    "after": {k: v for k, v in (final.get("scheduler") or {}).items()
                              if k.startswith(prefix)},
                    "delta": observed.get(feature, 0),
                }
            elif feature in latched:
                evidence = {"reason": latched[feature], "execution": execution}
            else:
                evidence = execution
            # A feature without an observation is a failed check, never a
            # KeyError that discards the whole run's evidence.
            check("feature_" + feature, observed.get(feature, 0) > 0, evidence)
        report["passed"] = True
    finally:
        if "final_status" not in report:
            try:
                report["failure_status"] = get("/v1/status")
            except Exception as exc:
                report["failure_status_error"] = str(exc)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Evidence: {args.output}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--run-import-guards"]:
        sys.exit(run_import_guards(sys.argv[2:]))
    main()
