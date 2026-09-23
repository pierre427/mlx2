#!/usr/bin/env python3
"""Exercise an exclusively owned candidate server and write its route evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


QUALIFICATION_HARNESS_SCHEMA = "mlx2.qualification-harness.v1"
APPROVED_ADAPTIVE_BENCHMARK_SHA256 = (
    "3a371843134d75c9e3e09e8532f2a15a07284e98a68a731c846ae8ed2e9bef4b"
)
PREFLIGHT_SCHEMA = "mlx2.qualification-preflight.v1"
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
    tail of the near-limit prompt was attended.
    """
    text = text.strip()
    return text.startswith("LONG_READY") or "compiler" in text.lower()


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


def qualification_harness_identity(path=None):
    source = Path(__file__) if path is None else Path(path)
    return {
        "schema": QUALIFICATION_HARNESS_SCHEMA,
        "name": "scripts/qualify_serving.py",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _tree_sha256(root, pattern):
    digest = hashlib.sha256()
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


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
        "test_source_sha256": _tree_sha256(root, "tests/**/*.py"),
    }


FULL_SUITE_PYTEST_ARGS = ("-m", "pytest")


def write_preflight_receipt(path, *, pytest_args=None, run=subprocess.run,
                            identity_fn=preflight_identity):
    command = [sys.executable, *FULL_SUITE_PYTEST_ARGS, *(pytest_args or [])]
    completed = run(command, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "passed": completed.returncode == 0,
        "timestamp": time.time(),
        "identity": identity_fn(),
        "test_command": command,
        "returncode": completed.returncode,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output_tail": output[-16000:],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not receipt["passed"]:
        raise AssertionError(f"full unit suite failed; see {path}")
    return receipt


def validate_preflight_receipt(path, active_runtime, *, identity_fn=preflight_identity):
    path = Path(path)
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != PREFLIGHT_SCHEMA or receipt.get("passed") is not True:
        raise AssertionError(f"preflight receipt is absent or failed: {path}")
    expected = identity_fn()
    if receipt.get("identity") != expected:
        raise AssertionError("preflight receipt does not match current git/runtime source/harness identity")
    if receipt["identity"].get("runtime") != active_runtime:
        raise AssertionError("preflight receipt runtime does not match active server")
    command = receipt.get("test_command")
    # --preflight-pytest-arg exists for scoped historical/control receipts.
    # Any extra argument can scope the run (--collect-only, -k, a test path,
    # --lf, --deselect, --ignore) and still exit 0 with passed=True, so only
    # the unscoped default command stands in for the unit_tests check.
    if (
        not isinstance(command, list)
        or len(command) != 1 + len(FULL_SUITE_PYTEST_ARGS)
        or not isinstance(command[0], str)
        or tuple(command[1:]) != FULL_SUITE_PYTEST_ARGS
    ):
        raise AssertionError(
            "preflight receipt did not run the full unit suite: "
            f"{command!r}: {path}"
        )
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "identity": expected, "test_command": command, "passed": True}


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


def feature_observations(final, kv_fidelity=None, adaptive_benchmark=None):
    execution = final.get("execution", {})
    segmented = execution.get("segmented_mtp", {})
    indexed = execution.get("indexed_qsa", {}).get("counts", {})
    scheduler = final.get("scheduler", {})
    fly_receipt_relaxed = 0
    for receipt in final.get("recent_receipts", ()):
        for key in ("mtp", "speculation"):
            verification = receipt.get(key) or {}
            if verification.get("verification") == "fly":
                fly_receipt_relaxed = max(
                    fly_receipt_relaxed,
                    int(verification.get("relaxed_accepts", 0)),
                )
    settings = final.get("settings", {})
    levers = execution.get("round_levers", {})
    ple_tables = execution.get("ple_tables", [])
    ple_compile = execution.get("ple_compile", {})
    ple_compile_counts = ple_compile.get("counts", {})
    fused_gdn = execution.get("fused_gdn", {})
    moe = execution.get("moe", {})
    compiled_ple_healthy = (
        ple_compile.get("enabled") is True
        and ple_compile_counts.get("builds", 0) > 0
        and ple_compile_counts.get("hits", 0) > 0
        and all(ple_compile_counts.get(name, 0) == 0
                for name in ("fallbacks", "overflow", "skips"))
    )
    # External-draft transactions publish their counters through the scheduler.
    # Native self-MTP publishes the same logical evidence through its segmented
    # execution counters instead.  A zero/partial accepted draft is a real
    # rollback: commit keeps only the accepted prefix (and target token), while
    # the unaccepted speculative suffix is discarded.  transaction_rejections
    # is narrower; it records an entire delivery/branch rejection and is not
    # expected during healthy nonterminal serving.
    native_segmented = settings.get("mtp") is True
    if native_segmented:
        segmented_transactions = segmented.get("transaction_branches", 0)
        segmented_rollbacks = (
            segmented.get("accepted_zero", 0)
            + segmented.get("accepted_partial", 0)
        )
    else:
        segmented_transactions = scheduler.get("segmented_transactions", 0)
        segmented_rollbacks = scheduler.get("segmented_rollbacks", 0)
    counts = final.get("counts", {})
    apcv2 = final.get("apcv2", {})
    apc_lifetime = apcv2.get("lifetime", {})
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
        "indexed_fused_merge": int(execution.get("indexed_qsa", {}).get("fused_merge", {}).get("engaged", False)),
        "indexed_output_gate": int(execution.get("indexed_qsa", {}).get("fused_merge", {}).get("gate_engaged", False)),
        "shared_qsa": segmented.get("shared_qsa_batched_selections", 0),
        "async_promotion": segmented.get("async_qsa_promotion_engaged", 0),
        "indexed_qsa": indexed.get("engaged", 0),
        "private_delta": segmented.get("private_delta_attention_calls", 0),
        "known_tail_prefetch": levers.get("ple_tail_prefetch_tables", 0),
        "file_backed_ple": sum(table.get("lookups", 0) for table in ple_tables),
        "compiled_ple": ple_compile_counts.get("hits", 0) if compiled_ple_healthy else 0,
        "pooled_qsa": levers.get("qsa_pooled_key_cache_hits", 0),
        "scatter_qsa": levers.get("qsa_scatter_chosen_calls", 0),
        "fused_gdn_decode": fused_gdn.get("fused_calls", 0),
        "fused_gdn_verify": fused_gdn.get("verify_calls", 0),
        "fused_gdn_replay_rollback": fused_gdn.get("replay_rollback_calls", 0),
        "eager_dispatch": levers.get("eager_async_evals", 0),
        "fused_moe": (sum(moe.get("dispatches", {}).values())
                      if moe.get("fused_gate_up_layers", 0) > 0 else 0),
        "external_draft": (scheduler.get("external_rounds", 0)
                           if scheduler.get("draft_fallbacks", 0) == 0 else 0),
        "proposal_distribution": scheduler.get("proposed_tokens", 0),
        "paired_draft_cache": scheduler.get("paired_cache_resumes", 0),
        "segmented_transaction": segmented_transactions,
        "segmented_rollback": segmented_rollbacks,
        "prompt_lookup": scheduler.get("pld_retrieval_cycles", 0),
        "prompt_lookup_proposals": scheduler.get("pld_proposed", 0),
        "prompt_lookup_rollback": scheduler.get("pld_rollbacks", 0),
        "prompt_lookup_batched_verify": scheduler.get("pld_batched_rounds", 0),
        "prompt_lookup_rotating_replay": scheduler.get("pld_rotating_replay_rounds", 0),
        # The bound policy benchmark proves only the features it selected.
        # Adaptive depth still needs bounded exploration, complete sampling,
        # and conditional recovery; handoff uses its independent fixed-depth
        # event/correctness/throughput gate.
        "adaptive_mtp_depth": adaptive_observed,
        "mtp_ordinary_handoff": handoff_observed,
        "self_mtp_copy_draft": scheduler.get("self_mtp_copy_rounds", 0),
        "fly_verification": max(
            int(scheduler.get("fly_relaxed_accepts", 0)), fly_receipt_relaxed
        ),
        "spomin_surgery": final.get("spomin_live_surgery", {}).get("counts", {}).get("applied", 0),
        "approximate_kv": final.get("approximate_kv", {}).get("applied", 0),
        # Int8 NAX prefill: engaged GEMMs.  required_feature_checks demands
        # feature_int8_prefill whenever the policy is enabled, so the
        # observation key must exist or the qualifier raises KeyError.
        "int8_prefill": final.get("int8_prefill", {}).get("counts", {}).get("engaged_calls", 0),
        "verify_bitexact": (
            (final.get("verify_bitexact") or {}).get("dispatches", 0)
            if (final.get("verify_bitexact") or {}).get("active") is True
            else 0
        ),
        "approximate_kv_mtp": final.get("approximate_kv", {}).get("mtp_lanes", 0),
        # Offline measurement (scripts/measure_kv_quant_fidelity.py) judged by
        # runtime/kv_quant_fidelity.py; 1 only when that verdict passed.
        "approximate_kv_fidelity": int(bool((kv_fidelity or {}).get("passed"))),
        "apc_interior_checkpoints": min(
            counts.get("apc_interior_checkpoints_captured", 0),
            counts.get("apc_interior_checkpoints_published", 0),
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
            idle_disk.get("parks", 0),
            idle_disk.get("resumes", 0),
            idle_disk.get("prefetch_restores_ok", 0),
            idle_disk.get("prefetch_hits", 0),
        ),
        # Each observation below counts the mechanism engaging, never the
        # policy being selected.  A hybrid rolling checkpoint must have been
        # published and later resumed from; the KV route's only publication
        # is a cancelled prefill's partial cache, which is its evidence.
        "apc_rolling_checkpoints": max(
            min(
                counts.get("apc_rolling_checkpoints_published", 0),
                apc_lifetime.get("rolling_hits", 0),
            ),
            counts.get("apc_rolling_checkpoints_cancel_published", 0),
        ),
        # A junction is useful only when a later diverging request hit it.
        "apc_junction_checkpoints": min(
            counts.get("apc_junction_checkpoints_published", 0),
            apc_lifetime.get("junction_hits", 0),
        ),
        # A bypass is an SRPT reorder (an older prompt was overtaken); a
        # forced bypass is bypass-capped service.  Slice clamps are neither.
        "prefill_scheduling": (
            scheduler.get("prefill_scheduling_bypasses", 0)
            + scheduler.get("prefill_scheduling_bypass_forced", 0)
        ),
        # The status snapshot carries a host reading only when the policy is
        # on and the Mach probe answered; psutil fallback leaves it absent.
        "host_memory_signals": int(
            type(host_available) is int and host_available > 0
        ),
        # A streamed model that never paged an expert in was fully resident.
        "moe_expert_streaming": counts.get("stream_page_ins_total", 0),
        # Preemption proves nothing unless the parked lane was replayed.
        "memory_preemption": min(
            counts.get("memory_preemptions", 0),
            counts.get("preempted_replays", 0),
        ),
        "tool_grammar_auto": counts.get(
            "constrained_tool_grammar_auto_engagements", 0
        ),
        "tool_grammar_streaming": counts.get("constrained_tool_grammar_streams", 0),
        "external_pairwise_selection": scheduler.get(
            "external_pairwise_selection_groups", 0
        ),
        "fused_gdn_dynamic_accept": fused_gdn.get(
            "replay_dynamic_rollback_calls", 0
        ),
    }


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
        throughput_tolerance > 0.08
        or max_probe_fraction > 0.075
        or min_bucket_rounds < 64
        or handoff_margin_threshold > 0.5
        or differential_alpha < 0.05
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
                        help="Use an exact passing preflight receipt instead of rerunning pytest")
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
        return response["choices"][0]["message"]["content"].strip()

    initial = get("/v1/status")
    assert initial["healthy"] and initial["inflight"] == 0
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
        assert condition, name

    from mlx2.qualification import required_feature_checks

    required_features = {
        name.removeprefix("feature_")
        for name in required_feature_checks(initial["settings"])
    } | set(args.require_feature)
    try:
        unobservable = unobservable_features(required_features)
        check(
            "feature_observability",
            not unobservable,
            {"required": sorted(required_features), "unobservable": unobservable},
        )
        if trusted_preflight:
            check("unit_tests", True, trusted_preflight)
        else:
            tests = subprocess.run(
                [sys.executable, "-m", "pytest"], capture_output=True, text=True
            )
            check("unit_tests", tests.returncode == 0, tests.stdout + tests.stderr)
        hermes = post(prompt("Reply with exactly HERMES_READY", options={"num_ctx": 262144}, reasoning_effort="none", think=False))
        check("hermes_client", content(hermes) == "HERMES_READY" and hermes["mlx2"]["request_controls"]["thinking"] is False, hermes)
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
        request = prompt("Reply with exactly MLX2_READY")
        cold, warm = post(request), post(request)
        report["responses"].update(cold=cold, warm=warm)
        check("cold_text", content(cold) == "MLX2_READY", cold["mlx2"])
        check(
            "warm_prefix",
            content(warm) == content(cold) and warm["mlx2"]["cached_tokens"] > 0,
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
            text == "MLX2_READY"
            and wire.endswith("data: [DONE]\n\n")
            and "mlx2" in chunks[-1],
            chunks,
        )
        tools = [
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
        tool_request = prompt(
            "Use the weather tool to get the weather in Toronto.",
            tools=tools,
            max_tokens=128,
        )
        tool_response = post(tool_request)
        choice = tool_response["choices"][0]
        calls = choice["message"].get("tool_calls", [])
        check(
            "tools",
            choice["finish_reason"] == "tool_calls"
            and len(calls) == 1
            and calls[0]["function"]["name"] == "weather"
            and json.loads(calls[0]["function"]["arguments"]).get("city") == "Toronto",
            tool_response,
        )
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
        stopped = post(prompt("Reply with exactly MLX2_READY", stop="_READY"))
        check(
            "stop",
            content(stopped) == "MLX2"
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
        # Four concurrent requests cover row lifetimes and observed compute width.
        # Fixed 160-token lanes measure batching, not reasoning: thinking is
        # turned off so a thinking-default model fills the same budget.
        batch_request = prompt(
            "Explain how a compiler works in detail.", max_tokens=160,
            reasoning_effort="none", think=False,
        )
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            batch = list(pool.map(post, [batch_request] * 4))
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
        mixed_prompts = [
            prompt(
                "Explain how a compiler works, in numbered sections.", max_tokens=64
            ),
            prompt(
                "Explain how a database transaction works, in numbered sections.",
                max_tokens=64,
            ),
        ]
        # The sequential warm-up is also the timing reference: two warm
        # requests together must finish faster than the same pair one after
        # the other.  A fixed wall clock failed models that simply answer at
        # length (Xing4.0: ~2.2K tokens each, 45-48 s concurrent).
        sequential_start = time.monotonic()
        for item in mixed_prompts:
            post(item)
        sequential = time.monotonic() - sequential_start
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            mixed = list(pool.map(post, mixed_prompts))
        concurrent = time.monotonic() - start
        check(
            "mixed_warm",
            all(content(r) and r["mlx2"]["cached_tokens"] > 0 for r in mixed)
            and concurrent < max(30.0, sequential),
            {
                "concurrent_seconds": concurrent,
                "sequential_seconds": sequential,
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
            # the budget reasoning and return empty content.
            return prompt(
                long_context_prompt(context_cap),
                max_tokens=completion_budget,
                reasoning_effort="none", think=False,
            )
        if not context_delegation:
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
                      == shared_completion_budget, primed)
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
                shared = list(pool.map(post, shared_pair))
            check(
                "shared_warm_requests",
                all(long_context_answer_passes(content(r))
                    and near_limit_usage_passes(
                        r["usage"], shared_context, shared_completion_budget
                    )
                    and r["mlx2"]["cached_tokens"]
                        > near_limit_prompt_floor(shared_context) - LONG_CONTEXT_CACHE_TOLERANCE
                    and r["mlx2"]["request_controls"].get("batch_cohort")
                        == shared_cohort
                    for r in shared),
                [r["mlx2"] for r in shared],
            )
            long_request = long_context_request(initial["max_context"])
            long = post(long_request)
            repeated = post(long_request)
            check(
                "context",
                long_context_answer_passes(content(long))
                and long_context_answer_passes(content(repeated))
                and near_limit_usage_passes(long["usage"], initial["max_context"])
                and repeated["mlx2"]["cached_tokens"] > n - LONG_CONTEXT_CACHE_TOLERANCE,
                {"cold": long, "warm": repeated},
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
                "Write a very long guide to compiler optimization.", max_tokens=8192
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
            content(recovered) == "MLX2_READY" and get("/health")["status"] == "ok",
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
        observed = feature_observations(
            final,
            kv_fidelity=kv_fidelity,
            adaptive_benchmark=(
                None if adaptive_benchmark is None else adaptive_benchmark[0]
            ),
        )
        report["feature_observations"] = observed
        for feature in sorted(required_features):
            evidence = (
                report.get("adaptive_benchmark")
                if feature in {"adaptive_mtp_depth", "mtp_ordinary_handoff"}
                else execution
            )
            # A feature without an observation is a failed check, never a
            # KeyError that discards the whole run's evidence.
            check("feature_" + feature, observed.get(feature, 0) > 0, evidence)
        check(
            "cache_leases", final["apcv2"]["cow"]["active_leases"] == 0, final["apcv2"]
        )
        from mlx2.serving import runtime_identity

        check(
            "runtime_stable",
            initial["runtime"] == final["runtime"] == runtime_identity(),
            initial["runtime"],
        )
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
    main()
