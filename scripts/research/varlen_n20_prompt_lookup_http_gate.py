#!/usr/bin/env python3
"""Bounded actual-HTTP N3 proof for N20 prompt-lookup verification.

The worker loads one source-bound Qwen3.8 27B service and sends three long chat
requests through the explicit native N route and ordinary greedy decoding.  Its
speculative profiles prove deterministic prompt-lookup K=2 verification.  The
prefill-only profile disables drafting and emits one bootstrap token so packed
prefill can be measured independently with stock quantized model math.  By
default this is a correctness/lifecycle gate.  The explicit performance mode
adds paired host-wall instrumentation; it does not make a qualification claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path.home() / "mlx-models/Qwen3.8-27B-MLX-4bit"
MANIFEST = Path("/tmp/mlx2-hybrid27b-artifact-1004.json")
WHEEL = Path.home() / ".cache/uv/sdists-v9/path/6b22317da775785a/d5Co9cAdGp8_XULv/mlx-0.32.2.dev20260919+39400a0d4-cp312-cp312-macosx_26_0_arm64.whl"
NATIVE = Path("/tmp/mlx2-n20-large-plane-build-1004/_paged_kv_native.cpython-312-darwin.so")
NATIVE_SHA256 = "1f67961f3185196c963505568a2443afde289ecb1d4fe6933a20a1132b63396a"
INPUTS = Path("/tmp/mlx2-spomin400-inputs-d2e10405e.json")
MAX_SECONDS = 180
MAX_RSS = 48 << 30
COHORT_SIZE = 3
MLP_EXPERIMENTS = (
    "staged_qmm", "single_eval_qmm", "staged_bf16", "single_eval_bf16",
    "tiled_q4_swiglu", "packed_gate_up_qmm")
# One bootstrap response precedes the K=2 verify, which then emits two
# accepted draft tokens and one target bonus response.
OUTPUT_CAP = 4
CAP_PROFILES = {
    "prefill-only": (1, 1, 1),
    "full-k2": (4, 4, 4),
    "shrinking-k2-k1-k0": (4, 3, 2),
}
FAILURE_ROOTS = []


def save(path, value):
    from varlen_hybrid_serving_smoke import save as atomic_save
    atomic_save(path, value)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def swapouts():
    output = subprocess.check_output(["vm_stat"], text=True)
    for line in output.splitlines():
        if line.startswith("Swapouts:"):
            return int(line.split(":", 1)[1].strip().rstrip("."))
    raise RuntimeError("host swapout counter unavailable")


def host_performance_context():
    def command(*argv):
        value = subprocess.run(
            argv, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        return {"returncode": value.returncode, "output": value.stdout.strip()}
    return {
        "monotonic_ns": time.monotonic_ns(),
        "swapouts": swapouts(),
        "swapusage": command("sysctl", "vm.swapusage"),
        "thermal": command("pmset", "-g", "therm"),
        "power": command("pmset", "-g", "batt"),
    }


def cap_contract(profile):
    caps = CAP_PROFILES.get(profile)
    if caps is None:
        raise ValueError("explicit HTTP cap profile required")
    drafts = tuple(max(0, min(2, cap - 2)) for cap in caps)
    queries = tuple(depth + 1 for depth in drafts)
    widths = tuple(sum(query > step for query in queries)
                   for step in range(max(queries)))
    return caps, drafts, queries, widths


def physical_counters(widths):
    depth = 16
    singleton_rounds = sum(width == 1 for width in widths)
    return {
        "grouped_n20_write_count": depth * len(widths),
        "grouped_n20_row_count": depth * sum(widths),
        "q1_stock_long_n20_partial_dispatch_count": depth * len(widths),
        "q1_stock_long_n20_reduce_dispatch_count": depth * len(widths),
        "q1_scalar_dispatch_count": 0,
        "q1_stock_long_n20_singleton_partial_dispatch_count": depth * singleton_rounds,
        "q1_stock_long_n20_singleton_reduce_dispatch_count": depth * singleton_rounds,
    }


def plan(cap_profile="full-k2", *, performance_measurement=False,
         arm_order="native-ordinary", diagnostic_stage_profile=False,
         ordinary_declared_cohort=False, ordinary_prefill_batch_size=2,
         native_mlp_experiment="staged_qmm", gdn_eval_wave_max_segments=1,
         gdn_eval_wave_expanded_charge=False):
    if ordinary_prefill_batch_size not in (2, COHORT_SIZE):
        raise ValueError("ordinary prefill batch size must be production B2 or research N3")
    if ordinary_prefill_batch_size == COHORT_SIZE and not ordinary_declared_cohort:
        raise ValueError("research ordinary N3 requires a declared atomic cohort")
    if native_mlp_experiment not in MLP_EXPERIMENTS:
        raise ValueError("unknown native MLP experiment")
    if type(gdn_eval_wave_max_segments) is not int or gdn_eval_wave_max_segments not in (1, 2, 4):
        raise ValueError("GDN evaluation wave must be 1, 2 or 4")
    if type(gdn_eval_wave_expanded_charge) is not bool or gdn_eval_wave_expanded_charge and gdn_eval_wave_max_segments == 1:
        raise ValueError("expanded GDN charge requires a grouped wave")
    caps, drafts, queries, widths = cap_contract(cap_profile)
    return {
        "schema": "mlx2.native-n20-prompt-lookup-http-gate.v3",
        "status": "planned",
        "gpu_executed": False,
        "implemented": True,
        "qualified": False,
        "selected_by_default": False,
        "route_observed_used": False,
        "prompt_lookup_source_observed_used": False,
        "performance_claim": False,
        "performance_measurement": performance_measurement,
        "diagnostic_stage_profile_enabled": diagnostic_stage_profile,
        "arm_order": arm_order,
        "ordinary_admission": (
            "declared_atomic_n3_research_prefill_override"
            if ordinary_declared_cohort and ordinary_prefill_batch_size == COHORT_SIZE
            else "declared_cohort_production_prefill_b2"
            if ordinary_declared_cohort
            else "independent_requests"),
        "ordinary_prefill_batch_size": ordinary_prefill_batch_size,
        "native_mlp_experiment": native_mlp_experiment,
        "gdn_eval_wave_max_segments": gdn_eval_wave_max_segments,
        "gdn_eval_wave_expanded_charge": gdn_eval_wave_expanded_charge,
        "measurement_scope": (
            "diagnostic stage-attributed actual loopback HTTP; evaluation fences inserted"
            if diagnostic_stage_profile else
            "paired actual loopback HTTP native N3 versus ordinary greedy host-wall performance"
            if performance_measurement else
            "actual loopback HTTP native N3 versus ordinary greedy correctness and lifecycle"),
        "cohort_size": COHORT_SIZE,
        "cap_profile": cap_profile,
        "max_tokens": max(caps),
        "max_tokens_by_lane": list(caps),
        "expected_draft_depths": list(drafts),
        "expected_query_lengths": list(queries),
        "expected_round_widths": list(widths),
        "prompt_lookup_depth": 0 if cap_profile == "prefill-only" else 2,
        "prompt_lookup_ngram": None if cap_profile == "prefill-only" else [1, 1],
        "hard_seconds": MAX_SECONDS,
        "max_rss_bytes": MAX_RSS,
    }


def selected_rows(inputs):
    from spomin_400case_native_suite import domain_rows, validate_inputs
    rows = tuple(domain_rows(validate_inputs(inputs), 0)[:COHORT_SIZE])
    if len(rows) != COHORT_SIZE or len({row["case_id"] for row in rows}) != COHORT_SIZE:
        raise ValueError("three distinct frozen source rows required")
    return rows


def request_body(row, *, model, native, inputs_sha256, cap=OUTPUT_CAP,
                 cohort_id="n20-prompt-lookup-http-n3",
                 declared_cohort=False):
    if type(cap) is not int or not 1 <= cap <= OUTPUT_CAP:
        raise ValueError("bounded HTTP cap1..4 required")
    body = {
        **row["body"],
        "model": model,
        "max_tokens": cap,
        "skip_writing_prefix_cache": True,
        # Keep event attribution and source identity identical in the control;
        # these fields do not select the native route without its explicit flag.
        "native_research_input_id": row["case_id"],
        "native_research_inputs_sha256": inputs_sha256,
    }
    if native or declared_cohort:
        body["batch_cohort"] = {"id": cohort_id, "size": COHORT_SIZE}
    if native:
        body["paged_native_packed_n20_research"] = True
    return body


def stage_receipt_delta(before, after):
    """Return the non-negative per-call delta of an observer receipt."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise TypeError("complete observer receipts required")
    result = {}
    names = set(before.get("stages", ())) | set(after.get("stages", ()))
    for name in sorted(names):
        left = before.get("stages", {}).get(
            name, {"calls": 0, "rows": 0, "elapsed_ns": 0})
        right = after.get("stages", {}).get(
            name, {"calls": 0, "rows": 0, "elapsed_ns": 0})
        delta = {key: int(right.get(key, 0)) - int(left.get(key, 0))
                 for key in ("calls", "rows", "elapsed_ns")}
        if any(value < 0 for value in delta.values()):
            raise RuntimeError("observer stage counters moved backwards")
        if any(delta.values()):
            result[name] = delta
    return result


def post(base, body):
    request = Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        with error:
            raw = error.read(1 << 20)
        try:
            return error.code, json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return error.code, {"raw_error_body": raw.decode(errors="replace")}


def run_three(base, bodies, telemetry=None):
    barrier = threading.Barrier(COHORT_SIZE + 1)
    pool = ThreadPoolExecutor(max_workers=COHORT_SIZE)
    starts = [None] * len(bodies)
    ends = [None] * len(bodies)

    def one(index, body):
        barrier.wait(timeout=10)
        starts[index] = time.monotonic_ns()
        try:
            return post(base, body)
        finally:
            ends[index] = time.monotonic_ns()

    try:
        futures = [pool.submit(one, index, body)
                   for index, body in enumerate(bodies)]
        barrier.wait(timeout=10)
        deadline = time.monotonic() + 130
        result = tuple(future.result(
            timeout=max(.001, deadline - time.monotonic())) for future in futures)
        if telemetry is not None:
            if any(value is None for value in (*starts, *ends)):
                raise RuntimeError("complete HTTP client timing required")
            telemetry.update(
                client_starts_ns=starts, client_ends_ns=ends,
                client_start_spread_seconds=(max(starts) - min(starts)) / 1e9)
        return result
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def timing_summary(events_by_case, client_timing):
    if (not isinstance(events_by_case, dict) or len(events_by_case) != COHORT_SIZE or
            any(not events for events in events_by_case.values()) or
            not isinstance(client_timing, dict)):
        raise ValueError("three complete event streams and client clocks required")
    starts = client_timing.get("client_starts_ns", [])
    ends = client_timing.get("client_ends_ns", [])
    if len(starts) != COHORT_SIZE or len(ends) != COHORT_SIZE:
        raise ValueError("three client start/end clocks required")
    events = [event for lane in events_by_case.values() for event in lane]
    first = [lane[0]["monotonic_ns"] for lane in events_by_case.values()]
    last = [lane[-1]["monotonic_ns"] for lane in events_by_case.values()]
    calls = {}
    for event in events:
        key = event["generator_call_id"]
        current = (event["generator_call_started_ns"],
                   event["generator_call_ended_ns"])
        if key in calls and calls[key] != current:
            raise RuntimeError("generator call timing attribution differs")
        calls[key] = current
    began = min(starts)
    ended = max(ends)
    first_any = min(first)
    first_all = max(first)
    last_any = max(last)
    cohort_wall = (ended - began) / 1e9
    decode_span = (last_any - first_any) / 1e9
    post_all_first = (last_any - first_all) / 1e9
    per_lane_spans = [
        (lane_last - lane_first) / 1e9
        for lane_first, lane_last in zip(first, last)
    ]
    output_tokens = len(events)
    decode_tokens = output_tokens - len(events_by_case)
    return {
        "scope": "host monotonic wall with synchronized generator completions; not device-only",
        "cohort_http_wall_seconds": cohort_wall,
        "client_start_spread_seconds": (max(starts) - began) / 1e9,
        "time_through_first_sample_seconds": (first_any - began) / 1e9,
        "time_through_all_first_samples_seconds": (first_all - began) / 1e9,
        "first_sample_to_last_sample_seconds": decode_span,
        "all_first_samples_to_last_sample_seconds": post_all_first,
        "per_lane_first_to_last_seconds": per_lane_spans,
        "mean_per_lane_first_to_last_seconds": (
            sum(per_lane_spans) / len(per_lane_spans)),
        "max_per_lane_first_to_last_seconds": max(per_lane_spans),
        "last_sample_to_http_complete_seconds": (ended - last_any) / 1e9,
        "output_tokens": output_tokens,
        "decode_tokens_after_first": decode_tokens,
        "complete_output_tokens_per_second": output_tokens / cohort_wall,
        "decode_tokens_per_second_after_first": (
            decode_tokens / decode_span if decode_span > 0 else None),
        "generator_calls": len(calls),
        "generator_call_wall_seconds": sum(
            (end - start) / 1e9 for start, end in calls.values()),
        "accepted_draft_tokens": sum(bool(event.get("from_draft"))
                                     for event in events),
        "execution_widths": sorted({
            int(event.get("execution_width", 1)) for event in events}),
    }


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(row, response, events, *, native, cap=OUTPUT_CAP,
              expected_draft=2, expected_query_lengths=(3, 3, 3),
              expected_round_widths=(3, 3, 3), prefill_only=False):
    status, body = response
    choices = body.get("choices", [])
    usage = body.get("usage", {})
    details = body.get("mlx2", {})
    receipt = details.get("route_receipt", {})
    tokens = [event["token"] for event in events]
    if (status != 200 or len(choices) != 1 or
            choices[0].get("finish_reason") != "length" or
            usage.get("completion_tokens") != cap or
            usage.get("prompt_tokens") != row["prompt_tokens"] or
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) != 0 or
            len(tokens) != cap):
        raise RuntimeError("actual HTTP status/usage/finish/sample contract differs")
    if native:
        if not isinstance(receipt, dict):
            raise RuntimeError("native HTTP route receipt is not structured")
        layout = receipt.get("ragged_verify_layout", {})
        lane_uids = layout.get("lane_uids", [])
        query_lengths = layout.get("query_lengths", [])
        uid = events[0].get("uid") if events else None
        lane_query = (query_lengths[lane_uids.index(uid)]
                      if uid in lane_uids and len(lane_uids) == len(query_lengths)
                      else None)
        proof = receipt.get("native_n20_graph_proof", {})
        base_differs = (
                receipt.get("route") != "native_hybrid_packed_n20_research" or
                receipt.get("selected") is not True or
                receipt.get("observed_used") is not True or
                receipt.get("qualified") is not False or
                receipt.get("price_usable") is not False or
                (receipt.get("native_n20_ragged_observed_used", False) is not False
                 if prefill_only else
                 receipt.get("native_n20_ragged_observed_used") is not
                 (expected_draft > 0)) or
                receipt.get("output_token_ids") != tokens or
                receipt.get("prefill_cohort_width") != COHORT_SIZE)
        speculative_differs = not prefill_only and (
                receipt.get("draft_source") != "prompt_lookup" or
                receipt.get("draft_proposed") != expected_draft or
                receipt.get("draft_accepted") != expected_draft or
                receipt.get("draft_configured_max_depth") != 2 or
                receipt.get("verifier_executed_rows") != expected_draft + 1 or
                receipt.get("verifier_round_widths") != list(expected_round_widths) or
                receipt.get("speculative_verification") is not True or
                receipt.get("state_publication") != "atomic_selected_executed_prefix" or
                sorted(query_lengths) != sorted(expected_query_lengths) or
                lane_query != expected_draft + 1 or
                proof.get("physical_counters") != physical_counters(expected_round_widths) or
                [event.get("from_draft") for event in events] !=
                [False, *([True] * expected_draft), False])
        prefill_only_differs = prefill_only and (
                receipt.get("draft_proposed", 0) != 0 or
                receipt.get("draft_accepted", 0) != 0 or
                receipt.get("speculative_verification", False) is not False or
                receipt.get("prefill_mode") != "native_packed_prefill" or
                receipt.get("prefill_layout") != "real_rows" or
                receipt.get("native_prefill_observed_used") is not True or
                any(event.get("from_draft") for event in events))
        if base_differs or speculative_differs or prefill_only_differs:
            observed = {key: receipt.get(key) for key in (
                "route", "selected", "observed_used", "qualified",
                "price_usable", "native_n20_ragged_observed_used",
                "draft_proposed", "draft_accepted", "speculative_verification",
                "prefill_mode", "prefill_layout", "native_prefill_observed_used",
                "output_token_ids", "prefill_cohort_width")}
            raise RuntimeError(
                "final HTTP native route receipt differs: " + repr(observed))
    elif ((isinstance(receipt, dict) and
           receipt.get("route") == "native_hybrid_packed_n20_research") or
          details.get("route") == "native_hybrid_packed_n20_research"):
        raise RuntimeError("ordinary HTTP control selected native N20")
    if details.get("qualification") != "unqualified":
        raise RuntimeError("HTTP qualification label differs")
    return {
        "case_id": row["case_id"],
        "tokens": tokens,
        "text": choices[0].get("message", {}).get("content", ""),
        "finish_reason": choices[0]["finish_reason"],
        "usage": usage,
        "route_receipt": receipt,
        "sample_events": events,
    }


def wait_idle(engine, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with engine.lock:
            idle = (not engine.jobs and not engine.queued_jobs and
                    engine.incoming.empty())
        with engine.submission_lock:
            idle = idle and not engine.pending_cohorts
        if idle:
            return {
                "jobs": 0, "queued_jobs": 0, "incoming": 0,
                "pending_cohorts": 0,
            }
        time.sleep(.005)
    raise RuntimeError("HTTP product queues did not become idle")


def retire_capture(capture, resources, initial_charge, timeout=5):
    owners, candidate, _boots = capture
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidate._serving_resources.reap()
        resources.reap_hybrid_admission_orphans()
        writer = candidate.backend.writer
        if (candidate._serving_resources.closed and
                not writer.pending_epochs and not writer.ledger.pending_count and
                not candidate.backend._orphaned_reads and
                writer.pool.allocated_count == 0 and
                all(owner.fully_retired for owner in owners) and
                resources._CHARGED == initial_charge):
            return {
                "resources_closed": True,
                "owners_fully_retired": True,
                "pending_epochs": 0,
                "ledger_pending": 0,
                "orphaned_reads": 0,
                "allocated_pages": 0,
                "native_charge_restored": True,
            }
        time.sleep(.005)
    raise RuntimeError("native cohort resources did not fully retire")


def execute(args, result):
    from varlen_pack_price_bench import _gpuq_owner
    result["gpuq_owner"] = _gpuq_owner()
    result["host_performance_context_before"] = host_performance_context()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != args.expected_source:
        raise RuntimeError("source revision differs from explicit gate binding")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("source worktree must be clean")
    if sha256(args.native) != args.native_sha256:
        raise RuntimeError("native binary hash differs")
    inputs = json.loads(args.inputs.read_text())
    rows = selected_rows(inputs)
    caps, drafts, queries, widths = cap_contract(args.cap_profile)

    sys.path.insert(0, str(args.native.parent))
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.runtime import hybrid_packed_prefill_n as factory
    os.environ.update(factory.environment(
        gdn_eval_wave_max_segments=args.gdn_eval_wave_max_segments))
    prompt_lookup = args.cap_profile != "prefill-only"
    os.environ.update(
        MLX2_PAGED_N20_RAGGED_PROMPT_LOOKUP="1" if prompt_lookup else "0",
        MLX2_PAGED_N20_RAGGED_DEPTH="2",
        MLX2_PAGED_N20_RAGGED_NGRAM_MIN="1",
        MLX2_PAGED_N20_RAGGED_NGRAM_MAX="1",
    )
    import _paged_kv_native as native
    native_preflight = factory.preflight_native_inputs(native, inputs)
    from mlx2.runtime.paged_price_identity import cached_live_price_identity
    identity = cached_live_price_identity(
        args.artifact_manifest, args.mlx_wheel, args.native,
        adapter_artifact_root=args.model.resolve())
    profile = factory.make_profile(
        identity, inputs,
        gdn_eval_wave_max_segments=args.gdn_eval_wave_max_segments,
        gdn_eval_wave_expanded_charge=args.gdn_eval_wave_expanded_charge,
        mlp_materialization_mode=args.native_mlp_experiment)
    save(args.profile, profile)
    os.environ.update(profile["required_environment"])
    os.environ.update(
        MLX2_NATIVE_PACKED_PREFILL_N20_PROFILE=str(args.profile),
        MLX2_NATIVE_PAGED_MANIFEST=str(args.artifact_manifest),
        MLX2_NATIVE_PAGED_MLX_WHEEL=str(args.mlx_wheel),
    )
    profile = factory.load_profile(
        args.profile, live_identity=identity,
        source_input_ids=tuple(row["case_id"] for row in rows),
        counts=tuple(row["prompt_tokens"] for row in rows),
        tokens=tuple(tuple(row["prompt_token_ids"]) for row in rows),
        environment_values=os.environ)

    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, configure_environment
    configure_environment()
    import mlx.core as mx
    from mlx2 import serving
    from mlx2.runtime import qwen35_paged_graph_factory as resources
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.ragged_verify_observation import (
        RaggedVerifyObserver, activate as activate_stage_observer)
    from mlx2.server import handler_for, validate_request
    from varlen_hybrid_packed_prefill_long_http_gate import bounded_cleanup

    result.update(
        status="running", gpu_executed=True, identity=identity,
        arm_order=args.arm_order,
        performance_measurement=args.performance_measurement,
        diagnostic_stage_profile_enabled=args.diagnostic_stage_profile,
        ordinary_admission=(
            "declared_atomic_n3_research_prefill_override"
            if (args.ordinary_declared_cohort and
                args.ordinary_prefill_batch_size == COHORT_SIZE)
            else "declared_cohort_production_prefill_b2"
            if args.ordinary_declared_cohort
            else "independent_requests"),
        ordinary_prefill_batch_size=args.ordinary_prefill_batch_size,
        native_mlp_experiment=args.native_mlp_experiment,
        gdn_eval_wave_max_segments=args.gdn_eval_wave_max_segments,
        gdn_eval_wave_expanded_charge=args.gdn_eval_wave_expanded_charge,
        native_sha256=args.native_sha256, inputs_sha256=inputs["inputs_sha256"],
        input_artifact_file_sha256=sha256(args.inputs),
        native_preload_capability_proof=native_preflight,
        selected_case_ids=[row["case_id"] for row in rows],
        max_tokens_by_lane=list(caps), expected_draft_depths=list(drafts),
        expected_query_lengths=list(queries), expected_round_widths=list(widths),
    )
    previous_cache_limit = mx.set_cache_limit(0)
    mx.clear_cache()
    initial_charge = resources._CHARGED
    captures = []
    events = {"native": {}, "ordinary": {}}
    current_arm = [None]
    generator_call_id = [0]
    stage_observers = {
        arm: (RaggedVerifyObserver(evaluator=mx.eval)
              if args.diagnostic_stage_profile else None)
        for arm in ("native", "ordinary")
    }
    diagnostic_generator_calls = {"native": [], "ordinary": []}
    native_factory_calls = []
    engine = server = server_thread = None
    original_next = BatchGenerator.next
    original_batch_init = BatchGenerator.__init__
    original_factory = factory.create_cold_packed_hybrid_n

    def research_batch_init(instance, *positional, **keywords):
        if args.ordinary_prefill_batch_size == COHORT_SIZE:
            if keywords.get("prefill_batch_size") != min(2, COHORT_SIZE):
                raise RuntimeError("ordinary production prefill width assumption drifted")
            keywords["prefill_batch_size"] = COHORT_SIZE
        return original_batch_init(instance, *positional, **keywords)

    def observed_next(batch, *positional, **keywords):
        call_id = generator_call_id[0]
        generator_call_id[0] += 1
        call_started = time.monotonic_ns()
        arm = current_arm[0]
        observer = stage_observers.get(arm)
        observer_before = observer.receipt() if observer is not None else None
        with activate_stage_observer(observer):
            prompts, responses = original_next(batch, *positional, **keywords)
        if arm is not None:
            mx.synchronize()
            call_ended = time.monotonic_ns()
            if observer is not None:
                observer_after = observer.receipt()
                diagnostic_generator_calls[arm].append({
                    "generator_call_id": call_id,
                    "wall_seconds": (call_ended - call_started) / 1e9,
                    "prompt_responses": len(prompts),
                    "sample_responses": len(responses),
                    "stage_delta": stage_receipt_delta(
                        observer_before, observer_after),
                })
            for response in responses:
                job = next((item for item in engine.jobs.values()
                            if item.uid == response.uid), None)
                if job is None:
                    raise RuntimeError("sample response lacks live HTTP Job")
                case_id = job.request.get("native_research_input_id", job.id)
                events[arm].setdefault(case_id, []).append({
                    "uid": int(response.uid),
                    "token": int(response.token),
                    "execution_width": int(getattr(response, "execution_width", 1)),
                    "from_draft": bool(getattr(response, "from_draft", False)),
                    "monotonic_ns": call_ended,
                    "generator_call_id": call_id,
                    "generator_call_started_ns": call_started,
                    "generator_call_ended_ns": call_ended,
                    "route_receipt": dict(response.mtp_receipt or {}),
                })
        return prompts, responses

    def captured_factory(*positional, **keywords):
        began = (time.monotonic_ns()
                 if args.diagnostic_stage_profile else None)
        value = original_factory(*positional, **keywords)
        if began is not None:
            mx.synchronize()
            native_factory_calls.append({
                "wall_seconds": (time.monotonic_ns() - began) / 1e9,
                "packed_prefill_proof": dict(
                    getattr(value[1], "_packed_prefill_receipt", {})),
            })
        captures.append(value)
        return value

    BatchGenerator.next = observed_next
    BatchGenerator.__init__ = research_batch_init
    factory.create_cold_packed_hybrid_n = captured_factory
    cleanup = None
    started = time.perf_counter()
    try:
        engine = serving.ServingEngine(
            args.model, adapter_factory=Qwen3827BAdapter,
            max_lanes=COHORT_SIZE, max_inflight=COHORT_SIZE,
            mtp=False, prompt_lookup=False, qualification_mode=False,
            prefill_step=8192, max_context=8192, cache_bytes=1 << 29,
            batch_cohort_timeout_ms=5000)
        deadline = time.monotonic() + 45
        while not engine.ready.is_set() and not engine.error and time.monotonic() < deadline:
            engine.ready.wait(.05)
        if engine.error or not engine.ready.is_set():
            raise RuntimeError("HTTP engine load failed: " + str(engine.error))
        result["load_seconds"] = time.perf_counter() - started
        if cached_live_price_identity(
                args.artifact_manifest, args.mlx_wheel, args.native,
                adapter_artifact_root=Path(engine.adapter.identity["path"]).resolve()) != identity:
            raise RuntimeError("loaded model/source identity changed")

        def phase_boundary(event):
            mx.synchronize()
            if int(mx.get_active_memory()) + int(mx.get_cache_memory()) > MAX_RSS:
                raise MemoryError("MLX active/cache bytes exceed gate ceiling")
            result.setdefault("native_phase_events", []).append({
                **event,
                "mlx_active_bytes": int(mx.get_active_memory()),
                "mlx_cache_bytes": int(mx.get_cache_memory()),
            })

        engine.adapter._native_n20_phase_boundary = phase_boundary
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        server.daemon_threads = True
        server.block_on_close = False
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base = "http://127.0.0.1:" + str(server.server_port)

        cohort_id = "n20-prompt-lookup-http-" + args.cap_profile
        native_bodies = tuple(validate_request(request_body(
            row, model=args.model.name, native=True,
            inputs_sha256=inputs["inputs_sha256"], cap=cap,
            cohort_id=cohort_id)) for row, cap in zip(rows, caps))
        ordinary_bodies = tuple(validate_request(request_body(
            row, model=args.model.name, native=False,
            inputs_sha256=inputs["inputs_sha256"], cap=cap,
            cohort_id=cohort_id + "-ordinary",
            declared_cohort=args.ordinary_declared_cohort))
            for row, cap in zip(rows, caps))
        arm_results = {}

        def run_arm(arm):
            before_captures = len(captures)
            client_timing = {}
            current_arm[0] = arm
            try:
                responses = run_three(
                    base, native_bodies if arm == "native" else ordinary_bodies,
                    telemetry=client_timing)
            finally:
                current_arm[0] = None
            if set(events[arm]) != {row["case_id"] for row in rows}:
                raise RuntimeError(arm + " HTTP event attribution differs")
            if arm == "native":
                if len(captures) != before_captures + 1:
                    raise RuntimeError("actual native HTTP cohort did not construct one shared candidate")
                summary = tuple(summarize(
                    row, response, events[arm][row["case_id"]], native=True,
                    cap=cap, expected_draft=draft,
                    expected_query_lengths=queries,
                    expected_round_widths=widths,
                    prefill_only=args.cap_profile == "prefill-only")
                    for row, response, cap, draft in zip(
                        rows, responses, caps, drafts))
                retirement = retire_capture(
                    captures[-1], resources, initial_charge)
            else:
                if len(captures) != before_captures or resources._CHARGED != initial_charge:
                    raise RuntimeError("ordinary control allocated native resources")
                summary = tuple(summarize(
                    row, response, events[arm][row["case_id"]], native=False,
                    cap=cap)
                    for row, response, cap in zip(rows, responses, caps))
                retirement = None
            arm_results[arm] = {
                "summary": summary,
                "timing": timing_summary(events[arm], client_timing),
                "idle": wait_idle(engine),
                "retirement": retirement,
            }

        for arm in args.arm_order.split("-"):
            run_arm(arm)
        native_summary = arm_results["native"]["summary"]
        ordinary_summary = arm_results["ordinary"]["summary"]
        native_idle = arm_results["native"]["idle"]
        ordinary_idle = arm_results["ordinary"]["idle"]
        native_retirement = arm_results["native"]["retirement"]
        for native_row, ordinary_row in zip(native_summary, ordinary_summary):
            if any(native_row[key] != ordinary_row[key]
                   for key in ("tokens", "text", "finish_reason", "usage")):
                raise RuntimeError("native/ordinary actual HTTP output differs")
        ordinary_widths = sorted({
            int(event["execution_width"])
            for lane in events["ordinary"].values() for event in lane})
        if (args.ordinary_prefill_batch_size == COHORT_SIZE and
                ordinary_widths != [COHORT_SIZE]):
            raise RuntimeError("research ordinary prefill override did not execute at N3")

        result.update(
            status="passed",
            route_observed_used=True,
            prompt_lookup_source_observed_used=prompt_lookup,
            native=native_summary,
            ordinary=ordinary_summary,
            exact_http_token_text_finish_usage_parity=True,
            ordinary_execution_widths=ordinary_widths,
            native_product_idle=native_idle,
            ordinary_product_idle=ordinary_idle,
            native_retirement=native_retirement,
            diagnostic_stage_profile={
                "scope": (
                    "diagnostic only; mx.eval fences inserted at observed model stages"
                    if args.diagnostic_stage_profile else "disabled"),
                "native": {
                    "observer": (stage_observers["native"].receipt()
                                 if stage_observers["native"] is not None else None),
                    "generator_calls": diagnostic_generator_calls["native"],
                    "factory_calls": native_factory_calls,
                },
                "ordinary": {
                    "observer": (stage_observers["ordinary"].receipt()
                                 if stage_observers["ordinary"] is not None else None),
                    "generator_calls": diagnostic_generator_calls["ordinary"],
                },
            },
            performance_timing={arm: arm_results[arm]["timing"]
                                for arm in ("native", "ordinary")},
            paired_speed_ratios={
                "cohort_http_wall_ordinary_over_native": safe_ratio(
                    arm_results["ordinary"]["timing"]["cohort_http_wall_seconds"],
                    arm_results["native"]["timing"]["cohort_http_wall_seconds"]),
                "time_through_all_first_ordinary_over_native": safe_ratio(
                    arm_results["ordinary"]["timing"]["time_through_all_first_samples_seconds"],
                    arm_results["native"]["timing"]["time_through_all_first_samples_seconds"]),
                "first_any_to_last_ordinary_over_native": safe_ratio(
                    arm_results["ordinary"]["timing"]["first_sample_to_last_sample_seconds"],
                    arm_results["native"]["timing"]["first_sample_to_last_sample_seconds"]),
                "all_first_to_last_ordinary_over_native": safe_ratio(
                    arm_results["ordinary"]["timing"]["all_first_samples_to_last_sample_seconds"],
                    arm_results["native"]["timing"]["all_first_samples_to_last_sample_seconds"]),
                "generator_call_wall_ordinary_over_native": safe_ratio(
                    arm_results["ordinary"]["timing"]["generator_call_wall_seconds"],
                    arm_results["native"]["timing"]["generator_call_wall_seconds"]),
            },
            model_loads=1,
            engine_instances=1,
            mlx_peak_bytes=int(mx.get_peak_memory()),
            elapsed_seconds=time.perf_counter() - started,
        )
    finally:
        current_arm[0] = None
        cleanup = bounded_cleanup(
            engine, server, server_thread, tuple(captures), result)
        deadline = time.monotonic() + 5
        while resources._CHARGED != initial_charge and time.monotonic() < deadline:
            resources.reap_hybrid_admission_orphans()
            time.sleep(.005)
        cleanup["native_charge_restored"] = resources._CHARGED == initial_charge
        cleanup["retained"] = cleanup["retained"] or not cleanup["native_charge_restored"]
        result["cleanup"] = cleanup
        result["host_performance_context_after"] = host_performance_context()
        result["lease_swapouts_delta"] = (
            result["host_performance_context_after"]["swapouts"] -
            result["host_performance_context_before"]["swapouts"])
        BatchGenerator.next = original_next
        BatchGenerator.__init__ = original_batch_init
        factory.create_cold_packed_hybrid_n = original_factory
        mx.set_cache_limit(previous_cache_limit)
        if cleanup["retained"]:
            FAILURE_ROOTS.append((engine, captures))
            if result.get("status") == "passed":
                raise RuntimeError("final cleanup retained native roots")


def supervise(args, result):
    save(args.output, result)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    process = subprocess.Popen(command, start_new_session=True)
    began = time.monotonic()
    peak_rss = 0
    try:
        while process.poll() is None:
            if time.monotonic() - began > MAX_SECONDS:
                raise TimeoutError("HTTP gate exceeded hard wall deadline")
            try:
                rss = int(subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(process.pid)], text=True).strip() or "0") * 1024
            except (subprocess.CalledProcessError, ValueError):
                rss = 0
            peak_rss = max(peak_rss, rss)
            if rss > MAX_RSS:
                raise MemoryError("HTTP gate worker exceeded 48 GiB RSS")
            time.sleep(.2)
        if args.output.is_file():
            result = json.loads(args.output.read_text())
        result.update(
            supervisor_peak_rss_bytes=peak_rss,
            supervisor_seconds=time.monotonic() - began,
        )
        save(args.output, result)
        return process.returncode
    except BaseException as error:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        if args.output.is_file():
            result = json.loads(args.output.read_text())
        result.update(
            status="failed", qualified=False, worker_killed=True,
            error=f"{type(error).__name__}: {error}",
            supervisor_peak_rss_bytes=peak_rss,
            supervisor_seconds=time.monotonic() - began,
        )
        save(args.output, result)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--artifact-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--mlx-wheel", type=Path, default=WHEEL)
    parser.add_argument("--native", type=Path, default=NATIVE)
    parser.add_argument("--native-sha256", default=NATIVE_SHA256)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--profile", type=Path,
                        default=Path("/tmp/mlx2-n20-prompt-lookup-http-profile-1004.json"))
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap-profile", choices=tuple(CAP_PROFILES),
                        default="full-k2")
    parser.add_argument("--arm-order",
                        choices=("native-ordinary", "ordinary-native"),
                        default="native-ordinary")
    parser.add_argument("--performance-measurement", action="store_true")
    parser.add_argument("--diagnostic-stage-profile", action="store_true")
    parser.add_argument("--ordinary-declared-cohort", action="store_true")
    parser.add_argument("--ordinary-prefill-batch-size", type=int,
                        choices=(2, COHORT_SIZE), default=2)
    parser.add_argument("--native-mlp-experiment", choices=MLP_EXPERIMENTS,
                        default="staged_qmm")
    parser.add_argument("--gdn-eval-wave-max-segments", type=int,
                        choices=(1, 2, 4), default=1)
    parser.add_argument("--gdn-eval-wave-expanded-charge", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    result = plan(
        args.cap_profile, performance_measurement=args.performance_measurement,
        arm_order=args.arm_order,
        diagnostic_stage_profile=args.diagnostic_stage_profile,
        ordinary_declared_cohort=args.ordinary_declared_cohort,
        ordinary_prefill_batch_size=args.ordinary_prefill_batch_size,
        native_mlp_experiment=args.native_mlp_experiment,
        gdn_eval_wave_max_segments=args.gdn_eval_wave_max_segments,
        gdn_eval_wave_expanded_charge=args.gdn_eval_wave_expanded_charge)
    if not args.execute:
        save(args.output, result)
        return 0
    if not args.worker:
        return supervise(args, result)
    code = 0
    try:
        execute(args, result)
    except BaseException as error:
        result.update(
            status="failed", qualified=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            retained_failure_roots=len(FAILURE_ROOTS),
        )
        code = 1
    save(args.output, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
