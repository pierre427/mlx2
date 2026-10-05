#!/usr/bin/env python3
"""Ten-minute deterministic arbitrary-traffic A/B for packed N20 research.

The immutable trace is prepared separately.  Every wave replays the same three
request bodies and arrival offsets through adaptive ordinary serving (A) and
packed N20 K=2 serving (B); arm order alternates by wave.  Runtime timing is an
observation, never an input to request selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
import varlen_n20_prompt_lookup_http_gate as G

MAX_RSS = 48 << 30
FAILURE_ROOTS = []


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_wave(base, bodies, offsets_ms):
    """Send one deterministic wave; offsets are identical in every arm."""
    if len(bodies) != len(offsets_ms) or not bodies:
        raise ValueError("one arrival offset per request is required")
    epoch = time.monotonic() + 0.1
    pool = ThreadPoolExecutor(max_workers=len(bodies))
    starts = [None] * len(bodies)
    ends = [None] * len(bodies)

    def one(index):
        target = epoch + offsets_ms[index] / 1000
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        starts[index] = time.monotonic_ns()
        try:
            return G.post(base, bodies[index])
        finally:
            ends[index] = time.monotonic_ns()

    try:
        futures = [pool.submit(one, index) for index in range(len(bodies))]
        responses = tuple(future.result(timeout=150) for future in futures)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return responses, {
        "client_starts_ns": starts,
        "client_ends_ns": ends,
        "wave_wall_seconds": (max(ends) - min(starts)) / 1e9,
        "start_spread_seconds": (max(starts) - min(starts)) / 1e9,
        "request_wall_seconds": [
            (end - start) / 1e9 for start, end in zip(starts, ends)
        ],
    }


def response_summary(row, response, *, native):
    status, body = response
    if not isinstance(body, dict):
        raise TypeError("arbitrary traffic HTTP response body is not an object")
    choices = body.get("choices", [])
    usage = body.get("usage", {})
    if not isinstance(choices, list):
        choices = []
    if not isinstance(usage, dict):
        usage = {}
    choice = choices[0] if len(choices) == 1 and isinstance(choices[0], dict) else {}
    details = body.get("mlx2", {})
    if not isinstance(details, dict):
        details = {}
    receipt = details.get("route_receipt", {})
    receipt_fields = receipt if isinstance(receipt, dict) else {}
    scheduler = details.get("scheduler", {})
    scheduler_fields = scheduler if isinstance(scheduler, dict) else {}
    if (
        status != 200
        or len(choices) != 1
        or not choice
        or choice.get("finish_reason") != "length"
        or usage.get("prompt_tokens") != row["prompt_tokens"]
        or usage.get("completion_tokens") != 4
    ):
        observed = {
            "case_id": row.get("case_id"),
            "native": native,
            "status": status,
            "finish_reason": (
                choice.get("finish_reason") if choice else None
            ),
            "choice_count": len(choices),
            "usage": usage,
            "error": body.get("error"),
        }
        raise RuntimeError(
            "arbitrary traffic HTTP response contract differs: " + repr(observed)
        )
    if native:
        if (
            receipt_fields.get("route") != "native_hybrid_packed_n20_research"
            or receipt_fields.get("selected") is not True
            or receipt_fields.get("observed_used") is not True
            or receipt_fields.get("qualified") is not False
            or receipt_fields.get("native_prefill_observed_used") is not True
            or receipt_fields.get("draft_source") != "prompt_lookup"
            or receipt_fields.get("draft_proposed") != 2
            or receipt_fields.get("draft_accepted") != 2
            or receipt_fields.get("speculative_verification") is not True
            or len(receipt_fields.get("output_token_ids", ())) != 4
        ):
            raise RuntimeError("native arbitrary-traffic route proof differs")
    elif (
        receipt_fields.get("route") == "native_hybrid_packed_n20_research"
    ):
        raise RuntimeError("ordinary arbitrary-traffic arm selected native N20")
    return {
        "case_id": row["case_id"],
        "text": (
            choice.get("message", {}).get("content", "")
            if isinstance(choice.get("message"), dict)
            else ""
        ),
        "finish_reason": choice["finish_reason"],
        "usage": usage,
        "output_token_ids": receipt_fields.get("output_token_ids"),
        "draft_proposed": receipt_fields.get("draft_proposed", 0),
        "draft_accepted": receipt_fields.get("draft_accepted", 0),
        "scheduler": scheduler_fields,
        "route_receipt": (
            receipt
            if native
            else {
                key: receipt_fields.get(key)
                for key in ("route", "selected", "observed_used", "qualified")
                if key in receipt_fields
            }
            if isinstance(receipt, dict)
            else receipt
        ),
    }


def wait_for_scheduler_snapshot(engine, *, timeout=2.0):
    """Read the asynchronously published scheduler counters after timing ends."""
    deadline = time.monotonic() + timeout
    scheduler = {}
    while time.monotonic() < deadline:
        status = engine.status()
        candidate = status.get("scheduler", {}) if isinstance(status, dict) else {}
        scheduler = candidate if isinstance(candidate, dict) else {}
        if scheduler.get("batch_geometry_rounds", 0) > 0:
            break
        time.sleep(0.05)
    return scheduler


def execute(args, result):
    from varlen_pack_price_bench import _gpuq_owner

    result["gpuq_owner"] = _gpuq_owner()
    result["host_performance_context_before"] = G.host_performance_context()
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if source != args.expected_source:
        raise RuntimeError("source revision differs from explicit campaign binding")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("source worktree must be clean")
    if file_sha256(args.native) != args.native_sha256:
        raise RuntimeError("native binary hash differs")
    inputs = json.loads(args.inputs.read_text())
    trace = json.loads(args.trace.read_text())
    embedded = inputs.get("deterministic_arbitrary_traffic")
    if (
        trace.get("schema") != "mlx2.deterministic-arbitrary-traffic.v1"
        or trace.get("inputs_sha256") != inputs.get("inputs_sha256")
        or embedded is None
        or embedded.get("requests") != trace.get("requests")
        or trace.get("wave_size") != 3
    ):
        raise ValueError("trace and source-bound input identity differ")
    rows_by_id = {row["case_id"]: row for row in inputs["rows"]}
    waves = []
    for wave_index in range(trace["wave_count"]):
        items = [
            item for item in trace["requests"] if item["wave"] == wave_index
        ]
        rows = tuple(rows_by_id[item["case_id"]] for item in items)
        if len(rows) != 3 or len({row["case_id"] for row in rows}) != 3:
            raise ValueError("each deterministic wave requires three unique rows")
        waves.append((items, rows))

    sys.path.insert(0, str(args.native.parent))
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.runtime import hybrid_packed_prefill_n as factory

    os.environ.update(factory.environment())
    os.environ.update(
        MLX2_PAGED_N20_RAGGED_PROMPT_LOOKUP="1",
        MLX2_PAGED_N20_RAGGED_DEPTH="2",
        MLX2_PAGED_N20_RAGGED_NGRAM_MIN="1",
        MLX2_PAGED_N20_RAGGED_NGRAM_MAX="1",
    )
    import _paged_kv_native as native

    native_preflight = factory.preflight_native_inputs(native, inputs)
    from mlx2.runtime.paged_price_identity import cached_live_price_identity

    identity = cached_live_price_identity(
        args.artifact_manifest,
        args.mlx_wheel,
        args.native,
        adapter_artifact_root=args.model.resolve(),
    )
    profile = factory.make_profile(identity, inputs)
    G.save(args.profile, profile)
    os.environ.update(profile["required_environment"])
    os.environ.update(
        MLX2_NATIVE_PACKED_PREFILL_N20_PROFILE=str(args.profile),
        MLX2_NATIVE_PAGED_MANIFEST=str(args.artifact_manifest),
        MLX2_NATIVE_PAGED_MLX_WHEEL=str(args.mlx_wheel),
    )
    _, first_rows = waves[0]
    factory.load_profile(
        args.profile,
        live_identity=identity,
        source_input_ids=tuple(row["case_id"] for row in first_rows),
        counts=tuple(row["prompt_tokens"] for row in first_rows),
        tokens=tuple(tuple(row["prompt_token_ids"]) for row in first_rows),
        environment_values=os.environ,
    )

    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, configure_environment

    configure_environment()
    import mlx.core as mx

    from mlx2 import serving
    from mlx2.runtime import qwen35_paged_graph_factory as resources
    from mlx2.server import handler_for, validate_request

    previous_cache_limit = mx.set_cache_limit(0)
    mx.clear_cache()
    initial_charge = resources._CHARGED
    captures = []
    engine = server = server_thread = None
    original_factory = factory.create_cold_packed_hybrid_n

    def captured_factory(*positional, **keywords):
        value = original_factory(*positional, **keywords)
        captures.append(value)
        return value

    factory.create_cold_packed_hybrid_n = captured_factory
    result.update(
        status="running",
        gpu_executed=True,
        source_revision=source,
        trace_sha256=file_sha256(args.trace),
        inputs_file_sha256=file_sha256(args.inputs),
        inputs_sha256=inputs["inputs_sha256"],
        seed=trace["seed"],
        configured_arrival_window_seconds=args.duration_seconds,
        native_preload_capability_proof=native_preflight,
        identity=identity,
        waves=[],
    )
    G.save(args.output, result)
    started = time.monotonic()
    try:
        engine = serving.ServingEngine(
            args.model,
            adapter_factory=Qwen3827BAdapter,
            max_lanes=6,
            max_inflight=20,
            mtp=False,
            prompt_lookup=False,
            qualification_mode=False,
            prefill_step=2048,
            max_context=8192,
            cache_bytes=1 << 29,
            batch_cohort_timeout_ms=5000,
            execution_policy={
                "batch_geometry": {
                    "token_budget": 4096,
                    "bucket_padding_fraction": 0.12,
                    "max_bucket_ratio": 1.5,
                }
            },
        )
        deadline = time.monotonic() + 45
        while not engine.ready.is_set() and not engine.error and time.monotonic() < deadline:
            engine.ready.wait(0.05)
        if engine.error or not engine.ready.is_set():
            raise RuntimeError("HTTP engine load failed: " + str(engine.error))
        result["load_seconds"] = time.monotonic() - started

        def phase_boundary(event):
            mx.synchronize()
            if int(mx.get_active_memory()) + int(mx.get_cache_memory()) > MAX_RSS:
                raise MemoryError("MLX active/cache bytes exceed campaign ceiling")
            result.setdefault("native_phase_events", []).append(
                {
                    **event,
                    "mlx_active_bytes": int(mx.get_active_memory()),
                    "mlx_cache_bytes": int(mx.get_cache_memory()),
                }
            )

        engine.adapter._native_n20_phase_boundary = phase_boundary
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        server.daemon_threads = True
        server.block_on_close = False
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base = "http://127.0.0.1:" + str(server.server_port)
        traffic_started = time.monotonic()
        launch = 0
        while time.monotonic() - traffic_started < args.duration_seconds:
            template_index = launch % len(waves)
            items, rows = waves[template_index]
            cohort_id = f"deterministic-arbitrary-{launch:04d}"
            offsets = [item["arrival_offset_ms"] for item in items]
            native_bodies = tuple(
                validate_request(
                    G.request_body(
                        row,
                        model=args.model.name,
                        native=True,
                        inputs_sha256=inputs["inputs_sha256"],
                        cap=4,
                        cohort_id=cohort_id,
                    )
                )
                for row in rows
            )
            ordinary_bodies = tuple(
                validate_request(
                    G.request_body(
                        row,
                        model=args.model.name,
                        native=False,
                        inputs_sha256=inputs["inputs_sha256"],
                        cap=4,
                        cohort_id=cohort_id + "-ordinary",
                    )
                )
                for row in rows
            )
            arm_order = (
                ("native", "ordinary")
                if launch % 2 == 0
                else ("ordinary", "native")
            )
            wave_result = {
                "launch": launch,
                "template_wave": template_index,
                "arm_order": list(arm_order),
                "requests": items,
                "arms": {},
                "launched_at_seconds": time.monotonic() - traffic_started,
            }
            summaries = {}
            for arm in arm_order:
                before_captures = len(captures)
                responses, timing = run_wave(
                    base,
                    native_bodies if arm == "native" else ordinary_bodies,
                    offsets,
                )
                summary = tuple(
                    response_summary(row, response, native=arm == "native")
                    for row, response in zip(rows, responses)
                )
                retirement = None
                if arm == "native":
                    if len(captures) != before_captures + 1:
                        raise RuntimeError("native wave did not construct one candidate")
                    retirement = G.retire_capture(
                        captures[-1], resources, initial_charge
                    )
                    # Retirement is proven above; retaining the factory capture
                    # would pin the candidate and its arena across the 10-minute
                    # campaign even though the route has relinquished ownership.
                    captures.pop()
                elif len(captures) != before_captures:
                    raise RuntimeError("ordinary wave constructed a native candidate")
                G.wait_idle(engine, timeout=10)
                summaries[arm] = summary
                wave_result["arms"][arm] = {
                    "timing": timing,
                    "summary": summary,
                    "retirement": retirement,
                }
            for native_row, ordinary_row in zip(
                summaries["native"], summaries["ordinary"]
            ):
                if any(
                    native_row[key] != ordinary_row[key]
                    for key in ("text", "finish_reason", "usage")
                ):
                    raise RuntimeError("arbitrary native/ordinary HTTP parity differs")
            native_wall = wave_result["arms"]["native"]["timing"][
                "wave_wall_seconds"
            ]
            ordinary_wall = wave_result["arms"]["ordinary"]["timing"][
                "wave_wall_seconds"
            ]
            wave_result.update(
                exact_http_text_finish_usage_parity=True,
                ordinary_over_native_wave_wall=(
                    ordinary_wall / native_wall if native_wall else None
                ),
                completed_at_seconds=time.monotonic() - traffic_started,
            )
            result["waves"].append(wave_result)
            result["completed_waves"] = len(result["waves"])
            result["traffic_elapsed_seconds"] = time.monotonic() - traffic_started
            G.save(args.output, result)
            launch += 1
        G.wait_idle(engine, timeout=10)
        ordinary_scheduler_final = wait_for_scheduler_snapshot(engine)
        adaptive_ordinary_observed_used = (
            ordinary_scheduler_final.get("batch_geometry_rounds", 0) > 0
        )
        if not adaptive_ordinary_observed_used:
            raise RuntimeError(
                "ordinary batch-geometry scheduler emitted no mechanism counter"
            )
        ratios = [wave["ordinary_over_native_wave_wall"] for wave in result["waves"]]
        result.update(
            status="passed",
            route_observed_used=True,
            adaptive_ordinary_observed_used=adaptive_ordinary_observed_used,
            ordinary_scheduler_final=ordinary_scheduler_final,
            prompt_lookup_source_observed_used=True,
            exact_http_text_finish_usage_parity=True,
            mean_ordinary_over_native_wave_wall=sum(ratios) / len(ratios),
            median_ordinary_over_native_wave_wall=sorted(ratios)[len(ratios) // 2],
            accepted_draft_tokens=sum(
                row["draft_accepted"]
                for wave in result["waves"]
                for row in wave["arms"]["native"]["summary"]
            ),
            mlx_peak_bytes=int(mx.get_peak_memory()),
        )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(5)
        if engine is not None:
            engine.close()
        deadline = time.monotonic() + 10
        while resources._CHARGED != initial_charge and time.monotonic() < deadline:
            resources.reap_hybrid_admission_orphans()
            time.sleep(0.01)
        result["cleanup"] = {
            "server_stopped": server_thread is None or not server_thread.is_alive(),
            "engine_closed": engine is None or not engine.thread.is_alive(),
            "native_charge_restored": resources._CHARGED == initial_charge,
        }
        result["host_performance_context_after"] = G.host_performance_context()
        result["lease_swapouts_delta"] = (
            result["host_performance_context_after"]["swapouts"]
            - result["host_performance_context_before"]["swapouts"]
        )
        factory.create_cold_packed_hybrid_n = original_factory
        mx.set_cache_limit(previous_cache_limit)
        if not all(result["cleanup"].values()):
            FAILURE_ROOTS.extend((engine, captures))
            if result.get("status") == "passed":
                raise RuntimeError("campaign cleanup remained incomplete")


def supervise(args, result):
    G.save(args.output, result)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    process = subprocess.Popen(command, start_new_session=True)
    began = time.monotonic()
    peak_rss = 0
    hard_seconds = args.duration_seconds + 300
    try:
        while process.poll() is None:
            if time.monotonic() - began > hard_seconds:
                raise TimeoutError("arbitrary traffic campaign exceeded hard deadline")
            try:
                rss = int(
                    subprocess.check_output(
                        ["ps", "-o", "rss=", "-p", str(process.pid)], text=True
                    ).strip()
                    or "0"
                ) * 1024
            except (subprocess.CalledProcessError, ValueError):
                rss = 0
            peak_rss = max(peak_rss, rss)
            if rss > MAX_RSS:
                raise MemoryError("campaign worker exceeded 48 GiB RSS")
            time.sleep(0.5)
        if args.output.is_file():
            result = json.loads(args.output.read_text())
        result.update(
            supervisor_peak_rss_bytes=peak_rss,
            supervisor_seconds=time.monotonic() - began,
        )
        G.save(args.output, result)
        return process.returncode
    except BaseException as error:  # noqa: BLE001 - supervisor must kill the process group
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        if args.output.is_file():
            result = json.loads(args.output.read_text())
        result.update(
            status="failed",
            qualified=False,
            worker_killed=True,
            error=f"{type(error).__name__}: {error}",
            supervisor_peak_rss_bytes=peak_rss,
            supervisor_seconds=time.monotonic() - began,
        )
        G.save(args.output, result)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=G.MODEL)
    parser.add_argument("--artifact-manifest", type=Path, default=G.MANIFEST)
    parser.add_argument("--mlx-wheel", type=Path, default=G.WHEEL)
    parser.add_argument("--native", type=Path, default=G.NATIVE)
    parser.add_argument("--native-sha256", default=G.NATIVE_SHA256)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 60 <= args.duration_seconds <= 3600:
        parser.error("duration must be between 60 and 3600 seconds")
    result = {
        "schema": "mlx2.deterministic-arbitrary-http-campaign.v1",
        "status": "planned",
        "implemented": True,
        "qualified": False,
        "selected_by_default": False,
        "gpu_executed": False,
        "duration_seconds": args.duration_seconds,
    }
    if not args.execute:
        G.save(args.output, result)
        return 0
    if not args.worker:
        return supervise(args, result)
    code = 0
    try:
        execute(args, result)
    except BaseException as error:  # noqa: BLE001 - persist every campaign failure
        result.update(
            status="failed",
            qualified=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            retained_failure_roots=len(FAILURE_ROOTS),
        )
        code = 1
    G.save(args.output, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
