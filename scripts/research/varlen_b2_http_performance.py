"""Source-bound six-cell ordinary/grouped B2 HTTP latency/decode screen.

This reports one research workload. It never creates a route price or
qualification. Run only inside one owned, externally capped GPUQ lease.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from varlen_b2_http_gate import _contexts, _prompt, _route
from varlen_pack_price_bench import MAX_RESIDENT_BYTES, _gpuq_owner, verify_artifact_manifest
from varlen_staged_graph_price import preflight_cpu

SCHEMA = "mlx2.varlen-b2-http-performance.v1"
ORDER = ("ordinary", "grouped", "grouped", "ordinary", "ordinary", "grouped")


def _timeout(_signum, _frame):
    raise TimeoutError("B2 HTTP performance screen exceeded 160 seconds")


def _post(base, body):
    request = Request(base + "/v1/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    start = time.perf_counter_ns()
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response), (time.perf_counter_ns() - start) / 1e6
    except HTTPError as error:
        with error:
            return error.code, json.load(error), (time.perf_counter_ns() - start) / 1e6


def _cell_requests(base, bodies, *, engine, cohort_id=None):
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(_post, base, bodies[0])
        if cohort_id is not None:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with engine.submission_lock:
                    staged = engine.pending_cohorts.get(("default", cohort_id))
                    if staged is not None and len(staged["jobs"]) == 1:
                        break
                time.sleep(.002)
            else:
                raise RuntimeError("first B2 HTTP job did not stage its cohort")
        second = workers.submit(_post, base, bodies[1])
        return first.result(timeout=35), second.result(timeout=35)


def _ready_ms(events, ids):
    ready = [event for event in events if
             any(job_id in ids and ordinal == 2
                 for job_id, ordinal in event["responses"])]
    if not ready or any(job_id not in ids for event in ready
                        for job_id, _ in event["responses"]):
        raise RuntimeError("HTTP ready-step event missing or crossed cells")
    return sum(event["elapsed_ms"] for event in ready), ready


def _decode_progress(events, ids, maximum, *, paired):
    """Count actual generator tokens after first response, with no fallback."""
    steps = [event for event in events if any(
        job_id in ids and ordinal >= 2 for job_id, ordinal in event["responses"])]
    observed = {job_id: [] for job_id in ids}
    for event in steps:
        responses = event["responses"]
        if (any(job_id not in ids for job_id, _ in responses) or
                paired and len(responses) != 2):
            raise RuntimeError("B2 decode steps lost a partner or crossed cells")
        for job_id, ordinal in responses:
            if ordinal >= 2:
                observed[job_id].append(ordinal)
    if any(ordinals != list(range(2, maximum + 1)) for ordinals in observed.values()):
        raise RuntimeError("B2 decode did not produce the pinned sustained token count")
    wall_ms = sum(event["elapsed_ms"] for event in steps)
    if wall_ms <= 0:
        raise RuntimeError("B2 decode wall time is missing")
    return {"generator_decode_ms": wall_ms,
            "generated_decode_tokens": 2 * (maximum - 1),
            "generator_decode_tokens_per_second": 2000 * (maximum - 1) / wall_ms,
            "decode_events": len(steps)}


def execute(manifest, profile_path):
    contexts = _contexts(manifest)
    preflight_cpu(manifest, context_validator=_contexts)
    gpuq_owner = _gpuq_owner()
    import _paged_kv_native
    import mlx.core as mx
    from mlx2 import serving
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime import qwen3_paged_graph_factory
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.paged_b2_research_profile import (
        COMBINED_FLAGS, DIRECT_FENCE_FLAG, STOCK_SDPA_FLAG, STRIPES_FLAG,
        SCHEMA_V2, SCHEMA_V4, SCHEMA_V5,
        SCHEMA_V6, SCHEMA_V7,
        SUSTAINED_FLAG, VECTOR_ROPE_FLAG, load_b2_research_profile)
    from mlx2.runtime.paged_price_identity import compute_live_price_identity
    from mlx2.server import handler_for
    from varlen_live_qwen3_request_driver import retired_native_state

    kernel = Path(_paged_kv_native.__file__).resolve()
    if kernel != Path(manifest["paths"]["kernel"]).resolve():
        raise RuntimeError("loaded native binary differs from manifest")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("GPU execution required")
    profile = load_b2_research_profile(
        profile_path, live_identity=manifest["identity"], context_lengths=contexts)
    if profile["schema"] not in (SCHEMA_V2, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6,
                                  SCHEMA_V7):
        raise RuntimeError("performance screen requires bounded or sustained B2 profile")
    sustained = profile["schema"] in (SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7)
    vector_rope = profile["schema"] == SCHEMA_V5
    direct_fence = profile["schema"] in (SCHEMA_V6, SCHEMA_V7)
    combined = profile["schema"] == SCHEMA_V7
    stripes = profile["q1_simd_stripes"] if combined else 4
    options = profile["combined_optimizations"] if combined else {
        "deferred_eval": False, "deferred_write_eval": False,
        "inline_metadata": False,
        "grouped_sampler": False, "stock_sdpa": False}
    model_root = Path(verify_artifact_manifest(
        Path(manifest["paths"]["artifact"]))["root"])
    events, sampled, effective, captures = [], {}, {}, []
    cell_state = {"arm": None, "first_tokens": set(), "prefill_profile": None,
                  "first_pair_generator_ms": 0.0, "first_pair_next_calls": 0,
                  "early_decode": False}
    original_next = BatchGenerator.next
    original_factory = qwen3_paged_graph_factory.create_shared_qwen3_graph_pack
    original_install = serving.install_explicit_native_qwen3_b2_cohort
    engine = server = server_thread = None

    def observed_next(batch, *args, **kwargs):
        start = time.perf_counter_ns()
        prompt, responses = original_next(batch, *args, **kwargs)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        before_first_pair = (cell_state["arm"] is not None and
                             len(cell_state["first_tokens"]) < 2)
        response_ids = []
        for response in responses:
            job = next((job for job in engine.jobs.values()
                        if job.uid == response.uid), None)
            if job is None:
                raise RuntimeError("generator response has no live HTTP job")
            sampled.setdefault(job.id, []).append(int(response.token))
            effective[job.id] = dict(job.effective_sampling or {})
            response_ids.append((job.id, len(sampled[job.id])))
            if before_first_pair and len(sampled[job.id]) >= 2:
                cell_state["early_decode"] = True
            if len(sampled[job.id]) == 1:
                cell_state["first_tokens"].add(job.id)
        if before_first_pair:
            cell_state["first_pair_generator_ms"] += elapsed_ms
            cell_state["first_pair_next_calls"] += 1
        if (cell_state["arm"] == "grouped" and
                len(cell_state["first_tokens"]) == 2 and
                cell_state["prefill_profile"] is None and captures):
            cell_state["prefill_profile"] = captures[-1][1].backend.profile_counters_snapshot()
        if response_ids:
            events.append({"elapsed_ms": elapsed_ms, "responses": response_ids})
        return prompt, responses

    def captured_factory(*args, **kwargs):
        owners, candidate = original_factory(*args, **kwargs)
        captures.append((owners, candidate))
        return owners, candidate

    def timed_native_install(*args, **kwargs):
        started = time.perf_counter_ns()
        try:
            return original_install(*args, **kwargs)
        finally:
            if cell_state["arm"] == "grouped":
                cell_state["native_cohort_install_calls"] += 1
                cell_state["native_cohort_install_ms"] = (
                    time.perf_counter_ns() - started) / 1e6

    class GateAdapter(StandardDecoderAdapter):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.model.apply(lambda parameter: parameter.astype(mx.float16))
            mx.eval(self.model.parameters())
            if (len(self.model.layers) != 28 or
                    self.model.model.embed_tokens.weight.dtype != mx.float16):
                raise RuntimeError("screen needs pinned fp16 Qwen3-0.6B")

    env_keys = ("MLX2_NATIVE_PAGED_B2_PROFILE", "MLX2_NATIVE_PAGED_MANIFEST",
                "MLX2_NATIVE_PAGED_MLX_WHEEL", "MLX2_PAGED_Q1_SIMD_TILE",
                "MLX2_PAGED_GROUPED_Q1_WRITE", "MLX2_PAGED_PRIVATE_TAIL_REUSE",
                "MLX2_PAGED_B2_PACKED_PREFILL", SUSTAINED_FLAG,
                VECTOR_ROPE_FLAG, DIRECT_FENCE_FLAG, STRIPES_FLAG,
                STOCK_SDPA_FLAG, *COMBINED_FLAGS)
    old_env = {key: os.environ.get(key) for key in env_keys}
    os.environ.update({
        "MLX2_NATIVE_PAGED_B2_PROFILE": str(profile_path),
        "MLX2_NATIVE_PAGED_MANIFEST": manifest["paths"]["artifact"],
        "MLX2_NATIVE_PAGED_MLX_WHEEL": manifest["paths"]["mlx_wheel"],
        "MLX2_PAGED_Q1_SIMD_TILE": "1", "MLX2_PAGED_GROUPED_Q1_WRITE": "1",
        "MLX2_PAGED_PRIVATE_TAIL_REUSE": "1",
        STRIPES_FLAG: str(stripes),
        STOCK_SDPA_FLAG: "1" if options["stock_sdpa"] else "0",
        **{key: "1" if enabled else "0" for key, enabled in zip(
            COMBINED_FLAGS, (options["deferred_eval"],
                             options["deferred_write_eval"],
                             options["inline_metadata"],
                             options["grouped_sampler"]))},
    })
    if sustained:
        os.environ["MLX2_PAGED_B2_PACKED_PREFILL"] = "1"
        os.environ[SUSTAINED_FLAG] = "1"
    if vector_rope:
        os.environ[VECTOR_ROPE_FLAG] = "1"
    if direct_fence:
        os.environ[DIRECT_FENCE_FLAG] = "1"
    BatchGenerator.next = observed_next
    qwen3_paged_graph_factory.create_shared_qwen3_graph_pack = captured_factory
    serving.install_explicit_native_qwen3_b2_cohort = timed_native_install
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(160)
    try:
        engine = serving.ServingEngine(
            str(model_root), adapter_factory=GateAdapter, max_lanes=2,
            max_inflight=2, mtp=False, qualification_mode=False,
            prefill_step=max(65, max(contexts)), cache_bytes=1 << 29)
        if not engine.ready.wait(30) or engine.error:
            raise RuntimeError(f"HTTP serving engine did not load: {engine.error}")
        live = compute_live_price_identity(
            manifest["paths"]["artifact"], manifest["paths"]["mlx_wheel"],
            kernel, adapter_artifact_root=engine.adapter.identity["path"])
        if live != manifest["identity"]:
            raise RuntimeError("loaded HTTP model/source/wheel differs")
        prompts = (_prompt(engine.adapter, 1000, contexts[0]),
                   _prompt(engine.adapter, 1001, contexts[1]))
        if prompts[0][1] == prompts[1][1]:
            raise RuntimeError("B2 prompts are not independent")
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        common = tuple({"model": model_root.name, "prompt": prompt,
                        "max_tokens": profile["max_tokens"], "temperature": 0,
                        "repetition_penalty": 1, "presence_penalty": 0,
                        "frequency_penalty": 0,
                        "skip_writing_prefix_cache": True}
                       for prompt, _ in prompts)
        baseline_tokens = baseline_text = baseline_sampling = None
        cells = []
        for index, arm in enumerate(ORDER):
            event_start, capture_start = len(events), len(captures)
            cell_state.update(arm=arm, first_tokens=set(), prefill_profile=None,
                              first_pair_generator_ms=0.0,
                              first_pair_next_calls=0, early_decode=False,
                              native_cohort_install_ms=None,
                              native_cohort_install_calls=0)
            cohort_id = f"perf-b2-{index}"
            bodies = tuple({**body, "paged_native_qwen3_b2": True,
                            "batch_cohort": {"id": cohort_id, "size": 2}}
                           for body in common) if arm == "grouped" else common
            responses = _cell_requests(base, bodies, engine=engine,
                                       cohort_id=cohort_id if arm == "grouped" else None)
            if any(status != 200 for status, _, _ in responses):
                raise RuntimeError(f"{arm} HTTP cell {index} failed: {responses}")
            payloads = [body for _, body, _ in responses]
            if any(body.get("mlx2", {}).get("qualification") != "unqualified"
                   for body in payloads):
                raise RuntimeError("HTTP service status is not unqualified")
            ids = [body.get("id") for body in payloads]
            if None in ids or len(set(ids)) != 2:
                raise RuntimeError("HTTP response IDs missing or repeated")
            tokens = [sampled.get(job_id) for job_id in ids]
            if any(type(value) is not list or len(value) != profile["max_tokens"]
                   for value in tokens):
                raise RuntimeError("HTTP sampled token IDs missing")
            text = [body["choices"][0]["text"] for body in payloads]
            sampling = [effective.get(job_id) for job_id in ids]
            if any(type(value) is not dict or value.get("temperature") != 0 or
                   value.get("repetition_penalty") != 1 or
                   value.get("presence_penalty") != 0 or
                   value.get("frequency_penalty") != 0 for value in sampling):
                raise RuntimeError("effective greedy sampler fields differ")
            if baseline_tokens is None:
                baseline_tokens, baseline_text, baseline_sampling = tokens, text, sampling
            elif (tokens != baseline_tokens or text != baseline_text or
                  sampling != baseline_sampling):
                raise RuntimeError("HTTP cell differs from ordinary output or sampler")
            ready_ms, ready_events = _ready_ms(events[event_start:], set(ids))
            first_pair_ms = cell_state["first_pair_generator_ms"]
            if len(cell_state["first_tokens"]) != 2 or first_pair_ms <= 0:
                raise RuntimeError("first-pair generator work boundary missing")
            first_pair_work = {
                "generator_wall_ms": first_pair_ms,
                "generator_next_calls": cell_state["first_pair_next_calls"],
                "prompt_tokens": sum(contexts),
                "includes_first_token_sampling": True,
                "includes_decode_before_both_first_tokens": cell_state["early_decode"],
                "includes_native_cohort_install": False,
                "effective_prompt_tokens_per_second": (
                    None if arm == "grouped" or cell_state["early_decode"] else
                    1000 * sum(contexts) / first_pair_ms),
            }
            native_install_ms = cell_state["native_cohort_install_ms"]
            if arm == "grouped" and (cell_state["native_cohort_install_calls"] != 1 or
                                     native_install_ms is None or native_install_ms <= 0):
                raise RuntimeError("native cohort install timing boundary missing")
            native_install_work = (None if arm != "grouped" else {
                "cohort_install_including_packed_prefill_ms": native_install_ms,
                "prompt_tokens": sum(contexts),
                "effective_prompt_tokens_per_second_including_install": (
                    1000 * sum(contexts) / native_install_ms),
                "scope": "shared_arena_allocation_packed_prefill_and_attachment",
            })
            decode_progress = (_decode_progress(
                events[event_start:], set(ids), profile["max_tokens"],
                paired=arm == "grouped") if sustained else None)
            receipts = ([_route(body) for body in payloads] if arm == "grouped"
                        else [{"route": body["mlx2"].get("route"),
                               "qualification": body["mlx2"].get("qualification")}
                              for body in payloads])
            native_host_profile = ready_host_ns = None
            if arm == "grouped":
                if len(captures) != capture_start + 1:
                    raise RuntimeError("grouped cell did not allocate one shared arena")
                for receipt in receipts:
                    if (receipt.get("route") != "native_qwen3_paged_b2" or
                            receipt.get("selected") is not True or
                            receipt.get("observed_used") is not True or
                            receipt.get("qualified") is not False or
                            receipt.get("price_usable") is not False or
                            receipt.get("admission_profile") != profile["profile_id"] or
                            receipt.get("native_read_delta") != 28 or
                            receipt.get("native_span_counts") != [2] * 28 or
                            receipt.get("native_read_calls") != 28 * (
                                profile["max_tokens"] + (0 if sustained else 1)) or
                            receipt.get("terminal_successes") != 28 * (
                                profile["max_tokens"] + (0 if sustained else 1)) or
                            receipt.get("prefill_mode") != (
                                "packed_staged" if sustained else "serial_native") or
                            receipt.get("vector_q1_rope_calls") != (
                                28 * (profile["max_tokens"] - 1) if vector_rope else 0) or
                            receipt.get("direct_grouped_fence_reads") != (
                                28 * (profile["max_tokens"] - 1) if direct_fence else 0) or
                            receipt.get("physical_dispatches") != {
                                "grouped_q1_writes": 28,
                                "native_write_dispatches": 0,
                                "q1_tile_dispatches": 0 if options["stock_sdpa"] else 28} or
                            receipt.get("combined_optimizations") != options or
                            receipt.get("q1_simd_stripes") != stripes or
                            receipt.get("combined_submission_delta") != {
                                "grouped_write_async_evals": 0 if options["deferred_eval"] or options["deferred_write_eval"] else 28,
                                "staged_read_async_evals": 0 if options["deferred_eval"] else 28,
                                "deferred_q1_write_roots": 28 if options["deferred_eval"] or options["deferred_write_eval"] else 0,
                                "deferred_q1_read_roots": 28 if options["deferred_eval"] or options["deferred_write_eval"] else 0,
                                "deferred_q1_final_evals": 1 if options["deferred_eval"] or options["deferred_write_eval"] else 0,
                                "deferred_q1_failure_flushes": 0,
                                "q1_metadata_dispatches": 28 if options["inline_metadata"] and not options["stock_sdpa"] else 0,
                                "q1_gather_dispatches": 28 if options["stock_sdpa"] else 0,
                                "stock_sdpa_graph_calls": 28 if options["stock_sdpa"] else 0,
                                **{f"q1_stripe_dispatches_{value}": 28 if not options["stock_sdpa"] and stripes == value else 0
                                   for value in (8, 16, 32)}} or
                            receipt.get("grouped_sampler_eval") is not options["grouped_sampler"]):
                        raise RuntimeError("grouped HTTP physical route proof differs")
                    if (sustained and receipt.get("native_prefill_proof") != {
                            "read_calls": 28, "terminal_successes": 28,
                            "span_counts": [2] * 28}):
                        raise RuntimeError("sustained packed prefill proof differs")
                if [receipt.get("output_token_ids") for receipt in receipts] != tokens:
                    raise RuntimeError("grouped route output IDs differ")
                owners, candidate = captures[-1]
                deadline = time.monotonic() + 3
                while (not all(owner.fully_retired for owner in owners) and
                       time.monotonic() < deadline):
                    time.sleep(.005)
                pending, retained, released = retired_native_state(candidate.backend.writer)
                if not released or not all(owner.fully_retired for owner in owners):
                    raise RuntimeError("grouped cell retained native pages or epochs")
                physical_total = candidate.backend.profile_snapshot()
                if sustained:
                    if (any(physical_total.get(key) != value for key, value in (
                        ("grouped_q1_writes", 28 * (profile["max_tokens"] - 1)),
                        ("q1_tile_dispatches", 0 if options["stock_sdpa"] else
                         28 * (profile["max_tokens"] - 1)),
                        ("q1_gather_dispatches",
                         28 * (profile["max_tokens"] - 1) if options["stock_sdpa"] else 0),
                        ("stock_sdpa_graph_calls",
                         28 * (profile["max_tokens"] - 1) if options["stock_sdpa"] else 0),
                        ("direct_grouped_fence_reads",
                         28 * (profile["max_tokens"] - 1) if direct_fence else 0),
                        ("q1_metadata_dispatches",
                         28 * (profile["max_tokens"] - 1) if options["inline_metadata"] and not options["stock_sdpa"] else 0),
                        *((f"q1_stripe_dispatches_{value}",
                           28 * (profile["max_tokens"] - 1) if not options["stock_sdpa"] and stripes == value else 0)
                          for value in (8, 16, 32)),
                    )) or type(physical_total.get("native_write_dispatches")) is not int or
                            physical_total["native_write_dispatches"] < 1):
                        raise RuntimeError("sustained physical dispatch totals differ")
                native_host_profile = receipts[0].get("native_host_profile")
                before_profile = cell_state["prefill_profile"]
                if (type(native_host_profile) is not dict or
                        type(native_host_profile.get("host_ns")) is not dict or
                        type(before_profile) is not dict or
                        type(before_profile.get("host_ns")) is not dict):
                    raise RuntimeError("grouped ready-step host profile boundary missing")
                ready_host_ns = {
                    key: value - before_profile["host_ns"].get(key, 0)
                    for key, value in native_host_profile["host_ns"].items()
                }
                if any(value < 0 for value in ready_host_ns.values()):
                    raise RuntimeError("grouped ready-step host profile regressed")
                # Capture detailed cumulative read work once, outside timing.
                native_host_profile = physical_total
                if (len(ready_events) != 1 or
                        sorted(ready_events[0]["responses"]) !=
                        sorted((job_id, 2) for job_id in ids)):
                    raise RuntimeError("grouped ready step did not return two responses")
            else:
                if len(captures) != capture_start or any(
                    receipt.get("route") == "native_qwen3_paged_b2"
                    for receipt in receipts
                ):
                    raise RuntimeError("ordinary cell allocated or selected native B2")
                pending = retained = 0
            cells.append({"index": index, "arm": arm, "context_tokens": list(contexts),
                          "request_ms": [elapsed for _, _, elapsed in responses],
                          "complete_pair_ms": max(elapsed for _, _, elapsed in responses),
                          "ready_step_ms": ready_ms, "ready_events": ready_events,
                          "first_pair_generator_work": first_pair_work,
                          "native_cohort_install_work": native_install_work,
                          "decode_progress": decode_progress,
                          "token_ids": tokens, "effective_sampling": sampling,
                          "route_receipts": receipts,
                          "native_host_profile_cumulative": native_host_profile,
                          "ready_host_ns": ready_host_ns,
                          "pending_native_epochs": pending,
                          "retained_pages": retained})
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RESIDENT_BYTES:
                raise RuntimeError("B2 HTTP screen exceeded memory cap")
        return {"schema": SCHEMA, "status": "passed", "gpu_executed": True,
                "identity": live, "gpuq_owner": gpuq_owner,
                "profile_id": profile["profile_id"], "order": list(ORDER),
                "max_tokens": profile["max_tokens"],
                "contexts": list(contexts), "cells": cells,
                "qualified": False, "price_usable": False,
                "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    finally:
        signal.alarm(0)
        BatchGenerator.next = original_next
        qwen3_paged_graph_factory.create_shared_qwen3_graph_pack = original_factory
        serving.install_explicit_native_qwen3_b2_cohort = original_install
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5)
        if engine is not None:
            engine.close()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    contexts = _contexts(manifest)
    if args.preflight:
        result = preflight_cpu(manifest, context_validator=_contexts)
        from mlx2.runtime.paged_b2_research_profile import (
            SCHEMA_V2, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7,
            load_b2_research_profile)
        profile = load_b2_research_profile(
            args.profile, live_identity=manifest["identity"], context_lengths=contexts)
        if profile["schema"] not in (SCHEMA_V2, SCHEMA_V4, SCHEMA_V5,
                                     SCHEMA_V6, SCHEMA_V7):
            raise RuntimeError("performance screen requires bounded or sustained B2 profile")
    else:
        if args.receipt is None:
            parser.error("GPU screen requires --receipt")
        try:
            result = execute(manifest, args.profile)
        except Exception as error:
            result = {"schema": SCHEMA, "status": "failed",
                      "identity": manifest.get("identity"),
                      "error_type": type(error).__name__, "error": str(error),
                      "qualified": False, "price_usable": False}
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            raise
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
