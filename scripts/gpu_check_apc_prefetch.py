#!/usr/bin/env python3
# ruff: noqa: EXE001
"""APCv2 resume/prefetch concurrency verification.

The real-artifact path is Metal-only and deliberately gated by
``--i-own-the-gpu``. ``--cpu`` runs the same ServingEngine/APCv2 lifecycle with
the built-in tiny Qwen4 fixture. The dry-run path imports no MLX modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

SCHEMA = "mlx2.apcv2-metal-prefetch-check.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-own-the-gpu",
        action="store_true",
        help="required acknowledgement for any Metal/model execution",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact plan without importing MLX or creating files",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="execute the built-in tiny Qwen4 fixture on CPU (no model path)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="optional real local artifact; otherwise use the built-in tiny Qwen4 fixture",
    )
    parser.add_argument("--dir", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--concurrent-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=600.0,
        help="bounded model-worker startup window (default: 600 seconds)",
    )
    return parser


def execution_plan(args) -> dict:
    model = (
        {"kind": "local_artifact", "path": str(args.model_path.expanduser().resolve())}
        if args.model_path
        else {"kind": "built_in_tiny_qwen4_hybrid_fixture"}
    )
    return {
        "schema": SCHEMA,
        "kind": "execution_plan",
        "will_execute": bool((args.cpu or args.i_own_the_gpu) and not args.dry_run),
        "requires_gpu_ownership": not args.cpu,
        "device": "cpu" if args.cpu else "metal",
        "model": model,
        "persistence_root": str(args.dir.expanduser().resolve()),
        "route": "ordinary_decode_with_apcv2",
        "requests": {
            "cold_max_tokens": args.max_tokens,
            "concurrent_max_tokens": args.concurrent_tokens,
            "warm_max_tokens": args.max_tokens,
            "temperature": 0,
        },
        "startup_timeout_seconds": args.startup_timeout_seconds,
        "steps": [
            "load one model in one ServingEngine",
            "cold-decode a session prompt and request a park of its exact APCv2 checkpoint",
            "wait boundedly for deferred COW leases to release and the worker idle boundary to complete the park",
            "start a different decode on the same model and pause the worker after a decode step",
            "queue session resume while that decode is active and assert no prefetch occurs",
            "release decode; wait for worker idle-boundary prefetch",
            "warm-decode the original prompt and compare captured token ids to cold decode",
            "quiesce with suspension, wait for the suspended state, and resume with an explicit session prefetch",
            "decode twice from the restored session and compare token ids to the pre-suspend warm decode",
        ],
        "pass_assertions": [
            "resume was queued while the same model worker was inside active decode",
            "deferred park completed with no pending entries at the first worker idle boundary",
            "prefetch restore count did not advance until the worker left active decode",
            "prefetch restore and subsequent APCv2 hit counters advanced",
            "warm token ids equal cold token ids exactly",
            "admin suspend/resume reaches suspended then resident and preserves exact decode token ids",
            "worker stayed alive with no MLX stream/thread error",
        ],
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class BatchProbe:
    """Capture response token ids and deterministically pause one decode step."""

    def __init__(self):
        self.tokens = defaultdict(list)
        self.lock = threading.Lock()
        self.block_next_response = False
        self.blocked_uid = None
        self.blocked = threading.Event()
        self.release = threading.Event()

    def arm(self):
        with self.lock:
            self.block_next_response = True
            self.blocked_uid = None
            self.blocked.clear()
            self.release.clear()

    def observe(self, responses):
        should_block = False
        with self.lock:
            for response in responses:
                self.tokens[int(response.uid)].append(int(response.token))
            if self.block_next_response and responses:
                self.block_next_response = False
                self.blocked_uid = int(responses[0].uid)
                should_block = True
                self.blocked.set()
        if should_block and not self.release.wait(timeout=120):
            raise TimeoutError("GPU check did not release the deliberately paused decode")


def _install_batch_probe(probe: BatchProbe):
    from mlx2.runtime.generate import BatchGenerator

    original = BatchGenerator.next

    def observed_next(batch, *args, **kwargs):
        prompts, responses = original(batch, *args, **kwargs)
        probe.observe(responses)
        return prompts, responses

    BatchGenerator.next = observed_next

    def restore():
        BatchGenerator.next = original

    return restore


def _tiny_adapter_factory():
    import mlx.core as mx

    from mlx2.runtime.models.qwen4_exp import Model, ModelArgs

    class Detokenizer:
        def __init__(self):
            self.last_segment = ""

        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = f"{int(token)} "

        def finalize(self):
            pass

    class Tokenizer:
        vocab_size = 64
        eos_token_ids: ClassVar[list[int]] = []

        @property
        def detokenizer(self):
            return Detokenizer()

    class Parser:
        stopped = False
        tool_count = 0

        def push(self, text, final=False):
            return [{"content": text}] if text else []

    class Adapter:
        max_context = 2048
        environment: ClassVar[dict[str, str]] = {}
        tokenizer = Tokenizer()

        def __init__(self, _path):
            text_config = dict(  # noqa: C408 - mirrors upstream model argument names
                model_type="qwen4_exp_text",
                hidden_size=32,
                intermediate_size=0,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=64,
                linear_num_value_heads=4,
                linear_num_key_heads=2,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_conv_kernel_dim=4,
                layer_types=["linear_attention", "full_attention"],
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=16,
                shared_expert_intermediate_size=16,
                hc_count=2,
                hc_lowrank=8,
                ple_layer_ids=[1],
                ple_embed_dim=32,
                ple_conv_kernel_size=4,
                ngram_size=3,
                heads_per_ngram=2,
                ngram_vocab_size_base=128,
                make_ngram_vocab_size_divisible_by=128,
                split_ngram_parts=1,
                indexer_n_heads=2,
                indexer_kv_heads=1,
                indexer_head_dim=8,
                indexer_budget=8,
                indexer_compress_ratio=2,
                mtp_num_hidden_layers=1,
                rope_parameters={
                    "type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 0.25,
                },
            )
            mx.random.seed(20260918)
            self.model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
            mx.eval(self.model.parameters())
            self.identity = {"fingerprint": "tiny-qwen4-apc-prefetch-check-v1"}
            self.layout = self.model.apc_v2_layout

        def profile_name(self, _mtp):
            return "tiny-qwen4-ordinary"

        def execution_config(self, **_kwargs):
            return {"num_draft": 0}

        def prompt_tokens(self, request):
            return list(request["tokens"])

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    return Adapter


def _engine_failure(engine, *, require_ready: bool) -> str | None:
    error = getattr(engine, "error", None)
    if error:
        return f"ServingEngine worker error: {error}"
    thread = getattr(engine, "thread", None)
    if thread is None or not thread.is_alive():
        return "ServingEngine worker thread exited"
    ready = getattr(engine, "ready", None)
    if require_ready and (ready is None or not ready.is_set()):
        return "ServingEngine worker is not ready"
    return None


def _wait_for_engine_ready(engine, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        failure = _engine_failure(engine, require_ready=False)
        if failure:
            raise RuntimeError(failure)
        if engine.ready.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic()))):
            failure = _engine_failure(engine, require_ready=True)
            if failure:
                raise RuntimeError(failure)
            return
    raise TimeoutError(f"ServingEngine did not become ready within {timeout}s")


def _assert_engine_alive(engine) -> None:
    failure = _engine_failure(engine, require_ready=True)
    if failure:
        raise RuntimeError(failure)


def _queue_empty_type():
    """Resolve the polling exception before work starts so CPU covers this path."""
    return queue.Empty


def _collect(job, *, timeout: float, engine=None) -> tuple[dict, str]:
    deadline = time.monotonic() + timeout
    queue_empty = _queue_empty_type()
    text = ""
    while True:
        if engine is not None:
            _assert_engine_alive(engine)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"request {job.id} did not finish")
        try:
            event = job.events.get(timeout=min(0.1, remaining))
        except queue_empty:
            continue
        if "error" in event:
            raise RuntimeError(f"request {job.id} failed: {event}")
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return event["receipt"], text


def _request(args, *, session_id: str | None, concurrent: bool = False) -> dict:
    max_tokens = args.concurrent_tokens if concurrent else args.max_tokens
    if args.model_path:
        prompt = (
            "Count upward from one, writing one integer per token for as long as allowed."
            if concurrent
            else "Reply with a short deterministic description of exact prefix caching."
        )
        value = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    else:
        tokens = list(range(21, 53)) if concurrent else [1, 7, 3, 9, 2, 8, 4, 6, 5]
        value = {"tokens": tokens, "max_tokens": max_tokens, "temperature": 0}
    if session_id is not None:
        value["session_id"] = session_id
    return value


def _wait_for(predicate, *, timeout: float, message: str, engine=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if engine is not None:
            _assert_engine_alive(engine)
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise TimeoutError(message)


def _wait_for_event(event, *, timeout: float, message: str, engine=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if engine is not None:
            _assert_engine_alive(engine)
        if event.wait(timeout=min(0.1, max(0.0, deadline - time.monotonic()))):
            return
    raise TimeoutError(message)


def run_engine_check(args) -> dict:
    import mlx.core as mx

    if args.cpu:
        mx.set_default_device(mx.cpu)
    from mlx2.serving import ServingEngine

    if not args.cpu and mx.default_device() == mx.cpu:
        raise RuntimeError("Metal check resolved to the CPU device")
    restore_cpu_admission = None
    if args.cpu:
        # The production admission reserve is sized for Apple-Silicon model
        # serving. The tiny CPU fixture still exercises the real scheduler and
        # APCv2; deterministic synthetic headroom keeps it out of a 60 s memory
        # wait that is unrelated to this lifecycle check.
        from mlx2 import memory
        from mlx2.runtime import os_memory

        restore_cpu_admission = (
            memory.execution_headroom,
            os_memory.physical_footprint_bytes,
        )
        memory.execution_headroom = lambda: 100 * 2**30
        os_memory.physical_footprint_bytes = lambda: 0
    scratch = args.dir.expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    persist = Path(tempfile.mkdtemp(prefix="mlx2-apc-metal-check-", dir=scratch))
    probe = BatchProbe()
    restore_probe = _install_batch_probe(probe)
    engine = None
    session_id = "apc-prefetch-metal-check"
    timings = {}
    try:
        engine = ServingEngine(
            str(args.model_path.expanduser().resolve()) if args.model_path else "tiny-qwen4",
            adapter_factory=None if args.model_path else _tiny_adapter_factory(),
            qualification_mode=True,
            mtp=False,
            max_inflight=4,
            max_lanes=2,
            max_context=4096 if args.model_path else 2048,
            prefill_step=128,
            cache_bytes=8 << 30,
            apc_persist_dir=str(persist),
            apc_session_pinned_disk_bytes=8 << 30,
            apc_session_pinned_disk_bytes_global=8 << 30,
            apc_session_pinned_resident_bytes=8 << 30,
        )
        _wait_for_engine_ready(
            engine,
            timeout=min(args.timeout_seconds, args.startup_timeout_seconds),
        )

        cold_started = time.perf_counter()
        cold_job = engine.submit(_request(args, session_id=session_id))
        cold_receipt, _ = _collect(
            cold_job, timeout=args.timeout_seconds, engine=engine
        )
        timings["cold_decode_seconds"] = time.perf_counter() - cold_started
        cold_tokens = list(probe.tokens[int(cold_job.uid)])
        if not cold_tokens:
            raise AssertionError("cold decode emitted no captured token ids")

        park_started = time.perf_counter()
        initial_park = engine.apc_session_park("default", session_id, ttl_seconds=300)
        park_wait_started = time.perf_counter()

        def completed_park():
            state = engine.apc_session_state("default", session_id)
            return state if (
                state["state"] == "disk"
                and state["park_pending"] == 0
                and state["disk_entries"] == state["entries"]
            ) else None

        parked = _wait_for(
            completed_park,
            timeout=args.timeout_seconds,
            message=f"session did not finish its deferred park: {initial_park}",
            engine=engine,
        )
        timings["park_wait_seconds"] = time.perf_counter() - park_wait_started
        timings["park_seconds"] = time.perf_counter() - park_started

        probe.arm()
        concurrent_started = time.perf_counter()
        concurrent_job = engine.submit(_request(args, session_id=None, concurrent=True))
        _wait_for_event(
            probe.blocked,
            timeout=args.timeout_seconds,
            message="concurrent decode never reached the deliberate worker pause",
            engine=engine,
        )
        if probe.blocked_uid != int(concurrent_job.uid):
            raise AssertionError("worker pause did not belong to the concurrent decode")

        prefetch_before = engine.apc.apc_stats["idle_disk"]["prefetch_restores_ok"]
        resume_started = time.perf_counter()
        resume_state = engine.apc_session_resume("default", session_id, ttl_seconds=60)
        timings["resume_enqueue_seconds"] = time.perf_counter() - resume_started
        prefetch_while_decode_blocked = engine.apc.apc_stats["idle_disk"][
            "prefetch_restores_ok"
        ]
        if prefetch_while_decode_blocked != prefetch_before:
            raise AssertionError("prefetch ran while the model worker was inside decode")
        if resume_state["state"] != "disk":
            raise AssertionError("resume unexpectedly restored outside the worker idle boundary")

        probe.release.set()
        _collect(concurrent_job, timeout=args.timeout_seconds, engine=engine)
        timings["concurrent_decode_seconds"] = time.perf_counter() - concurrent_started

        resident_started = time.perf_counter()
        _wait_for(
            lambda: engine.apc_session_state("default", session_id)["state"]
            == "resident",
            timeout=args.timeout_seconds,
            message="idle-boundary prefetch did not restore the session",
            engine=engine,
        )
        timings["decode_complete_to_prefetch_resident_seconds"] = (
            time.perf_counter() - resident_started
        )

        warm_started = time.perf_counter()
        warm_job = engine.submit(_request(args, session_id=session_id))
        warm_receipt, _ = _collect(
            warm_job, timeout=args.timeout_seconds, engine=engine
        )
        timings["warm_decode_seconds"] = time.perf_counter() - warm_started
        warm_tokens = list(probe.tokens[int(warm_job.uid)])
        if warm_tokens != cold_tokens:
            raise AssertionError(
                f"warm token ids differ from cold decode: cold={cold_tokens}, warm={warm_tokens}"
            )
        if int(warm_receipt.get("cached_tokens", 0)) <= 0:
            raise AssertionError("warm decode did not report an APCv2 prefix hit")
        stats = engine.apc.apc_stats["idle_disk"]
        if stats["prefetch_restores_ok"] <= prefetch_before:
            raise AssertionError("prefetch restore counter did not advance")
        if stats["prefetch_hits"] < 1:
            raise AssertionError("prefetched checkpoint was not consumed as a hit")
        if engine.error or not engine.thread.is_alive():
            raise AssertionError(f"model worker failed: {engine.error}")

        before_admin_suspend = engine.apc_session_state("default", session_id)
        admin_quiesce = engine.quiesce(
            drain_timeout_seconds=min(30.0, args.timeout_seconds),
            suspend=True,
        )
        if not engine.wait_for_quiesce(timeout=args.timeout_seconds):
            raise TimeoutError("admin quiesce did not reach its suspended boundary")
        admin_suspended = engine.service_state()
        if admin_suspended["state"] != "suspended":
            raise AssertionError(
                f"admin quiesce completed in {admin_suspended['state']!r}, not suspended"
            )
        suspended_session = engine.apc_session_state("default", session_id)
        if suspended_session["state"] != "disk":
            raise AssertionError("admin suspend did not park the session on disk")
        admin_resume = engine.resume(
            prefetch_sessions=(("default", session_id),)
        )
        if admin_resume["state"] != "serving":
            raise AssertionError("admin resume did not reopen serving admission")
        admin_resident = _wait_for(
            lambda: (
                state
                if (state := engine.apc_session_state("default", session_id))[
                    "state"
                ] == "resident"
                else None
            ),
            timeout=args.timeout_seconds,
            message="admin resume prefetch did not restore the session",
            engine=engine,
        )

        admin_warm_tokens = []
        admin_warm_receipts = []
        for _ in range(2):
            job = engine.submit(_request(args, session_id=session_id))
            receipt, _ = _collect(job, timeout=args.timeout_seconds, engine=engine)
            admin_warm_receipts.append(receipt)
            admin_warm_tokens.append(list(probe.tokens[int(job.uid)]))
        if any(tokens != warm_tokens for tokens in admin_warm_tokens):
            raise AssertionError(
                "admin suspend/resume changed deterministic decode token ids: "
                f"before={warm_tokens}, after={admin_warm_tokens}"
            )
        if any(int(receipt.get("cached_tokens", 0)) <= 0 for receipt in admin_warm_receipts):
            raise AssertionError("admin-restored warm decode did not report an APCv2 hit")

        return {
            "schema": SCHEMA,
            "passed": True,
            "model": execution_plan(args)["model"],
            "device": "cpu" if args.cpu else "metal",
            "timings": timings,
            "tokens": {
                "count": len(cold_tokens),
                "cold_sha256": hashlib.sha256(
                    json.dumps(cold_tokens, separators=(",", ":")).encode()
                ).hexdigest(),
                "warm_sha256": hashlib.sha256(
                    json.dumps(warm_tokens, separators=(",", ":")).encode()
                ).hexdigest(),
                "exact_match": True,
                "admin_suspend_resume_exact_match": True,
            },
            "receipts": {
                "cold_cached_tokens": cold_receipt.get("cached_tokens", 0),
                "warm_cached_tokens": warm_receipt.get("cached_tokens", 0),
                "admin_warm_cached_tokens": [
                    receipt.get("cached_tokens", 0)
                    for receipt in admin_warm_receipts
                ],
            },
            "apcv2": {
                "park_initial_state": initial_park["state"],
                "park_initial_pending": initial_park["park_pending"],
                "park_final_pending": parked["park_pending"],
                "park_disk_entries": parked["disk_entries"],
                "resume_queued_during_decode": True,
                "prefetch_count_while_decode_blocked": prefetch_while_decode_blocked,
                "prefetch_restores_ok": stats["prefetch_restores_ok"],
                "prefetch_hits": stats["prefetch_hits"],
                "admin_quiesce_initial_state": admin_quiesce["state"],
                "admin_suspended_state": admin_suspended["state"],
                "admin_suspended_session_state": suspended_session["state"],
                "admin_resumed_state": admin_resume["state"],
                "admin_resident_session_state": admin_resident["state"],
                "admin_entries_before_suspend": before_admin_suspend["entries"],
                "admin_entries_after_resume": admin_resident["entries"],
            },
            "mlx_stream_thread_errors": [],
        }
    finally:
        probe.release.set()
        if engine is not None:
            engine.close()
        restore_probe()
        if restore_cpu_admission is not None:
            from mlx2 import memory
            from mlx2.runtime import os_memory

            memory.execution_headroom, os_memory.physical_footprint_bytes = (
                restore_cpu_admission
            )
        shutil.rmtree(persist, ignore_errors=True)


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.max_tokens < 1 or args.concurrent_tokens < 2:
        parser.error("token counts must be positive and --concurrent-tokens must be at least 2")
    if args.timeout_seconds <= 0 or args.startup_timeout_seconds <= 0:
        parser.error("timeout values must be positive")
    if args.cpu and args.i_own_the_gpu:
        parser.error("--cpu and --i-own-the-gpu are mutually exclusive")
    if args.cpu and args.model_path:
        parser.error("--cpu uses only the built-in tiny Qwen4 fixture")
    if not args.dry_run and not args.cpu and not args.i_own_the_gpu:
        parser.error("refusing Metal/model execution without --i-own-the-gpu")


# Earlier name of run_engine_check (before the --cpu path); kept so existing
# callers and tests that patch it keep working.
run_metal_check = run_engine_check


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    plan = execution_plan(args)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    result = run_engine_check(args)
    if args.output:
        _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
