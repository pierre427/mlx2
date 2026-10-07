"""Opt-in, one-session APCv2 restore host-memory peak probe.

Dry-run performs source and argument preflight only. A live run needs the GPU
queue owner to hold both locks and pass --i-own-the-gpu. It creates one private
ServingEngine and never changes production policy or the APCv2 implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import queue
import shutil
import statistics
import subprocess
import threading
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APC_SOURCE = ROOT / "src/mlx2/runtime/apc_v2.py"
LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock/owner.json"), Path("/tmp/gpu.lock/owner.json"))
# The method holding the restore body (runtime/loop_trace wraps the public one).
RESTORE_METHOD = "_restore_entry_untraced_locked"
COUNTERS = ("restores", "bytes_read", "restore_budget_deferrals",
            "restore_transient_deferrals", "restore_failures", "resident_bytes")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_lines(source: Path = APC_SOURCE) -> dict[int, list[str]]:
    """Resolve the audited restore call sites without editing production code."""
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "APCv2")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    # The opt-in loop trace wraps the restore; its body lives in the untraced method.
    method = methods.get(RESTORE_METHOD) or methods["_restore_entry_locked"]
    phases: dict[int, list[str]] = {}

    def add(line: int, label: str) -> None:
        phases.setdefault(line, []).append(label)

    wanted = {"materialize_block_file", "load_prompt_cache", "mx.eval", "freeze_prompt_cache"}
    counts: dict[str, int] = {}
    calls = sorted((node for node in ast.walk(method) if isinstance(node, ast.Call)),
                   key=lambda node: (node.lineno, node.col_offset))
    for node in calls:
        name = ast.unparse(node.func)
        if name not in wanted:
            continue
        counts[name] = counts.get(name, 0) + 1
        label = name.replace(".", "_") + f"_{counts[name]}"
        add(node.lineno, label + "_before")
        add(node.end_lineno + 1, label + "_after")
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and ast.unparse(target) == "entry.prompt_cache"
            for target in node.targets
        ):
            add(node.lineno, "publication_before")
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is True:
            add(node.lineno, "publication_after")
    required = {"materialize_block_file", "load_prompt_cache", "mx.eval", "freeze_prompt_cache"}
    if set(counts) != required or not any("publication_before" in labels for labels in phases.values()) or not any(
        "publication_after" in labels for labels in phases.values()
    ):
        raise RuntimeError("APCv2 restore source changed; phase instrumentation refused")
    return phases


def gpu_owner() -> dict:
    owners = [json.loads(path.read_text()) for path in LOCKS]
    if not owners[0].get("lease_id") or owners[0] != owners[1]:
        raise RuntimeError("matching two-lock GPU owner receipts required")
    if not os.environ.get("GPUQ_SESSION") or owners[0].get("session") != os.environ["GPUQ_SESSION"]:
        raise RuntimeError("GPU queue session does not match the paired lock owner")
    return owners[0]


def parked_on_disk(state: dict) -> bool:
    """One session may have several persisted checkpoints after parking."""
    return (state["park_pending"] == 0 and state["resident_entries"] == 0
            and state["disk_entries"] >= 1)


def missing_restore_phases(observed: set[str]) -> list[str]:
    bases = ("materialize_block_file", "load_prompt_cache", "mx_eval",
             "freeze_prompt_cache", "publication")
    return [f"{base}{suffix}" for base in bases
            for suffix in ("_before", "_after")
            if not any(phase.startswith(base) and phase.endswith(suffix)
                       for phase in observed)]


class Sampler:
    """Independent 10–20 ms samples; phase stamps also sample immediately."""

    def __init__(self, read, interval_ms: int):
        self.read = read
        self.interval = interval_ms / 1000
        self.rows: list[dict] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.origin = time.monotonic_ns()

    def sample(self, phase: str) -> dict:
        with self.lock:
            row = {"elapsed_ns": time.monotonic_ns() - self.origin, "phase": phase,
                   "thread": threading.get_ident(), **self.read()}
            self.rows.append(row)
        return row

    def start(self) -> None:
        self.sample("pre_restore")

        def loop() -> None:
            while not self.stop_event.wait(self.interval):
                self.sample("interval")

        self.thread = threading.Thread(target=loop, daemon=True, name="apc-restore-memory-sampler")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.sample("post_restore")


def summarize(rows: list[dict]) -> dict:
    baseline = rows[0]
    fields = ("host_available_bytes", "physical_footprint_bytes",
              "mlx_active_bytes", "mlx_cache_bytes", "mlx_peak_bytes")
    result = {"samples": len(rows), "baseline": {field: baseline.get(field) for field in fields}}
    intervals = [row["elapsed_ns"] for row in rows if row["phase"] == "interval"]
    gaps_ms = [(later - earlier) / 1e6 for earlier, later in pairwise(intervals)]
    result["interval_samples"] = len(intervals)
    result["observed_interval_median_ms"] = statistics.median(gaps_ms) if gaps_ms else None
    result["observed_interval_max_ms"] = max(gaps_ms) if gaps_ms else None
    for field in fields:
        values = [row[field] for row in rows if type(row.get(field)) is int]
        extremum = (min(values) if field == "host_available_bytes" else max(values)) if values else None
        result[field] = {"sampled_extremum": extremum,
                         "delta_from_baseline": None if extremum is None or baseline.get(field) is None
                         else extremum - baseline[field]}
    return result


def collect(job, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    text = ""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request did not complete")
        try:
            event = job.events.get(timeout=min(remaining, 0.2))
        except queue.Empty:
            continue
        if "error" in event:
            raise RuntimeError(f"request failed: {event}")
        text += (event.get("delta") or {}).get("content", "")
        if "finish_reason" in event:
            return {"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "finish_reason": event["finish_reason"], "receipt": event.get("receipt") or {}}


def run(args, plan: dict) -> dict:
    owner = gpu_owner()
    import mlx.core as mx

    from mlx2.runtime import os_memory
    from mlx2.serving import ServingEngine

    if args.allocator_cache_limit_gib is not None:
        mx.set_cache_limit(int(args.allocator_cache_limit_gib * (1 << 30)))
    plan["gpu_owner"] = owner
    plan["mlx_cache_limit_bytes"] = mx.get_cache_limit() if hasattr(mx, "get_cache_limit") else None
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    phases = phase_lines()
    engine = None
    original_next = None
    sampler = None
    try:
        engine = ServingEngine(
            str(args.model.resolve()), qualification_mode=True, mtp=True,
            max_lanes=1, max_inflight=1, max_context=args.max_context,
            prefill_step=128, cache_bytes=args.cache_bytes,
            cache_dir=str(output / "cache"), apc_persist_dir=str(output / "cache"),
            apc_session_pinned_disk_bytes=args.cache_bytes,
            apc_session_pinned_disk_bytes_global=args.cache_bytes,
            apc_session_pinned_resident_bytes=args.cache_bytes,
            execution_policy={"host_memory_signals": {"enabled": True,
                "minimum_host_available_gib": args.host_floor_gib}},
        )
        startup_deadline = time.monotonic() + args.startup_timeout
        while not engine.ready.wait(0.1):
            if engine.error or time.monotonic() >= startup_deadline:
                raise RuntimeError(f"engine load failed: {engine.error or 'startup timeout'}")
        if engine.error:
            raise RuntimeError(f"engine load failed: {engine.error}")
        plan["cow_branching"] = bool(engine.apc._cow_branching)
        if not plan["cow_branching"]:
            raise RuntimeError("COW freezing is required for this restore overlap probe")
        plan["model_identity"] = getattr(engine.adapter, "identity", None)
        (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n")
        from mlx2.runtime.generate import BatchGenerator

        tokens: dict[int, list[int]] = {}
        original_next = BatchGenerator.next

        def capture_next(batch, *a, **kw):
            prompts, responses = original_next(batch, *a, **kw)
            for response in responses:
                tokens.setdefault(int(response.uid), []).append(int(response.token))
            return prompts, responses

        BatchGenerator.next = capture_next
        request = {"prompt": args.prompt, "session_id": "s1", "max_tokens": 3,
                   "min_tokens": 3, "temperature": 0}
        first_job = engine.submit(dict(request))
        first = collect(first_job, args.request_timeout)
        first["tokens"] = tokens.get(int(first_job.uid), [])
        engine.apc_session_park("default", "s1", ttl_seconds=600)
        park_deadline = time.monotonic() + args.request_timeout
        while True:
            parked = engine.apc_session_state("default", "s1")
            if parked_on_disk(parked):
                break
            if time.monotonic() >= park_deadline:
                raise TimeoutError(f"APCv2 session failed to park: {parked}")
            time.sleep(0.05)
        before = dict(engine.apc.apc_stats["idle_disk"])

        def read_memory() -> dict:
            host = os_memory.host_memory_snapshot()
            return {"host_available_bytes": None if host is None else host.available_bytes,
                    "physical_footprint_bytes": os_memory.physical_footprint_bytes(),
                    "mlx_active_bytes": int(mx.get_active_memory()),
                    "mlx_cache_bytes": int(mx.get_cache_memory()),
                    "mlx_peak_bytes": int(mx.get_peak_memory())}

        sampler = Sampler(read_memory, args.sample_ms)
        # Resolve this only after adapter load. Importing APCv2 here before
        # the Qwen adapter pins its import-time GDN profile loads models.base
        # too early and causes a guarded startup refusal.
        apc_type = type(engine.apc)
        restore_code = getattr(apc_type, RESTORE_METHOD, apc_type._restore_entry_locked).__code__

        def trace(frame, event, _arg):
            if frame.f_code is restore_code and event == "line":
                for label in phases.get(frame.f_lineno, ()):
                    sampler.sample(label)
            return trace

        sampler.start()
        threading.settrace_all_threads(trace)
        try:
            second_job = engine.submit(dict(request))
            second = collect(second_job, args.request_timeout)
            second["tokens"] = tokens.get(int(second_job.uid), [])
        finally:
            threading.settrace_all_threads(None)
            sampler.stop()
        after = dict(engine.apc.apc_stats["idle_disk"])
        if first["tokens"] != second["tokens"] or len(first["tokens"]) != 3:
            raise RuntimeError("restored response tokens differ or are incomplete")
        restore_count = after["restores"] - before["restores"]
        if restore_count < 1:
            raise RuntimeError("expected at least one APCv2 disk restore")
        if after.get("bytes_read", 0) <= before.get("bytes_read", 0):
            raise RuntimeError("APCv2 restore did not read a disk snapshot")
        for key in ("restore_budget_deferrals", "restore_transient_deferrals", "restore_failures"):
            if after.get(key, 0) != before.get(key, 0):
                raise RuntimeError(f"restore counter increased: {key}")
        state = engine.status()
        if state["inflight"] or state["queue_depth"]:
            raise RuntimeError("request remained active after restore")
        rows = list(sampler.rows)
        observed_phases = {row["phase"] for row in rows}
        missing = missing_restore_phases(observed_phases)
        if missing:
            raise RuntimeError(f"restore phase tracer missed {missing}")
        memory = summarize(rows)
        if any(memory[field]["sampled_extremum"] is None for field in (
            "host_available_bytes", "physical_footprint_bytes",
        )):
            raise RuntimeError("host availability or physical footprint unavailable")
        with (output / "samples.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        result = {"schema": "mlx2.mtplx-apcv2-restore-host-peak.v1",
                  "sample_interval_ms": args.sample_ms, "sampled_peak_only": True,
                  "disk_restore_count": restore_count,
                  "first": first, "second": second,
                  "idle_disk_before": {key: before.get(key) for key in COUNTERS},
                  "idle_disk_after": {key: after.get(key) for key in COUNTERS},
                  "parked": parked, "memory": memory,
                  "phase_counts": {phase: sum(row["phase"] == phase for row in rows)
                                   for labels in phases.values() for phase in labels},
                  "model_identity": getattr(engine.adapter, "identity", None),
                  "final_state": {"healthy": state["healthy"], "inflight": state["inflight"],
                                  "queue_depth": state["queue_depth"]},
                  "valid": True}
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        return result
    except BaseException as error:
        if sampler is not None:
            with (output / "samples.jsonl").open("w") as handle:
                for row in sampler.rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        (output / "failed.json").write_text(json.dumps({"valid": False, "error": repr(error)}, indent=2) + "\n")
        raise
    finally:
        if original_next is not None:
            BatchGenerator.next = original_next
        if engine is not None:
            engine.close()
        shutil.rmtree(output / "cache", ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="Reply with one short fact about apples.")
    parser.add_argument("--host-floor-gib", required=True, type=float)
    parser.add_argument("--sample-ms", type=int, choices=range(10, 21), default=15)
    parser.add_argument("--cache-bytes", type=int, default=8 << 30)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--allocator-cache-limit-gib", required=True, type=float)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=90)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not 0 < args.host_floor_gib < 128 or args.cache_bytes <= 0 or args.max_context <= 0:
        parser.error("floor, cache bytes, and context must be positive and bounded")
    if not 0 < args.allocator_cache_limit_gib <= 32:
        parser.error("allocator cache limit must be positive and at most 32 GiB")
    if not args.prompt.strip() or args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("prompt and timeouts must be positive")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error("model artifact with config.json required")
    phases = phase_lines()
    plan = {"schema": "mlx2.mtplx-apcv2-restore-host-peak-plan.v1",
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_apc_sha256": sha256(APC_SOURCE), "model": str(args.model.resolve()),
            "model_config_sha256": sha256(args.model / "config.json"),
            "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
            "prompt": args.prompt,
            "host_floor_gib": args.host_floor_gib, "cache_bytes": args.cache_bytes,
            "max_context": args.max_context, "sample_ms": args.sample_ms,
            "allocator_cache_limit_gib": args.allocator_cache_limit_gib,
            "session": "default/s1", "max_tokens": 3, "temperature": 0,
            "phase_lines": phases, "output_dir": str(args.output_dir.resolve()),
            "requires_gpu_owner": True, "execution_policy": {"host_memory_signals": {
                "enabled": True, "minimum_host_available_gib": args.host_floor_gib}}}
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.i_own_the_gpu:
        parser.error("live restore requires --i-own-the-gpu and an external GPU queue lease")
    run(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
