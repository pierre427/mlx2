#!/usr/bin/env python3
"""Source-bound HTTP A/B for padded versus live-row MLP-family prefill.

One loaded adapter serves both arms.  The research harness changes only the
model's adapter-installed varlen handle at idle arm boundaries; production
request routing never toggles it.  Every request disables APCv2 writes.

The dense family compacts individual dense MLPs.  The Qwen4 family compacts a
complete sparse-MoE branch, including routing, routed experts and its shared
expert.  The scheduler remains unaware of either model geometry.
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
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

import deterministic_arbitrary_http_campaign as campaign
import varlen_n20_prompt_lookup_http_gate as gate

MAX_SECONDS = 420
MAX_RSS_BY_FAMILY = {
    "qwen38-dense": 48 << 30,
    "qwen4-sparse": 72 << 30,
}
QWEN4_MODEL = (
    Path.home()
    / "mlx-models"
    / "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"
)


FAMILIES = {
    "qwen38-dense": {
        "policy_key": "varlen_dense_mlp",
        "adapter_field": "varlen_dense_mlp",
        "model_field": "_varlen_dense_mlp",
        "schema": "mlx2.varlen-dense-mlp.v1",
        "gate_schema": "mlx2.varlen-dense-mlp-http-gate.v1",
        "counter_prefix": "mlp",
    },
    "qwen4-sparse": {
        "policy_key": "varlen_sparse_moe",
        "adapter_field": "varlen_sparse_moe",
        "model_field": "_varlen_sparse_moe",
        "schema": "mlx2.varlen-sparse-moe.v1",
        "gate_schema": "mlx2.varlen-sparse-moe-http-gate.v1",
        "counter_prefix": "moe",
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counter_delta(before, after):
    keys = set(before) | set(after)
    result = {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}
    if any(value < 0 for value in result.values()):
        raise RuntimeError("varlen mechanism counters moved backwards")
    return {key: value for key, value in sorted(result.items()) if value}


def load_trace(inputs_path, trace_path):
    inputs = json.loads(Path(inputs_path).read_text())
    trace = json.loads(Path(trace_path).read_text())
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
        items = [item for item in trace["requests"] if item["wave"] == wave_index]
        rows = tuple(rows_by_id[item["case_id"]] for item in items)
        if len(rows) != 3 or len({row["case_id"] for row in rows}) != 3:
            raise ValueError("each deterministic wave requires three unique rows")
        waves.append((items, rows))
    return inputs, trace, tuple(waves)


def execute(args, result):
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if source != args.expected_source:
        raise RuntimeError("source revision differs from explicit gate binding")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("source worktree must be clean")

    inputs, trace, waves = load_trace(args.inputs, args.trace)
    if args.waves > len(waves):
        raise ValueError("requested wave count exceeds the frozen trace")

    sys.path.insert(0, str(ROOT / "src"))
    family = FAMILIES[args.family]
    if args.family == "qwen4-sparse":
        from mlx2.adapters.flash_next import FlashNextAdapter as Adapter
    else:
        from mlx2.adapters.qwen38_27b import Qwen3827BAdapter as Adapter
        from mlx2.adapters.qwen38_27b import configure_environment

        configure_environment()
    import mlx.core as mx
    from varlen_pack_price_bench import _gpuq_owner

    from mlx2 import serving
    from mlx2.server import handler_for, validate_request

    result.update(
        status="running",
        gpu_executed=True,
        source_revision=source,
        gpuq_owner=_gpuq_owner(),
        inputs_file_sha256=sha256(args.inputs),
        trace_sha256=sha256(args.trace),
        inputs_sha256=inputs["inputs_sha256"],
        seed=trace["seed"],
        host_performance_context_before=gate.host_performance_context(),
        waves=[],
    )
    gate.save(args.output, result)

    engine = server = server_thread = None
    handle = None
    try:
        loaded = time.monotonic()
        engine = serving.ServingEngine(
            args.model,
            adapter_factory=Adapter,
            max_lanes=6,
            max_inflight=20,
            mtp=args.mtp,
            prompt_lookup=False,
            # Explicit research permission for an unqualified candidate.  The
            # receipt remains qualified=false and no default is changed.
            qualification_mode=True,
            prefill_step=2048,
            max_context=8192,
            cache_bytes=1 << 29,
            batch_cohort_timeout_ms=5000,
            execution_policy={
                family["policy_key"]: {
                    "enabled": True,
                    "minimum_padding_rows": 1,
                    "minimum_padding_fraction": 0.0,
                }
            },
        )
        deadline = time.monotonic() + 45
        while not engine.ready.is_set() and not engine.error and time.monotonic() < deadline:
            engine.ready.wait(0.05)
        if engine.error or not engine.ready.is_set():
            raise RuntimeError("HTTP engine load failed: " + str(engine.error))
        result["load_seconds"] = time.monotonic() - loaded
        handle = getattr(engine.adapter, family["adapter_field"])
        if handle is None or handle["schema"] != family["schema"]:
            raise RuntimeError("adapter did not install the selected varlen candidate")

        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        server.daemon_threads = True
        server.block_on_close = False
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base = "http://127.0.0.1:" + str(server.server_port)

        def bodies(rows, cohort_id):
            return tuple(
                validate_request(
                    gate.request_body(
                        row,
                        model=args.model.name,
                        native=False,
                        inputs_sha256=inputs["inputs_sha256"],
                        cap=4,
                        cohort_id=cohort_id,
                        declared_cohort=True,
                    )
                )
                for row in rows
            )

        def run_arm(name, items, rows, cohort_id):
            campaign.wait_for_scheduler_snapshot(engine, timeout=0.1)
            object.__setattr__(
                engine.adapter.model,
                family["model_field"],
                handle if name == "varlen" else None,
            )
            before = dict(handle["counters"])
            responses, timing = campaign.run_wave(
                base,
                bodies(rows, cohort_id),
                [item["arrival_offset_ms"] for item in items],
            )
            gate.wait_idle(engine, timeout=10)
            mx.synchronize()
            after = dict(handle["counters"])
            delta = counter_delta(before, after)
            if name == "ordinary" and delta:
                raise RuntimeError("ordinary control engaged live-row compaction")
            if name == "varlen" and (
                delta.get(f"{family['counter_prefix']}_compaction_calls", 0) < 1
                or delta.get(f"{family['counter_prefix']}_compaction_calls")
                != delta.get(f"{family['counter_prefix']}_scatter_calls")
                or delta.get(f"{family['counter_prefix']}_padding_rows_skipped", 0)
                < 1
            ):
                raise RuntimeError("varlen arm emitted no complete mechanism proof")
            summaries = tuple(
                campaign.response_summary(row, response, native=False)
                for row, response in zip(rows, responses)
            )
            return {"timing": timing, "summary": summaries, "counters": delta}

        # Compile both graph geometries before measured work.  The warmup uses
        # frozen traffic but is not included in any performance aggregate.
        warm_items, warm_rows = waves[0]
        warmup = {}
        for name in ("ordinary", "varlen"):
            warmup[name] = run_arm(
                name, warm_items, warm_rows, f"{args.family}-warm-{name}"
            )
        if any(
            left[key] != right[key]
            for left, right in zip(
                warmup["ordinary"]["summary"], warmup["varlen"]["summary"]
            )
            for key in ("text", "finish_reason", "usage", "output_token_ids")
        ):
            raise RuntimeError("warmup HTTP parity differs")
        result["warmup"] = warmup
        handle["counters"].clear()
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()

        for index, (items, rows) in enumerate(waves[: args.waves]):
            ordinary_first = index % 2 == 0
            if args.reverse_order:
                ordinary_first = not ordinary_first
            order = (
                ("ordinary", "varlen")
                if ordinary_first
                else ("varlen", "ordinary")
            )
            wave = {
                "wave": index,
                "arm_order": list(order),
                "requests": items,
                "prompt_tokens": [row["prompt_tokens"] for row in rows],
                "arms": {},
            }
            for name in order:
                wave["arms"][name] = run_arm(
                    name,
                    items,
                    rows,
                    f"{args.family}-{index}-{name}",
                )
            ordinary = wave["arms"]["ordinary"]
            varlen = wave["arms"]["varlen"]
            if any(
                left[key] != right[key]
                for left, right in zip(ordinary["summary"], varlen["summary"])
                for key in ("text", "finish_reason", "usage", "output_token_ids")
            ):
                raise RuntimeError("measured HTTP parity differs")
            ordinary_wall = ordinary["timing"]["wave_wall_seconds"]
            varlen_wall = varlen["timing"]["wave_wall_seconds"]
            wave["ordinary_over_varlen_wave_wall"] = ordinary_wall / varlen_wall
            wave["exact_http_parity"] = True
            result["waves"].append(wave)
            gate.save(args.output, result)

        ordinary_total = sum(
            wave["arms"]["ordinary"]["timing"]["wave_wall_seconds"]
            for wave in result["waves"]
        )
        varlen_total = sum(
            wave["arms"]["varlen"]["timing"]["wave_wall_seconds"]
            for wave in result["waves"]
        )
        mechanism = dict(handle["counters"])
        result.update(
            status="passed",
            exact_http_parity=True,
            completed_paired_waves=len(result["waves"]),
            ordinary_wave_wall_seconds_sum=ordinary_total,
            varlen_wave_wall_seconds_sum=varlen_total,
            aggregate_ordinary_over_varlen=ordinary_total / varlen_total,
            varlen_wall_time_change_fraction=varlen_total / ordinary_total - 1.0,
            mechanism=mechanism,
            mechanism_observed_used=mechanism.get(
                f"{family['counter_prefix']}_compaction_calls", 0
            )
            > 0,
            adapter_status=engine.adapter.diagnostics().get(family["policy_key"]),
            mlx_peak_bytes=int(mx.get_peak_memory()),
        )
    finally:
        if engine is not None and handle is not None:
            object.__setattr__(engine.adapter.model, family["model_field"], handle)
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(5)
        if engine is not None:
            engine.close()
        result["cleanup"] = {
            "server_stopped": server_thread is None or not server_thread.is_alive(),
            "engine_closed": engine is None or not engine.thread.is_alive(),
        }
        result["host_performance_context_after"] = gate.host_performance_context()


def supervise(args, result):
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    process = subprocess.Popen(command, start_new_session=True)
    began = time.monotonic()
    peak = 0
    max_rss = MAX_RSS_BY_FAMILY[args.family]
    try:
        while process.poll() is None:
            if time.monotonic() - began > MAX_SECONDS:
                raise TimeoutError("varlen MLP-family HTTP gate exceeded 420 seconds")
            try:
                rss = int(
                    subprocess.check_output(
                        ["ps", "-o", "rss=", "-p", str(process.pid)], text=True
                    ).strip()
                    or "0"
                ) * 1024
            except (subprocess.CalledProcessError, ValueError):
                rss = 0
            peak = max(peak, rss)
            if rss > max_rss:
                raise MemoryError(
                    "varlen MLP-family HTTP gate exceeded its "
                    f"{max_rss / 2**30:.0f} GiB RSS cap"
                )
            time.sleep(0.2)
        if args.output.is_file():
            result = json.loads(args.output.read_text())
        result.update(
            supervisor_peak_rss_bytes=peak,
            supervisor_seconds=time.monotonic() - began,
        )
        gate.save(args.output, result)
        return process.returncode
    except BaseException as error:  # noqa: BLE001 - supervisor must kill worker tree
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        if args.output.is_file():
            result = json.loads(args.output.read_text())
        result.update(
            status="failed",
            qualified=False,
            worker_killed=True,
            error=f"{type(error).__name__}: {error}",
            supervisor_peak_rss_bytes=peak,
            supervisor_seconds=time.monotonic() - began,
        )
        gate.save(args.output, result)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILIES), default="qwen38-dense")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--waves", type=int, default=6)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.model is None:
        args.model = QWEN4_MODEL if args.family == "qwen4-sparse" else gate.MODEL
    if not 1 <= args.waves <= 6:
        parser.error("waves must be between 1 and 6")
    result = {
        "schema": FAMILIES[args.family]["gate_schema"],
        "status": "planned",
        "implemented": True,
        "qualified": False,
        "selected_by_default": False,
        "gpu_executed": False,
        "waves_requested": args.waves,
        "reverse_order": args.reverse_order,
        "family": args.family,
        "mtp": args.mtp,
    }
    if not args.execute:
        gate.save(args.output, result)
        return 0
    if not args.worker:
        return supervise(args, result)
    code = 0
    try:
        execute(args, result)
    except BaseException as error:  # noqa: BLE001 - persist complete failure evidence
        result.update(
            status="failed",
            qualified=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        code = 1
    gate.save(args.output, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
