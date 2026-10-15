#!/usr/bin/env python3
"""Bounded FOLD-only diagnostic in fresh processes; never changes admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from probe_fold_lifecycle import atomic_json, thermal

PHASES = ("reference", "cold", "warm", "seed", "continue", "rollback")


def array_record(value):
    """Bounded host digest of exact stored bits, with explicit shape and dtype."""
    import mlx.core as mx
    import numpy as np

    mx.eval(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(
            np.asarray(value.view(mx.uint8)).tobytes()
        ).hexdigest(),
    }


def state_record(value):
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return array_record(value)
    if isinstance(value, (list, tuple)):
        return [state_record(item) for item in value]
    if isinstance(value, dict):
        return {str(key): state_record(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise TypeError(f"unsupported state leaf {type(value).__name__}")


def cache_record(cache):
    records = []
    for entry in cache:
        offset = getattr(entry, "offset", None)
        state = entry.state
        # Descriptor rollback may preserve a grown backing array. Bytes beyond
        # the logical boundary are neither committed state nor a parity oracle.
        if (
            isinstance(offset, int)
            and isinstance(state, (tuple, list))
            and all(getattr(value, "ndim", None) == 4 for value in state)
        ):
            state = [value[..., :offset, :] for value in state]
        records.append(
            {
                "class": type(entry).__name__,
                "offset": offset,
                "metadata": state_record(getattr(entry, "meta_state", None)),
                "state": state_record(state),
            }
        )
    return records


def terminal_record(response):
    sidecar = response.cache_sidecar
    draft, tail = sidecar.state
    # KV physical spare capacity is not committed logical state.
    planes = [
        {
            "offset": int(entry.offset),
            "planes": [
                array_record(value[..., : entry.offset, :]) for value in entry.state
            ],
        }
        for entry in draft
    ]
    return {
        "target": cache_record(response.prompt_cache),
        "draft": planes,
        "tail": state_record(tail),
        "rng": state_record(sidecar.rng_key),
        "draws": sidecar.rng_draws,
        "covered": sidecar.covered_tokens,
        "binding": sidecar.binding,
        "history": response.all_tokens,
        "final_token": int(response.token),
    }


def checkpoint_record(cache, sidecar):
    return {
        "target": cache_record(cache),
        "draft": cache_record(sidecar.state[0]),
        "tail": state_record(sidecar.state[1]),
        "rng": state_record(sidecar.rng_key),
        "draws": sidecar.rng_draws,
        "covered": sidecar.covered_tokens,
        "binding": sidecar.binding,
    }


def phase(args, ownership):
    # No MLX import precedes trusted lease and paired ancestor-lock validation.
    import mlx.core as mx

    from mlx2.server import BoundedHTTPServer, handler_for
    from mlx2.serving import ServingEngine

    mx.set_default_device(mx.gpu)
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir / args.phase
    output.mkdir()
    policy = json.loads(args.policy.read_text())
    report = {
        "schema": "mlx2.fold-phased-diagnostic.v1",
        "phase": args.phase,
        "ownership": ownership,
        "policy": policy,
        "passed": False,
        "qualified": False,
        "performance_qualified": False,
        "mlx_version": mx.__version__,
        "device": mx.device_info(),
        "source_sha256": {},
        "thermal": [thermal(root)],
        "memory": [],
    }
    for name in (
        "src/mlx2/runtime/drafters/fold.py",
        "src/mlx2/runtime/drafters/dflash2.py",
        "src/mlx2/runtime/external_speculative.py",
        "src/mlx2/runtime/drafter_portfolio.py",
        "src/mlx2/adapters/external_draft_policy.py",
        "src/mlx2/runtime/apc_v2.py",
        "src/mlx2/serving.py",
        "scripts/probe_fold_phases.py",
    ):
        report["source_sha256"][name] = hashlib.sha256(
            (root / name).read_bytes()
        ).hexdigest()
    engine = server = None
    monitor_stop = threading.Event()
    monitor = None
    try:
        if report["thermal"][0]["thermal_state"] != 0:
            raise RuntimeError("phase must start at nominal thermal state")
        persistent = args.phase in ("cold", "warm")
        directory = str(args.output_dir / "http-cache") if persistent else None
        engine = ServingEngine(
            str(args.model),
            execution_policy=policy,
            mtp=False,
            max_lanes=1,
            max_inflight=1,
            max_context=args.context + 64,
            prefill_step=args.prefill_step,
            cache_bytes=512 << 20,
            cache_dir=directory,
            apc_persist_dir=directory,
            apc_persist_on_shutdown=persistent,
            apc_persist_shutdown_seconds=30,
            qualification_mode=True,
            lane_matmul="off",
            thinking_auto_calibration=False,
        )
        deadline = time.monotonic() + args.timeout
        while not engine.ready.wait(0.1):
            if engine.error or time.monotonic() >= deadline:
                raise RuntimeError(engine.error or "adapter startup deadline")
        if engine.error:
            raise RuntimeError(engine.error)
        # The adapter must pin its import-time model environment first.
        from mlx2.generation_trace import read_token_trace
        from mlx2.memory import host_available_bytes
        from mlx2.runtime.apc_v2 import APCKey, APCv2
        from mlx2.runtime.drafters.fold import FOLD_NUMERICS_VERSION, status
        from mlx2.runtime.generate import BatchGenerator
        from mlx2.runtime.os_memory import physical_footprint_bytes
        from mlx2.runtime.sample_utils import LaneRNG

        adapter = engine.adapter
        report.update(
            artifact_identity=json.loads(json.dumps(adapter.identity)),
            numerics=FOLD_NUMERICS_VERSION,
            runtime=engine.status()["runtime"],
            native_window=args.native_window,
        )
        if getattr(adapter.draft_model, "composition_policy", None) is not None:
            raise RuntimeError(
                "FOLD diagnostic requires an unwrapped, sole draft provider"
            )
        reference_path = args.output_dir / "reference" / "receipt.json"
        reference = (
            json.loads(reference_path.read_text())
            if args.phase != "reference"
            else None
        )
        if reference is not None:
            if (
                reference["artifact_identity"] != report["artifact_identity"]
                or reference["source_sha256"] != report["source_sha256"]
                or reference["policy"] != policy
                or reference["mlx_version"] != mx.__version__
            ):
                raise RuntimeError(
                    "phase artifact, policy, runtime or source identity changed"
                )
            prompt, text = reference["prompt_tokens"], reference["prompt_text"]
        else:
            text = "\n".join(
                f"Record {i}: gauge {i % 41}; seal intact; valve closed."
                for i in range(400)
            )
            tokens = adapter.prompt_tokens({"prompt": text})[: args.context]
            text = adapter.tokenizer.decode(tokens)
            prompt = adapter.prompt_tokens({"prompt": text})
        if not args.native_window < len(prompt) <= args.context + 16:
            raise RuntimeError(
                "round-tripped prompt is outside the bounded far-memory context"
            )
        report.update(
            prompt_text=text, prompt_tokens=prompt, context_tokens=len(prompt)
        )

        def memory(label, scheduler=None):
            raw = int(engine.admission_headroom_fn()())
            reserve = int(engine.hard_reserve_gib * (1 << 30))
            record = {
                "label": label,
                "active_bytes": int(mx.get_active_memory()),
                "cached_bytes": int(mx.get_cache_memory()),
                "footprint_bytes": physical_footprint_bytes(),
                "host_available_bytes": host_available_bytes(),
                "raw_headroom_bytes": raw,
                "hard_reserve_bytes": reserve,
                "usable_headroom_bytes": max(0, raw - reserve),
                "recommended_bytes": mx.device_info()[
                    "max_recommended_working_set_size"
                ],
            }
            if scheduler is not None:
                record.update(
                    reservation_bytes=scheduler.get("reservation_bytes", 0),
                    fold_extra_reservation_bytes=scheduler.get(
                        "draft_extra_reservation_bytes", 0
                    ),
                )
                record["shortfall_bytes"] = max(
                    0, record["reservation_bytes"] - record["usable_headroom_bytes"]
                )
                record["minimum_free_before_reserve_bytes"] = (
                    record["reservation_bytes"] + reserve
                )
            if len(report["memory"]) < 100:
                report["memory"].append(record)
            return record

        memory("adapter_ready")

        def watch():
            while not monitor_stop.wait(10):
                try:
                    snapshot = engine.status()
                    memory("live", snapshot.get("scheduler", {}))
                    report["live_status"] = snapshot
                    report["thermal"].append(thermal(root))
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    subprocess.SubprocessError,
                ) as error:
                    report["monitor_error"] = str(error)

        monitor = threading.Thread(target=watch, daemon=True)
        monitor.start()

        def drain(batch, uid):
            tokens = []
            for _ in range(128):
                prompts, responses = batch.next()
                for response in responses:
                    if response.uid == uid:
                        tokens.append(int(response.token))
                        if response.finish_reason:
                            return tokens, response
                if (
                    not prompts
                    and not responses
                    and getattr(batch, "scheduler_stats", {}).get("memory_deferred")
                ):
                    memory("direct_capacity_refusal", batch.scheduler_stats)
                    raise MemoryError(
                        "direct diagnostic cannot progress under unchanged serving admission"
                    )
            raise RuntimeError("bounded executor did not finish")

        if args.phase == "reference":

            def ordinary(_):
                batch = BatchGenerator(
                    adapter.model,
                    completion_batch_size=1,
                    prefill_batch_size=1,
                    prefill_step_size=args.prefill_step,
                    stop_tokens=[(int(t),) for t in adapter.tokenizer.eos_token_ids],
                )
                try:
                    (uid,) = batch.insert([prompt], max_tokens=[args.tokens])
                    return drain(batch, uid)[0]
                finally:
                    batch.close()

            report["tokens"] = engine._exclusive_adapter_operation(
                "fold_reference", ordinary, timeout=args.timeout
            )
        elif args.phase in ("cold", "warm"):
            before = status(adapter.draft_model)
            server = BoundedHTTPServer(("127.0.0.1", 0), handler_for(engine))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            body = {
                "model": args.model.name,
                "prompt": text,
                "max_tokens": args.tokens,
                "temperature": 0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "stream": False,
                "return_token_trace": True,
            }
            request = Request(
                f"http://127.0.0.1:{server.server_address[1]}/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(request, timeout=args.timeout) as response:
                    result = json.load(response)
            except HTTPError as error:
                report["http_error_body"] = error.read(65536).decode(errors="replace")
                raise
            route = result["mlx2"]
            report["route"] = route
            report["tokens"] = read_token_trace(route)["token_ids"]
            after = status(adapter.draft_model)
            report["far_memory_calls"] = (
                after["far_memory_calls"] - before["far_memory_calls"]
            )
            spec = route.get("speculation") or {}
            checks = {
                "ordinary_tokens_exact": report["tokens"] == reference["tokens"],
                "external_executed": route.get("route") == "external_draft"
                and spec.get("external_rounds", 0) > 0,
                "sole_fold_provider": not spec.get("proposal_composition"),
                "far_memory_executed": report["far_memory_calls"] > 0,
                "cache_boundary": (route.get("cached_tokens", 0) > 0)
                if args.phase == "warm"
                else route.get("cached_tokens", 0) == 0,
            }
            report["checks"] = checks
            if not all(checks.values()):
                raise RuntimeError("HTTP FOLD execution/parity/cache contract failed")
        else:

            def direct(_):
                key = APCKey(
                    "fold-phased-diagnostic",
                    revision=adapter.identity["fingerprint"],
                    cache_layout_fingerprint=adapter.layout,
                )
                directory = str(args.output_dir / "paired-cache")
                apc = APCv2(
                    layout_name=adapter.layout,
                    idle_disk_seconds=1,
                    idle_disk_dir=directory,
                    persist_dir=directory,
                    persist_identity=key,
                )
                batch = hit = None
                try:

                    def evict_checkpoint():
                        evicted = apc.evict_oldest_unleased()
                        if evicted:
                            engine._permit_allocator_reclaim_after_eviction()
                        return evicted

                    # Match serving.py's controller reserve; never grant the
                    # direct executor the larger raw admission callback.
                    batch = adapter.create_external_batch(
                        completion_batch_size=1,
                        prefill_step_size=args.prefill_step,
                        stop_tokens=(),
                        memory_headroom=lambda: max(
                            0,
                            engine.admission_headroom_fn()()
                            - engine.hard_reserve_gib * (1 << 30),
                        ),
                        reclaim_memory=lambda: (
                            engine._clear_allocator_cache_before_reject(
                                synchronize=True, settle=True
                            )
                        ),
                        evict_checkpoint=evict_checkpoint,
                    )
                    if args.phase == "seed":
                        (uid,) = batch.insert(
                            [prompt],
                            max_tokens=[3],
                            sampling_configs=[{"sampling_temp": 0.8}],
                            lane_rngs=[LaneRNG(53)],
                        )
                    else:
                        saved_seed = json.loads(
                            (args.output_dir / "seed" / "receipt.json").read_text()
                        )
                        seed = saved_seed["seed_boundary"]
                        hit = apc.lookup(key, seed["history"] + [seed["anchor"]])
                        if not hit.hit:
                            raise RuntimeError(
                                "paired persistent checkpoint was not restored"
                            )
                        hit.sidecar.validate(
                            adapter.identity["fingerprint"], len(seed["history"])
                        )
                        report["restored_state"] = checkpoint_record(
                            hit.cache, hit.sidecar
                        )
                        report["restore_exact"] = (
                            report["restored_state"] == saved_seed["seed_state"]
                        )
                        if not report["restore_exact"]:
                            raise RuntimeError(
                                "durable restore changed committed target/draft/tail/RNG state"
                            )
                        (uid,) = batch.insert(
                            [[seed["anchor"]]],
                            max_tokens=[4],
                            caches=[hit.cache],
                            all_tokens=[seed["history"]],
                            cache_states=[hit.sidecar],
                            resume_rng=True,
                            lane_rngs=[LaneRNG(999)],
                            sampling_configs=[{"sampling_temp": 0.8}],
                        )
                        if args.phase == "rollback":
                            lane = batch.lanes[uid]
                            batch._prefill(lane)
                            count = min(
                                batch.num_draft, lane.maximum - lane.generated - 1
                            )
                            if not batch._fit_cohort([lane], count + 1):
                                memory(
                                    "rollback_capacity_refusal", batch.scheduler_stats
                                )
                                raise MemoryError(
                                    "rollback diagnostic did not fit conservative admission"
                                )
                            before = {
                                "draft": cache_record(lane.draft_cache),
                                "target": cache_record(lane.cache),
                                "tail": state_record(lane.tail),
                                "rng": state_record(lane.rng.snapshot()),
                                "history": list(lane.history),
                            }
                            original = batch._propose

                            def injected(*a, **kw):
                                original(*a, **kw)
                                raise RuntimeError("diagnostic post-proposal fault")

                            batch._propose = injected
                            try:
                                batch._round([lane])
                            except RuntimeError as error:
                                if str(error) != "diagnostic post-proposal fault":
                                    raise
                            else:
                                raise RuntimeError("rollback fault was not reached")
                            finally:
                                batch._propose = original
                            after = {
                                "draft": cache_record(lane.draft_cache),
                                "target": cache_record(lane.cache),
                                "tail": state_record(lane.tail),
                                "rng": state_record(lane.rng.snapshot()),
                                "history": list(lane.history),
                            }
                            report["rollback_exact"] = (
                                before == after and not lane.ready
                            )
                            if not report["rollback_exact"]:
                                raise RuntimeError(
                                    "post-proposal failure changed committed state"
                                )
                    tokens, end = drain(batch, uid)
                    report.update(
                        tokens=tokens,
                        scheduler=dict(batch.scheduler_stats),
                        fold=status(adapter.draft_model),
                    )
                    if args.phase == "seed":
                        report["seed_state"] = checkpoint_record(
                            end.prompt_cache, end.cache_sidecar
                        )
                        apc.store(
                            key,
                            end.all_tokens,
                            end.prompt_cache,
                            sidecar=end.cache_sidecar,
                        )
                        report["seed_boundary"] = {
                            "history": end.all_tokens,
                            "anchor": int(end.token),
                        }
                        report["park"] = apc.close(
                            persist_resident=True,
                            time_budget_seconds=30,
                            release_memory=False,
                        )
                    else:
                        report["terminal"] = terminal_record(end)
                        if args.phase == "rollback":
                            expected = json.loads(
                                (
                                    args.output_dir / "continue" / "receipt.json"
                                ).read_text()
                            )
                            report["retry_exact"] = (
                                tokens == expected["tokens"]
                                and report["terminal"] == expected["terminal"]
                            )
                            if not report["retry_exact"]:
                                raise RuntimeError(
                                    "rollback retry diverged from unchanged persisted continuation"
                                )
                finally:
                    if batch is not None:
                        batch.close()
                    if hit is not None and hasattr(hit.cache, "close"):
                        hit.cache.close()
                    apc.close(release_memory=False)

            engine._exclusive_adapter_operation(
                "fold_" + args.phase, direct, timeout=args.timeout
            )
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=35)
        if engine is not None:
            report["final_status"] = engine.status()
        if server is not None:
            server.shutdown()
            server.server_close()
        if engine is not None:
            engine.close()
        report["swap_after"] = subprocess.check_output(
            ["/usr/sbin/sysctl", "vm.swapusage"], text=True
        ).strip()
        atomic_json(output / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2064)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--phase", choices=("all",) + PHASES, default="all")
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    if (
        not isinstance(policy.get("fold"), dict)
        or policy.get("proposal_composition") is not False
    ):
        parser.error("explicit FOLD and proposal_composition=false are required")
    args.native_window = int(
        json.loads((Path(policy["draft_model"]) / "config.json").read_text())[
            "sliding_window"
        ]
    )
    if (
        not args.native_window < args.context <= args.native_window + 128
        or not 1 <= args.tokens <= 8
        or not 1 <= args.prefill_step <= 256
    ):
        parser.error("context/token/prefill bound exceeded")
    if (
        args.output_dir if args.phase == "all" else args.output_dir / args.phase
    ).exists():
        parser.error("refusing to overwrite evidence")
    from qualification_gpu_ownership import require_qualification_lease

    ownership = require_qualification_lease(
        task_id=args.cpg_task,
        cpg_owner_lock=args.cpg_owner_lock,
        generation=args.cpg_generation,
    )
    if args.phase != "all":
        return 0 if phase(args, ownership)["passed"] else 1
    args.output_dir.mkdir(parents=True)
    result = {
        "ownership": ownership,
        "phases": [],
        "passed": False,
        "qualified": False,
        "performance_qualified": False,
    }
    try:
        for name in PHASES:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--phase",
                name,
            ]
            child = subprocess.run(command, timeout=args.timeout + 90, check=False)
            result["phases"].append({"phase": name, "returncode": child.returncode})
            if child.returncode:
                return 1
        result["passed"] = True
        return 0
    finally:
        atomic_json(args.output_dir / "summary.json", result)


if __name__ == "__main__":
    raise SystemExit(main())
