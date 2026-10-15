#!/usr/bin/env python3
"""Owned, bounded real-artifact FOLD lifecycle diagnostic; no qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def thermal(root):
    result = subprocess.run(
        ["/usr/bin/swift", str(root / "scripts/thermal_probe.swift")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def drain(batch, uid, *, bound=512):
    tokens, final = [], None
    for _ in range(bound):
        _, responses = batch.next()
        for response in responses:
            if response.uid == uid:
                tokens.append(int(response.token))
                if response.finish_reason:
                    final = response
        if final is not None:
            return tokens, final
    raise RuntimeError("bounded FOLD diagnostic executor stalled")


def run(args, ownership):
    # main validates the trusted lease and both live ancestor locks first.
    import copy

    import mlx.core as mx

    from mlx2.generation_trace import read_token_trace
    from mlx2.server import BoundedHTTPServer, handler_for
    from mlx2.serving import ServingEngine

    mx.set_default_device(mx.gpu)
    root = Path(__file__).resolve().parents[1]
    policy = json.loads(args.policy.read_text())
    if not isinstance(policy.get("fold"), dict):
        raise TypeError("an explicit FOLD policy is required")
    if args.context <= 2048:
        raise ValueError("context must exceed this artifact's native sliding window")
    receipt = {
        "schema": "mlx2.fold-lifecycle-diagnostic.v1",
        "ownership": ownership,
        "model": str(args.model),
        "policy": policy,
        "numerics": None,
        "mlx_version": mx.__version__,
        "device": mx.device_info()["device_name"],
        "implemented": True,
        "qualified": False,
        "performance_qualified": False,
        "scope": "one artifact, ordinary reference, HTTP cold/warm, direct paired persistence and rollback",
        "source_sha256": {},
        "thermal": [],
        "passed": False,
    }
    for relative in (
        "src/mlx2/runtime/drafters/fold.py",
        "src/mlx2/runtime/drafters/dflash2.py",
        "src/mlx2/runtime/external_speculative.py",
        "src/mlx2/adapters/external_draft_policy.py",
        "src/mlx2/runtime/apc_v2.py",
        "scripts/probe_fold_lifecycle.py",
    ):
        receipt["source_sha256"][relative] = hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest()
    first_thermal = thermal(root)
    receipt["thermal"].append(first_thermal)
    if int(first_thermal["thermal_state"]) != 0:
        raise RuntimeError("diagnostic must start at nominal thermal state")
    receipt["swap_before"] = subprocess.check_output(
        ["/usr/sbin/sysctl", "vm.swapusage"], text=True
    ).strip()
    engine = server = None
    watcher_stop = threading.Event()

    def watch():
        while not watcher_stop.wait(15):
            try:
                receipt["thermal"].append(thermal(root))
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                receipt["thermal"].append({"error": str(error)})

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        engine = ServingEngine(
            str(args.model),
            execution_policy=policy,
            mtp=False,
            max_lanes=1,
            max_inflight=1,
            max_context=args.context + 128,
            prefill_step=args.prefill_step,
            cache_bytes=1 << 30,
            cache_dir=str(args.output_dir / "http-cache"),
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

        from mlx2.runtime.apc_v2 import APCKey, APCv2
        from mlx2.runtime.drafters.fold import FOLD_NUMERICS_VERSION, status
        from mlx2.runtime.generate import BatchGenerator
        from mlx2.runtime.sample_utils import LaneRNG

        receipt["numerics"] = FOLD_NUMERICS_VERSION
        receipt["phase"] = "adapter_ready"
        atomic_json(args.output_dir / "receipt.json", receipt)

        def ordinary(adapter):
            text = "A ledger records each valve inspection in order. The valve is closed, the seal is intact, and the gauge reads zero. "
            seed = adapter.tokenizer.encode(text, add_special_tokens=False)
            sliced = (seed * (args.context // len(seed) + 1))[: args.context]
            prompt_text = adapter.tokenizer.decode(sliced)
            prompt = adapter.prompt_tokens({"prompt": prompt_text})
            batch = BatchGenerator(
                adapter.model,
                completion_batch_size=1,
                prefill_batch_size=1,
                prefill_step_size=args.prefill_step,
                stop_tokens=[
                    (int(token),) for token in adapter.tokenizer.eos_token_ids
                ],
            )
            try:
                (uid,) = batch.insert([prompt], max_tokens=[args.tokens])
                tokens, _ = drain(batch, uid)
                return prompt_text, prompt, tokens, dict(adapter.identity)
            finally:
                batch.close()

        prompt_text, prompt, reference, identity = engine._exclusive_adapter_operation(
            "fold_reference", ordinary, timeout=args.timeout
        )
        receipt.update(
            artifact_identity=identity,
            context_tokens=len(prompt),
            ordinary_reference_tokens=reference,
            phase="ordinary_reference_completed",
        )
        atomic_json(args.output_dir / "receipt.json", receipt)

        server = BoundedHTTPServer(("127.0.0.1", 0), handler_for(engine))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = {
            "model": args.model.name,
            "prompt": prompt_text,
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
        cells = []
        for label in ("cold", "warm"):
            before_fold = engine._exclusive_adapter_operation(
                "fold_counter",
                lambda adapter: status(adapter.draft_model),
                timeout=args.timeout,
            )
            request = Request(
                base + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(request, timeout=args.timeout) as response:
                    result = json.load(response)
            except HTTPError as error:
                receipt["http_error_body"] = error.read(65536).decode(errors="replace")
                raise
            after_fold = engine._exclusive_adapter_operation(
                "fold_counter",
                lambda adapter: status(adapter.draft_model),
                timeout=args.timeout,
            )
            route = result["mlx2"]
            trace = read_token_trace(route)
            cell = {
                "label": label,
                "receipt": route,
                "tokens": trace["token_ids"],
                "reference_equal": trace["token_ids"] == reference,
                "far_memory_calls": after_fold["far_memory_calls"]
                - before_fold["far_memory_calls"],
                "external_route_used": route.get("route") == "external_draft"
                and (route.get("speculation") or {}).get("external_rounds", 0) > 0,
            }
            cells.append(cell)
            receipt["http"] = cells
            receipt["phase"] = "http_" + label + "_completed"
            atomic_json(args.output_dir / "receipt.json", receipt)
        if not all(
            cell["external_route_used"] and cell["far_memory_calls"] > 0
            for cell in cells
        ):
            raise RuntimeError(
                "HTTP cells did not observe the selected external FOLD route"
            )
        if not all(cell["reference_equal"] for cell in cells):
            raise RuntimeError(
                "FOLD cold/warm output differs from ordinary reference; localize first divergence before logits"
            )
        if not cells[1]["receipt"].get("cached_tokens", 0):
            raise RuntimeError("warm HTTP request did not reuse paired APC state")

        def lifecycle(adapter):
            binding = adapter.identity["fingerprint"]

            def make():
                return adapter.create_external_batch(
                    completion_batch_size=1,
                    prefill_step_size=args.prefill_step,
                    stop_tokens=(),
                    memory_headroom=engine.admission_headroom_fn(),
                )

            source = make()
            (uid,) = source.insert(
                [prompt],
                max_tokens=[3],
                sampling_configs=[{"sampling_temp": 0.8}],
                lane_rngs=[LaneRNG(53)],
            )
            try:
                _, end = drain(source, uid)
                key = APCKey(
                    "fold-diagnostic",
                    revision=binding,
                    cache_layout_fingerprint=adapter.layout,
                )
                options = {
                    "layout_name": adapter.layout,
                    "idle_disk_seconds": 1,
                    "idle_disk_dir": str(args.output_dir / "paired-cache"),
                    "persist_dir": str(args.output_dir / "paired-cache"),
                    "persist_identity": key,
                }
                apc = APCv2(**options)
                apc.store(
                    key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar
                )
                apc.close(
                    persist_resident=True, time_budget_seconds=30, release_memory=False
                )
                apc = APCv2(**options)
                hit = apc.lookup(key, end.all_tokens + [end.token])
                if not hit.hit:
                    raise RuntimeError(
                        "fresh manager did not restore paired FOLD state"
                    )
                hit.sidecar.validate(binding, len(end.all_tokens))

                def continuation(cache, state, fail=False):
                    batch = make()
                    (uid,) = batch.insert(
                        [[end.token]],
                        max_tokens=[4],
                        caches=[cache],
                        all_tokens=[end.all_tokens],
                        cache_states=[state],
                        resume_rng=True,
                        lane_rngs=[LaneRNG(999)],
                        sampling_configs=[{"sampling_temp": 0.8}],
                    )
                    try:
                        lane = batch.lanes[uid]
                        batch._prefill(lane)
                        if fail:
                            rng, history = lane.rng.snapshot(), list(lane.history)
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
                                raise RuntimeError("rollback fault did not execute")
                            finally:
                                batch._propose = original
                            if (
                                lane.rng.snapshot() != rng
                                or lane.history != history
                                or lane.ready
                            ):
                                raise RuntimeError(
                                    "failed FOLD round changed committed control"
                                )
                        tokens, final = drain(batch, uid)
                        return tokens, final.cache_sidecar, dict(batch.scheduler_stats)
                    finally:
                        batch.close()

                try:
                    expected, expected_state, _ = continuation(
                        copy.deepcopy(end.prompt_cache), end.cache_sidecar
                    )
                    actual, actual_state, stats = continuation(
                        hit.cache, hit.sidecar, fail=True
                    )
                    expected_caches, expected_tail = expected_state.state
                    actual_caches, actual_tail = actual_state.state
                    checks = {
                        "boundary_exact": actual_state.covered_tokens
                        == expected_state.covered_tokens
                        and actual_state.binding == expected_state.binding,
                        "cache_geometry_exact": len(actual_caches)
                        == len(expected_caches)
                        == len(adapter.draft_model.layers)
                        and all(
                            x.offset == y.offset and len(x.state) == len(y.state) == 3
                            for x, y in zip(actual_caches, expected_caches)
                        ),
                        "pending_tail_exact": actual_tail.shape == expected_tail.shape
                        and actual_tail.dtype == expected_tail.dtype
                        and mx.array_equal(actual_tail, expected_tail).item(),
                        "tokens_exact": actual == expected,
                        "rng_exact": mx.array_equal(
                            actual_state.rng_key, expected_state.rng_key
                        ).item(),
                        "draws_exact": actual_state.rng_draws
                        == expected_state.rng_draws,
                        "planes_exact": all(
                            a.dtype == b.dtype
                            and a[..., : x.offset, :].shape
                            == b[..., : y.offset, :].shape
                            and mx.array_equal(
                                a[..., : x.offset, :], b[..., : y.offset, :]
                            ).item()
                            for x, y in zip(
                                actual_state.state[0], expected_state.state[0]
                            )
                            for a, b in zip(x.state, y.state)
                        ),
                    }
                    return {
                        "checks": checks,
                        "tokens": actual,
                        "scheduler": stats,
                        "fold": status(adapter.draft_model),
                        "passed": all(checks.values()),
                    }
                finally:
                    if hasattr(hit.cache, "close"):
                        hit.cache.close()
                    apc.close(release_memory=False)
            finally:
                source.close()

        receipt["paired_lifecycle"] = engine._exclusive_adapter_operation(
            "fold_lifecycle", lifecycle, timeout=args.timeout
        )
        receipt["peak_device_bytes"] = int(mx.get_peak_memory())
        receipt["phase"] = "paired_lifecycle_completed"
        receipt["passed"] = (
            receipt["paired_lifecycle"]["passed"]
            and receipt["paired_lifecycle"]["fold"]["observed_used"]
        )
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if engine is not None:
            engine.close()
        watcher_stop.set()
        watcher.join(timeout=35)
        receipt["swap_after"] = subprocess.check_output(
            ["/usr/sbin/sysctl", "vm.swapusage"], text=True
        ).strip()
        atomic_json(args.output_dir / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2304)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    args = parser.parse_args()
    if (
        not 2048 < args.context <= 8192
        or not 1 <= args.tokens <= 16
        or not 1 <= args.prefill_step <= 512
    ):
        parser.error("diagnostic exceeds bounded context/token/prefill limits")
    if args.output_dir.exists():
        parser.error("refusing to overwrite evidence directory")
    from qualification_gpu_ownership import require_qualification_lease

    ownership = require_qualification_lease(
        task_id=args.cpg_task,
        cpg_owner_lock=args.cpg_owner_lock,
        generation=args.cpg_generation,
    )
    args.output_dir.mkdir(parents=True)
    result = run(args, ownership)
    print(json.dumps({"passed": result["passed"], "output": str(args.output_dir)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
