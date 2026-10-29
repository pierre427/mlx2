#!/usr/bin/env python3
"""Owned single-token intervention diagnostic; explicit eval breaks fusion.

No throughput benchmark: nested exclusive costs include Python scheduling and
forced evaluation. The caller owns CPG, paired locks and thermal admission.
"""

from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from probe_deepseek_v4_packed import atomic, source_receipt, verify_ownership
from probe_deepseek_v4_ordinary import (
    load_request,
    memory,
    arrays_equal,
    arrays_numpy,
    first_logit_comparison,
)


def leaves(value, mx):
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, dict):
        return [a for v in value.values() for a in leaves(v, mx)]
    if isinstance(value, (tuple, list)):
        return [a for v in value for a in leaves(v, mx)]
    # Cache mutation must be included even when its output is not returned.
    if type(value).__module__.startswith("mlx2.runtime.models.deepseek_v4") and hasattr(
        value, "state"
    ):
        return leaves(value.state, mx)
    return []


class BoundaryProfiler:
    def __init__(self, mx, clock=time.perf_counter):
        self.mx, self.clock = mx, clock
        self.events, self.stack, self.layer = [], [], None

    def call(self, category, name, fn, args, kwargs, *, layer=None):
        previous = self.layer
        if layer is not None:
            self.layer = layer
        start = self.clock()
        self.mx.eval(leaves((args, kwargs), self.mx))
        ready = self.clock()
        frame = {"child_total": 0.0}
        self.stack.append(frame)
        try:
            result = fn(*args, **kwargs)
            self.mx.eval(leaves((result, args, kwargs), self.mx))
            end = self.clock()
        finally:
            self.stack.pop()
            self.layer = previous
        total = end - start
        if self.stack:
            self.stack[-1]["child_total"] += total
        self.events.append(
            {
                "category": category,
                "name": name,
                "layer": layer if layer is not None else previous,
                "inclusive_seconds": total,
                "input_barrier_seconds": ready - start,
                "exclusive_seconds": max(0.0, total - frame["child_total"]),
            }
        )
        return result

    def summary(self):
        groups = {}
        layers = {}
        for event in self.events:
            item = groups.setdefault(
                event["category"],
                {
                    "calls": 0,
                    "inclusive_seconds": 0.0,
                    "exclusive_seconds": 0.0,
                    "input_barrier_seconds": 0.0,
                },
            )
            item["calls"] += 1
            for key in (
                "inclusive_seconds",
                "exclusive_seconds",
                "input_barrier_seconds",
            ):
                item[key] += event[key]
            if event["layer"] is not None:
                row = layers.setdefault(str(event["layer"]), {})
                row[event["category"]] = (
                    row.get(event["category"], 0.0) + event["exclusive_seconds"]
                )
        return {
            "categories": groups,
            "layers_exclusive_seconds": layers,
            "events": self.events,
            "exclusive_total_seconds": sum(e["exclusive_seconds"] for e in self.events),
            "scope": "inclusive times overlap; only exclusive times are additive. Input barriers include deferred upstream arithmetic; no automatic overhead subtraction.",
        }


@contextmanager
def instrument(model, profiler):
    """Temporary hooks preserve the actual block graph and restore on failure."""
    from mlx2.runtime.models import deepseek_v4 as module

    patches = []

    def patch(owner, name, replacement):
        original = getattr(owner, name)
        patches.append((owner, name, original))
        setattr(owner, name, replacement)
        return original

    def method(cls, name, category, label):
        original = getattr(cls, name)

        def wrapped(obj, *args, **kwargs):
            return profiler.call(
                category,
                label(obj),
                lambda *a, **k: original(obj, *a, **k),
                args,
                kwargs,
            )

        patch(cls, name, wrapped)

    try:
        indices = {id(block): i for i, block in enumerate(model.layers)}
        original = module.Block.__call__

        def block(obj, *args, **kwargs):
            return profiler.call(
                "block_residual",
                obj.prefix,
                lambda *a, **k: original(obj, *a, **k),
                args,
                kwargs,
                layer=indices[id(obj)],
            )

        patch(module.Block, "__call__", block)
        method(module.Attention, "__call__", "attention", lambda obj: obj.prefix)
        method(
            module.HyperConnection,
            "__call__",
            "hc",
            lambda obj: "head" if obj.head else "mix_reduce",
        )
        for name, category in [("hc_expand", "hc"), ("route_experts", "routing")]:
            original_fn = getattr(module, name)

            def wrapped(
                *args, _fn=original_fn, _name=name, _category=category, **kwargs
            ):
                return profiler.call(_category, _name, _fn, args, kwargs)

            patch(module, name, wrapped)
        cls = type(model.weights)
        method(cls, "moe", "routed_moe", lambda obj: "packed_experts")
        method(cls, "grouped", "dense", lambda obj: "grouped_projection")
        method(cls, "embedding", "embedding", lambda obj: "embedding")
        linear = cls.linear

        def project(obj, name):
            fn = linear(obj, name)
            category = "shared_expert" if "_shexp." in name else "dense"
            return lambda *args, **kwargs: profiler.call(
                category, name, fn, args, kwargs
            )

        patch(cls, "linear", project)
        yield
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def profile_frontier(model, cache, token, mx):
    snapshots = [state.snapshot() for state in cache]
    mx.eval([s.arrays for s in snapshots])
    before_meta = [state.meta_state for state in cache]
    inp = mx.array([[token]], dtype=mx.uint32)

    def restore():
        for state, saved in zip(cache, snapshots):
            state.restore(saved)
        mx.eval([s.state for s in cache])

    try:
        start = time.perf_counter()
        reference = model(inp, cache=cache, logits_to_keep=0)
        mx.eval(reference, [s.state for s in cache])
        reference_seconds = time.perf_counter() - start
        ref = np.array(reference[0, -1])
        ref_states = [arrays_numpy(s.state) for s in cache]
        ref_meta = [s.meta_state for s in cache]
        restore()
        # Already-materialized array barrier control measures API overhead only.
        start = time.perf_counter()
        for _ in range(100):
            mx.eval(inp)
        barrier_seconds = (time.perf_counter() - start) / 100
        profiler = BoundaryProfiler(mx)
        start = time.perf_counter()
        with instrument(model, profiler):
            actual = model(inp, cache=cache, logits_to_keep=0)
            mx.eval(actual, [s.state for s in cache])
        profiled_seconds = time.perf_counter() - start
        value = np.array(actual[0, -1])
        state_equal = arrays_equal(ref_states, [arrays_numpy(s.state) for s in cache])
        meta_equal = ref_meta == [s.meta_state for s in cache]
        comparison = first_logit_comparison(value, ref)
        comparison["bitwise_equal"] = bool(np.array_equal(value, ref))
        result = {
            "input_token": token,
            "ordinary_seconds": reference_seconds,
            "profiled_seconds": profiled_seconds,
            "materialized_single_array_eval_control_seconds": barrier_seconds,
            "logits": comparison,
            "state_arrays_bitwise_equal": state_equal,
            "state_metadata_equal": meta_equal,
            **profiler.summary(),
        }
        result["unattributed_wall_seconds"] = (
            profiled_seconds - result["exclusive_total_seconds"]
        )
        return result
    finally:
        restore()
        if before_meta != [s.meta_state for s in cache] or not arrays_equal(
            [arrays_numpy(s.arrays) for s in snapshots],
            [arrays_numpy(s.state) for s in cache],
        ):
            raise RuntimeError("profiling failed to restore complete frontier")


def validate_args(args):
    if not args.run_gpu or args.ownership_window is None:
        raise ValueError("explicit GPU flag and owned window required")
    if args.output.exists():
        raise ValueError("refusing existing evidence directory")
    if not 60 <= getattr(args, "timeout_seconds", 1800) <= 1800:
        raise ValueError("timeout must be60..1800seconds")
    if not 1 <= args.prefill_step <= 512 or not 80 <= args.memory_limit_gib <= 112:
        raise ValueError("bounded prefill step1..512 and memory80..112GiB required")
    from mlx2.adapters.deepseek_v4_gguf import split_execution_policy

    _, external = split_execution_policy(args.execution_policy)
    if external:
        raise ValueError("layer cost probe is ordinary target only")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ownership-window", type=Path)
    parser.add_argument("--owner-path", type=Path, action="append")
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--execution-policy", type=json.loads, default={})
    parser.add_argument("--prefill-step", type=int, default=128)
    parser.add_argument("--memory-limit-gib", type=float, default=108)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    validate_args(args)
    owner_paths = args.owner_path or [
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        Path("/tmp/gpu.lock/owner.json"),
    ]
    if len(owner_paths) != 2:
        raise ValueError("two paired locks required")
    deadline = time.monotonic() + args.timeout_seconds

    def guard():
        if time.monotonic() > deadline:
            raise TimeoutError("bounded layer profile deadline exceeded")
        return verify_ownership(args.ownership_window, owner_paths)

    ownership, source = guard(), source_receipt()
    from mlx2.adapters.deepseek_v4_gguf_artifact import inspect_artifact

    artifact = inspect_artifact(args.model_path)
    request, request_binding = load_request(args.request_json)
    args.output.mkdir(parents=True)
    receipt = {
        "schema": "mlx2.deepseek-v4-layer-cost.v1",
        "status": "running",
        "qualification": False,
        "performance_claim": False,
        "source": source,
        "checkpoint": args.model_path.name + "@" + artifact["identity"]["fingerprint"],
        "ownership": ownership,
        "request_binding": request_binding,
        "policy": args.execution_policy,
        "scope": "single-token evaluated-boundary intervention diagnostic, not throughput benchmark; forced eval changes scheduling/fusion",
    }
    atomic(args.output / "results.json", receipt)
    adapter = model = cache = logits = None
    try:
        import mlx.core as mx
        from mlx2.process_env import require_process_numerics

        require_process_numerics("DeepSeek layer cost diagnostic")
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(int(args.memory_limit_gib * (1 << 30)))
        mx.set_cache_limit(256 << 20)
        from mlx2.adapters.deepseek_v4_gguf import DeepSeekV4GGUFAdapter
        from deepseek_v4_probe_residency import wire_preloaded_weights

        adapter = DeepSeekV4GGUFAdapter(
            str(args.model_path), execution_policy=args.execution_policy
        )
        if adapter.identity != artifact["identity"]:
            raise RuntimeError("artifact changed at load")
        model = adapter.model
        wire_preloaded_weights(mx, receipt)
        receipt["phase"] = "loaded_and_wired"
        atomic(args.output / "results.json", receipt)
        guard()
        tokens = list(adapter.prompt_tokens(request))
        if not 1 <= len(tokens) <= 512:
            raise ValueError("prompt must have1..512tokens")
        receipt["prompt_tokens"] = tokens
        receipt["resolved_policy"] = adapter.kernel_policy
        cache = model.make_cache()
        from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows

        for offset in range(0, len(tokens) - 1, args.prefill_step):
            guard()
            part = tokens[offset : min(offset + args.prefill_step, len(tokens) - 1)]
            with prefill_checkpoint_rows([len(part)], width=len(part)):
                logits = model.prefill_forward(
                    mx.array([part], dtype=mx.uint32), cache=cache
                )
            mx.eval(logits, [s.state for s in cache])
            for state in cache:
                state.finalize()
        logits = model(
            mx.array([[tokens[-1]]], dtype=mx.uint32), cache=cache, logits_to_keep=0
        )
        mx.eval(logits, [s.state for s in cache])
        first = int(mx.argmax(logits[0, -1]).item())
        # One ordinary warm decode token before the paired same-frontier check.
        logits = model(
            mx.array([[first]], dtype=mx.uint32), cache=cache, logits_to_keep=0
        )
        mx.eval(logits, [s.state for s in cache])
        warm = int(mx.argmax(logits[0, -1]).item())
        receipt["ordinary_reference"] = {
            "token_ids": [first, warm],
            "text": adapter.tokenizer.decode([first, warm]),
            "finite": bool(mx.all(mx.isfinite(logits)).item()),
        }
        receipt["phase"] = "prefilled_and_warm"
        atomic(args.output / "results.json", receipt)
        guard()
        receipt["profile"] = profile_frontier(model, cache, warm, mx)
        receipt["route"] = adapter.diagnostics()
        receipt["memory"] = memory(mx)
        receipt["status"] = "completed"
        receipt["passed"] = (
            receipt["ordinary_reference"]["finite"]
            and receipt["profile"]["logits"]["bitwise_equal"]
            and receipt["profile"]["state_arrays_bitwise_equal"]
            and receipt["profile"]["state_metadata_equal"]
        )
        guard()
    except BaseException as error:
        receipt.update(
            status="failed",
            passed=False,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        model = cache = logits = None
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                receipt.update(status="failed", passed=False, cleanup_error=str(error))
        adapter = None
        gc.collect()
        try:
            from deepseek_v4_probe_residency import restore_probe_wired_limit

            restore_probe_wired_limit(mx, receipt)
            mx.clear_cache()
        except Exception as error:
            receipt.update(status="failed", passed=False, cleanup_error=str(error))
        atomic(args.output / "results.json", receipt)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
