#!/usr/bin/env python3
"""Owned M3 resilience probe for generalized streamed-expert policies.

This is exploratory correctness evidence, not qualification or a benchmark.
The caller must provide the CPG/paired-lock ownership window.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def command_output(*command: str) -> str:
    return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()


def host_sample() -> dict:
    return {
        "at": time.time(),
        "swap": command_output("sysctl", "-n", "vm.swapusage"),
        "thermal": command_output("pmset", "-g", "therm"),
        "vm_stat": command_output("vm_stat"),
        "model_processes": command_output(
            "pgrep", "-af", r"mlx|python.*(serve|probe|bench|qualif)"
        ),
    }


def adapter_class(model_path: str):
    config = json.loads((Path(model_path) / "config.json").read_text())
    text_config = config.get("text_config", config)
    model_type = text_config.get("model_type", config.get("model_type"))
    if model_type == "qwen3_5_moe_text":
        from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

        return Qwen3635BA3BAdapter
    if model_type == "cohere2_moe":
        from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter

        return NorthMiniCodeAdapter
    raise ValueError(f"resilience probe has no autoregressive MoE adapter for {model_type!r}")


def corpus_tokens(tokenizer, count: int) -> tuple[list[int], dict]:
    text = "\n\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "docs").glob("*.md"))
    )
    # Keep tokenizer inputs bounded so a long documentation corpus does not emit
    # a misleading model-context warning before this probe slices its token set.
    ids = []
    for start in range(0, len(text), 16_384):
        ids.extend(tokenizer.encode(text[start : start + 16_384]))
        if len(ids) >= count:
            break
    if not ids:
        raise ValueError("probe corpus tokenized empty")
    return (ids * ((count + len(ids) - 1) // len(ids)))[:count], {
        "source": "docs/*.md in lexical order",
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def run_sequence(
    adapter,
    ids,
    *,
    chunk: int,
    decode: int,
    uid: int,
    capture_decode_index: int | None = None,
    logprob_window: int = 5,
) -> dict:
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.weight_stream import expert_prefill_admission

    cache = make_prompt_cache(adapter.model)
    logits = None
    cuts = []
    for start in range(0, len(ids), chunk):
        stop = min(len(ids), start + chunk)
        phase = "terminal" if stop == len(ids) else "intermediate"
        with expert_prefill_admission(phase, (uid,)):
            logits = adapter.model(
                mx.array(ids[start:stop], dtype=mx.int32)[None], cache=cache
            )
        mx.eval(logits, *[entry.state for entry in cache])
        cuts.append(stop)
    generated = []
    capture = None
    for decode_index in range(decode):
        token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        if decode_index == capture_decode_index:
            row = logits[0, -1].astype(mx.float32)
            width = min(logprob_window, int(row.shape[0]))
            top_ids = mx.argpartition(-row, kth=width - 1)[:width]
            top_logits = row[top_ids]
            order = mx.argsort(-top_logits)
            top_ids = top_ids[order]
            top_logprobs = top_logits[order] - mx.logsumexp(row)
            mx.eval(top_ids, top_logprobs)
            capture = {
                "decode_index": decode_index,
                "top_token_ids": np.asarray(top_ids).astype(int).tolist(),
                "top_logprobs": np.asarray(top_logprobs).tolist(),
            }
        mx.eval(token)
        generated.append(int(np.array(token).reshape(-1)[0]))
        logits = adapter.model(token.reshape(1, 1), cache=cache)
        mx.eval(logits, *[entry.state for entry in cache])
    # NumPy cannot expose MLX bfloat16 through the buffer protocol.  Normalize
    # only the evidence copy; model execution and comparisons remain unchanged.
    digest_logits = logits.astype(mx.float32)
    mx.eval(digest_logits)
    digest = hashlib.sha256(np.asarray(digest_logits).tobytes()).hexdigest()
    del logits, cache
    gc.collect()
    mx.clear_cache()
    result = {
        "generated_token_ids": generated,
        "final_logits_sha256": digest,
        "cuts": cuts,
    }
    if capture is not None:
        result["capture"] = capture
    return result


def abandoned_prefill(adapter, ids, *, chunk: int, uid: int) -> None:
    import mlx.core as mx

    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.weight_stream import expert_prefill_admission

    cache = make_prompt_cache(adapter.model)
    for start in range(0, min(len(ids), chunk * 3), chunk):
        with expert_prefill_admission("intermediate", (uid,)):
            logits = adapter.model(
                mx.array(ids[start : start + chunk], dtype=mx.int32)[None], cache=cache
            )
        mx.eval(logits, *[entry.state for entry in cache])
    del logits, cache
    gc.collect()
    mx.clear_cache()


def simulated_source_drift(manager) -> dict:
    from mlx2.runtime.weight_stream import WeightSourceChanged

    cache = next(iter(manager.caches.values()))
    missing = next(expert for expert in range(cache.reader.num_experts) if expert not in cache._entries)
    handles = cache.reader._handles
    original = handles.check

    def changed(_path):
        raise WeightSourceChanged("adversarial simulated source drift")

    handles.check = changed
    try:
        try:
            cache.acquire([missing])
        except WeightSourceChanged as error:
            return {"passed": True, "error": str(error), "expert": missing}
        raise AssertionError("source drift did not fail closed")
    finally:
        handles.check = original


def hint_fault_and_stale_storm(manager) -> dict:
    hints = manager._router_hints
    groups = tuple(dict.fromkeys(cache.group for cache in manager.caches.values()))
    target = groups[min(1, len(groups) - 1)]
    target_cache = next(cache for cache in manager.caches.values() if cache.group == target)
    original = target_cache.reader.read_bytes

    def failed(_expert):
        raise OSError("adversarial hint read failure")

    target_cache.reader.read_bytes = failed
    try:
        hints.schedule(target, [0])
        try:
            target_cache.acquire([0])
        except OSError as error:
            fault = {"passed": True, "error": str(error)}
        else:
            raise AssertionError("hint read fault did not fail closed")
    finally:
        target_cache.reader.read_bytes = original

    before = hints.counters()
    for offset in range(24):
        hints.schedule(target, [(offset + index) % target_cache.reader.num_experts
                                for index in range(manager.plan.top_k)])
    after = hints.counters()
    stale = sum(
        after[key] - before[key]
        for key in (
            "stream_router_hint_stale_cancelled_total",
            "stream_router_hint_stale_dropped_total",
            "stream_router_hint_declined_busy_total",
        )
    )
    if stale <= 0:
        raise AssertionError("stale hint storm engaged no cancellation or discard path")
    return {"fault": fault, "stale_events": stale, "before": before, "after": after}


def malformed_topology_refusal(manager) -> dict:
    groups = tuple(dict.fromkeys(cache.group for cache in manager.caches.values()))
    triggers = [
        next(path for path, cache in manager.caches.items()
             if cache.group == group and path.endswith(".down_proj"))
        for group in groups
    ]
    try:
        manager.configure_router_hint_order(tuple(reversed(groups)), trigger_paths=triggers)
    except ValueError as error:
        return {"passed": True, "error": str(error)}
    raise AssertionError("malformed adapter topology was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-gib", type=float, default=6.0)
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--short-prompt-tokens", type=int, default=257)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--capture-decode-index", type=int, default=-1)
    parser.add_argument("--logprob-window", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty evidence directory {out}")
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.runtime.streamed_load import (
        WeightStreamRequest,
        declared_router_hints,
    )

    Adapter = adapter_class(args.model)
    router_hints = declared_router_hints(Adapter)
    adapter_options = {}
    if "default_eager_dispatch_stride" in vars(Adapter):
        adapter_options["execution_policy"] = {"eager_dispatch_stride": 0}

    report = {
        "schema": "mlx2.slotstream-resilience-probe.v1",
        "state": "exploratory_unqualified_not_performance_evidence",
        "source_revision": os.environ.get("MLX2_SOURCE_REVISION", "unknown"),
        "source_dirty": os.environ.get("MLX2_SOURCE_DIRTY", "unknown"),
        "platform": platform.platform(),
        "device": mx.device_info(),
        "adapter": f"{Adapter.__module__}.{Adapter.__name__}",
        "adapter_options": adapter_options,
        "declared_router_hints": sorted(router_hints),
        "parameters": vars(args),
        "host_before": host_sample(),
        "arms": {},
        "failures": [],
    }
    atomic_json(out / "resilience.json", report)
    adapter = None
    try:
        # Fully resident ordinary path is the correctness oracle.
        adapter = Adapter(args.model, **adapter_options)
        ids, corpus = corpus_tokens(adapter.tokenizer, args.prompt_tokens)
        report["corpus"] = corpus
        short = min(args.short_prompt_tokens, args.prompt_tokens)
        scenarios = [(short, min(8, args.decode_tokens), 1001)]
        if args.prompt_tokens != short:
            scenarios.append((args.prompt_tokens, args.decode_tokens, 1002))
        sequence_options = {
            "capture_decode_index": (
                args.capture_decode_index if args.capture_decode_index >= 0 else None
            ),
            "logprob_window": args.logprob_window,
        }
        reference = {
            str(length): run_sequence(
                adapter,
                ids[:length],
                chunk=args.prefill_chunk,
                decode=decode,
                uid=uid,
                **sequence_options,
            )
            for length, decode, uid in scenarios
        }
        report["ordinary_reference"] = reference
        report["artifact"] = adapter.identity
        adapter.close()
        adapter = None
        gc.collect()
        mx.clear_cache()
        atomic_json(out / "resilience.json", report)

        arms = [
            ("stock_stream", "lru", "off"),
            ("terminal_hot", "terminal_hot", "off"),
        ]
        if "next_layer_reuse" in router_hints:
            arms.extend(
                (
                    ("router_hint", "lru", "next_layer_reuse"),
                    ("combined", "terminal_hot", "next_layer_reuse"),
                )
            )
        for arm_index, (name, admission, hint) in enumerate(arms):
            started = time.time()
            request = WeightStreamRequest(
                mode="moe_experts",
                budget_bytes=int(args.budget_gib * (1 << 30)),
                read_workers=args.read_workers,
                prefill_admission=admission,
                router_hint=hint,
            )
            adapter = Adapter(
                args.model,
                weight_streaming=request,
                **adapter_options,
            )
            manager = adapter.weight_stream
            arm = {
                "receipt": manager.receipt(),
                "plan": manager.plan.as_dict(),
                "counters_before": manager.counters(),
                "adversarial": {},
            }
            if name == "stock_stream":
                arm["adversarial"]["source_drift"] = simulated_source_drift(manager)
            if hint != "off":
                arm["adversarial"]["malformed_topology"] = malformed_topology_refusal(manager)
                arm["adversarial"]["hint_fault_and_stale"] = hint_fault_and_stale_storm(manager)
            abandoned_prefill(
                adapter, ids, chunk=args.prefill_chunk, uid=9000 + arm_index
            )
            outputs = {}
            for length, decode, uid in scenarios:
                outputs[str(length)] = run_sequence(
                    adapter,
                    ids[:length],
                    chunk=args.prefill_chunk,
                    decode=decode,
                    uid=uid + 100 * arm_index,
                    **sequence_options,
                )
                arm["outputs"] = outputs
                report["arms"][name] = arm
                atomic_json(out / "resilience.json", report)
                if outputs[str(length)] != reference[str(length)]:
                    raise AssertionError(f"{name} diverged from ordinary at {length} tokens")
            warm = run_sequence(
                adapter,
                ids[: scenarios[0][0]],
                chunk=args.prefill_chunk,
                decode=scenarios[0][1],
                uid=11000 + arm_index,
                **sequence_options,
            )
            if warm != reference[str(scenarios[0][0])]:
                raise AssertionError(f"{name} warm replay diverged from ordinary")
            arm["warm_replay"] = warm
            arm["counters_after"] = manager.counters()
            if admission == "terminal_hot":
                if arm["counters_after"]["stream_prefill_intermediate_calls_total"] <= 0:
                    raise AssertionError(f"{name} did not engage intermediate prefill")
                if arm["counters_after"]["stream_prefill_terminal_calls_total"] <= 0:
                    raise AssertionError(f"{name} did not engage terminal admission")
                if arm["counters_after"]["stream_prefill_request_resets_total"] <= 0:
                    raise AssertionError(f"{name} did not reset abandoned request state")
            if hint != "off":
                if arm["counters_after"]["stream_router_hint_issued_total"] <= 0:
                    raise AssertionError(f"{name} issued no router hints")
                if arm["counters_after"]["stream_router_hint_adopted_bytes_total"] <= 0:
                    raise AssertionError(f"{name} adopted no hinted bytes")
            arm["elapsed_diagnostic_seconds"] = time.time() - started
            arm["host_after"] = host_sample()
            report["arms"][name] = arm
            atomic_json(out / "resilience.json", report)
            adapter.close()
            adapter.close()
            adapter = None
            gc.collect()
            mx.clear_cache()
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        if adapter is not None:
            adapter.close()
        report["host_after"] = host_sample()
        report["finished_at"] = time.time()
        atomic_json(out / "resilience.json", report)


if __name__ == "__main__":
    main()
