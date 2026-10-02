#!/usr/bin/env python3
"""One-load, in-process serving memory ablation; never runs Metal without consent.

Defaults reproduce the resolved 2207af9a ladder environment, not today's
adapter defaults. Every arm starts with an empty APCv2 and derived projection /
compile caches, then accumulates distinct cold/warm prefixes for --cycles.
The coordinator owns GPU locking; the flag is an acknowledgement, not a lease.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import subprocess
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30
REF = "qualify/e8861bb5-uncensored"
RUN = "qualification/runs/qualify-e8861bb5-uncensored"
ARMS = (
    "baseline", "q1_credit_off", "hc_off", "shared_fold_off", "routed_off",
    "expert_views_off", "attn_rows_off", "moe_window_off", "batch_gdn_off",
    "compiled_off", "qsa_scores_off", "buffer_cache_zero", "gdn_prefill_off",
    "apc_trim_each_cycle",
)


def run_text(relative):
    """Read a file of the committed run directory.

    The run is committed on main, so the working tree has it. Snapshot clones
    (docs/QUALIFICATION.md Step 1) carry the qualify branch only as
    origin/<ref>, so the ref fallback tries both spellings.
    """
    path = ROOT / RUN / relative
    if path.is_file():
        return path.read_text()
    for ref in (REF, f"origin/{REF}"):
        try:
            return subprocess.check_output(["git", "show", f"{ref}:{RUN}/{relative}"],
                                           cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, OSError):
            continue
    raise FileNotFoundError(f"{RUN}/{relative}: not in the working tree, {REF} or origin/{REF}")


def served_config(ladder_file=None, policy_file=None):
    ladder = json.loads(Path(ladder_file).read_text() if ladder_file else run_text(
        "results/ladder-flash-next-uncensored-mtp2/ladder-short.json"))
    policy = json.loads(Path(policy_file).read_text() if policy_file else run_text(
        "policies/flash-next-uncensored-policy.json"))
    initial = ladder["initial"]
    # Freeze resolved choices that the three-key service policy leaves implicit.
    policy = {**initial["execution"]["policy"], **policy}
    # These default-on fields were added after the comparison's newer endpoint.
    policy.setdefault("qsa_fused_scores", False)
    policy.setdefault("fused_gdn_batch_verify", "off")
    # Server applies adapter-declared route defaults; direct ServingEngine does
    # not. Preserve these resolved settings too, especially interior stores.
    for key in ("host_memory_signals", "self_mtp_copy_draft", "prefill_scheduling",
                "apc_interior_checkpoints", "fly_verification"):
        if key in initial["settings"]:
            policy.setdefault(key, initial["settings"][key])
    return initial, policy


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(Path.home() / "mlx-models" /
                                         "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"))
    p.add_argument("--served-ladder", type=Path)
    p.add_argument("--policy", type=Path)
    p.add_argument("--runtime-src", type=Path,
                   help="use a pinned snapshot's src directory instead of current PYTHONPATH")
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="no MLX import or model load")
    p.add_argument("--summarize", type=Path, help="summarize probe JSONL without MLX")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--rounds", type=int, default=2, help="forward then reverse arm order")
    p.add_argument("--cycles", type=int, default=4, help="unique cold/warm prefix pairs per arm")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--widths", default="1,4")
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--out", type=Path, default=Path("/tmp/flashnext-peak-memory.jsonl"))
    a = p.parse_args(argv)
    a.arms = a.arms.split(",")
    a.widths = [int(w) for w in a.widths.split(",")]
    if set(a.arms) - set(ARMS) or "baseline" not in a.arms:
        p.error("arms must include baseline and use known arm names")
    if not a.widths or set(a.widths) - {1, 4}:
        p.error("widths must be 1 and/or 4")
    if min(a.rounds, a.cycles, a.context, a.gen) < 1 or a.timeout <= 0:
        p.error("rounds, cycles, context, gen and timeout must be positive")
    if not a.dry_run and not a.summarize and not a.i_own_the_gpu:
        p.error("refusing model load / Metal without --i-own-the-gpu")
    return a


def summarize(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    baselines = {(r["round"], r["width"], r["cycle"], r["phase"]):
                 [o["sha256"] for o in r["outputs"]]
                 for r in records if r["kind"] == "sample" and r["arm"] == "baseline"}
    summary = []
    for width, arm in sorted({(r["width"], r["arm"]) for r in records if r["kind"] == "arm_end"}):
        ends = [r for r in records if r["kind"] == "arm_end" and (r["width"], r["arm"]) == (width, arm)]
        samples = [r for r in records if r["kind"] == "sample" and (r["width"], r["arm"]) == (width, arm)]
        comparisons = [([o["sha256"] for o in r["outputs"]] == baselines[key])
                       for r in samples
                       if (key := (r["round"], width, r["cycle"], r["phase"])) in baselines]
        row = {"width": width, "arm": arm, "rounds": len(ends),
               "output_matches": sum(comparisons), "output_comparisons": len(comparisons)}
        starts = [r for r in records if r["kind"] == "arm_start" and
                  (r["width"], r["arm"]) == (width, arm) and
                  r["round"] in {end["round"] for end in ends}]
        if starts:
            row["start_active_gib"] = statistics.median(r["memory"]["metal_active_bytes"] / GIB for r in starts)
            row["start_apc_resident_gib"] = statistics.median(
                r["apcv2"]["idle_disk"]["resident_bytes"] / GIB for r in starts)
        for phase in ("before_clear", "after_clear"):
            row[phase] = {key.replace("_bytes", "_gib"): statistics.median(r[phase]["memory"][key] / GIB for r in ends)
                          for key in ("metal_peak_bytes", "metal_active_bytes", "metal_buffer_cache_bytes")}
            row[phase]["apc_resident_gib"] = statistics.median(
                r[phase]["apcv2"]["idle_disk"]["resident_bytes"] / GIB for r in ends)
            footprints = [r[phase]["memory"].get("process_physical_footprint_bytes") for r in ends]
            if all(v is not None for v in footprints):
                row[phase]["process_physical_footprint_gib"] = statistics.median(v / GIB for v in footprints)
        summary.append(row)
    baseline_peak = {r["width"]: r["before_clear"]["metal_peak_gib"]
                     for r in summary if r["arm"] == "baseline"}
    for row in summary:
        row["peak_delta_gib_vs_baseline"] = (row["before_clear"]["metal_peak_gib"] -
                                              baseline_peak.get(row["width"], 0))
    return {"units": "GiB", "arms": summary,
            "errors": [r for r in records if r["kind"] == "arm_error"]}


def memory_sample(mx):
    from mlx2.runtime.os_memory import physical_footprint_bytes
    from mlx2.memory import execution_headroom, host_available_bytes
    return {
        "metal_peak_bytes": int(mx.get_peak_memory()),
        "metal_active_bytes": int(mx.get_active_memory()),
        "metal_buffer_cache_bytes": int(mx.get_cache_memory()),
        "process_physical_footprint_bytes": physical_footprint_bytes(),
        "headroom_bytes": execution_headroom(host_signals=True),
        "host_available_bytes": host_available_bytes(True),
    }


def clear_derived(model, hc):
    """Release diagnostic caches, never resident source weights or live state."""
    counts = {}
    for _, module in model.named_modules():
        for name in ("_qsa_fused_cache", "_hc_decode_plan", "_mlx2_expert_views",
                     "_ple_compile_cache", "_ple_compile_seen"):
            if name in module.__dict__:
                counts[name] = counts.get(name, 0) + 1
                if name == "_qsa_fused_cache":
                    object.__setattr__(module, name, None)
                else:
                    del module.__dict__[name]
    hc._COMPILED.clear()
    return counts


def arm_switches(stack, arm, model, mx):
    """Use live setters; env-only edits would not change import-latched flags.

    All overrides are scoped to this standalone probe and restored at drain.
    Q1 has no production kill switch: substitute its pure credit calculation
    locally, which the engine's captured execution_headroom reads on each call.
    """
    from mlx2 import memory
    from mlx2.runtime.models import qwen4_hc_decode as hc
    from mlx2.runtime.models import qwen4_attn_rows as attn
    from mlx2.runtime.models import qwen4_routed_decode as routed
    try:
        qsa = importlib.import_module("mlx2.runtime.models.qwen4_qsa_scores")
    except ModuleNotFoundError as exc:
        if exc.name != "mlx2.runtime.models.qwen4_qsa_scores":
            raise
        qsa = None  # absent at both historical endpoints
    from mlx2.runtime.models import qwen4_exp as exp

    def setter(set_fn, old, new):
        set_fn(new)
        stack.callback(set_fn, old)

    touched = []
    if arm == "q1_credit_off":
        stack.enter_context(patch.object(memory, "host_term_reserve_credit_bytes", lambda *_: 0))
        touched.append("host_term_reserve_credit_bytes=0 (probe-local substitution)")
    if arm == "hc_off":
        setter(hc.set_hc_decode_enabled, hc.hc_decode_enabled(), False)
        touched.append("hc_decode_kernels=false")
    if arm == "expert_views_off":
        setter(routed.set_expert_views, routed._VIEWS, False)
        touched.append("expert_views=false")
    if arm == "attn_rows_off":
        setter(attn.set_enabled, attn.enabled(), False)
        touched.append("attn_fused_rows=false")
    if arm == "qsa_scores_off":
        if qsa is not None:
            setter(qsa.set_enabled, qsa.enabled(), False)
        touched.append("qsa_fused_scores=false (off or absent in endpoint receipt)")
    if arm == "compiled_off":
        mx.disable_compile()
        stack.callback(mx.enable_compile)
        stack.enter_context(patch.object(exp, "_PLE_COMPILE", False))
        touched.append("mx.disable_compile + PLE eager")
    if arm == "buffer_cache_zero":
        previous = mx.set_cache_limit(0)
        stack.callback(mx.set_cache_limit, previous)
        touched.append("mx.set_cache_limit(0)")
    for _, module in model.named_modules():
        for field, method, new in (
            ("moe_window_consumers", "set_moe_window_consumers", frozenset()),
            ("fused_gdn_batch_decode_mode", "set_fused_gdn_batch_decode_mode", "off"),
            ("fused_gdn_batch_verify_mode", "set_fused_gdn_batch_verify_mode", "off"),
            ("fused_gdn_prefill_mode", "set_fused_gdn_prefill_mode", "stock"),
        ):
            target = ("moe_window_off" if field.startswith("moe") else
                      "gdn_prefill_off" if field == "fused_gdn_prefill_mode" else "batch_gdn_off")
            if arm == target and hasattr(module, method):
                setter(getattr(module, method), getattr(module, field), new)
                touched.append(field)
        if arm in {"routed_off", "shared_fold_off"} and hasattr(module, "set_moe_routed_decode_mode"):
            setter(module.set_moe_routed_decode_mode, module.switch_mlp.routed_decode_mode,
                   "off" if arm == "routed_off" else "gate_up_down")
            touched.append("moe_routed_decode")
    return sorted(set(touched))


def requests(engine, context, width, cycle, gen):
    """Calibrate exact token count with the served tokenizer; same text per arm."""
    result = []
    for lane in range(width):
        header = f"Session memory-measurement-{cycle:04d}-{lane:02d}.\n"
        filler = "archival evidence "
        lo, hi = 0, context * 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            text = header + filler * mid
            n = len(engine.adapter.tokenizer.encode(text, add_special_tokens=False))
            if n <= context:
                lo = mid
            else:
                hi = mid - 1
        text = header + filler * lo
        # Fill a possible one-token gap without slicing decoded token text.
        while len(engine.adapter.tokenizer.encode(text, add_special_tokens=False)) < context:
            candidate = text + " x"
            if len(engine.adapter.tokenizer.encode(candidate, add_special_tokens=False)) > context:
                break
            text = candidate
        n = len(engine.adapter.tokenizer.encode(text, add_special_tokens=False))
        if n != context:
            raise RuntimeError(f"could not calibrate exact {context} tokens: got {n}")
        result.append({"prompt": text, "temperature": 0, "seed": 1,
                       "min_tokens": gen, "max_tokens": gen})
    return result


def run_cohort(engine, reqs, timeout):
    cohort = f"memory-{time.monotonic_ns()}"
    jobs = [engine.submit({**r, **({"batch_cohort": {"id": cohort, "size": len(reqs)}}
                                   if len(reqs) > 1 else {})}) for r in reqs]
    deadline = time.monotonic() + timeout
    outputs = []
    for job in jobs:
        text = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("cohort timed out")
            try:
                event = job.events.get(timeout=min(1, remaining))
            except queue.Empty:
                if engine.error:
                    raise RuntimeError(engine.error)
                continue
            if "error" in event:
                raise RuntimeError(f"cohort refused/failed: {event}")
            if "delta" in event:
                text.append(event["delta"].get("content", "") or "")
            if "text" in event:
                text.append(event["text"])
            if "finish_reason" in event:
                outputs.append({"sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
                                "receipt": event.get("receipt")})
                break
    return outputs


def main(argv=None):
    a = parse_args(argv)
    if a.summarize:
        print(json.dumps(summarize(a.summarize), indent=2))
        return
    initial, policy = served_config(a.served_ladder, a.policy)
    settings = initial["settings"]
    if settings["cache_bytes"] != 16 * GIB or not settings["mtp"]:
        raise ValueError("this probe requires the native-MTP / 16 GiB served receipt")
    metadata = {"schema": "mlx2.flashnext-memory-ablation.v1", "pid": os.getpid(),
                "harness_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "reference_runtime": initial["runtime"], "artifact": initial["artifact"],
                "policy": policy, "environment": settings["environment"],
                "arms": a.arms, "widths": a.widths, "context": a.context,
                "cycles": a.cycles, "rounds": a.rounds,
                "runtime_src": str(a.runtime_src) if a.runtime_src else None,
                "limitations": ["historical peaks are cumulative, probe peaks reset per arm",
                                 "disk streaming absent on this served route",
                                 "compiled/driver caches may survive arm cleanup",
                                 "coordinator must check locks and service ownership"]}
    if a.dry_run:
        print(json.dumps(metadata, indent=2))
        return
    # Guard above precedes every MLX or runtime import.
    if a.runtime_src:
        sys.path.insert(0, str(a.runtime_src.expanduser().resolve()))
    import mlx.core as mx
    from mlx2.adapters import flash_next
    from mlx2.adapters.flash_next_policy import FlashNextPolicy
    from mlx2.serving import ServingEngine

    for later_field in ("qsa_fused_scores", "fused_gdn_batch_verify"):
        if later_field not in FlashNextPolicy.__dataclass_fields__:
            if policy[later_field] not in (False, "off"):
                raise ValueError(f"pinned runtime cannot select {later_field}")
            del policy[later_field]

    environment = dict(settings["environment"])
    environment["MLX_QWEN4_PLE_NVME"] = str(Path(a.model).expanduser() / "ple_rows.bin")
    def historical_environment(path, policy=None):
        for name in tuple(os.environ):
            if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
                del os.environ[name]
        os.environ.update(environment)
        return dict(environment)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as out, ExitStack() as outer:
        def emit(record):
            out.write(json.dumps(record, default=str) + "\n")
            out.flush()
        emit({"kind": "metadata", **metadata})
        outer.enter_context(patch.object(flash_next, "configure_environment", historical_environment))
        # User-supplied allocator env stays in metadata; never silently assume unbounded.
        emit({"kind": "allocator_environment", "MLX2_CACHE_LIMIT_GIB": os.getenv("MLX2_CACHE_LIMIT_GIB")})
        import tempfile
        disk = outer.enter_context(tempfile.TemporaryDirectory(prefix="flashnext-memory-"))
        engine = ServingEngine(
            a.model, adapter_factory=flash_next.FlashNextAdapter,
            mtp=True, qualification_mode=True, execution_policy=policy,
            max_lanes=settings["max_lanes"], max_inflight=settings["max_inflight"],
            max_context=settings["max_context"], prefill_step=settings["prefill_step"],
            cache_bytes=settings["cache_bytes"], cache_dir=disk,
        )
        outer.callback(engine.close)
        if not engine.ready.wait(a.timeout) or engine.error:
            raise RuntimeError(engine.error or "model did not become ready")
        if engine.adapter.identity["fingerprint"] != initial["artifact"]:
            raise RuntimeError("loaded artifact differs from served ladder fingerprint")
        from mlx2.runtime.models import qwen4_hc_decode as hc
        from mlx2.runtime.segmented_self_mtp import segmented_self_mtp_stats
        emit({"kind": "loaded", "status": engine.status(), "memory": memory_sample(mx),
              "device_info": mx.device_info(),
              "actual_environment": {k: v for k, v in os.environ.items() if k.startswith("MLX")}})

        def drained(fn):
            return engine._exclusive_adapter_operation("memory_probe", fn, timeout=a.timeout)

        def snapshot(_):
            mx.synchronize()
            return {"memory": memory_sample(mx), "apcv2": engine.apc.apc_stats,
                    "execution": engine.adapter.diagnostics(),
                    "segmented_mtp": segmented_self_mtp_stats()}

        for round_index in range(a.rounds):
            order = a.arms if round_index % 2 == 0 else list(reversed(a.arms))
            for width in a.widths:
                reqs = [requests(engine, a.context, width, c, a.gen) for c in range(a.cycles)]
                for arm in order:
                    with ExitStack() as switches:
                        def prepare(adapter):
                            mx.synchronize()
                            engine.apc.clear()
                            engine.host_prompt_cache.clear()
                            dropped = clear_derived(adapter.model, hc)
                            touched = arm_switches(switches, arm, adapter.model, mx)
                            gc.collect()
                            mx.clear_cache()
                            mx.reset_peak_memory()
                            return {"dropped": dropped, "switches": touched, **snapshot(adapter)}
                        emit({"kind": "arm_start", "round": round_index, "width": width,
                              "arm": arm, **drained(prepare)})
                        error = None
                        try:
                            for cycle, cohort_requests in enumerate(reqs):
                                for phase in ("cold", "warm"):
                                    started = time.monotonic()
                                    outputs = run_cohort(engine, cohort_requests, a.timeout)
                                    emit({"kind": "sample", "round": round_index, "width": width,
                                          "arm": arm, "cycle": cycle, "phase": phase,
                                          "wall_seconds": time.monotonic() - started,
                                          "outputs": outputs, **drained(snapshot)})
                                if arm == "apc_trim_each_cycle":
                                    drained(lambda _: engine.apc.trim_to(n_bytes=0))
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            emit({"kind": "arm_error", "round": round_index, "width": width,
                                  "arm": arm, "error": error})
                        def finish(adapter):
                            before = snapshot(adapter)
                            mx.clear_cache()
                            after = snapshot(adapter)
                            switches.close()
                            return {"before_clear": before, "after_clear": after}
                        result = drained(finish)
                        emit({"kind": "arm_failed_end" if error else "arm_end",
                              "round": round_index, "width": width,
                              "arm": arm, **result})
    print(a.out)


if __name__ == "__main__":
    main()
