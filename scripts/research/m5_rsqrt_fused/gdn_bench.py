#!/usr/bin/env python3
"""Isolated actual GDN kernel rsqrt experiment; never alters production modules.

Kernel bodies, launch functions and factory arguments come from the local
MIT-licensed mlx2 implementation. Only selected precise::rsqrt tokens and the
kernel's cache-disambiguating name change. Source hashes accompany results.
Run exclusively through the CPG GPU lease and owned_exec.py wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from common import compare_arrays, require_ownership, time_variants


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def clone_function(fn, substitutions):
    cloned = types.FunctionType(
        fn.__code__, {**fn.__globals__, **substitutions}, fn.__name__,
        fn.__defaults__, fn.__closure__,
    )
    cloned.__kwdefaults__ = fn.__kwdefaults__
    return cloned


def variant_source(source, kind, expected_total):
    precise, fast = "metal::precise::rsqrt", "metal::fast::rsqrt"
    assert source.count(precise) == expected_total
    assert source.count(fast) == 0
    lines = source.splitlines(keepends=True)
    changed = []
    sites = []
    for i, line in enumerate(lines):
        if precise not in line:
            continue
        assert line.count(precise) == 1
        category = "output" if "po / (float)DV + norm_eps" in line else "qk"
        selected = kind == "fast_all" or kind == f"fast_{category}"
        sites.append({"line_in_kernel": i + 1, "category": category,
                      "original": line.strip(), "changed": selected})
        if selected:
            lines[i] = line.replace(precise, fast)
            changed.append(i + 1)
    result = "".join(lines)
    assert result.replace(fast, precise) == source
    assert result.count(fast) == len(changed)
    assert len(changed) == {"precise": 0, "fast_qk": expected_total - 1,
                            "fast_output": 1, "fast_all": expected_total}[kind]
    return result, sites


def build_variants(mx, module, route):
    """Use original factory metadata and original exported host launch code."""
    capture_mx = types.SimpleNamespace(
        fast=types.SimpleNamespace(metal_kernel=lambda **kwargs: kwargs)
    )
    original_factory = module._kernel.__wrapped__
    metadata = clone_function(original_factory, {"mx": capture_mx})()
    original_fn = getattr(module, f"qwen4_fused_gdn_{route}")
    variants, sources, calls = {}, {}, {}
    for kind in ("precise", "fast_qk", "fast_output", "fast_all"):
        source, sites = variant_source(metadata["source"], kind,
                                       5 if route == "decode" else 3)
        name = f"m5_rsqrt_{route}_{kind}_{sha(source)[:12]}"
        kernel = mx.fast.metal_kernel(**{**metadata, "source": source, "name": name})
        calls[kind] = 0

        def kernel_getter(kernel=kernel, kind=kind):
            calls[kind] += 1
            return kernel

        variants[kind] = clone_function(original_fn, {"_kernel": kernel_getter})
        sources[kind] = {"kernel_name": name, "source_sha256": sha(source),
                         "header_sha256": sha(metadata["header"]), "sites": sites}
    return variants, sources, calls


def make_inputs(mx, np, *, seed, steps, scale=1.0):
    rng = np.random.default_rng(seed)
    cd, vd, hv, dk, dv = 10240, 6144, 48, 128, 128

    def rand(shape, stdev=1.0, offset=0.0, dtype=None):
        a = (rng.standard_normal(shape) * stdev + offset).astype(np.float32)
        return mx.array(a, dtype=mx.bfloat16 if dtype is None else dtype)

    # Projected activations, convolution histories, recurrent state all nonzero
    # for nonzero scales. Gate and weight scales remain fixed across cases.
    return {
        "qkv": rand((1, steps, cd), scale),
        "z": rand((1, steps, vd), 1.0),
        "b": rand((1, steps, hv), 1.0),
        "a": rand((1, steps, hv), 1.0, -1.0),
        "conv_state": rand((1, 3, cd), scale),
        "conv_weight": rand((cd, 4, 1), 0.25),
        "A_log": rand((hv,), 0.25, -0.5, mx.float32),
        "dt_bias": rand((hv,), 0.3, -1.0),
        "recurrent_state": rand((1, hv, dv, dk), 0.05 * scale, dtype=mx.float32),
        "norm_weight": rand((dv,), 0.1, 1.0),
    }


def admitted(module, inputs, route, architecture):
    extra = dict(mask=None, spans=(), speculating=route == "verify",
                 training=False, sharded=False, num_key_heads=16,
                 num_value_heads=48, key_head_dim=128, value_head_dim=128,
                 conv_kernel=4, gate_activation="sigmoid" if architecture == "qwen4" else "swish")
    if route == "decode":
        extra["architecture"] = architecture
    result = getattr(module, f"admit_qwen4_fused_gdn_{route}")(**inputs, **extra)
    assert result.accepted, result.reason
    return result.reason


def evaluate(mx, fn, inputs, kwargs):
    result = fn(**inputs, norm_eps=1e-6, **kwargs)
    mx.eval(*result)
    return result


def output_metrics(mx, np, refs, actuals, route):
    names = ["output"] if route == "single" else ["output", "conv_state", "recurrent_state"]
    if route == "verify":
        names += ["state_snapshots", "conv_snapshots"]
    result = {}
    for name, ref, actual in zip(names, refs, actuals, strict=True):
        metric = compare_arrays(ref, actual)
        assert metric["nonfinite_reference"] == 0, (name, "nonfinite precise reference", metric)
        assert metric["nonfinite_candidate"] == 0, (name, "nonfinite candidate", metric)
        # Native storage bits distinguish a true exact match from numeric equality.
        bits_type = mx.uint16 if ref.dtype == mx.bfloat16 else mx.uint32
        rbits, abits = np.asarray(ref.view(bits_type)), np.asarray(actual.view(bits_type))
        metric["storage_bit_mismatch_count"] = int(np.count_nonzero(rbits != abits))
        metric["storage_bit_exact"] = bool(np.array_equal(rbits, abits))
        result[name] = metric
    return result


def chain_call(fn, inputs, kwargs, count):
    current = inputs
    output = None
    for _ in range(count):
        output = fn(**current, norm_eps=1e-6, **kwargs)
        current = {**current, "conv_state": output[1], "recurrent_state": output[2]}
    return output


def compiled_chain_call(mx, fn, inputs, kwargs, count):
    """Compile a dependent chain, accepting arrays as dynamic graph inputs.

    Every invocation starts from the same initial states and repeats the same
    synthetic projected block count times. This is an amortized kernel test,
    not end-to-end generation or a replay of different model token activations.
    """
    keys = tuple(inputs)
    values = tuple(inputs[key] for key in keys)

    def run(*arrays):
        return chain_call(fn, dict(zip(keys, arrays, strict=True)), kwargs, count)

    compiled = mx.compile(run)
    return lambda: compiled(*values)


def drift_test(mx, np, variants, architecture, threadgroup_y, seed, count):
    all_inputs = make_inputs(mx, np, seed=seed, steps=count, scale=1.0)
    mx.eval(*all_inputs.values())
    kwargs = {"threadgroup_y": threadgroup_y, "architecture": architecture}
    by_variant = {}
    checkpoints = sorted(set([1, min(8, count), min(32, count), count]))
    for name, fn in variants.items():
        current = all_inputs.copy()
        snapshots, outputs = {}, []
        for step in range(count):
            for field in ("qkv", "z", "a", "b"):
                current[field] = all_inputs[field][:, step:step + 1]
            out = evaluate(mx, fn, current, kwargs)
            outputs.append(out[0])
            current["conv_state"], current["recurrent_state"] = out[1:3]
            if step + 1 in checkpoints:
                snapshots[str(step + 1)] = out
        by_variant[name] = (mx.concatenate(outputs, axis=1), snapshots)
        mx.eval(by_variant[name][0])
    reference, ref_snapshots = by_variant["precise"]
    return {name: {
        "all_token_outputs": output_metrics(mx, np, (reference,), (out,), "single")["output"],
        "checkpoints": {step: output_metrics(mx, np, ref_snapshots[step], snapshots[step], "decode")
                        for step in ref_snapshots},
    } for name, (out, snapshots) in by_variant.items() if name != "precise"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--drift-steps", type=int, default=64)
    parser.add_argument("--chain", type=int, default=8)
    args = parser.parse_args()
    assert args.rounds > 0 and args.inner > 0 and args.seeds > 0 and args.chain > 0
    ownership = require_ownership()
    import numpy as np
    import mlx.core as mx
    from mlx2.runtime.models import qwen4_fused_gdn as decode
    from mlx2.runtime.models import qwen4_fused_gdn_verify as verify

    result = {
        "experiment": "actual_fused_gdn_rsqrt_only",
        "timing_kind": "interleaved warmed wall time including Python graph creation and mx.eval; not isolated GPU timestamps",
        "quality_scope": "synthetic tensors and recurrent drift only, no model logits/perplexity/acceptance qualification",
        "mlx_version": importlib.metadata.version("mlx"),
        "device": mx.metal.device_info(), "ownership": ownership,
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "common_sha256": hashlib.sha256(Path(__file__).with_name("common.py").read_bytes()).hexdigest(),
        "completed": False,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sources": {}, "variants": {}, "cases": [], "drift": {}, "calls": {},
        "chain_workload": "Each chain starts with the same initial states, then repeats the same projected synthetic block with sequential conv/recurrent state dependencies; no model inference.",
        "input_distribution": {
            "qkv_conv_state": "N(0,scale^2)", "z_b": "N(0,1)",
            "a": "N(-1,1)", "conv_weight": "N(0,0.25^2)",
            "A_log_float32": "N(-0.5,0.25^2)", "dt_bias": "N(-1,0.3^2)",
            "recurrent_state_float32": "N(0,(0.05*scale)^2)",
            "norm_weight": "N(1,0.1^2)", "other_dtype": "bfloat16", "norm_eps": 1e-6,
        },
    }
    provenance = json.loads((ROOT / "provenance/flashnext.json").read_text())
    for route, module in (("decode", decode), ("verify", verify)):
        file = Path(module.__file__)
        result["sources"][route] = {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                                      "provenance": provenance[f"models/{file.name}"]}
        variants, sources, calls = build_variants(mx, module, route)
        result["variants"][route] = sources
        configurations = [(1, "qwen4"), (1, "agnes")] if route == "decode" else [(s, "qwen4") for s in (2, 4, 8)]
        for steps, architecture in configurations:
            probe = decode.probe_qwen4_fused_gdn_decode(mx.bfloat16) if route == "decode" else verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, steps)
            if probe is None:
                raise RuntimeError(f"Original {route} width {steps} is unavailable")
            kwargs = {"threadgroup_y": probe}
            if route == "decode":
                kwargs["architecture"] = architecture
            key = f"{route}_{architecture}_{steps}"
            case = {"case": key, "steps": steps, "architecture": architecture,
                    "threadgroup_y": probe, "geometry": [1, steps, 16, 48, 128, 128],
                    "accuracy": []}
            result["cases"].append(case)
            print(json.dumps({"status": "start", "case": key}), flush=True)
            timing_inputs = None
            for seed_index in range(args.seeds):
                seed = args.seed + seed_index
                for scale in (0.0, 1e-4, 1.0, 8.0):
                    inputs = make_inputs(mx, np, seed=seed, steps=steps, scale=scale)
                    mx.eval(*inputs.values())
                    admitted(module, inputs, route, architecture)
                    reference = evaluate(mx, variants["precise"], inputs, kwargs)
                    exported = evaluate(mx, getattr(module, f"qwen4_fused_gdn_{route}"), inputs, kwargs)
                    export_parity = output_metrics(mx, np, exported, reference, route)
                    if not all(v["storage_bit_exact"] for v in export_parity.values()):
                        raise AssertionError(f"Cloned precise kernel disagrees with exported original: {key}")
                    checks = {name: output_metrics(mx, np, reference, evaluate(mx, fn, inputs, kwargs), route)
                              for name, fn in variants.items() if name != "precise"}
                    case["accuracy"].append({"seed": seed, "scale": scale,
                                             "exported_original_parity": export_parity, "vs_precise": checks})
                    if seed_index == 0 and scale == 1.0:
                        timing_inputs = inputs
            assert timing_inputs is not None
            timings = {name: (lambda fn=fn: fn(**timing_inputs, norm_eps=1e-6, **kwargs))
                       for name, fn in variants.items()}
            case["single_call_timing"] = time_variants(timings, rounds=args.rounds, inner=args.inner, seed=args.seed)
            chained = {name: (lambda fn=fn: chain_call(fn, timing_inputs, kwargs, args.chain))
                       for name, fn in variants.items()}
            case["dependent_chain_timing"] = time_variants(chained, rounds=args.rounds, inner=args.inner, seed=args.seed + 1)
            case["dependent_chain_calls_per_eval"] = args.chain
            compiled_chained = {name: compiled_chain_call(mx, fn, timing_inputs, kwargs, args.chain)
                                for name, fn in variants.items()}
            case["compiled_chain_parity"] = {}
            for name in variants:
                eager_chain, compiled_chain = chained[name](), compiled_chained[name]()
                parity = output_metrics(mx, np, eager_chain, compiled_chain, route)
                assert all(v["storage_bit_exact"] for v in parity.values()), (key, name, parity)
                case["compiled_chain_parity"][name] = parity
            case["compiled_dependent_chain_timing"] = time_variants(
                compiled_chained, rounds=args.rounds, inner=args.inner, seed=args.seed + 2)
            if route == "decode" and args.drift_steps > 0:
                result["drift"][key] = {"steps": args.drift_steps, "seed": args.seed + 100,
                    "vs_precise": drift_test(mx, np, variants, architecture, probe,
                                              args.seed + 100, args.drift_steps)}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"status": "case_complete", "case": key}), flush=True)
        assert all(n > 0 for n in calls.values()), calls
        result["calls"][route] = calls
    # Detect source edits concurrent with this experiment, rather than silently
    # attributing measurements to a source snapshot that no longer exists.
    for info in result["sources"].values():
        info["unchanged_at_end"] = hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"]
        assert info["unchanged_at_end"], f"Production source changed during benchmark: {info['path']}"
    assert result["harness_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert result["common_sha256"] == hashlib.sha256(Path(__file__).with_name("common.py").read_bytes()).hexdigest()
    result["peak_memory_bytes"] = mx.get_peak_memory()
    result["completed"] = True
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
