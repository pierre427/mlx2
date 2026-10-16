#!/usr/bin/env python3
"""Phase-1 feasibility probe: D3 edge trunk + readout on Core ML (CPU_AND_NE).

Never uses the GPU: MLX runs only in the reference child with
``mx.set_default_device(mx.cpu)`` before any MLX op, and Core ML uses only
``CPU_ONLY`` / ``CPU_AND_NE`` (conversion with ``CPU_ONLY`` and no model load).
The GPU lock is read, never taken.

Each stage runs in its own child process (one model per process) so MLX and
Core ML never share a process, the E5RT ANE-compile log is captured from the
child's native stderr, and peak resident memory is per stage:

    reference  token counts, MLX BF16 CPU reference (the served route's own
               D3Engine) and an FP32 numpy diagnostic reference
    build      per precision variant: one multifunction mlprogram (a function
               per bucket, shared weights), compiled, digested, compute plan
    parity     per variant x compute unit: every case in its own bucket and,
               for padding invariance, in the next larger bucket
    latency    per variant: interleaved ABBA CPU_ONLY / CPU_AND_NE warm medians
               and separate cold/warm load times

The evidence directory must not exist. Nothing is published or committed.

    PYTHONPATH=src ~/Desktop/mlx-uag/.venv/bin/python \\
        scripts/probe_coreml_d3.py --output <fresh evidence dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlx2.decisions.candidates import d3_coreml_build as B

UNITS = ("cpu_only", "cpu_and_ne")
SWAP_ABORT_PAGES = 100_000
STAGE_TIMEOUT_S = 900


# ------------------------------------------------------------------ utils ---

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def swapouts():
    text = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"Swapouts:\s+(\d+)", text)
    return int(match.group(1)) if match else None


def cosine(left, right):
    import numpy as np

    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def mark(text):
    """Stage marker on native stdout, so E5RT lines can be attributed by order."""
    sys.stderr.flush()
    sys.stdout.write(f"\n@@MARK {text}\n")
    sys.stdout.flush()


# ------------------------------------------------------------ child: ref ---

def stage_reference(args):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)  # before any MLX op: the GPU is never used
    import numpy as np

    from mlx2.decisions.candidates.d3 import D3Engine, prepare_question, render_prompt

    out = Path(args.output)
    started = time.perf_counter()
    engine = D3Engine(args.artifact)
    load_s = time.perf_counter() - started
    if mx.default_device() != mx.cpu:
        raise RuntimeError("reference must run on the MLX CPU device")
    weights = B.load_weights(args.artifact, verify_sha256=True)
    pad_id = B.pad_token_id(engine.tokenizer)
    cases, hidden_mlx, hidden_np = [], [], []
    for index, (state, question) in enumerate(B.PARITY_CASES):
        prepared, labels = prepare_question(question)
        tokens = render_prompt(engine.tokenizer, state, prepared, engine.codes,
                               max_length=engine.artifact["max_context"])
        tick = time.perf_counter()
        row = engine.model(mx.array([tokens]))[0, -1].astype(mx.float32)
        mx.eval(row)
        mlx_s = time.perf_counter() - tick
        row = np.array(row, dtype=np.float32)
        tick = time.perf_counter()
        row_np = B.numpy_forward(weights, weights.embed(tokens))[-1].astype(np.float32)
        numpy_s = time.perf_counter() - tick
        # The served law, recomputed exactly as D3Engine._predict_locked does.
        served = np.array(mx.softmax(mx.array(row) @ engine.head[: len(labels)].T
                                     / engine.temperature), dtype=np.float64)
        probs = B.decision_probabilities(row, weights.readout, len(labels), weights.temperature)
        probs_np = B.decision_probabilities(row_np, weights.readout, len(labels), weights.temperature)
        cases.append({
            "index": index, "type": question["type"], "labels": len(labels),
            "tokens": tokens, "token_count": len(tokens),
            "bucket": (B.bucket_for(len(tokens), tuple(args.buckets))
                       if len(tokens) <= args.buckets[-1] else None),
            "reference_probs": probs.tolist(), "reference_decision": int(probs.argmax()),
            "served_law_max_abs_diff": float(np.abs(served - probs).max()),
            "fp32_probs": probs_np.tolist(), "fp32_decision": int(probs_np.argmax()),
            "bf16_vs_fp32": {"hidden_cosine": cosine(row, row_np),
                             "max_abs_prob_diff": float(np.abs(probs - probs_np).max()),
                             "same_decision": int(probs.argmax()) == int(probs_np.argmax())},
            "seconds": {"mlx_cpu": round(mlx_s, 4), "numpy_fp32": round(numpy_s, 4)},
        })
        hidden_mlx.append(row)
        hidden_np.append(row_np)
    np.savez(out / "reference" / "hidden.npz", mlx_bf16_cpu=np.stack(hidden_mlx),
             numpy_fp32=np.stack(hidden_np))
    payload = {
        "reference": "mlx2 D3Engine (BF16 backbone, FP32 readout) on the MLX CPU device",
        "diagnostic": "independent FP32 numpy trunk (chunked GDN), same weights",
        "mlx_default_device": str(mx.default_device()),
        "engine_load_s": round(load_s, 3), "pad_token_id": pad_id,
        "weights_identity": weights.identity, "temperature": weights.temperature,
        "length_distribution": B.length_distribution([c["token_count"] for c in cases],
                                                     tuple(args.buckets)),
        "cases": cases, "peak_rss_bytes": peak_rss_bytes(),
    }
    engine.close()
    write_json(out / "reference" / "reference.json", payload)


# ---------------------------------------------------------- child: build ---

def stage_build(args):
    import shutil

    import coremltools as ct

    variant, out = args.variant, Path(args.output)
    root = out / "variants" / variant
    work = root / "packages"
    work.mkdir(parents=True)
    weights = B.load_weights(args.artifact, verify_sha256=False)  # verified by reference stage
    timings, packages = {}, {}
    for size in args.buckets:
        tick = time.perf_counter()
        program = B.build_program(weights, size)
        timings[f"build_seq{size}"] = round(time.perf_counter() - tick, 3)
        tick = time.perf_counter()
        packages[size] = B.convert_package(program, work / f"seq{size}.mlpackage", variant)
        timings[f"convert_seq{size}"] = round(time.perf_counter() - tick, 3)
        del program
    package_bytes = {str(size): B.tree_bytes(path) for size, path in packages.items()}
    tick = time.perf_counter()
    merged = B.merge_multifunction(packages, work / "d3.mlpackage")
    timings["merge"] = round(time.perf_counter() - tick, 3)
    tick = time.perf_counter()
    compiled = B.compile_package(merged, root / "d3.mlmodelc")
    timings["compile"] = round(time.perf_counter() - tick, 3)
    merged_bytes = B.tree_bytes(merged)
    functions = {B.function_name(size) for size in args.buckets}
    # MLComputePlan refuses multifunction models ("doesn't support the
    # multi-function description syntax"), so each bucket's identical
    # single-function package is compiled separately just for the plan.
    tick = time.perf_counter()
    plan, plan_error = {}, None
    try:
        for size, package in packages.items():
            single = B.compile_package(package, work / f"plan-seq{size}.mlmodelc")
            plan[B.function_name(size)] = B.compute_plan_counts(single, {"main"})["main"]
            shutil.rmtree(single)
    except Exception as exc:  # noqa: BLE001 - plan is optional evidence
        plan_error = repr(exc)
    timings["compute_plan"] = round(time.perf_counter() - tick, 3)
    shutil.rmtree(work)  # keep only the compiled multifunction model
    write_json(root / "build.json", {
        "variant": variant, "buckets": list(args.buckets), "functions": sorted(functions),
        "compiled": "d3.mlmodelc", "compiled_tree_sha256": B.tree_digest(compiled),
        "compiled_bytes": B.tree_bytes(compiled), "merged_package_bytes": merged_bytes,
        "per_bucket_package_bytes": package_bytes,
        "weights_shared_across_buckets": merged_bytes < 1.5 * max(package_bytes.values()),
        "coremltools": str(ct.__version__), "minimum_deployment_target": "macOS15",
        "compute_precision": variant, "convert_compute_units": "cpu_only",
        "gdn": {"chunk": B.DEFAULT_CHUNK, "inverse_block": B.INVERSE_BLOCK,
                "output_scale": B.GDN_OUTPUT_SCALE},
        "compute_plan": {"note": B.PLAN_NOTE, "error": plan_error, "per_function": plan,
                         "source": "per-bucket single-function compile of the merged packages"},
        "timings_s": timings, "peak_rss_bytes": peak_rss_bytes(),
    })


# --------------------------------------------------------- child: parity ---

def _reference(out):
    import numpy as np

    payload = json.loads((Path(out) / "reference" / "reference.json").read_text())
    hidden = np.load(Path(out) / "reference" / "hidden.npz")
    return payload, hidden["mlx_bf16_cpu"], hidden["numpy_fp32"]


def stage_parity(args):
    import numpy as np

    out, variant, unit = Path(args.output), args.variant, args.unit
    reference, ref_hidden, fp32_hidden = _reference(out)
    weights = B.load_weights(args.artifact, verify_sha256=False, include_trunk=False)
    pad_id = reference["pad_token_id"]
    compiled = out / "variants" / variant / "d3.mlmodelc"
    buckets = tuple(args.buckets)
    rows, loads = [], {}
    own_hidden = {}
    for size in buckets:
        name = B.function_name(size)
        mark(f"load {variant} {unit} {name}")
        tick = time.perf_counter()
        model = B.load_compiled(compiled, unit, name)
        loads[name] = round(time.perf_counter() - tick, 3)
        mark(f"predict {variant} {unit} {name}")
        for case in reference["cases"]:
            count, home = case["token_count"], case["bucket"]
            if home is None:
                continue
            larger = [b for b in buckets if b > home]
            if size != home and (not larger or size != larger[0]):
                continue
            inputs = B.graph_inputs(weights, case["tokens"], size, pad_id)
            tick = time.perf_counter()
            raw = model.predict(inputs)
            seconds = time.perf_counter() - tick
            labels = case["labels"]
            try:
                hidden, logits = B.checked_outputs(raw)
            except ValueError as error:  # nonfinite output is a recorded failure
                rows.append({"case": case["index"], "type": case["type"], "labels": labels,
                             "tokens": count, "bucket": size, "home_bucket": size == home,
                             "failure": str(error), "same_decision": False,
                             "predict_ms": round(seconds * 1000, 3)})
                continue
            probs = B.decision_probabilities(hidden, weights.readout, labels, weights.temperature)
            ref = np.asarray(case["reference_probs"])
            row = {
                "case": case["index"], "type": case["type"], "labels": labels, "tokens": count,
                "bucket": size, "home_bucket": size == home,
                "decision": int(probs.argmax()), "reference_decision": case["reference_decision"],
                "same_decision": int(probs.argmax()) == case["reference_decision"],
                "same_decision_fp32": int(probs.argmax()) == case["fp32_decision"],
                "max_abs_prob_diff": float(np.abs(probs - ref).max()),
                "abs_prob_diffs": np.abs(probs - ref).tolist(),
                "max_abs_prob_diff_fp32": float(np.abs(probs - np.asarray(case["fp32_probs"])).max()),
                "hidden_cosine": cosine(hidden, ref_hidden[case["index"]]),
                "hidden_cosine_fp32": cosine(hidden, fp32_hidden[case["index"]]),
                "graph_logits_vs_host_max_abs": float(
                    np.abs(logits - hidden @ weights.readout.T).max()),
                "predict_ms": round(seconds * 1000, 3),
            }
            if size == home:
                own_hidden[case["index"]] = hidden
            elif case["index"] in own_hidden:
                row["padding_cosine_vs_home_bucket"] = cosine(hidden, own_hidden[case["index"]])
            rows.append(row)
        del model
    write_json(out / "parity" / f"{variant}-{unit}.json", {
        "variant": variant, "compute_unit": unit, "load_s": loads, "rows": rows,
        "peak_rss_bytes": peak_rss_bytes(),
    })


# -------------------------------------------------------- child: latency ---

def stage_latency(args):
    import numpy as np

    out, variant = Path(args.output), args.variant
    reference, _, _ = _reference(out)
    weights = B.load_weights(args.artifact, verify_sha256=False, include_trunk=False)
    compiled = out / "variants" / variant / "d3.mlmodelc"
    result = {}
    for size in args.buckets:
        name = B.function_name(size)
        fitting = [c for c in reference["cases"] if c["bucket"] == size]
        if not fitting:
            continue
        case = max(fitting, key=lambda c: c["token_count"])
        inputs = B.graph_inputs(weights, case["tokens"], size, reference["pad_token_id"])
        models, row = {}, {"case": case["index"], "tokens": case["token_count"]}
        for unit in UNITS:
            mark(f"latency-load {variant} {unit} {name}")
            tick = time.perf_counter()
            first = B.load_compiled(compiled, unit, name)
            cold = time.perf_counter() - tick
            del first
            tick = time.perf_counter()
            models[unit] = B.load_compiled(compiled, unit, name)
            row[unit] = {"load_first_s": round(cold, 3),
                         "load_second_s": round(time.perf_counter() - tick, 3)}
        samples = {unit: [] for unit in UNITS}
        for index in range(args.warmup + args.iterations):
            order = UNITS if index % 2 == 0 else UNITS[::-1]
            for unit in order:
                tick = time.perf_counter()
                models[unit].predict(inputs)
                if index >= args.warmup:
                    samples[unit].append((time.perf_counter() - tick) * 1000)
        for unit in UNITS:
            values = samples[unit]
            row[unit].update({"warm_median_ms": round(statistics.median(values), 3),
                              "warm_min_ms": round(min(values), 3),
                              "warm_max_ms": round(max(values), 3),
                              "iterations": args.iterations, "warmup": args.warmup})
        for unit in UNITS:
            row[unit]["output_finite"] = bool(np.isfinite(
                np.asarray(models[unit].predict(inputs)["hidden"], np.float32)).all())
        row["schedule"] = "interleaved_abba"
        row["cpu_and_ne_speedup_vs_cpu_only"] = round(
            row["cpu_only"]["warm_median_ms"] / row["cpu_and_ne"]["warm_median_ms"], 3)
        models.clear()
        result[str(size)] = row
        del inputs
    write_json(out / "latency" / f"{variant}.json", {
        "variant": variant, "note": "exploratory warm latency, not a performance claim",
        "per_bucket": result, "peak_rss_bytes": peak_rss_bytes(),
    })


# ----------------------------------------------------------------- parent ---

def run_child(args, stage, label, **extra):
    out = Path(args.output)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
               "--output", str(out), "--artifact", str(args.artifact),
               "--buckets", *map(str, args.buckets),
               "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
    for key, value in extra.items():
        command += [f"--{key}", str(value)]
    before = swapouts()
    tick = time.perf_counter()
    with open(logs / f"{label}.stdout.txt", "wb") as stdout, \
            open(logs / f"{label}.stderr.txt", "wb") as stderr:
        try:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr,
                                       timeout=STAGE_TIMEOUT_S, check=False)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    after = swapouts()
    # E5RT writes its ANE-compile failure to native stdout, not stderr.
    text = "\n".join((logs / f"{label}.{stream}.txt").read_text(errors="replace")
                     for stream in ("stdout", "stderr"))
    record = {"label": label, "returncode": code,
              "seconds": round(time.perf_counter() - tick, 1),
              "swapouts_delta": None if before is None or after is None else after - before,
              "ane_compile_failed": B.ane_compile_failed(text),
              "ane_failures_after_marks": _attribute_failures(text)}
    print(json.dumps(record), flush=True)
    if record["swapouts_delta"] is not None and record["swapouts_delta"] > SWAP_ABORT_PAGES:
        raise SystemExit(f"swapouts rose by {record['swapouts_delta']} pages during {label}; abort")
    return record


def _attribute_failures(text):
    """Marks whose following stderr (until the next mark) has an ANE compile failure."""
    found, current = [], None
    for line in text.splitlines():
        if line.startswith("@@MARK "):
            current = line[len("@@MARK "):].strip()
        elif B.ane_compile_failed(line) and current not in found:
            found.append(current)
    return found


def summarize(out, variants, buckets, stages):
    import numpy as np

    reference = json.loads((out / "reference" / "reference.json").read_text())
    summary = {"variants": {}, "reference_bf16_vs_fp32": {}}
    bf = [c["bf16_vs_fp32"] for c in reference["cases"]]
    summary["reference_bf16_vs_fp32"] = {
        "cases": len(bf), "same_decisions": sum(r["same_decision"] for r in bf),
        "max_abs_prob_diff": max(r["max_abs_prob_diff"] for r in bf),
        "median_case_max_abs_prob_diff": statistics.median(r["max_abs_prob_diff"] for r in bf),
        "min_hidden_cosine": min(r["hidden_cosine"] for r in bf),
    }
    for variant in variants:
        entry = {}
        build = out / "variants" / variant / "build.json"
        if build.exists():
            data = json.loads(build.read_text())
            entry["build"] = {k: data[k] for k in ("compiled_bytes", "merged_package_bytes",
                                                    "per_bucket_package_bytes",
                                                    "weights_shared_across_buckets", "timings_s",
                                                    "peak_rss_bytes")}
            entry["compute_plan"] = {
                name: {k: row[k] for k in ("neural_engine", "cpu", "unknown",
                                           "neural_engine_compute_fraction", "compute_ops",
                                           "gdn_core_ops", "gdn_inverse_ops")}
                for name, row in data["compute_plan"]["per_function"].items()}
        for unit in UNITS:
            path = out / "parity" / f"{variant}-{unit}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            failures = [r for r in data["rows"] if "failure" in r]
            home = [r for r in data["rows"] if r["home_bucket"] and "failure" not in r]
            padded = [r for r in data["rows"] if not r["home_bucket"] and "failure" not in r]
            nonfinite = {"home": sorted(r["case"] for r in failures if r["home_bucket"]),
                         "padded": sorted(r["case"] for r in failures if not r["home_bucket"])}
            if not home:
                entry[unit] = {"cases": 0, "nonfinite_cases": nonfinite}
                continue
            diffs = [d for r in home for d in r["abs_prob_diffs"]]
            per_bucket = {}
            for size in buckets:
                sel = [r for r in home if r["bucket"] == size]
                if sel:
                    per_bucket[str(size)] = {
                        "cases": len(sel), "same_decisions": sum(r["same_decision"] for r in sel),
                        "max_abs_prob_diff": max(r["max_abs_prob_diff"] for r in sel),
                        "min_hidden_cosine": min(r["hidden_cosine"] for r in sel)}
            entry[unit] = {
                "cases": len(home) + len(nonfinite["home"]),
                "finite_cases": len(home), "nonfinite_cases": nonfinite,
                "same_decisions": sum(r["same_decision"] for r in home),
                "same_decisions_vs_fp32": sum(r["same_decision_fp32"] for r in home),
                "flipped_cases": [r["case"] for r in home if not r["same_decision"]],
                "max_abs_prob_diff": max(r["max_abs_prob_diff"] for r in home),
                "median_abs_prob_diff": float(np.median(diffs)),
                "median_case_max_abs_prob_diff": statistics.median(r["max_abs_prob_diff"] for r in home),
                "max_abs_prob_diff_vs_fp32": max(r["max_abs_prob_diff_fp32"] for r in home),
                "min_hidden_cosine": min(r["hidden_cosine"] for r in home),
                "median_hidden_cosine": statistics.median(r["hidden_cosine"] for r in home),
                "min_hidden_cosine_vs_fp32": min(r["hidden_cosine_fp32"] for r in home),
                "padding_rows": len(padded),
                "padding_same_decisions": sum(r["same_decision"] for r in padded),
                "padding_min_cosine_vs_home": min(
                    (r["padding_cosine_vs_home_bucket"] for r in padded
                     if "padding_cosine_vs_home_bucket" in r), default=None),
                "per_bucket": per_bucket, "load_s": data["load_s"],
                "peak_rss_bytes": data["peak_rss_bytes"],
            }
        path = out / "latency" / f"{variant}.json"
        if path.exists():
            entry["latency"] = json.loads(path.read_text())["per_bucket"]
        entry["ane_compile_failures"] = {
            label: rec["ane_failures_after_marks"] for label, rec in stages.items()
            if variant in label and rec["ane_compile_failed"]}
        summary["variants"][variant] = entry
    return summary


def parent(args):
    out = Path(args.output).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise SystemExit(f"refusing existing evidence path {out}")
    out.mkdir(parents=True)
    (out / "reference").mkdir()
    args.output = str(out)
    lock = B.gpu_lock_state()
    environment = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_lock_read_not_taken": lock, "gpu_used": False,
        "compute_units": list(UNITS), "macos": platform.mac_ver()[0], "machine": platform.machine(),
        "python": sys.executable, "artifact": str(args.artifact),
        "source_revision": B.SOURCE_REVISION, "builder_sha256": B._sha256_file(B.__file__),
        "probe_sha256": B._sha256_file(__file__), "swapouts_start": swapouts(),
        "buckets": list(args.buckets), "variants": list(args.variants),
    }
    write_json(out / "environment.json", environment)
    stages = {}
    stages["reference"] = run_child(args, "reference", "reference")
    if stages["reference"]["returncode"] != 0:
        raise SystemExit("reference stage failed; see logs")
    # Variant by variant (build, parity, latency) so an ANE-compile failure
    # or hang in a later variant cannot cost the evidence of an earlier one.
    latency_allowed = not args.skip_latency
    environment["gpu_lock_before_latency"] = {}
    for variant in args.variants:
        label = f"build-{variant}"
        stages[label] = run_child(args, "build", label, variant=variant)
        if stages[label]["returncode"] != 0:
            continue
        for unit in UNITS:
            label = f"parity-{variant}-{unit}"
            stages[label] = run_child(args, "parity", label, variant=variant, unit=unit)
        lock = B.gpu_lock_state()  # re-read right before each latency stage
        environment["gpu_lock_before_latency"][variant] = lock
        if latency_allowed and lock["latency_allowed"]:
            label = f"latency-{variant}"
            stages[label] = run_child(args, "latency", label, variant=variant)
    receipt = {
        "schema": "mlx2.d3-coreml-feasibility.v1",
        "qualified": False, "neural_engine_observed": False, "gpu": "excluded",
        "note": "phase-1 feasibility evidence; compute plans are anticipated placement",
        "environment": environment, "stages": stages,
        "latency_run": sorted(label for label in stages if label.startswith("latency-")),
        "latency_skipped_reason": None if latency_allowed else "--skip-latency",
        "summary": summarize(out, args.variants, args.buckets, stages),
        "swapouts_end": swapouts(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt["summary"], indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", default=str(B.DEFAULT_ARTIFACT))
    parser.add_argument("--buckets", type=int, nargs="+", default=list(B.DEFAULT_BUCKETS))
    # fp16 last: its ANE compile is the one expected to fail.
    parser.add_argument("--variants", nargs="+", default=["fp32_gdn", "fp32_inverse", "fp16"],
                        choices=B.PRECISION_VARIANTS)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--stage", choices=("reference", "build", "parity", "latency"))
    parser.add_argument("--variant", choices=B.PRECISION_VARIANTS)
    parser.add_argument("--unit", choices=UNITS)
    args = parser.parse_args(argv)
    args.buckets = list(B.validate_buckets(args.buckets))
    if not 1 <= args.iterations <= 200 or not 0 <= args.warmup <= 50:
        parser.error("iterations must be in [1, 200] and warmup in [0, 50]")
    stages = {"reference": stage_reference, "build": stage_build,
              "parity": stage_parity, "latency": stage_latency}
    if args.stage is None:
        parent(args)
    else:
        stages[args.stage](args)


if __name__ == "__main__":
    main()
