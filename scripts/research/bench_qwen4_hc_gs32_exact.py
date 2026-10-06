#!/usr/bin/env python3
"""Deferred Metal gate for mlx2's exact Qwen4 HC gs32 candidate.

``--describe`` is CPU/static-only and is safe to run without importing MLX.
``--run`` is deliberately harder: it requires a metadata-eligible real gs32
artifact, an explicit norm convention, both matching GPU lock receipts and an
ownership acknowledgement.  The candidate is installed only in this process;
production source and defaults remain unchanged.

This adopts oMLX #4245's group-addressing generalization, not #4248's faster
FP32 epilogue law.  Passing this component gate would still not qualify or
select a serving route; model-level ordinary/MTP decode gates remain required.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Direct ``python scripts/research/...py`` execution starts with this file's
# directory on sys.path, not the repository root that owns the scripts
# namespace package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.qwen4_hc_gs32_exact import (
    describe,
    exact_gs32_sources,
    inspect_artifact,
    source_constants,
)

SOURCE = REPO_ROOT / "src/mlx2/runtime/models/qwen4_hc_decode.py"
LOCKS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)


def require_gpu_lease(paths=LOCKS) -> dict:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing GPU ownership receipts: {missing}")
    receipts = [json.loads(path.read_text()) for path in paths]
    leases = [item.get("lease_id") for item in receipts]
    if not leases[0] or leases[0] != leases[1]:
        raise RuntimeError("GPU receipts do not name one matching lease_id")
    return {"lease_id": leases[0], "receipts": [str(path) for path in paths]}


def selected_modules(report: dict, limit: int) -> list[str]:
    eligible = [
        item["path"]
        for item in report["modules"]
        if item["metadata_eligible"]
        and "block_inject_weight" in item["projections"]
    ]
    if len(eligible) <= limit:
        return eligible
    # Evenly cover early/middle/late trunk geometry without loading a model.
    indexes = sorted({round(i * (len(eligible) - 1) / (limit - 1)) for i in range(limit)})
    return [eligible[index] for index in indexes]


def plan_receipt(artifact: Path, max_modules: int) -> dict:
    static = describe(SOURCE, [artifact])
    inspected = static["artifacts"][0]
    return {
        **static,
        "selected_modules": selected_modules(inspected, max_modules),
        "future_run": {
            "imports_mlx": True,
            "loads_model_tensors": True,
            "requires_real_gs32_artifact": True,
            "requires_explicit_norm_convention": True,
            "requires_matching_gpu_locks": [str(path) for path in LOCKS],
            "exactness": "candidate outputs must be raw-bit equal to module._composed",
            "performance": "counterbalanced component medians only",
            "model_qualification_remaining": [
                "ordinary decode tokens and logits",
                "MTP on/off acceptance and token parity",
                "prefill reference and route receipts",
                "controlled end-to-end performance",
            ],
        },
    }


def _install_process_candidate(hcd) -> None:
    values = source_constants(SOURCE)
    candidate = exact_gs32_sources(values)
    hcd.HEADER = candidate["HEADER"]
    hcd.NORM_DOWN_SOURCE = candidate["NORM_DOWN_SOURCE"]
    hcd.UP_MIX_SOURCE = candidate["UP_MIX_SOURCE"]
    hcd.GROUP_SIZE = 32
    original = hcd._build_plan

    def build_plan(module, law=0):
        plan = original(module, law)
        plan["a_template"] = [*plan["a_template"], ("GS", 32)]
        plan["b_template"] = [*plan["b_template"], ("GS", 32)]
        return plan

    hcd._build_plan = build_plan
    hcd._KERNELS.clear()
    hcd._COMPILED.clear()
    hcd.set_hc_decode_enabled(False)


def _bits_equal(mx, left, right) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and mx.array_equal(left.view(mx.uint16), right.view(mx.uint16)).item()
    )


def run(args, plan: dict, lease: dict) -> dict:  # pragma: no cover - Metal only
    from mlx2.adapters.flash_next import configure_environment

    configure_environment(args.artifact)
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.models import qwen4_exp as qwen4
    from mlx2.runtime.models import qwen4_hc_decode as hcd

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("Metal GPU is unavailable")
    inspected = inspect_artifact(args.artifact)
    if inspected["artifact_gate"] != "pass":
        raise RuntimeError("artifact metadata gate is blocked")
    _install_process_candidate(hcd)
    if hcd.hc_decode_enabled():
        raise AssertionError("candidate must remain default-off during direct component gate")

    config = json.loads((args.artifact / "config.json").read_text())
    text = config.get("text_config", config)
    model_args = SimpleNamespace(
        hc_count=text["hc_count"],
        hidden_size=text["hidden_size"],
        hc_lowrank=text["hc_lowrank"],
        rms_norm_eps=text["rms_norm_eps"],
    )
    index = json.loads(
        (args.artifact / "model.safetensors.index.json").read_text()
    )["weight_map"]
    module_geometry = {item["path"]: item for item in inspected["modules"]}
    shard_cache = {}

    def tensors(prefix):
        out = {}
        for key, shard in index.items():
            if not key.startswith(prefix + "."):
                continue
            if shard not in shard_cache:
                shard_cache[shard] = mx.load(str(args.artifact / shard))
            out[key[len(prefix) + 1 :]] = shard_cache[shard][key]
        return out

    def build(prefix):
        weights = tensors(prefix)
        geometry = module_geometry[prefix]["projections"]
        module = qwen4.GatedResidual(model_args, use_combine=True)

        def quantize(path, _layer):
            spec = geometry.get(path)
            if not spec or not spec["quantized"]:
                return False
            return {
                "group_size": spec["group_size"],
                "bits": spec["bits"],
                "mode": spec["mode"],
            }

        nn.quantize(module, class_predicate=quantize)
        if args.norm_convention == "raw":
            # mlx2 GroupRMSNorm stores the direct gamma. The normal loader's
            # norm-repair adds one to raw zero-centred tensors, preserving dtype.
            norm = weights["hc_norm.weight"]
            weights["hc_norm.weight"] = norm + mx.array(1, dtype=norm.dtype)
        module.load_weights(list(weights.items()), strict=True)
        module.eval()
        mx.eval(module.parameters())
        reason = hcd.static_admission(module)
        if reason is not None:
            raise AssertionError(f"{prefix}: native admission declined: {reason}")
        return module

    chosen = plan["selected_modules"]
    if not chosen:
        raise RuntimeError("no combined HC module survived the metadata gate")
    modules = {name: build(name) for name in chosen}
    width = model_args.hc_count * model_args.hidden_size
    exact = []
    samples = {}
    for module_index, (name, module) in enumerate(modules.items()):
        for case in range(args.cases):
            mx.random.seed(424500 + module_index * 100 + case)
            rows = (1, 2, 3, 8)[case % 4]
            sample = mx.random.normal((1, rows, width)).astype(mx.bfloat16)
            expected = module._composed(sample, False, False)
            mixed, inject = hcd.hc_decode_launch(module, sample.reshape(rows, width))
            actual = (
                mixed.reshape(1, rows, model_args.hidden_size),
                sample,
                inject.reshape(1, rows, model_args.hc_count),
            )
            mx.eval(*expected, *actual)
            equal = all(_bits_equal(mx, left, right) for left, right in zip(expected, actual))
            exact.append({"module": name, "rows": rows, "equal": equal})
            if rows == 1:
                samples[name] = sample
    if not exact or not all(item["equal"] for item in exact):
        raise AssertionError("candidate differs from the composed MLX reference")

    timings = {"composed": [], "candidate": []}
    first_name = next(iter(modules))
    module, sample = modules[first_name], samples[first_name]

    def composed():
        return module._composed(sample, False, False)

    def candidate():
        return hcd.hc_decode_launch(module, sample.reshape(1, width))[:2]

    for repetition in range(args.reps + 2):
        order = ("composed", "candidate") if repetition % 2 == 0 else ("candidate", "composed")
        for arm in order:
            fn = composed if arm == "composed" else candidate
            mx.synchronize()
            started = time.perf_counter_ns()
            for _ in range(args.iterations):
                result = fn()
                mx.eval(*result)
            elapsed = (time.perf_counter_ns() - started) / args.iterations / 1e3
            if repetition >= 2:
                timings[arm].append(elapsed)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "schema": "mlx2.qwen4-hc-gs32-exact-metal.v1",
        "artifact": str(args.artifact.resolve()),
        "artifact_report": inspected,
        "source_receipt": plan["source_hashes"],
        "upstream": plan["upstream"],
        "law": plan["law"],
        "gpu_lease": lease,
        "mlx": mx.__version__,
        "norm_convention": args.norm_convention,
        "modules": chosen,
        "exact_cases": exact,
        "all_bit_identical": True,
        "timings_us": timings,
        "median_us": medians,
        "candidate_speedup": medians["composed"] / medians["candidate"],
        "implemented": "process-local candidate only",
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "remaining": plan["future_run"]["model_qualification_remaining"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--max-modules", type=int, default=6)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=48)
    parser.add_argument("--norm-convention", choices=("raw", "converted"))
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.describe == args.run:
        parser.error("choose exactly one of --describe or --run")
    plan = plan_receipt(args.artifact, args.max_modules)
    if args.describe:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        parser.error("--run requires --i-own-the-gpu")
    if args.norm_convention is None:
        parser.error("--run requires --norm-convention raw|converted")
    if args.out is None:
        parser.error("--run requires --out")
    artifact = plan["artifacts"][0]
    if artifact["artifact_gate"] != "pass":
        parser.error("artifact is not complete, homogeneous affine gs32 HC")
    lease = require_gpu_lease()
    receipt = run(args, plan, lease)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
