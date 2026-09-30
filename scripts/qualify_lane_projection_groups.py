#!/usr/bin/env python3
"""Qualify adapter-declared lane projection groups without selecting them.

Two deliberately separate modes share one receipt schema:

* ``cpu`` replaces the Metal lane kernel with a dequantize-and-matmul
  reference and exercises tiny instances of the real Muse, HiLS and Xing
  attention classes.  Xing uses the checked-in tiny real-artifact fixture.
  This is an implementation/receipt smoke, never GPU qualification.
* ``metal`` uses the real lane kernel at the exact shipped projection shapes.
  It requires both lab GPU lock receipts before selecting the GPU.  A passing
  receipt is qualification evidence only; this script never changes policy,
  route qualification, or selection.

Every case records declared-versus-default-group parity, row invariance,
engagement counters, counterbalanced timing, and MLX active/peak memory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime import lane
from mlx2.runtime.lane import installer
from mlx2.runtime.models import muse_glimmer, olmo_hils, xing4_0

GPU_LOCK_RECEIPTS = (
    Path("/tmp/gpu.lock/owner.json"),
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
)
XING_FIXTURE = ROOT / "tests" / "fixtures" / "xing4_0_tiny"


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    profile: str
    weight_format: str
    expected_k: int
    expected_widths: tuple[int, ...]
    source: str


TINY_CASES = (
    Case("muse-q4-tiny", "muse", "tiny", "q4", 128, (128, 64, 64, 128),
         "generated tiny instance of shipped adapter class"),
    Case("hils-q6-tiny", "hils", "tiny", "q6", 128, (128, 128, 128, 64),
         "generated tiny instance of shipped adapter class"),
    Case("xing-bf16-tiny", "xing", "tiny", "bf16", 64, (32, 40),
         "tests/fixtures/xing4_0_tiny real weights"),
)

SHIPPED_CASES = (
    Case("muse-q4-shipped", "muse", "shipped", "q4", 6656,
         (4096, 256, 256, 4096), "exact shipped projection geometry"),
    Case("muse-q8-shipped", "muse", "shipped", "q8", 6656,
         (4096, 256, 256, 4096), "exact shipped projection geometry"),
    Case("muse-bf16-shipped", "muse", "shipped", "bf16", 6656,
         (4096, 256, 256, 4096), "exact shipped projection geometry"),
    Case("hils-q6-shipped", "hils", "shipped", "q6", 4096,
         (4096, 4096, 4096, 256), "exact shipped projection geometry"),
    Case("hils-bf16-shipped", "hils", "shipped", "bf16", 4096,
         (4096, 4096, 4096, 256), "exact shipped projection geometry"),
    Case("xing-q6-shipped", "xing", "shipped", "q6", 3584,
         (768, 576), "exact shipped projection geometry"),
    Case("xing-bf16-shipped", "xing", "shipped", "bf16", 3584,
         (768, 576), "exact shipped projection geometry"),
)


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def require_gpu_lock_receipts(paths=GPU_LOCK_RECEIPTS) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise RuntimeError(
            "Metal qualification requires CPG ownership and both GPU lock "
            f"receipts; missing {missing}"
        )


def _reference_lane_matmul(x, lw):
    if lw.bits == installer.UNQUANTIZED_BITS:
        y = x @ lw.weight.T
    else:
        weight = mx.dequantize(
            lw.weight,
            lw.scale_bias[..., 0].T,
            lw.scale_bias[..., 1].T,
            group_size=lw.group_size,
            bits=lw.bits,
        )
        y = (x.astype(mx.float32) @ weight.T.astype(mx.float32)).astype(x.dtype)
    return y if lw.bias is None else y + lw.bias


@contextmanager
def cpu_reference_backend():
    """Use installer semantics on CPU while replacing only its Metal kernel."""
    old_available = installer.available
    old_matmul = installer.lane_matmul
    installer.available = lambda: True
    installer.lane_matmul = _reference_lane_matmul
    try:
        yield
    finally:
        installer.available = old_available
        installer.lane_matmul = old_matmul
        installer.STATS.clear()
        lane.set_enabled(True, grouping=True)


def _muse(profile: str):
    if profile == "shipped":
        args = muse_glimmer.ModelArgs()
    else:
        args = muse_glimmer.ModelArgs(
            hidden_size=128, num_hidden_layers=1, intermediate_size=128,
            num_attention_heads=2, num_key_value_heads=1, head_dim=64,
            vocab_size=64, sliding_window=16,
        )
    return muse_glimmer.Attention(args, 0), muse_glimmer.lane_projection_groups()[0]


def _hils(profile: str):
    hidden = 4096 if profile == "shipped" else 128
    heads = 32 if profile == "shipped" else 2
    rank = 256 if profile == "shipped" else 64
    args = olmo_hils.ModelArgs(
        model_type="olmo_hils", hidden_size=hidden, num_hidden_layers=4,
        intermediate_size=11008 if profile == "shipped" else 128,
        num_attention_heads=heads, rms_norm_eps=1e-6, vocab_size=64,
        max_position_embeddings=8192 if profile == "shipped" else 512,
        sliding_window=512 if profile == "shipped" else 16,
        rope_theta=10000.0, chunk_size=64 if profile == "shipped" else 8,
        hils_topk=32 if profile == "shipped" else 2, lmk_q_lora_dim=rank,
    )
    return olmo_hils.HiLSAttention(args, 3), olmo_hils.lane_projection_groups()[0]


def _xing(profile: str):
    if profile == "tiny":
        config = json.loads((XING_FIXTURE / "config.json").read_text())
        model = xing4_0.Model(xing4_0.ModelArgs.from_dict(config))
        weights = model.sanitize(mx.load(str(XING_FIXTURE / "weights.safetensors")))
        model.load_weights(list(weights.items()), strict=True)
        parent = model.model.layers[0].self_attn
    else:
        # The shipped Xing artifact uses a 768-wide query LoRA.  ModelArgs'
        # upstream-compatible default is intentionally not the receipt shape.
        parent = xing4_0.Xing4_0Attention(xing4_0.ModelArgs(q_lora_rank=768))
    spec = next(
        item for item in xing4_0.lane_projection_groups()
        if item.name == "xing-mla-qa-kva"
    )
    return parent, spec


def _member(parent, path: str):
    value = parent
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def _members(parent, spec):
    return tuple(_member(parent, path) for path in spec.members)


def _format_parent(parent, spec, weight_format: str) -> None:
    parent.set_dtype(mx.bfloat16)
    if weight_format == "bf16":
        return
    bits = int(weight_format[1:])
    # Quantizing the real parent preserves the adapter's actual module tree;
    # only the declared members are used by this harness.
    nn.quantize(parent, group_size=64, bits=bits)
    if any(not isinstance(member, nn.QuantizedLinear) for member in _members(parent, spec)):
        raise RuntimeError(f"{weight_format} did not quantize every declared member")


def build_case(case: Case):
    parent, spec = {"muse": _muse, "hils": _hils, "xing": _xing}[case.family](case.profile)
    _format_parent(parent, spec, case.weight_format)
    members = _members(parent, spec)
    prepared = tuple(installer.prepare(member) for member in members)
    k = prepared[0].k
    widths = tuple(item.n for item in prepared)
    if k != case.expected_k or widths != case.expected_widths:
        raise RuntimeError(
            f"{case.name} shape drift: got K={k}, widths={widths}; "
            f"expected K={case.expected_k}, widths={case.expected_widths}"
        )
    mx.eval(parent.parameters())
    return parent, spec


def _project(parent, spec, x):
    return tuple(member(x) for member in _members(parent, spec))


def _eval_outputs(outputs) -> None:
    mx.eval(*outputs)


def _parity(actual, expected, *, atol: float, rtol: float) -> dict:
    max_abs = 0.0
    mean_abs = 0.0
    bitwise = True
    close = True
    finite = True
    for got, want in zip(actual, expected, strict=True):
        delta = mx.abs(got.astype(mx.float32) - want.astype(mx.float32))
        max_abs = max(max_abs, float(mx.max(delta).item()))
        mean_abs += float(mx.mean(delta).item())
        bitwise = bitwise and bool(mx.array_equal(got, want).item())
        close = close and bool(mx.all(delta <= atol + rtol * mx.abs(want)).item())
        finite = finite and bool(mx.all(mx.isfinite(got)).item())
    return {
        "bitwise_equal": bitwise,
        "allclose": close,
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs / len(actual),
        "atol": atol,
        "rtol": rtol,
    }


def _fp32_reference(parent, spec, x):
    outputs = []
    for member in _members(parent, spec):
        if isinstance(member, nn.QuantizedLinear):
            weight = mx.dequantize(
                member.weight, member.scales, member.biases,
                group_size=member.group_size, bits=member.bits,
            )
        else:
            weight = member.weight
        value = x.astype(mx.float32) @ weight.T.astype(mx.float32)
        bias = getattr(member, "bias", None)
        if bias is not None:
            value = value + bias.astype(mx.float32)
        outputs.append(value)
    result = tuple(outputs)
    _eval_outputs(result)
    return result


def _accuracy(actual, default, stock, reference) -> dict:
    def errors(values):
        deltas = [mx.abs(value.astype(mx.float32) - ref)
                  for value, ref in zip(values, reference, strict=True)]
        mx.eval(*deltas)
        return {
            "max_abs": max(float(mx.max(delta).item()) for delta in deltas),
            "mean_abs": sum(float(mx.mean(delta).item()) for delta in deltas) / len(deltas),
        }

    actual_error = errors(actual)
    default_error = errors(default)
    stock_error = errors(stock)
    scale = max(float(mx.max(mx.abs(ref)).item()) for ref in reference)
    # Same generous BF16 envelope as scripts/lane_matmul_gate.py: the lane
    # may be up to twice stock MLX's error plus one BF16 ulp at output scale.
    limit = 2.0 * stock_error["max_abs"] + scale * 2.0 ** -7
    return {
        "declared": actual_error,
        "default_grouped": default_error,
        "stock_mlx": stock_error,
        "reference_max_abs": scale,
        "declared_max_allowed": limit,
        "passed": actual_error["max_abs"] <= limit,
    }


def _row_invariance(parent, spec, x, rows) -> dict:
    failures = []
    for count in rows:
        together = _project(parent, spec, x[:count])
        _eval_outputs(together)
        per_member = [[] for _ in together]
        for row in range(count):
            alone = _project(parent, spec, x[row : row + 1])
            _eval_outputs(alone)
            for index, value in enumerate(alone):
                per_member[index].append(value)
        alone = tuple(mx.concatenate(values, axis=0) for values in per_member)
        _eval_outputs(alone)
        if not all(bool(mx.array_equal(a, b).item()) for a, b in zip(together, alone, strict=True)):
            failures.append(count)
    return {"rows": list(rows), "failure_rows": failures, "passed": not failures}


def _fresh_inputs(base, count=4):
    # A declared group keys reuse on object identity.  Timing the same object
    # repeatedly would measure the cached group output instead of a launch.
    values = tuple(base + mx.array(index * 0.0, dtype=base.dtype) for index in range(count))
    mx.eval(*values)
    if len({id(value) for value in values}) != len(values):
        raise RuntimeError("timing inputs must have distinct object identities")
    return values


def _timed(parent, spec, inputs, *, batch: int) -> float:
    start = time.perf_counter_ns()
    for index in range(batch):
        _eval_outputs(_project(parent, spec, inputs[index % len(inputs)]))
    return (time.perf_counter_ns() - start) / 1.0e6 / batch


def _timing(parent, spec, inputs, *, warmups: int, repeats: int, batch: int) -> dict:
    samples = {"default": [], "declared": []}
    for arm in ("default", "declared"):
        lane.install(parent, min_rows=1, declared=(spec,) if arm == "declared" else ())
        for _ in range(warmups):
            _timed(parent, spec, inputs, batch=batch)
    for repeat in range(repeats):
        order = ("default", "declared") if repeat % 2 == 0 else ("declared", "default")
        for arm in order:
            lane.install(parent, min_rows=1, declared=(spec,) if arm == "declared" else ())
            samples[arm].append(_timed(parent, spec, inputs, batch=batch))
    default = statistics.median(samples["default"])
    declared = statistics.median(samples["declared"])
    return {
        "method": "counterbalanced blocks; install excluded; distinct input objects",
        "warmups_per_arm": warmups,
        "repeats": repeats,
        "batch": batch,
        "default_ms": default,
        "declared_ms": declared,
        "speedup": default / declared if declared else None,
        "samples_ms": samples,
    }


def _memory_install(parent, spec) -> tuple[dict, dict]:
    lane.install(parent, min_rows=1)
    mx.reset_peak_memory()
    before = int(mx.get_active_memory())
    receipt = lane.install(parent, min_rows=1, declared=(spec,))
    after = int(mx.get_active_memory())
    peak = int(mx.get_peak_memory())
    return receipt, {
        "active_before_declared_install_bytes": before,
        "active_after_declared_install_bytes": after,
        "active_delta_bytes": after - before,
        "peak_bytes": peak,
        "peak_delta_from_before_bytes": max(0, peak - before),
        "scope": "declared regrouping only; model construction excluded",
    }


def run_case(case: Case, *, rows, timing_rows: int, warmups: int, repeats: int,
             batch: int, atol: float, rtol: float, mode: str) -> dict:
    parent, spec = build_case(case)
    max_rows = max(max(rows), timing_rows)
    key = mx.random.key(sum(ord(ch) for ch in case.name))
    x = mx.random.normal((max_rows, case.expected_k), key=key).astype(mx.bfloat16)
    inputs = _fresh_inputs(x[:timing_rows])

    stock = _project(parent, spec, x)
    reference = _fp32_reference(parent, spec, x)
    _eval_outputs(stock)
    default_receipt = lane.install(parent, min_rows=1)
    baseline = _project(parent, spec, x)
    _eval_outputs(baseline)
    declared_receipt = lane.install(parent, min_rows=1, declared=(spec,))
    declared = _project(parent, spec, x)
    _eval_outputs(declared)
    declared_vs_default = _parity(declared, baseline, atol=atol, rtol=rtol)
    declared_vs_stock = _parity(declared, stock, atol=atol, rtol=rtol)
    accuracy = _accuracy(declared, baseline, stock, reference)
    parity = {
        "declared_vs_default_grouped": declared_vs_default,
        "declared_vs_stock_mlx": declared_vs_stock,
        "accuracy_vs_fp32": accuracy,
        "passed": bool(declared_vs_default["allclose"]
                       and declared_vs_default["finite"] and accuracy["passed"]),
    }
    invariant = _row_invariance(parent, spec, x, rows)

    timing = _timing(
        parent, spec, inputs, warmups=warmups, repeats=repeats, batch=batch,
    )
    timing["performance_evidence"] = mode == "metal"
    timing["interpretation"] = (
        "real Metal kernel timing" if mode == "metal"
        else "CPU reference-backend harness timing; not performance evidence"
    )
    declared_receipt, memory = _memory_install(parent, spec)
    installer.STATS.clear()
    engagement_input = x[:timing_rows] + mx.array(0.0, dtype=x.dtype)
    _eval_outputs(_project(parent, spec, engagement_input))
    counters = lane.stats()
    observed = {
        "launches": counters.get(f"declared_launches:{spec.name}", 0),
        "reuses": counters.get(f"declared_reuses:{spec.name}", 0),
        "partial": counters.get(f"declared_partial:{spec.name}", 0),
    }
    expected = {"launches": 1, "reuses": len(spec.members) - 1, "partial": 0}
    formed = declared_receipt["declared_groups"][spec.name]["formed"]
    engagement = {
        "expected": expected,
        "observed": observed,
        "formed": formed,
        "passed": observed == expected and sum(formed.values()) == 1,
    }
    passed = bool(parity["passed"] and invariant["passed"] and engagement["passed"])
    lane.uninstall(parent)
    return {
        "case": asdict(case),
        "actual_shape": {"k": case.expected_k, "widths": list(case.expected_widths),
                         "stacked_n": sum(case.expected_widths)},
        "default_law_id": default_receipt["law_id"],
        "declared_law_id": declared_receipt["law_id"],
        "default_install_receipt": default_receipt,
        "declared_install_receipt": declared_receipt,
        "parity": parity,
        "row_invariance": invariant,
        "engagement": engagement,
        "counters": counters,
        "timing": timing,
        "memory": memory,
        "passed": passed,
    }


def run_suite(*, mode: str, cases, rows=(1, 2, 4, 8, 16, 32), timing_rows=16,
              warmups=2, repeats=7, batch=4, atol=2e-2, rtol=2e-2) -> dict:
    records = []
    context = cpu_reference_backend() if mode == "cpu" else nullcontext()
    with context:
        for case in cases:
            record = run_case(
                case, rows=rows, timing_rows=timing_rows, warmups=warmups,
                repeats=repeats, batch=batch, atol=atol, rtol=rtol, mode=mode,
            )
            records.append(record)
            print(json.dumps({
                "case": case.name, "passed": record["passed"],
                "speedup": record["timing"]["speedup"],
                "engagement": record["engagement"]["observed"],
            }), flush=True)
    passed = sum(bool(record["passed"]) for record in records)
    expected = SHIPPED_CASES if mode == "metal" else TINY_CASES
    complete_matrix = {case.name for case in cases} == {case.name for case in expected}
    all_passed = passed == len(records)
    candidate = mode == "metal" and complete_matrix and all_passed
    return {
        "schema": "mlx2.lane-declared-projection-groups-qualification.v1",
        "source_revision": _git_revision(),
        "execution": {
            "mode": mode,
            "device": str(mx.default_device()),
            "mlx_version": mx.__version__,
            "rows": list(rows),
            "timing_rows": timing_rows,
            "warmups": warmups,
            "repeats": repeats,
            "batch": batch,
        },
        "state": {
            "implemented": True,
            "tested_cases_passed": all_passed,
            "tested_case_set_complete": complete_matrix,
            "cpu_smoke_passed": mode == "cpu" and complete_matrix and all_passed,
            "metal_qualification_candidate_passed": candidate,
            "qualified": False,
            "selected": False,
            "observed_used": mode == "metal" and bool(records),
            "note": "A passing receipt does not modify route qualification or selection.",
        },
        "summary": {
            "cases": len(records), "passed": passed,
            "beneficial_timing_observations": sum(
                record["timing"]["speedup"] > 1.0 for record in records
            ),
        },
        "cases": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--profile", choices=("tiny", "shipped"), default=None)
    parser.add_argument("--case", action="append", default=[], help="case name; repeatable")
    parser.add_argument("--rows", default="1,2,4,8,16,32")
    parser.add_argument("--timing-rows", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    profile = args.profile or ("tiny" if args.mode == "cpu" else "shipped")
    if args.mode == "metal" and profile != "shipped":
        raise SystemExit("Metal qualification requires --profile shipped")
    if args.mode == "metal":
        try:
            require_gpu_lock_receipts()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        mx.set_default_device(mx.gpu)
        if not installer.available():
            raise SystemExit("lane matmul is unavailable on the selected Metal device")
    else:
        mx.set_default_device(mx.cpu)
    rows = tuple(int(value) for value in args.rows.split(",") if value)
    if not rows or min(rows) < 1 or max(rows) > installer.MAX_ROWS:
        raise SystemExit(f"rows must be within 1..{installer.MAX_ROWS}")
    if args.timing_rows < 1 or args.timing_rows > installer.MAX_ROWS:
        raise SystemExit(f"timing-rows must be within 1..{installer.MAX_ROWS}")
    if args.warmups < 0 or args.repeats < 1 or args.batch < 1:
        raise SystemExit("warmups must be nonnegative; repeats and batch must be positive")
    catalog = TINY_CASES if profile == "tiny" else SHIPPED_CASES
    selected = tuple(case for case in catalog if not args.case or case.name in args.case)
    unknown = sorted(set(args.case) - {case.name for case in catalog})
    if unknown or not selected:
        raise SystemExit(f"unknown or empty case selection for {profile}: {unknown}")
    receipt = run_suite(
        mode=args.mode, cases=selected, rows=rows, timing_rows=args.timing_rows,
        warmups=args.warmups, repeats=args.repeats, batch=args.batch,
        atol=args.atol, rtol=args.rtol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["summary"]), flush=True)
    return 0 if receipt["state"]["tested_cases_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
