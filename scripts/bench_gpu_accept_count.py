"""Device-count GDN rollback: CPU references plus the Metal equality/perf A/B.

``--device cpu`` (the default) checks only the NumPy/MLX references and is what
``tests/test_gpu_accept_count.py`` imports.  ``--device gpu`` is the GPU-queue
job: it compares the dynamic reconstruct kernel with the template kernel bit
for bit at every partial width and on ragged rows, compares the masked generic
replay with the sliced one, and times both paths (cold compile included).
"""
# ruff: noqa: B023 -- timed lambdas run immediately inside their loop body.
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts.bench_qwen4_gdn_replay_cpu import (
    bf16_round,
    reconstruct_state,
    replay_states,
)


def reconstruct_dynamic_reference(initial, keys, corrections, decay, accepted):
    """Mirror ``_RECONSTRUCT_DYNAMIC_SOURCE`` row by row in NumPy.

    ``initial`` is ``[B, HV, DV, DK]`` and every tape carries a leading row
    axis; row ``b`` replays ``clamp(accepted[b], 0, tape_steps)`` steps.
    """
    initial = np.asarray(initial, dtype=np.float32)
    rows = initial.shape[0]
    tape_steps = keys.shape[1]
    accepted = np.broadcast_to(np.asarray(accepted).reshape(-1), (rows,))
    out = np.empty_like(initial)
    for row in range(rows):
        steps = int(min(max(int(accepted[row]), 0), tape_steps))
        if steps == 0:
            out[row] = initial[row]
            continue
        out[row] = replay_states(
            initial[row],
            keys[row, :steps],
            corrections[row, :steps],
            decay[row, :steps],
        )[-1]
    return out


def ragged_case(width: int, rows: int, *, seed: int = 11, hk=2, hv=6, dk=16, dv=16):
    rng = np.random.default_rng(seed + 31 * width + rows)
    initial = rng.normal(size=(rows, hv, dv, dk)).astype(np.float32)
    keys = bf16_round(rng.normal(size=(rows, width - 1, hk, dk)).astype(np.float32))
    corrections = rng.normal(size=(rows, width - 1, hv, dv)).astype(np.float32)
    decay = rng.uniform(0.8, 1.0, size=(rows, width - 1, hv)).astype(np.float32)
    return initial, keys, corrections, decay


def check_reference(widths=(2, 3, 5, 8, 17)) -> dict:
    """Dynamic reference == template reference for every width and prefix."""
    results = []
    for width in widths:
        initial, keys, corrections, decay = ragged_case(width, 3)
        exact = True
        for accepted in range(1, width):
            want = np.stack(
                [
                    reconstruct_state(
                        initial[row], keys[row], corrections[row], decay[row], accepted
                    )
                    for row in range(3)
                ]
            )
            got = reconstruct_dynamic_reference(
                initial, keys, corrections, decay, [accepted]
            )
            exact &= bool(np.array_equal(got, want))
        results.append({"width": width, "all_prefixes_bit_exact": exact})
    return {"device": "cpu", "results": results}


def _metal_available() -> bool:
    import mlx.core as mx

    return bool(mx.metal.is_available())


def _timed(fn, repeats):
    import mlx.core as mx

    started = time.perf_counter_ns()
    for _ in range(repeats):
        mx.eval(fn())
    return (time.perf_counter_ns() - started) / repeats / 1e3


def gpu_kernel_ab(widths, repeats: int, rows: int) -> list[dict]:
    """Dynamic vs template reconstruct: bitwise equality, then latency."""
    import mlx.core as mx

    from mlx2.runtime.models import qwen4_fused_gdn_verify as fused

    hk, hv, dk, dv = (
        fused.NUM_KEY_HEADS,
        fused.NUM_VALUE_HEADS,
        fused.KEY_HEAD_DIM,
        fused.VALUE_HEAD_DIM,
    )
    ty = fused.probe_qwen4_fused_gdn_decode(mx.bfloat16)
    if ty is None:
        raise SystemExit("fused GDN decode probe declined on this device")
    report = []
    for width in widths:
        initial, keys, corrections, decay = ragged_case(
            width, rows, hk=hk, hv=hv, dk=dk, dv=dv
        )
        state = mx.array(initial)
        tape_keys = mx.array(keys).astype(mx.bfloat16)
        tape_corr = mx.array(corrections)
        tape_decay = mx.array(decay)
        mismatches = []
        # Uniform prefix, one row at a time against the template kernel.
        for accepted in range(width):
            for row in range(rows):
                args = (
                    state[row : row + 1],
                    tape_keys[row : row + 1],
                    tape_corr[row : row + 1],
                    tape_decay[row : row + 1],
                )
                got = fused.qwen4_fused_gdn_reconstruct(
                    *args, mx.array([accepted], mx.int32), threadgroup_y=ty
                )
                want = (
                    args[0]
                    if accepted == 0
                    else fused.qwen4_fused_gdn_reconstruct(
                        *args, accepted, threadgroup_y=ty
                    )
                )
                if not np.array_equal(np.asarray(got), np.asarray(want)):
                    mismatches.append({"accepted": accepted, "row": row})
        # Ragged rows in one dispatch.
        rng = np.random.default_rng(width)
        ragged = rng.integers(0, width, size=rows).astype(np.int32)
        got = fused.qwen4_fused_gdn_reconstruct(
            state, tape_keys, tape_corr, tape_decay, mx.array(ragged),
            threadgroup_y=ty,
        )
        ref = reconstruct_dynamic_reference(
            initial, np.asarray(tape_keys.astype(mx.float32)), corrections, decay,
            ragged,
        )
        ragged_max_abs = float(np.max(np.abs(np.asarray(got) - ref)))
        for row, accepted in enumerate(ragged.tolist()):
            if accepted == 0:
                want = state[row : row + 1]
            else:
                want = fused.qwen4_fused_gdn_reconstruct(
                    state[row : row + 1], tape_keys[row : row + 1],
                    tape_corr[row : row + 1], tape_decay[row : row + 1],
                    accepted, threadgroup_y=ty,
                )
            if not np.array_equal(np.asarray(got[row : row + 1]), np.asarray(want)):
                mismatches.append({"ragged": ragged.tolist(), "row": row})
        # Latency, B=1, middle prefix: template (warm) vs dynamic (warm), and
        # the cold cost of compiling every template width vs the one kernel.
        mid = max(1, (width - 1) // 2)
        one = (state[:1], tape_keys[:1], tape_corr[:1], tape_decay[:1])
        device_mid = mx.array([mid], mx.int32)
        template_us = _timed(
            lambda: fused.qwen4_fused_gdn_reconstruct(*one, mid, threadgroup_y=ty),
            repeats,
        )
        dynamic_us = _timed(
            lambda: fused.qwen4_fused_gdn_reconstruct(
                *one, device_mid, threadgroup_y=ty
            ),
            repeats,
        )
        ragged_template_us = _timed(
            lambda: [
                fused.qwen4_fused_gdn_reconstruct(
                    state[r : r + 1], tape_keys[r : r + 1], tape_corr[r : r + 1],
                    tape_decay[r : r + 1], max(1, int(ragged[r])), threadgroup_y=ty,
                )
                for r in range(rows)
            ],
            repeats,
        )
        device_ragged = mx.array(ragged)
        ragged_dynamic_us = _timed(
            lambda: fused.qwen4_fused_gdn_reconstruct(
                state, tape_keys, tape_corr, tape_decay, device_ragged,
                threadgroup_y=ty,
            ),
            repeats,
        )
        report.append(
            {
                "width": width,
                "rows": rows,
                "bitwise_mismatches": mismatches,
                "ragged_vs_reference_max_abs": ragged_max_abs,
                "template_us": template_us,
                "dynamic_us": dynamic_us,
                "ragged_template_loop_us": ragged_template_us,
                "ragged_dynamic_us": ragged_dynamic_us,
            }
        )
    return report


def gpu_probe_compile(widths) -> list[dict]:
    """Cold probe: every template width vs the single dynamic kernel."""
    import mlx.core as mx

    from mlx2.runtime.models import qwen4_fused_gdn_verify as fused

    out = []
    for width in widths:
        fused._PROBED_REPLAY_STEPS.pop(width, None)
        fused._PROBED_DYNAMIC_REPLAY_STEPS.pop(width, None)
        started = time.perf_counter()
        fused.probe_qwen4_fused_gdn_replay_verify(mx.bfloat16, width)
        template_s = time.perf_counter() - started
        started = time.perf_counter()
        fused.probe_qwen4_fused_gdn_replay_verify(
            mx.bfloat16, width, dynamic_accept=True
        )
        dynamic_s = time.perf_counter() - started
        out.append(
            {"width": width, "template_probe_s": template_s, "dynamic_probe_s": dynamic_s}
        )
    return out


def gpu_generic_masked_ab(widths, rows: int, repeats: int) -> list[dict]:
    """Generic path on Metal: masked full-width replay vs sliced replay."""
    import mlx.core as mx

    from mlx2.runtime.models import qwen3_5
    from mlx2.runtime.models.cache import ArraysCache

    args = qwen3_5.TextModelArgs(
        hidden_size=256,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    mx.random.seed(5)
    layer = qwen3_5.GatedDeltaNet(args)
    layer.set_dtype(mx.bfloat16)
    mx.eval(layer.parameters())
    out = []
    previous = qwen3_5._GDN_ARRAY_ACCEPT
    qwen3_5._GDN_ARRAY_ACCEPT = True
    try:
        for width in widths:
            cache = ArraysCache(2)
            warm = mx.random.normal((rows, 5, args.hidden_size)).astype(mx.bfloat16)
            layer(warm, cache=cache)
            cache.start_speculation()
            x = mx.random.normal((rows, width, args.hidden_size)).astype(mx.bfloat16)
            layer(x, cache=cache)
            record = cache._rollbacks[-1]
            mismatches = []
            for m in range(1, width):
                sliced = record.fn(m)
                masked = record.per_row_fn([m] * rows)
                for a, b in zip(sliced, masked):
                    if not np.array_equal(
                        np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
                    ):
                        mismatches.append(m)
                        break
            ragged = [(r % (width - 1)) + 1 for r in range(rows)]
            sliced_us = _timed(
                lambda: [record.fn(m) for m in sorted(set(ragged))], repeats
            )
            masked_us = _timed(lambda: record.per_row_fn(ragged), repeats)
            out.append(
                {
                    "width": width,
                    "rows": rows,
                    "bitwise_mismatch_prefixes": mismatches,
                    "per_distinct_length_sliced_us": sliced_us,
                    "one_graph_masked_us": masked_us,
                }
            )
    finally:
        qwen3_5._GDN_ARRAY_ACCEPT = previous
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--widths", default="2,3,4,5,8,17")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    widths = tuple(int(value) for value in args.widths.split(","))
    report = {"cpu_reference": check_reference(widths)}
    if args.device == "gpu":
        if not _metal_available():
            raise SystemExit("--device gpu needs Metal")
        report["probe_compile"] = gpu_probe_compile(widths)
        report["reconstruct_kernel_ab"] = gpu_kernel_ab(widths, args.repeats, args.rows)
        report["generic_masked_ab"] = gpu_generic_masked_ab(
            widths, args.rows, max(1, args.repeats // 10)
        )
        report["pass"] = all(
            not row["bitwise_mismatches"] for row in report["reconstruct_kernel_ab"]
        ) and all(
            not row["bitwise_mismatch_prefixes"] for row in report["generic_masked_ab"]
        )
    report["reference_pass"] = all(
        row["all_prefixes_bit_exact"] for row in report["cpu_reference"]["results"]
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)
    return 0 if report["reference_pass"] and report.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
