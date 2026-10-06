#!/usr/bin/env python3
"""Bounded GPU probe for exact Qwen4 QSA stage-one selector candidates.

This script is deliberately separate from production dispatch.  It compares:

* ``radix_copy``: the selected mlx2 selector, including its score copy;
* ``direct8``: the same eight-pass composite radix selector reading scores in place;
* ``direct4``: four score-key radix passes followed by deterministic segmented
  emission, higher-block-ID tie fill, and the same ascending canonical output;
* ``onepass``: the existing rejected one-pass candidate, retained as a control.

Run only through the shared GPU queue.  The script records both lock owners and
fails closed when it cannot prove that it owns both locks.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from mlx2.runtime.models import qwen4_qsa_stage1 as qsa

RATIO = 128
TOPK = 512
CANDIDATE_TOPK = 544
WIDTH = 1024


HEADER = r"""
#include <metal_stdlib>
using namespace metal;

inline uint probe_float_order_key(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

inline ulong probe_composite_key(float score, uint block_id) {
    return (ulong(probe_float_order_key(score)) << 32) | ulong(block_id);
}

inline bool probe_id_before(
    uint a_index, bool a_valid, uint b_index, bool b_valid) {
    if (a_valid != b_valid) {
        return a_valid;
    }
    return a_index < b_index;
}
"""


SORT_AND_STORE = r"""
    uint my_index = exchange_indices[lane];
    bool my_valid = exchange_valid[lane] != 0u;
    for (uint sequence = 2u; sequence <= WIDTH; sequence <<= 1u) {
        for (uint stride = sequence >> 1u; stride > 0u; stride >>= 1u) {
            exchange_indices[lane] = my_index;
            exchange_valid[lane] = my_valid ? 1u : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint partner = lane ^ stride;
            const uint other_index = exchange_indices[partner];
            const bool other_valid = exchange_valid[partner] != 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const bool is_lower = (lane & stride) == 0u;
            const uint a_index = is_lower ? my_index : other_index;
            const bool a_valid = is_lower ? my_valid : other_valid;
            const uint b_index = is_lower ? other_index : my_index;
            const bool b_valid = is_lower ? other_valid : my_valid;
            const bool lower_wants_before = (lane & sequence) == 0u;
            const bool swap = lower_wants_before
                ? probe_id_before(b_index, b_valid, a_index, a_valid)
                : probe_id_before(a_index, a_valid, b_index, b_valid);
            if (swap) {
                my_index = is_lower ? b_index : a_index;
                my_valid = is_lower ? b_valid : a_valid;
            }
        }
    }

    if (lane < TOP_K) {
        uint output_id = my_index;
        if (lane >= selected_valid) {
            const uint invalid_count = TOP_K - selected_valid;
            output_id = blocks - invalid_count + (lane - selected_valid);
        }
        block_ids[size_t(row) * TOP_K + lane] = output_id;
    }
"""


@lru_cache(maxsize=8)
def direct8_kernel(topk: int, ratio: int):
    width = 1 << (max(256, int(topk)) - 1).bit_length()
    header = (
        HEADER
        + f"\nconstant constexpr uint TOP_K = {topk};\n"
        + f"constant constexpr uint RATIO = {ratio};\n"
        + f"constant constexpr uint WIDTH = {width};\n"
        + "constant constexpr uint RADIX_BINS = 256;\n"
    )
    source = (
        r"""
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const uint blocks = uint(dims[0]);
        const int qpos = int(q_positions[row]);
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(blocks, complete);
        const uint selected_valid = metal::min(TOP_K, valid_count);
        const size_t score_base = size_t(row) * blocks;

        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup atomic_uint selected_count;
        threadgroup ulong radix_prefix;
        threadgroup uint radix_rank;
        threadgroup ulong threshold_key;

        if (lane == 0u) {
            radix_prefix = 0ul;
            radix_rank = selected_valid > 0u ? selected_valid - 1u : 0u;
            threshold_key = 0xfffffffffffffffful;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_valid > 0u) {
            for (uint pass = 0u; pass < 8u; ++pass) {
                if (lane < RADIX_BINS) {
                    atomic_store_explicit(
                        &radix_histogram[lane], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint shift = 56u - pass * 8u;
                const ulong prefix = radix_prefix;
                for (uint block = lane; block < valid_count; block += WIDTH) {
                    const ulong key = probe_composite_key(
                        scores[score_base + block], block);
                    bool prefix_matches = true;
                    if (pass > 0u) {
                        prefix_matches = (key >> (shift + 8u)) == prefix;
                    }
                    if (prefix_matches) {
                        atomic_fetch_add_explicit(
                            &radix_histogram[uint((key >> shift) & 0xfful)],
                            1u,
                            memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lane == 0u) {
                    uint rank = radix_rank;
                    uint chosen = 0u;
                    for (int digit = 255; digit >= 0; --digit) {
                        const uint count = atomic_load_explicit(
                            &radix_histogram[uint(digit)], memory_order_relaxed);
                        if (rank < count) {
                            chosen = uint(digit);
                            break;
                        }
                        rank -= count;
                    }
                    radix_prefix = (radix_prefix << 8u) | ulong(chosen);
                    radix_rank = rank;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0u) {
                threshold_key = radix_prefix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0u;
        if (lane == 0u) {
            atomic_store_explicit(&selected_count, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_valid > 0u) {
            const ulong threshold = threshold_key;
            for (uint block = lane; block < valid_count; block += WIDTH) {
                if (probe_composite_key(scores[score_base + block], block) >=
                    threshold) {
                    const uint slot = atomic_fetch_add_explicit(
                        &selected_count, 1u, memory_order_relaxed);
                    if (slot < TOP_K) {
                        exchange_indices[slot] = block;
                        exchange_valid[slot] = 1u;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    """
        + SORT_AND_STORE
    )
    return mx.fast.metal_kernel(
        name=f"mlx2_probe_qsa_direct8_k{topk}_r{ratio}",
        input_names=["scores", "q_positions", "dims"],
        output_names=["block_ids"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=8)
def direct4_kernel(topk: int, ratio: int):
    header = (
        HEADER
        + f"\nconstant constexpr uint TOP_K = {topk};\n"
        + f"constant constexpr uint RATIO = {ratio};\n"
        + f"constant constexpr uint WIDTH = {WIDTH};\n"
        + "constant constexpr uint RADIX_BINS = 256;\n"
        + "constant constexpr uint SIMD_GROUPS = WIDTH / 32;\n"
    )
    source = (
        r"""
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const uint simd_id = lane / 32u;
        const uint simd_lane = lane % 32u;
        const uint blocks = uint(dims[0]);
        const int qpos = int(q_positions[row]);
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(blocks, complete);
        const uint selected_valid = metal::min(TOP_K, valid_count);
        const size_t score_base = size_t(row) * blocks;

        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup uint partial_greater[SIMD_GROUPS];
        threadgroup uint partial_ties[SIMD_GROUPS];
        threadgroup uint radix_prefix;
        threadgroup uint radix_rank;
        threadgroup uint threshold_key;
        threadgroup uint total_greater;

        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0u;
        if (lane == 0u) {
            radix_prefix = 0u;
            radix_rank = selected_valid > 0u ? selected_valid - 1u : 0u;
            threshold_key = 0xffffffffu;
            total_greater = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_valid > 0u) {
            for (uint pass = 0u; pass < 4u; ++pass) {
                if (lane < RADIX_BINS) {
                    atomic_store_explicit(
                        &radix_histogram[lane], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint shift = 24u - pass * 8u;
                const uint prefix = radix_prefix;
                for (uint block = lane; block < valid_count; block += WIDTH) {
                    const uint key = probe_float_order_key(
                        scores[score_base + block]);
                    bool prefix_matches = true;
                    if (pass > 0u) {
                        prefix_matches = (key >> (shift + 8u)) == prefix;
                    }
                    if (prefix_matches) {
                        atomic_fetch_add_explicit(
                            &radix_histogram[(key >> shift) & 0xffu],
                            1u,
                            memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lane == 0u) {
                    uint rank = radix_rank;
                    uint chosen = 0u;
                    for (int digit = 255; digit >= 0; --digit) {
                        const uint count = atomic_load_explicit(
                            &radix_histogram[uint(digit)], memory_order_relaxed);
                        if (rank < count) {
                            chosen = uint(digit);
                            break;
                        }
                        rank -= count;
                    }
                    radix_prefix = (radix_prefix << 8u) | chosen;
                    radix_rank = rank;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0u) {
                threshold_key = radix_prefix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rake the row in reverse block-ID order.  This makes threshold ties
        // choose larger block IDs first, matching the existing composite key.
        const uint segment = (valid_count + WIDTH - 1u) / WIDTH;
        const uint reverse_start = lane * segment;
        const uint reverse_end = metal::min(reverse_start + segment, valid_count);
        uint local_greater = 0u;
        uint local_ties = 0u;
        for (uint reverse = reverse_start; reverse < reverse_end; ++reverse) {
            const uint block = valid_count - 1u - reverse;
            const uint key = probe_float_order_key(scores[score_base + block]);
            local_greater += key > threshold_key ? 1u : 0u;
            local_ties += key == threshold_key ? 1u : 0u;
        }

        uint prefix_greater = metal::simd_prefix_exclusive_sum(local_greater);
        uint prefix_ties = metal::simd_prefix_exclusive_sum(local_ties);
        if (simd_lane == 31u) {
            partial_greater[simd_id] = prefix_greater + local_greater;
            partial_ties[simd_id] = prefix_ties + local_ties;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u) {
            const uint group_greater = partial_greater[simd_lane];
            const uint group_ties = partial_ties[simd_lane];
            const uint group_prefix_greater =
                metal::simd_prefix_exclusive_sum(group_greater);
            const uint group_prefix_ties =
                metal::simd_prefix_exclusive_sum(group_ties);
            partial_greater[simd_lane] = group_prefix_greater;
            partial_ties[simd_lane] = group_prefix_ties;
            if (simd_lane == 31u) {
                total_greater = group_prefix_greater + group_greater;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint output_greater = partial_greater[simd_id] + prefix_greater;
        uint output_ties = total_greater + partial_ties[simd_id] + prefix_ties;
        if (local_greater > 0u ||
            (local_ties > 0u && output_ties < selected_valid)) {
            for (uint reverse = reverse_start; reverse < reverse_end; ++reverse) {
                const uint block = valid_count - 1u - reverse;
                const uint key = probe_float_order_key(scores[score_base + block]);
                if (key > threshold_key) {
                    if (output_greater < selected_valid) {
                        exchange_indices[output_greater] = block;
                        exchange_valid[output_greater] = 1u;
                    }
                    output_greater += 1u;
                } else if (key == threshold_key) {
                    if (output_ties < selected_valid) {
                        exchange_indices[output_ties] = block;
                        exchange_valid[output_ties] = 1u;
                    }
                    output_ties += 1u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    """
        + SORT_AND_STORE
    )
    return mx.fast.metal_kernel(
        name=f"mlx2_probe_qsa_direct4_k{topk}_r{ratio}",
        input_names=["scores", "q_positions", "dims"],
        output_names=["block_ids"],
        header=header,
        source=source,
    )


def _invoke(kernel, scores: mx.array, positions: mx.array, topk: int, width: int):
    rows, blocks = map(int, scores.shape)
    (selected,) = kernel(
        inputs=[
            scores,
            positions.astype(mx.int32),
            mx.array([blocks], dtype=mx.int32),
        ],
        grid=(rows * width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=[(rows, topk)],
        output_dtypes=[mx.uint32],
    )
    return selected


def select_direct8(
    scores: mx.array, positions: mx.array, *, topk: int, compress_ratio: int
) -> mx.array:
    width = 1 << (max(256, int(topk)) - 1).bit_length()
    return _invoke(direct8_kernel(topk, compress_ratio), scores, positions, topk, width)


def select_direct4(
    scores: mx.array, positions: mx.array, *, topk: int, compress_ratio: int
) -> mx.array:
    return _invoke(direct4_kernel(topk, compress_ratio), scores, positions, topk, WIDTH)


SELECTORS: dict[str, Callable[..., mx.array]] = {
    "radix_copy": qsa._select_scores,
    "direct8": select_direct8,
    "direct4": select_direct4,
    "onepass": qsa._select_scores_onepass,
}


def _lock_owner(path: str) -> dict:
    owner_path = Path(path) / "owner.json"
    if not owner_path.is_file():
        raise RuntimeError(f"missing GPU lock owner: {owner_path}")
    return json.loads(owner_path.read_text())


def _prove_gpu_ownership() -> dict:
    expected = os.environ.get("GPUQ_LEASE")
    session = os.environ.get("GPUQ_SESSION")
    if not expected:
        raise RuntimeError("GPUQ_LEASE does not own both locks")
    if not session:
        raise RuntimeError("GPUQ_SESSION does not own both locks")
    shared = _lock_owner("/Users/Shared/mlxuag/gpu.lock")
    temporary = _lock_owner("/tmp/gpu.lock")
    if any(owner.get("lease_id") != expected for owner in (shared, temporary)):
        raise RuntimeError("GPUQ_LEASE does not own both locks")
    if any(owner.get("session") != session for owner in (shared, temporary)):
        raise RuntimeError("GPUQ_SESSION does not own both locks")
    return {"shared": shared, "temporary": temporary}


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False, timeout=10
    )
    return (completed.stdout + completed.stderr).strip()


def _host_snapshot() -> dict:
    return {
        "vm_swapusage": _command_output(["sysctl", "-n", "vm.swapusage"]),
        "vm_stat": _command_output(["vm_stat"]),
        "thermal": _command_output(["pmset", "-g", "therm"]),
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
    }


def _array_equal(left: mx.array, right: mx.array) -> bool:
    result = mx.all(left == right)
    mx.eval(result)
    return bool(result.item())


def _score_case(kind: str, rows: int, blocks: int) -> mx.array:
    if kind == "zeros":
        return mx.zeros((rows, blocks), dtype=mx.float32)
    mx.random.seed(1729 + rows + blocks)
    scores = mx.random.uniform(shape=(rows, blocks), dtype=mx.float32)
    if kind == "ties":
        scores = mx.floor(scores * 16.0) / 16.0
    elif kind != "distinct":
        raise ValueError(kind)
    return scores


def correctness_suite() -> list[dict]:
    results = []
    cases = [
        ("distinct", 3, 4097, TOPK, "full"),
        ("ties", 5, 4097, TOPK, "full"),
        ("zeros", 2, 4097, TOPK, "full"),
        ("ties", 4, 2049, CANDIDATE_TOPK, "partial"),
        ("distinct", 3, 2049, TOPK, "underfilled"),
    ]
    for kind, rows, blocks, topk, position_mode in cases:
        scores = _score_case(kind, rows, blocks)
        if position_mode == "full":
            positions = mx.full((rows,), blocks * RATIO - 1, dtype=mx.int32)
        elif position_mode == "partial":
            positions = mx.array(
                [blocks * RATIO - 1, 1900 * RATIO - 1, 900 * RATIO - 1, 0],
                dtype=mx.int32,
            )
        else:
            positions = mx.array([0, 128 * RATIO - 1, 511 * RATIO - 1], dtype=mx.int32)
        mx.eval(scores, positions)
        reference = SELECTORS["radix_copy"](
            scores, positions, topk=topk, compress_ratio=RATIO
        )
        mx.eval(reference)
        row = {
            "kind": kind,
            "rows": rows,
            "blocks": blocks,
            "topk": topk,
            "position_mode": position_mode,
            "matches": {},
        }
        for name in ("direct8", "direct4", "onepass"):
            candidate = SELECTORS[name](
                scores, positions, topk=topk, compress_ratio=RATIO
            )
            row["matches"][name] = _array_equal(reference, candidate)
        results.append(row)
        del scores, positions, reference
        gc.collect()
        mx.clear_cache()
    return results


def _interleaved_timings(
    arms: dict[str, Callable[[], mx.array]], *, warmups: int, repetitions: int
) -> dict[str, dict]:
    names = list(arms)
    for offset in range(warmups):
        order = names[offset % len(names) :] + names[: offset % len(names)]
        for name in order:
            value = arms[name]()
            mx.eval(value)
    samples: dict[str, list[float]] = {name: [] for name in names}
    orders = []
    for offset in range(repetitions):
        order = names[offset % len(names) :] + names[: offset % len(names)]
        if offset % 2:
            order = list(reversed(order))
        orders.append(order)
        for name in order:
            started = time.perf_counter_ns()
            value = arms[name]()
            mx.eval(value)
            samples[name].append((time.perf_counter_ns() - started) / 1e6)
    return {
        name: {
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "samples_ms": values,
            "interleaved_orders": orders,
        }
        for name, values in samples.items()
    }


def selector_timing_suite() -> list[dict]:
    results = []
    for rows, blocks in ((64, 16256), (512, 16256), (512, 24576), (512, 32768)):
        scores = _score_case("distinct", rows, blocks)
        positions = mx.full((rows,), blocks * RATIO - 1, dtype=mx.int32)
        mx.eval(scores, positions)
        callables = {}
        for name in ("radix_copy", "direct8", "direct4", "onepass"):
            selector = SELECTORS[name]
            callables[name] = (
                lambda selector=selector, scores=scores, positions=positions: selector(
                    scores, positions, topk=TOPK, compress_ratio=RATIO
                )
            )
        cell = {
            "rows": rows,
            "blocks": blocks,
            "topk": TOPK,
            "arms": _interleaved_timings(callables, warmups=2, repetitions=15),
        }
        baseline = cell["arms"]["radix_copy"]["median_ms"]
        for arm in cell["arms"].values():
            arm["ratio_vs_radix_copy"] = arm["median_ms"] / baseline
        results.append(cell)
        del scores, positions
        gc.collect()
        mx.clear_cache()
    return results


def _full_route(
    selector: Callable[..., mx.array],
    q: mx.array,
    pooled: mx.array,
    positions: mx.array,
) -> mx.array:
    rows = int(q.shape[1])
    approximate = qsa._mpp_scores(q, pooled)
    candidate_ids = selector(
        approximate,
        positions,
        topk=CANDIDATE_TOPK,
        compress_ratio=RATIO,
    )
    candidate_keys = mx.take(pooled[0], candidate_ids, axis=0)
    exact = mx.einsum(
        "lhd,lcd->lch", q[0].astype(mx.float32), candidate_keys.astype(mx.float32)
    )
    exact = mx.sum(mx.maximum(exact, 0), axis=-1) / math.sqrt(q.shape[-1])
    selected_slots = selector(
        exact,
        mx.full((rows,), CANDIDATE_TOPK * RATIO - 1, dtype=mx.int32),
        topk=TOPK,
        compress_ratio=RATIO,
    )
    return mx.take_along_axis(candidate_ids, selected_slots, axis=-1)


def route_timing_suite() -> list[dict]:
    results = []
    for rows, blocks in ((64, 16256), (512, 16256), (512, 24576)):
        mx.random.seed(90210 + rows + blocks)
        query = mx.random.uniform(
            low=-1.0, high=1.0, shape=(1, rows, 4, 128), dtype=mx.float16
        )
        pooled = mx.random.uniform(
            low=-1.0, high=1.0, shape=(1, blocks, 128), dtype=mx.float16
        )
        positions = mx.full((rows,), blocks * RATIO - 1, dtype=mx.int32)
        mx.eval(query, pooled, positions)
        reference = _full_route(SELECTORS["radix_copy"], query, pooled, positions)
        mx.eval(reference)
        cell = {
            "rows": rows,
            "blocks": blocks,
            "topk": TOPK,
            "candidate_topk": CANDIDATE_TOPK,
            "arms": {},
            "matches": {},
        }
        callables = {}
        for name in ("radix_copy", "direct8", "direct4", "onepass"):
            selector = SELECTORS[name]
            candidate = _full_route(selector, query, pooled, positions)
            cell["matches"][name] = _array_equal(reference, candidate)
            callables[name] = (
                lambda selector=selector, query=query, pooled=pooled, positions=positions: (
                    _full_route(selector, query, pooled, positions)
                )
            )
        cell["arms"] = _interleaved_timings(callables, warmups=1, repetitions=9)
        baseline = cell["arms"]["radix_copy"]["median_ms"]
        for arm in cell["arms"].values():
            arm["ratio_vs_radix_copy"] = arm["median_ms"] / baseline
        results.append(cell)
        del query, pooled, positions, reference
        gc.collect()
        mx.clear_cache()
    return results


def memory_suite(include_largest: bool) -> list[dict]:
    shapes = [(4096, 32768), (8192, 16256)]
    if include_largest:
        shapes.append((8192, 32768))
    results = []
    for rows, blocks in shapes:
        scores = _score_case("distinct", rows, blocks)
        positions = mx.full((rows,), blocks * RATIO - 1, dtype=mx.int32)
        mx.eval(scores, positions)
        cell = {
            "rows": rows,
            "blocks": blocks,
            "input_score_bytes": rows * blocks * 4,
            "arms": {},
        }
        for name in ("radix_copy", "direct8", "direct4"):
            gc.collect()
            mx.clear_cache()
            active_before = int(mx.get_active_memory())
            mx.reset_peak_memory()
            started = time.perf_counter_ns()
            result = SELECTORS[name](scores, positions, topk=TOPK, compress_ratio=RATIO)
            mx.eval(result)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            peak = int(mx.get_peak_memory())
            cell["arms"][name] = {
                "elapsed_ms": elapsed_ms,
                "active_before_bytes": active_before,
                "peak_bytes": peak,
                "incremental_peak_bytes": max(0, peak - active_before),
            }
            del result
        results.append(cell)
        del scores, positions
        gc.collect()
        mx.clear_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-largest", action="store_true")
    args = parser.parse_args()

    locks = _prove_gpu_ownership()
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("Metal GPU is unavailable")

    report = {
        "schema_version": 1,
        "kind": "qwen4_qsa_stage1_selector_probe",
        "status": "running",
        "production_selection_changed": False,
        "revision": _command_output(["git", "rev-parse", "HEAD"]),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "gpu_locks": locks,
        "environment": {
            "GPUQ_SESSION": os.environ.get("GPUQ_SESSION"),
            "GPUQ_LEASE": os.environ.get("GPUQ_LEASE"),
        },
        "host_before": _host_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        report["correctness"] = correctness_suite()
        if not all(all(cell["matches"].values()) for cell in report["correctness"]):
            raise RuntimeError("candidate exactness gate failed")
        report["selector_timing"] = selector_timing_suite()
        report["route_timing"] = route_timing_suite()
        if not all(all(cell["matches"].values()) for cell in report["route_timing"]):
            raise RuntimeError("full-route exactness gate failed")
        report["memory"] = memory_suite(args.include_largest)
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["host_after"] = _host_snapshot()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
