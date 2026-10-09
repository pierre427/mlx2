"""Fused GroupRMSNorm for the Flash-Next hyper-connection stream.

Original mlx2 work; not mined from another tree.

``GroupRMSNorm.__call__`` (``qwen4_exp.py``) upcasts to fp32 and then runs each
step as a separate op, so every step materialises an fp32 copy of a tensor that
is 335 MB in bf16 at T=16384.  Counting the traffic gives 6710.9 MB per call,
and the measured 12.53 ms works out to 528 GB/s -- the op is memory-bandwidth
saturated, not arithmetic bound.  Two ``GatedResidual`` calls per layer over 48
layers makes that ~13.7% of a Flash-Next prefill.

This kernel reads bf16 once, accumulates the sum of squares in fp32 registers,
and writes bf16 once: 671 MB instead of 6711 MB.  The fp32 is an *accumulator*
choice, not a storage one -- production already returns bf16 -- and
``scripts/measure_rmsnorm_precision.py`` measured bf16-in/fp32-accumulate/
bf16-out as bit-identical to the eager path, so the precision is preserved and
only the traffic changes.

Whether the result is bit-exact still depends on matching MLX's reduction order
over the 2560-term group, which is a measured question and not assumed here.
``scripts/bench_fused_group_norm.py`` reports the delta in bf16 ULPs; do not
quote a speedup without it.

Geometry is locked to the Flash-Next hyper-connection stream and admission fails
closed on anything else, following ``qwen4_moe_router.py``:

  * stream width 10240 = ``hc_count(4) * hidden_size(2560)``
  * ``group_size`` 2560 = ``hidden_size``, so 4 groups per row
  * ``eps`` exactly 1e-6 (the config's ``rms_norm_eps``)
  * bf16 in and out, fp32 weight upcast inside

Threadgroup memory is 32 bytes (8 simdgroup partials), nowhere near the 32 KB
ceiling that rm03 found binding on JIT ``metal_kernel`` staging.
"""
from __future__ import annotations

import os
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx

STREAM_WIDTH = 10240
GROUP_SIZE = 2560
GROUPS_PER_ROW = STREAM_WIDTH // GROUP_SIZE
EPS = 1e-06
THREADS = 256
ELEMS_PER_THREAD = GROUP_SIZE // THREADS  # 10
QUALIFIED_ROW_COUNTS: tuple[int, ...] = ()
# Candidate widths this kernel has been A/B'd at.  A production serving run
# chunks prefill at its own prefill_step, so real row counts may not be in this
# list -- in which case admission declines and the eager path runs, which is the
# intended fail-closed behaviour rather than a silent approximation.
CANDIDATE_ROW_COUNTS: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384)

from .import_env import snapshot as _import_env_snapshot

_import_env_snapshot(__name__)
_ENABLE = os.environ.get("MLX_QWEN4_FUSED_GROUP_NORM", "0") == "1"


def fused_group_norm_enabled() -> bool:
    """Live read of the lever, so the toggle needs no module reload."""
    return _ENABLE


def set_fused_group_norm_enabled(enabled: bool) -> bool:
    global _ENABLE
    _ENABLE = bool(enabled)
    return _ENABLE


@dataclass(frozen=True)
class GroupNormAdmission:
    accepted: bool
    reason: str


def _enabled_rows(candidate_rows: Sequence[int]) -> tuple[int, ...]:
    requested = tuple(candidate_rows)
    unsupported = tuple(r for r in requested if r not in CANDIDATE_ROW_COUNTS)
    if unsupported:
        raise ValueError(
            f"unsupported fused group-norm candidate row counts {unsupported}; "
            f"available candidates are {CANDIDATE_ROW_COUNTS}"
        )
    return tuple(dict.fromkeys((*QUALIFIED_ROW_COUNTS, *requested)))


def admit_fused_group_norm(
    x,
    weight,
    *,
    eps: float,
    group_size: int | None,
    candidate_rows: Sequence[int] = (),
) -> GroupNormAdmission:
    """Fail closed unless this is exactly the locked hyper-connection geometry."""
    if not _ENABLE:
        return GroupNormAdmission(False, "fused group norm is not enabled")
    if group_size != GROUP_SIZE:
        return GroupNormAdmission(
            False, f"group_size {group_size!r} is not the locked {GROUP_SIZE}"
        )
    if eps != EPS:
        return GroupNormAdmission(False, f"eps {eps!r} is not the locked {EPS}")
    if x.dtype != mx.bfloat16:
        return GroupNormAdmission(False, "x must be bfloat16")
    if weight.dtype != mx.bfloat16:
        return GroupNormAdmission(False, "weight must be bfloat16")
    if x.ndim < 2:
        return GroupNormAdmission(False, "x needs at least a row and a feature axis")
    if x.shape[-1] != STREAM_WIDTH:
        return GroupNormAdmission(
            False, f"stream width {x.shape[-1]} is not the locked {STREAM_WIDTH}"
        )
    if weight.shape != (STREAM_WIDTH,):
        return GroupNormAdmission(False, "weight must be exactly [10240]")
    rows = 1
    for extent in x.shape[:-1]:
        rows *= extent
    if rows < 1:
        return GroupNormAdmission(False, "empty input")
    if rows not in _enabled_rows(candidate_rows):
        return GroupNormAdmission(
            False,
            f"row count {rows} is not qualified for the fused group norm "
            f"(qualified: {QUALIFIED_ROW_COUNTS}; explicit candidates: "
            f"{CANDIDATE_ROW_COUNTS})",
        )
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return GroupNormAdmission(False, "Metal GPU unavailable")
    return GroupNormAdmission(True, "eligible")


_HEADER = "\n#include <metal_stdlib>\nusing namespace metal;\n"
_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint group = thread_position_in_grid.z;
    const uint base = group * 2560 + tid * 10;
    // `w` is indexed by FEATURE position within the 10240-wide row, not by the
    // flat group index -- groups wrap every 4. Indexing it by `base` reads out
    // of bounds past the first row.
    const uint feat = (group % 4) * 2560 + tid * 10;

    threadgroup float partials[8];

    // Read bf16 once, hold in fp32 registers.  10 elements per thread x 256
    // threads = one 2560-element group per threadgroup.
    float v[10];
    float ss = 0.0f;
    for (uint i = 0; i < 10; ++i) {
        v[i] = float(x[base + i]);
        ss += v[i] * v[i];
    }

    ss = simd_sum(ss);
    if (lane == 0) partials[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float p = (lane < 8) ? partials[lane] : 0.0f;
        p = simd_sum(p);
        if (lane == 0) partials[0] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mx.mean divides by the extent; mirror that rather than multiplying by a
    // precomputed reciprocal, which differs in the last bit.
    const float mean = partials[0] / 2560.0f;
    const float r = metal::precise::rsqrt(mean + 1e-06f);

    // Eager order is (x * rsqrt) then * weight.fp32, then cast.  Match it.
    for (uint i = 0; i < 10; ++i) {
        const float scaled = v[i] * r;
        out[base + i] = T(scaled * float(w[feat + i]));
    }
"""
_KERNEL = mx.fast.metal_kernel(
    name="qwen4_fused_group_norm",
    input_names=["x", "w"],
    output_names=["out"],
    header=_HEADER,
    source=_SOURCE,
    ensure_row_contiguous=True,
)


def _rows_of(x) -> int:
    rows = 1
    for extent in x.shape[:-1]:
        rows *= extent
    return rows


def _launch(x, weight) -> mx.array:
    (out,) = _KERNEL(
        inputs=[x, weight],
        template=[("T", x.dtype)],
        grid=(THREADS, 1, _rows_of(x) * GROUPS_PER_ROW),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


def fused_group_norm(x, weight, *, eps: float, group_size: int | None,
                     candidate_rows: Sequence[int] = ()) -> mx.array:
    """Fused GroupRMSNorm.  Raises rather than silently falling back."""
    admission = admit_fused_group_norm(
        x, weight, eps=eps, group_size=group_size, candidate_rows=candidate_rows
    )
    if not admission.accepted:
        raise ValueError(f"fused group norm is not eligible: {admission.reason}")
    out = _launch(x, weight)
    _record("calls")
    return out


_STATS = Counter()
_STATS_LOCK = threading.Lock()
# Keep decline reasons bounded: the row-count reason embeds a number, so it is
# collapsed to one key rather than one key per observed width.  The distinct
# widths are retained separately, also bounded, for diagnosis.
_DECLINE_LIMIT = 16
_DECLINED_ROWS: set[str] = set()
_DECLINED_ROWS_LIMIT = 32


def _reason_key(reason: str) -> str:
    if reason.startswith("row count"):
        return "decline:row_count_not_qualified"
    return f"decline:{reason}"


def _record(key: str) -> None:
    with _STATS_LOCK:
        _STATS[key] += 1


def _record_decline(reason: str) -> None:
    key = _reason_key(reason)
    with _STATS_LOCK:
        if key == "decline:row_count_not_qualified" and (
            len(_DECLINED_ROWS) < _DECLINED_ROWS_LIMIT
        ):
            # Keep the actual width, bounded, so a coverage gap is diagnosable
            # instead of hiding behind a collapsed counter.
            _DECLINED_ROWS.add(reason.split()[2])
        distinct = sum(1 for k in _STATS if k.startswith("decline:"))
        if key not in _STATS and distinct >= _DECLINE_LIMIT:
            key = "decline:other"
        _STATS[key] += 1


def fused_group_norm_stats() -> dict:
    """Observed engagement.  A route must show counters, not assert a flag."""
    with _STATS_LOCK:
        out = dict(_STATS)
        if _DECLINED_ROWS:
            out["declined_row_counts"] = sorted(_DECLINED_ROWS, key=int)
        return out


def reset_fused_group_norm_stats() -> None:
    with _STATS_LOCK:
        _STATS.clear()
        _DECLINED_ROWS.clear()


def try_fused_group_norm(x, weight, group_size, eps,
                         candidate_rows: Sequence[int] = CANDIDATE_ROW_COUNTS):
    """Fused GroupRMSNorm, or ``None`` meaning "stay eager".

    Declining is not an error.  The eager arithmetic is the same math at a
    different reduction order, so the caller falls back to it exactly as
    ``qwen3_next._run_glue`` does for a compiled span.  Every decline is counted
    by reason so a route can prove which path actually ran.

    The disabled case returns before allocating a dataclass or taking the stats
    lock.  This runs on every ``GroupRMSNorm`` call, which at decode is ~96 calls
    per generated token, so the off path has to be free.
    """
    if not _ENABLE:
        return None
    admission = admit_fused_group_norm(
        x, weight, eps=eps, group_size=group_size, candidate_rows=candidate_rows
    )
    if not admission.accepted:
        _record_decline(admission.reason)
        return None
    out = _launch(x, weight)
    _record("calls")
    return out


_PROBE_COMPLETE = False
_PROBE_OK = False


def probe_fused_group_norm(rows: int = 1024) -> bool:
    """One-shot self-check that the compiled kernel matches the eager math."""
    global _PROBE_COMPLETE, _PROBE_OK
    if _PROBE_COMPLETE:
        return _PROBE_OK
    if not mx.metal.is_available():
        return False
    previous = _ENABLE
    try:
        set_fused_group_norm_enabled(True)
        mx.random.seed(0)
        x = (mx.random.normal((1, rows, STREAM_WIDTH)) * 1.7).astype(mx.bfloat16)
        w = (1.0 + 0.05 * mx.random.normal((STREAM_WIDTH,))).astype(mx.bfloat16)
        mx.eval(x, w)
        ref = eager_group_norm(x, w)
        got = fused_group_norm(x, w, eps=EPS, group_size=GROUP_SIZE,
                               candidate_rows=(rows,))
        mx.eval(ref, got)
        ok = bool(mx.array_equal(ref, got).item())
    except Exception as exc:  # noqa: BLE001 -- a Metal compile or admission
        # failure of any kind means "not usable", and the caller falls back to
        # the eager arithmetic, which is the same math.  A device fault is not
        # a verdict on the kernel: it propagates and nothing is cached.
        from .served_exp import is_device_fault

        if is_device_fault(exc):
            raise
        ok = False
    finally:
        set_fused_group_norm_enabled(previous)
    _PROBE_OK = ok
    _PROBE_COMPLETE = True
    return _PROBE_OK


def eager_group_norm(x, weight) -> mx.array:
    """The production arithmetic, standalone, as the correctness reference.

    Mirrors ``GroupRMSNorm.__call__`` for group_size=2560 and eps=1e-6 without
    the fast-path branch, which declines at any prefill width.
    """
    dtype = x.dtype
    xf = x.astype(mx.float32).reshape(*x.shape[:-1], GROUPS_PER_ROW, GROUP_SIZE)
    out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + EPS)
    out = out.reshape(*x.shape)
    return (out * weight.astype(mx.float32)).astype(dtype)
