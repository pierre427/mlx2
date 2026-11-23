# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash fp32 planes kept BF16/F16-resident (opt-in ``resident_bf16``).

Original mlx2 code.  Audit H5 (2026-11-22): ``sanitize`` stores several
planes fp32 once at load so no step re-reads a 16-bit copy to upcast it
(``hy_v4_flash._FP32_KEYS``), and the indexer caches its pooled group keys
fp32.  Every step then reads the fp32 bytes: the router (~258 MB per decode
step vs 129 MB bf16 on Hy4-flash), the indexer group keys (~16.8 MB per
indexed layer per lane at 128K), the quantized LM head's scales/biases and
the ``q2_0`` expert scales/biases (widened from the pack's F16).  Their
values are 16-bit-exact (checkpoint bf16 / F16, or the bf16 RoPE output of
the indexer), so this option keeps them 16-bit and widens them at use.

Parts (``resident_bf16``: true for all, or a list of these names):

``router``        ``mlp.gate.weight`` -> bf16, only when bf16-exact.  Read by
                  ``f32_matmul``: on the GPU, at most ``GEMV_MAX_ROWS`` rows
                  run a GEMV kernel that widens the weight in registers (fp32
                  sums, another order than MLX's: not bitwise); more rows,
                  and the CPU, widen with ``astype`` (bitwise the fp32 plane).
``head``          a dense fp32 LM head weight -> bf16 (same ``f32_matmul``);
                  a quantized head's fp32 scales/biases -> bf16.  The latter
                  are widened by ``astype`` inside ``Model.logits`` (MLX has
                  no mixed-dtype quantized matmul), so they save memory but
                  add a per-step copy: measure before using it.
``indexer_keys``  pooled group keys cached bf16 instead of fp32 (only for bf16
                  activations, where they are bf16-exact).  The reference
                  scores promote them (same values, bitwise); the v1 fused
                  kernel widens a copy; ``indexer_fused_scores: "simd"``
                  reads them in place.
``q2_scales``     ``q2_0`` expert scales/biases -> F16 (the pack's own
                  values; biases ``= -scales`` stay exact).  ``gather_qmm``
                  promotes them (bitwise on CPU); like the quantized head,
                  memory saved, a copy per call added.

A plane that is not exactly representable is left fp32 and recorded.  GPU
A/B pending for every part.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

PARTS = ("router", "head", "indexer_keys", "q2_scales")
GEMV_MAX_ROWS = 8
STATS = {"gemv_calls": 0, "widen_calls": 0}


def parse(value) -> tuple:
    """``resident_bf16`` policy value -> sorted part names (() = off)."""
    if value is False or value is None:
        return ()
    if value is True:
        return PARTS
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        unknown = sorted(set(value) - set(PARTS))
        if unknown:
            raise ValueError(f"Hy4-flash resident_bf16 parts must be among {PARTS}: {unknown}")
        return tuple(p for p in PARTS if p in value)
    raise ValueError(f"Hy4-flash resident_bf16 must be a boolean or a non-empty list of {PARTS}")


def layout_tag(parts) -> str | None:
    parts = parse(parts) if not isinstance(parts, tuple) else parts
    return f"resident-bf16-{'-'.join(p.replace('_', '') for p in parts)}-v1" if parts else None


def _narrow(array, dtype):
    """``array`` as ``dtype`` when every value round-trips exactly, else None."""
    if array.dtype != mx.float32:
        return None
    low = array.astype(dtype)
    return low if bool(mx.array_equal(low.astype(mx.float32), array).item()) else None


def install(model, parts) -> dict:
    """Narrow the selected planes of a loaded ``hy_v4_flash.Model`` in place.

    Returns (and keeps as ``model.resident_record``) bytes saved and the
    planes left fp32 per part.
    """
    from .hy_v4_flash import MoE, Q2ZeroSwitchLinear

    parts = parse(parts) if not isinstance(parts, tuple) else parts
    record = {"parts": list(parts), "saved_bytes": {p: 0 for p in parts}, "kept_fp32": {p: [] for p in parts}}

    def swap(owner, name, label, part, dtype=mx.bfloat16):
        value = owner[name]
        low = _narrow(value, dtype)
        if low is None:
            if value.dtype == mx.float32:
                record["kept_fp32"][part].append(label)
            return
        record["saved_bytes"][part] += value.nbytes - low.nbytes
        owner[name] = low

    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if "router" in parts and isinstance(mlp, MoE):
            swap(mlp.gate, "weight", f"layers.{i}.mlp.gate.weight", "router")
            mlp.gate.resident_gemv = True
        if "q2_scales" in parts and isinstance(mlp, MoE):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                bank = getattr(mlp.experts, proj, None)
                if isinstance(bank, Q2ZeroSwitchLinear):
                    for plane in ("scales", "biases"):
                        swap(bank, plane, f"layers.{i}.mlp.experts.{proj}.{plane}", "q2_scales",
                             dtype=mx.float16)
        if "indexer_keys" in parts and "compressed_indexer" in layer.self_attn:
            layer.self_attn.compressed_indexer.lowp_keys = True
    if "head" in parts and model.args.enable_lm_head_fp32:
        head = model.lm_head
        for plane in (("scales", "biases") if "scales" in head else ("weight",)):
            if plane in head:
                swap(head, plane, f"lm_head.{plane}", "head")
        model.resident_gemv = "scales" not in head
    record["saved_bytes_total"] = sum(record["saved_bytes"].values())
    object.__setattr__(model, "resident_record", record)
    return record


def status(model) -> dict:
    record = getattr(model, "resident_record", None) or {}
    parts = record.get("parts", [])
    keys = [layer.self_attn.compressed_indexer for layer in model.model.layers
            if "compressed_indexer" in layer.self_attn]
    return {"implemented": True, "qualified": False, "selected": bool(parts), "parts": parts,
            "saved_bytes": record.get("saved_bytes", {}), "saved_bytes_total": record.get("saved_bytes_total", 0),
            "kept_fp32": record.get("kept_fp32", {}),
            "indexer_key_groups_bf16": sum(getattr(k, "lowp_key_groups", 0) for k in keys),
            "gemv_calls": STATS["gemv_calls"], "widen_calls": STATS["widen_calls"],
            "observed_used": bool(parts) and (STATS["gemv_calls"] + STATS["widen_calls"] > 0
                                              or any(getattr(k, "lowp_key_groups", 0) for k in keys)),
            "exactness": ("values exact (planes narrowed only when they round-trip); CPU forwards bitwise; "
                          "GPU GEMV rows (<= 8) sum in another order than MLX's GEMV")}


# fp32 x [M, K] times a 16-bit weight [N, K] (rows of the output), widened in
# registers: one simdgroup per output column n, lanes stride K by 4, every
# row's partial reduced with simd_sum.
_GEMV_SRC = """
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int n = int(threadgroup_position_in_grid.x) * 8 + int(sg);
    const int N = int(w_shape[0]);
    const int K = int(w_shape[1]);
    const int Mr = int(x_shape[0]);
    if (n >= N) return;
    float acc[MAXR];
    for (int m = 0; m < MAXR; ++m) acc[m] = 0.0f;
    const device WT* wr = w + size_t(n) * K;
    for (int k = int(lane) * 4; k < K; k += 128) {
        const float w0 = float(wr[k]), w1 = float(wr[k + 1]), w2 = float(wr[k + 2]), w3 = float(wr[k + 3]);
        for (int m = 0; m < MAXR; ++m) {
            if (m < Mr) {
                const device float* xr = x + size_t(m) * K + k;
                acc[m] += xr[0] * w0 + xr[1] * w1 + xr[2] * w2 + xr[3] * w3;
            }
        }
    }
    for (int m = 0; m < MAXR; ++m) {
        const float s = simd_sum(acc[m]);
        if (lane == 0 && m < Mr) out[size_t(m) * N + n] = s;
    }
"""


@lru_cache(maxsize=None)
def _gemv_kernel():
    return mx.fast.metal_kernel(name="hy4_resident_gemv_f32_v1", input_names=["x", "w"],
                                output_names=["out"], source=_GEMV_SRC)


def gemv_f32(x2, w):
    """``x2 [M, K]`` fp32 (M <= GEMV_MAX_ROWS) times ``w [N, K]`` 16-bit -> ``[M, N]`` fp32 (Metal)."""
    M, K = x2.shape
    N = int(w.shape[0])
    return _gemv_kernel()(inputs=[x2, w], template=[("WT", w.dtype), ("MAXR", GEMV_MAX_ROWS)],
                          grid=(-(-N // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]


def f32_matmul(x, w, *, kernel: bool = False):
    """``x.astype(fp32) @ w.astype(fp32).T`` for an fp32 or a narrowed 16-bit ``w``.

    Without ``kernel`` (every caller unless ``install`` narrowed its plane)
    this is the reference expression unchanged.  With it, a 16-bit ``w``
    takes the GEMV kernel for at most ``GEMV_MAX_ROWS`` rows on the GPU, else
    the same expression (``astype`` widens to the exact fp32 values).
    """
    xf = x.astype(mx.float32)
    if not kernel or w.dtype == mx.float32:
        return xf @ w.astype(mx.float32).T
    rows = 1
    for n in xf.shape[:-1]:
        rows *= int(n)
    K = int(w.shape[-1])
    if (0 < rows <= GEMV_MAX_ROWS and K % 4 == 0 and mx.default_device() == mx.gpu
            and mx.metal.is_available()):
        STATS["gemv_calls"] += 1
        return gemv_f32(mx.contiguous(xf.reshape(rows, K)), w).reshape(*xf.shape[:-1], int(w.shape[0]))
    STATS["widen_calls"] += 1
    return xf @ w.astype(mx.float32).T
