# SPDX-License-Identifier: Apache-2.0
"""Opt-in Hy4-flash small-op fusions (default off; the unfused path is the reference).

Original mlx2 code.  Design references (read as PR descriptions only, no code
read or copied; provenance/hy-v4-flash-fusions.json): SGLang #51646 "fused
iHC post-pre" (previous block's post, this block's pre and input RMSNorm in
one pass), vLLM #65110 (split-K iHC pre for small token counts, partials
reduced in index order), vLLM #66225 (FlashTopK: indexer scoring fused so the
per-head score block is never written).

Four independent fusions, each behind its own execution-policy key
(``adapters/hy_v4_flash.py``):

``hc_seam_fused`` (bool)
    The iHC boundary ``merge(prev branch into streams) -> hc pre (mixes,
    sigmoid gates, stream collapse) -> RMSNorm`` of the next sub-block, in
    two custom Metal kernels per seam instead of ~15: a split-K partial pass
    (merge, sum of squares and the ``2n`` mix dot products over K blocks;
    each threadgroup loads its ``hc_fn`` tile once for a block of rows) and a
    per-row finalize (partials summed in index order, gates, collapse,
    RMSNorm).  The merged streams are bitwise the reference's; the fp32
    reduction order of the mixes and norms differs from MLX's GEMV and
    reductions, so gates and the normed input are numerically equivalent,
    not bitwise (tests quantify it).  ``mx.compile`` of the whole seam was
    measured first: MLX fuses no reduction or matmul into a compiled
    kernel, so it launched the same 15 kernels per seam (and on CPU its
    fused elementwise code contracts differently, so it is not even
    bitwise); it was dropped.  On a CPU stream the reference ops run.  The
    kernels have no gradient: a VJP/JVP through them raises ``ValueError``.
``mla_fused_input_proj`` (bool)
    One GEMM for every projection that reads the attention input:
    ``q_a_proj``, ``kv_a_proj_with_mqa``, ``linear_gate`` and, on indexed
    layers, the indexer's ``token_key_projection`` and
    ``head_weight_projection``.  Weights are concatenated on the output axis
    at load (quantized: only when bits / group size / mode match; otherwise
    the layer keeps its split projections and the reason is recorded).  One
    GEMM runs only up to a row bound measured at load on the running device
    (``calibrate``; static caps ``FUSED_MAX_ROWS``), split GEMMs above it.  The
    original modules become weightless views (``SegmentLinear``) that read
    their slice of the fused output, so every call site is unchanged.  The
    sigmoid-gate multiply in ``Attention.finish`` is compiled.
``engram_fused_proj`` (bool)
    One GEMM for the four per-stream key projections plus the value
    projection, the four ``norm_key`` / ``norm_query`` norms stacked into one
    norm each, and the gate math on ``[B, L, n]`` at once.
``indexer_fused_scores`` (bool)
    ``CompressedIndexer.select``: a Metal kernel computes the masked logits
    ``sum_h w_h * relu(q_h . k_g)`` directly (the fp32 ``[B, heads, rows, G]``
    score block is never written; the logits block is ``heads`` times
    smaller, so long chunks need ``heads`` times fewer bounded-block syncs).
    Top-k stays ``mx.argpartition`` on the logits, so ties break as before;
    only the logits' fp32 rounding differs (dot order), quantified in tests.

``engram_gate_rounding`` ("fp32" default | "torch") is a parity option, not
a fusion: see ``EngramLayer``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

ENGRAM_GATE_ROUNDINGS = ("fp32", "torch")


def on_gpu() -> bool:
    return mx.default_device() == mx.gpu


# ------------------------------------------------------------ HC seam --

# Split-K partial pass: one threadgroup per (K block, row block).  Each
# thread holds EPT columns of every hc_fn row in registers (loaded once per
# threadgroup) and walks the RB rows of its block.  Partials are written per
# (row, K block); the finalize sums them in index order (deterministic).
_PARTIAL_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint kb = threadgroup_position_in_grid.x;
    uint rb = threadgroup_position_in_grid.y;
    uint simd = tid / 32, lane = tid % 32;
    uint R = residual_shape[0];
    threadgroup float red[(OUT + 1) * (TG / 32)];
    float w[EPT][OUT];
    for (int e = 0; e < EPT; ++e) {
        int k = kb * (TG * EPT) + e * TG + tid;
        for (int o = 0; o < OUT; ++o) {
            w[e][o] = k < K ? hc_fn[o * K + k] : 0.0f;
        }
    }
    uint r1 = min(R, (rb + 1) * RB);
    for (uint r = rb * RB; r < r1; ++r) {
        float acc[OUT + 1];
        for (int o = 0; o <= OUT; ++o) acc[o] = 0.0f;
        for (int e = 0; e < EPT; ++e) {
            int k = kb * (TG * EPT) + e * TG + tid;
            if (k < K) {
                float s = float(residual[r * K + k]);
                /*MERGE*/
                acc[OUT] += s * s;
                for (int o = 0; o < OUT; ++o) acc[o] += s * w[e][o];
            }
        }
        for (int o = 0; o <= OUT; ++o) {
            float v = simd_sum(acc[o]);
            if (lane == 0) red[o * (TG / 32) + simd] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <= OUT) {
            float v = 0.0f;
            for (int s2 = 0; s2 < TG / 32; ++s2) v += red[tid * (TG / 32) + s2];
            part[(r * KB + kb) * (OUT + 1) + tid] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# Per-row finalize: mixes from the partials (index order), MLX's sigmoid
# form, the stream collapse in stream order, then RMSNorm as
# ``mx.fast.rms_norm`` rounds it (normalize, round to T, scale by the weight
# in T).  consts = [hc eps (rms), hc_eps, magnitude, norm eps].
_FINALIZE_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint r = threadgroup_position_in_grid.x;
    uint simd = tid / 32, lane = tid % 32;
    threadgroup float mix[OUT + 1];
    threadgroup float pre[N];
    threadgroup float red[TG / 32];
    if (tid <= OUT) {
        float v = 0.0f;
        for (int b = 0; b < KB; ++b) v += part[(r * KB + b) * (OUT + 1) + tid];
        mix[tid] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < OUT) {
        float msq = mix[OUT] * (1.0f / float(K));
        float inv = metal::precise::rsqrt(msq + consts[0]);
        float m = mix[tid] * inv;
        float zs = tid < N ? m * scale[0] : m * scale[1];
        float z = zs + base[tid];
        float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
        float sg = z < 0.0f ? y : 1.0f - y;
        if (tid < N) {
            pre[tid] = sg + consts[1];
        } else {
            float ms = consts[2] * sg;
            post_out[r * N + (tid - N)] = ms + consts[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float c[PER];
    float sq = 0.0f;
    for (int p = 0; p < PER; ++p) {
        int d = p * TG + tid;
        float v = 0.0f;
        if (d < D) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j) {
                float t = pre[j] * float(streams[r * K + j * D + d]);
                acc += t;
            }
            v = float(static_cast<T>(acc));
        }
        c[p] = v;
        sq += v * v;
    }
    sq = simd_sum(sq);
    if (lane == 0) red[simd] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int s2 = 0; s2 < TG / 32; ++s2) total += red[s2];
    float mean = total / float(D);
    float nrm = metal::precise::rsqrt(mean + consts[3]);
    for (int p = 0; p < PER; ++p) {
        int d = p * TG + tid;
        if (d < D) normed[r * D + d] = static_cast<T>(c[p] * nrm) * norm_w[d];
    }
"""

_MERGE_SRC = """
                {
                    int j = k / D;
                    int d = k - j * D;
                    // Separate statements: Metal contracts a*b+c inside one
                    // expression into an fma; MLX's compiled merge rounds the
                    // product first (bitwise checked on M5 Max).
                    float t = post_prev[r * N + j] * float(branch[r * D + d]);
                    T m = static_cast<T>(t + s);
                    merged[r * K + k] = m;
                    s = float(m);
                }
"""

HC_TG = 256
HC_EPT = 4
HC_RB = 8


@lru_cache(maxsize=None)
def _partial_kernel(merge: bool):
    inputs = ["residual", "hc_fn"] + (["post_prev", "branch"] if merge else [])
    outputs = (["merged"] if merge else []) + ["part"]
    return mx.fast.metal_kernel(name=f"hy4_hc_seam_partial_m{int(merge)}", input_names=inputs,
                                output_names=outputs,
                                source=_PARTIAL_SRC.replace("/*MERGE*/", _MERGE_SRC if merge else ""))


@lru_cache(maxsize=None)
def _finalize_kernel():
    return mx.fast.metal_kernel(
        name="hy4_hc_seam_finalize",
        input_names=["part", "streams", "base", "scale", "norm_w", "consts"],
        output_names=["post_out", "normed"], source=_FINALIZE_SRC)


NO_GRAD_MESSAGE = ("Hy4-flash hc_seam_fused runs custom Metal kernels without a gradient; load the "
                   "model without hc_seam_fused for any differentiated forward (e.g. an aMoEba carve)")


@lru_cache(maxsize=None)
def _seam_op(merge: bool):
    """The seam kernels as one ``mx.custom_function`` whose VJP fails loudly."""

    @mx.custom_function
    def op(*arrays):
        if merge:
            residual, hc_fn, base, scale, norm_w, consts, post_prev, branch = arrays
        else:
            (residual, hc_fn, base, scale, norm_w, consts), post_prev, branch = arrays, None, None
        return _hc_seam_kernels(residual, hc_fn, base, scale, norm_w, consts, post_prev, branch)

    @op.vjp
    def op_vjp(primals, cotangents, outputs):
        raise ValueError(NO_GRAD_MESSAGE)

    @op.jvp
    def op_jvp(primals, tangents):
        raise ValueError(NO_GRAD_MESSAGE)

    return op


def hc_seam_metal(residual, hc_fn, base, scale, norm_w, consts, post_prev=None, branch=None):
    """Metal seam: ``(streams, post, normed)``; ``streams`` is ``residual`` without a merge.

    No gradient: a VJP/JVP through it raises ``ValueError`` (``NO_GRAD_MESSAGE``).
    """
    arrays = (residual, hc_fn, base, scale, norm_w, consts)
    if post_prev is not None:
        arrays += (post_prev, branch)
    return tuple(_seam_op(post_prev is not None)(*arrays))


def _hc_seam_kernels(residual, hc_fn, base, scale, norm_w, consts, post_prev=None, branch=None):
    *lead, n, D = residual.shape
    R = 1
    for s in lead:
        R *= int(s)
    K, OUT = n * D, int(hc_fn.shape[0])
    dtype = residual.dtype
    KB = -(-K // (HC_TG * HC_EPT))
    merge = post_prev is not None
    tmpl = [("T", dtype), ("OUT", OUT), ("N", n), ("D", D), ("K", K),
            ("KB", KB), ("RB", HC_RB), ("TG", HC_TG), ("EPT", HC_EPT)]
    ins = [residual.reshape(R, K), hc_fn]
    shapes, dtypes = [(R, KB, OUT + 1)], [mx.float32]
    if merge:
        ins += [post_prev.reshape(R, n).astype(mx.float32), branch.reshape(R, D)]
        shapes, dtypes = [(R, K)] + shapes, [dtype] + dtypes
    outs = _partial_kernel(merge)(inputs=ins, template=tmpl, grid=(KB * HC_TG, -(-R // HC_RB), 1),
                                  threadgroup=(HC_TG, 1, 1), output_shapes=shapes, output_dtypes=dtypes)
    streams = outs[0] if merge else residual.reshape(R, K)
    part = outs[-1]
    per = -(-D // HC_TG)
    post, normed = _finalize_kernel()(
        inputs=[part, streams, base, scale, norm_w.astype(dtype), consts],
        template=[("T", dtype), ("OUT", OUT), ("N", n), ("D", D), ("K", K), ("KB", KB),
                  ("TG", HC_TG), ("PER", per)],
        grid=(R * HC_TG, 1, 1), threadgroup=(HC_TG, 1, 1),
        output_shapes=[(R, n), (R, D)], output_dtypes=[mx.float32, dtype])
    return (streams.reshape(*lead, n, D), post.reshape(*lead, n), normed.reshape(*lead, D))


class HCSeam:
    """Fused iHC boundaries of one ``DecoderLayer`` (a plain object, not a module).

    On a CPU stream, or an activation dtype the kernels do not take, each
    call runs the reference ops unchanged (exact), counted as ``reference``.
    """

    def __init__(self):
        self.calls = {"metal": 0, "reference": 0}
        self._consts = {}

    def _args(self, hc, norm):
        p = hc.hc_pre
        key = (float(hc.eps), float(hc.hc_eps), float(hc.magnitude), float(norm.eps))
        consts = self._consts.get(key)
        if consts is None:
            consts = self._consts[key] = mx.array(key, dtype=mx.float32)
        return (p.hc_fn.astype(mx.float32), p.hc_base.astype(mx.float32), p.hc_scale.astype(mx.float32),
                norm.weight, consts)

    @staticmethod
    def _metal(streams):
        return on_gpu() and streams.dtype in (mx.bfloat16, mx.float16, mx.float32)

    def pre(self, hc, norm, streams):
        """``post, norm(collapse)`` of ``hc(streams)`` (a block entry: nothing to merge)."""
        if not self._metal(streams):
            self.calls["reference"] += 1
            post, x = hc(streams)
            return post, norm(x)
        self.calls["metal"] += 1
        _, post, normed = hc_seam_metal(streams, *self._args(hc, norm))
        return post, normed

    def merge_pre(self, hc, norm, post_prev, branch, residual):
        """``streams = merge(post_prev, branch, residual)``; then ``post, norm(collapse)``."""
        if not self._metal(residual):
            from .hy_v4_flash import hc_merge

            self.calls["reference"] += 1
            streams = hc_merge(post_prev, branch, residual)
            post, x = hc(streams)
            return streams, post, norm(x)
        self.calls["metal"] += 1
        return hc_seam_metal(residual, *self._args(hc, norm), post_prev, branch)


# ------------------------------------------------- fused input projections --

def _linear_format(module) -> tuple:
    if isinstance(module, nn.QuantizedLinear):
        return ("quantized", int(module.bits), int(module.group_size), str(getattr(module, "mode", "affine")),
                "biases" in module, "bias" in module)
    if type(module) is nn.Linear:
        return ("dense", str(module.weight.dtype), "bias" in module)
    return ("unsupported", type(module).__name__)


def fusion_refusal(modules: Sequence[nn.Module]) -> Optional[str]:
    """Why these linears cannot share one GEMM, or None."""
    formats = {_linear_format(m) for m in modules}
    if any(f[0] == "unsupported" for f in formats):
        return "unsupported_module:" + ",".join(sorted(f[1] for f in formats if f[0] == "unsupported"))
    if len(formats) != 1:
        return "format_mismatch:" + ";".join(sorted(str(f) for f in formats))
    widths = {int(m.weight.shape[-1]) for m in modules}
    if len(widths) != 1:
        return "input_width_mismatch"
    return None


# Rows (product of the leading axes) up to which one concatenated GEMM is
# bitwise the split projections on Metal (M5 Max, MLX 0.32.4, every segment
# compared, x [1, R, 3072] bf16, the real Hy4 projection widths): affine
# 3/4/6/8-bit quantized_matmul through 12 rows (its vector kernel; from 13
# rows the matrix kernel's tiling depends on the output width), bf16 matmul
# through 64 rows.  Larger inputs run each segment as its own GEMM (the
# split route), so the fusion never changes numerics; it serves decode and
# small verify forwards, which is where launch count matters.  These are
# caps, not promises: the kernel choice can depend on the chip and the MLX
# version, so ``calibrate`` re-measures each fused shape at load on the
# running device and lowers the bound (to 0: never fused) where it differs.
FUSED_MAX_ROWS = {"quantized": 12, "dense": 64}


class FusedLinear(nn.Module):
    """Concatenated linears (output axis); one GEMM, sliced per segment."""

    def __init__(self, modules: Sequence[nn.Module], names: Sequence[str], max_rows: Optional[int] = None):
        super().__init__()
        first = modules[0]
        self.quantized = isinstance(first, nn.QuantizedLinear)
        planes = ["weight"] + (["scales"] if self.quantized else []) + (
            ["biases"] if self.quantized and "biases" in first else []) + (["bias"] if "bias" in first else [])
        for p in planes:
            setattr(self, p, mx.concatenate([m[p] for m in modules], axis=0))
        if self.quantized:
            self.bits, self.group_size = int(first.bits), int(first.group_size)
            self.mode = str(getattr(first, "mode", "affine"))
        bounds, a = [], 0
        for m in modules:
            b = a + int(m["scales" if self.quantized else "weight"].shape[0])
            bounds.append((a, b))
            a = b
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "bounds", tuple(bounds))
        object.__setattr__(self, "_x", None)
        object.__setattr__(self, "_out", None)
        object.__setattr__(self, "bypass", False)
        object.__setattr__(self, "stats", {"fused_calls": 0, "segment_calls": 0})
        object.__setattr__(self, "_max_rows", max_rows)

    def _matmul(self, x, a=None, b=None):
        sl = slice(a, b)
        if self.quantized:
            out = mx.quantized_matmul(x, self["weight"][sl], self["scales"][sl],
                                      self["biases"][sl] if "biases" in self else None, transpose=True,
                                      group_size=self.group_size, bits=self.bits, mode=self.mode)
            if "bias" in self:
                out = out + self["bias"][sl]  # as nn.QuantizedLinear
            return out
        if "bias" in self:
            return mx.addmm(self["bias"][sl], x, self["weight"][sl].T)  # as nn.Linear
        return x @ self["weight"][sl].T

    @property
    def max_rows(self) -> int:
        if self._max_rows is not None:
            return self._max_rows
        return FUSED_MAX_ROWS["quantized" if self.quantized else "dense"]

    def signature(self) -> tuple:
        first = ("quantized", self.bits, self.group_size, self.mode) if self.quantized else ("dense",)
        return first + (int(self["weight"].shape[-1]), self.bounds, "bias" in self, self.max_rows)

    def self_check(self, dtype, seeds=(0, 1)) -> int:
        """Largest row count (up to ``max_rows``) at which the fused GEMM is bitwise the split ones.

        Runs this device's kernels on random input at the real widths: the
        MLX vector-vs-matrix kernel choice that sets the bound can depend
        on the chip, the MLX version and the output width.  Every row count
        is checked up to 16, then 24, 32, 48, 64, ... (the first mismatch
        stops the scan), on two random inputs.  A pass is evidence, not proof:
        where the fused and split shapes take different kernels, mismatches
        are data dependent and can be rare (bf16 dense at 8 rows differed on
        one input and not on another), so the static caps stay as the upper
        limit and the quantized vector kernel (per-row identical by
        construction) is the case the bound is meant for.
        """
        width = int(self["scales"].shape[-1]) * self.group_size if self.quantized else int(self["weight"].shape[-1])
        candidates = [r for r in list(range(1, 17)) + [24, 32, 48, 64, 128, 256, 512, 1024, 2048]
                      if r <= self.max_rows]
        good = 0
        for rows in candidates:
            same = True
            for seed in seeds:
                x = mx.random.normal((1, rows, width), key=mx.random.key(seed * 4099 + rows)).astype(dtype)
                full = self._matmul(x)
                parts = [self._matmul(x, a, b) for a, b in self.bounds]
                same = all(mx.array_equal(full[..., a:b], part).item()
                           for (a, b), part in zip(self.bounds, parts))
                if not same:
                    break
            if not same:
                break
            good = rows
        return good

    def set_max_rows(self, rows: int):
        object.__setattr__(self, "_max_rows", int(rows))

    @staticmethod
    def _rows(x) -> int:
        rows = 1
        for s in x.shape[:-1]:
            rows *= int(s)
        return rows

    def segments(self, x) -> List[mx.array]:
        """Every segment of ``x``'s projection: one GEMM up to ``max_rows``, else one per segment."""
        if self._rows(x) <= self.max_rows:
            self.stats["fused_calls"] += 1
            full = self._matmul(x)
            return [full[..., a:b] for a, b in self.bounds]
        self.stats["segment_calls"] += len(self.bounds)
        return [self._matmul(x, a, b) for a, b in self.bounds]

    def segment(self, x, index: int, *, lead: bool = False, last: bool = False):
        """Segment ``index`` of ``x``'s projection (one fused GEMM per input object)."""
        a, b = self.bounds[index]
        if self._x is x:
            out = self._out[..., a:b]
        elif lead and not self.bypass and self._rows(x) <= self.max_rows:
            full = self._matmul(x)
            object.__setattr__(self, "_x", x)
            object.__setattr__(self, "_out", full)
            self.stats["fused_calls"] += 1
            out = full[..., a:b]
        else:
            # Not the leading call on this input (a lane slice, or a call
            # site outside the fused order): this segment alone, exactly
            # the split projection.
            self.stats["segment_calls"] += 1
            out = self._matmul(x, a, b)
        if last:
            self.release()
        return out

    def release(self):
        object.__setattr__(self, "_x", None)
        object.__setattr__(self, "_out", None)


class SegmentLinear(nn.Module):
    """A weightless stand-in for one fused projection (its slice of ``FusedLinear``)."""

    def __init__(self, owner: FusedLinear, index: int, *, lead: bool = False, last: bool = False):
        super().__init__()
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "lead", lead)
        object.__setattr__(self, "last", last)

    def __call__(self, x):
        return self.owner.segment(x, self.index, lead=self.lead, last=self.last)


@lru_cache(maxsize=None)
def _gate_mul_kernel():
    def fn(out, gate):
        return out * mx.sigmoid(gate)
    return mx.compile(fn)


def gated_output(out, gate):
    """``out * sigmoid(gate)`` compiled (``Attention.finish``'s gate)."""
    return _gate_mul_kernel()(out, gate)


def fuse_mla_inputs(attn) -> Optional[str]:
    """Install one fused input GEMM on ``attn`` (an ``Attention``); refusal reason or None.

    Order of the fused rows: q_a, kv_a, [token_key, head_weight], linear_gate.
    ``q_a_proj`` leads (``project`` calls it first on the attention input);
    ``linear_gate`` is the last consumer (``finish``) and releases the cache.
    """
    names = ["q_a_proj", "kv_a_proj_with_mqa"]
    mods = [attn.q_a_proj, attn.kv_a_proj_with_mqa]
    ix = attn.get("compressed_indexer") if "compressed_indexer" in attn else None
    if ix is not None:
        extra = [ix.token_key_projection, ix.head_weight_projection]
        if fusion_refusal(mods + extra) is None:
            names += ["compressed_indexer.token_key_projection", "compressed_indexer.head_weight_projection"]
            mods += extra
    names.append("linear_gate")
    mods.append(attn.linear_gate)
    reason = fusion_refusal(mods)
    if reason is not None:
        return reason
    fused = FusedLinear(mods, names)
    mx.eval(fused.parameters())  # the split planes are released once the proxies replace them
    attn.fused_input_proj = fused
    last = len(names) - 1
    for i, name in enumerate(names):
        proxy = SegmentLinear(fused, i, lead=i == 0, last=i == last)
        if name.startswith("compressed_indexer."):
            setattr(ix, name.split(".", 1)[1], proxy)
        else:
            setattr(attn, name, proxy)
    return None


def fuse_engram_inputs(layer) -> Optional[str]:
    """One GEMM for ``key_projs[0..n)`` + ``value_proj`` of an ``EngramLayer``."""
    mods = list(layer.key_projs) + [layer.value_proj]
    reason = fusion_refusal(mods)
    if reason is not None:
        return reason
    # Dense bf16 addmm at the Engram widths (1536 -> 5 x 3072) was bitwise
    # the split projections from 1 through 2048 rows on Metal; quantized
    # (the 4.6-bit pack stores them 8-bit affine g64 with bias) only through
    # 12 rows (13-64 rows differ), so a quantized bank takes the quantized
    # bound.  Either bound is still lowered by the load-time self-check.
    quantized = isinstance(mods[0], nn.QuantizedLinear)
    fused = FusedLinear(mods, [f"key_projs.{c}" for c in range(layer.n)] + ["value_proj"],
                        max_rows=None if quantized else 2048)
    mx.eval(fused.parameters())
    layer.fused_proj = fused
    # Stacked norm gammas, built once (plain attributes, not parameters).
    for name in ("norm_key", "norm_query"):
        stacked = mx.stack([norm.weight for norm in layer[name]])
        mx.eval(stacked)
        object.__setattr__(layer, "stacked_" + name, stacked)
    layer.key_projs = [SegmentLinear(fused, c) for c in range(layer.n)]
    layer.value_proj = SegmentLinear(fused, layer.n)
    return None


# ------------------------------------------------------ indexer scores --

# logits[b, l, g] = sum_h w[b, l, h] * relu(q[b, h, l, :] . gk[b, g, :]) for
# g < own(l), else -inf.  One thread per group key; RB query rows per
# threadgroup share each key read, their q rows staged in threadgroup memory.
# The head sum runs in head order (the reference's reduction axis order);
# each dot runs in lane order over d, unlike the reference's GEMM tiles.
_INDEXER_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint g = threadgroup_position_in_grid.x * TG + tid;
    uint lb = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    uint L = w_shape[1];
    uint G = gk_shape[1];
    threadgroup float qs[RB * HEADS * DIM];
    uint l0 = lb * RB;
    for (uint i = tid; i < RB * HEADS * DIM; i += TG) {
        uint rr = i / (HEADS * DIM);
        uint h = (i / DIM) % HEADS;
        uint d = i % DIM;
        uint l = l0 + rr;
        qs[i] = l < L ? q[((b * HEADS + h) * L + l) * DIM + d] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g >= G) return;
    float acc[RB][HEADS];
    for (int rr = 0; rr < RB; ++rr)
        for (int h = 0; h < HEADS; ++h) acc[rr][h] = 0.0f;
    const device float4* kp = (const device float4*)(gk + (b * G + g) * DIM);
    for (int d4 = 0; d4 < DIM / 4; ++d4) {
        float4 kv = kp[d4];
        for (int rr = 0; rr < RB; ++rr) {
            for (int h = 0; h < HEADS; ++h) {
                const threadgroup float* qq = qs + (rr * HEADS + h) * DIM + d4 * 4;
                acc[rr][h] += qq[0] * kv.x + qq[1] * kv.y + qq[2] * kv.z + qq[3] * kv.w;
            }
        }
    }
    for (int rr = 0; rr < RB; ++rr) {
        uint l = l0 + rr;
        if (l >= L) break;
        float out;
        if (int(g) < own[l]) {
            out = 0.0f;
            for (int h = 0; h < HEADS; ++h) {
                float t = w[(b * L + l) * HEADS + h] * metal::max(acc[rr][h], 0.0f);
                out += t;
            }
        } else {
            out = -INFINITY;
        }
        logits[(b * L + l) * G + g] = NEG ? -out : out;
    }
"""

IDX_TG = 256
IDX_RB = 2


@lru_cache(maxsize=None)
def _indexer_kernel():
    return mx.fast.metal_kernel(name="hy4_indexer_logits", input_names=["q", "gk", "w", "own"],
                                output_names=["logits"], source=_INDEXER_SRC)


def indexer_logits_metal(q, gk, w, own, negate: bool = False):
    """``q [B, H, l, d]`` f32, ``gk [B, G, d]`` f32, ``w [B, l, H]`` f32, ``own [l]`` int32 -> ``[B, l, G]``.

    Masked logits (``-inf`` at and after each row's own group); ``negate``
    returns their exact negation (``+inf`` masked), the operand of top-k.
    """
    B, Hh, L, d = q.shape
    G = int(gk.shape[1])
    if d % 4 or int(gk.shape[-1]) != d:
        raise ValueError("indexer_fused_scores needs a head dim divisible by 4 shared by queries and keys")
    # The kernel reads keys as float4: fp32, row-contiguous (the input copy guarantees the latter).
    return _indexer_kernel()(
        inputs=[q.astype(mx.float32), gk.astype(mx.float32), w.astype(mx.float32), own.astype(mx.int32)],
        template=[("HEADS", int(Hh)), ("DIM", int(d)), ("TG", IDX_TG), ("RB", IDX_RB), ("NEG", bool(negate))],
        grid=(-(-G // IDX_TG) * IDX_TG, -(-L // IDX_RB), B), threadgroup=(IDX_TG, 1, 1),
        output_shapes=[(B, L, G)], output_dtypes=[mx.float32])[0]


def indexer_logits_reference(q, gk, w, own):
    """The unfused math, as ``CompressedIndexer.select`` writes it (CPU fallback)."""
    scores = nn.relu(q @ gk[:, None].swapaxes(-1, -2))
    logits = (w.swapaxes(1, 2)[..., None] * scores).sum(axis=1)
    valid = mx.arange(gk.shape[1])[None, None, :] < own[None, :, None]
    return mx.where(valid, logits, -mx.inf)


# --------------------------------------------------------------- install --

def install(model, *, hc_seam: bool = False, mla_inputs: bool = False, engram_inputs: bool = False,
            indexer_scores: bool = False, engram_gate_rounding: str = "fp32") -> Dict:
    """Install the selected fusions on a loaded ``hy_v4_flash.Model`` (after its strict load).

    Returns (and keeps as ``model.fusion_record``) the load record.
    """
    if engram_gate_rounding not in ENGRAM_GATE_ROUNDINGS:
        raise ValueError(f"engram_gate_rounding must be one of {ENGRAM_GATE_ROUNDINGS}")
    record: Dict = {"hc_seam_fused": bool(hc_seam), "mla_fused_input_proj": {}, "engram_fused_proj": {},
                    "indexer_fused_scores": bool(indexer_scores), "engram_gate_rounding": engram_gate_rounding}
    for index, layer in enumerate(model.model.layers):
        if hc_seam:
            layer.hc_seam = HCSeam()
        if mla_inputs and not isinstance(getattr(layer.self_attn, "fused_input_proj", None), FusedLinear):
            reason = fuse_mla_inputs(layer.self_attn)
            record["mla_fused_input_proj"][index] = reason or "fused"
        if "compressed_indexer" in layer.self_attn:
            layer.self_attn.compressed_indexer.fused_scores = bool(indexer_scores)
        if "engram_layer" in layer:
            eng = layer.engram_layer
            eng.gate_rounding = engram_gate_rounding
            if engram_inputs and not isinstance(getattr(eng, "fused_proj", None), FusedLinear):
                reason = fuse_engram_inputs(eng)
                # Only a fused layer takes the fused path; a refused one keeps its split projections.
                eng.fused = reason is None
                record["engram_fused_proj"][index] = reason or "fused"
    record["row_bounds"] = calibrate(model)
    record["installed_all"] = bool(hc_seam and mla_inputs and engram_inputs)
    object.__setattr__(model, "fusion_record", record)
    return record


def _fused_modules(model):
    out = []
    for index, layer in enumerate(model.model.layers):
        f = getattr(layer.self_attn, "fused_input_proj", None)
        if isinstance(f, FusedLinear):
            out.append((f"layers.{index}.self_attn", f))
        if "engram_layer" in layer:
            f = getattr(layer.engram_layer, "fused_proj", None)
            if isinstance(f, FusedLinear):
                out.append((f"layers.{index}.engram_layer", f))
    return out


def calibrate(model) -> Dict:
    """Load-time self-check: set each fused GEMM's row bound on this device (cached per shape).

    Returns the receipt record: device, MLX version, and per shape the
    static bound, the measured one and the layers it applies to.
    """
    modules = _fused_modules(model)
    if not modules:
        return {}
    dtype = model.model.embed_tokens.weight.dtype
    if dtype not in (mx.bfloat16, mx.float16, mx.float32):
        dtype = mx.bfloat16
    try:
        device = mx.device_info().get("device_name") if on_gpu() else "cpu"
    except Exception:  # noqa: BLE001
        device = None
    shapes: Dict[tuple, Dict] = {}
    for name, fused in modules:
        sig = fused.signature()
        entry = shapes.get(sig)
        if entry is None:
            static = fused.max_rows
            measured = fused.self_check(dtype)
            entry = shapes[sig] = {"format": str(sig[:-4]), "input_width": sig[-4],
                                   "output_widths": [b - a for a, b in fused.bounds],
                                   "static_bound": static, "measured_bound": measured, "modules": []}
        fused.set_max_rows(entry["measured_bound"])
        object.__setattr__(fused, "calibrated_rows", entry["measured_bound"])
        entry["modules"].append(name)
    return {"device": device, "mlx": mx.__version__, "dtype": str(dtype), "shapes": list(shapes.values())}


def set_active(model, *, hc_seam: bool = False, mla_inputs: bool = False, engram_inputs: bool = False,
               indexer_scores: bool = False, engram_gate_rounding: str = "fp32") -> Dict:
    """Switch installed fusions on or off in place (research A/B: one load, several arms).

    Requires ``install`` with every fusion first (``hc_seam``, ``mla_inputs``,
    ``engram_inputs`` true).  Off restores the reference exactly: the seam
    runs the unfused ops, a fused GEMM's row bound becomes 0 (each segment is
    its own GEMM, the split route), the Engram layer takes its split path,
    the indexer its unfused scores and the gate the default rounding.
    Returns the active settings (also kept in ``model.fusion_record``).
    """
    if engram_gate_rounding not in ENGRAM_GATE_ROUNDINGS:
        raise ValueError(f"engram_gate_rounding must be one of {ENGRAM_GATE_ROUNDINGS}")
    record = getattr(model, "fusion_record", None)
    if not record or not record.get("installed_all"):
        raise ValueError("set_active needs install(hc_seam=True, mla_inputs=True, engram_inputs=True) first")
    for layer in model.model.layers:
        layer.hc_seam = HCSeam() if hc_seam else None
        fused = getattr(layer.self_attn, "fused_input_proj", None)
        if isinstance(fused, FusedLinear):
            fused.set_max_rows(fused.calibrated_rows if mla_inputs else 0)
        if "compressed_indexer" in layer.self_attn:
            layer.self_attn.compressed_indexer.fused_scores = bool(indexer_scores)
        if "engram_layer" in layer:
            eng = layer.engram_layer
            eng.gate_rounding = engram_gate_rounding
            proj = getattr(eng, "fused_proj", None)
            if isinstance(proj, FusedLinear):
                eng.fused = bool(engram_inputs)
                proj.set_max_rows(proj.calibrated_rows if engram_inputs else 0)
    active = {"hc_seam_fused": bool(hc_seam), "mla_fused_input_proj": bool(mla_inputs),
              "engram_fused_proj": bool(engram_inputs), "indexer_fused_scores": bool(indexer_scores),
              "engram_gate_rounding": engram_gate_rounding}
    record["active"] = active
    record["hc_seam_fused"] = bool(hc_seam)
    record["indexer_fused_scores"] = bool(indexer_scores)
    record["engram_gate_rounding"] = engram_gate_rounding
    return active


def status(model) -> Dict:
    """Route record (implemented / qualified / selected / observed_used) of every fusion."""
    record = getattr(model, "fusion_record", None) or {}
    layers = list(model.model.layers)
    seams = [layer.hc_seam for layer in layers if getattr(layer, "hc_seam", None) is not None]
    seam_calls = {k: sum(s.calls[k] for s in seams) for k in ("metal", "reference")}
    mla = [layer.self_attn.fused_input_proj for layer in layers
           if isinstance(getattr(layer.self_attn, "fused_input_proj", None), FusedLinear)]
    eng = [layer.engram_layer.fused_proj for layer in layers
           if "engram_layer" in layer and isinstance(getattr(layer.engram_layer, "fused_proj", None), FusedLinear)]
    ix = [layer.self_attn.compressed_indexer for layer in layers if "compressed_indexer" in layer.self_attn]
    ix_calls = sum(getattr(i, "fused_score_calls", 0) for i in ix)
    rounding = record.get("engram_gate_rounding", "fp32")
    eng_layers = [layer.engram_layer for layer in layers if "engram_layer" in layer]
    return {
        "hc_seam_fused": {
            "implemented": True, "qualified": False, "selected": bool(record.get("hc_seam_fused")),
            "calls": seam_calls, "observed_used": seam_calls["metal"] > 0,
            "exactness": ("merged streams bitwise; post gates differ by fp32 split-K reassociation "
                          "(<= ~3e-6), normed input flips 1 bf16 ulp on ~0.01-0.02% of elements "
                          "(tests/test_hy_v4_flash_fusions_metal.py); CPU runs the reference"),
        },
        "mla_fused_input_proj": {
            "implemented": True, "qualified": False, "selected": bool(record.get("mla_fused_input_proj")),
            "layers": dict(record.get("mla_fused_input_proj", {})),
            "fused_calls": sum(f.stats["fused_calls"] for f in mla),
            "segment_calls": sum(f.stats["segment_calls"] for f in mla),
            "observed_used": sum(f.stats["fused_calls"] for f in mla) > 0,
            "row_bounds": record.get("row_bounds", {}),
            "exactness": ("bitwise: one GEMM only up to the row bound the load-time self-check measured "
                          "on this device (static caps FUSED_MAX_ROWS), split GEMMs above it"),
        },
        "engram_fused_proj": {
            "implemented": True, "qualified": False, "selected": bool(record.get("engram_fused_proj")),
            "layers": dict(record.get("engram_fused_proj", {})),
            "fused_calls": sum(f.stats["fused_calls"] for f in eng),
            "segment_calls": sum(f.stats["segment_calls"] for f in eng),
            "observed_used": sum(f.stats["fused_calls"] for f in eng) > 0,
            "exactness": ("bitwise: one GEMM only up to the row bound the load-time self-check measured "
                          "on this device (static caps: 12 rows quantized, 2048 dense), split GEMMs above"),
        },
        "indexer_fused_scores": {
            "implemented": True, "qualified": False, "selected": bool(record.get("indexer_fused_scores")),
            "fused_score_calls": ix_calls, "observed_used": ix_calls > 0,
            "exactness": ("logits differ from the reference by fp32 dot reassociation; top-k is the same "
                          "argpartition, so selections differ only at near-ties of the k-th logit"),
        },
        "engram_gate_rounding": {
            "implemented": True, "qualified": False, "mode": rounding, "selected": rounding != "fp32",
            "observed_used": rounding != "fp32" and any(getattr(e, "gate_calls", 0) for e in eng_layers),
            "applies_to": "engram_precision transformers with bf16/fp16 activations",
        },
    }
