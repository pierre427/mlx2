# SPDX-License-Identifier: Apache-2.0 AND MIT
# Row/chain/combine structure adapted from jundot/omlx PR #4052 (Apache-2.0)
# at d05afc7a; the per-row window layout from jundot/omlx PRs #4041 / #4105
# (Apache-2.0; heads 7ef4d997 / 93e708a8, ``attn_fused.dense_sdpa_gate``).
# The arithmetic inside the Metal sources is transcribed from MLX 39400a0d4
# ``sdpa_vector.h`` (MIT).  See docs/PROVENANCE.md,
# provenance/omlx-4052-qwen4-attn-rows.json,
# provenance/omlx-4041-4105-qwen4-attn-window.json,
# provenance/muse-d128-attention-window.json and
# provenance/sdpa-vector-d512-tk2-20261122.json.
"""One generator for mlx2's transcriptions of MLX's vector SDPA (Metal text).

The arithmetic is MLX 39400a0d4 ``sdpa_vector.h`` (``sdpa_vector``,
``sdpa_vector_2pass_1``, ``sdpa_vector_2pass_2``; MIT, see
docs/PROVENANCE.md).  mlx2 carried it three times -- Flash-Next attention
rows (``qwen4_attn_rows``, adapted from jundot/omlx #4052), the Flash-Next
row-exact window (``qwen4_attn_window``, patching the rows' text), Muse's
D128 window (``muse_attn_window``) -- plus the TK-2 head-dim-512 decode pair
(``sdpa_vector_d512``).  Every one of those sources is now produced here, and
``tests/test_technologies_window_attention_sources.py`` proves each generated
string is byte-identical to the string the old module compiled
(``tests/fixtures/window_attention_legacy_kernels.json``).

What a source varies in (all named for what they do, none for a model):

``key_count``
    ``"cache_view"``: a row attends to every key of the K/V view
    (``N = keys_shape[2]``).  ``"window_rows"``: row ``r`` of a verify window
    attends to exactly ``n_first + r`` keys -- the physical reduction domain
    of its serial one-token decode step.
``two_pass_rows``
    Where pass 1 finds its query row: ``"threadgroup"`` (simdgroup ``z`` of
    the threadgroup, MLX's layout for several rows of one call) or
    ``"grid"`` (grid ``z`` divided by ``BLOCKS``, one independent one-row
    plan per row).
``mask``
    ``"none"``, ``"broadcast"`` (a ``[1|B, 1|H, 1|R, 1|N]`` bool array read
    through its strides, MLX's ``bmask``), or ``"window_concat"`` (the
    window rows' one-token mask rows concatenated flat, row ``r`` holding
    ``n_first + r`` entries).
``store``
    ``"plain"`` or ``"gate"`` (the closing store times an already-sigmoided
    gate, Flash-Next's output gate folded into the reduction).
``output``
    ``"rows"`` (``[R, H * D]`` written by (head, row) threadgroups) or
    ``"single_row"`` (MLX's ``[B * H, D]`` two-pass output).
:class:`SourceStyle`
    Text-only conventions of each historical transcription (macro name,
    comments, blank lines, indentation).  They change no compiled
    arithmetic; they are kept so every source stays byte-identical to the
    one whose GPU evidence exists.  Folding them is a zero-numerics cleanup
    that would only need a source-identity rebind (Muse's canary key).

Head dim, GQA, heads per simdgroup, simdgroups per threadgroup, block count
and dtype stay Metal template arguments (``D``, ``GQA``, ``HPT``, ``SG``,
``BLOCKS``, ``T``), exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SourceStyle",
    "VectorKernelFamily",
    "ROWS_STYLE",
    "COMPACT_STYLE",
    "key_step_header",
    "one_pass_chains_source",
    "two_pass_chains_source",
    "one_pass_combine_source",
    "two_pass_combine_source",
    "strided_tk_pass1_source",
    "kernel_kwargs",
    "KEY_COUNTS",
    "TWO_PASS_ROWS",
    "MASKS",
    "STORES",
    "OUTPUTS",
]

KEY_COUNTS = ("cache_view", "window_rows")
TWO_PASS_ROWS = ("threadgroup", "grid")
MASKS = ("none", "broadcast", "window_concat")
STORES = ("plain", "gate")
OUTPUTS = ("rows", "single_row")


@dataclass(frozen=True)
class SourceStyle:
    """Text conventions of one transcription (no arithmetic effect)."""

    key_step_macro: str
    # Spaces between the macro's parameter list and its line continuation.
    define_pad: int
    # Inline ``//`` comments of the annotated transcription.
    comments: bool
    # Blank lines between the two-pass combine's reduction phases.
    combine_blank_lines: bool
    # Indentation of the combine's store statement.
    store_indent: str


# omlx #4052's annotated transcription (Flash-Next rows and window).
ROWS_STYLE = SourceStyle("ATTN_ROWS_KEY_STEP", 20, True, True, "")
# Muse's compact copy of the same text.
COMPACT_STYLE = SourceStyle("MUSE_D128_KEY_STEP", 19, False, False, " " * 12)


def _check(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}")


# --------------------------------------------------------------------------
# The per-key update both passes share (sdpa_vector / sdpa_vector_2pass_1)
# --------------------------------------------------------------------------

_KEY_STEP_BODY = r"""    {                                                                    \
        U score = 0;                                                     \
        for (int e = 0; e < PT; e++) {                                   \
            score += QV[e] * KP[e];                                      \
        }                                                                \
        score = simd_sum(score);                                         \
        U new_max = max(MAXV, score);                                    \
        U factor = fast::exp(MAXV - new_max);                            \
        U exp_score = fast::exp(score - new_max);                        \
        MAXV = new_max;                                                  \
        SUMV = SUMV * factor + exp_score;                                \
        for (int e = 0; e < PT; e++) {                                   \
            OV[e] = OV[e] * factor + exp_score * VP[e];                  \
        }                                                                \
    }
"""


def key_step_header(style: SourceStyle) -> str:
    """The ``#define`` of the online-softmax key step (kernel ``header``)."""
    return (
        f"\n#define {style.key_step_macro}(QV, KP, VP, MAXV, SUMV, OV)"
        + " " * style.define_pad
        + "\\\n"
        + _KEY_STEP_BODY
    )


# --------------------------------------------------------------------------
# Pass sources (templates; ``@...@`` markers are specialised below)
# --------------------------------------------------------------------------

_N = {
    "cache_view": "const int N = int(keys_shape[2]);",
    "window_rows": "const int N = n_first[0] + row;",
}

# sdpa_vector: simdgroup s of a head's threadgroup visits keys s, s + 32, ...
# Here chain s of HPT query heads (sharing one KV head) runs in its own
# simdgroup; its state goes to the combine launch.
_ONE_PASS_CHAINS = r"""
    constexpr int BN = 32;
    constexpr int PT = D / 32;
    typedef float U;
    const uint lane = thread_index_in_simdgroup;
    const uint f = threadgroup_position_in_grid.x * SG + simdgroup_index_in_threadgroup;
    const int s = int(f % BN);
    const int hg = int(f / BN);
    const int kv = threadgroup_position_in_grid.y;
    const int row = threadgroup_position_in_grid.z;
    const int rows = threadgroups_per_grid.z;
    const int H = GQA * int(threadgroups_per_grid.y);
    @N@
    const float sc = scale[0];

    U q[HPT][PT];
    U o[HPT][PT];
    U max_score[HPT];
    U sum_exp_score[HPT];
    for (int j = 0; j < HPT; j++) {
        const int head = kv * GQA + hg * HPT + j;
        const device T* qp = queries + (size_t)head * queries_strides[1]
            + (size_t)row * queries_strides[2] + lane * PT;
        for (int e = 0; e < PT; e++) {
            q[j][e] = static_cast<U>(sc) * qp[e];
            o[j][e] = 0;
        }
        max_score[j] = Limits<U>::finite_min;
        sum_exp_score[j] = 0;
    }
    const device T* kp = keys + kv * keys_strides[1] + s * keys_strides[2] + lane * PT;
    const device T* vp = values + kv * values_strides[1] + s * values_strides[2] + lane * PT;
    const int k_step = BN * int(keys_strides[2]);
    const int v_step = BN * int(values_strides[2]);
@MASK_SETUP@
    for (int i = s; i < N; i += BN) {
        if (@USE_KEY@) {
            for (int j = 0; j < HPT; j++) {
                @MACRO@(q[j], kp, vp, max_score[j], sum_exp_score[j], o[j])
            }
        }
        kp += k_step;
        vp += v_step;
        @MASK_ADVANCE@
    }

    for (int j = 0; j < HPT; j++) {
        const int head = kv * GQA + hg * HPT + j;
        const size_t c = (size_t)(row * H + head) * BN + s;
        if (lane == 0) {
            maxs[c] = max_score[j];
            sums[c] = sum_exp_score[j];
        }
@C:        // [row, head, dim, chain]: the combine reads one dim's 32 chains.
        for (int e = 0; e < PT; e++) {
            outs[((size_t)(row * H + head) * D + lane * PT + e) * BN + s] = o[j][e];
        }
    }
"""

_TWO_PASS_ROW_LAYOUT = {
    "threadgroup": (
        "const int block = threadgroup_position_in_grid.z;",
        "const int row = thread_position_in_threadgroup.z;",
        "const int rows = threads_per_threadgroup.z;",
    ),
    "grid": (
        "const int block = int(threadgroup_position_in_grid.z) % BLOCKS;",
        "const int row = int(threadgroup_position_in_grid.z) / BLOCKS;",
        "const int rows = int(threadgroups_per_grid.z) / BLOCKS;",
    ),
}

# sdpa_vector_2pass_1: threadgroup (KV head, block), simdgroup (query head g of
# that KV head, row); keys block, block + BLOCKS, ...
_TWO_PASS_CHAINS = r"""
    constexpr int PT = D / 32;
    typedef float U;
    const uint lane = thread_index_in_simdgroup;
    const int kv = threadgroup_position_in_grid.x;
    @BLOCK@
    const int g = thread_position_in_threadgroup.y;
    @ROW@
    @ROWS@
    const int head = GQA * kv + g;
    @N@
    const float sc = scale[0];
    const int o_offset = head * rows + row;

    U q[PT];
    U o[PT] = {0};
    const device T* qp = queries + (size_t)head * queries_strides[1]
        + (size_t)row * queries_strides[2] + lane * PT;
    for (int e = 0; e < PT; e++) {
        q[e] = static_cast<U>(sc) * qp[e];
    }
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    const device T* kp = keys + kv * keys_strides[1] + block * keys_strides[2] + lane * PT;
    const device T* vp = values + kv * values_strides[1] + block * values_strides[2] + lane * PT;
    const int k_step = BLOCKS * int(keys_strides[2]);
    const int v_step = BLOCKS * int(values_strides[2]);
@MASK_SETUP@
    for (int i = block; i < N; i += BLOCKS) {
        if (@USE_KEY@) {
            @MACRO@(q, kp, vp, max_score, sum_exp_score, o)
        }
        kp += k_step;
        vp += v_step;
        @MASK_ADVANCE@
    }
    const size_t c = (size_t)o_offset * BLOCKS + block;
    if (lane == 0) {
        sums[c] = sum_exp_score;
        maxs[c] = max_score;
    }
    for (int e = 0; e < PT; e++) {
        partials[c * D + lane * PT + e] = static_cast<T>(o[e]);
    }
"""

# sdpa_vector's closing reduction for one (head, row).
_ONE_PASS_COMBINE = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int PT = D / BD;
    typedef float U;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int head = threadgroup_position_in_grid.x;
    const int row = threadgroup_position_in_grid.y;
    const int H = threadgroups_per_grid.x;
    const size_t c0 = (size_t)(row * H + head) * BN;

    U max_score = maxs[c0 + simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    U sum_exp_score = simd_sum(sums[c0 + simd_lid] * factor);
    U o[PT];
    for (int i = 0; i < PT; i++) {
@C:        // MLX's outputs[simd_gid * BD + simd_lid]: dim simd_gid * PT + i of
@C:        // chain simd_lid.
        o[i] = simd_sum(
            outs[((size_t)(row * H + head) * D + simd_gid * PT + i) * BN + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
    if (simd_lid == 0) {
        const size_t base = (size_t)row * (H * D) + head * D + simd_gid * PT;
        for (int i = 0; i < PT; i++) {
@STORE@
        }
    }
"""

_TWO_PASS_COMBINE_INDEX = {
    "rows": r"""    const int head = threadgroup_position_in_grid.x;
    const int row = threadgroup_position_in_grid.y;
    const int H = threadgroups_per_grid.x;
    const int rows = threadgroups_per_grid.y;
    const size_t q_offset = (size_t)head * rows + row;
    const device T* pp = partials + q_offset * BLOCKS * D + simd_gid * D
        + simd_lid * elem_per_thread;
    const device float* sp = sums + q_offset * BLOCKS;
    const device float* mp = maxs + q_offset * BLOCKS;
""",
    "single_row": r"""    const int q_offset = threadgroup_position_in_grid.x;
    const device T* pp = partials + q_offset * BLOCKS * D + simd_gid * D + simd_lid * elem_per_thread;
    const device float* sp = sums + q_offset * BLOCKS;
    const device float* mp = maxs + q_offset * BLOCKS;
    device T* op = out + q_offset * D + simd_gid * elem_per_thread;
""",
}

_TWO_PASS_COMBINE_STORE = {
    "rows": r"""    if (simd_lid == 0) {
        const size_t base = (size_t)row * (H * D) + head * D + simd_gid * elem_per_thread;
        for (int i = 0; i < elem_per_thread; i++) {
@STORE@
        }
    }
""",
    "single_row": r"""    if (simd_lid == 0) {
        for (int i = 0; i < elem_per_thread; i++) {
@STORE@
        }
    }
""",
}

# sdpa_vector_2pass_2 for one (head, row) or one MLX output row.
_TWO_PASS_COMBINE = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D / BD;
    typedef float U;

    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
@INDEX@
    U sum_exp_score = 0.0;
    U max_score = Limits<U>::finite_min;
@SEP@
    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, mp[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);
@SEP@
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * sp[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);
@SEP@
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_gid] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(pp[i]);
        }
        mp += BN;
        sp += BN;
        pp += BN * D;
    }
@SEP@
    for (int i = 0; i < elem_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
@SEP@
@STORE_BLOCK@"""


def _comments(source: str, style: SourceStyle) -> str:
    lines = []
    for line in source.split("\n"):
        if line.startswith("@C:"):
            if style.comments:
                lines.append(line[3:])
            continue
        lines.append(line)
    return "\n".join(lines)


def _mask(source: str, mask: str, chain: str, step: str) -> str:
    """MLX's ``use_key = bmask[0]`` walk for the requested mask layout."""
    _check("mask", mask, MASKS)
    if mask == "broadcast":
        setup = (
            "const int m_row = mask_shape[2] > 1 ? int(mask_strides[2]) : 0;\n"
            "    const int m_key = mask_shape[3] > 1 ? int(mask_strides[3]) : 0;\n"
            # Tiny masks arrive in the constant address space.
            f"    auto mp = mask + row * m_row + {chain} * m_key;\n"
            f"    const int m_step = {step} * m_key;"
        )
        use_key, advance = "mp[0]", "mp += m_step;"
    elif mask == "window_concat":
        # Row ``row`` of the group: its mask starts after the masks of the
        # rows before it, whose lengths are n_first, n_first + 1, ...
        setup = (
            "auto mp = masks + mask_base[0] + row * n_first[0]"
            f" + (row * (row - 1)) / 2 + {chain};\n"
            f"    const int m_step = {step};"
        )
        use_key, advance = "mp[0]", "mp += m_step;"
    else:
        setup, use_key, advance = "", "true", ""
    return (
        source.replace("@MASK_SETUP@", setup)
        .replace("@USE_KEY@", use_key)
        .replace("@MASK_ADVANCE@", advance)
    )


def one_pass_chains_source(*, key_count: str, mask: str, style: SourceStyle) -> str:
    """``sdpa_vector`` chains: one simdgroup per (32-key stride, HPT heads)."""
    _check("key_count", key_count, KEY_COUNTS)
    source = _ONE_PASS_CHAINS.replace("@N@", _N[key_count]).replace("@MACRO@", style.key_step_macro)
    return _comments(_mask(source, mask, "s", "BN"), style)


def two_pass_chains_source(
    *, key_count: str, two_pass_rows: str, mask: str, style: SourceStyle
) -> str:
    """``sdpa_vector_2pass_1``: one simdgroup per (query head, row) per block."""
    _check("key_count", key_count, KEY_COUNTS)
    _check("two_pass_rows", two_pass_rows, TWO_PASS_ROWS)
    block, row, rows = _TWO_PASS_ROW_LAYOUT[two_pass_rows]
    source = (
        _TWO_PASS_CHAINS.replace("@BLOCK@", block)
        .replace("@ROW@", row)
        .replace("@ROWS@", rows)
        .replace("@N@", _N[key_count])
        .replace("@MACRO@", style.key_step_macro)
    )
    return _comments(_mask(source, mask, "block", "BLOCKS"), style)


def _store(store: str, style: SourceStyle, target: str = "out[base + i]") -> str:
    _check("store", store, STORES)
    value = "static_cast<T>(o[i])"
    if store == "gate":
        value += " * gate[base + i]"
    return f"{style.store_indent}{target} = {value};"


def one_pass_combine_source(*, store: str, style: SourceStyle) -> str:
    """``sdpa_vector``'s closing reduction over a head's 32 chains."""
    return _comments(_ONE_PASS_COMBINE.replace("@STORE@", _store(store, style)), style)


def two_pass_combine_source(*, store: str, output: str, style: SourceStyle) -> str:
    """``sdpa_vector_2pass_2`` over ``BLOCKS`` partials."""
    _check("output", output, OUTPUTS)
    if output == "single_row" and store != "plain":
        raise ValueError("the single-row two-pass combine has no gated store")
    target = "out[base + i]" if output == "rows" else "op[i]"
    sep = "\n" if style.combine_blank_lines else ""
    return (
        _TWO_PASS_COMBINE.replace("@INDEX@\n", _TWO_PASS_COMBINE_INDEX[output] + "\n")
        .replace("@SEP@\n", sep)
        .replace("@STORE_BLOCK@", _TWO_PASS_COMBINE_STORE[output])
        .replace("@STORE@", _store(store, style, target))
    )


# --------------------------------------------------------------------------
# MLX 0.32.4 sdpa_vector_2pass_1 with a TK-key unroll, strided addressing
# --------------------------------------------------------------------------

# One query row per (batch, head); every input is addressed through its
# strides, including the element stride, with 64-bit token offsets; the scale
# is read from a buffer; ``TK`` is a template argument (``sdpa_vector_d512``
# compiles it at 2, MLX at 4 above head dim 256).  The key visit order is
# MLX's for any TK.
_STRIDED_TK_PASS1 = r"""
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = D / BD;
    typedef float U;

    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    const int kv_head_idx = threadgroup_position_in_grid.x;
    const int batch_idx = threadgroup_position_in_grid.y;
    const int block_idx = threadgroup_position_in_grid.z;
    const int blocks = threadgroups_per_grid.z;
    const int gqa_factor = threads_per_threadgroup.y;
    const int q_head_idx = gqa_factor * kv_head_idx + thread_position_in_threadgroup.y;
    const int num_kv_heads = threadgroups_per_grid.x;
    const int num_q_heads = num_kv_heads * gqa_factor;
    const int o_offset = batch_idx * num_q_heads + q_head_idx;
    const uint simd_lid = thread_index_in_simdgroup;
    const int N = keys_shape[2];
    const int64_t k_seq_stride = keys_strides[2];
    const int64_t v_seq_stride = values_strides[2];
    const int64_t qd = queries_strides[3];
    const int64_t kd = keys_strides[3];
    const int64_t vd = values_strides[3];

    const device T* qp = queries + batch_idx * queries_strides[0]
        + q_head_idx * queries_strides[1] + simd_lid * qk_per_thread * qd;
    const device T* kp = keys + batch_idx * keys_strides[0] + kv_head_idx * keys_strides[1]
        + block_idx * k_seq_stride + simd_lid * qk_per_thread * kd;
    const device T* vp = values + batch_idx * values_strides[0]
        + kv_head_idx * values_strides[1] + block_idx * v_seq_stride
        + simd_lid * v_per_thread * vd;
    device T* op = out + o_offset * blocks * D + block_idx * D + simd_lid * v_per_thread;

    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = static_cast<U>(scale[0]) * qp[i * qd];
    }

    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;

    const int64_t stride_k_tok = blocks * k_seq_stride;
    const int64_t stride_v_tok = blocks * v_seq_stride;

    int i = block_idx;
    for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
        T k_reg[TK][qk_per_thread];
        T v_reg[TK][v_per_thread];
        for (int t = 0; t < TK; t++) {
            const device T* kt = kp + t * stride_k_tok;
            for (int j = 0; j < qk_per_thread; j++) { k_reg[t][j] = kt[j * kd]; }
        }
        for (int t = 0; t < TK; t++) {
            const device T* vt = vp + t * stride_v_tok;
            for (int j = 0; j < v_per_thread; j++) { v_reg[t][j] = vt[j * vd]; }
        }
        for (int t = 0; t < TK; t++) {
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) {
                score += q[j] * static_cast<U>(k_reg[t][j]);
            }
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < v_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
            }
        }
        kp += TK * stride_k_tok;
        vp += TK * stride_v_tok;
    }

    for (; i < N; i += blocks) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) { score += q[j] * kp[j * kd]; }
        score = simd_sum(score);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vp[j * vd];
        }
        kp += stride_k_tok;
        vp += stride_v_tok;
    }

    if (simd_lid == 0) {
        sums[o_offset * blocks + block_idx] = sum_exp_score;
        maxs[o_offset * blocks + block_idx] = max_score;
    }
    for (int j = 0; j < v_per_thread; j++) {
        op[j] = static_cast<T>(o[j]);
    }
"""


def strided_tk_pass1_source() -> str:
    """MLX's one-row ``sdpa_vector_2pass_1`` with a ``TK`` unroll (no mask,
    no causal, no sinks), every input addressed through its strides."""
    return _STRIDED_TK_PASS1


# --------------------------------------------------------------------------
# Kernel families: the ``mx.fast.metal_kernel`` arguments of each kernel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorKernelFamily:
    """One set of compiled vector-SDPA kernels (a kernel name prefix plus
    the source parameters its chains use).

    ``chain_mask`` is the layout a ``masked=True`` chain reads; the unmasked
    chain is always generated with ``mask="none"``.
    """

    name_prefix: str
    key_count: str
    two_pass_rows: str
    chain_mask: str
    style: SourceStyle

    def __post_init__(self):
        _check("key_count", self.key_count, KEY_COUNTS)
        _check("two_pass_rows", self.two_pass_rows, TWO_PASS_ROWS)
        _check("chain_mask", self.chain_mask, MASKS)

    @property
    def chain_inputs(self) -> list:
        names = ["queries", "keys", "values", "scale"]
        if self.key_count == "window_rows":
            names.append("n_first")
        return names

    @property
    def mask_inputs(self) -> list:
        return ["mask"] if self.chain_mask == "broadcast" else ["masks", "mask_base"]


STAGES = ("one_chains", "two_chains", "one_combine", "two_combine")


def kernel_kwargs(
    family: VectorKernelFamily, stage: str, *, masked: bool = False, gated: bool = False
) -> dict:
    """The exact ``mx.fast.metal_kernel`` keyword arguments of one kernel."""
    _check("stage", stage, STAGES)
    style = family.style
    passes = stage[:3]
    number = "1" if passes == "one" else "2"
    if stage.endswith("chains"):
        if gated:
            raise ValueError("chains have no gate")
        mask = family.chain_mask if masked else "none"
        if passes == "one":
            source = one_pass_chains_source(key_count=family.key_count, mask=mask, style=style)
            outputs = ["maxs", "sums", "outs"]
        else:
            source = two_pass_chains_source(
                key_count=family.key_count,
                two_pass_rows=family.two_pass_rows,
                mask=mask,
                style=style,
            )
            outputs = ["partials", "sums", "maxs"]
        return dict(
            name=f"{family.name_prefix}{number}_chains" + ("_mask" if masked else ""),
            input_names=family.chain_inputs + (family.mask_inputs if masked else []),
            output_names=outputs,
            header=key_step_header(style),
            source=source,
            ensure_row_contiguous=False,
        )
    if masked:
        raise ValueError("combines read no mask")
    store = "gate" if gated else "plain"
    if passes == "one":
        source = one_pass_combine_source(store=store, style=style)
        inputs = ["maxs", "sums", "outs"]
    else:
        source = two_pass_combine_source(store=store, output="rows", style=style)
        inputs = ["partials", "sums", "maxs"]
    return dict(
        name=f"{family.name_prefix}{number}_combine" + ("_gate" if gated else ""),
        input_names=inputs + (["gate"] if gated else []),
        output_names=["out"],
        source=source,
    )


# The compiled families.  Names are the historical kernel names.
QWEN4_ROWS = VectorKernelFamily(
    "mlx2_qwen4_attn_rows_sdpa", "cache_view", "threadgroup", "broadcast", ROWS_STYLE
)
QWEN4_WINDOW = VectorKernelFamily(
    "mlx2_qwen4_attn_window_sdpa", "window_rows", "grid", "window_concat", ROWS_STYLE
)
MUSE_D128_WINDOW = VectorKernelFamily(
    "mlx2_muse_d128_window_sdpa", "window_rows", "grid", "window_concat", COMPACT_STYLE
)

# One-row TK-unrolled decode at a head dim MLX unrolls at TK 4.
D512_TK_STYLE = SourceStyle("", 0, False, True, " " * 12)


def strided_tk_kwargs(stage: str) -> dict:
    """``mx.fast.metal_kernel`` arguments of the strided TK pass pair."""
    if stage == "pass1":
        return dict(
            name="mlx2_sdpa_vector_d512_pass1",
            input_names=["queries", "keys", "values", "scale"],
            output_names=["out", "sums", "maxs"],
            source=strided_tk_pass1_source(),
            ensure_row_contiguous=False,
        )
    if stage == "pass2":
        return dict(
            name="mlx2_sdpa_vector_d512_pass2",
            input_names=["partials", "sums", "maxs"],
            output_names=["out"],
            source=two_pass_combine_source(store="plain", output="single_row", style=D512_TK_STYLE),
        )
    raise ValueError(f"stage must be pass1 or pass2; got {stage!r}")
