# SPDX-License-Identifier: Apache-2.0 AND MIT
# Window layout adapted from jundot/omlx PRs #4041 / #4105 (Apache-2.0; heads
# 7ef4d997 / 93e708a8: ``attn_fused.dense_sdpa_gate`` -- every row of a verify
# window runs the vector SDPA plan of its own serial decode step, row ``r``
# seeing ``N - R + r + 1`` keys).  The arithmetic is L3's transcription of MLX
# 39400a0d4's vector SDPA (``qwen4_attn_rows``; ``sdpa_vector.h``, MIT).  See
# docs/PROVENANCE.md and provenance/omlx-4041-4105-qwen4-attn-window.json.
"""Row-exact attention window for Flash-Next self-MTP verify (default off).

A row-exact verify window (``qwen4_row_exact``) must give every verify row the
bits of the one-token decode step at its position.  The first version ran the
attention core as R separate one-token forwards: per row the QSA selection,
the q/k norm + RoPE launch, the K/V append and MLX's vector SDPA (two launches
past 1,024 keys).  The one-token step runs with the fused attention rows
(policy ``attn_fused_rows``), whose SDPA is ``qwen4_attn_rows.sdpa_gate``:
MLX's vector kernel for ONE query row over that row's ``N`` keys.  This
module runs the same per-row arithmetic for every row of a window:

``window_sdpa_gate``
    One launch pair per run of consecutive rows that share a vector plan (one
    pass below 1,024 keys; two passes with ``select_sdpa_blocks(N, GQA)``
    partitions above).  Row ``j`` reads keys ``0 .. n_first + j - 1`` of the
    window's cache view and, when given, its own boolean mask row (the
    one-token step's mask, concatenated flat).  The chain and combine sources
    are L3's with only the row indexing changed: a simdgroup's arithmetic is
    the one-row kernel's.
``dense_window``
    Below the indexer budget the one-token step selects every block (the
    fused rows' dense short-circuit), so its mask is the all-valid causal row
    and its state effects are the raw index-key and K/V appends.  The window
    runs the norm + RoPE launch, the index-key append and the K/V append once
    for all rows, then ``window_sdpa_gate`` (no mask: an all-true mask is the
    same arithmetic).  Admitted only when every row is dense by construction,
    checked before any state changes.
``deferred rows``
    Past the budget each row keeps its one-token forward for the selection,
    the QSA mask and the cache append, and hands its SDPA (the call
    ``sdpa_gate`` would make) to the window, which runs all of them in one
    ``window_sdpa_gate``.  Rows whose one-token step takes another arm
    (indexed, NAX, gather, MLX's SDPA) compute themselves as before.

Fail closed: anything not admitted keeps the per-row one-token forwards
(still row-exact); the window records which form ran.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import mlx.core as mx

from . import qwen4_attn_rows as AR

ENV_NAME = "MLX_QWEN4_ROW_EXACT_ATTN_WINDOW"
# Rows per window (copy-draft windows reach 17; omlx serves 16).
WINDOW_MAX_ROWS = 32


def _parse_flag(raw: Optional[str]) -> bool:
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{ENV_NAME} must be 0/off or 1/on; got {raw!r}")


# Default off (the Flash-Next policy field ``row_exact_window_kernels`` sets
# it); off, or ``set_enabled(False)``, the route keeps the per-row one-token
# forwards.
_ENABLED = _parse_flag(os.environ.get(ENV_NAME))


def enabled() -> bool:
    return _ENABLED


def set_enabled(value: bool) -> bool:
    global _ENABLED
    previous = _ENABLED
    _ENABLED = bool(value)
    return previous


# --------------------------------------------------------------------------
# Metal sources: L3's chains with per-row key counts
# --------------------------------------------------------------------------


def _replace(source: str, old: str, new: str) -> str:
    if old not in source:  # pragma: no cover - guards against L3 drift
        raise RuntimeError(f"qwen4_attn_window: L3 source changed near {old!r}")
    return source.replace(old, new)


def _one_pass_source(masked: bool) -> str:
    source = _replace(
        AR._ONE_PASS_CHAINS,
        "const int N = int(keys_shape[2]);",
        "const int N = n_first[0] + row;",
    )
    if masked:
        # Row ``row`` of the group: its mask starts after the masks of the
        # rows before it, whose lengths are n_first, n_first + 1, ...
        setup = (
            "auto mp = masks + mask_base[0] + row * n_first[0]"
            " + (row * (row - 1)) / 2 + s;\n"
            "    const int m_step = BN;"
        )
        use_key, advance = "mp[0]", "mp += m_step;"
    else:
        setup, use_key, advance = "", "true", ""
    source = _replace(source, "@MASK_SETUP(s, BN)@", setup)
    return source.replace("@USE_KEY@", use_key).replace("@MASK_ADVANCE@", advance)


def _two_pass_source(masked: bool) -> str:
    source = AR._TWO_PASS_CHAINS
    source = _replace(
        source,
        "const int block = threadgroup_position_in_grid.z;",
        "const int block = int(threadgroup_position_in_grid.z) % BLOCKS;",
    )
    source = _replace(
        source,
        "const int row = thread_position_in_threadgroup.z;",
        "const int row = int(threadgroup_position_in_grid.z) / BLOCKS;",
    )
    source = _replace(
        source,
        "const int rows = threads_per_threadgroup.z;",
        "const int rows = int(threadgroups_per_grid.z) / BLOCKS;",
    )
    source = _replace(
        source, "const int N = int(keys_shape[2]);", "const int N = n_first[0] + row;"
    )
    if masked:
        setup = (
            "auto mp = masks + mask_base[0] + row * n_first[0]"
            " + (row * (row - 1)) / 2 + block;\n"
            "    const int m_step = BLOCKS;"
        )
        use_key, advance = "mp[0]", "mp += m_step;"
    else:
        setup, use_key, advance = "", "true", ""
    source = _replace(source, "@MASK_SETUP(block, BLOCKS)@", setup)
    return source.replace("@USE_KEY@", use_key).replace("@MASK_ADVANCE@", advance)


_KERNELS: dict = {}


def _kernel(name: str):
    kernel = _KERNELS.get(name)
    if kernel is not None:
        return kernel
    masked = name.endswith("_mask")
    extra = ["masks", "mask_base"] if masked else []
    if name.startswith("one"):
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_attn_window_sdpa1_chains" + ("_mask" if masked else ""),
            input_names=["queries", "keys", "values", "scale", "n_first"] + extra,
            output_names=["maxs", "sums", "outs"],
            header=AR._KEY_STEP,
            source=_one_pass_source(masked),
            ensure_row_contiguous=False,
        )
    else:
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_attn_window_sdpa2_chains" + ("_mask" if masked else ""),
            input_names=["queries", "keys", "values", "scale", "n_first"] + extra,
            output_names=["partials", "sums", "maxs"],
            header=AR._KEY_STEP,
            source=_two_pass_source(masked),
            ensure_row_contiguous=False,
        )
    _KERNELS[name] = kernel
    return kernel


_INTS: dict = {}


def _int(value: int) -> mx.array:
    array = _INTS.get(value)
    if array is None:
        array = mx.array([int(value)], dtype=mx.int32)
        if len(_INTS) < 4096:
            _INTS[value] = array
    return array


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------


def row_plans(n_first: int, rows: int, heads: int, kv_heads: int, head_dim: int):
    """The one-row vector plan of each row (``None`` where MLX would not run
    the transcribed kernel for that row's one-token step)."""
    return [AR.sdpa_plan(n_first + j, 1, heads, kv_heads, head_dim) for j in range(rows)]


def plan_runs(plans: Sequence) -> List[tuple]:
    """``[(start, stop, plan)]``: maximal runs of consecutive rows sharing a
    plan."""
    runs = []
    start = 0
    for j in range(1, len(plans) + 1):
        if j == len(plans) or plans[j] != plans[start]:
            runs.append((start, j, plans[start]))
            start = j
    return runs


def window_supported(queries, keys, values, n_first: int) -> Optional[str]:
    """``None`` when ``window_sdpa_gate`` reproduces, for every row, the
    one-row ``sdpa_gate`` call of its one-token step."""
    if not AR.metal_ready():
        return "device"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "rank"
    batch, heads, rows, head_dim = queries.shape
    if batch != 1 or keys.shape[0] != 1 or values.shape[0] != 1:
        return "batch"
    if not 1 <= rows <= WINDOW_MAX_ROWS:
        return "rows"
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return "dtype"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "kv_dtype"
    if keys.shape[-1] != head_dim or values.shape[-1] != head_dim:
        return "kv_width"
    if keys.shape[1] != values.shape[1] or keys.shape[2] != values.shape[2]:
        return "kv_shape"
    if n_first < 1 or keys.shape[2] < n_first + rows - 1:
        return "key_count"
    plans = row_plans(n_first, rows, heads, keys.shape[1], head_dim)
    if any(plan is None for plan in plans):
        return "plan"
    return None


def window_sdpa_gate(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    *,
    n_first: int,
    gate: mx.array,
    masks: Optional[mx.array] = None,
) -> mx.array:
    """``[1, R, H * D]``: row ``j`` is ``sdpa_gate(queries[:, :, j:j+1],
    keys[:, :, :n_first + j], values[:, :, :n_first + j], scale, mask_j,
    gate_j)`` bit for bit.

    ``queries`` [1, H, R, D] (unit-stride last axis), ``keys``/``values`` cache
    views [1, KVH, >= n_first + R - 1, D], ``gate`` [1, R, H, D] or
    [1, R, H * D] (already the sigmoid), ``masks`` the R one-token mask rows
    concatenated flat (bool, ``n_first + j`` entries for row ``j``) or None
    (every mask all-true)."""
    _, heads, rows, head_dim = queries.shape
    kv_heads = keys.shape[1]
    gqa = heads // kv_heads
    dtype = queries.dtype
    scale_array = AR._scalar(float(scale), mx.float32)
    gate = gate.reshape(rows, heads * head_dim)
    plans = row_plans(n_first, rows, heads, kv_heads, head_dim)
    suffix = "_mask" if masks is not None else ""
    outputs = []
    for start, stop, plan in plan_runs(plans):
        count = stop - start
        q = queries[:, :, start:stop]
        g = gate[start:stop]
        first = n_first + start
        extra = []
        if masks is not None:
            base = start * n_first + (start * (start - 1)) // 2
            extra = [masks, _int(base)]
        passes, blocks = plan
        if passes == 1:
            hpt, sg = AR._ONE_PASS_HPT, AR._ONE_PASS_SG
            chains = 32 * (gqa // hpt)
            maxs, sums, outs = _kernel("one" + suffix)(
                inputs=[q, keys, values, scale_array, _int(first)] + extra,
                template=[("T", dtype), ("D", head_dim), ("GQA", gqa), ("HPT", hpt), ("SG", sg)],
                grid=(32 * chains, kv_heads, count),
                threadgroup=(32 * sg, 1, 1),
                output_shapes=[
                    (count * heads * 32,),
                    (count * heads * 32,),
                    (count * heads * head_dim * 32,),
                ],
                output_dtypes=[mx.float32] * 3,
            )
            out = AR._kernel("one_combine_gate")(
                inputs=[maxs, sums, outs, g],
                template=[("T", dtype), ("D", head_dim)],
                grid=(heads * 1024, count, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(count, heads * head_dim)],
                output_dtypes=[dtype],
            )[0]
        else:
            partials, sums, maxs = _kernel("two" + suffix)(
                inputs=[q, keys, values, scale_array, _int(first)] + extra,
                template=[("T", dtype), ("D", head_dim), ("GQA", gqa), ("BLOCKS", blocks)],
                grid=(32 * kv_heads, gqa, blocks * count),
                threadgroup=(32, gqa, 1),
                output_shapes=[
                    (heads * count * blocks * head_dim,),
                    (heads * count * blocks,),
                    (heads * count * blocks,),
                ],
                output_dtypes=[dtype, mx.float32, mx.float32],
            )
            out = AR._kernel("two_combine_gate")(
                inputs=[partials, sums, maxs, g],
                template=[("T", dtype), ("D", head_dim), ("BLOCKS", blocks)],
                grid=(heads * 1024, count, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(count, heads * head_dim)],
                output_dtypes=[dtype],
            )[0]
        outputs.append(out)
    out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
    return out.reshape(1, rows, heads * head_dim)


# --------------------------------------------------------------------------
# Window forms used by qwen4_row_exact._RowExactAttention
# --------------------------------------------------------------------------


def _gate_heads(attention, qg, rows):
    """``mx.sigmoid`` of the gate half, as the one-token fused path builds it
    (``mx.split(qg.reshape(B, L, H, -1), 2, axis=-1)[1]``)."""
    (_, gate) = mx.split(qg.reshape(1, rows, attention.num_heads, -1), 2, axis=-1)
    return mx.sigmoid(gate)


def _host_offset(cache):
    """The cache's physical key count as a host int, or a refusal reason.

    A plain ``QSAKVCache`` (int offset), or the ordinary B=1 route's
    ``BatchQSAKVCache`` with one row and no staged right padding, whose keys
    view is ``[:_idx]``."""
    from . import qwen4_exp as Q

    if type(cache) is Q.QSAKVCache:
        offset = cache.offset
        if not isinstance(offset, int) or isinstance(offset, bool):
            return "offset"
        return offset
    if type(cache) is Q.BatchQSAKVCache:
        if int(cache.left_padding.shape[0]) != 1:
            return "batch_rows"
        # The route's one-token rows run with ``_ordinary_b1_mask``: every
        # physical key ``[:_idx + j + 1]`` valid (no left padding, which is
        # what the ordinary B=1 route's lane has).  The window computes the
        # same; it refuses only when the host mirror KNOWS the padding is not
        # zero (reading the device value would cost a sync per layer).
        padding = cache._left_padding_rows()
        unpadded = getattr(cache, "_unpadded_ref", None) is cache.left_padding
        if not unpadded and padding is not None and padding != [0]:
            return "left_padding_nonzero"
        if getattr(cache, "_right_padding", None) is not None:
            return "right_padding"
        return int(cache._idx)
    return "cache_type"


def dense_admission(attention, x, cache, rows: int) -> Optional[str]:
    """``None`` when every row's one-token step is the fused rows' dense arm
    and ``dense_window`` may run; checked before any state changes."""
    from . import qwen4_exp as Q

    if not AR.enabled():
        return "attn_fused_rows_off"
    if attention.training:
        return "training"
    if int(x.shape[0]) != 1 or x.dtype not in (mx.bfloat16, mx.float16):
        return "rows_layout"
    if rows > WINDOW_MAX_ROWS:
        return "rows"
    if Q._QSA_GATHER_KV:
        return "gather_kv"
    offset = _host_offset(cache)
    if isinstance(offset, str):
        return offset
    if hasattr(cache, "bits"):
        return "quantized_cache"
    if getattr(cache, "attention_backend", "sdpa") != "sdpa":
        return "attention_backend"
    if getattr(cache, "_mtp_shared_topk", None) is not None:
        return "shared_topk"
    ledger = cache.index_keys
    if (0 if ledger is None else int(ledger.shape[1])) != offset:
        return "index_ledger"
    n_blocks = (offset + rows) // attention.indexer.compress_ratio
    if n_blocks and not attention.indexer._dense_by_construction(n_blocks, None):
        return "past_budget"
    return None


def dense_window(attention, x, cache, projected, window) -> Optional[mx.array]:
    """The window's attention output before ``o_proj`` ([1, R, H * D], gated),
    or ``None`` (nothing changed) when not admitted."""
    rows = int(x.shape[1])
    reason = dense_admission(attention, x, cache, rows)
    (qg, k_flat, v_flat, index_qk) = projected
    offset = _host_offset(cache) if reason is None else 0
    prepped = None
    if reason is None:
        # ``cache.offset`` as the one-token step passes it (an int, or the
        # batch cache's one-entry array): the same RoPE positions.
        prepped = attention._attn_rows_prep(qg, k_flat, cache.offset)
        if prepped is None:
            reason = "prep"
    if reason is None:
        (q, k) = prepped
        # The cache views after the append: [1, KVH, offset + R, D].
        after = _ShapeOnly((1, attention.num_kv_heads, offset + rows, attention.head_dim), q.dtype)
        reason = window_supported(q, after, after, offset + 1)
    if reason is not None:
        window.note("attention_window_dense_decline", reason)
        return None
    selection = attention.indexer(
        x, None, cache, projected_qk=index_qk, dense_shortcircuit=True, fused_query=True
    )
    if selection.kind != "implicit_all":  # pragma: no cover - admission bug
        raise RuntimeError(
            "row-exact dense attention window: the indexer did not select every "
            f"block ({selection.kind}) after dense admission; the raw index keys "
            "are already appended, so the window cannot fall back"
        )
    v = v_flat.reshape(1, rows, attention.num_kv_heads, attention.head_dim).transpose(0, 2, 1, 3)
    (keys, values) = cache.update_and_fetch(k, v)
    out = window_sdpa_gate(
        q,
        keys,
        values,
        attention.scale,
        n_first=offset + 1,
        gate=_gate_heads(attention, qg, rows),
    )
    window.note("attention", "window_dense", rows)
    return out


class _ShapeOnly:
    """Shape/dtype stand-in for ``window_supported`` before the append."""

    __slots__ = ("shape", "dtype", "ndim")

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.ndim = len(self.shape)


class DeferredSDPA:
    """Collects the one-row ``sdpa_gate`` calls of a window's one-token
    forwards (``Attention.__call__(..., _defer_sdpa=...)``)."""

    __slots__ = ("calls",)

    def __init__(self):
        self.calls: List[tuple] = []

    def append(self, queries, keys, values, scale, mask, gate) -> None:
        self.calls.append((queries, keys, values, float(scale), mask, gate))


def _flat_mask(mask, n_keys: int):
    """The one-token mask as a flat [n_keys] bool row, or ``None`` when it is
    not a plain all-row mask of that width (then the row runs on its own)."""
    if mask is None:
        return mx.ones((n_keys,), dtype=mx.bool_)
    if not isinstance(mask, mx.array) or mask.dtype != mx.bool_ or mask.ndim > 4:
        return None
    shape = (1,) * (4 - mask.ndim) + tuple(mask.shape)
    if shape[:3] != (1, 1, 1) or shape[3] != n_keys:
        return None
    return mask.reshape(n_keys)


def run_deferred(deferred: DeferredSDPA, window) -> List[mx.array]:
    """Outputs ([1, 1, H * D] each, gated) of the deferred one-row calls, in
    order: one ``window_sdpa_gate`` when the rows form a window (consecutive
    key counts over one cache view), else each call on its own."""
    calls = deferred.calls
    if not calls:
        return []
    reason = None
    first_keys = int(calls[0][1].shape[2])
    keys, values = calls[-1][1], calls[-1][2]
    scale = calls[0][3]
    masks = []
    for j, (q, k, v, s, mask, gate) in enumerate(calls):
        if int(k.shape[2]) != first_keys + j or s != scale or q.shape[2] != 1:
            reason = "row_geometry"
            break
        flat = _flat_mask(mask, first_keys + j)
        if flat is None:
            reason = "mask_layout"
            break
        masks.append(None if mask is None else flat)
    if reason is None and len(calls) > 1:
        queries = mx.concatenate([c[0] for c in calls], axis=2)
        reason = window_supported(queries, keys, values, first_keys)
    if reason is not None or len(calls) == 1:
        if reason is not None:
            window.note("attention_window_deferred_decline", reason)
        return [
            AR.sdpa_gate(q, k, v, s, mask=mask, gate=gate)
            for (q, k, v, s, mask, gate) in calls
        ]
    flat_masks = None
    if any(m is not None for m in masks):
        flat_masks = mx.concatenate(
            [
                m if m is not None else mx.ones((first_keys + j,), dtype=mx.bool_)
                for j, m in enumerate(masks)
            ]
        )
    gate = mx.concatenate([c[5].reshape(1, 1, -1) for c in calls], axis=1)
    out = window_sdpa_gate(
        queries, keys, values, scale, n_first=first_keys, gate=gate, masks=flat_masks
    )
    window.note("attention_sdpa", "window_deferred", len(calls))
    return [out[:, j : j + 1] for j in range(len(calls))]
