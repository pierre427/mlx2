"""One-token QSA decode attention over the selected blocks only (prototype).

The served one-token path builds a dense visibility mask over every cached
position and runs SDPA across the whole K/V cache: at 32K that reads ~64 MB
per layer to use the ~4 MB the indexer selected
(qualification/runs/fn-lily-attn-core-20261029).  This kernel walks the
selected block ids directly (flash-decoding):

* one threadgroup per (KV head, chunk of selected-id slots), one simdgroup
  per query head of that KV head (GQA 12), 8 of the 256 dims per lane;
* a selected block is visible when complete at the query position
  (``block * 4 + 3 <= q_pos``); the last chunk also takes the incomplete tail
  block ``[(q_pos + 1) // 4 * 4, q_pos]`` -- exactly the visible set of
  ``QSASelection.dense_mask`` for one row.  Positions are logical; a batch
  cache's left padding shifts the buffer row (physical = logical + pad);
* per-chunk online-softmax partials (max, sum, output) in float32, merged by a
  small MLX combine, then the output gate.

Different arithmetic order from the masked SDPA: NOT bit-exact.  Default off
(``MLX_QWEN4_QSA_B1_SELECTED``); unqualified.
"""

from __future__ import annotations

import os

import mlx.core as mx

B1_SELECTED_ENV = "MLX_QWEN4_QSA_B1_SELECTED"
BLOCK_SIZE = 4
HEAD_DIM = 256
SLOTS_PER_CHUNK = 8
_KERNELS: dict = {}
_STATS = {"calls": 0, "declines": {}}


def enabled_from_env() -> bool:
    raw = os.environ.get(B1_SELECTED_ENV, "0").strip().lower()
    if raw in {"", "0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    raise ValueError(f"{B1_SELECTED_ENV}={raw!r}: expected 0/off or 1/on")


_ENABLED = enabled_from_env()
# Record the environment this module read its switch from, so a profile pinned
# after a stale import fails closed in import_env.assert_profile_applied
# instead of serving the other decode law while the receipt says selected.
from .import_env import snapshot as _import_env_snapshot  # noqa: E402

_import_env_snapshot(__name__)


def enabled() -> bool:
    return _ENABLED


def set_enabled(value: bool) -> bool:
    global _ENABLED
    previous, _ENABLED = _ENABLED, bool(value)
    return previous


def status(*, reset: bool = False) -> dict:
    report = {"enabled": _ENABLED, "calls": _STATS["calls"], "declines": dict(_STATS["declines"])}
    if reset:
        _STATS["calls"] = 0
        _STATS["declines"].clear()
    return report


def decline(reason: str) -> None:
    _STATS["declines"][reason] = _STATS["declines"].get(reason, 0) + 1


SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;   // query head within the KV group
    const uint chunk = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const uint nchunks = threadgroups_per_grid.y;
    const int q_pos = meta[0];                         // logical position
    const int cap = meta[1];
    const int pad = meta[2];                           // left padding: physical = logical + pad
    const int NSEL = meta[3];                          // selected slots (runtime: one pipeline)
    const uint qh = kvh * GQA + sg;
    constexpr int VPL = D / 32;                        // values per lane

    float qv[VPL];
    for (int i = 0; i < VPL; i++) {
      qv[i] = float(q[qh * D + lane * VPL + i]) * scale[0];
    }
    float m = -INFINITY;
    float l = 0.0f;
    float o[VPL];
    for (int i = 0; i < VPL; i++) {
      o[i] = 0.0f;
    }
    const device T* kbase = keys + size_t(kvh) * size_t(cap) * D + lane * VPL;
    const device T* vbase = values + size_t(kvh) * size_t(cap) * D + lane * VPL;

    const int first = int(chunk) * SPC;
    const int last = min(first + SPC, NSEL + 1);       // slot NSEL is the tail
    for (int slot = first; slot < last; slot++) {
      int t0, t1;
      if (slot < NSEL) {
        const int b = int(blocks[slot]);
        t0 = b * BS;
        t1 = t0 + BS;
        if (t1 - 1 > q_pos) {
          continue;                                    // incomplete or invalid block
        }
      } else {
        t0 = (q_pos + 1) / BS * BS;
        t1 = q_pos + 1;
      }
      for (int t = t0; t < t1; t++) {
        const device T* kr = kbase + size_t(t + pad) * D;
        float s = 0.0f;
        for (int i = 0; i < VPL; i++) {
          s += qv[i] * float(kr[i]);
        }
        s = simd_sum(s);
        const float m_new = max(m, s);
        const float corr = metal::exp(m - m_new);
        const float p = metal::exp(s - m_new);
        l = l * corr + p;
        const device T* vr = vbase + size_t(t + pad) * D;
        for (int i = 0; i < VPL; i++) {
          o[i] = o[i] * corr + p * float(vr[i]);
        }
        m = m_new;
      }
    }
    const uint part = qh * nchunks + chunk;
    if (lane == 0) {
      out_m[part] = m;
      out_l[part] = l;
    }
    for (int i = 0; i < VPL; i++) {
      out_o[size_t(part) * D + lane * VPL + i] = o[i];
    }
"""


# v2: the chunk's visible tokens are listed once, then K/V tiles of TT tokens
# are staged in threadgroup memory with 16-byte loads and shared by the GQA
# query-head simdgroups; each tile rescales the running softmax once.
SOURCE_V2 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint chunk = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const uint nchunks = threadgroups_per_grid.y;
    constexpr int NT = 32 * GQA;
    constexpr int VPL = D / 32;
    constexpr int V8 = D / 8;                          // uint4 (8 x 16-bit) per row
    const int q_pos = meta[0];                         // logical position
    const int cap = meta[1];
    const int pad = meta[2];                           // left padding: physical = logical + pad
    const int NSEL = meta[3];                          // selected slots (runtime: one pipeline)
    const uint qh = kvh * GQA + sg;

    threadgroup int toks[SPC * BS + BS];
    threadgroup int ntok[1];
    threadgroup uint4 ks4[TT * V8];
    threadgroup uint4 vs4[TT * V8];
    const threadgroup T* ks = (const threadgroup T*)ks4;
    const threadgroup T* vs = (const threadgroup T*)vs4;

    if (tid == 0) {
      int n = 0;
      const int first = int(chunk) * SPC;
      const int last = min(first + SPC, NSEL + 1);
      for (int slot = first; slot < last; slot++) {
        if (slot < NSEL) {
          const int b = int(blocks[slot]);
          if (b * BS + BS - 1 > q_pos) {
            continue;
          }
          for (int j = 0; j < BS; j++) {
            toks[n++] = b * BS + j;
          }
        } else {
          for (int t = (q_pos + 1) / BS * BS; t <= q_pos; t++) {
            toks[n++] = t;
          }
        }
      }
      ntok[0] = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int n = ntok[0];

    float qv[VPL];
    for (int i = 0; i < VPL; i++) {
      qv[i] = float(q[qh * D + lane * VPL + i]) * scale[0];
    }
    float m = -INFINITY;
    float l = 0.0f;
    float o[VPL];
    for (int i = 0; i < VPL; i++) {
      o[i] = 0.0f;
    }
    const device uint4* kh4 = (const device uint4*)(keys + size_t(kvh) * size_t(cap) * D);
    const device uint4* vh4 = (const device uint4*)(values + size_t(kvh) * size_t(cap) * D);

    for (int base = 0; base < n; base += TT) {
      const int tn = min(TT, n - base);
      for (int e = int(tid); e < tn * V8; e += NT) {
        const int r = e / V8;
        const int c = e - r * V8;
        const size_t src = size_t(toks[base + r] + pad) * V8 + c;
        ks4[e] = kh4[src];
        vs4[e] = vh4[src];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float sc[TT];
      float tmax = -INFINITY;
      for (int r = 0; r < tn; r++) {
        float acc = 0.0f;
        const threadgroup T* kr = ks + r * D + lane * VPL;
        for (int i = 0; i < VPL; i++) {
          acc += qv[i] * float(kr[i]);
        }
        sc[r] = simd_sum(acc);
        tmax = max(tmax, sc[r]);
      }
      const float m_new = max(m, tmax);
      const float corr = metal::exp(m - m_new);
      l *= corr;
      for (int i = 0; i < VPL; i++) {
        o[i] *= corr;
      }
      for (int r = 0; r < tn; r++) {
        const float p = metal::exp(sc[r] - m_new);
        l += p;
        const threadgroup T* vr = vs + r * D + lane * VPL;
        for (int i = 0; i < VPL; i++) {
          o[i] += p * float(vr[i]);
        }
      }
      m = m_new;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint part = qh * nchunks + chunk;
    if (lane == 0) {
      out_m[part] = m;
      out_l[part] = l;
    }
    for (int i = 0; i < VPL; i++) {
      out_o[size_t(part) * D + lane * VPL + i] = o[i];
    }
"""
VARIANT = "v1"  # v2 (tiled K/V) measured no faster: fn-lily-b1-opt-20261029
TILE_TOKENS = 16

# Fused combine: one threadgroup per query head, one thread per output dim.
# Each thread recomputes the chunk max and weights (NCH small), accumulates its
# dim over the chunks, divides, casts to T and applies the optional gate.
COMBINE_SOURCE = r"""
    const uint h = threadgroup_position_in_grid.x;
    const uint d = thread_position_in_threadgroup.x;
    const int NCH = nch[0];
    float top = -INFINITY;
    for (int c = 0; c < NCH; c++) {
      top = max(top, out_m[h * NCH + c]);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (int c = 0; c < NCH; c++) {
      const float w = metal::exp(out_m[h * NCH + c] - top);
      denom += w * out_l[h * NCH + c];
      acc += w * out_o[(size_t(h) * NCH + c) * D + d];
    }
    T y = static_cast<T>(acc / denom);
    if (GATED) {
      y = y * gate[h * D + d];
    }
    out[h * D + d] = y;
"""


def _kernel(variant: str):
    kernel = _KERNELS.get(variant)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_qsa_b1_selected_partials_" + variant,
            input_names=["q", "keys", "values", "blocks", "meta", "scale"],
            output_names=["out_m", "out_l", "out_o"],
            source=SOURCE if variant == "v1" else SOURCE_V2,
            ensure_row_contiguous=True,
        )
        _KERNELS[variant] = kernel
    return kernel


def selection_refusal(selection) -> str | None:
    """The kernel encodes plain causal visibility and four-token blocks.

    A custom mask may exclude tokens inside a selected block. Its shape is
    not a proof of plain causality, so leave it with the ordinary consumer.
    """
    from .qwen4_qsa_visibility import QSAVisibility
    from .qsdpa_verify_metal import NOT_CAUSAL, causal_left_padding

    if selection.block_size != BLOCK_SIZE:
        return "block size"
    mask = selection.causal_mask
    if isinstance(mask, QSAVisibility):
        return None
    if causal_left_padding(
        mask, selection.batch, selection.length, selection.physical_width
    ) is NOT_CAUSAL:
        return "custom attention mask"
    return None


def refusal(q, keys, values, block_ids, q_pos) -> str | None:
    """Why this call cannot be served, or None."""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1:
        return "not one query row"
    if keys.ndim != 4 or keys.shape[0] != 1 or values.shape != keys.shape:
        return "cache layout"
    if q.shape[-1] != HEAD_DIM or keys.shape[-1] != HEAD_DIM:
        return "head dim"
    if keys.shape[1] < 1 or q.shape[1] < 1 or q.shape[1] % keys.shape[1]:
        return "query heads not a multiple of KV heads"
    gqa = q.shape[1] // keys.shape[1]
    if gqa * 32 > 1024:
        return "GQA too wide for one threadgroup"
    if q.dtype != keys.dtype or keys.dtype != values.dtype or q.dtype not in (mx.bfloat16, mx.float16):
        return "dtype"
    if block_ids.ndim != 3 or block_ids.shape[:2] != (1, 1):
        return "block ids shape"
    if block_ids.dtype != mx.uint32:
        return "block ids dtype"
    if isinstance(q_pos, int):
        if q_pos < 0 or q_pos >= keys.shape[2]:
            return "query position"
    elif not isinstance(q_pos, mx.array) or q_pos.size != 1:
        return "query position"
    return None


def own_position(cache, left_pad=0):
    """Logical position of the query row ``update_and_fetch`` just wrote.

    The cache has written rows ``[0, _idx)`` (physical), so the query is row
    ``_idx - 1`` and its logical position is that minus the left padding.
    Not ``selection.offset``: a batch cache's offset is an ``mx.array`` that
    ``update_and_fetch`` advanced in place, so the selection's alias already
    counts this token; read as the query position it made the kernel take one
    unwritten row (zeros, or past the buffer when the step filled it), which
    broke width-1 cold/warm parity (qualification/runs/fn-b1-coldwarm-20261029).
    """
    written = getattr(cache, "_idx", None)
    if written is None:
        written = int(cache.offset)
    if isinstance(left_pad, mx.array):
        return (written - 1) - left_pad.reshape(-1)[:1].astype(mx.int32)
    return written - 1 - int(left_pad)


def _meta(q_pos, left_pad, capacity: int, nsel: int):
    """[q_pos, capacity, left_pad, nsel] as int32 without reading device
    values: a batch cache's offset and left padding stay on the GPU.  The
    selected-slot count is a runtime value, not a template constant: it grows
    every BLOCK_SIZE tokens below the indexer budget, and each template value
    compiled its own Metal library (omlx#5025's hazard)."""
    def one(value):
        if isinstance(value, mx.array):
            return value.reshape(-1)[:1].astype(mx.int32)
        return mx.array([int(value)], dtype=mx.int32)

    return mx.concatenate([one(q_pos), one(capacity), one(left_pad), one(nsel)])


def partials(q, keys, values, block_ids, *, q_pos, scale: float, left_pad=0,
             variant: str | None = None, slots_per_chunk: int | None = None):
    """The per-chunk online-softmax partials ``(m, l, o)``."""
    variant = VARIANT if variant is None else variant
    spc = SLOTS_PER_CHUNK if slots_per_chunk is None else int(slots_per_chunk)
    nqh, nkv = int(q.shape[1]), int(keys.shape[1])
    gqa = nqh // nkv
    nsel = int(block_ids.shape[-1])
    nchunks = (nsel + 1 + spc - 1) // spc
    meta = _meta(q_pos, left_pad, int(keys.shape[2]), nsel)
    template = [("T", q.dtype), ("D", HEAD_DIM), ("GQA", gqa), ("SPC", spc), ("BS", BLOCK_SIZE)]
    if variant != "v1":
        template.append(("TT", TILE_TOKENS))
    out_m, out_l, out_o = _kernel(variant)(
        inputs=[q.reshape(-1), keys, values, block_ids.reshape(-1).astype(mx.uint32),
                meta, mx.array([float(scale)], dtype=mx.float32)],
        template=template,
        grid=(32 * gqa, nchunks, nkv),
        threadgroup=(32 * gqa, 1, 1),
        output_shapes=[(nqh * nchunks,), (nqh * nchunks,), (nqh * nchunks * HEAD_DIM,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return (out_m.reshape(nqh, nchunks), out_l.reshape(nqh, nchunks),
            out_o.reshape(nqh, nchunks, HEAD_DIM))


def _combine_kernel():
    kernel = _KERNELS.get("combine")
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_qsa_b1_selected_combine",
            input_names=["out_m", "out_l", "out_o", "gate", "nch"],
            output_names=["out"],
            source=COMBINE_SOURCE,
            ensure_row_contiguous=True,
        )
        _KERNELS["combine"] = kernel
    return kernel


FUSED_COMBINE = True


def combine(m, l, o, *, dtype, gate=None, fused: bool | None = None):
    """[1, 1, H * D] from the partials, gated when ``gate`` is given."""
    nqh = int(m.shape[0])
    if FUSED_COMBINE if fused is None else fused:
        nchunks = int(m.shape[1])
        gate_flat = (gate.reshape(-1).astype(dtype) if gate is not None
                     else mx.zeros((1,), dtype=dtype))
        out = _combine_kernel()(
            inputs=[m.reshape(-1), l.reshape(-1), o.reshape(-1), gate_flat,
                    mx.array([nchunks], dtype=mx.int32)],
            template=[("T", dtype), ("D", HEAD_DIM), ("GATED", int(gate is not None))],
            grid=(HEAD_DIM * nqh, 1, 1),
            threadgroup=(HEAD_DIM, 1, 1),
            output_shapes=[(nqh * HEAD_DIM,)],
            output_dtypes=[dtype],
        )[0]
        return out.reshape(1, 1, nqh * HEAD_DIM)
    top = mx.max(m, axis=-1, keepdims=True)
    weight = mx.exp(m - top)
    denom = mx.sum(weight * l, axis=-1, keepdims=True)
    out = (mx.sum(weight[..., None] * o, axis=1) / denom).astype(dtype)  # [H, D]
    out = out.reshape(1, 1, nqh, HEAD_DIM)
    if gate is not None:
        out = out * gate
    return out.reshape(1, 1, nqh * HEAD_DIM)


def attention(q, keys, values, block_ids, *, q_pos, scale: float, gate=None, left_pad=0,
              variant: str | None = None, slots_per_chunk: int | None = None):
    """[1, 1, H * D] attention output for one query row (gated when ``gate``
    is the [1, 1, H, D] sigmoid gate).  ``keys``/``values`` are the cache's
    step buffers [1, KVH, capacity, D]; positions above ``q_pos`` are unused."""
    m, l, o = partials(q, keys, values, block_ids, q_pos=q_pos, scale=scale, left_pad=left_pad,
                       variant=variant, slots_per_chunk=slots_per_chunk)
    _STATS["calls"] += 1
    return combine(m, l, o, dtype=q.dtype, gate=gate)
