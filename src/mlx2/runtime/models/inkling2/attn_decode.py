# SPDX-License-Identifier: Apache-2.0
"""One-token decode front end of the Inkling 2 ``Attention`` block in one
Metal launch (``inkling2_attn_decode_kernel``; original mlx2 code,
2026-11-22; default off, unqualified).

Why: at B1 decode the shipped attention (Inkling-2-Mini@440956c8, mlx2
306adac5e) costs 12.3 ms of a 29.2 ms step at 1K and barely grows with
context, while its weights stream in about 5 ms. The rest is a chain of
small launches per visit: on the GPU graph of one call, 38 kernels on a
sliding visit and 55 on a global one (``scripts/research/
inkling2_attn_decode_launches.py``). ``mx.compile`` cannot merge the
reductions in that chain (three L2 clips, the value RMS norm, the bias
GEMM), so it leaves most of them.

What the kernel does, for one query row (B = L = 1), between the input
projections and the cache append:

- the k and v short convolutions over ``[state; x]`` (static taps on
  sliding visits, static plus input-dependent taps on global ones), their
  ``(conv + x) / sqrt(.)`` tail and BF16 rounding, and the next conv states;
- the L2 clips of q (with ``q_scale``), k and r (with ``rel_scale``), the
  gain-free value RMS norm, the log-position scale of q and r on global
  visits, the relative bias rows ``r @ rel_proj`` and the FP32 sink logit.

Caches: KVCache and RotatingKVCache (direct API) and the one-row
BatchKVCache and BatchRotatingKVCache the batch generator merges even a
single serving lane into. The batch caches' device ``left_padding`` is
passed on exactly as the composed path passes it (to the split-K kernel,
cropped with the window, and subtracted from the log-tau position), so no
host proof of zero padding is needed; the conv cache must still give no
padding mask (``_cache_padding_mask`` returns None when its host mirror
proves every row valid), as the composed sconv kernels require.

The attention itself stays the shipped split-K kernel (``splitk.py``,
called as ``Attention.__call__`` calls it) and the output gate and o_proj
are unchanged. The K/V append, ring read and window crop are the block's
own methods. Every call outside the decode shape, or any state the
composed path would treat differently, declines before anything is
written and runs the composed path; declines are counted by reason
(``kernel_receipt()["attn_decode"]``).

Numerics: each value follows the reference op for op in FP32 with the same
rounding points, but the reductions (the clip norms, the RMS norm, the
16-term bias dot) sum in the kernel's own order, so it is near-exact, not
bit-identical. An L2 clip whose norm is below its threshold multiplies by
exactly 1, so the q/k/r clips are usually exact; the value norm and the
bias are the likely sources of last-bit differences.
"""

from __future__ import annotations

import contextlib
import math

import mlx.core as mx

# Tests on the CPU replace the Metal kernel with ``prep_reference`` (the
# same law in MLX ops) and the split-K kernel with its reference law, to
# check the plumbing (conv states, cache append, ring crop, receipts).
EMULATE = False

_SOURCE = r"""
    constexpr int DL = D / 32;
    const uint g = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const float qk_thr = fp[0];
    const float rel_thr = fp[1];
    const float inv = fp[2];
    const float eps = fp[3];
    float tau = 1.0f;
    if (TAU) {
        const int padding = HAS_LP ? int(lp[0]) : 0;
        const float p = (float(pos[0]) - float(padding)) / fp[4];
        tau = 1.0f + fp[5] * metal::precise::log(max(p, 1.0f));
    }
    if (g < HQ) {
        // Query head g: q clip (scaled), tau; r clip (scaled), tau, bias.
        float xq[DL];
        float ss = 0.0f;
        for (int i = 0; i < DL; ++i) {
            const int c = g * D + lane * DL + i;
            const float t = float(q_raw[c]) * qs[c];
            xq[i] = t;
            ss += t * t;
        }
        ss = simd_sum(ss);
        const float fq = min(qk_thr / (metal::precise::sqrt(ss) + 1e-6f), 1.0f);
        for (int i = 0; i < DL; ++i) {
            const int c = g * D + lane * DL + i;
            float y = float(T(xq[i] * fq));
            if (TAU) y = float(T(y * tau));
            q_out[c] = T(y);
        }
        float rt = 0.0f;
        if (lane < DR) rt = float(r_raw[g * DR + lane]) * rs[g * DR + lane];
        const float rss = simd_sum(rt * rt);
        const float fr = min(rel_thr / (metal::precise::sqrt(rss) + 1e-6f), 1.0f);
        float rv = float(T(rt * fr));
        if (TAU) rv = float(T(rv * tau));
        float rr[DR];
        for (int j = 0; j < DR; ++j) rr[j] = simd_shuffle(rv, ushort(j));
        for (int e = lane; e < E; e += 32) {
            float acc = 0.0f;
            for (int j = 0; j < DR; ++j) acc += rr[j] * float(relp[j * E + e]);
            rel_out[g * E + e] = T(acc);
        }
        if (lane == 0) sink_out[g] = float(sink_raw[g]);
        return;
    }
    // KV head h: both short convolutions, k clip, v RMS norm, next states.
    const int h = int(g) - HQ;
    float yk[DL], yv[DL];
    float ssk = 0.0f, ssv = 0.0f;
    for (int i = 0; i < DL; ++i) {
        const int c = h * D + lane * DL + i;
        float ck[4], cv[4];
        for (int j = 0; j < 3; ++j) {
            ck[j] = HAS_STATE ? kst[j * C + c] : 0.0f;
            cv[j] = HAS_STATE ? vst[j * C + c] : 0.0f;
        }
        ck[3] = float(kx[c]);
        cv[3] = float(vx[c]);
        float ak = 0.0f, av = 0.0f;
        if (DYN) {
            // weights[lag] = static[lag, c] + dynamic[h, lag]; lag 0 is x.
            for (int lag = 0; lag < 4; ++lag) {
                ak = ak + ck[3 - lag] * (kw[lag * C + c] + float(kd[h * 4 + lag]));
                av = av + cv[3 - lag] * (vw[lag * C + c] + float(vd[h * 4 + lag]));
            }
        } else {
            for (int j = 0; j < 4; ++j) {
                ak += kw[c * 4 + j] * ck[j];
                av += vw[c * 4 + j] * cv[j];
            }
        }
        yk[i] = float(T((ak + ck[3]) / inv));
        yv[i] = float(T((av + cv[3]) / inv));
        ssk += yk[i] * yk[i];
        ssv += yv[i] * yv[i];
        for (int j = 0; j < 3; ++j) {
            kst_out[j * C + c] = ck[j + 1];
            vst_out[j * C + c] = cv[j + 1];
        }
    }
    ssk = simd_sum(ssk);
    ssv = simd_sum(ssv);
    const float fk = min(qk_thr / (metal::precise::sqrt(ssk) + 1e-6f), 1.0f);
    const float nv = metal::precise::rsqrt(ssv / float(D) + eps);
    for (int i = 0; i < DL; ++i) {
        const int c = h * D + lane * DL + i;
        k_out[c] = T(yk[i] * fk);
        v_out[c] = VNORM ? T(yv[i] * nv) : T(yv[i]);
    }
"""
_kernel = None


def _metal():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="inkling2_attn_decode_prep",
            input_names=["q_raw", "r_raw", "sink_raw", "kx", "vx", "kst", "vst", "kw", "vw",
                         "kd", "vd", "qs", "rs", "relp", "fp", "pos", "lp"],
            output_names=["q_out", "k_out", "v_out", "rel_out", "sink_out", "kst_out",
                          "vst_out"],
            source=_SOURCE,
            # No product is fused into an add the composed path rounds
            # separately (fuse_kernels.py's _SAFE_HEADER contract line).
            header="#pragma METAL fp contract(off)\n",
        )
    return _kernel


def count(attn, key):
    counts = attn.__dict__.setdefault("attn_decode_counts", {})
    counts[key] = counts.get(key, 0) + 1


def _device_ok():
    return EMULATE or mx.default_device() == mx.gpu


_CACHES = ("KVCache", "RotatingKVCache", "BatchKVCache", "BatchRotatingKVCache")


def decline_reason(attn, x, key_input, value_input, cache, conv_mask):
    """None when the one-launch front end may take this call, else why not.

    Checked before anything runs, so a decline leaves every state as it was
    (the projection dtype check in ``call`` follows the projections, which
    write nothing)."""
    from . import fuse as _fuse
    from . import splitk
    from .language import DynamicShortConvolution, RotatingKVCache

    if not _device_ok():
        return "cpu"
    if x.shape[0] != 1 or x.shape[1] != 1:
        return "shape"
    if cache is None:
        return "no_cache"
    kv, conv = cache
    if attn.__dict__.get("_kv_q8"):
        return "kv_q8"
    if type(kv).__name__ not in _CACHES or type(kv) not in attn._SPLITK_CACHES:
        return "cache_type"
    if getattr(kv, "deferred_rows", 0):
        # A ring with uncommitted deferred rows (inkling2_verify_deferred_commit)
        # is read through the attention's deferred view, never in place.
        return "deferred_pending"
    held = getattr(kv, "keys", None)
    if isinstance(held, mx.array) and held.dtype == mx.int8:
        return "cache_format"
    if not attn.__dict__.get("_verify_splitk"):
        # The front end hands its rows to the split-K kernel only.
        return "splitk_off"
    if (_fuse.on(attn, "fuse_attention") and type(kv) is RotatingKVCache
            and not attn.__dict__.get("_sink_sdpa") and not attn.__dict__.get("_flash_prefill")):
        # The composed path takes its fused ring read here, which split-K
        # declines (cache_format), and attends with SDPA.
        return "fused_ring_sdpa"
    if x.dtype not in (mx.bfloat16, mx.float32) or any(
            a.dtype != x.dtype for a in (key_input, value_input)):
        return "dtype"
    cfg = attn.config
    if attn.head_dim % 32 or cfg.d_rel > 32 or attn.n_heads % attn.n_kv:
        return "geometry"
    if not EMULATE:
        # The composed gate on what it would read: the cache buffers (the
        # fetched K/V are views of them) or, on an empty cache, rows in the
        # activation dtype (the front end writes K/V in x.dtype).
        probe = mx.zeros((1, attn.n_heads, 1, attn.head_dim), x.dtype)
        keys = held if isinstance(held, mx.array) else probe
        values = getattr(kv, "values", None)
        values = values if isinstance(values, mx.array) else probe
        if isinstance(held, mx.array) and held.dtype != x.dtype:
            return "splitk:dtype"
        reason = splitk.decline_reason(probe, keys, values, n_heads=attn.n_heads,
                                       n_kv=attn.n_kv, head_dim=attn.head_dim)
        if reason is not None:
            return "splitk:" + reason
    if conv is None:
        return "no_conv"
    if conv_mask is not None:
        return "conv_mask"
    if getattr(conv, "speculating", False):
        return "speculating"
    channels = attn.n_kv * attn.head_dim
    for module in (attn.k_sconv, attn.v_sconv):
        if module.kernel_size != 4:
            return "conv_kernel"
        if isinstance(module, DynamicShortConvolution) and module.num_heads != attn.n_kv:
            return "conv_heads"
    states = [conv[attn.k_sconv.conv_idx], conv[attn.v_sconv.conv_idx]]
    if (states[0] is None) != (states[1] is None):
        return "conv_state"
    for state in states:
        if state is not None and (state.shape != (1, 3, channels) or state.dtype != mx.float32):
            return "conv_state"
    return None


def _rel_proj(attn, dtype):
    weight = attn.rel_proj
    if weight.dtype == dtype:
        return weight
    held = attn.__dict__.get("_attn_decode_relp")
    if held is None or held[0] is not weight or held[1].dtype != dtype:
        held = (weight, weight.astype(dtype))
        object.__setattr__(attn, "_attn_decode_relp", held)
    return held[1]


def _fparams(attn, dynamic):
    cfg = attn.config
    key = dynamic
    held = attn.__dict__.get("_attn_decode_fp")
    if held is None or held[0] != key:
        k = cfg.sconv_kernel_size
        inv = math.sqrt(2 * k + 1) if dynamic else math.sqrt(k + 1)
        values = [cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold, inv, cfg.rms_norm_eps,
                  float(1.0 if attn.log_floor is None else attn.log_floor),
                  cfg.log_scaling_alpha]
        held = (key, mx.array(values, mx.float32))
        mx.eval(held[1])
        object.__setattr__(attn, "_attn_decode_fp", held)
    return held[1]


def prep(attn, q, r, sink, kx, vx, k_state, v_state, k_dyn, v_dyn, tau_position,
         left_padding=None):
    """The front end on one row: ``(q, k, v, rel, sink, k_state, v_state)``.

    q, r, sink, kx, vx: the raw projection rows ([1, 1, n]); k/v_state the
    conv states ([1, 3, C] FP32, or None for zeros); k/v_dyn the global
    visits' ``weight_proj`` rows (None on sliding visits); tau_position the
    query's position ``offset + 1`` on global visits (None: no log scale),
    from which the row's ``left_padding`` ([1] int, device) is subtracted.
    Returns q [1, H, 1, D], k and v [1, Hk, 1, D], rel [1, 1, H, E] in the
    activation dtype, sink [1, 1, H] FP32 and the next conv states."""
    from .language import _fp32

    if EMULATE:
        return prep_reference(attn, q, r, sink, kx, vx, k_state, v_state, k_dyn, v_dyn,
                              tau_position, left_padding)
    cfg = attn.config
    dtype = q.dtype
    hq, hkv, dim, dr = attn.n_heads, attn.n_kv, attn.head_dim, cfg.d_rel
    channels = hkv * dim
    relp = _rel_proj(attn, dtype)
    extent = relp.shape[-1]
    dynamic = k_dyn is not None
    if dynamic:
        kw = _fp32(attn.k_sconv, "static_weight")
        vw = _fp32(attn.v_sconv, "static_weight")
    else:
        kw = _fp32(attn.k_sconv.conv, "weight")
        vw = _fp32(attn.v_sconv.conv, "weight")
    placeholder = _placeholder(attn)
    has_state = k_state is not None
    tau = tau_position is not None
    has_lp = tau and left_padding is not None
    outs = _metal()(
        inputs=[q, r, sink, kx, vx,
                k_state if has_state else placeholder, v_state if has_state else placeholder,
                kw, vw, k_dyn if dynamic else placeholder, v_dyn if dynamic else placeholder,
                _fp32(attn, "q_scale"), _fp32(attn, "rel_scale"), relp,
                _fparams(attn, dynamic),
                mx.array([int(tau_position) if tau else 0], mx.int32),
                left_padding if has_lp else _placeholder_int(attn)],
        template=[("T", dtype), ("D", dim), ("HQ", hq), ("DR", dr), ("E", extent),
                  ("C", channels), ("DYN", dynamic), ("TAU", tau), ("HAS_STATE", has_state), ("HAS_LP", has_lp),
                  ("VNORM", bool(cfg.use_value_norm_no_gamma))],
        grid=((hq + hkv) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, hq, 1, dim), (1, hkv, 1, dim), (1, hkv, 1, dim),
                       (1, 1, hq, extent), (1, 1, hq), (1, 3, channels), (1, 3, channels)],
        output_dtypes=[dtype, dtype, dtype, dtype, mx.float32, mx.float32, mx.float32],
    )
    return tuple(outs)


def _placeholder_int(attn):
    held = attn.__dict__.get("_attn_decode_zero_int")
    if held is None:
        held = mx.zeros((1,), mx.int32)
        mx.eval(held)
        object.__setattr__(attn, "_attn_decode_zero_int", held)
    return held


def _placeholder(attn):
    held = attn.__dict__.get("_attn_decode_zero")
    if held is None:
        held = mx.zeros((1,), mx.float32)
        mx.eval(held)
        object.__setattr__(attn, "_attn_decode_zero", held)
    return held


def prep_reference(attn, q, r, sink, kx, vx, k_state, v_state, k_dyn, v_dyn, tau_position,
                   left_padding=None):
    """``prep`` in MLX ops, rounding where the composed path rounds (the CPU
    stand-in for the kernel, and the Metal tests' reference)."""
    from .language import _fp32, l2_clip

    cfg = attn.config
    dtype = q.dtype
    hq, hkv, dim, dr = attn.n_heads, attn.n_kv, attn.head_dim, cfg.d_rel
    channels = hkv * dim
    k = cfg.sconv_kernel_size

    def conv(x, state, module, dyn):
        xf = x.astype(mx.float32).reshape(1, 1, channels)
        if state is None:
            state = mx.zeros((1, k - 1, channels), mx.float32)
        context = mx.concatenate([state, xf], axis=1)
        if dyn is None:
            w = _fp32(module.conv, "weight").reshape(channels, k)
            out = mx.zeros((1, 1, channels), mx.float32)
            for j in range(k):
                out = out + w[:, j] * context[:, j : j + 1]
            inv = math.sqrt(k + 1)
        else:
            w = (_fp32(module, "static_weight").T.reshape(hkv, -1, k)
                 + dyn.astype(mx.float32).reshape(1, 1, hkv, 1, k))
            out = mx.zeros((1, 1, hkv, dim), mx.float32)
            for lag in range(k):
                values = context[:, k - 1 - lag : k - lag].reshape(out.shape)
                out = out + values * w[..., lag]
            out = out.reshape(1, 1, channels)
            inv = math.sqrt(2 * k + 1)
        y = ((out + xf) / inv).astype(dtype)
        return y, context[:, -(k - 1):]

    kc, k_next = conv(kx, k_state, attn.k_sconv, k_dyn)
    vc, v_next = conv(vx, v_state, attn.v_sconv, v_dyn)
    qh = l2_clip(q.reshape(1, 1, hq, dim), cfg.qk_l2_clip_threshold, attn.q_scale)
    kh = l2_clip(kc.reshape(1, 1, hkv, dim), cfg.qk_l2_clip_threshold)
    vh = vc.reshape(1, 1, hkv, dim)
    if cfg.use_value_norm_no_gamma:
        vh = mx.fast.rms_norm(vh, None, cfg.rms_norm_eps)
    rh = l2_clip(r.reshape(1, 1, hq, dr), cfg.rel_l2_clip_threshold, attn.rel_scale)
    if tau_position is not None:
        p = mx.array(float(tau_position), mx.float32)
        if left_padding is not None:
            p = p - left_padding.reshape(()).astype(mx.float32)
        p = p / attn.log_floor
        tau = 1 + cfg.log_scaling_alpha * mx.log(mx.maximum(p, 1))
        qh = (qh.astype(mx.float32) * tau).astype(dtype)
        rh = (rh.astype(mx.float32) * tau).astype(dtype)
    rel = rh @ _rel_proj(attn, dtype)
    return (qh.transpose(0, 2, 1, 3), kh.transpose(0, 2, 1, 3), vh.transpose(0, 2, 1, 3),
            rel, sink.astype(mx.float32), k_next, v_next)


def _position_base(kv):
    """The composed path's ``offset + 1`` for the one query row, before the
    append (KVCache counts in ``offset``, the batch caches in ``_idx``)."""
    from .language import KVCache

    return (kv.offset if type(kv) is KVCache else kv._idx) + 1


def call(attn, x, key_input, value_input, cache, conv_mask):
    """The decode visit through the front end, or None (declined, counted)
    for the composed path."""
    from . import splitk
    from .language import DynamicShortConvolution, _count_ring

    reason = decline_reason(attn, x, key_input, value_input, cache, conv_mask)
    if reason is not None:
        count(attn, "declined:" + reason)
        return None
    kv, conv = cache
    q, r, gate, sink, kx, vx = attn._project_inputs(x, key_input, value_input)
    if any(a.dtype != x.dtype for a in (q, r, gate, sink, kx, vx)):
        # Nothing is written yet; the lazy projections are dropped.
        count(attn, "declined:projection_dtype")
        return None
    dynamic = isinstance(attn.k_sconv, DynamicShortConvolution)
    k_dyn = attn.k_sconv.weight_proj(key_input) if dynamic else None
    v_dyn = attn.v_sconv.weight_proj(value_input) if dynamic else None
    left_padding = getattr(kv, "left_padding", None)
    # Global visits scale q and r by log position; the query sits at
    # ``offset + 1 - left_padding`` (the composed path's positions). Below the
    # floor tau is exactly 1 and the composed path skips it when unpadded.
    tau_position = None
    if attn.log_floor is not None:
        base = _position_base(kv)
        if left_padding is not None or base > attn.log_floor:
            tau_position = base
    ki, vi = attn.k_sconv.conv_idx, attn.v_sconv.conv_idx
    q, k, v, rel, sink, k_state, v_state = prep(
        attn, q, r, sink, kx, vx, conv[ki], conv[vi], k_dyn, v_dyn, tau_position, left_padding)
    conv[ki] = k_state
    conv[vi] = v_state
    ring_phys, ring_start = None, 0
    if attn._ring_direct(kv, 1):
        kv.update_and_fetch(k, v)
        ring_phys = attn._ring_layout(kv)
        if ring_phys is None:
            _count_ring(attn, "unmapped")
            k, v = attn._chronological(kv, None)
        else:
            k = v = None
    else:
        k, v = attn._fetch(kv, k, v)
    # Re-read after the append, as the composed path does: a rotating batch
    # cache updates its padding during the append (in place today, but do not
    # depend on that).
    left_padding = getattr(kv, "left_padding", None)
    size = k.shape[2] if ring_phys is None else ring_phys[1]
    if attn.sliding and size > attn.sliding:
        start = size - attn.sliding
        if ring_phys is None:
            k, v = k[:, :, start:], v[:, :, start:]
        ring_start = start
        size -= start
        if left_padding is not None:
            left_padding = mx.maximum(left_padding - start, 0)
    offset = size - 1
    counts = attn.__dict__.setdefault("verify_splitk_counts", {})
    counts["calls"] = counts.get("calls", 0) + 1
    ring_arg = None
    if ring_phys is not None:
        _count_ring(attn, "calls")
        k, v = kv.keys, kv.values
        ring_arg = ((ring_phys[0] + ring_start) % k.shape[2], size)
    attend = splitk.reference_attention if EMULATE else splitk.gqa_splitk_attention
    out = attend(q, k, v, rel, sink, offset=offset, left_padding=left_padding,
                 window=attn.sliding, scale=attn.scale, ring=ring_arg)
    count(attn, "calls")
    return attn._finish(out, gate, 1, 1, x.dtype)


def composed_reference(attn, x, key_input, value_input, cache):
    """The composed path's own front end on the same live inputs, for the
    Metal tests and the checkpoint check: ``(q, k, v, rel, sink, k_state,
    v_state)`` as ``prep`` returns them, computed by the modules and helpers
    ``Attention.__call__`` uses (the sconv kernel, the compiled dynamic
    convolution, ``l2_clip``, the value norm, ``fuse.log_tau``, the bias
    GEMM). Writes no cache: the conv modules get a copy of the conv slots.
    Diagnostic only (it bumps the modules' fuse counters)."""
    from . import fuse as _fuse
    from .language import l2_clip

    kv, conv = cache
    cfg = attn.config
    b, length = 1, 1
    key_input = x if key_input is None else key_input
    value_input = x if value_input is None else value_input
    q, r, _gate, sink, k, v = attn._project_inputs(x, key_input, value_input)
    slots = _Slots([conv[i] for i in range(4)])
    if attn.is_sliding:
        k = attn.k_sconv(k, slots, None)
        v = attn.v_sconv(v, slots, None)
    else:
        k = attn.k_sconv(k, key_input, slots, None)
        v = attn.v_sconv(v, value_input, slots, None)
    q = l2_clip(q.reshape(b, length, attn.n_heads, attn.head_dim),
                cfg.qk_l2_clip_threshold, attn.q_scale)
    k = l2_clip(k.reshape(b, length, attn.n_kv, attn.head_dim), cfg.qk_l2_clip_threshold)
    v = v.reshape(b, length, attn.n_kv, attn.head_dim)
    fa = _fuse.on(attn, "fuse_attention")
    if cfg.use_value_norm_no_gamma and fa and v.dtype == x.dtype:
        v = mx.fast.rms_norm(v, None, cfg.rms_norm_eps)
    elif cfg.use_value_norm_no_gamma:
        v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(x.dtype)
    r = l2_clip(r.reshape(b, length, attn.n_heads, cfg.d_rel), cfg.rel_l2_clip_threshold,
                attn.rel_scale)
    q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
    left_padding = getattr(kv, "left_padding", None)
    if attn.log_floor is not None:
        offset = _position_base(kv) - 1
        if fa or _fuse.on(attn, "fuse_shared_tau"):
            q, r = _fuse.log_tau(attn, q, r, length, offset, left_padding, x.dtype)
        else:
            positions = (mx.arange(length) + offset + 1).astype(mx.float32)[None, :]
            if left_padding is not None:
                positions = positions - left_padding[:, None].astype(mx.float32)
            tau = 1 + cfg.log_scaling_alpha * mx.log(mx.maximum(positions / attn.log_floor, 1))
            tau = tau[:, :, None, None]
            q = (q.astype(mx.float32) * tau.transpose(0, 2, 1, 3)).astype(x.dtype)
            r = (r.astype(mx.float32) * tau).astype(x.dtype)
    rel = r @ attn.rel_proj.astype(x.dtype)
    return (q, k, v, rel, sink.astype(mx.float32), slots[attn.k_sconv.conv_idx],
            slots[attn.v_sconv.conv_idx])


class _Slots(list):
    """Conv-cache stand-in for ``composed_reference`` (indexable, assignable,
    never speculating)."""

    speculating = False


OUTPUTS = ("q", "k", "v", "rel", "sink", "k_state", "v_state")


@contextlib.contextmanager
def parity_probe(records):
    """Diagnostic: while active, every engaged call also runs
    ``composed_reference`` on the same live inputs and appends
    ``{"kind": "sliding"|"global", "front": prep outputs, "composed": ...}``
    to ``records`` (the caller compares them). Not for serving."""
    global call, prep
    original_call, original_prep = call, prep

    def probed_call(attn, x, key_input, value_input, cache, conv_mask):
        if decline_reason(attn, x, key_input, value_input, cache, conv_mask) is not None:
            return original_call(attn, x, key_input, value_input, cache, conv_mask)
        composed = composed_reference(attn, x, key_input, value_input, cache)
        held = {}

        def captured(*args, **kwargs):
            held["front"] = original_prep(*args, **kwargs)
            return held["front"]

        globals()["prep"] = captured
        try:
            out = original_call(attn, x, key_input, value_input, cache, conv_mask)
        finally:
            globals()["prep"] = original_prep
        if "front" in held:
            records.append({"kind": "sliding" if attn.is_sliding else "global",
                            "front": dict(zip(OUTPUTS, held["front"])),
                            "composed": dict(zip(OUTPUTS, composed))})
        return out

    call = probed_call
    try:
        yield records
    finally:
        call, prep = original_call, original_prep
