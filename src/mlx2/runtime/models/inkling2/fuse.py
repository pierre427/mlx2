# SPDX-License-Identifier: Apache-2.0
"""Opt-in launch-count fusions for the Inkling 2 decoder (``inkling2_fuse_*``).

Original mlx2 code (2026-11-22).  Every option here is default off and
keeps the reference path selectable.  The ``exact`` options rearrange the
same primitives (``mx.compile`` over elementwise chains, cached index and
zero arrays, one concatenation where the reference made two, the batched
form of three identical small GEMVs); they are meant to be bit-identical
and the CPU tests check that bitwise.  The ``*_kernel`` options are small
Metal kernels whose reductions run in a different order: near-exact, GPU
only, and kept separate.

Two CPU facts shape what is fused (both measured on MLX 0.32.4):

- ``tanh``, ``log`` and ``rsqrt`` give different bits when ``mx.compile``
  fuses them with other ops on the CPU (a vectorized versus a scalar
  implementation), so they stay outside the compiled chains here.
- ``rsqrt`` over a small array is vectorized in groups of 8 with a scalar
  tail, so a row's RMS norm depends on how many rows share the call.  The
  batched controller and norm forms therefore engage only for one-row calls
  (B1 decode), where both sides use the scalar path.

Options (``Model.configure_kernels`` keyword, adapter ``inkling2_fuse_*``):

- ``fuse_controllers``: the three q/k/v loop-message controllers of a visit
  as one batched chain (shared RMS norms, one concatenation, three batched
  GEMVs) plus the three input norms as one norm and one scale-and-cast;
  the MLP controller reuses the visit's message norms.  One-row calls only;
  other calls keep the reference ops with a compiled blend
  (``batched_rows_ok`` says why multi-row calls are not batched).
- ``fuse_residual``: attention-residual mixes read FP32 copies of the BF16
  blocks made once per forward, instead of promoting every block on every
  mix (one cast per block per call in the reference).
- ``fuse_sconv``: short convolutions reuse the convolution context for the
  next state (the reference concatenates the same values twice), and the
  output scale, cast and decoder residual adds run as one compiled kernel;
  the dynamic global-layer convolution's four-lag loop is one compiled
  chain.
- ``fuse_attention``: the relative-bias mask from index and band arrays
  built once per forward and shared by all visits with the same key (one
  gather plus one compiled select per visit), log-position scaling skipped
  below the floor (tau is exactly 1) and compiled above it, the output gate
  compiled, the ring's chronological read and the zero sink row as one
  concatenation, cached zero rows.
- ``fuse_sconv_kernel`` (GPU; default on since 2026-11-22): one Metal kernel
  per static short convolution call of up to 8 rows (conv, scale, residual
  tail and the next state).  Its own summation order makes it near-exact by
  construction; it was observed bitwise on the tested shapes (Metal module
  tests, and checkpoint decode and last-prefill logits at 1K/16K).
- ``fuse_residual_kernel`` (near-exact, GPU): one Metal kernel per one-row
  attention-residual mix (norms, scores, softmax, weighted sum).
"""

from __future__ import annotations

import math

import mlx.core as mx

EXACT = ("fuse_controllers", "fuse_residual", "fuse_sconv", "fuse_attention")
NEAR_EXACT = ("fuse_sconv_kernel", "fuse_residual_kernel")
# Added 2026-11-22 from omlx#5538 / Rapid-MLX #5918-#5919 (reimplemented):
# intended bit-identical (cached gather lhs indices, FP32 RMSNorm weight
# copies, one log-tau per forward) and near-exact kernels (dynamic conv,
# loop-message mix). All default off pending the GPU A/B.
EXACT_2 = ("fuse_gather_lhs", "fuse_norm_fp32", "fuse_shared_tau")
NEAR_EXACT_2 = ("fuse_dconv_kernel", "fuse_message_kernel")
OPTIONS = EXACT + NEAR_EXACT + EXACT_2 + NEAR_EXACT_2
NONE = frozenset()


def count(module, key, n=1):
    """Engagement counter (cumulative since load, like the other Inkling 2
    kernel counters; ``kernel_receipt()["fuse_counts"]``)."""
    counts = module.__dict__.setdefault("fuse_counts", {})
    counts[key] = counts.get(key, 0) + n


def on(module, name):
    return name in getattr(module, "_fuse", NONE)


# ----- compiled elementwise chains (no tanh/log/rsqrt inside) -----


@mx.compile
def blend(forget, gain, x, message):
    """``(1 - forget) * x + gain * message`` (tanh already applied)."""
    return (1 - forget) * x + gain * message


@mx.compile
def scale_cast(normed, weight, dtype_ref):
    """The second half of ``fast.rms_norm(x, w)`` (``rms_norm(x, None) * w``)
    and the decoder's cast to the activation dtype."""
    return (normed * weight).astype(dtype_ref.dtype)


def _sconv_tail_factory():
    @mx.compile
    def plain(out, xf, inv, dtype_ref):
        return ((out + xf) / inv).astype(dtype_ref.dtype)

    @mx.compile
    def attn_residual(out, xf, inv, dtype_ref, x):
        # DecoderLayer: attn_sconv(attn).astype(f32) + x.astype(f32)
        return ((out + xf) / inv).astype(dtype_ref.dtype).astype(mx.float32) + x.astype(
            mx.float32)

    @mx.compile
    def attn_boundary(out, xf, inv, dtype_ref):
        return ((out + xf) / inv).astype(dtype_ref.dtype).astype(mx.float32)

    @mx.compile
    def mlp_residual(out, xf, inv, dtype_ref, residual, out_ref):
        # DecoderLayer: (residual + mlp_sconv(h).astype(f32)).astype(x.dtype)
        return (residual + ((out + xf) / inv).astype(dtype_ref.dtype).astype(mx.float32)
                ).astype(out_ref.dtype)

    return {"plain": plain, "attn_residual": attn_residual,
            "attn_boundary": attn_boundary, "mlp_residual": mlp_residual}


SCONV_TAILS = _sconv_tail_factory()


def sconv_tail(kind, out, xf, k, dtype_ref, *extra):
    # The reference divides by the Python float ``math.sqrt(k + 1)``; an FP32
    # scalar array of the same value divides identically.
    inv = mx.array(math.sqrt(k + 1), mx.float32)
    return SCONV_TAILS[kind](out, xf, inv, dtype_ref, *extra)


def dynamic_conv(context, xf, static_t, dynamic, k, length, num_heads, out_dtype_ref):
    """``DynamicShortConvolution``'s weight sum, lag loop and tail."""
    return _dynamic_conv(context, xf, static_t, dynamic,
                         mx.array(math.sqrt(2 * k + 1), mx.float32), out_dtype_ref,
                         k, length, num_heads)


@mx.compile
def _dynamic_conv(context, xf, static_t, dynamic, inv, dtype_ref, k, length, num_heads):
    b = xf.shape[0]
    weights = static_t.reshape(num_heads, -1, k) + dynamic
    out = mx.zeros_like(xf).reshape(b, length, num_heads, -1)
    for lag in range(k):
        values = context[:, k - 1 - lag : k - 1 - lag + length].reshape(out.shape)
        out = out + values * weights[..., lag]
    return ((out.reshape(xf.shape) + xf) / inv).astype(dtype_ref.dtype)


@mx.compile
def gate_finish(out, gate, dtype_ref):
    return (out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(
        dtype_ref.dtype)


@mx.compile
def _tau_pre(positions, floor):
    return mx.maximum(positions / floor, 1)


@mx.compile
def _tau_apply(logp, alpha, q, r):
    tau = 1 + alpha * logp
    tau = tau[:, :, None, None]
    q2 = (q.astype(mx.float32) * tau.transpose(0, 2, 1, 3)).astype(q.dtype)
    r2 = (r.astype(mx.float32) * tau).astype(r.dtype)
    return q2, r2


def log_tau(attn, q, r, length, offset, left_padding, dtype):
    """Reference: ``tau = 1 + alpha * log(max(pos / floor, 1))`` with
    ``pos = arange(L) + offset + 1`` (minus the row's left padding)."""
    cfg = attn.config
    if left_padding is None and offset + length <= attn.log_floor:
        # Every position is at most offset + L, so below the floor tau is
        # exactly 1 and the reference's FP32 multiply by 1.0 and cast back
        # return q and r unchanged.  (A merged batch can carry negative
        # left padding, which raises positions, so padded calls compute.)
        count(attn, "tau_skipped")
        return q, r
    logp = None
    shared = on(attn, "fuse_shared_tau")
    if shared:
        # Every global visit of one forward holds the same positions, so one
        # log(max(pos / floor, 1)) serves them all (omlx#5538); the values
        # are those each visit would compute.
        memo = forward_memo(attn)
        key = ("tau", int(length), int(offset), float(attn.log_floor),
               None if left_padding is None else id(left_padding))
        held = memo.get(key)
        if held is not None and held[0] is left_padding:
            logp = held[1]
            count(attn, "tau_shared_hits")
    if logp is None:
        ar = _arange(attn, length)
        positions = (ar + offset + 1).astype(mx.float32)[None, :]
        if left_padding is not None:
            positions = positions - left_padding[:, None].astype(mx.float32)
        logp = mx.log(_tau_pre(positions, attn.log_floor))
        if shared and forward_memo(attn) is not None:
            forward_memo(attn)[key] = (left_padding, logp)
    q2, r2 = _tau_apply(logp, cfg.log_scaling_alpha, q.astype(dtype), r.astype(dtype))
    return q2, r2


_DTYPE_REFS = {}


def dtype_ref(dtype):
    """An evaluated scalar carrying ``dtype`` into a compiled function (its
    value is never read), so no array is built per call."""
    ref = _DTYPE_REFS.get(dtype)
    if ref is None:
        ref = _DTYPE_REFS[dtype] = mx.zeros((), dtype, stream=mx.cpu)
        mx.eval(ref)
    return ref


def _arange(module, n):
    held = module.__dict__.setdefault("_fuse_arange", {})
    a = held.get(n)
    if a is None:
        a = held[n] = mx.arange(n)
    return a


def zeros(module, shape, dtype):
    held = module.__dict__.setdefault("_fuse_zeros", {})
    key = (tuple(shape), dtype)
    z = held.get(key)
    if z is None:
        z = held[key] = mx.zeros(shape, dtype)
    return z


# ----- relative-bias mask (inkling v1 ``banded_additive_mask``, CPU form) -----


@mx.compile
def _mask_select(pb, far, masked, zero, neg):
    return mx.where(masked, neg, mx.where(far, zero, pb))


@mx.compile
def _mask_pad(mask, valid, neg):
    return mx.where(valid, mask, neg)


def forward_memo(module):
    """The per-forward memo shared by every visit of one model call
    (``TextModel._forward`` clears it); an empty dict when none is set."""
    holder = getattr(module, "_fuse_holder", None)
    memo = holder.get("memo") if holder is not None else None
    return memo if memo is not None else {}


def fp32_block(module, block):
    """FP32 copy of a residual block, made once per forward."""
    if block.dtype == mx.float32:
        return block
    memo = forward_memo(module)
    key = ("f32", id(block))
    held = memo.get(key)
    if held is None or held[0] is not block:
        held = memo[key] = (block, block.astype(mx.float32))
        count(module, "block_casts")
    return held[1]


def banded_mask(attn, rel, proj, q_offset, size, sliding, rel_extent, left_padding=None):
    """Same values as ``inkling.language.banded_additive_mask`` (non-fused
    branch).  The integer index and the beyond-extent and masked bands
    depend only on (rows, keys, offset, window, extent), so they are built
    once per forward and shared by every visit with that key (all 40 local
    visits of a decode token share one, the 8 global visits another), and
    the per-visit work is one gather and one compiled select."""
    b, lq, h, _ = rel.shape
    dtype = rel.dtype
    key = ("mask", lq, int(size), int(q_offset), int(sliding), int(rel_extent))
    memo = forward_memo(attn)
    held = memo.get(key)
    if held is None:
        qp = mx.arange(lq) + int(q_offset)
        kp = mx.arange(int(size))
        dist = qp[:, None] - kp[None, :]
        gidx = mx.clip(dist, 0, rel_extent - 1)[None, None]
        far = (dist >= rel_extent)[None, None]
        masked = dist < 0
        if sliding > 0:
            masked = masked | (dist >= sliding)
        held = memo[key] = (gidx, far, masked[None, None])
        count(attn, "mask_index_built")
    else:
        count(attn, "mask_index_reused")
    gidx, far, masked = held
    rl = (rel @ proj).transpose(0, 2, 1, 3)
    pb = mx.take_along_axis(rl, mx.broadcast_to(gidx, (b, h, lq, int(size))), axis=-1)
    mask = _mask_select(pb, far, masked, mx.array(0.0, dtype), mx.array(-1e30, dtype))
    if left_padding is not None:
        pkey = ("pad", int(size), id(left_padding))
        valid = memo.get(pkey)
        if valid is None or valid[0] is not left_padding:
            valid = memo[pkey] = (left_padding, (
                mx.arange(int(size))[None, :] >= left_padding[:, None])[:, None, None, :])
        mask = _mask_pad(mask, valid[1], mx.array(-1e30, dtype))
    return mask.astype(dtype)


# ----- ring read plus the zero sink row in one concatenation -----


def ring_with_sink(kv, start, zero):
    """``concat([temporal_order(buf)[..., start:, :], zero])`` for one
    RotatingKVCache buffer pair, in one concatenation per tensor.  The
    parts mirror ``RotatingKVCache._temporal_order`` case for case."""
    keep, idx, offset = kv.keep, kv._idx, kv.offset
    outs = []
    for buf in (kv.keys, kv.values):
        width = buf.shape[2]
        if idx == width:
            parts = [(buf, 0, width)]
        elif idx < offset:
            parts = [(buf, 0, keep), (buf, idx, width), (buf, keep, idx)]
        else:
            parts = [(buf, 0, idx)]
        # Drop the first ``start`` keys of the chronological sequence.
        skip, views = start, []
        for src, a, e in parts:
            n = e - a
            if n <= 0:
                continue
            if skip >= n:
                skip -= n
                continue
            views.append(src[..., a + skip : e, :])
            skip = 0
        outs.append(mx.concatenate([*views, zero], axis=2))
    return outs


# ----- loop-message controllers -----


def controller_stack(layer):
    """``(down [3, r, 2D], forget [3, D, r] fp32, gain [3, D, r] fp32)`` for
    the attn/k/v controllers.  The controllers' own weights become row views
    of the stacks (the FP32 forget/gain caches too), so memory does not grow
    and the reference path reads the same bytes."""
    from .language import _fp32

    names = ("attn", "k", "v")
    ctrls = [layer.controllers[n] for n in names]
    key = tuple(c[w] for c in ctrls for w in ("down_weight", "forget_weight", "gain_weight"))
    held = layer.__dict__.get("_fuse_ctrl")
    if held is not None and all(a is b for a, b in zip(held[0], key)):
        return held[1]
    down = mx.stack([c.down_weight for c in ctrls])
    forget = mx.stack([_fp32(c, "forget_weight") for c in ctrls])
    gain = mx.stack([_fp32(c, "gain_weight") for c in ctrls])
    mx.eval(down, forget, gain)
    for i, c in enumerate(ctrls):
        c.down_weight = down[i]
        for name, stack in (("forget_weight", forget), ("gain_weight", gain)):
            if c[name].dtype == mx.float32:
                c[name] = stack[i]
            else:
                object.__setattr__(c, "_fp32_" + name, (c[name], stack[i]))
    key = tuple(c[w] for c in ctrls for w in ("down_weight", "forget_weight", "gain_weight"))
    object.__setattr__(layer, "_fuse_ctrl", (key, (down, forget, gain)))
    return down, forget, gain


def rms(module, norm, x):
    """``norm(x)`` for an ``nn.RMSNorm``; with ``fuse_norm_fp32`` an FP32
    input reads a cached FP32 copy of the weight, so ``fast.rms_norm`` does
    not cast the BF16 weight on every call (omlx#5538). Same values, so the
    same result."""
    if on(module, "fuse_norm_fp32") and x.dtype == mx.float32 and norm.weight.dtype != mx.float32:
        held = norm.__dict__.get("_fuse_fp32_weight")
        if held is None or held[0] is not norm.weight:
            copy = norm.weight.astype(mx.float32)
            mx.eval(copy)
            held = (norm.weight, copy)
            object.__setattr__(norm, "_fuse_fp32_weight", held)
        count(module, "norm_fp32")
        return mx.fast.rms_norm(x, held[1], norm.eps)
    return norm(x)


def norm_stack(layer):
    """The three input-norm weights as one ``[3, D]`` array (local layers
    repeat their one norm)."""
    names = (("input_layernorm",) * 3 if layer.is_sliding
             else ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm"))
    key = tuple(layer[n].weight for n in names)
    held = layer.__dict__.get("_fuse_norms")
    if held is None or not all(a is b for a, b in zip(held[0], key)):
        held = (key, mx.stack(list(key)))
        object.__setattr__(layer, "_fuse_norms", held)
    return held[1]


def batched_rows_ok(x):
    """Whether the batched controller and norm forms may take this call:
    one-row calls only (decode at B1), where both sides use the same paths.

    Multi-row calls keep the reference ops.  Widening this to GPU calls of up
    to 32 rows (DSpark verify, B>1, short prefill chunks; fc9ac7ed8) was
    tried and reverted on 2026-11-22: the Metal tests at 16eed37f5
    (qualification/runs/inkling2-gpu-20261122/16eed37f5-fusemetal) were not
    bitwise.  ``qkv_controllers`` at the checkpoint's width and rank differed
    from the reference controllers at every FP32 row count 2..32 (max 2.3e-5)
    and at BF16 16 and 32 rows (max 1.4e-3); whole tiny/wide models drifted
    from the first multi-row chunk (BF16 wide max |dlogit| 4.1).  The batch-3
    GEMM picks a different matmul kernel than the three batch-1 GEMMs.  It
    bought nothing measurable either: DSpark GSM8K on Inkling-2-Mini@440956c8
    decoded at a median 39.7 tok/s with it against 40.2 without (one run per
    arm, per-request SD 8-9 tok/s, so within noise; same 40/40 outputs)."""
    return x.shape[0] * x.shape[1] == 1


def qkv_controllers(layer, mixed, messages, message_norms):
    """One-row (B = L = 1) attn/k/v controllers in one batched chain.

    mixed: [1, 1, nq, D] FP32 (nq = 1 or 3); messages: [1, 1, 4, D].
    Returns the blended inputs [1, 1, 3, D] FP32."""
    down, forget, gain = controller_stack(layer)
    ctrl = layer.controllers["attn"]
    width = mixed.shape[-1]
    nx = mx.fast.rms_norm(mixed, None, ctrl.eps)
    if mixed.shape[-2] == 1:
        nx = mx.broadcast_to(nx, (*nx.shape[:-2], 3, width))
    pair = mx.concatenate([nx, message_norms[..., :3, :]], axis=-1).astype(down.dtype)
    pair = pair.reshape(3, 1, 2 * width)
    hidden = (pair @ down.swapaxes(-1, -2)).astype(mx.float32)
    count(layer, "controllers_batched")
    if on(layer, "fuse_message_kernel"):
        from . import fuse_kernels

        out = fuse_kernels.message_mix(
            layer, hidden.reshape(3, -1), forget, gain,
            mixed.reshape(mixed.shape[-2], width), messages[..., :3, :].reshape(3, width))
        if out is not None:
            return out.reshape(1, 1, 3, width)
    f = mx.tanh(hidden @ forget.swapaxes(-1, -2)).reshape(1, 1, 3, width)
    g = mx.tanh(hidden @ gain.swapaxes(-1, -2)).reshape(1, 1, 3, width)
    return blend(f, g, mixed, messages[..., :3, :])


def one_controller(ctrl, x, message, x_norm=None, message_norm=None, owner=None):
    """Reference ``MessageController`` with optional precomputed norms and
    the blend compiled; ``owner`` (the decoder layer) carries the fuse set
    for ``fuse_message_kernel`` on one-row calls."""
    from .language import _fp32

    nx = mx.fast.rms_norm(x, None, ctrl.eps) if x_norm is None else x_norm
    nm = mx.fast.rms_norm(message, None, ctrl.eps) if message_norm is None else message_norm
    pair = mx.concatenate([nx, nm], axis=-1)
    hidden = (pair.astype(ctrl.down_weight.dtype) @ ctrl.down_weight.T).astype(mx.float32)
    width = x.shape[-1]
    if (owner is not None and on(owner, "fuse_message_kernel") and x.size == width
            and message.size == width and x.dtype == mx.float32):
        from . import fuse_kernels

        out = fuse_kernels.message_mix(
            owner, hidden.reshape(1, -1), _fp32(ctrl, "forget_weight")[None],
            _fp32(ctrl, "gain_weight")[None], x.reshape(1, width),
            message.astype(mx.float32).reshape(1, width))
        if out is not None:
            return out.reshape(x.shape)
    forget = mx.tanh(hidden @ _fp32(ctrl, "forget_weight").T)
    gain = mx.tanh(hidden @ _fp32(ctrl, "gain_weight").T)
    return blend(forget, gain, x, message)
