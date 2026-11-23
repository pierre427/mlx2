"""GPTQ-style quantization to MLX affine planes.

Original mlx2 code, written from the published algorithm (Frantar et al.,
"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
Transformers", 2022, Algorithm 1 with lazy batch updates); no implementation
was copied.  Model-agnostic: it quantizes weight matrices given calibration
Hessians and emits the standard MLX affine planes (``weight`` packed uint32
low bits first, per-group ``scales`` and ``biases``) so every MLX kernel
(``quantized_matmul``, ``gather_qmm``, ``dequantize``) runs the result.

For a weight ``W [O, I]`` with layer input rows ``x`` (``H = sum x x^T``):

1. dead input columns (``H_ii == 0``) are zeroed and their diagonal set to 1;
2. ``H += damp * mean(diag(H)) * I``;
3. ``U = cholesky(H^-1)`` (upper);
4. columns are quantized left to right in blocks of ``block``.  At each
   group start (``group_size`` columns, MLX's contiguous input-axis groups)
   the group's affine ``(scale, bias)`` (MLX's ``mx.quantize`` law) are
   taken from the *current* (error-compensated) weights, rounded to
   the storage dtype so the feedback sees exactly what will be stored;
   ``q = clip(round((w - bias) / scale))``; the column error ``(w - q) /
   U_ii`` is propagated to the remaining columns of the block through row
   ``i`` of ``U``, and each finished block's errors to every later column.

``act_order`` quantizes columns in decreasing ``diag(H)`` order with static
groups (group parameters from the original weights, so groups stay
contiguous on the input axis, as MLX requires).

Expert banks ``W [E, O, I]`` take one Hessian per expert, from the rows
routed to that expert (:class:`InputRecorder`); experts are processed in
chunks to bound memory.  Linear algebra runs on the CPU stream (MLX's
``linalg`` is CPU-only); the column loop runs on the default device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

PACKABLE_BITS = (2, 4, 8)


@dataclass
class GPTQSettings:
    bits: int
    group_size: int = 64
    damp: float = 0.01
    block: int = 128
    act_order: bool = False
    expert_chunk: int = 32
    # Experts with fewer routed calibration rows than this fall back to
    # round-to-nearest (their Hessian says nothing useful).
    min_rows: int = 8

    def validate(self, in_dim: int):
        if self.bits not in PACKABLE_BITS:
            raise ValueError(f"GPTQ packs bits {PACKABLE_BITS} (got {self.bits})")
        if in_dim % self.group_size:
            raise ValueError(f"input width {in_dim} is not a multiple of group_size {self.group_size}")
        if self.block % self.group_size:
            raise ValueError("block must be a multiple of group_size")
        if not 0 <= self.damp < 1:
            raise ValueError("damp must be in [0, 1)")


# ----------------------------------------------------------------- packing --

def pack(q, bits: int):
    """Integer codes ``[..., I]`` (0 .. 2^bits-1) -> MLX packed uint32 ``[..., I * bits / 32]``."""
    import mlx.core as mx

    per = 32 // bits
    if q.shape[-1] % per:
        raise ValueError("input width does not fill whole uint32 words")
    q = q.astype(mx.uint32).reshape(*q.shape[:-1], q.shape[-1] // per, per)
    shifts = (mx.arange(per, dtype=mx.uint32) * bits)
    return (q << shifts).sum(axis=-1).astype(mx.uint32)


def _group_params(w, bits, dtype):
    """Affine ``(scale, bias)`` of a group ``[..., G]``, rounded to ``dtype``.

    MLX's own affine law (``mx.quantize``): the edge of larger magnitude is
    made exactly representable (the scale is stretched to ``edge / q0`` and
    may be negative), so round-to-nearest here is ``mx.quantize``.  A plain
    min/max law has ~33% more 2-bit weight error, which eats most of GPTQ's
    gain against an ``mx.quantize`` baseline.
    """
    import mlx.core as mx

    w = w.astype(mx.float32)
    lo = w.min(axis=-1)
    hi = w.max(axis=-1)
    levels = float(2 ** bits - 1)
    mask = mx.abs(lo) > mx.abs(hi)
    scale = mx.maximum((hi - lo) / levels, 1e-7)
    scale = mx.where(mask, scale, -scale)
    edge = mx.where(mask, lo, hi)
    q0 = mx.round(edge / scale)
    scale = mx.where(q0 != 0, edge / q0, scale)
    bias = mx.where(q0 == 0, mx.zeros_like(edge), edge)
    return scale.astype(dtype).astype(mx.float32), bias.astype(dtype).astype(mx.float32)


def _codes(w, scale, bias, bits):
    import mlx.core as mx

    return mx.clip(mx.round((w - bias) / scale), 0, 2 ** bits - 1)


def rtn(W, *, bits: int, group_size: int, dtype):
    """Round-to-nearest with the same group law (the GPTQ baseline / fallback)."""
    import mlx.core as mx

    *lead, I = W.shape
    G = I // group_size
    g = W.astype(mx.float32).reshape(*lead, G, group_size)
    scale, bias = _group_params(g, bits, dtype)
    q = _codes(g, scale[..., None], bias[..., None], bits).reshape(*lead, I)
    return q, scale, bias


# ---------------------------------------------------------------- the core --

def gptq_batch(W, H, settings: GPTQSettings, *, dtype):
    """GPTQ for a batch ``W [B, O, I]`` with Hessians ``H [B, I, I]`` (fp32).

    Returns integer codes ``[B, O, I]`` and ``(scales, biases) [B, O, I/G]``.
    """
    import mlx.core as mx

    B, O, I = W.shape
    s = settings
    s.validate(I)
    W = W.astype(mx.float32)
    H = H.astype(mx.float32)
    diag = mx.diagonal(H, axis1=-2, axis2=-1)                      # [B, I]
    dead = diag == 0
    eye = mx.eye(I, dtype=mx.float32)
    H = H + mx.where(dead[:, :, None], eye, 0.0) * eye
    W = mx.where(dead[:, None, :], 0.0, W)
    diag = mx.diagonal(H, axis1=-2, axis2=-1)
    G = I // s.group_size
    static_scale = static_bias = None
    perm = None
    if s.act_order:
        g = W.reshape(B, O, G, s.group_size)
        static_scale, static_bias = _group_params(g, s.bits, dtype)        # [B, O, G]
        perm = mx.argsort(-diag, axis=-1)                                   # [B, I]
        W = mx.take_along_axis(W, perm[:, None, :], axis=-1)
        H = mx.take_along_axis(mx.take_along_axis(H, perm[:, :, None], axis=1), perm[:, None, :], axis=2)
        diag = mx.take_along_axis(diag, perm, axis=-1)
    damp = s.damp * diag.mean(axis=-1)                                      # [B]
    H = H + damp[:, None, None] * eye
    mx.eval(W, H)
    Hinv = mx.linalg.inv(H, stream=mx.cpu)
    U = mx.linalg.cholesky(Hinv, upper=True, stream=mx.cpu)                 # [B, I, I]
    mx.eval(U)
    Q = mx.zeros((B, O, I), dtype=mx.float32)
    scales = mx.zeros((B, O, G), dtype=mx.float32)
    biases = mx.zeros((B, O, G), dtype=mx.float32)
    if s.act_order:
        col_group = (perm // s.group_size)                                  # [B, I] group of each permuted column
    cols = []
    for i1 in range(0, I, s.block):
        i2 = min(i1 + s.block, I)
        W1 = W[:, :, i1:i2]
        U1 = U[:, i1:i2, i1:i2]
        errs = []
        q_cols = []
        sc = bi = None
        for j in range(i2 - i1):
            col = i1 + j
            w = W1[:, :, j]                                                 # [B, O]
            if s.act_order:
                gi = col_group[:, col]                                      # [B]
                sc = mx.take_along_axis(static_scale, gi[:, None, None], axis=-1)[..., 0]
                bi = mx.take_along_axis(static_bias, gi[:, None, None], axis=-1)[..., 0]
            elif col % s.group_size == 0:
                gscale, gbias = _group_params(W1[:, :, j:j + s.group_size], s.bits, dtype)
                g = col // s.group_size
                scales[:, :, g] = gscale
                biases[:, :, g] = gbias
                sc, bi = gscale, gbias
            q = _codes(w, sc, bi, s.bits)
            dq = q * sc + bi
            d = U1[:, j, j]                                                 # [B]
            err = (w - dq) / d[:, None]                                     # [B, O]
            if j + 1 < i2 - i1:
                W1 = W1.at[:, :, j + 1:].add(-err[:, :, None] * U1[:, j, j + 1:][:, None, :])
            q_cols.append(q)
            errs.append(err)
        Qb = mx.stack(q_cols, axis=-1)
        Err = mx.stack(errs, axis=-1)                                       # [B, O, block]
        if i2 < I:
            W = W.at[:, :, i2:].add(-(Err @ U[:, i1:i2, i2:]))
        cols.append(Qb)
        mx.eval(Qb, W, scales, biases)
    Q = mx.concatenate(cols, axis=-1)
    if s.act_order:
        inverse = mx.argsort(perm, axis=-1)
        Q = mx.take_along_axis(Q, inverse[:, None, :], axis=-1)
        scales, biases = static_scale, static_bias
    return Q, scales, biases


def quantize_bank(W, hessians, settings: GPTQSettings, *, dtype, rows=None):
    """GPTQ for a weight ``[O, I]`` or an expert bank ``[E, O, I]``.

    ``hessians`` is a callable ``h(start, stop) -> [n, I, I]`` for experts
    ``start .. stop-1`` (one expert for a 2-D weight); ``rows[e]`` (optional)
    is how many calibration rows built expert ``e``'s Hessian: experts under
    ``settings.min_rows`` are quantized round-to-nearest.  Returns MLX planes
    ``(weight uint32, scales, biases)`` in the input's leading shape, and a
    receipt.
    """
    import mlx.core as mx

    squeeze = W.ndim == 2
    bank = W[None] if squeeze else W
    E, O, I = bank.shape
    settings.validate(I)
    q_parts, s_parts, b_parts = [], [], []
    fallback = 0
    for start in range(0, E, settings.expert_chunk):
        stop = min(start + settings.expert_chunk, E)
        Wc = bank[start:stop].astype(mx.float32)
        Hc = hessians(start, stop)
        q, sc, bi = gptq_batch(Wc, Hc, settings, dtype=dtype)
        if rows is not None:
            few = [e - start for e in range(start, stop) if rows[e] < settings.min_rows]
            if few:
                fallback += len(few)
                rq, rs, rb = rtn(Wc, bits=settings.bits, group_size=settings.group_size, dtype=dtype)
                mask = mx.array([e in few for e in range(stop - start)])
                q = mx.where(mask[:, None, None], rq, q)
                sc = mx.where(mask[:, None, None], rs, sc)
                bi = mx.where(mask[:, None, None], rb, bi)
        q_parts.append(pack(q, settings.bits))
        s_parts.append(sc.astype(dtype))
        b_parts.append(bi.astype(dtype))
        mx.eval(q_parts[-1], s_parts[-1], b_parts[-1])
        del Wc, Hc, q, sc, bi
        mx.clear_cache()  # per-chunk Hessians/inverses are GBs at MoE-bank size
    weight = mx.concatenate(q_parts, axis=0)
    scales = mx.concatenate(s_parts, axis=0)
    biases = mx.concatenate(b_parts, axis=0)
    if squeeze:
        weight, scales, biases = weight[0], scales[0], biases[0]
    receipt = {"method": "gptq", "bits": settings.bits, "group_size": settings.group_size,
               "damp": settings.damp, "block": settings.block, "act_order": settings.act_order,
               "experts": E if not squeeze else None, "rtn_fallback_experts": fallback,
               "min_rows": settings.min_rows}
    if rows is not None:
        import numpy as np

        counts = np.asarray(rows)
        receipt["calibration_rows"] = {"total": int(counts.sum()), "min": int(counts.min()),
                                       "median": float(np.median(counts)), "max": int(counts.max())}
    return (weight, scales, biases), receipt


# ------------------------------------------------------------- calibration --

class StopForward(Exception):
    """Raised by a recorder once every recorder of its stage has seen the
    current calibration text: the rest of the forward is not needed."""


class HessianAccumulator:
    """Wraps a linear or an expert bank and accumulates ``H = sum x x^T``.

    No input rows are kept: a bank keeps one fp32 ``[I, I]`` Hessian per
    expert (a Python list, so adding to one expert never copies the
    others) and a row count; a linear keeps one Hessian.  ``stage`` (a
    :class:`Stage`) makes the forward stop as soon as every accumulator of
    the stage has recorded the current text.
    """

    def __init__(self, module, *, bank: bool, experts: Optional[int] = None, stage=None):
        self.module = module
        self.bank = bank
        self.experts = experts
        self.stage = stage
        self.calls = 0
        self.H = None
        self.counts = None
        self.eval_experts = 8  # experts per evaluation (bounds the transient peak)

    def _rows(self, x, indices):
        import mlx.core as mx

        xs = x[..., 0, :] if x.ndim >= 2 and x.shape[-2] == 1 else x
        target = mx.broadcast_shapes(xs.shape[:-1], indices.shape)
        rows = mx.broadcast_to(xs, (*target, xs.shape[-1])).reshape(-1, xs.shape[-1])
        return rows, mx.broadcast_to(indices, target).reshape(-1)

    def __call__(self, x, indices=None, sorted_indices=False):
        import mlx.core as mx
        import numpy as np

        I = x.shape[-1]
        if self.bank:
            rows, ids = self._rows(x, indices)
            if self.H is None:
                self.H = [None] * self.experts
                self.counts = [0] * self.experts
            ids_np = np.array(ids)
            order = np.argsort(ids_np, kind="stable")
            bounds = np.searchsorted(ids_np[order], np.arange(self.experts + 1))
            rows = rows[mx.array(order)].astype(mx.float32)
            # Evaluate a few experts at a time: one lazy graph over every
            # routed expert holds the old Hessians, the updates and the new
            # Hessians at once (about 3x the bank's fp32 Hessians, ~14 GB
            # on a 512-expert 1536-wide bank).  Same sums, same order.
            pending = []
            for e in np.nonzero(bounds[1:] > bounds[:-1])[0].tolist():
                a, b = int(bounds[e]), int(bounds[e + 1])
                block = rows[a:b]
                update = block.T @ block
                self.H[e] = update if self.H[e] is None else self.H[e] + update
                self.counts[e] += b - a
                pending.append(self.H[e])
                if len(pending) >= self.eval_experts:
                    mx.eval(pending)
                    pending = []
            if pending:
                mx.eval(pending)
            del rows
        else:
            rows = x.reshape(-1, I).astype(mx.float32)
            update = rows.T @ rows
            self.H = update if self.H is None else self.H + update
            self.counts = [int(rows.shape[0]) + (self.counts[0] if self.counts else 0)]
            mx.eval(self.H)
        self.calls += 1
        out = self.module(x, indices, sorted_indices=sorted_indices) if self.bank else self.module(x)
        if self.stage is not None:
            self.stage.check()
        return out

    def __getattr__(self, name):
        return getattr(self.module, name)

    def hessians(self, I: int):
        """``(h(start, stop), rows_per_expert)``; experts never routed get zeros."""
        import mlx.core as mx

        def h(start, stop):
            if not self.bank:
                return self.H[None]
            return mx.stack([self.H[e] if self.H[e] is not None else mx.zeros((I, I), dtype=mx.float32)
                             for e in range(start, stop)])

        return h, (list(self.counts) if self.counts is not None else [0])

    def release(self):
        self.H = None


class Stage:
    """The accumulators of one calibration pass and the text being run."""

    def __init__(self):
        self.members = []
        self.text = 0

    def check(self):
        if all(m.calls > self.text for m in self.members):
            raise StopForward()


def accumulate(targets: Sequence[tuple], texts, run_text) -> list:
    """One calibration pass: wrap ``targets`` (``(parent, attribute, module,
    bank, experts)``), run ``run_text(text)`` for each text, stopping each
    forward once every target has recorded it; originals always restored."""
    stage = Stage()
    installed = []
    try:
        for parent, attribute, module, bank, experts in targets:
            if attribute in vars(parent):
                raise ValueError(f"{attribute} is already shadowed on its parent")
            acc = HessianAccumulator(module, bank=bank, experts=experts, stage=stage)
            object.__setattr__(parent, attribute, acc)
            stage.members.append(acc)
            installed.append((parent, attribute, acc))
        for index, text in enumerate(texts):
            stage.text = index
            try:
                run_text(text)
            except StopForward:
                pass
            # run_text's own ``finally`` (between-texts hooks such as a swap
            # guard) has run; a guard's exception propagates from there.
            for acc in stage.members:
                if acc.calls <= index:
                    raise RuntimeError(f"calibration text {index} never reached a recorded module")
    finally:
        for parent, attribute, _acc in installed:
            vars(parent).pop(attribute, None)
    return [acc for *_rest, acc in installed]


class InputRecorder:
    """Wraps a linear or an expert bank and records its input rows.

    ``bank(x, indices, sorted_indices=False)`` (expert banks: ``x
    [..., 1, I]`` broadcast against ``indices``) records one row per
    (position, routed expert) with its expert id; ``linear(x)`` records the
    rows of ``x [..., I]``.  Rows are kept in bf16 on the default device.
    """

    def __init__(self, module, *, bank: bool):
        self.module = module
        self.bank = bank
        self.rows = []
        self.ids = []

    def __call__(self, x, indices=None, sorted_indices=False):
        import mlx.core as mx

        if self.bank:
            xs = x[..., 0, :] if x.ndim >= 2 and x.shape[-2] == 1 else x
            target = mx.broadcast_shapes(xs.shape[:-1], indices.shape)
            rows = mx.broadcast_to(xs, (*target, xs.shape[-1])).reshape(-1, xs.shape[-1])
            ids = mx.broadcast_to(indices, target).reshape(-1)
            self.rows.append(rows.astype(mx.bfloat16))
            self.ids.append(ids.astype(mx.int32))
            mx.eval(self.rows[-1], self.ids[-1])
            return self.module(x, indices, sorted_indices=sorted_indices)
        self.rows.append(x.reshape(-1, x.shape[-1]).astype(mx.bfloat16))
        mx.eval(self.rows[-1])
        return self.module(x)

    def __getattr__(self, name):
        return getattr(self.module, name)

    def hessians(self, experts: Optional[int] = None):
        """``(h(start, stop), rows_per_expert)`` from the recorded rows."""
        import mlx.core as mx

        X = mx.concatenate(self.rows, axis=0)
        if not self.bank:
            def h_dense(start, stop):
                Xf = X.astype(mx.float32)
                return (Xf.T @ Xf)[None]
            return h_dense, [int(X.shape[0])]
        ids = mx.concatenate(self.ids, axis=0)
        import numpy as np

        ids_np = np.array(ids)
        order = np.argsort(ids_np, kind="stable")
        sorted_ids = ids_np[order]
        n = int(experts)
        bounds = np.searchsorted(sorted_ids, np.arange(n + 1))
        Xs = X[mx.array(order)]
        counts = (bounds[1:] - bounds[:-1]).tolist()

        def h_bank(start, stop):
            I = X.shape[-1]
            out = []
            for e in range(start, stop):
                a, b = int(bounds[e]), int(bounds[e + 1])
                if b > a:
                    Xe = Xs[a:b].astype(mx.float32)
                    out.append(Xe.T @ Xe)
                else:
                    out.append(mx.zeros((I, I), dtype=mx.float32))
            return mx.stack(out)

        return h_bank, counts


def record(model, targets: Sequence[tuple], forward):
    """Run ``forward()`` with ``InputRecorder`` wrappers on ``targets``.

    ``targets`` are ``(parent, attribute, module, bank)``.  The recorder is
    set as an instance attribute of ``parent`` (it shadows the module's dict
    entry, which parameters()/update() keep seeing) and removed in
    ``finally``.  Returns the recorders in target order.
    """
    recorders = []
    try:
        for parent, attribute, module, bank in targets:
            if attribute in vars(parent):
                raise ValueError(f"{attribute} is already shadowed on its parent")
            recorder = InputRecorder(module, bank=bank)
            object.__setattr__(parent, attribute, recorder)
            recorders.append((parent, attribute, recorder))
        forward()
    finally:
        for parent, attribute, _recorder in recorders:
            vars(parent).pop(attribute, None)
    return [r for *_rest, r in recorders]
