"""Use row-local arithmetic for serial and drafted forwards alike; gdn_post and mlp_act match lane_qmm group sums, while norm_xs projections always use its four-part sums."""

from __future__ import annotations

import hashlib
from threading import Lock
from typing import Any

import mlx.core as mx

# Kernel source comments belong to the decoder version hash and must change only with the arithmetic.
_NORM_XS = r"""
  // Match MLX rms_looped for K > 4096 and rms_single_row otherwise: four
  // consecutive reads per thread, the same simd_sum hierarchy, and precise
  // rsqrt. The second pass keeps the BF16 normalization boundary before the
  // BF16 gain multiply. XS is derived after XO and cannot alter RMSNorm.
  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  constexpr int E = 4;
  constexpr int TPG = K > 4096 ? 1024 : 32 * ((K + 127) / 128);
  threadgroup float red[32];
  if (int(m) >= M) {
    if (t < K / 64) XS[t * MP + m] = 0.0f;
    return;
  }
  float ss = 0.0f;
  for (int r = 0; r < K; r += TPG * E) {
    const int base = int(m) * K + r + int(t) * E;
    for (int i = 0; i < E; i++) if (r + int(t) * E + i < K) {
      bfloat h = H[base + i];
      RESIDUAL_ADD
      const float hf = float(h);
      ss += hf * hf;
    }
  }
  ss = simd_sum(ss);
  if (simdgroup_index_in_threadgroup == 0) red[thread_index_in_simdgroup] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_index_in_threadgroup == 0) {
    ss = simd_sum(red[thread_index_in_simdgroup]);
    if (thread_index_in_simdgroup == 0)
      red[0] = metal::precise::rsqrt(ss / float(K) + eps[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = red[0];
  for (int r = 0; r < K; r += TPG * E) {
    const int base = int(m) * K + r + int(t) * E;
    for (int i = 0; i < E; i++) if (r + int(t) * E + i < K) {
      bfloat h = H[base + i];
      RESIDUAL_RELOAD
      const bfloat normalized = bfloat(float(h) * inv);
      XO[base + i] = bfloat(Wt[r + int(t) * E + i] * normalized);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < K / 64) {
    float gs = 0.0f;
    for (int i = 0; i < 64; i++) gs += float(XO[int(m) * K + int(t) * 64 + i]);
    XS[t * MP + m] = gs;
  }
"""

_GDN_PRE = r"""
  // one simdgroup per (row w, head): q heads [0, NK), k heads [NK, 2 NK), v heads [2 NK, 2 NK + NV)
  const uint lane = thread_index_in_simdgroup;
  const uint head = threadgroup_position_in_grid.y;
  const uint w = threadgroup_position_in_grid.z;
  constexpr int C = 2 * NK * DK + NV * DV;
  const bool isq = head < NK, isk = !isq && head < 2 * NK;
  const int c0 = isq ? int(head) * DK : (isk ? NK * DK + (int(head) - NK) * DK : 2 * NK * DK + (int(head) - 2 * NK) * DV);
  constexpr int PER = DK / 32;                          // channels per lane (DK == DV)
  float vals[PER];
  for (int j = 0; j < PER; j++) {
    const int c = c0 + int(lane) * PER + j;
    float acc = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int row = windows[w * TAPS + tap];          // into [conv state rows; window rows]
      const float xv = row < TAPS - 1 ? float(CS[row * C + c]) : float(QKV[(row - (TAPS - 1)) * C + c]);
      acc += float(CW[c * TAPS + tap]) * xv;
    }
    // Compiled nn.silu on the served Qwen3.8 path evaluates the stable,
    // symmetric fast-exp sigmoid in bf16, then rounds the product to bf16.
    const bfloat conv = bfloat(acc);
    const bfloat sigmoid_exp = bfloat(metal::exp(metal::abs(conv)));
    const bfloat sigmoid_low = bfloat(bfloat(1.0f) / bfloat(bfloat(1.0f) + sigmoid_exp));
    const bfloat sigmoid = conv < bfloat(0.0f)
        ? sigmoid_low : bfloat(bfloat(1.0f) - sigmoid_low);
    vals[j] = float(bfloat(conv * sigmoid));
  }
  if (isq || isk) {
    float ss = 0.0f;
    for (int j = 0; j < PER; j++) ss += vals[j] * vals[j];
    ss = simd_sum(ss);
    // mlx_lm expresses FLA's L2 epsilon through RMSNorm: dividing the sum by
    // DK must divide the 1e-6 epsilon by DK as well.
    const float norm_eps = 1e-6f / float(DK);
    const float inv = metal::precise::rsqrt(ss / float(DK) + norm_eps);
    // mlx_lm: q = (DK^-0.5)^2 * rms_norm(q), k = DK^-0.5 * rms_norm(k), scales rounded to bf16
    const float scale = isq ? float(bfloat(1.0f / float(DK)))
                            : float(bfloat(metal::precise::rsqrt(float(DK))));
    for (int j = 0; j < PER; j++) {
      const bfloat out = bfloat(scale * float(bfloat(vals[j] * inv)));
      if (isq) Q[(w * NK + head) * DK + lane * PER + j] = out;
      else Kout[(w * NK + head - NK) * DK + lane * PER + j] = out;
    }
  } else {
    const int hv = int(head) - 2 * NK;
    for (int j = 0; j < PER; j++) Vout[(w * NV + hv) * DV + lane * PER + j] = bfloat(vals[j]);
    if (lane == 0) {
      // compute_g keeps a + dt_bias and softplus in bf16.  MLX's softplus is
      // max/min stabilized and uses a compensated log1p; both exponentials
      // around it are precise fp32 operations.
      const bfloat av = bfloat(float(Ain[w * NV + hv]) + float(DT[hv]));
      const bfloat zero = bfloat(0.0f);
      const bfloat hi = metal::max(av, zero);
      const bfloat lo = metal::min(av, zero);
      const bfloat softplus_arg = bfloat(metal::exp(lo - hi));
      const float arg = float(softplus_arg);
      const float arg_plus_one = 1.0f + arg;
      const float log1p = arg_plus_one == 1.0f
          ? arg : arg * (metal::log(arg_plus_one) / (arg_plus_one - 1.0f));
      const bfloat sp = metal::isnan(av)
          ? metal::numeric_limits<bfloat>::quiet_NaN()
          : ((lo == -metal::numeric_limits<bfloat>::infinity()
              || hi == metal::numeric_limits<bfloat>::infinity())
             ? hi : bfloat(hi + bfloat(log1p)));
      G[w * NV + hv] = metal::precise::exp(
          -metal::precise::exp(float(ALOG[hv])) * float(sp));
      const float beta_x = float(Bin[w * NV + hv]);
      const float beta_y = 1.0f / (1.0f + metal::precise::exp(metal::abs(beta_x)));
      BETA[w * NV + hv] = beta_x < 0.0f ? beta_y : 1.0f - beta_y;
    }
  }
"""

_GDN_POST = r"""
  // one simdgroup per (row, v head): out = SiLU(z) * RMSNorm(y) * w, and the out projection's group sums
  const uint lane = thread_index_in_simdgroup;
  const uint hv = threadgroup_position_in_grid.y;
  const uint m = threadgroup_position_in_grid.z;
  const int M = dims[0], MP = dims[1];
  constexpr int PER = DV / 32;
  constexpr int GPH = DV / 64;                          // 64-groups per head
  threadgroup bfloat ob[DV];
  if (int(m) >= M) {
    if (lane < GPH) XS[(hv * GPH + lane) * MP + m] = 0.0f;
    return;
  }
  float yv[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; j++) {
    yv[j] = float(Y[(m * NV + hv) * DV + lane * PER + j]);
    ss += yv[j] * yv[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::precise::rsqrt(ss / float(DV) + eps[0]);
  for (int j = 0; j < PER; j++) {
    const int d = int(lane) * PER + j;
    // mx.fast.rms_norm materializes bf16 before its bf16 gain multiply.
    const bfloat normalized = bfloat(yv[j] * inv);
    const bfloat normed = bfloat(NW[d] * normalized);
    const float zf = float(Z[m * NV * DV + hv * DV + d]);
    // Qwen3.8's swish gate widens z to fp32 and compiled nn.silu uses the
    // stable symmetric sigmoid; its exp is respelled to the served one.
    const float gate_exp = metal::exp(metal::abs(zf));
    const float gate_low = 1.0f / (1.0f + gate_exp);
    const float gate_sigmoid = zf < 0.0f ? gate_low : 1.0f - gate_low;
    const float gate = zf * gate_sigmoid;
    const bfloat o = bfloat(float(normed) * gate);
    OUT[m * NV * DV + hv * DV + d] = o;
    ob[d] = o;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < GPH) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(ob[lane * 64 + i]);
    XS[(hv * GPH + lane) * MP + m] = acc;
  }
"""

_MLP_ACT = r"""
  // 64 threads per (row, 64-group): h = SiLU(gate) * up, and the down projection's group sums
  const uint t = thread_position_in_threadgroup.x;
  const uint g = threadgroup_position_in_grid.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  threadgroup bfloat hb[64];
  if (int(m) >= M) {
    if (t == 0) XS[g * MP + m] = 0.0f;
    return;
  }
  const int e = int(m) * N + int(g) * 64 + int(t);
  const bfloat gf = GATE[e];
  // compiled swiglu's stable symmetric bf16 sigmoid; its exp is respelled to the served one.
  const bfloat mlp_sigmoid_exp = bfloat(metal::exp(metal::abs(gf)));
  const bfloat mlp_sigmoid_low = bfloat(bfloat(1.0f) / bfloat(bfloat(1.0f) + mlp_sigmoid_exp));
  const bfloat mlp_sigmoid = gf < bfloat(0.0f)
      ? mlp_sigmoid_low : bfloat(bfloat(1.0f) - mlp_sigmoid_low);
  const bfloat activated = bfloat(gf * mlp_sigmoid);
  const bfloat h = bfloat(activated * UP[e]);
  HOUT[e] = h;
  hb[t] = h;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(hb[i]);
    XS[g * MP + m] = acc;
  }
"""

# mlx2 modification (2026-10-17): served-SiLU spelling.  TensorFold wrote the
# mlp_act and gdn_post sigmoids with ``metal::exp``.  MLX 0.32.4 serves the
# compiled ``swiglu``/``nn.silu`` with ``metal::precise::exp``; the two bf16
# sigmoids round apart at -6.84375, which real MLP gates reach, so a fixed
# spelling made tree rows differ from ordinary decode on that build while
# random-input parity checks passed.  Each kernel's own sigmoid text (sliced
# below) is probed against the served op over every bf16 encoding (fp32 for
# the widened gdn_post gate: every widened bf16 plus a random-bits sample)
# with ``served_exp.ServedExpGate``, the kernel compiles the matching
# spelling, and a once-per-process kernel canary on a value sweep against the
# served eager ops must pass.  On a refusal the entry points compute the
# served eager op instead (exact, no fused group sums) and count it.
_EXP = "metal::exp"


def _slice(source: str, start: str, stop: str) -> str:
    begin = source.index(start)
    return source[begin:source.index(stop, begin)]


_MLP_SIGMOID = _slice(_MLP_ACT, "  const bfloat mlp_sigmoid_exp", "  const bfloat h =")
_POST_SIGMOID = _slice(_GDN_POST, "    const float gate_exp", "    const bfloat o =")
for _name, _text in (("mlp_act", _MLP_ACT), ("gdn_post", _GDN_POST)):
    if _text.count(_EXP) != 1:
        raise RuntimeError(f"lane_glue.{_name}: expected one {_EXP} site")
_SERVED: dict[str, Any] = {}       # kernel -> (ServedExpGate, probe dtype)
_PARITY: dict[tuple, str | None] = {}
_FALLBACK_CALLS = {"mlp_act": 0, "gdn_post": 0}
_PARITY_LOCK = Lock()
_GATE_LOCK = Lock()   # separate: a parity check holds _PARITY_LOCK while it builds kernels


def _served_swiglu(x: mx.array) -> mx.array:
    from mlx2.runtime.models.activations import swiglu

    return swiglu(x, mx.ones_like(x))


def _gate(name: str) -> tuple[Any, Any]:
    """The kernel's ``ServedExpGate`` and the dtype its sigmoid runs in (one per process)."""

    hit = _SERVED.get(name)
    if hit is not None:
        return hit
    with _GATE_LOCK:
        if name not in _SERVED:
            _SERVED[name] = _new_gate(name)
        return _SERVED[name]


def _new_gate(name: str) -> tuple[Any, Any]:
    import mlx.nn as nn

    from mlx2.runtime.models.served_exp import ServedExpGate

    if name == "mlp_act":
        gate = ServedExpGate("tensorfold_lane_glue_mlp_act", served_name="compiled swiglu",
                             served=_served_swiglu,
                             body="  const bfloat gf = x;\n" + _MLP_SIGMOID + "  y = activated;")
        return gate, mx.bfloat16
    if name == "gdn_post":
        gate = ServedExpGate("tensorfold_lane_glue_gdn_post", served_name="compiled nn.silu (fp32 gate)",
                             served=nn.silu,
                             body="    const float zf = float(x);\n" + _POST_SIGMOID + "    y = gate;")
        return gate, mx.float32
    raise KeyError(name)


def served_spelling(name: str) -> str | None:
    """The exp spelling that reproduces ``name``'s served op (Metal), or None."""

    gate, dtype = _gate(name)
    return gate.spelling(dtype)


def served_source(name: str) -> str:
    """``mlp_act``/``gdn_post`` source with its sigmoid spelled as the served op."""

    from mlx2.runtime.models.served_exp import respell

    source = {"mlp_act": _MLP_ACT, "gdn_post": _GDN_POST}[name]
    spelling = served_spelling(name)
    if spelling is None:
        gate, dtype = _gate(name)
        raise RuntimeError(f"lane_glue.{name}: {gate.adaptation_refusal(dtype)}")
    return source if spelling == _EXP else respell(source, _EXP, spelling)


def _sweep(count: int) -> mx.array:
    """Every finite bf16 value at least once, tiled to ``count`` elements in a fixed order."""

    import numpy as np

    from mlx2.runtime.models.served_exp import covering_rows

    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    finite = np.isfinite((bits.astype(np.uint32) << 16).view(np.float32))
    grid = mx.array(bits[finite]).view(mx.bfloat16)
    return covering_rows(grid, 1, count, 10170).reshape(count)


def _bits_equal(a: mx.array, b: mx.array) -> bool:
    from mlx2.runtime.models.served_exp import same_bits

    return same_bits(a, b)


def mlp_act_reference(gate: mx.array, up: mx.array) -> mx.array:
    """The served (ordinary decode) MLP activation: compiled ``swiglu``."""

    from mlx2.runtime.models.activations import swiglu

    return swiglu(gate, up)


def gdn_post_reference(y: mx.array, z: mx.array, weight: mx.array, eps: float) -> mx.array:
    """The served Qwen3.8 GDN norm gate (``Qwen3NextRMSNormGated``), [1, W, nv * dv]."""

    from mlx2.runtime.models.qwen3_next import _precise_swiglu

    _, W, nv, dv = (int(s) for s in y.shape)
    out = _precise_swiglu(y, z.reshape(1, W, nv, dv), mx.fast.rms_norm(y, weight, eps))
    return out.reshape(1, W, nv * dv)


def parity_case(name: str) -> tuple[tuple, mx.array]:
    """Value-sweep inputs for ``name`` and the served eager op's output on them.

    ``mlp_act``: every finite bf16 gate value (16 rows of 4096) times a random
    up; ``gdn_post``: every finite bf16 z value (11 rows at the Qwen3.8 27B
    48 x 128 head geometry) against a random y and norm weight.
    """

    if name == "mlp_act":
        rows, n = 16, 4096
        gate = _sweep(rows * n).reshape(1, rows, n)
        up = mx.random.normal((1, rows, n), key=mx.random.key(10171)).astype(mx.bfloat16)
        return (gate, up), mlp_act_reference(gate, up)
    if name == "gdn_post":
        rows, nv, dv, eps = 11, 48, 128, 1e-6
        z = _sweep(rows * nv * dv).reshape(1, rows, nv * dv)
        y = mx.random.normal((1, rows, nv, dv), key=mx.random.key(10172)).astype(mx.bfloat16)
        weight = (1 + 0.1 * mx.random.normal((dv,), key=mx.random.key(10173))).astype(mx.bfloat16)
        return (y, z, weight, eps), gdn_post_reference(y, z, weight, eps)
    raise KeyError(name)


def _parity(name: str) -> str | None:
    (args, want) = parity_case(name)
    have = (_mlp_act_kernel if name == "mlp_act" else _gdn_post_kernel)(*args)
    if not _bits_equal(have, want):
        return f"lane_glue.{name}: kernel differs from the served eager op on the value sweep"
    return None


def checked_parity(key: tuple, check: Any) -> str | None:
    """Memoize one parity verdict per process (device faults are re-raised)."""

    if key in _PARITY:
        return _PARITY[key]
    with _PARITY_LOCK:
        if key not in _PARITY:
            from mlx2.runtime.models.served_exp import is_device_fault

            try:
                _PARITY[key] = check()
            except Exception as exc:  # noqa: BLE001 - fail closed
                if is_device_fault(exc):
                    raise
                _PARITY[key] = f"{key[0]}.{key[1]}: parity probe failed: {type(exc).__name__}"
        return _PARITY[key]


def refusal(name: str) -> str | None:
    """Why ``mlp_act``/``gdn_post`` may not stand in for the served op, or None.

    Once per process (Metal): the sigmoid expression against the served op
    (``ServedExpGate``), then the compiled kernel against the served eager op
    on a value sweep (``parity_case``).
    """

    gate, dtype = _gate(name)
    reason = gate.adaptation_refusal(dtype)
    if reason is not None:
        return f"lane_glue.{name}: {reason}"
    return checked_parity(("lane_glue", name, served_spelling(name)), lambda: _parity(name))


def stats() -> dict[str, Any]:
    """Spelling, parity verdict and eager-fallback count per gated kernel."""

    out: dict[str, Any] = {"backend": "mlx2_tensorfold_lane_glue_served_exp_v1"}
    for name in ("mlp_act", "gdn_post"):
        gate, dtype = _SERVED.get(name, (None, None))
        spelling = gate._spellings.get(str(dtype)) if gate is not None else None
        verdict = _PARITY.get(("lane_glue", name, spelling), "unchecked")
        out[name] = {
            "exp": spelling,
            "parity": "exact" if verdict is None else verdict,
            "fallback_calls": _FALLBACK_CALLS[name],
        }
    return out


_kernels: dict[str, Any] = {}


def _named(base: str, source: str) -> str:
    return f"{base}_{hashlib.sha256(source.encode()).hexdigest()[:16]}"


def _kernel(name: str) -> Any:
    if name in ("mlp_act", "gdn_post"):
        key = f"{name}:{served_spelling(name)}"
    else:
        key = name
    if key not in _kernels:
        spec = {
            "norm": (_NORM_XS.replace("RESIDUAL_ADD", "h = bfloat(float(h) + float(R[base + i]));\n      HO[base + i] = h;")
                     .replace("RESIDUAL_RELOAD", "h = HO[base + i];"),
                     ["H", "R", "Wt", "eps", "dims"], ["HO", "XO", "XS"]),
            "norm_nores": (_NORM_XS.replace("RESIDUAL_ADD", "").replace("RESIDUAL_RELOAD", ""),
                           ["H", "Wt", "eps", "dims"], ["XO", "XS"]),

            "gdn_pre": (_GDN_PRE, ["QKV", "CS", "CW", "windows", "Ain", "Bin", "ALOG", "DT"], ["Q", "Kout", "Vout", "G", "BETA"]),
            "gdn_post": (None, ["Y", "Z", "NW", "eps", "dims"], ["OUT", "XS"]),
            "mlp_act": (None, ["GATE", "UP", "dims"], ["HOUT", "XS"]),
        }[name]
        source, inputs, outputs = spec
        if source is None:
            source = served_source(name)
        _kernels[key] = mx.fast.metal_kernel(name=_named("lane_glue_" + name, source), input_names=inputs,
                                             output_names=outputs, source=source)
    return _kernels[key]


_consts: dict[Any, mx.array] = {}


def _const(key: Any, make: Any) -> mx.array:
    if key not in _consts:
        _consts[key] = make()
    return _consts[key]


def _dims(m: int) -> mx.array:
    mp = 16 * ((m + 15) // 16)
    return _const(("dims", m), lambda: mx.array([m, mp], dtype=mx.int32))


def _eps(eps: float) -> mx.array:
    return _const(("eps", float(eps)), lambda: mx.array([float(eps)], dtype=mx.float32))


def remember(x: mx.array, xs: mx.array) -> mx.array:
    """Hand ``xs`` to the lane matmul as ``x``'s group sums (its XSUM kernel then does not run)."""

    from . import lane_qmm

    lane_qmm._xs_cache[id(x)] = (x, xs)
    while len(lane_qmm._xs_cache) > 4:
        lane_qmm._xs_cache.pop(next(iter(lane_qmm._xs_cache)))
    return x


def norm_xs(hidden: mx.array, residual: mx.array | None, weight: mx.array, eps: float
            ) -> tuple[mx.array, mx.array]:
    """Return h = hidden + residual, or hidden if residual is None, and RMSNorm(h) * weight as (1, M, K) bf16 arrays with norm output group sums."""

    lead = hidden.shape[:-1]
    K = int(hidden.shape[-1])
    M = 1
    for d in lead:
        M *= int(d)
    MP = 16 * ((M + 15) // 16)
    if K % 512 or K > 16384:
        raise ValueError(f"norm_xs: the hidden size must be a multiple of 512 up to 16384, got {K}")
    tpg = 1024 if K > 4096 else 32 * ((K + 127) // 128)
    common = {"grid": (tpg, MP, 1), "threadgroup": (tpg, 1, 1)}
    if residual is None:
        x, xs = _kernel("norm_nores")(
            inputs=[hidden.reshape(M, K), weight, _eps(eps), _dims(M)],
            template=[("K", K)],
            output_shapes=[(M, K), (K // 64, MP)], output_dtypes=[mx.bfloat16, mx.float32], **common)
        h = hidden
    else:
        h, x, xs = _kernel("norm")(
            inputs=[hidden.reshape(M, K), residual.reshape(M, K), weight, _eps(eps), _dims(M)],
            template=[("K", K)],
            output_shapes=[(M, K), (M, K), (K // 64, MP)], output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32],
            **common)
        h = h.reshape(*lead, K)
    x = x.reshape(*lead, K)
    return h, remember(x, xs)


def gdn_pre(qkv: mx.array, conv_state: mx.array, conv_weight: mx.array, windows: mx.array, a: mx.array,
            b: mx.array, a_log: mx.array, dt_bias: mx.array, *, nk: int, nv: int, dk: int, dv: int
            ) -> tuple[mx.array, ...]:
    """Return q/k [1, W, nk, dk], v [1, W, nv, dv], and fp32 g/beta [1, W, nv]; windows index [conv_state; qkv] rows."""

    W = int(qkv.shape[-2])
    C = int(qkv.shape[-1])
    taps = int(conv_weight.shape[1])
    if dk != dv or dk % 32:
        raise ValueError("gdn_pre: needs head_k_dim == head_v_dim, a multiple of 32")
    q, k, v, g, beta = _kernel("gdn_pre")(
        inputs=[qkv.reshape(W, C), conv_state.reshape(taps - 1, C), conv_weight.reshape(C, taps), windows,
                a.reshape(W, nv), b.reshape(W, nv), a_log, dt_bias],
        template=[("NK", nk), ("NV", nv), ("DK", dk), ("DV", dv), ("TAPS", taps)],
        grid=(32, 2 * nk + nv, W), threadgroup=(32, 1, 1),
        output_shapes=[(1, W, nk, dk), (1, W, nk, dk), (1, W, nv, dv), (1, W, nv), (1, W, nv)],
        output_dtypes=[qkv.dtype, qkv.dtype, qkv.dtype, mx.float32, mx.float32])
    return q, k, v, g, beta


def gdn_post(y: mx.array, z: mx.array, weight: mx.array, eps: float) -> mx.array:
    """SiLU(z) * RMSNorm(y) * weight per head: y [1, W, nv, dv], z [1, W, nv * dv] -> [1, W, nv * dv]."""

    if refusal("gdn_post") is not None:
        _FALLBACK_CALLS["gdn_post"] += 1
        return gdn_post_reference(y, z, weight, eps)
    return _gdn_post_kernel(y, z, weight, eps)


def _gdn_post_kernel(y: mx.array, z: mx.array, weight: mx.array, eps: float) -> mx.array:
    _, W, nv, dv = (int(s) for s in y.shape)
    MP = 16 * ((W + 15) // 16)
    out, xs = _kernel("gdn_post")(
        inputs=[y, z.reshape(W, nv * dv), weight, _eps(eps), _dims(W)],
        template=[("NV", nv), ("DV", dv)],
        grid=(32, nv, MP), threadgroup=(32, 1, 1),
        output_shapes=[(1, W, nv * dv), (nv * dv // 64, MP)], output_dtypes=[y.dtype, mx.float32])
    return remember(out, xs)


def mlp_act(gate: mx.array, up: mx.array) -> mx.array:
    """SiLU(gate) * up, (1, W, N) bf16, with its group sums remembered for the down projection."""

    if refusal("mlp_act") is not None:
        _FALLBACK_CALLS["mlp_act"] += 1
        return mlp_act_reference(gate, up)
    return _mlp_act_kernel(gate, up)


def _mlp_act_kernel(gate: mx.array, up: mx.array) -> mx.array:
    N = int(gate.shape[-1])
    W = int(gate.size // N)
    MP = 16 * ((W + 15) // 16)
    h, xs = _kernel("mlp_act")(
        inputs=[gate.reshape(W, N), up.reshape(W, N), _dims(W)], template=[("N", N)],
        grid=(64 * (N // 64), MP, 1), threadgroup=(64, 1, 1),
        output_shapes=[(W, N), (N // 64, MP)], output_dtypes=[gate.dtype, mx.float32])
    return remember(h.reshape(gate.shape), xs)


__all__ = ["gdn_post", "gdn_post_reference", "gdn_pre", "mlp_act", "mlp_act_reference", "norm_xs", "refusal",
           "remember", "served_source", "served_spelling", "stats"]
