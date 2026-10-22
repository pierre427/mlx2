"""mlx2-owned corrected Qwen3.8 tree/cohort GDN kernels.

Ported from the pinned corrected TensorFold source; see
``provenance/qwen38-unified-tree-gdn.json``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .served_exp import EXP_SPELLINGS, ServedExpGate, is_device_fault, respell


def ints(values: Sequence[int]) -> mx.array:
    return mx.array([int(value) for value in values], dtype=mx.int32)


def tree_paths(parents: Sequence[int]) -> tuple[list[int], list[list[int]]]:
    """Validate parent-first topology and return depths plus root paths."""

    depths: list[int] = []
    paths: list[list[int]] = []
    for row, parent in enumerate(parents):
        parent = int(parent)
        if parent < 0:
            path = [row]
        else:
            if parent >= row:
                raise ValueError("parents must come before their children")
            path = paths[parent] + [row]
        paths.append(path)
        depths.append(len(path) - 1)
    return depths, paths


_COUNTERS = {
    "pre_calls": 0,
    "pre_rows": 0,
    "tree_calls": 0,
    "tree_rows": 0,
    "tree_streams": 0,
    "replay_calls": 0,
    "replay_streams": 0,
    "tail_calls": 0,
    "tail_streams": 0,
}


def stats() -> dict[str, int | str | None]:
    # ``pre_silu_exp`` is the conv SiLU spelling the pre kernel compiled (None
    # until it first ran), ``pre_parity`` its once-per-process verdict.
    spellings = [spelling for spelling in EXP_SPELLINGS if spelling in _kernel_cache]
    built = spellings[0] if len(spellings) == 1 else (None if not spellings else ",".join(spellings))
    parity = _PARITY.get(built) if built in _PARITY else None
    return {
        "backend": "mlx2_qwen38_unified_tree_gdn_v1",
        **_COUNTERS,
        "pre_silu_exp": built,
        "pre_parity": None if built is None else ("exact" if parity is None else parity),
    }


MAX_STREAMS = 8  # conv-state buffers per launch

# The conv SiLU's sigmoid, as written by the corrected TensorFold source with
# ``metal::exp``.  MLX 0.32.4 serves compiled ``nn.silu`` with
# ``metal::precise::exp``; the two bf16 sigmoids round apart on one input
# (-6.84375, which real conv outputs reach), so a fixed spelling made the tree
# rows' q/k/v differ from the ordinary decode and prefill paths on that build.
# ``_SILU_GATE`` probes this exact text against the served ``nn.silu`` over
# every bf16 encoding and the kernel compiles the matching spelling.
_PRE_SILU = """    const bfloat sigmoid_exp = bfloat(metal::exp(metal::abs(conv)));
    const bfloat sigmoid_low = bfloat(bfloat(1.0f) / bfloat(bfloat(1.0f) + sigmoid_exp));
    const bfloat sigmoid = conv < bfloat(0.0f)
        ? sigmoid_low : bfloat(bfloat(1.0f) - sigmoid_low);
"""
_SILU_GATE = ServedExpGate(
    "qwen38_tree_gdn_pre_silu",
    served_name="compiled nn.silu",
    served=nn.silu,
    body="    const bfloat conv = x;\n" + _PRE_SILU + "    y = bfloat(conv * sigmoid);",
)

# Use lane_fuse gdn_pre arithmetic with each stream's conv state and window rows in the grouped layout.
_PRE = (
    r"""
  // one simdgroup per (row w, head): q heads [0, NK), k heads [NK, 2 NK), v heads [2 NK, 2 NK + NV)
  const uint lane = thread_index_in_simdgroup;
  const uint head = threadgroup_position_in_grid.y;
  const uint w = threadgroup_position_in_grid.z;
  const int st = row_stream[w];                         // the row's stream: its conv state
  const device bfloat16_t* CSb = CS0;
  switch (st) {
"""
    + "".join(f"    case {i}: CSb = CS{i}; break;\n" for i in range(1, MAX_STREAMS))
    + r"""  }
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
      const float xv = row < TAPS - 1 ? float(CSb[row * C + c]) : float(QKV[(row - (TAPS - 1)) * C + c]);
      acc += float(CW[c * TAPS + tap]) * xv;
    }
    // Compiled nn.silu on the served Qwen3.8 path evaluates the stable,
    // symmetric sigmoid in bf16, then rounds the product to bf16.  Its exp
    // spelling is the served one (_PRE_SILU, respelled per MLX build).
    const bfloat conv = bfloat(acc);
"""
    + "__PRE_SILU__"
    + r"""    vals[j] = float(bfloat(conv * sigmoid));
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
      const bfloat av = bfloat(float(Ain[w * ZS + AO + hv]) + float(DT[hv]));
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
      const float beta_x = float(Bin[w * ZS + BO + hv]);
      const float beta_y = 1.0f / (1.0f + metal::precise::exp(metal::abs(beta_x)));
      BETA[w * NV + hv] = beta_x < 0.0f ? beta_y : 1.0f - beta_y;
    }
  }
"""
)

KERNEL_SILU_EXP = "metal::exp"
if "__PRE_SILU__" not in _PRE:
    raise RuntimeError("stream_gdn pre source lost its SiLU site")

_INPUTS = [
    "QKV",
    *[f"CS{i}" for i in range(MAX_STREAMS)],
    "CW",
    "windows",
    "Ain",
    "Bin",
    "ALOG",
    "DT",
    "row_stream",
]
_kernel_cache: dict[str, Any] = {}
_PARITY: dict[str, str | None] = {}


def source(spelling: str = KERNEL_SILU_EXP) -> str:
    """The pre kernel source with its conv SiLU spelled ``spelling``."""
    if spelling not in EXP_SPELLINGS:
        raise ValueError(f"unknown exp spelling {spelling!r}")
    silu = _PRE_SILU if spelling == KERNEL_SILU_EXP else respell(_PRE_SILU, KERNEL_SILU_EXP, spelling)
    return _PRE.replace("__PRE_SILU__", silu)


def _build(spelling: str) -> Any:
    if spelling not in _kernel_cache:
        text = source(spelling)
        # The name carries the source digest: MLX caches libraries by name.
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        _kernel_cache[spelling] = mx.fast.metal_kernel(
            name=f"stream_gdn_pre_{digest}",
            input_names=_INPUTS,
            output_names=["Q", "Kout", "Vout", "G", "BETA"],
            source=text,
        )
    return _kernel_cache[spelling]


def served_silu_spelling() -> str | None:
    """The conv SiLU spelling that reproduces the served ``nn.silu`` (Metal)."""
    return _SILU_GATE.spelling(mx.bfloat16)


def pre_refusal() -> str | None:
    """Why the pre kernel may not stand in for the ordinary GDN prework, or None.

    Two once-per-process checks (Metal): the SiLU expression against the served
    ``nn.silu`` over every bf16 encoding (``_SILU_GATE``), then the compiled
    kernel itself against the eager chain on a sweep of conv outputs covering
    every bf16 value with magnitude in [2**-6, 32] (``_decode_parity``).  A
    random draw almost never reaches the inputs where the spellings round
    apart, which is how the fixed ``metal::exp`` passed every parity check
    while serving MLX 0.32.4.
    """
    refusal = _SILU_GATE.adaptation_refusal(mx.bfloat16)
    if refusal is not None:
        return f"stream_gdn.gdn_pre: {refusal}"
    spelling = served_silu_spelling()
    if spelling not in _PARITY:
        try:
            _PARITY[spelling] = _decode_parity(spelling)
        except Exception as exc:  # noqa: BLE001 - fail closed
            if is_device_fault(exc):
                raise
            _PARITY[spelling] = f"stream_gdn.gdn_pre: parity probe failed: {type(exc).__name__}"
    return _PARITY[spelling]


def _kernel() -> Any:
    refusal = pre_refusal()
    if refusal is not None:
        raise RuntimeError(refusal)
    return _build(served_silu_spelling())


def _parity_inputs(rows: int, nk: int, nv: int, dk: int, dv: int, taps: int):
    """Sweep conv outputs (identity last tap, zero state) plus fixed gates."""
    import numpy as np

    channels = 2 * nk * dk + nv * dv
    encodings = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    widened = (encodings.astype(np.uint32) << 16).view(np.float32)
    keep = np.isfinite(widened) & (np.abs(widened) >= 2.0 ** -6) & (np.abs(widened) <= 32.0)
    grid = mx.array(encodings[keep]).view(mx.bfloat16)
    from .served_exp import covering_rows

    qkv = covering_rows(grid, rows, channels, 9300)[None]
    weight = mx.zeros((channels, taps, 1), dtype=mx.bfloat16)
    weight[:, taps - 1, 0] = 1
    k1, k2, k3 = mx.random.split(mx.random.key(9400), 3)
    a = (mx.random.normal((1, rows, nv), key=k1) * 3).astype(mx.bfloat16)
    b = (mx.random.normal((1, rows, nv), key=k2) * 3).astype(mx.bfloat16)
    a_log = mx.linspace(-2.0, 1.0, nv).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((nv,), key=k3) * 0.5).astype(mx.bfloat16)
    state = mx.zeros((1, taps - 1, channels), dtype=mx.bfloat16)
    return qkv, state, weight, a, b, a_log, dt_bias


def _decode_parity(spelling: str, *, nk=16, nv=48, dk=128, dv=128, taps=4) -> str | None:
    """Compiled pre kernel vs the eager Qwen3.5-family chain, bit for bit.

    The eager chain is what the ordinary decode and prefill paths run:
    ``nn.silu(conv1d(.))``, ``normalize_gdn_qk``, ``compute_g`` and the
    widened ``gate_sigmoid(b)``.  Checked at the Qwen3.8 27B geometry.
    """
    from .gated_delta import compute_g, normalize_gdn_qk
    from .precise_ops import gate_sigmoid

    rows = 2
    qkv, state, weight, a, b, a_log, dt_bias = _parity_inputs(rows, nk, nv, dk, dv, taps)
    windows = ints([t + r for r in range(rows) for t in range(taps)])
    q, k, v, g, beta = _launch(
        _build(spelling), qkv, [state], weight, windows, a, b, a_log, dt_bias,
        mx.zeros((rows,), dtype=mx.int32), nk=nk, nv=nv, dk=dk, dv=dv, zs=nv, ao=0, bo=0,
    )
    conv = nn.silu(mx.conv1d(mx.concatenate([state, qkv], axis=1), weight, groups=int(qkv.shape[-1])))
    eq, ek, ev = mx.split(conv, [nk * dk, 2 * nk * dk], -1)
    eq, ek = normalize_gdn_qk(eq.reshape(1, rows, nk, dk), ek.reshape(1, rows, nk, dk))
    expected = {
        "q": eq, "k": ek, "v": ev.reshape(1, rows, nv, dv),
        "g": compute_g(a_log, a, dt_bias).astype(mx.float32),
        "beta": gate_sigmoid(b.astype(mx.float32)),
    }
    got = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    for name, want in expected.items():
        have = got[name]
        width = mx.uint16 if have.dtype.size == 2 else mx.uint32
        if have.shape != want.shape or have.dtype != want.dtype or not bool(
            mx.all(have.view(width) == want.view(width)).item()
        ):
            return f"stream_gdn.gdn_pre: {name} differs from the eager GDN prework ({spelling})"
    return None


def _launch(kernel, qkv, states, conv_weight, windows, a, b, a_log, dt_bias, row_stream, *,
            nk, nv, dk, dv, zs, ao, bo):
    rows = int(qkv.shape[-2])
    channels = int(qkv.shape[-1])
    taps = int(conv_weight.shape[1])
    spare = states[0]
    cs = [s.reshape(taps - 1, channels) for s in states] + [spare.reshape(taps - 1, channels)] * (
        MAX_STREAMS - len(states)
    )
    return tuple(
        kernel(
            inputs=[
                qkv.reshape(rows, channels),
                *cs,
                conv_weight.reshape(channels, taps),
                windows,
                a.reshape(rows, -1),
                b.reshape(rows, -1),
                a_log,
                dt_bias,
                row_stream,
            ],
            template=[
                ("NK", nk),
                ("NV", nv),
                ("DK", dk),
                ("DV", dv),
                ("TAPS", taps),
                ("ZS", zs),
                ("AO", ao),
                ("BO", bo),
            ],
            grid=(32, 2 * nk + nv, rows),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (1, rows, nk, dk),
                (1, rows, nk, dk),
                (1, rows, nv, dv),
                (1, rows, nv),
                (1, rows, nv),
            ],
            output_dtypes=[qkv.dtype, qkv.dtype, qkv.dtype, mx.float32, mx.float32],
        )
    )


class ConvPlan:
    """Index each row's final n_keep + 1 conv inputs from its stream's conv state and ancestor path into state rows plus all window rows."""

    def __init__(self, parents: Sequence[Sequence[int]], n_keep: int) -> None:
        if not 1 <= len(parents) <= MAX_STREAMS:
            raise ValueError(f"stream_gdn: 1 to {MAX_STREAMS} streams a launch")
        flat: list[int] = []
        streams: list[int] = []
        first = 0
        for s, rp in enumerate(parents):
            _, paths = tree_paths(rp)
            for path in paths:
                rows = list(range(n_keep)) + [n_keep + first + r for r in path]
                flat += rows[-(n_keep + 1) :]
                streams.append(s)
            first += len(rp)
        self.rows, self.streams = first, len(parents)
        self.n_keep = n_keep
        self.windows = ints(flat)
        self.row_stream = ints(streams)


def _require_taps(where, taps, plan_taps, state_sizes, channels):
    """The pre kernel reads TAPS - 1 conv-state rows and indexes windows by
    TAPS: one tap leaves an empty state (its row clamp would be -1), and a
    plan or state built for another tap count reads out of step."""
    if taps < 2:
        raise ValueError(f"stream_gdn.{where}: conv taps must be at least 2, got {taps}")
    if plan_taps != taps:
        raise ValueError(f"stream_gdn.{where}: plan built for {plan_taps} taps, weight has {taps}")
    if any(size != (taps - 1) * channels for size in state_sizes):
        raise ValueError(f"stream_gdn.{where}: conv state must hold taps - 1 = {taps - 1} rows")


def gdn_pre(
    qkv: mx.array,
    states: Sequence[mx.array],
    conv_weight: mx.array,
    plan: ConvPlan,
    zba: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    *,
    nk: int,
    nv: int,
    dk: int,
    dv: int,
) -> tuple[mx.array, ...]:
    """Return q/k [1, R, nk, dk], v [1, R, nv, dv], and fp32 g/beta [1, R, nv] from grouped projections and per-stream conv states."""

    R = int(qkv.shape[-2])
    C = int(qkv.shape[-1])
    taps = int(conv_weight.shape[1])
    zs = int(zba.shape[-1])
    _require_taps("gdn_pre", taps, plan.n_keep + 1, (int(s.size) for s in states), C)
    if (
        R != plan.rows
        or dk != dv
        or dk % 32
        or zs != nv * dv + 2 * nv
        or len(states) > MAX_STREAMS
    ):
        raise ValueError(
            "stream_gdn.gdn_pre: rows, head sizes or [z | b | a] width do not match"
        )
    zba2 = zba.reshape(R, zs)
    result = _launch(
        _kernel(), qkv, states, conv_weight, plan.windows, zba2, zba2, a_log, dt_bias,
        plan.row_stream, nk=nk, nv=nv, dk=dk, dv=dv, zs=zs, ao=nv * dv + nv, bo=nv * dv,
    )
    _COUNTERS["pre_calls"] += 1
    _COUNTERS["pre_rows"] += R
    return result


def gdn_pre_split(
    qkv: mx.array,
    conv_state: mx.array,
    conv_weight: mx.array,
    windows: mx.array,
    a: mx.array,
    b: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    *,
    nk: int,
    nv: int,
    dk: int,
    dv: int,
) -> tuple[mx.array, ...]:
    """Corrected single-stream pre-processing for split ``a``/``b`` inputs."""

    rows = int(qkv.shape[-2])
    channels = int(qkv.shape[-1])
    taps = int(conv_weight.shape[1])
    _require_taps(
        "gdn_pre_split", taps, int(windows.size) // max(rows, 1), (int(conv_state.size),), channels
    )
    if int(windows.size) != rows * taps:
        raise ValueError("stream_gdn.gdn_pre_split: windows must hold rows x taps entries")
    if (
        dk != dv
        or dk % 32
        or int(a.shape[-2]) != rows
        or int(b.shape[-2]) != rows
        or int(a.shape[-1]) != nv
        or int(b.shape[-1]) != nv
    ):
        raise ValueError("stream_gdn.gdn_pre_split: rows or head sizes do not match")
    row_stream = mx.zeros((rows,), dtype=mx.int32)
    result = _launch(
        _kernel(), qkv, [conv_state], conv_weight, windows, a, b, a_log, dt_bias,
        row_stream, nk=nk, nv=nv, dk=dk, dv=dv, zs=nv, ao=0, bo=0,
    )
    _COUNTERS["pre_calls"] += 1
    _COUNTERS["pre_rows"] += rows
    return result


# Walk nodes parents-first from each stream's committed state using mlx_lm gated_delta_step arithmetic.
_TREE = (
    r"""
        auto n = thread_position_in_grid.z;                 // (stream, head)
        const int st = int(n) / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        const device float* Sb = S0;
        switch (st) {
"""
    + "".join(f"          case {i}: Sb = S{i}; break;\n" for i in range(1, MAX_STREAMS))
    + r"""        }
        auto i_state = Sb + (hv_idx * Dv + dv_idx) * Dk;
        float s0[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          s0[i] = static_cast<float>(i_state[s_idx]);
        }
        float states[MAXW][n_per_t];
        const int first = meta[2 * st], W = meta[2 * st + 1];   // the stream's rows among all rows
        for (int node = 0; node < W; ++node) {
          const int parent = parents[first + node];
          float state[n_per_t];
          // a chain keeps one slot: each node's parent is the node before it
          for (int i = 0; i < n_per_t; ++i) state[i] = parent < 0 ? s0[i] : states[CHAIN ? 0 : parent][i];
          const int row = first + node;
          auto q_ = q + (row * Hk + hk_idx) * Dk;
          auto k_ = k + (row * Hk + hk_idx) * Dk;
          auto v_ = v + (row * Hv + hv_idx) * Dv;
          const float g_ = static_cast<float>(g[row * Hv + hv_idx]);
          const float beta_ = static_cast<float>(beta[row * Hv + hv_idx]);
          // --- mlx_lm gated_delta_step, one step, verbatim arithmetic ---
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_;
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (v_[dv_idx] - kv_mem) * beta_;
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[(row * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }
          for (int i = 0; i < n_per_t; ++i) states[CHAIN ? 0 : node][i] = state[i];
        }
"""
)

MAX_TREE = (
    32  # a branching window's rows (per-thread state slots); chains take up to 128
)

# Replay each stream's kept rows with lane_tree arithmetic, one threadgroup column per stream and head.
_REPLAY = (
    r"""
        auto n = thread_position_in_grid.z;                 // (stream, head)
        const int st = int(n) / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        const device float* Sb = S0;
        device float* Ob = O0;
        switch (st) {
"""
    + "".join(
        f"          case {i}: Sb = S{i}; Ob = O{i}; break;\n"
        for i in range(1, MAX_STREAMS)
    )
    + r"""        }
        auto i_state = Sb + (hv_idx * Dv + dv_idx) * Dk;
        auto o_state = Ob + (hv_idx * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        const int steps = meta[2 * st + 1];
        for (int j = 0; j < steps; ++j) {                    // the accepted path's rows, in order
          const int row = rows[meta[2 * st] + j];
          auto q_ = q + (row * Hk + hk_idx) * Dk;
          auto k_ = k + (row * Hk + hk_idx) * Dk;
          auto v_ = v + (row * Hv + hv_idx) * Dv;
          const float g_ = static_cast<float>(g[row * Hv + hv_idx]);
          const float beta_ = static_cast<float>(beta[row * Hv + hv_idx]);
          // --- mlx_lm gated_delta_step, one step, verbatim arithmetic ---
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_;
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (v_[dv_idx] - kv_mem) * beta_;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
          }
        }
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = state[i];
"""
)

# each stream's next conv state: the last taps - 1 rows of [its conv state; its kept rows], copied
_TAILS = (
    r"""
  const uint c = thread_position_in_grid.x;
  const uint t = thread_position_in_grid.y;
  const uint st = thread_position_in_grid.z;
  const device bfloat16_t* Cb = CS0;
  device bfloat16_t* Tb = T0;
  switch (st) {
"""
    + "".join(
        f"    case {i}: Cb = CS{i}; Tb = T{i}; break;\n" for i in range(1, MAX_STREAMS)
    )
    + r"""  }
  const int src = tails[st * NKEEP + t];              // < NKEEP: a conv state row; else NKEEP + a row of all rows
  if (int(c) < C) Tb[t * C + c] = src < NKEEP ? Cb[src * C + c] : QKV[(src - NKEEP) * C + c];
"""
)

def _fma_order(source: str) -> str:
    """The same step with every multiply-add an explicit ``metal::fma`` in loop order.

    Served sources leave contraction and reassociation of the step to the
    compiler, which compiles the tree walk (two state-slot variants) and the
    commit replay as separate kernels.  Spelled out, the walk's step and the
    replay's step are one sequence of roundings, so a position's recurrent
    state does not depend on whether a commit replay or an in-tree step
    produced it (``tensorfold-tree-position-v1``; Kapitsa 55b6153's GDN
    update order, ported).
    """

    steps = (
        ("kv_mem += state[i] * k_[s_idx];",
         "kv_mem = metal::fma(state[i], static_cast<float>(k_[s_idx]), kv_mem);"),
        ("auto delta = (v_[dv_idx] - kv_mem) * beta_;",
         "const float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_;"),
        ("state[i] = state[i] + k_[s_idx] * delta;",
         "state[i] = metal::fma(static_cast<float>(k_[s_idx]), delta, state[i]);"),
        ("out += state[i] * q_[s_idx];",
         "out = metal::fma(state[i], static_cast<float>(q_[s_idx]), out);"),
    )
    for old, new in steps:
        if old in source:
            source = source.replace(old, new)
        elif "out +=" not in old:
            raise RuntimeError(f"GDN step source lost {old!r}")
    if "metal::fma(state[i], static_cast<float>(k_" not in source:
        raise RuntimeError("GDN fma order did not apply")
    return source


GDN_ORDERS = ("served", "fma")
_commit_kernels: dict[str, Any] = {}


def _commit_kernel(name: str, order: str = "served") -> Any:
    if order not in GDN_ORDERS:
        raise ValueError(f"GDN step order must be one of {GDN_ORDERS}")
    key = name if order == "served" else f"{name}:{order}"
    if key not in _commit_kernels:
        _commit_kernels[key] = _build_commit_kernel(name, order)
    return _commit_kernels[key]


def _build_commit_kernel(name: str, order: str) -> Any:
    if name == "tree":
        inputs = ["q", "k", "v", "g", "beta", *[f"S{i}" for i in range(MAX_STREAMS)], "parents", "meta"]
        source, outputs = _TREE, ["y"]
    elif name == "replay":
        inputs = ["q", "k", "v", "g", "beta", *[f"S{i}" for i in range(MAX_STREAMS)], "rows", "meta"]
        source, outputs = _REPLAY, [f"O{i}" for i in range(MAX_STREAMS)]
    else:
        if order != "served":
            raise ValueError("conv tails are copies: they have no step order")
        inputs = [*[f"CS{i}" for i in range(MAX_STREAMS)], "QKV", "tails"]
        source, outputs = _TAILS, [f"T{i}" for i in range(MAX_STREAMS)]
    if order == "fma":
        source = _fma_order(source)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name=f"stream_gdn_{name}_{digest}",
        input_names=inputs,
        output_names=outputs,
        source=source,
    )


class TreePlan:
    """Share each stream's window and parent indices across recurrent layers, with parents first and -1 denoting the root."""

    def __init__(self, parents: Sequence[Sequence[int]]) -> None:
        if not 1 <= len(parents) <= MAX_STREAMS:
            raise ValueError(f"stream_gdn: 1 to {MAX_STREAMS} streams a launch")
        flat: list[int] = []
        meta: list[int] = []
        chain = True
        widest = 1
        for rp in parents:
            tree_paths(rp)  # parents come before their children
            meta += [len(flat), len(rp)]
            flat += [int(q) for q in rp]
            chain = chain and list(rp) == list(range(-1, len(rp) - 1))
            widest = max(widest, len(rp))
        if widest > (128 if chain else MAX_TREE):
            raise ValueError(
                f"stream_gdn: windows of {widest} rows (trees take up to {MAX_TREE}, chains 128)"
            )
        self.streams, self.rows, self.chain = len(parents), len(flat), chain
        self.slots = (
            1 if chain else (16 if widest <= 16 else MAX_TREE)
        )  # per-thread state slots (compiled variants)
        self.parents, self.meta = ints(flat), ints(meta)


def tree(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    states: Sequence[mx.array],
    plan: TreePlan,
    *,
    order: str = "served",
) -> mx.array:
    """Return recurrence outputs [1, R, Hv, Dv] by walking each row's path from its stream's own fp32 state."""

    _, R, Hk, Dk = (int(d) for d in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    if R != plan.rows or any(s.dtype != mx.float32 for s in states):
        raise ValueError("stream_gdn.tree: rows or state dtype do not match")
    S = len(states)
    result = _commit_kernel("tree", order)(
        inputs=[
            q,
            k,
            v,
            g,
            beta,
            *states,
            *[states[0]] * (MAX_STREAMS - S),
            plan.parents,
            plan.meta,
        ],
        template=[
            ("InT", q.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("MAXW", plan.slots),
            ("CHAIN", plan.chain),
        ],
        grid=(32, Dv, Hv * S),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, R, Hv, Dv)],
        output_dtypes=[q.dtype],
    )[0]
    _COUNTERS["tree_calls"] += 1
    _COUNTERS["tree_rows"] += R
    _COUNTERS["tree_streams"] += S
    return result


class CommitPlan:
    """Locate each stream's kept rows and next conv tail among grouped rows, with paths ordered root first."""

    def __init__(
        self, paths: Sequence[Sequence[int]], firsts: Sequence[int], n_keep: int
    ) -> None:
        if not 1 <= len(paths) <= MAX_STREAMS:
            raise ValueError(f"stream_gdn: 1 to {MAX_STREAMS} streams a launch")
        rows: list[int] = []
        meta: list[int] = []
        tails: list[int] = []
        for path, first in zip(paths, firsts):
            meta += [len(rows), len(path)]
            rows += [int(first) + int(r) for r in path]
            tails += (
                list(range(n_keep)) + [n_keep + int(first) + int(r) for r in path]
            )[-n_keep:]
        self.streams, self.n_keep = len(paths), n_keep
        self.rows, self.meta, self.tails = ints(rows), ints(meta), ints(tails)


def replay(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    states: Sequence[mx.array],
    plan: CommitPlan,
    *,
    order: str = "served",
) -> list[mx.array]:
    """Return each stream's state after serial steps over its kept rows using lane_tree replay arithmetic."""

    _, _, Hk, Dk = (int(d) for d in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    if any(s.dtype != mx.float32 for s in states):
        raise ValueError("stream_gdn.replay: fp32 states")
    S = len(states)
    spare = states[0]
    outs = _commit_kernel("replay", order)(
        inputs=[
            q,
            k,
            v,
            g,
            beta,
            *states,
            *[spare] * (MAX_STREAMS - S),
            plan.rows,
            plan.meta,
        ],
        template=[("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, Hv * S),
        threadgroup=(32, 4, 1),
        output_shapes=[states[0].shape] * S + [(16,)] * (MAX_STREAMS - S),
        output_dtypes=[mx.float32] * MAX_STREAMS,
    )
    _COUNTERS["replay_calls"] += 1
    _COUNTERS["replay_streams"] += S
    return list(outs[:S])


def conv_tails(
    states: Sequence[mx.array], qkv: mx.array, plan: CommitPlan
) -> list[mx.array]:
    """Each stream's next conv state [1, n_keep, C]: the last n_keep rows of [its conv state; its kept rows]."""

    C = int(qkv.shape[-1])
    S = len(states)
    spare = states[0]
    outs = _commit_kernel("tails")(
        inputs=[*states, *[spare] * (MAX_STREAMS - S), qkv, plan.tails],
        template=[("C", C), ("NKEEP", plan.n_keep)],
        grid=(C, plan.n_keep, S),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, plan.n_keep, C)] * S + [(16,)] * (MAX_STREAMS - S),
        output_dtypes=[qkv.dtype] * MAX_STREAMS,
    )
    _COUNTERS["tail_calls"] += 1
    _COUNTERS["tail_streams"] += S
    return list(outs[:S])


def sources(order: str = "served") -> dict[str, str]:
    if order not in GDN_ORDERS:
        raise ValueError(f"GDN step order must be one of {GDN_ORDERS}")
    if order == "fma":
        return {"pre": _PRE, "tree": _fma_order(_TREE), "replay": _fma_order(_REPLAY), "tails": _TAILS}
    return {"pre": _PRE, "tree": _TREE, "replay": _REPLAY, "tails": _TAILS}


__all__ = [
    "GDN_ORDERS",
    "MAX_STREAMS",
    "MAX_TREE",
    "CommitPlan",
    "ConvPlan",
    "TreePlan",
    "conv_tails",
    "gdn_pre",
    "gdn_pre_split",
    "replay",
    "source",
    "sources",
    "stats",
    "tree",
]
