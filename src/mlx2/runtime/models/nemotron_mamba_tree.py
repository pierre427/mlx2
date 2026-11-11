# SPDX-License-Identifier: MIT
"""Parent-indexed Nemotron-H Mamba-2 tree kernels (conv4 + gated SSM).

A speculative tree tile of up to 16 rows runs every Mamba layer in two
dispatches instead of one one-token chain per node:

* ``fused_mamba_conv_tree`` is ``nemotron_mamba_conv._SHORT_SOURCE`` with
  indexed loads.  Node ``n`` convolves the window ``[a3, a2, a1, self]`` of
  its own root-to-node path, where ancestors above the tile come from the
  three committed conv-state rows.  The per-node arithmetic is the one-row
  step of the short kernel, spelled identically.
* ``fused_mamba_ssm_tree`` is ``nemotron_mamba_ssm._SOURCE`` (the S=1 decode
  kernel on the exact verify routes) walking nodes in parent-first order.
  Each thread keeps one register slot per node (Ds/32 = 4 floats); a node
  starts from its parent's slot, or from ``state_in`` for the root, and runs
  the identical softplus / ``fast::exp`` / state step, readout and optional
  served-SiLU gate.  Only gated ``y`` leaves the kernel: per-node recurrent
  states (4 MiB per node per layer at the Super geometry) are never written.
  The accepted path's committed state is replayed afterwards with the
  existing ``fused_mamba_ssm_short``, the arithmetic of chain rollback.

The Metal compiler, not the source text, decides FMA contraction, so a
separately compiled kernel could round differently from the decode kernel.
``tree_kernel_parity`` therefore probes lazily, once per process for each
(dtype, time-step limit, geometry, tile rows S) it is asked about: the tree
kernels against serial S=1 fused calls along every path of a branched S-row
probe tree at that geometry, and the S-row commit replay against S one-token
calls, bit for bit.  A tile of S rows needs every size 1..S.  Any mismatch
(or a compile failure) refuses, and the caller keeps its serial per-node
backend.
Provenance: provenance/nemotron-h-mamba-tree.json.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from . import nemotron_mamba_conv as _conv
from . import nemotron_mamba_ssm as _ssm

TREE_MAX_ROWS = 16
TREE_KERNEL_LAW = "nemotron-h-mamba-tree-parent-slot-v1"


@dataclass(frozen=True)
class TreeKernelAdmission:
    accepted: bool
    reason: str


_CONV_TREE_SOURCE = r"""
  const uint channel = thread_position_in_grid.x;
  if (channel >= uint(C)) return;
  const device T* wc = weight + size_t(channel) * 4u;
  for (int node = 0; node < S; ++node) {
    // windows[node] names the three predecessors on this node's own path:
    // 0..2 are committed conv-state rows, 3.. are tile rows.
    const int w0 = windows[node * 3 + 0];
    const int w1 = windows[node * 3 + 1];
    const int w2 = windows[node * 3 + 2];
    T s0 = w0 < 3 ? state[uint(w0) * uint(C) + channel]
                  : input[uint(w0 - 3) * uint(C) + channel];
    T s1 = w1 < 3 ? state[uint(w1) * uint(C) + channel]
                  : input[uint(w1 - 3) * uint(C) + channel];
    T s2 = w2 < 3 ? state[uint(w2) * uint(C) + channel]
                  : input[uint(w2 - 3) * uint(C) + channel];
    T in = input[uint(node) * uint(C) + channel];
    float acc = float(s0) * float(wc[0]);
    acc += float(s1) * float(wc[1]);
    acc += float(s2) * float(wc[2]);
    acc += float(in) * float(wc[3]);
    T conv = static_cast<T>(acc);
    T xb = conv + bias[channel];
    output[uint(node) * uint(C) + channel] =
        xb * mlx_sigmoid_fast(xb);
  }
"""


_SSM_TREE_SOURCE = r"""
  auto n = thread_position_in_grid.z;
  auto h_idx = n % H;
  auto b_idx = n / H;
  auto grp = h_idx / G;
  constexpr int n_per_t = Ds / 32;
  constexpr int HB = H / G;

  auto i_state = state_in + n * Dh * Ds;
  auto ds_idx = thread_position_in_threadgroup.x;
  auto d_idx = thread_position_in_grid.y;
  auto A = -fast::exp(static_cast<float>(A_log[h_idx]));

  float init[n_per_t];
  for (int i = 0; i < n_per_t; ++i)
    init[i] = i_state[d_idx * Ds + n_per_t * ds_idx + i];
  // One register slot per node; a node reads only its parent's slot.
  float slots[S][n_per_t];

  for (int s = 0; s < S; ++s) {
    const int parent = parents[s];
    float st[n_per_t];
    for (int i = 0; i < n_per_t; ++i)
      st[i] = parent < 0 ? init[i] : slots[parent][i];
    auto x_ = static_cast<float>(
        X[((b_idx * S + s) * H + h_idx) * Dh + d_idx]);
    float raw_dt = static_cast<float>(dt[(b_idx * S + s) * H + h_idx]);
    float dt_ = mlx_softplus_fast<float>(raw_dt + float(dt_bias[h_idx]));
    if constexpr (CLAMP_DT)
      dt_ = metal::clamp(dt_, 0.001f, 100.0f);
    auto dA = fast::exp(A * dt_);
    auto B_ = B + ((b_idx * S + s) * HB + grp) * Ds;
    auto C_ = C + ((b_idx * S + s) * HB + grp) * Ds;

    float acc = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * ds_idx + i;
      auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
      st[i] = dA * st[i] + dB_by_x;
      acc += st[i] * C_[s_idx];
    }
    for (int i = 0; i < n_per_t; ++i)
      slots[s][i] = st[i];
    acc = simd_sum(acc);
    if (thread_index_in_simdgroup == 0) {
      auto out_idx = ((b_idx * S + s) * H + h_idx) * Dh + d_idx;
      auto y = static_cast<T>(acc + x_ * D[h_idx]);
      if constexpr (APPLY_GATE) {
        auto g = gate[out_idx];
        auto silu = static_cast<T>(g * mlx_sigmoid_fast<T>(g));
        out[out_idx] = static_cast<T>(silu * y);
      } else {
        out[out_idx] = y;
      }
    }
  }
"""

_CONV_KERNEL = None
_SSM_KERNEL = None
_KERNEL_LOCK = threading.Lock()


def _conv_kernel():
    global _CONV_KERNEL
    with _KERNEL_LOCK:
        if _CONV_KERNEL is None:
            header, suffix = _conv._served_header()
            _CONV_KERNEL = mx.fast.metal_kernel(
                name="mlx2_nemotron_mamba_conv4_silu_tree" + suffix,
                input_names=["input", "state", "weight", "bias", "windows"],
                output_names=["output"],
                header=header,
                source=_CONV_TREE_SOURCE,
                ensure_row_contiguous=True,
            )
        return _CONV_KERNEL


def _ssm_kernel():
    global _SSM_KERNEL
    with _KERNEL_LOCK:
        if _SSM_KERNEL is None:
            header, suffix = _ssm._served_header()
            _SSM_KERNEL = mx.fast.metal_kernel(
                name="mlx2_nemotron_mamba_ssm_tree_dt" + suffix,
                input_names=[
                    "X", "A_log", "B", "C", "D", "dt", "dt_bias", "state_in",
                    "gate", "parents",
                ],
                output_names=["out"],
                header=header,
                source=_SSM_TREE_SOURCE,
                ensure_row_contiguous=True,
            )
        return _SSM_KERNEL


def validate_tree_parents(parents) -> tuple[int, ...]:
    parents = tuple(parents)
    if not 1 <= len(parents) <= TREE_MAX_ROWS:
        raise ValueError("Nemotron Mamba tree requires 1..16 rows")
    for row, parent in enumerate(parents):
        if type(parent) is not int or (
            (row == 0 and parent != -1) or (row and not 0 <= parent < row)
        ):
            raise ValueError("Nemotron Mamba tree requires one parent-first root")
    return parents


def conv_windows(parents, kernel_size: int = 4) -> list[list[int]]:
    """Per-node predecessor rows in ``[committed state rows; tile rows]``."""

    parents = validate_tree_parents(parents)
    count = kernel_size - 1
    paths = []
    windows = []
    for row, parent in enumerate(parents):
        path = [row] if parent < 0 else paths[parent] + [row]
        paths.append(path)
        history = [*range(count), *(count + index for index in path[:-1])]
        windows.append(history[-count:])
    return windows


def fused_mamba_conv_tree(conv_input, conv_state, weight, bias, windows):
    """Conv4 + bias + SiLU for every tree node from its own path window.

    ``windows`` is an ``[N, 3]`` int32 array from :func:`conv_windows`.
    Returns the ``[1, N, C]`` conv outputs; no state is written.
    """

    steps = int(conv_input.shape[1])
    channels = int(conv_input.shape[-1])
    (output,) = _conv_kernel()(
        inputs=[conv_input, conv_state, weight, bias, windows],
        template=[("T", conv_input.dtype), ("C", channels), ("S", steps)],
        grid=(channels, 1, 1),
        threadgroup=(min(256, channels), 1, 1),
        output_shapes=[conv_input.shape],
        output_dtypes=[conv_input.dtype],
    )
    return output


def fused_mamba_ssm_tree(
    x, A_log, B, C, D, dt, dt_bias, state, parents, time_step_limit, gate=None
):
    """Parent-indexed gated SSM readout for every node; no state written.

    ``parents`` is an ``[N]`` int32 array (``-1`` for the root).  With
    ``gate`` the served SiLU gate is applied in-kernel exactly as
    ``fused_mamba_ssm_gate_short`` does.
    """

    limits = tuple(float(value) for value in time_step_limit)
    if limits == (0.001, 100.0):
        clamp_dt = True
    elif limits == (0.0, float("inf")):
        clamp_dt = False
    else:
        raise ValueError(
            "fused Nemotron tree SSM requires served time_step_limit "
            "(0, inf) or (0.001, 100.0)"
        )
    batch, steps, heads, head_dim = x.shape
    groups, state_dim = B.shape[-2:]
    (out,) = _ssm_kernel()(
        inputs=[x, A_log, B, C, D, dt, dt_bias, state,
                x if gate is None else gate, parents],
        template=[
            ("T", x.dtype),
            ("Dh", head_dim),
            ("Ds", state_dim),
            ("H", heads),
            ("G", heads // groups),
            ("S", steps),
            ("CLAMP_DT", clamp_dt),
            ("APPLY_GATE", gate is not None),
        ],
        grid=(32, head_dim, heads * batch),
        threadgroup=(32, 8, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


def admit_tree_conv(
    *, conv_input, conv_state, weight, bias, training, ssm_geometry, time_step_limit,
    param_dtypes=None,
) -> TreeKernelAdmission:
    admission = _conv.admit_fused_mamba_conv_short(
        conv_input=conv_input, conv_state=conv_state, weight=weight, bias=bias,
        mask=None, training=training, kernel_size=int(weight.shape[1]),
    )
    if not admission.accepted:
        return TreeKernelAdmission(False, admission.reason)
    refusal = _conv.fused_mamba_conv_refusal(conv_input.dtype) or tree_kernel_parity(
        conv_input.dtype, time_step_limit,
        geometry=(*ssm_geometry, int(conv_input.shape[-1])),
        rows=int(conv_input.shape[1]), param_dtypes=param_dtypes,
    )
    if refusal is not None:
        return TreeKernelAdmission(False, refusal)
    return TreeKernelAdmission(True, "eligible")


def admit_tree_ssm(
    *, x, B, C, dt, A_log, D, dt_bias, state, gate, channels, time_step_limit
) -> TreeKernelAdmission:
    if gate is None:
        admission = _ssm.admit_fused_mamba_ssm_short(
            x=x, B=B, C=C, dt=dt, A_log=A_log, D=D, dt_bias=dt_bias, state=state,
            mask=None, lengths=None,
        )
    else:
        admission = _ssm.admit_fused_mamba_ssm_gate_short(
            x=x, B=B, C=C, dt=dt, A_log=A_log, D=D, dt_bias=dt_bias, state=state,
            gate=gate, mask=None, lengths=None,
        )
    if not admission.accepted:
        return TreeKernelAdmission(False, admission.reason)
    geometry = (int(x.shape[2]), int(x.shape[3]), int(B.shape[2]), int(B.shape[3]), int(channels))
    refusal = (
        _ssm.fused_mamba_ssm_gate_refusal(x.dtype) if gate is not None else None
    ) or tree_kernel_parity(
        x.dtype, time_step_limit, geometry=geometry, rows=int(x.shape[1]),
        param_dtypes=(A_log.dtype, dt_bias.dtype),
    )
    if refusal is not None:
        return TreeKernelAdmission(False, refusal)
    return TreeKernelAdmission(True, "eligible")


# -- once-per-process parity gate -------------------------------------------

# A branched 16-row probe: depth 7, chain-prefix rows, siblings at the root's
# child, a node whose window crosses the committed state, and a late branch.
# A tile of S rows is probed on this tree's first S rows (still parent-first).
PARITY_PARENTS = (-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 1, 10, 2, 12, 12, 7)
# (heads, head_dim, groups, state_dim, conv channels) of Nemotron 3.5 Super.
SUPER_GEOMETRY = (128, 64, 8, 128, 128 * 64 + 2 * 8 * 128)
_PARITY = {}
_PARITY_LOCK = threading.Lock()
# Probes compile and launch kernels: one at a time, never twice per key.
_PROBE_LOCK = threading.Lock()


def _bits_equal(left, right) -> bool:
    import numpy as np

    mx.eval(left, right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    view = mx.uint16 if left.dtype in (mx.bfloat16, mx.float16) else mx.uint32
    return bool(np.array_equal(np.asarray(left.view(view)), np.asarray(right.view(view))))


def _run_parity(dtype, limit, geometry, rows, param_dtypes=(mx.float32, mx.float32),
                seed: int = 20261110) -> str | None:
    """Tree conv / tree SSM (gated and plain) at ``rows`` vs serial S=1 decode
    along every path, and the S=``rows`` replay vs ``rows`` S=1 calls."""
    heads, head_dim, groups, state_dim, channels = geometry
    intermediate = heads * head_dim
    parents_list = PARITY_PARENTS[:rows]
    key = mx.random.key(seed + 131 * rows)
    keys = mx.random.split(key, 12)
    conv_input = (mx.random.normal((1, rows, channels), key=keys[0]) * 2).astype(dtype)
    conv_state = (mx.random.normal((1, 3, channels), key=keys[1]) * 2).astype(dtype)
    weight = (mx.random.normal((channels, 4, 1), key=keys[2]) * 0.5).astype(dtype)
    bias = (mx.random.normal((channels,), key=keys[3]) * 0.5).astype(dtype)
    x = mx.random.normal((1, rows, heads, head_dim), key=keys[4]).astype(dtype)
    B = mx.random.normal((1, rows, groups, state_dim), key=keys[5]).astype(dtype)
    C = mx.random.normal((1, rows, groups, state_dim), key=keys[6]).astype(dtype)
    dt = (mx.random.normal((1, rows, heads), key=keys[7]) - 1).astype(dtype)
    gate = (mx.random.normal((1, rows, intermediate), key=keys[8]) * 2).astype(dtype)
    # The served A_log / dt_bias dtypes: each input-dtype set compiles its own
    # kernel, so the probe must run the one decode and trees will run.
    a_log_dtype, dt_bias_dtype = param_dtypes
    A_log = mx.random.uniform(-1, 2.5, (heads,), key=keys[9]).astype(a_log_dtype)
    D = mx.random.normal((heads,), key=keys[10]).astype(dtype)
    dt_bias = mx.random.normal((heads,), key=keys[11]).astype(dt_bias_dtype)
    state = mx.random.normal((1, heads, head_dim, state_dim), key=key).astype(mx.float32)

    windows = conv_windows(parents_list)
    tree_conv = fused_mamba_conv_tree(
        conv_input, conv_state, weight, bias, mx.array(windows, dtype=mx.int32)
    )
    source = mx.concatenate([conv_state, conv_input], axis=1)
    serial_conv = []
    for row, window in enumerate(windows):
        out, _ = _conv.fused_mamba_conv_short(
            conv_input[:, row : row + 1],
            mx.take(source, mx.array(window, dtype=mx.int32), axis=1),
            weight, bias, capture_checkpoints=False,
        )
        serial_conv.append(out)
    if not _bits_equal(tree_conv, mx.concatenate(serial_conv, axis=1)):
        return f"tree conv kernel differs from serial S=1 short conv (S={rows})"

    def step(gated, start, sl):
        if gated:
            return _ssm.fused_mamba_ssm_gate_short(
                x[:, sl], A_log, B[:, sl], C[:, sl], D, dt[:, sl], dt_bias,
                start, gate[:, sl], limit,
            )
        return _ssm.fused_mamba_ssm_short(
            x[:, sl], A_log, B[:, sl], C[:, sl], D, dt[:, sl], dt_bias, start, limit,
        )

    parents = mx.array(parents_list, dtype=mx.int32)
    for gated in (True, False):
        tree = fused_mamba_ssm_tree(
            x, A_log, B, C, D, dt, dt_bias, state, parents, limit,
            gate=gate if gated else None,
        )
        serial = [None] * rows
        states = [None] * rows
        for row, parent in enumerate(parents_list):
            start = state if parent < 0 else states[parent]
            serial[row], states[row] = step(gated, start, slice(row, row + 1))
        if not _bits_equal(tree, mx.concatenate(serial, axis=1)):
            return (
                "tree SSM kernel differs from serial S=1 fused calls "
                f"(S={rows}, gate={gated}, limit={limit})"
            )
        # The commit replays an accepted path of this length with the decode
        # kernel at S = rows; it must equal that many S=1 decode calls.
        chained = state
        for row in range(rows):
            _, chained = step(gated, chained, slice(row, row + 1))
        _, replayed = step(gated, state, slice(0, rows))
        if not _bits_equal(replayed, chained):
            return (
                f"short SSM kernel at S={rows} differs from {rows} S=1 "
                f"calls (gate={gated}, limit={limit})"
            )
    return None


def _static_refusal(dtype, limit) -> str | None:
    if dtype != mx.bfloat16:
        return "tree kernels are established for bf16 activations only"
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "tree kernels require the Metal GPU"
    if limit not in ((0.0, float("inf")), (0.001, 100.0)):
        return f"time_step_limit {limit} has no fused kernel"
    return None


def _verdict(dtype, limit, geometry, rows, param_dtypes) -> str | None:
    name = (
        f"{dtype}:{limit[0]!r}:{limit[1]!r}:{'x'.join(map(str, geometry))}:S{rows}"
        f":A_log={param_dtypes[0]}:dt_bias={param_dtypes[1]}"
    )
    with _PARITY_LOCK:
        if name in _PARITY:
            return _PARITY[name]
    with _PROBE_LOCK:
        with _PARITY_LOCK:
            if name in _PARITY:
                return _PARITY[name]
        try:
            verdict = _run_parity(dtype, limit, geometry, rows, param_dtypes)
        except Exception as error:  # compile or launch failure refuses
            verdict = f"tree kernel parity probe failed: {type(error).__name__}: {error}"
        with _PARITY_LOCK:
            _PARITY[name] = verdict
    return verdict


def tree_kernel_parity(
    dtype=mx.bfloat16, time_step_limit=(0.0, float("inf")), *,
    geometry=SUPER_GEOMETRY, rows=TREE_MAX_ROWS, param_dtypes=None,
) -> str | None:
    """``None`` when a ``rows``-row tile is exact at ``geometry``; else why not.

    Every template value compiles its own kernel, and the compiler decides
    FMA contraction, so a verdict holds only for the exact (dtype, limit,
    geometry, S, A_log / dt_bias dtypes) it probed.  A tile of S rows needs every size 1..S: its
    accepted path, replayed at S = path length, is at most S rows long.
    Probes run lazily, once per key and process, one at a time.  Device and
    dtype refusals are not cached (another thread may run on the GPU).
    """
    limit = tuple(float(value) for value in time_step_limit)
    refusal = _static_refusal(dtype, limit)
    if refusal is not None:
        return refusal
    geometry = tuple(int(value) for value in geometry)
    if type(rows) is not int or not 1 <= rows <= TREE_MAX_ROWS:
        return f"tree rows {rows!r}"
    param_dtypes = tuple(param_dtypes or (mx.float32, mx.float32))
    if len(param_dtypes) != 2:
        return "tree parity needs (A_log, dt_bias) dtypes"
    for size in range(1, rows + 1):
        verdict = _verdict(dtype, limit, geometry, size, param_dtypes)
        if verdict is not None:
            return verdict
    return None


def parity_status() -> dict[str, Any]:
    with _PARITY_LOCK:
        return {
            "law": TREE_KERNEL_LAW,
            "probe_parents": list(PARITY_PARENTS),
            "verdicts": {key: value or "exact" for key, value in _PARITY.items()},
        }


def reset_parity_for_tests() -> None:
    with _PARITY_LOCK:
        _PARITY.clear()


__all__ = [
    "TREE_KERNEL_LAW",
    "TREE_MAX_ROWS",
    "admit_tree_conv",
    "admit_tree_ssm",
    "conv_windows",
    "fused_mamba_conv_tree",
    "fused_mamba_ssm_tree",
    "parity_status",
    "tree_kernel_parity",
]
