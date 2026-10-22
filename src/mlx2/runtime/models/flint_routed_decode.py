# SPDX-License-Identifier: Apache-2.0
"""One-token routed experts for Flint decode without whole-table binds.

Flint's stock one-token MoE runs three ``gather_qmm`` launches (gate, up,
down) over 256-expert tables, plus the compiled SwiGLU.  Each gather binds a
whole expert table (134 MB per projection for the Q4 g64 artifact), and MLX
commits its command buffer once the bound inputs pass the size cap, so every
one of those launches also ends a command buffer.  This module serves the
same products from one-expert views (the arrays alias the table, so the
kernels index every expert from the bound pointer):

``gate_up``
    ``swiglu(gate_j(x), up_j(x))`` for the six routed experts in one launch:
    ``qwen4_routed_decode.SPLIT_GATE_UP_SOURCE`` (mined from jundot/omlx, see
    provenance/omlx-fn-2026-09-25.json), MLX's ``qmv_fast`` lane partition and
    add order, the compiled SwiGLU epilogue.
``down_rows``
    ``down_j(h_j)`` for each slot, rounded to the activation dtype exactly as
    ``gather_qmm`` writes it (one simdgroup per four output rows, the same
    ``qmv_fast`` arithmetic).  Original mlx2 source.

Flint's own combine law stays in the model: the six bf16 expert rows are
multiplied by the fp32 renormalized sigmoid weights and summed in fp32.  The
router stays the model's fp32 sigmoid + argpartition.

Exactness is proven, not assumed: the first admitted call in a process runs
both kernels and the stock ``gather_qmm`` path on the live inputs and compares
them bitwise, once per QMV input-sum transcription (MLX 0.32.4 adds the
bf16 inputs in the activation type; the mlx2 fork added them in float).  The
first transcription that matches is used; if none matches, the route is
refused for the process and the stock path serves every call.
"""

from __future__ import annotations

import mlx.core as mx

from . import qwen4_routed_decode as RD

MODES = ("off", "gate_up", "gate_up_down")
BITS = 4
GROUP_SIZE = 64
DOWN_ROWS = 4            # output rows per simdgroup (MLX qmv_fast results_per_simdgroup)
DOWN_SIMDGROUPS = 2

# MLX 0.32.4 ``load_vector`` sums the activation values in T before adding to
# the fp32 accumulator; the mlx2 fork (2d6705923) casts each to float first.
_FORK_SUM = "sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);"
_STOCK_SUM = "sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];"
TRANSCRIPTIONS = ("stock_sum", "fork_sum")

DOWN_ROWS_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    const size_t row0 = size_t(rhs[tid.z]) * N + out_row;
    const int lane_w = int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const int lane_g = int(simd_lid) / SCALE_STEP_PER_THREAD;

    const device uint8_t* ws = (const device uint8_t*)w + row0 * in_vec_size_w + lane_w;
    const device T* sl = scales + row0 * in_vec_size_g + lane_g;
    const device T* bl = biases + row0 * in_vec_size_g + lane_g;
    const device T* xp = x + size_t(tid.z) * K + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[RPS] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < RPS; row++) {
        float s = sl[row * in_vec_size_g];
        float b = bl[row * in_vec_size_g];
        result[row] += qdot_n(ws + row * in_vec_size_w, x_thread, s, b, sum, VALUES_PER_THREAD);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sl += BLOCK_SIZE / GS;
      bl += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }

    for (int row = 0; row < RPS; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0) {
      device T* yp = y + size_t(tid.z) * N + out_row;
      for (int row = 0; row < RPS; row++) {
        yp[row] = static_cast<T>(result[row]);
      }
    }
"""

_KERNELS: dict = {}
# Process verdict: None = not probed; a transcription name; or "refused: ...".
_VERIFIED: dict = {"verdict": None, "probes": []}


def _header(transcription: str, *, swiglu: bool) -> str:
    header = (
        RD.QMV_HEADER.replace("__BITS__", str(BITS))
        .replace("__GS__", str(GROUP_SIZE))
        .replace("__FAST__", "1")
    )
    if _FORK_SUM not in header:
        raise RuntimeError("routed-decode QMV header changed; transcriptions need review")
    if transcription == "stock_sum":
        header = header.replace(_FORK_SUM, _STOCK_SUM)
    if swiglu:
        header += RD.SWIGLU_GATE.adapt(RD.SIGMOID, mx.bfloat16)
    return header


def _kernel(kind: str, transcription: str):
    key = (kind, transcription)
    if key not in _KERNELS:
        if kind == "gate_up":
            spelling = RD.SWIGLU_GATE.spelling(mx.bfloat16) or "unknown"
            _KERNELS[key] = mx.fast.metal_kernel(
                name=f"mlx2_flint_gate_up_swiglu_q4g64_{transcription}"
                + ("_precise" if "precise" in spelling else "_fast"),
                input_names=["x", "wg", "sg", "bg", "wu", "su", "bu", "rhs"],
                output_names=["y"],
                header=_header(transcription, swiglu=True),
                source=RD.SPLIT_GATE_UP_SOURCE,
            )
        else:
            _KERNELS[key] = mx.fast.metal_kernel(
                name=f"mlx2_flint_down_rows_q4g64_{transcription}",
                input_names=["x", "w", "scales", "biases", "rhs"],
                output_names=["y"],
                header=_header(transcription, swiglu=False),
                source=DOWN_ROWS_SOURCE,
            )
    return _KERNELS[key]


# Refusals that depend on the call, not on the block: a later call may admit.
TRANSIENT_REFUSALS = frozenset({"not one token", "activation must be bfloat16"})


def admission(x, switch) -> str | None:
    """Structural refusal for one-token routed decode, or None.  Never
    evaluates an array."""
    from .switch_layers import QuantizedSwitchLinear

    hidden = x.shape[-1]
    if x.size != hidden:
        return "not one token"
    if x.dtype != mx.bfloat16:
        return "activation must be bfloat16"
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = switch.get(name) if hasattr(switch, "get") else getattr(switch, name, None)
        if type(layer) is not QuantizedSwitchLinear:
            return f"{name}: not a resident QuantizedSwitchLinear"
        reason = RD._quantized_ok(layer, bits=BITS, group_size=GROUP_SIZE)
        if reason is not None:
            return f"{name}: {reason}"
    gate, down = switch.gate_proj, switch.down_proj
    experts, inter = int(gate["weight"].shape[0]), int(gate["weight"].shape[1])
    if hidden % 512 or inter % 512:
        return "hidden and intermediate must be multiples of 512 (MLX qmv_fast)"
    table = (experts, inter, hidden * BITS // 32)
    if tuple(gate["weight"].shape) != table or tuple(switch.up_proj["weight"].shape) != table:
        return "gate/up tables differ"
    if tuple(down["weight"].shape) != (experts, hidden, inter * BITS // 32):
        return "down table shape"
    if inter % (RD.GATE_UP_ROWS * RD.GATE_UP_SIMDGROUPS) or hidden % (DOWN_ROWS * DOWN_SIMDGROUPS):
        return "row tiling"
    if not RD.runtime_supported():
        return "Metal runtime unavailable"
    reason = RD.served_swiglu_refusal()
    if reason is not None:
        return reason
    verdict = _VERIFIED["verdict"]
    if isinstance(verdict, str) and verdict.startswith("refused"):
        return verdict
    return None


def gate_up(x, indices, switch, transcription):
    top_k = int(indices.size)
    hidden = x.shape[-1]
    inter = int(switch.gate_proj["weight"].shape[1])
    rows = RD.GATE_UP_ROWS * RD.GATE_UP_SIMDGROUPS
    return _kernel("gate_up", transcription)(
        inputs=[
            x.reshape(hidden),
            *RD.expert_operands(switch.gate_proj),
            *RD.expert_operands(switch.up_proj),
            indices.reshape(top_k).astype(mx.uint32),
        ],
        template=[("T", x.dtype), ("K", hidden), ("NI", inter), ("RPS", RD.GATE_UP_ROWS),
                  ("NSG", RD.GATE_UP_SIMDGROUPS), ("TOPK", top_k)],
        grid=(32, RD.GATE_UP_SIMDGROUPS * inter // rows, top_k),
        threadgroup=(32, RD.GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(top_k, inter)],
        output_dtypes=[x.dtype],
    )[0]


def down_rows(h, indices, switch, transcription):
    top_k, inter = int(h.shape[0]), int(h.shape[1])
    hidden = int(switch.down_proj["weight"].shape[1])
    rows = DOWN_ROWS * DOWN_SIMDGROUPS
    return _kernel("down", transcription)(
        inputs=[
            h,
            *RD.expert_operands(switch.down_proj),
            indices.reshape(top_k).astype(mx.uint32),
        ],
        template=[("T", h.dtype), ("K", inter), ("N", hidden), ("RPS", DOWN_ROWS),
                  ("NSG", DOWN_SIMDGROUPS)],
        grid=(32, DOWN_SIMDGROUPS * hidden // rows, top_k),
        threadgroup=(32, DOWN_SIMDGROUPS, 1),
        output_shapes=[(top_k, hidden)],
        output_dtypes=[h.dtype],
    )[0]


def _stock(x, indices, switch):
    """The stock SwitchGLU rows ``[top_k, hidden]`` and its activation."""
    xe = mx.expand_dims(x, (-2, -3))
    up = switch.up_proj(xe, indices)
    gate = switch.gate_proj(xe, indices)
    act = switch.activation(up, gate)
    return switch.down_proj(act, indices).reshape(int(indices.size), -1), act.reshape(int(indices.size), -1)


def _probe(x, indices, switch):
    """Bitwise canary on live inputs; sets the process verdict."""
    stock_rows, stock_act = _stock(x, indices, switch)
    mx.eval(stock_rows, stock_act)
    for transcription in TRANSCRIPTIONS:
        try:
            h = gate_up(x, indices, switch, transcription)
            y = down_rows(stock_act, indices, switch, transcription)
            mx.eval(h, y)
        except Exception as exc:  # noqa: BLE001 - an unbuildable kernel is a refusal
            _VERIFIED["probes"].append({"transcription": transcription, "error": f"{type(exc).__name__}: {exc}"})
            continue
        gate_equal = bool(mx.array_equal(h, stock_act).item())
        down_equal = bool(mx.array_equal(y, stock_rows).item())
        _VERIFIED["probes"].append({"transcription": transcription, "gate_up_bitwise": gate_equal,
                                    "down_bitwise": down_equal})
        if gate_equal and down_equal:
            _VERIFIED["verdict"] = transcription
            return transcription
    _VERIFIED["verdict"] = "refused: no transcription matched stock gather_qmm bitwise"
    return None


def routed_rows(x, indices, switch, mode):
    """``[top_k, hidden]`` expert rows for one token, or None (stock serves)."""
    verdict = _VERIFIED["verdict"]
    if verdict is None:
        verdict = _probe(x, indices, switch)
        if verdict is None:
            return None
    h = gate_up(x, indices, switch, verdict)
    if mode == "gate_up":
        act = h.reshape(1, 1, int(indices.size), 1, -1)
        return switch.down_proj(act, indices).reshape(int(indices.size), -1)
    return down_rows(h, indices, switch, verdict)


def status() -> dict:
    return {"verdict": _VERIFIED["verdict"], "probes": list(_VERIFIED["probes"])}


def reset_for_tests():
    _VERIFIED["verdict"] = None
    _VERIFIED["probes"].clear()
