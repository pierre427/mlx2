# SPDX-License-Identifier: Apache-2.0
"""Adapter-owned row-exact progressive target rows for the Qwen hybrids.

Original mlx2 implementation (generalized from the 2026-10-17 default-off
Qwen3.6 candidate in this repository; no external source is mined).  One
executor covers both Qwen hybrid layouts that interleave gated-delta (GDN)
recurrent layers with full attention:

* Qwen3.6 35B-A3B (sparse MoE MLPs, ``qwen36_35b``);
* Qwen3.8 27B (dense SwiGLU MLPs, ``qwen38_27b``).

It is selected only through the adapters' explicit external-draft policy key
``target_verify_row_exact: true`` (default off).  Installing it class-swaps the
trunk's leaves and exposes, on that model instance only,
``progressive_external_verify_protocol`` (``row-exact-b1s1-cache-state-v1``)
and ``progressive_external_target_rows``, and routes that instance's
non-prefill ``forward_with_taps`` (standard verify, plain rounds, reference
tree paths, trusted rounds) through the same executor, as Muse's selected
route does; body-only prefill stays ordinary.  Every verify row is computed with
the arithmetic of the model's ordinary one-token (B1/S1) decode at the same
position:

==============  ===========================================================
stage           route (noted on the open ``row_exact_verify`` window)
==============  ===========================================================
projections     every trunk ``QuantizedLinear``/``Linear`` and the LM head
                through the shared ``row_exact.ops`` leaves (row-exact qmv /
                broadcast matvec where the per-build canary holds, one
                one-row call per row otherwise); GDN q/k/v/z/b/a grouped
norms           decoder ``RMSNorm`` through ``ops.row_independent`` (one
                batched launch where its canary holds, per row otherwise)
attention       serial B1/S1 per valid row against the authoritative row
                cache, with the ordinary B=1 mask; or, when the target-owned
                ``attention_window_row_exact`` candidate is selected, the
                existing exact q/k-prep + vector-SDPA window independently
                per lane while K/V still append one token at a time;
                ragged lineages never write padding into a row cache
GDN core        Metal: the layer's own fused verify kernel only when this
                build's decode-parity canary (verify kernel vs the layer's
                serial ordinary one-token route: outputs, final state and
                every rollback point) passes for that kind, lane count and
                step count.  Otherwise (always on CPU): an exact serial path
                that runs the layer's ordinary one-token recurrent step per
                valid row on a private scratch state (the decode arithmetic
                itself) and stages one ``record_rollback`` per row whose
                replay returns the exact per-step states, so
                ``HybridVerifyRows`` commit/trim/abort are exact
MoE (3.6)       one one-token block call per row (the decode route, routed
                kernels included), or the block's own ``row_exact`` window
                kernels when that consumer is explicitly selected together
                with routed decode (engagement counted per call)
dense MLP       its projections above; SwiGLU is row-independent
taps            post-block residuals of the same rows (same row law)
==============  ===========================================================

Fail closed: a launch attests ``row_exact=True`` only when every stage of
that launch took an exact route.  Anything else (a stage that fell back to
its stock multi-row form, a class swap replaced after install, a lane
preparation installed later, an unknown cache contract, more than eight
physical lanes, more than sixteen token rows) is recorded on the window and
the launch is returned with ``row_exact=False``, which the generic engine
refuses for exact fidelity.

Evidence state: implemented and CPU-tested.  A focused Q8 Qwen3.6 MoE Metal
gate and bounded serving smoke pass, but the complete real-artifact route
comparison at widths 1/2/4/8 on both artifacts is still pending.  The route
therefore remains unqualified and its receipts say so.
"""

from __future__ import annotations

import threading
from numbers import Integral
from typing import Any

import mlx.core as mx
from mlx import nn

from .. import row_exact
from .. import row_exact_verify as REV
from ..row_exact import ops
from ..segmented_plain_kv import SegmentedBatchKVCache
from . import row_exact_qmv as REQ
from .base import create_causal_mask
from .cache import KVCache
from .row_exact_components import (
    RowExactGatedDeltaNetProjectionsMixin,
    RowExactLinearMixin,
    RowExactQuantizedLinearMixin,
    row_count,
    row_exact_subclass,
)

SCHEMA = "mlx2.qwen-hybrid-row-exact-target.v1"
ALGORITHM = "qwen-hybrid-b1s1-row-exact-v1"
# Safety boundary, not qualification (the Flash-Next row-exact route saw
# B=16 divergence despite clean stage counters; the engine's lane-exact
# cohort forward is bounded to 2..8 lanes and 16 rows as well).
MAX_PHYSICAL_LANES = 8
MAX_TOKEN_ROWS = 16
GPU_EVIDENCE = (
    "focused_q8_qwen36_moe_pass;_full_route_metal_bitwise_widths_1_2_4_8_pending"
)
SOURCE_PATHS = (
    "src/mlx2/runtime/models/qwen_hybrid_row_exact.py",
    "src/mlx2/runtime/models/qwen4_attn_rows.py",
    "src/mlx2/runtime/models/qwen4_attn_window.py",
    "src/mlx2/runtime/models/row_exact_components.py",
    "src/mlx2/runtime/row_exact/ops.py",
    "src/mlx2/runtime/row_exact/canary.py",
    "src/mlx2/runtime/models/row_exact_qmv.py",
    "src/mlx2/runtime/hybrid_verify_rows.py",
)
_ROUTED_DECODE_MODES = ("gate_up_down", "gate_up_down_shared")


def _ordinary_b1_mask(cache):
    """The ordinary B=1 route's explicit all-valid one-token mask.

    The ordinary serving route's B=1 ``BatchKVCache`` supplies
    ``create_causal_mask(1, offset, left_padding=[0])`` for a one-token
    full-attention forward (see ``KVCache.make_mask``); a bare B1 ``KVCache``
    gives none, and MLX's SDPA need not produce the same bits for the two.
    ``None`` when the physical offset is not a host integer.
    """
    offset = getattr(cache, "offset", None)
    if not isinstance(offset, Integral):
        return None
    return create_causal_mask(1, offset=int(offset), left_padding=mx.array([0]))


def _fused_runtime_supported() -> bool:
    """Whether a fused GDN verify kernel can run here at all (Metal only)."""
    from .qwen4_fused_gdn import fused_gdn_runtime_supported

    return fused_gdn_runtime_supported()


def _strip(x):
    rows = row_count(x)
    return x.reshape(1, rows, x.shape[-1])


# -- norms ----------------------------------------------------------------------
class RowExactRMSNormMixin:
    """Decoder RMSNorm over a window's rows through the row-independent canary."""

    def __call__(self, x):
        if REV.current() is None or row_count(x) <= 1:
            return super().__call__(x)
        call = super().__call__
        y = ops.row_independent(
            call,
            _strip(x),
            stage="norms",
            key=("rms_norm", float(self.eps), tuple(self.weight.shape),
                 str(self.weight.dtype)),
        )
        return y.reshape(x.shape)


# -- attention ------------------------------------------------------------------
class RowExactQwen3NextAttentionMixin:
    """Qwen3Next attention using one B1/S1 state transition per valid row.

    Q/K/V and O projections keep a packed call surface (their installed
    projection mixins perform one-row arithmetic).  Attention, RoPE, cache
    append, q/k normalization, gating and SDPA then run once per token
    against the authoritative B1 row cache with the ordinary B=1 mask.  Ragged
    padding never reaches a row cache.  Inside a window this holds for a
    single row as well, so a one-row launch takes the same mask law.
    """

    def _row_exact_attention_window(self, qg, k_flat, v_flat, row_cache, valid):
        """One lane's consecutive B1/S1 rows in the exact vector-SDPA window.

        Admission is complete before the first cache write.  K/V are then
        appended one token at a time: ``KVCache`` grows its backing allocation
        differently for a multi-token append that crosses a 256-row boundary,
        so a bulk append would not preserve the ordinary B1 cache state.
        After mutation, a kernel error propagates to the owning transaction;
        it must never fall back after appending the rows.
        """
        route = getattr(self, "_mlx2_qwen_hybrid_route", None)
        if (
            route is None
            or not route.attention_window_row_exact
            or valid < 2
            or valid > MAX_TOKEN_ROWS
        ):
            return None
        from . import qwen4_attn_rows as AR
        from . import qwen4_attn_window as AW
        from .precise_ops import gate_sigmoid

        reason = None
        if type(row_cache) is not KVCache:
            reason = "cache_type"
        elif not isinstance(row_cache.offset, Integral) or row_cache.offset < 0:
            reason = "offset"
        elif (
            type(self.q_norm) is not nn.RMSNorm
            or type(self.k_norm) is not nn.RMSNorm
        ):
            reason = "norm_type"
        elif self.q_norm.eps != self.k_norm.eps:
            reason = "norm_eps"
        elif (
            qg.ndim != 3
            or k_flat.ndim != 3
            or v_flat.ndim != 3
            or qg.shape[:2] != (1, valid)
            or k_flat.shape[:2] != (1, valid)
            or v_flat.shape
            != (1, valid, self.num_key_value_heads * self.head_dim)
            or v_flat.dtype != qg.dtype
        ):
            reason = "projected_geometry"
        elif row_cache.keys is not None and (
            row_cache.values is None
            or row_cache.keys.shape[:2] != (1, self.num_key_value_heads)
            or row_cache.values.shape[:2] != (1, self.num_key_value_heads)
            or row_cache.keys.shape[-1] != self.head_dim
            or row_cache.values.shape[-1] != self.head_dim
            or row_cache.keys.dtype != qg.dtype
            or row_cache.values.dtype != qg.dtype
            or row_cache.offset > row_cache.keys.shape[2]
            or row_cache.offset > row_cache.values.shape[2]
        ):
            reason = "cache_geometry"
        else:
            reason = AR.prep_qk_supported(
                qg,
                k_flat,
                self.q_norm.weight,
                self.k_norm.weight,
                heads=self.num_attention_heads,
                kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                rope=self.rope,
            )
        if reason is not None:
            return None, reason

        base = int(row_cache.offset)
        queries, keys = AR.prep_qk(
            qg,
            k_flat,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.eps,
            base,
            heads=self.num_attention_heads,
            kv_heads=self.num_key_value_heads,
            rope=self.rope,
        )
        values = v_flat.reshape(
            1, valid, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        final = base + valid
        shape = (1, self.num_key_value_heads, final, self.head_dim)
        after = AW._ShapeOnly(shape, queries.dtype)
        reason = AW.window_supported(queries, after, after, base + 1)
        if reason is not None:
            return None, reason

        fetched = None
        for position in range(valid):
            fetched = row_cache.update_and_fetch(
                keys[..., position : position + 1, :],
                values[..., position : position + 1, :],
            )
        if fetched is None:  # pragma: no cover - valid >= 2 above
            raise RuntimeError("row-exact attention window appended no cache rows")
        (_, gate) = mx.split(
            qg.reshape(1, valid, self.num_attention_heads, -1), 2, axis=-1
        )
        output = AW.window_sdpa_gate(
            queries,
            fetched[0],
            fetched[1],
            self.scale,
            n_first=base + 1,
            gate=gate_sigmoid(gate),
        )
        return output, None

    def _row_exact_serial_attention(self, x, cache, lengths):
        window = REV.current()
        _, width, _ = x.shape
        q_all = self.q_proj(x)
        k_all = self.k_proj(x)
        v_all = self.v_proj(x)
        lane_outputs = []
        window_rows = 0
        serial_rows = 0
        for lane, (row_cache, valid) in enumerate(zip(cache, lengths)):
            candidate = self._row_exact_attention_window(
                q_all[lane : lane + 1, :valid],
                k_all[lane : lane + 1, :valid],
                v_all[lane : lane + 1, :valid],
                row_cache,
                valid,
            )
            if candidate is not None:
                output, reason = candidate
                if output is not None:
                    window_rows += valid
                    lane_outputs.append(
                        mx.pad(output, [(0, 0), (0, width - valid), (0, 0)])
                        if valid < width
                        else output
                    )
                    continue
                window.note("attention_window_fallback", reason)
            outputs = []
            for position in range(valid):
                row_mask = _ordinary_b1_mask(row_cache)
                if row_mask is None:
                    window.fail("attention_offset_not_host")
                    return None
                outputs.append(
                    self._attend(
                        q_all[lane : lane + 1, position : position + 1],
                        k_all[lane : lane + 1, position : position + 1],
                        v_all[lane : lane + 1, position : position + 1],
                        1,
                        1,
                        row_mask,
                        row_cache,
                    )
                )
            if outputs:
                output = mx.concatenate(outputs, axis=1)
            else:
                output = mx.zeros(
                    (1, 0, self.num_attention_heads * self.head_dim),
                    dtype=x.dtype,
                )
            if valid < width:
                output = mx.pad(output, [(0, 0), (0, width - valid), (0, 0)])
            lane_outputs.append(output)
            serial_rows += valid
        if window_rows:
            window.note("attention", "window_b1s1", window_rows)
        if serial_rows:
            window.note("attention", "serial_b1s1", serial_rows)
        layer_route = (
            "mixed_window_serial_b1s1"
            if window_rows and serial_rows
            else ("window_b1s1" if window_rows else "serial_b1s1")
        )
        window.note("attention_layers", layer_route)
        return self.o_proj(mx.concatenate(lane_outputs, axis=0))

    def __call__(self, x, mask=None, cache=None):
        window = REV.current()
        if window is None:
            return super().__call__(x, mask, cache)
        batch, width, _ = x.shape
        if isinstance(cache, SegmentedBatchKVCache):
            lengths = getattr(cache, "_step_lengths", None)
            if (
                lengths is None
                or len(lengths) != batch
                or width != cache.step_width
                or len(cache.rows) != batch
            ):
                window.fail("attention_segmented_step_unprepared")
                return super().__call__(x, mask, cache)
            if getattr(cache, "_updated", False):
                window.fail("attention_segmented_step_already_consumed")
                return super().__call__(x, mask, cache)
            result = self._row_exact_serial_attention(x, cache.rows, tuple(lengths))
            if result is None:
                return super().__call__(x, mask, cache)
            # The authoritative rows were updated directly.  Mirror the
            # wrapper's one-step lifecycle so a second consumer is refused
            # and transaction finalization can clear it normally.
            cache._updated = True
            cache._value_dim = self.head_dim
            cache.offset = mx.array([row.offset for row in cache.rows])
            cache.note_attention()
            return result
        if batch == 1 and _ordinary_b1_mask(cache) is not None:
            result = self._row_exact_serial_attention(x, (cache,), (width,))
            if result is not None:
                return result
        window.fail("attention_b1s1_cache_contract_unavailable")
        return super().__call__(x, mask, cache)


# -- GDN ------------------------------------------------------------------------
def _scratch(conv, state):
    """A private, non-speculating B1 recurrent cache holding ``conv``/``state``."""
    from .cache import ArraysCache

    scratch = ArraysCache(size=2)
    scratch[0] = conv
    scratch[1] = state
    return scratch


def ordinary_one_token_step(
    layer, qkv, z, b, a, scratch, dtype, *, probe=False
):
    """The layer's ordinary B1/S1 decode step after its input projections.

    Exactly what ``GatedDeltaNet.__call__`` runs for a one-token B1 forward
    on a non-speculating cache: the layer's own fused decode shortcut when it
    admits the step, else the reference recurrent core.  Returns
    ``(route, value)``: ``"fused_decode"`` with the projected ``[1, 1, H]``
    output, or ``"stock_decode"`` with the pre-``out_proj`` ``[1, 1, V]``.
    """
    fused_decode = (
        getattr(layer, "row_exact_probe_fused_decode", layer._try_fused_decode)
        if probe
        else layer._try_fused_decode
    )
    fused = fused_decode(qkv, z, b, a, None, scratch)
    if fused is not None:
        return "fused_decode", fused
    core = layer._recurrent_core(qkv, z, b, a, None, scratch, dtype=dtype)
    return "stock_decode", core.reshape(1, 1, -1)


def _gdn_geometry(layer):
    return (
        int(layer.num_k_heads),
        int(layer.num_v_heads),
        int(layer.head_k_dim),
        int(layer.head_v_dim),
        int(layer.conv_kernel_size),
    )


def _flat(*values):
    return mx.concatenate([v.astype(mx.float32).reshape(-1) for v in values])


def gdn_verify_parity_probe(layer, kind, lanes, dtype, state_dtype):
    """``probe(steps) -> (fused verify, serial ordinary decode)`` packed.

    Both sides start from the same random conv/recurrent state and inputs
    (ragged step counts across lanes for a batched kind).  The verify side
    is the layer's own speculating multi-token path (``_try_fused_decode``
    on a speculating scratch cache, i.e. exactly what a verify window runs);
    the reference is the layer's ordinary one-token step per row.  Packed:
    every valid output row (both through one-row ``out_proj`` arithmetic),
    every lane's final conv/recurrent state, and every rollback point the
    verify side staged.  A refused or failing verify side returns a
    mismatching pair so the verdict fails closed.
    """
    from .cache import ArraysCache

    refused = (mx.zeros((1,), dtype=mx.float32), mx.ones((1,), dtype=mx.float32))
    conv_dim = int(layer.conv_dim)
    value_dim = int(layer.value_dim)
    heads = int(layer.num_v_heads)
    keep = int(layer.conv_kernel_size) - 1

    def probe(steps):
        key = mx.random.key(5113 + 97 * steps + lanes)
        keys = mx.random.split(key, 6)
        qkv = (mx.random.normal((lanes, steps, conv_dim), key=keys[0]) * 0.5).astype(dtype)
        z = (mx.random.normal((lanes, steps, value_dim), key=keys[1]) * 0.5).astype(dtype)
        b = mx.random.normal((lanes, steps, heads), key=keys[2]).astype(dtype)
        a = mx.random.normal((lanes, steps, heads), key=keys[3]).astype(dtype)
        conv0 = (mx.random.normal((lanes, keep, conv_dim), key=keys[4]) * 0.5).astype(dtype)
        state0 = (
            mx.random.normal(
                (lanes, heads, int(layer.head_v_dim), int(layer.head_k_dim)),
                key=keys[5],
            )
            * 0.05
        ).astype(state_dtype)
        counts = [steps] + [max(1, steps - 1 - (lane % 2)) for lane in range(1, lanes)]
        verify = ArraysCache(size=2)
        verify[0], verify[1] = conv0, state0
        mask = None
        if lanes > 1:
            verify.prepare(lengths=counts)
            mask = verify.make_mask(steps)
        verify.start_speculation()
        private = REV.Window(lanes * steps)
        try:
            with REV.window(private):
                fused_decode = getattr(
                    layer, "row_exact_probe_fused_decode", layer._try_fused_decode
                )
                out = fused_decode(qkv, z, b, a, mask, verify)
        except Exception:  # noqa: BLE001 - a canary never raises mid-forward
            return refused
        if out is None or not verify._rollbacks:
            return refused
        record = verify._rollbacks[-1]
        multi, reference = [], []
        for lane in range(lanes):
            scratch = _scratch(conv0[lane : lane + 1], state0[lane : lane + 1])
            states = []
            for position in range(counts[lane]):
                cut = slice(position, position + 1)
                route, value = ordinary_one_token_step(
                    layer,
                    qkv[lane : lane + 1, cut],
                    z[lane : lane + 1, cut],
                    b[lane : lane + 1, cut],
                    a[lane : lane + 1, cut],
                    scratch,
                    dtype,
                    probe=True,
                )
                if route == "stock_decode":
                    value = layer.out_proj(value)
                reference.append(value)
                multi.append(out[lane : lane + 1, cut])
                states.append((scratch[0], scratch[1]))
            multi += [verify[0][lane : lane + 1], verify[1][lane : lane + 1]]
            reference += list(states[-1])
            for m in range(1, counts[lane]):
                conv_m, state_m = record.fn(m)
                multi += [conv_m[lane : lane + 1], state_m[lane : lane + 1]]
                reference += list(states[m - 1])
        if private.failures:
            return refused
        return _flat(*multi), _flat(*reference)

    return probe


class RowExactQwenGatedDeltaNetMixin(RowExactGatedDeltaNetProjectionsMixin):
    """GDN layer: grouped row-exact projections, then an exact core route.

    Requires the layer class to declare ``row_exact_fused_verify(lanes,
    steps)``, ``row_exact_fused_verify_calls()`` and
    ``row_exact_decode_route()`` (both Qwen hybrid GDN classes do); the
    executor owns everything else.
    """

    def __call__(self, inputs, mask=None, cache=None):
        window = REV.current()
        if window is None:
            return super().__call__(inputs, mask, cache)
        route = getattr(self, "_mlx2_qwen_hybrid_route", None)
        if route is None:
            window.fail("gdn_route_not_installed")
            return super().__call__(inputs, mask, cache)
        result = route.gdn_forward(self, inputs, mask, cache, window)
        if result is None:
            return super().__call__(inputs, mask, cache)
        return result


# -- MoE ------------------------------------------------------------------------
class RowExactQwenSparseMoEMixin:
    """Qwen3.6 sparse block: each row runs the one-token block call.

    The one-token call is the decode route (routed decode kernels and the
    shared fold included), so a per-row loop is exact by construction.  When
    the block's own ``row_exact`` window consumer is explicitly selected with
    routed decode, its window kernels (each row the one-token block
    arithmetic) run instead, and the launch is credited only when the
    window actually engaged.
    """

    def __call__(self, x):
        window = REV.current()
        rows = row_count(x)
        if window is None or rows <= 1:
            return super().__call__(x)
        if self.row_exact_window_selected():
            refusal = self.row_exact_window_refusal(x)
            if refusal is not None:
                y = ops.per_row(super().__call__, _strip(x))
                window.note("moe", "per_row_one_token_block", rows)
                window.note("moe_layers", "per_row_one_token_block")
                window.note("moe_window_fallback", refusal)
                return y.reshape(x.shape)
            before = int(self.moe_window_calls.get("row_exact", 0))
            y = super().__call__(x)
            if int(self.moe_window_calls.get("row_exact", 0)) != before + 1:
                reason = getattr(self, "qwen36_decode_last_fallback", None)
                window.fail(
                    "moe_row_exact_window_not_engaged"
                    + (f":{reason}" if reason else "")
                )
            else:
                window.note("moe", "row_exact_window", rows)
            window.note("moe_layers", "row_exact_window")
            return y
        y = ops.per_row(super().__call__, _strip(x))
        window.note("moe", "per_row_one_token_block", rows)
        window.note("moe_layers", "per_row_one_token_block")
        return y.reshape(x.shape)


# -- executor -------------------------------------------------------------------
class QwenHybridRowExactRoute:
    """Installed route: class swaps on the trunk plus the target-row hook."""

    def __init__(self, model):
        from .qwen3_5 import GatedDeltaNet as ReferenceGatedDeltaNet
        from .qwen38_27b import Qwen3NextAttention

        self.model = model
        self.attention_window_row_exact = False
        self._lock = threading.Lock()
        self._swapped: list[tuple[Any, type, type]] = []
        self._gdn: list[Any] = []
        self._attention: list[Any] = []
        self._moe: list[Any] = []
        self.counts: dict[str, Any] = {
            "launches": 0,
            "row_exact": 0,
            "not_exact": 0,
            "rows": 0,
            "stages": {},
            "failures": {},
            "physical_width_hist": {},
        }
        self.static_refusals: dict[str, int] = {}
        trunk = getattr(getattr(model, "language_model", None), "model", None)
        final_norm = getattr(trunk, "norm", None)
        for name, module in model.named_modules():
            cls = type(module)
            if cls is nn.QuantizedLinear:
                mixin = RowExactQuantizedLinearMixin
            elif cls is nn.Linear:
                mixin = RowExactLinearMixin
            elif cls is nn.RMSNorm and (
                module is final_norm
                or name.rsplit(".", 1)[-1]
                in ("input_layernorm", "post_attention_layernorm")
            ):
                # Token-row norms only.  q/k norms run inside the serial
                # attention at their one-token shape and stay untouched.
                mixin = RowExactRMSNormMixin
            elif cls is Qwen3NextAttention:
                mixin = RowExactQwen3NextAttentionMixin
                self._attention.append(module)
            elif isinstance(module, ReferenceGatedDeltaNet):
                if not all(
                    callable(getattr(cls, name, None))
                    for name in (
                        "row_exact_fused_verify",
                        "row_exact_fused_verify_calls",
                        "row_exact_decode_route",
                    )
                ):
                    self._refuse("undeclared_gdn_class:" + cls.__name__)
                    continue
                mixin = RowExactQwenGatedDeltaNetMixin
                self._gdn.append(module)
            elif callable(getattr(cls, "row_exact_window_selected", None)):
                mixin = RowExactQwenSparseMoEMixin
                self._moe.append(module)
            else:
                continue
            swapped = row_exact_subclass(mixin, cls)
            self._swapped.append((module, cls, swapped))
            module.__class__ = swapped
        for module in self._gdn:
            object.__setattr__(module, "_mlx2_qwen_hybrid_route", self)
        for module in self._attention:
            object.__setattr__(module, "_mlx2_qwen_hybrid_route", self)
        self._audit()
        self._had = {
            name: name in model.__dict__
            for name in (
                "progressive_external_target_rows",
                "progressive_external_verify_protocol",
                "progressive_external_exact_tile_widths",
                "progressive_external_exact_terminal_input_widths",
                "progressive_external_exact_max_lanes",
                "forward_with_taps",
            )
        }
        from ..progressive_external_verify import PROGRESSIVE_TARGET_PROTOCOL_VERSION

        object.__setattr__(model, "progressive_external_target_rows", self.execute)
        # Safety/admission capabilities of this executor, not qualification or
        # a performance claim.  ``execute`` refuses shapes outside these same
        # constants before it can attest a launch as row-exact.
        object.__setattr__(
            model,
            "progressive_external_exact_tile_widths",
            tuple(range(1, MAX_TOKEN_ROWS + 1)),
        )
        object.__setattr__(
            model,
            "progressive_external_exact_terminal_input_widths",
            tuple(range(2, MAX_TOKEN_ROWS + 1)),
        )
        object.__setattr__(
            model, "progressive_external_exact_max_lanes", MAX_PHYSICAL_LANES
        )
        object.__setattr__(model, "forward_with_taps", self.forward_with_taps)
        object.__setattr__(
            model,
            "progressive_external_verify_protocol",
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        )

    # -- install-time audit ------------------------------------------------------
    def _refuse(self, reason):
        self.static_refusals[reason] = self.static_refusals.get(reason, 0) + 1

    def set_attention_window_row_exact(self, selected: bool) -> None:
        if type(selected) is not bool:
            raise ValueError("attention_window_row_exact must be a boolean")
        self.attention_window_row_exact = selected

    def _audit(self):
        swapped = {id(module) for module, _, _ in self._swapped}
        for _, module in self.model.named_modules():
            if id(module) in swapped:
                continue
            if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                self._refuse("unswapped_projection:" + type(module).__name__)
            elif hasattr(module, "switch_mlp"):
                self._refuse("undeclared_moe_block:" + type(module).__name__)
        for layer in self._gdn:
            if hasattr(layer, "in_proj_fused"):
                # Ordinary decode would project through the fused table.
                self._refuse("gdn_fused_input_projection_installed")
            if layer.sharding_group is not None:
                self._refuse("gdn_sharded")
        language = getattr(self.model, "language_model", None)
        if getattr(getattr(language, "args", None), "tie_word_embeddings", False):
            self._refuse("tied_embedding_head")
        trunk = getattr(language, "model", None)
        if getattr(trunk, "pipeline_size", 1) != 1:
            self._refuse("pipeline_parallel")
        if not self._gdn or not self._attention:
            self._refuse("not_a_qwen_hybrid_trunk")

    def _launch_refusals(self, record):
        """Per-launch drift checks (cheap: identity and one dict read each)."""
        for module, _, swapped in self._swapped:
            if type(module) is not swapped:
                record.fail("class_swap_replaced_after_install")
                return
            if REQ.lane_preparation(module) is not None:
                # The one-row reference inside a window is the stock call,
                # ordinary decode would run the lane law.
                record.fail("lane_prepared_projection")
                return

    # -- GDN core ----------------------------------------------------------------
    def _gdn_fused_admitted(self, layer, window, kind, lanes, steps, dtype, state_dtype):
        from ..row_exact.canary import CANARIES

        key = (
            "qwen_hybrid_gdn_verify_vs_decode",
            type(layer).__dict__.get("_mlx2_row_exact_base", type(layer)).__qualname__,
            kind,
            int(lanes),
            str(dtype),
            str(state_dtype),
            tuple(layer.row_exact_decode_route()),
            _gdn_geometry(layer),
        )
        exact = CANARIES.exact(
            key,
            gdn_verify_parity_probe(layer, kind, lanes, dtype, state_dtype),
            rows=int(steps),
        )
        if not exact:
            window.note("gdn_canary", f"{kind}_differs_from_decode")
        return exact

    def gdn_forward(self, layer, inputs, mask, cache, window):
        if layer.sharding_group is not None:
            window.fail("gdn_sharded")
            return None
        batch, steps, _ = inputs.shape
        if cache is None or cache[1] is None:
            window.fail("gdn_uninitialized_cache")
            return None
        from ..segmented_batch_cache import SegmentedBatchArraysCache

        if isinstance(cache, SegmentedBatchArraysCache):
            rows = list(cache.rows)
            if len(rows) != batch:
                window.fail("gdn_segmented_rows_mismatch")
                return None
        elif batch == 1:
            rows = [cache]
        else:
            window.fail("gdn_cache_contract_unavailable")
            return None
        spans = cache.rollback_spans(steps, None)
        if spans is None:
            window.fail("gdn_rollback_geometry_undescribable")
            return None
        lengths = [steps] * batch if spans == () else [int(n) for n in spans]
        if len(lengths) != batch or any(not 1 <= n <= steps for n in lengths):
            window.fail("gdn_row_lengths_invalid")
            return None
        if any(row[1] is None for row in rows):
            window.fail("gdn_uninitialized_row")
            return None
        speculating = bool(getattr(cache, "speculating", False))
        qkv, z, b, a = self._project_gdn(layer, inputs)
        state_dtype = rows[0][1].dtype
        kind = None
        if speculating and steps > 1 and _fused_runtime_supported():
            kind = layer.row_exact_fused_verify(batch, steps)
        if kind is not None and self._gdn_fused_admitted(
            layer, window, kind, batch, steps, qkv.dtype, state_dtype
        ):
            before = int(layer.row_exact_fused_verify_calls())
            out = layer._try_fused_decode(qkv, z, b, a, mask, cache)
            if out is not None:
                if int(layer.row_exact_fused_verify_calls()) != before + 1:
                    window.fail("gdn_fused_route_unaccounted")
                else:
                    window.note("gdn", f"fused_{kind}_decode_parity", batch * steps)
                    window.note("gdn_layers", f"fused_{kind}")
                return out
            # Refused before touching the cache: the serial route below.
            window.note("gdn_fallback", f"fused_{kind}_refused")
        return self._serial_gdn(
            layer, qkv, z, b, a, cache, rows, lengths, speculating, inputs.dtype,
            window,
        )

    @staticmethod
    def _project_gdn(layer, inputs):
        return layer._input_projections(inputs)

    def _serial_gdn(
        self, layer, qkv, z, b, a, cache, rows, lengths, speculating, dtype, window
    ):
        batch, steps = qkv.shape[:2]
        projected = {}
        cores = {}
        finals = []
        staged = []
        routes = {}
        for lane, (row, valid) in enumerate(zip(rows, lengths)):
            conv0, state0 = row[0], row[1]
            scratch = _scratch(conv0, state0)
            states = []
            for position in range(valid):
                cut = slice(position, position + 1)
                route, value = ordinary_one_token_step(
                    layer,
                    qkv[lane : lane + 1, cut],
                    z[lane : lane + 1, cut],
                    b[lane : lane + 1, cut],
                    a[lane : lane + 1, cut],
                    scratch,
                    dtype,
                )
                routes[route] = routes.get(route, 0) + 1
                (projected if route == "fused_decode" else cores)[lane, position] = value
                states.append((scratch[0], scratch[1]))
            finals.append(states[-1])
            staged.append((row, [conv0, state0], states))
        # Stock-core rows share one row-exact out_proj; a fused decode row
        # already carries its own one-row out_proj.
        core_out = None
        if cores:
            padded = []
            zero = None
            for lane in range(batch):
                for position in range(steps):
                    value = cores.get((lane, position))
                    if value is None:
                        if zero is None:
                            zero = mx.zeros_like(next(iter(cores.values())))
                        value = zero
                    padded.append(value)
            core_out = layer.out_proj(
                mx.concatenate(padded, axis=1).reshape(batch, steps, -1)
            )
        lane_outputs = []
        hidden = int(layer.hidden_size)
        for lane, valid in enumerate(lengths):
            pieces = []
            for position in range(valid):
                value = projected.get((lane, position))
                if value is None:
                    value = core_out[lane : lane + 1, position : position + 1]
                pieces.append(value)
            output = mx.concatenate(pieces, axis=1)
            if valid < steps:
                output = mx.pad(output, [(0, 0), (0, steps - valid), (0, 0)])
            lane_outputs.append(output)
        result = mx.concatenate(lane_outputs, axis=0)
        if result.shape[-1] != hidden:
            window.fail("gdn_serial_output_geometry")
            return None
        # Stage the exact per-row rollback, then publish the final states.
        if speculating:
            for row, snapshot, states in staged:

                def replay(m, _states=states):
                    m = int(m)
                    if not 1 <= m <= len(_states):
                        raise RuntimeError(
                            f"serial GDN replay outside its staged rows: {m}"
                        )
                    return list(_states[m - 1])

                row.record_rollback(steps, replay, snapshot)
        if len(rows) == 1 and rows[0] is cache:
            cache[0], cache[1] = finals[0]
        else:
            cache[0] = mx.concatenate([conv for conv, _ in finals], axis=0)
            cache[1] = mx.concatenate([state for _, state in finals], axis=0)
        cache.advance(steps)
        for route, count in routes.items():
            window.note("gdn", f"serial_{route}", count)
        window.note("gdn_layers", "serial_one_token")
        return result

    # -- the hook ----------------------------------------------------------------
    def execute(self, tokens, caches, capture_layers):
        from ..progressive_external_verify import (
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            ProgressiveTargetRows,
        )
        from .moe_nax_gather import prefill_scope

        lanes = int(tokens.shape[0])
        width = int(tokens.shape[1])
        rows = lanes * width
        record = REV.Window(rows)
        for reason in self.static_refusals:
            record.fail(reason)
        stock = type(self.model).forward_with_taps
        if lanes > MAX_PHYSICAL_LANES or width > MAX_TOKEN_ROWS:
            # Outside the evidenced shape: the stock forward, never credited
            # (and never a serial per-token walk over a prefill-sized block).
            record.fail(
                "physical_width_above_evidenced_limit"
                if lanes > MAX_PHYSICAL_LANES
                else "token_width_above_limit"
            )
            record.note("target", "stock_outside_window")
            logits, features = stock(self.model, tokens, caches, capture_layers)
        else:
            self._launch_refusals(record)
            # A verify launch is never a prefill forward (NAX MoE gather etc.).
            with REV.window(record), prefill_scope(False):
                logits, features = stock(self.model, tokens, caches, capture_layers)
            layers = record.stages
            if sum(layers.get("gdn_layers", {}).values()) != len(self._gdn):
                record.fail("gdn_layer_not_engaged")
            if sum(layers.get("attention_layers", {}).values()) != len(
                self._attention
            ):
                record.fail("attention_layer_not_engaged")
            if rows > 1 and sum(layers.get("moe_layers", {}).values()) != len(
                self._moe
            ):
                record.fail("moe_layer_not_engaged")
        layers = record.stages
        gdn_routes = sorted(layers.get("gdn_layers", {}))
        moe_routes = sorted(layers.get("moe_layers", {}))
        with self._lock:
            counts = self.counts
            counts["launches"] += 1
            counts["rows"] += rows
            counts["row_exact" if record.exact else "not_exact"] += 1
            key = str(lanes)
            counts["physical_width_hist"][key] = (
                counts["physical_width_hist"].get(key, 0) + 1
            )
            for stage, routes in record.stages.items():
                target = counts["stages"].setdefault(stage, {})
                for route, n in routes.items():
                    target[route] = target.get(route, 0) + n
            for reason, n in record.failures.items():
                counts["failures"][reason] = counts["failures"].get(reason, 0) + n
            launch = counts["launches"]
        return ProgressiveTargetRows(
            logits=logits,
            features=features,
            protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            row_exact=record.exact,
            detail={
                "schema": SCHEMA,
                "algorithm": ALGORITHM,
                "launch": launch,
                "rows": rows,
                "physical_width": lanes,
                "token_width": width,
                "gdn_route": "+".join(gdn_routes) or None,
                "moe_route": "+".join(moe_routes) or None,
                "failure_count": sum(record.failures.values()),
                "reason": next(iter(sorted(record.failures)), None),
                "gpu_evidence": GPU_EVIDENCE,
            },
        )

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False,
                          **kwargs):
        """The selected model's ``forward_with_taps``: every target forward
        that produces logits (standard verify, plain rounds, reference-tree
        paths, trusted rounds) takes the same row law as progressive verify,
        as Muse's selected route does.  Body-only prefill stays ordinary."""
        stock = type(self.model).forward_with_taps
        last_logits_only = bool(kwargs.pop("last_logits_only", False))
        if kwargs:
            raise TypeError(f"unexpected forward_with_taps options {sorted(kwargs)}")
        if body_only or REV.current() is not None:
            options = {"body_only": True} if body_only else {}
            if last_logits_only:
                options["last_logits_only"] = True
            return stock(self.model, inputs, cache, capture_layers, **options)
        result = self.execute(inputs, cache, capture_layers)
        logits = result.logits[:, -1:] if last_logits_only else result.logits
        return logits, result.features

    # -- receipts ----------------------------------------------------------------
    def status(self):
        with self._lock:
            counts = {
                **self.counts,
                "stages": {k: dict(v) for k, v in self.counts["stages"].items()},
                "failures": dict(self.counts["failures"]),
                "physical_width_hist": dict(self.counts["physical_width_hist"]),
            }
        return {
            "schema": SCHEMA,
            "algorithm": ALGORITHM,
            "implemented": True,
            "qualified": False,
            "selected": bool(self._swapped),
            "attention_window_row_exact": self.attention_window_row_exact,
            "observed_used": bool(counts["launches"]),
            "gpu_evidence": GPU_EVIDENCE,
            "static_refusals": dict(self.static_refusals),
            "layers": {
                "gdn": len(self._gdn),
                "attention": len(self._attention),
                "moe": len(self._moe),
            },
            "counts": counts,
            **row_exact.status(),
        }

    def receipt(self):
        status = self.status()
        counts = status["counts"]
        return {
            "target_verify_row_exact": {
                "algorithm": ALGORITHM,
                "implemented": True,
                "qualified": False,
                "selected": True,
                "attention_window_row_exact": self.attention_window_row_exact,
                "performance_claim": False,
                "gpu_evidence": GPU_EVIDENCE,
                "observed_used": counts["launches"] > 0,
                "launches": counts["launches"],
                "row_exact_launches": counts["row_exact"],
                "nonexact_launches": counts["not_exact"],
                "physical_query_rows": counts["rows"],
                "physical_width_hist": counts["physical_width_hist"],
                "routes": {
                    stage: dict(sorted(routes.items()))
                    for stage, routes in sorted(counts["stages"].items())
                },
                "failures": counts["failures"],
                "static_refusals": status["static_refusals"],
            }
        }

    def remove(self):
        for module, cls, _ in reversed(self._swapped):
            module.__class__ = cls
        for module in self._gdn:
            module.__dict__.pop("_mlx2_qwen_hybrid_route", None)
        for module in self._attention:
            module.__dict__.pop("_mlx2_qwen_hybrid_route", None)
        self._swapped = []
        for name, had in self._had.items():
            if not had and name in self.model.__dict__:
                object.__delattr__(self.model, name)


def install(model) -> QwenHybridRowExactRoute:
    """Install the explicit, unqualified Qwen hybrid row-exact target route."""

    existing = model.__dict__.get("_mlx2_qwen_hybrid_row_exact")
    if existing is not None:
        return existing
    route = QwenHybridRowExactRoute(model)
    object.__setattr__(model, "_mlx2_qwen_hybrid_row_exact", route)
    return route


def installed(model):
    return getattr(model, "__dict__", {}).get("_mlx2_qwen_hybrid_row_exact")


def uninstall(model) -> None:
    route = installed(model)
    if route is not None:
        route.remove()
        object.__delattr__(model, "_mlx2_qwen_hybrid_row_exact")
