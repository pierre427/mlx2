# SPDX-License-Identifier: Apache-2.0
"""Row-exact self-MTP verify for Flash-Next (qwen4_exp); opt-in, default off.

With MTP on, the target model verifies ``R`` rows (the pending token plus the
drafts) in one forward.  On the stock verify path several components change
bits with ``R`` (a Metal inventory is in scripts/gpu_check_row_exact_
components.py), so near-tie tokens flip and greedy MTP-on output differs from
MTP-off output.  This route computes every verify row with the arithmetic of
the one-token decode step at the same position:

==============  ============================================================
stage           row-exact implementation
==============  ============================================================
projections     every trunk ``QuantizedLinear`` and the LM head: the
                row-exact qmv kernel (``row_exact_qmv``; one weight pass,
                one-row ``qmv_fast``/``qmv`` bits per row), or one stock
                one-row call per row where the kernel does not cover the
                layout
dense           every trunk dense ``nn.Linear`` (mixed-precision artifacts
projections     keep the HC ``block_inject_weight`` and the MoE
                ``shared_expert_gate`` bf16): one one-token call per row
                (MLX's dense matmul is ``gemv`` at one row and
                ``gemv_wide``/GEMM tiling at more, so the composed M=R call
                is not the one-token arithmetic)
norms           ``GroupRMSNorm`` sees the one-token width (the width-8 fast
                RMSNorm gate would move wider windows to the eager norm)
attention       batched row-exact projections, then per row the one-token
                QSA selection, cache append, RoPE and SDPA with the ordinary
                B=1 mask, and the one-row output projection
GDN             the fused verify kernel (already bit-identical to the
                one-token fused decode kernel, rollback included; Metal-
                checked at 2..17 rows); a window whose GDN falls back to the
                stock multi-row block is not row-exact
MoE             router gate and shared expert through the kernel, routing
                and combine batched (per-row reductions), routed experts one
                one-token call per row; with the MoE row window selected
                (policy ``moe_window_row_exact``, qwen4_moe_window) the
                whole block runs in the window launches instead, each row
                still bit-identical to its one-token call
==============  ============================================================

Everything else in the window (embedding, PLE gather, HC elementwise and
stream mean, fast RMSNorm, logsumexp) was checked row-invariant on Metal.

Fail closed: each window records the route every stage took.  A window is
counted row-exact only when no stage fell back to a width-dependent path;
``status()`` reports both counts, and a receipt may say ``row_exact`` only
when the not-exact count did not move.  A projection module the route does
not swap (a ``Linear``/``QuantizedLinear`` subclass installed by another
mechanism, or a tied embedding head) would run its stock multi-row call
unrecorded, so its presence fails every window (``static_refusals``).
The whole target batch is also refused above eight lanes: B=16 token
divergence is observed despite successful stage counters. Batched one-token
passthroughs are recorded as not exact, including depth-zero target rounds;
their projections and head run outside the row-exact arithmetic window.
"""
from __future__ import annotations

import threading
import weakref
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .. import row_exact_verify as REV
from . import row_exact_qmv as REQ

SCHEMA = "mlx2.qwen4-row-exact-verify.v1"
ENV = "MLX_QWEN4_ROW_EXACT_VERIFY"
# Safety boundary, not qualification: B=4/8 matched the B=1 oracle, whereas
# B=16 diverged despite every verify stage reporting success. See
# qualification/runs/batched-gdn-verify-20261001/README.md. Do not raise this
# until the complete target path (including one-token rounds) is evidenced.
MAX_EVIDENCED_LANES = 8


def _rows(x) -> int:
    return int(x.size // x.shape[-1]) if x.ndim else 1


class _RowExactQuantizedLinear:
    """Mixin: route multi-row calls inside a window through the kernel."""

    def __call__(self, x):
        window = REV.current()
        if window is None or _rows(x) <= 1:
            return super().__call__(x)
        stock = super().__call__
        (y, route) = REQ.quantized_linear(self, x, call=stock)
        window.note("projections", route)
        return y


class _RowExactLinear:
    """Mixin: a dense (unquantized) ``nn.Linear`` inside a window, one
    one-token-shaped call per row.

    MLX runs a bf16 ``x @ W.T`` as ``gemv`` at one row but as ``gemv_wide`` or
    a tiled GEMM at more, so the composed multi-row call does not carry the
    one-token step's bits (W3, Metal: the uncensored artifact's bf16 HC
    inject at 17 rows).  Each row here IS the one-token call."""

    def __call__(self, x):
        window = REV.current()
        rows = _rows(x)
        if window is None or rows <= 1:
            return super().__call__(x)
        call = super(_RowExactLinear, self).__call__
        flat = x.reshape(rows, 1, 1, x.shape[-1])
        out = mx.concatenate([call(flat[r]) for r in range(rows)], axis=0)
        window.note("projections", "dense_per_row", rows)
        return out.reshape(*x.shape[:-1], out.shape[-1])


class _RowExactMoE:
    """Mixin: the MoE block of a window.

    The router gate and shared expert are projections (row-exact kernel) and
    the routing (softmax, top-k) and combine are per-row reductions and
    elementwise ops, so the block runs batched and only the routed experts
    (``_RowExactSwitch``) run one row at a time.  The fused router kernel
    (policy ``moe_router_kernel``) admits only the one-token shape, so with it
    selected each row runs the whole one-token block.
    """

    def __call__(self, x):
        window = REV.current()
        rows = _rows(x)
        if window is None or rows <= 1:
            return super().__call__(x)
        if getattr(self, "moe_router_mode", "stock") != "fused":
            from . import qwen3_next as Q3N

            if Q3N._MOE_GATE_COMPILE and rows > Q3N._MOE_GATE_COMPILE_MAX_TOKENS:
                # The one-token step routes through the compiled router; a
                # window this wide runs the eager routing instead.
                window.fail("moe_router_compiled_one_token_eager_window")
            window.note("moe_router", "batched", rows)
            return super().__call__(x)
        flat = x.reshape(rows, 1, 1, x.shape[-1])
        out = mx.concatenate([super(_RowExactMoE, self).__call__(flat[r]) for r in range(rows)], axis=0)
        window.note("moe", "per_row_block", rows)
        return out.reshape(*x.shape[:-1], out.shape[-1])


class _RowExactSwitch:
    """Mixin: the routed experts of a window, one one-token call per row."""

    def __call__(self, x, indices, scores=None, variant="scalar"):
        window = REV.current()
        rows = _rows(indices)
        if window is None or rows <= 1:
            return super().__call__(x, indices, scores=scores, variant=variant)
        lead = indices.shape[:-1]
        xs = x.reshape(rows, 1, 1, x.shape[-1])
        ids = indices.reshape(rows, 1, 1, indices.shape[-1])
        ws = None if scores is None else scores.reshape(rows, 1, 1, scores.shape[-1])
        call = super(_RowExactSwitch, self).__call__
        out = mx.concatenate(
            [
                call(xs[r], ids[r], scores=None if ws is None else ws[r], variant=variant)
                for r in range(rows)
            ],
            axis=0,
        )
        window.note("moe_experts", "per_row", rows)
        return out.reshape(*lead, *out.shape[2:])


def _ordinary_b1_mask(cache):
    """The mask the ordinary B=1 route gives a one-token attention forward.

    Its ``BatchKVCache`` supplies an explicit all-valid mask
    (``create_causal_mask(1, offset, left_padding=[0])``); the MTP rows' B1
    caches give none, and MLX's SDPA does not produce the same bits for the
    two.  ``None`` when the physical offset is not a host integer.
    """
    from .base import create_causal_mask

    offset = getattr(cache, "offset", None)
    if isinstance(offset, mx.array):
        if offset.size != 1:
            return None
        offset = getattr(cache, "_idx", None)
    if not isinstance(offset, int):
        return None
    return create_causal_mask(1, offset=offset, left_padding=mx.array([0]))


def _grouped(window, linears, x):
    (outputs, route) = REQ.quantized_linears(linears, x)
    window.note("projections", route, 1 if route == "group_kernel" else len(linears))
    return outputs


class _RowExactAttention:
    """Mixin: split a window's attention into one-token forwards."""

    def _project_segmented_qsa(self, x):
        window = REV.current()
        if window is None or _rows(x) <= 1:
            return super()._project_segmented_qsa(x)
        from . import qwen4_exp as Q

        if Q._QSA_FUSED_PROJ:
            # The stock form multiplies one concatenated table at M=R, which
            # is not the one-row arithmetic: not row-exact.
            window.fail("attention_fused_projection_table")
            return super()._project_segmented_qsa(x)
        return _grouped(
            window,
            (self.q_proj, self.k_proj, self.v_proj, self.indexer.index_qk_proj),
            x,
        )

    def __call__(
        self,
        x,
        mask,
        cache,
        *,
        _projected=None,
        _return_pre_o=False,
        _selection=None,
        _fetched_kv=None,
    ):
        window = REV.current()
        length = int(x.shape[1])
        stock = super().__call__
        if window is None or length <= 1:
            return stock(
                x, mask, cache, _projected=_projected, _return_pre_o=_return_pre_o,
                _selection=_selection, _fetched_kv=_fetched_kv,
            )
        if getattr(cache, "segmented_attention", None) is not None and _projected is None:
            # The segmented consumer projects the slab (row-exact projections)
            # and calls back per lineage with ``_projected``.
            return stock(x, mask, cache)
        refusal = None
        if _selection is not None or _fetched_kv is not None:
            refusal = "attention_preselected"
        elif int(x.shape[0]) != 1:
            refusal = "attention_batch_rows"
        elif _return_pre_o:
            refusal = "attention_pre_o_requested"
        if refusal is not None:
            window.fail(refusal)
            return stock(
                x, mask, cache, _projected=_projected, _return_pre_o=_return_pre_o,
                _selection=_selection, _fetched_kv=_fetched_kv,
            )
        if _projected is None:
            _projected = self._project_segmented_qsa(x)
        from . import qwen4_attn_window as AW
        from .qwen4_qsa_indexed_merge import fused_gate_enabled

        # Unless the indexed kernel may fuse the output gate (policy
        # indexed_output_gate, default off), a one-token forward ends in
        # ``o_proj(out * sigmoid(gate))``: stop each row before the output
        # projection and run it once for the window with the row-exact kernel.
        pre_o = not fused_gate_enabled()
        windowed = pre_o and AW.enabled()
        if windowed:
            # Every row dense by construction: one norm/RoPE launch, one
            # append, one windowed SDPA (qwen4_attn_window.dense_window).
            out = AW.dense_window(self, x, cache, _projected, window)
            if out is not None:
                return self.o_proj(out)
        deferred = AW.DeferredSDPA() if windowed else None
        outputs, gates = [], []
        for j in range(length):
            row_mask = _ordinary_b1_mask(cache)
            if row_mask is None:
                window.fail("attention_offset_not_host")
            result = stock(
                x[:, j : j + 1],
                row_mask,
                cache,
                _projected=tuple(value[:, j : j + 1] for value in _projected),
                _return_pre_o=pre_o,
                _defer_sdpa=deferred,
            )
            if result is None:
                # Deferred: the row's one-row SDPA runs with the window below.
                outputs.append(None)
                gates.append(None)
            elif pre_o:
                outputs.append(result[0])
                gates.append(result[1])
            else:
                outputs.append(result)
        window.note("attention", "per_row", length)
        if not pre_o:
            window.note("attention_o_proj", "per_row", length)
            return mx.concatenate(outputs, axis=1)
        ran = iter(AW.run_deferred(deferred, window) if deferred is not None else ())
        gated = [
            next(ran) if out is None else out * mx.sigmoid(gate)
            for out, gate in zip(outputs, gates)
        ]
        return self.o_proj(mx.concatenate(gated, axis=1))


class _RowExactGatedDeltaNet:
    """Mixin: the four GDN input projections in one row-exact launch.

    Each output column runs the same one-row arithmetic whether the serial
    step uses the separate projections or the fused input table, so one
    grouped launch matches either.
    """

    def _input_projections(self, inputs):
        window = REV.current()
        if window is None or _rows(inputs) <= 1:
            return super()._input_projections(inputs)
        return _grouped(
            window,
            (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a),
            inputs,
        )


_SUBCLASSES: Dict[type, type] = {}


def _subclass(mixin, cls):
    key = (mixin, cls)
    sub = _SUBCLASSES.get(key)
    if sub is None:
        # The swapped-out class, for kernels that admit only an exact layer
        # class: ``_row_exact_base`` (qwen4_hc_decode) and
        # ``_mlx2_row_exact_base`` (the shared-fold admission, qwen3_next).
        sub = type(
            "RowExact" + cls.__name__,
            (mixin, cls),
            {"_row_exact_base": cls, "_mlx2_row_exact_base": cls},
        )
        _SUBCLASSES[key] = sub
    return sub


class _RequestStart:
    """A request's receipt snapshot.  Live snapshots are tracked weakly so a
    request that overlaps another is never credited with its windows (the
    window counters are process-wide)."""

    __slots__ = ("windows", "windows_not_exact", "enabled", "overlapped", "__weakref__")

    def __init__(self, windows: int, windows_not_exact: int, enabled: bool):
        self.windows = windows
        self.windows_not_exact = windows_not_exact
        self.enabled = enabled
        self.overlapped = False

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class RowExactVerify:
    """Installed route: class swaps on the trunk plus the verify hooks."""

    def __init__(self, model):
        from .qwen3_next import (
            FusedDownSwitchGLU,
            FusedGateUpSwitchGLU,
            Qwen3NextSparseMoeBlock,
        )
        from .qwen4_exp import Attention, GatedDeltaNet

        self.model = model
        self.enabled = False
        self._lock = threading.Lock()
        self._swapped = []
        self._pending: Optional[REV.Window] = None
        self._live = weakref.WeakSet()
        self.counts: Dict[str, Any] = {
            "windows": 0,
            "windows_row_exact": 0,
            "windows_not_exact": 0,
            "rows": 0,
            "one_row_passthrough": 0,
            "stages": {},
            "failures": {},
        }
        language = model.language_model
        self._gdn = [
            module for module in language.model.layers
            if isinstance(getattr(module, "linear_attn", None), GatedDeltaNet)
        ]
        for _, module in language.named_modules():
            cls = type(module)
            if cls is nn.QuantizedLinear:
                mixin = _RowExactQuantizedLinear
            elif cls is nn.Linear:
                mixin = _RowExactLinear
            elif cls is Attention:
                mixin = _RowExactAttention
            elif cls is Qwen3NextSparseMoeBlock:
                mixin = _RowExactMoE
            elif cls is GatedDeltaNet:
                mixin = _RowExactGatedDeltaNet
            elif cls in (FusedGateUpSwitchGLU, FusedDownSwitchGLU):
                mixin = _RowExactSwitch
            else:
                continue
            self._swapped.append((module, cls))
            module.__class__ = _subclass(mixin, cls)
        self.static_refusals = self._audit()
        object.__setattr__(model, "mtp_verify_backbone", self.verify_backbone)
        object.__setattr__(model, "mtp_verify_logits", self.verify_logits)

    def _audit(self) -> Dict[str, int]:
        """Projection modules a window would reach without a row-exact form.

        The window runs ``language_model`` (trunk and LM head).  Every
        ``Linear``/``QuantizedLinear`` there must be one this route swapped;
        a subclass installed by another mechanism (for example a prefill
        projection class) keeps its stock multi-row call, and a tied head
        multiplies the embedding table at M=R.  Each is a reason every window
        fails closed."""
        language = self.model.language_model
        swapped = {id(module) for module, _ in self._swapped}
        refusals: Dict[str, int] = {}
        for _, module in language.named_modules():
            if id(module) in swapped:
                continue
            if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                key = f"unswapped_projection:{type(module).__name__}"
                refusals[key] = refusals.get(key, 0) + 1
        if getattr(getattr(language, "args", None), "tie_word_embeddings", False):
            refusals["tied_embedding_head"] = 1
        return refusals

    # -- control ---------------------------------------------------------------
    def enable(self, flag: bool = True) -> None:
        self.enabled = bool(flag)
        if self.enabled:
            # Re-audit: a mechanism may have swapped classes since install.
            self.static_refusals = self._audit()

    def remove(self) -> None:
        for module, cls in self._swapped:
            module.__class__ = cls
        self._swapped = []
        for name in ("mtp_verify_backbone", "mtp_verify_logits"):
            if name in self.model.__dict__:
                object.__delattr__(self.model, name)
        self.enabled = False

    # -- verify hooks ----------------------------------------------------------
    def _gdn_engaged(self) -> int:
        # A batched window's GDN is engaged when the batched fused verify ran
        # (policy fused_gdn_batch_verify): every lane's rows are its B=1
        # fused verify bits, which are the one-token decode bits.
        return sum(
            int(layer.linear_attn.fused_gdn_verify_calls)
            + int(getattr(layer.linear_attn, "fused_gdn_batch_verify_calls", 0))
            for layer in self._gdn
        )

    def verify_backbone(self, tokens, cache):
        model = self.model
        rows = int(tokens.size)
        lanes = int(tokens.shape[0])
        width = int(tokens.shape[-1])
        if not self.enabled or width <= 1:
            if self.enabled:
                self.counts["one_row_passthrough"] += 1
            self._retire_pending()
            if self.enabled and lanes > 1:
                # One token per lane is still a multi-row target forward.
                # Preserve the stock arithmetic, but never silently credit it
                # with a previous exact verify window's claim.
                record = REV.Window(rows)
                self._admit_batch(record, lanes)
                record.note("target_backbone", "batched_one_token_passthrough")
                record.note("target_logits", "outside_window")
                record.fail("batched_one_token_target_not_row_exact")
                self._close(record)
            return model.mtp_backbone(tokens, cache=cache)
        from . import qwen4_exp as Q

        self._retire_pending()
        record = REV.Window(rows)
        self._admit_batch(record, lanes)
        for reason in self.static_refusals:
            record.fail(reason)
        before = self._gdn_engaged()
        with REV.window(record), Q._declared_width(1):
            out = model.language_model.model(tokens, cache, return_hyper=True)
        engaged = self._gdn_engaged() - before
        if engaged != len(self._gdn):
            record.fail("gdn_fused_verify_not_engaged")
        record.note("gdn", "fused_verify", engaged)
        self._pending = record
        return out

    def _admit_batch(self, record: REV.Window, lanes: int) -> None:
        if lanes > MAX_EVIDENCED_LANES:
            record.note("target_batch", "above_evidenced_lane_limit")
            record.fail("target_batch_lane_limit")

    def verify_logits(self, hidden):
        record, self._pending = self._pending, None
        if record is None:
            return self.model.logits(hidden)
        with REV.window(record):
            logits = self.model.logits(hidden)
        self._close(record)
        return logits

    def _retire_pending(self) -> None:
        """A window whose logits never ran is closed as not row-exact, so a
        second backbone call cannot silently replace (and drop) it."""
        record, self._pending = self._pending, None
        if record is not None:
            record.fail("verify_window_not_consumed")
            self._close(record)

    def _close(self, record: REV.Window) -> None:
        with self._lock:
            counts = self.counts
            counts["windows"] += 1
            counts["rows"] += record.rows
            counts["windows_row_exact" if record.exact else "windows_not_exact"] += 1
            for stage, routes in record.stages.items():
                target = counts["stages"].setdefault(stage, {})
                for route, n in routes.items():
                    target[route] = target.get(route, 0) + n
            for reason, n in record.failures.items():
                counts["failures"][reason] = counts["failures"].get(reason, 0) + n

    # -- receipts --------------------------------------------------------------
    def snapshot(self) -> _RequestStart:
        start = _RequestStart(
            self.counts["windows"], self.counts["windows_not_exact"], self.enabled
        )
        with self._lock:
            live = list(self._live)
            for other in live:
                other.overlapped = True
            start.overlapped = bool(live)
            self._live.add(start)
        return start

    def receipt(self, start: Optional[dict]) -> dict:
        """Fail closed: ``row_exact`` only when every window since ``start``
        was row-exact and at least one ran (process-global counters)."""
        reason = None
        windows = not_exact = 0
        if not start:
            reason = "no_request_snapshot"
        else:
            windows = self.counts["windows"] - int(start["windows"])
            not_exact = self.counts["windows_not_exact"] - int(start["windows_not_exact"])
            if isinstance(start, _RequestStart):
                with self._lock:
                    self._live.discard(start)
            if not (start.get("enabled") and self.enabled):
                reason = "route_not_enabled"
            elif start.get("overlapped"):
                reason = "concurrent_requests"
            elif not_exact:
                reason = "window_fell_back"
            elif windows <= 0:
                reason = "no_verify_window_observed"
        return {
            "schema": SCHEMA,
            "row_exact": reason is None,
            "reason": reason,
            "windows": max(0, windows),
            "windows_not_exact": max(0, not_exact),
        }

    def status(self) -> dict:
        # HTTP threads call this while the generation worker's _close()
        # inserts stage/route keys under the lock: copy under it too, or a
        # resize mid-iteration 500s /v1/status (sweep 2026-10-02 V3).
        with self._lock:
            counts = self.counts
            return {
                "schema": SCHEMA,
                "enabled": self.enabled,
                "max_evidenced_lanes": MAX_EVIDENCED_LANES,
                "installed_modules": len(self._swapped),
                "static_refusals": dict(self.static_refusals),
                "gdn_layers": len(self._gdn),
                "windows": counts["windows"],
                "windows_row_exact": counts["windows_row_exact"],
                "windows_not_exact": counts["windows_not_exact"],
                "rows": counts["rows"],
                "one_row_passthrough": counts["one_row_passthrough"],
                "stages": {k: dict(v) for k, v in counts["stages"].items()},
                "failures": dict(counts["failures"]),
            }


def install(model) -> RowExactVerify:
    """Install the route on a loaded qwen4_exp ``Model`` (disabled until
    ``enable``)."""
    existing = getattr(model, "_mlx2_row_exact_verify", None)
    if existing is not None:
        return existing
    handle = RowExactVerify(model)
    object.__setattr__(model, "_mlx2_row_exact_verify", handle)
    return handle
