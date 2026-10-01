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
norms           ``GroupRMSNorm`` sees the one-token width (the width-8 fast
                RMSNorm gate would move wider windows to the eager norm)
attention       batched row-exact projections, then per row the one-token
                QSA selection, cache append, RoPE and SDPA with the ordinary
                B=1 mask, and the one-row output projection
GDN             the fused verify kernel (already bit-identical to the
                one-token fused decode kernel, rollback included; Metal-
                checked at 2..17 rows); a window whose GDN falls back to the
                stock multi-row block is not row-exact
MoE             one call of the one-token MoE block per row (router, routed
                experts, shared expert); wave 2 can fuse this once the
                routed kernel covers verify windows
==============  ============================================================

Everything else in the window (embedding, PLE gather, HC elementwise and
stream mean, fast RMSNorm, logsumexp) was checked row-invariant on Metal.

Fail closed: each window records the route every stage took.  A window is
counted row-exact only when no stage fell back to a width-dependent path;
``status()`` reports both counts, and a receipt may say ``row_exact`` only
when the not-exact count did not move.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .. import row_exact_verify as REV
from . import row_exact_qmv as REQ

SCHEMA = "mlx2.qwen4-row-exact-verify.v1"
ENV = "MLX_QWEN4_ROW_EXACT_VERIFY"


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


class _RowExactMoE:
    """Mixin: one call of the one-token MoE block per window row."""

    def __call__(self, x):
        window = REV.current()
        rows = _rows(x)
        if window is None or rows <= 1:
            return super().__call__(x)
        flat = x.reshape(rows, 1, 1, x.shape[-1])
        out = mx.concatenate([super(_RowExactMoE, self).__call__(flat[r]) for r in range(rows)], axis=0)
        window.note("moe", "per_row", rows)
        return out.reshape(*x.shape[:-1], out.shape[-1])


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


class _RowExactAttention:
    """Mixin: split a window's attention into one-token forwards."""

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
        if refusal is None and _projected is None:
            from . import qwen4_exp as Q

            if Q._QSA_FUSED_PROJ:
                refusal = "attention_fused_projection_table"
        if refusal is not None:
            window.fail(refusal)
            return stock(
                x, mask, cache, _projected=_projected, _return_pre_o=_return_pre_o,
                _selection=_selection, _fetched_kv=_fetched_kv,
            )
        if _projected is None:
            _projected = (
                self.q_proj(x),
                self.k_proj(x),
                self.v_proj(x),
                self.indexer.index_qk_proj(x),
            )
        from .qwen4_qsa_indexed_merge import fused_gate_enabled

        # Unless the indexed kernel may fuse the output gate (policy
        # indexed_output_gate, default off), a one-token forward ends in
        # ``o_proj(out * sigmoid(gate))``: stop each row before the output
        # projection and run it once for the window with the row-exact kernel.
        pre_o = not fused_gate_enabled()
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
            )
            if pre_o:
                outputs.append(result[0])
                gates.append(result[1])
            else:
                outputs.append(result)
        window.note("attention", "per_row", length)
        if not pre_o:
            window.note("attention_o_proj", "per_row", length)
            return mx.concatenate(outputs, axis=1)
        out = mx.concatenate(outputs, axis=1)
        gate = mx.concatenate(gates, axis=1)
        return self.o_proj(out * mx.sigmoid(gate))


_SUBCLASSES: Dict[type, type] = {}


def _subclass(mixin, cls):
    key = (mixin, cls)
    sub = _SUBCLASSES.get(key)
    if sub is None:
        sub = type("RowExact" + cls.__name__, (mixin, cls), {})
        _SUBCLASSES[key] = sub
    return sub


class RowExactVerify:
    """Installed route: class swaps on the trunk plus the verify hooks."""

    def __init__(self, model):
        from .qwen3_next import Qwen3NextSparseMoeBlock
        from .qwen4_exp import Attention, GatedDeltaNet

        self.model = model
        self.enabled = False
        self._lock = threading.Lock()
        self._swapped = []
        self._pending: Optional[REV.Window] = None
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
            elif cls is Attention:
                mixin = _RowExactAttention
            elif cls is Qwen3NextSparseMoeBlock:
                mixin = _RowExactMoE
            else:
                continue
            self._swapped.append((module, cls))
            module.__class__ = _subclass(mixin, cls)
        object.__setattr__(model, "mtp_verify_backbone", self.verify_backbone)
        object.__setattr__(model, "mtp_verify_logits", self.verify_logits)

    # -- control ---------------------------------------------------------------
    def enable(self, flag: bool = True) -> None:
        self.enabled = bool(flag)

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
        return sum(int(layer.linear_attn.fused_gdn_verify_calls) for layer in self._gdn)

    def verify_backbone(self, tokens, cache):
        model = self.model
        rows = int(tokens.size)
        if not self.enabled or int(tokens.shape[-1]) <= 1:
            if self.enabled:
                self.counts["one_row_passthrough"] += 1
            self._pending = None
            return model.mtp_backbone(tokens, cache=cache)
        from . import qwen4_exp as Q

        record = REV.Window(rows)
        before = self._gdn_engaged()
        with REV.window(record), Q._declared_width(1):
            out = model.language_model.model(tokens, cache, return_hyper=True)
        engaged = self._gdn_engaged() - before
        if engaged != len(self._gdn):
            record.fail("gdn_fused_verify_not_engaged")
        record.note("gdn", "fused_verify", engaged)
        self._pending = record
        return out

    def verify_logits(self, hidden):
        record, self._pending = self._pending, None
        if record is None:
            return self.model.logits(hidden)
        with REV.window(record):
            logits = self.model.logits(hidden)
        self._close(record)
        return logits

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
    def snapshot(self) -> dict:
        return {
            "windows": self.counts["windows"],
            "windows_not_exact": self.counts["windows_not_exact"],
            "enabled": self.enabled,
        }

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
            if not (start.get("enabled") and self.enabled):
                reason = "route_not_enabled"
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
        counts = self.counts
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "installed_modules": len(self._swapped),
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
