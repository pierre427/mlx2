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
                one-row ``qmv_fast``/``qmv`` bits per row) through the shared
                package's canary-gated ``row_exact.ops.quantized_linear`` /
                ``quantized_linears``, or one stock one-row call per row
                where the kernel does not cover the layout or its canary
                fails
dense           every trunk dense ``nn.Linear`` (mixed-precision artifacts
projections     keep the HC ``block_inject_weight`` and the MoE
                ``shared_expert_gate`` bf16): R broadcast matrix-vector
                products (``row_exact.ops.dense_linear``, canary-checked per
                build against the one-token call; a failed canary, a
                biased layer or a one-output layer such as the bf16 shared
                gate runs one one-token call per row).  MLX's dense
                matmul is ``gemv`` at one row and ``gemv_wide``/GEMM tiling
                at more, so the composed M=R call is not the one-token
                arithmetic
norms           ``GroupRMSNorm`` sees the one-token width (the width-8 fast
                RMSNorm gate would move wider windows to the eager norm)
attention       batched row-exact projections, then per row the one-token
                QSA selection, cache append, RoPE and SDPA with the ordinary
                B=1 mask, and the one-row output projection
GDN             the fused verify kernel (bit-identical to the one-token
                fused decode kernel, rollback included, when both compile
                the same served-SiLU header; Metal-checked at 2..17 rows); a
                window whose GDN falls back to the stock multi-row block, or
                whose verify kernel kind fails its per-build decode-parity
                canary (``decode_parity_probe``), is not row-exact
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
from . import row_exact_components as REC
from . import row_exact_qmv as REQ

_RowExactGatedDeltaNet = REC.RowExactGatedDeltaNetProjectionsMixin
_RowExactLinear = REC.RowExactLinearMixin
_RowExactMoE = REC.RowExactMoEMixin
_RowExactQuantizedLinear = REC.RowExactQuantizedLinearMixin
_RowExactSwitch = REC.RowExactSwitchMixin
_grouped = REC.grouped_quantized_linears
_rows = REC.row_count
_subclass = REC.row_exact_subclass

SCHEMA = "mlx2.qwen4-row-exact-verify.v1"
ENV = "MLX_QWEN4_ROW_EXACT_VERIFY"
# Safety boundary, not qualification: B=4/8 matched the B=1 oracle, whereas
# B=16 diverged despite every verify stage reporting success. See
# qualification/runs/batched-gdn-verify-20261001/README.md. Do not raise this
# until the complete target path (including one-token rounds) is evidenced.
MAX_EVIDENCED_LANES = 8
# Layer names a per-row label lists (the counts cover all of them).
_LABEL_LAYERS = 16


def _has_segmented_layer(cache) -> bool:
    try:
        layers = list(cache) if cache is not None else []
    except TypeError:
        layers = [cache]
    return any(
        getattr(layer, "segmented_attention", None) is not None for layer in layers
    )


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
        if window is None or length < 1:
            return stock(
                x, mask, cache, _projected=_projected, _return_pre_o=_return_pre_o,
                _selection=_selection, _fetched_kv=_fetched_kv,
            )
        if length == 1 and not (
            getattr(cache, "segmented_attention", None) is not None
            and _projected is None
        ):
            # A one-row lineage inside a window (a ragged lane on its last
            # token, or K=0): its B1 row cache gives no mask, but the ordinary
            # B=1 route's SDPA reads an explicit all-valid one and the two
            # differ in bits (sweep 2026-10-02 S1).
            if _selection is not None or _fetched_kv is not None:
                window.fail("attention_preselected")
            elif int(x.shape[0]) != 1:
                window.fail("attention_batch_rows")
            else:
                row_mask = _ordinary_b1_mask(cache)
                if row_mask is None:
                    window.fail("attention_offset_not_host")
                else:
                    mask = row_mask
                    window.note("attention", "per_row", 1)
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

    # Set at install (``_per_row_inventory``); None on a bare handle.
    per_row_layers: Optional[Dict[str, Any]] = None
    # Set at install with ``low_bit_lane`` (``_prepare_low_bit_lane``).
    lane_layers: Optional[Dict[str, Any]] = None

    def __init__(self, model, *, low_bit_lane: bool = False):
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
        self.low_bit_lane = bool(low_bit_lane)
        if self.low_bit_lane:
            try:
                self.lane_layers = self._prepare_low_bit_lane()
            except ValueError:
                self.remove()  # restore every class swapped above
                raise
        self.per_row_layers = self._per_row_inventory()
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

    def _per_row_inventory(self) -> Dict[str, Any]:
        """Swapped projections whose verify rows run one one-row call each.

        ``slow_path`` (and the receipt label) is raised by quantized layers
        only; see the comment below.  A quantized layout the row-exact kernel does not cover (2/3-bit
        weights above all; ``row_exact_qmv.layout_decline_reason``) computes
        each verify row as its own one-token call: exact, but the weights are
        read once per row, so a wide verify window costs about that many
        decode steps.  Static: the layout decides, not the device (off Metal
        every layer runs per row anyway).  Pierre 2026-10-14: keep serving,
        say so.  Dense ``nn.Linear`` layers (``_RowExactLinear``) are listed
        for completeness: they run the shared package's broadcast matvec
        rows where its parity canary holds and one one-token call per row
        where it does not (a biased layer always); the route a window took is
        in ``status()["stages"]["projections"]``."""
        language = self.model.language_model
        names = {id(module): name for name, module in language.named_modules()}
        quantized: Dict[str, Any] = {"count": 0, "by_format": {}, "by_reason": {}, "layers": []}
        dense: Dict[str, Any] = {"count": 0, "by_format": {}, "layers": []}
        for module, cls in self._swapped:
            if cls is nn.QuantizedLinear:
                reason = REQ.layout_decline_reason(module)
                if reason is None or REQ.lane_preparation(module) is not None:
                    continue
                entry = quantized
                entry["by_reason"][reason] = entry["by_reason"].get(reason, 0) + 1
            elif cls is nn.Linear:
                entry = dense
            else:
                continue
            label = REQ.layout_label(module)
            entry["count"] += 1
            entry["by_format"][label] = entry["by_format"].get(label, 0) + 1
            entry["layers"].append(names.get(id(module), "?"))
        for entry in (quantized, dense):
            entry["layers"] = sorted(entry["layers"])[:_LABEL_LAYERS]
        return {
            "exact": True,
            # Kernel-declined quantized layers are the slow path that
            # matters (whole trunk or head matrices re-read per row).  Dense
            # layers are listed but do not raise the flag: every Flash-Next
            # artifact keeps tiny dense HC mixers (K = streams), and mixed
            # artifacts a bf16 gate or inject, whose per-row calls are the
            # designed one-token arithmetic at negligible weight bytes.
            "slow_path": bool(quantized["count"]),
            "path": "one stock one-row call per verify row; weights read once per row",
            "quantized": quantized,
            "dense": dense,
        }

    def _prepare_low_bit_lane(self) -> Dict[str, Any]:
        """Prepare the lane matmul for every layer the row-exact kernel
        declines (2/3-bit above all), opt-in (``low_bit_lane``).

        Such a layer then runs one lane law for its one-token decode and its
        verify rows instead of one stock call per verify row.  That law is
        not stock MLX: MTP-on output still equals MTP-off output of this
        route, but both differ from a route without the option, so the
        receipt binds ``law_id`` and the adapter's numerics contract binds
        it into APCv2.  On M1-M4 only the single-kernel ``affine_rows``
        family is admitted (the scalar/matrix twins are checked by the lane
        installer, which this route does not run).  Unqualified.

        Explicit and all-or-nothing: a selection that would leave any
        declined layer on the per-row path (no lane backend, a refused
        layout), or that finds no declined layer at all, raises before any
        layer is prepared, so a route never runs the option half-engaged."""
        from ..lane import matmul as LM
        from ..lane import simd as LS
        from ..lane.installer import LAW_IDS

        backend = LM.backend()
        declined = [
            module for module, cls in self._swapped
            if cls is nn.QuantizedLinear and REQ.layout_decline_reason(module) is not None
        ]
        refused: Dict[str, int] = {}
        prepared = []
        for module in declined:
            reason = None
            if backend is None:
                reason = "lane_unavailable"
            else:
                try:
                    lw = LM.prepare(module, backend)
                except LM.LaneUnsupported as error:
                    reason = str(error)
                else:
                    if backend == "simd" and LS.path(
                        lw.bits, lw.group_size, lw.n, lw.k, lw.scales_dtype, lw.biases.dtype
                    ) != "affine":
                        reason = "simd_twin_unchecked"
            if reason is not None:
                refused[reason] = refused.get(reason, 0) + 1
            else:
                prepared.append((module, lw))
        if not declined:
            raise ValueError(
                "row_exact_low_bit_lane is selected but the model has no layer the "
                "row-exact kernel declines"
            )
        if refused:
            raise ValueError(
                "row_exact_low_bit_lane cannot prepare every declined layer "
                f"({len(prepared)} of {len(declined)}; refused {refused})"
            )
        by_format: Dict[str, int] = {}
        for module, lw in prepared:
            object.__setattr__(module, REQ.LANE_ATTR, lw)
            label = REQ.layout_label(module)
            by_format[label] = by_format.get(label, 0) + 1
        return {
            "backend": backend,
            "law_id": f"{LAW_IDS[backend]}+row-exact-low-bit",
            "intended": len(declined),
            "count": len(prepared),
            "by_format": by_format,
            "refused": {},
            "qualified": False,
        }

    # -- control ---------------------------------------------------------------
    def enable(self, flag: bool = True) -> None:
        self.enabled = bool(flag)
        if self.enabled:
            # Re-audit: a mechanism may have swapped classes since install.
            self.static_refusals = self._audit()

    def remove(self) -> None:
        for module, cls in self._swapped:
            module.__class__ = cls
            module.__dict__.pop(REQ.LANE_ATTR, None)
        self._swapped = []
        for name in ("mtp_verify_backbone", "mtp_verify_logits"):
            if name in self.model.__dict__:
                object.__delattr__(self.model, name)
        self.enabled = False

    # -- verify hooks ----------------------------------------------------------
    def _gdn_kinds(self) -> Dict[str, int]:
        """Fused verify launches so far, by kernel kind (``DECODE_PARITY_KINDS``).

        A batched window's GDN is engaged when the batched fused verify ran
        (policy fused_gdn_batch_verify): every lane runs its B=1 verify
        arithmetic.  Either way the rows are the one-token decode bits only
        if that kernel kind passes its decode-parity canary on this build
        (``verify_backbone``)."""
        kinds = dict.fromkeys(("replay", "snapshot", "batch_replay", "batch_snapshot"), 0)
        for layer in self._gdn:
            attn = layer.linear_attn
            verify = int(attn.fused_gdn_verify_calls)
            replay = int(getattr(attn, "fused_gdn_replay_verify_calls", 0))
            batch = int(getattr(attn, "fused_gdn_batch_verify_calls", 0))
            compact = int(getattr(attn, "fused_gdn_batch_verify_compact_calls", 0))
            kinds["replay"] += replay
            kinds["snapshot"] += verify - replay
            kinds["batch_replay"] += compact
            kinds["batch_snapshot"] += batch - compact
        return kinds

    def _gdn_engaged(self) -> int:
        return sum(self._gdn_kinds().values())

    def _gdn_state_dtype(self):
        dtype = getattr(self._gdn[0].linear_attn, "_gdn_state_dtype", None) if self._gdn else None
        # ``None`` is the default fp32 class (mx.Dtype refuses ``== None``).
        return mx.float16 if dtype is not None and dtype == mx.float16 else mx.float32

    def _gdn_decode_parity(self, kind: str) -> bool:
        """Whether ``kind``'s verify kernel reproduces serial fused decode on
        this device and MLX build (one canary per kind and state class per
        process; the shared registry reports it in ``status()``)."""
        from ..row_exact.canary import CANARIES
        from .qwen4_fused_gdn_verify import decode_parity_probe

        state_dtype = self._gdn_state_dtype()
        key = ("qwen4_fused_gdn_verify_vs_decode", kind, str(state_dtype))
        return CANARIES.exact(key, decode_parity_probe(kind, state_dtype=state_dtype))

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
            elif self.enabled and _has_segmented_layer(cache):
                # A single lane at K=0 through a segmented cache: each B1 row
                # cache gives its attention no mask, not the ordinary B=1
                # route's explicit one (sweep 2026-10-02 S1).  Not row-exact.
                record = REV.Window(rows)
                record.note("target_backbone", "segmented_one_token_passthrough")
                record.fail("segmented_one_token_target_not_row_exact")
                self._close(record)
            return model.mtp_backbone(tokens, cache=cache)
        from . import qwen4_exp as Q

        self._retire_pending()
        record = REV.Window(rows)
        self._admit_batch(record, lanes)
        for reason in self.static_refusals:
            record.fail(reason)
        before = self._gdn_kinds()
        with REV.window(record), Q._declared_width(1):
            out = model.language_model.model(tokens, cache, return_hyper=True)
        after = self._gdn_kinds()
        launched = {kind: after[kind] - before[kind] for kind in after}
        engaged = sum(launched.values())
        if engaged != len(self._gdn):
            record.fail("gdn_fused_verify_not_engaged")
        record.note("gdn", "fused_verify", engaged)
        for kind, count in launched.items():
            if count and not self._gdn_decode_parity(kind):
                # The kernel ran, but its rows are not the decode step's bits
                # on this build: the window is not row-exact.
                record.fail(f"gdn_{kind}_verify_differs_from_decode")
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
        receipt = {
            "schema": SCHEMA,
            "row_exact": reason is None,
            "reason": reason,
            "windows": max(0, windows),
            "windows_not_exact": max(0, not_exact),
        }
        if (self.per_row_layers or {}).get("slow_path"):
            # Only when some layer takes the per-row path, so receipts of
            # fully covered models are unchanged.
            receipt["per_row_layers"] = self.per_row_layers
        if (self.lane_layers or {}).get("count"):
            receipt["low_bit_lane"] = self.lane_layers
        return receipt

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
                "per_row_layers": self.per_row_layers,
                "low_bit_lane": self.lane_layers,
                "gdn_layers": len(self._gdn),
                "windows": counts["windows"],
                "windows_row_exact": counts["windows_row_exact"],
                "windows_not_exact": counts["windows_not_exact"],
                "rows": counts["rows"],
                "one_row_passthrough": counts["one_row_passthrough"],
                "stages": {k: dict(v) for k, v in counts["stages"].items()},
                "failures": dict(counts["failures"]),
                # Parity-canary verdicts of the shared row_exact ops this
                # route's projections run through (a failed form ran per row).
                **_package_status(),
            }


def _package_status() -> dict:
    from .. import row_exact

    return row_exact.status()


def install(model, *, low_bit_lane: bool = False) -> RowExactVerify:
    """Install the route on a loaded qwen4_exp ``Model`` (disabled until
    ``enable``).  ``low_bit_lane`` (default off) runs the layers the row-exact
    kernel declines on the lane law instead of one stock call per row."""
    existing = getattr(model, "_mlx2_row_exact_verify", None)
    if existing is not None:
        if bool(getattr(existing, "low_bit_lane", False)) != bool(low_bit_lane):
            raise ValueError("row-exact verify is already installed with another low_bit_lane")
        return existing
    handle = RowExactVerify(model, low_bit_lane=low_bit_lane)
    object.__setattr__(model, "_mlx2_row_exact_verify", handle)
    return handle
