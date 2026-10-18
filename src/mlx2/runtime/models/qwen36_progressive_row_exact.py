# SPDX-License-Identifier: Apache-2.0
"""Default-off Qwen3.6 progressive target-row executor candidate.

This module is deliberately not a protocol declaration or qualification.
It supplies a CPU-testable, adapter-owned implementation of the
``Qwen3NextAttention`` B1/S1 arithmetic and installs the shared row-exact
projection/MoE/GDN-projection components explicitly.  The complete executor
continues to attest ``row_exact=False`` until the Qwen3.6 recurrent transition
and real-artifact Metal widths 2/4/8 have passed bitwise serial comparison.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import mlx.core as mx
from mlx import nn

from .. import row_exact
from .. import row_exact_verify as REV
from ..segmented_plain_kv import SegmentedBatchKVCache
from .base import create_causal_mask
from .row_exact_components import (
    RowExactGatedDeltaNetProjectionsMixin,
    RowExactLinearMixin,
    RowExactMoEMixin,
    RowExactQuantizedLinearMixin,
    RowExactSwitchMixin,
    row_count,
    row_exact_subclass,
)

SCHEMA = "mlx2.qwen36-progressive-row-exact-candidate.v1"
CANDIDATE_PROTOCOL = "qwen36-b1s1-progressive-target-candidate-v1"
MAX_PENDING_METAL_WIDTH = 8


def _ordinary_b1_mask(cache):
    offset = getattr(cache, "offset", None)
    if not isinstance(offset, Integral):
        return None
    return create_causal_mask(
        1,
        offset=int(offset),
        left_padding=mx.array([0]),
    )


class RowExactQwen3NextAttentionMixin:
    """Qwen3Next attention using one B1/S1 state transition per valid row.

    Q/K/V and O projections retain a packed call surface, but their installed
    projection mixins perform one-row arithmetic.  Attention, RoPE, cache
    append, normalization, gating, and SDPA are then invoked once per token
    against the authoritative B1 cache.  Ragged padding never reaches a row
    cache.
    """

    def _row_exact_serial_attention(self, x, cache, lengths):
        window = REV.current()
        _, width, _ = x.shape
        q_all = self.q_proj(x)
        k_all = self.k_proj(x)
        v_all = self.v_proj(x)
        lane_outputs = []
        for lane, (row_cache, valid) in enumerate(zip(cache, lengths)):
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
        window.note("attention", "serial_b1s1", sum(lengths))
        return self.o_proj(mx.concatenate(lane_outputs, axis=0))

    def __call__(self, x, mask=None, cache=None):
        window = REV.current()
        if window is None or row_count(x) <= 1:
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


class Qwen36ProgressiveRowExactCandidate:
    """Explicitly installed candidate; never a qualification/default switch."""

    def __init__(self, model):
        from .qwen3_next import FusedDownSwitchGLU, FusedGateUpSwitchGLU
        from .qwen36_35b import GatedDeltaNet, Qwen3NextAttention
        from .qwen36_moe_decode import Qwen36SparseMoeBlock

        self.model = model
        self._swapped = []
        self.counts: dict[str, Any] = {
            "launches": 0,
            "row_exact": 0,
            "not_exact": 0,
            "failures": {},
        }
        self._had_hook = "progressive_external_target_rows" in model.__dict__
        self._previous_hook = model.__dict__.get("progressive_external_target_rows")
        self._gdn = []
        for _, module in model.named_modules():
            cls = type(module)
            if cls is nn.QuantizedLinear:
                mixin = RowExactQuantizedLinearMixin
            elif cls is nn.Linear:
                mixin = RowExactLinearMixin
            elif cls is Qwen3NextAttention:
                mixin = RowExactQwen3NextAttentionMixin
            elif cls is Qwen36SparseMoeBlock:
                mixin = RowExactMoEMixin
            elif cls is GatedDeltaNet:
                mixin = RowExactGatedDeltaNetProjectionsMixin
                self._gdn.append(module)
            elif cls in (FusedGateUpSwitchGLU, FusedDownSwitchGLU):
                mixin = RowExactSwitchMixin
            else:
                continue
            self._swapped.append((module, cls))
            module.__class__ = row_exact_subclass(mixin, cls)
        self.static_refusals = self._audit()
        object.__setattr__(
            model,
            "progressive_external_target_rows",
            self.execute,
        )

    def _audit(self):
        swapped = {id(module) for module, _ in self._swapped}
        refusals: dict[str, int] = {}
        for _, module in self.model.named_modules():
            if id(module) in swapped:
                continue
            if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                reason = "unswapped_projection:" + type(module).__name__
                refusals[reason] = refusals.get(reason, 0) + 1
        language = getattr(self.model, "language_model", None)
        if getattr(getattr(language, "args", None), "tie_word_embeddings", False):
            refusals["tied_embedding_head"] = 1
        return refusals

    def execute(self, tokens, caches, capture_layers):
        from ..progressive_external_verify import ProgressiveTargetRows

        rows = int(tokens.shape[0] * tokens.shape[1])
        record = REV.Window(rows)
        # These are evidence gates, not inferred implementation defects.  The
        # candidate cannot attest exactness until the artifact matrix closes.
        record.fail("real_artifact_metal_widths_2_4_8_unverified")
        for reason in self.static_refusals:
            record.fail(reason)
        if int(tokens.shape[0]) > MAX_PENDING_METAL_WIDTH:
            record.fail("physical_width_above_pending_evidence")
        before_gdn = sum(
            int(layer.fused_gdn_verify_calls)
            + int(getattr(layer, "fused_gdn_batch_verify_calls", 0))
            for layer in self._gdn
        )
        with REV.window(record):
            logits, features = self.model.forward_with_taps(
                tokens,
                caches,
                capture_layers,
            )
        engaged_gdn = (
            sum(
                int(layer.fused_gdn_verify_calls)
                + int(getattr(layer, "fused_gdn_batch_verify_calls", 0))
                for layer in self._gdn
            )
            - before_gdn
        )
        record.note("gdn", "fused_verify", engaged_gdn)
        if engaged_gdn != len(self._gdn):
            record.fail("gdn_fused_verify_not_engaged")
        self.counts["launches"] += 1
        self.counts["row_exact" if record.exact else "not_exact"] += 1
        for reason, count in record.failures.items():
            failures = self.counts["failures"]
            failures[reason] = failures.get(reason, 0) + count
        first_reason = next(iter(sorted(record.failures)), None)
        return ProgressiveTargetRows(
            logits=logits,
            features=features,
            protocol=CANDIDATE_PROTOCOL,
            row_exact=record.exact,
            detail={
                "schema": SCHEMA,
                "candidate": True,
                "launch": self.counts["launches"],
                "rows": rows,
                "physical_width": int(tokens.shape[0]),
                "token_width": int(tokens.shape[1]),
                "failure_count": sum(record.failures.values()),
                "reason": first_reason,
            },
        )

    def status(self):
        return {
            "schema": SCHEMA,
            "installed": bool(self._swapped),
            "qualified": False,
            "selected": False,
            "observed_used": bool(self.counts["launches"]),
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "pending_evidence": "real_artifact_metal_bitwise_widths_2_4_8",
            "static_refusals": dict(self.static_refusals),
            "counts": {
                **self.counts,
                "failures": dict(self.counts["failures"]),
            },
            **row_exact.status(),
        }

    def remove(self):
        for module, cls in reversed(self._swapped):
            module.__class__ = cls
        self._swapped = []
        if self._had_hook:
            object.__setattr__(
                self.model,
                "progressive_external_target_rows",
                self._previous_hook,
            )
        elif "progressive_external_target_rows" in self.model.__dict__:
            object.__delattr__(self.model, "progressive_external_target_rows")


def install(model):
    """Install the explicit, unqualified Qwen3.6 executor candidate."""

    return Qwen36ProgressiveRowExactCandidate(model)
