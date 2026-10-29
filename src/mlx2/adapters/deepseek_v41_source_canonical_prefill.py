# SPDX-License-Identifier: MIT
"""Explicit ordinary-B1 prefill law for exact Source prefix reuse.

The outer scheduler may cut anywhere, including an APC interior checkpoint.
Every token still uses the ordinary one-row program.  This deliberately trades
prefill throughput for a partition-independent numerical profile; it does not
change expert memory admission or qualify a serving route.
"""
import hashlib
from pathlib import Path

POLICY = 'canonical_m1_v1'
ADAPTIVE = 'adaptive_v1'


def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def partition_receipt(policy):
    if policy not in (ADAPTIVE, POLICY):
        raise ValueError('Unsupported Source prefill partition policy')
    canonical = policy == POLICY
    return dict(policy=policy, arithmetic_rows=1 if canonical else 'adaptive',
                outer_chunk_changes_numerics=not canonical,
                final_token_law='ordinary_B1', adaptive_expert_residency=True,
                apcv2_reuse_enabled=canonical, qualification='unqualified',
                prefill_performance='ordinary B1 per token' if canonical else 'adaptive chunks')


def canonical_forward(model, forward, inputs, cache=None, *, logits_to_keep=None,
                      capture_layers=(), body_only=False):
    """Run the original B1 forward in token order, retaining original ownership.

    Validate the complete input before the first mutation. Each ordinary call
    owns its existing completion/memory/bank scope. Failure leaves its existing
    incomplete-state guard in force; there is no approximate publication or
    silent switch back to a multi-row arithmetic law.
    """
    import mlx.core as mx
    from ..runtime.prefill_checkpoint_rows import select_checkpoint_rows
    if (model.identity != model._source_canonical_identity
            or getattr(model.weights, 'source_prefill_partition_policy', None) != POLICY):
        raise ValueError('Source canonical prefill owner/profile changed')
    manager = getattr(model, '_source_ast_bank_manager', None)
    if manager is not None:
        manager.before_ordinary()
    cache, _ = model.validate_forward(inputs, cache, logits_to_keep)
    rows = inputs.shape[1]
    if rows == 1:
        return forward(inputs, cache, logits_to_keep=logits_to_keep,
                       capture_layers=capture_layers, body_only=body_only)
    logits, features = [], []
    keep = rows if logits_to_keep in (None, 0) else min(rows, logits_to_keep)
    try:
        for row in range(rows):
            wanted = row >= rows - keep
            out, taps = forward(inputs[:, row:row + 1], cache, logits_to_keep=0,
                                capture_layers=capture_layers,
                                body_only=body_only or not wanted)
            if out is not None:
                logits.append(out)
            if taps is not None:
                features.append(taps)
    except BaseException:
        # A partially completed outer call must never become an APC entry,
        # even if the failing B1 call failed before setting its own marker.
        for state in cache:
            state._in_forward = True
        raise
    result = mx.concatenate(logits, axis=1) if logits else None
    # The existing prompt guard may request just its last row. Apply that
    # request after serial B1 heads, preserving each head's ordinary row law.
    if result is not None and logits_to_keep is None:
        result = select_checkpoint_rows(result)
    stats = model.scheduled_kv_calls
    stats['source_canonical_m1_prefill_tokens'] = stats.get('source_canonical_m1_prefill_tokens', 0) + rows
    stats['source_canonical_m1_prefill_outer_calls'] = stats.get('source_canonical_m1_prefill_outer_calls', 0) + 1
    return result, mx.concatenate(features, axis=1) if features else None


def canonical_model_class():
    """Keep the model's own layer/self references; no proxy or global patch."""
    from ..runtime.models.deepseek_v41 import DeepSeekV41Model

    class CanonicalM1SourceModel(DeepSeekV41Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._source_canonical_identity = self.identity

        def _forward(self, inputs, cache=None, *, logits_to_keep=None,
                     capture_layers=(), body_only=False):
            return canonical_forward(self, super()._forward, inputs, cache,
                logits_to_keep=logits_to_keep, capture_layers=capture_layers, body_only=body_only)

    return CanonicalM1SourceModel
