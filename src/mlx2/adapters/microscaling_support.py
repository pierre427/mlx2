"""Adapter-owned admission metadata for opt-in SC-MXFP4 prefill.

These declarations select an implementation seam, not a qualification. The
shared installer still validates every selected tensor's format and range.
"""
from __future__ import annotations

from collections.abc import Mapping


_PROJECTION_OWNERS = (
    'lane_dense', 'lane_matmul', 'sp_qmm', 'tensorfold_prefill',
    'tensorfold_qmv_rows', 'invariant_prefill', 'row_exact_verify',
    'row_exact_low_bit_lane', 'target_verify_row_exact',
    'varlen_dense_mlp', 'varlen_sparse_moe', 'compact_moe_prefill',
)


def projection_law_compatible(adapter) -> bool:
    """Decline other projection owners and opaque streamed weight templates."""
    if getattr(adapter, 'weight_stream', None) is not None:
        return False
    for policy_name in ('policy', 'external_policy'):
        policy = getattr(adapter, policy_name, None)
        for name in _PROJECTION_OWNERS:
            value = policy.get(name, False) if isinstance(policy, Mapping) else getattr(policy, name, False)
            if value:
                return False
    return not any(getattr(adapter, name, None) for name in (
        'tensorfold_prefill', 'tensorfold_qmv', 'invariant_prefill',
        'varlen_dense_mlp', 'varlen_sparse_moe',
    ))


def decode_row_multiplier(config) -> int:
    """Maximum routed-expert expansion, derived from validated artifact data."""
    if not isinstance(config, Mapping):
        raise ValueError('microscaling requires the adapter configuration')
    text = config.get('text_config', config)
    if not isinstance(text, Mapping):
        raise ValueError('microscaling text configuration must be an object')
    experts = text.get('num_experts', config.get('num_experts', 0))
    if type(experts) is not int or experts < 0:
        raise ValueError('microscaling expert count must be a nonnegative integer')
    if not experts:
        return 1
    values = [source[name] for source in (config, text)
              for name in ('num_experts_per_tok', 'num_experts_per_token')
              if name in source]
    if not values or any(type(v) is not int or not 1 <= v <= experts for v in values):
        raise ValueError('microscaling MoE decode requires a valid expert top-k')
    return max(values)
