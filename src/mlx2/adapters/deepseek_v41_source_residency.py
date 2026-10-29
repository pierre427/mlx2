# SPDX-License-Identifier: MIT
"""Settable Source expert residency: split pools, pre-eviction, live maximum.

Exact modes keep the frozen hot-map precision law. ``dynamic_precision_v1`` is
an explicit approximate profile whose precision depends on cache history.
See provenance/deepseek-v41-source-expert-residency.json.
"""
import math

RESIDENCY_SHARED = 'shared_v1'
RESIDENCY_SPLIT = 'split_v1'
RESIDENCY_DYNAMIC = 'dynamic_precision_v1'
RESIDENCY_MODES = (RESIDENCY_SHARED, RESIDENCY_SPLIT, RESIDENCY_DYNAMIC)
CACHE_MAXIMUM_POLICIES = ('explicit', 'live_v1')
EXPLICIT_MAXIMUM_BYTES = 90_000_000_000
LIVE_MAXIMUM_CLAMP_BYTES = 120_000_000_000
# Historical measured default; used when live availability cannot be read.
UNKNOWN_MEMORY_FALLBACK_BYTES = 75_000_000_000
# Automatic expert-cache maximum: the Hot60 map (60 GB) plus the measured
# ~17.7 GB Q1 working set; 78 GB made warm requests read zero expert bytes in
# the 2026-10-28 A/B. The adaptive target still shrinks under live pressure.
AUTO_MAXIMUM_BYTES = 78_000_000_000
HIGH_FRACTION_BOUNDS = (0.05, 0.95)
PRE_EVICT_MAXIMUM = 16
DEFAULTS = dict(high_fraction='auto', pre_evict_slots=0, live_reserve_bytes=16 << 30,
                promote_slots=8, promote_decay=0.95, promote_min_score=2.0, promote_hysteresis=1.5)
RESIDENCY_KEYS = frozenset(('residency', 'high_fraction', 'pre_evict_slots', 'cache_maximum',
                            'live_reserve_bytes', 'promote_slots', 'promote_decay',
                            'promote_min_score', 'promote_hysteresis', 'prefill_residency'))
SPLIT_KEYS = frozenset(('high_fraction', 'pre_evict_slots'))
DYNAMIC_KEYS = frozenset(('promote_slots', 'promote_decay', 'promote_min_score', 'promote_hysteresis'))
# Contracts that assume one fixed precision per expert for a model lifetime,
# or that publish/reuse state, refuse dynamic precision.
DYNAMIC_REFUSALS = (
    ('deepseek_physical_batch', 'b1'), ('deepseek_source_dspark_ast', 'off'),
    ('deepseek_source_dspark_prepare', 'off'), ('deepseek_source_resident_gather', 'reference'),
    ('deepseek_source_resident_prefill', 'reference'), ('deepseek_source_native_mx', 'reference'),
    ('deepseek_source_frame_reuse', 'reference'), ('deepseek_source_row_packet', 'reference'),
    ('deepseek_source_prefill_down', 'reference'), ('deepseek_source_prefill_partition', 'adaptive_v1'),
    ('deepseek_source_prefill_order', 'reference'), ('deepseek_ced', 'reference'),
)
DYNAMIC_APC_REASON = 'dynamic_precision_v1_output_depends_on_cache_history'


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(result, *, adaptive, original_bytes):
    """Validate residency keys of a parsed expert policy in place."""
    residency = result.get('residency', RESIDENCY_SHARED)
    if residency not in RESIDENCY_MODES:
        raise ValueError('Source expert residency must be one of ' + ', '.join(RESIDENCY_MODES))
    if residency != RESIDENCY_SHARED:
        if result.get('mode') != 'hot_original_q1':
            raise ValueError('Split/dynamic Source expert residency requires hot_original_q1')
        if not adaptive:
            raise ValueError('Split/dynamic Source expert residency requires adaptive residency')
    present = set(result)
    if residency == RESIDENCY_SHARED and present & (SPLIT_KEYS | DYNAMIC_KEYS):
        raise ValueError('high_fraction/pre_evict_slots/promotion options require split_v1 or dynamic_precision_v1')
    if residency != RESIDENCY_DYNAMIC and present & DYNAMIC_KEYS:
        raise ValueError('promotion options require dynamic_precision_v1')
    if 'high_fraction' in result:
        value = result['high_fraction']
        low, high = HIGH_FRACTION_BOUNDS
        if value != 'auto' and (type(value) is not float or not low <= value <= high):
            raise ValueError(f'high_fraction must be "auto" or a float in [{low}, {high}]')
    if 'pre_evict_slots' in result:
        value = result['pre_evict_slots']
        if type(value) is not int or not 0 <= value <= PRE_EVICT_MAXIMUM:
            raise ValueError(f'pre_evict_slots must be an integer 0..{PRE_EVICT_MAXIMUM}')
    if 'promote_slots' in result and (type(result['promote_slots']) is not int or not 1 <= result['promote_slots'] <= 64):
        raise ValueError('promote_slots must be an integer 1..64')
    if 'promote_decay' in result and (type(result['promote_decay']) is not float or not 0.5 <= result['promote_decay'] <= 0.999):
        raise ValueError('promote_decay must be a float in [0.5, 0.999]')
    if 'promote_min_score' in result and (not _number(result['promote_min_score']) or type(result['promote_min_score']) is bool
                                          or not 0 < result['promote_min_score'] <= 1e6):
        raise ValueError('promote_min_score must be positive')
    if 'promote_hysteresis' in result and (type(result['promote_hysteresis']) is not float
                                           or not 1.0 <= result['promote_hysteresis'] <= 4.0):
        raise ValueError('promote_hysteresis must be a float in [1.0, 4.0]')
    from .deepseek_v41_source_prefill_residency import validate as validate_prefill_residency
    validate_prefill_residency(result, adaptive=adaptive)
    maximum = result.get('cache_maximum', 'explicit')
    if maximum not in CACHE_MAXIMUM_POLICIES:
        raise ValueError('cache_maximum must be explicit or live_v1')
    if maximum == 'live_v1' and not adaptive:
        raise ValueError('live_v1 expert maximum requires adaptive residency')
    if 'live_reserve_bytes' in result:
        value = result['live_reserve_bytes']
        if maximum != 'live_v1' or type(value) is not int or not 1 << 30 <= value <= 64 << 30:
            raise ValueError('live_reserve_bytes requires live_v1 and 1..64 GiB')
    limit = LIVE_MAXIMUM_CLAMP_BYTES if maximum == 'live_v1' else EXPLICIT_MAXIMUM_BYTES
    budget = result.get('budget_bytes')
    if type(budget) is not int or not original_bytes*2 <= budget <= limit:
        raise ValueError('Source expert maximum must fit one expert plus staging and be at most'
                         + ('120GB (live_v1 clamp)' if maximum == 'live_v1' else '90GB'))
    return result


def settings(policy):
    """Resolved residency settings; absent keys keep today's shared behavior."""
    residency = policy.get('residency', RESIDENCY_SHARED)
    result = dict(residency=residency, cache_maximum=policy.get('cache_maximum', 'explicit'))
    if residency != RESIDENCY_SHARED:
        result.update(high_fraction=policy.get('high_fraction', DEFAULTS['high_fraction']),
                      pre_evict_slots=policy.get('pre_evict_slots', DEFAULTS['pre_evict_slots']))
    else:
        result.update(high_fraction=None, pre_evict_slots=0)
    if residency == RESIDENCY_DYNAMIC:
        result.update({key: policy.get(key, DEFAULTS[key]) for key in DYNAMIC_KEYS})
    if 'prefill_residency' in policy:
        result['prefill_residency'] = policy['prefill_residency']
    if result['cache_maximum'] == 'live_v1':
        result['live_reserve_bytes'] = policy.get('live_reserve_bytes', DEFAULTS['live_reserve_bytes'])
    return result


def resolve_high_fraction(value, *, hot_objects, original_bytes, payload_maximum):
    """``auto``: the frozen hot map's bytes over the payload maximum, clamped.

    At the measured 75 GB maximum the Hot60 map (3,191 x 18.8 MB = 60.0 GB)
    gives 0.80: every hot original fits, and the remaining 15 GB is Q1.
    """
    if value != 'auto':
        return value, 'explicit'
    low, high = HIGH_FRACTION_BOUNDS
    if payload_maximum <= 0:
        return low, 'auto_hot_map_bytes_over_maximum'
    return min(high, max(low, hot_objects*original_bytes/payload_maximum)), 'auto_hot_map_bytes_over_maximum'


def derive_live_maximum(clamp, plan, *, reserve_bytes, available_bytes, minimum):
    """Expert-cache maximum from live memory at load, never above ``clamp``.

    ``plan`` is a ``weight_plan`` result: dense payload, prepared expansion,
    dense staging and metadata are charged before the operator reserve that
    covers KV, workspace and the OS. Unknown memory falls back to the smaller
    of the clamp and the historical 75 GB default; the adaptive live target
    still shrinks to measured memory afterwards.
    """
    try:
        available = available_bytes() if callable(available_bytes) else available_bytes
    except Exception:
        available = None
    known = type(available) is int and available >= 0
    dense = (plan['dense_payload_bytes']+plan['dense_prepared_bound_bytes']
             + plan['dense_staging_bound_bytes']+plan.get('metadata_bound_bytes', 0))
    charged = dense+reserve_bytes
    if not known:
        derived, reason = min(clamp, UNKNOWN_MEMORY_FALLBACK_BYTES), 'availability_unknown_fallback'
    else:
        derived = min(clamp, available-charged)
        reason = 'clamped_to_maximum' if derived == clamp else 'derived_from_live_memory'
        if derived < minimum:
            derived, reason = minimum, 'live_memory_below_progress_floor'
    return derived, dict(policy='live_v1', clamp_bytes=clamp, measured_available_bytes=available if known else None,
                         availability_known=known, dense_charged_bytes=dense, reserve_bytes=reserve_bytes,
                         derived_maximum_bytes=derived, reason=reason)


def validate_combination(policy, expert):
    """Fail closed on profiles whose contracts assume fixed precision."""
    if expert is None:
        return
    residency = expert.get('residency', RESIDENCY_SHARED)
    if residency != RESIDENCY_DYNAMIC:
        return
    blocked = [name for name, reference in DYNAMIC_REFUSALS if policy.get(name, reference) != reference]
    if policy.get('draft_model') is not None:
        blocked.append('draft_model')
    if policy.get('ast_prompt_lookup', False):
        blocked.append('ast_prompt_lookup')
    if policy.get('deepseek_verify_commit', 'replay') not in ('replay', None):
        blocked.append('deepseek_verify_commit')
    if blocked:
        raise ValueError('Source dynamic_precision_v1 cannot combine with: '+', '.join(blocked))
