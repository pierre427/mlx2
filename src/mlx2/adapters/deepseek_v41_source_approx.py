# SPDX-License-Identifier: MIT
"""Opt-in APPROXIMATE Source operations: policy, identity salt and refusals.

Two prototype operations live behind this module; both are default off,
labelled approximate and unqualified, and never publish to APCv2:

* ``deepseek_source_prefill_approx=yoco_v1`` -- long-prefill decoder skip,
  implemented in :mod:`deepseek_v41_source_yoco`.
* ``deepseek_source_route_bias=resident_v1|resident_guarded_v1`` -- decode
  expert selection biased toward cache-resident experts, implemented in
  :mod:`deepseek_v41_source_route_bias`.

Isolation contract (the same for both operations):

1. Selecting either one salts ``numerics_revision``, ``profile``,
   ``numeric_profile`` and therefore the model identity that every cache row
   carries. Rows produced under the salt cannot be consumed by an exact model
   (``validate_forward`` compares identities) and live in a separate APCv2
   namespace.
2. Rows that an approximate operation actually touched are additionally
   stamped with a private numerical profile (the cache's existing
   ``ast_private_profile`` publication condition), which structurally refuses
   checkpoint publication even if a route-level gate were bypassed.
3. The route refuses APCv2 reuse and publication outright
   (``execution_config['apcv2_reuse_disabled_reason']`` and
   ``apc_publication_refusal``). :data:`PUBLICATION_QUALIFIED` is the single
   switch a future qualification would have to flip; it is ``False`` and no
   code path here enables publication.

Composition (2026-11-09). Route bias, route drop and the reduced expert
tiers (``deepseek_source_experts.tiers``/``q1_on_miss``) may run together:

* One router chain per Block: native router -> bias re-selection (combine
  weights from the original scores) -> drop on the biased selection ->
  redirect dropped slots to the row's kept top expert. Each wrapper installs
  idempotently and the drop wrapper resolves the bias router at call time, so
  installation order does not matter and nothing runs twice.
* With reduced tiers on, the bias counts reduced-pool members as resident
  (``route_drop.resident_or_pooled``): an expert held in a Q1/Q2 pool costs no
  read. The tiers then serve the biased (and dropped) selection as usual.
* Identity: when two router operations or bias + tiers are selected, the
  salt binding gains a composition field, so a composed run never shares
  identity with either operation alone. Each operation alone hashes exactly
  the binding it always did. The reduced-tier salt chains after this one and
  both stamp one combined private profile (``q1_on_miss.combined_profile``).

Provenance: provenance/deepseek-v41-source-yoco-approx.json and
provenance/deepseek-v41-source-route-bias.json.
"""
import hashlib
import json
import math
from pathlib import Path

YOCO_KEY = 'deepseek_source_prefill_approx'
YOCO_LAYERS = 'deepseek_source_prefill_approx_layers'
YOCO_KEEP = 'deepseek_source_prefill_approx_keep_rows'
YOCO_MIN_ROWS = 'deepseek_source_prefill_approx_min_rows'
YOCO_CHOICES = ('off', 'yoco_v1')
YOCO_DEFAULTS = {YOCO_LAYERS: [21, 39], YOCO_KEEP: 140, YOCO_MIN_ROWS: 4096}

ROUTE_KEY = 'deepseek_source_route_bias'
ROUTE_LAMBDA = 'deepseek_source_route_bias_lambda'
ROUTE_ADAPTIVE = 'deepseek_source_route_bias_adaptive'
ROUTE_MARGIN = 'deepseek_source_route_bias_margin'
ROUTE_CHOICES = ('off', 'resident_v1', 'resident_guarded_v1')
ROUTE_ADAPTIVE_CHOICES = ('off', 'margin_v1')
ROUTE_DEFAULT_LAMBDA = 0.35
# Placeholder bounds: NOT measured on the checkpoint. The GPU script reports
# their effect; nothing here claims they are good values.
ROUTE_DEFAULT_MARGIN = {'minimum': 0.1, 'maximum': 0.35, 'hard_gap': 1.0, 'easy_gap': 4.0}
ROUTE_LAMBDA_MAXIMUM = 3.0

YOCO_PARAMETER_KEYS = (YOCO_LAYERS, YOCO_KEEP, YOCO_MIN_ROWS)
ROUTE_PARAMETER_KEYS = (ROUTE_LAMBDA, ROUTE_ADAPTIVE, ROUTE_MARGIN)

# Soft-reap probe (Pierre 2026-11-08): zero the lowest-weight routed expert on
# a layer range for B1 decode rows; see deepseek_v41_source_route_drop.
DROP_KEY = 'deepseek_source_route_drop'
DROP_LAYERS = 'deepseek_source_route_drop_layers'
DROP_CHOICES = ('off', 'drop_min_v1', 'drop_miss_v1')
# drop_miss_v1 (Pierre 2026-11-09): drop only selected experts that are NOT resident, never the
# top PROTECT ranks, and only below MAX_WEIGHT of the row; the span count caps drops per row.
DROP_PROTECT = 'deepseek_source_route_drop_protect'
DROP_MAX_WEIGHT = 'deepseek_source_route_drop_max_weight'
DROP_COUNT = 'deepseek_source_route_drop_count'
DROP_DEFAULT_LAYERS = [13, 26]    # middle third of 40 layers; not a measured choice
# Per-span counts (Pierre 2026-11-08: un-ablate specific layers after damage).
# A list of {"layers": [first, last] | layer, "count": 0..5}; later entries
# override earlier ones layer by layer; count 0 restores native routing.
DROP_SPANS = 'deepseek_source_route_drop_spans'
DROP_PARAMETER_KEYS = (DROP_LAYERS, DROP_COUNT, DROP_SPANS, DROP_PROTECT, DROP_MAX_WEIGHT)
POLICY_KEYS = frozenset((YOCO_KEY, ROUTE_KEY, DROP_KEY) + YOCO_PARAMETER_KEYS + ROUTE_PARAMETER_KEYS
                        + DROP_PARAMETER_KEYS)

# The one switch a future qualification would flip. Nothing sets it.
PUBLICATION_QUALIFIED = False
QUALIFICATION = 'unqualified'
APC_REASON_PREFIX = 'approximate_source_operation_unqualified:'
PRIVATE_PROFILE_PREFIX = 'mlx2.ds41.source-approx.v1:'
# Bound into the salt only when operations compose (alone: absent, binding unchanged).
ROUTE_DROP_COMPOSITION = 'native>route_bias>route_drop>redirect.v1'
ROUTE_TIERS_RESIDENCY = 'original_cache+leased+reduced_pools.v1'

# Forward/state laws whose composition with an approximate operation has not
# been implemented. Each (key, reference value) must hold.
_COMMON_REFUSALS = (
    ('deepseek_physical_batch', 'b1'), ('deepseek_vision', 'reference'),
    ('deepseek_source_dspark_ast', 'off'), ('deepseek_source_dspark_prepare', 'off'),
    ('deepseek_source_dspark_target_forward', 'reference'),
    ('deepseek_source_ast_kv_preparation', 'reference'),
    ('deepseek_source_resident_gather', 'reference'), ('deepseek_source_resident_prefill', 'reference'),
)
YOCO_REFUSALS = _COMMON_REFUSALS + (
    ('deepseek_ced', 'reference'), ('deepseek_kv_storage', 'reference'),
    ('deepseek_candidate_score', 'reference'),
    ('deepseek_source_prefill_partition', 'adaptive_v1'), ('deepseek_source_prefill_order', 'reference'),
    ('deepseek_source_frame_reuse', 'reference'), ('deepseek_source_row_packet', 'reference'),
)
ROUTE_REFUSALS = _COMMON_REFUSALS
DROP_REFUSALS = _COMMON_REFUSALS


def _plain_int(value):
    return type(value) is int


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def split_policy(value):
    """Pop every approximate-operation key out of a raw policy (in place)."""
    return {key: value.pop(key) for key in list(value) if key in POLICY_KEYS}


def parse_layers(value):
    if isinstance(value, str):
        parts = value.split('-')
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            raise ValueError(YOCO_LAYERS + ' must be "first-last" or [first, last]')
        value = [int(parts[0]), int(parts[1])]
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(_plain_int(v) for v in value) or not 1 <= value[0] <= value[1] <= 1023):
        raise ValueError(YOCO_LAYERS + ' must be two integer layer IDs 1 <= first <= last')
    return [value[0], value[1]]


def _validate_yoco(raw):
    mode = raw.get(YOCO_KEY, 'off')
    if type(mode) is not str or mode not in YOCO_CHOICES:
        raise ValueError('invalid Source policy: ' + YOCO_KEY)
    present = [k for k in YOCO_PARAMETER_KEYS if k in raw]
    if mode == 'off':
        if present:
            raise ValueError(', '.join(present) + ' require ' + YOCO_KEY + '=yoco_v1')
        return {}
    layers = parse_layers(raw.get(YOCO_LAYERS, YOCO_DEFAULTS[YOCO_LAYERS]))
    keep = raw.get(YOCO_KEEP, YOCO_DEFAULTS[YOCO_KEEP])
    if not _plain_int(keep) or not 1 <= keep <= 1024:
        raise ValueError(YOCO_KEEP + ' must be an integer 1..1024 (one ordinary row chunk)')
    minimum = raw.get(YOCO_MIN_ROWS, YOCO_DEFAULTS[YOCO_MIN_ROWS])
    if not _plain_int(minimum) or minimum < 2048 or minimum <= 2 * keep:
        raise ValueError(YOCO_MIN_ROWS + ' must be an integer >= 2048 and > 2 x keep rows')
    return {YOCO_KEY: mode, YOCO_LAYERS: layers, YOCO_KEEP: keep, YOCO_MIN_ROWS: minimum}


def _validate_margin(value):
    if not isinstance(value, dict) or set(value) != set(ROUTE_DEFAULT_MARGIN):
        raise ValueError(ROUTE_MARGIN + ' requires exactly minimum, maximum, hard_gap, easy_gap')
    if not all(_finite_number(v) and type(v) is not bool for v in value.values()):
        raise ValueError(ROUTE_MARGIN + ' values must be finite numbers')
    low, high = float(value['minimum']), float(value['maximum'])
    hard, easy = float(value['hard_gap']), float(value['easy_gap'])
    if not 0.0 <= low <= high <= ROUTE_LAMBDA_MAXIMUM or not 0.0 <= hard < easy:
        raise ValueError(ROUTE_MARGIN + ' requires 0 <= minimum <= maximum <= 3 and 0 <= hard_gap < easy_gap')
    return {'minimum': low, 'maximum': high, 'hard_gap': hard, 'easy_gap': easy}


def _validate_route(raw):
    mode = raw.get(ROUTE_KEY, 'off')
    if type(mode) is not str or mode not in ROUTE_CHOICES:
        raise ValueError('invalid Source policy: ' + ROUTE_KEY)
    present = [k for k in ROUTE_PARAMETER_KEYS if k in raw]
    if mode == 'off':
        if present:
            raise ValueError(', '.join(present) + ' require an enabled ' + ROUTE_KEY)
        return {}
    adaptive = raw.get(ROUTE_ADAPTIVE, 'off')
    if type(adaptive) is not str or adaptive not in ROUTE_ADAPTIVE_CHOICES:
        raise ValueError('invalid Source policy: ' + ROUTE_ADAPTIVE)
    result = {ROUTE_KEY: mode, ROUTE_ADAPTIVE: adaptive}
    if adaptive == 'off':
        if ROUTE_MARGIN in raw:
            raise ValueError(ROUTE_MARGIN + ' requires ' + ROUTE_ADAPTIVE + '=margin_v1')
        lam = raw.get(ROUTE_LAMBDA, ROUTE_DEFAULT_LAMBDA)
        if type(lam) is bool or not _finite_number(lam) or not 0.0 < lam <= ROUTE_LAMBDA_MAXIMUM:
            raise ValueError(ROUTE_LAMBDA + ' must be a finite number in (0, 3]')
        result[ROUTE_LAMBDA] = float(lam)
    else:
        if ROUTE_LAMBDA in raw:
            raise ValueError(ROUTE_LAMBDA + ' is fixed-mode only; margin_v1 uses ' + ROUTE_MARGIN)
        result[ROUTE_MARGIN] = _validate_margin(raw.get(ROUTE_MARGIN, dict(ROUTE_DEFAULT_MARGIN)))
    return result


def resolve_drop_spans(spans):
    """Validate per-span drop counts; return canonical maximal runs [[first, last, count>0], ...]."""
    if not isinstance(spans, (list, tuple)) or not spans:
        raise ValueError(DROP_SPANS + ' must be a non-empty list of {"layers": ..., "count": ...}')
    counts = {}
    for entry in spans:
        if not isinstance(entry, dict) or set(entry) != {'layers', 'count'}:
            raise ValueError(DROP_SPANS + ' entries need exactly "layers" and "count"')
        layers, count = entry['layers'], entry['count']
        if _plain_int(layers):
            layers = [layers, layers]
        if (not isinstance(layers, (list, tuple)) or len(layers) != 2 or not all(_plain_int(v) for v in layers)
                or not 0 <= layers[0] <= layers[1] <= 1023):
            raise ValueError(DROP_SPANS + ' layers must be a layer ID or [first, last] with 0 <= first <= last')
        if not _plain_int(count) or not 0 <= count <= 5:
            raise ValueError(DROP_SPANS + ' count must be an integer 0..5')
        for layer in range(layers[0], layers[1] + 1):
            counts[layer] = count
    runs = []
    for layer in sorted(k for k, v in counts.items() if v):
        if runs and runs[-1][1] == layer - 1 and runs[-1][2] == counts[layer]:
            runs[-1][1] = layer
        else:
            runs.append([layer, layer, counts[layer]])
    if not runs:
        raise ValueError(DROP_SPANS + ' drops nothing; turn ' + DROP_KEY + ' off instead')
    return runs


def drop_counts(policy):
    """Per-layer drop counts {layer: count} for either policy form."""
    if DROP_SPANS in policy:
        return {layer: c for first, last, c in policy[DROP_SPANS] for layer in range(first, last + 1)}
    first, last = policy[DROP_LAYERS]
    return {layer: policy.get(DROP_COUNT, 1) for layer in range(first, last + 1)}


def _validate_drop(raw):
    mode = raw.get(DROP_KEY, 'off')
    if type(mode) is not str or mode not in DROP_CHOICES:
        raise ValueError('invalid Source policy: ' + DROP_KEY)
    if mode == 'off':
        present = [k for k in DROP_PARAMETER_KEYS if k in raw]
        if present:
            raise ValueError(', '.join(present) + ' require ' + DROP_KEY + '=drop_min_v1|drop_miss_v1')
        return {}
    extra = {}
    if mode == 'drop_miss_v1':
        protect = raw.get(DROP_PROTECT, 2)
        maxw = raw.get(DROP_MAX_WEIGHT, 0.10)
        if not _plain_int(protect) or not 1 <= protect <= 5:
            raise ValueError(DROP_PROTECT + ' must be an integer 1..5 (top ranks never dropped)')
        if type(maxw) is bool or not _finite_number(maxw) or not 0.0 < maxw <= 1.0:
            raise ValueError(DROP_MAX_WEIGHT + ' must be a number in (0, 1]')
        extra = {DROP_PROTECT: protect, DROP_MAX_WEIGHT: float(maxw)}
    elif DROP_PROTECT in raw or DROP_MAX_WEIGHT in raw:
        raise ValueError(DROP_PROTECT + '/' + DROP_MAX_WEIGHT + ' require ' + DROP_KEY + '=drop_miss_v1')
    if DROP_SPANS in raw:
        if DROP_LAYERS in raw or DROP_COUNT in raw:
            raise ValueError(DROP_SPANS + ' replaces ' + DROP_LAYERS + '/' + DROP_COUNT + '; give one form')
        return {DROP_KEY: mode, DROP_SPANS: resolve_drop_spans(raw[DROP_SPANS]), **extra}
    value = raw.get(DROP_LAYERS, DROP_DEFAULT_LAYERS)
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_plain_int(v) for v in value)
            or not 0 <= value[0] <= value[1] <= 1023):
        raise ValueError(DROP_LAYERS + ' must be two integer layer IDs 0 <= first <= last')
    count = raw.get(DROP_COUNT, 1)
    if not _plain_int(count) or not 1 <= count <= 5:
        raise ValueError(DROP_COUNT + ' must be an integer 1..5 (lowest-weight experts zeroed per row; 5 keeps only the top expert)')
    return {DROP_KEY: mode, DROP_LAYERS: [value[0], value[1]], DROP_COUNT: count, **extra}


def _blocked(policy, refusals):
    blocked = [name for name, reference in refusals if policy.get(name, reference) != reference]
    if policy.get('draft_model') not in (None, 'none'):
        blocked.append('draft_model')
    if policy.get('ast_prompt_lookup', False):
        blocked.append('ast_prompt_lookup')
    expert = policy.get('deepseek_source_experts') or {}
    if expert.get('residency', 'shared_v1') == 'dynamic_precision_v1':
        blocked.append('deepseek_source_experts.residency=dynamic_precision_v1')
    return blocked


def validate_policy(raw, policy):
    """Validate split approximate keys against the parsed Source policy.

    Returns only the keys to merge; an all-off request returns ``{}`` so the
    default policy is byte-for-byte unchanged.
    """
    if not isinstance(raw, dict) or set(raw) - POLICY_KEYS:
        raise ValueError('unknown approximate Source policy key')
    result = {}
    yoco = _validate_yoco(raw)
    if yoco:
        blocked = _blocked(policy, YOCO_REFUSALS)
        if blocked:
            raise ValueError('Source approximate yoco_v1 cannot combine with: ' + ', '.join(blocked))
        result.update(yoco)
    route = _validate_route(raw)
    if route:
        blocked = _blocked(policy, ROUTE_REFUSALS)
        if policy.get('deepseek_source_experts') is None:
            blocked.append('deepseek_source_experts(required: residency is read from the expert store)')
        if blocked:
            raise ValueError('Source approximate route bias cannot combine with: ' + ', '.join(blocked))
        result.update(route)
    drop = _validate_drop(raw)
    if drop:
        # Route bias composes (2026-11-09): one router chain, bias then drop.
        blocked = _blocked(policy, DROP_REFUSALS)
        if blocked:
            raise ValueError('Source approximate route drop cannot combine with: ' + ', '.join(blocked))
        result.update(drop)
    return result


def selected_operations(policy):
    ops = []
    if policy.get(YOCO_KEY, 'off') != 'off':
        ops.append(policy[YOCO_KEY])
    if policy.get(ROUTE_KEY, 'off') != 'off':
        ops.append('route_' + policy[ROUTE_KEY])
    if policy.get(DROP_KEY, 'off') != 'off':
        ops.append('route_' + policy[DROP_KEY])
    return tuple(ops)


def _file_identity(names):
    root = Path(__file__).resolve().parents[1]
    return hashlib.sha256(''.join(name + ':' + hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in names).encode()).hexdigest()


def _reduced_tiers(policy):
    from .deepseek_v41_source_q1_on_miss import enabled
    return enabled(policy)


def implementation_identity(policy):
    names = ['adapters/deepseek_v41_source_approx.py', 'runtime/models/deepseek_v41.py',
             'runtime/models/deepseek_v41_cache.py']
    if policy.get(YOCO_KEY, 'off') != 'off':
        names += ['adapters/deepseek_v41_source_yoco.py', 'runtime/models/deepseek_v41_ced.py']
    if policy.get(ROUTE_KEY, 'off') != 'off':
        names += ['adapters/deepseek_v41_source_route_bias.py', 'runtime/models/deepseek_v4.py',
                  'runtime/models/deepseek_v41_fused.py']
    if policy.get(DROP_KEY, 'off') != 'off':
        names += ['adapters/deepseek_v41_source_route_drop.py', 'runtime/models/deepseek_v4.py',
                  'runtime/models/deepseek_v41_fused.py']
    elif policy.get(ROUTE_KEY, 'off') != 'off' and _reduced_tiers(policy):
        names += ['adapters/deepseek_v41_source_route_drop.py']   # bias reads resident_or_pooled
    # De-duplicate in order; an operation alone has no duplicates, so its hash is unchanged.
    return _file_identity(list(dict.fromkeys(names)))


def composition(policy):
    """Composition fields bound into the salt; ``{}`` for any operation alone."""
    value = {}
    route = policy.get(ROUTE_KEY, 'off') != 'off'
    if route and policy.get(DROP_KEY, 'off') != 'off':
        value['router_chain'] = ROUTE_DROP_COMPOSITION
    if route and _reduced_tiers(policy):
        value['route_bias_residency'] = ROUTE_TIERS_RESIDENCY
    return value


def identity_salt(policy):
    """Deterministic salt over every selected operation parameter + code."""
    ops = selected_operations(policy)
    if not ops:
        return None
    binding = {key: policy[key] for key in sorted(POLICY_KEYS) if key in policy}
    binding['implementation'] = implementation_identity(policy)
    binding['qualification'] = QUALIFICATION
    composed = composition(policy)
    if composed:
        binding['composition'] = composed
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def private_profile(policy):
    salt = identity_salt(policy)
    return None if salt is None else PRIVATE_PROFILE_PREFIX + '+'.join(selected_operations(policy)) + ':' + salt[:32]


def apc_refusal_reason(policy):
    ops = selected_operations(policy)
    if not ops or PUBLICATION_QUALIFIED:
        # PUBLICATION_QUALIFIED is False; this branch only returns None when
        # no approximate operation is selected.
        return None
    return APC_REASON_PREFIX + '+'.join(ops)


def bind_identity(adapter):
    """Salt the adapter's numerical identity; a no-op when all operations are off."""
    policy = adapter.kernel_policy
    salt = identity_salt(policy)
    if salt is None:
        return None
    ops = '+'.join(selected_operations(policy))
    adapter.numerics_revision += ':source_approx=' + ops + ':salt=' + salt
    adapter.profile += ':approx-' + salt[:16]
    adapter.numeric_profile += ':' + ops + '_approximate_unqualified'
    return salt


def stamp_cache(rows, profile):
    """Mark rows an approximate operation touched; refuses publication."""
    for row in rows:
        current = getattr(row, 'ast_private_profile', None)
        if current not in (None, profile):
            raise ValueError('cache row already carries another private numerical profile')
        row.ast_private_profile = profile


def receipt(policy, model=None):
    ops = selected_operations(policy)
    value = dict(selected=list(ops), approximate=bool(ops), qualification=QUALIFICATION,
                 default_selected=False, publication_qualified=PUBLICATION_QUALIFIED,
                 apcv2_publication=('refused: ' + apc_refusal_reason(policy)) if ops else 'not affected',
                 identity_salt=identity_salt(policy), private_cache_profile=private_profile(policy))
    if composition(policy):
        value['composition'] = composition(policy)
    combined = getattr(model, 'source_combined_private_profile', None)
    if ops and combined is not None:
        value['combined_private_cache_profile'] = combined
        value['combined_identity_salt'] = getattr(model, 'source_combined_identity_salt', None)
    if policy.get(YOCO_KEY, 'off') != 'off':
        value['yoco'] = dict(policy=policy[YOCO_KEY], layers=policy[YOCO_LAYERS],
                             keep_rows=policy[YOCO_KEEP], min_prompt_rows=policy[YOCO_MIN_ROWS],
                             last_wave=getattr(model, 'source_prefill_approx_last', None),
                             approximated_state='decoder local sliding-window KV and decoder hidden rows before the retained tail',
                             qualification=QUALIFICATION)
    if policy.get(ROUTE_KEY, 'off') != 'off':
        bias = getattr(model, 'source_route_bias', None)
        value['route_bias'] = dict(policy=policy[ROUTE_KEY], adaptive=policy[ROUTE_ADAPTIVE],
                                   **({'lambda': policy[ROUTE_LAMBDA]} if ROUTE_LAMBDA in policy else {}),
                                   **({'margin': policy[ROUTE_MARGIN], 'margin_values_measured': False}
                                      if ROUTE_MARGIN in policy else {}),
                                   scope='B1 decode rows only; selection only; combine weights are original probabilities',
                                   residency=('original cache, lease or reduced pool' if _reduced_tiers(policy)
                                              else 'original cache or lease'),
                                   stats=None if bias is None else bias.receipt(),
                                   qualification=QUALIFICATION)
    if policy.get(DROP_KEY, 'off') != 'off':
        drop = getattr(model, 'source_route_drop', None)
        form = ({'spans': policy[DROP_SPANS]} if DROP_SPANS in policy
                else {'layers': policy[DROP_LAYERS], 'count': policy[DROP_COUNT]})
        value['route_drop'] = dict(policy=policy[DROP_KEY], **form,
                                   stats=None if drop is None else drop.receipt(),
                                   qualification=QUALIFICATION)
    return value


def approximate_model_class(base, policy):
    """Adapter-owned subclass of the selected model type, or ``base`` when off."""
    yoco = policy.get(YOCO_KEY, 'off') != 'off'
    route = policy.get(ROUTE_KEY, 'off') != 'off'
    drop = policy.get(DROP_KEY, 'off') != 'off'
    if not (yoco or route or drop):
        return base

    class ApproximateSourceModel(base):
        source_prefill_approx_enabled = yoco
        source_route_bias_enabled = route
        source_route_drop_enabled = drop

        def _forward(self, inputs, cache=None, **options):
            bias = getattr(self, 'source_route_bias', None)
            if bias is not None:
                from .deepseek_v41_source_route_bias import install_router_wrappers
                install_router_wrappers(self)
            reap = getattr(self, 'source_route_drop', None)
            if reap is not None:
                from .deepseek_v41_source_route_drop import install_router_wrappers as install_drop
                install_drop(self)
            if getattr(self, 'source_prefill_approx_plan', None) is not None:
                from .deepseek_v41_source_yoco import yoco_forward
                result = yoco_forward(self, super()._forward, inputs, cache, **options)
            else:
                result = super()._forward(inputs, cache, **options)
            if bias is not None:
                bias.after_forward(self, inputs, cache, result, options)
            if reap is not None:
                reap.after_forward(self, cache)
            return result

    return ApproximateSourceModel


def configure_model(adapter):
    """Attach the selected operations to the constructed adapter model."""
    policy, model = adapter.kernel_policy, adapter.model
    if not selected_operations(policy):
        return
    model.source_approx_private_profile = private_profile(policy)
    if policy.get(YOCO_KEY, 'off') != 'off':
        from .deepseek_v41_source_yoco import configure_yoco
        configure_yoco(model, policy)
    if policy.get(ROUTE_KEY, 'off') != 'off':
        from .deepseek_v41_source_route_bias import ResidentRouteBias
        model.source_route_bias = ResidentRouteBias.from_policy(policy, adapter.weight_stream, model.args)
    if policy.get(DROP_KEY, 'off') != 'off':
        from .deepseek_v41_source_route_drop import RankDrop
        model.source_route_drop = RankDrop.from_policy(policy, store=adapter.weight_stream)
