# SPDX-License-Identifier: MIT
"""Opt-in APPROXIMATE Source decode: reduced-precision expert tiers.

Two policy keys of ``deepseek_source_experts`` (``mode: original_lfu``, an
absolute top-level ``q1_root``), mutually exclusive, one engine:

``q1_on_miss`` -- weight rule, no map::

    "q1_on_miss": {"max_weight": 0.10, "layers": [0, 39],
                   "pool_bytes": 15000000000, "overrides": [...]}

``tiers`` -- frozen precision map (q4 = original MXFP4, q2, q1)::

    "tiers": {"usage_map": "/abs/usage.json", "q2_root": "/abs/Q2-g32h",
              "q2_format": "g32h", "q2_bytes": 15000000000, "q1_bytes": 15000000000,
              "max_weight": 1.0, "layers": [0, 39], "overrides": [...]}

Q2 format. ``q2_format`` (default ``"g64"``, only with ``q2_root``) names the
bank format in the policy so budgets resolve before any bank is opened (the
per-expert payload sizes ``q2_experts`` overrides and the q2 band). At load
the bank's manifest must match the named entry of
``deepseek_v41_source_tiers.Q2_FORMATS`` exactly (quantization revision,
group size, bits, rotation, per-expert payload); anything else is refused:

* ``g64``  -- ``source-q2-mlx-affine-g64-f32q-f16-v2``, affine 2-bit g64,
  no rotation, 11,059,200 B/expert.
* ``g32h`` -- ``source-q2-mlx-affine-g32-had256-mseclip-f16-v3``, affine
  2-bit g32 over W' = W H (orthonormal Hadamard over 256-wide blocks of the
  input dim), 13,271,040 B/expert; the runtime multiplies by H x.

The format, revision and manifest fingerprint are bound into the identity salt.

Budgets. ``budget_bytes`` stays the TOTAL expert cache the server admits.
The reduced pools (Q1 ``pool_bytes``/``q1_bytes``, Q2 ``q2_bytes``) are
carved out of it; the original cache maximum is the remainder. Each reduced
pool is split per layer (default even split; overrides per layer or span,
bytes or expert counts, later entries win -- see deepseek_v41_source_tiers)
and evicts LRU within its layer only. Under live-memory pressure the
adaptive target is sized over the total (pool bytes counted reclaimable) and
every pool's per-layer targets scale with the original share, so the pools
shrink with the original cache, never on top of it.

Decode rule (B1 single-row MoE calls in ``layers``; every multi-row call,
i.e. prefill, and every layer outside the range is exact original MXFP4).
Let w be the selected expert's combine weight renormalised over the top-k.

1. Resident original (cached, already leased this window, or a prefill
   transient) -> exact original hit, whatever its tier.
2. ``q1_on_miss``: w < max_weight -> Q1; otherwise an ordinary original miss.
   ``tiers``: the map's tier. q4 -> ordinary original miss (LFU insert).
   q2/q1 with w < max_weight -> that tier; q2/q1 with w >= max_weight ->
   original read *transiently* (leased for the call, not cached): a guard for
   a heavy selection without letting non-q4 experts churn the q4 pool.
3. A reduced tier looks in its per-layer pool (LRU hit); a pool miss reads the
   sealed bank and inserts (LRU eviction inside that layer only); with no
   pool, or a zero layer budget, the object is served uncached.

A reduced copy whose original later becomes resident stays until its layer's
LRU ages it out; it is never served while the original is resident (rule 1).

Approximate contract (as deepseek_v41_source_approx.py, separate module):
the salt binds settings, bank fingerprints, the usage-map digest and the
resolved layout identity (so two override sets never share identity); it is
appended to numerics_revision/profile/numeric_profile and the store
apc_revision; forwards that served a reduced tier stamp cache rows with a
private profile; APCv2 reuse and publication are refused.

Known limit: a forward with exactly one row is treated as decode (the same
scope test as ``deepseek_source_route_bias``).

Status: implemented_untested_unqualified (CPU tests only). Provenance:
provenance/deepseek-v41-source-q1-on-miss.json (original code, no mining).
"""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path

from . import deepseek_v41_source_tiers as tiers_mod

KEY = 'q1_on_miss'
TIERS_KEY = 'tiers'
KEYS = (KEY, TIERS_KEY)
NAME = 'q1_on_miss_v1'
TIERS_NAME = 'reduced_tiers_v1'
Q1_FIELDS = frozenset(('max_weight', 'layers', 'pool_bytes', 'overrides'))
TIER_FIELDS = frozenset(('max_weight', 'layers', 'usage_map', 'q2_root', 'q2_format', 'q2_bytes', 'q1_bytes',
                         'overrides'))
QUALIFICATION = 'unqualified'
STATUS = 'implemented_untested_unqualified'
PUBLICATION_QUALIFIED = False
APC_REASON = 'approximate_source_operation_unqualified:' + NAME
TIERS_APC_REASON = 'approximate_source_operation_unqualified:' + TIERS_NAME
PRIVATE_PROFILE_PREFIX = 'mlx2.ds41.source-q1-on-miss.v1:'
Q2_FORMATS = tiers_mod.Q2_FORMATS
Q2_REVISION = Q2_FORMATS['g64']['revision']      # the default format's revision
Q2_BYTES = tiers_mod.Q2_BYTES                    # the default format's payload

# Forward/state laws not composed with a per-call substitution. Each
# (key, reference value) must hold; mirrors the approximate-route refusals
# plus the expert-I/O paths that read originals ahead of the rule.
REFUSALS = (
    ('deepseek_physical_batch', 'b1'), ('deepseek_vision', 'reference'),
    ('deepseek_source_dspark_ast', 'off'), ('deepseek_source_dspark_prepare', 'off'),
    ('deepseek_source_dspark_target_forward', 'reference'),
    ('deepseek_source_ast_kv_preparation', 'reference'),
    ('deepseek_source_resident_gather', 'reference'), ('deepseek_source_resident_prefill', 'reference'),
    ('deepseek_source_expert_io', 'reference'), ('deepseek_source_prefill_readahead', 'reference'),
    ('deepseek_source_row_packet', 'reference'),
    ('deepseek_source_prefill_approx', 'off'),
)
# deepseek_source_route_drop composes (Pierre 2026-11-08): a dropped slot arrives as a
# zero-weight duplicate of the row's top expert (route_drop.redirect_dropped); arm()
# collapses it, so tiers rank only the kept experts and the dropped one is never read.
# deepseek_source_route_bias composes (2026-11-09): the bias counts this rule's pool members
# as resident (route_drop.resident_or_pooled) and only changes the selection; arm() then sees
# an ordinary six-expert route with original-score combine weights.


def _abs(value):
    return isinstance(value, str) and Path(value).is_absolute()


def _weight(value, default=None):
    if value is None:
        return default
    if type(value) not in (int, float) or type(value) is bool or not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError('max_weight must be a finite number in (0, 1]')
    return float(value)


def _layers(value):
    layers = [0, 1023] if value is None else value
    if (not isinstance(layers, (list, tuple)) or len(layers) != 2
            or not all(type(v) is int for v in layers) or not 0 <= layers[0] <= layers[1] <= 1023):
        raise ValueError('layers must be two integer layer IDs 0 <= first <= last')
    return [layers[0], layers[1]]


def _bytes(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a non-negative integer byte count')
    return value


def parse(value):
    """Normalise a ``q1_on_miss`` object; raises on anything unexpected."""
    if not isinstance(value, dict) or set(value) - Q1_FIELDS or 'max_weight' not in value:
        raise ValueError('q1_on_miss must be {"max_weight": float, "layers": [first, last], "pool_bytes": int}')
    try:
        result = {'max_weight': _weight(value['max_weight']), 'layers': _layers(value.get('layers'))}
    except ValueError as error:
        raise ValueError('q1_on_miss.' + str(error)) from None
    if 'pool_bytes' in value:
        result['pool_bytes'] = _bytes('q1_on_miss.pool_bytes', value['pool_bytes'])
    if 'overrides' in value:
        if 'pool_bytes' not in value:
            raise ValueError('q1_on_miss.overrides require pool_bytes')
        overrides = tiers_mod.validate_overrides(value['overrides'], tiers=('q4', 'q1'))
        if any(set(e) - {'layers', 'q1_bytes', 'q1_experts'} for e in overrides):
            raise ValueError('q1_on_miss.overrides accept only q1_bytes/q1_experts budget entries')
        result['overrides'] = overrides
    return result


def parse_tiers(value):
    if not isinstance(value, dict) or set(value) - TIER_FIELDS or 'usage_map' not in value:
        raise ValueError('tiers must name an absolute usage_map; keys: ' + ', '.join(sorted(TIER_FIELDS)))
    if not _abs(value['usage_map']):
        raise ValueError('tiers.usage_map must be absolute')
    result = {'usage_map': value['usage_map'], 'max_weight': _weight(value.get('max_weight'), 1.0),
              'layers': _layers(value.get('layers')), 'q1_bytes': _bytes('tiers.q1_bytes', value.get('q1_bytes', 0))}
    if 'q2_root' in value:
        if not _abs(value['q2_root']):
            raise ValueError('tiers.q2_root must be absolute')
        result['q2_root'] = value['q2_root']
        fmt = value.get('q2_format', tiers_mod.DEFAULT_Q2_FORMAT)
        try:
            tiers_mod.q2_format(fmt)
        except ValueError as error:
            raise ValueError('tiers.' + str(error)) from None
        result['q2_format'] = fmt
        result['q2_bytes'] = _bytes('tiers.q2_bytes', value.get('q2_bytes', 0))
    elif 'q2_bytes' in value or 'q2_format' in value:
        raise ValueError('tiers.q2_bytes and tiers.q2_format require tiers.q2_root')
    configured = ('q4', 'q2', 'q1') if 'q2_root' in result else ('q4', 'q1')
    result['overrides'] = tiers_mod.validate_overrides(value.get('overrides'), tiers=configured)
    return result


def configured_tiers(settings, mode):
    if mode == KEY:
        return ('q4', 'q1')
    return ('q4', 'q2', 'q1') if 'q2_root' in settings else ('q4', 'q1')


def totals(settings, mode):
    if mode == KEY:
        return {'q1': settings.get('pool_bytes', 0)}
    value = {'q1': settings['q1_bytes']}
    if 'q2_root' in settings:
        value['q2'] = settings['q2_bytes']
    return value


def q2_spec(settings):
    """The configured Q2 format spec (name included), or None without a Q2 bank."""
    if 'q2_root' not in settings:
        return None
    name = settings.get('q2_format', tiers_mod.DEFAULT_Q2_FORMAT)
    return dict(tiers_mod.q2_format(name), format=name)


def resolve_layout(settings, mode, *, budget_bytes, usage=None, layers=40, experts=384):
    """Per-layer budgets (+ map for tiers) through the one pure resolver."""
    spec = q2_spec(settings) if mode == TIERS_KEY else None
    return tiers_mod.resolve(budget_bytes=budget_bytes, totals=totals(settings, mode), usage=usage,
                             overrides=settings.get('overrides'), layers=layers, experts=experts,
                             tiers=configured_tiers(settings, mode),
                             q2_bytes=None if spec is None else spec['payload_bytes'])


def validate_expert_policy(result):
    """In-place check inside a parsed ``deepseek_source_experts`` policy."""
    present = [key for key in KEYS if key in result]
    if not present:
        return result
    if len(present) > 1:
        raise ValueError('q1_on_miss and tiers are mutually exclusive')
    mode = present[0]
    if result.get('mode') != 'original_lfu':
        raise ValueError(mode + ' requires mode original_lfu (hot_original_q1 already serves Q1)')
    if not _abs(result.get('q1_root')):
        raise ValueError(mode + ' requires an absolute q1_root')
    if result.get('direct_reads', False):
        raise ValueError(mode + ' cannot combine with direct_reads')
    settings = result[mode] = parse(result[mode]) if mode == KEY else parse_tiers(result[mode])
    from . import deepseek_v41_source_lfu as lfu
    carved = sum(totals(settings, mode).values())
    if mode == KEY and 'pool_bytes' in settings and not lfu.Q1_BYTES <= settings['pool_bytes']:
        raise ValueError('q1_on_miss.pool_bytes must hold one Q1 expert')
    if carved >= result['budget_bytes'] - 2 * lfu.ORIGINAL_BYTES:
        raise ValueError(mode + ' reduced pools must leave budget_bytes room for one original expert plus staging')
    # Budgets only (the usage map is read at load): fails closed on overflow.
    # Selection entries need the map, so they are checked structurally here
    # (validate_overrides in parse_tiers) and resolved in full at load.
    budgets_only = dict(settings, overrides=[entry for entry in settings.get('overrides') or []
                                             if tiers_mod._entry_kind(entry) != 'selection'])
    resolve_layout(budgets_only, mode, budget_bytes=result['budget_bytes'])
    return result


def mode_of(policy):
    expert = policy.get('deepseek_source_experts') if isinstance(policy, dict) else None
    if not isinstance(expert, dict):
        return None
    return next((key for key in KEYS if key in expert), None)


def enabled(policy):
    return mode_of(policy) is not None


def validate_combination(policy):
    """Fail closed on route/state laws this rule has not been composed with."""
    mode = mode_of(policy)
    if mode is None:
        return
    blocked = [name for name, reference in REFUSALS if policy.get(name, reference) != reference]
    if policy.get('draft_model') not in (None, 'none'):
        blocked.append('draft_model')
    if policy.get('ast_prompt_lookup', False):
        blocked.append('ast_prompt_lookup')
    if policy.get('deepseek_verify_rows', 'serial') not in ('serial', 'reference'):
        blocked.append('deepseek_verify_rows')
    if blocked:
        raise ValueError('Source ' + mode + ' cannot combine with: ' + ', '.join(blocked))


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    names = ('adapters/deepseek_v41_source_q1_on_miss.py', 'adapters/deepseek_v41_source_tiers.py',
             'adapters/deepseek_v41_source_lfu.py', 'runtime/gguf/source_q1.py',
             # Q2 reference law (affine 2-bit matmul, rotation, middle, rounding).
             'adapters/deepseek_v41_source_weights.py')
    return hashlib.sha256(''.join(name + ':' + hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in names).encode()).hexdigest()


def identity_salt(settings, bank_binding, *, mode=KEY, banks=None, layout_identity=None, usage_binding=None,
                  bank_formats=None):
    binding = dict(operation=NAME if mode == KEY else TIERS_NAME, settings=settings, q1_bank=bank_binding,
                   implementation=implementation_identity(), qualification=QUALIFICATION)
    if banks:
        binding['banks'] = banks
    if bank_formats:
        binding['bank_formats'] = bank_formats
    if layout_identity is not None:
        binding['layout'] = layout_identity
    if usage_binding is not None:
        binding['usage_map'] = usage_binding
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def private_profile(salt):
    return PRIVATE_PROFILE_PREFIX + salt[:32]


def store_rule(adapter):
    stream = getattr(adapter, 'weight_stream', None)
    return getattr(stream, 'q1_on_miss', None)


def bind_identity(adapter):
    """Salt the adapter's numerical identity; no-op when the rule is off."""
    mode = mode_of(adapter.kernel_policy)
    if mode is None:
        return None
    rule = store_rule(adapter)
    if rule is None:
        raise ValueError(mode + ' selected but the expert store did not bind it')
    adapter.numerics_revision += ':source_q1_on_miss=' + rule.name + ':salt=' + rule.salt
    adapter.profile += ':q1miss-' + rule.salt[:16]
    adapter.numeric_profile += ':' + rule.name + '_approximate_unqualified'
    return rule.salt


def apc_refusal_reason(policy):
    mode = mode_of(policy)
    if mode is None or PUBLICATION_QUALIFIED:
        return None
    return APC_REASON if mode == KEY else TIERS_APC_REASON


def model_class(base, policy):
    """Stamp cache rows of forwards that served a reduced tier; ``base`` when off."""
    if not enabled(policy):
        return base

    class Q1OnMissSourceModel(base):
        source_q1_on_miss_enabled = True

        def _forward(self, inputs, cache=None, **options):
            rule = getattr(self, 'source_q1_on_miss', None)
            before = rule.stats['q1_served'] if rule is not None else 0
            result = super()._forward(inputs, cache, **options)
            if rule is not None and rule.stats['q1_served'] != before:
                rule.stats['stamped_forwards'] += 1
                if cache is not None:
                    from .deepseek_v41_source_approx import stamp_cache
                    stamp_cache(cache, getattr(self, 'source_combined_private_profile', None)
                                or private_profile(rule.salt))
            return result

    return Q1OnMissSourceModel


def combined_profile(approx_profile, salt):
    return approx_profile + '+' + private_profile(salt)


def combined_identity_salt(approx_salt, salt):
    """One digest over the chained salts (approximate operations first, then this rule)."""
    return hashlib.sha256(json.dumps(dict(approximate=approx_salt, reduced=salt),
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configure_model(adapter):
    if enabled(adapter.kernel_policy):
        model = adapter.model
        rule = model.source_q1_on_miss = store_rule(adapter)
        approx = getattr(model, 'source_approx_private_profile', None)
        if approx is not None:
            # Composed with an approximate operation (e.g. route drop): both stamp the same
            # combined profile, so a row they both touch is not a profile conflict.
            # (2026-11-08: drop + tiers failed every request with "another private profile".)
            model.source_combined_private_profile = combined_profile(approx, rule.salt)
            model.source_approx_private_profile = model.source_combined_private_profile
            from .deepseek_v41_source_approx import identity_salt as approx_salt
            model.source_combined_identity_salt = combined_identity_salt(approx_salt(adapter.kernel_policy),
                                                                         rule.salt)


def diagnostics(adapter):
    policy = adapter.kernel_policy
    mode = mode_of(policy)
    if mode is None:
        return {'selected': False}
    rule = store_rule(adapter)
    value = rule.receipt() if rule is not None else {'selected': True, 'bound': False}
    value.update(apcv2_publication='refused: ' + apc_refusal_reason(policy))
    model = getattr(adapter, 'model', None)
    combined = getattr(model, 'source_combined_private_profile', None)
    if combined is not None:
        # Composed with an approximate operation: cache rows carry this profile, not the
        # tier-only private_cache_profile (mirrors the approximate-operations receipt).
        value['combined_private_cache_profile'] = combined
        value['combined_identity_salt'] = getattr(model, 'source_combined_identity_salt', None)
    return value


def payload_bytes(arrays):
    amount = getattr(arrays, 'allocation_nbytes', None)
    return amount if amount is not None else sum(v.nbytes for v in arrays.values())


# ---- Banks ---------------------------------------------------------------

def manifest_q2_format(manifest):
    """The admitted format name whose spec this manifest matches exactly, else raises."""
    for name, spec in Q2_FORMATS.items():
        if (manifest.get('quantization') == spec['revision'] and manifest.get('group_size') == spec['group_size']
                and manifest.get('weight_bits', spec['bits']) == spec['bits']
                and manifest.get('rotation') == spec['rotation']):
            return name
    raise ValueError('Q2 bank format is not admitted: ' + repr(manifest.get('quantization')))


def load_q2_manifest(root, source, *, expected_experts=15360, q2_format=None):
    """(root, records, fingerprint). ``q2_format`` (the policy's) must match the manifest;
    None admits either known format. Each record carries its bank's ``q2_format``."""
    from .deepseek_v41_source_artifact import checked_file
    from .deepseek_v41_mlx_weights import _stamp
    root = Path(root).resolve(); manifest = json.loads(checked_file(root, 'manifest.json').read_text())
    digest = manifest.pop('fingerprint')
    if hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != digest:
        raise ValueError('Q2 manifest fingerprint mismatch')
    if (manifest.get('schema') != 'mlx2.source-expert-q2.v1' or manifest['source'] != source
            or len(manifest['experts']) != expected_experts or manifest.get('dense_modified') is not False):
        raise ValueError('Q2 bank is not the admitted Source expert-only derivative')
    fmt = manifest_q2_format(manifest)
    if q2_format is not None and fmt != q2_format:
        raise ValueError(f'Q2 bank is format {fmt} but the policy names q2_format {q2_format}')
    payload = Q2_FORMATS[fmt]['payload_bytes']
    records = {}
    for record in manifest['experts']:
        key = (record['layer'], record['expert'])
        if (key in records or not 0 <= key[0] < 40 or not 0 <= key[1] < 384 or record['payload_bytes'] != payload
                or record.get('quantization', manifest['quantization']) != manifest['quantization']):
            raise ValueError('invalid Q2 expert inventory')
        path = checked_file(root, record['file'])
        records[key] = {**record, 'stat': _stamp(path.stat()), 'q2_format': fmt}
        if records[key]['stat']['size'] != record['size']:
            raise ValueError('Q2 size changed')
    return root, records, digest


def q2_read(root, record):
    """Verified Q2 host tensors: {proj.q2weight|q2scales|q2biases}.

    One open per read (header parsed from the same bytes); the content hash is
    checked once per file and stat seal, later reads are stat-sealed
    (deepseek_v41_mlx_weights.read_sealed_bank_file). Geometry follows the
    record's ``q2_format`` (default g64).
    """
    from .deepseek_v41_mlx_weights import read_sealed_bank_file
    from .deepseek_v41_source_weights import PROJECTIONS
    import numpy as np
    spec = tiers_mod.q2_format(record.get('q2_format', tiers_mod.DEFAULT_Q2_FORMAT))
    raw = read_sealed_bank_file(root, record['file'], record)
    expected = {p + '.' + f for p in PROJECTIONS for f in ('qweight', 'scales', 'biases')}
    if set(raw) != expected or sum(v.nbytes for v in raw.values()) != spec['payload_bytes']:
        raise ValueError('Q2 tensor inventory mismatch')
    result = {}
    for name, v in raw.items():
        n, k = (5120, 2304) if name.startswith('down.') else (2304, 5120)
        shape = (n, k // (16 if name.endswith('qweight') else spec['group_size']))
        dtype = np.dtype('<u4' if name.endswith('qweight') else '<f2')
        if v.shape != shape or v.dtype != dtype:
            raise ValueError('Q2 tensor geometry mismatch')
        proj, field = name.split('.')
        result[proj + '.q2' + {'qweight': 'weight', 'scales': 'scales', 'biases': 'biases'}[field]] = v
    return result


class Q2Expert(dict):
    """Q2 arrays under distinct names: code unaware of Q2 fails closed (KeyError).

    Carries its bank's quantization law: ``group_size`` and ``rotation`` (None,
    or ``block_hadamard_256``: the activation is rotated before each matmul).
    """
    source_tier = 'q2'
    group_size = 64
    rotation = None
    q2_format = 'g64'

    def __init__(self, arrays, *, q2_format=None):
        super().__init__(arrays)
        if q2_format is not None:
            spec = tiers_mod.q2_format(q2_format)
            self.q2_format, self.group_size, self.rotation = q2_format, spec['group_size'], spec['rotation']


ROTATION_BLOCKS = {'block_hadamard_256': 256}


def rotate_activation(x, rotation):
    """H x for a block-Hadamard bank (orthonormal, over the last dim); x unchanged when None."""
    if rotation is None:
        return x
    import mlx.core as mx
    block = ROTATION_BLOCKS.get(rotation)
    k = x.shape[-1]
    if block is None or k % block:
        raise ValueError('unsupported Q2 rotation ' + repr(rotation) + ' for width ' + str(k))
    return mx.hadamard_transform(x.reshape(*x.shape[:-1], k // block, block)).reshape(x.shape)


def is_q2(data):
    return getattr(data, 'source_tier', None) == 'q2'


class Bank:
    """A sealed reduced bank; ``size`` is its per-expert payload, ``q2_format`` its format (Q2 only)."""

    def __init__(self, tier, root, records, binding, size, q2_format=None):
        self.tier, self.root, self.records, self.binding, self.size = tier, root, records, binding, size
        self.q2_format = q2_format

    def format_binding(self):
        if self.q2_format is None:
            return None
        spec = tiers_mod.q2_format(self.q2_format)
        return dict(format=self.q2_format, revision=spec['revision'], group_size=spec['group_size'],
                    rotation=spec['rotation'], payload_bytes=spec['payload_bytes'], fingerprint=self.binding)


# ---- Per-layer LRU pool --------------------------------------------------

class TierPool:
    """One reduced tier: a per-layer LRU with its own byte budget per layer."""

    def __init__(self, tier, budgets):
        self.tier = tier
        self.budgets = list(budgets)
        self.capacity = sum(self.budgets)
        self.targets = list(self.budgets)
        self.layers = [OrderedDict() for _ in self.budgets]
        self.live = [0] * len(self.budgets)
        self.stats = {name: [0] * len(self.budgets) for name in
                      ('hits', 'misses', 'insertions', 'evictions', 'pressure_evictions', 'uncached')}

    @property
    def live_bytes(self):
        return sum(self.live)

    def lookup(self, key):
        layer = self.layers[key[0]]
        arrays = layer.get(key[1])
        if arrays is None:
            self.stats['misses'][key[0]] += 1
            return None
        layer.move_to_end(key[1])
        self.stats['hits'][key[0]] += 1
        return arrays

    def evict_layer_to(self, index, limit, protected, *, pressure=False):
        layer, evicted = self.layers[index], 0
        for expert in list(layer):
            if self.live[index] <= limit:
                break
            arrays = layer[expert]
            if protected.get((index, expert)) is arrays:
                continue
            del layer[expert]
            amount = payload_bytes(arrays)
            self.live[index] -= amount; evicted += amount
            self.stats['evictions'][index] += 1
            if pressure:
                self.stats['pressure_evictions'][index] += 1
        return evicted

    def insert(self, key, arrays, size, protected):
        index = key[0]
        target = self.targets[index]
        if size > target:
            self.stats['uncached'][index] += 1
            return False
        self.evict_layer_to(index, target - size, protected)
        if self.live[index] + size > target:
            self.stats['uncached'][index] += 1
            return False
        self.layers[index][key[1]] = arrays
        self.live[index] += size
        self.stats['insertions'][index] += 1
        return True

    def scale(self, fraction, protected):
        evicted = 0
        for index, budget in enumerate(self.budgets):
            self.targets[index] = min(budget, int(budget * fraction))
            evicted += self.evict_layer_to(index, self.targets[index], protected, pressure=True)
        return evicted

    def objects(self):
        for index, layer in enumerate(self.layers):
            for expert, arrays in layer.items():
                yield (index, expert), arrays

    def clear(self):
        for layer in self.layers:
            layer.clear()
        self.live = [0] * len(self.layers)

    def receipt(self):
        total = {name: sum(values) for name, values in self.stats.items()}
        return dict(tier=self.tier, eviction='lru_within_layer', capacity_bytes=self.capacity,
                    live_bytes=self.live_bytes, live_target_bytes=sum(self.targets), objects=sum(map(len, self.layers)),
                    **total, per_layer=dict(budget_bytes=list(self.budgets), target_bytes=list(self.targets),
                                            live_bytes=list(self.live), objects=[len(l) for l in self.layers],
                                            **{name: list(values) for name, values in self.stats.items()}))


# ---- Rule ------------------------------------------------------------------

class Q1OnMiss:
    """Per-store rule state: settings, banks, per-layer pools, layout, counters."""

    def __init__(self, settings, *, root, records, bank_binding, layers, mode=KEY, budget_bytes=None,
                 q2_bank=None, usage=None, usage_binding=None, experts=384):
        self.mode = mode
        self.name = NAME if mode == KEY else TIERS_NAME
        self.max_weight = settings['max_weight']
        first, last = settings['layers']
        if first >= layers:
            raise ValueError(mode + '.layers outside this checkpoint')
        self.first, self.last = first, min(last, layers - 1)
        self.settings = dict(settings, layers=[self.first, self.last])
        self.banks = {'q1': Bank('q1', root, records, bank_binding, None)}
        if q2_bank is not None:
            self.banks['q2'] = q2_bank
        if mode == TIERS_KEY and ('q2_root' in settings) != (q2_bank is not None):
            raise ValueError('tiers q2_root and the loaded Q2 bank disagree')
        if q2_bank is not None and q2_bank.q2_format is not None:
            spec = q2_spec(settings)
            if q2_bank.q2_format != spec['format'] or q2_bank.size != spec['payload_bytes']:
                raise ValueError('tiers q2_format and the loaded Q2 bank format disagree')
        if mode == TIERS_KEY and usage is None:
            raise ValueError('tiers requires the usage map')
        self.root, self.records, self.bank_binding = root, records, bank_binding
        self.layout = None
        pooled = mode == TIERS_KEY or 'pool_bytes' in settings
        if pooled:
            if budget_bytes is None:
                raise ValueError('reduced pools need the admitted total expert budget')
            self.layout = resolve_layout(settings, mode, budget_bytes=budget_bytes, usage=usage,
                                         layers=layers, experts=experts)
        self.pools = {}
        if self.layout is not None:
            for tier in configured_tiers(settings, mode):
                if tier == 'q4' or (mode == KEY and tier != 'q1'):
                    continue
                self.pools[tier] = TierPool(tier, [b[tier] for b in self.layout['budgets']])
        self.pool_capacity = sum(p.capacity for p in self.pools.values()) if self.pools else None
        self.usage_binding = usage_binding
        self.bank_formats = {t: b.format_binding() for t, b in self.banks.items()
                             if b.format_binding() is not None}
        self.salt = identity_salt(self.settings, bank_binding, mode=mode,
                                  banks={t: b.binding for t, b in self.banks.items()},
                                  layout_identity=None if self.layout is None else self.layout['identity'],
                                  usage_binding=usage_binding, bank_formats=self.bank_formats or None)
        self.armed = None          # {expert: renormalised weight} for the current call
        self.armed_layer = None
        self.served = {}           # key -> reduced/transient arrays leased in the current CALL
        self.served_ids = {}       # key -> id(arrays): re-acquire guard for the rest of the window
        self.per_layer = [0] * layers
        self.stats = dict(decode_calls=0, armed_calls=0, decode_selections=0, below_threshold=0,
                          below_threshold_resident=0, q1_served=0, q1_reads=0, q1_bytes_read=0,
                          q1_payload_bytes=0, stamped_forwards=0, invalid_route_calls=0)
        if mode == TIERS_KEY:
            self.stats.update(tier_q4_misses=0, tier_q2_served=0, tier_q1_served=0, q2_reads=0,
                              q2_bytes_read=0, original_transient_reads=0, original_transient_bytes=0)
        self.weight_considered = 0.0
        self.weight_q1 = 0.0
        self.armed_rank = {}
        self.counted = set()
        self.served_calls = {t: [0] * layers for t in ('q4', 'q2', 'q1')}
        self.weight_tier = {'q2': 0.0, 'q1': 0.0}

    # Back-compat single-pool view (q1_on_miss): the Q1 pool, or None.
    @property
    def pool(self):
        return self.pools.get('q1')

    @property
    def pool_live(self):
        return sum(p.live_bytes for p in self.pools.values())

    @classmethod
    def from_policy(cls, policy, *, source_binding, layers, budget_bytes=None):
        from .deepseek_v41_source_lfu import load_q1_manifest
        mode = next(key for key in KEYS if key in policy)
        settings = policy[mode]
        root, records, binding = load_q1_manifest(policy['q1_root'], source_binding)
        q2_bank = usage = usage_binding = None
        if mode == TIERS_KEY:
            if 'q2_root' in settings:
                spec = q2_spec(settings)
                q2_root, q2_records, q2_binding = load_q2_manifest(settings['q2_root'], source_binding,
                                                                   q2_format=spec['format'])
                q2_bank = Bank('q2', q2_root, q2_records, q2_binding, spec['payload_bytes'], spec['format'])
            usage, usage_binding = tiers_mod.load_usage(settings['usage_map'], source_binding)
        return cls(settings, root=root, records=records, bank_binding=binding, layers=layers, mode=mode,
                   budget_bytes=budget_bytes, q2_bank=q2_bank, usage=usage, usage_binding=usage_binding)

    def active_for(self, layer):
        return self.first <= layer <= self.last

    def q2_active_for(self, layer):
        return 'q2' in self.banks and self.active_for(layer)

    def arm(self, layer, ids, weights):
        """Arm one B1 decode call; ``weights`` are the raw top-k combine weights."""
        self.stats['decode_calls'] += 1
        if not self.active_for(layer):
            return False
        ids = [int(v) for v in ids]
        weights = [float(v) for v in weights]
        total = sum(weights)
        valid = (len(ids) == len(weights) and bool(ids) and math.isfinite(total) and total > 0
                 and all(math.isfinite(w) and w >= 0 for w in weights))
        if valid and len(set(ids)) != len(ids):
            # Route-drop redirects: a repeated id is allowed only as a zero-weight copy of a
            # kept slot. Collapse it so ranking sees each kept expert once.
            kept = {}
            for e, w in zip(ids, weights):
                if e in kept and w != 0.0 and kept[e] != 0.0:
                    valid = False; break
                kept[e] = max(kept.get(e, 0.0), w)
            if valid:
                self.stats['dropped_slots_collapsed'] = self.stats.get('dropped_slots_collapsed', 0) + len(ids) - len(kept)
                ids, weights = list(kept), list(kept.values())
                if any(w == 0.0 for w in weights):
                    valid = False
        if not valid:
            # Malformed routing for this call: serve it exact (never poison
            # the provider); the counter makes the event visible.
            self.stats['invalid_route_calls'] += 1
            return False
        normalised = [w / total for w in weights]
        self.stats['armed_calls'] += 1
        self.stats['decode_selections'] += len(ids)
        self.weight_considered += 1.0
        self.armed = dict(zip(ids, normalised))
        order = sorted(range(len(ids)), key=lambda i: -normalised[i])   # stable: ties by slot
        self.armed_rank = {ids[i]: rank for rank, i in enumerate(order)}
        self.armed_layer = layer
        self.stats['below_threshold'] += sum(w < self.max_weight for w in normalised)
        return True

    def disarm(self):
        self.armed = None
        self.armed_layer = None
        self.served = {}

    def candidate(self, key):
        """Renormalised weight when ``key`` is in the armed call, else None."""
        if self.armed is None or key[0] != self.armed_layer:
            return None
        return self.armed.get(key[1])

    def tier_for(self, key, weight):
        """Serving decision for an armed, NOT resident expert.

        Returns 'q1', 'q2', 'original_transient', or None (ordinary original miss).
        """
        if self.mode == KEY:
            return 'q1' if weight < self.max_weight else None
        mapped = tiers_mod.tier_of(self.layout, key[0], key[1])
        wanted = self.requested_tier(key, weight)
        if wanted == 'q4':
            # The identity map decides which originals may enter the q4 pool.
            if mapped == 'q4':
                self.stats['tier_q4_misses'] += 1
                return None
            return 'original_transient'
        return wanted

    def requested_tier(self, key, weight):
        """Precision requested for an armed selection (tiers mode)."""
        rank_tiers = self.layout['rank_tiers'][key[0]]
        if rank_tiers is not None:
            rank = self.armed_rank[key[1]]
            if rank < rank_tiers['q4']:
                return 'q4'
            if rank < rank_tiers['q4'] + rank_tiers['q2']:
                return 'q2'
            return rank_tiers['rest']
        mapped = tiers_mod.tier_of(self.layout, key[0], key[1])
        return mapped if mapped == 'q4' or weight < self.max_weight else 'q4'

    def strict_cap(self, key, weight):
        """True when a strict rank-tier layer asks for a reduced tier for this selection."""
        if self.mode != TIERS_KEY:
            return False
        rank_tiers = self.layout['rank_tiers'][key[0]]
        return bool(rank_tiers and rank_tiers.get('strict')) and self.requested_tier(key, weight) != 'q4'

    def would_reduce(self, key, weight):
        if self.mode == KEY:
            return weight < self.max_weight
        return self.requested_tier(key, weight) != 'q4'

    def count(self, layer, tier, weight):
        """Per-layer decode selections by served precision (+ reduced weight)."""
        self.served_calls[tier][layer] += 1
        if tier != 'q4':
            self.weight_tier[tier] += weight

    def record_resident(self):
        self.stats['below_threshold_resident'] += 1

    def record_read(self, tier, read_bytes, payload):
        """One read of a reduced bank (an uncached serve or a pool miss)."""
        if tier == 'q1':
            self.stats['q1_reads'] += 1
            self.stats['q1_bytes_read'] += read_bytes
            self.stats['q1_payload_bytes'] += payload
        else:
            self.stats['q2_reads'] += 1
            self.stats['q2_bytes_read'] += read_bytes

    def record_io(self, tier, read_ns, upload_ns):
        """Wall time of one reduced-tier read (also in demand_wait_ns) and its upload."""
        for name, value in (('_read_wait_ns', read_ns), ('_upload_ns', upload_ns)):
            self.stats[tier + name] = self.stats.get(tier + name, 0) + value

    def record_served(self, key, weight, tier='q1'):
        """One selection served from a reduced tier (bank read or pool hit)."""
        self.stats['q1_served'] += 1   # all reduced tiers: drives cache-row stamping
        if self.mode == TIERS_KEY:
            self.stats['tier_' + tier + '_served'] += 1
        self.per_layer[key[0]] += 1
        self.weight_q1 += weight

    def record_transient(self, read_bytes):
        self.stats['original_transient_reads'] += 1
        self.stats['original_transient_bytes'] += read_bytes

    def scale_pools(self, fraction, protected):
        return sum(pool.scale(fraction, protected) for pool in self.pools.values())

    def tier_objects(self):
        """Every reduced/transient object the rule owns (pools + current leases)."""
        for pool in self.pools.values():
            yield from list(pool.objects())
        yield from list(self.served.items())

    def pool_clear(self):
        for pool in self.pools.values():
            pool.clear()

    def receipt(self):
        selections = self.stats['decode_selections']
        layout = None
        if self.layout is not None:
            layout = dict(identity=self.layout['identity'], budget_bytes=self.layout['budget_bytes'],
                          totals=self.layout['totals'], per_layer_budgets=self.layout['budgets'],
                          per_layer_expert_counts=self.layout['counts'])
        return dict(selected=True, policy=self.name, mode=self.mode, max_weight=self.max_weight,
                    layers=[self.first, self.last],
                    banks={t: b.binding for t, b in self.banks.items()}, q1_bank_binding=self.bank_binding,
                    bank_formats=self.bank_formats or None,
                    usage_map_binding=self.usage_binding, layout=layout,
                    pools={t: p.receipt() for t, p in self.pools.items()} or None,
                    pool=None if self.pool is None else self.pool.receipt(),
                    identity_salt=self.salt, private_cache_profile=private_profile(self.salt),
                    approximate=True, qualification=QUALIFICATION, status=STATUS, default_selected=False,
                    publication_qualified=PUBLICATION_QUALIFIED,
                    scope='B1 single-row MoE calls only; prefill and resident experts exact original MXFP4',
                    counters=dict(self.stats),
                    weight_served_from_q1=self.weight_q1,
                    mean_weight_from_q1_per_armed_call=(self.weight_q1 / self.weight_considered
                                                        if self.weight_considered else 0.0),
                    q1_fraction_of_decode_selections=(self.stats['q1_served'] / selections if selections else 0.0),
                    per_layer_q1_served=list(self.per_layer),
                    per_layer_served_calls={t: list(v) for t, v in self.served_calls.items()},
                    weight_served={t: v for t, v in self.weight_tier.items()},
                    rank_tiers=None if self.layout is None else self.layout['rank_tiers'])
