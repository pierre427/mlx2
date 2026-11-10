# SPDX-License-Identifier: MIT
"""Pure tier-layout resolver for Source reduced-precision expert tiers.

One function, :func:`resolve`, turns (default usage map, per-tier totals,
overrides) into a per-layer layout: the byte budget of each tier
(``q4`` = original MXFP4, ``q2`` = MLX affine 2-bit in one of the
:data:`Q2_FORMATS` bank formats, ``q1`` = affine 1-bit g64) and, when a usage map is given, the tier of every expert.
No I/O and no MLX: the serving store, CPU tests and
``scripts/research/ds41_atlas/tier_layout.py`` all call the same code.

Layers are addressed as one integer or an inclusive ``[first, last]`` span.
Overrides apply in list order; a later entry wins over an earlier one and
every entry wins over the default even split.

Budget entry (any subset of tiers; bytes or expert counts, not both per tier)::

    {"layers": [13, 26], "q4_bytes": 2000000000, "q2_experts": 40, "q1_bytes": 0}

Selection entry (applied on top of the usage-derived bands)::

    {"layers": [0, 7], "tier": "q4", "top": 120}     # top-N by usage -> tier
    {"layers": 5, "tier": "q2", "experts": [3, 17]}  # explicit experts -> tier
    {"layers": [30, 39], "only": "q1"}               # whole layer(s) -> tier

Per-token rank entry (decode rows only; precedence below)::

    {"layers": 32, "rank_tiers": {"q4": 1, "q2": 0, "rest": "q1"}}

Bands (identity map). Per layer, experts ranked by usage (descending, ties
by expert id); the first ``q4_experts`` (= ``q4_bytes // ORIGINAL``) are q4,
the next ``q2_experts`` (= ``q2_bytes // Q2``) are q2, and the REMAINDER
RULE is fixed: every lower-ranked expert is q1 (served through the per-layer
Q1 pool; ranks beyond what ``q1_bytes`` holds simply miss the pool and are
read from the Q1 bank). So ``{"q4_experts": 1, "q1_experts": 383}`` makes the
layer's single most-used expert q4 and the other 383 q1. Counts given in one
entry must sum to at most ``experts``, and the resolved q4+q2 band of every
layer must fit in ``experts``. Budgets size caches; a selection override may
put more experts in a tier than its cache holds -- the pool is then an LRU
over that larger set.

Precedence when a per-token rank entry and the identity map both cover a
layer: ``rank_tiers`` chooses the precision of each decode selection by its
combine-weight rank inside the token's top-k (rank 1 = largest weight, ties
by slot); the identity map then only decides which originals may enter the
q4 pool -- a q4 request for a q4-mapped expert is an ordinary cached miss,
for any other expert a transient (uncached) original read. The guard
``max_weight`` does not apply on rank layers. A resident original always
serves exact, whatever the rank.
"""
import hashlib
import json

TIERS = ('q4', 'q2', 'q1')
ORIGINAL_BYTES = 18800640
Q2_BYTES = 11059200          # the default (g64) Q2 format; see Q2_FORMATS
Q1_BYTES = 6635520
TIER_BYTES = {'q4': ORIGINAL_BYTES, 'q2': Q2_BYTES, 'q1': Q1_BYTES}
# Admitted Q2 expert-bank formats (policy ``tiers.q2_format``; the loaded bank's
# manifest must match its entry exactly). ``rotation`` names the input-side
# transform the runtime applies to the activation before each Q2 matmul.
Q2_FORMATS = {
    'g64': dict(revision='source-q2-mlx-affine-g64-f32q-f16-v2', group_size=64, bits=2, rotation=None,
                payload_bytes=11059200),
    'g32h': dict(revision='source-q2-mlx-affine-g32-had256-mseclip-f16-v3', group_size=32, bits=2,
                 rotation='block_hadamard_256', payload_bytes=13271040),
}
DEFAULT_Q2_FORMAT = 'g64'


def q2_format(name):
    """The admitted Q2 format spec for ``name``; refuses anything else."""
    if not isinstance(name, str) or name not in Q2_FORMATS:
        raise ValueError('q2_format must be one of: ' + ', '.join(sorted(Q2_FORMATS)))
    return Q2_FORMATS[name]


def tier_bytes(q2_bytes=None):
    """Per-expert bytes by tier, with the configured Q2 format's size."""
    return dict(TIER_BYTES, q2=Q2_BYTES if q2_bytes is None else q2_bytes)
CODE = {'q4': '4', 'q2': '2', 'q1': '1'}
BUDGET_KEYS = frozenset(t + s for t in TIERS for s in ('_bytes', '_experts'))
SELECTION_KEYS = frozenset(('tier', 'top', 'experts', 'only'))
RANK_KEY = 'rank_tiers'
TOP_K = 6
LAYOUT_SCHEMA = 'mlx2.ds41-source-tier-layout.v1'


def _int(value):
    return type(value) is int


def parse_span(value, layers):
    """``n`` or ``[first, last]`` (inclusive) -> range; refuses outside 0..layers-1."""
    if _int(value):
        value = [value, value]
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_int(v) for v in value)
            or not 0 <= value[0] <= value[1] < layers):
        raise ValueError(f'tier override layers must be n or [first, last] within 0..{layers - 1}')
    return range(value[0], value[1] + 1)


def _entry_kind(entry):
    if not isinstance(entry, dict) or 'layers' not in entry:
        raise ValueError('tier override must be an object with "layers"')
    keys = set(entry) - {'layers'}
    if keys and keys <= BUDGET_KEYS:
        return 'budget'
    if keys and keys <= SELECTION_KEYS:
        return 'selection'
    if keys == {RANK_KEY}:
        return 'rank'
    raise ValueError('tier override mixes or misses keys: ' + ', '.join(sorted(keys)))


def validate_overrides(overrides, *, layers=40, experts=384, tiers=TIERS):
    """Structural check (no budgets); returns a normalised copy."""
    if overrides is None:
        return []
    if not isinstance(overrides, list) or len(overrides) > 4096:
        raise ValueError('tier overrides must be a list')
    result = []
    for entry in overrides:
        kind = _entry_kind(entry)
        parse_span(entry['layers'], layers)
        item = dict(entry)
        if kind == 'budget':
            for tier in TIERS:
                if tier + '_bytes' in entry and tier + '_experts' in entry:
                    raise ValueError(f'tier override sets both {tier}_bytes and {tier}_experts')
            for key, value in entry.items():
                if key != 'layers' and (not _int(value) or value < 0):
                    raise ValueError(f'tier override {key} must be a non-negative integer')
                if key != 'layers' and key[:2] not in tiers:
                    raise ValueError(f'tier override {key}: tier {key[:2]} is not configured')
            counted = sum(v for k, v in entry.items() if k.endswith('_experts'))
            if counted > experts:
                raise ValueError(f'tier override expert counts sum to {counted} > {experts} experts per layer')
        elif kind == 'rank':
            rank = entry[RANK_KEY]
            if (not isinstance(rank, dict) or set(rank) - {'q4', 'q2', 'rest', 'strict'} or 'rest' not in rank
                    or type(rank.get('strict', False)) is not bool
                    or rank['rest'] not in tiers
                    or any(not _int(rank.get(t, 0)) or rank.get(t, 0) < 0 for t in ('q4', 'q2'))
                    or rank.get('q4', 0) + rank.get('q2', 0) > TOP_K
                    or (rank.get('q2', 0) and 'q2' not in tiers)):
                raise ValueError('rank_tiers must be {"q4": n, "q2": m, "rest": tier} with n+m <= '
                                 f'{TOP_K} and only configured tiers')
            item[RANK_KEY] = {'q4': rank.get('q4', 0), 'q2': rank.get('q2', 0), 'rest': rank['rest']}
            if rank.get('strict', False):
                # Precision cap (Pierre 2026-11-08: "q2 as the highest quant"): a resident
                # original does not override the requested reduced tier at decode.
                item[RANK_KEY]['strict'] = True
        else:
            if 'only' in entry:
                if set(entry) != {'layers', 'only'} or entry['only'] not in tiers:
                    raise ValueError('"only" takes one configured tier and nothing else')
            else:
                if entry.get('tier') not in tiers or ('top' in entry) == ('experts' in entry):
                    raise ValueError('selection override needs a configured "tier" and exactly one of "top"/"experts"')
                if 'top' in entry and (not _int(entry['top']) or not 0 <= entry['top'] <= experts):
                    raise ValueError(f'"top" must be an integer 0..{experts}')
                if 'experts' in entry:
                    ids = entry['experts']
                    if (not isinstance(ids, list) or not all(_int(e) and 0 <= e < experts for e in ids)
                            or len(set(ids)) != len(ids)):
                        raise ValueError(f'"experts" must be distinct integers 0..{experts - 1}')
        result.append(item)
    return result


def resolve(*, budget_bytes, totals, usage=None, overrides=None, layers=40, experts=384, tiers=TIERS,
            q2_bytes=None):
    """(usage map, per-tier totals, overrides) -> per-layer budgets and tier map.

    ``totals`` maps each configured tier to its total bytes (q4 may be omitted:
    it then gets whatever ``budget_bytes`` leaves after the reduced tiers).
    ``usage`` is ``layers`` x ``experts`` non-negative counts, or None (no map).
    ``q2_bytes`` is the configured Q2 format's per-expert payload (default g64);
    it sizes ``q2_experts`` overrides and the q2 band.
    Raises ValueError when the resolved per-layer total exceeds budget_bytes.
    """
    sizes = tier_bytes(q2_bytes)
    if not _int(sizes['q2']) or sizes['q2'] <= 0:
        raise ValueError('q2_bytes must be a positive integer')
    overrides = validate_overrides(overrides, layers=layers, experts=experts, tiers=tiers)
    if not _int(budget_bytes) or budget_bytes <= 0:
        raise ValueError('budget_bytes must be a positive integer')
    totals = dict(totals)
    for tier, value in totals.items():
        if tier not in tiers or not _int(value) or value < 0:
            raise ValueError(f'invalid tier total {tier}')
    if 'q4' not in totals:
        totals['q4'] = budget_bytes - sum(v for t, v in totals.items() if t != 'q4')
    budgets = [{t: (totals.get(t, 0) // layers) for t in TIERS} for _ in range(layers)]
    for entry in overrides:
        if _entry_kind(entry) != 'budget':
            continue
        for layer in parse_span(entry['layers'], layers):
            for key, value in entry.items():
                if key == 'layers':
                    continue
                tier, unit = key[:2], key[3:]
                budgets[layer][tier] = value if unit == 'bytes' else value * sizes[tier]
    for layer in budgets:
        for tier in TIERS:
            if tier not in tiers:
                layer[tier] = 0
    total = sum(sum(layer.values()) for layer in budgets)
    if any(budgets[l]['q4'] < 0 for l in range(layers)):
        raise ValueError('reduced-tier totals exceed budget_bytes')
    if total > budget_bytes:
        raise ValueError(f'resolved per-layer tier budgets total {total} B exceeds budget_bytes {budget_bytes} B')
    for index, layer in enumerate(budgets):
        bands = layer['q4'] // ORIGINAL_BYTES + layer['q2'] // sizes['q2']
        if bands > experts:
            raise ValueError(f'layer {index}: q4+q2 bands hold {bands} > {experts} experts')
    rank_tiers = [None] * layers
    for entry in overrides:
        if _entry_kind(entry) == 'rank':
            for layer in parse_span(entry['layers'], layers):
                rank_tiers[layer] = dict(entry[RANK_KEY])
    tier_map = None
    if usage is not None:
        if len(usage) != layers or any(len(row) != experts for row in usage):
            raise ValueError('usage map geometry must be layers x experts')
        tier_map = []
        for layer in range(layers):
            row = usage[layer]
            ranked = sorted(range(experts), key=lambda e: (-row[e], e))
            n4 = budgets[layer]['q4'] // ORIGINAL_BYTES
            n2 = budgets[layer]['q2'] // sizes['q2'] if 'q2' in tiers else 0
            assign = ['q1' if 'q1' in tiers else ('q2' if 'q2' in tiers else 'q4')] * experts
            for rank, expert in enumerate(ranked):
                if rank < n4:
                    assign[expert] = 'q4'
                elif rank < n4 + n2:
                    assign[expert] = 'q2'
            tier_map.append((assign, ranked))
        for entry in overrides:
            if _entry_kind(entry) != 'selection':
                continue
            for layer in parse_span(entry['layers'], layers):
                assign, ranked = tier_map[layer]
                if 'only' in entry:
                    assign[:] = [entry['only']] * experts
                elif 'top' in entry:
                    for expert in ranked[:entry['top']]:
                        assign[expert] = entry['tier']
                else:
                    for expert in entry['experts']:
                        assign[expert] = entry['tier']
        tier_map = [''.join(CODE[t] for t in assign) for assign, _ in tier_map]
    elif any(_entry_kind(e) == 'selection' for e in overrides):
        raise ValueError('expert selection overrides need a usage map')
    layout = dict(schema=LAYOUT_SCHEMA, layers=layers, experts=experts, tiers=[t for t in TIERS if t in tiers],
                  budget_bytes=budget_bytes, budgets=budgets, rank_tiers=rank_tiers, expert_bytes=sizes,
                  totals={t: sum(b[t] for b in budgets) for t in TIERS}, map=tier_map,
                  counts=None if tier_map is None else [
                      {t: row.count(CODE[t]) for t in TIERS} for row in tier_map])
    layout['identity'] = identity(layout)
    return layout


def identity(layout):
    body = {k: layout[k] for k in ('schema', 'layers', 'experts', 'tiers', 'budget_bytes', 'budgets', 'map', 'rank_tiers')}
    if layout.get('expert_bytes', TIER_BYTES) != TIER_BYTES:
        body['expert_bytes'] = layout['expert_bytes']   # a non-default Q2 format sizes the bands
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tier_of(layout, layer, expert):
    if layout['map'] is None:
        return None
    return {'4': 'q4', '2': 'q2', '1': 'q1'}[layout['map'][layer][expert]]


USAGE_SCHEMA = 'mlx2.ds41-source-expert-usage.v1'


def load_usage(path, source):
    """Source-bound per-layer usage counts: (rows, sha256 of the file bytes)."""
    from pathlib import Path
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get('schema') != USAGE_SCHEMA or value.get('source') != source
            or not isinstance(value.get('counts'), list)):
        raise ValueError('expert usage map does not bind this Source')
    rows = value['counts']
    if not all(isinstance(r, list) and all(_int(c) and c >= 0 for c in r) for r in rows):
        raise ValueError('expert usage counts must be non-negative integers')
    return rows, hashlib.sha256(raw).hexdigest()


def summary(layout):
    """Compact human table (one line per layer) for CPU printing."""
    lines = [f"budget_bytes={layout['budget_bytes']} totals={layout['totals']} identity={layout['identity'][:16]}"]
    for layer, budget in enumerate(layout['budgets']):
        counts = '' if layout['counts'] is None else ' experts ' + ' '.join(
            f"{t}={layout['counts'][layer][t]}" for t in TIERS)
        rank = layout['rank_tiers'][layer]
        rank = '' if rank is None else f" rank_tiers={rank}"
        lines.append(f"L{layer:02d} " + ' '.join(f"{t}={budget[t] / 1e9:.3f}GB" for t in TIERS) + counts + rank)
    return '\n'.join(lines)
