# SPDX-License-Identifier: MIT
"""Source hot-map schema, provenance and offline recalibration from an atlas.

A hot map is the exact-mode precision law: listed experts are served as the
original MXFP4 codes, all others as Q1g64. The 2026-10-27 measured map is
pinned by digest; a recalibrated map is accepted by structure and bound to
the same Source fingerprint and hot budget. Its content digest enters the
effective artifact identity either way. See
provenance/deepseek-v41-source-expert-residency.json.
"""
import hashlib
import json
from pathlib import Path

SCHEMA = 'mlx2.source-expert-hot-map.v1'
RECALIBRATION_SCHEMA = 'mlx2.source-expert-hot-map-recalibration.v1'
ORIGINAL_EXPERT_BYTES = 18800640
LAYERS, EXPERTS = 40, 384
_HEX = frozenset('0123456789abcdef')


def _digest(value):
    return type(value) is str and len(value) == 64 and set(value) <= _HEX


def validate_recalibrated(mapping, *, source, hot_budget_bytes):
    """Raise ValueError unless ``mapping`` is a complete recalibrated map."""
    if type(mapping) is not dict or mapping.get('schema') != SCHEMA:
        raise ValueError('recalibrated hot map schema mismatch')
    if mapping.get('source') != source or mapping.get('hot_budget_bytes') != hot_budget_bytes:
        raise ValueError('recalibrated hot map does not bind this Source/hot budget')
    if mapping.get('original_expert_bytes') != ORIGINAL_EXPERT_BYTES:
        raise ValueError('recalibrated hot map expert size mismatch')
    block = mapping.get('recalibration')
    if (type(block) is not dict or block.get('schema') != RECALIBRATION_SCHEMA
            or not _digest(block.get('atlas_sha256'))
            or type(block.get('calibration_token_selections')) is not int
            or block['calibration_token_selections'] <= 0
            or block.get('parent_hot_map_sha256') is not None and not _digest(block['parent_hot_map_sha256'])):
        raise ValueError('recalibrated hot map lacks a valid recalibration record')
    experts = mapping.get('experts')
    if type(experts) is not list or not experts:
        raise ValueError('recalibrated hot map has no experts')
    seen = set()
    for row in experts:
        if (type(row) is not list or len(row) != 2 or any(type(v) is not int for v in row)
                or not 0 <= row[0] < LAYERS or not 0 <= row[1] < EXPERTS):
            raise ValueError('recalibrated hot map expert geometry')
        seen.add(tuple(row))
    if len(seen) != len(experts) or len(seen)*ORIGINAL_EXPERT_BYTES > hot_budget_bytes:
        raise ValueError('recalibrated hot map duplicate or over-budget experts')
    return mapping


def provenance(raw, mapping, *, measured_sha256, source, hot_budget_bytes):
    """``measured``, ``recalibrated`` or ``unrecognized``; receipts keep them distinct."""
    digest = hashlib.sha256(raw).hexdigest()
    result = dict(sha256=digest, measured_sha256=measured_sha256)
    if digest == measured_sha256:
        return dict(result, kind='measured', recalibration=None)
    try:
        validate_recalibrated(mapping, source=source, hot_budget_bytes=hot_budget_bytes)
    except ValueError as error:
        return dict(result, kind='unrecognized', reason=str(error), recalibration=None)
    return dict(result, kind='recalibrated', recalibration=dict(mapping['recalibration']))


def build_recalibrated(atlas_path, output, *, hot_bytes, source=None, parent_hot_map=None):
    """Rank the accumulated atlas and write a new map; never overwrite."""
    from .deepseek_v41_source_lfu import read_atlas
    atlas_path, output = Path(atlas_path), Path(output)
    raw = atlas_path.read_bytes()
    if source is None:
        source = json.loads(raw)['source']
    atlas = read_atlas(atlas_path, source)
    rows = {(r['layer'], r['expert']): r for r in atlas['counts']}
    total = sum(r['token_selections'] for r in rows.values())
    if total <= 0:
        raise ValueError('atlas has no token selections to calibrate from')
    if type(hot_bytes) is not int or hot_bytes < ORIGINAL_EXPERT_BYTES:
        raise ValueError('hot budget must admit one original expert')
    keys = sorted(((l, e) for l in range(LAYERS) for e in range(EXPERTS)),
                  key=lambda k: (-rows.get(k, {}).get('token_selections', 0),
                                 -rows.get(k, {}).get('last_access', 0), k[1], k[0]))
    hot = keys[:hot_bytes//ORIGINAL_EXPERT_BYTES]
    selected = sum(rows.get(k, {}).get('token_selections', 0) for k in hot)
    parent = None
    if parent_hot_map is not None:
        parent = hashlib.sha256(Path(parent_hot_map).read_bytes()).hexdigest()
    value = {'schema': SCHEMA, 'source': source, 'original_expert_bytes': ORIGINAL_EXPERT_BYTES,
             'hot_budget_bytes': hot_bytes, 'experts': [list(k) for k in hot],
             'atlas_sha256': hashlib.sha256(raw).hexdigest(),
             'calibration_token_selections': total, 'hot_token_selections': selected,
             'calibration_hot_fraction': selected/total,
             'precision_selection': 'fixed for model lifetime; usage changes residency ranking and subsequent startup maps only',
             'recalibration': {'schema': RECALIBRATION_SCHEMA, 'atlas_sha256': hashlib.sha256(raw).hexdigest(),
                               'atlas_clock': atlas['clock'], 'calibration_token_selections': total,
                               'parent_hot_map_sha256': parent,
                               'builder': 'mlx2.adapters.deepseek_v41_source_hotmap.build_recalibrated'}}
    validate_recalibrated(value, source=source, hot_budget_bytes=hot_bytes)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    return {'output': str(output), 'hot_map_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'original_objects': len(hot), 'original_payload_bytes': len(hot)*ORIGINAL_EXPERT_BYTES,
            'calibration_hot_fraction': selected/total, 'parent_hot_map_sha256': parent}
