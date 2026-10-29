# SPDX-License-Identifier: MIT
"""Transaction-local preparation of selected KV, shared by borrowing layers.

This caches a gather, not attention outputs or numerical state. Selection arrays
are immutable within their published generation. Physical reads and latency
savings require a separate measurement; logical payload accounting is not DRAM
traffic. Provenance: deepseek-v41-ast-selected-kv-preparation.json.
"""
import hashlib
from pathlib import Path

POLICY = 'descendant_reuse_v1'
_KEY = '_ast_exact_selected_preparation'


def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prepare_selected_class(xp, owner, shared, nodes, *, check_live, profile):
    """Return the existing exact gather and validity with a reuse receipt.

    ``shared`` is private to one forward transaction. The actual capability
    validates source revision, COW generation and lifetime before each reuse.
    All dynamic index and validity data stay on the device. Owners and arrays
    remain in the transaction's normal shared roots until completion or drain.
    """
    if not callable(check_live) or not isinstance(profile, str) or not profile:
        raise ValueError('Selected KV preparation requires a live profile-bound capability')
    if check_live() is False:
        raise RuntimeError('Selected KV preparation transaction is no longer live')
    owner.assert_current()
    nodes = tuple(nodes)
    if (len(shared) != 16 or len(owner.logical_rows) != 16
            or owner.pool_table is None or not nodes or len(set(nodes)) != len(nodes)
            or any(type(n) is not int or not 0 <= n < 16 for n in nodes)
            or len({id(row) for row in shared}) != 16):
        raise ValueError('Selected KV preparation needs distinct canonical tree rows')
    selection = []
    for node in nodes:
        row = shared[node]
        indices, valid = row.get('indices'), row.get('selected_mask')
        generation = row.get('_index_generation', 0)
        if (indices is None or valid is None or indices.ndim != 3
                or indices.shape[:2] != (1, 1) or valid.shape != indices.shape
                or not owner.logical_rows[node] or type(generation) is not int or generation < 0):
            raise ValueError('Selected KV preparation has no complete chronological selection')
        selection.append((indices, valid, generation))
    count = selection[0][0].shape[-1]
    if not count or any(item[0].shape[-1] != count for item in selection):
        raise ValueError('Selected KV preparation requires one selection count class')
    marker = (id(check_live), profile, id(owner), owner._stamp(),
              id(owner.pool_table), tuple((id(row), len(row)) for row in owner.logical_rows),
              nodes, tuple((id(ids), id(valid), generation)
                           for ids, valid, generation in selection))
    cache = shared[0].setdefault(_KEY, {})
    if marker in cache:
        values, valid = cache[marker]
        return values, valid, dict(reused=True, built_payload_bytes=0,
                                   reused_payload_bytes=values.nbytes)
    logical = xp.concatenate([item[0] for item in selection], axis=0)[:, 0]
    valid = xp.concatenate([item[1] for item in selection], axis=0)[:, 0]
    width = max(len(owner.logical_rows[node]) for node in nodes)
    mapping = xp.array([list(owner.logical_rows[node]) +
                        [0] * (width - len(owner.logical_rows[node]))
                        for node in nodes], xp.int32)
    physical = xp.take_along_axis(mapping, logical, axis=1)
    values = owner.pool_table[0, physical].astype(xp.float32)
    cache[marker] = (values, valid)
    return values, valid, dict(reused=False, built_payload_bytes=values.nbytes,
                               reused_payload_bytes=0)
