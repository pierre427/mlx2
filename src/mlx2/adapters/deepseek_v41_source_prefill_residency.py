# SPDX-License-Identifier: MIT
"""Opt-in Source prefill admission that keeps the decode-resident expert set.

``prefill_residency`` is a key of the Source ``deepseek_source_experts``
policy. ``evict_v1`` (the default, also when absent) is today's behavior: a
long prefill's misses evict through the ordinary LFU/split-pool order, which
on the Hot60/Q1g64 checkpoint costs about 11,600 evictions per 4K request and
leaves decode re-reading what prefill evicted.

``protect_decode_v1`` snapshots the experts resident when the first wide MoE
call (``PROTECT_MIN_ROWS`` or more rows) of a prefill begins, and shields them
from demand eviction until the first narrow MoE call (decode or verify).
A prefill miss first evicts unshielded objects in the ordinary order. When
only shielded objects could make room, the expert is uploaded through the
ordinary read/upload path but served as a bounded *transient* object: it is
leased for the call and kept only while consecutive calls stay on the same
layer (layer-major sub-chunks), never inserted into the cache. Transients
count against the configured hard maximum and a fixed byte cap; when either
would be exceeded the shield yields and the ordinary eviction runs (counted).
Live-memory pressure eviction is never shielded.

Considered and rejected: snapshot plus background restore after prefill (the
antirez/ds4 ``ds4_expert_restore_async`` design). Here an upload is an MLX
evaluation under the store lock, so a background restore would race demand
uploads, pending-upload ownership and the bounded-completion window, and would
re-read bytes prefill had just evicted. Keeping the set needs no new thread,
no new upload site and no re-read.

Exactness: precision is the frozen hot map, bytes and kernels are the
ordinary ones, only which objects stay resident changes. Dynamic precision is
refused because its kind assignment depends on cache history.
Provenance: provenance/deepseek-v41-source-prefill-residency.json.
"""
POLICY_KEY = 'prefill_residency'
EVICT = 'evict_v1'
PROTECT = 'protect_decode_v1'
POLICIES = (EVICT, PROTECT)
# Wide calls start (or keep) the shield. Decode (1), DSpark verify (<=6) and
# AST tiles (<=16) rows are narrow and release it.
PROTECT_MIN_ROWS = 64
# Bound on transient (stream-through) objects held between same-layer calls:
# one layer of Q1 experts (384 x 6.6 MB) plus slack for its originals.
TRANSIENT_MAX_BYTES = 4 << 30
COUNTERS = ('prefill_shield_snapshots', 'prefill_shield_snapshot_objects', 'prefill_shield_releases',
            'prefill_shield_blocks', 'prefill_shield_yields', 'prefill_shield_yield_bytes',
            'prefill_transient_uploads', 'prefill_transient_bytes', 'prefill_transient_hits',
            'prefill_transient_peak_bytes', 'prefill_shield_retained_hits')


def validate(result, *, adaptive):
    """Validate the key inside a parsed Source expert policy."""
    if POLICY_KEY not in result:
        return result
    value = result[POLICY_KEY]
    if value not in POLICIES:
        raise ValueError('prefill_residency must be evict_v1 or protect_decode_v1')
    if value == PROTECT:
        if result.get('mode') != 'hot_original_q1' or not adaptive:
            raise ValueError('protect_decode_v1 requires adaptive hot_original_q1 residency')
        if result.get('residency') == 'dynamic_precision_v1':
            raise ValueError('protect_decode_v1 cannot combine with dynamic_precision_v1')
    return result


def receipt(store):
    policy = getattr(store, '_prefill_residency', EVICT)
    value = dict(policy=policy, qualification='unqualified')
    if policy == PROTECT:
        shield = getattr(store, '_prefill_shield', None)
        value.update(min_rows=PROTECT_MIN_ROWS, transient_max_bytes=TRANSIENT_MAX_BYTES,
                     shield_active=shield is not None, shield_objects=len(shield or ()),
                     transient_objects=len(getattr(store, '_prefill_transient', {})),
                     transient_held_bytes=getattr(store, '_prefill_transient_bytes', 0),
                     counters={k: store.stats.get(k, 0) for k in COUNTERS},
                     contract='decode-resident set shielded from prefill demand eviction; '
                              'bytes, precision and kernels unchanged; pressure eviction unshielded')
    return value
