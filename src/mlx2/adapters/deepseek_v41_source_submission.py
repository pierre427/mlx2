# SPDX-License-Identifier: MIT
"""Explicit shared-before-route scheduling; arithmetic and I/O are unchanged.

Provenance: provenance/deepseek-v41-plan-submission.json.
The final layer output consumes both branches. Completion owners must still
retain every independent root; this helper is not a completion marker.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

POLICY = 'shared_first_v1'


@lru_cache(maxsize=1)
def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    paths = ('adapters/deepseek_v41_source_submission.py',
             'adapters/deepseek_v41_source_weights.py',
             'runtime/models/deepseek_v41.py')
    records = '\n'.join(p + ':' + hashlib.sha256((root/p).read_bytes()).hexdigest()
                        for p in paths)
    return hashlib.sha256((POLICY+'\n'+records).encode()).hexdigest()


def shared_before_route(shared, routed, *, submit, counters):
    """Build and submit the independent shared branch before routed ID readback.

    ``submit`` is the current backend's async submission, never a new stream or
    independent ownership window. Exceptions propagate to the enclosing Source
    provider scope, whose existing final/error drain retains all resources.
    """
    value = shared()
    submit(value)
    counters['source_shared_first_submissions'] = counters.get(
        'source_shared_first_submissions', 0) + 1
    result = routed()
    return result, value
