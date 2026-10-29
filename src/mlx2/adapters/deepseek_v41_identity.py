# SPDX-License-Identifier: MIT
"""Effective weight identity for immutable Source expert derivatives.

Residency budgets, eviction order and filesystem locations do not change the
model. A source/Q1-bank/hot-map combination does. Keep the digest compatible
with existing Source/Q1 research receipts while exposing its constituents.
"""

import hashlib
import json
import re


def effective_source_identity(base_identity, *, q1_bank=None, hot_map=None):
    """Return a copy; never relabel the source provider's immutable artifact."""
    result = dict(base_identity)
    source = result["fingerprint"]
    if q1_bank is None and hot_map is None:
        result["base_fingerprint"] = source
        return result
    bindings = {"source": source, "q1": q1_bank, "hotmap": hot_map}
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in bindings.values()):
        raise ValueError("Source derivative identity requires all three sealed SHA256 bindings")
    # desktop_runner.digest used this serialization in the original campaign.
    fingerprint = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
    result.update(
        fingerprint=fingerprint,
        weight_sha256=fingerprint,
        base_fingerprint=source,
        derivation={"schema": "mlx2.v41.source-expert-derivative.v1", **bindings},
    )
    return result
