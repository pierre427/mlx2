"""Exact request-private prefix reuse for the Nemotron-H target cache.

Nemotron-H mixes Mamba recurrent state and attention KV.  A reusable sibling
boundary is valid only after a B1 target verification has used the model's
ordinary tokenwise path and the hybrid transaction has committed one exact
prefix.  MTP cache planes and the separate TwoTower architecture are refused.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


class NemotronPrefixReuseUnsupported(RuntimeError):
    """The model or cache geometry cannot prove exact shared-prefix reuse."""


def _validate_target_geometry(model, cache) -> tuple[int, int]:
    from .models.cache import ArraysCache, KVCache

    if getattr(model, "model_type", None) != "nemotron_h":
        raise NemotronPrefixReuseUnsupported(
            "exact prefix reuse supports the Nemotron-H target, not TwoTower"
        )
    if not bool(getattr(model, "supports_exact_prefix_cascade", False)):
        raise NemotronPrefixReuseUnsupported(
            "model does not declare exact B1 prefix-cascade state"
        )
    expected = model.make_cache()
    row = list(cache)
    if not row or len(row) != len(expected):
        raise NemotronPrefixReuseUnsupported(
            "cache does not match the target-layer topology"
        )
    recurrent = attention = 0
    for actual, declared in zip(row, expected):
        if type(actual) is not type(declared):
            raise NemotronPrefixReuseUnsupported(
                "cache includes an MTP, merged, or foreign state plane"
            )
        if isinstance(actual, ArraysCache):
            recurrent += 1
            if actual.batch_size != 1 and not actual.empty():
                raise NemotronPrefixReuseUnsupported(
                    "recurrent shared-prefix state must have exactly one row"
                )
        elif type(actual) is KVCache:
            attention += 1
            if actual.keys is not None and int(actual.keys.shape[0]) != 1:
                raise NemotronPrefixReuseUnsupported(
                    "attention shared-prefix state must have exactly one row"
                )
        else:
            raise NemotronPrefixReuseUnsupported(
                f"unsupported Nemotron target cache plane: {type(actual).__name__}"
            )
    if recurrent == 0 or attention == 0:
        raise NemotronPrefixReuseUnsupported(
            "Nemotron shared-prefix reuse requires recurrent and attention state"
        )
    return recurrent, attention


def exact_prefix_reuse_geometry(
    model,
    cache,
    *,
    state_revision: str,
    cache_layout: str,
):
    """Return a revision-bound exact B1 target-cache attestation."""

    from .models.cache import KVCache
    from .proposal_cascade import PrefixReuseGeometry

    recurrent, attention = _validate_target_geometry(model, cache)
    offsets = {
        int(entry.offset)
        for entry in cache
        if type(entry) is KVCache
    }
    if len(offsets) != 1:
        raise NemotronPrefixReuseUnsupported(
            "attention shared-prefix state must have one aligned position"
        )
    return PrefixReuseGeometry(
        state_revision=state_revision,
        cache_layout=cache_layout,
        position=offsets.pop(),
        layer_count=len(cache),
        state_components=("recurrent_layers", "attention_kv_layers"),
        component_widths=(recurrent, attention),
    )


def _clone_private_cache(cache):
    """Clone one committed B1 row; future appends remain branch-private."""

    return copy.deepcopy(list(cache))


@dataclass(slots=True)
class NemotronPrefixVerification:
    """One evaluated longest-path transaction awaiting a prefix decision."""

    model: Any
    transaction: Any
    logits: Any
    features: Any
    verified_tokens: tuple[int, ...]
    recurrent_layers: int
    attention_layers: int
    _closed: bool = False

    def commit_and_fork(self, accepted_tokens: int, *, sibling_count: int):
        """Commit the exact accepted prefix and clone it for viable siblings.

        The common tokens are not forwarded again.  Returned caches are
        request-private and are not APCv2 entries; publication remains the
        responsibility of the ordinary request lifecycle after a final path
        commits.
        """

        if self._closed or self.transaction.closed:
            raise RuntimeError("Nemotron prefix verification is closed")
        if type(accepted_tokens) is not int or not 1 <= accepted_tokens <= len(
            self.verified_tokens
        ):
            raise ValueError("accepted_tokens must be a nonempty verified prefix")
        if type(sibling_count) is not int or sibling_count < 1:
            raise ValueError("sibling_count must be a positive integer")
        rows = self.transaction.commit([accepted_tokens])
        self._closed = True
        row = rows[0]
        recurrent, attention = _validate_target_geometry(self.model, row)
        for entry in row:
            if getattr(entry, "speculating", False):
                raise RuntimeError("committed prefix retained a speculation owner")
        branches = tuple(_clone_private_cache(row) for _ in range(sibling_count))
        receipt = {
            "schema": "mlx2.nemotron-exact-prefix-reuse.v2",
            "accepted_tokens": accepted_tokens,
            "branches": sibling_count,
            "common_tokens_recomputed": 0,
            "recurrent_layers": recurrent,
            "attention_layers": attention,
            "request_private": True,
            "apcv2_published": False,
            "serving_route_implemented": False,
            "qualified": False,
            "selected": False,
            "observed_used": True,
            "observed_use_scope": "direct_request_private_prefix_primitive",
            "observed_used_in_serving": False,
        }
        return branches, receipt

    def abort(self) -> None:
        if not self._closed:
            self.transaction.abort()
            self._closed = True


def verify_longest_prefix(model, cache, tokens, *, capture_layers=()):
    """Evaluate one B1 path with canonical tokenwise Nemotron-H state updates."""

    from .hybrid_verify_rows import HybridVerifyRows

    token_ids = tuple(tokens)
    if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("verification tokens must be a nonempty token-id sequence")
    recurrent, attention = _validate_target_geometry(model, cache)
    owner = HybridVerifyRows([cache])
    transaction = owner.begin([len(token_ids)])
    try:
        import mlx.core as mx

        capture = tuple(capture_layers)
        if not capture:
            capture = (len(model.layers) - 1,)
        logits, features = model.forward_with_taps(
            mx.array([token_ids], dtype=mx.uint32),
            transaction.caches,
            capture,
        )
        mx.eval(logits, features)
    except BaseException:
        transaction.abort()
        raise
    return NemotronPrefixVerification(
        model,
        transaction,
        logits,
        features,
        token_ids,
        recurrent,
        attention,
    )


__all__ = [
    "NemotronPrefixReuseUnsupported",
    "NemotronPrefixVerification",
    "exact_prefix_reuse_geometry",
    "verify_longest_prefix",
]
