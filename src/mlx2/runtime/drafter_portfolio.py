"""Revision-bound multi-drafter cache plumbing and host-only admission.

The portfolio never mixes proposal distributions.  Each resident drafter owns
its tensor math and cache slice; the proposal pool compares complete token
paths, while the processed target law remains the only verification authority.
"""

from __future__ import annotations

import copy
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from types import SimpleNamespace

_PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class PortfolioProvider:
    provider_id: str
    mechanism: str
    revision: str
    draft: object
    lane_cost: int = 1
    tile_rows: int = 1
    fallback: bool = False

    def __post_init__(self):
        if not isinstance(self.provider_id, str) or not _PROVIDER_ID.fullmatch(
            self.provider_id
        ):
            raise ValueError("portfolio provider_id must be a bounded stable name")
        if not isinstance(self.mechanism, str) or not self.mechanism:
            raise ValueError("portfolio mechanism must be nonempty")
        if (
            not isinstance(self.revision, str)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.revision) is None
        ):
            raise ValueError("portfolio provider revision must be pinned")
        if type(self.lane_cost) is not int or self.lane_cost < 1:
            raise ValueError("portfolio lane_cost must be positive")
        if type(self.tile_rows) is not int or self.tile_rows < 1:
            raise ValueError("portfolio tile_rows must be positive")
        if type(self.fallback) is not bool:
            raise ValueError("portfolio fallback must be boolean")
        config = getattr(self.draft, "config", None)
        if config is None or not getattr(config, "target_layer_ids", None):
            raise ValueError("portfolio drafter must declare target_layer_ids")
        for method in (
            "make_cache",
            "batch_caches",
            "append_context",
            "draft_distributions",
        ):
            if not callable(getattr(self.draft, method, None)):
                raise TypeError(f"portfolio drafter lacks {method}")


@dataclass(frozen=True)
class PortfolioAdmissionPolicy:
    max_active_providers: int = 5
    lane_slots: int = 5
    tile_rows: int = 75
    minimum_expected_tokens: float = 0.25
    minimum_observations: int = 8
    park_rounds: int = 16
    reprobe_interval: int = 16
    pld_fallback: bool = True

    @classmethod
    def from_value(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("invalid multi_drafter admission policy")
        policy = cls(**value)
        for name in (
            "max_active_providers",
            "lane_slots",
            "tile_rows",
            "minimum_observations",
            "park_rounds",
            "reprobe_interval",
        ):
            item = getattr(policy, name)
            if type(item) is not int or item < 1:
                raise ValueError(f"multi_drafter {name} must be positive")
        if (
            type(policy.minimum_expected_tokens) not in (int, float)
            or not math.isfinite(policy.minimum_expected_tokens)
            or policy.minimum_expected_tokens < 0
        ):
            raise ValueError(
                "multi_drafter minimum_expected_tokens must be finite and nonnegative"
            )
        if type(policy.pld_fallback) is not bool:
            raise ValueError("multi_drafter pld_fallback must be boolean")
        return policy

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class DrafterAdmissionPlanner:
    """Choose providers before generation from lagged committed evidence only."""

    def __init__(self, policy=None, *, max_requests=1024):
        self.policy = PortfolioAdmissionPolicy.from_value(policy)
        self.max_requests = int(max_requests)
        self._rounds = OrderedDict()
        self._parked = OrderedDict()

    def snapshot(self):
        """Capture bounded host state before an executor transaction starts."""
        return self._rounds.copy(), self._parked.copy()

    def restore(self, snapshot):
        """Undo failed-round parking, reprobes and request-LRU evictions."""
        rounds, parked = snapshot
        self._rounds, self._parked = rounds.copy(), parked.copy()

    def reserve_runtime_provider(self, admission, prediction, *, reason):
        """Reserve fallback/reprobe work before invoking its provider.

        Empty results still consumed work. Keep those reservations charged so
        successive empty providers cannot bypass the per-round limits.
        """
        source_id = prediction["source_id"]
        fallback = prediction.get("fallback", False)
        lanes = int(prediction["lane_cost"])
        rows = int(prediction["tile_rows"])
        if (
            source_id in admission["admitted"]
            or (fallback and not self.policy.pld_fallback)
            or len(admission["admitted"]) >= self.policy.max_active_providers
            or admission["lane_slots_used"] + lanes > self.policy.lane_slots
            or admission["tile_rows_used"] + rows > self.policy.tile_rows
        ):
            return None
        return {
            **admission,
            "admitted": (*admission["admitted"], source_id),
            "lane_slots_used": admission["lane_slots_used"] + lanes,
            "tile_rows_used": admission["tile_rows_used"] + rows,
            "fallback_only": bool(fallback),
            reason: source_id,
        }

    def _request_round(self, request_id):
        value = self._rounds.get(request_id, 0)
        self._rounds[request_id] = value + 1
        self._rounds.move_to_end(request_id)
        while len(self._rounds) > self.max_requests:
            evicted, _ = self._rounds.popitem(last=False)
            for key in tuple(self._parked):
                if key[0] == evicted:
                    self._parked.pop(key, None)
        return value

    def choose(self, request_id, predictions):
        """Return an immutable admission receipt with a bounded provider set."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("multi_drafter admission requires request identity")
        predictions = tuple(dict(item) for item in predictions)
        if not predictions:
            raise ValueError("multi_drafter admission requires providers")
        current_round = self._request_round(request_id)
        candidates = []
        parked = []
        for item in predictions:
            source_id = item["source_id"]
            key = (request_id, source_id)
            parked_until = self._parked.get(key, 0)
            observed = int(item.get("observations", 0))
            expected = float(item["expected_accepted_tokens"])
            fallback = bool(item.get("fallback", False))
            reprobe = bool(parked_until) and current_round >= parked_until
            if parked_until > current_round:
                parked.append(
                    {
                        "source_id": source_id,
                        "mechanism": item["mechanism"],
                        "reason": "low_committed_acceptance",
                        "reprobe_round": parked_until,
                    }
                )
                continue
            if reprobe:
                self._parked.pop(key, None)
            elif (
                not fallback
                and observed >= self.policy.minimum_observations
                and expected < self.policy.minimum_expected_tokens
            ):
                parked_until = current_round + max(
                    self.policy.park_rounds, self.policy.reprobe_interval
                )
                self._parked[key] = parked_until
                parked.append(
                    {
                        "source_id": source_id,
                        "mechanism": item["mechanism"],
                        "reason": "low_committed_acceptance",
                        "reprobe_round": parked_until,
                    }
                )
                continue
            # Prefer useful expected depth per reserved tile row.  Source id is
            # a final stable tie break; no current provider score participates.
            efficiency = expected / max(1, int(item["tile_rows"]))
            candidates.append(
                (
                    -efficiency,
                    -expected,
                    fallback,
                    source_id,
                    item,
                )
            )
        candidates.sort(key=lambda row: row[:4])
        selected = []
        used_lanes = used_rows = 0
        for _efficiency, _expected, _fallback_order, _source_id, item in candidates:
            if item.get("fallback", False):
                continue
            if len(selected) >= self.policy.max_active_providers:
                break
            lanes = int(item["lane_cost"])
            rows = int(item["tile_rows"])
            if used_lanes + lanes > self.policy.lane_slots:
                continue
            if used_rows + rows > self.policy.tile_rows:
                continue
            selected.append(item)
            used_lanes += lanes
            used_rows += rows
        if not any(not item.get("fallback", False) for item in selected):
            fallbacks = [
                row[-1] for row in candidates if row[-1].get("fallback", False)
            ]
            if self.policy.pld_fallback and fallbacks and fallbacks[0] not in selected:
                fallback = fallbacks[0]
                if (
                    int(fallback["lane_cost"]) <= self.policy.lane_slots
                    and int(fallback["tile_rows"]) <= self.policy.tile_rows
                ):
                    selected = [fallback]
                    used_lanes = int(fallback["lane_cost"])
                    used_rows = int(fallback["tile_rows"])
        return {
            "round": current_round,
            "admitted": tuple(item["source_id"] for item in selected),
            "parked": tuple(parked),
            "lane_slots_used": used_lanes,
            "lane_slots_available": self.policy.lane_slots,
            "tile_rows_used": used_rows,
            "tile_rows_available": self.policy.tile_rows,
            "fallback_only": bool(selected)
            and all(item.get("fallback", False) for item in selected),
        }


class DrafterPortfolio:
    """One cache/tap facade over one to 32 independently loaded drafters."""

    proposal_distribution = "deterministic_point_mass"
    requires_processor_histories = True

    def __init__(self, providers):
        providers = tuple(providers)
        if not 1 <= len(providers) <= 32 or any(
            not isinstance(item, PortfolioProvider) for item in providers
        ):
            raise ValueError("drafter portfolio requires 1..32 provider records")
        if len({item.provider_id for item in providers}) != len(providers):
            raise ValueError("drafter portfolio provider ids must be unique")
        vocabs = {int(item.draft.config.vocab_size) for item in providers}
        hidden = {int(item.draft.config.hidden_size) for item in providers}
        if len(vocabs) != 1 or len(hidden) != 1:
            raise ValueError("drafter portfolio target vocabulary/hidden size mismatch")
        self.providers = providers
        self._by_id = {item.provider_id: item for item in providers}
        self._cache_slices = {}
        cursor = 0
        for item in providers:
            count = len(item.draft.make_cache())
            self._cache_slices[item.provider_id] = slice(cursor, cursor + count)
            cursor += count
        target_layers = tuple(
            sorted(
                {
                    int(layer)
                    for item in providers
                    for layer in item.draft.config.target_layer_ids
                }
            )
        )
        self._layer_index = {layer: index for index, layer in enumerate(target_layers)}
        self._hidden_slices = {
            item.provider_id: tuple(
                self._layer_index[int(layer)]
                for layer in item.draft.config.target_layer_ids
            )
            for item in providers
        }
        configs = [item.draft.config for item in providers]
        self.config = SimpleNamespace(
            block_size=min(int(config.block_size) for config in configs),
            vocab_size=vocabs.pop(),
            hidden_size=hidden.pop(),
            target_layer_ids=target_layers,
            num_hidden_layers=sum(int(config.num_hidden_layers) for config in configs),
            num_key_value_heads=max(
                int(config.num_key_value_heads) for config in configs
            ),
            head_dim=max(int(config.head_dim) for config in configs),
            intermediate_size=max(int(config.intermediate_size) for config in configs),
        )

    @property
    def receipt_settings(self):
        return {
            "schema": "mlx2.multi-drafter-portfolio.v1",
            "providers": [
                {
                    "provider_id": item.provider_id,
                    "mechanism": item.mechanism,
                    "revision": item.revision,
                    "lane_cost": item.lane_cost,
                    "tile_rows": item.tile_rows,
                    "fallback": item.fallback,
                    "settings": copy.deepcopy(item.draft.receipt_settings),
                }
                for item in self.providers
            ],
            "target_layer_ids": list(self.config.target_layer_ids),
            "cache_ownership": "provider_private_flattened_slices",
            "proposal_combination": "complete_path_dedup_and_target_verification",
        }

    def _hidden(self, provider_id, hidden):
        import mlx.core as mx

        size = self.config.hidden_size
        return mx.concatenate(
            [
                hidden[..., index * size : (index + 1) * size]
                for index in self._hidden_slices[provider_id]
            ],
            axis=-1,
        )

    def context_for(self, provider_id, context):
        cache_slice = self._cache_slices[provider_id]
        return replace(
            context,
            draft_cache=context.draft_cache[cache_slice],
            pending_features=self._hidden(provider_id, context.pending_features),
        )

    def make_cache(self):
        return [cache for item in self.providers for cache in item.draft.make_cache()]

    def batch_caches(self, rows):
        result = []
        for item in self.providers:
            cache_slice = self._cache_slices[item.provider_id]
            result.extend(item.draft.batch_caches([row[cache_slice] for row in rows]))
        return result

    def append_context(self, hidden, cache):
        for item in self.providers:
            cache_slice = self._cache_slices[item.provider_id]
            item.draft.append_context(
                self._hidden(item.provider_id, hidden), cache[cache_slice]
            )

    def extra_memory_reservation(self, cache, *, append, pending_tokens, prefill):
        required = 0
        for item in self.providers:
            reserve = getattr(item.draft, "extra_memory_reservation", None)
            if callable(reserve):
                extra = reserve(
                    cache[self._cache_slices[item.provider_id]], append=append,
                    pending_tokens=pending_tokens, prefill=prefill,
                )
                if type(extra) is not int or extra < 0:
                    raise ValueError("drafter memory reservation must be nonnegative bytes")
                required += extra
        return required

    def draft_distributions(
        self, anchors, hidden, cache, proposal_length, rngs, temperatures, **kwargs
    ):
        from .speculative_sampling import RequestRNG

        snapshots = [
            rng.snapshot() if callable(getattr(rng, "snapshot", None)) else None
            for rng in rngs
        ]
        outputs = None
        for ordinal, item in enumerate(self.providers):
            cache_slice = self._cache_slices[item.provider_id]
            provider_rngs = rngs
            if ordinal:
                provider_rngs = [
                    RequestRNG(state=state) if state is not None else copy.deepcopy(rng)
                    for rng, state in zip(rngs, snapshots, strict=True)
                ]
            result = item.draft.draft_distributions(
                anchors,
                self._hidden(item.provider_id, hidden),
                cache[cache_slice],
                proposal_length,
                provider_rngs,
                temperatures,
                **kwargs,
            )
            if outputs is None:
                outputs = result
        return outputs


class PortfolioContinuationSource:
    """Bind an existing complete-path provider to one portfolio cache slice."""

    def __init__(self, portfolio, provider_id, source):
        if provider_id not in portfolio._by_id or not callable(source):
            raise ValueError("unknown portfolio continuation provider")
        self.portfolio = portfolio
        self.provider_id = provider_id
        self.source = source
        self.mechanism = source.mechanism

    def __call__(self, context, limit):
        return self.source(self.portfolio.context_for(self.provider_id, context), limit)


__all__ = [
    "DrafterAdmissionPlanner",
    "DrafterPortfolio",
    "PortfolioAdmissionPolicy",
    "PortfolioContinuationSource",
    "PortfolioProvider",
]
