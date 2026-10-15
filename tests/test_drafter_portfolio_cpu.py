"""CPU-only contracts for bounded multi-drafter admission and cache plumbing."""

from types import SimpleNamespace

import mlx.core as mx

from mlx2.runtime.drafter_portfolio import (
    DrafterAdmissionPlanner,
    DrafterPortfolio,
    PortfolioProvider,
)
from mlx2.runtime.speculative_sampling import RequestRNG

REVISION = "a" * 64


class FakeDraft:
    proposal_distribution = "deterministic_point_mass"

    def __init__(self, layer, token):
        self.config = SimpleNamespace(
            block_size=8,
            vocab_size=32,
            hidden_size=2,
            target_layer_ids=(layer,),
            num_hidden_layers=1,
            num_key_value_heads=1,
            head_dim=2,
            intermediate_size=4,
        )
        self.token = token
        self.draws = []

    @property
    def receipt_settings(self):
        return {"token": self.token}

    def make_cache(self):
        return [f"cache-{self.token}"]

    def batch_caches(self, rows):
        return [tuple(row[0] for row in rows)]

    def append_context(self, hidden, cache):
        self.appended = (tuple(hidden.shape), tuple(cache))

    def draft_distributions(
        self, anchors, hidden, cache, proposal_length, rngs, temperatures, **kwargs
    ):
        self.draws.append(tuple(rng.uniform() for rng in rngs))
        return [[self.token] * proposal_length for _ in anchors], [[] for _ in anchors]


def provider(name, mechanism, layer, token, **kwargs):
    return PortfolioProvider(
        name, mechanism, REVISION, FakeDraft(layer, token), **kwargs
    )


def prediction(name, mechanism, expected, *, fallback=False, observations=0, **kwargs):
    return {
        "source_id": name,
        "mechanism": mechanism,
        "expected_accepted_tokens": expected,
        "observations": observations,
        "lane_cost": kwargs.get("lane_cost", 1),
        "tile_rows": kwargs.get("tile_rows", 1),
        "fallback": fallback,
    }


def test_portfolio_supports_five_providers_and_order_invariant_private_rngs():
    records = tuple(
        provider(f"p{index}", mechanism, index, index + 1)
        for index, mechanism in enumerate(
            ("xpress", "lilicorr", "dflash2", "dflash2_fold", "dspark")
        )
    )
    portfolio = DrafterPortfolio(records)
    assert portfolio.config.target_layer_ids == (0, 1, 2, 3, 4)
    assert len(portfolio.make_cache()) == 5
    assert [row["mechanism"] for row in portfolio.receipt_settings["providers"]] == [
        "xpress",
        "lilicorr",
        "dflash2",
        "dflash2_fold",
        "dspark",
    ]

    rng = RequestRNG(7)
    portfolio.draft_distributions(
        [1], mx.zeros((1, 1, 10)), portfolio.make_cache(), 2, [rng], [0.0]
    )
    assert len({record.draft.draws[0] for record in records}) == 1
    assert rng.draws == 1


def test_admission_respects_lanes_tiles_and_uses_pld_only_as_fallback():
    planner = DrafterAdmissionPlanner(
        {
            "max_active_providers": 5,
            "lane_slots": 3,
            "tile_rows": 5,
            "minimum_observations": 8,
        }
    )
    result = planner.choose(
        "request",
        [
            prediction("wide", "dflash2", 3.0, lane_cost=2, tile_rows=4),
            prediction("compact", "lilicorr", 2.0, tile_rows=1),
            prediction("extra", "dspark", 1.0, lane_cost=2, tile_rows=2),
            prediction("pld", "prompt_lookup", 9.0, fallback=True),
        ],
    )
    assert result["admitted"] == ("compact", "wide")
    assert result["lane_slots_used"] == 3
    assert result["tile_rows_used"] == 5
    assert not result["fallback_only"]


def test_low_acceptance_latch_parks_then_reprobes_before_pld_fallback():
    planner = DrafterAdmissionPlanner(
        {
            "minimum_expected_tokens": 0.5,
            "minimum_observations": 2,
            "park_rounds": 2,
            "reprobe_interval": 2,
        }
    )
    rows = [
        prediction("fold", "dflash2_fold", 0.1, observations=9),
        prediction("pld", "prompt_lookup", 0.2, fallback=True),
    ]
    first = planner.choose("request", rows)
    second = planner.choose("request", rows)
    reprobe = planner.choose("request", rows)
    assert first["admitted"] == second["admitted"] == ("pld",)
    assert first["parked"][0]["source_id"] == "fold"
    assert reprobe["admitted"] == ("fold",)
