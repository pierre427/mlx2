"""Muse's q4 row-exact verify must refuse a lane law over its one-row calls.

``row_exact_qmv`` gives every multi-row verify row the stock one-row qmv
arithmetic of ordinary decode and bypasses module ``__call__``.  The guard only
looked at ``nn.Linear``, so lane ``exact`` (or a q4 threshold of one row)
installed over the q4 target: ordinary decode ran the lane law while verify
rows ran stock one-row arithmetic, and the row-exact receipt still claimed
equality.  A crossover that keeps one-row calls on stock is consistent.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.lane.policy import detect, resolve
from mlx2.runtime.models.muse_glimmer import Model
from mlx2.serving import row_exact_target_mutation_guard


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def q4_muse_row_exact_target():
    mx.random.seed(18)
    model = Model(
        ModelArgs(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            vocab_size=128,
            sliding_window=4,
            max_position_embeddings=2048,
            tie_word_embeddings=False,
        )
    )
    model.set_dtype(mx.bfloat16)
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda _name, module: isinstance(module, nn.Linear),
    )
    model.eval()
    mx.eval(model.parameters())
    model.configure_target_verify_row_exact(True)
    return SimpleNamespace(model=model, descriptor=SimpleNamespace(family="muse-glimmer"))


def test_lane_over_one_row_q4_calls_is_refused_for_a_row_exact_target(monkeypatch):
    adapter = q4_muse_row_exact_target()
    assert detect(adapter.model)["formats"] == {"q4": 17}
    exact = resolve(detect(adapter.model), family="muse-glimmer", mode="exact")
    one_row = resolve(
        detect(adapter.model), family="muse-glimmer", mode="crossover",
        overrides={"min_rows": {"q4": 1}},
    )
    crossover = resolve(detect(adapter.model), family="muse-glimmer", mode="crossover")
    classes = [(name, type(module)) for name, module in adapter.model.named_modules()]
    monkeypatch.setattr("mlx2.runtime.lane.available", lambda: True)
    # One-row calls stay stock: verify and decode keep one law.
    for allowed in ({**exact, "mode": "off"}, {**exact, "skip": ["*"]}, crossover):
        row_exact_target_mutation_guard(adapter, lane_policy=allowed)
    for policy in (exact, one_row):
        with pytest.raises(ValueError, match="native projections"):
            row_exact_target_mutation_guard(adapter, lane_policy=policy)
    assert classes == [
        (name, type(module)) for name, module in adapter.model.named_modules()
    ]
