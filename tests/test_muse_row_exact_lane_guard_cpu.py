"""Muse row-exact verify and an installed lane law: the guard's contract.

``row_exact_qmv`` gives every multi-row verify row the stock one-row qmv
arithmetic and bypasses module ``__call__``.  Before bbc8f0bad that made any
lane law over one-row calls (lane ``exact``, or a q4 threshold of one row)
unsound: ordinary decode ran the lane law while verify rows ran stock
arithmetic, and the guard refused it.

Muse's verify now follows the installed law per projection: a lane-installed
module whose one-row call runs the lane law (min_rows 1) is called itself
over every verify row (row-invariant lane law), and every other projection
keeps the stock one-row arithmetic its one-token call runs.  The model names
those projections (``row_exact_lane_followers``) and the guard admits lane
policies over exactly them; any covered projection outside that set, or a
model that declares none, is still refused.  The bit-level half of the
contract (verify rows == one-token decode under the installed law) is the
Metal gate tests/test_muse_row_exact_stacked_metal.py.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.lane.policy import detect, resolve
from mlx2.runtime.models import muse_glimmer as MG
from mlx2.runtime.models.muse_glimmer import Model
from mlx2.serving import row_exact_target_mutation_guard


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def q4_muse_row_exact_target(tie=False):
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
            tie_word_embeddings=tie,
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


def _policies(model):
    exact = resolve(detect(model), family="muse-glimmer", mode="exact")
    one_row = resolve(
        detect(model), family="muse-glimmer", mode="crossover",
        overrides={"min_rows": {"q4": 1}},
    )
    crossover = resolve(detect(model), family="muse-glimmer", mode="crossover")
    return exact, one_row, crossover


def _projections(model):
    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.QuantizedLinear, nn.Linear))
    }


def test_followers_are_every_projection_verify_reaches():
    """The declaration must name every covered projection, and only those."""
    for tie in (False, True):
        model = q4_muse_row_exact_target(tie).model
        followers = set(model.row_exact_lane_followers())
        assert followers == _projections(model)
        assert ("lm_head" in followers) is (not tie)


def test_every_admissible_lane_policy_is_admitted_without_mutation(monkeypatch):
    adapter = q4_muse_row_exact_target()
    assert detect(adapter.model)["formats"] == {"q4": 17}
    exact, one_row, crossover = _policies(adapter.model)
    classes = [(name, type(module)) for name, module in adapter.model.named_modules()]
    monkeypatch.setattr("mlx2.runtime.lane.available", lambda: True)
    for policy in (
        {**exact, "mode": "off"},
        {**exact, "skip": ["*"]},
        crossover,
        exact,
        one_row,
    ):
        row_exact_target_mutation_guard(adapter, lane_policy=policy)
    assert classes == [
        (name, type(module)) for name, module in adapter.model.named_modules()
    ]


@pytest.mark.parametrize("dropped", ["lm_head", "model.layers.1.mlp.down_proj"])
def test_falsifier_a_projection_verify_bypasses_is_refused(monkeypatch, dropped):
    """A covered projection outside the declaration (one whose verify call
    would not follow the law) keeps the refusal under a one-row lane law."""
    adapter = q4_muse_row_exact_target()
    exact, one_row, crossover = _policies(adapter.model)
    full = adapter.model.row_exact_lane_followers()
    monkeypatch.setattr(
        type(adapter.model),
        "row_exact_lane_followers",
        lambda self: tuple(name for name in full if name != dropped),
    )
    monkeypatch.setattr("mlx2.runtime.lane.available", lambda: True)
    row_exact_target_mutation_guard(adapter, lane_policy=crossover)
    for policy in (exact, one_row):
        with pytest.raises(ValueError, match="native projections"):
            row_exact_target_mutation_guard(adapter, lane_policy=policy)
    # Skipping the bypassed projection keeps it on stock at every width.
    row_exact_target_mutation_guard(adapter, lane_policy={**exact, "skip": [dropped]})


def test_falsifier_a_model_without_the_declaration_keeps_the_old_refusal(monkeypatch):
    adapter = q4_muse_row_exact_target()
    exact, one_row, crossover = _policies(adapter.model)
    monkeypatch.delattr(MG.Model, "row_exact_lane_followers")
    monkeypatch.setattr("mlx2.runtime.lane.available", lambda: True)
    row_exact_target_mutation_guard(adapter, lane_policy=crossover)
    for policy in (exact, one_row):
        with pytest.raises(ValueError, match="native projections"):
            row_exact_target_mutation_guard(adapter, lane_policy=policy)


def test_verify_follows_the_law_per_module(monkeypatch):
    """Routing half of the contract: a projection whose one-row call runs the
    lane law is called itself; any other takes the stock one-row form."""
    adapter = q4_muse_row_exact_target()
    lane = {"model.layers.0.mlp.down_proj"}
    names = {id(m): n for n, m in adapter.model.named_modules()}
    monkeypatch.setattr(MG, "_lane_exact", lambda module, value: names.get(id(module)) in lane)
    from mlx2.runtime.models import row_exact_qmv as REQ

    stock_calls = []
    real = REQ.quantized_linear

    def counted(linear, x, call=None):
        stock_calls.append(names.get(id(linear)))
        return real(linear, x, call=call)

    monkeypatch.setattr(REQ, "quantized_linear", counted)
    from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows

    cache = adapter.model.make_cache()
    mx.eval(adapter.model(mx.array([[1, 2, 3]]), cache=cache))
    tx = SegmentedKVRows([cache]).begin([2])
    MG.ROW_EXACT_ROUTES.clear()
    adapter.model.forward_with_taps(mx.array([[4, 5]]), tx.caches, (0, 1))
    tx.abort()
    assert "model.layers.0.mlp.down_proj" not in stock_calls
    assert "model.layers.1.mlp.down_proj" in stock_calls
    assert MG.ROW_EXACT_ROUTES["projection:lane_module"] == 1
