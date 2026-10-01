"""Prefix-only diffusion mechanics, not proposal quality qualification."""

from dataclasses import replace

import pytest

mx = pytest.importorskip("mlx.core")
from mlx import nn
from mlx.utils import tree_flatten

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.model import DiffusionStudent, Model
from mlx2.experimental.hysparse2.train import loss


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(42)
    yield
    mx.set_default_device(previous)


def test_future_teacher_states_are_not_used():
    c = replace(
        Config.smoke(),
        diffusion_conditioning="prefix",
        diffusion_trunk_gradient_scale=1.0,
    )
    model = Model(c)
    student = DiffusionStudent(c)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    teacher = mx.random.normal((1, 8, c.hidden_size))
    a, mask = student(tokens, model.embedding, teacher)
    changed = mx.concatenate([teacher[:, :4], teacher[:, 4:] + 100], axis=1)
    b, other_mask = student(tokens, model.embedding, changed)
    assert a.shape == (1, 4, c.vocab_size)
    assert bool(mx.all(mask == other_mask).item())
    assert float(mx.max(mx.abs(a - b)).item()) == 0
    _, gradients = nn.value_and_grad(model, loss)(model, tokens)
    for key in [
        "self_decoder.0.attention.q.weight",
        "diffusion_student.condition.weight",
    ]:
        assert float(mx.sum(mx.abs(dict(tree_flatten(gradients))[key])).item()) > 0


def test_fully_masked_suffix_can_distinguish_positions():
    c = replace(Config.smoke(), diffusion_conditioning="prefix")
    legacy = Model(c)
    positioned = Model(replace(c, diffusion_position_encoding="sinusoidal"))
    positioned.load_weights(tree_flatten(legacy.parameters()), strict=True)
    tokens = mx.ones((1, 8), dtype=mx.int32)
    mask = mx.ones(tokens.shape, dtype=mx.bool_)
    teacher = mx.ones((1, 1, c.hidden_size))
    level = mx.ones((1, 1, 1))
    a = legacy.diffusion_student.denoise(tokens, legacy.embedding, teacher, mask, level)
    b = positioned.diffusion_student.denoise(
        tokens, positioned.embedding, teacher, mask, level
    )
    assert float(mx.max(mx.abs(a[:, :1] - a[:, 1:])).item()) < 1e-6
    assert float(mx.max(mx.abs(b[:, :1] - b[:, 1:])).item()) > 1e-4
    legacy.eval()
    positioned.eval()
    assert float(mx.max(mx.abs(legacy(tokens)[0] - positioned(tokens)[0])).item()) == 0
    with pytest.raises(ValueError, match="position encoding"):
        replace(c, diffusion_position_encoding="invalid")


def test_bounded_proposal_preserves_causal_reference_and_cache():
    c = replace(Config.smoke(), diffusion_conditioning="prefix")
    model = Model(c)
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    _, cache = model.prefill(tokens)
    boundary = cache.boundary
    calls = (cache.self_layer_calls, cache.cross_layer_calls)
    proposal, receipt = model.diffusion_propose(cache, count=4, steps=3)
    repeat, _ = model.diffusion_propose(cache, count=4, steps=3)
    assert proposal.shape == (2, 4) and bool(mx.all(proposal == repeat).item())
    assert cache.length == 5 and cache.boundary is boundary
    assert (cache.self_layer_calls, cache.cross_layer_calls) == calls
    assert not receipt["target_verified"] and not receipt["kv_committed"]
    for count, steps in [(0, 1), (65, 1), (4, 0), (4, 17)]:
        with pytest.raises(ValueError):
            model.diffusion_propose(cache, count=count, steps=steps)
    historical = Model(Config.smoke())
    historical.load_weights(tree_flatten(model.parameters()), strict=True)
    historical.eval()
    assert float(mx.max(mx.abs(model(tokens)[0] - historical(tokens)[0])).item()) == 0
    with pytest.raises(ValueError, match="prefix-conditioned"):
        historical.diffusion_propose(cache)


@pytest.mark.parametrize("damage", ["missing_layer", "offset", "boundary", "history"])
def test_incomplete_prefix_rejected_before_diffusion(damage):
    model = Model(replace(Config.smoke(), diffusion_conditioning="prefix"))
    model.eval()
    _, cache = model.prefill(mx.array([[1, 2, 3, 4]]))
    if damage == "missing_layer":
        del cache.self_kv[0]
    elif damage == "offset":
        k, v, start = cache.cross_kv[0][0]
        cache.cross_kv[0][0] = k, v, start + 1
    elif damage == "boundary":
        cache.boundary = cache.boundary[:, :, :1]
    else:
        cache.ple_history = None
    before = (cache.length, cache.self_layer_calls, cache.cross_layer_calls)
    with pytest.raises(ValueError, match="endpoint state"):
        model.diffusion_propose(cache)
    assert before == (cache.length, cache.self_layer_calls, cache.cross_layer_calls)
