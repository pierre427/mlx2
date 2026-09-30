"""CPU episode contracts; GPU mechanics use the same checks on M3."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten

from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.lora import LoRAEpisode
from mlx2.experimental.hysparse2.model import Model
from mlx2.experimental.hysparse2.train import loss


@pytest.fixture(autouse=True)
def cpu():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(42)
    yield
    mx.set_default_device(device)


def test_episode_updates_only_lora_and_rolls_back(tmp_path):
    model = Model(Config.smoke())
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    original = dict(tree_flatten(model.parameters()))
    mx.eval(original)
    baseline = model(tokens)[0]
    mx.eval(baseline)
    _, cache = model.prefill(tokens[:, :4])
    episode = LoRAEpisode(
        model,
        ["self_decoder.0.attention.q", "semantic_ple.value"],
        base_revision="checkpoint-test",
    )
    assert set(dict(tree_flatten(model.trainable_parameters()))) == set(
        episode.weights()
    )
    model.eval()
    assert float(mx.max(mx.abs(model(tokens)[0] - baseline)).item()) == 0
    with pytest.raises(ValueError, match="another model"):
        model.decode(tokens[:, 4:5], cache)
    _, active_cache = model.prefill(tokens[:, :4])
    revision = model.adapter_revision
    episode.step(tokens, loss)
    assert model.adapter_revision != revision
    model.eval()
    with pytest.raises(ValueError, match="another model"):
        model.decode(tokens[:, 4:5], active_cache)
    assert float(mx.max(mx.abs(model(tokens)[0] - baseline)).item()) > 0
    for key, base in episode.originals.items():
        assert (
            float(mx.max(mx.abs(base.weight - original[key + ".weight"])).item()) == 0
        )
    episode.export(tmp_path / "candidate")
    episode.close()
    assert model.adapter_revision is None
    assert model.training is False
    assert set(dict(tree_flatten(model.parameters()))) == set(original)
    assert float(mx.max(mx.abs(model(tokens)[0] - baseline)).item()) == 0
    episode.close()


def test_reject_and_promote_gates():
    model = Model(Config.smoke())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    episode = LoRAEpisode(model, ["self_decoder.0.attention.q"], base_revision="r")
    try:
        episode.step(tokens, loss)

        def constant(_model, _tokens):
            return mx.array(1.0)

        assert not episode.evaluate(tokens, tokens, constant)["promoted"]

        def witness(m, t):
            # Gate mechanics only: independently supplied evaluator detects wrappers.
            return mx.array(
                0.5 if hasattr(m.self_decoder[0].attention.q, "lora_a") else 1.0
            )

        assert episode.evaluate(tokens, tokens, witness)["promoted"]
    finally:
        episode.close()
    assert hasattr(model.self_decoder[0].attention.q, "lora_a")
    assert model.adapter_revision is not None
    episode.rollback()
    assert model.adapter_revision is None
    assert not hasattr(model.self_decoder[0].attention.q, "lora_a")


def test_invalid_keys_do_not_mutate():
    model = Model(Config.smoke())
    keys = set(dict(tree_flatten(model.parameters())))
    with pytest.raises(ValueError):
        LoRAEpisode(model, ["semantic_ple.embedding"], base_revision="r")
    assert set(dict(tree_flatten(model.parameters()))) == keys


@pytest.mark.parametrize("fail_baseline", [False, True])
def test_evaluation_baseline_state_cannot_escape(fail_baseline):
    model = Model(Config.smoke())
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    episode = LoRAEpisode(model, ["self_decoder.0.attention.q"], base_revision="r")
    try:
        episode.step(tokens, loss)
        revision = model.adapter_revision
        captured = []

        def evaluate(m, t):
            _, cache = m.prefill(t[:, :4])
            baseline = not hasattr(m.self_decoder[0].attention.q, "lora_a")
            captured.append((baseline, m.adapter_revision, cache))
            if baseline and fail_baseline:
                raise RuntimeError("evaluator failed")
            return mx.array(1.0)

        if fail_baseline:
            with pytest.raises(RuntimeError, match="evaluator failed"):
                episode.evaluate(tokens, tokens, evaluate)
        else:
            episode.evaluate(tokens, tokens, evaluate)
        assert model.adapter_revision == revision
        assert hasattr(model.self_decoder[0].attention.q, "lora_a")
        for baseline, observed_revision, cache in captured:
            assert observed_revision == (None if baseline else revision)
            with pytest.raises(ValueError, match="another model"):
                model.decode(tokens[:, 4:5], cache)
        _, valid = model.prefill(tokens[:, :4])
        model.decode(tokens[:, 4:5], valid)
    finally:
        episode.close()


def test_candidate_revision_and_budget(tmp_path):
    model = Model(Config.smoke())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    episode = LoRAEpisode(
        model, ["self_decoder.0.attention.q"], base_revision="r", max_steps=1
    )
    episode.step(tokens, loss)
    with pytest.raises(ValueError, match="trainable"):
        episode.step(tokens, loss)
    episode.export(tmp_path / "candidate")
    episode.close()
    with pytest.raises(ValueError, match="base revision"):
        LoRAEpisode.load_candidate(model, tmp_path / "candidate", base_revision="wrong")
    loaded = LoRAEpisode.load_candidate(
        model, tmp_path / "candidate", base_revision="r"
    )
    assert loaded.steps == 1
    loaded.close()
    import json

    path = tmp_path / "candidate" / "adapter_config.json"
    config = json.loads(path.read_text())
    config["lora_parameters"]["scale"] *= 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="content revision"):
        LoRAEpisode.load_candidate(model, tmp_path / "candidate", base_revision="r")
    assert model.adapter_revision is None
