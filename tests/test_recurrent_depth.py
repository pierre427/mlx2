"""CPU-safe contract tests for model-agnostic recurrent-depth orchestration."""

import pytest

from mlx2.runtime.recurrent_depth import (
    RecurrentDepthConfig,
    recurrent_depth_forward,
)


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def recurrent_depth_hidden(self, inputs, *, cache, input_embeddings=None, **kwargs):
        hidden = inputs if input_embeddings is None else input_embeddings
        result = hidden + cache["increment"]
        self.calls.append(
            {
                "cache": cache["name"],
                "input_embeddings": input_embeddings,
                "kwargs": kwargs,
                "result": result,
            }
        )
        return result

    def recurrent_depth_logits(self, hidden):
        return hidden * 10


def test_recurrent_depth_chains_hidden_state_and_isolates_conditioning():
    adapter = FakeAdapter()
    caches = [
        {"name": "pass-1", "increment": 2},
        {"name": "pass-2", "increment": 3},
    ]

    output = recurrent_depth_forward(
        adapter,
        5,
        caches,
        config=RecurrentDepthConfig(passes=2),
        first_pass_kwargs={"capsule": "memory"},
    )

    assert output.hidden == 10
    assert output.logits == 100
    assert adapter.calls == [
        {
            "cache": "pass-1",
            "input_embeddings": None,
            "kwargs": {"capsule": "memory"},
            "result": 7,
        },
        {
            "cache": "pass-2",
            "input_embeddings": 7,
            "kwargs": {},
            "result": 10,
        },
    ]
    assert output.receipt == {
        "schema": "mlx2.recurrent-depth-forward.v1",
        "passes": 2,
        "input_mode": "direct_hidden",
        "cache_stacks": 2,
        "conditioned_passes": 1,
    }


@pytest.mark.parametrize(
    ("passes", "error"),
    [(False, TypeError), (0, ValueError), (9, ValueError), (2.0, TypeError)],
)
def test_recurrent_depth_rejects_invalid_pass_counts(passes, error):
    with pytest.raises(error, match="passes"):
        RecurrentDepthConfig(passes=passes)


def test_recurrent_depth_rejects_shared_or_wrong_cache_stacks():
    adapter = FakeAdapter()
    shared = {"name": "shared", "increment": 1}
    config = RecurrentDepthConfig(passes=2)
    with pytest.raises(ValueError, match="cache count"):
        recurrent_depth_forward(adapter, 1, [shared], config=config)
    with pytest.raises(ValueError, match="distinct cache"):
        recurrent_depth_forward(adapter, 1, [shared, shared], config=config)


def test_recurrent_depth_fails_closed_without_adapter_seams():
    with pytest.raises(TypeError, match="does not expose"):
        recurrent_depth_forward(
            object(),
            1,
            [{"pass": 1}],
            config=RecurrentDepthConfig(passes=1),
        )


def test_recurrent_depth_rejects_reserved_conditioning_keys():
    with pytest.raises(ValueError, match="may not override"):
        recurrent_depth_forward(
            FakeAdapter(),
            1,
            [{"name": "pass-1", "increment": 1}],
            config=RecurrentDepthConfig(passes=1),
            first_pass_kwargs={"cache": "wrong"},
        )
