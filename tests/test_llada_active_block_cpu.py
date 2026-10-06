"""CPU parity gates for the exact LLaDA active-block postprocessing candidate."""

import mlx.core as mx
import pytest

from mlx2.runtime.models.llada import Model, ModelArgs, generate


def _tiny_model():
    mx.random.seed(7)
    return Model(
        ModelArgs(
            model_type="llada",
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            mlp_hidden_size=64,
            vocab_size=64,
            embedding_size=64,
            weight_tying=False,
        )
    )


def _ids(value):
    return [int(token) for token in value.reshape(-1).tolist()]


def test_active_block_postprocessing_matches_full_row_reference_on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _tiny_model()
        prompt = mx.array([[1, 2, 3, 4]])
        common = {
            "steps": 8,
            "gen_length": 8,
            "block_length": 4,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "remasking": "low_confidence",
            "mask_id": 63,
            "return_stats": True,
        }
        reference, reference_stats = generate(model, prompt, **common)
        candidate, candidate_stats = generate(
            model, prompt, active_block_postprocess=True, **common
        )
    finally:
        mx.set_default_device(previous)

    assert _ids(candidate) == _ids(reference)
    assert reference_stats["active_block_postprocess"] is False
    assert reference_stats["postprocess_rows_per_forward"] == 12
    assert candidate_stats["active_block_postprocess"] is True
    assert candidate_stats["postprocess_rows_per_forward"] == 4
    assert candidate_stats["forwards"] == reference_stats["forwards"] == 8


def test_verified_active_head_matches_full_head_reference_on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _tiny_model()
        prompt = mx.array([[1, 2, 3, 4]])
        common = {
            "steps": 8,
            "gen_length": 8,
            "block_length": 4,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "remasking": "low_confidence",
            "mask_id": 63,
            "return_stats": True,
        }
        reference, _ = generate(model, prompt, **common)
        candidate, stats = generate(
            model,
            prompt,
            active_block_postprocess=True,
            active_block_head=True,
            verify_active_block_head=True,
            **common,
        )
    finally:
        mx.set_default_device(previous)

    assert _ids(candidate) == _ids(reference)
    assert stats["active_block_head"] is True
    assert stats["lm_head_rows_per_forward"] == 4
    assert stats["active_block_head_parity_checks"] == 8


def test_ordinary_generate_preserves_single_argument_model_contract():
    class SingleArgumentModel:
        def __call__(self, value):
            batch, rows = value.shape
            return mx.zeros((batch, rows, 64))

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        result = generate(
            SingleArgumentModel(),
            mx.array([[1, 2]]),
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=63,
        )
    finally:
        mx.set_default_device(previous)

    assert result.shape == (1, 4)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"temperature": 0.1}, "temperature == 0"),
        ({"parallel_threshold": 0.9}, "fixed schedule"),
        ({"remasking": "random"}, "low_confidence remasking"),
    ],
)
def test_active_block_candidate_fails_closed_outside_exact_scope(overrides, match):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        kwargs = {
            "steps": 4,
            "gen_length": 4,
            "block_length": 4,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "mask_id": 63,
            "active_block_postprocess": True,
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=match):
            generate(_tiny_model(), mx.array([[1, 2]]), **kwargs)
    finally:
        mx.set_default_device(previous)


def test_active_head_requires_active_postprocessing():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        with pytest.raises(ValueError, match="requires active_block_postprocess"):
            generate(
                _tiny_model(),
                mx.array([[1, 2]]),
                steps=4,
                gen_length=4,
                block_length=4,
                mask_id=63,
                active_block_head=True,
            )
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"verify_active_block_head": True}, "requires active_block_head"),
        ({"active_block_head": True, "kv_cache": True}, "does not support cache"),
        ({"active_block_head": True, "cfg_scale": 0.1}, "does not support cache"),
    ],
)
def test_active_head_fails_closed_for_unsupported_routes(overrides, match):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        kwargs = {
            "steps": 4,
            "gen_length": 4,
            "block_length": 4,
            "mask_id": 63,
            "active_block_postprocess": True,
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=match):
            generate(_tiny_model(), mx.array([[1, 2]]), **kwargs)
    finally:
        mx.set_default_device(previous)


def test_model_logit_slice_fails_closed_for_invalid_requests():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _tiny_model()
        inputs = mx.array([[1, 2, 3]])
        with pytest.raises(ValueError, match="requires logit_slice"):
            model(inputs, verify_logit_slice=True)
        with pytest.raises(ValueError, match="outside the hidden-state rows"):
            model(inputs, logit_slice=(2, 4))
        with pytest.raises(ValueError, match="incompatible with return_kv"):
            model(inputs, return_kv=True, logit_slice=(0, 1))
    finally:
        mx.set_default_device(previous)
