"""Independent Qwen4 HC tap and cache-transaction oracles on CPU."""

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from test_batched_mtp import _tiny_qwen4_model

from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models.cache import ArraysCache


@pytest.fixture(autouse=True)
def cpu():
    before = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(before)


def _manual_mix(mixer, hyper):
    # Independent expansion of the existing learned HC mix. The declared
    # feature convention includes its grouped normalization and learned gates.
    normed = mixer.hc_norm(hyper)
    gates = mx.sigmoid(
        mixer.input_mix_weight_up(
            nn.silu(mixer.input_mix_weight_down(normed) / mixer.hc_count)
        )
    )
    shape = (*hyper.shape[:-1], mixer.hc_count, mixer.hidden_size)
    return mx.mean(gates.reshape(shape) * normed.reshape(shape), axis=-2)


def _manual_body(model, inputs, caches, layers):
    from mlx2.runtime.models.base import create_attention_mask, create_ssm_mask

    body = model.language_model.model
    hidden = mx.tile(body.embed_tokens(inputs), (1, 1, body.args.hc_count))
    fa_mask = create_attention_mask(hidden, caches[body.fa_idx], return_array=True)
    if fa_mask is not None and fa_mask.ndim == 2:
        fa_mask = fa_mask[None, None]
    ssm_mask = create_ssm_mask(hidden, caches[body.ssm_idx])
    taps = []
    for index, (layer, cache) in enumerate(zip(body.layers, caches)):
        hidden = layer(hidden, inputs, fa_mask, cache, ssm_mask)
        if index in layers:
            taps.append(_manual_mix(body.hyper_connection_mixer, hidden))
    mixed = _manual_mix(body.hyper_connection_mixer, hidden)
    language = model.language_model
    logits = (
        body.embed_tokens.as_linear(mixed)
        if language.args.tie_word_embeddings
        else language.lm_head(mixed)
    )
    return logits, mx.concatenate(taps, axis=-1)


def _assert_close(actual, expected):
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=3e-5)


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_taps_use_learned_shared_hc_mixer_and_preserve_target_logits(
    monkeypatch, short, tied
):
    monkeypatch.setattr(qwen4_exp, "_SHAPE_STABLE_SHORT_FORWARD", short)
    mx.random.seed(142)
    model = _tiny_qwen4_model()
    model.language_model.args.tie_word_embeddings = tied
    assert model.external_feature_convention == "post_layer_shared_hc_mixer"
    inputs = mx.array([[1, 5, 8, 7]])
    cache, reference = model.make_cache(), model.make_cache()
    actual, features = model.forward_with_taps(inputs, cache, [0, 1])
    if short:
        outputs = [
            _manual_body(model, inputs[:, i : i + 1], reference, [0, 1])
            for i in range(4)
        ]
        expected = tuple(
            mx.concatenate([output[i] for output in outputs], axis=1) for i in range(2)
        )
    else:
        expected = _manual_body(model, inputs, reference, [0, 1])
    _assert_close(actual, expected[0])
    _assert_close(features, expected[1])
    assert features.shape == (1, 4, 64)
    ordinary = model(inputs, cache=model.make_cache())
    _assert_close(actual, ordinary)


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("lengths", [[3, 3], [3, 2]])
def test_b2_transaction_taps_match_independent_kept_prefix_oracles(
    monkeypatch, short, lengths
):
    monkeypatch.setattr(qwen4_exp, "_SHAPE_STABLE_SHORT_FORWARD", short)
    mx.random.seed(153)
    model = _tiny_qwen4_model()
    prompts = [[2, 7, 3, 1], [6, 8, 3, 9, 5, 4, 2]]
    rows, oracles = [], []
    for prompt in prompts:
        row, oracle = model.make_cache(), model.make_cache()
        model(mx.array([prompt]), cache=row)
        model(mx.array([prompt]), cache=oracle)
        rows.append(row)
        oracles.append(oracle)
    transaction = HybridVerifyRows(rows).begin(lengths)
    inputs = mx.array([[3, 2, 6], [7, 5, 0]])
    actual, features = model.forward_with_taps(inputs, transaction.caches, [0, 1])
    mx.eval(actual, features)
    # Compare each accepted prefix with an independent plain B1 cache and
    # manual body; this also checks the captured layers precede final logits.
    kept = [2, 1]
    expected = [
        _manual_body(model, inputs[index : index + 1, :count], oracle, [0, 1])
        for index, (count, oracle) in enumerate(zip(kept, oracles))
    ]
    transaction.commit(kept)
    assert features.shape == (2, 3, 64)
    for index, count in enumerate(kept):
        _assert_close(actual[index : index + 1, :count], expected[index][0])
        _assert_close(features[index : index + 1, :count], expected[index][1])
        for live, oracle in zip(rows[index], oracles[index]):
            if isinstance(live, ArraysCache):
                for value, wanted in zip(live.cache, oracle.cache):
                    if value is not None or wanted is not None:
                        _assert_close(value, wanted)
                assert not live.speculating and not live._rollbacks
            else:
                assert live.offset == oracle.offset == len(prompts[index]) + count
                assert live.index_keys.shape[1] == live.offset
                _assert_close(
                    live.keys[:, :, : live.offset], oracle.keys[:, :, : oracle.offset]
                )
                _assert_close(
                    live.values[:, :, : live.offset],
                    oracle.values[:, :, : oracle.offset],
                )
                _assert_close(live.index_keys, oracle.index_keys)
                if (
                    live._qsa_pooled_keys is not None
                    or oracle._qsa_pooled_keys is not None
                ):
                    _assert_close(live._qsa_pooled_keys, oracle._qsa_pooled_keys)
                assert not live._mtp_share_topk and live._mtp_shared_topk is None


def test_body_only_skips_lm_head_and_capture_validation_precedes_state_mutation(
    monkeypatch,
):
    model = _tiny_qwen4_model()
    cache = model.make_cache()
    inputs = mx.array([[1, 3]])
    before = [entry.empty() for entry in cache]
    for layers in ([], [1, 0], [0, 0], [-1], [2], [True]):
        with pytest.raises(ValueError, match="capture"):
            model.forward_with_taps(inputs, cache, layers)
        assert [entry.empty() for entry in cache] == before
    with pytest.raises(ValueError, match="cover every layer"):
        model.forward_with_taps(inputs, cache[:1], [0])

    def forbidden(*args, **kwargs):
        raise AssertionError("body-only prefill reached LM head")

    monkeypatch.setattr(model.language_model, "logits", forbidden)
    features = model.prefill_body(inputs, cache, [0, 1])
    mx.eval(features)
    assert features.shape == (1, 2, 64)
