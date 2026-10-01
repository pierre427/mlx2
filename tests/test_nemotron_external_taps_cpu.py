"""Real tiny Nemotron-H Mamba2 target tap/rollback contracts, CPU only."""

import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.nemotron_h import Model, ModelArgs


def tiny_model():
    mx.random.seed(28)
    args = ModelArgs(
        model_type="nemotron_h",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=5,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_bias=False,
        mamba_num_heads=2,
        mamba_head_dim=4,
        mamba_proj_bias=False,
        ssm_state_size=4,
        conv_kernel=3,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        hybrid_override_pattern=["-", "M", "*", "-", "M"],
    )
    return Model(args)


def assert_state_equal(actual, expected, tolerance=2e-6):
    for a, e in zip(actual, expected):
        if isinstance(a, ArraysCache):
            assert isinstance(e, ArraysCache)
            for av, ev in zip(a.cache, e.cache):
                if av is None or ev is None:
                    assert av is ev
                else:
                    np.testing.assert_allclose(
                        np.asarray(av), np.asarray(ev), rtol=1e-5, atol=tolerance
                    )
        else:
            assert a.offset == e.offset
            for av, ev in zip(a.state, e.state):
                np.testing.assert_allclose(
                    np.asarray(av)[..., : a.offset, :],
                    np.asarray(ev)[..., : e.offset, :],
                    rtol=1e-5,
                    atol=tolerance,
                )


def test_compact_cache_mapping_and_postlayer_taps_match_independent_layer_walk():
    from mlx2.runtime.models.base import create_attention_mask, create_ssm_mask

    model = tiny_model()
    tokens = mx.array([[1, 2, 3]])
    caches = model.make_cache()
    assert [type(c) for c in caches] == [ArraysCache, KVCache, ArraysCache]
    x = model.backbone.embeddings(tokens)
    masks = {
        "*": create_attention_mask(x, caches[1]),
        "M": create_ssm_mask(x, caches[0]),
    }
    captures = []
    index = 0
    for layer_id, layer in enumerate(model.layers):
        c = caches[index] if layer.block_type in ("M", "*") else None
        if c is not None:
            index += 1
        x = layer(x, mask=masks.get(layer.block_type), cache=c)
        if layer_id in (0, 1, 4):
            captures.append(x)
    expected = mx.concatenate(captures, -1)
    logits, taps = model.forward_with_taps(tokens, model.make_cache(), [0, 1, 4])
    np.testing.assert_array_equal(np.asarray(taps), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(logits), np.asarray(model(tokens, model.make_cache()))
    )
    np.testing.assert_array_equal(
        np.asarray(model.prefill_body(tokens, model.make_cache(), [0, 1, 4])),
        np.asarray(expected),
    )
    assert taps.shape == (1, 3, 24)
    from mlx2.adapters.xpress import XPressConfig
    from mlx2.runtime.drafters.xpress import XPressDraftModel

    draft = XPressDraftModel(
        XPressConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=16,
            mask_token_id=15,
            num_target_layers=5,
            target_layer_ids=[0, 1, 4],
            block_size=4,
            layer_types=["full_attention"],
            xpress_rank=3,
            xpress_mlp_hidden=5,
            xpress_num_passes=3,
        )
    ).bind(model)
    assert (
        draft.embed_tokens is model.backbone.embeddings
        and draft.lm_head is model.lm_head
    )
    from mlx.utils import tree_flatten

    names = {name for name, _ in tree_flatten(model.parameters())}
    assert "backbone.embeddings.weight" in names
    assert not any(name.startswith("embed_tokens.") for name in names)
    for invalid in ([], [1, 0], [1, 1], [5], [True]):
        with pytest.raises(ValueError, match="capture"):
            model.forward_with_taps(tokens, model.make_cache(), invalid)
    with pytest.raises(ValueError, match="topology"):
        model.forward_with_taps(tokens, [KVCache()] * 5, [0])


@pytest.mark.parametrize("keep", [0, 1, 2, 3])
def test_actual_mamba2_speculative_replay_matches_independent_accepted_prefix(keep):
    model = tiny_model()
    base = model.make_cache()
    model(mx.array([[1, 2, 3]]), base)
    reference = copy.deepcopy(base)
    candidate = copy.deepcopy(base)
    verify = mx.array([[4, 5, 6]])
    for index in range(keep):
        model(verify[:, index : index + 1], reference)
    for cache in candidate:
        cache.start_speculation()
    model.forward_with_taps(verify, candidate, [0, 1, 4])
    mx.eval([c.state for c in candidate])
    for cache in candidate:
        assert cache.trim(3 - keep) == 3 - keep
        cache.stop_speculation()
    assert_state_equal(candidate, reference)
    for actual, expected in zip(candidate, reference):
        if isinstance(actual, ArraysCache):
            for a, e in zip(actual.cache, expected.cache):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(e))
    np.testing.assert_allclose(
        np.asarray(model(mx.array([[7]]), candidate)),
        np.asarray(model(mx.array([[7]]), reference)),
        rtol=1e-5,
        atol=3e-6,
    )


@pytest.mark.parametrize("merged", [False, True])
def test_single_row_verification_matches_bitwise_serial_taps_and_states(merged):
    from mlx2.runtime.models.cache import BatchKVCache

    model = tiny_model()
    base = model.make_cache()
    model(mx.array([[1, 2, 3]]), base)
    reference = copy.deepcopy(base)
    candidate = copy.deepcopy(base)
    if merged:
        candidate = [type(c).merge([c]) for c in candidate]
        for c in candidate:
            c.prepare(lengths=[3], right_padding=[0])
        assert isinstance(candidate[1], BatchKVCache)
    for c in candidate:
        c.start_speculation()
    tokens = mx.array([[4, 5, 6]])
    serial_logits = []
    serial_taps = []
    for index in range(3):
        logits, taps = model.forward_with_taps(
            tokens[:, index : index + 1], reference, [0, 1, 4]
        )
        serial_logits.append(logits)
        serial_taps.append(taps)
    logits, taps = model.forward_with_taps(tokens, candidate, [0, 1, 4])
    np.testing.assert_array_equal(
        np.asarray(logits), np.asarray(mx.concatenate(serial_logits, 1))
    )
    np.testing.assert_array_equal(
        np.asarray(taps), np.asarray(mx.concatenate(serial_taps, 1))
    )
    for c in candidate:
        if isinstance(c, ArraysCache):
            assert c._rollback_position == 3 or c._rollback_positions == [3]
            if merged:
                assert c._length_vector() == [0]
        assert "_nemotron_unpadded_verify" not in c.__dict__
    for actual, expected in zip(candidate, reference):
        if isinstance(actual, ArraysCache):
            for a, e in zip(actual.cache, expected.cache):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(e))
        else:
            for a, e in zip(actual.state[:2], expected.state):
                np.testing.assert_array_equal(
                    np.asarray(a)[..., : actual.size(), :],
                    np.asarray(e)[..., : expected.size(), :],
                )
    assert model.external_execution_receipt["batched_numerical_qualification"] is False


@pytest.mark.parametrize("lengths,kept", [([3, 3], [1, 2]), ([3, 2], [1, 1])])
def test_actual_hybrid_cohort_different_contexts_and_ragged_rejection_replay(
    lengths, kept
):
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    model = tiny_model()
    prefixes = [[1, 2], [2, 3, 4, 5]]
    rows = []
    for prefix in prefixes:
        caches = model.make_cache()
        model(mx.array([prefix]), caches)
        rows.append(caches)
    references = [copy.deepcopy(row) for row in rows]
    verify = [[6, 7, 8], [8, 7, 6]]
    for row, ids, keep in zip(references, verify, kept):
        model(mx.array([ids[:keep]]), row)
    owner = HybridVerifyRows(rows)
    transaction = owner.begin(lengths)
    cache = transaction.caches
    logits, taps = model.forward_with_taps(mx.array(verify), cache, [0, 1, 4])
    mx.eval(logits, taps, [c.state for c in cache])
    restored = transaction.commit(kept)
    for actual, expected in zip(restored, references):
        assert_state_equal(actual, expected)
    for row, expected in zip(restored, references):
        np.testing.assert_allclose(
            np.asarray(model(mx.array([[9]]), row)),
            np.asarray(model(mx.array([[9]]), expected)),
            rtol=2e-5,
            atol=3e-6,
        )
    assert taps.shape == (2, 3, 24)
