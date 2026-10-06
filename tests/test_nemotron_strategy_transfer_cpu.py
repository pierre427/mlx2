"""CPU/static gates for the Nemotron-H exact-prefix strategy transfer."""

import copy

import numpy as np
import pytest

from mlx2.runtime.exact_prefix_cascade import (
    longest_first_paths,
    next_cascade_stage,
)


def _tiny_model():
    import mlx.core as mx

    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    return Model(
        ModelArgs.from_dict(
            dict(  # noqa: C408 - compact parity with existing tiny-model fixtures
                model_type="nemotron_h",
                vocab_size=64,
                hidden_size=16,
                intermediate_size=16,
                hybrid_override_pattern="ME*",
                mtp_hybrid_override_pattern="*E",
                num_hidden_layers=3,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                attention_bias=False,
                mamba_num_heads=2,
                mamba_head_dim=8,
                mamba_proj_bias=False,
                use_bias=False,
                ssm_state_size=8,
                conv_kernel=3,
                n_groups=1,
                chunk_size=128,
                use_conv_bias=True,
                time_step_limit=None,
                mlp_bias=False,
                mlp_hidden_act="relu2",
                layer_norm_epsilon=1e-5,
                n_routed_experts=2,
                num_experts_per_tok=1,
                n_group=1,
                topk_group=1,
                norm_topk_prob=True,
                routed_scaling_factor=5.0,
                n_shared_experts=1,
                moe_latent_size=8,
                moe_intermediate_size=16,
                moe_shared_expert_intermediate_size=16,
                num_nextn_predict_layers=1,
                max_position_embeddings=4096,
                rope_theta=10000,
                partial_rotary_factor=1.0,
                tie_word_embeddings=False,
            )
        )
    )


def _assert_cache_equal(left, right):
    import mlx.core as mx

    from mlx2.runtime.models.cache import ArraysCache

    assert [type(item) for item in left] == [type(item) for item in right]
    for actual, expected in zip(left, right):
        if isinstance(actual, ArraysCache):
            for a, e in zip(actual.cache, expected.cache):
                if a is None or e is None:
                    assert a is e
                else:
                    mx.eval(a, e)
                    np.testing.assert_allclose(np.asarray(a), np.asarray(e), atol=0, rtol=0)
        else:
            assert actual.offset == expected.offset
            for a, e in zip(actual.keys_and_values(), expected.keys_and_values()):
                mx.eval(a, e)
                np.testing.assert_allclose(np.asarray(a), np.asarray(e), atol=0, rtol=0)


def test_longest_first_prunes_siblings_outside_the_exact_prefix():
    paths = [(1, 2), (1, 2, 3, 4), (1, 7, 8), (1, 2, 9), (1, 2)]
    assert longest_first_paths(paths) == (
        (1, 2, 3, 4),
        (1, 7, 8),
        (1, 2, 9),
        (1, 2),
    )
    stage = next_cascade_stage(paths, (1, 2), attempted=(0,))
    assert stage.path == (1, 2, 9)
    assert stage.suffix == (9,)
    assert stage.viable_indices == (2,)
    assert set(stage.pruned_indices) == {1, 3}


@pytest.mark.parametrize("paths", [[], [()], [(1, -1)], [(True, 2)]])
def test_invalid_exact_prefix_paths_fail_closed(paths):
    with pytest.raises(ValueError):
        longest_first_paths(paths)


def test_nemotron_adapter_owns_prefill_and_declares_unqualified_state_boundary():
    from mlx2.adapters.nemotron3_super import Nemotron3SuperAdapter
    from mlx2.adapters.nemotron35_lightning import Nemotron35LightningAdapter

    for adapter_type in (Nemotron3SuperAdapter, Nemotron35LightningAdapter):
        adapter = object.__new__(adapter_type)
        assert adapter.prefill_step_default() == 2048
        contract = adapter.exact_prefix_cascade_contract()
        assert contract["accepted_prefix_state"] == "b1_tokenwise_hybrid_transaction"
        assert contract["common_tokens_recomputed"] is False
        assert contract["mtp_cache_reuse"] is False
        assert contract["twotower_reuse"] is False
        assert contract["qualified"] is contract["selected"] is False


def test_exact_b1_prefix_is_forked_without_recomputing_common_tokens_cpu():
    import mlx.core as mx

    from mlx2.runtime.nemotron_prefix_reuse import verify_longest_prefix

    model = _tiny_model()
    base = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]], dtype=mx.uint32), cache=base))
    reference = copy.deepcopy(base)
    mx.eval(model(mx.array([[4]], dtype=mx.uint32), cache=reference))
    mx.eval(model(mx.array([[5]], dtype=mx.uint32), cache=reference))

    verification = verify_longest_prefix(model, copy.deepcopy(base), [4, 5, 6])
    assert verification.logits.shape == (1, 3, 64)
    branches, receipt = verification.commit_and_fork(2, sibling_count=2)
    assert receipt["accepted_tokens"] == 2
    assert receipt["branches"] == 2
    assert receipt["common_tokens_recomputed"] == 0
    assert receipt["recurrent_layers"] == 1
    assert receipt["attention_layers"] == 1
    assert receipt["apcv2_published"] is False
    _assert_cache_equal(branches[0], reference)
    _assert_cache_equal(branches[1], reference)

    untouched = copy.deepcopy(branches[1])
    expected = model(mx.array([[9]], dtype=mx.uint32), cache=reference)
    actual = model(mx.array([[9]], dtype=mx.uint32), cache=branches[0])
    mx.eval(expected, actual)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0, rtol=0)
    _assert_cache_equal(branches[0], reference)
    _assert_cache_equal(branches[1], untouched)


def test_nemotron_receipt_scopes_direct_use_separately_from_serving(monkeypatch):
    from mlx2.runtime import nemotron_prefix_reuse as prefix_reuse

    class Transaction:
        closed = False

        def commit(self, accepted):
            assert accepted == [2]
            self.closed = True
            return [[{"state": "committed"}]]

    monkeypatch.setattr(
        prefix_reuse,
        "_validate_target_geometry",
        lambda model, cache: (23, 6),
    )
    verification = prefix_reuse.NemotronPrefixVerification(
        model=object(),
        transaction=Transaction(),
        logits=None,
        features=None,
        verified_tokens=(11, 12, 13),
        recurrent_layers=23,
        attention_layers=6,
    )

    branches, receipt = verification.commit_and_fork(2, sibling_count=2)

    assert len(branches) == 2
    assert receipt["schema"] == "mlx2.nemotron-exact-prefix-reuse.v2"
    assert receipt["observed_used"] is True
    assert receipt["observed_use_scope"] == (
        "direct_request_private_prefix_primitive"
    )
    assert receipt["serving_route_implemented"] is False
    assert receipt["observed_used_in_serving"] is False
    assert receipt["qualified"] is receipt["selected"] is False


def test_mtp_and_twotower_cache_boundaries_are_refused_cpu():
    from mlx2.runtime.nemotron_prefix_reuse import (
        NemotronPrefixReuseUnsupported,
        verify_longest_prefix,
    )

    model = _tiny_model()
    with pytest.raises(NemotronPrefixReuseUnsupported, match="target-layer topology"):
        verify_longest_prefix(model, model.make_mtp_cache(), [1])
    model.model_type = "nemotron_twotower"
    with pytest.raises(NemotronPrefixReuseUnsupported, match="not TwoTower"):
        verify_longest_prefix(model, model.make_cache(), [1])
