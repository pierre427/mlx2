"""Repair-first CPU evidence for progressive external verify portability.

These tests deliberately do not claim qualification.  They establish that
the hybrid transaction can commit/abort exact prefixes, while the current
Qwen3.8 and Qwen3.6 target forwards do not supply the bit-exact wide-row law
required by progressive verification.  The adapters therefore remain
fail-closed instead of substituting a predictably slower serial target loop.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.adapters.qwen36_35b import EXTERNAL_POLICY_KEYS as QWEN36_EXTERNAL_KEYS
from mlx2.adapters.qwen38_27b import EXTERNAL_POLICY_KEYS as QWEN38_EXTERNAL_KEYS
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models.qwen36_35b import Model as Qwen36Model
from mlx2.runtime.models.qwen38_27b import ModelArgs
from test_qwen38_dflash2_cpu import tiny_pair, tiny_target


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny_qwen36(seed=7):
    text = dict(
        model_type="qwen3_5_moe",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5,
        rope_parameters=None,
        max_position_embeddings=512,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    mx.random.seed(seed)
    model = Qwen36Model(
        ModelArgs(model_type="qwen3_5_moe", text_config=text)
    )
    model.eval()
    mx.eval(model.parameters())
    return model


def _serial_rows(model, tokens, cache):
    logits, taps = [], []
    for index in range(int(tokens.shape[1])):
        row_logits, row_taps = model.forward_with_taps(
            tokens[:, index : index + 1], cache, [1, 6]
        )
        mx.eval(row_logits, row_taps)
        logits.append(row_logits)
        taps.append(row_taps)
    result = mx.concatenate(logits, axis=1), mx.concatenate(taps, axis=1)
    mx.eval(*result)
    return result


@pytest.mark.parametrize("factory", [tiny_target, tiny_qwen36])
def test_wide_target_rows_are_not_the_ordinary_b1_law(factory):
    model = factory()
    base = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=base))
    tokens = mx.array([[4, 5, 6, 7]])

    wide_logits, wide_taps = model.forward_with_taps(
        tokens, copy.deepcopy(base), [1, 6]
    )
    serial_logits, serial_taps = _serial_rows(
        model, tokens, copy.deepcopy(base)
    )
    mx.eval(wide_logits, wide_taps)

    # Progressive tiles must retain the ordinary B1 target law bit for bit.
    # Both current family implementations change reduction geometry at S>1.
    assert not np.array_equal(np.asarray(wide_logits), np.asarray(serial_logits))
    assert not np.array_equal(np.asarray(wide_taps), np.asarray(serial_taps))


def _next_logits(model, cache):
    value = model(mx.array([[9]]), cache=copy.deepcopy(cache))
    mx.eval(value)
    return np.asarray(value)


@pytest.mark.parametrize("factory", [tiny_target, tiny_qwen36])
def test_hybrid_transaction_partial_commit_and_abort_are_exact(factory):
    model = factory()
    base = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=base))
    tokens = mx.array([[4, 5, 6, 7]])

    owner = HybridVerifyRows([copy.deepcopy(base)])
    transaction = owner.begin([4])
    _serial_rows(model, tokens, transaction.caches)
    committed = transaction.commit([2])[0]
    reference = copy.deepcopy(base)
    for index in range(2):
        mx.eval(model(tokens[:, index : index + 1], cache=reference))
    np.testing.assert_array_equal(
        _next_logits(model, committed), _next_logits(model, reference)
    )

    owner = HybridVerifyRows([copy.deepcopy(base)])
    transaction = owner.begin([4])
    _serial_rows(model, tokens, transaction.caches)
    transaction.abort()
    np.testing.assert_array_equal(
        _next_logits(model, owner.rows[0]), _next_logits(model, base)
    )


def test_qwen_exact_progressive_remains_fail_closed_without_wide_row_protocol():
    candidate_keys = {
        "progressive_verification_tile",
        "progressive_multilane_draft_cap",
        "progressive_verification_fidelity",
    }
    # Qwen3.8 exposes only the separate, explicit approximate candidate.  The
    # exact selector below must still refuse it because the model does not
    # declare the row-exact wide target protocol.
    assert candidate_keys <= QWEN38_EXTERNAL_KEYS
    assert candidate_keys.isdisjoint(QWEN36_EXTERNAL_KEYS)
    assert not hasattr(tiny_target(), "progressive_external_verify_protocol")
    assert not hasattr(tiny_qwen36(), "progressive_external_verify_protocol")

    target, draft = tiny_pair()
    with pytest.raises(ValueError, match="row-exact progressive target protocol"):
        ExternalDraftBatchGenerator(
            target,
            draft_model=draft,
            binding="unsupported-progressive-qwen",
            num_draft=3,
            progressive_verification_tile=1,
        )


def test_flash_next_has_no_external_progressive_policy_surface():
    with pytest.raises(ValueError, match="unknown execution policy fields"):
        FlashNextPolicy.from_mapping({"progressive_verification_tile": 2})
