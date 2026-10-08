"""CPU transaction gates for Muse's opt-in row-exact target verifier."""

from __future__ import annotations

import copy

import mlx.core as mx
import pytest

from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.models.muse_glimmer import Model
from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def model_and_cache():
    mx.random.seed(18)
    model = Model(
        ModelArgs(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=128,
            sliding_window=4,
            max_position_embeddings=2048,
        )
    )
    model.eval()
    mx.eval(model.parameters())
    cache = model.make_cache()
    for token in (1, 2, 3, 4, 5, 6):
        mx.eval(model(mx.array([[token]]), cache=cache))
    return model, cache


def assert_cache_equal(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert type(got) is type(want)
        assert got.meta_state == want.meta_state
        assert got.offset == want.offset
        if got.keys is None:
            assert want.keys is None and want.values is None
            continue
        if hasattr(got, "_temporal_order"):
            got_values = (got._temporal_order(got.keys), got._temporal_order(got.values))
            want_values = (
                want._temporal_order(want.keys),
                want._temporal_order(want.values),
            )
        else:
            got_values = (got.keys[..., : got.offset, :], got.values[..., : got.offset, :])
            want_values = (
                want.keys[..., : want.offset, :],
                want.values[..., : want.offset, :],
            )
        assert all(
            mx.array_equal(a, b)
            for a, b in zip(got_values, want_values, strict=True)
        )


@pytest.mark.parametrize("accepted", [0, 1, 3, 7])
def test_row_exact_matches_serial_taps_logits_and_partial_commit(accepted):
    model, base = model_and_cache()
    tokens = mx.array([[11, 12, 13, 14, 15, 16, 17]])
    serial_cache = copy.deepcopy(base)
    serial_logits, serial_taps = [], []
    for position in range(tokens.shape[1]):
        logits, taps = model.forward_with_taps(
            tokens[:, position : position + 1], serial_cache, (0, 3)
        )
        serial_logits.append(logits)
        serial_taps.append(taps)
    serial_logits = mx.concatenate(serial_logits, axis=1)
    serial_taps = mx.concatenate(serial_taps, axis=1)

    expected_cache = copy.deepcopy(base)
    for position in range(accepted):
        mx.eval(model(tokens[:, position : position + 1], cache=expected_cache))

    candidate_cache = copy.deepcopy(base)
    transaction = SegmentedKVRows([candidate_cache]).begin([tokens.shape[1]])
    model.configure_target_verify_row_exact(True)
    logits, taps = model.forward_with_taps(tokens, transaction.caches, (0, 3))
    mx.eval(logits, taps)
    committed = transaction.commit([accepted])[0]
    model.configure_target_verify_row_exact(False)

    assert mx.array_equal(logits, serial_logits)
    assert mx.array_equal(taps, serial_taps)
    assert_cache_equal(committed, expected_cache)
    assert mx.array_equal(
        model(mx.array([[19]]), cache=copy.deepcopy(committed)),
        model(mx.array([[19]]), cache=copy.deepcopy(expected_cache)),
    )
    receipt = model.external_execution_receipt
    assert receipt is None


def test_row_exact_receipt_marks_observed_use_without_qualification():
    model, cache = model_and_cache()
    model.configure_target_verify_row_exact(True)
    tx = SegmentedKVRows([copy.deepcopy(cache)]).begin([2])
    model.forward_with_taps(mx.array([[7, 8]]), tx.caches, (0, 3))
    tx.abort()
    receipt = model.external_execution_receipt["target_verify_row_exact"]
    assert receipt["implemented"] and receipt["selected"]
    assert receipt["observed_used"] and receipt["executed_forwards"] == 1
    assert receipt["physical_query_rows"] == 2
    assert not receipt["qualified"] and not receipt["performance_claim"]
