"""Opt-in segmented self-MTP ledger cache (spec-round Phase 1a slice 1a-3), CPU only.

``MLX2_SEGMENTED_LEDGER_CACHE=1`` memoizes cache-contract fingerprints by value
and per-pass cache declarations in the segmented ledger. Pins: identical
fingerprints and decode output, and the ledger's fail-closed checks (out-of-
lineage mutation, cache swap, stale branch) still raise with the cache on.
"""

import mlx.core as mx
import pytest

import test_segmented_mtp as seg
from mlx2.runtime import cache_operations as ops
from mlx2.runtime import segmented_self_mtp as ssm


@pytest.fixture
def ledger_cache(monkeypatch):
    monkeypatch.setattr(ops, "_LEDGER_CACHE", True)
    monkeypatch.setattr(ssm, "_LEDGER_CACHE", True)
    ops._FINGERPRINTS.clear()
    yield
    ops._FINGERPRINTS.clear()


def test_off_by_default():
    import os

    if os.environ.get("MLX2_SEGMENTED_LEDGER_CACHE") is None:
        assert ops._LEDGER_CACHE is False and ssm._LEDGER_CACHE is False


def test_fingerprint_memo_is_exact(ledger_cache):
    for contract in (ops.KV_OPERATIONS, ops.RECURRENT_OPERATIONS, ops.ROTATING_KV_OPERATIONS):
        expected = contract._compute_fingerprint()
        assert contract.fingerprint == expected
        assert contract.fingerprint == expected  # served from the memo
    assert len(ops._FINGERPRINTS) == 3


def test_pass_memo_never_outlives_its_pass(ledger_cache):
    with ssm._ledger_pass():
        assert ssm._PASS.memo == {}
        with ssm._ledger_pass():  # nested passes share the outer memo
            ssm._PASS.memo["x"] = 1
        assert ssm._PASS.memo == {"x": 1}
    assert ssm._PASS.memo is None


@pytest.mark.parametrize("name", [
    "test_generation_stale_parallel_branch_is_rejected",
    "test_zero_delivery_refuses_changed_state_before_cheap_rejection",
    "test_each_position_bearing_target_cache_must_be_aligned",
    "test_reused_transaction_rejects_same_position_cache_swap",
])
def test_fail_closed_checks_still_raise(ledger_cache, name):
    getattr(seg, name)()


@pytest.mark.parametrize("mutation", ["cur", "qsa", "gdn"])
def test_out_of_lineage_mutation_still_raises(ledger_cache, mutation):
    seg.test_reattach_rejects_same_position_out_of_lineage_state_mutation(mutation)


def test_decode_output_unchanged(monkeypatch):
    from test_batched_mtp import _tiny_qwen4_model
    from test_draft_loop_cpu import _run

    prompt = [3, 9, 14, 2, 27, 5, 11, 20, 8, 17, 6, 25, 4, 13]
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(7)
        model = _tiny_qwen4_model()
        off, _, _ = _run(model, [prompt], 24, self_mtp={"num_draft": 2})
        monkeypatch.setattr(ops, "_LEDGER_CACHE", True)
        monkeypatch.setattr(ssm, "_LEDGER_CACHE", True)
        on, _, _ = _run(model, [prompt], 24, self_mtp={"num_draft": 2})
    finally:
        mx.set_default_device(previous)
        ops._FINGERPRINTS.clear()
    assert on == off
