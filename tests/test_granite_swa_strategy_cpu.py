"""Granite SWA strategy transfer: exact rotating state, CPU only."""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.granite_swa import GraniteSWAAdapter
from mlx2.runtime.continuation_verification import (
    ExactPrefixFrontier,
    SelectedContinuationTransaction,
    sample_continuations,
)
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.models.granitemoe_swa import Model, ModelArgs
from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def tiny_model():
    mx.random.seed(19)
    model = Model(
        ModelArgs(
            model_type="granitemoe_swa",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=8,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            shared_intermediate_size=16,
            max_position_embeddings=64,
            rms_norm_eps=1e-5,
            embedding_multiplier=1.0,
            attention_multiplier=0.5,
            residual_multiplier=1.0,
            logits_scaling=1.0,
            sliding_window=4,
            layer_types=[
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
        )
    )
    model.eval()
    mx.eval(model.parameters())
    return model


def ordinary(model, cache, anchor, count):
    tokens = []
    for _ in range(count):
        logits = model(mx.array([[anchor]], mx.uint32), cache=cache)[0, -1]
        mx.eval(logits)
        anchor = int(mx.argmax(logits).item())
        tokens.append(anchor)
    return tuple(tokens), anchor


def test_adapter_owns_bounded_prefill_and_keeps_strategy_unselected():
    adapter = object.__new__(GraniteSWAAdapter)
    assert adapter.prefill_step_default() == 512
    assert GraniteSWAAdapter.descriptor.metadata["proposal_strategy"] == {
        "algorithm": "longest-first-exact-prefix-v1",
        "implemented": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
    }
    assert adapter.diagnostics() == {
        "route": "ordinary",
        "qualification": "pending",
        "proposal_strategy": GraniteSWAAdapter.descriptor.metadata[
            "proposal_strategy"
        ],
    }


def test_longest_first_frontier_prunes_impossible_siblings():
    paths = (
        (1, 2, 3, 4),
        (0, 2, 3, 4),
        (1, 2, 9, 4),
        (1, 2, 3, 8),
    )
    frontier = ExactPrefixFrontier(paths)
    first = frontier.next_attempt()
    assert first.index == 0 and first.suffix == paths[0]
    from mlx2.runtime.continuation_verification import ContinuationOutcome

    frontier.record(
        first,
        ContinuationOutcome(0, 3, (1, 2, 3, 7), 1, 5, (5,), ()),
    )
    # The accepted prefix plus target correction makes every sibling
    # impossible; none may be launched after this authoritative frontier.
    assert frontier.next_attempt() is None
    assert frontier.receipt() == {
        "algorithm": "longest-first-exact-prefix-v1",
        "frontier_tokens": 4,
        "attempted_paths": 1,
        "pruned_paths": 3,
        "shared_prefix_tokens_reused": 0,
    }


def test_wrapped_swa_cascade_reuses_committed_prefix_without_recompute():
    model = tiny_model()
    cache = model.make_cache()
    prompt = [1, 2, 3, 4, 5, 6, 7]
    mx.eval(model(mx.array([prompt[:-1]], mx.uint32), cache=cache))
    assert all(entry.offset == 6 for entry in cache)
    assert all(
        entry.offset > entry.max_size
        for kind, entry in zip(model.args.layer_types, cache)
        if kind == "sliding_attention"
    )

    contract = model.exact_prefix_reuse_contract(cache)
    assert contract == {
        "algorithm": "granite-full-swa-segmented-kv-v1",
        "full_layers": 2,
        "sliding_layers": 2,
        "sliding_window": 4,
        "committed_offset": 6,
        "transaction": "SegmentedKVRows exact rotating-window snapshot/replay",
    }
    adapter = object.__new__(GraniteSWAAdapter)
    adapter.model = model
    strategy = adapter.proposal_verification_strategy(cache)
    assert strategy["implemented"] and not strategy["qualified"]
    assert not strategy["selected"] and not strategy["observed_used"]

    reference_cache = copy.deepcopy(cache)
    target, reference_anchor = ordinary(model, reference_cache, prompt[-1], 5)
    wrong = (target[2] + 1) % model.args.vocab_size
    if wrong == target[2]:
        wrong = (wrong + 1) % model.args.vocab_size
    paths = (
        (target[0], target[1], wrong, target[3]),
        target[:4],
        ((target[0] + 1) % model.args.vocab_size, *target[1:4]),
    )
    frontier = ExactPrefixFrontier(paths)
    anchor = prompt[-1]
    emitted = []
    attempted_rows = []
    while len(emitted) < 5:
        attempt = frontier.next_attempt()
        assert attempt is not None
        inputs = [anchor, *attempt.suffix]
        attempted_rows.append(len(inputs))
        transaction = SegmentedKVRows([cache]).begin([len(inputs)])
        logits = model(mx.array([inputs], mx.uint32), cache=transaction.caches)
        mx.eval(logits)
        outcome = sample_continuations(
            (attempt.suffix,),
            logits,
            lambda row, _prefix: int(mx.argmax(row).item()),
            maximum=5 - len(emitted),
        )
        kept = min(outcome.accepted + 1, len(outcome.emitted))
        cache = SelectedContinuationTransaction(transaction, 0, 1).commit([kept])[0]
        frontier.record(attempt, outcome)
        emitted.extend(outcome.emitted)
        anchor = emitted[-1]
        if outcome.accepted == len(attempt.suffix):
            break

    assert tuple(emitted) == target
    # The second candidate starts after the three-token authoritative frontier:
    # anchor plus one remaining proposal token, not anchor plus the full path.
    assert attempted_rows == [5, 2]
    receipt = frontier.receipt()
    assert receipt["shared_prefix_tokens_reused"] == 3
    assert receipt["pruned_paths"] == 1

    # Compare the next ordinary target law from independently replayed and
    # cascade-committed full/SWA state. This crosses a wrapped window boundary.
    actual = model(mx.array([[anchor]], mx.uint32), cache=copy.deepcopy(cache))[0, -1]
    expected = model(
        mx.array([[reference_anchor]], mx.uint32), cache=copy.deepcopy(reference_cache)
    )[0, -1]
    mx.eval(actual, expected)
    assert int(mx.argmax(actual).item()) == int(mx.argmax(expected).item())
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5
    )


def test_granite_prefix_reuse_refuses_wrong_sliding_geometry():
    model = tiny_model()
    cache = model.make_cache()
    cache[1] = KVCache()
    with pytest.raises(ValueError, match="sliding layer 1"):
        model.exact_prefix_reuse_contract(cache)
    cache = model.make_cache()
    cache[1] = RotatingKVCache(max_size=8)
    with pytest.raises(ValueError, match="sliding layer 1"):
        model.exact_prefix_reuse_contract(cache)
