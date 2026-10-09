"""Complete-path verification uses one target draw and private exact branches."""

import copy

import mlx.core as mx
import numpy as np
import pytest
from test_standard_xpress_serving_cpu import reference, tiny

from mlx2.runtime.continuation_verification import (
    SelectedContinuationTransaction,
    prepare_continuations,
    prepare_longest_first_continuations,
    sample_continuations,
    verify_longest_first_continuations,
)
from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows as SegmentedVerifyRows
from mlx2.runtime.speculative_sampling import RequestRNG, softmax


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def prepared(monkeypatch):
    target, _ = tiny()
    prompt = [1, 2, 3]
    cache = target.make_cache()
    mx.eval(target(mx.array([prompt[:-1]]), cache=cache))
    expected = reference(target, prompt, 4)
    paths = [
        tuple((t + 1) % 9 for t in expected[:3]),
        tuple(expected[:3]),
        (expected[0], (expected[1] + 1) % 9, expected[2]),
    ]
    calls = []
    original = target.forward_with_taps

    def forward(tokens, *a, **kw):
        calls.append(tuple(tokens.shape))
        return original(tokens, *a, **kw)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    result = prepare_continuations(
        target, mx, cache, prompt[-1], paths, [0, 2], SegmentedVerifyRows, max_depth=3
    )
    return target, prompt, cache, expected, calls, result


def assert_cache(actual, expected):
    for a, b in zip(actual, expected):
        assert a.offset == b.offset
        for x, y in zip(a.state, b.state):
            np.testing.assert_allclose(
                np.asarray(x[..., : a.offset, :]),
                np.asarray(y[..., : b.offset, :]),
                atol=2e-6,
                rtol=2e-6,
            )


def test_one_forward_matches_second_ranked_path_and_commits_only_matching_prefix(
    monkeypatch,
):
    target, prompt, cache, expected, calls, (paths, logits, features, tx) = prepared(
        monkeypatch
    )
    draws = []

    def sample(row, prefix):
        draws.append(prefix)
        return int(mx.argmax(row).item())

    result = sample_continuations(paths, logits, sample, maximum=4)
    assert calls == [(3, 4)] and list(result.emitted) == expected
    assert draws == [tuple(expected[:i]) for i in range(4)]
    assert result.selected_index == 1 and result.accepted == 3
    selected = SelectedContinuationTransaction(
        tx, result.selected_index, len(paths)
    ).commit([4])[0]
    fresh = target.make_cache()
    mx.eval(target(mx.array([prompt + expected[:3]]), cache=fresh))
    assert_cache(selected, fresh)
    assert all(c.offset == len(prompt) - 1 for c in cache)
    assert features.shape[:2] == (3, 4)


def test_sampled_target_law_is_ordinary_at_each_actual_prefix_and_drawn_once(
    monkeypatch,
):
    target, prompt, cache, _, calls, (paths, logits, _, tx) = prepared(monkeypatch)
    rng = RequestRNG(26)
    draws = []

    def sample(row, prefix):
        law = softmax(np.asarray(row), 0.8)
        ordinary = target(mx.array([prompt + list(prefix)]), cache=target.make_cache())[
            0, -1
        ]
        np.testing.assert_allclose(
            law, softmax(np.asarray(ordinary), 0.8), atol=2e-6, rtol=2e-5
        )
        draws.append(prefix)
        return rng.sample(law)

    result = sample_continuations(paths, logits, sample, maximum=4)
    assert rng.draws == len(result.emitted) == len(draws) and calls == [(3, 4)]
    kept = min(result.accepted + 1, len(result.emitted))
    selected = SelectedContinuationTransaction(
        tx, result.selected_index, len(paths)
    ).commit([kept])[0]
    fresh = target.make_cache()
    mx.eval(target(mx.array([prompt + list(result.emitted[:-1])]), cache=fresh))
    assert_cache(selected, fresh)
    assert all(c.offset == len(prompt) - 1 for c in cache)


@pytest.mark.parametrize("terminal", ["stop", "budget"])
def test_terminal_draw_selects_highest_ranked_path_containing_emitted(terminal):
    # The draw at position 1 matches only the lower-ranked path, then ends the
    # walk.  The outcome must name that path, not the rank-0 path whose token
    # there differs: receipts credit the selected path's sources.
    paths = ((5, 6, 7), (5, 9, 10))
    draws = (5, 9, 10, 11)
    kwargs = (
        {"maximum": 8, "stop_tokens": (9,)} if terminal == "stop" else {"maximum": 2}
    )
    outcome = sample_continuations(
        paths, mx.zeros((2, 4, 4)), lambda row, prefix: draws[len(prefix)], **kwargs
    )
    assert outcome.emitted == (5, 9) and outcome.accepted == 2
    selected = paths[outcome.selected_index]
    assert selected[: outcome.accepted] == outcome.emitted[: outcome.accepted]
    assert outcome.selected_index == 1
    assert outcome.selected_index in outcome.matched_indices


def test_stop_and_single_remaining_token_do_not_commit_unreached_branch_tokens(
    monkeypatch,
):
    target, prompt, cache, expected, _, (paths, logits, _, tx) = prepared(monkeypatch)
    result = sample_continuations(
        paths,
        logits,
        lambda row, prefix: int(mx.argmax(row).item()),
        maximum=1,
        stop_tokens=[expected[0]],
    )
    assert result.emitted == (expected[0],)
    selected = SelectedContinuationTransaction(
        tx, result.selected_index, len(paths)
    ).commit([1])[0]
    fresh = target.make_cache()
    mx.eval(target(mx.array([prompt]), cache=fresh))
    assert_cache(selected, fresh)
    assert all(c.offset == len(prompt) - 1 for c in cache)


def test_failed_branch_forward_and_aborted_success_leave_original_cache_unchanged(
    monkeypatch,
):
    target, prompt, cache, _, _, (paths, _, _, tx) = prepared(monkeypatch)
    snapshot = copy.deepcopy(cache)
    tx.abort()
    assert_cache(cache, snapshot)
    original = target.forward_with_taps

    def fail(*a, **kw):
        original(*a, **kw)
        raise RuntimeError("branch write failure")

    monkeypatch.setattr(target, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="branch write"):
        prepare_continuations(
            target,
            mx,
            cache,
            prompt[-1],
            paths,
            [0, 2],
            SegmentedVerifyRows,
            max_depth=3,
        )
    assert_cache(cache, snapshot)


def test_fifteen_complete_sequences_have_fifteen_physical_target_rows(monkeypatch):
    target, _ = tiny()
    prompt = [1, 2, 3]
    cache = target.make_cache()
    mx.eval(target(mx.array([prompt[:-1]]), cache=cache))
    expected = reference(target, prompt, 4)
    paths = [tuple(expected[:3])] + [(i // 9, i % 9, 8) for i in range(14)]
    calls = []
    original = target.forward_with_taps

    def forward(tokens, *a, **kw):
        calls.append(tuple(tokens.shape))
        return original(tokens, *a, **kw)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    paths, logits, _, tx = prepare_continuations(
        target, mx, cache, prompt[-1], paths, [0, 2], SegmentedVerifyRows, max_depth=3
    )
    outcome = sample_continuations(
        paths, logits, lambda row, prefix: int(mx.argmax(row).item()), maximum=4
    )
    assert calls == [(15, 4)] and outcome.physical_width == 15
    assert list(outcome.emitted) == expected
    SelectedContinuationTransaction(tx, outcome.selected_index, len(paths)).commit([4])


def test_longest_first_prunes_impossible_sibling_and_reuses_exact_prefix():
    target, _ = tiny()
    prompt = [1, 2, 3]
    cache = target.make_cache()
    mx.eval(target(mx.array([prompt[:-1]]), cache=cache))
    before = copy.deepcopy(cache)
    expected = reference(target, prompt, 4)
    vocab = target.args.vocab_size
    longest_bad = (
        expected[0],
        (expected[1] + 1) % vocab,
        (expected[2] + 2) % vocab,
        (expected[3] + 3) % vocab,
    )
    impossible = ((expected[0] + 1) % vocab, expected[1], expected[2])
    good = tuple(expected[:3])
    attempts = []

    def prepare_attempt(branch, anchor, suffix):
        attempts.append((anchor, tuple(suffix)))
        _paths, logits, features, transaction = prepare_continuations(
            target,
            mx,
            branch,
            anchor,
            (suffix,),
            [0, 2],
            SegmentedVerifyRows,
            max_sequences=1,
            max_depth=4,
        )
        return logits, features, transaction

    outcome, feature_slices, transaction = verify_longest_first_continuations(
        (longest_bad, impossible, good),
        cache,
        prompt[-1],
        prepare_attempt,
        lambda row, _prefix: int(mx.argmax(row).item()),
        maximum=4,
    )
    assert list(outcome.emitted) == expected
    assert outcome.selected_index == 2 and outcome.accepted == 3
    assert outcome.attempted_indices == (0, 2)
    assert outcome.pruned_siblings == 1
    assert outcome.shared_prefix_tokens_reused == 1
    assert outcome.physical_width == 1 and outcome.physical_span == 7
    assert attempts == [
        (prompt[-1], longest_bad),
        (expected[1], (expected[2],)),
    ]
    assert [part.shape[1] for part in feature_slices] == [2, 2]
    selected = transaction.commit([4])[0]
    fresh = target.make_cache()
    mx.eval(target(mx.array([prompt + expected[:3]]), cache=fresh))
    assert_cache(selected, fresh)
    assert_cache(cache, before)


def test_longest_first_reuses_reached_prefix_without_recomputing_common_rows(
    monkeypatch,
):
    target, prompt, cache, expected, calls, (paths, _logits, _features, tx) = prepared(
        monkeypatch
    )
    tx.abort()
    outcome, hidden, transaction = prepare_longest_first_continuations(
        target,
        mx,
        cache,
        prompt[-1],
        paths,
        [0, 2],
        SegmentedVerifyRows,
        lambda row, _prefix: int(mx.argmax(row).item()),
        maximum=4,
        max_depth=3,
    )
    assert outcome.emitted == tuple(expected)
    assert outcome.algorithm == "longest_first_exact_prefix_v1"
    assert outcome.launches == 2
    assert outcome.input_lengths == (4, 3)
    assert outcome.target_rows == 7
    assert outcome.shared_prefix_reused_tokens == 1
    assert calls[-2:] == [(1, 4), (1, 3)]
    assert hidden.shape[1] == 4
    selected = transaction.commit([4])[0]
    fresh = target.make_cache()
    mx.eval(target(mx.array([prompt + expected[:3]]), cache=fresh))
    assert_cache(selected, fresh)


@pytest.mark.parametrize("kind", ["gdn", "qsa", "mamba"])
def test_real_hybrid_branch_selection_commits_only_selected_recurrent_history(
    kind, monkeypatch
):
    from test_external_adaptive_hybrid_cpu import assert_state

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    if kind == "gdn":
        from test_qwen38_dflash2_cpu import tiny_target

        target = tiny_target()
        layers = [1, 6]
    elif kind == "qsa":
        from test_batched_mtp import _tiny_qwen4_model

        from mlx2.runtime.models import qwen4_exp

        monkeypatch.setattr(qwen4_exp, "_QSA_POOLED_KEY_CACHE", True)
        monkeypatch.setattr(qwen4_exp, "_QSA_APC_SUMMARIES", True)

        target = _tiny_qwen4_model()
        layers = [0, 1]
    else:
        from test_nemotron_external_taps_cpu import tiny_model

        target = tiny_model()
        layers = [0, 1, 4]
    owner = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    owner.model = target
    owner.scheduler_stats = {}
    prompt = [1, 2, 3]
    cache = target.make_cache()
    mx.eval(target(mx.array([prompt[:-1]]), cache=cache))
    before = copy.deepcopy(cache)
    expected = reference(target, prompt, 4)
    vocab = (
        target.speculative_args.vocab_size
        if hasattr(target, "speculative_args")
        else target.args.vocab_size
    )
    paths = [tuple((t + 1) % vocab for t in expected[:3]), tuple(expected[:3])]
    paths, logits, _, tx = prepare_continuations(
        target, mx, cache, prompt[-1], paths, layers, owner._target_owner, max_depth=3
    )
    outcome = sample_continuations(
        paths, logits, lambda row, prefix: int(mx.argmax(row).item()), maximum=4
    )
    assert list(outcome.emitted) == expected and outcome.selected_index == 1
    selected = SelectedContinuationTransaction(
        tx, outcome.selected_index, len(paths)
    ).commit([4])[0]
    fresh = target.make_cache()
    mx.eval(target(mx.array([prompt + expected[:3]]), cache=fresh))
    assert_state(selected, fresh)
    assert_state(cache, before)
    if kind == "qsa":
        from test_external_qsa_verify_rows_cpu import _assert_qsa

        _assert_qsa(selected[-1], fresh[-1])
        _assert_qsa(cache[-1], before[-1])
