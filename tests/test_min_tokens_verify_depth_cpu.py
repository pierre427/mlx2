"""min_tokens counts draft depth on every speculative verify row (CPU).

sglang #51903 -> #52054 made penalties and ``min_new_tokens`` per verify row.
mlx2 applies each lane's logits processors to every verify row's own history,
so a row at draft depth ``d`` sees ``generated + d`` completion tokens and
``minimum_tokens_processor`` masks EOS there iff ``generated + d < min_tokens``.
These tests put the boundary inside one draft block (generated=5,
min_tokens=8, six drafts: rows d=0..2 masked, d>=3 open) and check the mask
row by row on the external chain verifier (DFlash2 and prompt lookup share
``_verify``), the external tree verifier (row-wise and batched, with trees
whose row index differs from node depth), the parked plain-step mask, and the
native self-MTP route end to end against ordinary decode.

No model checkpoint: fake logits, or a tiny randomly initialised CPU model.
"""

from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.external_speculative import (
    ExternalDraftBatchGenerator,
    HostDraftRow,
    TreeDraftRow,
)
from mlx2.runtime.speculative_sampling import RequestRNG
from mlx2.serving import minimum_tokens_processor

EOS = 0
VOCAB = 8
PROMPT = [3, 4, 5, 6]
GENERATED = [2, 2, 2, 2, 2]  # five completion tokens already emitted
MINIMUM = 8
DRAFTS = 6


def _executor(stops=()):
    batch = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    batch.mx = mx
    batch.stops = set(stops)
    batch.fly_verification = SimpleNamespace(enabled=False)
    batch.tree_gates = {"logprobs_on_request": True}
    batch.scheduler_stats = Counter()
    return batch


def _lane():
    # The external lane keeps the last emitted token as ``anchor``, outside
    # ``history``; verify row j therefore sees len(history) + 1 + j tokens.
    tokens = PROMPT + GENERATED
    return SimpleNamespace(
        uid=1, history=tokens[:-1], anchor=tokens[-1],
        processors=[minimum_tokens_processor(mx, [EOS], len(PROMPT), MINIMUM)],
        sampling={"sampling_temp": 0.0, "emit_logprobs": True},
        rng=RequestRNG(3),
    )


def _law(token):
    law = np.full(VOCAB, 0.01)
    law[token] = 1.0
    return law / law.sum()


def _row_logits(rows, *, eos_first):
    # Token 1 and EOS lead; ``eos_first`` makes EOS the raw argmax.
    row = [-4.0] * VOCAB
    row[1], row[EOS] = (2.0, 3.0) if eos_first else (3.0, 2.0)
    return mx.array([[row] * rows])


def _masked(row):
    return bool(np.isneginf(np.asarray(row.astype(mx.float32))[EOS]))


def _expected_masked(depth):
    return len(GENERATED) + depth < MINIMUM


def test_chain_verify_masks_eos_by_row_depth():
    lane = _lane()
    block = HostDraftRow([1] * DRAFTS, [_law(1)] * DRAFTS)
    (decision,) = _executor()._verify(
        [lane], [block], _row_logits(DRAFTS + 1, eos_first=False)
    )
    assert decision.emitted == [1] * (DRAFTS + 1)
    masked = [_masked(row) for row in decision.response_logprobs]
    assert masked == [_expected_masked(d) for d in range(DRAFTS + 1)]
    assert masked == [True, True, True, False, False, False, False]


def test_chain_verify_accepts_eos_at_first_allowed_depth_and_stops():
    lane = _lane()
    drafts = [1, 1, 1, EOS, 1, 1]
    block = HostDraftRow(drafts, [_law(token) for token in drafts])
    (decision,) = _executor(stops=[EOS])._verify(
        [lane], [block], _row_logits(DRAFTS + 1, eos_first=True)
    )
    # Rows 0..2 forbid EOS (target picks 1 == draft); row 3 picks EOS == draft.
    assert decision.accepted == 4
    assert decision.emitted == [1, 1, 1, EOS]
    assert len(GENERATED) + len(decision.emitted) == MINIMUM + 1


def test_chain_verify_rejects_eos_drafted_before_the_minimum():
    lane = _lane()
    drafts = [1, EOS, 1, 1, 1, 1]
    block = HostDraftRow(drafts, [_law(token) for token in drafts])
    (decision,) = _executor(stops=[EOS])._verify(
        [lane], [block], _row_logits(DRAFTS + 1, eos_first=True)
    )
    # Depth 1 is still below the minimum: the drafted EOS is rejected and
    # the target's correction there is the masked argmax, token 1.
    assert decision.accepted == 1
    assert decision.emitted == [1, 1]


def _tree_block():
    # A six-deep chain of 1s plus a sibling (token 3) under the first chain
    # node: target row 7 sits at depth 2, so it must stay masked even though
    # its row index (7) is past the boundary.
    tokens = [1] * DRAFTS + [3]
    parents = [-1, 0, 1, 2, 3, 4, 0]
    return TreeDraftRow(tokens, parents)


@pytest.mark.parametrize("batched", [False, True])
def test_tree_verify_masks_eos_by_node_depth(batched):
    batch, lane, block = _executor(), _lane(), _tree_block()
    logits = _row_logits(len(block.tokens) + 1, eos_first=False)
    # Target row r verifies the children of node r; its depth is its path
    # length minus one (row 0 is the anchor).
    paths = batch._tree_paths(batch._target_tree_parents(block))
    depths = [len(path) - 1 for path in paths]
    assert depths == [0, 1, 2, 3, 4, 5, 6, 2]
    if batched:
        fence, state = batch._launch_tree_laws(lane, block, logits)
        mx.eval(*fence)
        masked = [_masked(row) for row in state["values"]]
        assert masked == [_expected_masked(d) for d in depths]
        assert masked[7] is True  # depth 2, not row index 7
        decision = batch._verify_tree_batched(lane, block, state)
    else:
        decision = batch._verify_tree(lane, block, logits)
    assert decision.emitted == [1] * (DRAFTS + 1)
    walked = [_masked(row) for row in decision.response_logprobs]
    assert walked == [_expected_masked(d) for d in range(DRAFTS + 1)]
    assert walked == [True, True, True, False, False, False, False]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("shape", ["sibling_last", "interleaved"])
def test_tree_verify_accepts_eos_at_first_allowed_depth(batched, shape):
    batch = _executor(stops=[EOS])
    lane = _lane()
    if shape == "sibling_last":
        # Accepted path on rows 0..4 (index == depth), sibling at row 7, depth 2.
        tokens = [1, 1, 1, EOS, 1, 1, 3]
        parents = [-1, 0, 1, 2, 3, 4, 0]
    else:
        # A rejected root child (token 2) first shifts the accepted path to
        # rows 0, 2, 3, 4 at depths 0..3, so row index != depth on the walk:
        # indexing the mask by row would open EOS at row 3 (depth 2) and
        # emit [1, 1, EOS] one token early.
        tokens = [2, 1, 1, 1, EOS]
        parents = [-1, -1, 1, 2, 3]
    block = TreeDraftRow(tokens, parents)
    logits = _row_logits(len(tokens) + 1, eos_first=True)
    if batched:
        fence, state = batch._launch_tree_laws(lane, block, logits)
        mx.eval(*fence)
        decision = batch._verify_tree_batched(lane, block, state)
    else:
        decision = batch._verify_tree(lane, block, logits)
    assert decision.emitted == [1, 1, 1, EOS]


@pytest.mark.parametrize("generated, offset, masked", [
    (MINIMUM - 1, 1, True),   # this anchor's row: completion length 7
    (MINIMUM - 1, 2, False),  # following dispatch: one more in-flight anchor
    (MINIMUM - 2, 2, True),
    (MINIMUM, 1, False),
])
def test_parked_forbidden_mask_counts_the_dispatched_row(generated, offset, masked):
    from mlx2.runtime.external_speculative import _parked_forbidden_mask

    def lane(processors):
        tokens = PROMPT + [2] * generated
        return SimpleNamespace(history=tokens[:-1], anchor=tokens[-1], processors=processors)

    minimum = minimum_tokens_processor(mx, [EOS], len(PROMPT), MINIMUM)
    cohort = [lane([minimum]), lane([])]
    rows = _parked_forbidden_mask(cohort, VOCAB, offset)
    if not masked:
        assert rows is None
        return
    expected = np.zeros((2, VOCAB), dtype=np.bool_)
    expected[0, EOS] = True
    np.testing.assert_array_equal(rows, expected)


# Native self-MTP: end to end on a tiny CPU model, against ordinary decode.


def _forcing_processor():
    """Make token 1 and EOS the only legal ids, EOS preferred when allowed.

    With the MTP draft probes seeing it too, every draft is accepted, so a
    round verifies the full draft depth and the boundary lands inside it.
    """

    def force(tokens, logits):
        vocabulary = mx.arange(logits.shape[-1])
        return mx.where(
            vocabulary == EOS, 1.0, mx.where(vocabulary == 1, 0.0, -float("inf"))
        ).astype(logits.dtype) + mx.zeros_like(logits)

    force.probe = force
    force.history_pure = True
    return force


class _Spy:
    """Record (completion length, EOS masked) for every real processor call."""

    history_pure = True

    def __init__(self, prompt_len):
        self.prompt_len = prompt_len
        self.calls = []

    def __call__(self, tokens, logits):
        row = np.asarray(logits.astype(mx.float32)).reshape(-1)
        self.calls.append(
            (int(tokens.shape[-1]) - self.prompt_len, bool(np.isneginf(row[EOS])))
        )
        return logits

    def probe(self, tokens, logits):
        return logits


def _run(self_mtp, minimum):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG
    from test_batched_mtp import _tiny_qwen4_model

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = None
    try:
        mx.random.seed(923)
        model = _tiny_qwen4_model()
        prompt = [1, 7, 3]
        spy = _Spy(len(prompt))
        processors = [
            _forcing_processor(),
            minimum_tokens_processor(mx, [EOS], len(prompt), minimum),
            spy,
        ]
        kwargs = {}
        if self_mtp:
            kwargs["self_mtp"] = {"num_draft": DRAFTS, "persistent": True}
        generator = BatchGenerator(
            model, completion_batch_size=1, prefill_batch_size=1,
            prefill_step_size=32, stop_tokens=[[EOS]], **kwargs,
        )
        insert = {"max_tokens": [16], "logits_processors": [processors]}
        if self_mtp:
            insert.update(
                lane_rngs=[LaneRNG(11)],
                self_mtp_configs=[{"sampling_temp": 0.0}],
            )
        generator.insert([prompt], **insert)
        emitted, rounds, finish = [], [], None
        for _ in range(40):
            before = len(spy.calls)
            _, responses = generator.next()
            rounds.append(spy.calls[before:])
            emitted.extend(int(response.token) for response in responses)
            finish = next(
                (r.finish_reason for r in responses if r.finish_reason), finish
            )
            if finish is not None:
                break
        return emitted, finish, rounds, spy.calls
    finally:
        if generator is not None:
            generator.close()
        mx.set_default_device(previous)


@pytest.mark.parametrize("minimum", [3, 5, MINIMUM])
def test_self_mtp_min_tokens_boundary_inside_draft_block_matches_ordinary(minimum):
    ordinary, ordinary_finish, _, _ = _run(False, minimum)
    assert ordinary == [1] * minimum + [EOS]
    assert ordinary_finish == "stop"

    output, finish, rounds, calls = _run(True, minimum)
    assert output == ordinary
    assert finish == ordinary_finish
    # Every verify row masks EOS exactly while its completion length is short.
    assert calls
    assert all(masked == (length < minimum) for length, masked in calls)
    # The prefill row emits token 1; the first full round verifies rows at
    # completion lengths 1..7, so minimum 3 and 5 put the boundary inside it
    # (8 falls on the next round's first row).
    straddled = any(
        len(round_calls) > 1
        and any(length == minimum - 1 for length, _ in round_calls)
        and any(length == minimum for length, _ in round_calls)
        for round_calls in rounds
    )
    assert straddled == (minimum < 1 + DRAFTS + 1), rounds
