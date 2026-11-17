"""Batched sampled verify laws (sglang#48949 lesson) match the row-wise path (CPU)."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.external_speculative import (
    CompactDraftRow,
    ExternalDraftBatchGenerator,
    HostDraftRow,
    LaneFailure,
)
from mlx2.runtime.speculative_sampling import FLyVerificationPolicy, RequestRNG

VOCAB = 64


def _generator(batched):
    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen.mx = mx
    gen.stops = set()
    gen.fly_verification = FLyVerificationPolicy()
    gen.scheduler_stats = {}
    gen.batched_sampled_laws = batched
    return gen


def _penalty(tokens, logits):
    return logits - 0.5 * (mx.arange(logits.shape[-1]) % 3 == 0)


def _lanes(seed):
    def lane(uid, sampling, processors=()):
        return SimpleNamespace(
            uid=uid, anchor=1 + uid, history=[0, uid], processors=list(processors),
            rng=RequestRNG(seed + uid),
            sampling={"emit_logprobs": True, **sampling},
        )

    return [
        lane(0, {"sampling_temp": 0.7, "top_p": 0.95}),
        lane(1, {"sampling_temp": 0.0}),
        lane(2, {"sampling_temp": 1.0, "top_k": 8, "emit_logprobs": False}),
        lane(3, {"sampling_temp": 0.8}, [_penalty]),
        lane(4, {"sampling_temp": 0.6, "min_p": 0.05}),
    ]


def _blocks(rng, logits, depths):
    blocks = []
    for row, depth in enumerate(depths):
        target = np.asarray(mx.argmax(logits[row], axis=-1)).tolist()
        drafts = [int(t if rng.random() < 0.6 else rng.integers(VOCAB)) for t in target[:depth]]
        if row % 2:
            ids = np.array([[d, (d + 1) % VOCAB, (d + 2) % VOCAB] for d in drafts])
            blocks.append(CompactDraftRow(drafts, ids, np.tile([0.6, 0.3, 0.1], (depth, 1)), 3))
        else:
            laws = rng.dirichlet(np.ones(VOCAB), size=depth)
            for law, d in zip(laws, drafts):
                law[d] += 1.0
            blocks.append(HostDraftRow(drafts, [law / law.sum() for law in laws]))
    return blocks


def _run(batched, seed, depths):
    rng = np.random.default_rng(seed)
    logits = mx.array(2.0 * rng.normal(size=(5, max(depths) + 1, VOCAB)), dtype=mx.float32)
    lanes = _lanes(seed)
    gen = _generator(batched)
    decisions = gen._verify(lanes, _blocks(rng, logits, depths), logits)
    return gen, lanes, decisions


@pytest.mark.parametrize("seed", range(12))
def test_batched_sampled_laws_are_bit_and_rng_identical(seed):
    depths = [3, 4, 2, 3, 5][seed % 5:] + [3, 4, 2, 3, 5][: seed % 5]
    ref_gen, ref_lanes, ref = _run(False, seed, depths)
    gen, lanes, got = _run(True, seed, depths)
    for a, b, la, lb in zip(ref, got, ref_lanes, lanes):
        assert (a.accepted, list(a.emitted)) == (b.accepted, list(b.emitted))
        assert la.rng.draws == lb.rng.draws
        assert la.rng.uniform() == lb.rng.uniform()
        if a.target_laws is None:
            assert b.target_laws is None
        else:
            assert len(a.target_laws) == len(b.target_laws)
            for x, y in zip(a.target_laws, b.target_laws):
                assert (x is None and y is None) or np.array_equal(x, y)
        if a.response_logprobs is None:
            assert b.response_logprobs is None
        else:
            assert len(a.response_logprobs) == len(b.response_logprobs)
            for x, y in zip(a.response_logprobs, b.response_logprobs):
                assert (x is None) == (y is None)
                if x is not None:
                    assert np.array_equal(np.asarray(x), np.asarray(y))
    assert "external_batched_sampled_law_rounds" not in ref_gen.scheduler_stats
    assert gen.scheduler_stats["external_batched_sampled_law_rounds"] == 1
    # Lanes 0, 2 and 4 are sampled and processor-free; 1 is greedy, 3 has a processor.
    assert gen.scheduler_stats["external_batched_sampled_law_rows"] == sum(
        depths[row] + 1 for row in (0, 2, 4)
    )


def _nonfinite(batched, bad_rows):
    logits = np.zeros((5, 3, VOCAB), dtype=np.float32)
    for row in bad_rows:
        logits[row, 1, 7] = np.nan
    logits = mx.array(logits)
    lanes = _lanes(0)
    blocks = [HostDraftRow([2, 3], [np.full(VOCAB, 1 / VOCAB)] * 2) for _ in lanes]
    with pytest.raises(LaneFailure) as caught:
        _generator(batched)._verify(lanes, blocks, logits)
    return caught.value


@pytest.mark.parametrize("bad_rows", [(0,), (4,), (2, 4), (1, 4), (3, 0)])
def test_batched_nonfinite_fails_the_same_lane_as_reference(bad_rows):
    ref = _nonfinite(False, bad_rows)
    got = _nonfinite(True, bad_rows)
    assert ref.uid == got.uid
    assert "not a probability distribution" in str(got)


def _two_faults(batched):
    logits = np.zeros((5, 3, VOCAB), dtype=np.float32)
    logits[0, 1, 7] = np.nan
    lanes = _lanes(0)
    blocks = [HostDraftRow([2, 3], [np.full(VOCAB, 1 / VOCAB)] * 2) for _ in lanes]
    # Lane 1 is greedy: its prepare step refuses a draft its law cannot draw.
    blocks[1] = CompactDraftRow(
        [2, 3], np.array([[2, 3, 4], [3, 4, 5]]), np.array([[0.0, 0.6, 0.4]] * 2), 3,
    )
    with pytest.raises(Exception) as caught:
        _generator(batched)._verify(lanes, blocks, mx.array(logits))
    return caught.value


def test_earlier_lane_failure_wins_over_a_later_malformed_proposal():
    ref, got = _two_faults(False), _two_faults(True)
    assert type(ref) is type(got) is LaneFailure
    assert ref.uid == got.uid == 0


# ------------------------------------------------- end to end, tiny models --
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_external_dflash2_cpu import (
    _external_engine,
    _greedy_reference,
    drain,
    tiny,
)
from test_external_dflash2_cpu import (
    _serve as _serve_request,
)

from mlx2.runtime.sample_utils import LaneRNG


def _serve(batched, progressive_tile=None):
    model, draft = tiny()
    kwargs = {}
    if progressive_tile is not None:
        model.configure_target_verify_row_exact(True)
        kwargs = {"progressive_verification_tile": progressive_tile, "completion_batch_size": 3}
    batch = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="batched-laws-test", num_draft=3,
        prefill_step_size=3, **kwargs,
    )
    batch.batched_sampled_laws = batched
    uids = batch.insert(
        [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12]], max_tokens=[12, 12, 12],
        sampling_configs=[
            {"sampling_temp": 0.8, "top_p": 0.9},
            {"sampling_temp": 0},
            {"sampling_temp": 1.0, "top_k": 6},
        ],
        lane_rngs=[LaneRNG(400 + index) for index in range(3)],
    )
    output, _final = drain(batch)
    return [output[uid] for uid in uids], batch.scheduler_stats


@pytest.mark.parametrize("tile", [None, 2])
def test_batched_sampled_laws_serve_the_same_tokens(tile):
    reference, ref_stats = _serve(False, tile)
    got, stats = _serve(True, tile)
    assert got == reference
    assert "external_batched_sampled_law_rounds" not in ref_stats
    assert stats["external_batched_sampled_law_rounds"] > 0
    if tile is not None:
        assert stats["external_progressive_verify_launches"] > 0


def _poison_batched(monkeypatch, temperature):
    """NaN the batched rows of lanes sampling at ``temperature`` only."""
    real = ExternalDraftBatchGenerator._batched_sampled_laws

    def poisoned(self, entries):
        entries = [
            (key, lane, [row * float("nan") for row in rows])
            if float(lane.sampling.get("sampling_temp", 0)) == temperature
            else (key, lane, rows)
            for key, lane, rows in entries
        ]
        return real(self, entries)

    monkeypatch.setattr(ExternalDraftBatchGenerator, "_batched_sampled_laws", poisoned)


@pytest.mark.parametrize("tile", [None, 2])
def test_batched_nonfinite_law_fails_only_its_lane(monkeypatch, tile):
    _poison_batched(monkeypatch, 0.8)
    model, draft = tiny()
    kwargs = {}
    if tile is not None:
        model.configure_target_verify_row_exact(True)
        kwargs = {"progressive_verification_tile": tile, "completion_batch_size": 2}
    batch = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="batched-laws-test", num_draft=3,
        prefill_step_size=3, **kwargs,
    )
    batch.batched_sampled_laws = True
    poisoned, healthy = batch.insert(
        [[1, 2, 3, 4, 5], [6, 7, 8]], max_tokens=[6, 6],
        sampling_configs=[{"sampling_temp": 0.8}, {"sampling_temp": 1.0}],
        lane_rngs=[LaneRNG(7), LaneRNG(8)],
    )
    output, final = drain(batch)
    failures = batch.take_lane_failures()
    assert [failure["uid"] for failure in failures] == [poisoned]
    assert "probability distribution" in failures[0]["reason"]
    assert healthy in final and len(output[healthy]) == 6


def test_flag_off_never_enters_the_batched_builder(monkeypatch):
    def refuse(self, entries):
        raise AssertionError("batched builder used with the switch off")

    monkeypatch.setattr(ExternalDraftBatchGenerator, "_batched_sampled_laws", refuse)
    for tile in (None, 2):
        _serve(False, tile)


def test_batched_invalid_law_is_a_request_error_not_a_worker_death(monkeypatch):
    monkeypatch.setenv("MLX2_SPEC_BATCHED_SAMPLED_LAWS", "1")
    _poison_batched(monkeypatch, 0.7)
    poisoned, calls = ExternalDraftBatchGenerator._batched_sampled_laws, []

    def counted(self, entries):
        calls.append(len(entries))
        return poisoned(self, entries)

    monkeypatch.setattr(ExternalDraftBatchGenerator, "_batched_sampled_laws", counted)
    m, d = tiny(sliding_window=16)
    engine = _external_engine(monkeypatch, m, d)
    try:
        failed = _serve_request(
            engine, {"tokens": [1, 2, 3], "max_tokens": 4, "temperature": 0.7, "seed": 3}
        )
        after = _serve_request(
            engine, {"tokens": [1, 2, 3], "max_tokens": 4, "temperature": 0}
        )
        alive, error = engine.thread.is_alive(), engine.error
    finally:
        engine.close()
    assert calls and calls[0] == 1  # the batched builder served the sampled lane
    assert "probability distribution" in failed.get("error", "")
    assert alive and error is None
    assert after["tokens"] == _greedy_reference(m, [1, 2, 3], 4)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), ("1", True), ("0", False), ("", True), ("true", True)],
)
def test_switch_is_default_on_and_only_zero_turns_it_off(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MLX2_SPEC_BATCHED_SAMPLED_LAWS", raising=False)
    else:
        monkeypatch.setenv("MLX2_SPEC_BATCHED_SAMPLED_LAWS", value)
    model, draft = tiny()
    batch = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="batched-laws-env", num_draft=3,
        prefill_step_size=3,
    )
    assert batch.batched_sampled_laws is expected
