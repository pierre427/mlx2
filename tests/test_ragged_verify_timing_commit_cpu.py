"""Timed MTP rounds must stay committable.

With ``MLX2_RAGGED_VERIFY_TIMING=1`` the proposal wrapper returns a copy of
the round result carrying its stage receipt.  Commit checks the open proposal
by identity, so the batch must point at that copy; it used to keep the
original and every timed round failed with "commit requires the currently
open self-MTP proposal".
"""
from types import SimpleNamespace

from mlx2.runtime import hybrid_speculative as hs


def _result():
    return hs.SelfMTPCycleResult(
        membership_epoch=0, lane_uids=(1,), draft_depths=(1,),
        accepted_lengths=(1,), target_drops=(0,), head_drops=(0,), outputs=((),),
    )


def test_timed_round_returns_the_open_proposal(monkeypatch):
    monkeypatch.setenv("MLX2_RAGGED_VERIFY_TIMING", "1")
    lane = SimpleNamespace(stats=hs.HybridStats())
    batch = SimpleNamespace(lanes=[lane], _open_proposal=None)
    inner = _result()

    def round_(model, batch_):
        batch_._open_proposal = inner
        return inner

    monkeypatch.setattr(hs, "_propose_batched_self_mtp_round", round_)
    returned = hs._propose_batched_self_mtp_impl(None, batch)
    assert returned is batch._open_proposal
    assert returned is not inner and returned.verify_observation
    assert lane.stats.ragged_verify_diagnostic_rounds == 1


def test_untimed_round_is_returned_unchanged(monkeypatch):
    monkeypatch.delenv("MLX2_RAGGED_VERIFY_TIMING", raising=False)
    batch = SimpleNamespace(lanes=[], _open_proposal=None)
    inner = _result()

    def round_(model, batch_):
        batch_._open_proposal = inner
        return inner

    monkeypatch.setattr(hs, "_propose_batched_self_mtp_round", round_)
    assert hs._propose_batched_self_mtp_impl(None, batch) is inner is batch._open_proposal
