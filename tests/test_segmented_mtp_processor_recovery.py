"""Committed MTP recovery retains serving's processor ownership."""

import mlx.core as mx
import pytest
from test_batched_mtp import _prepare_lane, _tiny_qwen4_model
from test_segmented_mtp import _ABTokenizer

from mlx2.runtime import hybrid_speculative as mtp
from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
from mlx2.runtime.generate import BatchGenerator
from mlx2.structured_output import StructuredOutputProcessor, compile_constraint
from mlx2.thinking_guard import ThinkingGuard


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _processor():
    return StructuredOutputProcessor(
        _ABTokenizer(), 4, compile_constraint(grammar="ab")
    )


@pytest.mark.parametrize("true_batch", [False, True])
@pytest.mark.parametrize("interruption", ["abort", "proposal", "commit"])
def test_recovery_restores_processor_state_without_replacing_serving_owners(
    monkeypatch, true_batch, interruption,
):
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", str(int(true_batch)))
    model = _tiny_qwen4_model()
    rows = [_prepare_lane(model, uid, [1, 2, 3, 4]) for uid in range(2)]
    processors = [_processor() for _ in rows]
    for row, processor in zip(rows, processors):
        row.lane.cur = 63 if row.lane.uid == 0 else 10
        row.lane.sampling_temp = 0.0
        row.lane.logprob_transform = None
        row.lane.logits_processors = [processor]
    before = [processor.verify_ledger() for processor in processors]
    state = mtp.attach_segmented_self_mtp_lanes(model, None, rows)
    rngs = [lane.rng for lane in state.lanes]

    def fail(*_args, **_kwargs):
        raise RuntimeError("interrupted CPU round")

    try:
        with monkeypatch.context() as injected:
            if interruption == "proposal":
                propose = mtp._propose_batched_self_mtp_impl

                def interrupt_after_verify(*args, **kwargs):
                    propose(*args, **kwargs)
                    fail()

                injected.setattr(mtp, "_propose_batched_self_mtp_impl", interrupt_after_verify)
                with pytest.raises(RuntimeError, match="interrupted CPU round"):
                    mtp.propose_batched_self_mtp(model, state)
            else:
                proposal = mtp.propose_batched_self_mtp(model, state)
                assert processors[0].failure is not None
                assert processors[1].constrained_steps > 0
                if interruption == "abort":
                    mtp.abort_batched_self_mtp(state, proposal)
                else:
                    # Fail after the first row publishes, forcing whole-cohort
                    # recovery of caches and processor state.
                    injected.setattr(state.transactions[1], "publish", fail)
                    with pytest.raises(RuntimeError, match="interrupted CPU round"):
                        mtp.commit_batched_self_mtp(
                            state, proposal,
                            emitted_counts=[len(row) for row in proposal.outputs],
                            terminal=[False, False],
                        )
        assert not state.poisoned and not state.proposal_open
        assert [lane.uid for lane in state.lanes] == [0, 1]
        assert all(lane.rng is rng for lane, rng in zip(state.lanes, rngs))
        for lane, processor, ledger in zip(state.lanes, processors, before):
            assert lane.logits_processors[0] is processor
            assert processor.verify_ledger() == ledger
        # A real post-recovery grammar dead end must reach the serving latch.
        proposal = mtp.propose_batched_self_mtp(model, state)
        assert "no valid token continuation" in processors[0].failure
        assert processors[1].failure is None
        mtp.abort_batched_self_mtp(state, proposal)
        assert all(processor.failure is None for processor in processors)
    finally:
        mtp.close_segmented_self_mtp_state(state)


def test_resyncing_guard_is_never_copied_during_abort_and_retry(monkeypatch):
    model = _tiny_qwen4_model()
    row = _prepare_lane(model, 0, [1, 2, 3, 4])
    guard = ThinkingGuard(4, (7,), budget=100)
    row.lane.logits_processors = [guard]

    def refuse_copy(*_args):
        pytest.fail("a rollback-resyncing processor must retain its live owner")

    monkeypatch.setattr(ThinkingGuard, "__deepcopy__", refuse_copy, raising=False)
    state = mtp.attach_segmented_self_mtp_lanes(model, None, [row])
    try:
        for _ in range(2):
            proposal = mtp.propose_batched_self_mtp(model, state)
            mtp.abort_batched_self_mtp(state, proposal)
            assert not state.poisoned, state.poison_reason
            assert state.lanes[0].logits_processors[0] is guard
    finally:
        mtp.close_segmented_self_mtp_state(state)


@pytest.mark.parametrize("cancel_first", [False, True])
def test_recovered_processors_follow_ordinary_handoff_and_survivor(monkeypatch, cancel_first):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    batch = BatchGenerator(
        _tiny_qwen4_model(), completion_batch_size=2,
        self_mtp={
            "persistent": True, "num_draft": 2,
            "segment_aware_live_tip": True,
            "segment_aware_async_qsa_promotion": False,
        },
    )
    processors = [_processor(), _processor()]
    uids = batch.insert(
        [[1, 2, 3, 4], [5, 6, 7, 8]], max_tokens=[4, 4],
        logits_processors=[[processor] for processor in processors],
    )
    try:
        assert batch.next()[1] == []
        executor = batch._generation_batch
        proposal = mtp.propose_batched_self_mtp(batch.model, executor.state)
        mtp.abort_batched_self_mtp(executor.state, proposal)
        assert not executor.state.poisoned
        executor.ordinary_handoff_policy = MTPOrdinaryHandoffPolicy.from_value(
            {"enabled": True, "max_mtp_width": 1}
        )
        executor._handoff_all_to_plain(
            decision={"reason": "width"}, projected_width=2,
        )
        if cancel_first:
            batch.remove([uids[0]])
        selected = uids[1:] if cancel_first else uids
        initial = batch._migrate_plain_fallbacks()
        assert [response.uid for response in initial] == selected
        tokens = {response.uid: [int(response.token)] for response in initial}
        plain = batch._plain_fallback_batch
        assert plain.uids == selected
        for uid, owned in zip(plain.uids, plain.logits_processors):
            assert owned[0] is processors[uids.index(uid)]
        finished = {}
        for _ in range(8):
            for response in batch.next()[1]:
                tokens[response.uid].append(int(response.token))
                if response.finish_reason:
                    finished[response.uid] = response.mtp_receipt
            if len(finished) == len(selected):
                break
        assert list(finished) == selected
        assert all(output == [10, 11, 0, 0] for output in tokens.values())
        assert all(receipt["mtp_ordinary_handoff"]["one_way"] for receipt in finished.values())
        assert all(processors[uids.index(uid)].constrained_steps > 0 for uid in selected)
        assert all(processor.failure is None for processor in processors)
    finally:
        batch.close()
