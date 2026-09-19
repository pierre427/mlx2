"""Optional physical copies yield before active routes or useful APC entries."""
from dataclasses import asdict
import pytest
from mlx2.runtime.generate import MTPGenerationBatch
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController, _make_self_mtp_admission_callback
from mlx2.serving import reclaim_deferred_cache
from test_segmented_mtp import _FakeModel, _FakeMatcher, _detached


def setup_batch(free, *, delayed=False, lanes=1, saturation=16):
    bank = dict(free=free, retired=0.0, release=not delayed, events=[], decision={})
    def reclaim():
        bank['events'].append('reclaim')
        if bank['release']:
            bank['free'] += bank['retired']; bank['retired'] = 0.0
    def evict():
        bank['events'].append('evict');bank['free'] += 12.0;return True
    callback = _make_self_mtp_admission_callback(
        SelfMTPLaneAdmissionController(cache_estimator=lambda n: 1 << 20, saturation_lane_cap=saturation),
        free_memory=lambda: bank['free'], reclaim_memory=reclaim,
        evict_unused_cache=evict, observer=lambda d: bank.update(decision=asdict(d)))
    batch = MTPGenerationBatch(_FakeModel(), [_detached(i) for i in range(lanes)], [None] * lanes, [_FakeMatcher() for _ in range(lanes)],
                               segmented_live_tip=True, mtp_admission=callback)
    class Ticket:
        reserved_bytes = 8 << 30
        pending_bytes = reserved_bytes
        def settle_for_admission(self):
            bank['events'].append('settle');self.pending_bytes = 0
        def cancel_and_drain(self):
            bank['events'].append('cancel');bank['retired'] = 8.0
    batch._async_qsa_ticket = Ticket()
    return batch, bank


@pytest.mark.parametrize('free,expected', [(40.0, []), (23.0, ['settle']),
                                         (21.0, ['settle', 'cancel', 'reclaim'])])
def test_normal_overlap_settled_residency_and_cancel_before_eviction(free, expected):
    batch, bank = setup_batch(free)
    try:
        assert batch._apply_admission() is True
        assert bank['events'] == expected
        assert batch.uids == [0] and not batch._paused and not batch._plain_ready
        assert bank['decision']['stage'] == 'full'
        if free == 40:
            assert batch.mtp_cycle_state()[0][5] == 8.0  # No forced sync.
        elif free == 23:
            assert batch._async_qsa_ticket is not None
            assert batch.mtp_cycle_state()[0][5] == 0.0
        else:
            assert batch._async_qsa_ticket is None
    finally:
        batch.close()


def test_old_admission_order_counterfactual_evicts_checkpoint():
    batch, bank = setup_batch(21.0)
    try:
        # Before this repair the mutation-capable callback ran first.
        assert batch.mtp_admission(tuple(batch.mtp_cycle_state())) == {0: 2}
        assert 'evict' in bank['events']
        assert 'cancel' not in bank['events']
    finally:
        batch.close()


def test_delayed_host_accounting_yields_without_apc_eviction_or_migration():
    batch, bank = setup_batch(21.0, delayed=True)
    try:
        assert batch.next() == []
        assert bank['events'] == ['settle', 'cancel', 'reclaim']
        assert batch.uids == [0] and not batch._paused and not batch._plain_ready
        assert bank['decision']['optional_reclaim_wait'] is True
        class APC:
            def __len__(self): return 1
            def evict_oldest_unleased(self): raise AssertionError('grace evicted APC')
        # The outer serving loop must also honor this explicit wait reason.
        assert not reclaim_deferred_cache(APC(), bank['decision'], lambda: None)
        bank['release'] = True
        batch._optional_reclaim_retry_at = 0
        assert batch._apply_admission() is True
        assert bank['decision']['stage'] == 'full'
        assert bank['decision']['optional_reclaim_wait'] is False
        assert 'evict' not in bank['events']
    finally:
        batch.close()


def test_optional_reclaim_grace_expires_to_real_memory_policy():
    batch, bank = setup_batch(21.0, delayed=True)
    try:
        assert batch._apply_admission() is False
        batch._optional_reclaim_deadline = 1.0
        assert batch._apply_admission() is True
        assert 'evict' in bank['events']  # Actual persistent pressure, no credit.
        assert bank['decision']['optional_reclaim_wait'] is False
    finally:
        batch.close()


def test_capacity_only_split_never_waits_for_memory_recovery():
    batch, bank = setup_batch(40.0, lanes=2, saturation=1)
    try:
        assert batch._apply_admission() is True
        assert bank['events'] == ['reclaim', 'cancel']  # Existing policy reclaim and membership change, no pressure settle.
        assert batch._optional_reclaim_deadline == 0
        assert batch.uids == [0] and set(batch._paused) == {1}
        assert bank['decision']['optional_reclaim_wait'] is False
    finally:
        batch.close()


def test_pressure_drain_failure_propagates_before_eviction(monkeypatch):
    batch, bank = setup_batch(21.0)
    ticket = batch._async_qsa_ticket
    cancel = ticket.cancel_and_drain
    def fail(): raise RuntimeError('drain failed')
    monkeypatch.setattr(ticket, 'cancel_and_drain', fail)
    try:
        with pytest.raises(RuntimeError, match='drain failed'):
            batch._apply_admission()
        assert batch._async_qsa_ticket is ticket
        assert bank['events'] == ['settle']
    finally:
        monkeypatch.setattr(ticket, 'cancel_and_drain', cancel)
        batch.close()
