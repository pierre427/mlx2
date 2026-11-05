"""Synthetic byte-cache oracles; no checkpoint or device execution."""
import pytest

from mlx2.runtime.streamed_load import WeightStreamRequest
from mlx2.runtime.weight_stream import ExpertLRU, StreamStats, expert_prefill_admission


class Reader:
    expert_bytes = 8
    num_experts = 32

    def __init__(self):
        self.fail = False

    def check_sources(self):
        if self.fail:
            raise RuntimeError("source changed")

    def outstanding_bytes(self):
        return 0

    def iter_many(self, experts):
        for expert in experts:
            yield expert, bytes([expert]) * self.expert_bytes

    def materialize(self, payload):
        return (payload,)


def cache(policy="scan_resistant", capacity=4):
    return ExpertLRU(Reader(), capacity_experts=capacity, stats=StreamStats(),
                     prefill_admission=policy)


def test_prefill_scan_preserves_hot_decode_payloads_and_recency():
    candidate, ordinary = cache(), cache("lru")
    for bank in (candidate, ordinary):
        bank.acquire([0, 1, 2, 3])
        bank.acquire([0, 1, 2])
    for expert in range(4, 20):
        with expert_prefill_admission("intermediate", ["request"]):
            for bank in (candidate, ordinary):
                assert bank.acquire([expert])[expert] == (bytes([expert]) * 8,)
    assert {0, 1, 2} <= set(candidate._entries)
    assert not {0, 1, 2} & set(ordinary._entries)
    assert len(candidate._protected) == 3
    assert candidate.stats.resident_bytes == 32


@pytest.mark.parametrize("phase", ["intermediate", "terminal", "mixed"])
def test_scan_hit_does_not_promote_or_refresh_decode_recency(phase):
    bank = cache()
    bank.acquire([0, 1, 2, 3])
    before = list(bank._entries), set(bank._protected)
    with expert_prefill_admission(phase, ["a"]):
        bank.acquire([0, 1])
    assert (list(bank._entries), bank._protected) == before
    assert bank.stats.as_dict()["stream_scan_calls_total"] == 1
    assert bank.stats.as_dict()["stream_scan_protected_hits_total"] == 1


def test_decode_promotes_probation_and_demotes_old_protected_rows():
    bank = cache()
    bank.acquire([0, 1, 2, 3])
    assert bank._protected == {1, 2, 3}
    bank.acquire([0])
    assert bank._protected == {0, 2, 3}
    assert list(bank._entries)[-1] == 0


def test_wide_scan_returns_every_payload_and_preserves_held_working_set():
    bank = cache()
    held = bank.acquire([0, 1, 2, 3])
    with expert_prefill_admission("terminal", ["a"]):
        result = bank.acquire(list(range(16)))
    assert len(result) == 16 and len(bank._entries) == 4
    assert all(result[e] == (bytes([e]) * 8,) for e in range(16))
    assert held[0] == (b"\0" * 8,)
    bank.note_gather(16)
    assert bank.stats.resident_bytes == 32
    assert bank.stats.peak_transient_bytes == (16 + 12) * 8
    assert bank._protected <= bank._entries.keys()


def test_source_failure_never_publishes_new_payload_and_shrink_bounds_protection():
    bank = cache()
    bank.acquire([0, 1, 2, 3])
    bank.reader.fail = True
    with pytest.raises(RuntimeError, match="source changed"):
        bank.acquire([4])
    assert 4 not in bank._entries and not bank._busy
    bank.capacity = 1
    bank._trim_to_capacity()
    assert len(bank._entries) == len(bank._protected) == 1
    bank.clear()
    assert not bank._protected and bank.stats.resident_bytes == 0


def test_selection_parses_through_existing_streamed_load_contract():
    from mlx2.serving import moe_expert_streaming_policy

    policy = moe_expert_streaming_policy({"enabled": True, "cache_gib": 1,
                                         "prefill_admission": "scan_resistant"})
    assert policy["prefill_admission"] == "scan_resistant"
    request = WeightStreamRequest(mode="moe_experts", budget_bytes=1024,
                                  prefill_admission="scan_resistant")
    assert request.prefill_admission == "scan_resistant"
    with pytest.raises(ValueError, match="MoE-only"):
        WeightStreamRequest(mode="dense_mlp", budget_bytes=1024,
                            prefill_admission="scan_resistant")
    with pytest.raises(ValueError, match="adaptive residency"):
        ExpertLRU(Reader(), capacity_experts=4, stats=StreamStats(),
                  prefill_admission="scan_resistant", adaptive_residency={"enabled": True})


def test_unknown_direct_cache_policy_refuses():
    with pytest.raises(ValueError, match="unknown"):
        cache("typo")


def test_elastic_shrink_retains_a_probation_segment():
    bank = cache()
    bank.acquire([0, 1, 2, 3])
    bank.capacity = 2
    bank._trim_to_capacity()
    assert len(bank._entries) == 2 and len(bank._protected) == 1
