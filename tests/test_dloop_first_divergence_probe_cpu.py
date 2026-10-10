"""CPU contracts for the bounded DLoop first-divergence probe."""

from scripts.dloop_first_divergence_probe import (
    _first_difference,
    _first_divergence_state_evidence,
    _scan_limits,
    _top_two_logprobs,
    _trace_window,
)


class _Vector:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        if isinstance(key, _Vector):
            return _Vector([self.values[index] for index in key.values])
        if isinstance(key, slice):
            return _Vector(self.values[key])
        return self.values[key]

    def tolist(self):
        return list(self.values)


class _FakeMx:
    @staticmethod
    def argpartition(values, kth):
        assert kth == -2
        return _Vector(sorted(range(len(values)), key=values.values.__getitem__))

    @staticmethod
    def argsort(values):
        return _Vector(sorted(range(len(values)), key=values.values.__getitem__))


def test_top_two_logprobs_preserve_rank_and_margin():
    result = _top_two_logprobs(_Vector([-4.0, -1.0, -3.0, -2.5]), _FakeMx)
    assert result == {
        "entries": [
            {"token_id": 1, "logprob": -1.0},
            {"token_id": 3, "logprob": -2.5},
        ],
        "margin_nats": 1.5,
    }


def test_first_difference_and_common_prefix_are_exact():
    assert _first_difference([4, 5, 6], [4, 5, 9]) == 2
    assert _first_difference([4, 5], [4, 5, 9]) == 2
    assert _first_difference([4, 5], [4, 5]) is None


def test_trace_window_is_bounded_to_divergence_neighbors():
    trace = [[{"token_id": index, "logprob": 0.0}] for index in range(8)]
    first = _trace_window(trace, 0, [10, 11], [20, 21, 22])
    assert list(first) == ["0", "1"]
    assert first["1"]["generated_prefix_tokens"] == [20]
    middle = _trace_window(trace, 4, [10, 11], [20, 21, 22, 23, 24])
    assert list(middle) == ["3", "4", "5"]
    assert middle["4"]["generated_prefix_tokens"] == [20, 21, 22, 23]
    assert _trace_window(trace, None, [10], [20]) == {}


def test_scan_limits_bound_known_short_and_late_divergences():
    assert _scan_limits([2, 7], 80, []) == {2: 8, 7: 80}
    assert _scan_limits([2, 7], 80, ["2=12", "7=76"]) == {2: 12, 7: 76}


def test_boundary_rerun_mismatch_is_retained_without_state_claim(monkeypatch):
    import scripts.dloop_first_divergence_probe as probe

    monkeypatch.setattr(
        probe,
        "_run_arm",
        lambda *args, **kwargs: {"tokens": [10, 99], "terminal_state": None},
    )
    report = _first_divergence_state_evidence(
        model="model",
        prompt_tokens=[1, 2],
        prompt_index=2,
        arms={"ordinary": None},
        earliest=2,
        expected_prefix=[10, 11],
        n=16,
    )
    assert report == {
        "status": "boundary_rerun_mismatch",
        "arms": [
            {
                "arm": "ordinary",
                "status": "boundary_rerun_failed",
                "failure": "rerun_tokens_differ_from_scanned_common_prefix",
                "boundary_tokens": 2,
                "expected_common_prefix_tokens": [10, 11],
                "actual_boundary_tokens": [10, 99],
                "state_continuation": None,
            }
        ],
    }


def test_index_zero_divergence_records_unavailable_state_boundary():
    result = _first_divergence_state_evidence(
        model=None,
        prompt_tokens=[1, 2],
        prompt_index=2,
        arms={},
        earliest=0,
        expected_prefix=[],
        n=16,
    )
    assert result == {
        "status": "unavailable_no_common_prefix_boundary",
        "reason": "first_divergence_at_generated_token_index_0",
        "arms": [],
    }
