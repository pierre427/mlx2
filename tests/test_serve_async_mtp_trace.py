from types import SimpleNamespace

from scripts.serve_async_mtp_trace import _proposal_rows


def test_proposal_rows_are_request_local_and_preserve_observed_depths():
    proposal = SimpleNamespace(
        lane_uids=(7, 11),
        draft_depths=(2, 4),
        accepted_lengths=(1, 4),
    )
    rounds = {7: 3}
    rows = _proposal_rows(
        proposal,
        context_lengths=(1024, 2048),
        cycle_ns=99,
        round_indices=rounds,
    )
    assert rows == [
        {
            "request_id": "7",
            "round_index": 3,
            "context_length": 1024,
            "batch_size": 2,
            "draft_depth": 2,
            "accepted_prefix": 1,
            "cycle_ns": 99,
        },
        {
            "request_id": "11",
            "round_index": 0,
            "context_length": 2048,
            "batch_size": 2,
            "draft_depth": 4,
            "accepted_prefix": 4,
            "cycle_ns": 99,
        },
    ]
    assert rounds == {7: 4, 11: 1}
