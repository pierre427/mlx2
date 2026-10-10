"""Offline DLoop oracle: exact lookahead labels from the committed stream."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dloop_oracle.py"
SPEC = importlib.util.spec_from_file_location("dloop_oracle", SCRIPT)
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def _record(request, position, tokens, top1, accepted, emitted):
    return {
        "schema": oracle.LOG_SCHEMA,
        "uid": 0,
        "features": [[p, 0.0, 1.0] for p in top1],
        "tokens": list(tokens),
        "prev_token": 0,
        "verify_depth": 2,
        "accepted": accepted,
        "relaxed": 0,
        "labels": [],
        "emitted": list(emitted),
        "position": position,
        "request": request,
    }


def _write(tmp_path, records):
    path = tmp_path / "accept.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_lookahead_positions_are_labelled_from_the_committed_stream(tmp_path):
    # Committed stream for request 7: 10 11 12 13 14 15 16 17.
    records = [
        # Verifies 2, accepts both (+ bonus 12); lookahead 12 13 matches too.
        _record(7, 0, [10, 11, 12, 13], [0.99] * 4, 2, [10, 11, 12]),
        # Accepts 1 (+ correction 14); lookahead would have been wrong.
        _record(7, 3, [13, 99, 14, 15], [0.9, 0.2, 0.9, 0.9], 1, [13, 14]),
        _record(7, 5, [15, 16, 17, 0], [0.99, 0.99, 0.5, 0.5], 2, [15, 16, 17]),
    ]
    rows = oracle.split_requests(oracle.load_rows([_write(tmp_path, records)]))
    streams, ends, problems = oracle.build_streams(rows)
    assert problems == []
    assert ends == {"7.0": 8}
    samples, dropped = oracle.label_cycles(rows, streams, ends, 4)
    # The last cycle's fourth draft lands past the end of the request.
    assert dropped == 1
    assert [s["match"] for s in samples] == [4, 1]


def test_stream_conflicts_are_reported(tmp_path):
    records = [
        _record(1, 0, [5, 6], [0.9, 0.9], 2, [5, 6, 7]),
        _record(1, 2, [8, 9], [0.9, 0.9], 0, [8]),
    ]
    rows = oracle.split_requests(oracle.load_rows([_write(tmp_path, records)]))
    _, _, problems = oracle.build_streams(rows)
    assert ("stream_conflict", "1.0", 2) in problems


def test_rows_without_positions_are_refused(tmp_path):
    record = _record(1, 0, [5], [0.9], 1, [5, 6])
    record.pop("position")
    rows = oracle.split_requests(oracle.load_rows([_write(tmp_path, [record])]))
    with pytest.raises(SystemExit, match="position/request"):
        oracle.build_streams(rows)


def test_gate_extends_only_while_the_stage_score_clears_the_threshold():
    sample = {"match": 9, "logq": [math.log(0.95)] * 3 + [math.log(0.5)] * 3 + [0.0] * 3}
    # Stage 1 scores 3*log(0.95) = -0.154: extend.  Stage 2 scores -2.08: stop.
    assert oracle.gated_depth(sample, 3, 3, -0.5) == 6
    assert oracle.gated_depth(sample, 3, 3, -0.1) == 3
    assert oracle.gated_depth(sample, 3, 3, -3.0) == 9
    assert oracle.oracle_depth({"match": 5, "logq": []}, 3, 3) == 6
    assert oracle.oracle_depth({"match": 2, "logq": []}, 3, 3) == 3


def test_policies_use_measured_cycle_costs(tmp_path):
    records = []
    for request in range(4):
        # Every request: 13 committed tokens, drafts all correct and confident.
        stream = list(range(100 * request, 100 * request + 13))
        for start, stop in ((0, 3), (3, 6), (6, 13)):
            records.append(
                _record(request, start, stream[start:start + 6], [0.99] * 6, 2,
                        stream[start:stop])
            )
    log = _write(tmp_path, records)
    cost = tmp_path / "cost.json"
    raw = [{"rep": 0, "lanes": 1, "depth": d, "seconds_per_cycle": s}
           for d, s in ((0, 0.10), (2, 0.12), (4, 0.13), (6, 0.20))]
    cost.write_text(json.dumps({"schema": oracle.COST_SCHEMA, "raw": raw}))
    args = oracle.build_parser().parse_args(
        ["--log", str(log), "--cost", str(cost), "--stage", "2", "--max-stages", "3",
         "--thresholds", "-0.5", "--gate-sync-ms", "0"]
    )
    report = oracle.run(args)
    policies = report["policies"]
    assert policies["fixed_2"]["tau"] == 3
    assert policies["fixed_6"]["tau"] == 7
    # Interpolated depth 3 cost sits between the measured depths.
    assert 120 < policies["fixed_3"]["ms_per_cycle"] < 130
    # The gate extends to 6 on every (confident, fully accepted) cycle.
    assert policies["gate_-0.5"]["depths"] == {6: 12}
    assert policies["gate_-0.5"]["speedup_vs_first_boundary"] == pytest.approx(
        (120 / 3) / (200 / 7)
    )


def test_reused_uids_split_into_requests_when_position_resets(tmp_path):
    # Two sequential requests on the server's B1 lane, both uid 0.
    records = [
        _record(0, 1, [1, 2], [0.9, 0.9], 2, [1, 2, 3]),
        _record(0, 4, [4, 5], [0.9, 0.9], 2, [4, 5, 6]),
        _record(0, 1, [7, 8], [0.9, 0.9], 2, [7, 8, 9]),
    ]
    rows = oracle.split_requests(oracle.load_rows([_write(tmp_path, records)]))
    assert [r["request_key"] for r in rows] == ["0.0", "0.0", "0.1"]
    streams, ends, problems = oracle.build_streams(rows)
    assert problems == []
    # Streams start at position 1 (the prefill token is not a cycle output).
    assert ends == {"0.0": 7, "0.1": 4}
    samples, dropped = oracle.label_cycles(rows, streams, ends, 2)
    assert [s["match"] for s in samples] == [2, 2, 2] and dropped == 0


def test_explicit_boundaries_stop_at_the_cost_step():
    sample = {"match": 9, "logq": [math.log(0.95)] * 3 + [math.log(0.9)] * 4 + [-5.0] * 2}
    # A stage's score decides whether the next stage is drafted: stage 1
    # (3 drafts) scores -0.154, stage 2 (4 drafts) scores -0.421.
    assert oracle.gated_depth(sample, 0, 0, -0.5, [3, 7]) == 7
    assert oracle.gated_depth(sample, 0, 0, -0.1, [3, 7]) == 3
    assert oracle.gated_depth(sample, 0, 0, -0.5, [3, 7, 9]) == 9
    assert oracle.gated_depth(sample, 0, 0, -0.4, [3, 7, 9]) == 7
    assert oracle.oracle_depth({"match": 6, "logq": []}, 0, 0, [3, 7]) == 7
    assert oracle.oracle_depth({"match": 2, "logq": []}, 0, 0, [3, 7]) == 3
