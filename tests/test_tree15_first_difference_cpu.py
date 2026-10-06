"""CPU-only contracts for the bounded B1 diagnostic's comparison records."""

import importlib.util
import json
from pathlib import Path

import numpy as np


SOURCE = (Path(__file__).resolve().parents[1] / "qualification" / "runs"
          / "tree15-b1-discriminator-20261003" / "first_difference.py")
SPEC = importlib.util.spec_from_file_location("tree15_first_difference", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_row_margin_records_near_tie_and_changed_winner():
    row = probe.rank_row(np.array([1.0, 1.0002, 0.0]),
                         np.array([1.0001, 1.0, 0.0]))
    assert row["reference_top2"] == [0, 1]
    assert not row["argmax_equal"]
    assert np.isclose(row["reference_margin"], 0.0001, atol=1e-7)
    assert np.isclose(row["reference_candidate_gap"], 0.0001, atol=1e-7)


def test_state_comparison_checks_every_gdn_and_semantic_kv_plane():
    class Gdn:
        def __init__(self):
            self.cache = [np.ones((2,)), np.ones((2,))]

    class Kv:
        def __init__(self):
            self.offset = 2
            self.keys = np.ones((1, 1, 4, 2))
            self.values = np.ones((1, 1, 4, 2))

    left = [Gdn() if (i + 1) % 4 else Kv() for i in range(64)]
    right = [Gdn() if (i + 1) % 4 else Kv() for i in range(64)]
    # Uncommitted KV capacity is immaterial; the 2 committed positions agree.
    right[3].keys[..., 3, :] = 99
    assert probe.state_equal(probe.compare_state(left, right))
    right[46].cache[1][0] = 0
    rows = probe.compare_state(left, right)
    assert len([r for r in rows if r["kind"] == "gdn"]) == 48
    assert len([r for r in rows if r["kind"] == "kv"]) == 16
    assert not probe.state_equal(rows)
    assert not rows[46]["state"]["bitwise_equal"]


def test_capture_taps_preserve_layer_identity():
    left = np.ones((1, 2, 6))
    right = left.copy()
    right[0, 1, 4] = 2
    rows = probe.compare_taps(left, right, [5, 12, 19])
    assert rows[1]["captures"][2]["layer"] == 19
    assert not rows[1]["captures"][2]["bitwise_equal"]
    assert rows[0]["captures"][2]["bitwise_equal"]
    # This was the live probe's asymmetric slice: TensorFold [W,F] against
    # serial [1,W,F]. Keep rejecting it instead of silently broadcasting.
    try:
        probe.compare_taps(left[0], right, [5, 12, 19])
    except ValueError as error:
        assert "candidate=(2, 6), reference=(1, 2, 6)" in str(error)
    else:
        raise AssertionError("mixed tap shapes were accepted")


def test_gpuq_guard_requires_matching_session_and_lease(tmp_path):
    paths = (tmp_path / "host.json", tmp_path / "tmp.json")
    owner = {"session": "s", "lease_id": "s-spot", "pid": 123}
    for path in paths:
        path.write_text(json.dumps(owner))
    env = {"GPUQ_SESSION": "s", "GPUQ_LEASE": "s-spot"}
    assert probe.gpu_owner(paths, env) == owner
    try:
        probe.gpu_owner(paths, {**env, "GPUQ_LEASE": "another"})
    except ValueError:
        pass
    else:
        raise AssertionError("lease mismatch was accepted")
    paths[1].write_text(json.dumps({**owner, "pid": 456}))
    try:
        probe.gpu_owner(paths, env)
    except ValueError:
        pass
    else:
        raise AssertionError("owner mismatch was accepted")


def test_bounded_route_rejects_explicit_topology_overrides():
    probe.require_bounded_route_env({"MLX2_TENSORFOLD_SOURCE": "/pinned/source"})
    for name in probe.BOUNDED_ROUTE_OVERRIDES:
        try:
            probe.require_bounded_route_env({name: "1"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} override was accepted")
