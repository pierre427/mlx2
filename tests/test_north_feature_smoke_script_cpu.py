from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_north_feature_smoke.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_north_feature_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_policy_is_explicit_and_default_off():
    smoke = load_script()
    assert smoke.execution_policy(
        expert_gather_sort="auto", batch_row_exact_q4=False
    ) == {}
    assert smoke.execution_policy(
        expert_gather_sort="auto", batch_row_exact_q4=True
    ) == {"batch_row_exact_q4": True}
    assert smoke.execution_policy(
        expert_gather_sort="unsorted_decode", batch_row_exact_q4=False
    ) == {"expert_gather_sort": "unsorted_decode"}


def test_route_gate_requires_complete_native_engagement_without_refusal():
    smoke = load_script()
    route = {
        "selected": True,
        "observed_used": True,
        "counts": {
            "started_forwards": 3,
            "complete_forwards": 3,
            "kernel": 10,
            "group_kernel": 20,
            "per_row": 0,
            "refusals": 0,
        },
    }
    assert smoke.batch_row_exact_q4_engaged(route)
    for field, value in (
        ("kernel", 0),
        ("group_kernel", 0),
        ("complete_forwards", 0),
        ("per_row", 1),
        ("refusals", 1),
    ):
        broken = {**route, "counts": {**route["counts"], field: value}}
        assert not smoke.batch_row_exact_q4_engaged(broken), field
    incomplete = {
        **route,
        "counts": {**route["counts"], "started_forwards": 4},
    }
    assert not smoke.batch_row_exact_q4_engaged(incomplete)
    assert not smoke.batch_row_exact_q4_engaged({**route, "selected": False})
    assert not smoke.batch_row_exact_q4_engaged({**route, "observed_used": False})


def test_candidate_flag_is_forwarded_only_to_server_policy():
    source = SCRIPT.read_text()
    smoke_start = source.index("smoke_command = [")
    server_start = source.index("server_command = [")
    policy_start = source.index("policy = execution_policy(", server_start)
    ordinary_command = source[smoke_start:server_start]
    server_policy = source[server_start:policy_start + 200]
    assert "execution-policy" not in ordinary_command
    assert "batch_row_exact_q4=args.batch_row_exact_q4" in server_policy
