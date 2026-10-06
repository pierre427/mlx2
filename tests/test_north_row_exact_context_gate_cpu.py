from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_north_row_exact_context_gate.py"


def load_script():
    spec = importlib.util.spec_from_file_location("north_row_exact_context_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(**counts):
    return {
        "selected": True,
        "observed_used": True,
        "qualified": False,
        "counts": {
            "started_forwards": 3,
            "complete_forwards": 3,
            "kernel": 7,
            "group_kernel": 5,
            "per_row": 0,
            "refusals": 0,
            **counts,
        },
    }


def response(identifier, ids, *, width, cached=900, qualification="candidate"):
    return {
        "choices": [{
            "message": {"content": f"{identifier} exact"},
            "logprobs": {"content": [{"id": value} for value in ids]},
        }],
        "mlx2": {
            "cache": "apcv2",
            "cached_tokens": cached,
            "ordinary_compute_width": width,
            "qualification": qualification,
        },
    }


def valid_cell(gate):
    identifiers = [f"NEEDLE-ABCDEF{i:04d}-{i}" for i in range(4)]
    ids = list(range(16))
    prime = [response(value, ids, width=1, cached=0) for value in identifiers]
    b1 = [response(value, ids, width=1) for value in identifiers]
    b4 = [response(value, ids, width=4) for value in identifiers]
    return identifiers, prime, b1, b4


def test_contexts_policy_and_layout_are_fixed_to_candidate_contract():
    gate = load_script()
    assert gate.REQUESTED_CONTEXTS == (1024, 4096, 16384)
    assert gate.WIDTH == 4
    assert gate.MAX_CONTEXT == 16384
    assert gate.POLICY == {"batch_row_exact_q4": True}
    assert gate.LAYOUT_SUFFIX == ":north-batch-row-exact-q4-v1"


def test_identifier_normalizes_only_dash_punctuation():
    gate = load_script()
    identifier = "NEEDLE-ABC-0"
    assert gate.identifier_present("NEEDLE‑ABC—0 exact", identifier)
    assert not gate.identifier_present("needle-ABC-0 exact", identifier)
    assert not gate.identifier_present("NEEDLE-ABD-0 exact", identifier)


def test_route_delta_requires_positive_balanced_native_engagement():
    gate = load_script()
    initial = route(
        started_forwards=0,
        complete_forwards=0,
        kernel=0,
        group_kernel=0,
    )
    initial["observed_used"] = False
    final = route()
    assert gate.counter_delta(initial, final)["kernel"] == 7
    assert gate.candidate_delta_engaged(initial, final)

    assert not gate.candidate_delta_engaged(initial, route(group_kernel=0))
    assert not gate.candidate_delta_engaged(initial, route(per_row=1))
    assert not gate.candidate_delta_engaged(initial, route(refusals=1))
    assert not gate.candidate_delta_engaged(initial, route(started_forwards=4))
    regressed = route(kernel=-1)
    assert not gate.candidate_delta_engaged(initial, regressed)


def test_cell_requires_exact_b1_b4_tokens_identifiers_width_and_apcv2():
    gate = load_script()
    identifiers, prime, b1, b4 = valid_cell(gate)
    result = gate.evaluate_cell(
        identifiers=identifiers,
        prime=prime,
        b1=b1,
        b4=b4,
        max_tokens=16,
    )
    assert result["passed"]
    assert all(result["checks"].values())

    broken = [*b4]
    broken[2] = response(identifiers[2], list(range(15)) + [99], width=4)
    assert not gate.evaluate_cell(
        identifiers=identifiers,
        prime=prime,
        b1=b1,
        b4=broken,
        max_tokens=16,
    )["passed"]

    missing = [*b4]
    missing[0] = response("WRONG", list(range(16)), width=4)
    assert not gate.evaluate_cell(
        identifiers=identifiers,
        prime=prime,
        b1=b1,
        b4=missing,
        max_tokens=16,
    )["checks"]["identifier_parity"]

    no_apc = [*b4]
    no_apc[0] = response(identifiers[0], list(range(16)), width=4, cached=0)
    assert not gate.evaluate_cell(
        identifiers=identifiers,
        prime=prime,
        b1=b1,
        b4=no_apc,
        max_tokens=16,
    )["checks"]["apcv2"]

    wrong_width = [*b4]
    wrong_width[0] = response(identifiers[0], list(range(16)), width=3)
    assert not gate.evaluate_cell(
        identifiers=identifiers,
        prime=prime,
        b1=b1,
        b4=wrong_width,
        max_tokens=16,
    )["checks"]["b4_width"]


def test_cell_refuses_partial_cohorts_and_non_candidate_receipts():
    gate = load_script()
    identifiers, prime, b1, b4 = valid_cell(gate)
    partial = gate.evaluate_cell(
        identifiers=identifiers,
        prime=prime,
        b1=b1,
        b4=b4[:3],
        max_tokens=16,
    )
    assert partial == {
        "passed": False,
        "failure": "expected exactly four rows per phase",
    }

    ordinary = [*b4]
    ordinary[0] = response(
        identifiers[0], list(range(16)), width=4, qualification="qualified"
    )
    result = gate.evaluate_cell(
        identifiers=identifiers,
        prime=prime,
        b1=b1,
        b4=ordinary,
        max_tokens=16,
    )
    assert not result["checks"]["candidate_receipts"]


def test_script_is_candidate_only_and_preserves_existing_control():
    source = SCRIPT.read_text()
    assert '"--execution-policy"' in source
    assert '"--ordinary"' in source
    assert '"--qualification-mode"' in source
    assert "run_single_model_smoke_ladder.py" not in source
    assert "thermal_ladder.py" not in source
    assert "refusing to overwrite nonempty evidence directory" in source
    assert "validate_ownership(args.session, args.label) == owner" in source
    assert "stop_server(server)" in source


def test_contamination_reasons_fail_closed_for_each_environment_gate():
    gate = load_script()
    clean = {
        "thermal_pre": {"stable": True},
        "thermal_post": {"breached": False},
        "swapouts_before": 10,
        "swapouts_after": 10,
        "foreign_activity": [],
        "swapout_tolerance_pages": 0,
    }
    assert gate.contamination_reasons(**clean) == []

    cases = (
        ({"thermal_pre": {"stable": False}}, "thermal admission failed"),
        ({"thermal_post": {"breached": True}}, "two consecutive"),
        ({"swapouts_after": 11}, "swapouts rose by 1 pages"),
        ({"swapouts_before": -1}, "swapout counter unavailable"),
        ({"foreign_activity": [{"pid": 42}]}, "foreign GPU-capable process"),
    )
    for replacement, expected in cases:
        arguments = {**clean, **replacement}
        reasons = gate.contamination_reasons(**arguments)
        assert any(expected in reason for reason in reasons)


def test_thermal_admission_reuses_ladder_policy(monkeypatch):
    gate = load_script()
    samples = [{"thermal_state": 1, "virtual_temperature_c": 33.0}]
    observed = []

    def stabilize(policy):
        observed.append(policy)
        return samples

    monkeypatch.setattr(gate.thermal.matrix, "stabilize_thermal", stabilize)
    result = gate.thermal_admission()
    assert result["stable"] is True
    assert result["samples"] == samples
    assert observed == [gate.thermal.ADMISSION_POLICY]
    assert "fair (1)" in result["interpretation"]


def test_contamination_window_excludes_owned_tree_and_applies_post_rule(monkeypatch):
    gate = load_script()
    own_snapshots = iter(({1, 2, 3}, {1, 2, 3, 4}))
    foreign_snapshots = iter((
        {90: {"cpu_seconds": 4.0, "command": "mlx2.server foreign"}},
        {90: {"cpu_seconds": 4.5, "command": "mlx2.server foreign"}},
    ))
    swaps = iter((100, 100))
    monkeypatch.setattr(gate, "owned_process_tree", lambda _pid: next(own_snapshots))
    monkeypatch.setattr(gate.thermal, "foreign_snapshot", lambda _own: next(foreign_snapshots))
    monkeypatch.setattr(gate.thermal, "swapouts", lambda: next(swaps))
    monkeypatch.setattr(
        gate.thermal,
        "foreign_activity",
        lambda before, after, threshold: [],
    )
    post = {
        "samples": [{"thermal_state": 1}],
        "breached": False,
        "rule": "invalid only after two consecutive throttled samples",
    }
    monkeypatch.setattr(gate.thermal, "post_run_thermal", lambda: post)

    start = gate.begin_contamination_window(123)
    result = gate.finish_contamination_window(
        start,
        server_pid=123,
        foreign_cpu_threshold=2.0,
        swapout_tolerance_pages=0,
        thermal_pre={"stable": True},
    )
    assert result["passed"] is True
    assert result["thermal_post"] == post
    assert result["swapouts"]["delta"] == 0
    assert result["process_contamination"]["owned_pids_before"] == [1, 2, 3]
    assert result["process_contamination"]["owned_pids_after"] == [1, 2, 3, 4]


def test_script_thermal_brackets_every_context_cell():
    source = SCRIPT.read_text()
    loop = source[source.index("for requested in REQUESTED_CONTEXTS") :]
    admission = loop.index("thermal_pre = thermal_admission()")
    window = loop.index("environment_start = begin_contamination_window(server_pid)")
    request = loop.index("prime = [chat(")
    finish = loop.index("environment = finish_contamination_window(")
    assert admission < window < request < finish
    assert "post_run_thermal()" in source
    assert "thermal.matrix.stabilize_thermal(thermal.ADMISSION_POLICY)" in source
    assert "thermal.process_tree({server_pid})" in source
