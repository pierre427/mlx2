from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "qualification/runs/varlen-closeout-current-main-20261005"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_post_thermal_samples_retain_two_failing_samples(monkeypatch):
    arm = load("thermal_arm", RUN / "run_lane_budget_p25_thermal_arm.py")
    rows = iter([{"stable": False}, {"stable": False}])
    monkeypatch.setattr(
        arm.thermal, "sample_thermal", lambda _command=None: next(rows)
    )
    monkeypatch.setattr(
        arm.thermal,
        "thermally_stable",
        lambda row, _policy: row["stable"],
    )

    observed, passed = arm.post_thermal_samples(
        {"post_sample_interval_seconds": 0}
    )

    assert observed == [{"stable": False}, {"stable": False}]
    assert passed is False


def test_arm_summary_keeps_acceptance_and_ttft_evidence(tmp_path):
    abba = load("thermal_abba", RUN / "run_lane_budget_p25_abba.py")
    raw = tmp_path / "arm.json"
    raw.write_text("{}\n")
    arm = {
        "rounds": [
            {
                "phase": "timed",
                "wall_seconds": 2.0,
                "rows": [
                    {
                        "usage": {"completion_tokens": 5, "prompt_tokens": 10},
                        "receipt": {
                            "ttft_seconds": 0.2,
                            "speculation": {"proposed": 4, "accepted": 3},
                        },
                    },
                    {
                        "usage": {"completion_tokens": 7, "prompt_tokens": 20},
                        "receipt": {
                            "ttft_seconds": 0.4,
                            "speculation": {"proposed": 6, "accepted": 2},
                        },
                    },
                ],
            }
        ],
        "status_after": {
            "execution": {"varlen_dense_mlp": {"selected": 3}},
            "scheduler": {
                "external_tensorfold_tree_rows": 16,
                "unrelated": 9,
            },
        },
    }

    summary = abba.arm_summary(raw, arm)

    assert summary["timed_tokens"] == 12
    assert summary["tokens_per_second"] == 6.0
    assert summary["timed_prompt_tokens"] == 30
    assert summary["median_ttft_seconds"] == pytest.approx(0.3)
    assert summary["prompt_tokens_per_request_ttft_second"] == pytest.approx(50.0)
    assert summary["proposed_tokens"] == 10
    assert summary["accepted_tokens"] == 5
    assert summary["proposal_acceptance_rate"] == 0.5
    assert summary["tensorfold_scheduler"] == {
        "external_tensorfold_tree_rows": 16
    }


def test_probe_sigterm_is_routed_through_cleanup_finally():
    probe = load("thermal_probe_arm", RUN / "probe_varlen_server_arm.py")

    with pytest.raises(KeyboardInterrupt):
        probe._interrupt_for_cleanup(15, None)


def test_pair_rejects_mixed_source_commits():
    summarize = load(
        "thermal_summarize", RUN / "summarize_lane_budget_p25_thermal_reps.py"
    )
    receipts = [
        {
            "arm": "control",
            "source": "a" * 40,
            "result": {"sha256": "1" * 64},
        },
        {
            "arm": "candidate",
            "source": "b" * 40,
            "result": {"sha256": "2" * 64},
        },
    ]
    raw = [
        {"arm": "control", "source": "a" * 40},
        {"arm": "candidate", "source": "b" * 40},
    ]
    summaries = [{"sha256": "1" * 64}, {"sha256": "2" * 64}]

    errors = summarize.pair_identity_errors(receipts, raw, summaries)

    assert errors == [
        {"kind": "mixed_source", "sources": ["a" * 40, "b" * 40]}
    ]


def test_pair_rejects_changed_request_body_before_output_comparison():
    abba = load("thermal_abba_identity", RUN / "run_lane_budget_p25_abba.py")
    common = {
        "phase": "timed",
        "round": 0,
        "request": 0,
        "task": "code",
        "group": 0,
        "cohort_id": "cohort-0",
        "planned_offset_ns": 0,
        "output_sha256": "same-output",
    }
    left = {
        "rounds": [
            {
                "phase": "timed",
                "round": 0,
                "rows": [{**common, "body_sha256": "left-body"}],
            }
        ]
    }
    right = {
        "rounds": [
            {
                "phase": "timed",
                "round": 0,
                "rows": [{**common, "body_sha256": "right-body"}],
            }
        ]
    }

    mismatch = abba.mismatches(left, right)

    assert mismatch == [
        {
            "phase": "timed",
            "round": 0,
            "request": 0,
            "kind": "request_identity",
            "differences": [
                {
                    "field": "body_sha256",
                    "left": "left-body",
                    "right": "right-body",
                    "left_present": True,
                    "right_present": True,
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    "environment,message",
    [
        ({}, "GPUQ_LEASE"),
        ({"GPUQ_LEASE": "lease-1"}, "GPUQ_SESSION"),
        (
            {"GPUQ_LEASE": "foreign", "GPUQ_SESSION": "session-1"},
            "GPUQ_LEASE",
        ),
        (
            {"GPUQ_LEASE": "lease-1", "GPUQ_SESSION": "foreign"},
            "GPUQ_SESSION",
        ),
    ],
)
def test_thermal_gpu_gate_refuses_missing_or_foreign_owner(
    tmp_path, environment, message
):
    abba = load("thermal_abba_ownership", RUN / "run_lane_budget_p25_abba.py")
    owner = {"lease_id": "lease-1", "session": "session-1", "pid": 123}
    paths = (tmp_path / "shared.json", tmp_path / "temporary.json")
    for path in paths:
        path.write_text(json.dumps(owner))

    with pytest.raises(RuntimeError, match=message):
        abba.require_gpu_owners(environ=environment, paths=paths)


def test_thermal_gpu_gate_accepts_only_the_matching_pair(tmp_path):
    abba = load("thermal_abba_owned", RUN / "run_lane_budget_p25_abba.py")
    owner = {"lease_id": "lease-1", "session": "session-1", "pid": 123}
    paths = (tmp_path / "shared.json", tmp_path / "temporary.json")
    for path in paths:
        path.write_text(json.dumps(owner))

    assert abba.require_gpu_owners(
        environ={"GPUQ_LEASE": "lease-1", "GPUQ_SESSION": "session-1"},
        paths=paths,
    ) == owner
