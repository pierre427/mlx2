from types import SimpleNamespace as NS

import pytest

from mlx2 import memory, serving
from mlx2.qualification import required_feature_checks
from mlx2.runtime import apc_v2, generate, os_memory
from mlx2.server import build_parser
from mlx2.serving import ServingEngine, decode_time_fairness_policy


def test_cli_adaptive_mtp_depth_is_explicit_and_default_off():
    parser = build_parser()
    assert not parser.parse_args(["--model", "fixture"]).adaptive_mtp_depth
    assert parser.parse_args(
        ["--model", "fixture", "--adaptive-mtp-depth"]
    ).adaptive_mtp_depth


@pytest.mark.parametrize(
    ("external_draft", "prompt_lookup", "enabled"),
    [
        (False, False, True),  # ordinary and native self-MTP use BatchGenerator
        (True, False, False),
        (False, True, False),
        (True, True, False),
    ],
)
def test_decode_time_fairness_identity_matches_selected_route(
    external_draft, prompt_lookup, enabled
):
    policy = decode_time_fairness_policy(
        external_draft=external_draft, prompt_lookup=prompt_lookup
    )
    assert policy["enabled"] is enabled


def test_qualification_requires_observed_adaptive_mtp_engagement():
    assert "feature_adaptive_mtp_depth" in required_feature_checks(
        {
            "mtp": True,
            "adaptive_mtp_depth": {"enabled": True},
            "execution_policy": {},
            "environment": {},
            "max_context": 1024,
        }
    )
    assert "feature_adaptive_mtp_depth" not in required_feature_checks(
        {
            "mtp": True,
            "adaptive_mtp_depth": {"enabled": False},
            "execution_policy": {},
            "environment": {},
            "max_context": 1024,
        }
    )


def test_completed_adaptive_receipt_publishes_bounded_qualification_evidence():
    from scripts.qualify_serving import feature_observations

    batch = NS(scheduler_stats={})
    response = NS(
        mtp_receipt={
            "adaptive_depth": {
                "selected": True,
                "counters": {
                    "boundaries": 7,
                    "depth_changes": 2,
                    "depth_decreases_concurrent": 1,
                    "depth_recoveries_alone": 1,
                },
                "cost_model": {
                    "active_bucket": "5-8",
                    "buckets": {
                        "5-8": {
                            "chosen_depth": 0,
                            "goodput_tokens_per_second": {"0": 107.35, "2": 98.72},
                        }
                    },
                },
            }
        }
    )
    generate.BatchGenerator._observe_adaptive_mtp_responses(
        batch, [response, response]
    )
    assert batch.scheduler_stats["adaptive_mtp_boundaries"] == 7
    assert batch.scheduler_stats["adaptive_mtp_depth_changes"] == 2
    assert batch.scheduler_stats["adaptive_mtp_depth_decreases_concurrent"] == 1
    assert batch.scheduler_stats["adaptive_mtp_depth_recoveries_alone"] == 1
    assert batch.scheduler_stats["adaptive_mtp_cost_model"]["buckets"]["5-8"][
        "chosen_depth"
    ] == 0
    benchmark = {"adaptive_qualification": {
        "passed": True,
        "features": {
            "adaptive_mtp_depth": {"selected": True, "passed": True},
        },
    }}
    assert feature_observations(
        {"settings": {"mtp": True}, "scheduler": batch.scheduler_stats},
        adaptive_benchmark=benchmark,
    )["adaptive_mtp_depth"] == 1
    assert feature_observations(
        {
            "settings": {"mtp": True},
            "scheduler": {
                "adaptive_mtp_depth_decreases_concurrent": 3,
                "adaptive_mtp_depth_recoveries_alone": 0,
            },
        }
    )["adaptive_mtp_depth"] == 0


def test_adaptive_mtp_serving_selection_runs_unqualified_or_qualified():
    # Adaptive depth is exact; without a receipt it runs, labelled
    # unqualified (AGENTS.md).  Qualifying it still needs benchmark evidence.
    ServingEngine.validate_arguments("unused", adaptive_mtp_depth=True)
    ServingEngine.validate_arguments(
        "unused",
        qualification="qualified.json",
        adaptive_mtp_depth=True,
    )
    with pytest.raises(ValueError, match="native self-MTP"):
        ServingEngine(
            "unused",
            qualification_mode=True,
            mtp=False,
            adaptive_mtp_depth=True,
        )
    with pytest.raises(ValueError, match="unknown adaptive MTP settings"):
        ServingEngine(
            "unused",
            qualification_mode=True,
            adaptive_mtp_depth={"enabled": True, "mystery": 1},
        )


def test_execution_policy_can_select_adaptive_mtp_and_is_fail_closed():
    ServingEngine.validate_arguments(
        "unused",
        execution_policy={"adaptive_mtp_depth": {"enabled": True}},
    )
    ServingEngine.validate_arguments(
        "unused",
        qualification="qualified.json",
        execution_policy={"adaptive_mtp_depth": {"enabled": True}},
    )
    with pytest.raises(ValueError, match="conflicts with disabled"):
        ServingEngine.validate_arguments(
            "unused",
            qualification_mode=True,
            adaptive_mtp_depth=True,
            execution_policy={"adaptive_mtp_depth": {"enabled": False}},
        )


def test_mtp_ordinary_handoff_is_explicit_qualified_and_native_only():
    selected = {
        "mtp_ordinary_handoff": {
            "enabled": True,
            "max_mtp_width": 8,
        }
    }
    assert "feature_mtp_ordinary_handoff" in required_feature_checks(
        {
            "mtp": True,
            "mtp_ordinary_handoff": selected["mtp_ordinary_handoff"],
            "execution_policy": {},
            "environment": {},
            "max_context": 1024,
        }
    )
    with pytest.raises(ValueError, match="enabled.*true"):
        ServingEngine.validate_arguments(
            "unused",
            qualification_mode=True,
            execution_policy={"mtp_ordinary_handoff": {"max_mtp_width": 8}},
        )
    # Runs unqualified without a receipt; qualifying it still requires the
    # feature check above.
    ServingEngine.validate_arguments("unused", execution_policy=selected)
    ServingEngine.validate_arguments(
        "unused", qualification="qualified.json", execution_policy=selected
    )
    with pytest.raises(ValueError, match="native self-MTP"):
        ServingEngine.validate_arguments(
            "unused",
            qualification_mode=True,
            mtp=False,
            execution_policy=selected,
        )


def test_handoff_observation_requires_safe_live_benchmark_evidence():
    from scripts.qualify_serving import feature_observations

    evidence = {
        "adaptive_qualification": {
            "passed": True,
            "features": {
                "adaptive_mtp_depth": {"selected": False, "passed": True},
                "mtp_ordinary_handoff": {"selected": True, "passed": True},
            },
            "handoff": {
                "selected": True,
                "passed": True,
                "events": 1,
                "comparison_count": 8,
                "exact_match_fraction": 0.875,
                "unsafe_divergences": [],
            },
        }
    }
    assert feature_observations(
        {}, adaptive_benchmark=evidence
    )["mtp_ordinary_handoff"] == 1
    assert feature_observations(
        {}, adaptive_benchmark=evidence
    )["adaptive_mtp_depth"] == 0
    evidence["adaptive_qualification"]["handoff"]["unsafe_divergences"] = [
        {"divergence_phase": "post_handoff"}
    ]
    assert feature_observations(
        {}, adaptive_benchmark=evidence
    )["mtp_ordinary_handoff"] == 0


def test_qualification_engine_propagates_adaptive_mtp_identity_and_policy(
    monkeypatch,
):
    captured = {}

    class APC:
        def __init__(self, **_kwargs):
            self.apc_stats = {}

        def key(self, *_args, **_kwargs):
            return "key"

        def spill_idle_entries(self):
            pass

        def clear(self):
            pass

    class Batch:
        scheduler_stats = {}

        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)
            self.lanes = {}

        def close(self):
            pass

    class Adapter:
        max_context = 64
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake-layout"
        model = None
        tokenizer = NS(vocab_size=32, eos_token_ids=[])

        def __init__(self, _path):
            pass

        def profile_name(self, _mtp):
            return "fake-mtp"

        def execution_config(self, **_kwargs):
            return {"num_draft": 2}

        def diagnostics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(
        serving, "runtime_identity", lambda: {"source_sha256": "source"}
    )
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = ServingEngine(
        "fake",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=True,
        max_lanes=2,
        max_inflight=4,
        adaptive_mtp_depth={
            "enabled": True,
            "ewma_alpha": 0.5,
            "loss_rounds": 2,
        },
    )
    try:
        assert engine.ready.wait(5)
        assert engine.error is None
        assert engine.snapshot["profile"] == "fake-mtp-adaptive-mtp-depth"
        selected = engine.snapshot["settings"]["adaptive_mtp_depth"]
        assert selected["enabled"] is True
        assert "mtp_ordinary_handoff" not in engine.snapshot["settings"]
        assert selected["ewma_alpha"] == 0.5
        assert selected["loss_rounds"] == 2
        assert engine.snapshot["settings"]["decode_time_fairness"]["enabled"]
        assert captured["adaptive_mtp_depth"] == {
            "ewma_alpha": 0.5,
            "shrink_gate": 0.35,
            "grow_gate": 0.8,
            "loss_rounds": 2,
                "gain_rounds": 3,
                "park_rounds": 4,
                "goodput_alpha": 0.25,
                "goodput_hysteresis": 0.05,
                "min_samples_per_depth": 3,
                "goodput_window": 8,
                "probe_interval": 16,
                "stale_rounds": 128,
            }
        assert captured["decode_time_fairness"]["enabled"]
    finally:
        engine.close()
