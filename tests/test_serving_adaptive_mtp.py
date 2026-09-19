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
                "counters": {"boundaries": 7, "depth_changes": 2},
            }
        }
    )
    generate.BatchGenerator._observe_adaptive_mtp_responses(
        batch, [response, response]
    )
    assert batch.scheduler_stats["adaptive_mtp_boundaries"] == 7
    assert batch.scheduler_stats["adaptive_mtp_depth_changes"] == 2
    assert feature_observations(
        {"settings": {"mtp": True}, "scheduler": batch.scheduler_stats}
    )["adaptive_mtp_depth"] == 7


def test_adaptive_mtp_serving_selection_is_qualification_only():
    with pytest.raises(ValueError, match="qualification mode"):
        ServingEngine("unused", adaptive_mtp_depth=True)
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
        }
        assert captured["decode_time_fairness"]["enabled"]
    finally:
        engine.close()
