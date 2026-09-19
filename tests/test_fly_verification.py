from types import SimpleNamespace as NS

import numpy as np
import pytest

from mlx2.runtime.speculative_sampling import (
    FLyVerificationPolicy,
    RequestRNG,
    verify_proposals,
)


class _FixedRNG:
    def __init__(self, uniforms):
        self.uniforms = iter(uniforms)

    def uniform(self):
        return float(next(self.uniforms))

    def sample(self, law):
        # The tests care about accepted length and relaxed count; keep the
        # correction/bonus draw deterministic without consuming another coin.
        return int(np.argmax(law))


def _law(*values):
    return np.asarray(values, dtype=np.float64)


def test_fly_relaxes_only_when_entropy_probability_and_window_all_pass():
    tokens = [0, 1]
    proposals = [_law(1, 0, 0), _law(0, 1, 0)]
    targets = [
        _law(0.2, 0.4, 0.4),
        _law(0, 1, 0),
        _law(0, 0, 1),
    ]
    policy = {
        "enabled": True,
        "entropy_threshold": 1.0,
        "window": 1,
        "min_prob": 0.2,
    }
    accepted = verify_proposals(
        tokens,
        proposals,
        targets,
        _FixedRNG([0.9, 0.1]),
        fly_verification=policy,
    )
    assert accepted.accepted == 2
    assert accepted.relaxed_accepts == 1
    assert accepted.emitted == (0, 1, 2)

    failures = [
        {**policy, "entropy_threshold": 1.1},
        {**policy, "min_prob": 0.21},
    ]
    for failed_policy in failures:
        rejected = verify_proposals(
            tokens,
            proposals,
            targets,
            _FixedRNG([0.9, 0.1]),
            fly_verification=failed_policy,
        )
        assert rejected.accepted == rejected.relaxed_accepts == 0

    rejected_window = verify_proposals(
        tokens,
        proposals,
        [targets[0], _law(0.45, 0.1, 0.45), targets[2]],
        _FixedRNG([0.9, 0.9]),
        fly_verification=policy,
    )
    assert rejected_window.accepted == rejected_window.relaxed_accepts == 0


def test_fly_default_off_preserves_rng_and_output_bit_exactly():
    tokens = [0, 1]
    proposals = [_law(0.6, 0.2, 0.2), _law(0.1, 0.8, 0.1)]
    targets = [_law(0.5, 0.3, 0.2), _law(0.2, 0.7, 0.1), _law(0.1, 0.2, 0.7)]
    baseline_rng = RequestRNG(37)
    disabled_rng = RequestRNG(37)
    baseline = verify_proposals(tokens, proposals, targets, baseline_rng)
    disabled = verify_proposals(
        tokens,
        proposals,
        targets,
        disabled_rng,
        fly_verification={"enabled": False},
    )
    assert baseline.emitted == disabled.emitted
    assert baseline.accepted == disabled.accepted
    assert baseline_rng.snapshot() == disabled_rng.snapshot()
    for left, right in zip(
        baseline.target_probabilities, disabled.target_probabilities
    ):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    "value,error",
    [
        ({"mystery": 1}, "unknown FLy"),
        ({"enabled": 1}, "enabled must be boolean"),
        ({"entropy_threshold": -1}, "nonnegative"),
        ({"window": 0}, "positive integer"),
        ({"min_prob": 1.1}, r"\[0, 1\]"),
    ],
)
def test_fly_policy_rejects_unknown_and_invalid_values(value, error):
    with pytest.raises(ValueError, match=error):
        FLyVerificationPolicy.from_value(value)


def test_serving_execution_policy_propagates_fly_and_rejects_invalid(monkeypatch):
    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="unknown FLy"):
        ServingEngine(
            "unused",
            execution_policy={"fly_verification": {"mystery": 1}},
        )

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
        def __init__(self, *_args, **kwargs):
            self.scheduler_stats = {}
            captured.update(kwargs)
            self.lanes = {}

        def close(self):
            pass

    class Adapter:
        max_context = 64
        layout = "fake-layout"
        model = None
        tokenizer = NS(vocab_size=32, eos_token_ids=[])

        def __init__(self, _path):
            self.identity = {"fingerprint": "fake"}
            self.environment = {}

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
    selected = {
        "enabled": True,
        "entropy_threshold": 1.5,
        "window": 3,
        "min_prob": 0.02,
    }
    engine = ServingEngine(
        "fake",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=True,
        max_lanes=2,
        max_inflight=4,
        execution_policy={"fly_verification": selected},
    )
    try:
        assert engine.ready.wait(5)
        assert engine.error is None
        assert engine.snapshot["settings"]["fly_verification"] == selected
        assert captured["fly_verification"].as_dict() == selected
    finally:
        engine.close()


def test_selected_fly_requires_observed_qualification():
    from mlx2.qualification import required_feature_checks

    base = {
        "mtp": True,
        "speculation": "self_mtp",
        "execution_policy": {},
        "environment": {},
        "max_context": 1024,
    }
    assert "feature_fly_verification" not in required_feature_checks(base)
    assert "feature_fly_verification" in required_feature_checks(
        {**base, "fly_verification": {"enabled": True}}
    )
    assert "feature_fly_verification" in required_feature_checks(
        {
            **base,
            "mtp": False,
            "speculation": "external_draft",
            "fly_verification": {"enabled": True},
        }
    )


def test_qualifier_observes_fly_only_from_engaged_receipts_or_counter():
    from scripts.qualify_serving import feature_observations

    enabled = feature_observations(
        {
            "scheduler": {"fly_relaxed_accepts": 0},
            "recent_receipts": [
                {"mtp": {"verification": "fly", "relaxed_accepts": 2}},
                {"speculation": {"verification": "exact", "relaxed_accepts": 9}},
            ],
        }
    )
    disabled = feature_observations(
        {
            "scheduler": {"fly_relaxed_accepts": 0},
            "recent_receipts": [
                {"mtp": {"verification": "exact", "relaxed_accepts": 3}},
                {"speculation": {"verification": "exact", "relaxed_accepts": 4}},
            ],
        }
    )
    counter_only = feature_observations(
        {"scheduler": {"fly_relaxed_accepts": 5}, "recent_receipts": []}
    )
    assert enabled["fly_verification"]==2
    assert disabled["fly_verification"]==0
    assert counter_only["fly_verification"]==5
