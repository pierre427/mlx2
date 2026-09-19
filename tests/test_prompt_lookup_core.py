import pytest

from mlx2.runtime.prompt_lookup import (
    AdaptiveLookback,
    IndexedPromptLookup,
    plan_proposal_around_verify_cliff,
    verify_prompt_lookup,
)


def test_indexed_oracle_hot_segment_and_rejection_ttl():
    oracle = IndexedPromptLookup([1, 2, 3, 9, 1, 2, 3], ngram_min=3, ngram_max=3)
    assert oracle.propose(4) == [9, 1, 2, 3]
    oracle.feedback(4, 0)
    assert oracle.propose(4) == []
    oracle.add_hot_segment([8, 1, 2, 3, 7, 6])
    assert oracle.propose(2) == [7, 6]


def test_verifier_accepts_prefix_and_returns_exact_boundary():
    full = verify_prompt_lookup([4, 5], [4, 5, 6])
    assert full.accepted == (4, 5) and full.next_token == 6
    assert full.receipt["bonus"]
    partial = verify_prompt_lookup([4, 8], [4, 5, 6])
    assert partial.accepted == (4,) and partial.next_token == 5
    assert partial.receipt["verified"]


def test_adaptive_lookback_widens_on_miss_and_narrows_on_reject():
    policy = AdaptiveLookback((8, 32), misses=2, rejects=1)
    policy.observe(0, 0)
    policy.observe(0, 0)
    assert policy.current == 32 and policy.widen_events == 1
    policy.observe(3, 0)
    assert policy.current == 8 and policy.narrow_events == 1


def test_verify_cliff_planner_extends_or_snaps_without_inventing_capacity():
    assert plan_proposal_around_verify_cliff(8, 15) == 15
    assert plan_proposal_around_verify_cliff(8, 10) == 7
    assert plan_proposal_around_verify_cliff(4, 20) == 4
    with pytest.raises(ValueError):
        plan_proposal_around_verify_cliff(-1, 2)


def test_prompt_lookup_policy_rejects_unknown_keys():
    import pytest

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    assert PromptLookupBatchGenerator.validate_policy({"num_draft": 4}) == {"num_draft": 4}
    with pytest.raises(ValueError, match="unknown prompt_lookup policy keys: numdraft"):
        PromptLookupBatchGenerator.validate_policy({"numdraft": 4})
    with pytest.raises(ValueError, match="must be an object"):
        PromptLookupBatchGenerator.validate_policy([1])


def test_prompt_lookup_policy_rejects_coerced_or_nonpositive_num_draft():
    import pytest

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    for invalid in (True, False, 0, -1, 1.9, "3", None):
        with pytest.raises(ValueError, match="positive integer"):
            PromptLookupBatchGenerator.validate_policy({"num_draft": invalid})

    assert PromptLookupBatchGenerator.validate_policy({"num_draft": 3}) == {
        "num_draft": 3
    }


def test_prompt_lookup_policy_strictly_validates_every_runtime_value():
    import math

    import pytest

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    invalid = (
        {"ngram_min": True},
        {"ngram_max": 2.5},
        {"ngram_min": 4, "ngram_max": 3},
        {"hot_segments": -1},
        {"reject_ttl": "8"},
        {"lookback_misses": 0},
        {"lookback_rejects": False},
        {"adaptive_warmup": -1},
        {"adaptive": "false"},
        {"adaptive_gate": "nan"},
        {"adaptive_gate": math.nan},
        {"adaptive_gate": math.inf},
        {"adaptive_gate": -0.1},
        {"adaptive_gate": 1.1},
        {"deferred_admission": 1},
        {"cliff_aware_span": "true"},
        {"admission_window": 0},
        {"admission_confirm_windows": False},
        {"admission_reprobe_interval": -1},
        {"admission_gate": "0.5"},
        {"admission_gate": math.nan},
        {"admission_gate": math.inf},
        {"admission_gate": -0.1},
        {"admission_gate": 1.1},
        {"verify_cliff_start": 1, "verify_cliff_end": 1},
        {"verify_cliff_end": 2.5},
        {"verify_cliff_start": 10, "verify_cliff_end": 9},
        {"lookback_ladder": "8,32"},
        {"lookback_ladder": []},
        {"lookback_ladder": [8, "32"]},
        {"lookback_ladder": [8, 8]},
        {"retrieval_segments": "1,2"},
        {"retrieval_segments": [[]]},
        {"retrieval_segments": [[1, "2"]]},
        {"retrieval_segments": [[-1]]},
    )
    for policy in invalid:
        with pytest.raises(ValueError):
            PromptLookupBatchGenerator.validate_policy(policy)

    assert PromptLookupBatchGenerator.validate_policy(
        {
            "ngram_min": 2,
            "ngram_max": 4,
            "hot_segments": 0,
            "reject_ttl": 0,
            "lookback_ladder": (8, 32),
            "lookback_misses": 1,
            "lookback_rejects": 1,
            "retrieval_segments": ((1, 2),),
            "adaptive": False,
            "adaptive_warmup": 0,
            "adaptive_gate": 1,
            "deferred_admission": True,
            "admission_window": 8,
            "admission_gate": 0.5,
            "admission_confirm_windows": 2,
            "admission_reprobe_interval": 0,
            "cliff_aware_span": True,
            "verify_cliff_start": 9,
            "verify_cliff_end": 15,
        }
    ) == {
        "ngram_min": 2,
        "ngram_max": 4,
        "hot_segments": 0,
        "reject_ttl": 0,
        "lookback_ladder": [8, 32],
        "lookback_misses": 1,
        "lookback_rejects": 1,
        "retrieval_segments": [[1, 2]],
        "adaptive": False,
        "adaptive_warmup": 0,
        "adaptive_gate": 1.0,
        "deferred_admission": True,
        "admission_window": 8,
        "admission_gate": 0.5,
        "admission_confirm_windows": 2,
        "admission_reprobe_interval": 0,
        "cliff_aware_span": True,
        "verify_cliff_start": 9,
        "verify_cliff_end": 15,
    }


def test_prompt_lookup_lane_policy_overrides_use_the_same_strict_contract():
    import pytest

    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    generator = PromptLookupBatchGenerator(object())
    with pytest.raises(ValueError, match="adaptive must be a boolean"):
        generator.insert(
            [[1]],
            caches=[[KVCache()]],
            prompt_lookup_configs=[{"adaptive": "false"}],
        )


def test_prompt_lookup_policy_allowlist_covers_every_key_the_generator_reads():
    import re
    from pathlib import Path

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    source = Path(PromptLookupBatchGenerator.__module__.replace(".", "/") + ".py")
    source = Path(__file__).resolve().parents[1] / "src" / source
    read_keys = set(re.findall(r'config(?:\.get\(|\[)"([a-z_]+)"', source.read_text()))
    assert read_keys, "expected the generator to read policy keys"
    assert read_keys <= PromptLookupBatchGenerator.POLICY_KEYS


def test_prompt_lookup_policy_null_is_still_a_misplaced_key(monkeypatch):
    """An explicit ``"prompt_lookup": null`` must not slip past the route check."""
    from types import SimpleNamespace as NS

    import pytest

    from mlx2 import serving

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})

    class Adapter:
        max_context = 1000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tokenizer = NS(vocab_size=10, eos_token_ids=[])

        def __init__(self, _path, execution_policy=None):
            pass

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, **_kw):
            return {"num_draft": 0}

        def diagnostics(self):
            return {}

        def close(self):
            pass

    engine = serving.ServingEngine(
        "fake", adapter_factory=Adapter, qualification_mode=True, mtp=False,
        execution_policy={"prompt_lookup": None},
    )
    try:
        engine.thread.join(5)
        assert engine.error and "prompt-lookup route is not selected" in engine.error
        with pytest.raises(RuntimeError):
            engine.submit({"prompt": "hi"})
    finally:
        engine.close()
