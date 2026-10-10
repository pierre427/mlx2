"""APCv2 lookup roles must not be confused with output publication receipts."""
import copy
import importlib.util
from pathlib import Path

import pytest

from mlx2.cache_receipts import prompt_boundary_replay_evidence

SHA = "a" * 64


def _pair():
    return (
        {"prompt_tokens": 20, "cached_tokens": 0, "cache_checkpoint_role": None},
        {"prompt_tokens": 20, "cached_tokens": 19,
         "cache_checkpoint_role": "committed_prompt_boundary"},
    )


def _check(cold, warm, **kwargs):
    return prompt_boundary_replay_evidence(
        cold, warm, expected_prompt_tokens=kwargs.get("tokens", 20),
        cold_prompt_sha256=SHA, warm_prompt_sha256=kwargs.get("sha", SHA),
    )


def test_cold_null_and_warm_committed_role_prove_boundary_reuse():
    cold, warm = _pair()
    original = copy.deepcopy((cold, warm))
    evidence = _check(cold, warm)
    assert evidence["passed"]
    assert evidence["publication_evidence"] == "same_prompt_warm_lookup"
    assert evidence["cold_lookup_role"] is None
    assert (cold, warm) == original


@pytest.mark.parametrize("arm,field,value", [
    (0, "cached_tokens", 1), (0, "cached_tokens", False),
    (0, "cache_checkpoint_role", "committed_prompt_boundary"),
    (1, "cached_tokens", 18), (1, "cache_checkpoint_role", "completion"),
    (1, "cache_checkpoint_role", None), (1, "prompt_tokens", 21),
])
def test_wrong_or_partial_lookup_evidence_fails(arm, field, value):
    pair = _pair()
    pair[arm][field] = value
    assert not _check(*pair)["passed"]


@pytest.mark.parametrize("arm", [0, 1])
def test_missing_role_field_is_not_an_explicit_cold_null(arm):
    pair = _pair()
    del pair[arm]["cache_checkpoint_role"]
    assert not _check(*pair)["passed"]


@pytest.mark.parametrize("tokens", [None, "20", True, 1])
def test_invalid_prompt_length_fails_without_guessing(tokens):
    assert not _check(*_pair(), tokens=tokens)["passed"]


def test_different_prompt_cannot_witness_cold_publication():
    assert not _check(*_pair(), sha="b" * 64)["passed"]


@pytest.mark.parametrize("publish", [True, False])
def test_real_serving_receipt_role_is_lookup_provenance(monkeypatch, publish):
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False)
    request = {"tokens": list(range(1, 21)), "max_tokens": 3, "temperature": 0}
    try:
        cold = run(engine, {**request, "skip_writing_prefix_cache": not publish})
        warm = run(engine, request)
        assert cold["finish"] == warm["finish"] == "length"
        assert cold["receipt"]["cached_tokens"] == 0
        assert cold["receipt"]["cache_checkpoint_role"] is None
        evidence = _check(cold["receipt"], warm["receipt"])
        assert evidence["passed"] is publish
        if publish:
            assert warm["receipt"]["cached_tokens"] == 19
            assert warm["receipt"]["cache_checkpoint_role"] == "committed_prompt_boundary"
        else:
            assert warm["receipt"]["cached_tokens"] == 0
            assert warm["receipt"]["cache_checkpoint_role"] is None
    finally:
        engine.close()


def test_offline_checker_preserves_failed_source_and_rejects_nonempty_cache():
    path = Path(__file__).parents[1] / "scripts/check_prompt_boundary_replay.py"
    spec = importlib.util.spec_from_file_location("_cache_gate_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cold, warm = _pair()
    report = {
        "state": "failed", "failures": ["cold: committed prompt boundary absent"],
        "prompt_recipes": [{"lane": 0, "prompt_tokens": 20}],
        "arms": [{"arm": name, "rows": [{"lane": 0, "route_receipt": receipt,
                                         "prompt_sha256": SHA}]}
                 for name, receipt in (("cold", cold), ("warm", warm))],
        "server_initial_status": {"apcv2": {"lookups": 0, "hits": 0, "stores": 0,
                                            "persistence": {"enabled": False}}},
    }
    original = copy.deepcopy(report)
    result = module.assess(report)
    assert result["prompt_boundary_gate"] == "passed"
    assert result["original_state"] == "failed"
    assert result["qualification_changed"] is False
    assert report == original
    report["server_initial_status"]["apcv2"]["stores"] = 1
    assert module.assess(report)["prompt_boundary_gate"] == "failed"

    report = copy.deepcopy(original)
    report["arms"][0]["rows"].append(copy.deepcopy(report["arms"][0]["rows"][0]))
    assert module.assess(report)["prompt_boundary_gate"] == "failed"
