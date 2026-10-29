"""Synthetic identity-contract tests; no checkpoint or model execution."""

import hashlib
import json

import pytest

from mlx2.adapters.deepseek_v41_identity import effective_source_identity


def test_original_identity_is_preserved_without_mutating_provider():
    source = {"fingerprint": "a" * 64, "weight_sha256": "a" * 64, "path": "/source"}
    original = source.copy()
    result = effective_source_identity(source)
    assert source == original
    assert result["fingerprint"] == result["base_fingerprint"] == source["fingerprint"]
    assert "derivation" not in result


def test_derivative_matches_existing_research_digest_and_binds_all_weights():
    source = {"fingerprint": "a" * 64, "weight_sha256": "a" * 64, "path": "/source"}
    bindings = {"source": "a" * 64, "q1": "b" * 64, "hotmap": "c" * 64}
    result = effective_source_identity(source, q1_bank=bindings["q1"], hot_map=bindings["hotmap"])
    expected = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
    assert result["fingerprint"] == result["weight_sha256"] == expected
    assert result["base_fingerprint"] == source["fingerprint"]
    assert {k: result["derivation"][k] for k in bindings} == bindings
    for changed in ({"q1_bank": "d" * 64, "hot_map": "c" * 64},
                    {"q1_bank": "b" * 64, "hot_map": "d" * 64}):
        assert effective_source_identity(source, **changed)["fingerprint"] != expected
    moved = {**source, "path": "/relocated-source"}
    assert effective_source_identity(moved, q1_bank="b" * 64, hot_map="c" * 64)["fingerprint"] == expected
    assert source["fingerprint"] == "a" * 64


@pytest.mark.parametrize("q1,hot", [(None, "c" * 64), ("b" * 64, None), ("", "c" * 64),
                                   ("B" * 64, "c" * 64), ("b" * 64, 1)])
def test_partial_or_unsealed_derivative_refused(q1, hot):
    with pytest.raises(ValueError, match="sealed SHA256"):
        effective_source_identity({"fingerprint": "a" * 64}, q1_bank=q1, hot_map=hot)
