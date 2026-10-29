"""Ladder verdicts keep functional, exact replay and width contracts separate."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v41_ladder", ROOT/"scripts/research/deepseek_v41_context_ladder.py")
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)


def arm(code, hit, tokens=(1, 2, 3)):
    return {"text": code, "reasoning": "", "token_ids": list(tokens),
            "token_trace": {"execution_widths": [1]*len(tokens)},
            "receipt": {"cache": "apcv2", "cached_tokens": hit, "prompt_tokens": 960,
                        "route": "external_draft", "route_receipt": "unqualified",
                        "speculation": {"verification": "exact"}}}


def test_uncontrolled_ladder_exact_pair_and_functional_failure_are_distinct():
    cold, warm = arm("V41-ABC", 0), arm("V41-ABC", 959)
    assert ladder.assess(cold, warm, "V41-ABC", 960)["passed"]
    cold["text"] = warm["text"] = "wrong"
    verdict = ladder.assess(cold, warm, "V41-ABC", 960)
    assert verdict["contract_passed"] and not verdict["functional_passed"]
    assert verdict["classification"] == "functional_outcome_failure"


def test_first_divergence_is_localized_and_hit_credit_stays_exact():
    cold, warm = arm("V41-ABC", 0), arm("V41-ABC", 959, (1, 9, 3))
    verdict = ladder.assess(cold, warm, "V41-ABC", 960)
    assert verdict["first_divergence"] == {"zero_based_position": 1, "shared_prefix": [1], "cold_token": 2, "warm_token": 9}
    warm = arm("V41-ABC", 900)
    assert not ladder.assess(cold, warm, "V41-ABC", 960)["gates"]["warm_exact_hit"]


def test_identifier_normalization_preserves_case_and_other_characters():
    assert ladder.normalize_code(" V41–ABC\n") == "V41-ABC"
    assert ladder.normalize_code("v41-ABC") != "V41-ABC"
    assert ladder.normalize_code('"V41-ABC"') != "V41-ABC"


def test_model_free_draft_empty_cache_survives_disk_roundtrip(tmp_path):
    import io

    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    path = tmp_path/"empty-draft.safetensors"
    save_prompt_cache(str(path), [], metadata={"binding": "synthetic@empty-draft-v1"})
    assert load_prompt_cache(str(path)) == []
    assert load_prompt_cache(io.BytesIO(path.read_bytes()), return_metadata=True) == (
        [], {"binding": "synthetic@empty-draft-v1"})


def test_empty_cache_marker_never_hides_nonempty_arrays(tmp_path):
    import mlx.core as mx
    import pytest

    from mlx2.runtime.models.cache import load_prompt_cache

    mx.set_default_device(mx.cpu)
    path = tmp_path/"corrupt-empty.safetensors"
    mx.save_safetensors(str(path), {"unexpected": mx.array([1])}, {"mlx2.empty_prompt_cache.v1": "{}"})
    with pytest.raises(ValueError, match="unexpected data"):
        load_prompt_cache(str(path))


def test_apcv2_disk_refault_restores_model_free_draft_sidecar(tmp_path):
    import mlx.core as mx

    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.external_speculative import ExternalDraftState
    from mlx2.runtime.models.cache import KVCache

    mx.set_default_device(mx.cpu)
    clock = [0.0]
    apc = APCv2(max_size=4, max_bytes=1 << 20, layout_name="synthetic-empty-draft-v1",
                idle_disk_seconds=1, idle_disk_dir=str(tmp_path), now_fn=lambda: clock[0])
    target = KVCache()
    values = mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1)
    target.update_and_fetch(values, values)
    sidecar = ExternalDraftState(([], mx.ones((1, 1, 4))), covered_tokens=3, binding="synthetic@copy")
    key = APCKey("synthetic@empty-draft-refault")
    hit = None
    try:
        apc.store(key, [1, 2, 3], [target], sidecar=sidecar)
        clock[0] = 2
        assert apc.spill_idle_entries() == 1
        hit = apc.lookup(key, [1, 2, 3, 4], require_sidecar=True)
        assert hit.cached_tokens == 3 and hit.sidecar.state[0] == []
        hit.sidecar.validate("synthetic@copy", 3)
        assert mx.array_equal(hit.cache[0].keys[..., :3, :], values).item()
        assert apc.apc_stats["idle_disk"]["restores"] == 1
    finally:
        if hit is not None and hit.cache is not None:
            hit.cache.close()
        apc.close()
