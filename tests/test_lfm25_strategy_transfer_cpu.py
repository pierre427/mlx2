"""CPU/static gates for the LFM2.5-VL exact-prefix strategy transfer."""

import pytest

from mlx2.adapters.lfm25_vl import (
    CACHE_LAYOUT,
    SOURCE_REVISION,
    LFM25VLAdapter,
)
from mlx2.runtime.exact_prefix_cascade import (
    longest_first_paths,
    next_cascade_stage,
)


def _adapter():
    adapter = object.__new__(LFM25VLAdapter)
    adapter.identity = {"fingerprint": "target-fingerprint"}
    adapter._media_proof_key = b"lfm-strategy-cpu-media-key"
    return adapter


def _binding(**updates):
    adapter = _adapter()
    media_tokens = [10, 124907, 11, 12]
    media_fingerprint = "ordered-image-or-frame-fingerprint"
    media_end = 2
    value = {
        "execution_domain": "autoregressive_language_only",
        "cache_layout": CACHE_LAYOUT,
        "artifact_fingerprint": "target-fingerprint",
        "source_revision": SOURCE_REVISION,
        "checkpoint_kind": "ordinary_exact",
        "prefix_start_position": 120,
        "checkpoint_position": 122,
        "state_planes": (
            "attention_kv",
            "recurrent",
            "rng",
            "transcript",
        ),
        "attention_layers": 8,
        "recurrent_layers": 22,
        "batch_size": 1,
        "media_token_end": media_end,
        "media_fingerprint": media_fingerprint,
        "media_prompt_tokens": media_tokens,
        "media_proof": adapter._media_checkpoint_proof(
            media_tokens, media_fingerprint, media_end
        ),
        "dspark_state": "absent",
    }
    value.update(updates)
    return value


def test_longest_first_is_stable_and_deduplicates_complete_paths():
    assert longest_first_paths(
        [(1,), (2, 3, 4), (5, 6), (2, 3, 4), (7, 8)]
    ) == ((2, 3, 4), (5, 6), (7, 8), (1,))


def test_prefix_pruning_and_suffix_only_reuse_are_model_neutral():
    paths = [(1, 2, 3, 4), (1, 2, 9), (1, 7, 8, 9), (6, 7)]
    first = next_cascade_stage(paths)
    assert first.path == (1, 2, 3, 4)
    assert first.suffix == first.path
    assert first.reused_prefix_tokens == 0

    second = next_cascade_stage(paths, (1, 2), attempted=(0,))
    assert second.path == (1, 2, 9)
    assert second.suffix == (9,)
    assert second.reused_prefix_tokens == 2
    assert second.viable_indices == (2,)
    assert set(second.pruned_indices) == {1, 3}


def test_lfm_contract_is_language_only_and_unselected():
    adapter = _adapter()
    assert adapter.prefill_step_default() is None
    contract = adapter.exact_prefix_cascade_contract()
    assert contract["execution_domain"] == "autoregressive_language_only"
    assert contract["verification_order"] == "longest_first"
    assert contract["invalid_sibling_pruning"] is True
    assert contract["shared_prefix_reuse"] == "suffix_only_after_exact_geometry_gate"
    assert contract["media_prefill"] == "bound_below_language_checkpoint"
    assert contract["dspark_state"] == "excluded"
    assert contract["apcv2_publication"] is False
    assert contract["qualified"] is contract["selected"] is False
    assert contract["observed_used"] is False


def test_lfm_reuses_exact_hybrid_prefix_without_recomputing_common_tokens():
    adapter = _adapter()
    paths = [(1, 2, 3, 4), (1, 2, 9), (1, 7, 8)]
    stage = adapter.plan_exact_prefix_cascade(
        paths,
        (1, 2),
        attempted=(0,),
        state_binding=_binding(),
    )
    assert stage.path == (1, 2, 9)
    assert stage.suffix == (9,)
    assert stage.reused_prefix_tokens == 2


def test_initial_lfm_stage_does_not_claim_reusable_state():
    stage = _adapter().plan_exact_prefix_cascade([(1, 2, 3), (4, 5)])
    assert stage.accepted_prefix == ()
    assert stage.suffix == (1, 2, 3)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"checkpoint_position": 123}, "checkpoint position"),
        ({"cache_layout": "other"}, "cache layout"),
        ({"artifact_fingerprint": "other"}, "artifact fingerprint"),
        ({"source_revision": "other"}, "source revision"),
        ({"checkpoint_kind": "multirow_verify"}, "checkpoint kind"),
        ({"state_planes": ("attention_kv", "recurrent")}, "state planes"),
        ({"attention_layers": 7}, "hybrid layer geometry"),
        ({"batch_size": 2}, "batch size"),
        ({"dspark_state": "bound"}, "DSpark state"),
        ({"media_token_end": 121}, "media boundary"),
        ({"media_fingerprint": None}, "media fingerprint"),
        ({"media_proof": "forged"}, "media proof"),
    ],
)
def test_lfm_shared_prefix_reuse_fails_closed_on_geometry_drift(updates, match):
    with pytest.raises(ValueError, match=match):
        _adapter().plan_exact_prefix_cascade(
            [(1, 2, 3), (1, 2, 9)],
            (1, 2),
            attempted=(0,),
            state_binding=_binding(**updates),
        )


def test_text_only_binding_has_no_media_or_dspark_state():
    stage = _adapter().plan_exact_prefix_cascade(
        [(1, 2, 3), (1, 2, 9)],
        (1, 2),
        attempted=(0,),
        state_binding=_binding(
            media_token_end=0,
            media_fingerprint=None,
            media_prompt_tokens=None,
            media_proof=None,
        ),
    )
    assert stage.suffix == (9,)


@pytest.mark.parametrize("paths", [[], [()], [(1, -1)], [(True, 2)]])
def test_invalid_cascade_paths_fail_closed(paths):
    with pytest.raises(ValueError):
        longest_first_paths(paths)
