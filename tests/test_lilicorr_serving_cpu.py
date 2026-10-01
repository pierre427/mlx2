"""Initial-prompt, chunked prefill and revision-bound APC pending anchor parity."""

import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from test_standard_xpress_serving_cpu import drain, reference, tiny

from mlx2.adapters.lilicorr import LiLiCorrConfig
from mlx2.runtime.drafters.lilicorr import LiLiCorrDraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator


def pair():
    model, _ = tiny()
    config = LiLiCorrConfig(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=9,
        mask_token_id=8,
        num_target_layers=3,
        target_layer_ids=[0, 2],
        block_size=4,
        layer_types=["full_attention"],
        lilicorr_hidden_size=6,
        lilicorr_candidate_topk=2,
        lilicorr_num_layers=2,
        lilicorr_num_heads=2,
        lilicorr_mlp_ratio=1.5,
        lilicorr_factor_dim=3,
    )
    draft = LiLiCorrDraftModel(config).bind(model)
    return model, draft


def engine(model, draft, step=3):
    return ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="tiny-lilicorr-v1",
        num_draft=2,
        prefill_step_size=step,
    )


def capture_anchor(draft):
    captured = []
    original = draft.lilicorr.score

    def score(embeddings, probs, hidden, anchor, valid):
        captured.append((np.asarray(anchor).copy(), np.asarray(valid).copy()))
        return original(embeddings, probs, hidden, anchor, valid)

    draft.lilicorr.score = score
    return captured


@pytest.mark.parametrize("step", [1, 3, 32])
def test_first_draft_preserves_actual_committed_anchor_and_ordinary_output(step):
    model, draft = pair()
    scheduler = engine(model, draft, step)
    prompt = [1, 2, 3, 4, 5]
    expected_taps = model.prefill_body(
        mx.array([prompt[:-1]]), model.make_cache(), [0, 2]
    )
    expected_anchor = np.asarray(draft.hidden_norm(draft.fc(expected_taps[:, -1])))
    captured = capture_anchor(draft)
    scheduler.insert([prompt], max_tokens=[6])
    lane = scheduler.lanes[0]
    while lane.remaining:
        scheduler._prefill(lane)
    assert lane.tail.shape[1] > 0
    assert lane.draft_cache[0].offset + lane.tail.shape[1] == len(lane.history)
    outputs, finish = drain(scheduler)
    np.testing.assert_allclose(captured[0][0], expected_anchor, rtol=1e-5, atol=1e-6)
    assert captured[0][1].tolist() == [True]
    assert outputs[0] == reference(model, prompt, 6)
    assert finish[0].speculative_receipt["kind"] == "external_lilicorr"
    assert scheduler.scheduler_stats["draft_fallbacks"] == 0


def test_apcv2_first_prompt_boundary_restores_same_pending_anchor(
    tmp_path, monkeypatch
):
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    model, draft = pair()
    initial = engine(model, draft, step=2)
    prompt = [1, 2, 3, 4, 5]
    initial.insert([prompt], max_tokens=[4])
    lane = initial.lanes[0]
    while lane.remaining:
        initial._prefill(lane)
    pending = np.asarray(lane.tail).copy()
    sidecar = initial._sidecar(lane)
    apc = APCv2(max_size=2, layout_name="tiny-lilicorr")
    key = APCKey(
        "qwen3+lilicorr",
        revision="tiny-source-v1",
        cache_layout_fingerprint="tiny-lilicorr",
    )
    apc.store(key, lane.history, copy.deepcopy(lane.cache), sidecar=sidecar)
    hit = apc.lookup(key, prompt)
    assert hit.hit and hit.hit_kind == "external_draft_sidecar"
    resumed = engine(model, draft)
    resumed.insert(
        [[lane.anchor]],
        max_tokens=[4],
        caches=[hit.cache],
        all_tokens=[lane.history],
        cache_states=[hit.sidecar],
    )
    np.testing.assert_array_equal(np.asarray(resumed.lanes[0].tail), pending)
    captured = capture_anchor(draft)
    outputs, finished = drain(resumed)
    expected = np.asarray(draft.hidden_norm(draft.fc(mx.array(pending)[:, -1])))
    np.testing.assert_array_equal(captured[0][0], expected)
    assert captured[0][1].tolist() == [True]
    assert outputs[0] == reference(model, prompt, 4)
    finished[0].cache_sidecar.validate("tiny-lilicorr-v1", len(finished[0].all_tokens))
    assert resumed.scheduler_stats["paired_cache_resumes"] == 1
    hit.cache.close()
    initial.close()
    apc.clear(release_memory=False)


def test_cached_context_without_pending_anchor_fails_closed():
    _model, draft = pair()
    cache = draft.make_cache()
    draft.append_context(mx.ones((1, 2, 16)), cache)
    with pytest.raises(ValueError, match="missing pending anchor"):
        draft.draft_distributions(
            [1], mx.zeros((1, 0, 16)), cache, 2, [object()], [0.0]
        )
