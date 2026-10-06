"""Real tiny standard Qwen3 + XPress lifecycle tests, strictly CPU only."""

import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from mlx2.adapters.xpress import XPressConfig
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.base import create_attention_mask
from mlx2.runtime.models.standard_decoder import Model, ModelArgs
from mlx2.runtime.sample_utils import LaneRNG
from mlx2.runtime.speculative_sampling import RequestRNG, softmax, verify_proposals


def tiny(tied=False):
    mx.random.seed(13)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=8,
            num_hidden_layers=3,
            intermediate_size=12,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=9,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=128,
            rope_theta=10000,
            tie_word_embeddings=tied,
        )
    )
    from mlx2.runtime.drafters.xpress import XPressDraftModel

    draft = XPressDraftModel(
        XPressConfig(
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
            xpress_rank=3,
            xpress_mlp_hidden=5,
            xpress_num_passes=3,
        )
    ).bind(model)
    return model, draft


def generator(model, draft, **kwargs):
    return ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="tiny-xpress-v1",
        num_draft=2,
        prefill_step_size=3,
        **kwargs,
    )


def drain(engine):
    outputs, finishes = {}, {}
    for _ in range(100):
        _, responses = engine.next()
        for response in responses:
            outputs.setdefault(response.uid, []).append(response.token)
            if response.finish_reason:
                finishes[response.uid] = response
        if not engine.lanes:
            return outputs, finishes
    raise AssertionError("tiny XPress scheduler stalled")


def reference(model, prompt, length):
    cache, tokens = model.make_cache(), list(prompt)
    output = []
    for step in range(length):
        logits = model(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        output.append(token)
        tokens.append(token)
    return output


@pytest.mark.parametrize("tied", [False, True])
def test_post_layer_taps_match_independent_layer_walk_and_ordinary(tied):
    model, _ = tiny(tied)
    inputs = mx.array([[1, 2, 3], [3, 4, 5]])
    cache = model.make_cache()
    hidden = model.model.embed_tokens(inputs)
    mask = create_attention_mask(hidden, cache[0])
    captured = []
    for index, layer in enumerate(model.layers):
        hidden = layer(hidden, mask, cache[index])
        if index in (0, 2):
            captured.append(hidden)
    expected = mx.concatenate(captured, axis=-1)
    logits, taps = model.forward_with_taps(inputs, model.make_cache(), [0, 2])
    ordinary = model(inputs, cache=model.make_cache())
    np.testing.assert_array_equal(np.asarray(taps), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(ordinary))
    body = model.prefill_body(inputs, model.make_cache(), [0, 2])
    np.testing.assert_array_equal(np.asarray(body), np.asarray(expected))
    assert taps.shape == (2, 3, 16)
    for invalid in ([], [2, 0], [0, 0], [-1], [3], [True]):
        with pytest.raises(ValueError, match="capture"):
            model.forward_with_taps(inputs, model.make_cache(), invalid)


@pytest.mark.parametrize("tied", [False, True])
def test_real_xpress_greedy_matches_standard_reference_batched_and_budget(tied):
    model, draft = tiny(tied)
    engine = generator(model, draft)
    prompts, budgets = [[1, 2, 3, 4, 5], [2, 3], [1]], [7, 5, 1]
    ids = engine.insert(prompts, max_tokens=budgets)
    outputs, finishes = drain(engine)
    for uid, prompt, budget in zip(ids, prompts, budgets):
        assert outputs[uid] == reference(model, prompt, budget)
        end = finishes[uid]
        assert end.finish_reason == "length"
        end.cache_sidecar.validate("tiny-xpress-v1", len(end.all_tokens))
        assert end.speculative_receipt["kind"] == "external_xpress"
        settings = end.speculative_receipt["draft_settings"]
        assert {
            key: settings[key] for key in draft.receipt_settings
        } == draft.receipt_settings
        assert settings["minimum_proposal_length"] == 1
        assert settings["proposal_floor_raises"] == 0
        assert settings["terminal_exhaustion_may_shorten"] is True
        assert (
            end.speculative_receipt["draft_settings"]["proposal_distribution"]
            == "deterministic_point_mass"
        )
    assert engine.scheduler_stats["external_rounds"] > 0
    assert engine.scheduler_stats["target_max_width"] >= 2
    assert draft.stats["jacobi_passes"] == draft.stats["backbone_blocks"] * 3


def test_sampled_target_uses_actual_point_mass_q_and_target_law(monkeypatch):
    import mlx2.runtime.external_speculative as external

    model, draft = tiny()
    engine = generator(model, draft)
    prompt = [1, 2, 3]
    target_logits = np.asarray(
        model(mx.array([prompt]), cache=model.make_cache())[0, -1]
    )
    expected = softmax(target_logits, 0.8)
    captured = []

    def checked(tokens, proposals, targets, rng, **kwargs):
        if tokens:
            for token, law in zip(tokens, proposals):
                assert law[token] == 1 and np.count_nonzero(law) == 1
            captured.append(
                (list(tokens), copy.deepcopy(proposals), copy.deepcopy(targets))
            )
        return verify_proposals(tokens, proposals, targets, rng, **kwargs)

    monkeypatch.setattr(external, "verify_proposals", checked)
    engine.insert(
        [prompt],
        max_tokens=[5],
        lane_rngs=[LaneRNG(22)],
        sampling_configs=[{"sampling_temp": 0.8}],
    )
    outputs, _ = drain(engine)
    assert len(outputs[0]) == 5 and captured
    tokens, proposals, targets = captured[0]
    np.testing.assert_allclose(targets[0], expected, rtol=1e-6, atol=1e-7)
    counts = np.zeros(9)
    rng = RequestRNG(82)
    for _ in range(10000):
        result = verify_proposals(tokens, proposals, targets, rng)
        counts[result.emitted[0]] += 1
    np.testing.assert_allclose(counts / counts.sum(), expected, atol=0.016)


def test_real_xpress_target_failure_restores_both_planes_rng_and_processors(
    monkeypatch,
):
    model, draft = tiny()
    engine = generator(model, draft)
    engine.insert(
        [[1, 2, 3]],
        max_tokens=[5],
        lane_rngs=[LaneRNG(19)],
        sampling_configs=[{"sampling_temp": 0.7}],
    )
    lane = engine.lanes[0]
    engine._prefill(lane)
    before = copy.deepcopy(lane.__dict__)
    original = model.forward_with_taps

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("after target cache append")

    monkeypatch.setattr(model, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="cache append"):
        engine._round([lane])
    assert not lane.ready and lane.history == before["history"]
    assert lane.rng.snapshot() == before["rng"].snapshot()
    for field in ("cache", "draft_cache"):
        for now, old in zip(getattr(lane, field), before[field]):
            assert now.offset == old.offset
            for actual, expected in zip(now.state, old.state):
                if actual is not None:
                    np.testing.assert_array_equal(
                        np.asarray(actual)[..., : now.offset, :],
                        np.asarray(expected)[..., : old.offset, :],
                    )
    np.testing.assert_array_equal(np.asarray(lane.tail), np.asarray(before["tail"]))
    monkeypatch.setattr(model, "forward_with_taps", original)
    output, finish = drain(engine)
    assert len(output[0]) == 5
    finish[0].cache_sidecar.validate("tiny-xpress-v1", len(finish[0].all_tokens))


def test_real_xpress_apcv2_cow_paired_resume_and_revision_failure(
    tmp_path, monkeypatch
):
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    model, draft = tiny()
    engine = generator(model, draft)
    engine.insert([[1, 2, 3]], max_tokens=[4])
    _, finished = drain(engine)
    end = finished[0]
    apc = APCv2(max_size=2, layout_name="tiny-standard-xpress")
    key = APCKey(
        "qwen3+xpress",
        revision="bound-revision",
        cache_layout_fingerprint="tiny-standard-xpress",
    )
    apc.store(key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar)
    hit = apc.lookup(key, end.all_tokens + [end.token])
    assert hit.hit and hit.hit_kind == "external_draft_sidecar"
    hit.sidecar.validate("tiny-xpress-v1", len(end.all_tokens))
    with pytest.raises(ValueError, match="revision"):
        hit.sidecar.validate("different-xpress", len(end.all_tokens))
    assert not apc.lookup(
        APCKey(
            "qwen3+xpress",
            revision="other",
            cache_layout_fingerprint="tiny-standard-xpress",
        ),
        end.all_tokens + [end.token],
    ).hit
    resumed = generator(model, draft)
    resumed.insert(
        [[end.token]],
        max_tokens=[4],
        caches=[hit.cache],
        all_tokens=[end.all_tokens],
        cache_states=[hit.sidecar],
    )
    actual, final = drain(resumed)
    assert actual[0] == reference(model, end.all_tokens + [end.token], 4)
    assert resumed.scheduler_stats["paired_cache_resumes"] == 1
    final[0].cache_sidecar.validate("tiny-xpress-v1", len(final[0].all_tokens))
    hit.cache.close()
    assert apc.apc_stats["cow"]["active_leases"] == 0
    apc.clear(release_memory=False)
