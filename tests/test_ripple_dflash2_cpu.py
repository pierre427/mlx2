"""Ripple target + DFlash2 external draft: exact greedy parity on CPU.

A tiny FP32 Ripple target (four-token window, so verify and rollback cross
the rotating-cache wrap) with a tiny DFlash2 draft that, like the vendor's
Ripple drafter, mixes sliding layers with a NoPE global layer. External
speculation must emit exactly the ordinary greedy tokens.
"""
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.drafters.dflash2_config import DFlash2Config
from mlx2.runtime.drafters.dflash_base import NO_ROPE
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from test_external_dflash2_cpu import drain
from test_ripple_pld_cpu import greedy, tiny


def draft_args(**overrides):
    values = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=8, vocab_size=64, num_target_layers=5,
                  target_layer_ids=[1, 3], conv_kernel_size=2, conv_group_size=4, selector_rank=4,
                  selector_top_k=4, block_size=4, mask_token_id=63, max_position_embeddings=256,
                  sliding_window=6, layer_types=["sliding_attention", "full_attention"],
                  no_rope_layers=[1])
    values.update(overrides)
    return DFlash2Config(**values)


def pair():
    model = tiny()
    mx.random.seed(11)
    draft = DFlash2DraftModel(draft_args()).bind(model)
    mx.eval(draft.parameters())
    return model, draft


def test_no_rope_layers_parse_validate_and_reach_attention():
    config = DFlash2Config.from_dict({
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 64, "num_target_layers": 5,
        "max_position_embeddings": 256, "sliding_window": 6,
        "layer_types": ["sliding_attention", "full_attention"], "no_rope_layers": [1],
        "dflash_config": {"block_size": 4, "mask_token_id": 63, "target_layer_ids": [1, 3],
                          "conv_kernel_size": 2, "conv_group_size": 4, "selector_rank": 4,
                          "selector_top_k": 4}})
    assert config.no_rope_layers == [1]
    draft = DFlash2DraftModel(config)
    assert [layer.self_attn.use_rope for layer in draft.layers] == [True, False]
    with pytest.raises(ValueError, match="no_rope_layers"):
        draft_args(no_rope_layers=[2])
    with pytest.raises(ValueError, match="no_rope_layers"):
        draft_args(no_rope_layers=[1, 1])
    x = mx.ones((1, 2, 3, 8))
    assert NO_ROPE(x, offset=5) is x


def test_target_taps_are_post_block_pre_norm_and_concatenated():
    model = tiny()
    tokens = mx.array([[3, 7, 11, 5, 9, 2, 8]])
    logits, features = model.forward_with_taps(tokens, model.make_cache(), (1, 3))
    reference = model(tokens, cache=model.make_cache())
    assert mx.array_equal(logits, reference)
    taps = dict.fromkeys((1, 3))
    model.model(tokens, model.make_cache(), capture=taps)
    assert mx.array_equal(features, mx.concatenate([taps[1], taps[3]], axis=-1))
    assert features.shape == (1, 7, 2 * 32)
    body = model.prefill_body(tokens, model.make_cache(), (1, 3))
    assert mx.array_equal(body, features)
    with pytest.raises(ValueError, match="capture layers"):
        model.forward_with_taps(tokens, model.make_cache(), (3, 1))


@pytest.mark.parametrize("num_draft", [2, 3])
def test_external_dflash2_greedy_matches_ordinary_across_window_wrap(num_draft):
    model, draft = pair()
    prompts = [[3, 7, 11, 5, 9, 2, 8, 1, 4], [5, 1, 6]]
    count = 14
    generator = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test",
                                            num_draft=num_draft, prefill_step_size=4)
    ids = generator.insert(prompts, max_tokens=[count] * 2, sampling_configs=[{"sampling_temp": 0}] * 2)
    got, final = drain(generator)
    for uid, prompt in zip(ids, prompts, strict=True):
        assert got[uid] == greedy(model, prompt, count)
        final[uid].cache_sidecar.validate("test", len(final[uid].all_tokens))
    stats = generator.scheduler_stats
    assert stats["external_rounds"] > 0 and stats["proposed_tokens"] > 0


def test_draft_uses_unscaled_target_embeddings_like_the_vendor():
    # Vendor SGLang shares get_input_embeddings() (the raw table) with the
    # DFlash worker; the target's own embed_scale applies only to the target.
    model, draft = pair()
    assert model.args.embed_scale != 1.0
    assert draft.embed_scale == 1.0
    assert draft.embed_tokens is model.model.embed_tokens
    assert draft.lm_head is model.lm_head


def test_policy_split_routes_numerics_options_and_external_keys():
    from mlx2.adapters.ripple import RippleAdapter, split_policy

    numerics, options, external = split_policy({
        "activation_precision": "bf16", "fused_elementwise": True,
        "draft_model": "/models/draft", "num_draft": 5})
    assert numerics == {"activation_precision": "bf16"}
    assert options == {"fused_elementwise": True}
    assert external == {"draft_model": "/models/draft", "num_draft": 5}
    adapter = object.__new__(RippleAdapter)
    assert adapter._parse_external_policy({}, family="Ripple") is False
    with pytest.raises(ValueError, match="external draft_model"):
        adapter._parse_external_policy({"num_draft": 5}, family="Ripple")
    with pytest.raises(ValueError, match="external draft_model"):
        adapter._parse_external_policy({"draft_model": "x", "speed": 1}, family="Ripple")


def test_pairing_accepts_only_dflash2_and_pins_its_content_revision(monkeypatch):
    from mlx2.adapters import dflash2, drafter_pairing

    monkeypatch.setattr(dflash2, "inspect_drafter", lambda d, t: {"files": [], "weight_sha256": []})
    monkeypatch.setattr(dflash2, "content_revision", lambda record: "a" * 64)
    target = {"adapter": "RippleAdapter", "path": "/target"}
    pairing = drafter_pairing.pair_drafter(
        target, {"path": "/draft", "architectures": ["DFlash2DraftModel"], "name": "Ripple-DFlash2"})
    assert pairing["execution_policy"] == {"draft_model": "/draft", "draft_revision": "a" * 64, "num_draft": 7}
    assert pairing["draft"] == "Ripple-DFlash2@sha256:" + "a" * 64
    with pytest.raises(ValueError, match="DFlash2DraftModel only"):
        drafter_pairing.pair_drafter(target, {"path": "/d", "architectures": ["Eagle3Speculator"], "name": "e"})


def test_adapter_admits_the_draft_revision_pin_only_with_a_draft():
    from mlx2.adapters.ripple import RippleAdapter

    adapter = object.__new__(RippleAdapter)
    assert adapter._parse_external_policy({"draft_model": "/d", "draft_revision": "a" * 64}, family="Ripple")
    with pytest.raises(ValueError, match="external draft_model"):
        adapter._parse_external_policy({"draft_revision": "a" * 64}, family="Ripple")


def test_best_route_table_selects_dflash2_for_the_measured_oq4e_checkpoint_only():
    from mlx2.best_route_defaults import find_entry, load_table

    table = load_table()
    oq4e = "8ddd93fab94b25551a5a9533efdcb0f9edfbb198bfcde2888e0515933208f7da"
    entry = find_entry(table, "RippleAdapter", oq4e)
    assert entry["status"] == "selected_unqualified"
    assert entry["default"]["kind"] == "ast" and entry["default"]["execution_policy"] == {"num_draft": 7}
    assert entry["default"]["drafter"]["architectures"] == ["DFlash2DraftModel"]
    assert "Operator decision" in entry["reason"]
    assert find_entry(table, "RippleAdapter", "0" * 64) is None
