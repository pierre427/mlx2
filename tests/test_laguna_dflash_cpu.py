"""CPU contract tests for the causal Laguna DFlash drafter; tiny random weights."""
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.drafters.laguna_dflash import (
    LagunaDFlashConfig,
    LagunaDFlashDraftModel,
    expected_weight_shapes,
)
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.laguna import Model, ModelArgs
from mlx2.runtime.speculative_sampling import RequestRNG

DFLASH_ROOT = Path(
    "~/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-DFlash/snapshots"
)


def _snapshot():
    found = sorted(DFLASH_ROOT.glob("*/config.json")) if DFLASH_ROOT.exists() else []
    return found[0].parent if found else None


def tiny(seed=5):
    mx.random.seed(seed)
    target = Model(ModelArgs(
        vocab_size=48, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, sliding_window=4,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, max_position_embeddings=256,
    ))
    draft = LagunaDFlashDraftModel(LagunaDFlashConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, vocab_size=48, sliding_window=4, block_size=6,
        mask_token_id=47, target_layer_ids=[1, 3], num_target_layers=4,
    ))
    mx.eval(target.parameters(), draft.parameters())
    return target, draft.bind(target)


def drain(batch):
    output, final = {}, {}
    for _ in range(200):
        _, responses = batch.next()
        for response in responses:
            output.setdefault(response.uid, []).append(response.token)
            if response.finish_reason:
                final[response.uid] = response
        if not batch.lanes:
            return output, final
    raise AssertionError("scheduler stalled")


def greedy_reference(target, prompt, count):
    cache = target.make_cache()
    tokens, out = list(prompt), []
    for step in range(count):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        out.append(token)
    return out


def test_post_block_taps_match_ordinary_forward():
    target, _ = tiny()
    x = mx.array([[1, 2, 3, 4, 5]])
    logits, features = target.forward_with_taps(x, target.make_cache(), [1, 3])
    np.testing.assert_allclose(
        np.asarray(logits), np.asarray(target(x, cache=target.make_cache())), atol=1e-5
    )
    assert features.shape == (1, 5, 32)
    with pytest.raises(ValueError):
        target.forward_with_taps(x, target.make_cache(), [4])


def test_block_laws_are_exact_and_only_context_is_committed():
    target, draft = tiny()
    features = target.prefill_body(mx.array([[1, 2, 3, 4, 5, 6]]), target.make_cache(), [1, 3])
    cache = draft.make_cache()
    tokens, laws = draft.draft_distributions([7], features, cache, 4, [RequestRNG(2)], [0.7])
    assert len(tokens[0]) == 4
    for token, law in zip(tokens[0], laws[0]):
        assert law[token] > 0 and abs(law.sum() - 1) < 1e-9
    assert all(entry.offset == 6 and entry.keys.shape[2] == 3 for entry in cache)
    assert draft.stats["block_drafts"] == 1


def test_block_is_causal_later_positions_do_not_change_earlier_logits():
    target, draft = tiny()
    features = target.prefill_body(mx.array([[1, 2, 3]]), target.make_cache(), [1, 3])
    short, long = draft.make_cache(), draft.make_cache()
    draft.append_context(features, short)
    draft.append_context(features, long)
    a = draft._block_logits([4], [short], 2)
    b = draft._block_logits([4], [long], 4)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b[:, :2]), atol=1e-5)


def test_external_laguna_greedy_matches_ordinary_and_pairs_sidecar():
    target, draft = tiny()
    batch = ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="test", num_draft=4, prefill_step_size=3
    )
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 2]]
    ids = batch.insert(prompts, max_tokens=[10, 10], sampling_configs=[{"sampling_temp": 0}] * 2)
    got, final = drain(batch)
    for uid, prompt in zip(ids, prompts):
        assert got[uid] == greedy_reference(target, prompt, 10)
        final[uid].cache_sidecar.validate("test", len(final[uid].all_tokens))
        assert final[uid].speculative_receipt["kind"] == "external_laguna_dflash"
    stats = batch.scheduler_stats
    assert stats["external_rounds"] > 0 and stats["proposed_tokens"] > 0
    assert stats["external_context_token_pairings"] == 0  # block drafter: no pairing
    assert draft.stats["block_drafts"] > 0


def _header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))


def test_real_laguna_dflash_config_and_header_schema():
    snapshot = _snapshot()
    if snapshot is None:
        pytest.skip("poolside Laguna DFlash artifact not in HF cache")
    raw = json.loads((snapshot / "config.json").read_text())
    config = LagunaDFlashConfig.from_hf(raw)
    assert config.target_layer_ids == [1, 13, 25, 33, 39]
    assert (config.block_size, config.mask_token_id, config.sliding_window) == (16, 12, 512)
    header = _header(snapshot / "model.safetensors")
    observed = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    assert observed == expected_weight_shapes(config)
    from mlx2.adapters.laguna_dflash import inspect_drafter

    record = inspect_drafter(snapshot, target_config={
        "hidden_size": 2048, "vocab_size": 100352, "num_hidden_layers": 40})
    assert record["args"].num_hidden_layers == 5
    with pytest.raises(ValueError):
        LagunaDFlashConfig.from_hf({**raw, "dflash_config": {**raw["dflash_config"], "causal": False}})
    with pytest.raises(ValueError):
        LagunaDFlashConfig.from_hf({**raw, "eagle_aux_hidden_state_layer_ids": [1, 13, 25, 33, 39]})


def test_laguna_adapter_external_policy_fails_closed_and_keeps_ordinary_config(monkeypatch):
    import mlx2.adapters.laguna_xs21 as laguna

    with pytest.raises(ValueError, match="no qualified overrides"):
        laguna.LagunaXS21Adapter("/missing", execution_policy={"num_draft": 3})
    adapter = object.__new__(laguna.LagunaXS21Adapter)
    assert "backend" not in adapter.execution_config(max_lanes=2, prefill_step=64)
    adapter.draft_model, adapter.external_policy = object(), {"draft_model": "x"}
    config = adapter.execution_config(max_lanes=2, prefill_step=64)
    assert config["backend"] == "external_draft" and config["num_draft"] == 7
    assert laguna.LagunaXS21Adapter.external_profile_name(False) == "laguna-xs21-apcv2-laguna-dflash"
    snapshot = _snapshot()
    if snapshot is None:
        return
    target = {"hidden_size": 2048, "vocab_size": 100352, "num_hidden_layers": 40}
    monkeypatch.setattr(laguna, "inspect_artifact", lambda path: {"config": target})
    for count in (0, 16):
        with pytest.raises(ValueError, match="num_draft"):
            laguna.LagunaXS21Adapter(
                "/unused", execution_policy={"draft_model": str(snapshot), "num_draft": count}
            )


def test_laguna_forced_acceptance_commits_multi_token_context_exactly():
    from collections import Counter

    target, draft = tiny()
    prompt = [1, 2, 3, 4, 5, 6]
    reference = greedy_reference(target, prompt, 16)
    favourite = Counter(reference).most_common(1)[0][0]
    head = draft.lm_head
    draft.lm_head = lambda hidden: head(hidden) + mx.where(mx.arange(48) == favourite, 1e4, 0.0)
    batch = ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="test", num_draft=4, prefill_step_size=3
    )
    uid = batch.insert([prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0}])[0]
    got, final = drain(batch)
    assert got[uid] == reference
    assert batch.scheduler_stats["accepted_proposals"] > 0
    final[uid].cache_sidecar.validate("test", len(final[uid].all_tokens))
