"""CPU contract tests for Muse Glimmer's original DFlash drafter (assistant).

Tiny random target/drafter weights; the real BF16/Q8/Q4 assistant artifacts
are inspected from headers only when present.
"""
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.drafters.muse_dflash import MuseDFlashConfig, MuseDFlashDraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.muse_glimmer import Model
from mlx2.runtime.speculative_sampling import RequestRNG

MODELS = Path.home() / "mlx-models"
VOCAB = 96


def tiny(seed=11, window=8):
    mx.random.seed(seed)
    target = Model(ModelArgs(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=VOCAB, sliding_window=8, max_position_embeddings=512,
    ))
    draft = MuseDFlashDraftModel(MuseDFlashConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        vocab_size=VOCAB, num_target_layers=4, target_layer_ids=[1, 3],
        mask_token_id=VOCAB - 1, block_size=6, sliding_window=window,
    ))
    mx.eval(target.parameters(), draft.parameters())
    return target, draft.bind(target)


def greedy_reference(target, prompt, count):
    cache = target.make_cache()
    tokens, out = list(prompt), []
    for step in range(count):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        out.append(token)
    return out


def drain(batch):
    output, final = {}, {}
    for _ in range(300):
        _, responses = batch.next()
        for response in responses:
            output.setdefault(response.uid, []).append(response.token)
            if response.finish_reason:
                final[response.uid] = response
        if not batch.lanes:
            return output, final
    raise AssertionError("scheduler stalled")


def test_laws_are_exact_and_only_committed_context_is_stored():
    target, draft = tiny()
    features = target.prefill_body(mx.array([[1, 2, 3, 4, 5, 6]]), target.make_cache(), [1, 3])
    cache = draft.make_cache()
    tokens, laws = draft.draft_distributions([7], features, cache, 4, [RequestRNG(3)], [0.8])
    assert len(tokens[0]) == 4
    for token, law in zip(tokens[0], laws[0]):
        assert law[token] > 0 and abs(law.sum() - 1) < 1e-9
    # Window 8 keeps the last 7 committed positions; proposals are never stored.
    assert all(entry.offset == 6 and entry.keys.shape[2] == 6 for entry in cache)


def test_block_is_causal_and_honours_the_sliding_window():
    target, draft = tiny(window=4)
    features = target.prefill_body(mx.array([[1, 2, 3, 4, 5, 6, 7]]), target.make_cache(), [1, 3])
    short, long = draft.make_cache(), draft.make_cache()
    draft.append_context(features, short)
    draft.append_context(features, long)
    a = draft._block_logits([4], [short], 2)
    b = draft._block_logits([4], [long], 4)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b[:, :2]), atol=1e-5)
    # Resident ring holds window - 1 keys; later block rows see fewer of them.
    assert short[0].keys.shape[2] == 3


def test_external_assistant_greedy_matches_ordinary():
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
        assert final[uid].speculative_receipt["kind"] == "external_muse_dflash_assistant"
    assert batch.scheduler_stats["proposed_tokens"] > 0


def test_forced_acceptance_commits_multi_token_context_exactly():
    target, draft = tiny()
    prompt = [1, 2, 3, 4, 5, 6]
    reference = greedy_reference(target, prompt, 16)
    favourite = Counter(reference).most_common(1)[0][0]
    logits = draft._logits
    object.__setattr__(
        draft, "_logits",
        lambda hidden: logits(hidden) + mx.where(mx.arange(VOCAB) == favourite, 1e4, 0.0),
    )
    batch = ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="test", num_draft=4, prefill_step_size=3
    )
    uid = batch.insert([prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0}])[0]
    got, final = drain(batch)
    assert got[uid] == reference
    assert batch.scheduler_stats["accepted_proposals"] > 0
    final[uid].cache_sidecar.validate("test", len(final[uid].all_tokens))


def test_quantized_drafter_loads_through_sanitized_encoder_names(tmp_path):
    from mlx2.adapters import muse_dflash

    _, draft = tiny()
    weights = dict(nn.utils.tree_flatten(draft.parameters()))
    renamed = {
        ("encoder.fc." + k[3:] if k.startswith("fc.") else
         "encoder.output_norm_enc." + k[12:] if k.startswith("hidden_norm.") else k): v
        for k, v in weights.items()
    }
    assert draft.sanitize(renamed).keys() == weights.keys()
    with pytest.raises(ValueError, match="encoder"):
        draft.sanitize({"encoder.extra.weight": mx.zeros((1,))})
    assert muse_dflash.is_assistant_artifact(tmp_path) is False


@pytest.mark.parametrize("name", [
    "Muse-Glimmer-30B-assistant", "Muse-Glimmer-30B-assistant-q8", "Muse-Glimmer-30B-assistant-q4",
])
def test_real_assistant_artifacts_inspect_from_headers(name):
    from mlx2.adapters.muse_dflash import inspect_drafter, is_assistant_artifact

    draft, target = MODELS / name, MODELS / "Muse-Glimmer-30B-mlx-8bit"
    if not (draft / "config.json").is_file() or not (target / "config.json").is_file():
        pytest.skip("local Muse artifacts absent")
    assert is_assistant_artifact(draft)
    record = inspect_drafter(draft, target)
    args = record["args"]
    assert args.target_layer_ids == [1, 13, 25, 37, 49]
    assert (args.block_size, args.mask_token_id, args.sliding_window) == (16, 201818, 2048)
    assert args.vocab_size == 202048 and args.num_target_layers == 52
    assert (args.quantization or {}).get("bits") == {"": None, "-q8": 8, "-q4": 4}[name[len("Muse-Glimmer-30B-assistant"):]]
    assert len(record["fingerprint"]) == 64


def test_muse_adapter_refuses_assistant_with_dflash2_only_options(tmp_path):
    import json

    import mlx2.adapters.muse_glimmer as muse

    draft = tmp_path / "assistant"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({"architectures": ["MuseGlimmerAssistantModel"]}))
    adapter = object.__new__(muse.MuseGlimmerAdapter)
    for policy in (
        {"draft_model": str(draft), "target_verify_row_exact": True},
        {"draft_model": str(draft), "pairwise_selection": "batched"},
    ):
        with pytest.raises(ValueError, match="linear host route"):
            adapter._init_muse("/unused", execution_policy=policy)
