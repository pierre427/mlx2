"""Opt-in early verify build behind the drafter: output and law unchanged.

``MLX2_EXTERNAL_VERIFY_PREBUILD=1`` lets a B1 greedy LiLiCorr chain round
queue its verify while the drafter runs; the scheduler adopts it only when
the host proposal equals it.  Emitted tokens must equal ordinary greedy
decode and the prebuild-off run, with and without proposal composition, on
the exact Nemotron-H chain verify.
"""

import copy
from contextlib import closing

import mlx.core as mx
import pytest

from mlx2.adapters.lilicorr import LiLiCorrConfig
from mlx2.runtime.drafters.lilicorr import LiLiCorrDraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from tests.test_nemotron_tree_cpu import CAPTURES, _model, _prefilled


def _drafter(model):
    mx.random.seed(3)
    config = LiLiCorrConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=16, mask_token_id=15, num_target_layers=5,
        target_layer_ids=list(CAPTURES), block_size=6,
        layer_types=["full_attention"], lilicorr_hidden_size=6,
        lilicorr_candidate_topk=4, lilicorr_num_layers=1, lilicorr_num_heads=2,
        lilicorr_mlp_ratio=1.5, lilicorr_factor_dim=3,
    )
    return LiLiCorrDraftModel(config).bind(model)


def _serve(model, draft, prompt, tokens):
    with closing(ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="prebuild-cpu", num_draft=5,
        completion_batch_size=1, prefill_step_size=8, cost_per_commit_handoff=False,
    )) as batch:
        batch.insert([prompt], max_tokens=[tokens],
                     sampling_configs=[{"sampling_temp": 0.0, "emit_logprobs": False}])
        emitted = []
        for _ in range(200):
            _, responses = batch.next()
            emitted.extend(response.token for response in responses)
            if not batch.lanes:
                break
        return emitted, dict(batch.scheduler_stats)


@pytest.mark.parametrize("composed", [False, True])
def test_prebuilt_verify_serves_ordinary_greedy_tokens(monkeypatch, composed):
    mx.set_default_device(mx.cpu)
    model = _model()
    prompt = [1, 2, 3, 4, 5, 6, 1, 2, 3]
    cache = _prefilled(model, prompt[:-1])
    expected, token = [], prompt[-1]
    for _ in range(20):
        token = int(mx.argmax(model(mx.array([[token]]), cache)[0, -1]).item())
        expected.append(token)

    def drafter():
        draft = _drafter(model)
        if composed:
            from mlx2.runtime.proposal_composition import ComposedDraftModel

            from mlx2.adapters.external_draft_policy import (
                DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION,
            )

            draft = ComposedDraftModel(
                draft,
                dict(DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION),
            )
        return draft

    monkeypatch.delenv("MLX2_EXTERNAL_VERIFY_PREBUILD", raising=False)
    off, off_stats = _serve(model, drafter(), prompt, 20)
    monkeypatch.setenv("MLX2_EXTERNAL_VERIFY_PREBUILD", "1")
    on, on_stats = _serve(model, drafter(), prompt, 20)
    assert off == expected and on == expected
    assert off_stats.get("external_verify_prebuild_built", 0) == 0
    assert on_stats.get("external_verify_prebuild_built", 0) > 0
    assert on_stats.get("external_verify_prebuild_adopted", 0) > 0
    assert (on_stats.get("external_verify_prebuild_adopted", 0)
            + on_stats.get("external_verify_prebuild_discarded", 0)
            == on_stats["external_verify_prebuild_built"])
    for key in ("segmented_transactions", "external_rounds"):
        if key in off_stats:
            assert on_stats.get(key) == off_stats[key]
