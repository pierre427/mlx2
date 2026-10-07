"""Shared-prefix APCv2 reuse on the speculative routes (2026-10-06 sweep).

GPU evidence: on Qwen3.8 27B a request sharing an 8K document prefix with the
previous one got no APCv2 reuse on ``--prompt-lookup`` and ``--external-draft``
while ``--ordinary`` and ``--native-mtp`` resumed from it.  A hybrid (GDN)
boundary entry can only be trimmed back to a recorded state checkpoint, and
the ordinary prefill loop records them at chunk boundaries; the prompt-lookup
prefill loop never did.
"""

import pytest

from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

DOC = [(7 * i + 3) % 120 + 1 for i in range(70)]
FIRST = DOC + [5, 6, 7, 8, 9]
SECOND = DOC + [11, 12, 13]


def _request(tokens):
    return {"tokens": list(tokens), "max_tokens": 3, "temperature": 0}


@pytest.mark.parametrize("prompt_lookup", [False, True])
def test_shared_prefix_hit_on_hybrid_route(monkeypatch, prompt_lookup):
    patch_host(monkeypatch)
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "16")
    model, vocab = tiny_qwen38_mtp()
    cold_engine = make_engine(model, vocab, mtp=False, prompt_lookup=prompt_lookup)
    try:
        cold = run(cold_engine, _request(SECOND))
    finally:
        cold_engine.close()
    engine = make_engine(model, vocab, mtp=False, prompt_lookup=prompt_lookup)
    try:
        run(engine, _request(FIRST))
        warm = run(engine, _request(SECOND))
    finally:
        engine.close()
    assert "error" not in warm, warm
    # The deepest recorded checkpoint at or below the 70-token shared prefix.
    assert warm["receipt"]["cached_tokens"] == 64
    assert warm["tokens"] == cold["tokens"]


def _kv_target_external_pair(vocab=128):
    # A sliding window wider than every prompt keeps the target trimmable, so
    # APCv2 can land a target-only hit inside the stored prompt boundary.
    import mlx.core as mx

    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config
    from mlx2.runtime.models.muse_glimmer import Model

    mx.random.seed(8)
    model = Model(ModelArgs(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=vocab, sliding_window=512, max_position_embeddings=512,
    ))
    draft = DFlash2DraftModel(DFlash2Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=vocab, num_target_layers=4, target_layer_ids=[0, 3],
        conv_kernel_size=2, conv_group_size=2, selector_rank=4,
        selector_top_k=4, block_size=4, mask_token_id=vocab - 1,
        max_position_embeddings=512, sliding_window=8,
        layer_types=["sliding_attention"] * 2,
    )).bind(model)
    mx.eval(model.parameters(), draft.parameters())
    return model, draft, vocab


def test_external_draft_receipt_does_not_claim_a_target_only_prefix(monkeypatch):
    # The external generator re-prefills the whole transcript when a hit has
    # no paired draft state; the receipt claimed the 70 shared tokens anyway.
    from mlx2.runtime import external_speculative
    from route_harness import make_external_engine

    patch_host(monkeypatch)
    starts = []
    original = external_speculative.ExternalDraftBatchGenerator._prefill

    def prefill(self, lane, *, step=None):
        starts.append(len(lane.history))
        return original(self, lane, step=step)

    monkeypatch.setattr(
        external_speculative.ExternalDraftBatchGenerator, "_prefill", prefill
    )
    model, draft, vocab = _kv_target_external_pair()
    engine = make_external_engine(model, draft, vocab)
    try:
        run(engine, _request(FIRST))
        del starts[:]
        warm = run(engine, _request(SECOND))
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert "error" not in warm, warm
    assert starts[0] == 0  # the whole prompt was prefilled again
    assert warm["receipt"]["cached_tokens"] == 0
    assert counts["external_draft_sidecar_missing_misses"] == 1
