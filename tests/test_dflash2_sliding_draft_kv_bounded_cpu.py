"""Sliding-window DFlash2 drafters keep their draft KV at the window.

vLLM's Q4 2026 list asks for KV memory efficiency across draft/target
attention types (an SWA drafter beside a full-attention verifier).  mlx2's
served DFlash2 drafters (Qwen3.8-27B, Flash-Next, Qwen3.6-35B) are all
``sliding_attention`` with window 2048; their draft planes are rotating
caches.  This checks, through the real external-draft generator, that the
draft KV never grows with the context: every lane's draft plane stays at
``sliding_window - 1`` positions after each scheduler step, at B1 and B2,
over a prompt many times the window.  (The opt-in FOLD far-memory policy
keeps full keys by design and is not covered here.)
"""
import mlx.core as mx
import pytest

from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.drafters.dflash2_config import DFlash2Config
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.cache import RotatingKVCache
from mlx2.runtime.models.muse_glimmer import Model

WINDOW = 4


def _tiny():
    mx.random.seed(8)
    target = Model(ModelArgs(hidden_size=8, intermediate_size=16, num_hidden_layers=4,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                             vocab_size=32, sliding_window=3, max_position_embeddings=256))
    draft = DFlash2DraftModel(DFlash2Config(
        hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, vocab_size=32, num_target_layers=4,
        target_layer_ids=[0, 3], conv_kernel_size=2, conv_group_size=2, selector_rank=4,
        selector_top_k=4, block_size=4, mask_token_id=31, max_position_embeddings=256,
        sliding_window=WINDOW, layer_types=["sliding_attention"] * 2)).bind(target)
    return target, draft


def _planes(lane):
    for entry in lane.draft_cache:
        yield from (entry.rows if hasattr(entry, "rows") else [entry])


@pytest.mark.parametrize("lanes", [1, 2])
def test_sliding_draft_kv_stays_at_window_over_long_context(lanes):
    target, draft = _tiny()
    assert all(isinstance(c, RotatingKVCache) and c.max_size == WINDOW - 1
               for c in draft.make_cache())
    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="test",
                                        num_draft=2, prefill_step_size=7)
    prompts = [[(3 * i + 5 * j) % 29 + 1 for i in range(60)] for j in range(lanes)]
    try:
        batch.insert(prompts, max_tokens=[24] * lanes)
        widest, offsets, steps = 0, 0, 0
        while batch.lanes and steps < 200:
            batch.next()
            steps += 1
            for lane in batch.lanes.values():
                for plane in _planes(lane):
                    if plane.keys is not None:
                        widest = max(widest, int(plane.keys.shape[2]))
                        offsets = max(offsets, int(plane.offset))
        assert not batch.lanes, "scheduler stalled"
    finally:
        batch.close()
    # The context reached far past the window, yet no draft plane held more.
    assert offsets > 10 * WINDOW
    assert widest <= WINDOW - 1
