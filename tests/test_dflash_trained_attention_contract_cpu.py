"""DFlash drafters must be served with the attention layout they were trained with.

vLLM #61537: speculators-format DFlash checkpoints had their full-attention
draft layers served causally although they were trained bidirectionally
(acceptance length -18%).  oMLX #4508/#4525 and TensorRT-LLM #19988 found the
same family of silent contract mismatches: block layout (``sample_from_anchor``),
partial RoPE, and trained mask embeddings.

mlx2 serves two block drafters:

* DFlash2 (Qwen3.8 / Qwen3.6 / Muse Glimmer checkpoints, ``is_causal: false``,
  Splash reference "non-causal block"): every block position attends to the
  whole retained context and the whole block.
* Laguna DFlash (``dflash_config.causal: true``): the block is causal
  (``tests/test_laguna_dflash_cpu.py`` locks that in).

The first tests lock the DFlash2 mask against an explicit reference mask and
show the check has teeth (a causal mask fails it).  The remaining tests make
the config parsers refuse checkpoint metadata that declares a different
contract from the one mlx2 implements, instead of serving it silently with
the wrong one.
"""

import copy
import glob
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.drafters.dflash2_config import DFlash2Config
from mlx2.runtime.drafters.dflash_base import DFlashAttention
from mlx2.runtime.drafters.laguna_dflash import LagunaDFlashConfig
from mlx2.runtime.models.cache import KVCache

QWEN38_DFLASH2 = Path.home() / "mlx-models/Qwen3.8-27B-DFlash2/config.json"
LAGUNA_GLOB = str(
    Path.home() / ".cache/huggingface/hub"
    / "models--poolside--Laguna-XS-2.1-DFlash/snapshots/*/config.json"
)


def _tiny_dflash2_config(**overrides):
    values = dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=64, num_target_layers=4, target_layer_ids=[0, 3],
        conv_kernel_size=2, conv_group_size=2, selector_rank=4,
        selector_top_k=4, block_size=4, mask_token_id=63,
        max_position_embeddings=256, sliding_window=8,
        layer_types=["sliding_attention"],
    )
    values.update(overrides)
    return DFlash2Config(**values)


def _reference_attention(attention, x, x_ctx, rope, mask):
    """Explicit-mask attention over [context; block] with the module's weights."""
    B, L, _ = x.shape
    S = x_ctx.shape[1]

    def heads(t, n, norm=None):
        t = t.reshape(B, t.shape[1], n, -1)
        if norm is not None:
            t = norm(t)
        return t.transpose(0, 2, 1, 3)

    q = heads(attention.q_proj(x), attention.n_heads, attention.q_norm)
    ck = heads(attention.k_proj(x_ctx), attention.n_kv_heads, attention.k_norm)
    pk = heads(attention.k_proj(x), attention.n_kv_heads, attention.k_norm)
    cv = heads(attention.v_proj(x_ctx), attention.n_kv_heads)
    pv = heads(attention.v_proj(x), attention.n_kv_heads)
    q = rope(q, offset=S)
    ck = rope(ck, offset=0)
    pk = rope(pk, offset=S)
    keys = mx.concatenate([ck, pk], axis=2)
    values = mx.concatenate([cv, pv], axis=2)
    repeat = attention.n_heads // attention.n_kv_heads
    keys = mx.repeat(keys, repeat, axis=1)
    values = mx.repeat(values, repeat, axis=1)
    scores = (q @ keys.transpose(0, 1, 3, 2)) * attention.scale
    scores = mx.where(mask, scores, -mx.inf)
    out = mx.softmax(scores.astype(mx.float32), axis=-1) @ values
    return attention.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


def _attention_case():
    from mlx2.runtime.drafters.dflash_base import _build_rope

    config = _tiny_dflash2_config()
    mx.random.seed(3)
    attention = DFlashAttention(config, 0)
    mx.eval(attention.parameters())
    x = mx.random.normal((1, 4, config.hidden_size))
    x_ctx = mx.random.normal((1, 5, config.hidden_size))
    rope = _build_rope(config)
    return attention, x, x_ctx, rope


def _masks(context, block):
    full = mx.ones((block, context + block), dtype=mx.bool_)
    causal = mx.concatenate(
        [mx.ones((block, context), dtype=mx.bool_),
         mx.tril(mx.ones((block, block), dtype=mx.bool_))], axis=-1,
    )
    return full, causal


def test_dflash2_block_attends_bidirectionally_as_trained():
    attention, x, x_ctx, rope = _attention_case()
    served = attention(x, x_ctx, rope, KVCache())
    full, causal = _masks(x_ctx.shape[1], x.shape[1])
    bidirectional = _reference_attention(attention, x, x_ctx, rope, full)
    causal_ref = _reference_attention(attention, x, x_ctx, rope, causal)
    np.testing.assert_allclose(np.asarray(served), np.asarray(bidirectional), atol=1e-5)
    # Teeth: the causal law the vLLM bug served is measurably different on
    # every block row but the last, so this comparison would catch it.
    gap = np.abs(np.asarray(served) - np.asarray(causal_ref)).max(axis=-1)[0]
    assert (gap[:-1] > 1e-3).all() and gap[-1] < 1e-5


def test_dflash2_early_block_rows_see_later_block_rows():
    attention, x, x_ctx, rope = _attention_case()
    base = np.asarray(attention(x, x_ctx, rope, KVCache()))
    perturbed = x.at[:, 3].add(1.0)
    moved = np.asarray(attention(perturbed, x_ctx, rope, KVCache()))
    # Row 0 (the anchor) depends on the last mask row only through
    # bidirectional block attention.
    assert np.abs(base[0, 0] - moved[0, 0]).max() > 1e-4


def _qwen38_dflash2_config():
    if not QWEN38_DFLASH2.exists():
        pytest.skip("Qwen3.8 DFlash2 checkpoint not on disk")
    return json.loads(QWEN38_DFLASH2.read_text())


def test_real_dflash2_checkpoint_declares_the_bidirectional_contract():
    raw = _qwen38_dflash2_config()
    assert raw["is_causal"] is False
    assert set(raw["layer_types"]) == {"sliding_attention"}
    config = DFlash2Config.from_dict(raw)
    assert config.is_causal is False


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda c: c["dflash_config"].update(causal=True), id="dflash_config.causal"),
        pytest.param(lambda c: c.update(dflash_query_causal=True), id="dflash_query_causal"),
        pytest.param(lambda c: c.update(sample_from_anchor=True), id="sample_from_anchor"),
        pytest.param(
            lambda c: c["dflash_config"].update(sample_from_anchor=True),
            id="dflash_config.sample_from_anchor",
        ),
        pytest.param(lambda c: c.update(partial_rotary_factor=0.5), id="partial_rotary_factor"),
        pytest.param(
            lambda c: c["rope_parameters"].update(partial_rotary_factor=0.5),
            id="rope_parameters.partial_rotary_factor",
        ),
    ],
)
def test_dflash2_refuses_checkpoint_metadata_for_another_contract(mutate):
    raw = copy.deepcopy(_qwen38_dflash2_config())
    mutate(raw)
    with pytest.raises(ValueError):
        DFlash2Config.from_dict(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda c: c["dflash_config"].update(causal=False), id="causal_false"),
        pytest.param(lambda c: c["dflash_config"].update(sample_from_anchor=True), id="sample_from_anchor"),
        pytest.param(lambda c: c.update(partial_rotary_factor=0.5), id="partial_rotary_factor"),
        pytest.param(
            lambda c: c.update(rope_parameters={
                "rope_type": "yarn", "factor": 32.0, "rope_theta": 500000.0,
                "partial_rotary_factor": 0.5,
            }),
            id="rope_parameters",
        ),
        pytest.param(
            lambda c: c.update(rope_scaling={"rope_type": "yarn", "factor": 32.0}),
            id="rope_scaling",
        ),
    ],
)
def test_laguna_refuses_checkpoint_metadata_for_another_contract(mutate):
    found = sorted(glob.glob(LAGUNA_GLOB))
    if not found:
        pytest.skip("poolside Laguna DFlash artifact not in HF cache")
    raw = json.loads(Path(found[0]).read_text())
    LagunaDFlashConfig.from_hf(raw)  # the real checkpoint still loads
    mutated = copy.deepcopy(raw)
    mutate(mutated)
    with pytest.raises(ValueError):
        LagunaDFlashConfig.from_hf(mutated)
