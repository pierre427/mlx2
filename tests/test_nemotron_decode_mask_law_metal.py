"""Real-Metal gate: Nemotron MTP rows keep ordinary decode bits past 8K keys.

MLX 0.32.4's one-query SDPA gives different bits for an all-true boolean
mask than for no mask once there are 8192 or more keys (M3 Pro and M5 Max:
0 mismatches at 8191 keys, about half of the output elements from 8192 to
262144).  Ordinary decode on ``BatchKVCache`` attends with that mask
(``MLX2_DECODE_MASK=array``, the default); MTP verify rows and a lane's
prompt-tail token arrived unmasked.  So past 8K the MTP route left its
ordinary reference, and a lane that fell back from MTP to plain decode
switched law mid-stream: the Nemotron 3.5 Super VL mtp-d2 256K ladder cell
failed width-1 cold/warm exactness that way.  A tiny hybrid with the Super
attention geometry (32 query heads over 2 KV heads of 128) checks ordinary
decode, pure MTP and MTP with fallback at 0 and 3 committed tokens, bit for
bit, in every logprob row and the final target cache, under both mask laws.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

PROMPT_TOKENS = 8400


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _bits(a):
    a = mx.contiguous(a)
    mx.eval(a)
    view = a.view(mx.uint16) if a.dtype in (mx.bfloat16, mx.float16) else a.view(mx.uint32)
    return np.asarray(view)


def test_sdpa_mask_law_is_the_premise():
    """Pins why the law matters on this build; skips once MLX unifies them."""
    q = mx.random.normal((1, 32, 1, 128), key=mx.random.key(1)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, PROMPT_TOKENS, 128), key=mx.random.key(2)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, PROMPT_TOKENS, 128), key=mx.random.key(3)).astype(mx.bfloat16)
    bare = mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5, mask=None)
    full = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=128 ** -0.5, mask=mx.ones((1, 1, 1, PROMPT_TOKENS), dtype=mx.bool_)
    )
    if np.array_equal(_bits(bare), _bits(full)):
        pytest.skip("this MLX build gives equal bits for both masks")


def _tiny_mtp_model():
    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    mx.random.seed(2026)
    layers = ["mamba", "moe", "attention", "mamba"]
    model = Model(ModelArgs.from_dict({
        "model_type": "nemotron_h", "vocab_size": 128, "hidden_size": 128,
        "intermediate_size": 128, "max_position_embeddings": 16384,
        "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        "attention_bias": False, "mamba_num_heads": 8, "mamba_head_dim": 16,
        "mamba_proj_bias": False, "ssm_state_size": 32, "conv_kernel": 4,
        "n_groups": 2, "mlp_bias": False, "layer_norm_epsilon": 1e-5,
        "use_bias": False, "use_conv_bias": True, "layers_block_type": layers,
        "num_hidden_layers": len(layers), "moe_latent_size": 64,
        "moe_intermediate_size": 64, "moe_shared_expert_intermediate_size": 64,
        "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
        "routed_scaling_factor": 2.0, "time_step_limit": (0.0, float("inf")),
        "num_nextn_predict_layers": 1, "mtp_layers_block_type": ["attention", "moe"],
    }))
    model.load_weights(
        [(name, value.astype(mx.bfloat16)) for name, value in tree_flatten(model.parameters())]
    )
    mx.eval(model.parameters())
    return model


def _greedy(model, prompt, *, mtp, fallback_after=None, max_tokens=12):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    kwargs = dict(completion_batch_size=1, prefill_batch_size=1, prefill_step_size=2048)
    insert = dict(max_tokens=[max_tokens])
    if mtp:
        kwargs["self_mtp"] = {
            "num_draft": 2, "persistent": True,
            "segment_aware_live_tip": True, "segment_aware_cohort_size": 1,
        }
        insert.update(lane_rngs=[LaneRNG(3)], self_mtp_configs=[{"sampling_temp": 0.0}])
    if fallback_after is not None:
        kwargs["mtp_admission"] = lambda rows: {
            row[0]: "plain" if int(row[1]) - len(prompt) >= fallback_after else 2
            for row in rows
        }
    batch = BatchGenerator(model, **kwargs)
    batch.insert([prompt], **insert)
    rows = []
    for _ in range(200):
        _, responses = batch.next()
        for response in responses:
            rows.append((int(response.token), _bits(response.logprobs.astype(mx.float32))))
            if response.finish_reason is not None:
                route = (getattr(response, "mtp_receipt", None) or {}).get("route")
                arrays = []
                for entry in response.prompt_cache:
                    if isinstance(getattr(entry, "keys", None), mx.array):
                        arrays += [entry.keys[..., : entry.offset, :],
                                   entry.values[..., : entry.offset, :]]
                    else:
                        arrays += [a for a in entry.cache if a is not None]
                return route, rows, [_bits(a) for a in arrays]
    pytest.fail("generation did not finish")


@pytest.mark.parametrize("law", ["array", "none"])
def test_ordinary_mtp_and_fallback_agree_past_8k_keys(monkeypatch, law):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX2_DECODE_MASK", law)
    model = _tiny_mtp_model()
    prompt = [int(t) for t in np.random.default_rng(8).integers(1, 128, size=PROMPT_TOKENS)]
    _, ordinary, ordinary_cache = _greedy(model, prompt, mtp=False)
    arms = {"mtp": _greedy(model, prompt, mtp=True)}
    for k in (0, 3):
        arms[f"fallback@{k}"] = _greedy(model, prompt, mtp=True, fallback_after=k)
    assert arms["mtp"][0] == "segmented_self_mtp"
    for name, (route, rows, cache) in arms.items():
        if name != "mtp":
            assert route == "ordinary_after_mtp_fallback"
        assert [t for t, _ in rows] == [t for t, _ in ordinary], name
        for index, ((_, want), (_, have)) in enumerate(zip(ordinary, rows, strict=True)):
            assert np.array_equal(have, want), f"{name}: logprob row {index} differs"
        for want, have in zip(ordinary_cache, cache, strict=True):
            np.testing.assert_array_equal(have, want, err_msg=name)


def test_padded_batch_one_row_equals_ordinary_decode_past_8k_keys(monkeypatch):
    """B=2 with left padding: an unmasked one-row call gets the cache's own
    decode mask (false on padding), bit-equal to ordinary masked decode."""
    import copy

    from mlx2.runtime.models.cache import BatchKVCache
    from mlx2.runtime.models.nemotron_h import ModelArgs, NemotronHAttention

    monkeypatch.delenv("MLX2_DECODE_MASK", raising=False)
    mx.random.seed(5)
    layer = NemotronHAttention(ModelArgs.from_dict({
        "model_type": "nemotron_h", "vocab_size": 16, "hidden_size": 128,
        "intermediate_size": 128, "num_attention_heads": 32, "num_key_value_heads": 2,
        "head_dim": 128, "attention_bias": False, "mamba_num_heads": 8,
        "mamba_head_dim": 16, "ssm_state_size": 32, "conv_kernel": 4, "n_groups": 2,
        "layers_block_type": ["attention"], "num_hidden_layers": 1,
        "layer_norm_epsilon": 1e-5, "max_position_embeddings": 16384,
        "mamba_proj_bias": False, "mlp_bias": False, "use_bias": False,
        "use_conv_bias": True,
    }))
    layer.load_weights(
        [(name, value.astype(mx.bfloat16)) for name, value in tree_flatten(layer.parameters())]
    )
    reference = BatchKVCache([300, 0])
    prompt = mx.random.normal((2, PROMPT_TOKENS, 128), key=mx.random.key(6)).astype(mx.bfloat16)
    mx.eval(layer(prompt, mask=reference.make_mask(PROMPT_TOKENS, return_array=True), cache=reference))
    candidate = copy.deepcopy(reference)
    x = mx.random.normal((2, 1, 128), key=mx.random.key(7)).astype(mx.bfloat16)
    want = layer(x, mask=reference.make_mask(1), cache=reference)
    have = layer(x, mask=None, cache=candidate)
    np.testing.assert_array_equal(_bits(have), _bits(want))
