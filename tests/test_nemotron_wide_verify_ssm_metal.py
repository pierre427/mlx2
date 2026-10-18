"""Real-Metal gate: wide Nemotron verify strips keep one-token decode bits.

On Metal ``ssm_update`` uses a sequential kernel for 2..8 rows and the
matrix form (``ssm_attn``) for 9..31.  Before the fix a 9+ row verify strip
(copy-draft span 8, ``num_draft >= 8``) committed a recurrent state that
serial decode never produces: on M3 Metal, a mini Super (production layer
geometry) at ``num_draft`` 8 or 12 ended with a different target cache than
ordinary decode, and a forced MTP->plain fallback gave a third one.  Here a
tiny hybrid with the same dispatch law checks both the component and the
whole greedy route, bit for bit.
"""

from __future__ import annotations


import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _bits(a):
    mx.eval(a)
    view = a.view(mx.uint16) if a.dtype in (mx.bfloat16, mx.float16) else a.view(mx.uint32)
    return np.asarray(view)


@pytest.mark.parametrize("steps", [9, 12, 16])
def test_serial_exact_strip_matches_serial_decode(steps):
    from mlx2.runtime.models.nemotron_h import _serial_exact_ssm_update
    from mlx2.runtime.models.ssm import ssm_update

    keys = mx.random.split(mx.random.key(11), 6)
    heads, head_dim, groups, state_dim = 8, 16, 2, 32
    A_log = mx.log(mx.arange(1, heads + 1, dtype=mx.float32)).astype(mx.bfloat16)
    D = mx.ones((heads,), dtype=mx.bfloat16)
    dt_bias = (mx.random.normal((heads,), key=keys[0]) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((1, heads, head_dim, state_dim), key=keys[1]) * 0.1)
    x = (mx.random.normal((1, steps, heads, head_dim), key=keys[2]) * 0.5).astype(mx.bfloat16)
    B = (mx.random.normal((1, steps, groups, state_dim), key=keys[3]) * 0.5).astype(mx.bfloat16)
    C = (mx.random.normal((1, steps, groups, state_dim), key=keys[4]) * 0.5).astype(mx.bfloat16)
    dt = mx.random.normal((1, steps, heads), key=keys[5]).astype(mx.bfloat16)
    limit = (0.0, float("inf"))
    rows, serial = [], state
    for t in range(steps):
        y, serial = ssm_update(
            x[:, t : t + 1], A_log, B[:, t : t + 1], C[:, t : t + 1], D,
            dt[:, t : t + 1], dt_bias, serial, limit,
        )
        mx.eval(y, serial)
        rows.append(y)
    y, strip = _serial_exact_ssm_update(x, A_log, B, C, D, dt, dt_bias, state, limit, None)
    np.testing.assert_array_equal(_bits(y), _bits(mx.concatenate(rows, axis=1)))
    np.testing.assert_array_equal(_bits(strip), _bits(serial))


def _tiny_mtp_model():
    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    mx.random.seed(2026)
    layers = ["mamba", "moe", "attention", "mamba"]
    model = Model(ModelArgs.from_dict({
        "model_type": "nemotron_h", "vocab_size": 128, "hidden_size": 64,
        "intermediate_size": 128, "max_position_embeddings": 4096,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
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
    model.update(
        {"lm_head": {"weight": mx.zeros_like(model.lm_head.weight)}}
    )  # every draft is accepted: verify strips are num_draft + 1 rows wide
    # The served Super policies, so verify runs whole strips through the
    # layerwise segment walk instead of one-token calls.  At this geometry
    # the fused kernels refuse and the Mamba strip takes stock ssm_update:
    # the dispatch the served route reaches when the B1 cache carries lengths.
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.set_conv_policy("fused_short")
            layer.mixer.set_ssm_policy("fused_short_gate")
            layer.mixer.set_projection_policy("short_row_exact")
    model.set_short_segment_policy("layerwise_short_exact")
    mx.eval(model.parameters())
    return model


def _final_target_cache(model, prompt, *, num_draft=None, fallback_after=None):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    kwargs = dict(completion_batch_size=1, prefill_batch_size=1, prefill_step_size=64)
    insert = dict(max_tokens=[24])
    if num_draft is not None:
        kwargs["self_mtp"] = {
            "num_draft": num_draft, "persistent": True,
            "segment_aware_live_tip": True, "segment_aware_cohort_size": 1,
        }
        if fallback_after is not None:
            kwargs["mtp_admission"] = lambda rows: {
                row[0]: "plain" if int(row[1]) - len(prompt) >= fallback_after else num_draft
                for row in rows
            }
        insert.update(lane_rngs=[LaneRNG(3)], self_mtp_configs=[{"sampling_temp": 0.0}])
    batch = BatchGenerator(model, **kwargs)
    batch.insert([prompt], **insert)
    for _ in range(200):
        _, responses = batch.next()
        for response in responses:
            if response.finish_reason is not None:
                route = (response.mtp_receipt or {}).get("route")
                arrays = []
                for entry in response.prompt_cache:
                    if hasattr(entry, "keys") and isinstance(entry.keys, mx.array):
                        arrays += [entry.keys[..., : entry.offset, :],
                                   entry.values[..., : entry.offset, :]]
                    else:
                        arrays += [a for a in entry.cache if a is not None]
                return route, [_bits(mx.contiguous(a)) for a in arrays]
    pytest.fail("generation did not finish")


@pytest.mark.parametrize("num_draft", [2, 8, 12])
def test_wide_verify_and_fallback_keep_ordinary_target_bits(monkeypatch, num_draft):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    model = _tiny_mtp_model()
    prompt = [int(t) for t in np.random.default_rng(4).integers(1, 128, size=40)]
    _, ordinary = _final_target_cache(model, prompt)
    route, mtp = _final_target_cache(model, prompt, num_draft=num_draft)
    assert route == "segmented_self_mtp"
    for want, have in zip(ordinary, mtp, strict=True):
        np.testing.assert_array_equal(have, want)
    route, fallback = _final_target_cache(
        model, prompt, num_draft=num_draft, fallback_after=num_draft + 2
    )
    assert route == "ordinary_after_mtp_fallback"
    for want, have in zip(ordinary, fallback, strict=True):
        np.testing.assert_array_equal(have, want)
