# SPDX-License-Identifier: MIT
"""North (Cohere2Moe) normalization choice: RMSNorm, not LayerNorm.

Why this file exists
--------------------
Until 2026-09-20 mlx2 built North's ``input_layernorm`` and ``model.norm`` as
``nn.LayerNorm(hidden, eps=layer_norm_eps, bias=False)``.  That is the
mean-centred Cohere norm, and it is the wrong one for this checkpoint.

The reference (HF transformers ``models/cohere2_moe/modeling_cohere2_moe.py``,
decoder layer and final norm) selects the norm *class* from ``rms_norm_eps``::

    Cohere2MoeRMSNorm(hidden_size, eps=config.rms_norm_eps)
    if config.rms_norm_eps is not None
    else Cohere2MoeLayerNorm(hidden_size, eps=config.layer_norm_eps)

``North-Mini-Code-1.0`` ships ``rms_norm_eps: 1e-06``, so the reference runs
RMSNorm with eps 1e-6.  The ``layer_norm_eps: 1e-05`` alongside it is only
``Cohere2MoeConfig``'s own default being serialized and is inert here.
mlx-vlm's independent ``cohere2_moe`` port makes the same choice.

The weights cannot arbitrate and must not be read as agreement: RMSNorm and the
Cohere LayerNorm have *identical* parameter shapes (a single ``[hidden]`` gamma,
no bias).  The real checkpoint carries 49 ``input_layernorm.weight`` plus one
``model.norm.weight``, all ``[2048]``, and zero ``*.bias`` tensors, so the wrong
norm loads cleanly and silently.  Shape checks can never catch this; only this
test can.  Do not "simplify" ``_norm_layer`` back to a single LayerNorm.
"""

import mlx.core as mx
import pytest
from mlx import nn


from mlx2.runtime.models.cohere2_moe import Cohere2MoeModel, ModelArgs, norm_counters


def tiny(**over) -> ModelArgs:
    base = dict(
        hidden_size=16,
        head_dim=4,
        num_hidden_layers=4,
        intermediate_size=8,
        prefix_dense_intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        sliding_window=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-06,
    )
    base.update(over)
    return ModelArgs(**base)


def test_rms_norm_eps_is_a_declared_field_and_survives_from_dict():
    # BaseModelArgs.from_dict filters keys it does not declare.  Before the fix
    # rms_norm_eps was not declared, so the checkpoint's 1e-06 was dropped on
    # the floor and the model fell through to LayerNorm.
    args = ModelArgs.from_dict({"model_type": "cohere2_moe", "rms_norm_eps": 1e-06})
    assert args.rms_norm_eps == 1e-06


def test_north_config_selects_rms_norm():
    model = Cohere2MoeModel(tiny())
    assert isinstance(model.norm, nn.RMSNorm)
    assert all(isinstance(l.input_layernorm, nn.RMSNorm) for l in model.layers)
    assert model.norm.eps == 1e-06


def test_absent_rms_norm_eps_still_selects_layer_norm():
    # The other branch of the reference conditional stays reachable for a
    # Cohere2Moe checkpoint that genuinely omits rms_norm_eps.
    model = Cohere2MoeModel(tiny(rms_norm_eps=None))
    assert isinstance(model.norm, nn.LayerNorm)
    assert all(isinstance(l.input_layernorm, nn.LayerNorm) for l in model.layers)


def test_counters_report_which_norm_was_built():
    before = norm_counters()
    Cohere2MoeModel(tiny())
    after = norm_counters()
    # 4 decoder layers + the final norm.
    assert after["rms"] - before["rms"] == 5
    assert after["layernorm"] == before["layernorm"]


def test_legacy_override_is_opt_in_and_counted(monkeypatch):
    monkeypatch.setenv("MLX2_NORTH_NORM", "legacy_layernorm")
    before = norm_counters()
    model = Cohere2MoeModel(tiny())
    after = norm_counters()
    assert isinstance(model.norm, nn.LayerNorm)
    assert after["legacy_override"] - before["legacy_override"] == 5
    assert after["rms"] == before["rms"]


def test_the_two_norms_are_not_interchangeable():
    # Guards the premise: if these agreed numerically the bug would be cosmetic.
    # A residual stream with a non-zero mean is normalized differently.
    x = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    rms = nn.RMSNorm(4, eps=1e-06)
    ln = nn.LayerNorm(4, eps=1e-05, bias=False)
    assert not mx.allclose(rms(x), ln(x), atol=1e-3)


def test_rms_matches_the_reference_formula():
    args = tiny()
    model = Cohere2MoeModel(args)
    x = mx.random.normal((1, 3, args.hidden_size)) + 0.7
    gamma = model.norm.weight
    x32 = x.astype(mx.float32)
    want = gamma * (
        x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + args.rms_norm_eps)
    )
    assert mx.allclose(model.norm(x), want.astype(x.dtype), atol=1e-5)


# --------------------------------------------------------------------------
# The Cohere EAGLE drafter had the same mistake in its final norm.
# --------------------------------------------------------------------------


def _eagle_config():
    from mlx2.runtime.drafters.cohere_eagle import CohereEagleConfig

    return CohereEagleConfig(
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        sliding_window=8,
        num_target_layers=4,
    )


def test_eagle_drafter_final_norm_is_rms():
    """The drafter's final norm must match the geometry its head reads.

    EAGLE-1's final hidden is projected by the *bound target's* lm_head, and
    the target's final norm is RMSNorm.  The drafter's own decoder layers
    already used RMSNorm from the same config; only the final norm was a
    mean-centred LayerNorm, which is inconsistent within one model and feeds
    the shared head a differently-shaped distribution.  Its published config
    declares `norm_type: "rms_norm"` once for the whole model.

    The weights cannot catch this either: `model.norm.weight` is a bare
    [hidden] gamma with no bias, which both norms accept.
    """
    from mlx2.runtime.drafters.cohere_eagle import CohereEagleDraftModel

    config = _eagle_config()
    model = CohereEagleDraftModel(config)
    assert isinstance(model.norm, nn.RMSNorm)
    assert model.norm.eps == config.rms_norm_eps == 1e-6
    assert all(isinstance(l.input_layernorm, nn.RMSNorm) for l in model.layers)


def test_eagle_from_hf_pins_norm_type():
    from mlx2.runtime.drafters.cohere_eagle import CohereEagleConfig

    published = {
        "architectures": ["EagleCohereForCausalLM"],
        "transformer_block_type": "parallel",
        "use_qk_norm": False,
        "position_embedding_type": "rope_gptj",
        "rope_scaling": None,
        "attention_bias": False,
        "hidden_act": "silu",
        "use_gated_activation": True,
        "norm_type": "rms_norm",
        "hidden_size": 16,
        "intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
        "sliding_window": 8,
        "rms_norm_eps": 1e-06,
        "layer_types": ["sliding_attention", "sliding_attention"],
    }
    config = CohereEagleConfig.from_hf(published, num_target_layers=4)
    assert config.rms_norm_eps == 1e-06

    for bad in ("layer_norm", None):
        broken = dict(published)
        if bad is None:
            del broken["norm_type"]
        else:
            broken["norm_type"] = bad
        with pytest.raises(ValueError, match="norm_type"):
            CohereEagleConfig.from_hf(broken, num_target_layers=4)
