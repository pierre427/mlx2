"""int8 prefill declarations and selection on the dense 8-bit adapters.

CPU only, no Metal kernel and no ``mx.eval``: each test builds the served
module tree of a small synthetic model of the architecture, quantizes it the
way the checkpoint loader does (8-bit gs64 affine), and checks the scopes the
adapter declares and the projections ``checkpoint_census`` / the adapter's
selector pick.
"""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import int8_prefill as ip


def _quantize_8bit(model, skip_prefix=()):
    model.set_dtype(mx.bfloat16)  # checkpoint scales/biases are bf16
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda path, module: (
            hasattr(module, "to_quantized")
            and not path.startswith(skip_prefix)
            and module.weight.shape[-1] % 64 == 0
        ),
    )


def _chosen(adapter, scope):
    selector = getattr(adapter, "int8_prefill_select", None)
    chooser = (selector(scope) if callable(selector) else None) or ip.default_select(scope)
    return sorted(
        path
        for path, module, spec, _ in ip._projection_candidates(adapter.model, chooser)
        if spec is not None and ip.q8_inplace_eligible(spec, module)[0]
    )


# --------------------------------------------------------------------------
# standard decoder (Qwen3 dense)
# --------------------------------------------------------------------------


def _standard_config(**overrides):
    config = {
        "model_type": "qwen3",
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "intermediate_size": 512,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
    }
    config.update(overrides)
    return config


def _standard_adapter(config, external_policy=None, *, build=True):
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter

    adapter = object.__new__(StandardDecoderAdapter)
    adapter.config = config
    adapter.external_policy = dict(external_policy or {})
    if build:
        from mlx2.runtime.models.standard_decoder import Model, ModelArgs

        adapter.model = Model(ModelArgs.from_dict(config))
        _quantize_8bit(adapter.model)
    return adapter


def test_standard_decoder_declares_dense_qwen3_only():
    assert ip.adapter_scopes(_standard_adapter(_standard_config(), build=False)) == {
        "mlp",
        "all",
    }
    for family in ("qwen2", "llama"):
        config = _standard_config(model_type=family)
        assert ip.adapter_scopes(_standard_adapter(config, build=False)) == frozenset()
    moe = _standard_config(model_type="qwen3_moe", num_experts=8)
    assert ip.adapter_scopes(_standard_adapter(moe, build=False)) == frozenset()
    row_exact = _standard_adapter(
        _standard_config(), {"target_verify_row_exact": True}, build=False
    )
    assert ip.adapter_scopes(row_exact) == frozenset()
    assert ip.auto_policy(row_exact) is None


def test_standard_decoder_census_selects_decoder_projections_not_the_head():
    adapter = _standard_adapter(_standard_config())
    assert not hasattr(adapter, "int8_prefill_select")  # default classifier
    mlp = [f"model.layers.{i}.mlp.{p}" for i in range(2)
           for p in ("down_proj", "gate_proj", "up_proj")]
    attn = [f"model.layers.{i}.self_attn.{p}" for i in range(2)
            for p in ("k_proj", "o_proj", "q_proj", "v_proj")]
    assert _chosen(adapter, "mlp") == sorted(mlp)
    assert _chosen(adapter, "all") == sorted(mlp + attn)
    assert ip.checkpoint_census(adapter, "mlp") == {"q8": 6, "q45": 0, "other_quantized": 0, "linear": 0}
    assert ip.checkpoint_census(adapter, "all") == {"q8": 14, "q45": 0, "other_quantized": 0, "linear": 0}
    # the head is an eligible-shaped 8-bit projection: excluded by name
    assert isinstance(adapter.model.lm_head, nn.QuantizedLinear)
    assert ip.auto_policy(adapter).scope == "all"


def test_standard_decoder_default_routes_stay_below_the_row_threshold():
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter

    ordinary = ip.max_decode_rows(max_lanes=4, config={}, speculation="ordinary")
    external = ip.max_decode_rows(
        max_lanes=4,
        config={"num_draft": StandardDecoderAdapter.EXTERNAL_DEFAULT_NUM_DRAFT},
        speculation="external_draft",
    )
    assert ordinary == 8 and external == 68
    assert max(ordinary, external) < ip.DEFAULT_ROW_THRESHOLD


# --------------------------------------------------------------------------
# Gemma 4
# --------------------------------------------------------------------------


def _tiny_gemma4():
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.gemma4.gemma4 import Model

    # The 31B layout at eligible toy widths: a sliding layer with v_proj and a
    # full-attention layer whose K doubles as V (no v_proj).
    text = TextConfig(
        hidden_size=256, num_hidden_layers=2, intermediate_size=512,
        num_attention_heads=2, head_dim=128, global_head_dim=128,
        num_key_value_heads=2, num_global_key_value_heads=2,
        attention_k_eq_v=True, num_kv_shared_layers=0,
        hidden_size_per_layer_input=0, vocab_size=512,
        vocab_size_per_layer_input=512, sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        use_double_wide_mlp=False,
    )
    vision = VisionConfig(
        hidden_size=256, intermediate_size=512, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=128,
        global_head_dim=128, default_output_length=4, pooling_kernel_size=1,
        patch_size=2, position_embedding_size=16,
    )
    model = Model(ModelConfig(text_config=text, vision_config=vision, image_token_id=7))
    # As the checkpoint: the vision tower stays bf16; the multimodal
    # projector and the embedding are 8-bit.
    _quantize_8bit(model, skip_prefix=("vision_tower",))
    return model


def _gemma_adapter(cls, model):
    from mlx2.adapters.gemma4 import _Gemma4LogitsModel

    adapter = object.__new__(cls)
    adapter.model = _Gemma4LogitsModel(model)
    return adapter


def test_gemma4_declares_on_the_dense_31b_only():
    from mlx2.adapters.gemma4 import Gemma4A4BAdapter, Gemma431BAdapter

    assert ip.adapter_scopes(Gemma431BAdapter) == {"mlp", "all"}
    assert ip.adapter_scopes(Gemma4A4BAdapter) == frozenset()


def test_gemma4_census_selects_text_decoder_projections_only():
    from mlx2.adapters.gemma4 import Gemma431BAdapter

    model = _tiny_gemma4()
    adapter = _gemma_adapter(Gemma431BAdapter, model)
    assert isinstance(model.embed_vision.embedding_projection, nn.QuantizedLinear)
    prefix = "language_model.model.layers"
    mlp = [f"{prefix}.{i}.mlp.{p}" for i in range(2)
           for p in ("down_proj", "gate_proj", "up_proj")]
    attn = [f"{prefix}.0.self_attn.{p}" for p in ("k_proj", "o_proj", "q_proj", "v_proj")]
    attn += [f"{prefix}.1.self_attn.{p}" for p in ("k_proj", "o_proj", "q_proj")]
    assert _chosen(adapter, "mlp") == sorted(mlp)
    assert _chosen(adapter, "all") == sorted(mlp + attn)
    assert ip.checkpoint_census(adapter, "all") == {"q8": 13, "q45": 0, "other_quantized": 0, "linear": 0}
    # The default classifier would also take the multimodal projector.
    assert "embed_vision.embedding_projection" in [
        path for path, _, spec, _ in ip._projection_candidates(
            adapter.model, ip.default_select("all")) if spec is not None
    ]
    select = Gemma431BAdapter.int8_prefill_select("all")
    for path in (
        "embed_vision.embedding_projection",
        "vision_tower.encoder.layers.0.mlp.gate_proj.linear",
        "language_model.model.embed_tokens",
        f"{prefix}.0.per_layer_input_gate",
        f"{prefix}.0.per_layer_projection",
        f"{prefix}.0.router.proj",
        f"{prefix}.0.experts.switch_glu.gate_proj",
    ):
        assert not select(path, None), path


def test_gemma4_ordinary_route_stays_below_the_row_threshold():
    from mlx2.adapters.gemma4 import Gemma431BAdapter

    bound = ip.max_decode_rows(max_lanes=4, config={}, speculation="ordinary")
    assert bound < ip.DEFAULT_ROW_THRESHOLD
    # The served 512-row chunk reaches the threshold, so full chunks take int8.
    assert Gemma431BAdapter.default_prefill_step >= ip.DEFAULT_ROW_THRESHOLD


# --------------------------------------------------------------------------
# Muse Glimmer (text-only port)
# --------------------------------------------------------------------------


def _muse_adapter(external_policy=None, *, build=True):
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter

    adapter = object.__new__(MuseGlimmerAdapter)
    adapter.external_policy = dict(external_policy or {})
    if build:
        from mlx2.adapters.muse_glimmer_config import ModelArgs
        from mlx2.runtime.models.muse_glimmer import Model

        args = ModelArgs.from_dict({
            "model_type": "muse_glimmer",
            "text_config": {
                "model_type": "muse_glimmer_text", "num_hidden_layers": 4,
                "hidden_size": 256, "intermediate_size": 512,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 128,
                "vocab_size": 1024,
            },
        })
        adapter.model = Model(args)
        _quantize_8bit(adapter.model)
    return adapter


def test_muse_declares_unless_row_exact():
    assert ip.adapter_scopes(_muse_adapter(build=False)) == {"mlp", "all"}
    row_exact = _muse_adapter({"target_verify_row_exact": True}, build=False)
    assert ip.adapter_scopes(row_exact) == frozenset()
    assert ip.auto_policy(row_exact) is None


def test_muse_census_selects_text_decoder_projections_with_the_attention_gate():
    adapter = _muse_adapter()
    mlp = [f"model.layers.{i}.mlp.{p}" for i in range(4)
           for p in ("down_proj", "gate_proj", "up_proj")]
    attn = [f"model.layers.{i}.self_attn.{p}" for i in range(4)
            for p in ("gate_proj", "k_proj", "o_proj", "q_proj", "v_proj")]
    assert _chosen(adapter, "mlp") == sorted(mlp)
    assert _chosen(adapter, "all") == sorted(mlp + attn)
    assert isinstance(adapter.model.lm_head, nn.QuantizedLinear)
    select = adapter.int8_prefill_select("all")
    for path in ("lm_head", "vision_adapter.fc1", "vision_projection",
                 "vision_tower.layers.0.mlp.fc1", "vision_tower.layers.0.attn.proj"):
        assert not select(path, None), path


def test_muse_external_draft_routes_stay_below_the_row_threshold():
    from mlx2.adapters.muse_glimmer import DFLASH2_DEFAULT_NUM_DRAFT

    # Default DFlash2 width 3, the qualified profile's 4, and the widest
    # admitted width (below the drafter's 16-token block).
    for num_draft in (DFLASH2_DEFAULT_NUM_DRAFT, 4, 15):
        bound = ip.max_decode_rows(
            max_lanes=4, config={"num_draft": num_draft}, speculation="external_draft"
        )
        assert bound < ip.DEFAULT_ROW_THRESHOLD, num_draft


def test_qwen36_27b_opts_out_of_int8_prefill():
    # Measured no-go on tool-call text (int8-dense8-e2e-20261009); the
    # Qwen3.8-27B parent keeps its declaration.
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime.int8_prefill import adapter_scopes

    assert adapter_scopes(Qwen3627BAdapter) == frozenset()
    assert adapter_scopes(Qwen3827BAdapter) == {"mlp", "all"}
