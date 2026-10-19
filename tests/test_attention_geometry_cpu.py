"""Declared attention geometry on the real model classes; CPU, tiny shapes."""
import mlx.core as mx

from mlx2.runtime.attention_geometry import declared_attention_geometry


def test_muse_declares_sliding_and_full_layers_under_their_windows():
    from test_external_dflash2_cpu import tiny

    mx.set_default_device(mx.cpu)
    model, _ = tiny(sliding_window=3)
    geometry = model.progressive_external_attention_geometry
    kinds = list(model.args.layer_types)
    per_layer = 2 * model.args.num_key_value_heads * model.args.head_dim * 2
    assert geometry["kv_bytes_per_token"] == kinds.count("full_attention") * per_layer
    assert geometry["windows"] == ({3: kinds.count("sliding_attention") * per_layer}
                                   if "sliding_attention" in kinds else {})
    assert geometry["full_attention_layers"] + geometry["sliding_attention_layers"] == len(kinds)


def test_qwen38_27b_counts_only_full_attention_layers():
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

    mx.set_default_device(mx.cpu)
    text = dict(model_type="qwen3_5", hidden_size=32, intermediate_size=64, num_hidden_layers=8,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=128,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                linear_value_head_dim=8, linear_conv_kernel_dim=3, full_attention_interval=4,
                mtp_num_hidden_layers=0, partial_rotary_factor=0.5, rope_parameters=None,
                max_position_embeddings=128)
    model = Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": text}))
    geometry = model.progressive_external_attention_geometry
    # Eight layers at interval four: two keep a KV cache, six are recurrent.
    assert geometry["full_attention_layers"] == 2 and geometry["sliding_attention_layers"] == 0
    assert geometry["kv_bytes_per_token"] == 2 * 2 * 2 * 8 * 2
    assert geometry["windows"] == {}
    assert geometry == declared_attention_geometry(full_layers=2, kv_heads=2, head_dim=8)


def test_flash_next_declares_full_attention_layers_only():
    from test_batched_mtp import _tiny_qwen4_model

    mx.set_default_device(mx.cpu)
    model = _tiny_qwen4_model()
    args = model.language_model.args
    geometry = model.progressive_external_attention_geometry
    full = sum(kind != "linear_attention" for kind in args.layer_types)
    assert geometry["full_attention_layers"] == full > 0
    assert geometry["kv_bytes_per_token"] == full * 2 * args.num_key_value_heads * args.head_dim * 2
    assert geometry["windows"] == {}
