"""The Agnes adapter must import its model before loading any weights."""


def test_agnes_model_import_and_tiny_attention_on_cpu():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx2.runtime.models.agnes import (
        LAYER_GLOBAL,
        Model,
        ModelArgs,
        Qwen3NextAttention,
        TextModelArgs,
    )

    args = TextModelArgs(
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        vocab_size=32,
        layer_types=[LAYER_GLOBAL],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rope_parameters={"type": "default", "partial_rotary_factor": 0.25},
    )
    attention = Qwen3NextAttention(args)
    assert attention.num_attention_heads == 2
    model = Model(ModelArgs(model_type="agnes", text_config=args.__dict__))
    assert len(model.make_cache()) == 1
