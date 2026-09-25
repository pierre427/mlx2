"""The original Hugging Face Muse checkpoint loads exactly like the MLX conversion.

The HF original nests everything under ``model.`` (``model.language_model.*``,
``model.vision_tower.*``, ...) and keeps ``lm_head.weight`` at the top level;
the MLX conversion uses ``language_model.model.*``, ``language_model.lm_head.*``
and top-level ``vision_*``. Tensors are otherwise identical, so ``sanitize`` is
a pure rename plus a vision-tower drop. Runs on CPU with a tiny synthetic model.
"""

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters.muse_glimmer import inspect_artifact
from mlx2.adapters.muse_glimmer_config import ModelArgs


TEXT_CONFIG = dict(
    hidden_size=64, num_hidden_layers=4, intermediate_size=64,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    vocab_size=128, sliding_window=8, max_position_embeddings=4096,
    hidden_activation="silu", eos_token_id=1,
    layer_types=["sliding_attention"] * 3 + ["full_attention"],
)

# One representative of every vision family the HF original carries.
VISION = {
    "vision_tower.layers.0.attn.q_proj.weight": (8, 8),
    "vision_tower.layers.0.attn.q_proj.bias": (8,),
    "vision_tower.layers.0.norm1.weight": (8,),
    "vision_tower.ln_pre.weight": (8,),
    "vision_tower.patch_embedder.patch_embedding.weight": (8, 12),
    "vision_tower.patch_embedder.position_embedding_table.weight": (4, 8),
    "vision_adapter.fc1.weight": (8, 8),
    "vision_adapter.fc2.weight": (8, 8),
    "vision_projection.weight": (64, 8),
}


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _reference():
    from mlx2.runtime.models.muse_glimmer import Model

    mx.random.seed(3)
    model = Model(ModelArgs.from_dict({"text_config": TEXT_CONFIG}))
    model.eval()
    mx.eval(model.parameters())
    return model


def _hf_name(name):
    if name.startswith("model."):
        return "model.language_model." + name[len("model."):]
    return name  # lm_head.weight stays at the top level


def _mlx_name(name):
    if name.startswith("model."):
        return "language_model.model." + name[len("model."):]
    return "language_model." + name


def _write(tmp_path, model, rename, vision_prefix):
    tensors = {rename(k): v for k, v in tree_flatten(model.parameters())}
    tensors.update(
        {vision_prefix + k: mx.zeros(shape, dtype=mx.bfloat16) for k, shape in VISION.items()}
    )
    names = sorted(tensors)
    shards = [names[: len(names) // 2], names[len(names) // 2 :]]
    weight_map = {}
    for i, shard in enumerate(shards, 1):
        file = f"model-0000{i}-of-00002.safetensors"
        mx.save_safetensors(str(tmp_path / file), {k: tensors[k] for k in shard})
        weight_map.update({k: file for k in shard})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MuseGlimmerForConditionalGeneration"],
                "model_type": "muse_glimmer",
                "text_config": TEXT_CONFIG,
                "vision_config": {"hidden_size": 8},
            }
        )
    )
    return tmp_path


def _load(path):
    from mlx2.runtime.models.muse_glimmer import Model
    from mlx2.runtime.ubc_evict import load_shards_evicting

    identity = inspect_artifact(path)
    config = json.loads((path / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    files = [path / record[0] for record in identity["files"]]
    weights = load_shards_evicting(files, sanitize=model.sanitize)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model


def test_hf_original_key_layout_loads_like_the_mlx_conversion(tmp_path):
    reference = _reference()
    (tmp_path / "hf").mkdir()
    (tmp_path / "mlx").mkdir()
    hf = _load(_write(tmp_path / "hf", reference, _hf_name, "model."))
    converted = _load(_write(tmp_path / "mlx", reference, _mlx_name, ""))
    expected = dict(tree_flatten(reference.parameters()))
    for model in (hf, converted):
        loaded = dict(tree_flatten(model.parameters()))
        assert loaded.keys() == expected.keys()
        for name, value in expected.items():
            assert mx.array_equal(loaded[name], value), name
    tokens = mx.array([[5, 9, 17, 33, 2, 64, 7, 11, 90, 3]], dtype=mx.uint32)
    a = hf(tokens, cache=hf.make_cache())
    b = converted(tokens, cache=converted.make_cache())
    assert mx.array_equal(a, b)


def test_sanitize_drops_model_prefixed_vision_and_renames_text_tower():
    reference = _reference()
    params = dict(tree_flatten(reference.parameters()))
    names = {_hf_name(k): k for k in params}
    names.update({"model." + k: None for k in VISION})
    out = reference.sanitize(dict(names))
    assert set(out) == set(params)
    assert all(out[k] == k for k in out)
