"""RecursiveSI/flint-int4 layout: no index, uint8-packed rows, export-declared law."""

import hashlib
import json

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import flint


def export(tmp_path, *, embed_bits=8, extra_shard=False, config_digest=None):
    w, s, b = mx.quantize(mx.random.normal((8, 128)), group_size=64, bits=4)
    e, es, eb = mx.quantize(mx.random.normal((16, 128)), group_size=64, bits=embed_bits)
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.asarray(w).view(np.uint8),
        "model.layers.0.self_attn.q_proj.scales": np.asarray(s.astype(mx.float32)),
        "model.layers.0.self_attn.q_proj.biases": np.asarray(b.astype(mx.float32)),
        "model.embed_tokens.weight": np.asarray(e).view(np.uint8),
        "model.embed_tokens.scales": np.asarray(es.astype(mx.float32)),
        "model.embed_tokens.biases": np.asarray(eb.astype(mx.float32)),
        "model.norm.weight": np.ones((128,), dtype=np.float32),
    }
    mx.save_safetensors(str(tmp_path / "model-00000.safetensors"),
                        {k: mx.array(v) for k, v in tensors.items()}, metadata={"format": "mlx"})
    if extra_shard:
        mx.save_safetensors(str(tmp_path / "model-00001.safetensors"), {"x": mx.zeros((1,))},
                            metadata={"format": "mlx"})
    config = {"quantization": {"bits": 4, "group_size": 64},
              "mlx_export": {"bits": 4, "routed_expert_bits": 4, "embedding_bits": 8}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    digest = config_digest or hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    (tmp_path / flint.UPSTREAM_INT4_MANIFEST).write_text(json.dumps({
        "weight_format": flint.UPSTREAM_INT4_FORMAT,
        "checkpoint_files": {"config.json": digest, "model-00000.safetensors": "x"}}))
    return config, w


def test_upstream_layout_maps_shards_and_declares_the_int8_embedding(tmp_path):
    config, _ = export(tmp_path)
    weights, law = flint._upstream_int4_layout(tmp_path, config)
    assert weights["model.embed_tokens.weight"] == "model-00000.safetensors"
    assert law == {"group_size": 64, "bits": 4, "mode": "affine",
                   "model.embed_tokens": {"group_size": 64, "bits": 8, "mode": "affine"}}


def test_uint8_rows_view_back_to_the_mlx_words_bit_for_bit(tmp_path):
    _, words = export(tmp_path)
    loaded = mx.load(str(tmp_path / "model-00000.safetensors"))
    rows = loaded["model.layers.0.self_attn.q_proj.weight"]
    assert rows.dtype == mx.uint8
    assert mx.array_equal(rows.view(mx.uint32), words).item()


@pytest.mark.parametrize("case", ["embed_packed_4bit", "shard_not_in_manifest", "config_digest"])
def test_upstream_layout_fails_closed(tmp_path, case):
    kwargs = {"embed_packed_4bit": {"embed_bits": 4}, "shard_not_in_manifest": {"extra_shard": True},
              "config_digest": {"config_digest": "0" * 64}}[case]
    config, _ = export(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="Flint upstream INT4"):
        flint._upstream_int4_layout(tmp_path, config)
