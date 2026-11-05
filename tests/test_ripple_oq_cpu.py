"""Synthetic CPU coverage of BF16/affine storage and the Ripple oQe walk.

No real-checkpoint serving claim. The run receipt binds exact fixture and
mlx2 revisions. External oQ integration also binds the provenance revision.
"""

import copy
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx2.adapters.ripple import (
    affine_specs,
    source_weights,
    tensor_schema,
    validate_config,
    validate_headers,
)
from mlx2.conversion.ripple_oq import forward_microbatches, stream_calibrate
from mlx2.runtime.models.ricursive import Model, ModelArgs


def config():
    c = json.loads((Path(__file__).parent / "fixtures/ripple/config.json").read_text())
    c.pop("quantization_config")
    c.pop("dtype")
    return c


def args():
    return ModelArgs(hidden_size=64, intermediate_size=128, num_hidden_layers=5,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                     vocab_size=128, sliding_window=4, embed_scale=8.0)


def test_bf16_and_mixed_affine_configuration():
    c = config()
    validate_config(c)
    assert len(tensor_schema(c)) == 1203
    c["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine",
                         "lm_head": {"bits": 8, "group_size": 64, "mode": "affine"},
                         "model.layers.3.mlp.down_proj": {"bits": 6, "group_size": 64, "mode": "affine"}}
    c["quantization_config"] = copy.deepcopy(c["quantization"])
    validate_config(c)
    assert affine_specs(c)["lm_head"]["bits"] == 8
    assert tensor_schema(c)["model.layers.3.mlp.down_proj.weight"] == ("U32", [5120, 7680])
    assert len(tensor_schema(c)) == 2487
    for key, value in [("bits", 7), ("group_size", 16), ("mode", "nvfp4"), ("unknown", {})]:
        broken = copy.deepcopy(c)
        broken["quantization"][key] = value
        broken["quantization_config"] = copy.deepcopy(broken["quantization"])
        with pytest.raises(ValueError):
            validate_config(broken)
    broken = copy.deepcopy(c)
    broken["quantization_config"]["bits"] = 8
    with pytest.raises(ValueError, match="disagree"):
        validate_config(broken)


@pytest.mark.parametrize("quantized", [False, True])
def test_full_bf16_or_mixed_affine_weights_roundtrip_and_logits(tmp_path, quantized):
    mx.random.seed(84)
    a = args()
    model = Model(a, nvfp4=False)
    model.set_dtype(mx.bfloat16)
    c = asdict(a)
    if quantized:
        c["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine",
                             "lm_head": {"bits": 8, "group_size": 64, "mode": "affine"},
                             "model.layers.3.mlp.down_proj": {"bits": 6, "group_size": 64, "mode": "affine"}}
        specs = affine_specs(c)
        nn.quantize(model, class_predicate=lambda name, module: specs.get(name, False))
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    artifact = {"identity": {"path": str(tmp_path)}, "weight_map": dict.fromkeys(weights, "model.safetensors")}
    validate_headers(tmp_path, artifact["weight_map"], tensor_schema(c))
    loaded = Model(a, nvfp4=False)
    if quantized:
        nn.quantize(loaded, class_predicate=lambda name, module: specs.get(name, False))
    loaded.load_weights(list(source_weights(artifact)), strict=True)
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    np.testing.assert_array_equal(np.asarray(model(tokens).astype(mx.float32)),
                                  np.asarray(loaded(tokens).astype(mx.float32)))
    if quantized:
        linear = loaded.layers[3].mlp.down_proj
        x = mx.ones((1, 128), dtype=mx.bfloat16)
        dense = mx.dequantize(linear.weight, linear.scales, linear.biases, bits=6, group_size=64)
        np.testing.assert_allclose(np.asarray(linear(x).astype(mx.float32)),
                                   np.asarray((x @ dense.T).astype(mx.float32)), rtol=.025, atol=.02)


def test_microbatch_walk_keeps_original_embedding_seed():
    mx.random.seed(9)
    model = Model(args(), nvfp4=False)
    model.set_dtype(mx.bfloat16)
    tokens = mx.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    seed = model.model.embed_tokens(tokens) * model.args.embed_scale
    h = seed
    for layer in model.layers:
        h = forward_microbatches(layer, h, seed, 1)
    expected = mx.concatenate([model(tokens[i:i+1]) for i in range(2)], axis=0)
    actual = model.lm_head(model.model.norm(h))
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32)))


def test_pinned_oq_collector_matches_native_forward_and_restores_weights(tmp_path):
    oq = pytest.importorskip("omlx.oq", reason="offline pinned oQ integration dependency")
    assert oq.HAS_MLX
    mx.random.seed(90)
    model = Model(args(), nvfp4=False)
    model.set_dtype(mx.bfloat16)
    weights = dict(tree_flatten(model.parameters()))
    mx.eval(weights)
    original = {name: np.asarray(value.view(mx.uint16)).copy() for name, value in weights.items()}
    tokens = mx.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])

    class Reader:
        def read(self, name):
            return weights[name]

        def layer(self, _args, index):
            return model.layers[index]

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [1, 2, 3]

        def decode(self, ids):
            return str(ids)

    def draw(_tokenizer, dataset, num_samples, seq_length):
        return tokens[:, :seq_length]

    with patch.object(oq, "_load_builtin_calibration", draw):
        captured, metadata = stream_calibrate(Reader(), asdict(args()), Tokenizer(), oq,
            evidence=tmp_path, micro_batch_size=1, num_samples=2, imatrix_length=6, sensitivity_length=3)
    assert len(captured) == 41 and len(metadata["sensitivity_map"]) == 5
    after = dict(tree_flatten(model.parameters()))
    for name, value in after.items():
        np.testing.assert_array_equal(np.asarray(value.view(mx.uint16)), original[name])
    collector = oq.OQImatrixCollector()
    collector.install(model)
    try:
        for i in range(2):
            mx.eval(model(tokens[i:i+1]))
    finally:
        collector.restore(model)
    assert captured.keys() == collector.entries.keys()
    for name, entry in captured.items():
        np.testing.assert_array_equal(entry.counts, collector.entries[name].counts)
        np.testing.assert_allclose(entry.in_sum2, collector.entries[name].in_sum2, rtol=1e-6, atol=1e-6)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(args())))
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    data = oq.OQImatrixData(captured, {"collection": metadata}, "synthetic", False)
    output = tmp_path / "converted"
    with (patch.object(oq, "_load_or_collect_imatrix", lambda *a, **kw: data),
          patch.object(oq, "_build_model_sanitizer", lambda *a, **kw: lambda w: w),
          patch.object(oq, "_MAX_SHARD_BYTES", 100_000)):
        oq.quantize_oq_streaming(str(source), str(output), 4, enhanced=True,
            imatrix_strict=True, sensitivity_map_override=metadata["sensitivity_map"],
            stream_calibration=False, auto_proxy_sensitivity=False)
    report = json.loads((output / "oq_imatrix_report.json").read_text())
    assert not report["missing"] and not report["mismatched"]
    assert len(report["applied"]) == 41
    emitted_config = json.loads((output / "config.json").read_text())
    index = json.loads((output / "model.safetensors.index.json").read_text())
    validate_headers(output, index["weight_map"], tensor_schema(emitted_config))
    loaded = Model(args(), nvfp4=False)
    specs = affine_specs(emitted_config)
    nn.quantize(loaded, class_predicate=lambda name, module: specs.get(name, False))
    artifact = {"identity": {"path": str(output)}, "weight_map": index["weight_map"]}
    loaded.load_weights(list(source_weights(artifact)), strict=True)
    logits = loaded(tokens)
    assert np.isfinite(np.asarray(logits.astype(mx.float32))).all()
