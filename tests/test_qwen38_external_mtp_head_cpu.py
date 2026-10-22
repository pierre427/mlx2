"""External Qwen3.8 27B MTP head directories on tiny CPU models.

A separately shipped head (``model_type: qwen3_5_mtp`` + model.safetensors,
or a bare ``mtp.safetensors``) is bound to a target before any tensor loads:
geometry, exact schema, quantization derived from packed shapes, degenerate
payloads and targets that already ship a head are refused.  Loaded through
the adapter's own weight path it equals the embedded head bit for bit, and
native-MTP drafting from it reproduces ordinary greedy output.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)

from test_qwen38_dflash2_cpu import drain, ordinary_greedy

from mlx2.adapters import qwen38_27b
from mlx2.adapters.qwen38_external_mtp_head import (
    attach_mtp_head,
    head_quantization_predicate,
    inspect_mtp_head,
    is_mtp_head_artifact,
)

VOCAB = 128
TEXT = {
    "model_type": "qwen3_5", "hidden_size": 32, "intermediate_size": 64,
    "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 8, "vocab_size": VOCAB, "linear_num_key_heads": 2,
    "linear_num_value_heads": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
    "mtp_num_hidden_layers": 1, "partial_rotary_factor": 0.5,
    "rope_parameters": None, "max_position_embeddings": 512,
    "attn_output_gate": True, "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
}


def tiny_model(seed=3, mtp=1):
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

    mx.random.seed(seed)
    model = Model(ModelArgs(model_type="qwen3_5", text_config={**TEXT, "mtp_num_hidden_layers": mtp}))
    model.eval()
    mx.eval(model.parameters())
    return model


def split_weights(model):
    flat = dict(tree_flatten(model.parameters()))
    head = {k.removeprefix("language_model.mtp."): v for k, v in flat.items()
            if k.startswith("language_model.mtp.")}
    trunk = {k: v for k, v in flat.items() if not k.startswith("language_model.mtp.")}
    return trunk, head


def target_artifact(*, has_mtp=False, text=None):
    config = {"model_type": "qwen3_5", "text_config": dict(text or TEXT)}
    return {"config": config, "has_mtp": has_mtp,
            "identity": {"path": "/t", "fingerprint": "f" * 64, "files": []}}


def write_head(path: Path, head, *, config=True, name="model.safetensors", text=None, **overrides):
    path.mkdir(parents=True, exist_ok=True)
    if config:
        head_text = {**(text or TEXT), "model_type": "qwen3_5_text", **overrides}
        (path / "config.json").write_text(json.dumps(
            {"model_type": "qwen3_5_mtp", "block_size": 3, "text_config": head_text}))
    mx.save_safetensors(str(path / name), head, metadata={"format": "mlx"})
    return path


# -- inspection and refusals ----------------------------------------------------


def test_head_binds_and_identity_names_the_pair(tmp_path):
    _, head = split_weights(tiny_model())
    path = write_head(tmp_path / "head", head)
    assert is_mtp_head_artifact(path)
    record = inspect_mtp_head(path, target_artifact())
    assert record["quantization"] == {} and record["declared_geometry"]
    bound = attach_mtp_head(target_artifact(), record)
    assert bound["has_mtp"] and bound["config"]["text_config"]["mtp_num_hidden_layers"] == 1
    identity = bound["identity"]
    assert identity["target_fingerprint"] == "f" * 64
    assert identity["fingerprint"] != "f" * 64 and identity["mtp_head"]["fingerprint"] == record["fingerprint"]
    assert bound["mtp_sidecar"]["external"] and Path(bound["mtp_sidecar"]["file"]).is_absolute()
    # A different head payload is a different pair identity.
    other = write_head(tmp_path / "other", {**head, "norm.weight": head["norm.weight"] + 1})
    assert attach_mtp_head(target_artifact(), inspect_mtp_head(other, target_artifact()))[
        "identity"]["fingerprint"] != identity["fingerprint"]


@pytest.mark.parametrize("key, value, match", [
    ("hidden_size", 64, "hidden_size"),
    ("vocab_size", VOCAB + 1, "vocab_size"),
    ("num_hidden_layers", 12, "num_hidden_layers"),
    ("mtp_num_hidden_layers", 2, "exactly one MTP layer"),
    ("attn_output_gate", False, "attn_output_gate"),
])
def test_declared_geometry_mismatch_is_refused(tmp_path, key, value, match):
    _, head = split_weights(tiny_model())
    path = write_head(tmp_path / "head", head, **{key: value})
    with pytest.raises(ValueError, match=match):
        inspect_mtp_head(path, target_artifact())


def test_schema_degenerate_payload_and_existing_head_are_refused(tmp_path):
    _, head = split_weights(tiny_model())
    with pytest.raises(ValueError, match="outside the single-layer schema"):
        inspect_mtp_head(write_head(tmp_path / "extra", {**head, "layers.1.mlp.up_proj.weight":
                                                         head["layers.0.mlp.up_proj.weight"]}),
                         target_artifact())
    with pytest.raises(ValueError, match="q_proj shape"):
        inspect_mtp_head(write_head(tmp_path / "shape", {
            **head, "layers.0.self_attn.q_proj.weight": mx.zeros((16, 32))}), target_artifact())
    zeros = {k: mx.zeros_like(v) for k, v in head.items()}
    with pytest.raises(ValueError, match="all zeros"):
        inspect_mtp_head(write_head(tmp_path / "zeros", zeros), target_artifact())
    with pytest.raises(ValueError, match="ships its own MTP head"):
        inspect_mtp_head(write_head(tmp_path / "ok", head), target_artifact(has_mtp=True))


def test_quantization_is_derived_from_packed_shapes_and_must_match_config(tmp_path):
    model = tiny_model()
    nn.quantize(model.language_model.mtp.layers[0], group_size=32, bits=4)
    _, head = split_weights(model)
    path = write_head(tmp_path / "q", head, config=False, name="mtp.safetensors")
    assert is_mtp_head_artifact(path)
    record = inspect_mtp_head(path, target_artifact())
    assert not record["declared_geometry"]
    assert record["quantization"]["layers.0.mlp.down_proj"] == {"bits": 4, "group_size": 32, "mode": "affine"}
    assert "fc" not in record["quantization"]          # fc stayed float
    mismatch = write_head(tmp_path / "declared", head)
    config = json.loads((mismatch / "config.json").read_text())
    config["quantization"] = {"group_size": 32, "bits": 8}
    (mismatch / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="quantization"):
        inspect_mtp_head(mismatch, target_artifact())
    decide = head_quantization_predicate(record["quantization"])
    assert decide("language_model.mtp.layers.0.mlp.up_proj") == {"bits": 4, "group_size": 32, "mode": "affine"}
    assert decide("language_model.mtp.fc") is False
    assert decide("language_model.model.layers.0.mlp.up_proj") is None


def test_adapter_policy_refusals_before_any_load(tmp_path):
    adapter = object.__new__(qwen38_27b.Qwen3827BAdapter)
    with pytest.raises(ValueError, match="external or prompt-lookup drafter"):
        adapter._init_qwen38("/unused", execution_policy={"mtp_head": str(tmp_path), "draft_model": "pld"})
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter

    other = object.__new__(Qwen3627BAdapter)
    with pytest.raises(ValueError, match="does not attach external MTP heads"):
        other._init_qwen38("/unused", execution_policy={"mtp_head": str(tmp_path)})
    assert not Qwen3627BAdapter.is_external_mtp_head(tmp_path)


# -- loading and greedy equality ------------------------------------------------


def _load_with_external_head(tmp_path, monkeypatch, reference, *, quantize_head=False):
    trunk, head = split_weights(reference)
    mx.save_safetensors(str(tmp_path / "trunk.safetensors"), trunk)
    path = write_head(tmp_path / "head", head, config=not quantize_head,
                      name="mtp.safetensors" if quantize_head else "model.safetensors")
    record = inspect_mtp_head(path, target_artifact())
    bound = attach_mtp_head(target_artifact(), record)
    monkeypatch.setattr("mlx2.runtime.ubc_evict.load_shards_evicting",
                        lambda files, **_: dict(mx.load(str(files[0]))))
    model = tiny_model(seed=99)  # different init: every tensor must come from disk
    sidecar = bound["mtp_sidecar"]
    weights = qwen38_27b._load_target_weights(
        [tmp_path / "trunk.safetensors"], external_draft=False, sanitize=model.sanitize,
        mtp_sidecar={**sidecar, "path": Path("/ignored") / sidecar["file"]},
    )
    qwen38_27b._quantize_target(model, None, weights, sidecar["quantization"])
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def test_external_head_loads_bit_identical_to_the_embedded_head(tmp_path, monkeypatch):
    reference = tiny_model()
    model = _load_with_external_head(tmp_path, monkeypatch, reference)
    ours, theirs = dict(tree_flatten(model.parameters())), dict(tree_flatten(reference.parameters()))
    assert ours.keys() == theirs.keys()
    for key, value in ours.items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(theirs[key]), err_msg=key)


def _native_mtp_greedy(model, prompt, count, num_draft=3):
    from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    model.select_external_feature_source("mtp_seed")
    draft = NativeMTPDraftModel(model, num_draft=num_draft, revision="a" * 64)
    batch = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test",
                                        num_draft=num_draft, prefill_step_size=4)
    uid = batch.insert([prompt], max_tokens=[count], sampling_configs=[{"sampling_temp": 0}])[0]
    got, _final = drain(batch)
    return got[uid], batch.scheduler_stats


@pytest.mark.parametrize("quantize_head", [False, True])
def test_native_mtp_from_external_head_equals_ordinary_greedy(tmp_path, monkeypatch, quantize_head):
    reference = tiny_model()
    if quantize_head:
        nn.quantize(reference.language_model.mtp.layers[0], group_size=32, bits=4)
        mx.eval(reference.parameters())
    model = _load_with_external_head(tmp_path, monkeypatch, reference, quantize_head=quantize_head)
    if quantize_head:
        layer = model.language_model.mtp.layers[0]
        assert isinstance(layer.mlp.down_proj, nn.QuantizedLinear) and layer.mlp.down_proj.bits == 4
        assert not isinstance(model.language_model.mtp.fc, nn.QuantizedLinear)
    prompt = [3, 9, 27, 81, 15, 44, 2, 7, 7, 7]
    got, stats = _native_mtp_greedy(model, prompt, 16)
    assert got == ordinary_greedy(model, prompt, 16)
    assert stats["proposed_tokens"] > 0
