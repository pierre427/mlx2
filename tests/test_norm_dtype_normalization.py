"""Float32 RMSNorm gammas must not promote the Qwen3.5/3.6/3.8 compute stream.

The Qwen3.8-27B CRACK conversions store 305 norm/GDN tensors as float32 where
the official MLX conversions use bf16. ``mx.fast.rms_norm(bf16, f32)`` returns
float32, so without the targeted cast in the Qwen3.8 ``sanitize`` the residual
stream and the KV cache run in float32 (1.66x slower quantized, ~4x on bf16).

Everything here runs on the MLX CPU stream with tiny topologies.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx2.runtime.models.dtype_normalize import (
    check_compute_dtype,
    normalize_norm_dtypes,
    resolve_compute_dtype,
)

# The seven tensor groups the CRACK artifacts store as float32.
CRACK_FP32_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".linear_attn.norm.weight",
    ".linear_attn.A_log",
    ".linear_attn.dt_bias",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    "model.norm.weight",
)
# Only these may be cast; GDN tensors stay float32 (they do not promote).
PROMOTING = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "model.norm.weight",
)
GDN_FP32 = (".linear_attn.A_log", ".linear_attn.dt_bias", ".linear_attn.norm.weight")

TEXT = dict(
    model_type="qwen3_5_text",
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=128,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=3,
    full_attention_interval=4,
    mtp_num_hidden_layers=0,
    partial_rotary_factor=0.5,
    rope_parameters=None,
    max_position_embeddings=128,
    dtype="bfloat16",
)
MOE = dict(
    TEXT,
    model_type="qwen3_5_moe_text",
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
)
PROMPT = [3, 17, 42, 9, 101, 5]


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


def _module(moe: bool):
    if moe:
        from mlx2.runtime.models.qwen36_35b import Model, ModelArgs

        return Model, ModelArgs, {"model_type": "qwen3_5_moe", "text_config": MOE}
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

    return Model, ModelArgs, {"model_type": "qwen3_5", "text_config": TEXT}


def _artifact(moe: bool, fp32_suffixes=CRACK_FP32_SUFFIXES):
    """bf16 weights with the CRACK float32 groups, as a converted artifact."""
    Model, ModelArgs, config = _module(moe)
    mx.random.seed(11)
    source = Model(ModelArgs.from_dict(config))
    if moe:
        # MLX's CPU GatherMM is float32-only; a quantized artifact (as the
        # 35B-A3B ships) runs bf16 through gather_qmm on the CPU.
        nn.quantize(source, group_size=32, bits=4)
    weights = {}
    for key, value in tree_flatten(source.parameters()):
        if key.endswith(("norm.weight", "layernorm.weight")):
            # Converted-layout gammas near 1 with a visible spread.
            value = 1.0 + 0.3 * mx.random.normal(value.shape)
        if value.dtype == mx.uint32:
            weights[key] = value
            continue
        wide = key.endswith(fp32_suffixes)
        weights[key] = value.astype(mx.float32 if wide else mx.bfloat16)
    mx.eval(weights)
    return weights


def _load(moe: bool, weights, *, sanitize=True):
    Model, ModelArgs, config = _module(moe)
    model = Model(ModelArgs.from_dict(config))
    loaded = model.sanitize(dict(weights)) if sanitize else dict(weights)
    if moe:
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda name, module: hasattr(module, "to_quantized")
            and f"{name}.scales" in loaded,
        )
    model.load_weights(list(loaded.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def _trunk_dtypes(model):
    text = model.language_model
    cache = text.make_cache()
    hidden = text.model(mx.array([PROMPT]), cache=cache)
    kv = {c.keys.dtype for c in cache if getattr(c, "keys", None) is not None}
    return hidden.dtype, kv


def _greedy(model, steps=12):
    cache = model.make_cache()
    logits = model(mx.array([PROMPT]), cache=cache)
    out = []
    for _ in range(steps):
        token = int(mx.argmax(logits[0, -1]).item())
        out.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return out, logits


@pytest.mark.parametrize("moe", [False, True], ids=["qwen38-dense", "qwen36-moe"])
def test_unpatched_float32_norms_promote_the_stream(moe):
    """The defect the cast fixes: loading without it runs in float32."""
    model = _load(moe, _artifact(moe), sanitize=False)
    hidden, kv = _trunk_dtypes(model)
    assert hidden == mx.float32
    assert kv == {mx.float32}
    check = check_compute_dtype(model.language_model, mx.bfloat16)
    assert check["ok"] is False
    assert check["observed"]["hidden"] == "float32"


@pytest.mark.parametrize("moe", [False, True], ids=["qwen38-dense", "qwen36-moe"])
def test_sanitize_casts_promoting_norms_and_stream_stays_bf16(moe):
    weights = _artifact(moe)
    model = _load(moe, weights)
    hidden, kv = _trunk_dtypes(model)
    assert hidden == mx.bfloat16
    assert kv == {mx.bfloat16}

    text = model.language_model
    promoting = [k for k in weights if k.endswith(PROMOTING)]
    # 4 input + 4 post-attention + 1 q_norm + 1 k_norm + final norm.
    assert len(promoting) == 11
    assert text.dtype_normalization == {
        "cast": 11,
        "from": "float32",
        "to": "bfloat16",
    }
    params = dict(tree_flatten(model.parameters()))
    for key, value in params.items():
        if key.endswith(PROMOTING):
            assert value.dtype == mx.bfloat16, key
        if key.endswith(GDN_FP32):
            # Left as stored: none of these promotes the stream.
            assert value.dtype == mx.float32, key
    assert check_compute_dtype(text, text.compute_dtype) == {
        "ok": True,
        "expected": "bfloat16",
        "observed": {"hidden": "bfloat16", "kv": ["bfloat16"]},
    }


def test_gdn_float32_tensors_alone_do_not_promote():
    """Evidence for leaving A_log, dt_bias and linear_attn.norm float32."""
    weights = _artifact(False, fp32_suffixes=GDN_FP32)
    model = _load(False, weights, sanitize=False)
    hidden, kv = _trunk_dtypes(model)
    assert hidden == mx.bfloat16
    assert kv == {mx.bfloat16}


@pytest.mark.parametrize("moe", [False, True], ids=["qwen38-dense", "qwen36-moe"])
def test_greedy_matches_the_same_weights_stored_bf16(moe):
    """Casting at load equals a converter that stored the norms as bf16.

    The cast rounds each float32 gamma to bf16 exactly as a bf16 conversion
    would, so logits are bit-identical and greedy output is token-for-token.
    """
    fp32_model = _load(moe, _artifact(moe))
    bf16_weights = {
        k: (v.astype(mx.bfloat16) if k.endswith(PROMOTING) else v)
        for k, v in _artifact(moe).items()
    }
    bf16_model = _load(moe, bf16_weights)
    assert bf16_model.language_model.dtype_normalization["cast"] == 0
    tokens_a, logits_a = _greedy(fp32_model)
    tokens_b, logits_b = _greedy(bf16_model)
    assert tokens_a == tokens_b
    assert mx.array_equal(logits_a, logits_b).item()


def test_raw_layout_fold_runs_in_float32_before_the_cast():
    """Raw HF gammas: +1 is added in float32, then rounded once to bf16."""
    weights = _artifact(False)
    raw = {}
    for key, value in weights.items():
        if key.endswith("conv1d.weight"):
            value = value.moveaxis(1, 2)  # HF layout: last dim is the kernel
        elif key.endswith(PROMOTING):
            value = value - 1.0  # raw convention stores w for 1 + w
        raw[key] = value
    model = _load(False, raw)
    params = dict(tree_flatten(model.parameters()))
    for key in weights:
        if key.endswith(PROMOTING):
            expected = (raw[key] + 1.0).astype(mx.bfloat16)
            assert params[key].dtype == mx.bfloat16, key
            assert mx.array_equal(params[key], expected).item(), key


def test_float32_compute_dtype_and_unknown_dtype_are_left_alone():
    weights = {"model.layers.0.input_layernorm.weight": mx.ones(4)}
    assert normalize_norm_dtypes(dict(weights), PROMOTING, mx.float32) is None
    assert normalize_norm_dtypes(dict(weights), PROMOTING, None) is None
    assert resolve_compute_dtype(None, {}) is None
    assert resolve_compute_dtype("float16") == mx.float16
    inferred = resolve_compute_dtype(
        None, {"model.embed_tokens.scales": mx.ones(2, dtype=mx.bfloat16)}
    )
    assert inferred == mx.bfloat16


def test_intentional_float32_tensors_outside_the_suffixes_are_untouched():
    """Routing biases and hyper-connection scales keep float32."""
    weights = {
        "model.layers.2.mlp.gate.e_score_correction_bias": mx.zeros(8),
        "model.layers.0.attn_hc.hc_scale": mx.ones(3),
        "model.layers.0.ffn_hc.hc_base": mx.ones(3),
        "model.layers.0.input_layernorm.weight": mx.ones(4),
    }
    receipt = normalize_norm_dtypes(weights, PROMOTING, mx.bfloat16)
    assert receipt == {"cast": 1, "from": "float32", "to": "bfloat16"}
    assert weights["model.layers.0.input_layernorm.weight"].dtype == mx.bfloat16
    for key in (
        "model.layers.2.mlp.gate.e_score_correction_bias",
        "model.layers.0.attn_hc.hc_scale",
        "model.layers.0.ffn_hc.hc_base",
    ):
        assert weights[key].dtype == mx.float32, key


def test_xing_sanitize_keeps_its_float32_tensors():
    """Xing4.0 does not share the Qwen sanitize: nothing float32 is narrowed."""
    import json
    from pathlib import Path

    from mlx2.runtime.models.xing4_0 import Model, ModelArgs

    fixture = Path(__file__).parent / "fixtures" / "xing4_0_tiny"
    config = json.loads((fixture / "config.json").read_text())
    config["dtype"] = "bfloat16"
    raw = mx.load(str(fixture / "weights.safetensors"))
    model = Model(ModelArgs.from_dict(config))
    sanitized = model.sanitize(dict(raw))
    intentional = [
        k
        for k in sanitized
        if any(s in k for s in ("hc_fn", "hc_base", "hc_scale", "e_score_correction"))
    ]
    assert intentional
    for key, value in sanitized.items():
        if key in raw and raw[key].dtype == mx.float32:
            assert value.dtype == mx.float32, key


def test_mismatch_is_logged_as_a_warning(caplog):
    model = _load(False, _artifact(False), sanitize=False)
    with caplog.at_level(logging.WARNING):
        check = check_compute_dtype(model.language_model, mx.bfloat16)
    assert check["ok"] is False
    assert "compute dtype promoted at load" in caplog.text


def test_adapter_records_the_receipt_and_load_check():
    """The receipt and the load-time check surface in adapter diagnostics."""
    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    for cls in (Qwen3827BAdapter, Qwen359BAdapter):
        adapter = object.__new__(cls)
        adapter.model = _load(False, _artifact(False))
        adapter._record_load_dtype()
        assert adapter.dtype_diagnostics() == {
            "dtype_normalized": {"cast": 11, "from": "float32", "to": "bfloat16"},
            "load_dtype_check": {
                "ok": True,
                "expected": "bfloat16",
                "observed": {"hidden": "bfloat16", "kv": ["bfloat16"]},
            },
        }
