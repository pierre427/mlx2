import types

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.fp32_head import enable_fp32_head_logits

# Tiny shapes; keep them off the shared GPU.
mx.set_default_device(mx.cpu)


def _language_model(*, quantized=True, tied=False, vocab=256, hidden=128):
    mx.random.seed(0)
    head = nn.Linear(hidden, vocab, bias=False)
    head.set_dtype(mx.bfloat16)
    if quantized:
        head = nn.QuantizedLinear.from_linear(head, group_size=64, bits=4)
    lm = nn.Module()
    lm.args = types.SimpleNamespace(tie_word_embeddings=tied)
    lm.lm_head = head
    return lm


def test_fp32_head_writes_fp32_sums_of_the_same_weights():
    lm = _language_model()
    head = lm.lm_head
    x = (mx.random.normal((2, 3, 128)) * 4).astype(mx.bfloat16)
    before = head(x)
    names = [name for name, _ in tree_flatten(head.parameters())]
    weight = head["weight"]
    receipt = enable_fp32_head_logits(lm)
    after = lm.lm_head(x)
    assert before.dtype == mx.bfloat16
    assert after.dtype == mx.float32
    assert lm.lm_head is head
    assert [name for name, _ in tree_flatten(head.parameters())] == names
    assert head["weight"] is weight
    # Exact reference: fp32 hidden times the dequantized head.
    w = mx.dequantize(
        head["weight"], head["scales"], head["biases"], group_size=64, bits=4
    ).astype(mx.float32)
    reference = x.astype(mx.float32) @ w.T
    assert float(mx.max(mx.abs(after - reference))) < 1e-3
    assert receipt["enabled"] and receipt["bits"] == 4
    # bf16 -> fp32 scales and biases of the same shape: one fp32 scales' worth.
    assert receipt["extra_resident_bytes"] == head["scales"].nbytes


def test_fp32_head_is_idempotent():
    lm = _language_model()
    enable_fp32_head_logits(lm)
    again = enable_fp32_head_logits(lm)
    assert again["extra_resident_bytes"] == 0
    assert lm.lm_head["scales"].dtype == mx.float32


@pytest.mark.parametrize(
    "kwargs, message",
    [({"tied": True}, "untied"), ({"quantized": False}, "quantized")],
)
def test_fp32_head_fails_closed(kwargs, message):
    lm = _language_model(**kwargs)
    with pytest.raises(ValueError, match=message):
        enable_fp32_head_logits(lm)


def test_flash_next_policy_key_is_opt_in_and_hidden_when_off():
    assert "fp32_head_logits" not in FlashNextPolicy().as_dict()
    assert FlashNextPolicy.from_mapping({"fp32_head_logits": True}).as_dict()[
        "fp32_head_logits"
    ] is True
    with pytest.raises(ValueError, match="boolean"):
        FlashNextPolicy.from_mapping({"fp32_head_logits": 1})


def test_fp32_head_changes_serving_execution_identity():
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    ordinary = FlashNextPolicy().batch_config(max_lanes=4, prefill_step=2048)
    fp32 = FlashNextPolicy(fp32_head_logits=True).batch_config(
        max_lanes=4, prefill_step=2048
    )
    assert "fp32_head_logits" not in ordinary
    assert fp32 == {**ordinary, "fp32_head_logits": True}

    adapter = object.__new__(Qwen3827BAdapter)
    adapter.descriptor = types.SimpleNamespace(capabilities=frozenset())
    adapter.fp32_head = None
    ordinary = adapter.execution_config(max_lanes=4, prefill_step=2048)
    adapter.fp32_head = {"enabled": True}
    assert adapter.execution_config(max_lanes=4, prefill_step=2048) == {
        **ordinary,
        "fp32_head_logits": True,
    }


def test_qwen38_27b_policy_validates_before_loading():
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    with pytest.raises(ValueError, match="boolean"):
        Qwen3827BAdapter("/nonexistent", execution_policy={"fp32_head_logits": "yes"})
    with pytest.raises(ValueError, match="fp32_head_logits"):
        Qwen3827BAdapter("/nonexistent", execution_policy={"bogus": True})


@pytest.mark.parametrize("mode", ["mxfp4", "mxfp8", "nvfp4"])
def test_non_affine_heads_are_refused_before_they_are_touched(mode):
    """Their uint8 scale codes were cast to float32, and every later forward
    raised; the promise is to fail closed before changing the head."""
    from types import SimpleNamespace

    from mlx2.runtime.fp32_head import enable_fp32_head_logits

    mx.set_default_device(mx.cpu)
    group = {"mxfp4": 32, "mxfp8": 32, "nvfp4": 16}[mode]
    bits = {"mxfp4": 4, "mxfp8": 8, "nvfp4": 4}[mode]
    head = nn.QuantizedLinear(64, 32, bias=False, group_size=group, bits=bits, mode=mode)
    model = SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=False), lm_head=head)
    scales = head["scales"]
    with pytest.raises(ValueError, match="affine"):
        enable_fp32_head_logits(model)
    assert head["scales"].dtype == scales.dtype
