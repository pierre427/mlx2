from types import SimpleNamespace

import pytest
from mlx import nn

from mlx2.adapters.nemotron35_draft_quantization import (
    AFFINE_Q4_G64,
    AFFINE_Q8_G64,
    apply_runtime_quantization,
    normalize_runtime_quantization,
)


def test_runtime_quantization_value_is_closed():
    assert normalize_runtime_quantization(None) is None
    assert normalize_runtime_quantization(AFFINE_Q8_G64) == AFFINE_Q8_G64
    with pytest.raises(ValueError, match="affine_q8_g64"):
        normalize_runtime_quantization("q4")


def test_runtime_q8_selects_only_group64_compatible_matrices(monkeypatch):
    selected = []

    class Module:
        def __init__(self, shape, quantizable=True):
            self.weight = SimpleNamespace(ndim=2, shape=shape)
            if quantizable:
                self.to_quantized = object()

    modules = {
        "large": Module((128, 4096)),
        "small_input": Module((1024, 5)),
        "plain": Module((128, 4096), quantizable=False),
    }

    def quantize(model, *, group_size, bits, mode, class_predicate):
        assert model is draft
        assert (group_size, bits, mode) == (64, 8, "affine")
        selected.extend(
            path for path, module in modules.items() if class_predicate(path, module)
        )

    monkeypatch.setattr(nn, "quantize", quantize)
    draft = SimpleNamespace()
    receipt = apply_runtime_quantization(draft, AFFINE_Q8_G64)

    assert selected == ["large"]
    assert receipt["modules"] == ["large"]
    assert receipt["module_count"] == 1
    assert receipt["source_artifact_unchanged"] is True
    assert draft.runtime_weight_quantization == receipt


def test_runtime_q4_profile_is_accepted_and_bound_to_four_bits(monkeypatch):
    assert normalize_runtime_quantization(AFFINE_Q4_G64) == AFFINE_Q4_G64
    for rejected in ("affine_q4_g32", "q4", "affine_q4"):
        with pytest.raises(ValueError, match="affine_q4_g64"):
            normalize_runtime_quantization(rejected)

    seen = []

    def quantize(model, *, group_size, bits, mode, class_predicate):
        seen.append((group_size, bits, mode))

    monkeypatch.setattr(nn, "quantize", quantize)
    receipt = apply_runtime_quantization(SimpleNamespace(), AFFINE_Q4_G64)
    assert seen == [(64, 4, "affine")]
    assert receipt["kind"] == AFFINE_Q4_G64
    assert (receipt["mode"], receipt["bits"], receipt["group_size"]) == (
        "affine", 4, 64,
    )
    assert receipt["qualified"] is False


class _TinyDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(96, 128)
        self.proj = nn.Linear(128, 192, bias=False)
        self.small = nn.Linear(5, 64, bias=False)
        self.lm_head = nn.Linear(192, 96, bias=False)


def _bf16_draft():
    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.random.seed(4064)
    draft = _TinyDraft()
    draft.load_weights(
        [(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(draft.parameters())],
        strict=True,
    )
    return draft


def test_runtime_q4_converts_real_modules_deterministically_and_keeps_small_bf16():
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    first, second = _bf16_draft(), _bf16_draft()
    receipt = apply_runtime_quantization(first, AFFINE_Q4_G64)
    assert apply_runtime_quantization(second, AFFINE_Q4_G64) == receipt
    assert receipt["modules"] == ["embed", "lm_head", "proj"]
    assert receipt["module_count"] == 3
    for name in ("proj", "lm_head"):
        module = getattr(first, name)
        assert isinstance(module, nn.QuantizedLinear)
        assert (module.bits, module.group_size, module.mode) == (4, 64, "affine")
    assert isinstance(first.embed, nn.QuantizedEmbedding)
    assert (first.embed.bits, first.embed.group_size) == (4, 64)
    assert isinstance(first.small, nn.Linear)
    assert not isinstance(first.small, nn.QuantizedLinear)
    assert first.small.weight.dtype == mx.bfloat16
    left, right = dict(tree_flatten(first.parameters())), dict(
        tree_flatten(second.parameters())
    )
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key], right[key]
        if a.dtype == mx.bfloat16:
            a, b = a.view(mx.uint16), b.view(mx.uint16)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # 4-bit packing: 8 values per uint32 word.
    assert first.proj.weight.dtype == mx.uint32
    assert tuple(first.proj.weight.shape) == (192, 128 // 8)


def test_runtime_q4_differs_from_q8_receipt():
    q8 = apply_runtime_quantization(_bf16_draft(), AFFINE_Q8_G64)
    q4 = apply_runtime_quantization(_bf16_draft(), AFFINE_Q4_G64)
    assert q8["modules"] == q4["modules"]
    assert (q8["kind"], q8["bits"]) == (AFFINE_Q8_G64, 8)
    assert (q4["kind"], q4["bits"]) == (AFFINE_Q4_G64, 4)
