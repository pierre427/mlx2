from types import SimpleNamespace

import pytest
from mlx import nn

from mlx2.adapters.nemotron35_draft_quantization import (
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
