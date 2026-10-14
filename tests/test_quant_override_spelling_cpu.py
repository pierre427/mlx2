"""A per-module quantization override missed by its spelling fails the load.

TensorFold#755: oQ checkpoints key per-module widths as
``language_model.model.layers.N...`` while a loader asked for
``model.layers.N...``, so every override was missed and the tensor was read
at the global width.  mlx2 adapters look overrides up by the exact runtime
module name and then load strictly (tests/test_qwen36_35b_port.py pins
``strict=True``); this pins why that is enough: a missed override builds the
module at the global width, and the strict load refuses the stored shapes
instead of reading them at the wrong width.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten


class _Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(128, 64, bias=False)]


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = _Text()


def _stored(bits):
    """Checkpoint tensors for the one Linear, quantized at ``bits``/g64."""
    reference = nn.QuantizedLinear.from_linear(
        nn.Linear(128, 64, bias=False), group_size=64, bits=bits
    )
    flat = dict(tree_flatten(reference.parameters()))
    return reference, {f"language_model.model.layers.0.{k}": v for k, v in flat.items()}


def _adapter_load(quant, weights):
    """The adapters' predicate (exact runtime name, then 'has scales'), strict load."""
    model = _Wrapper()

    def predicate(name, module):
        if name in quant:
            return quant[name]
        return hasattr(module, "to_quantized") and f"{name}.scales" in weights

    nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"], class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    return model


@pytest.mark.parametrize(
    "spelling", ["model.layers.0", "layers.0", "model.language_model.layers.0"]
)
def test_missed_override_spelling_refuses_the_load(spelling):
    _, weights = _stored(bits=8)
    quant = {"group_size": 64, "bits": 4, spelling: {"group_size": 64, "bits": 8}}
    with pytest.raises(ValueError, match="shape"):
        _adapter_load(quant, weights)


def test_runtime_spelling_loads_at_the_override_width():
    reference, weights = _stored(bits=8)
    quant = {
        "group_size": 64,
        "bits": 4,
        "language_model.model.layers.0": {"group_size": 64, "bits": 8},
    }
    model = _adapter_load(quant, weights)
    x = mx.random.normal((2, 128), key=mx.random.key(0))
    assert mx.array_equal(model.language_model.model.layers[0](x), reference(x)).item()
