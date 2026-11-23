"""``moe_top_k``: route each token to its best k experts (opt-in, approximate) (CPU)."""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import TOKENS, build, checkpoint, tiny_config  # noqa: E402

from mlx2.adapters import hy_v4_flash as A  # noqa: E402


@pytest.fixture()
def model():
    config = tiny_config()
    m, _ = build(config, checkpoint(config))
    return m


def _routers(model):
    return [layer.mlp.gate for layer in model.model.layers if hasattr(getattr(layer, "mlp", None), "gate")]


def test_default_leaves_the_checkpoint_k(model):
    native = _routers(model)[0].top_k
    receipt = A.set_moe_top_k(model, None)
    assert receipt == {"selected": False, "k": native, "native_k": native, "approximate": False}


def test_native_k_is_exact_and_fewer_experts_changes_the_output(model):
    ids = mx.array([TOKENS[:12]])
    native = _routers(model)[0].top_k
    base = np.array(model(ids))
    assert A.set_moe_top_k(model, native)["approximate"] is False
    np.testing.assert_array_equal(np.array(model(ids)), base)
    receipt = A.set_moe_top_k(model, native - 1)
    assert receipt["selected"] and receipt["k"] == native - 1 and receipt["approximate"]
    assert all(r.top_k == native - 1 for r in _routers(model))
    assert not np.array_equal(np.array(model(ids)), base)
    indices, weights = _routers(model)[0](mx.zeros((1, 3, model.args.hidden_size)) + 0.1)
    assert indices.shape[-1] == native - 1


def test_refusals_and_namespace(model):
    native = _routers(model)[0].top_k
    with pytest.raises(ValueError, match="exceeds"):
        A.set_moe_top_k(model, native + 1)
    for bad in (0, -1, 2.0, "8"):
        with pytest.raises(ValueError, match="moe_top_k"):
            A._parse_policy({"moe_top_k": bad})
    A._parse_policy({"moe_top_k": 8})
    assert A.numerics_layout("x", moe_top_k=8) == "x+moe-top8"
    assert A.numerics_layout("x") == "x"
    assert "moe_top_k" not in A.HYV4FlashAdapter.route_neutral_policy_keys
