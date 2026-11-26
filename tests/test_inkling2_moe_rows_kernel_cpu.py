"""``inkling2_moe_rows_kernel``, CPU only.

The Metal kernels run on the GPU only (scripts/research/inkling2_moe_rows_kernel_ab.py
checks them on the real checkpoint).  Here: the option defaults off and is an
alternative to the panel kernel; off never asks the kernels; on, the CPU
declines every call (counted) and stays the shipped route bitwise; with the
device gate and the launch patched to the kernels' law (each row through the
one-token kernel's law), the plumbing engages on 2-16 row calls of the NVFP4
layers, declines one-token calls, the MXFP8 layers, more than 16 rows and
calls the shipped route would stream or combine through the weighted sum
(counted), and the logits stay bitwise.
"""

import mlx.core as mx
import pytest
from test_inkling2_cpu import _run, _tiny_model
from test_inkling2_nonbitwise_cpu import _quantized_model

from mlx2.runtime.models.inkling2 import moe_rows_kernel as mrk

NVFP4_VISITS = 6 + 12 * 3 + 2  # layers 0-5, the looped 6-17 three times, 18-19
MXFP8_VISITS = 4  # layers 20-23


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _keys(receipt):
    keys = {"declined:" + k: v for k, v in receipt["moe_rows_kernel_declines"].items()}
    if receipt["moe_rows_kernel_calls"]:
        keys["rows_kernel"] = receipt["moe_rows_kernel_calls"]
    return keys


def _law_on_cpu(monkeypatch):
    from mlx2.runtime.models.inkling2 import moe_decode_kernel as mdk

    monkeypatch.setattr(mrk, "device_ok", lambda: True)
    monkeypatch.setattr(mdk, "device_ok", lambda: True)
    monkeypatch.setattr(mrk, "launch", lambda moe, experts, x, order, scores, **_: mrk.law(
        moe, experts, x, order, scores))


def test_defaults_off_and_validation():
    from mlx2.adapters.inkling2 import KERNEL_DEFAULTS, validate_policy

    assert validate_policy(None)[0]["moe_rows_kernel"] is False
    assert "moe_rows_kernel" not in KERNEL_DEFAULTS
    assert validate_policy({"inkling2_moe_rows_kernel": True})[0]["moe_rows_kernel"] is True
    with pytest.raises(ValueError, match="moe_rows_kernel requires inkling2_moe_fused"):
        validate_policy({"inkling2_moe_rows_kernel": True, "inkling2_moe_fused": False})
    with pytest.raises(ValueError, match="alternatives"):
        validate_policy({"inkling2_moe_rows_kernel": True, "inkling2_moe_panel_kernel": True})
    with pytest.raises(ValueError, match="logdomain_renorm"):
        validate_policy({"inkling2_moe_rows_kernel": True, "inkling2_moe_logdomain_renorm": True})
    model = _tiny_model()
    with pytest.raises(ValueError, match="moe_rows_kernel requires moe_fused"):
        model.configure_kernels(moe_rows_kernel=True)
    with pytest.raises(ValueError, match="alternatives"):
        model.configure_kernels(moe_fused=True, moe_rows_kernel=True, moe_panel_kernel=True)
    model.configure_kernels(moe_fused=True)
    assert model.kernel_policy["moe_rows_kernel"] is False
    assert not any(layer.mlp.__dict__.get("_moe_rows_kernel") for layer in model.model.layers)


def test_numerics_contract_names_the_sources():
    from mlx2.adapters.inkling2 import Inkling2Adapter

    adapter = Inkling2Adapter.__new__(Inkling2Adapter)
    adapter.kernel_policy = {"moe_fused": True, "moe_rows_kernel": True}
    names = adapter.execution_numerics_contract()["source_sha256"]
    for name in ("inkling2/moe_rows_kernel.py", "inkling2/moe_decode_kernel.py",
                 "inkling2/moe_panel_kernel.py"):
        assert name in names
    adapter.kernel_policy = {"moe_fused": True}
    assert "inkling2/moe_rows_kernel.py" not in adapter.execution_numerics_contract()[
        "source_sha256"]


def test_option_off_never_launches(monkeypatch):
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 24), key=mx.random.key(1))
    plan = [(0, 7), (7, 17), (17, 18), (18, 22), (22, 24)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)

    def refuse(*_args, **_kwargs):
        raise AssertionError("rows kernel asked with the option off")

    model.configure_kernels(moe_fused=True, moe_rows_kernel=True)
    model.configure_kernels(moe_fused=True, moe_rows_kernel=False)
    monkeypatch.setattr(mrk, "call", refuse)
    assert mx.array_equal(_run(model, tokens, plan), base)
    assert _keys(model.kernel_receipt()) == {}


def test_option_on_cpu_declines_every_call_and_is_bitwise():
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 24), key=mx.random.key(2))
    plan = [(0, 7), (7, 17), (17, 18), (18, 22), (22, 24)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    model.configure_kernels(moe_fused=True, moe_rows_kernel=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    assert _keys(model.kernel_receipt()) == {"declined:not_gpu": 48 * len(plan)}


def test_plumbing_engages_on_nvfp4_rows_and_stays_bitwise(monkeypatch):
    """Tiny model: 8 experts, top 2.  2-15 rows engage on the NVFP4 visits;
    the MXFP8 visits decline (expert_format); 16 rows reach MLX's streaming
    floor (B >= 16, B / E >= 4); 17+ rows and one-token calls decline (rows)."""
    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 64), key=mx.random.key(3))
    plan = [(0, 7), (7, 17), (17, 29), (29, 45), (45, 46), (46, 49), (49, 50), (50, 64)]
    model.configure_kernels(moe_fused=True)
    base = _run(model, tokens, plan)
    _law_on_cpu(monkeypatch)
    model.configure_kernels(moe_fused=True, moe_rows_kernel=True)
    got = _run(model, tokens, plan)
    keys = _keys(model.kernel_receipt())
    engaged_calls = [7, 10, 12, 3, 14]  # rows per call that engage on NVFP4
    assert keys["rows_kernel"] == len(engaged_calls) * NVFP4_VISITS, keys
    # MXFP8 visits decline on the format first, the 16-row call included.
    assert keys["declined:expert_format"] == (len(engaged_calls) + 1) * MXFP8_VISITS
    # 16 rows on the NVFP4 visits: the shipped route streams; 1 row: rows.
    assert keys["declined:rows"] == 2 * 48
    assert keys.get("declined:rhs_route", 0) + keys.get("declined:rhs_pad", 0) == NVFP4_VISITS, keys
    assert mx.array_equal(got, base)


def test_b2_and_b4_decode_engage(monkeypatch):
    for lanes in (2, 4):
        model = _quantized_model()
        tokens = mx.random.randint(0, 500, (lanes, 12), key=mx.random.key(4 + lanes))
        plan = [(0, 8)] + [(i, i + 1) for i in range(8, 12)]
        model.configure_kernels(moe_fused=True)
        base = _run(model, tokens, plan)
        _law_on_cpu(monkeypatch)
        model.configure_kernels(moe_fused=True, moe_rows_kernel=True)
        got = _run(model, tokens, plan)
        keys = _keys(model.kernel_receipt())
        assert keys["rows_kernel"] == NVFP4_VISITS * (len(plan) - 1), keys
        assert mx.array_equal(got, base)
        monkeypatch.undo()


def test_weighted_sum_calls_decline(monkeypatch):
    from mlx2.runtime.models.inkling2 import moe_panel_kernel as mpk

    model = _quantized_model()
    tokens = mx.random.randint(0, 500, (1, 10), key=mx.random.key(7))
    _law_on_cpu(monkeypatch)
    monkeypatch.setattr(mpk, "shipped_weighted_sum_engages", lambda *a, **k: True)
    model.configure_kernels(moe_fused=True, moe_rows_kernel=True)
    _run(model, tokens, [(0, 6), (6, 10)])
    keys = _keys(model.kernel_receipt())
    assert keys == {"declined:weighted_sum_order_unproven": 2 * NVFP4_VISITS,
                    "declined:expert_format": 2 * MXFP8_VISITS}


@pytest.mark.parametrize("rows", [2, 3, 8, 15])
def test_law_matches_shipped_fused_call(rows):
    """Each row's one-token law against the shipped multi-row fused call on an
    NVFP4 layer: the target the GPU kernels must meet."""
    from mlx2.runtime.models.inkling2 import moe_decode_kernel as mdk
    from mlx2.runtime.models.inkling2 import moe_fast
    from mlx2.runtime.models.inkling2.language import _fp32

    model = _quantized_model()
    model.configure_kernels(moe_fused=True)
    moe = model.model.layers[3].mlp
    experts, refusal = moe._fast_experts()
    assert refusal is None
    for seed in range(4):
        x = (mx.random.normal((1, rows, model.config.hidden_size), key=mx.random.key(seed))
             ).astype(mx.bfloat16)
        ref = moe_fast.fused_call(moe, experts, x)
        order, scores = mdk.route_compiled(x, _fp32(moe, "gate_weight"),
                                           moe.e_score_correction_bias, moe.top_k)
        got = mrk.law(moe, experts, x, order, scores)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        assert mx.array_equal(got, ref)
