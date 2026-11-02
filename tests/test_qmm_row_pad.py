"""CPU tests for the split-K row padding (qmm_row_pad).

The speed and arithmetic of the padded route need a NAX GPU
(qualification/runs/qmm-row-pad-20261030 checks them there).  These tests pin
what can be checked on CPU:

- the replicated MLX 0.32.4 route model (qmv limit, split factor, pad target)
  on the shapes the M5 Max sweep measured;
- padding preserves the product (zero rows sliced back off);
- eligibility, instance scoping, removal, build/device admission and the
  serving switch.
"""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx2.runtime.models import qmm_row_pad as rp  # noqa: E402


@pytest.mark.parametrize(
    "K, N, M, split, target",
    [
        # 27B down 17408 -> 5120: split 2 until the NAX qmm at 33 rows.
        (17408, 5120, 13, 2, 33),
        (17408, 5120, 32, 2, 33),
        (17408, 5120, 33, 1, None),
        # 27B o 6144 -> 5120: split 3.
        (6144, 5120, 16, 3, 33),
        # 35B q_gate 2048 -> 8192: split 2 from 13 rows.
        (2048, 8192, 16, 2, 33),
        # Flash-Next o 6144 -> 2560: split 6 (kept on stock), then 3 and 2.
        (6144, 2560, 16, 6, None),
        (6144, 2560, 48, 3, 97),
        (6144, 2560, 96, 2, 97),
        (6144, 2560, 128, 1, None),
        # 35B o 4096 -> 2048: qmv below 25 rows, split 8 (stock), split 4 (pad).
        (4096, 2048, 24, None, None),
        (4096, 2048, 32, 8, None),
        (4096, 2048, 64, 4, 129),
        # Narrow outputs stay on stock until the split falls to 4 (5 tied).
        (2560, 512, 32, 20, None),
        (5120, 1024, 64, 8, None),
        (5120, 1024, 96, 5, None),
        (5120, 1024, 128, 4, 257),
        (640, 2560, 32, 5, None),
        (640, 2560, 40, 2, 97),
        # Below the qmv batch limit nothing is touched.
        (17408, 5120, 4, None, None),
    ],
)
def test_route_model_matches_m5max_sweep(K, N, M, split, target):
    if split is not None:
        assert M >= rp.vector_limit(K, N)
        assert rp.split_k(M, N, K, 64) == split
    else:
        assert M < rp.vector_limit(K, N)
    assert rp.pad_target(M, K, N, 64) == target


def test_pad_target_lands_on_nax_qmm():
    for K, N in [(17408, 5120), (6144, 2560), (4096, 2048), (5120, 1024)]:
        for M in range(1, 300):
            target = rp.pad_target(M, K, N, 64, max_split=64)
            if target is not None:
                assert target > M
                assert rp.split_k(target, N, K, 64) == 1
                assert rp.split_k(target - 1, N, K, 64) > 1


def test_unaligned_k_stays_on_stock():
    assert rp.pad_target(16, 17408 + 32, 5120, 32) is None


def _model(K=6144, N=2560, bits=4, group_size=64, bias=False):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(K, N, bias=bias)
            self.small = nn.Linear(K, N, bias=False)

    m = M()
    nn.quantize(m, group_size=group_size, bits=bits)
    return m


@pytest.mark.parametrize("bias", [False, True])
def test_padding_preserves_product(bias):
    m = _model(bias=bias)
    x = mx.random.normal((2, 24, 6144), key=mx.random.key(0)).astype(mx.bfloat16)
    ref = m.proj(x)
    handle = rp.apply(m)
    try:
        before = dict(rp.STATS)
        got = m.proj(x)
        assert rp.STATS["routed"] == before["routed"] + 1
    finally:
        rp.remove(handle)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    # CPU runs one kernel for every M, so padded rows give the same numbers.
    assert mx.array_equal(got, ref)
    assert type(m.proj) is nn.QuantizedLinear


def test_stock_calls_are_counted_not_padded():
    m = _model()
    handle = rp.apply(m)
    try:
        before = dict(rp.STATS)
        m.proj(mx.zeros((4, 6144), dtype=mx.bfloat16))  # qmv band
        m.proj(mx.zeros((16, 6144), dtype=mx.float32))  # fp32 activations
        assert rp.STATS["routed"] == before["routed"]
        assert rp.STATS["stock"] == before["stock"] + 2
    finally:
        rp.remove(handle)


def test_apply_skips_other_formats_and_swapped_instances():
    assert rp.apply(_model(bits=8)) == []
    assert rp.apply(_model(group_size=32)) == []
    m = _model()

    class Owned(nn.QuantizedLinear):
        pass

    m.small.__class__ = Owned
    handle = rp.apply(m)
    assert [module for module, _ in handle] == [m.proj]
    assert type(m.small) is Owned


def test_refusal_admits_only_modelled_build_and_device():
    assert rp.refusal("0.32.4", "applegpu_g17s") is None
    assert rp.refusal("0.32.4", "applegpu_g18p") is None
    assert "0.32.4" in rp.refusal("0.32.5", "applegpu_g17s")
    assert rp.refusal("0.32.4", "applegpu_g16s") is not None
    assert rp.refusal("0.32.4", "applegpu_g17d") is not None
    assert rp.refusal("0.32.4", "") is not None


def test_serving_switch_is_boolean_default_off(monkeypatch):
    from mlx2.serving import qmm_row_pad_selected

    monkeypatch.delenv("MLX2_QMM_ROW_PAD", raising=False)
    assert qmm_row_pad_selected(None) is False
    monkeypatch.setenv("MLX2_QMM_ROW_PAD", "1")
    assert qmm_row_pad_selected({}) is True
    assert qmm_row_pad_selected({"qmm_row_pad": False}) is False
    with pytest.raises(ValueError):
        qmm_row_pad_selected({"qmm_row_pad": 1})


def test_numerics_identity_binds_row_pad():
    from mlx2.runtime.apc_numerics import execution_numerics_identity

    assert execution_numerics_identity({}) is None
    assert execution_numerics_identity({}, qmm_row_pad=True) == {"version": 1, "qmm_row_pad": True}


def test_weight_streaming_refuses_row_pad():
    from mlx2.serving import weight_streaming_refusals

    reasons = weight_streaming_refusals(
        lane_matmul="auto", sp_qmm=False, int8_prefill=False, verify_bitexact=False,
        multi_lora=False, lora_root=None, execution_policy={}, qmm_row_pad=True,
    )
    assert "qmm_row_pad" in reasons


def test_qualification_observes_only_routed_calls_in_the_run():
    from scripts.qualify_serving import feature_observations

    settings = {"mtp": False, "qmm_row_pad": {"modules": 3, "policy": rp.POLICY_ID}}
    status = {"enabled": True, "modules": 3, "routed_calls": 5, "stock_calls": 2}
    initial = {"settings": settings, "qmm_row_pad": dict(status)}
    final = {"settings": settings, "qmm_row_pad": dict(status, routed_calls=11)}
    assert feature_observations(final, initial=initial)["qmm_row_pad"] == 6
    assert feature_observations(final)["qmm_row_pad"] == 0
    final["qmm_row_pad"]["enabled"] = False
    assert feature_observations(final, initial=initial)["qmm_row_pad"] == 0
