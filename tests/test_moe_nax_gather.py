"""CPU tests for the NAX sorted MoE gather port (omlx #3995/#4022/#4029).

The kernels themselves only run on an M5 GPU (scripts/bench_moe_nax_gather.py
and qualification/runs/port-nax-gather-20261002); here: the default-off
policy, the admission and its counted refusals, the lazy sort's equivalence
to ``_gather_sort``, the plan table, and that every route keeps the stock
output when the kernel declines.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models import moe_nax_gather as nax
from mlx2.runtime.models import qwen3_next as qn
from mlx2.runtime.models import switch_layers as sl


@pytest.fixture(autouse=True)
def _reset():
    old = nax.MODE
    nax.status(reset=True)
    yield
    nax.set_mode(old)
    nax.status(reset=True)


def _qlinear(E=8, out=128, inp=128, bits=4):
    lin = sl.SwitchLinear(inp, out, E, bias=False)
    return lin.to_quantized(group_size=64, bits=bits)


def _sorted_rows(T=40, k=2, E=8, D=128, seed=0):
    mx.random.seed(seed)
    x = mx.random.normal((T, D)).astype(mx.bfloat16)
    inds = mx.random.randint(0, E, (T, k)).astype(mx.uint32)
    return x, inds


# --- policy -----------------------------------------------------------------


def test_default_mode_is_off():
    assert nax.mode_from_env({}) == "off"
    assert nax.MODES == ("off", "gather", "fused")


@pytest.mark.parametrize(
    "raw,mode",
    [("0", "off"), ("off", "off"), ("", "off"), ("1", "fused"), ("on", "fused"),
     ("gather", "gather"), ("FUSED", "fused")],
)
def test_mode_parsing(raw, mode):
    assert nax.mode_from_env({nax.ENV_MODE: raw}) == mode


def test_invalid_mode_refuses():
    with pytest.raises(ValueError):
        nax.mode_from_env({nax.ENV_MODE: "maybe"})
    with pytest.raises(ValueError):
        nax.set_mode("maybe")


def test_set_mode_returns_previous():
    nax.set_mode("off")
    assert nax.set_mode("gather") == "off"
    assert nax.MODE == "gather"


def test_env_prefix_is_import_guarded():
    from mlx2.runtime.models.import_env import PREFIXES

    assert nax.ENV_MODE.startswith(PREFIXES)


# --- admission --------------------------------------------------------------


def test_rows_ok_matches_mlx_rhs_gate():
    assert nax.rows_ok(16, 4)
    assert not nax.rows_ok(15, 1)
    assert not nax.rows_ok(5119, 1280)
    assert nax.rows_ok(5120, 512)  # 512-token Flash-Next chunk, top-10
    assert not nax.rows_ok(2047, 512)  # under 4 rows/expert: gather_qmv
    assert not nax.rows_ok(16, 0)


def _supports(x, lin, idx, **over):
    args = dict(
        x=x, w=lin["weight"], scales=lin["scales"], biases=lin.get("biases"),
        indices=idx, group_size=lin.group_size, bits=lin.bits, mode=lin.mode,
    )
    args.update(over)
    return nax.supports(**args)


def test_supports_accepts_flash_next_layout_and_refuses_others():
    lin = _qlinear()
    lin.scales = lin.scales.astype(mx.bfloat16)
    lin.biases = lin.biases.astype(mx.bfloat16)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    idx = mx.zeros((64,), mx.uint32)
    assert _supports(x, lin, idx)
    assert not _supports(x.astype(mx.float32), lin, idx)  # fp32 activations
    assert not _supports(x, lin, idx.astype(mx.int32))  # index dtype
    assert not _supports(x[:8], lin, idx)  # rows != indices
    assert not _supports(x, lin, idx, biases=None)  # affine needs biases
    assert not _supports(x, lin, idx, bits=3)
    assert not _supports(x[:4], lin, idx[:4])  # < 8 rows
    rm = mx.zeros((64,), mx.uint32)
    assert _supports(x[:5], lin, idx, row_map=rm)  # token rows through a map
    assert not _supports(x[:5], lin, idx, row_map=rm.astype(mx.int32))


def test_plan_table():
    seg, db = nax._SCHED_SEG, nax._SCHED_DB
    # Flash-Next gate_up (K=2560, N=1280) at 10 rows/expert: db 64, plain grid.
    assert nax._plan(5120, 512, 2560, 1280) == nax.Plan(db, 64, 64, 0, 0)
    # 8192-token chunk: 160 rows/expert -> 128-row seg, tile-on-x.
    assert nax._plan(81920, 512, 2560, 1280) == nax.Plan(seg, 128, 128, 32, 8192)
    # down (K=640) at 160 rows/expert -> 96-row seg.
    assert nax._plan(81920, 512, 640, 2560) == nax.Plan(seg, 96, 128, 32, 0)
    # ragged K keeps plain 64-row seg tiles.
    assert nax._plan(81920, 512, 608, 2560) == nax.Plan(seg, 64, 64, 0, 0)


def test_plan_env_pin(monkeypatch):
    monkeypatch.setenv(nax._ENV_PLAN, "seg,96,64,0,0")
    assert nax._plan(10, 1, 64, 64) == nax.Plan(nax._SCHED_SEG, 96, 64, 0, 0)
    monkeypatch.setenv(nax._ENV_PLAN, "bogus")
    assert nax._plan(10, 1, 64, 64).bm in nax._TILE_ROWS


def test_try_gather_counts_refusals(monkeypatch):
    lin = _qlinear(E=4)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    small = mx.zeros((8,), mx.uint32)
    assert nax.try_gather(x[:8], lin, small) is None
    assert nax.status()["not_candidates"] == 1
    assert nax.status()["fallbacks"] == {}
    idx = mx.zeros((64,), mx.uint32)
    monkeypatch.setattr(nax, "_nax_host", False)
    assert nax.try_gather(x, lin, idx) is None
    assert nax.status()["fallbacks"] == {"not_nax_host": 1}
    monkeypatch.setattr(nax, "_nax_host", True)
    assert nax.try_gather(x, lin, idx) is None  # conftest: CPU default device
    st = nax.status()
    assert st["fallbacks"]["cpu_device"] == 1 and st["last_fallback"] == "cpu_device"
    assert not any(st["calls"].values())


def test_try_gather_unsupported_and_unverified(monkeypatch):
    monkeypatch.setattr(nax, "_nax_host", True)
    monkeypatch.setattr(nax.mx, "default_device", lambda: nax.mx.gpu)
    lin = _qlinear(E=4)  # fp32 scales: layout refused for bf16 x
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    idx = mx.zeros((64,), mx.uint32)
    assert nax.try_gather(x, lin, idx) is None
    assert nax.status()["fallbacks"] == {"unsupported_layout": 1}
    lin.scales = lin.scales.astype(mx.bfloat16)
    lin.biases = lin.biases.astype(mx.bfloat16)
    monkeypatch.setattr(nax, "sorted_gather_qmm", lambda *a, **k: None)
    assert nax.try_gather(x, lin, idx) is None
    assert nax.status()["fallbacks"]["kernel_unverified"] == 1


def test_try_swiglu_falls_back_from_row_map(monkeypatch):
    monkeypatch.setattr(nax, "_nax_host", True)
    monkeypatch.setattr(nax.mx, "default_device", lambda: nax.mx.gpu)
    seen = []

    def fake(x, *a, row_map=None, **k):
        seen.append(row_map is not None)
        return None if row_map is not None else mx.ones((64, 1, 64))

    monkeypatch.setattr(nax, "sorted_gather_qmm_swiglu", fake)
    proj = _qlinear(E=4)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    idx = mx.zeros((64,), mx.uint32)
    out = nax.try_swiglu(proj, x, idx, (x[:16], mx.zeros((64,), mx.uint32)))
    assert out is not None and seen == [True, False]
    st = nax.status()
    assert st["fallbacks"] == {"row_map_declined": 1}
    assert st["calls"]["swiglu"] == 1 and st["calls"]["swiglu_map"] == 0


def test_status_reset():
    nax._fallback("x")
    nax.calls["gather"] += 2
    st = nax.status(reset=True)
    assert st["fallbacks"] == {"x": 1} and st["calls"]["gather"] == 2
    st = nax.status()
    assert st["fallbacks"] == {} and st["calls"]["gather"] == 0 and st["last_fallback"] is None


# --- lazy sort ------------------------------------------------------------


@pytest.mark.parametrize("T,k", [(40, 2), (3281, 10)])  # second: n > 32768, tail pad
def test_sort_routes_equals_gather_sort(T, k):
    x, inds = _sorted_rows(T=T, k=k, E=64, D=8)
    xe = mx.expand_dims(x, (-2, -3))
    ref_x, ref_idx, ref_inv = sl._gather_sort(xe, inds)
    xs, idx, inv, x_tok, row_map = sl._sort_routes(xe, inds)
    assert mx.array_equal(xs, ref_x).item()
    assert mx.array_equal(idx, ref_idx).item()
    assert mx.array_equal(inv, ref_inv).item()
    assert mx.array_equal(x_tok[row_map], ref_x).item()
    assert row_map.shape == idx.shape
    if T * k > 32768:
        assert idx.size % 64 == 0 and idx.size > T * k


# --- wiring: off and declined routes keep the stock output --------------------


def test_off_never_consults_the_kernel(monkeypatch):
    nax.set_mode("off")

    def boom(*a, **k):
        raise AssertionError("kernel consulted while off")

    monkeypatch.setattr(nax, "try_gather", boom)
    monkeypatch.setattr(nax, "try_swiglu", boom)
    lin = _qlinear(E=4)
    x = mx.random.normal((64, 1, 128))
    idx = mx.sort(mx.random.randint(0, 4, (64,)).astype(mx.uint32))
    lin(x, idx, sorted_indices=True)


def test_gather_mode_declined_matches_stock():
    lin = _qlinear(E=4)
    x = mx.random.normal((64, 1, 128))
    idx = mx.sort(mx.random.randint(0, 4, (64,)).astype(mx.uint32))
    nax.set_mode("off")
    ref = lin(x, idx, sorted_indices=True)
    nax.set_mode("gather")
    out = lin(x, idx, sorted_indices=True)
    assert mx.array_equal(out, ref).item()
    assert nax.status()["fallbacks"].get("cpu_device", 0) + nax.status()["fallbacks"].get(
        "not_nax_host", 0) == 1


def _fused_glu(E=8, D=128, H=64, bits=4):
    m = qn.FusedGateUpSwitchGLU(D, H, E)
    m.gate_up_proj = m.gate_up_proj.to_quantized(group_size=64, bits=bits)
    m.down_proj = m.down_proj.to_quantized(group_size=64, bits=bits)
    m.eval()
    return m


def test_fused_mode_declined_matches_stock():
    m = _fused_glu()
    x, inds = _sorted_rows(T=40, k=2, E=8, D=128)
    x = x.astype(mx.float32)
    nax.set_mode("off")
    ref = m(x, inds)
    nax.set_mode("fused")
    assert qn._nax_swiglu_candidate(m)
    out = m(x, inds)
    assert mx.array_equal(out, ref).item()
    st = nax.status()
    assert not any(st["calls"].values())
    assert sum(st["fallbacks"].values()) >= 1


def test_fused_candidate_refusals():
    m = _fused_glu()
    nax.set_mode("gather")
    assert not qn._nax_swiglu_candidate(m)  # gather mode leaves gate/up split
    nax.set_mode("fused")
    m.train()
    assert not qn._nax_swiglu_candidate(m)
    m.eval()

    class OtherAct(sl.SwiGLU):
        pass

    m.activation = OtherAct()
    assert not qn._nax_swiglu_candidate(m)
    m.activation = sl.SwiGLU()

    class Streamed(sl.QuantizedSwitchLinear):
        pass

    real = m.gate_up_proj
    s = Streamed.__new__(Streamed)
    s.__dict__.update(real.__dict__)
    m.gate_up_proj = s
    assert not qn._nax_swiglu_candidate(m)
    m.gate_up_proj = real
    assert qn._nax_swiglu_candidate(m)


def test_decode_width_is_not_sorted_so_untouched(monkeypatch):
    nax.set_mode("fused")

    def boom(*a, **k):
        raise AssertionError("kernel consulted for an unsorted decode call")

    monkeypatch.setattr(nax, "try_gather", boom)
    monkeypatch.setattr(nax, "try_swiglu", boom)
    m = _fused_glu()
    x, inds = _sorted_rows(T=1, k=10, E=8, D=128)  # 10 < sort minimum 20
    m(x.astype(mx.float32), inds)


def _split_glu(E=8, D=128, H=64, bits=4):
    m = qn.FusedDownSwitchGLU(D, H, E)
    m.gate_proj = m.gate_proj.to_quantized(group_size=64, bits=bits)
    m.up_proj = m.up_proj.to_quantized(group_size=64, bits=bits)
    m.down_proj = m.down_proj.to_quantized(group_size=64, bits=bits)
    m.eval()
    return m


def test_split_fused_mode_declined_matches_stock():
    m = _split_glu()
    x, inds = _sorted_rows(T=40, k=2, E=8, D=128)
    x = x.astype(mx.float32)
    nax.set_mode("off")
    ref = m(x, inds)
    nax.set_mode("fused")
    assert qn._nax_split_swiglu_candidate(m)
    out = m(x, inds)
    assert mx.array_equal(out, ref).item()
    assert not any(nax.status()["calls"].values())


def test_split_candidate_refusals():
    m = _split_glu()
    nax.set_mode("gather")
    assert not qn._nax_split_swiglu_candidate(m)
    nax.set_mode("fused")
    m.train()
    assert not qn._nax_split_swiglu_candidate(m)
    m.eval()
    real = m.up_proj
    m.up_proj = sl.SwitchLinear(128, 64, 8, bias=False)  # unquantized
    assert not qn._nax_split_swiglu_candidate(m)
    m.up_proj = real
    assert qn._nax_split_swiglu_candidate(m)


def test_try_swiglu_split_refuses_mixed_formats(monkeypatch):
    monkeypatch.setattr(nax, "_nax_host", True)
    monkeypatch.setattr(nax.mx, "default_device", lambda: nax.mx.gpu)
    gate = _qlinear(E=4, bits=4)
    up = _qlinear(E=4, bits=8)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    idx = mx.zeros((64,), mx.uint32)
    assert nax.try_swiglu_split(gate, up, x, idx) is None
    assert nax.status()["fallbacks"] == {"gate_up_formats_differ": 1}


def test_split_api_refuses_mismatched_tables(monkeypatch):
    monkeypatch.setattr(nax, "_nax_host", True)
    gate = _qlinear(E=4)
    up = _qlinear(E=4, out=64)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    idx = mx.zeros((64,), mx.uint32)
    t = lambda p: (p["weight"], p["scales"], p.get("biases"))
    assert nax.sorted_gather_qmm_swiglu_split(
        x, t(gate), t(up), idx, group_size=64, bits=4) is None
