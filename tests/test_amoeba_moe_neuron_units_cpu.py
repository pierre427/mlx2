"""--moe-units neuron in probe_amoeba_unit_box.install_units: exactness against the stock MoE block and the
expert-unit path, on tiny synthetic blocks (CPU, fp32 and 4-bit switch projections)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import probe_amoeba_unit_box as pb

from mlx2.runtime.models import qwen3_next as qn
from mlx2.runtime.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
)

D, F, E, K = 64, 64, 8, 2


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    pb.ST.update(active=False, allow=None, record=None)
    yield
    pb.ST.update(active=False, allow=None, record=None)
    mx.set_default_device(prev)


class Layer(nn.Module):
    def __init__(self, name, m):
        super().__init__()
        setattr(self, name, m)


def _qwen(variant):
    args = SimpleNamespace(hidden_size=D, moe_intermediate_size=F, shared_expert_intermediate_size=F,
                           norm_topk_prob=True, num_experts=E, num_experts_per_tok=K)
    old = (qn._MOE_FUSED_GATE_UP, qn._MOE_SHARED_IN_GATHER)
    qn._MOE_FUSED_GATE_UP = variant != "split"
    qn._MOE_SHARED_IN_GATHER = variant == "folded"
    try:
        m = qn.Qwen3NextSparseMoeBlock(args)
    finally:
        qn._MOE_FUSED_GATE_UP, qn._MOE_SHARED_IN_GATHER = old
    assert m.shared_folded == (variant == "folded")
    return m


def _ling():
    from mlx2.runtime.models import bailing_hybrid as bh
    args = SimpleNamespace(hidden_size=D, moe_intermediate_size=F, num_experts=E, num_experts_per_tok=K, use_bias=False,
                           moe_shared_expert_intermediate_size=F, num_shared_experts=1, intermediate_size=F, n_group=1,
                           topk_group=1, norm_topk_prob=True, routed_scaling_factor=1.0, score_function="sigmoid",
                           moe_router_enable_expert_bias=True, vendor_v32_laws=False, router_dtype=None)
    return bh.SparseMoeBlock(args)


def _gemma4(bias=False):
    from mlx2.runtime.models import gemma4 as g4
    m = g4.Experts(SimpleNamespace(hidden_size=D, moe_intermediate_size=F, num_experts=E))
    if bias:
        m.switch_glu = SwitchGLU(D, F, E, activation=g4.GeGLU(), bias=True)
    return m


BUILD = {"qwen_fused": lambda: _qwen("fused"), "qwen_split": lambda: _qwen("split"),
         "qwen_folded": lambda: _qwen("folded"), "ling": _ling, "gemma4": _gemma4}


def _make(name, quant, seed=0, dtype=mx.float32, bits=4):
    mx.random.seed(seed)
    m = BUILD[name]()
    if dtype != mx.float32:
        m.set_dtype(dtype)
    if quant:
        nn.quantize(m, group_size=32, bits=bits, class_predicate=lambda p, mod: isinstance(mod, SwitchLinear))
    mx.eval(m.parameters())
    return m


def _install(name, quant, moe_units, seed=0, dtype=mx.float32, bits=4):
    m = _make(name, quant, seed, dtype, bits)
    attr = "experts" if name == "gemma4" else "mlp"
    sites = pb.install_units(mx, [Layer(attr, m)], moe_units=moe_units)
    assert len(sites) == 1 and sites[0][1] is not None
    return m, sites[0]


def _inputs(name, T, seed=1, dtype=mx.float32):
    mx.random.seed(seed)
    x = mx.random.normal((1, T, D)).astype(dtype)
    if name != "gemma4":
        return (x,)
    inds = mx.argsort(mx.random.uniform(shape=(1, T, E)), axis=-1)[..., :K].astype(mx.uint32)
    w = mx.softmax(mx.random.normal((1, T, K)), axis=-1).astype(dtype)
    return (x, inds, w)


def _run(m, a, allow=None, record=False):
    pb.ST.update(active=True, allow=None if allow is None else mx.array(allow[None]), record=[None] if record else None)
    y = m(*a)
    mx.eval(y)
    rec = pb.ST["record"]
    pb.ST.update(active=False, allow=None, record=None)
    return np.array(y.astype(mx.float32)), (rec[0] if rec else None)


CASES = [(n, q, T) for n in BUILD for q in (False, True) for T in (3, 16)]  # T*K = 6 (unsorted) / 32 (sorted gather)


@pytest.mark.parametrize("name,quant,T", CASES)
def test_neuron_all_allowed_matches_stock_block(name, quant, T):
    m, site = _install(name, quant, "neuron")
    assert site[2] == E * F and site[3] == 3 * D
    a = _inputs(name, T)
    ref = np.array(m(*a))  # ST inactive: the original module call
    for allow in (None, np.ones(E * F, np.float32)):
        y, _ = _run(m, a, allow)
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name,quant,T", CASES)
def test_neuron_expert_drop_equals_expert_mode_drop(name, quant, T):
    mn, _ = _install(name, quant, "neuron")
    me, se = _install(name, quant, "expert")
    assert se[2] == E
    a = _inputs(name, T)
    for drop in ([0], [3, 5], list(range(E))):
        an = np.ones(E * F, np.float32)
        ae = np.ones(E, np.float32)
        for e in drop:
            an[e * F:(e + 1) * F] = 0
            ae[e] = 0
        yn, _ = _run(mn, a, an)
        ye, _ = _run(me, a, ae)
        np.testing.assert_allclose(yn, ye, rtol=1e-5, atol=1e-5)


def _dense(proj):
    if isinstance(proj, QuantizedSwitchLinear):
        return np.array(mx.dequantize(proj.weight, proj.scales, proj.biases, proj.group_size, proj.bits))
    return np.array(proj.weight)


def _reference_hidden(m, name, a):
    """Router (via the same helpers) + numpy SwiGLU/GeGLU per selected expert: inds, w, a [T, K, F], Wd [E, D, F]."""
    sw = m.switch_glu if name == "gemma4" else m.switch_mlp
    if name == "gemma4":
        x, inds, w = a
    else:
        x = a[0]
        fn = pb._qwen_next_rows if name.startswith("qwen") else pb._ling_rows
        inds, w, _, _ = fn(mx, m, x)
    x, inds, w = np.array(x)[0], np.array(inds)[0].astype(int), np.array(w.astype(mx.float32))[0]
    if hasattr(sw, "gate_up_proj"):
        gu = _dense(sw.gate_up_proj)
        Wg, Wu = gu[:, :F], gu[:, F:]
    else:
        Wg, Wu = _dense(sw.gate_proj), _dense(sw.up_proj)
    Wd = _dense(sw.down_proj)
    g = np.einsum("tkfd,td->tkf", Wg[inds], x)
    u = np.einsum("tkfd,td->tkf", Wu[inds], x)
    if name == "gemma4":
        act = np.array(nn.gelu_approx(mx.array(g))) * u
    else:
        act = g / (1 + np.exp(-g)) * u
    return inds, w, act, Wd


@pytest.mark.parametrize("name,quant,T", CASES)
def test_neuron_record_mass_and_relation_to_expert_mass(name, quant, T):
    mn, _ = _install(name, quant, "neuron")
    me, _ = _install(name, quant, "expert")
    a = _inputs(name, T)
    _, rn = _run(mn, a, record=True)
    _, re_ = _run(me, a, record=True)
    assert rn[0] == "moe_neuron" and re_[0] == "moe"
    tot = np.array(rn[1]).reshape(E, F)
    inds, w, act, Wd = _reference_hidden(mn, name, a)
    colnorm = np.linalg.norm(Wd[:E], axis=1)                        # [E, F]
    c = w[..., None] * np.abs(act) * colnorm[inds]                    # [T, K, F]
    ref = np.zeros((E, F))
    np.add.at(ref, inds, c)
    np.testing.assert_allclose(tot, ref, rtol=2e-4, atol=1e-5)
    # per-token top-32 flat ids name selected experts and carry their contributions
    ids, topc = np.array(rn[2])[0], np.array(rn[3])[0].astype(np.float32)
    for t in range(T):
        assert set((ids[t] // F).tolist()) <= set(inds[t].tolist())
        np.testing.assert_allclose(topc[t].max(), c[t].max(), rtol=2e-3)
    # expert mass ||w_e y_e|| is the norm of the sum of the neuron writes; the neuron masses sum to an
    # upper bound of it (triangle inequality), equal only when the writes are parallel
    writes = np.einsum("tkf,tkdf->tkd", w[..., None] * act, Wd[inds])
    en = np.linalg.norm(writes, axis=-1)
    np.testing.assert_allclose(np.array(re_[3])[0].astype(np.float32), en, rtol=2e-3, atol=1e-3)
    emass = np.zeros(E)
    np.add.at(emass, np.array(re_[1])[0].astype(int), np.array(re_[3])[0].astype(np.float64))
    assert np.all(tot.sum(1) >= emass * (1 - 2e-3) - 1e-4)
    assert tot.sum() > emass.sum() * 1.05  # random weights: neuron writes are far from parallel


def test_neuron_record_respects_mask():
    m, _ = _install("qwen_fused", False, "neuron")
    a = _inputs("qwen_fused", 16)
    allow = np.ones(E * F, np.float32)
    off = np.random.default_rng(0).choice(E * F, E * F // 3, replace=False)
    allow[off] = 0
    _, rec = _run(m, a, allow, record=True)
    assert np.all(np.array(rec[1])[off] == 0)


def test_neuron_mode_fails_closed_on_biased_down_proj():
    with pytest.raises(ValueError, match="biased down_proj"):
        pb.install_units(mx, [Layer("experts", _gemma4(bias=True))], moe_units="neuron")
    pb.install_units(mx, [Layer("experts", _gemma4(bias=True))], moe_units="expert")  # expert mode unchanged


def test_neuron_mode_fails_closed_without_switch_glu():
    m = _gemma4()
    m.switch_glu = SimpleNamespace(down_proj=m.switch_glu.down_proj, num_experts=E)
    with pytest.raises(ValueError, match="supports only SwitchGLU, FusedDownSwitchGLU, FusedGateUpSwitchGLU"):
        pb.install_units(mx, [Layer("experts", m)], moe_units="neuron")


def test_unknown_moe_units_rejected():
    with pytest.raises(ValueError):
        pb.install_units(mx, [], moe_units="expert_neuron")


def _old_mass_matched(m, b):
    out = np.zeros(len(m), bool)
    free = [u for u in np.argsort(m) if not b[u]]
    fm = m[free]
    used = np.zeros(len(free), bool)
    for u in np.flatnonzero(b)[np.argsort(-m[b])]:
        d = np.abs(fm - m[u]) + used * 1e300
        j = int(np.argmin(d))
        if used[j]:
            break
        used[j] = True
        out[free[j]] = True
    return out


@pytest.mark.parametrize("n,k", [(50, 5), (400, 60), (300, 200), (30, 29)])
def test_fast_mass_matched_twin_equals_reference_rule(n, k):
    donor = pytest.importorskip("amoeba_cwe_donor")
    rng = np.random.default_rng(n + k)
    m = rng.gamma(0.5, size=n)
    b = np.zeros(n, bool)
    b[rng.choice(n, k, replace=False)] = True
    fast = donor.mass_matched_twin(m, b, n + 7)
    assert fast.shape == (n + 7,) and not fast[n:].any()
    np.testing.assert_array_equal(fast[:n], _old_mass_matched(m, b))
    assert not (fast[:n] & b).any() and fast.sum() == min(k, n - k)


# ---------------------------------------------------------------- bf16 + 4-bit: patched == stock, bit for bit

BF16 = [(n, T) for n in BUILD for T in (3, 16)]


@pytest.mark.parametrize("name,T", BF16)
@pytest.mark.parametrize("units", ["expert", "neuron"])
def test_bf16_4bit_patched_equals_stock_bitwise(name, T, units):
    m, site = _install(name, True, units, dtype=mx.bfloat16)
    a = _inputs(name, T, dtype=mx.bfloat16)
    ref = m(*a)
    assert ref.dtype == mx.bfloat16
    ref = np.array(ref.astype(mx.float32))
    n = site[2]
    for allow in (None, np.ones(n, np.float32)):
        y, _ = _run(m, a, allow)
        np.testing.assert_array_equal(y, ref)
    pb.ST.update(active=True, record=[None])  # recording does not change the output either
    y = np.array(m(*a).astype(mx.float32))
    pb.ST.update(active=False, record=None)
    np.testing.assert_array_equal(y, ref)


@pytest.mark.parametrize("name,T", BF16)
def test_bf16_4bit_neuron_expert_drop_bitwise(name, T):
    mn, _ = _install(name, True, "neuron", dtype=mx.bfloat16)
    me, _ = _install(name, True, "expert", dtype=mx.bfloat16)
    a = _inputs(name, T, dtype=mx.bfloat16)
    for drop in ([1], [2, 6, 7]):
        an, ae = np.ones(E * F, np.float32), np.ones(E, np.float32)
        for e in drop:
            an[e * F:(e + 1) * F], ae[e] = 0, 0
        yn, _ = _run(mn, a, an)
        ye, _ = _run(me, a, ae)
        np.testing.assert_array_equal(yn, ye)
        # the override runs the expert rule on the neuron-installed module (the donor's parity check)
        pb.ST["moe_units_override"] = "expert"
        try:
            yo, _ = _run(mn, a, np.concatenate([ae, np.ones(E * F - E, np.float32)]))
        finally:
            pb.ST["moe_units_override"] = None
        np.testing.assert_array_equal(yo, ye)


def test_neuron_override_refused_on_expert_site():
    m, _ = _install("qwen_fused", False, "expert")
    pb.ST["moe_units_override"] = "neuron"
    try:
        with pytest.raises(RuntimeError, match="installed with expert units"):
            _run(m, _inputs("qwen_fused", 3))
    finally:
        pb.ST["moe_units_override"] = None


@pytest.mark.parametrize("bits", [3, 4, 5, 6, 8])
@pytest.mark.parametrize("name", ["qwen_fused", "qwen_split", "ling", "gemma4"])
def test_params_per_unit_any_bit_width(name, bits):
    _, sn = _install(name, True, "neuron", bits=bits)
    _, se = _install(name, True, "expert", bits=bits)
    assert sn[3] == 3 * D and se[3] == 3 * D * F


def test_neuron_mode_fails_closed_on_switch_subclass():
    class OddGLU(SwitchGLU):
        pass

    m = _gemma4()
    m.switch_glu = OddGLU(D, F, E, activation=m.switch_glu.activation)
    with pytest.raises(ValueError, match="found .*OddGLU"):
        pb.install_units(mx, [Layer("experts", m)], moe_units="neuron")


def test_hub_transfer_refuses_box_of_other_granularity(tmp_path, monkeypatch):
    import amoeba_hub_transfer as ht
    run = tmp_path / "box"
    run.mkdir()
    (run / "box.json").write_text('{"checkpoint": "x@y", "moe_units": "neuron", "boxes": {}}')
    for extra, msg in (([], "moe_units='neuron'"),):
        monkeypatch.setattr(sys, "argv", ["amoeba_hub_transfer.py", "--donor", "/nonexistent", "--out",
                                          str(tmp_path / "out"), "--donor-box-run", str(run)] + extra)
        with pytest.raises(SystemExit, match=msg):
            ht.main()
    (run / "box.json").write_text('{"checkpoint": "x@y", "boxes": {}}')  # pre-field boxes are expert boxes
    monkeypatch.setattr(sys, "argv", ["amoeba_hub_transfer.py", "--donor", "/nonexistent", "--out",
                                      str(tmp_path / "out2"), "--donor-box-run", str(run), "--moe-units", "neuron"])
    with pytest.raises(SystemExit, match="moe_units='expert'"):
        ht.main()
