"""The box pack's lazy donor: slices read from an unread mx.load view match the eager slices, and a packed
layer's projections are released afterwards (child_pack._release); Hy4-flash's lazy_load policy validation."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


class QSw(nn.Module):                                       # [E, out, in] affine-quantized switch bank
    def __init__(self, w, s, b):
        super().__init__()
        self.weight, self.scales, self.biases = w, s, b
        self.group_size, self.bits, self.mode = 32, 4, "affine"


def test_lazy_view_slices_match_eager_and_release_empties(tmp_path):
    from mlx2.experimental.amoeba.child_pack import _check_slice, _release, quant_slice
    mx.set_default_device(mx.cpu)
    w, s, b = mx.quantize(mx.random.normal((3, 16, 256), key=mx.random.key(0)), group_size=32, bits=4)
    mx.save_safetensors(str(tmp_path / "bank.safetensors"), {"w": w, "s": s, "b": b})
    lazy = mx.load(str(tmp_path / "bank.safetensors"))          # unread views, as lazy_load leaves them
    eager_q = quant_slice(QSw(w, s, b), expert=1, rows=[2, 9, 11])
    sw = QSw(lazy["w"], lazy["s"], lazy["b"])
    lazy_q = quant_slice(sw, expert=1, rows=[2, 9, 11])
    _check_slice(lazy_q, sw, 1, rows=[2, 9, 11])
    for k in ("w", "s", "b"):
        np.testing.assert_array_equal(lazy_q[k], eager_q[k])
    _release(sw)
    assert all(sw[k].size == 0 for k in ("weight", "scales", "biases"))
    assert sw.group_size == 32 and sw.bits == 4                  # non-array attributes are kept


def test_hy4_lazy_load_policy_is_validated():
    from mlx2.adapters import hy_v4_flash as A
    model, _, _ = A._parse_policy({"lazy_load": True})
    assert model["lazy_load"] is True
    with pytest.raises(ValueError, match="lazy_load must be a boolean"):
        A._parse_policy({"lazy_load": 1})
    with pytest.raises(ValueError, match="lazy_load"):
        A._parse_policy({"lazy_load": True, "moe_fused_gate_up": True})


def test_host_loads_resident_when_the_adapter_declares_no_lazy_load(monkeypatch):
    """resident=False is a request: an adapter without LAZY_LOAD_POLICY gets no foreign policy keys."""
    import types

    from research.amoeba.carve import host as H
    seen = {}

    class NoLazy:
        def __init__(self, path, **kw):
            seen.update(kw)
            raise RuntimeError("stop after construction")

    monkeypatch.setattr("mlx2.adapters.registry.inspect_model",
                        lambda p: types.SimpleNamespace(adapter_type=NoLazy))
    logs = []
    with pytest.raises(RuntimeError, match="stop after construction"):
        H.Host("/nonexistent", log=logs.append, resident=False)
    assert seen == {} and any("loading resident" in m for m in logs)


def test_pack_refuses_layers_without_unit_sites(tmp_path, monkeypatch):
    import argparse
    import types

    from mlx2.experimental.amoeba import child_pack
    from research.amoeba.carve import host as H
    from research.amoeba.carve import ops as O

    site = lambda l, kind: types.SimpleNamespace(layer=l, kind=kind, n=4)  # noqa: E731
    fake = types.SimpleNamespace(L=2, widths=[4, 4], sites=[site(0, None), site(1, "dense")], resident=False,
                                 model=types.SimpleNamespace(lazy_shards=[]))
    monkeypatch.setattr(H, "Host", lambda *a, **k: fake)
    monkeypatch.setattr(O, "footprint_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(O, "setup_mlx", lambda *a, **k: None)
    np.savez(tmp_path / "m.npz", box=np.ones((2, 4), np.uint8))
    for layers in ("0", "5", ","):
        a = argparse.Namespace(model="m", mask=f"{tmp_path / 'm.npz'}:box", moe_units="neuron", cpu=True,
                               memory_gb=1, watchdog_gb=1, resident=False, layers=layers)
        with pytest.raises(SystemExit, match="--layers"):
            child_pack._pack(a, tmp_path / "out")


def test_layer_major_gate_refuses_vacuous_nan_and_out_of_tolerance_records():
    from mlx2.experimental.amoeba.child_removal import gate_passes

    def rec(box_lm, box_res, rnd=(0.02, 0.02), rel=1e-5, top1=1.0):
        arm = lambda lm, res: {"per_window": {"layer_major": list(lm), "resident": list(res)},  # noqa: E731
                               "mean_kl_resident": float(np.mean(res))}
        return {"full_logits_max_rel": rel, "full_top1_agreement": top1,
                "arms": {"box": arm(box_lm, box_res), "random": arm(rnd, rnd)}}
    assert gate_passes(rec([0.5, 0.3], [0.505, 0.3]))[0]                 # 1% of the KL: inside 2%
    assert not gate_passes(rec([0.5, 0.3], [0.55, 0.3]))[0]              # 10%: outside
    assert not gate_passes(rec([0.0, 0.0], [0.0005, 0.0002]))[0]         # vacuous: box KL under 1e-2
    assert not gate_passes(rec([0.5], [0.5], rel=float("nan")))[0]
    assert not gate_passes(rec([0.5], [float("nan")]))[0]
    assert not gate_passes(rec([0.5], [0.5], top1=0.99))[0]


def test_layer_major_scorer_refuses_windows_past_the_band_threshold():
    from research.amoeba.carve import layer_major as LM
    sc = LM.LayerMajorScorer.__new__(LM.LayerMajorScorer)
    with pytest.raises(SystemExit, match="band"):
        sc.kl_many([(None, list(range(LM.MAX_SEQ_LEN + 1)), 1)], {})
