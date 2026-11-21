"""Bug sweep 2026-11-21, lane amoeba-carve (AMCV-1..9): CPU regressions on toy hosts, no checkpoint."""
from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_amoeba_carve_cpu as TC  # noqa: E402
from research.amoeba.carve import __main__ as Dm  # noqa: E402
from research.amoeba.carve import evals as Ev  # noqa: E402
from research.amoeba.carve import export as X  # noqa: E402
from research.amoeba.carve import ops as O  # noqa: E402
from research.amoeba.carve.host import Host  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _host(m, **kw):
    return Host(model=m, tokenizer=TC.Tok(), fingerprint="toy:abc", log=lambda *_: None, **kw)


def _corpus(tmp_path, n=60):
    rng = np.random.default_rng(0)
    rows = [{"text": "".join(chr(97 + int(c)) for c in rng.integers(0, 26, 80)), "domain": "text", "source": f"s{i}"}
            for i in range(n)]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


# ------------------------------------------------------------------ AMCV-1
def _biased_toy(seed=0):
    """The carve toy with an unquantized, biased dense up_proj and biased stacked expert projections."""
    from mlx2.runtime.models.switch_layers import SwitchGLU
    mx.random.seed(seed)
    m = TC.Toy(True)
    for i, layer in enumerate(m.model.layers):
        if i % 2 == 1:
            layer.mlp.switch_mlp = SwitchGLU(TC.D, 16, TC.E, bias=True)
            for proj in (layer.mlp.switch_mlp.gate_proj, layer.mlp.switch_mlp.up_proj, layer.mlp.switch_mlp.down_proj):
                proj.bias = mx.random.normal(proj.bias.shape) * 0.5
        else:
            layer.mlp.up_proj = nn.Linear(TC.D, TC.H, bias=True)
            layer.mlp.up_proj.bias = mx.random.normal((TC.H,)) * 0.5
    mx.eval(m.parameters())
    return m


def test_export_plan_lists_weight_and_bias_of_unquantized_biased_projections(monkeypatch):
    from research.amoeba.carve.units import Site
    keys = {f"model.layers.0.mlp.up_proj.{k}": "m.safetensors" for k in ("weight", "bias")}
    keys.update({f"model.layers.1.mlp.experts.{e}.down_proj.{k}": "m.safetensors" for e in (0, 1) for k in ("weight", "bias")})
    keys.update({f"model.layers.2.mlp.up_proj.{k}": "m.safetensors" for k in ("weight", "scales", "biases", "bias")})
    monkeypatch.setattr(X, "_index", lambda src: keys)
    parent = types.SimpleNamespace(up_proj=object())
    host = types.SimpleNamespace(sites=[Site(0, "dense", 4, 1, None, parent), Site(1, "router_experts", 2, 1, None, None),
                                        Site(2, "dense", 4, 1, None, parent)])
    core = [np.array([1, 0, 1, 0], np.float32), np.array([0, 1], np.float32), np.array([0, 1, 1, 1], np.float32)]
    got = {k: (how, idx) for k, how, idx in X.plan(host, core, Path("/nonexistent"))}
    assert got == {"model.layers.0.mlp.up_proj.weight": ("rows", [1, 3]),
                   "model.layers.0.mlp.up_proj.bias": ("rows", [1, 3]),
                   "model.layers.1.mlp.experts.0.down_proj.weight": ("whole", None),
                   "model.layers.1.mlp.experts.0.down_proj.bias": ("whole", None),
                   # quantized: scales and biases rows and the Linear's own bias; the packed weight stays
                   "model.layers.2.mlp.up_proj.scales": ("rows", [0]),
                   "model.layers.2.mlp.up_proj.biases": ("rows", [0]),
                   "model.layers.2.mlp.up_proj.bias": ("rows", [0])}


def test_export_of_a_biased_host_makes_dropped_units_contribute_nothing(tmp_path):
    m = _biased_toy()
    src = tmp_path / "src"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), dict(nn.utils.tree_flatten(m.parameters())),
                        metadata={"format": "mlx"})
    (src / "config.json").write_text(json.dumps({"model_type": "toy"}))
    h = _host(m)
    rng = np.random.default_rng(3)
    core = [(rng.random(n) > 0.4).astype(np.float32) for n in h.widths]
    rep = X.export_dense_core(h, src, core, tmp_path / "dst", log=lambda *_: None, verify_ids=TC.ids_batch().tolist())
    ref = np.load(tmp_path / "dst" / "carve_verify_reference.npz")
    out = mx.load(str(tmp_path / "dst" / "model.safetensors"))
    m2 = _biased_toy()
    m2.load_weights(list(out.items()))
    got = np.array(m2(mx.array(ref["ids"])).astype(mx.float32))
    assert np.abs(got - ref["logits"]).max() < 1e-4           # the export is the masked research forward
    assert np.abs(np.array(m(mx.array(ref["ids"])).astype(mx.float32)) - got).max() > 1e-3
    # dropped units: weight row and bias both exactly zero, on the unit axis of each projection
    d0 = np.where(core[0] == 0)[0]
    assert not np.array(out["model.layers.0.mlp.up_proj.weight"])[d0].any()
    assert not np.array(out["model.layers.0.mlp.up_proj.bias"])[d0].any()
    e1 = np.where(core[1] == 0)[0]
    assert not np.array(out["model.layers.1.mlp.switch_mlp.down_proj.weight"])[e1].any()
    assert not np.array(out["model.layers.1.mlp.switch_mlp.down_proj.bias"])[e1].any()
    k1 = np.where(core[1] == 1)[0]
    assert np.array_equal(np.array(out["model.layers.1.mlp.switch_mlp.down_proj.bias"])[k1],
                          np.array(m.model.layers[1].mlp.switch_mlp.down_proj.bias)[k1])
    assert "model.layers.0.mlp.up_proj.weight" in rep["edited_keys"] and "bias" in rep["zeroing"]


def test_export_neuron_units_zero_a_stacked_up_bias(tmp_path):
    NU = pytest.importorskip("test_amoeba_carve_neuron_units_cpu")
    m = NU.qn_toy()
    for i in (1, 3):
        up = NU_switch(m, i).up_proj
        up.bias = mx.random.normal((up.weight.shape[0], up.weight.shape[1])) * 0.5
    mx.eval(m.parameters())
    src = tmp_path / "src"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), dict(nn.utils.tree_flatten(m.parameters())),
                        metadata={"format": "mlx"})
    (src / "config.json").write_text(json.dumps({"model_type": "toy"}))
    h = NU.host(m)
    rng = np.random.default_rng(5)
    core = [(rng.random(n) > 0.4).astype(np.float32) for n in h.widths]
    X.export_dense_core(h, src, core, tmp_path / "dst", log=lambda *_: None, verify_ids=NU.ids_batch().tolist())
    ref = np.load(tmp_path / "dst" / "carve_verify_reference.npz")
    out = mx.load(str(tmp_path / "dst" / "model.safetensors"))
    key = next(k for k in out if ".layers.1." in k and k.endswith("up_proj.bias"))
    drop = np.where(core[1] == 0)[0]
    b = np.array(out[key])
    assert not b[drop // NU.FF, drop % NU.FF].any()
    w = np.array(out[key[:-len("bias")] + "weight"])
    assert not w[drop // NU.FF, drop % NU.FF].any()
    m2 = NU.qn_toy()
    for i in (1, 3):
        up = NU_switch(m2, i).up_proj
        up.bias = mx.zeros((up.weight.shape[0], up.weight.shape[1]))
    m2.load_weights(list(out.items()))
    got = np.array(m2(mx.array(ref["ids"])).astype(mx.float32))
    assert np.abs(got - ref["logits"]).max() < 1e-4


def NU_switch(m, i):
    mlp = m.model.layers[i].mlp
    return getattr(mlp, "switch_mlp", None) or mlp.switch_glu


# ------------------------------------------------------------------ AMCV-2, AMCV-7 (rank), completion marker
def test_tied_ranking_receipts_record_the_kept_fraction_and_the_marker_is_last(tmp_path):
    Z = -np.ones((4, TC.H), np.float32)
    Z[:, :30] = 3.0                                    # 30 of 48 units per layer tied at the top
    np.savez(tmp_path / "tied.npz", Z=Z)
    run = tmp_path / "run"
    fac = lambda log: _host(TC.toy(moe=False))  # noqa: E731
    Dm.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--corpus", f"jsonl:{_corpus(tmp_path)}",
             "--targets", "0.0001", "--bisect-iters", "2", "--ablations", "zero", "--stages", "bisect",
             "--extra-ranking", f"zero={tmp_path / 'tied.npz'}", "--size-targets", "0.1", "--out", str(run)],
            host_factory=fac)
    rep = json.loads((run / "carve.json").read_text())
    units = rep["shape"]["units"]
    t = np.load(run / "tier_zero_size0.1.npz")
    row = rep["bisect"]["zero"]["size+0.1"]
    assert row["kept_frac"] == 0.1 and float(t["frac"]) == 0.1            # the requested size, unchanged
    assert int(t["core"].sum()) == 120 == row["kept_units"] == int(t["kept_units"])   # ties at the cut stay kept
    assert row["kept_frac_actual"] == pytest.approx(120 / units) == pytest.approx(float(t["frac_actual"]))
    e = np.load(run / "export_core.npz")
    ec = rep["export_core"]
    assert ec["kept_units"] == int(e["core"].sum()) == int(e["kept_units"])
    assert ec["kept_frac_actual"] == pytest.approx(float(e["core"].sum()) / units) == pytest.approx(float(e["frac_actual"]))
    assert ec["kept_frac"] == float(e["frac"])
    lr = rep["bisect"]["zero"]["0.0001"]
    assert lr["learned_kept_frac_actual"] == pytest.approx(lr["learned_kept_units"] / units)
    assert lr["learned_kept_frac_actual"] == pytest.approx(float(np.mean(lr["learned_per_layer_kept"])))
    # completion marker: written after the final receipt
    mk = json.loads((run / "carve_complete.json").read_text())
    assert mk["schema"] == "mlx2.amoeba-carve-complete.v1" and mk["mlx2_source_revision"] == rep["mlx2_source_revision"]
    assert mk["completed_at"] and mk["stages_done"] == rep["stages_done"] == ["bisect"]
    assert (run / "carve_complete.json").stat().st_mtime_ns >= (run / "carve.json").stat().st_mtime_ns
    assert not list(run.glob("carve_complete.json.tmp*"))


def test_no_completion_marker_when_the_carve_dies_after_export_core(tmp_path, monkeypatch):
    Z = np.random.default_rng(1).normal(size=(4, TC.H)).astype(np.float32)
    np.savez(tmp_path / "z.npz", Z=Z)
    run = tmp_path / "run"
    real = Dm._random_like
    calls = []

    def dying(M, fixed, pool, host, seed):              # the export core's random control runs after export_core.npz
        calls.append(seed)
        if (run / "export_core.npz").exists():
            raise RuntimeError("killed between export_core.npz and its held evaluations")
        return real(M, fixed, pool, host, seed)
    monkeypatch.setattr(Dm, "_random_like", dying)
    with pytest.raises(RuntimeError, match="killed"):
        Dm.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--corpus", f"jsonl:{_corpus(tmp_path)}",
                 "--targets", "", "--bisect-iters", "2", "--ablations", "zero", "--stages", "bisect",
                 "--extra-ranking", f"zero={tmp_path / 'z.npz'}", "--out", str(run)],
                host_factory=lambda log: _host(TC.toy(moe=False)))
    assert (run / "export_core.npz").exists() and not (run / "carve_complete.json").exists()


def test_resumed_carve_that_dies_drops_the_earlier_marker(tmp_path, monkeypatch):
    run = tmp_path / "run"
    base = ["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--corpus", f"jsonl:{_corpus(tmp_path)}",
            "--targets", "", "--bisect-iters", "2", "--ablations", "zero", "--out", str(run)]
    fac = lambda log: _host(TC.toy(moe=False))  # noqa: E731
    Dm.main(base + ["--stages", "mean"], host_factory=fac)
    assert json.loads((run / "carve_complete.json").read_text())["stages_done"] == ["mean"]

    def dying(*a, **k):
        raise RuntimeError("killed in the bisect")
    monkeypatch.setattr(Dm, "_random_like", dying)
    np.savez(tmp_path / "z.npz", Z=np.random.default_rng(1).normal(size=(4, TC.H)).astype(np.float32))
    with pytest.raises(RuntimeError, match="killed"):
        Dm.main(base + ["--stages", "bisect", "--resume", "--extra-ranking", f"zero={tmp_path / 'z.npz'}"],
                host_factory=fac)
    assert (run / "export_core.npz").exists() and not (run / "carve_complete.json").exists()


# ------------------------------------------------------------------ AMCV-3
class AttnOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(TC.D)
        self.q, self.k, self.v, self.o = (nn.Linear(TC.D, TC.D, bias=False) for _ in range(4))

    def __call__(self, x, mask=None, cache=None):
        h = self.input_layernorm(x)
        B, T, _ = h.shape
        q, k, v = (p(h).reshape(B, T, 2, TC.D // 2).transpose(0, 2, 1, 3) for p in (self.q, self.k, self.v))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(TC.D // 2) ** -0.5, mask=mask)
        return x + self.o(a.transpose(0, 2, 1, 3).reshape(B, T, TC.D))


def test_hybrid_donor_with_siteless_layers_is_ragged():
    m = TC.toy(moe=False)
    m.model.layers[1], m.model.layers[3] = AttnOnly(), AttnOnly()
    mx.eval(m.parameters())
    h = _host(m)
    assert [s.kind for s in h.sites] == ["dense", None, "dense", None] and h.widths == [TC.H, 0, TC.H, 0]
    assert h.N is None                                  # equal sited widths, but not a uniform [L, N] layout
    assert sum(int(r.size) for r in h.ones()) == sum(h.widths)       # no phantom units in siteless layers
    ids = TC.ids_batch()
    h.check_paths(ids)
    mean = Ev.mean_acts(h, np.array(ids).tolist())
    assert [int(r.shape[0]) for r in mean] == h.widths
    assert h.check_layerwise_grad(ids[:1], tol=2e-3), h.paths


# ------------------------------------------------------------------ AMCV-4, AMCV-5
def test_sweep_driver_options_bind_to_every_family_loader():
    from research.amoeba.sweep.family import FAMILIES, _loader, family_options
    asked = {"expert_budget_gb": 24, "ffn_texts": 1}
    took = {}
    for name in sorted(FAMILIES):
        opts, dropped = family_options(name, asked)
        assert set(opts) | dropped == set(asked) and not set(opts) & dropped
        mod, loader = _loader(name)
        sig = inspect.signature(loader)
        sig.bind_partial("/nonexistent", **opts)
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):     # forwarded to the family class
            cls = getattr(mod, sig.return_annotation)
            inspect.signature(cls.__init__).bind_partial(None, "/nonexistent", **opts)
        took[name] = set(opts)
    assert took["deepseek_v41"] == {"expert_budget_gb", "ffn_texts"} and took["glm5_next"] == {"expert_budget_gb", "ffn_texts"}
    assert took["mimo_v2"] == set() and took["hy_v4_flash"] == set()        # the TypeError of the old driver
    assert took["kimi_linear"] == set() and took["nemotron_h"] == set()


def test_sweep_driver_refuses_begin_layer_families_before_loading(tmp_path):
    from research.amoeba.sweep import capture as C
    import signal
    (tmp_path / "c.jsonl").write_text("")
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}    # main installs its own
    try:
        with pytest.raises(SystemExit, match="child_rolling"):
            C.main(["--family", "kimi_linear", "--model", "/nonexistent-model-dir", "--corpus", str(tmp_path / "c.jsonl"),
                    "--out", str(tmp_path / "o"), "--state-dir", str(tmp_path / "s")])
    finally:
        for sig, h in saved.items():
            signal.signal(sig, h)


def test_sweep_nll_rows_scores_targets_2_to_T_minus_1():
    from research.amoeba.carve.carving import ce_rows
    from research.amoeba.sweep import capture as C
    rng = np.random.default_rng(7)
    logits = mx.array(rng.normal(size=(3, 9, 17)).astype(np.float32) * 3)
    ids = mx.array(rng.integers(0, 17, (3, 9)).astype(np.int32))
    got = np.array(C.nll_rows(logits, ids))
    want = np.array(ce_rows(logits[:, 1:-1], ids[:, 2:]).mean(1))          # TopicsBackend.nll
    assert np.allclose(got, want, atol=1e-5)
    old = np.array(ce_rows(logits[:, :-1], ids[:, 1:]).mean(1))
    assert not np.allclose(got, old, atol=1e-3)
    assert C.NLL_CONVENTION == "targets 2..T-1"


# ------------------------------------------------------------------ AMCV-6
def test_later_watchdog_calls_tighten_and_record_what_is_in_force(monkeypatch):
    from research.amoeba.carve import memwatch as M
    monkeypatch.setenv("MLX2_AMOEBA_WATCHDOG", "on")
    monkeypatch.setattr(O, "_WATCHDOG_STARTED", False)
    monkeypatch.setattr(O, "_WATCHER", [None])
    monkeypatch.setattr(O, "WATCH", {})
    monkeypatch.setattr(O, "swap_used_gb", lambda: 1.0)
    monkeypatch.setattr(M, "readings", lambda pid: {"swap_used_gib": 1.0, "compressor_gib": 3.0})

    class Stub:                                           # no thread, no watcher process
        def __init__(self, lim, log, **kw):
            self.lim, self.stats = lim, kw["stats"]

        def rebase(self):
            self.lim["swap0_gib"] = 1.0
    monkeypatch.setattr(M, "Watch", Stub)
    logs = []
    O.footprint_watchdog(92, logs.append)                 # a batch starts the process's watchdog
    lim = O._WATCHER[0].lim
    assert (lim["limit_gib"], lim["hard_gib"], lim["swap_growth_gib"]) == (92.0, 98.0, 2.0)
    O.footprint_watchdog(60, logs.append, swap_growth_gb=6.0, min_avail_gb=20.0)     # a carve inside the batch
    assert (lim["limit_gib"], lim["hard_gib"], lim["min_avail_gib"]) == (60.0, 66.0, 20.0)     # stricter: enforced
    assert lim["swap_growth_gib"] == 2.0                                          # looser: not applied
    assert O.WATCH["limit_gb"] == 60.0 and O.WATCH["swap_growth_gb"] == 2.0 and O.WATCH["min_avail_gb"] == 20.0
    assert O.WATCH["requested_ignored"] == {"swap_growth_gb": {"requested": 6.0, "in_force": 2.0}}
    assert any("looser limits not applied" in str(x) for x in logs)
    O.footprint_watchdog(100, logs.append)                # looser cap: the 60 stays, and the receipt says so
    assert lim["limit_gib"] == 60.0 and lim["hard_gib"] == 66.0
    assert O.WATCH["requested_ignored"] == {"limit_gb": {"requested": 100.0, "in_force": 60.0},
                                            "min_avail_gb": {"requested": 15.0, "in_force": 20.0}}
    O.footprint_watchdog(60, logs.append, min_avail_gb=20.0)
    assert "requested_ignored" not in O.WATCH


def test_carve_waiver_receipt_records_the_swap_limit_in_force(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "WATCH", {"swap_growth_gb": 2.0, "limit_gb": 92.0})   # an earlier, stricter watchdog
    run = tmp_path / "run"
    Dm.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--corpus", f"jsonl:{_corpus(tmp_path)}",
             "--stages", "mean", "--swap-growth-gb", "6", "--swap-waiver-note", "test waiver", "--out", str(run)],
            host_factory=lambda log: _host(TC.toy(moe=False)))
    w = json.loads((run / "carve.json").read_text())["watchdog_waiver"]
    assert w == {"swap_growth_gb": 2.0, "requested_swap_growth_gb": 6.0, "reason": "test waiver"}


# ------------------------------------------------------------------ AMCV-7
@pytest.mark.parametrize("stages,want", [("fisher", 4), ("fisher,rank", 12), ("mean", 0)])
def test_inputs_manifest_flags_the_fisher_capture_sequences(tmp_path, stages, want):
    run = tmp_path / "run"
    Dm.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--corpus", f"jsonl:{_corpus(tmp_path)}",
             "--windows", "2", "--ablations", "zero", "--stages", stages, "--bs", "4", "--out", str(run)],
            host_factory=lambda log: _host(TC.toy(moe=False)))
    rows = [json.loads(l) for l in (run / "inputs.jsonl").read_text().splitlines()]
    fit = [r for r in rows if r["split"] == "fit"]
    rep = json.loads((run / "carve.json").read_text())
    inp = rep["corpus"]["inputs"]
    if stages == "fisher":                               # smoke: the capture reads fit[:4]; no rank walk
        assert rep["fisher"]["capture"]["zero"] is not None
        assert inp["trained_by"] == {"rank": 0, "fisher_capture": 4}
    assert sum(1 for r in fit if r["trained"]) == min(want, len(fit))
    assert inp["trained_sequences"] == want


# ------------------------------------------------------------------ AMCV-8
def test_init_from_honours_the_npz_key(tmp_path):
    h = _host(TC.toy(moe=False))
    rng = np.random.default_rng(2)
    Z, seed = rng.normal(size=(4, TC.H)).astype(np.float32), rng.normal(size=(4, TC.H)).astype(np.float32)
    np.savez(tmp_path / "f.npz", Z=Z, seed=seed)
    np.savez(tmp_path / "noz.npz", seed=seed)
    got = np.array(Dm._init_z(f"{tmp_path / 'f.npz'}:seed", "zero", tmp_path, h, raw=True))
    assert np.array_equal(got, np.clip(seed, -3, 3))
    assert np.array_equal(np.array(Dm._init_z(str(tmp_path / "f.npz"), "zero", tmp_path, h, raw=True)), np.clip(Z, -3, 3))
    assert np.array_equal(np.array(Dm._init_z(f"{tmp_path / 'noz.npz'}:seed", "zero", tmp_path, h, raw=True)),
                          np.clip(seed, -3, 3))
    with pytest.raises(SystemExit, match="no array 'Z'"):
        Dm._init_z(str(tmp_path / "noz.npz"), "zero", tmp_path, h, raw=True)


# ------------------------------------------------------------------ AMCV-9
def test_m0_heldout_windows_come_from_the_held_rows(tmp_path, monkeypatch):
    m0 = pytest.importorskip("research.amoeba.m0")
    monkeypatch.setattr(m0, "DATA", tmp_path)
    for fi, name in enumerate(m0.Mix.WEIGHTS):
        n = 6100 if fi < 2 else 50                        # two files with a held-out part, three without
        with open(tmp_path / name, "w") as f:
            for r in range(n):
                f.write(json.dumps({"text": f"{fi}:{r}"}) + "\n")

    def tok(text, add_special_tokens=False):              # one token per row: file * 10000 + row
        fi, r = text.split(":")
        return {"input_ids": [int(fi) * 10000 + int(r)]}
    mix = m0.Mix(tok, T=8, max_rows=40000)
    ids, label = mix.held_batch(16)
    got = set(np.array(ids).reshape(-1).tolist()) - {151643}
    train = set()
    for s in mix.streams.values():
        train |= set(s.tolist())
    assert got and not got & (train - {151643})           # disjoint from every training stream
    assert all(v % 10000 >= 4100 and v // 10000 < 2 for v in got) and label.startswith("heldout rows of ")
    assert set(np.array(mix.batch(16)).reshape(-1).tolist()) <= train
    for name in list(m0.Mix.WEIGHTS)[:2]:                 # no file has a held-out part: labelled, not claimed
        (tmp_path / name).write_text("\n".join(json.dumps({"text": f"0:{r}"}) for r in range(50)) + "\n")
    _, label = m0.Mix(tok, T=8).held_batch(4)
    assert "not held out" in label


# ------------------------------------------------------------------ review R4: the nll_full label follows the array

def test_nll_convention_is_recorded_only_by_the_sweep_that_saved_the_array(tmp_path):
    from types import SimpleNamespace
    from research.amoeba.sweep import capture as C
    rng = np.random.default_rng(11)
    emb = mx.array(rng.normal(size=(23, 6)).astype(np.float32))
    head = mx.array(rng.normal(size=(6, 23)).astype(np.float32))
    fam = SimpleNamespace(L=2, widths=[4, 4], kinds=[None, None], embed=lambda ids: emb[ids],
                          run_layer=lambda l, c: (c["h"] * 1.5, {}), release_layer=lambda l: None,
                          final_logits=lambda c: c @ head)
    seqs = [[int(t) for t in rng.integers(0, 23, 7)] for _ in range(3)]
    out, st = tmp_path / "out", tmp_path / "state"
    out.mkdir()
    np.save(out / "mu.npy", np.zeros((2, 4), np.float32))
    receipt = {}
    assert C.nll_meta(receipt) == {}
    C.sweep(fam, seqs, ["fit"] * 3, out, st, log=lambda m: None, receipt=receipt, parity=True)
    assert (out / "nll_full.npy").exists() and receipt["nll_full"] == {"convention": C.NLL_CONVENTION}
    assert C.nll_meta(receipt)["nll_full"]["convention"] == "targets 2..T-1"
    # an older capture (no label on its receipt) resumed after its last layer: this run writes no nll_full.npy
    before = (out / "nll_full.npy").read_bytes()
    old = {}
    final = C.sweep(fam, seqs, ["fit"] * 3, out, st, log=lambda m: None, receipt=old, resume=True, parity=True)
    assert final is not None and (out / "nll_full.npy").read_bytes() == before
    assert "nll_full" not in old and C.nll_meta(old) == {}
