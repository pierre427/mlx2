"""CPU tests of the generic amoeba carve pipeline on a toy donor (dense + ling-style MoE layers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.amoeba.carve import export as X  # noqa: E402
from research.amoeba.carve import fisher as F  # noqa: E402
from research.amoeba.carve.carving import kl_rows  # noqa: E402
from research.amoeba.carve.host import Host, TopicsBackend  # noqa: E402

D, H, E, K, V = 32, 48, 6, 2, 64


class Args:
    tie_word_embeddings = False


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(D, H, bias=False), nn.Linear(D, H, bias=False), nn.Linear(H, D, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    training_indices = False          # as Kimi's router: stop gradients on ids only when asked

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D, E, bias=False)

    def __call__(self, x):
        p = mx.softmax(self.proj(x).astype(mx.float32), axis=-1)
        ids = mx.argpartition(-p, kth=K - 1, axis=-1)[..., :K]
        if self.training_indices:
            ids = mx.stop_gradient(ids)
        return ids, mx.take_along_axis(p, ids, axis=-1)


class Moe(nn.Module):
    def __init__(self):
        super().__init__()
        from mlx2.runtime.models.switch_layers import SwitchGLU
        self.gate = Gate()
        self.switch_mlp = SwitchGLU(D, 16, E)
        self.shared_experts = MLP()

    def __call__(self, x):
        ids, w = self.gate(x)
        y = self.switch_mlp(x, ids)
        return (y.astype(mx.float32) * w[..., None]).sum(-2).astype(x.dtype) + self.shared_experts(x)


class Layer(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = nn.RMSNorm(D), nn.RMSNorm(D)
        self.q, self.k, self.v, self.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        self.mlp = Moe() if moe else MLP()

    def __call__(self, x, mask=None, cache=None):
        h = self.input_layernorm(x)
        B, T, _ = h.shape
        q, k, v = (p(h).reshape(B, T, 2, D // 2).transpose(0, 2, 1, 3) for p in (self.q, self.k, self.v))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // 2) ** -0.5, mask=mask)
        x = x + self.o(a.transpose(0, 2, 1, 3).reshape(B, T, D))
        return x + self.mlp(self.post_attention_layernorm(x))


class Inner(nn.Module):
    def __init__(self, moe=True):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, D)
        self.layers = [Layer(moe=moe and (i % 2 == 1)) for i in range(4)]
        self.norm = nn.RMSNorm(D)

    def __call__(self, ids):
        h = self.embed_tokens(ids)
        for layer in self.layers:
            h = layer(h, mask="causal")
        return self.norm(h)


class Toy(nn.Module):
    def __init__(self, moe=True):
        super().__init__()
        self.model = Inner(moe)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.args = Args()

    def __call__(self, ids):
        return self.lm_head(self.model(ids))


class Tok:
    def encode(self, text, add_special_tokens=False):
        return [b % V for b in text.encode()]

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


def toy(seed=0, moe=True):
    mx.random.seed(seed)
    m = Toy(moe)
    mx.eval(m.parameters())
    return m


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def ids_batch(n=2, T=12, seed=1):
    return mx.array(np.random.default_rng(seed).integers(0, V, (n, T)).astype(np.int32))


def test_sites_paths_and_parity():
    h = Host(model=toy(), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    assert [s.kind for s in h.sites] == ["dense", "ling", "dense", "ling"]
    assert h.widths == [H, E, H, E] and h.N is None
    ids = ids_batch()
    p = h.check_paths(ids)
    assert p["fast_head"] and p["replay"]
    par = h.check_ones_parity(ids)
    assert par["top1_agree"] == 1.0 and par["max_abs"] < 1e-4


def test_dense_mask_matches_zeroed_up_row():
    m = toy()
    h = Host(model=m, tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    ids = ids_batch()
    core = [mx.ones((n,)) for n in h.widths]
    core[0] = mx.array((np.arange(H) % 3 != 0).astype(np.float32))
    h.ctl.reset()
    h.ctl.core = core
    masked = h.logits_at(ids)
    h.ctl.reset()
    w = np.array(m.model.layers[0].mlp.up_proj.weight)
    w[np.arange(H) % 3 == 0] = 0
    m.model.layers[0].mlp.up_proj.weight = mx.array(w)
    zeroed = h.logits_at(ids)
    assert float(mx.abs(masked - zeroed).max()) < 1e-5


def test_layerwise_grad_matches_whole_graph():
    h = Host(model=toy(), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    assert h.check_layerwise_grad(ids, tol=1e-3)


def test_fisher_capture_and_carve(tmp_path):
    h = Host(model=toy(), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    seqs = [[int(t) for t in np.random.default_rng(i).integers(0, V, 12)] for i in range(24)]
    h.check_paths(mx.array(seqs[:2]))
    h.check_layerwise_grad(mx.array(seqs[:1]))
    F.capture(h, seqs, tmp_path / "U", seed=0, log=lambda *_: None)
    Us = F.load(tmp_path / "U", h.L)
    assert [u.shape for u in Us] == [(24, n) for n in h.widths]
    fixed = [np.zeros(n, np.float32) for n in h.widths]
    fixed[0][:5] = 1
    Zs, rep = F.carve(Us, fixed, None, lam=1e-4, T1=11, steps=50, log=lambda *_: None)
    assert all((Zs[0][:5] == 3.0))                          # fixed units never move
    assert 0 < rep["kept"] <= 1
    # the second-order estimate tracks the real KL for small drops (same order of magnitude)
    full = h.logits_at(mx.array(seqs))
    pred, real = [], []
    for l in (0, 2):
        for i in range(0, H, 7):
            drop = np.zeros(H, np.float32)
            drop[i] = 1
            pred.append(F.cost(Us[l], drop, 11))
            core = [mx.ones((n,)) for n in h.widths]
            core[l] = mx.array(1 - drop)
            h.ctl.reset(); h.ctl.core = core; h.ctl.sink_layers = h.L
            lm = h.logits_at(mx.array(seqs))
            h.ctl.reset()
            real.append(float(kl_rows(full[:, 1:], lm[:, 1:]).mean()))
    pred, real = np.array(pred), np.array(real)
    assert np.corrcoef(np.log(pred + 1e-12), np.log(real + 1e-12))[0, 1] > 0.5


def test_export_zeroing_reproduces_mask(tmp_path):
    m = toy()
    src = tmp_path / "src"
    src.mkdir()
    flat = dict(nn.utils.tree_flatten(m.parameters()))
    # HF-like names: model.layers.{l}.mlp.* (dense) and model.layers.{l}.mlp.switch_mlp.* (stacked experts)
    mx.save_safetensors(str(src / "model.safetensors"), flat, metadata={"format": "mlx"})
    (src / "config.json").write_text(json.dumps({"model_type": "toy"}))
    h = Host(model=m, tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    rng = np.random.default_rng(3)
    core = [(rng.random(n) > 0.4).astype(np.float32) for n in h.widths]
    rep = X.export_dense_core(h, src, core, tmp_path / "dst", log=lambda *_: None, verify_ids=ids_batch().tolist())
    ref = np.load(tmp_path / "dst" / "carve_verify_reference.npz")
    m2 = Toy()
    m2.load_weights(list(mx.load(str(tmp_path / "dst" / "model.safetensors")).items()))
    got = np.array(m2(mx.array(ref["ids"])).astype(mx.float32))
    assert np.abs(got - ref["logits"]).max() < 1e-4
    assert rep["edits"] == 4


def test_topics_backend_interface():
    b = TopicsBackend(None, host=Host(model=toy(), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None))
    ids = ids_batch().tolist()
    st = b.stats(ids)
    assert [s.shape for s in st] == [(2, H), (2, E), (2, H), (2, E)]
    nll0 = b.nll(ids)
    assert nll0.shape == (len(ids),)              # one mean NLL per text (targets 2..T-1)
    b.set_mask([np.zeros(n) for n in b.h.widths])
    assert not np.allclose(b.nll(ids), nll0)


@pytest.mark.parametrize("moe", [False, True])
def test_driver_end_to_end_with_tiers_and_resume(tmp_path, moe):
    """Every stage through the driver on the toy donor (legacy corpus re-encoded by the toy tokenizer)."""
    from research.amoeba.carve import __main__ as D
    fac = lambda log: Host(model=toy(moe=moe), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    stages = "inspect,mean,rank,fisher,bisect,sink" + ("" if moe else ",atlas")
    run = tmp_path / "t1"
    base = ["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--fisher-seqs", "4", "--targets", "0.5",
            "--bisect-iters", "2", "--atlas-seqs", "16"]
    D.main(base + ["--out", str(run), "--stages", stages], host_factory=fac)
    rep = json.loads((run / "carve.json").read_text())
    assert set(rep["stages_done"]) == set(stages.split(","))
    assert (run / "export_core.npz").exists() and "chosen_stand_in" in rep["sink"]
    assert (run / "core27b.json").exists() == (not moe)
    assert {"zero", "mean", "fisher_zero", "fisher_mean"} <= set(rep["bisect"])
    # resume: nothing re-runs
    D.main(base + ["--out", str(run), "--stages", stages, "--resume"], host_factory=fac)
    assert len(json.loads((run / "carve.json").read_text())["runs"]) == 2
    # T2: T1 fixed; fixed units stay kept in every cut and in the random control
    t1 = np.load(run / "export_core.npz")["core"]
    run2 = tmp_path / "t2"
    D.main(base + ["--out", str(run2), "--stages", "mean,rank,bisect", "--ablations", "zero",
                   "--fixed", f"{run}/export_core.npz:core", "--size-targets", "0.05"], host_factory=fac)
    for f in run2.glob("tier_*.npz"):
        assert (np.load(f)["core"][t1 > 0] == 1).all()
    assert (np.load(run2 / "export_core.npz")["core"][t1 > 0] == 1).all()
    # a non-empty run dir is refused without --resume
    with pytest.raises(SystemExit):
        D.main(base + ["--out", str(run), "--stages", "mean"], host_factory=fac)


def _tiny_qwen38_27b():
    """The real Qwen3.8-27B classes (GDN + full attention, multimodal wrapper) at toy width, with the
    embedding and MLPs quantized as in the q8 build: catches what the toy donor cannot (head owner,
    nested text_config, packed embedding)."""
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
    src = Path.home() / "mlx-models/Qwen3.8-27B-BF16-q8-mlx2/config.json"
    if not src.exists():
        pytest.skip("27B q8 config not on disk")
    cfg = json.loads(src.read_text())
    cfg.pop("quantization", None)
    cfg.pop("quantization_config", None)
    t = dict(cfg["text_config"])
    t.update(hidden_size=128, intermediate_size=256, num_hidden_layers=4, head_dim=32, num_attention_heads=4,
             num_key_value_heads=2, linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2,
             linear_num_value_heads=4, vocab_size=512, mtp_num_hidden_layers=0, tie_word_embeddings=False)
    t["layer_types"] = t["layer_types"][:4]
    cfg["text_config"] = t
    mx.random.seed(0)
    m = Model(ModelArgs.from_dict(cfg))
    nn.quantize(m, group_size=64, bits=8, class_predicate=lambda p, mod: hasattr(mod, "to_quantized")
                and getattr(mod, "weight", None) is not None and mod.weight.shape[-1] % 64 == 0)
    mx.eval(m.parameters())
    return m


def test_real_27b_classes_head_and_layerwise():
    m = _tiny_qwen38_27b()
    h = Host(model=m, tokenizer=Tok(), fingerprint="tiny27:abc", log=lambda *_: None)
    assert [s.kind for s in h.sites] == ["dense"] * 4 and h.N == 256 and h.d == 128
    ids = mx.array(np.random.default_rng(0).integers(0, 512, (2, 12)).astype(np.int32))
    p = h.check_paths(ids)
    assert p["fast_head"] and p["replay"], p
    assert h.check_ones_parity(ids)["max_abs"] == 0.0
    assert h.check_layerwise_grad(ids[:1], tol=2e-3), h.paths


def test_pool_units_never_leak_into_cuts(tmp_path):
    """--pool smaller than the cut: units outside pool | fixed must stay dropped (review finding 4)."""
    from research.amoeba.carve import __main__ as D
    fac = lambda log: Host(model=toy(moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    pool = np.zeros((4, H), np.float32)
    pool[:, :6] = 1
    np.savez(tmp_path / "pool.npz", pool=pool)
    run = tmp_path / "r"
    D.main(["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--targets", "0.0001", "--bisect-iters", "2",
            "--ablations", "zero", "--stages", "rank,bisect", "--pool", f"{tmp_path}/pool.npz:pool",
            "--size-targets", "0.5", "--out", str(run)], host_factory=fac)
    for f in [run / "export_core.npz", *run.glob("tier_*.npz")]:
        assert (np.load(f)["core"][pool == 0] == 0).all(), f


def _fisher_setup(tmp_path, n=48):
    h = Host(model=toy(moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    seqs = [[int(t) for t in np.random.default_rng(100 + i).integers(0, V, 12)] for i in range(n)]
    h.check_paths(mx.array(seqs[:2]))
    h.check_layerwise_grad(mx.array(seqs[:1]))
    F.capture(h, seqs, tmp_path / "U", seed=0, log=lambda *_: None)
    return h, seqs, F.load(tmp_path / "U", h.L)


def _kl_of(h, core, seqs):
    full = h.logits_at(mx.array(seqs))
    h.ctl.reset(); h.ctl.core = core; h.ctl.sink_layers = h.L
    lm = h.logits_at(mx.array(seqs))
    h.ctl.reset()
    return float(kl_rows(full[:, 1:], lm[:, 1:]).mean())


def test_fisher_core_beats_random_same_counts(tmp_path):
    h, seqs, Us = _fisher_setup(tmp_path)
    zeros = [np.zeros(n, np.float32) for n in h.widths]
    Zs, rep = F.carve(Us, zeros, None, lam=2e-3, T1=11, steps=200, log=lambda *_: None)
    keep = [(z > 0).astype(np.float32) for z in Zs]
    assert 0.2 < rep["kept"] < 0.95, rep["kept"]
    held = [[int(t) for t in np.random.default_rng(900 + i).integers(0, V, 12)] for i in range(24)]  # not carved on
    kl_f = _kl_of(h, mx.stack([mx.array(k) for k in keep]), held)
    rng = np.random.default_rng(7)
    kl_r = []
    for _ in range(5):
        R = np.zeros((h.L, H), np.float32)
        for l, k in enumerate(keep):
            R[l, rng.choice(H, int(k.sum()), replace=False)] = 1
        kl_r.append(_kl_of(h, mx.array(R), held))
    assert kl_f < min(kl_r), (kl_f, kl_r)


def test_fisher_partial_capture_is_not_resumed(tmp_path, monkeypatch):
    h = Host(model=toy(moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    seqs = [[int(t) for t in np.random.default_rng(i).integers(0, V, 12)] for i in range(6)]
    h.check_paths(mx.array(seqs[:2]))
    calls = {"n": 0}
    real = h.value_and_grad

    def dying(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("window killed")
        return real(*a, **k)
    monkeypatch.setattr(h, "value_and_grad", dying)
    with pytest.raises(KeyboardInterrupt):
        F.capture(h, seqs, tmp_path / "U", log=lambda *_: None)
    assert not (tmp_path / "U").exists() and (tmp_path / "U.partial").exists()
    monkeypatch.setattr(h, "value_and_grad", real)
    F.capture(h, seqs, tmp_path / "U", log=lambda *_: None)       # restarts cleanly over the partial dir
    assert (tmp_path / "U" / "U_L00.npy").exists() and not (tmp_path / "U.partial").exists()


def test_fisher_carve_outside_pool_is_dropped(tmp_path):
    h, seqs, Us = _fisher_setup(tmp_path, n=8)
    fixed = [np.zeros(n, np.float32) for n in h.widths]
    fixed[0][:3] = 1
    pool = [np.zeros(n, np.float32) for n in h.widths]
    for p in pool:
        p[10:20] = 1
    Zs, _ = F.carve(Us, fixed, pool, lam=0.0, T1=11, steps=20, log=lambda *_: None)
    for l, z in enumerate(Zs):
        outside = (pool[l] == 0) & (fixed[l] == 0)
        assert (z[outside] < 0).all()
    assert (Zs[0][:3] > 0).all()


class LagunaLikeMoe(nn.Module):
    """Laguna's block shape: router -> (ids, weights), routed * scale + ungated shared expert."""

    def __init__(self):
        super().__init__()
        from mlx2.runtime.models.switch_layers import SwitchGLU
        self.scale = 2.5
        self.gate = Gate()
        self.switch_mlp = SwitchGLU(D, 16, E)
        self.shared_expert = MLP()

    def __call__(self, x):
        ids, w = self.gate(x)
        return mx.sum(self.switch_mlp(x, ids) * w[..., None], axis=-2) * self.scale + self.shared_expert(x)


def test_laguna_block_shape_is_a_unit_site():
    m = toy()
    for i in (1, 3):
        m.model.layers[i].mlp = LagunaLikeMoe()
    mx.eval(m.parameters())
    h = Host(model=m, tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    assert [s.kind for s in h.sites] == ["dense", "laguna", "dense", "laguna"]
    ids = ids_batch()
    h.check_paths(ids)
    par = h.check_ones_parity(ids)
    assert par["top1_agree"] == 1.0 and par["max_abs"] < 1e-4
    assert h.check_layerwise_grad(ids[:1], tol=1e-3)


def test_grad_compare_metrics():
    from research.amoeba.carve.host import grad_compare
    b = np.array([10.0, 1.0, -2.0])
    c = grad_compare([b + np.array([0.1, 0, 0])], [b])
    assert abs(c["rel_max"] - 0.01) < 1e-9 and c["rel_l2"] < 0.01 and c["cosine"] > 0.9999
    assert grad_compare(-b, b)["cosine"] < -0.9999


def test_fp32_activations_reference_on_bf16_donor():
    """With the residual stream in float32 the bf16 donor's mask gradients move toward the float32
    donor's; the context restores the embedding afterwards."""
    from research.amoeba.carve.host import grad_compare
    ids = ids_batch(1, 10)
    m32 = toy(moe=False)
    m16 = toy(moe=False)
    m16.set_dtype(mx.bfloat16)
    hosts = [Host(model=m, tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None) for m in (m32, m16)]
    rng = np.random.default_rng(0)
    core = mx.array((rng.random((len(hosts[0].widths), H)) > 0.3).astype(np.float32))

    def grad(h):
        h.ctl.reset()
        pos = list(range(1, ids.shape[1]))
        lc = mx.stop_gradient(h.logits_at(ids, pos).astype(mx.float32))
        _, g = h._layerwise(ids, {"core": core}, lambda l, sl: h.set_layer_masks(l, sl, None),
                            lambda hid: kl_rows(lc, h.head(hid[:, mx.array(pos)]).astype(mx.float32)).mean())
        return np.array(g["core"].astype(mx.float32))
    g32 = grad(hosts[0])
    g16 = grad(hosts[1])
    emb_cls = type(hosts[1]._embed())
    with hosts[1].fp32_activations():
        assert hosts[1].hidden(ids).dtype == mx.float32
        g16_32 = grad(hosts[1])
    assert type(hosts[1]._embed()) is emb_cls and hosts[1].hidden(ids).dtype == mx.bfloat16
    assert grad_compare(g16_32, g32)["rel_l2"] < grad_compare(g16, g32)["rel_l2"]


def test_driver_refuses_whole_graph_fallback(tmp_path, monkeypatch):
    from research.amoeba.carve import __main__ as D
    fac = lambda log: Host(model=toy(moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    monkeypatch.setattr(Host, "check_layerwise_grad", lambda self, *a, **k: False)
    monkeypatch.setattr(Host, "check_directional_grad", lambda self, *a, **k: False)
    base = ["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--stages", "inspect,rank", "--ablations", "zero", "--steps", "2"]
    with pytest.raises(SystemExit, match="whole-graph"):
        D.main(base + ["--out", str(tmp_path / "a")], host_factory=fac)
    D.main(base + ["--out", str(tmp_path / "b"), "--allow-whole-graph"], host_factory=fac)


def test_driver_refuses_on_resume_path_and_skips_without_backprop(tmp_path, monkeypatch):
    from research.amoeba.carve import __main__ as D
    fac = lambda log: Host(model=toy(moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    base = ["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--out", str(tmp_path / "r")]
    D.main(base + ["--stages", "inspect"], host_factory=fac)
    monkeypatch.setattr(Host, "check_layerwise_grad", lambda self, *a, **k: False)
    monkeypatch.setattr(Host, "check_directional_grad", lambda self, *a, **k: False)
    D.main(base + ["--stages", "inspect,mean", "--resume"], host_factory=fac)        # no backprop stage
    with pytest.raises(SystemExit, match="whole-graph"):
        D.main(base + ["--stages", "inspect,mean,rank", "--resume"], host_factory=fac)


def test_layer_gate_catches_one_wrong_layer():
    from research.amoeba.carve.host import grad_compare
    rng = np.random.default_rng(0)
    b = [rng.normal(size=50) * (1.0 if l else 1e-3) for l in range(4)]
    a = [x.copy() for x in b]
    a[0] = -a[0]                                   # tiny layer, sign-flipped: invisible in pooled L2
    c = grad_compare(a, b)
    assert c["rel_l2"] < 2e-2 and c["min_layer_cosine"] < -0.99


def test_layerwise_gate_still_fails_a_replay_defect(monkeypatch):
    """The loosened relative-L2 tolerance must not pass a broken replay: one layer's gradient wrong."""
    h = Host(model=toy(moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)
    ids = ids_batch(1, 10)
    h.check_paths(ids)
    real = Host._layerwise

    def broken(self, *a, **k):
        loss, g = real(self, *a, **k)
        c = g["core"]
        return loss, {"core": mx.concatenate([c[:1], -c[1:2], c[2:]]) if isinstance(c, mx.array) else
                      [c[0], -c[1], *c[2:]]}
    assert h.check_layerwise_grad(ids)
    monkeypatch.setattr(Host, "_layerwise", broken)
    assert not h.check_layerwise_grad(ids), h.paths["layerwise_grad"]
