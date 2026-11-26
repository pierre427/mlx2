"""H45c LoopRes read (model.Core._looped_res, loop_train arms loopres / loopres_content, h45c_kill): CPU oracles."""
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import families as F
from research.ens import h45c_kill as K
from research.ens import loop_train as L
from research.ens.model import BusCall, Core, CoreConfig, loopres_roles, value_logits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _cfg(**kw):
    return CoreConfig(d=32, layers=6, heads=2, ffn=64, bus_width=8, loop_body=(2, 4), **kw)


def _core(read="loopres", seed=0):
    mx.random.seed(seed)
    return Core(_cfg(loop_read=read))


def _batch(n=6):
    rng = random.Random(0)
    rows = [F.sample_row(F.BY_ID["graph.6-2"], rng) for _ in range(n)]
    tokens, q, f, a, ans, zs = C.batch(rows, "bus")
    return tokens, q, f, a, ans, mx.random.normal((n, 8), key=mx.random.key(5))


def test_default_read_is_byte_identical_to_the_committed_h45_core(tmp_path):
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True, text=True).stdout  # noqa
    rev = "HEAD"
    if "loop_read" in git("show", "HEAD:research/ens/model.py"):      # once H45c is committed: the revision before
        rev = git("log", "-n1", "--format=%H", "-S", "loop_read: str", "--", "research/ens/model.py").strip() + "^"
    src = git("show", f"{rev}:research/ens/model.py")
    assert "loop_read" not in src
    (tmp_path / "old_model.py").write_text(src)
    spec = importlib.util.spec_from_file_location("old_model", tmp_path / "old_model.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules["old_model"] = old                    # dataclasses resolve annotations through sys.modules
    try:
        spec.loader.exec_module(old)
    finally:
        del sys.modules["old_model"]
    tokens, q, f, a, ans, msg = _batch()
    mixed = mx.array([1, 2, 3, 1, 3, 2])
    mx.random.seed(7)
    want = old.Core(old.CoreConfig(d=32, layers=6, heads=2, ffn=64, bus_width=8, loop_body=(2, 4)))
    mx.random.seed(7)
    got = Core(_cfg())
    assert [k for k, _ in tree_flatten(got.parameters())] == [k for k, _ in tree_flatten(want.parameters())]
    for passes in (None, mixed):
        b = BusCall("early", q, f, msg)
        lw = want(tokens, b, passes=passes) if passes is not None else want(tokens, b)
        lg = got(tokens, b, passes=passes) if passes is not None else got(tokens, b)
        assert bool(mx.array_equal(lw, lg).item())


def test_backbone_init_and_global_rng_match_the_loop_arm():
    mx.random.seed(11)
    plain = Core(_cfg())
    after_plain = mx.random.uniform(shape=(4,))
    mx.random.seed(11)
    res = Core(_cfg(loop_read="loopres", loop_key_seed=3))
    after_res = mx.random.uniform(shape=(4,))
    assert bool(mx.array_equal(after_plain, after_res).item())        # role keys come from their own key
    for (k, v), (_, w) in zip(tree_flatten(plain.backbone.parameters()), tree_flatten(res.backbone.parameters())):
        assert bool(mx.array_equal(v, w).item()), k
    assert res.loopres_keys.shape == (len(loopres_roles(2)), 32)


def test_untrained_read_is_a_uniform_mix_and_gets_gradients():
    core = _core()
    assert float(mx.max(mx.abs(core.loopres_qc)).item()) == 0.0 and float(mx.max(mx.abs(core.loopres_qr)).item()) == 0.0
    vs = [mx.random.normal((2, 5, 32), key=mx.random.key(i)) for i in range(4)]
    srcs = [(i, v, mx.rsqrt(mx.mean(v * v, axis=-1) + 1e-5)) for i, v in enumerate(vs)]
    assert mx.allclose(core._loopres_read(1, srcs), sum(vs) / 4, atol=1e-6).item()
    content = _core("loopres_content")
    assert "loopres_qr" not in content and "loopres_keys" not in content
    assert mx.allclose(content._loopres_read(1, srcs), sum(vs) / 4, atol=1e-6).item()

    tokens, q, f, a, ans, msg = _batch()
    passes = mx.array([1, 2, 3, 1, 3, 2])

    def loss(model):
        logits = model(tokens, BusCall("early", q, f, msg), passes=passes)
        return nn.losses.cross_entropy(value_logits(logits, a), ans, reduction="mean")
    for m in (core, content):
        g = dict(tree_flatten(nn.value_and_grad(m, loss)(m)[1]))
        assert float(mx.sum(mx.abs(g["loopres_qc"])).item()) > 0
        for r in range(5):                                   # every sub-layer query and the readout learn
            assert float(mx.sum(mx.abs(g["loopres_qc"][r])).item()) > 0, r
        if "loopres_qr" in m:
            assert float(mx.sum(mx.abs(g["loopres_qr"])).item()) > 0
            # keys see gradient through q_role only: zero at init, nonzero once the queries move
            assert float(mx.max(mx.abs(g["loopres_keys"])).item()) == 0.0
    core.loopres_qr = core.loopres_qr + 0.1
    g = dict(tree_flatten(nn.value_and_grad(core, loss)(core)[1]))
    assert float(mx.sum(mx.abs(g["loopres_keys"])).item()) > 0


def test_loop_body_has_no_residual_stream(monkeypatch):
    """loopres (paper form): every sub-layer input is exactly the uniform (untrained) mix of p, the previous loop's
    block outputs, this loop's earlier block outputs and the block's partial sum; block outputs are sums of
    sub-layer outputs; the coda reads the loop output mix. No running state enters anywhere."""
    from research.ens import model as Mo
    core = _core()
    calls = []
    real_a, real_m = Mo.Block.attn_update, Mo.Block.mlp_update

    def rec(kind, fn):
        def f(self, x):
            y = fn(self, x)
            calls.append((kind, x, y))
            return y
        return f
    monkeypatch.setattr(Mo.Block, "attn_update", rec("a", real_a))
    monkeypatch.setattr(Mo.Block, "mlp_update", rec("m", real_m))
    tokens = _batch()[0]
    logits = core(tokens, passes=mx.full((6,), 2))
    assert [c[0] for c in calls] == ["a", "m"] * 4
    x = [c[1] for c in calls]
    y = [c[2] for c in calls]
    p = core.prefix(tokens, "early")                   # the prelude output: blocks 0-1 (loop body is 2-3)
    b0, b1 = y[0] + y[1], y[2] + y[3]
    c0, c1 = y[4] + y[5], y[6] + y[7]
    mean = lambda *v: sum(v) / len(v)                  # noqa: E731
    want = [p, mean(p, y[0]), mean(p, b0), mean(p, b0, y[2]),
            mean(p, b0, b1), mean(p, b0, b1, y[4]), mean(p, b0, b1, c0), mean(p, b0, b1, c0, y[6])]
    for i, (got, w) in enumerate(zip(x, want)):
        assert mx.allclose(got, w, atol=1e-5).item(), i
    bb = core.backbone
    out = mean(p, b0, b1, c0, c1)
    for blk in bb.blocks[4:]:
        out = blk(out)
    assert mx.allclose(logits, bb.head(bb.norm(out)), atol=1e-4).item()
    # the additive variant keeps H45's stream: its first MLP reads (p + p + a) / 2, not (p + a) / 2
    calls.clear()
    add = _core("loopres_add")
    add(tokens, passes=mx.full((6,), 1))
    assert mx.allclose(calls[1][1], mean(p, p + calls[0][2]), atol=1e-5).item()


@pytest.mark.parametrize("read", ["loopres", "loopres_content", "loopres_add"])
def test_per_row_passes_and_unseen_pass_counts(read):
    core = _core(read)
    core.loopres_qc = mx.random.normal(core.loopres_qc.shape, key=mx.random.key(1)) * 0.3   # a non-uniform read
    if "loopres_qr" in core:
        core.loopres_qr = mx.random.normal(core.loopres_qr.shape, key=mx.random.key(2))
    tokens, q, f, a, ans, msg = _batch()
    bus = BusCall("early", q, f, msg)
    mixed = mx.array([1, 2, 3, 1, 3, 2])
    got = core(tokens, bus, passes=mixed)
    for k in (1, 2, 3):
        uni = core(tokens, bus, passes=mx.full((6,), k))
        rows = mx.array([i for i in range(6) if int(mixed[i].item()) == k])
        assert float(mx.max(mx.abs(got[rows] - uni[rows])).item()) < 1e-5
    assert bool(mx.array_equal(core(tokens, bus, passes=mixed, max_passes=3), got).item())
    six = core(tokens, bus, passes=mx.full((6,), 6))             # never trained: same roles, same parameters
    three = core(tokens, bus, passes=mx.full((6,), 3))
    assert bool(mx.all(mx.isfinite(six)).item()) and float(mx.max(mx.abs(six - three)).item()) > 0
    _, states = core.looped(tokens, mx.full((6,), 6), bus, return_states=True)
    assert states.shape[0] == 6
    with pytest.raises(ValueError):                              # the bus still writes before the loop body only
        core(tokens, BusCall("middle", q, f, msg), passes=mixed)
    with pytest.raises(ValueError):
        Core(_cfg(loop_read="loopres", loop_gain="decay"))


def test_one_pass_accuracy_runs_the_read():
    """A LoopRes core has no plain one-pass graph: acc(passes=None) runs its loop port at one pass."""
    core = _core()
    tokens, q, f, a, ans, msg = _batch()
    bus = BusCall("early", q, f, msg)
    one = core(tokens, bus, passes=mx.ones((6,), dtype=mx.int32))
    assert bool(mx.all(mx.isfinite(one)).item())
    assert L.acc(core, tokens, q, f, a, ans, msg, None) == L.acc(core, tokens, q, f, a, ans, msg,
                                                                mx.ones((6,), dtype=mx.int32))


def test_few_steps_overfit_a_tiny_set():
    core = _core()
    tokens, q, f, a, ans, msg = _batch(8)
    passes = mx.array([1, 2, 3, 2, 1, 3, 2, 1])
    opt = optim.Adam(learning_rate=3e-3)

    def loss(model):
        logits = model(tokens, BusCall("early", q, f, msg), passes=passes, max_passes=3)
        return nn.losses.cross_entropy(value_logits(logits, a), ans, reduction="mean")
    vg = nn.value_and_grad(core, loss)
    first = None
    for _ in range(40):
        l, g = vg(core)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(core, g)
        mx.eval(core.parameters(), opt.state)
        first = float(l.item()) if first is None else first
    assert float(loss(core).item()) < 0.5 * first


def test_accum_two_matches_accum_one(tmp_path):
    base = ["--arm", "loopres", "--seed", "0", "--steps", "4", "--batch", "8", "--log-every", "100",
            "--stop-after", "2"]
    for acc_k in (1, 2):
        with pytest.raises(SystemExit) as e:
            L.main(base + ["--accum", str(acc_k), "--out", str(tmp_path / f"a{acc_k}")])
        assert e.value.code == 3
    w1 = mx.load(str(tmp_path / "a1" / "resume" / "core.safetensors"))
    w2 = mx.load(str(tmp_path / "a2" / "resume" / "core.safetensors"))
    assert "loopres_keys" in w1 and "loopres_qc" in w1
    init = Core(CoreConfig(bus_width=64, tau=0.5, layers=6, d=256, ffn=1024, heads=4, loop_body=L.LOOP_BODY,
                           loop_read="loopres"))
    assert float(mx.max(mx.abs(w1["loopres_qc"])).item()) > 0          # the read trained
    assert init.loopres_qc.shape == w1["loopres_qc"].shape
    for k in w1:
        assert mx.allclose(w1[k], w2[k], atol=2e-5, rtol=1e-4).item(), k


def test_receipt_records_the_read(tmp_path):
    with pytest.raises(SystemExit) as e:
        L.main(["--arm", "loopres_content", "--seed", "1", "--steps", "2", "--batch", "8", "--log-every", "1",
                "--out", str(tmp_path / "r")])
    assert e.value.code == 0
    r = json.loads((tmp_path / "r" / "receipt.json").read_text())
    assert r["loop_read"] == "loopres_content" and r["config"]["loop_read"] == "loopres_content"
    assert r["config"]["loop_key_seed"] == 1 and r["blocks_per_row_train"] == L.BLOCKS_PER_ROW["loop"]
    s = r["summary"]["ALL.unseen"]
    assert {"bus", "ordinary", "bus_one_pass", "bus_minus1_vs_s"} <= set(s)


def _fake(root, arm, seed, unseen, trained, ordinary=0.9, one=0.5, **args):
    d = root / f"{arm}-s{seed}"
    d.mkdir(parents=True)
    blk = lambda acc: {"bus": acc, "ordinary": ordinary, "bus_one_pass": one, "ordinary_one_pass": one,  # noqa
                       "op_guess": 0.1}
    summ = {"ALL.unseen": blk(unseen), "ALL.trained": blk(trained)}
    for c in ("GRAPH", "FST", "PROGRAM"):
        summ[f"{c}.unseen"] = {"bus": unseen}
    a = {"steps": 48000, "batch": 256, "accum": 4, "lr": 1e-3, "p_ordinary": 0.25, "warmup": 500} | args
    (d / "receipt.json").write_text(json.dumps({"summary": summ, "args": a, "machine": "x86_64",
                                                "device": "Device(gpu, 0)"}))


def test_kill_script_on_fake_receipts(tmp_path):
    import io
    for s in (0, 1):
        _fake(tmp_path, "loop", s, 0.64, 0.90, one=0.40)
        _fake(tmp_path, "loopres", s, 0.72 if s == 0 else 0.70, 0.89, ordinary=0.80, one=0.46)
        _fake(tmp_path, "loopres_content", s, 0.66, 0.90)
    log = io.StringIO()
    out = K.kills(tmp_path, [0, 1], log=log)
    assert "loopres-s1" in log.getvalue() and '"accum": 4' in log.getvalue()
    assert out["K1_depth"]["loopres_minus_loop"] == 0.07 and not out["K1_depth"]["killed"]
    # K2 is bus only: an ordinary drop of 0.10 is reported, not a kill
    assert not out["K2_cost"]["killed"] and not out["any_killed"]
    assert out["ordinary_cost_reported"]["loop_minus_loopres_trained_ordinary"] == 0.1
    assert out["one_pass_attribution"]["unseen.bus_one_pass"] == 0.06
    assert out["content_ablation"]["loopres_minus_loopres_content_unseen"] == 0.05
    assert "compute_control" not in out
    # a small gain kills K1; a trained-depth bus cost kills K2
    small = tmp_path / "small"
    for s in (0, 1):
        _fake(small, "loop", s, 0.64, 0.90)
        _fake(small, "loopres", s, 0.67, 0.87)
        _fake(small, "uniform3", s, 0.63, 0.90)
    out = K.kills(small, [0, 1], log=io.StringIO())
    assert out["K1_depth"]["killed"] and out["K2_cost"]["killed"] and out["any_killed"]
    assert "content_ablation" not in out and out["compute_control"]["uniform3_unseen_bus"] == 0.63
    with pytest.raises(SystemExit, match="missing loop receipts"):
        K.kills(small, [0, 1, 2], log=io.StringIO())
    # different training settings are refused; a missing value is an error, not a dropped seed
    bad = tmp_path / "bad"
    _fake(bad, "loop", 0, 0.64, 0.90, accum=1)
    _fake(bad, "loopres", 0, 0.70, 0.90)
    with pytest.raises(SystemExit, match="different training settings"):
        K.kills(bad, [0], log=io.StringIO())
    gap = tmp_path / "gap"
    _fake(gap, "loop", 0, 0.64, 0.90)
    _fake(gap, "loopres", 0, 0.70, 0.90)
    r = json.loads((gap / "loopres-s0" / "receipt.json").read_text())
    del r["summary"]["ALL.trained"]["bus"]
    (gap / "loopres-s0" / "receipt.json").write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="missing"):
        K.kills(gap, [0], log=io.StringIO())
    part = tmp_path / "part"
    for s in (0, 1):
        _fake(part, "loop", s, 0.64, 0.90)
        _fake(part, "loopres", s, 0.70, 0.90)
    _fake(part, "uniform3", 0, 0.63, 0.90)
    with pytest.raises(SystemExit, match="uniform3"):
        K.kills(part, [0, 1], log=io.StringIO())
