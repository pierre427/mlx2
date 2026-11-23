"""Growth operators: function preservation at the moment of growth (with and without the bus), bus-site tracking,
and that the grown parts can learn (non-zero gradients)."""
from __future__ import annotations

import json
import os
import random

os.environ.setdefault("ENS_DEVICE", "cpu")

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from research.ens import common as C  # noqa: F401
from research.ens import families as F
from research.ens import growth as G
from research.ens.model import BusCall, Core, CoreConfig, value_logits

CFG = CoreConfig(d=64, layers=4, heads=4, ffn=128, bus_width=16)


def _rows(n=16):
    import random
    rng = random.Random(0)
    fams = F.split("train")
    return [F.sample_row(fams[rng.randrange(len(fams))], rng) for _ in range(n)]


def _check_same(old, new, atol):
    tokens, q, f, a, ans, zs = C.batch(_rows(), "bus")
    msg = mx.random.normal((tokens.shape[0], CFG.bus_width), key=mx.random.key(3))
    assert mx.allclose(old(tokens), new(tokens), atol=atol).item()                       # bypass
    for site in ("early", "middle"):
        assert mx.allclose(old(tokens, BusCall(site, q, f, msg)), new(tokens, BusCall(site, q, f, msg)),
                           atol=atol).item(), site


@pytest.mark.parametrize("at", [0, 1, 2, 4])
def test_insert_block_is_identity_and_tracks_sites(at):
    mx.random.seed(0)
    core = Core(CFG)
    new = G.insert_block(core, at)
    assert new.cfg.layers == CFG.layers + 1
    _check_same(core, new, 1e-5)


def test_widen_ffn_is_exact():
    mx.random.seed(1)
    core = Core(CFG)
    _check_same(core, G.widen_ffn(core, 192), 1e-5)


def test_widen_d_preserves_function_and_new_dims_can_learn():
    mx.random.seed(2)
    core = Core(CFG)
    new = G.widen_d(core, 2)
    assert new.cfg.d == CFG.d + 2 * (CFG.d // CFG.heads) and new.cfg.heads == CFG.heads + 2
    _check_same(core, new, 1e-5)
    tokens, q, f, a, ans, zs = C.batch(_rows(), "bus")
    g = nn.value_and_grad(new, lambda m: nn.losses.cross_entropy(value_logits(m(tokens), a), ans, reduction="mean"))(new)[1]
    flat = dict(tree_flatten(g))
    # the writes into the new dimensions start at zero but receive gradient, so training can use them
    assert float(mx.abs(flat["backbone.embed.weight"][:, CFG.d:]).sum().item()) > 0
    assert float(mx.abs(flat["backbone.blocks.0.o.weight"][CFG.d:]).sum().item()) > 0
    assert float(mx.abs(flat["backbone.blocks.0.down.weight"][CFG.d:]).sum().item()) > 0
    # and the new heads' outputs can start to be used
    assert float(mx.abs(flat["backbone.blocks.0.o.weight"][:CFG.d, CFG.d:]).sum().item()) > 0


def test_grown_core_keeps_core001(tmp_path):
    mx.random.seed(4)
    core = G.widen_d(G.widen_ffn(G.insert_block(Core(CFG), 2), 160), 1)
    tokens, *_ = C.batch(_rows(), "ordinary")
    assert mx.array_equal(core(tokens), core.backbone(tokens)).item()


def test_remove_block_shifts_sites():
    core = Core(CFG)
    new = G.remove_block(core, 0)
    assert new.cfg.layers == CFG.layers - 1
    assert dict(new.cfg.sites)["early"] == dict(CFG.sites)["early"] - 1


def test_growth_refuses_inexact_cases():
    import dataclasses
    import pytest
    with pytest.raises(ValueError):
        G.widen_d(Core(dataclasses.replace(CFG, reader_kind="mlp")), 1)
    with pytest.raises(ValueError):
        G.widen_ffn(Core(dataclasses.replace(CFG, lora_rank=4, lora_site="early")), 160)


def test_remove_block_moves_site_at_removed_index():
    sites = dict(CFG.sites)
    i = max(j for j in sites.values() if 0 <= j < CFG.layers)
    new = dict(G.remove_block(Core(CFG), i).cfg.sites)
    for n, j in sites.items():
        assert new[n] == (j - 1 if j >= i and j >= 0 else j)


def test_controller_with_every_window_rejected_equals_no_grow(tmp_path):
    from research.ens import grow_train as GT
    common = ["--d", "32", "--heads", "2", "--layers", "2", "--bus-width", "16", "--steps", "60", "--batch", "16",
              "--warmup", "5", "--window", "10", "--eval-every", "100", "--trial-steps", "5", "--trial-budget", "0.9"]
    def run(extra, out):
        GT.main(common + extra + ["--out", str(tmp_path / out)])
        return json.loads((tmp_path / out / "receipt.json").read_text())
    g = run(["--stall", "10", "--margin", "1e9"], "g")
    f = run(["--no-grow"], "f")
    assert g["growth_events"] and not any(e["accepted"] for e in g["growth_events"])
    assert g["params"] == f["params"]
    # rejected windows are a true continuation: same data stream, same optimiser, same weights
    assert g["log"][-1]["val_loss"] == f["log"][-1]["val_loss"]


@pytest.mark.parametrize("op,args", [(G.insert_block, (1,)), (G.widen_ffn, (160,)), (G.widen_d, (1,))])
def test_map_optimizer_keeps_old_moments(op, args):
    import mlx.optimizers as optim
    mx.random.seed(5)
    core = Core(CFG)
    opt = optim.AdamW(learning_rate=1e-3)
    tokens, q, f, a, ans, zs = C.batch(_rows(), "bus")
    lossf = lambda m: nn.losses.cross_entropy(value_logits(m(tokens), a), ans, reduction="mean")
    for _ in range(3):
        opt.update(core, nn.value_and_grad(core, lossf)(core)[1])
    grown = op(core, *args)
    new = G.map_optimizer(op, core, opt, grown, lambda: optim.AdamW(learning_rate=1e-3), *args)
    old_s, new_s = dict(tree_flatten(opt.state)), dict(tree_flatten(new.state))
    assert int(new.state["step"].item()) == int(opt.state["step"].item()) == 3
    for n, p in tree_flatten(grown.parameters()):
        assert new_s[f"{n}.m"].shape == p.shape and new_s[f"{n}.v"].shape == p.shape
    # every old moment entry survives somewhere (the total mass is preserved exactly: new entries are zero)
    for k in ("m", "v"):
        o = sum(float(mx.abs(v).sum().item()) for n, v in old_s.items() if n.endswith("." + k))
        g = sum(float(mx.abs(v).sum().item()) for n, v in new_s.items() if n.endswith("." + k))
        assert abs(o - g) <= 1e-4 * max(o, 1.0)
    new.update(grown, nn.value_and_grad(grown, lossf)(grown)[1])     # and training continues


@pytest.mark.parametrize("op,args", [(G.insert_block, (2,)), (G.widen_ffn, (160,)), (G.widen_d, (1,))])
def test_map_optimizer_places_each_moment_where_its_weight_went(op, args):
    """Tag every moment with its weight (m = w + 10, v = (w + 10)^2): the mapped moments must equal the
    weights-mode image of the tagged core at every old position and be 0 everywhere else (reviewer probe)."""
    import mlx.optimizers as optim
    from mlx.utils import tree_unflatten
    mx.random.seed(1)
    core = Core(CFG)
    P = dict(tree_flatten(core.parameters()))
    opt = optim.AdamW(learning_rate=1e-3)
    opt.init(core.trainable_parameters())
    opt.state = tree_unflatten([("step", mx.array(7, mx.uint64)), ("learning_rate", mx.array(1e-3))]
                               + [(f"{n}.m", w + 10) for n, w in P.items()]
                               + [(f"{n}.v", (w + 10) ** 2) for n, w in P.items()])
    tag = Core(CFG)
    tag.load_weights([(n, w + 10) for n, w in P.items()])
    ref = dict(tree_flatten(op(tag, *args).parameters()))
    grown = op(core, *args)
    new = G.map_optimizer(op, core, opt, grown, lambda: optim.AdamW(learning_rate=1e-3), *args)
    S = dict(tree_flatten(new.state))
    for n, r in ref.items():
        if op is G.widen_d and (n.endswith("n1.weight") or n.endswith("n2.weight") or n == "backbone.norm.weight"):
            d1 = r.shape[0]
            r = mx.concatenate([r[:CFG.d] / (CFG.d / d1) ** 0.5, mx.zeros(d1 - CFG.d)])   # undo the gain
        old = mx.abs(r) > 5
        m, v = S[f"{n}.m"], S[f"{n}.v"]
        assert mx.all(mx.where(old, mx.abs(m - r) < 1e-4, m == 0)).item(), n
        assert mx.all(mx.where(old, mx.abs(v - r * r) < 1e-2, v == 0)).item(), n


def test_controller_counts_flops_for_all_arms(tmp_path):
    from research.ens import grow_train as GT
    common = ["--d", "32", "--heads", "2", "--layers", "2", "--bus-width", "16", "--steps", "40", "--batch", "16",
              "--warmup", "5", "--window", "10", "--eval-every", "100", "--trial-steps", "5", "--trial-budget", "0.9"]
    GT.main(common + ["--no-grow", "--out", str(tmp_path / "f")])
    f = json.loads((tmp_path / "f" / "receipt.json").read_text())
    tokens = 16 * 32                                        # batch x seq_len
    assert f["train_flops"] == 6 * f["params"] * tokens * 40
    for margin, name in (("1e9", "rejected"), ("-1e9", "accepted")):
        GT.main(common + ["--stall", "10", f"--margin={margin}", "--out", str(tmp_path / name)])
        g = json.loads((tmp_path / name / "receipt.json").read_text())
        assert g["growth_events"]
        # closed form: each main window at that window's params; each growth window = every arm's params x trial
        rows = g["log"]
        main = sum((r["step"] - prev_step - trial) * r["params"]
                   for r, prev_step, trial in zip(rows, [0] + [x["step"] for x in rows[:-1]],
                                                  [0] + [5 if any(e["step"] == x["step"] for e in g["growth_events"])
                                                         else 0 for x in rows[:-1]]))
        trials = sum(5 * sum(e["trial_params"].values()) for e in g["growth_events"])
        assert g["train_flops"] == 6 * tokens * (main + trials), name
        assert rows[-1]["step"] == 40 and rows[-1]["train_flops"] == g["train_flops"]   # the curve ends at the total
    assert any(e["accepted"] for e in g["growth_events"]) and g["params"] > f["params"]


def test_decide_rules_on_fixed_cases():
    from research.ens.grow_train import decide
    P = {"control": 100, "big": 150, "same": 100}
    # control regresses (-0.030), the larger candidate regresses a bit more (-0.033): never accepted (review case)
    assert decide(2.0, {"control": 2.030, "big": 2.033}, P, "per-flop", 0.005)[2] is False
    assert decide(2.0, {"control": 2.030, "big": 2.033}, P, "loss", 0.005)[2] is False
    # lower absolute loss but worse gain per FLOP: loss accepts, per-flop rejects
    r = {"control": 1.90, "big": 1.86}                       # control gain 0.10; big gain 0.14 < 1.5 * 0.10
    assert decide(2.0, r, P, "loss", 0.005)[2] is True
    assert decide(2.0, r, P, "per-flop", 0.005)[2] is False
    # enough gain to pay for its size: accepted
    b, g, a = decide(2.0, {"control": 1.90, "big": 1.80}, P, "per-flop", 0.005)    # 0.20 - 1.5 * 0.10 = 0.05
    assert b == "big" and abs(g - 0.05) < 1e-9 and a
    # at equal cost both rules pick the same candidate
    r = {"control": 1.95, "same": 1.93, "big": 1.92}
    assert decide(2.0, {k: r[k] for k in ("control", "same")}, P, "loss", 0)[0] == \
        decide(2.0, {k: r[k] for k in ("control", "same")}, P, "per-flop", 0)[0] == "same"
    # control regresses, candidate improves in absolute terms: per-flop accepts on the absolute gain
    b, g, a = decide(2.0, {"control": 2.01, "big": 1.99}, P, "per-flop", 0.005)
    assert b == "big" and abs(g - 0.01) < 1e-9 and a


def test_per_flop_rule_is_wired_through(tmp_path):
    from research.ens import grow_train as GT
    common = ["--d", "32", "--heads", "2", "--layers", "2", "--bus-width", "16", "--steps", "40", "--batch", "16",
              "--warmup", "5", "--window", "10", "--eval-every", "100", "--trial-steps", "5", "--trial-budget", "0.9",
              "--stall", "10", "--margin=-1e9", "--accept", "per-flop"]
    GT.main(common + ["--out", str(tmp_path / "g")])
    g = json.loads((tmp_path / "g" / "receipt.json").read_text())
    assert g["growth_events"]
    for e in g["growth_events"]:
        res = dict(e["trial_val"])
        assert e["best"] == GT.decide(e["pre_trial_sel"], res, e["trial_params"], "per-flop", -1e9)[0]


def test_prune_ffn_drops_only_the_removed_units():
    """Pruning units whose down columns are zero leaves the function unchanged; kept units keep their weights."""
    mx.random.seed(7)
    core = G.widen_ffn(Core(CFG), 160)                       # units 128..159 have zero down columns: removable exactly
    keep = [list(range(128)) for _ in range(CFG.layers)]
    new = G.prune_ffn(core, keep)
    assert new.cfg.ffn == 128
    _check_same(core, new, 1e-5)
    # an arbitrary keep set selects the same rows/columns in every block
    keep = [sorted(random.Random(b).sample(range(160), 100)) for b in range(CFG.layers)]
    new = G.prune_ffn(core, keep)
    for b in range(CFG.layers):
        idx = mx.array(keep[b])
        ou, nu = core.backbone.blocks[b].up.weight, new.backbone.blocks[b].up.weight
        assert mx.array_equal(nu[:100], ou[:160][idx]).item() and mx.array_equal(nu[100:], ou[160:][idx]).item()
        assert mx.array_equal(new.backbone.blocks[b].down.weight, core.backbone.blocks[b].down.weight[:, idx]).item()


def test_unit_importance_ranks_units_by_removal_cost():
    """The first-order Taylor score is a ranking heuristic (it over-estimates magnitudes): removing the
    top-scored unit of a block must cost more loss than removing the bottom-scored one."""
    from research.ens.grow_train import unit_importance
    mx.random.seed(8)
    core = Core(CFG)
    tokens, q, f, a, ans, zs = C.batch(_rows(), "bus")
    lossf = lambda m: nn.losses.cross_entropy(value_logits(m(tokens), a), ans, reduction="mean")
    _, gr = nn.value_and_grad(core, lossf)(core)
    units, blocks = unit_importance(core, gr)
    assert units.shape == (CFG.layers, CFG.ffn) and blocks.shape == (CFG.layers,)
    l0 = float(lossf(core).item())
    w = dict(tree_flatten(core.parameters()))

    def removal_cost(b, j):
        up, dn = w[f"backbone.blocks.{b}.up.weight"], w[f"backbone.blocks.{b}.down.weight"]
        c = Core(CFG)
        c.load_weights(list(w.items()))
        c.load_weights([(f"backbone.blocks.{b}.up.weight", up.at[j].multiply(0.0).at[CFG.ffn + j].multiply(0.0)),
                        (f"backbone.blocks.{b}.down.weight", dn.at[:, j].multiply(0.0))], strict=False)
        return abs(float(lossf(c).item()) - l0)
    for b in range(CFG.layers):
        top, bot = int(mx.argmax(units[b]).item()), int(mx.argmin(units[b]).item())
        assert removal_cost(b, top) > removal_cost(b, bot)


def test_balloon_mode_grows_then_prunes_and_counts_flops(tmp_path):
    from research.ens import grow_train as GT
    GT.main(["--mode", "balloon", "--d", "32", "--heads", "2", "--layers", "2", "--bus-width", "16", "--steps", "300",
             "--batch", "16", "--warmup", "5", "--window", "25", "--eval-every", "100", "--balloon-until", "0.5",
             "--unit-thresh", "10", "--out", str(tmp_path / "b")])
    r = json.loads((tmp_path / "b" / "receipt.json").read_text())
    ops = [o for e in r["growth_events"] for o in e["ops"]]
    assert any(o.startswith("grow:") for o in ops) and any(o.startswith("prune_ffn") for o in ops)
    rows = r["log"]
    main = sum((x["step"] - p) * x["params"] for x, p in zip(rows, [0] + [y["step"] for y in rows[:-1]]))
    assert r["train_flops"] == 6 * 16 * 32 * main            # balloon has no trials: every step is a main step
    assert rows[-1]["step"] == 300
