"""ENS A0 harness: CORE-001, TRAIN-001, PKG-001, ABI/registry, bound, and family checks (CPU)."""
from __future__ import annotations

import copy
import json
import random

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from research.ens import abi
from research.ens import common as C
from research.ens import families as F
from research.ens import vocab as V
from research.ens.model import Backbone, BusCall, Core, CoreConfig, bounded_update, value_logits

SMALL = CoreConfig(d=64, layers=6, heads=4, ffn=128, bus_width=16)


def _batch(n=8, mode="bus", seed=0):
    rng = random.Random(seed)
    fams = F.split("train")
    rows = [F.sample_row(fams[rng.randrange(len(fams))], rng) for _ in range(n)]
    return rows, C.batch(rows, mode)


# ------------------------------------------------------------------ CORE-001
def test_core001_empty_admission_is_bitwise_ordinary_graph(tmp_path):
    mx.random.seed(0)
    core = Core(SMALL)
    # perturb bus parameters so a leak would show
    core.ports["middle"].gate = mx.array([3.0])
    _, (tokens, *_rest) = _batch()
    # strip the artifact to its backbone and reload into a bare ordinary graph
    path = tmp_path / "bb.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(core.backbone.parameters())))
    ordinary = Backbone(SMALL)
    ordinary.load_weights(str(path))
    reg = abi.Registry()
    admission = reg.admit([], core_artifact="x@y")
    assert admission.empty
    # the empty admission resolves to no bus call; a nonempty one would build BusCalls
    bus = None if admission.empty else BusCall("middle", None, None, None)
    a = core(tokens, bus)
    b = ordinary(tokens)
    assert mx.array_equal(a, b).item()
    # the perturbed gate is live on the bus path, so a leak into the empty path would have shown
    _, (_t, q, f, *_r) = _batch()
    msg = mx.random.normal((tokens.shape[0], SMALL.bus_width))
    assert not mx.array_equal(core(tokens, BusCall("middle", q, f, msg)), b).item()


def test_core001_detach_reattach_fresh_request():
    mx.random.seed(1)
    core = Core(SMALL)
    _, (tokens, q, f, a, ans, zs) = _batch()
    base = core(tokens)
    msg = mx.random.normal((tokens.shape[0], SMALL.bus_width))
    with_bus = core(tokens, BusCall("middle", q, f, msg))
    assert not mx.array_equal(base, with_bus).item()
    assert mx.array_equal(core(tokens), base).item()


def test_bus_write_is_causal_and_local():
    mx.random.seed(2)
    core = Core(SMALL)
    _, (tokens, q, f, a, ans, zs) = _batch()
    msg = mx.random.normal((tokens.shape[0], SMALL.bus_width))
    base, w = core(tokens), core(tokens, BusCall("middle", q, f, msg))
    for i in range(tokens.shape[0]):
        fi = int(f[i].item())
        assert mx.array_equal(base[i, :fi], w[i, :fi]).item()      # positions before the field untouched


# ------------------------------------------------------------------ bound
def test_bound_projects_onto_ball_and_zero_rule():
    h = mx.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    d = mx.array([[30.0, 40.0], [1.0, 1.0], [0.1, 0.0]])
    out = bounded_update(d, h, 0.5)
    n = mx.linalg.norm(out, axis=-1)
    assert abs(n[0].item() - 2.5) < 1e-5        # clipped to tau*||h||
    assert n[1].item() == 0.0                   # h = 0 -> no update
    assert abs(n[2].item() - 0.1) < 1e-6        # inside the ball: unchanged


def test_nonfinite_message_refused():
    core = Core(SMALL)
    _, (tokens, q, f, *_r) = _batch(n=2)
    bad = mx.full((2, SMALL.bus_width), float("nan"))
    with pytest.raises(ValueError):
        core(tokens, BusCall("middle", q, f, bad), check_finite=True)


# ------------------------------------------------------------------ TRAIN-001
def _grads(core, tokens, q, f, a, ans, msg):
    def loss(m):
        return nn.losses.cross_entropy(value_logits(m(tokens, BusCall("middle", q, f, msg)), a), ans,
                                       reduction="mean")
    _, g = nn.value_and_grad(core, loss)(core)
    port = g["ports"]["middle"]
    return float(mx.abs(port["gate"]).sum().item()), float(mx.abs(port["reader"]["weight"]).sum().item())


@pytest.mark.parametrize("gate,reader_zero,expect_gate,expect_reader", [
    (0.0, False, True, False),     # zero gate, nonzero reader: identity, gate learns
    (1.0, True, False, True),      # nonzero gate, zero reader: identity, reader learns
    (0.0, True, False, False),     # both zero: identity and stuck - must never be used at onboarding
])
def test_train001_gradient_flow(gate, reader_zero, expect_gate, expect_reader):
    mx.random.seed(3)
    core = Core(SMALL)
    port = core.ports["middle"]
    port.gate = mx.array([gate])
    if reader_zero:
        port.reader.weight = mx.zeros_like(port.reader.weight)
    _, (tokens, q, f, a, ans, zs) = _batch()
    msg = C.codebook(SMALL.bus_width)[zs]
    assert mx.allclose(core(tokens, BusCall("middle", q, f, msg)), core(tokens), atol=0, rtol=0).item()
    g_gate, g_reader = _grads(core, tokens, q, f, a, ans, msg)
    assert (g_gate > 0) == expect_gate
    assert (g_reader > 0) == expect_reader
    assert all(mx.all(mx.isfinite(v)).item() for _, v in tree_flatten(core.parameters()))


def test_onboarding_default_is_open_and_nonzero():
    port = Core(SMALL).ports["middle"]
    assert port.gate.item() == 1.0
    assert mx.abs(port.reader.weight).sum().item() > 0


# ------------------------------------------------------------------ PKG-001 / ABI
def _manifest(weight_bytes=b"\x00" * 64, tmp=None):
    p = tmp / "weights.safetensors"
    p.write_bytes(weight_bytes)
    return {
        "schema": abi.SCHEMA, "address": "ens://0/a0.local/perm.10@0.1.0", "core_artifact": "core@abc",
        "neural_abi_major": 0, "execution_abi_major": 0, "law": "stateless.residual.v1",
        "implementation_digest": "d" * 64,
        "ports": [{"direction": "consume", "site": "middle.tap.query", "width": 16},
                  {"direction": "produce", "site": "middle.residual.field", "width": 16}],
        "state": {"kind": "stateless", "clock": "none", "rollback": "none"},
        "composition": {"writers": "exclusive.v1", "bound": "final_l2.v1", "tau_milli": 500},
        "limits": {"max_batch": 4, "max_positions": 32, "max_workspace_bytes": 1048576},
        "artifacts": [abi.artifact_descriptor(p)],
    }


def test_pkg001_key_order_weight_byte_and_receipt(tmp_path):
    m = _manifest(tmp=tmp_path)
    d0 = abi.package_digest(m)
    reordered = json.loads(json.dumps(dict(reversed(list(m.items())))))
    assert abi.package_digest(reordered) == d0
    m2 = _manifest(weight_bytes=b"\x00" * 63 + b"\x01", tmp=tmp_path)
    assert abi.package_digest(m2) != d0
    assert abi.contract_digest(m2) == abi.contract_digest(m)        # weights change, contract does not
    m3 = copy.deepcopy(m)
    m3["receipts"] = [{"qualified": True, "sig": "x"}]
    m3["signatures"] = ["sig"]
    assert abi.package_digest(m3) == d0


def test_abi_refusals(tmp_path):
    m = _manifest(tmp=tmp_path)
    with pytest.raises(abi.ManifestError):
        abi.parse('{"a": 1, "a": 2}')
    for mutate in (
        lambda x: x.update(required_features=["ens.future.v9"]),
        lambda x: x["artifacts"].append(dict(x["artifacts"][0])),
        lambda x: x["artifacts"][0].update(name="../escape"),
        lambda x: x["composition"].update(tau=0.1),
        lambda x: x.pop("law"),
        lambda x: x["ports"].append(dict(x["ports"][0])),
    ):
        bad = copy.deepcopy(m)
        mutate(bad)
        with pytest.raises(abi.ManifestError):
            abi.package_digest(bad)


def test_registry_admission_is_immutable_and_allocates_no_tensors(tmp_path):
    reg = abi.Registry()
    base = _manifest(tmp=tmp_path)
    for i in range(10_000):
        m = dict(base, address=f"ens://0/synthetic/m{i}@0.1.0")
        reg.register(m)
    assert len(reg) == 10_000
    assert not any(isinstance(v, mx.array) for _, v in tree_flatten(reg._manifests))
    adm = reg.admit(["ens://0/synthetic/m1@0.1.0"], core_artifact="core@abc")
    pinned = adm.digests
    other = dict(base, address="ens://0/synthetic/m1@0.1.0", implementation_digest="e" * 64)
    with pytest.raises(abi.ManifestError):
        reg.register(other)                                  # alias collision needs an explicit move
    new = reg.register(dict(other, address="ens://0/synthetic/m1b@0.1.0"))
    reg.move_alias("ens://0/synthetic/m1@0.1.0", new)
    assert adm.digests == pinned                             # admitted set unchanged by catalog mutation
    assert reg.admit(["ens://0/synthetic/m1@0.1.0"], core_artifact="core@abc").digests != pinned
    with pytest.raises(abi.ManifestError):
        reg.admit(["ens://0/synthetic/m2@0.1.0"], core_artifact="other@core")


# ------------------------------------------------------------------ families
def test_family_registry_structure():
    assert F.check_structure() == []
    assert len(F.split("train")) >= 16 and len(F.split("dev")) == 4 and len(F.split("confirm")) == 4
    assert len(F.split("gen_dev")) == 4 and len(F.split("gen_confirm")) == 6 and len(F.split("route_dev")) == 2
    train = {(f.cls, f.knobs) for f in F.split("train")}
    for f in F.split("dev") + F.split("confirm"):          # extension: seen structure, fresh law
        assert (f.cls, f.knobs) in train and f.law and f.fid.endswith("." + f.law)
    for f in F.split("gen_dev") + F.split("gen_confirm"):  # generalization: unseen structure
        assert (f.cls, f.knobs) not in train and not f.law


def test_training_family_digests_unchanged_by_rescope():
    # pilot-2 cores (b5eca3e13) were trained on these digests; the re-scope must not move them
    assert F.BY_ID["fst.2-2"].digest.startswith("941ac865b8c6")
    assert F.BY_ID["program.ADD"].digest.startswith("c5a4cc37d1ec")


def test_extension_laws_are_distinct():
    ext = F.split("dev") + F.split("confirm")
    laws = {}
    for f in ext:
        base = F.BY_ID[f.fid.rsplit(".", 1)[0]]
        assert f.digest != base.digest and f.hidden != base.hidden
        assert laws.setdefault((f.cls, f.knobs, f.hidden), f.fid) == f.fid   # two laws on one structure differ
    assert len({(f.cls, f.knobs) for f in ext}) == len(ext) == 8            # no structure twice
    dev = {(f.cls, f.knobs) for f in F.split("dev")}
    assert not dev & {(f.cls, f.knobs) for f in F.split("confirm")}

def test_family_counterfactual_scoring_matches_generator():
    rng = random.Random(5)
    for fam in F.FAMILIES:
        for _ in range(20):
            r = F.sample_row(fam, rng)
            assert F.answer_for(fam, r, r.z) == r.answer
            # the token re-scorer must agree with the generator's own closure at every z
            for z2 in range(fam.z_domain):
                twin = F.sample_row(fam, random.Random(hash((fam.fid, r.op, 1)) & 0xFFFF), z=z2)
                assert F.answer_for(fam, twin, z2) == twin.answer
            # and every op depends on z
            assert len({F.answer_for(fam, r, z) for z in range(fam.z_domain)}) > 1


def test_module_tap_cannot_see_the_op():
    # cycling a small training pool is safe only because the query position precedes the op tokens
    rng = random.Random(3)
    for fam in F.split("dev") + F.split("confirm"):
        r = F.sample_row(fam, rng)
        ids, q, fi, ai = F.layout(r, "bus")
        assert q == 2 and ids[q] == V.key(r.x) and q < 3 <= fi


def test_held_program_rows_pass_witness_filter():
    fam = F.BY_ID["program.XOR.ADD.MUL"]
    short = F._program_tables_up_to(2)
    for r in F.family_rows(fam, "held")[:200]:
        table = tuple(F.answer_for(fam, r, z) for z in range(F.PROGRAM_Z))
        assert table not in short


def test_contributor_pools_disjoint_and_hidden_law_balanced():
    for fam in F.split("dev") + F.split("confirm") + F.split("gen_confirm"):
        tr, he = F.family_rows(fam, "train"), F.family_rows(fam, "held")
        assert len({r.op for r in he}) >= F.MIN_HELD_OPS and len({r.x for r in he}) == 16 and len(tr) == F.TRAIN_POOL
        assert not {r.signature for r in tr} & {r.signature for r in he}
        assert all(r.z == fam.hidden[r.x] for r in tr + he)
        counts = [fam.hidden.count(z) for z in range(fam.z_domain)]
        assert max(counts) - min(counts) <= 1


def test_layout_geometry_matches_across_modes():
    rng = random.Random(0)
    r = F.sample_row(F.BY_ID["fst.4-2"], rng)
    lays = {m: F.layout(r, m) for m in ("bus", "typed", "ordinary")}
    assert len({(l[1], l[2], l[3]) for l in lays.values()}) == 1
    assert len({len(l[0]) for l in lays.values()}) == 1


def test_gen_dev_two_op_program_is_not_a_one_op_program():
    fam = F.BY_ID["program.XOR.ADD"]
    one = F._program_tables_up_to(1)
    for r in F.family_rows(fam, "held")[:300]:
        assert tuple(F.answer_for(fam, r, z) for z in range(F.PROGRAM_Z)) not in one


def test_pairs_have_different_targets():
    from research.ens.evaluate import pair_rows
    for fid in ("fst.3-2", "graph.8-2", "perm.8"):
        fam = F.BY_ID[fid]
        rows = F.family_rows(fam, "held")
        partners = pair_rows(fam, rows)
        assert all(j >= 0 for j in partners)
        for r, j in zip(rows, partners):
            if j >= 0:
                assert rows[j].z != r.z and F.answer_for(fam, r, rows[j].z) != r.answer


def test_family_digest_ignores_file_hash_but_tracks_law():
    fam = F.BY_ID["perm.10"]
    rec = fam.record()
    assert "generator_sha256" in rec and rec["law_version"] == F.LAW_VERSION
    import dataclasses
    assert dataclasses.replace(fam, knobs=(9,)).digest != fam.digest


# ------------------------------------------------------------------ arm matching (3-step CPU smoke)
def _train(tmp_path, arm, extra=()):
    from research.ens import train_core
    out = tmp_path / arm
    argv = ["--arm", arm, "--seed", "0", "--steps", "3", "--batch", "16", "--warmup", "1", "--d", "32",
            "--heads", "2", "--bus-width", "8", "--log-every", "100", "--out", str(out), *extra]
    train_core.main(argv)
    return out


def test_arms_share_data_order_and_freeze_correctly(tmp_path):
    from research.ens import train_core
    fams = F.split("train")
    M = C.codebook(8)
    r6, j6, r7, j7 = random.Random(1), random.Random(2), random.Random(1), random.Random(2)
    for _ in range(4):
        x6 = train_core.build_batch(r6, j6, fams, 16, "c6", 0.25, M)
        x7 = train_core.build_batch(r7, j7, fams, 16, "c7", 0.25, M)
        assert mx.array_equal(x6[4], x7[4]).item()          # same answers -> same rows, every batch
        assert mx.array_equal(x6[3], x7[3]).item()
    c6, c7 = _train(tmp_path, "c6"), _train(tmp_path, "c7")
    c6r = _train(tmp_path, "c6r", ("--from-core", str(c7)))
    loaded = {n: C.load_core(d)[0] for n, d in (("c6", c6), ("c7", c7), ("c6r", c6r))}
    base7 = loaded["c7"]

    def changed(a, b, prefix):
        fa, fb = dict(tree_flatten(a.parameters())), dict(tree_flatten(b.parameters()))
        return any(not mx.array_equal(fa[k], fb[k]).item() for k in fa if k.startswith(prefix))
    site = "ports." + train_core.DEFAULT_SITE
    mx.random.seed(0)
    init = Core(CoreConfig(d=32, heads=2, ffn=128, bus_width=8))       # the same init train_core builds
    assert not changed(base7, init, "ports")                            # c7 never trains a port
    assert changed(base7, init, "backbone")
    assert changed(loaded["c6"], init, site + ".reader") and not changed(loaded["c6"], init, site + ".tap")
    # c6r touches only the site reader/gate of its c7 base
    assert not changed(loaded["c6r"], base7, "backbone")
    assert changed(loaded["c6r"], base7, site + ".reader")
    assert not changed(loaded["c6r"], base7, site + ".tap")
    rec = json.loads((c6 / "receipt.json").read_text())
    assert rec["site"] == train_core.DEFAULT_SITE and rec["checkpoint"].startswith("ens-a0-core-c6-")


def test_module_messages_leave_at_unit_rms():
    from research.ens.contrib import Module, OracleTable
    mx.random.seed(0)
    for m, x in ((Module(16, 32), mx.random.normal((8, 16)) * 50), (OracleTable(16, 4), mx.array([0, 1, 2, 3]))):
        out = m(x)
        assert mx.allclose(mx.sqrt(mx.mean(out * out, axis=-1)), mx.ones(out.shape[0]), atol=1e-3).item()


def test_zblind_bayes_known_values():
    from research.ens.evaluate import zblind_bayes
    # ADD is a bijection of z and H is balanced over 16 values: without z every answer is equally likely
    fam = F.BY_ID["program.ADD.h1"]
    assert abs(zblind_bayes(fam, F.family_rows(fam, "held")) - 1 / 16) < 1e-9
    # GRAPH squashes z, so the op alone beats chance (1 / z_domain)
    g = F.BY_ID["graph.4-2.h1"]
    assert zblind_bayes(g, F.family_rows(g, "held")) > 1 / g.z_domain + 0.1


def test_sealed_families_refuse_oracle_eval_on_any_split(tmp_path):
    from research.ens import evaluate
    fam = F.split("confirm")[0]
    with pytest.raises(SystemExit):
        evaluate.main(["--core", str(tmp_path), "--family", fam.fid, "--split", "train", "--out", str(tmp_path / "o")])


def test_summarize_reads_eval_receipts(tmp_path):
    from research.ens import summarize
    d = tmp_path / "eval-x"
    d.mkdir()
    (d / "receipt.json").write_text(json.dumps({"schema": "ens.a0.eval_receipt.v1", "core_checkpoint": "c@1",
        "arm": "c6", "family": "graph.4-2.h1", "module_kind": "mlp",
        "results": {"on": 0.9, "bypass": 0.5, "zblind": 0.55, "both": 0.8, "oracle_codebook": 1.0, "ceiling_ok": True}}))
    (r,) = list(summarize.rows(tmp_path))
    assert r["bar"] == 0.55 and abs(r["margin"] - 0.35) < 1e-9


def test_key_table_reads_only_the_key_and_quantizes(tmp_path):
    from research.ens.contrib import KeyTable, message
    from research.ens.evaluate import quantize_table
    fam = F.BY_ID["graph.4-2.h1"]
    rows = F.family_rows(fam, "held")[:8]
    tokens, q, f, a, ans, zs = C.batch(rows, "bus")
    m = KeyTable(64)
    out = message(m, tokens, zs * 0)                       # z is never read
    assert mx.array_equal(out, m(mx.array([r.x for r in rows]))).item()
    for mode, b in (("int8", 64), ("int4", 32), ("codebook", 1)):
        mm = KeyTable(64)
        mm, eb = quantize_table(mm, mode, 64)
        assert eb == b
    M = C.codebook(64)
    assert all(any(mx.allclose(row, M[i]).item() for i in range(64)) for row in mm.table.weight[:3])


def test_snapped_key_table_emits_codebook_rows_with_gradient():
    from research.ens.contrib import KeyTable
    import mlx.nn as nn
    m = KeyTable(64, snap=True)
    out = m(mx.array([0, 1, 2]))
    M = C.codebook(64)
    assert all(any(mx.allclose(r, M[i], atol=1e-5).item() for i in range(64)) for r in out)
    g = nn.value_and_grad(m, lambda mm: mx.sum(mm(mx.array([0, 1])) * M[5]))(m)[1]
    assert float(mx.abs(g["table"]["weight"]).sum().item()) > 0


def test_key_choice_serves_codebook_rows_and_learns():
    from research.ens.contrib import build
    import mlx.nn as nn
    m = build("key-choice", 64, hidden=0, z_domain=16)
    M = C.codebook(64)
    g = nn.value_and_grad(m, lambda mm: -mx.sum(mm(mx.array([0, 1])) * M[7]))(m)[1]
    assert float(mx.abs(g["logits"]).sum().item()) > 0
    m.logits = m.logits.at[0, 7].add(5.0)
    m.hard = True
    assert mx.allclose(m(mx.array([0]))[0], M[7]).item()


def _fake_confirm(tmp_path, gap, noise=0.0, drop=None, undone=()):
    import json as _j
    rng = random.Random(1)
    for seed in range(5):
        sd = tmp_path / f"seed{seed}"
        for fam in ("program.MUL.h1", "program.XOR.h1", "graph.6-1.h1", "graph.8-1.h1"):
            for arm, base in (("c6", 0.9), ("c6r", 0.9), ("c7", 0.9 - gap)):
                for kind in ("mlp", "oracle-table"):
                    if kind == "mlp" and (arm, fam, seed) == drop:
                        continue
                    d = sd / f"eval-{arm}-{kind}-{fam}"
                    d.mkdir(parents=True)
                    on = 0.99 if kind == "oracle-table" else base + rng.uniform(-noise, noise)
                    d.joinpath("receipt.json").write_text(_j.dumps({"schema": "ens.a0.eval_receipt.v1",
                        "core_checkpoint": f"{arm}@{seed}", "family_digest": F.BY_ID[fam].digest, "family": fam, "module_kind": kind, "arm": arm,
                        "split": "held", "quant": "none", "results": {
                        "on": on, "bypass": 0.3, "zblind": 0.3, "both": 0.8, "wrong": 0.2, "ordinary_suite": 0.7}}))
        for arm in ("c6", "c6r", "c7"):
            (sd / f"core-{arm}").mkdir(parents=True, exist_ok=True)
            (sd / f"core-{arm}" / "receipt.json").write_text(
                _j.dumps({"arm": arm, "seed": seed, "checkpoint": f"{arm}@{seed}"}))
            (sd / f"core-{arm}" / "core001.json").write_text(_j.dumps({"pass": True, "core_checkpoint": f"{arm}@{seed}"}))
        if seed not in undone:
            (sd / "DONE").write_text("")


def test_decide_verdicts(tmp_path):
    from research.ens import decide
    _fake_confirm(tmp_path / "a", 0.30, 0.02)
    _fake_confirm(tmp_path / "s", 0.00, 0.02)
    ra, rs = decide.decide(*decide.load(tmp_path / "a")), decide.decide(*decide.load(tmp_path / "s"))
    assert ra["verdict"] == "advance" and abs(ra["c6_minus_c7"] - 0.30) < 0.02
    assert abs(ra["c6_minus_c6r"]) < 0.02 and ra["ci95"][0] > decide.DELTA
    assert rs["verdict"] == "stop" and rs["ci95"][1] < decide.DELTA


def test_decide_fails_closed_and_excludes_unfinished_seeds(tmp_path):
    from research.ens import decide
    _fake_confirm(tmp_path / "m", 0.30, 0.02, drop=("c7", "graph.8-1.h1", 2))   # a gating arm (option A)
    with pytest.raises(SystemExit):
        decide.decide(*decide.load(tmp_path / "m"))
    r = decide.decide(*decide.load(tmp_path / "m"), missing_as_loss=True)
    assert r["missing_cells_counted_as_loss"]
    _fake_confirm(tmp_path / "u", 0.30, 0.02, undone=(4,))
    r = decide.decide(*decide.load(tmp_path / "u"))
    assert r["verdict"] == "incomplete"
    assert r["excluded_seeds"] == [4] and r["seeds"] == [0, 1, 2, 3]


def test_c6a_adapter_arm_trains_only_the_adapter_and_keeps_core001(tmp_path):
    from research.ens import core001
    c7 = _train(tmp_path, "c7")
    c6a = _train(tmp_path, "c6a", ("--from-core", str(c7)))
    a, _ = C.load_core(c6a)
    b, _ = C.load_core(c7)
    assert a.cfg.reader_kind == "mlp" and "adapter" in a.ports["early"]
    fa, fb = dict(tree_flatten(a.backbone.parameters())), dict(tree_flatten(b.backbone.parameters()))
    assert all(mx.array_equal(fa[k], fb[k]).item() for k in fb)          # backbone frozen
    assert core001.check(c6a)["pass"]


def test_c6a_state_adapter_zero_message_writes_nothing(tmp_path):
    from research.ens import core001
    c7 = _train(tmp_path, "c7")
    c6a = _train(tmp_path, "c6a", ("--from-core", str(c7), "--adapter-kind", "mlp_state"))
    core, _ = C.load_core(c6a)
    assert core.cfg.reader_kind == "mlp_state"
    _, (tokens, q, f, a, ans, zs) = _batch()
    zero = mx.zeros((tokens.shape[0], core.cfg.bus_width))
    assert mx.array_equal(core(tokens, BusCall("early", q, f, zero)), core(tokens)).item()
    assert core001.check(c6a)["pass"]


def test_c6l_lora_only_on_bus_path_and_core001(tmp_path):
    from research.ens import core001
    c7 = _train(tmp_path, "c7")
    c6l = _train(tmp_path, "c6l", ("--from-core", str(c7), "--lora-rank", "2", "--reader-init", "0.01"))
    core, _ = C.load_core(c6l)
    assert core.cfg.lora_rank == 2 and "lora" in core
    base, _ = C.load_core(c7)
    fa, fb = dict(tree_flatten(core.backbone.parameters())), dict(tree_flatten(base.backbone.parameters()))
    assert all(mx.array_equal(fa[k], fb[k]).item() for k in fb)                 # backbone frozen
    lp = dict(tree_flatten(core.parameters()["lora"]))
    assert any(float(mx.abs(v).sum().item()) > 0 for k, v in lp.items() if k.endswith("b.weight"))  # LoRA trained
    assert core001.check(c6l)["pass"]                                            # no bus -> ordinary graph


def test_matryoshka_batches_cut_only_bus_rows_top_half():
    from research.ens import train_core
    M = C.codebook(8)
    fams = F.split("train")
    *_, msg = train_core.build_batch(random.Random(1), random.Random(2), fams, 64, "c6", 0.0, M, matryoshka=1.0)
    assert float(mx.abs(msg[:, 4:]).sum().item()) == 0 and float(mx.abs(msg[:, :4]).sum().item()) > 0
    *_, m0 = train_core.build_batch(random.Random(1), random.Random(2), fams, 64, "c6", 0.0, M, matryoshka=0.0)
    assert float(mx.abs(m0[:, 4:]).sum().item()) > 0


def test_family_eval_logs_every_training_family(tmp_path):
    out = _train(tmp_path, "c6", ("--family-eval", "--log-every", "3"))
    r = json.loads((out / "receipt.json").read_text())
    fams = r["log"][-1]["families"]
    assert set(fams) == {f.fid for f in F.split("train")} and all({"ord", "native"} <= set(v) for v in fams.values())


def test_confirm_rejects_mismatched_evaluation_and_core001(tmp_path):
    from research.ens import decide
    _fake_confirm(tmp_path, 0.3)
    cells = ("eval-c6-mlp-graph.6-1.h1", "eval-c6r-mlp-program.XOR.h1", "eval-c7-mlp-graph.8-1.h1")
    for cell, update in zip(cells, ({"core_checkpoint": "other@0"}, {"family_digest": "stale"},
                                    {"results": {"half": "low"}})):
        p = tmp_path / "seed0" / cell / "receipt.json"
        record = json.loads(p.read_text())
        if "results" in update:
            record["results"].update(update["results"])
        else:
            record.update(update)
        p.write_text(json.dumps(record))
    oracle = tmp_path / "seed0/core-c6/core001.json"
    oracle.write_text(json.dumps({"pass": True, "core_checkpoint": "other@0"}))
    data, meta = decide.load(tmp_path)
    assert len(meta["rejected"]) == 3 and not meta["core001"]["c6/seed0"]
    with pytest.raises(SystemExit, match="incomplete grid"):
        decide.decide(data, meta)


def test_registry_copies_manifest_on_input_and_output(tmp_path):
    reg = abi.Registry()
    m = _manifest(tmp=tmp_path)
    digest = reg.register(m)
    m["core_artifact"] = "other"
    reg.manifest(digest)["ports"][0]["width"] = 99
    assert abi.package_digest(reg.manifest(digest)) == digest
    with pytest.raises(abi.ManifestError):
        reg.admit([m["address"]], core_artifact="other")


def test_key_choice_load_matches_exported_indices_and_checks_bytes(tmp_path):
    from research.ens import contrib
    fam = F.BY_ID["program.ADD.h1"]
    mod = contrib.build("key-choice", SMALL.bus_width, hidden=1, z_domain=fam.z_domain)
    mod.logits = mod.logits.at[0, 7].add(1)
    weights = tmp_path / "module.safetensors"
    mx.save_safetensors(str(weights), dict(tree_flatten(mod.parameters())))
    man = contrib.manifest("test@0", SMALL, "middle", fam, weights, "key-choice")
    assert man["ports"][0]["site"] == "prompt.key" and man["address"].endswith(".choice@0.1.0")
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    (tmp_path / "receipt.json").write_text(json.dumps({
        "recipe": contrib.RECIPES["key-choice"], "kind": "key-choice", "family": fam.record(),
        "core_checkpoint": "test@0", "package_digest": abi.package_digest(man)}))
    loaded = contrib.load_module(tmp_path, SMALL.bus_width)
    assert mx.array_equal(loaded(mx.array([0]))[0], C.codebook(SMALL.bus_width)[7]).item()
    weights.write_bytes(weights.read_bytes() + b"!")
    with pytest.raises(SystemExit, match="digest"):
        contrib.load_module(tmp_path, SMALL.bus_width)


def test_core_rejects_substituted_weights_and_wrong_retrofit_source(tmp_path):
    core = _train(tmp_path, "c6")
    with pytest.raises(SystemExit, match="must be a c7"):
        _train(tmp_path, "c6r", ("--from-core", str(core)))
    weights = core / "core.safetensors"
    weights.write_bytes(weights.read_bytes() + b"!")
    with pytest.raises(ValueError, match="weights or configuration"):
        C.load_core(core)


def test_m3_owner_stops_nested_work_before_releasing_lock(tmp_path):
    import fcntl
    import os
    import signal
    import subprocess
    import sys
    import time

    # Only the temporary locks and a CPU-only sleeping process are exercised here.
    worker_lock = tmp_path / "worker.lock"
    ready = tmp_path / "ready"
    worker = ("import fcntl,os,pathlib,time; "
              f"f=open({str(worker_lock)!r}, 'w'); fcntl.flock(f, fcntl.LOCK_EX); "
              f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)")
    middle = f"import subprocess,sys; subprocess.run([sys.executable,'-c',{worker!r}])"
    command = f"""
import sys
from pathlib import Path
from types import SimpleNamespace
from research.ens import m3
m3.LOCK = Path({str(tmp_path / 'lease')!r})
m3.OWNER = Path({str(tmp_path / 'owner')!r})
m3.run(SimpleNamespace(cmd=[sys.executable, '-c', {middle!r}], job='cpu-test', wait_s=2))
"""
    with subprocess.Popen([sys.executable, "-c", command]) as owner:
        try:
            deadline = time.monotonic() + 8
            while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ready.exists()
            owner.terminate()
            assert owner.wait(timeout=5) == 143
            assert not (tmp_path / "owner").exists()
            with worker_lock.open("r") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if owner.poll() is None:
                owner.kill()
            if ready.exists():
                try:
                    os.kill(int(ready.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_research_probes_reject_unmeasurable_inputs(tmp_path, monkeypatch):
    from research.ens import chain, claim, depth
    with pytest.raises(ValueError, match="at least two"):
        chain._rows("GRAPH", 1, 2, 4, 0)
    with pytest.raises(ValueError, match="unknown depth arm"):
        depth._parse("attent2")
    with pytest.raises(SystemExit, match="2"):
        claim.main(["--core", "unused", "--modules", "a", "--out", str(tmp_path / "out")])
    fam = F.BY_ID["program.ADD.h1"]
    core = Core(SMALL)
    monkeypatch.setattr(C, "load_core", lambda _: (core, "core@cpu"))
    monkeypatch.setattr(claim, "check_core_families", lambda _: None)
    monkeypatch.setattr(claim, "core_site", lambda _: "middle")
    module = tmp_path / "oracle"
    module.mkdir()
    (module / "receipt.json").write_text(json.dumps({"kind": "oracle-table", "family": fam.record(),
        "core_checkpoint": "core@cpu", "family_digest": fam.digest}))
    with pytest.raises(SystemExit, match="oracle tables"):
        claim.main(["--core", "unused", "--modules", str(module), "other", "--out", str(tmp_path / "claim")])


def test_zero_message_mlp_and_unreachable_port():
    from dataclasses import replace
    from research.ens.evaluate import other_port
    core = Core(replace(SMALL, reader_kind="mlp", layers=2))
    q = mx.zeros((1, SMALL.bus_width))
    port = core.ports["early"]
    port.adapter.layers[0].bias = mx.ones_like(port.adapter.layers[0].bias)
    assert mx.array_equal(port.read(q), mx.zeros((1, SMALL.d))).item()
    assert other_port(core, "early") == "ingress"
    _, (tokens, qi, fi, *_) = _batch(1)
    with pytest.raises(KeyError, match="unreachable"):
        core(tokens, BusCall("late", qi, fi, q))


def test_confirm_missing_cell_uses_own_core_baseline():
    from research.ens import decide
    data = {(a, k, f, 0): {"bypass": 0.1 if a == "c6" else 0.4, "ordinary_suite": 0.6, "zblind": 0.5}
            for a in decide.ARMS for f in decide.CONFIRM for k in decide.KINDS}
    key = ("c7", "mlp", decide.CONFIRM[0], 0)
    del data[key]
    decide.complete_grid(data, [0], missing_as_loss=True)
    assert data[key]["on"] == 0.4
    del data[key], data["c7", "oracle-table", decide.CONFIRM[0], 0]
    with pytest.raises(SystemExit, match="measured baseline"):
        decide.complete_grid(data, [0], missing_as_loss=True)


def test_decide_rejects_cores_off_protocol_and_c6r_does_not_gate(tmp_path):
    from research.ens import decide
    old = decide.PROTOCOL
    try:
        decide.PROTOCOL = None
        _fake_confirm(tmp_path / "p", 0.30, 0.02)
        data, meta = decide.load(tmp_path / "p")
        assert data            # with no protocol set, fake cores are accepted (unit-test path)
        for k in list(data):
            if k[0] == "c6r":
                del data[k]    # C6R cells missing must not make the grid incomplete
        r = decide.decide(data, meta)
        assert r["primary"] == "c6_minus_c7"
    finally:
        decide.PROTOCOL = old


@pytest.mark.parametrize("kind", ["mlp", "oracle-table", "key-table"])
def test_batched_contrib_equals_sequential(tmp_path, kind):
    from research.ens import contrib, contrib_batched
    core_dir = _train(tmp_path, "c6", ("--bus-width", "64"))
    fams = ["program.ADD.h1", "graph.4-2.h1", "fst.3-1.h1"]
    common = ["--core", str(core_dir), "--kind", kind, "--seed", "3", "--steps", "4", "--batch", "8",
              "--hidden", "16", "--log-every", "2"]
    for fid in fams:
        contrib.main(common + ["--family", fid, "--out", str(tmp_path / f"seq-{fid}")])
    contrib_batched.main(common + ["--families", *fams, "--out-root", str(tmp_path), "--prefix", "bat-"])
    for fid in fams:
        a = mx.load(str(tmp_path / f"seq-{fid}" / "module.safetensors"))
        b = mx.load(str(tmp_path / f"bat-{fid}" / "module.safetensors"))
        assert set(a) == set(b)
        assert all(mx.allclose(a[k], b[k], atol=1e-5).item() for k in a), fid
        ra = json.loads((tmp_path / f"seq-{fid}" / "receipt.json").read_text())
        rb = json.loads((tmp_path / f"bat-{fid}" / "receipt.json").read_text())
        assert ra["package_digest"] != "" and rb["trainer"] == "contrib_batched"
        assert [r["loss"] for r in ra["log"]] == pytest.approx([r["loss"] for r in rb["log"]], abs=1e-4)


def test_prefix_suffix_equals_bus_forward(tmp_path):
    mx.random.seed(0)
    core = Core(SMALL)
    _, (tokens, q, f, a, ans, zs) = _batch()
    msg = mx.random.normal((tokens.shape[0], SMALL.bus_width))
    for site in ("ingress", "early", "middle"):
        bus = BusCall(site, q, f, msg)
        assert mx.allclose(core(tokens, bus), core.suffix(core.prefix(tokens, site), bus), atol=1e-5).item(), site
