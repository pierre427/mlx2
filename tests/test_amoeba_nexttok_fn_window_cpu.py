"""The next-token gating FN window (scripts/amoeba_nexttok_fn_window.py), Part Q end to end on the carve CPU
toy with a fake partition and corpora; and the carve units' precomputed per-token gates (token_gate)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

from research.amoeba.carve.host import Host  # noqa: E402
from test_amoeba_carve_cpu import Tok, toy  # noqa: E402
import amoeba_nexttok_fn_window as W  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _host():
    return Host(model=toy(0, moe=False), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None)


def test_token_gate_matches_static_masks_and_core():
    h = _host()
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 64, 16).tolist() for _ in range(4)]
    L, N = h.L, h.N
    own = rng.integers(0, 3, size=(L, N))
    core = own == 0
    groups = {("a",): own == 1, ("b",): own == 2}
    masks = W.device_masks(core, groups)
    ones = {k: np.ones((4, 16), bool) for k in groups}
    zeros = {k: np.zeros((4, 16), bool) for k in groups}
    full = W.masked_ce(h, seqs, None, None, 4)
    allopen = W.masked_ce(h, seqs, masks, ones, 4)
    assert np.allclose(full, allopen, atol=1e-5)               # core + every group open == the full model
    c = h.ctl
    import research.amoeba.carve.evals as E
    cfg = E.configure(h, mx.array(core.astype(np.float32)))
    cfg()
    ids = mx.array(seqs)
    from research.amoeba.carve.carving import ce_rows
    ref = np.array(ce_rows(h.logits_at(ids, list(range(3, 15))), ids[:, 4:])).mean(1)
    c.reset()
    assert np.allclose(W.masked_ce(h, seqs, masks, zeros, 4), ref, atol=1e-5)
    # dense layers run whole whatever the gates say: with every layer dense, the full model
    assert np.allclose(W.masked_ce(h, seqs, masks, zeros, 4, dense_layers=range(L)), full, atol=1e-5)


def test_token_gate_on_moe_neuron_units_matches_static_masks():
    """The MoE neuron path (_neuron_keep, the one FN uses): token_gate all ones == the static box mask, all
    zeros == the core alone, and a per-token mix equals running each row with its own static mask."""
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    h = Host(model=qn_toy(), tokenizer=Tok(), fingerprint="toy:abc", log=lambda *_: None, moe_units="neuron")
    rng = np.random.default_rng(3)
    ids = mx.array(rng.integers(0, 64, (2, 10)).tolist())
    own = [rng.integers(0, 2, n) for n in h.widths]
    core = [mx.array((o == 0).astype(np.float32)) for o in own]
    box = [mx.array((o == 1).astype(np.float32)) for o in own]
    c = h.ctl

    def run(boxes):
        c.reset()
        c.core, c.boxes = core, boxes
        out = h.full_logits(ids)
        mx.eval(out)
        c.reset()
        return np.array(out)
    static = run([{"name": "b", "mask": box}])
    alone = run([])
    ones = run([{"name": "b", "mask": box, "token_gate": mx.ones((2, 10, 1))}])
    zeros = run([{"name": "b", "mask": box, "token_gate": mx.zeros((2, 10, 1))}])
    assert np.allclose(ones, static, atol=1e-5) and np.allclose(zeros, alone, atol=1e-5)
    assert not np.allclose(static, alone, atol=1e-3)
    g = np.zeros((2, 10, 1), np.float32)
    g[1] = 1                                                  # row 0 closed, row 1 open
    mix = run([{"name": "b", "mask": box, "token_gate": mx.array(g)}])
    assert np.allclose(mix[0], alone[0], atol=1e-5) and np.allclose(mix[1], static[1], atol=1e-5)


def test_hysteresis_and_gate_arms_are_causal():
    S = np.zeros((12, 2), bool)
    S[5, 0] = True
    O = W.hysteresis(S, S, 3, 2)
    assert O[:, 0].tolist() == [False] * 5 + [True] * 4 + [False] * 3
    rng = np.random.default_rng(1)
    Hv, Hf = rng.standard_normal((10, 8)), rng.standard_normal((10, 8))
    A = (rng.standard_normal((2, 8)), np.zeros(2))
    B = (rng.standard_normal((2, 8)), np.zeros(2))
    arms = W.gate_arms(Hv, Hf, list(range(10)), A, B, lambda t: np.full((len(t), 2), 0.6), 2, 0.2, hyst=(0,))
    same = arms["same"]
    assert np.array_equal(arms["lag/N0"][2:], same[1:-1])                         # step t-1's own gates
    hid = (1 / (1 + np.exp(-(Hf @ B[0].T)))) > 0.5
    assert np.array_equal(arms["hidden/N0"][2:], hid[1:-1])                      # never token t's own state
    assert arms["token/N0"][2:].all()


def test_part_q_end_to_end_on_the_toy(tmp_path):
    h = _host()
    L, N = h.L, h.N
    rng = np.random.default_rng(2)
    own = rng.integers(0, 4, size=(L, N))
    tiles = {"layers": np.arange(L), "core": own == 0, "box:x": own == 1, "box:y": own == 2,
             "pair:x__z": own == 3}
    pd = tmp_path / "part"
    pd.mkdir()
    np.savez(pd / "tiles.npz", **tiles)
    alpha = {"x": "aeiou", "y": "klmnp", "english": "rstvw"}

    def corpus(name):
        p = tmp_path / f"{name}.jsonl"
        rows = []
        for split in ("fit", "held"):
            for _ in range(8):
                rows.append({"split": split, "text": "".join(rng.choice(list(alpha[name]), 40))})
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return str(p)
    cx, cy, ce = corpus("x"), corpus("y"), corpus("english")
    (pd / "partition.json").write_text(json.dumps({"fingerprint": "toy:abc", "checkpoint": "toy@abc", "rule": "t",
                                                   "boxes": {"x": {"target_corpus": cx}, "y": {"target_corpus": cy}}}))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"held": {"x": {"spec": f"x={cx}@40"}, "y": {"spec": f"y={cy}@40"}}}))
    out = tmp_path / "out"
    out.mkdir()
    a = W.parser().parse_args(["--out", str(out), "--partition", str(pd), "--boxes", "x,y", "--battery-meta",
                               str(meta), "--english", ce, "--fit-docs", "6", "--held-docs", "4", "--switch-docs", "4",
                               "--tokens", "24", "--prompt", "4", "--probe-layer", "1", "--quality-arms",
                               "lag/N0,hidden/N16,token/N16", "--kernel-arms", "lag/N16", "--kernel-streams", "2"])
    receipt = {}
    streams, A, B = W.part_q(a, h, print, receipt)
    q = receipt["part_q"]
    assert set(q["held_ce"]) == {"full", "core", "oracle", "same", "lag/N0", "hidden/N16", "token/N16"}
    assert q["held_docs_per_set"] == {"x": 4, "y": 4, "english": 4, "switch": 4}
    assert sorted(map(tuple, q["gate_groups"])) == [("x",), ("y",)]               # pair x__z opens with x
    for arm in q["held_ce"]:
        assert abs(q["held_ce"][arm]["dce_vs_full"]["x"]) < 50
    assert q["held_ce"]["full"]["dce_vs_full"]["x"] == 0.0
    assert q["arms"]["same"]["missed_per_needed"] == 0.0 and q["arms"]["same"]["exact_set_match_rate"] == 1.0
    assert 0.0 <= q["arms"]["lag/N16"]["missed_per_needed"] <= q["arms"]["lag/N0"]["missed_per_needed"] + 1e-9
    assert len(streams["same"]) == 2 and streams["same"][0].shape == (24, 2)
    assert A[0].shape == (2, h.d) and (out / "probes.npz").exists()


def test_window_hysteresis_matches_the_runtime_gate():
    """The window script's hysteresis() and NextTokenGate (prefill, then advance per step) apply one rule."""
    from mlx2.runtime.amoeba_blocks import layout as L
    from mlx2.runtime.amoeba_blocks import nexttok as NT
    from mlx2.runtime.amoeba_blocks import ops as O
    g = L.TileGraph(["a", "b"], ["core", "box:a", "box:b"], [(), (0,), (1,)])
    t = O.GateTables.from_graph(g)
    rng = np.random.default_rng(4)
    T, prompt = 30, 5
    S = rng.random((T, 2)) < 0.3
    P = rng.random((T, 2)) < 0.3
    prior = NT.TokenPrior.from_probs(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32))
    for N in (0, 2, 7):
        want = W.hysteresis(P, S, N, prompt)
        gate = NT.NextTokenGate(t, 1, NT.NextTokenConfig(hysteresis=N, source="token"), prior=prior,
                                gate_fn=NT.reference_tile_gates)
        gate.prefill(0, box_open=mx.array(S[:prompt].astype(np.uint8)))
        for s in range(prompt, T):
            gate.advance(tokens=mx.array([int(P[s, 0]) + 2 * int(P[s, 1])]))
            assert np.array(gate.current())[0, 1:].astype(bool).tolist() == want[s].tolist(), (N, s)


def test_window_wires_and_runs_the_watcher(tmp_path, monkeypatch):
    """2026-11-19 19:43 stop (footprint 90 GB > 88, 69.3 GB MLX active, weights unwired): --wired-gb (env
    AMOEBA_WIRED_GB, default auto) is parsed and handed to ops.prepare_wiring, which is called before the watchdog
    and setup_mlx (the first Metal call); the watchdog runs with its watcher and a stop receipt; a bad value fails at
    parse time. (No part runs here, so wire_weights after the Host load is not reached: Part Q needs the model.)"""
    from research.amoeba.carve import ops as CO
    calls = []
    monkeypatch.setattr(CO, "prepare_wiring", lambda spec: calls.append(("prepare", spec)) or {})
    monkeypatch.setattr(CO, "footprint_watchdog", lambda *a, **k: calls.append(("watchdog", k.get("watcher"),
                                                                                 Path(k["receipt"]).name)))
    monkeypatch.setattr(CO, "setup_mlx", lambda *a, **k: calls.append(("setup",)))
    monkeypatch.setenv("AMOEBA_WIRED_GB", "77")
    assert W.parser().parse_args([]).wired_gb == "77"
    with pytest.raises(SystemExit):
        W.parser().parse_args(["--wired-gb", "nan"])
    monkeypatch.setattr(sys, "argv", ["w", "--out", str(tmp_path / "o"), "--parts", ""])
    assert W.main() == 0
    assert calls == [("prepare", "77"), ("watchdog", True, "watchdog-stop.json"), ("setup",)]
    assert "watch" in json.loads((tmp_path / "o" / "receipt.json").read_text())
