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


# ---------------------------------------------------------------------------
# Memory (2026-11-20 watchdog stops): the chunked paths give the numbers of the whole-matrix ones
# ---------------------------------------------------------------------------


def _fit_logistic_unchunked(X, Y, steps=400, lr=0.05, l2=1e-3):
    """fit_logistic as it was before the chunked path (47214c256), verbatim: the reference."""
    import mlx.nn as nn
    import mlx.optimizers as optim
    mu = X.mean(0, dtype=np.float64).astype(np.float32)
    sd = (X.std(0, dtype=np.float64) + 1e-4).astype(np.float32)
    Xs = mx.array((X - mu) / sd)
    Yt = mx.array(Y.astype(np.float32))
    pos = np.clip(Y.mean(0), 1e-4, 1 - 1e-4)
    lin = nn.Linear(X.shape[1], Y.shape[1])
    lin.weight = mx.zeros_like(lin.weight)
    lin.bias = mx.array(np.log(pos / (1 - pos)).astype(np.float32))
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(m):
        z = m(Xs)
        return nn.losses.binary_cross_entropy(z, Yt, with_logits=True, reduction="mean") + l2 * (m.weight ** 2).sum()
    step = nn.value_and_grad(lin, loss_fn)
    for _ in range(steps):
        loss, g = step(lin)
        opt.update(lin, g)
        mx.eval(lin.parameters(), loss)
    W_ = np.array(lin.weight) / sd[None]
    b = np.array(lin.bias) - (W_ * mu[None]).sum(1)
    return W_.astype(np.float32), b.astype(np.float32), float(loss)


def _states(n=13, T=11, d=40, seed=5):
    rng = np.random.default_rng(seed)
    H = (rng.standard_normal((n, T, d)) * rng.uniform(0.1, 30, d) + rng.standard_normal(d) * 5).astype(np.float16)
    Y = rng.random((n, 3)) < 0.4
    return H, Y


@pytest.mark.parametrize("docs", [1, 2, 5, 13, 100])
def test_column_moments_are_numpys_bit_for_bit(docs):
    H, _ = _states()
    for lo, hi in ((1, None), (None, -1)):
        X = H[:, lo:hi].reshape(-1, H.shape[-1]).astype(np.float32)
        mean, std = W.column_moments(W.Rows(H, lo, hi, docs))
        assert np.array_equal(mean, X.mean(0, dtype=np.float64))
        assert np.array_equal(std, X.std(0, dtype=np.float64))
        assert np.array_equal(np.concatenate(list(W.Rows(H, lo, hi, docs).chunks())), X)
    for d in (1, 9000):                       # one column (numpy sums it pairwise); past numpy's 8192-element buffer
        Hd = (np.random.default_rng(d).standard_normal((5, 300 if d == 1 else 7, d)) * 3 + 1).astype(np.float16)
        for lo, hi in ((1, None), (None, -1)):
            X = Hd[:, lo:hi].reshape(-1, d).astype(np.float32)
            mean, std = W.column_moments(W.Rows(Hd, lo, hi, docs))
            assert np.array_equal(mean, X.mean(0, dtype=np.float64)) and np.array_equal(std, X.std(0, dtype=np.float64))
    X2 = H.reshape(-1, H.shape[-1]).astype(np.float32)
    m2, s2 = W.column_moments(W.Rows(X2, rows=docs * 7))
    assert np.array_equal(m2, X2.mean(0, dtype=np.float64)) and np.array_equal(s2, X2.std(0, dtype=np.float64))


@pytest.mark.parametrize("docs", [1, 4, 13])
def test_chunked_probe_fit_equals_the_whole_matrix_fit(docs):
    H, flab = _states()
    T = H.shape[1]
    Y = np.repeat(flab, T - 1, axis=0)
    for lo, hi in ((1, None), (None, -1)):
        X = H[:, lo:hi].reshape(-1, H.shape[-1]).astype(np.float32)
        ref = _fit_logistic_unchunked(X, Y, steps=60)
        got = W.fit_logistic(W.Rows(H, lo, hi, docs), Y, steps=60)
        assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1]) and ref[2] == got[2]
    assert all(np.array_equal(p, q) for p, q in zip(W.fit_logistic(X, Y, steps=60)[:2],
                                                     _fit_logistic_unchunked(X, Y, steps=60)[:2]))


def test_chunked_probe_fit_holds_no_whole_host_copy():
    """Host peak (numpy reports to tracemalloc): the unchunked fit makes float32 + float64 + standardized copies of
    X (>= 3 x its float32 bytes); the chunked one holds one chunk.  Device: the standardized rows once (+ a chunk)."""
    import tracemalloc
    rng = np.random.default_rng(0)
    n, T, d = 64, 33, 256
    H = rng.standard_normal((n, T, d)).astype(np.float16)
    Y = np.repeat(rng.random((n, 2)) < 0.5, T - 1, axis=0)
    xbytes = n * (T - 1) * d * 4

    def host_peak(fn):
        tracemalloc.start()
        fn()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak
    old = host_peak(lambda: _fit_logistic_unchunked(H[:, 1:].reshape(-1, d).astype(np.float32), Y, steps=2))
    mx.reset_peak_memory()
    new = host_peak(lambda: W.fit_logistic(W.Rows(H, 1, None, 4), Y, steps=2))
    dev = mx.get_peak_memory()
    assert old >= 3 * xbytes
    assert new <= 0.4 * xbytes and new * 8 < old, (new, old, xbytes)   # a 4-doc chunk of 64 docs + its float64 copies
    assert dev <= 1.25 * xbytes, (dev, xbytes)                 # the standardized rows, filled in place


def _fn_tiny_host():
    import test_amoeba_map_units_cpu as MU
    from mlx2.runtime.models.qwen4_exp import Model, ModelArgs
    cfg = MU._qwen_cfg(MU.MODELS / "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP", hc_lowrank=16, ple_embed_dim=32,
                       ple_layer_ids=[2], ngram_vocab_size_base=1000, split_ngram_parts=4, indexer_n_heads=2,
                       indexer_head_dim=32, indexer_budget=16, mtp={"num_hidden_layers": 0})
    mx.random.seed(0)
    m = MU.randomize_zeros(Model(ModelArgs.from_dict(cfg)))
    return Host(model=m, tokenizer=MU.Tok(), fingerprint="tiny:00000000", log=lambda *_: None, moe_units="neuron")


def _capture_all_layers(host, seqs, v, bs=8):
    """capture as it was before the one-layer tap (Host.record keeps every layer input): the reference."""
    Hv, Hf = [], []
    for i in range(0, len(seqs), bs):
        host.ctl.reset()
        recs, out = host.record(mx.array(seqs[i:i + bs]))
        hv = recs[v][0].astype(mx.float16)
        hf = (out[0] if isinstance(out, tuple) else out).astype(mx.float16)
        mx.eval(hv, hf)
        Hv.append(np.array(hv))
        Hf.append(np.array(hf))
    return np.concatenate(Hv), np.concatenate(Hf)


@pytest.mark.parametrize("which", ["toy", "flash_next_tiny"])
def test_one_layer_tap_capture_equals_the_all_layer_record(which):
    h = _host() if which == "toy" else _fn_tiny_host()
    rng = np.random.default_rng(7)
    seqs = [rng.integers(0, 64, 12).tolist() for _ in range(7)]
    for v in (0, 1, h.L - 1):
        for bs in (8, 3):                                     # same batch: bitwise (the batch itself may round)
            ref = _capture_all_layers(h, seqs, v, bs)
            got = W.capture(h, seqs, v, bs)
            assert got[0].dtype == np.float16 and np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])
    if which == "flash_next_tiny":
        assert ref[0].shape[-1] == h.cfg["hc_count"] * h.d          # the hyper-connection residual is tapped


def test_memory_estimate_for_the_flash_next_window():
    """Flash-Next shapes of the 2026-11-20 run: fit 216 docs, held 144 + 24 switch, T 256, layer-3 residual 10240
    (4 x 2560 hyper-connections), final hidden 2560, 48 layers x 327680 neuron units, 36 gate groups, vocab 248320.
    The script-owned peak stays well under 10 GiB above the model."""
    est = W.memory_estimate(216, 168, 256, 32, 10240, 2560, 48, 327680, 36, 248320)
    assert est["peak"] <= 6.0, est
    assert 3.0 <= est["probe_A_same_step"] <= 4.0                     # 1.3 fit states + 2.1 device rows + chunk
    assert est["masked_ce"] <= 5.0                                     # 1.1 device masks + 3.4 logits (B 8)
    small = W.memory_estimate(216, 168, 256, 32, 10240, 2560, 48, 327680, 36, 248320, ce_batch=4)
    assert small["masked_ce"] < est["masked_ce"]


def _masked_ce_unchunked(host, seqs, masks, gates, prompt, dense_layers=(), bs=8):
    """masked_ce as it was (host.logits_at: the [B, T, V] float32 logits, then the gather): the reference."""
    from research.amoeba.carve.carving import ce_rows
    out = []
    for i in range(0, len(seqs), bs):
        ids = mx.array(seqs[i:i + bs])
        c = host.ctl
        c.reset()
        if masks is not None:
            c.core = masks[0]
            c.dense_layers = tuple(dense_layers)
            c.boxes = [{"name": k, "mask": m, "token_gate": mx.array(gates[k][i:i + bs].astype(np.float16))[..., None]}
                       for k, m in masks[1].items()]
        T = ids.shape[1]
        lg = host.logits_at(ids, list(range(prompt - 1, T - 1)))
        ce = ce_rows(lg, ids[:, prompt:])
        mx.eval(ce)
        out.extend(np.array(ce).mean(1).tolist())
        c.reset()
    return out


@pytest.mark.parametrize("which", ["toy", "flash_next_tiny"])
def test_masked_ce_gather_before_cast_is_bitwise(which):
    h = _host() if which == "toy" else _fn_tiny_host()
    rng = np.random.default_rng(8)
    seqs = [rng.integers(0, 64, 14).tolist() for _ in range(5)]
    widths = h.widths
    own = [rng.integers(0, 3, n) for n in widths]
    L = h.L
    core = np.stack([o == 0 for o in own]) if len(set(widths)) == 1 else None
    if core is None:
        pytest.skip("non-uniform widths")
    groups = {("a",): np.stack([o == 1 for o in own]), ("b",): np.stack([o == 2 for o in own])}
    masks = W.device_masks(core, groups)
    gates = {k: rng.random((5, 14)) < 0.5 for k in groups}
    for m, g, dense in ((None, None, ()), (masks, gates, ()), (masks, gates, range(1))):
        assert W.masked_ce(h, seqs, m, g, 4, dense) == _masked_ce_unchunked(h, seqs, m, g, 4, dense)
    assert L == len(widths)
    # FN's logits are bf16 (the CPU MoE gather cannot run bf16): gather-then-cast == cast-then-gather, bitwise
    from research.amoeba.carve import host as H
    lg = mx.array(rng.standard_normal((3, 14, 50)).astype(np.float32) * 20).astype(mx.bfloat16)
    pos = list(range(3, 13))
    assert np.array_equal(np.array(H._take(mx, lg, pos).astype(mx.float32)), np.array(H._take(mx, lg.astype(mx.float32), pos)))


def test_part_q_is_chunk_invariant(tmp_path):
    """Part Q end to end on the toy: the probe-fit chunking changes memory only; every reported number is the same
    (the memory readings and the estimate aside).  (Capture and CE keep their batch of 8: a forward's rounding can
    depend on its batch, so those are compared at the same batch, above.)"""
    h = _host()
    L, N = h.L, h.N
    rng = np.random.default_rng(2)
    own = rng.integers(0, 4, size=(L, N))
    pd = tmp_path / "part"
    pd.mkdir()
    np.savez(pd / "tiles.npz", layers=np.arange(L), core=own == 0, **{"box:x": own == 1, "box:y": own == 2,
                                                                       "pair:x__z": own == 3})
    alpha = {"x": "aeiou", "y": "klmnp", "english": "rstvw"}

    def corpus(name):
        p = tmp_path / f"{name}.jsonl"
        rows = [{"split": s, "text": "".join(rng.choice(list(alpha[name]), 40))} for s in ("fit", "held")
                for _ in range(8)]
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return str(p)
    cx, cy, ce = corpus("x"), corpus("y"), corpus("english")
    (pd / "partition.json").write_text(json.dumps({"fingerprint": "toy:abc", "checkpoint": "toy@abc", "rule": "t",
                                                   "boxes": {"x": {"target_corpus": cx}, "y": {"target_corpus": cy}}}))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"held": {"x": {"spec": f"x={cx}@40"}, "y": {"spec": f"y={cy}@40"}}}))

    def run(tag, *extra):
        out = tmp_path / tag
        out.mkdir()
        a = W.parser().parse_args(["--out", str(out), "--partition", str(pd), "--boxes", "x,y", "--battery-meta",
                                   str(meta), "--english", ce, "--fit-docs", "6", "--held-docs", "4",
                                   "--switch-docs", "4", "--tokens", "24", "--prompt", "4", "--probe-layer", "1",
                                   "--quality-arms", "lag/N0,hidden/N16,token/N16", "--kernel-arms", "lag/N16",
                                   "--kernel-streams", "2", *extra])
        receipt = {}
        streams, A, B = W.part_q(a, h, lambda *_: None, receipt)
        phases = [r["phase"] for r in receipt["memory"]]
        assert phases[:4] == ["docs", "fit capture, first batch before clear", "fit capture", "probe A"], phases
        assert "CE full, first batch before clear" in phases and "CE same, first batch before clear" in phases
        assert receipt["memory_estimate_gib"]["peak"] >= 0
        return receipt["part_q"], streams, A, B
    whole = run("whole", "--fit-chunk-docs", "1000")
    small = run("small", "--fit-chunk-docs", "1")
    default = run("default")
    for got in (small, default):
        assert json.dumps(got[0], sort_keys=True) == json.dumps(whole[0], sort_keys=True)
        assert all(np.array_equal(p, q) for k in whole[1] if k != "names" for p, q in zip(whole[1][k], got[1][k]))
        assert all(np.array_equal(p, q) for p, q in zip(whole[2] + whole[3], got[2] + got[3]))
