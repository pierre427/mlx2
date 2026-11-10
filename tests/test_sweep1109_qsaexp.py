"""Bug sweep 2026-11-09, lane qsaexp (QX-1..QX-7): CPU regressions.

QX-1/2/6/7 exercise the DS4.1 research simulators on synthetic traces (no
capture, no checkpoint); QX-3/4/5 the QSA indexer trainer helpers.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)
import mlx.optimizers as optim
from mlx import nn

from mlx2.runtime.models import qsa_indexer_train as qt

ATLAS = Path(__file__).resolve().parents[1] / "scripts" / "research" / "ds41_atlas"
_RESEARCH_MODULES = ("common", "costs", "eventsim", "heads", "heads_prev", "cache_sim")


@pytest.fixture
def research(monkeypatch):
    """Import the research scripts by their own flat names, then forget them."""
    saved = {name: sys.modules.pop(name) for name in _RESEARCH_MODULES if name in sys.modules}
    monkeypatch.syspath_prepend(str(ATLAS))
    monkeypatch.syspath_prepend(str(ATLAS / "sched"))
    try:
        yield lambda name: importlib.import_module(name)
    finally:
        for name in _RESEARCH_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


# ---------------------------------------------------------------- eventsim (QX-1, QX-2)
L, E = 40, 384


def _trace(token_rows):
    """Two warm-up prefill rows (experts 0..5 at every layer) then decode rows."""
    base = [[0, 1, 2, 3, 4, 5]] * L
    ids = np.array([base, base] + token_rows, np.int16)
    T = len(ids)
    steps = [(0, 0, [np.unique(ids[0:2, layer]).tolist() for layer in range(L)])]
    steps += [(1, r) for r in range(2, T)]
    return {"ids": ids, "steps": steps, "freq_train": np.zeros(L * E, np.int64),
            "tok_split": np.ones(T, np.int8), "tok_decode": np.r_[0, 0, np.ones(T - 2)].astype(np.int8),
            "tok_req": np.zeros(T, np.int32)}


def _token(cold):
    """One decode row: experts 0..5 everywhere except ``{layer: cold expert}``."""
    row = [[0, 1, 2, 3, 4, 5] for _ in range(L)]
    for (layer, expert) in cold.items():
        row[layer] = [0, 1, 2, 3, 4, expert]
    return row


def _run(ES, C, tok, **kw):
    ES._TR = _trace([tok])
    ES._PR = {}
    sim = ES.Sim(C.make(), ram_experts=4000, **kw)
    out = sim.run()
    return out["heldout"]["stall_s_per_token"] * 1e3, sim.st["loads_prefetch"]


def test_eventsim_demand_read_wins_over_prefetch_issued_at_same_router_instant(research):
    ES, C = research("eventsim"), research("costs")
    svc = C.expert_ms("nvme", C.make())
    try:
        # Cold expert at layer 0 (demand) and at layer 1 (perfect d1 hint issued at layer 0's router).
        (stall, loads) = _run(ES, C, _token({0: 100, 1: 200}), policy="lru", prefetch="oracle_d1", k=6)
        assert loads == 1
        assert stall == pytest.approx(svc, rel=1e-9)  # was 2 x svc: demand waited behind the prefetch
        (stall, _) = _run(ES, C, _token({0: 100, 1: 200}), policy="lru", prefetch="none")
        assert stall == pytest.approx(2 * svc, rel=1e-9)
        (stall, loads) = _run(ES, C, _token({1: 200}), policy="lru", prefetch="oracle_d1", k=6)
        assert (stall, loads) == (0.0, 1)
    finally:
        ES._TR = ES._PR = None


def test_eventsim_pinned_layer_still_issues_d1_and_xtok_hints(research):
    ES, C = research("eventsim"), research("costs")
    try:
        (stall, loads) = _run(ES, C, _token({1: 200}), policy="lru", prefetch="oracle_d1", k=6,
                              pin_layers=[0])
        assert loads == 1 and stall == 0.0  # was 0 loads, one full miss at layer 1
        # xtokAB is issued at layer 0's router instant: pinning layer 0 must not disable it.
        (stall, loads) = _run(ES, C, _token({20: 300}), policy="lru", prefetch="oracle_xtok", k=6,
                              pin_layers=[0])
        assert loads == 1 and stall == 0.0
    finally:
        ES._TR = ES._PR = None


# ---------------------------------------------------------------- trainer (QX-3, QX-4, QX-5)
def _indexer(hidden, seed=0, **kw):
    mx.random.seed(seed)
    idx = qt.TrainableIndexer(hidden, heads=2, head_dim=16, rope_base=1e4, **kw)
    mx.eval(idx.parameters())
    return idx


def test_hybrid_kl_without_free_block_refuses_instead_of_nan_gradients():
    length, hidden = 40, 16
    idx = _indexer(hidden)
    x = mx.random.normal((length, hidden))
    target = qt.block_teacher(mx.random.normal((4, length, 8)), mx.random.normal((2, length, 8)), 0.5)
    fixed = qt.fixed_blocks(length, 16, 128)
    with pytest.raises(ValueError, match="no query has a valid block"):
        nn.value_and_grad(idx, lambda m: qt.kl_loss(m.scores(x), target, fixed))(idx)
    # One free block is enough: finite loss and gradients.
    fixed = qt.fixed_blocks(length, 4, 5)
    (loss, grads) = nn.value_and_grad(idx, lambda m: qt.kl_loss(m.scores(x), target, fixed))(idx)
    from mlx.utils import tree_flatten

    assert np.isfinite(float(loss.item()))
    assert all(np.isfinite(np.array(g)).all() for (_, g) in tree_flatten(grads))


def _ckpt(directory, indexers, opts, step):
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for (i, idx) in indexers.items():
        tensors.update(idx.export(f"model.layers.{i}.self_attn"))
    mx.save_safetensors(str(directory / "qsa-indexers.safetensors"), tensors)
    mx.save_safetensors(str(directory / "trainer-state.safetensors"), qt.trainer_state(indexers, opts))
    (directory / "config.json").write_text(json.dumps({"qsa": {"trained_steps": step}}))


def test_resume_refuses_a_learning_rate_the_checkpoint_would_override(tmp_path):
    hidden, length = 32, 64
    x = mx.random.normal((length, hidden))
    target = mx.where(qt.block_valid(length),
                      mx.softmax(_indexer(hidden, seed=7).scores(x) * 0.05, axis=-1), 0.0)

    def fresh(lr):
        return ({5: _indexer(hidden, seed=1)},
                {5: optim.AdamW(learning_rate=lr, betas=[0.9, 0.95], weight_decay=0.0,
                                bias_correction=True)})

    (idx, opt) = fresh(3e-4)
    (_, g) = nn.value_and_grad(idx[5], lambda m: qt.kl_loss(m.scores(x), target))(idx[5])
    opt[5].update(idx[5], g)
    mx.eval(idx[5].parameters(), opt[5].state)
    _ckpt(tmp_path / "step1", idx, opt, 1)
    with pytest.raises(ValueError, match="learning rate"):
        qt.load_trainer_checkpoint(tmp_path / "step1", *fresh(1e-5), step=1)
    receipt = qt.load_trainer_checkpoint(tmp_path / "step1", *fresh(3e-4), step=1)
    assert receipt["learning_rate"] == pytest.approx(3e-4, rel=1e-6)


def test_mass_breakdown_hybrid_columns_follow_budget_and_variant_selection():
    rng = np.random.default_rng(3)
    length = 1200
    n_blocks = length // 4
    raw = rng.random((length, n_blocks)).astype(np.float32) ** 4
    valid = np.array(qt.block_valid(length))
    t = np.where(valid, raw, 0.0)
    t = t / np.maximum(t.sum(-1, keepdims=True), 1e-12)
    target = mx.array(t)
    s = rng.normal(size=(length, n_blocks)).astype(np.float32)
    scores = {"main": mx.where(qt.block_valid(length), mx.array(s), -mx.inf)}
    # Budget 64 blocks < default 16 sink + 128 recent: no hybrid column may appear (was > 1.0).
    out = qt.mass_breakdown(target, scores, 64)
    assert not any(k.startswith("hybrid") for k in out)
    assert all(v <= 1.0 + 1e-5 for (k, v) in out.items() if k != "rows")
    # A variant's own rule (sink 2, recent 3) at budget 16: independent numpy reference.
    topk, sink, recent = 16, 2, 3
    out = qt.mass_breakdown(target, scores, topk,
                            selections={"main": {"sink_blocks": sink, "recent_blocks": recent}})
    first = (topk + 1) * 4 - 1
    want_arm, want_oracle = [], []
    for p in range(first, length):
        newest = (p + 1) // 4
        complete = np.arange(n_blocks) < newest
        fixed = (np.arange(n_blocks) < sink) | ((np.arange(n_blocks) >= newest - recent) & complete)
        free = complete & ~fixed
        rest = topk - sink - recent
        base = t[p][fixed].sum()
        cand = np.where(free)[0]
        want_arm.append(base + t[p][cand[np.argsort(-s[p][cand])[:rest]]].sum())
        want_oracle.append(base + np.sort(t[p][cand])[::-1][:rest].sum())
    assert out["hybrid:main"] == pytest.approx(float(np.mean(want_arm)), rel=1e-4)
    assert out["hybrid:oracle:main"] == pytest.approx(float(np.mean(want_oracle)), rel=1e-4)
    assert "hybrid:oracle" not in out  # default 16+128 rule does not fit a 16-block budget


# ---------------------------------------------------------------- cache_sim cycle (QX-6)
def test_cache_sim_cycle_reuse_ema_decays_with_elapsed_steps(research):
    cs = research("cache_sim")
    # Key A: five accesses long ago; key B: two recent accesses.  Same layer, capacity 2.
    acc = [(3, 1, s) for s in range(5)] + [(3, 2, 200), (3, 2, 201), (3, 9, 202), (3, 2, 203)]
    miss = cs.simulate(acc, 2, "cycle", alpha=0.1)
    # Inserting key 9 must evict the stale A (decayed EMA), so B's next access hits.
    # A lifetime-count EMA (1 - 0.9^n) evicted B instead.
    assert not miss[-1]


# ---------------------------------------------------------------- heads_prev beta (QX-7)
def _heads_prev_fixture(tmp_path, research, w_seed):
    hp = research("heads_prev")
    rng = np.random.default_rng(0)
    layers, hidden, n_req, per = 2, 8, 10, 30
    T = n_req * per
    ids = np.stack([np.stack([rng.choice(384, 6, replace=False) for _ in range(layers)])
                    for _ in range(T)]).astype(np.int16)
    req = np.repeat(np.arange(n_req), per).astype(np.int32)
    split = np.where(req >= 8, 1, 0).astype(np.int8)
    dec = np.ones(T, np.int8)
    dec[::per] = 0
    xs = {L: rng.normal(size=(T, hidden)).astype(np.float32) for L in (0, 1, 39)}
    gates = [(rng.normal(size=(384, hidden)).astype(np.float32), np.zeros(384, np.float32))
             for _ in range(layers)]
    out = tmp_path / f"out{w_seed}"
    cap = tmp_path / f"cap{w_seed}"
    out.mkdir()
    cap.mkdir()
    (cap / "replies.jsonl").write_text("".join(
        json.dumps({"split": "heldout" if r >= 8 else "train"}) + "\n" for r in range(n_req)))
    wr = np.random.default_rng(100 + w_seed)
    np.savez(out / "xtok_heads.npz", W_AB_h1=(wr.normal(size=(2 * hidden, layers * 384)) * 50).astype(np.float16),
             mu_AB_h1=np.zeros(2 * hidden, np.float32), mr_AB_h1=np.zeros(layers * 384, np.float32),
             lam_AB_h1=np.array(10.0))
    (out / "heads_eval.json").write_text(json.dumps({"evals": {}}))
    np.savez(out / "preds.npz", pred_d1=np.zeros((1,), np.int16))
    hp.OUT, hp.CAPTURE, hp.LAYERS = str(out), str(cap), layers
    hp.load_trace = lambda: {"ids": ids, "tok_req": req, "tok_decode": dec, "tok_split": split}
    hp.X = lambda L: xs[L]
    hp.load_gate = lambda L: gates[L]
    hp.main()
    return json.loads((out / "heads_eval.json").read_text())["evals"]["xtok_ABp_h1"]


def test_heads_prev_beta_is_selected_without_the_saved_all_train_ridge(tmp_path, research):
    # The saved ridge saw the validation rows; beta selection must not depend on it at all.
    a = _heads_prev_fixture(tmp_path, research, 1)
    b = _heads_prev_fixture(tmp_path, research, 2)
    assert a["beta_by_val_recall8"] == b["beta_by_val_recall8"]
    assert "FIT" in a["beta_selection"]


def test_review_trainer_free_block_check_uses_the_block_constant():
    src = (Path(__file__).resolve().parents[1] / "scripts/train_qsa_indexers.py").read_text()
    assert "int(train.shape[1]) // qt.BLOCK <= sel[" in src
    assert "int(train.shape[1]) // 4 <=" not in src
