"""H56 / H56b side tile on a tiny random bf16 Qwen3 (CPU only; no checkpoint, no GPU, synthetic documents)."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from research.ens import corpus_tile as CT
from research.ens import corpus_tile_eval as E
from research.ens import lm_bus as LB
from research.ens import qwen3 as Q
from research.ens import side_tile as ST

SPECIALS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")
_SPLIT = re.compile("(" + "|".join(map(re.escape, SPECIALS)) + ")")
WORDS = ("lattice gradient photon kernel anneal ladder tile donor bus stitch neuron cache prefix scheduler quantum "
         "orbit vector sparse dense mirror harbor copper meadow lantern river signal anchor margin cipher").split()


class FakeTok:
    """Byte-level tokenizer with the Qwen chat specials as single ids (256..260)."""

    def __init__(self):
        self.ids = {s: 256 + i for i, s in enumerate(SPECIALS)}
        self.pad = self.ids["<|endoftext|>"]
        self.t = self

    def encode(self, s):
        out = []
        for part in _SPLIT.split(s):
            out += [self.ids[part]] if part in self.ids else list(part.encode())
        return out

    def token_to_id(self, s):
        return self.ids.get(s)

    def decode(self, ids):
        inv = {v: k for k, v in self.ids.items()}
        out, buf = [], []
        for i in ids:
            if i in inv:
                out.append(bytes(buf).decode(errors="replace"))
                buf = []
                out.append(inv[i])
            elif i < 256:
                buf.append(i)
        out.append(bytes(buf).decode(errors="replace"))
        return "".join(out)


@pytest.fixture(autouse=True)
def cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def tiny(layers=4, d=64):
    c = Q.Qwen3Config(hidden_size=d, intermediate_size=2 * d, num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=d // 4, rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=264,
                      tie_word_embeddings=True)
    mx.random.seed(0)
    m = Q.Qwen3(c)
    m.set_dtype(mx.bfloat16)
    m.freeze()
    mx.eval(m.parameters())
    return m


def prose(rng, n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words)).capitalize() + "."


@pytest.fixture
def world(tmp_path, monkeypatch):
    rng = random.Random(1)
    text = tmp_path / "text.jsonl"
    with text.open("w") as f:
        for _ in range(60):
            f.write(json.dumps({"text": " ".join(prose(rng, 12) for _ in range(40))}) + "\n")
    monkeypatch.setattr(LB, "TEXT", text)
    docs = []
    for i in range(8):
        docs.append({"id": f"d{i}", "title": f"Note {i}", "batch": "tuning" if i % 2 else "course",
                     "text": " ".join(prose(rng, 10) for _ in range(12))})
    split = {"train": ["d0", "d1", "d2", "d3"], "held_out": ["d4", "d5", "d6", "d7"]}
    return ST.world_from(tiny(), FakeTok(), "tiny-qwen3@000000000000", docs, split, {"corpus.jsonl": "synthetic"})


def targs(out, *extra):
    return ST.parser().parse_args(["train", "--corpus", "unused", "--out", str(out), "--width", "32", "--heads", "2",
                                   "--no-memory-gate", "--stat-rows", "32", "--log-every", "5", "--save-every", "0",
                                   "--a-len", "128", "--batch", "3", *extra])


def stats_for(W, width=32, every=1):
    lay = ST.aligned_layers(W["cfg"].num_hidden_layers, every)
    return ST.ladder_stats(W, lay, width, seed=0, n_rows=32)


def tile_for(W, *extra, tmp=Path("/nonexistent")):
    a = targs(tmp, *extra)
    mx.random.seed(0)
    t = ST.make_tile(W, stats_for(W, a.width, a.every), a)
    if t.align:
        ST.align_init(t, a.seed)                         # as train() on a fresh run
    return t, a


def toks(W, n=3, L=96):
    rng = random.Random(4)
    return mx.array([W["tok"].encode(prose(rng, 40))[:L] for _ in range(n)])


def _f32(x):
    return np.array(x.astype(mx.float32))


# ------------------------------------------------------------------ exactness and wiring
@pytest.mark.parametrize("extra", [(), ("--every", "2"), ("--write-site", "mid"), ("--gate-per-layer",),
                                   ("--arm", "align_hard"), ("--arm", "align_soft"), ("--arm", "align_grow"),
                                   ("--arm", "align_act"), ("--width", "64", "--heads", "4"),
                                   ("--alpha-b", "0")])
def test_untrained_tile_is_the_original_bitwise(world, extra):
    m = world["model"]
    T = toks(world)
    ref = _f32(m(T))
    tile, _ = tile_for(world, *extra)
    got, _ = ST.forward(m, T, tile)
    assert np.array_equal(_f32(got), ref)
    assert np.array_equal(_f32(CT.forward(m, T, None)[0]), ref)
    # a trained-like write changes the logits; zero_values (tile off) is the original again
    for u in tile.U:
        u.weight = mx.random.normal(u.weight.shape) * 0.05
    assert not np.array_equal(_f32(ST.forward(m, T, tile)[0]), ref)
    assert np.array_equal(_f32(ST.forward(m, T, tile, zero_values=True)[0]), ref)


def test_no_lora_on_the_donor(world, monkeypatch):
    assert ST.DONOR_LORA is False
    m = world["model"]
    assert not any("lora" in k for k, _ in __import__("mlx.utils").utils.tree_flatten(m.parameters()))
    tile, _ = tile_for(world, "--gate-per-layer")
    tile.set_phase("B")
    keys = [k for k, _ in __import__("mlx.utils").utils.tree_flatten(tile.parameters())]
    assert keys and not any("lora" in k for k in keys)
    assert C_trainable(m) == 0
    seen = []
    orig = Q.Block.__call__

    def spy(self, h, mask="causal", lora=None):
        seen.append(lora)
        return orig(self, h, mask, lora)
    monkeypatch.setattr(Q.Block, "__call__", spy)
    for u in tile.U:
        u.weight = mx.random.normal(u.weight.shape) * 0.05
    T = toks(world)
    import mlx.nn as nn
    vg = nn.value_and_grad(tile, lambda t: ST.forward(m, T, t)[0].astype(mx.float32).mean())
    _, g = vg(tile)
    assert seen and all(x is None for x in seen)
    assert any(float(mx.abs(v).sum().item()) > 0 for _, v in __import__("mlx.utils").utils.tree_flatten(g))


def C_trainable(m):
    from research.ens import common as C
    return C.n_params(m, trainable=True)


def test_alpha_one_is_teacher_forced(world):
    tile, _ = tile_for(world, "--alpha-b", "1.0")
    st, _ = ST.donor_pass(world["model"], toks(world), set(tile._lay))

    def capture(alpha, perturb):
        ins = []
        tile.run(st, alpha, on_input=lambda j, x: ins.append(_f32(x)),
                 on_output=(lambda j, s: s + 3.0 * mx.random.normal(s.shape)) if perturb else None)
        return ins
    mx.random.seed(1)
    a, b = capture(1.0, False), capture(1.0, True)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    a, b = capture(0.5, False), capture(0.5, True)
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[1], b[1])


def test_phase_b_default_reads_donor_at_depth(world):
    tile, a = tile_for(world)
    assert a.alpha_b == 0.5 and tile.alpha == 0.5
    tile.set_phase("B")
    assert all(not p.trainable_parameters() == {} for p in tile.P[:-1])       # every ladder projector trains in B
    for u in tile.U:
        u.weight = mx.random.normal(u.weight.shape) * 0.05
    m = world["model"]
    T = toks(world)
    L = world["cfg"].num_hidden_layers
    deep = tile._lay[-2]                                                     # deepest ladder read (block L - 1 input)
    assert tile.needed_layers() >= set(tile._lay[:-1]) and deep == L - 1
    st, _ = ST.donor_pass(m, T, tile.needed_layers())
    base = [_f32(s) for s in tile.run(st)]
    noisy = dict(st)
    noisy[deep] = st[deep] + mx.random.normal(st[deep].shape).astype(st[deep].dtype)
    out = [_f32(s) for s in tile.run(noisy)]
    assert all(np.array_equal(x, y) for x, y in zip(base[:-1], out[:-1]))     # causal in depth
    assert not np.array_equal(base[-1], out[-1])
    # the full phase-B forward (pass 2 from the write site) changes too when only a middle ladder state changes
    mid = tile._lay[2]
    noisy = dict(st)
    noisy[mid] = st[mid] + mx.random.normal(st[mid].shape).astype(st[mid].dtype)
    assert not np.array_equal(_f32(ST.forward(m, T, tile, states=st)[0]), _f32(ST.forward(m, T, tile, states=noisy)[0]))


def test_alpha_zero_ignores_donor_after_the_first_ladder(world):
    tile, _ = tile_for(world, "--alpha-b", "0")
    st, _ = ST.donor_pass(world["model"], toks(world), set(tile._lay))
    base = [_f32(s) for s in tile.run(st, 0.0)]
    noisy = {l: (h if l == tile._lay[0] else h + mx.random.normal(h.shape).astype(h.dtype)) for l, h in st.items()}
    assert all(np.array_equal(x, _f32(y)) for x, y in zip(base, tile.run(noisy, 0.0)))
    noisy0 = dict(st)
    noisy0[0] = st[0] + mx.random.normal(st[0].shape).astype(st[0].dtype)
    assert not np.array_equal(base[0], _f32(tile.run(noisy0, 0.0)[0]))
    assert tile.needed_layers(0.0) == {0, world["cfg"].num_hidden_layers - 1}


def test_alpha_schedule():
    assert ST.alpha_a(1, 100) == 1.0 and ST.alpha_a(25, 100) == 1.0
    assert ST.alpha_a(50, 100) == pytest.approx(0.5)
    assert ST.alpha_a(75, 100) == 0.0 and ST.alpha_a(100, 100) == 0.0
    assert ST.alpha_a(50, 100, end=0.5) == pytest.approx(0.75) and ST.alpha_a(100, 100, end=0.5) == 0.5
    assert ST.annealing(50, 100, 0.25) and not ST.annealing(10, 100, 0.25) and not ST.annealing(90, 100, 0.25)


# ------------------------------------------------------------------ training
def test_imitation_loss_falls(world, tmp_path):
    a = targs(tmp_path / "imit", "--arm", "imitate", "--steps-a", "40", "--steps", "0", "--lr-a", "3e-3",
              "--alpha-hold", "0.5")
    assert ST.train(a, world=world) == 0
    r = json.loads((tmp_path / "imit/train_receipt.json").read_text())
    tf = [x["tf_err"] for x in r["log"] if "tf_err" in x]
    assert len(tf) >= 4 and tf[-1] < 0.7 * tf[0]
    assert r["donor_lora"] is False and r["steps_a"] == 40
    assert "end_of_A" in r["log"][-1] and len(r["log"][-1]["end_of_A"]["drift_by_layer"]) == 4
    assert r["base_checkpoint"] == "tiny-qwen3@000000000000" and "@" in r["tile_checkpoint"]


def test_corpus_training_reduces_nll_and_eval_path(world, tmp_path):
    out = tmp_path / "corr"
    a = targs(out, "--arm", "correction", "--steps", "60", "--lr", "3e-3", "--p-text", "0.34")
    assert ST.train(a, world=world) == 0
    qd = tmp_path / "q"
    qd.mkdir()
    (qd / "recall.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"qid": "r1", "doc_split": "train", "question": "What is note 0 about?", "rephrase": "Note 0 topic?",
         "answer": "lattice", "aliases": []},
        {"qid": "r2", "doc_split": "held_out", "question": "What is note 4 about?", "rephrase": "Note 4 topic?",
         "answer": "harbor", "aliases": []}]) + "\n")
    (qd / "crossdoc.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"qid": "x1", "type": "synthesis", "split_mix": "all_train", "question": "What links notes 0 and 1?",
         "rephrase": "Notes 0 and 1 share?", "answer": "photon"},
        {"qid": "x2", "type": "graph", "split_mix": "all_train", "question": "Which note follows note 0?",
         "rephrase": "After note 0?", "choices": ["Note 1", "Note 2", "Note 3", "Note 5"], "choice_answer": "Note 1"}
    ]) + "\n")
    ev = ST.parser().parse_args(["eval", "--run", str(out), "--questions", str(qd), "--no-memory-gate"])
    assert ST.evaluate(ev, world=world) == 0
    r = json.loads((out / "receipt.json").read_text())["eval"]
    td = r["nll"]["train_docs"]
    assert td["tile"] < td["base"] - 0.1
    assert td["lora_only"] == td["base"]                      # tile off = base exactly (no donor LoRA)
    assert {"held_docs", "ordinary_text", "recall_gain_trained_minus_held", "memory_share_trained"} <= set(r["nll"])
    qa = r["qa"]
    assert any(k.endswith("recall.train.question") for k in qa) and any(".graph_mc.all_train." in k for k in qa)
    for v in qa.values():
        assert {"base", "tile", "lora_only", "n"} <= set(v)
    assert r["state_error"]["free_running"][0] > 0


@pytest.mark.parametrize("arm", ["imitate", "align_hard", "align_soft", "align_act", "align_grow"])
def test_resume_equals_uninterrupted(world, tmp_path, arm):
    common = ("--arm", arm, "--steps-a", "4", "--steps", "4", "--match-every", "2", "--tau-on", "0.2",
              "--tau-off", "0.1", "--p-text", "0.34", "--lr-a", "3e-3", "--lr", "3e-3")
    assert ST.train(targs(tmp_path / "full", *common), world=world) == 0
    cut = tmp_path / "cut"
    assert ST.train(targs(cut, *common, "--max-steps-this-launch", "3"), world=world) == 3
    assert ST.train(targs(cut, *common, "--max-steps-this-launch", "3"), world=world) == 3
    assert ST.train(targs(cut, *common), world=world) == 0
    a = mx.load(str(tmp_path / "full/tile.safetensors"))
    b = mx.load(str(cut / "tile.safetensors"))
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(np.array(a[k]), np.array(b[k])), k
    ra = json.loads((tmp_path / "full/train_receipt.json").read_text())
    rb = json.loads((cut / "train_receipt.json").read_text())
    assert ra["tile_checkpoint"] == rb["tile_checkpoint"]


# ------------------------------------------------------------------ H56b matching
def test_match_ties_threshold_hysteresis_and_uniqueness():
    on, off = 0.5, 0.3
    s = np.full((6, 5), 0.05)
    mask = np.array([1, 0, 1, 0, 0, 0], bool)
    idx = np.array([0, 0, 1, 0, 0, 0])
    s[0, 0] = 0.4             # tied, between off and on: stays tied
    s[1, 4] = 0.4             # free, between: stays free
    s[2, 1] = 0.2             # tied, below off: unties
    s[3, 2] = 0.7             # free, above on: ties to 2
    s[4, 2], s[4, 3] = 0.6, 0.55   # wants 2 (taken by the better row 3): ties to 3
    s[5, 0] = 0.9             # wants 0, held by a kept tie: never stolen
    m, i = ST.match_ties(s, mask, idx, on, off)
    assert m.tolist() == [True, False, False, True, True, False]
    assert i[0] == 0 and i[3] == 2 and i[4] == 3
    m2, i2 = ST.match_ties(s, m, i, on, off)                 # no churn on a rerun
    assert m2.tolist() == m.tolist() and (i2[m2] == i[m]).all()
    assert len(set(i[m].tolist())) == m.sum()


def _opt_with_moments(tile):
    import mlx.optimizers as optim
    tile.set_phase("A")
    opt = optim.AdamW(learning_rate=1e-3)
    opt.init(tile.trainable_parameters())
    for blk in opt.state["blocks"]:
        node = blk["mlp"]["down_proj"]["weight"] if "down_proj" in blk["mlp"] else blk["mlp"]["coef"]
        node["m"], node["v"] = mx.ones_like(node["m"]), mx.ones_like(node["v"])
    return opt


def test_aligned_arms_start_tied_in_place(world):
    for arm in ("align_hard", "align_act", "align_grow"):
        tile, _ = tile_for(world, "--arm", arm)
        assert tile.tied_fraction() == 1.0
        for j, blk in enumerate(tile.blocks):
            idx = np.array(blk.mlp.tie_idx)
            assert len(set(idx.tolist())) == idx.size                       # distinct donor columns
            assert np.allclose(ST._np(blk.mlp.down_proj.weight), ST._np(tile.dictionary(j, blk.mlp.tie_idx)))


def test_align_hard_untie_retie_resets_moments(world):
    tile, a = tile_for(world, "--arm", "align_hard", "--tau-on", "0.95", "--tau-off", "0.3")
    opt = _opt_with_moments(tile)
    mlp = tile.blocks[1].mlp
    D = ST._np(tile.dictionary(1))
    a.match_every = 1
    rep = ST.align_step(tile, a, opt=opt)                                     # ties hold at the start
    assert rep["tied_frac"] == 1.0
    own = int(mlp.tie_idx[5].item())
    W = ST._np(mlp.down_proj.weight)
    W[:, 5] = -D[:, own]                                                      # the shadow drifted away
    mlp.down_proj.weight = mx.array(W)
    ST.align_step(tile, a, opt=opt)
    assert float(mlp.tie_mask[5].item()) == 0.0
    m = np.array(opt.state["blocks"][1]["mlp"]["down_proj"]["weight"]["m"])
    assert (m[:, 5] == 0).all() and (m[:, 6] == 1).all()
    free_cols = sorted(set(range(D.shape[1])) - set(np.array(mlp.tie_idx)[np.array(mlp.tie_mask) > 0.5].tolist()))
    c = free_cols[0]
    W = ST._np(mlp.down_proj.weight)
    W[:, 5] = D[:, c] + 1e-3 * np.random.default_rng(0).normal(size=D.shape[0])
    mlp.down_proj.weight = mx.array(W)
    opt.state["blocks"][1]["mlp"]["down_proj"]["weight"]["m"] = mx.ones_like(mx.array(m))
    ST.align_step(tile, a, opt=opt)
    assert float(mlp.tie_mask[5].item()) == 1.0 and int(mlp.tie_idx[5].item()) == c
    assert np.allclose(ST._np(tile.down_eff(1))[:, 5], D[:, c], atol=1e-6)
    assert (np.array(opt.state["blocks"][1]["mlp"]["down_proj"]["weight"]["m"])[:, 5] == 0).all()


def test_align_soft_starts_full_rank(world):
    tile, _ = tile_for(world, "--arm", "align_soft")
    for j, blk in enumerate(tile.blocks):
        idx = np.array(blk.mlp.soft_idx)
        assert all(len(set(r.tolist())) == idx.shape[1] for r in idx)       # distinct per neuron
        assert len({tuple(r.tolist()) for r in idx}) > 1
        D = ST._np(tile.dictionary(j))
        dict_part = np.einsum("wfk,fk->wf", D[:, idx], ST._np(blk.mlp.coef))
        free = ST._np(blk.mlp.free_b.weight) @ ST._np(blk.mlp.free_a.weight)
        assert np.linalg.matrix_rank(dict_part) > 1
        assert (dict_part ** 2).sum() > (free ** 2).sum()


def test_align_grow_frees_are_capped_skipped_in_anneal_and_start_at_the_donor(world):
    tile, a = tile_for(world, "--arm", "align_grow")
    n = sum(b.mlp.tie_mask.shape[0] for b in tile.blocks)
    before = [ST._np(tile.down_eff(j)) for j in range(len(tile.blocks))]
    tile.grow_state = mx.array([1.0, 1.0, 0.0])                               # stalled
    r = ST.align_step(tile, a, annealing=True)
    assert r["grow_freed"] == 0 and tile.tied_fraction() == 1.0
    tile.grow_state = mx.array([1.0, 1.0, 0.0])
    r = ST.align_step(tile, a)
    assert r["grow_freed"] > 0 and r["grow_freed_total"] == r["grow_freed"] and tile.tied_fraction() < 1.0
    for j in range(len(tile.blocks)):                                         # freed neurons write what they wrote
        assert np.allclose(ST._np(tile.down_eff(j)), before[j], atol=1e-6)
    cap = int(a.grow_max_frac * n)
    tile.grow_state = mx.array([1.0, 1.0, float(cap - 2)])
    r = ST.align_step(tile, a)
    assert r["grow_freed"] == 2 and r["grow_freed_total"] == cap
    tile.grow_state = mx.array([1.0, 1.0, float(cap)])
    assert ST.align_step(tile, a)["grow_freed"] == 0


def test_phase_b_batches_match_across_arms_and_a_ends_at_alpha_b(world, tmp_path):
    shas = {}
    for arm in ("correction", "imitate", "align_hard"):
        a = targs(tmp_path / arm, "--arm", arm, "--steps-a", "6", "--steps", "4", "--p-text", "0.34",
                  "--match-every", "3")
        assert ST.train(a, world=world) == 0
        log = json.loads((tmp_path / arm / "train_receipt.json").read_text())["log"]
        shas[arm] = [r["batch_sha"] for r in log if "batch_sha" in r]
        if arm != "correction":
            last_a = [r for r in log if r["phase"] == "A"][-1]
            assert last_a["alpha"] == 0.5 and last_a["step"] == 6
        if arm == "align_hard":
            assert all("tied_frac" in r and r["tied_frac"] is not None for r in log if r["step"] % 3 == 0)
    assert len(shas["correction"]) == 3 and shas["correction"] == shas["imitate"] == shas["align_hard"]


def test_compute_matched_correction_adds_b_steps(world, tmp_path):
    a = targs(tmp_path / "cm", "--arm", "correction", "--steps", "3", "--steps-a", "4", "--compute-matched")
    assert ST.train(a, world=world) == 0
    r = json.loads((tmp_path / "cm/train_receipt.json").read_text())
    assert r["steps_a"] == 0 and r["steps_b"] == 5 and r["compute_matched_extra_b"] == 2


def test_gate_per_layer_writes_each_block_once():
    L = 28
    lay = ST.aligned_layers(L, 1)
    sites = ST.write_sites(L, lay, "last", True)
    assert len({i for _, i in sites}) == len(sites) and sites[-1] == (28, 28)


def test_hungarian_checked_at_startup(world, tmp_path):
    try:
        import scipy  # noqa: F401
        pytest.skip("scipy present")
    except ImportError:
        pass
    with pytest.raises(SystemExit, match="scipy"):
        ST.train(targs(tmp_path / "h", "--arm", "align_hard", "--match", "hungarian"), world=world)
    assert not (tmp_path / "h").exists()


def test_align_soft_refresh_keeps_the_write_close(world):
    tile, a = tile_for(world, "--arm", "align_soft")
    before = ST._np(tile.down_eff(0))
    rep = ST.align_step(tile, a)
    after = ST._np(tile.down_eff(0))
    assert 0 <= rep["dict_share"] <= 1
    assert np.linalg.norm(after - before) <= np.linalg.norm(before) + 1e-6


# ------------------------------------------------------------------ parameters
def test_param_counts_reported_and_near_h51_at_06b_shape():
    cfg = Q.Qwen3Config(hidden_size=1024, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16,
                        num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=151936,
                        tie_word_embeddings=True)
    a = ST.parser().parse_args(["params"])
    rep = ST.count_params(cfg, a, arms=("correction", "imitate", "align_hard", "align_soft"))
    for arm in ("correction", "imitate", "align_hard"):
        assert abs(rep[arm]["stored"] - ST.H51_TILE_PARAMS) / ST.H51_TILE_PARAMS < 0.03, rep[arm]
        assert rep[arm]["trainable_A"] == rep[arm]["blocks"] - (28 * 2 * 768 if arm == "align_hard" else 0)
    assert rep["align_soft"]["stored"] < rep["imitate"]["stored"]
    for arm in ("correction", "imitate", "align_hard"):                        # default alpha_b 0.5: ladders active
        assert abs(rep[arm]["trainable_B"] - ST.H51_TILE_PARAMS) / ST.H51_TILE_PARAMS < 0.03, rep[arm]


# ------------------------------------------------------------------ corpus_tile path unchanged
def test_corpus_tile_eval_default_forward_unchanged(world):
    m, tok = world["model"], world["tok"]
    mx.random.seed(2)
    d = world["cfg"].hidden_size
    ct = CT.CorpusTile(m, 1, mx.random.normal((5, d)), mx.zeros((d,)), n_free=8, dk=16, k=4, rank=4)
    ct.slot_values = mx.random.normal(ct.slot_values.shape) * 0.5
    for blk in ct.lora.values():
        for lo in blk.values():
            lo.b.weight = mx.random.normal(lo.b.weight.shape) * 0.05
    a = targs(Path("/nonexistent"))
    r0 = CT.nll_eval(world, ct, a, n=4)
    r1 = CT.nll_eval(world, ct, a, n=4, fwd=CT.forward)
    assert r0 == r1 and r0["train_docs"]["tile"] != r0["train_docs"]["base"]
    p = [tok.encode(E.QA.chat("What is note 0 about? Answer briefly."))]
    assert E.generate(m, tok, ct, p, max_new=6) == E.generate(m, tok, ct, p, max_new=6, fwd=CT.forward)
    assert E.choice_pick(m, tok, ct, False, "Which?", ["a b", "c d"]) == \
        E.choice_pick(m, tok, ct, False, "Which?", ["a b", "c d"], CT.forward)


# ------------------------------------------------------------------ gradient accumulation
def _grads_rel_diff(g1, g2):
    from mlx.utils import tree_flatten
    a, b = dict(tree_flatten(g1)), dict(tree_flatten(g2))
    assert set(a) == set(b)
    return max(float(mx.abs(a[n] - b[n]).max().item()) / (float(mx.abs(a[n]).max().item()) + 1e-30) for n in a)


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_accum_gradients_equal_full_batch(world, k):
    """One step's loss and gradient at accum k equal accum 1 (float summation order only), for phase B (correction:
    4 corpus rows then 2 text rows, groups crossing the corpus/text boundary) and phase A (imitate, 6 rows)."""
    m, tok = world["model"], world["tok"]
    tile, _ = tile_for(world, "--arm", "correction")
    tile.set_phase("B")
    for u in tile.U:
        u.weight = mx.random.normal(u.weight.shape) * 0.05
    ch = CT.doc_chunks(tok, world["docs"], world["train_ids"])[:4]
    T = mx.array([w[:-1] for _, w, _ in ch])
    Y = mx.array([w[1:] for _, w, _ in ch])
    Wm = mx.array([[1.0] * (n - 1) + [0.0] * (CT.CHUNK - n + 1) for _, _, n in ch])
    ttok = mx.array(CT.text_chunks(tok, "train", n=2, seed=0))
    keep = tile.needed_layers() | {min(i for _, i in tile._sites)}
    l1, n1, k1, g1 = ST._accum_corpus(tile, m, T, Y, Wm, ttok, keep, 1.0, 1)
    lk, nk, kk, gk = ST._accum_corpus(tile, m, T, Y, Wm, ttok, keep, 1.0, k)
    assert abs(float(l1.item()) - float(lk.item())) < 1e-5 and _grads_rel_diff(g1, gk) < 1e-5
    # the accum-1 helper is the default step's loss
    import mlx.nn as nn
    st_c, _ = ST.donor_pass(m, T, keep)
    st_t, tref = ST.donor_pass(m, ttok, keep, logits=True)
    tref = tref.astype(mx.float32)

    def full(t):
        lg, _ = ST.forward(m, T, t, states=st_c)
        ce = nn.losses.cross_entropy(lg.astype(mx.float32), Y, reduction="none")
        tl = ST.forward(m, ttok, t, states=st_t)[0].astype(mx.float32)
        kl = mx.mean(mx.sum(mx.softmax(tref, -1) * (LB._log_softmax(tref, -1) - LB._log_softmax(tl, -1)), -1))
        return mx.sum(ce * Wm) / mx.sum(Wm) + kl
    lf, gf = nn.value_and_grad(tile, full)(tile)
    assert abs(float(lf.item()) - float(lk.item())) < 1e-5 and _grads_rel_diff(gf, gk) < 1e-5
    # phase A
    tile, _ = tile_for(world, "--arm", "imitate")
    tile.set_phase("A")
    TA = mx.array(CT.text_chunks(tok, "train", n=6, n_tok=128, seed=2))
    st, _ = ST.donor_pass(m, TA, set(tile._lay))
    (la, _), ga = nn.value_and_grad(tile, lambda t: ST.imitation_loss(t, st, 0.5))(tile)
    lb, gb = ST._accum_imitation(tile, m, TA, 0.5, k)
    assert abs(float(la.item()) - float(lb.item())) < 1e-5 and _grads_rel_diff(ga, gb) < 1e-5


@pytest.mark.parametrize("arm,extra,k2", [("correction", ("--steps", "3"), "2"),
                                          ("imitate", ("--steps-a", "3", "--steps", "0"), "2"),
                                          ("align_hard", ("--steps-a", "2", "--steps", "2", "--match-every", "2"), "2"),
                                          ("correction", ("--steps", "3", "--batch", "6"), "6"),
                                          ("imitate", ("--steps-a", "3", "--steps", "0", "--batch", "6"), "6")])
def test_accum_training_matches_full_batch(world, tmp_path, arm, extra, k2):
    """3 training steps. Losses match; weights match within 1e-5 except a small share of elements whose gradient is
    ~0, where Adam's first steps (update ~ lr x sign(g)) turn float-order noise into up to ~lr per step."""
    lr = 3e-3
    base = ("--arm", arm, "--batch", "4", "--p-text", "0.5", "--log-every", "1", "--lr", str(lr), "--lr-a", str(lr),
            *extra)
    runs = {}
    for k in ("1", k2):
        assert ST.train(targs(tmp_path / f"k{k}", *base, "--accum", k), world=world) == 0
        runs[k] = json.loads((tmp_path / f"k{k}/train_receipt.json").read_text())["log"]
    l1, l2 = [r["loss"] for r in runs["1"]], [r["loss"] for r in runs[k2]]
    # 5e-5: logged losses are rounded to 5 decimals, and the gradient-norm clip (default 1.0) rescales by a norm whose
    # float order differs between accum paths
    assert len(l1) >= 3 and np.allclose(l1, l2, rtol=0, atol=5e-5), (l1, l2)
    if arm == "correction":
        assert [r["batch_sha"] for r in runs["1"]] == [r["batch_sha"] for r in runs[k2]]
    a = mx.load(str(tmp_path / "k1/tile.safetensors"))
    b = mx.load(str(tmp_path / f"k{k2}/tile.safetensors"))
    tot = bad = 0
    for key in a:
        x = np.abs(np.array(a[key].astype(mx.float32)) - np.array(b[key].astype(mx.float32)))
        tot, bad = tot + x.size, bad + int((x > 1e-5).sum())
        assert x.max() <= 2 * lr * len(l1) + 1e-6, key
    assert bad / tot < 0.01, (bad, tot)


def test_accum_resume_and_validation(world, tmp_path):
    common = ("--arm", "imitate", "--batch", "4", "--p-text", "0.5", "--steps-a", "3", "--steps", "3",
              "--accum", "2")
    assert ST.train(targs(tmp_path / "full", *common), world=world) == 0
    cut = tmp_path / "cut"
    assert ST.train(targs(cut, *common, "--max-steps-this-launch", "4"), world=world) == 3
    assert ST.train(targs(cut, *common), world=world) == 0
    a, b = mx.load(str(tmp_path / "full/tile.safetensors")), mx.load(str(cut / "tile.safetensors"))
    assert all(np.array_equal(np.array(a[k]), np.array(b[k])) for k in a)
    with pytest.raises(SystemExit, match="accum"):
        ST.train(targs(tmp_path / "bad", "--batch", "4", "--p-text", "0.5", "--accum", "5"), world=world)
    with pytest.raises(SystemExit, match="accum"):
        ST.train(targs(tmp_path / "bad0", "--accum", "0"), world=world)
    # a run written before --accum existed resumes as accum 1
    old = tmp_path / "old"
    oc = ("--arm", "correction", "--steps", "3")
    assert ST.train(targs(old, *oc, "--max-steps-this-launch", "1"), world=world) == 3
    saved = json.loads((old / "args.json").read_text())
    del saved["accum"]
    (old / "args.json").write_text(json.dumps(saved))
    assert ST.train(targs(old, *oc), world=world) == 0
    assert ST.train(targs(tmp_path / "old2", *oc, "--max-steps-this-launch", "1"), world=world) == 3
    with pytest.raises(SystemExit, match="differ"):                          # accum is not a per-launch argument
        ST.train(targs(tmp_path / "old2", *oc, "--accum", "2"), world=world)


# ------------------------------------------------------------------ non-finite regression (2026-11-25, Qwen3-0.6B NaN)
def test_deep_tile_backward_is_finite_at_the_sink(world):
    """The tile's position 0 must never be an exactly-zero state: with zeros there, every tile RMSNorm's gradient at
    that position is 1/sqrt(eps) = 1000x and the product over a 28-block tile overflowed fp32 (NaN in blocks 0-7 at
    step 2 of the first Qwen3-0.6B run). Tiny donor at full depth, trained-like U."""
    import mlx.nn as nn
    m28 = tiny(layers=28)
    W = dict(world, model=m28, cfg=m28.config)
    tile, _ = tile_for(W)
    tile.set_phase("B")
    for u in tile.U:
        u.weight = mx.random.normal(u.weight.shape) * 0.05
    T = toks(W, n=2, L=64)
    st, _ = ST.donor_pass(m28, T, tile.needed_layers())
    ins = []
    tile.run(st, on_input=lambda j, x: ins.append(float(mx.abs(x[:, 0]).max().item())))
    assert min(ins) > 0.1                                                    # no zero state at the sink, any block
    loss, g = nn.value_and_grad(tile, lambda t: ST.forward(m28, T, t, states=st)[0].astype(mx.float32).std())(tile)
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(g))
    assert all(bool(mx.isfinite(v).all().item()) for v in flat.values())
    assert float(mx.abs(flat["blocks.0.self_attn.k_proj.weight"]).max().item()) < 1e3


def test_nonfinite_step_stops_with_diagnostics(world, tmp_path, monkeypatch):
    out = tmp_path / "nan"
    orig = ST._accum_corpus

    def poisoned(*a, **k):
        loss, nll, kl, g = orig(*a, **k)
        return loss * mx.array(float("nan")), nll, kl, g
    monkeypatch.setattr(ST, "_accum_corpus", poisoned)
    rc = ST.train(targs(out, "--arm", "correction", "--steps", "3", "--accum", "2", "--batch", "4", "--p-text", "0.5"),
                  world=world)
    assert rc == 4
    d = json.loads((out / "nonfinite.json").read_text())
    assert d["step"] == 1 and d["loss_finite"] is False and "bad_params" in d
    assert not (out / "train_receipt.json").exists()


# ------------------------------------------------------------------ held-out NLL tracking during training
def test_track_every_off_by_default_and_on(world, tmp_path):
    off = tmp_path / "off"
    assert ST.train(targs(off, "--arm", "imitate", "--steps-a", "2", "--steps", "4"), world=world) == 0
    r = json.loads((off / "train_receipt.json").read_text())
    assert r["args"]["track_every"] == 0 and "track" not in r
    assert not any(k in row for row in r["log"] for k in ST.TRACK_KEYS)
    on = tmp_path / "on"
    assert ST.train(targs(on, "--arm", "imitate", "--steps-a", "2", "--steps", "6", "--track-every", "2"),
                    world=world) == 0
    r = json.loads((on / "train_receipt.json").read_text())
    tr = r["track"]
    assert [x["step"] for x in tr] == [4, 6, 8]                              # phase-B steps 2, 4, 6
    for x in tr:
        assert set(ST.TRACK_KEYS) <= set(x)
        assert x["gain_train"] == pytest.approx(x["base_train"] - x["track_train"], abs=2e-4)
        assert x["gain_train_minus_held"] == pytest.approx(x["gain_train"] - x["gain_held"], abs=2e-4)
    assert len({x["base_train"] for x in tr}) == 1                           # the base is computed once
    # the fixed sample and the tracked numbers agree with a direct computation on the final tile
    tile = ST.load_run(on, world, argparse_ns(r["args"]))
    trs, hes = ST.track_sample(world["tok"], world, r["args"]["seed"])
    assert 0 < len(trs) <= ST.TRACK_N and 0 < len(hes) <= ST.TRACK_N        # tiny corpus: fewer chunks than 8
    assert ST.track_nll(world["model"], tile, trs) == pytest.approx(tr[-1]["track_train"], abs=1e-3)
    assert ST.track_nll(world["model"], None, trs) == pytest.approx(tr[-1]["base_train"], abs=1e-3)


def argparse_ns(d):
    import argparse
    return argparse.Namespace(**d)


def test_track_history_survives_resume(world, tmp_path):
    common = ("--arm", "correction", "--steps", "6", "--track-every", "2")
    assert ST.train(targs(tmp_path / "full", *common), world=world) == 0
    cut = tmp_path / "cut"
    assert ST.train(targs(cut, *common, "--max-steps-this-launch", "3"), world=world) == 3
    assert ST.train(targs(cut, *common), world=world) == 0
    strip = lambda rows: [{k: v for k, v in x.items() if k != "track_s"} for x in rows]   # noqa: E731
    a = json.loads((tmp_path / "full/train_receipt.json").read_text())["track"]
    b = json.loads((cut / "train_receipt.json").read_text())["track"]
    assert [x["step"] for x in b] == [2, 4, 6] and strip(a) == strip(b)
