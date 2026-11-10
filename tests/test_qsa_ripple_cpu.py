"""QSA on Ripple: the MLX indexer trainer against the inference indexer and
numpy oracles, and the sidecar installed on a tiny synthetic Ripple model.

Synthetic CPU oracles; no checkpoint.  The GPU probe records checkpoint and
commit for any model-path result.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim  # noqa: E402
from mlx import nn  # noqa: E402

from mlx2.runtime.models import qsa_indexer_train as qt  # noqa: E402
from mlx2.runtime.models import qsa_sidecar as qs  # noqa: E402
from mlx2.runtime.models.ricursive import Model, ModelArgs  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _indexer(hidden, *, heads=2, head_dim=16, rope_dim=0, seed=0):
    mx.random.seed(seed)
    idx = qt.TrainableIndexer(hidden, heads=heads, head_dim=head_dim, rope_dim=rope_dim,
                              rope_base=1e4)
    idx.q_norm = mx.random.normal((head_dim,)) * 0.1
    idx.k_norm = mx.random.normal((head_dim,)) * 0.1
    idx.k_block_norm = mx.random.normal((head_dim,)) * 0.1
    return idx


def _save(tmp_path, layers: dict, *, heads, head_dim, rope_dim):
    tensors = {}
    for index, idx in layers.items():
        tensors.update(idx.export(f"model.layers.{index}.self_attn"))
    mx.save_safetensors(str(tmp_path / "qsa-indexers.safetensors"), tensors)
    (tmp_path / "config.json").write_text(json.dumps({"qsa": {
        "indexer_heads": heads, "indexer_head_dim": head_dim, "rope_dim": rope_dim,
        "rope_base": 1e4, "block_size": 4}}))
    return tmp_path


@pytest.mark.parametrize("rope_dim", [0, 8])
def test_trainable_scores_match_inference_indexer(tmp_path, rope_dim):
    hidden, length = 24, 45
    idx = _indexer(hidden, rope_dim=rope_dim)
    _save(tmp_path, {3: idx}, heads=2, head_dim=16, rope_dim=rope_dim)
    sidecar = qs.QSASidecar(tmp_path, budget=8, mode="mask")
    assert sidecar.geometry["rope_dim"] == rope_dim
    x = mx.random.normal((length, hidden))
    want = np.array(idx.scores(x))
    live = sidecar.indexers[3]
    got = []
    offset = 0
    for size in (7, 1, 20, 17):  # uneven chunks exercise the pooled-key tail
        part = x[offset:offset + size]
        live.append_keys(part, offset)
        s = np.array(live.scores(live.queries(part, offset)))
        row = np.full((size, want.shape[1]), -np.inf, np.float32)
        row[:, : s.shape[1]] = s
        got.append(row)
        offset += size
    got = np.concatenate(got)
    valid = np.isfinite(want)
    np.testing.assert_allclose(got[valid], want[valid], rtol=1e-4, atol=1e-4)


def test_block_teacher_matches_numpy():
    rng = np.random.default_rng(1)
    (hq, hkv, length, dim) = (6, 2, 37, 8)
    q = rng.standard_normal((hq, length, dim)).astype(np.float32)
    k = rng.standard_normal((hkv, length, dim)).astype(np.float32)
    got = np.array(qt.block_teacher(mx.array(q), mx.array(k), 0.35, query_chunk=8))
    group = hq // hkv
    nb = length // 4
    want = np.zeros((length, nb), np.float32)
    for t in range(length):
        probs = np.zeros(t + 1)
        for h in range(hq):
            logit = q[h, t] @ k[h // group, : t + 1].T * 0.35
            e = np.exp(logit - logit.max())
            probs += e / e.sum()
        for j in range(nb):
            if 4 * j + 3 <= t:
                want[t, j] = probs[4 * j: 4 * j + 4].max()
        if want[t].sum() > 0:
            want[t] /= want[t].sum()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_kl_loss_matches_numpy_and_grads_are_finite():
    hidden, length = 16, 33
    idx = _indexer(hidden)
    x = mx.random.normal((length, hidden))
    target = qt.block_teacher(mx.random.normal((4, length, 8)), mx.random.normal((2, length, 8)),
                              0.5)
    scores = idx.scores(x)
    got = float(qt.kl_loss(scores, target).item())
    s, t = np.array(scores), np.array(target)
    rows = []
    for r in range(length):
        if t[r].sum() <= 0:
            continue
        v = np.isfinite(s[r])
        logp = s[r][v] - np.log(np.exp(s[r][v] - s[r][v].max()).sum()) - s[r][v].max()
        tr = t[r][v]
        rows.append(np.sum(np.where(tr > 0, tr * (np.log(np.where(tr > 0, tr, 1)) - logp), 0)))
    assert got == pytest.approx(float(np.mean(rows)), rel=1e-4)
    (_, grads) = nn.value_and_grad(idx, lambda m: qt.kl_loss(m.scores(x), target))(idx)
    from mlx.utils import tree_flatten

    assert all(np.isfinite(np.array(g)).all() for (_, g) in tree_flatten(grads))


def test_distillation_fits_a_learnable_teacher():
    """The objective can fit at all: a student learns another indexer's blocks."""
    hidden, length = 32, 96
    x = mx.random.normal((length, hidden))
    teacher = _indexer(hidden, seed=7)
    target = mx.softmax(teacher.scores(x) * 4.0, axis=-1)
    target = mx.where(qt.block_valid(length), target, 0.0)
    student = _indexer(hidden, seed=3)
    opt = optim.AdamW(learning_rate=3e-3, betas=[0.9, 0.95], weight_decay=0.0)
    step = nn.value_and_grad(student, lambda m: qt.kl_loss(m.scores(x), target))
    first = None
    for _ in range(150):
        (loss, grads) = step(student)
        opt.update(student, grads)
        mx.eval(student.parameters(), opt.state)
        first = float(loss.item()) if first is None else first
    final = float(qt.kl_loss(student.scores(x), target).item())
    assert final < 0.5 * first
    (mass, rows) = qt.capture(student.scores(x), target, topk=8)
    assert rows > 0 and 0.0 < mass <= 1.0 + 1e-5


def test_global_clip_scales_every_layer_together():
    grads = [{"a": mx.array([3.0, 0.0])}, {"b": mx.array([0.0, 4.0])}]
    (clipped, norm) = qt.global_clip(grads, 1.0)
    assert norm == pytest.approx(5.0)
    assert np.array(clipped[0]["a"]).tolist() == pytest.approx([0.6, 0.0], abs=1e-6)
    assert np.array(clipped[1]["b"]).tolist() == pytest.approx([0.0, 0.8], abs=1e-6)


# ------------------------------------------------------------ Ripple install


def _ripple():
    args = ModelArgs(hidden_size=16, intermediate_size=32, num_hidden_layers=5,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                     vocab_size=32, sliding_window=4, embed_scale=2.0,
                     ffn_clamp_threshold=0.2)
    mx.random.seed(11)
    model = Model(args, nvfp4=False)
    # Random gates and value blend so every term of the layer is live.
    for layer in model.layers:
        layer.self_attn.value_res_lambda = mx.array([0.3])
    mx.eval(model.parameters())
    return model


def _run(model, tokens, steps):
    cache = model.make_cache()
    rows = []
    start = 0
    for size in steps:
        h = model.model(mx.array([tokens[start:start + size]], dtype=mx.uint32), cache)
        mx.eval(h)
        rows.append(np.array(h[0].astype(mx.float32)))
        start += size
    return np.concatenate(rows)


def _sidecar(tmp_path, *, budget, mode):
    _save(tmp_path, {4: _indexer(16, heads=2, head_dim=16)}, heads=2, head_dim=16,
          rope_dim=0)
    return qs.QSASidecar(tmp_path, budget=budget, mode=mode)


TOKENS = [int(t) for t in np.random.default_rng(5).integers(0, 32, 41)]


def test_ripple_install_targets_only_global_layers(tmp_path):
    model = _ripple()
    sidecar = _sidecar(tmp_path, budget=8, mode="dense")
    assert qs.install(model, sidecar) == [4]
    assert type(model.layers[4].self_attn).__name__ == "QSASidecarAttention"
    assert type(model.layers[3].self_attn).__name__ == "Attention"
    bad = qs.QSASidecar(tmp_path, budget=8, mode="dense")
    bad.indexers = {3: bad.indexers[4]}  # a sliding layer has no indexer seam
    with pytest.raises(ValueError, match="no full attention"):
        qs.install(_ripple(), bad)


def test_ripple_dense_and_selfdense_equal_base(tmp_path):
    steps = (17, 1, 9, 1, 13)
    base = _run(_ripple(), TOKENS, steps)
    for mode in ("dense", "selfdense"):
        qs.STATUS.clear()
        model = _ripple()
        qs.install(model, _sidecar(tmp_path, budget=8, mode=mode))
        got = _run(model, TOKENS, steps)
        np.testing.assert_array_equal(got, base)
    assert qs.status()["dense_by_construction"] == len(steps)


def test_ripple_sparse_mask_engages_and_gather_matches_mask(tmp_path):
    steps = (24, 1, 1, 1)
    base = _run(_ripple(), TOKENS, steps)
    qs.STATUS.clear()
    masked = _ripple()
    qs.install(masked, _sidecar(tmp_path, budget=8, mode="mask"))
    got_mask = _run(masked, TOKENS, steps)
    status = qs.status(reset=True)
    assert status["sparse_mask_rows"] == sum(steps)
    assert not np.array_equal(got_mask, base)  # 2 of 6 blocks: selection is live
    # The first rows see at most two complete blocks: dense by construction.
    np.testing.assert_allclose(got_mask[:11], base[:11], rtol=1e-5, atol=1e-5)
    # Gather is the decode law (prefill rows go to the Metal NAX kernel): prefill
    # under mask, then decode under gather.
    gathered = _ripple()
    sidecar = _sidecar(tmp_path, budget=8, mode="mask")
    qs.install(gathered, sidecar)
    cache = gathered.make_cache()
    gathered.model(mx.array([TOKENS[:24]], dtype=mx.uint32), cache)
    sidecar.mode = "gather"
    rows = []
    for i in range(24, 27):
        h = gathered.model(mx.array([[TOKENS[i]]], dtype=mx.uint32), cache)
        rows.append(np.array(h[0].astype(mx.float32)))
    np.testing.assert_allclose(np.concatenate(rows), got_mask[24:27], rtol=1e-5, atol=1e-5)


def test_ripple_refuses_batches(tmp_path):
    model = _ripple()
    qs.install(model, _sidecar(tmp_path, budget=8, mode="mask"))
    cache = model.make_cache()
    with pytest.raises(qs.QSASidecarRefused, match="one plain cached sequence"):
        model.model(mx.array([TOKENS[:8], TOKENS[8:16]], dtype=mx.uint32), cache)


def test_fixed_blocks_and_hybrid_kl():
    length = 61
    fixed = np.array(qt.fixed_blocks(length, sink_blocks=2, recent_blocks=3))
    for t in (0, 3, 20, 60):
        newest = (t + 1) // 4
        want = {j for j in range(15) if j < 2 or newest - 3 <= j < newest}
        assert {j for j in range(15) if fixed[t, j]} == want
    idx = _indexer(16)
    x = mx.random.normal((length, 16))
    target = qt.block_teacher(mx.random.normal((4, length, 8)), mx.random.normal((2, length, 8)),
                              0.5)
    mfixed = qt.fixed_blocks(length, 2, 3)
    got = float(qt.kl_loss(idx.scores(x), target, mfixed).item())
    s, t = np.array(idx.scores(x)), np.array(target)
    rows = []
    for r in range(length):
        allowed = np.isfinite(s[r]) & ~fixed[r]
        if not allowed.any():
            continue
        tr = np.where(allowed, t[r], 0.0)
        if tr.sum() <= 0:
            continue
        tr = tr[allowed] / tr.sum()
        sr = s[r][allowed]
        logp = sr - sr.max() - np.log(np.exp(sr - sr.max()).sum())
        rows.append(np.sum(np.where(tr > 0, tr * (np.log(np.where(tr > 0, tr, 1)) - logp), 0)))
    assert got == pytest.approx(float(np.mean(rows)), rel=1e-4)
    (_, grads) = nn.value_and_grad(idx, lambda m: qt.kl_loss(m.scores(x), target, mfixed))(idx)
    from mlx.utils import tree_flatten

    assert all(np.isfinite(np.array(g)).all() for (_, g) in tree_flatten(grads))


def test_sidecar_hybrid_selection_takes_fixed_blocks_then_indexer(tmp_path):
    hidden, length = 24, 90
    idx = _indexer(hidden, rope_dim=8)
    _save(tmp_path, {3: idx}, heads=2, head_dim=16, rope_dim=8)
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["qsa"]["selection"] = {"sink_blocks": 2, "recent_blocks": 3}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    sidecar = qs.QSASidecar(tmp_path, budget=32, mode="mask")  # 8 blocks: 5 fixed + 3 learned
    assert sidecar.selection == {"sink_blocks": 2, "recent_blocks": 3}
    x = mx.random.normal((length, hidden))
    (ids, ok, q_pos, n_blocks, _) = qs._selection(sidecar.indexers[3], x, 0, 8,
                                                  selection=sidecar.selection)
    scores = np.array(idx.scores(x))
    ids, ok = np.array(ids), np.array(ok)
    for t in range(40, length):
        newest = (t + 1) // 4
        fixed = {0, 1, newest - 3, newest - 2, newest - 1}
        got = {int(j) for j, v in zip(ids[t], ok[t]) if v}
        assert fixed <= got
        rest = [j for j in range(newest) if j not in fixed]
        want = set(sorted(rest, key=lambda j: -scores[t, j])[:3])
        assert got - fixed == want
    with pytest.raises(qs.QSASidecarRefused, match="hybrid"):
        qs._selection(sidecar.indexers[3], x[:4], length, 8, fused=True,
                      selection=sidecar.selection)


def test_sidecar_dense_layers_run_dense(tmp_path):
    steps = (24, 1, 1)
    base = _run(_ripple(), TOKENS, steps)
    model = _ripple()
    sidecar = _sidecar(tmp_path, budget=8, mode="mask")
    sidecar.dense_layers = {4}
    qs.install(model, sidecar)
    qs.STATUS.clear()
    np.testing.assert_allclose(_run(model, TOKENS, steps), base, rtol=1e-5, atol=1e-5)
    assert "sparse_mask_rows" not in qs.status(reset=True)


def test_training_temperature_scales_loss_only_and_moves_at_its_own_pace():
    hidden, length = 32, 96
    x = mx.random.normal((length, hidden))
    teacher = _indexer(hidden, seed=7)
    # A diffuse teacher: much flatter than raw RMS-normed scores.
    target = mx.softmax(teacher.scores(x) * 0.05, axis=-1)
    target = mx.where(qt.block_valid(length), target, 0.0)
    student = _indexer(hidden, seed=3)
    student_t = qt.TrainableIndexer(hidden, heads=2, head_dim=16, rope_base=1e4,
                                    temperature={"init": 0.1, "lr_mult": 30.0})
    student_t.update({k: v for k, v in student.parameters().items()})
    assert student_t.temperature() == pytest.approx(0.1, rel=1e-5)
    np.testing.assert_allclose(np.array(student_t.loss_scores(x)),
                               np.array(student.scores(x)) * 0.1, rtol=1e-5)
    assert not any("temp" in k for k in student_t.export("p"))
    finals = []
    for model in (student, student_t):
        opt = optim.AdamW(learning_rate=3e-4, betas=[0.9, 0.95], weight_decay=0.0,
                          bias_correction=True)
        step = nn.value_and_grad(model, lambda m: qt.kl_loss(m.loss_scores(x), target))
        for _ in range(60):
            (_, grads) = step(model)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
        finals.append(float(qt.kl_loss(model.loss_scores(x), target).item()))
    assert student_t.temperature() != pytest.approx(0.1, rel=1e-3)
    assert finals[1] < 0.5 * finals[0]


def _write_ckpt(directory, indexers, opts, step):
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for (i, idx) in indexers.items():
        tensors.update(idx.export(f"model.layers.{i}.self_attn"))
    mx.save_safetensors(str(directory / "qsa-indexers.safetensors"), tensors)
    mx.save_safetensors(str(directory / "trainer-state.safetensors"),
                        qt.trainer_state(indexers, opts))
    (directory / "config.json").write_text(json.dumps({"qsa": {
        "trained_steps": step,
        "training_temperature": {str(i): idx.temperature() for (i, idx) in indexers.items()}}}))


def test_resume_from_checkpoint_is_exact(tmp_path):
    from mlx.utils import tree_flatten

    hidden, length = 32, 64
    x = mx.random.normal((length, hidden))
    target = mx.where(qt.block_valid(length),
                      mx.softmax(_indexer(hidden, seed=7).scores(x) * 0.05, axis=-1), 0.0)
    temp = {"init": 0.1, "lr_mult": 30.0}

    def fresh():
        mx.random.seed(1)
        idx = {5: qt.TrainableIndexer(hidden, heads=2, head_dim=16, rope_dim=8,
                                      rope_base=1e4, temperature=temp)}
        opt = {5: optim.AdamW(learning_rate=3e-3, betas=[0.9, 0.95], weight_decay=0.0,
                              bias_correction=True)}
        return idx, opt

    def train(idx, opt, n):
        for _ in range(n):
            (_, g) = nn.value_and_grad(idx[5], lambda m: qt.kl_loss(m.loss_scores(x), target))(idx[5])
            opt[5].update(idx[5], g)
            mx.eval(idx[5].parameters(), opt[5].state)

    (straight, so) = fresh()
    train(straight, so, 5)
    (first, fo) = fresh()
    train(first, fo, 3)
    _write_ckpt(tmp_path / "step3", first, fo, 3)
    (resumed, ro) = fresh()
    receipt = qt.load_trainer_checkpoint(tmp_path / "step3", resumed, ro, step=3)
    assert receipt["optimizer_state"] == "restored"
    train(resumed, ro, 2)
    want = dict(tree_flatten(straight[5].parameters()))
    got = dict(tree_flatten(resumed[5].parameters()))
    for key in want:
        np.testing.assert_array_equal(np.array(got[key]), np.array(want[key]))
    with pytest.raises(ValueError, match="holds step 3"):
        qt.load_trainer_checkpoint(tmp_path / "step3", *fresh(), step=4)
    # A pre-trainer-state checkpoint: weights + temperature, AdamW fresh.
    (tmp_path / "step3" / "trainer-state.safetensors").unlink()
    (old, oo) = fresh()
    receipt = qt.load_trainer_checkpoint(tmp_path / "step3", old, oo, step=3)
    assert receipt["optimizer_state"].startswith("fresh")
    assert old[5].temperature() == pytest.approx(first[5].temperature(), rel=1e-5)
    np.testing.assert_array_equal(np.array(old[5].q_proj.weight), np.array(first[5].q_proj.weight))
