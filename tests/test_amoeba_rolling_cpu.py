"""CPU tests of the rolling layer-window capture (mlx2.experimental.amoeba.child_rolling): the layer-major schedule
equals the ordinary text-major capture on toy dense, MoE-expert and MoE-neuron donors; chunking, prefetch, the
per-layer byte map, and the plan's automatic choice."""
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

import test_amoeba_carve_cpu as TC  # noqa: E402
import test_amoeba_carve_neuron_units_cpu as TN  # noqa: E402
from research.amoeba.carve.host import Host  # noqa: E402

from mlx2.experimental.amoeba import child_rolling as R  # noqa: E402
from mlx2.experimental.amoeba import config as K  # noqa: E402
from mlx2.experimental.amoeba import plan as P  # noqa: E402
from mlx2.experimental.amoeba.child_capture import peaks  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _seqs(n, T, vocab, seed=3):
    rng = np.random.default_rng(seed)
    lens = [T, T, T - 3, T - 3, T][:n]
    return [rng.integers(0, vocab, size=t).tolist() for t in lens]


def _resident(h, seqs, fit):
    """The ordinary capture's arithmetic: mu on fit texts (positions >= 1), then per-text peaks."""
    L, W = h.L, max(h.widths)
    mu = np.zeros((L, W), np.float32)
    acc, cnt = {}, {}

    def st(l, kind, p):
        if kind == "dense":
            acc[l] = acc.get(l, 0) + np.array(p["act"][:, 1:].astype(mx.float32).sum((0, 1)))
            cnt[l] = cnt.get(l, 0) + p["act"].shape[0] * (p["act"].shape[1] - 1)
    for s, f in zip(seqs, fit):
        if f:
            h.ctl.reset()
            h.ctl.stats = st
            mx.eval(h.hidden(mx.array([s])))
    h.ctl.reset()
    for l in acc:
        mu[l, :h.widths[l]] = acc[l] / cnt[l]
    return mu, [peaks(h, [s], mu) for s in seqs]


@pytest.mark.parametrize("make,units,vocab", [(lambda: TC.toy(moe=True), "expert", TC.V),
                                              (lambda: TC.toy(moe=False), "expert", TC.V),
                                              (lambda: TN.qn_toy(), "neuron", TN.V)])
@pytest.mark.parametrize("chunks,carry", [(1, "resident"), (2, "resident"), (1, "memmap")])
def test_rolling_equals_the_ordinary_capture(tmp_path, make, units, vocab, chunks, carry):
    from research.amoeba.sweep.host_family import HostFamily
    h = Host(model=make(), tokenizer=TC.Tok(), fingerprint="toy:abc", log=lambda *_: None, moe_units=units)
    fam = HostFamily(h)
    seqs = _seqs(5, 12, vocab)
    fit = [True, False, True, True, False]
    rec, mu, nll = R.run(fam, seqs, fit, tmp_path, window=2, chunks=chunks, carry_mode=carry, max_texts=2,
                         log=lambda *_: None)
    mu0, ref = _resident(h, seqs, fit)
    assert rec["chunks"] == chunks and len(rec["layers"]) == h.L * chunks and rec["carry"] == carry
    assert np.allclose(mu, mu0, atol=1e-5)
    for l in range(h.L):
        got = np.load(tmp_path / f"resp_L{l:02d}.npy").astype(np.float32)
        want = np.stack([(r[l][0] if r[l] is not None else np.zeros(h.widths[l])) for r in ref]).astype(np.float16)
        assert np.allclose(got, want.astype(np.float32), atol=2e-3, rtol=2e-3), (l, h.sites[l].kind)
    # the full-model NLL from the schedule's final state equals the ordinary forward's (TopicsBackend.nll targets)
    from research.amoeba.carve.host import TopicsBackend
    be = TopicsBackend(None, host=h)
    want_nll = np.array([be.nll([s])[0] for s in seqs])
    assert np.allclose(nll, want_nll, atol=1e-5)
    par = R.parity(fam, seqs[:3], None, log=lambda *_: None)     # the fresh parity gate, every run
    assert par["ok"] and all(t["bitwise_equal"] for t in par["texts"])
    par2 = R.parity(fam, seqs[:2], 1, log=lambda *_: None)       # first block only (big models)
    assert par2["ok"] and par2["layers"] == 1 and fam.L == h.L


def test_parity_gate_refuses_a_family_that_does_not_reproduce_the_forward(tmp_path):
    from research.amoeba.sweep.host_family import HostFamily
    h = Host(model=TC.toy(moe=False), tokenizer=TC.Tok(), fingerprint="toy:abc", log=lambda *_: None)
    fam = HostFamily(h)
    orig = fam.run_layer

    def broken(l, carry):                      # e.g. a layer passing state to the next beyond the residual stream
        c, p = orig(l, carry)
        return ({"h": c["h"] * 1.01, "ids": c["ids"]} if l == 1 else c), p
    fam.run_layer = broken
    par = R.parity(fam, _seqs(2, 12, TC.V), None, log=lambda *_: None)
    assert not par["ok"]


def test_neuron_units_light_only_routed_experts(tmp_path):
    h = Host(model=TN.qn_toy(), tokenizer=TC.Tok(), fingerprint="toy:abc", log=lambda *_: None, moe_units="neuron")
    s = _seqs(1, 12, TN.V)
    r = peaks(h, s, None)
    for l, site in enumerate(h.sites):
        if site.units != "neuron":
            continue
        got = r[l][0].reshape(site.E, site.F)
        routed = set()

        def st(l_, kind, p, _l=l):
            if l_ == _l:
                routed.update(np.array(p["inds"][:, 1:]).ravel().tolist())
        h.ctl.reset()
        h.ctl.stats = st
        mx.eval(h.hidden(mx.array(s)))
        h.ctl.reset()
        lit = set(np.nonzero(got.max(1) > 0)[0].tolist())
        assert lit and lit <= routed                  # only experts that some position routed to carry signal


def test_chunks_balance_tokens_and_cover_every_text():
    seqs = [[0] * n for n in (10, 9, 8, 3, 2, 2, 1)]
    g = R.chunk_texts(seqs, 3)
    assert sorted(j for x in g for j in x) == list(range(7)) and len(g) == 3
    tok = [sum(len(seqs[j]) for j in x) for x in g]
    assert max(tok) - min(tok) <= 10


def test_prefetcher_reads_layer_ranges_from_safetensors(tmp_path):
    from mlx2.experimental.amoeba.inspection import layer_bytes
    w = {f"model.layers.{i}.mlp.down_proj.weight": mx.ones((64, 32)) * i for i in range(3)}
    w["model.embed_tokens.weight"] = mx.ones((10, 32))
    w["model.layers.1.mlp.experts.0.up_proj.weight"] = mx.ones((8, 32))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), w)
    lb = layer_bytes(tmp_path, 3)
    assert lb["layered"] and lb["per_layer"][0] == 64 * 32 * 4 and lb["rest"] == 10 * 32 * 4
    assert lb["experts_per_layer"] == [0, 8 * 32 * 4, 0]
    rg = R.layer_ranges(tmp_path, 3, experts_only=False)
    assert sorted(rg) == [0, 1, 2] and sum(e - b for _, b, e in rg[1]) == 64 * 32 * 4 + 8 * 32 * 4
    assert sorted(R.layer_ranges(tmp_path, 3, experts_only=True)) == [1]
    pf = R.Prefetcher(rg, chunk=1000)
    try:
        assert pf.wait(0) == 0.0                      # never requested: no wait
        pf.request(1)
        pf.request(2)
        assert pf.wait(2) >= 0.0 and pf.done.get(1) and pf.done.get(2)
        assert pf.bytes == sum(e - b for l in (1, 2) for _, b, e in rg[l])
    finally:
        pf.close()


def _info(weights_gb, L=40, d=5120, F=17408, moe=False):
    per = int(weights_gb * 2 ** 30 * 0.97 / L)
    return {"path": "/m", "checkpoint": "big@0", "fingerprint": "0" * 64, "adapter": "A", "kind": "moe" if moe else "dense",
            "layers": L, "hidden": d, "intermediate": F, "vocab": 151936, "moe": None, "file_backed_ple": False,
            "weight_bytes": int(weights_gb * 2 ** 30),
            "units": {"dense_neurons": L * F, "moe_expert_units": 0, "moe_neuron_units": 0, "estimate": True},
            "layer_bytes": {"per_layer": [per] * L, "max_layer": per, "rest": int(weights_gb * 2 ** 30 * 0.03),
                            "experts_per_layer": [int(per * 0.9) if moe else 0] * L,
                            "max_layer_experts": int(per * 0.9) if moe else 0, "layered": True}}


def test_plan_picks_rolling_automatically_and_never_falls_back(monkeypatch):
    monkeypatch.setattr(P, "physical_gb", lambda: 128.0)
    cfg = K.resolve("full", {}, {}, "")
    p = P.make_plan(_info(300), cfg, texts=46000)
    st = p["stream"]
    assert st["mode"] == "rolling" and st["chosen"].startswith("automatic") and p["fits"]
    assert st["window"] >= 1 and st["chunks"] >= 1 and "refused" in st["carve"]
    assert any("rolling" in n for n in p["notes"])
    small = P.make_plan(_info(10), cfg, texts=1000)
    assert small["stream"]["mode"] == "resident"
    forced = P.make_plan(_info(10), cfg, texts=1000, stream="rolling", window=2)
    assert forced["stream"]["mode"] == "rolling" and forced["stream"]["window"] == 2
    huge = P.make_plan(_info(300), cfg, texts=46000, window=500)
    assert not huge["fits"] and "window" in huge["refusals"][0]
    moe = P.make_plan(_info(300, moe=True), cfg, texts=46000)    # no whole-layer MoE streaming: refuse, never re-read
    assert not moe["fits"] and "whole-layer" in moe["refusals"][0]
    assert st["bs"] >= 1 and p["batches"]["capture"]["bs"] == 0        # the rolling child never gets bs 0


def test_checkpoint_without_an_adapter_maps_through_its_sweep_family(tmp_path, monkeypatch):
    import types
    import research.amoeba.sweep.family as FAM
    from mlx2.experimental.amoeba.inspection import inspect_checkpoint
    mod = types.ModuleType("toyfam_mod")
    mod.checkpoint_fingerprint = lambda path: "f" * 64
    monkeypatch.setitem(sys.modules, "toyfam_mod", mod)
    monkeypatch.setitem(FAM.FAMILIES, "toyfam_next", "toyfam_mod:load")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "toyfam_next", "num_hidden_layers": 2,
                                                      "hidden_size": 8, "intermediate_size": 16, "vocab_size": 10}))
    mx.save_safetensors(str(tmp_path / "model.safetensors"),
                        {f"model.layers.{i}.mlp.down_proj.weight": mx.ones((8, 16)) for i in range(2)})
    info = inspect_checkpoint(tmp_path)
    assert info["no_adapter"]["family"] == "toyfam_next" and info["sweep_family"] == "toyfam_next"
    assert info["fingerprint"] == "f" * 64 and info["checkpoint"].endswith("@ffffffff")
    cfg = K.resolve("quick", {}, {}, "")
    p = P.make_plan(info, cfg, texts=10)
    assert p["stream"]["mode"] == "rolling" and p["stream"]["family"] == "toyfam_next" and p["fits"]
    r = P.make_plan(info, cfg, texts=10, stream="resident")
    assert not r["fits"] and "no mlx2 adapter" in r["refusals"][0]
    monkeypatch.delitem(FAM.FAMILIES, "toyfam_next")
    with pytest.raises(SystemExit, match="no sweep family"):
        inspect_checkpoint(tmp_path)


QWEN06 = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


@pytest.mark.skipif(not (QWEN06 / "config.json").exists(), reason="Qwen3-0.6B not on disk")
def test_real_dense_streaming_reads_each_layer_once(tmp_path):
    """Regression (review of 55f501af2): with the stream manager's own policy every layer after the first was loaded
    once per BATCH. The host family holds one acquire per layer: each streamed layer is read once per chunk."""
    from research.amoeba.sweep.host_family import HostFamily
    host, stream = R.load_streamed_host(QWEN06, stream_mode="dense_layers", ceiling_bytes=int(0.1 * 2 ** 30),
                                        resident=False, moe_units="neuron", log=lambda *_: None)
    fam = HostFamily(host, stream=stream, model_path=QWEN06, stream_mode="dense_layers")
    seqs = _seqs(5, 16, 1000)
    before = stream.stats.page_ins
    rec, mu, nll = R.run(fam, seqs, [True] * 5, tmp_path, window=2, chunks=1, max_texts=2, log=lambda *_: None)
    streamed = len(fam._blocks)
    loads = stream.stats.page_ins - before
    assert streamed == host.L - 1                              # layer 0 resident
    assert loads == streamed                                   # one load per streamed layer per chunk (was ~1 per batch)
    par = R.parity(fam, seqs[:3], None, log=lambda *_: None)
    assert par["ok"]


def test_get_weight_reads_nested_quantized_params():
    nn = pytest.importorskip("mlx.nn")

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(4, 4)
    b = Blk()
    w = R._get_weight(b, "mlp.weight")
    assert w is not None and w.shape == (4, 4)
    assert R._get_weight(b, "mlp.missing") is None
    from mlx2.runtime.layer_stream import _set_weights
    _set_weights(b, {"mlp.weight": None})            # an unloaded streamed tensor reads back as None
    assert R._get_weight(b, "mlp.weight") is None


def test_rolling_keeps_the_resident_batches():
    """27B parity 2026-11-16: interleaved chunks changed batch neighbours and sizes, and with them the numbers."""
    import sys
    sys.path.insert(0, "scripts")
    import amoeba_topics_lib as T
    from mlx2.experimental.amoeba import child_rolling as R
    lens = [128] * 62 + [105, 51] + [128] * 5
    seqs = [[0] * n for n in lens]
    res = [sorted(b) for b in T.batches_by_len(seqs, 8)]
    assert [sorted(b) for b in R.resident_batches(seqs, 8)] == res
    for chunks in (1, 2, 3, 5):
        groups = R.chunk_batches(seqs, R.resident_batches(seqs, 8), chunks)
        got = sorted(sorted(b) for g in groups for b in g)
        assert got == sorted(res)                       # every resident batch, whole, exactly once
        for g in groups:                                # longest first within a chunk (memmap carry sizing)
            assert [len(seqs[b[0]]) for b in g] == sorted((len(seqs[b[0]]) for b in g), reverse=True)
    assert sorted(j for g in R.chunk_texts(seqs, 2) for j in g) == list(range(len(seqs)))


def _resident_batched_mu(h, seqs, fit, bs):
    """child_capture's mu exactly: the fit texts in their own length batches (batches_by_len(fit, bs)), a float32
    sum per batch added in mx float32 in batch order, then np.array(acc) / cnt."""
    import amoeba_topics_lib as T
    fseqs = [s for s, f in zip(seqs, fit) if f]
    L, W = h.L, max(h.widths)
    acc, cnt = [None] * L, np.zeros(L, np.float64)

    def st(l, kind, p):
        if kind == "dense":
            v = p["act"][:, 1:].astype(mx.float32).sum((0, 1))
            acc[l] = v if acc[l] is None else acc[l] + v
            cnt[l] += p["act"].shape[0] * (p["act"].shape[1] - 1)
    for g in T.batches_by_len(fseqs, bs):
        h.ctl.reset()
        h.ctl.stats = st
        mx.eval(h.hidden(mx.array([fseqs[j] for j in g])), *[x for x in acc if x is not None])
    h.ctl.reset()
    mu = np.zeros((L, W), np.float32)
    for l in range(L):
        if acc[l] is not None and cnt[l]:
            mu[l, :h.widths[l]] = np.array(acc[l]) / cnt[l]
    return mu


@pytest.mark.parametrize("carry", ["resident", "memmap"])
def test_fit_pass_mu_is_the_resident_mu(tmp_path, carry):
    """The 27B parity (2026-11-16) failed mu only: the resident capture takes mu from fit-only batches, the inline
    pass from the main batches' fit rows. The fit pass reproduces the resident batching and accumulation order."""
    from research.amoeba.sweep.host_family import HostFamily
    h = Host(model=TC.toy(moe=False), tokenizer=TC.Tok(), fingerprint="toy:abc", log=lambda *_: None)
    fam = HostFamily(h)
    rng = np.random.default_rng(7)
    seqs = [rng.integers(0, TC.V, size=t).tolist() for t in (12, 12, 12, 9, 12, 9, 12, 12, 9)]
    fit = [True, True, False, True, True, True, False, True, True]
    (tmp_path / carry).mkdir()
    (tmp_path / (carry + "-2")).mkdir()
    rec, mu, _ = R.run(fam, seqs, fit, tmp_path / carry, window=2, chunks=1, carry_mode=carry, max_texts=3,
                       log=lambda *_: None)
    assert rec["mu_pass"].startswith("fit") and rec["batch"] == 3
    want = _resident_batched_mu(h, seqs, fit, 3)
    dense = [l for l in range(h.L) if fam.kinds[l] == "dense"]
    assert dense and all(np.array_equal(mu[l], want[l]) for l in dense)       # bitwise
    rec2, mu2, _ = R.run(fam, seqs, fit, tmp_path / (carry + "-2"), window=2, chunks=2, carry_mode=carry,
                         max_texts=3, log=lambda *_: None)
    assert rec2["chunks"] == 2 and all(np.array_equal(mu2[l], want[l]) for l in dense)   # contiguous runs: bitwise


def test_inline_mu_pass_still_available(tmp_path):
    from research.amoeba.sweep.host_family import HostFamily
    h = Host(model=TC.toy(moe=False), tokenizer=TC.Tok(), fingerprint="toy:abc", log=lambda *_: None)
    seqs = _seqs(5, 12, TC.V)
    rec, mu, _ = R.run(HostFamily(h), seqs, [True, False, True, True, False], tmp_path, window=2, chunks=1,
                       max_texts=2, log=lambda *_: None, mu_pass="inline")
    assert rec["mu_pass"] == "inline" and np.abs(mu).sum() > 0



def test_contiguous_groups_keep_the_resident_order():
    seqs = [[0] * n for n in (9, 9, 9, 12, 12, 12, 12, 12, 12, 12)]
    batches = R.resident_batches(seqs, 2)
    for chunks in (1, 2, 3, 7):
        runs = R.contiguous_groups(seqs, batches, chunks)
        assert 1 <= len(runs) <= min(chunks, len(batches))     # never more runs than main chunks
        flat = [b for r in runs for b in sorted(r, key=lambda js: batches.index(js))]
        assert flat == batches                          # runs in order reproduce the resident order
    assert R.contiguous_groups(seqs, [], 3) == []


def test_fit_pass_stops_after_the_last_dense_layer(tmp_path, monkeypatch):
    from research.amoeba.sweep.host_family import HostFamily
    h = Host(model=TC.toy(moe=False), tokenizer=TC.Tok(), fingerprint="toy:abc", log=lambda *_: None)
    fam = HostFamily(h)
    dense = [l for l in range(h.L) if fam.kinds[l] == "dense"]
    fam.kinds = [k if l <= dense[0] else None for l, k in enumerate(fam.kinds)]   # only the first dense layer
    seen = []
    real = R._fit_layer
    monkeypatch.setattr(R, "_fit_layer", lambda f, l, *a: seen.append(l) or real(f, l, *a))
    R.run(fam, _seqs(5, 12, TC.V), [True] * 5, tmp_path, window=2, chunks=1, max_texts=2, log=lambda *_: None)
    assert seen == list(range(dense[0] + 1))


def test_carry_dirs_removed_on_failure(tmp_path):
    from research.amoeba.sweep.host_family import HostFamily
    h = Host(model=TC.toy(moe=False), tokenizer=TC.Tok(), fingerprint="toy:abc", log=lambda *_: None)
    fam = HostFamily(h)
    real = fam.run_layer

    def boom(l, carry):
        if l == 1:
            raise RuntimeError("layer failed")
        return real(l, carry)
    fam.run_layer = boom
    with pytest.raises(RuntimeError):
        R.run(fam, _seqs(5, 12, TC.V), [True] * 5, tmp_path, window=2, chunks=1, carry_mode="memmap", max_texts=2,
              log=lambda *_: None)
    assert not any((tmp_path / ".carry").glob("*/*"))


def test_clipped_batch_is_reported(tmp_path):
    from research.amoeba.sweep.host_family import HostFamily
    h = Host(model=TC.toy(moe=False), tokenizer=TC.Tok(), fingerprint="toy:abc", log=lambda *_: None)
    fam = HostFamily(h)
    fam.batch_rows = 12                                 # 12 rows / 12 tokens: batch 1 instead of 2
    logs = []
    rec, _, _ = R.run(fam, _seqs(5, 12, TC.V), [True] * 5, tmp_path, window=2, chunks=1, max_texts=2,
                      log=logs.append)
    assert rec["batch"] == 1 and rec["batch_matches_resident"] is False and any("clipped" in x for x in logs)


def test_plan_counts_the_fit_carry_only_with_dense_units():
    base = {"hidden": 10}
    assert P._fit_carry({**base, "units": {"dense_neurons": 5}}) == 2
    assert P._fit_carry({**base, "units": {"dense_neurons": 0}}) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_parity_requires_usable_logits(value):
    from types import SimpleNamespace

    logits = mx.full((1, 2, 3), value)
    family = SimpleNamespace(L=1, embed=lambda ids: ids, run_layer=lambda l, c: (c, None),
                             release_layer=lambda l: None, final_logits=lambda c: logits,
                             reference_logits=lambda ids: logits)
    assert not R.parity(family, [[1, 2]], None, log=lambda *_: None)["ok"]


def test_prefetch_read_error_reaches_caller(tmp_path, monkeypatch):
    shard = tmp_path / "shard"
    shard.write_bytes(b"x")

    def failed_read(*args):
        raise OSError("disk read failed")

    monkeypatch.setattr(R.os, "preadv", failed_read)
    pf = R.Prefetcher({1: [(shard, 0, 1)], 2: [(shard, 0, 1)]}, chunk=1)
    try:
        pf.request(1)
        pf.request(2)
        with pytest.raises(OSError, match="disk read failed"):
            pf.wait(2, timeout=0.25)
    finally:
        pf.close()


def test_negative_infinity_vocab_mask_preserves_finite_parity():
    from types import SimpleNamespace
    ref = mx.array([[[1.0, 2.0, -float("inf")]]])
    family = SimpleNamespace(L=1, embed=lambda ids: ids, run_layer=lambda l, c: (c, None),
                             release_layer=lambda l: None, final_logits=lambda _: ref + 1e-5,
                             reference_logits=lambda _: ref)
    receipt = R.parity(family, [[1]], None, log=lambda *_: None)
    assert receipt["ok"] and not receipt["texts"][0]["bitwise_equal"]


@pytest.mark.parametrize("change", ["none", "mu_nan", "short_rows", "different_ids", "signed_zero"])
def test_rolling_capture_comparison(tmp_path, monkeypatch, change):
    from mlx2.experimental.amoeba import cli, runner
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"A"}\n{"id":"B"}\n')
    out = tmp_path / "out"
    info = _info(1)
    monkeypatch.setattr(cli, "inspect_checkpoint", lambda _: info)
    monkeypatch.setattr(cli, "check_commit", lambda _: {"commit": "test"})

    def capture(cmd, **kw):
        dest = Path(cmd[cmd.index("--out") + 1])
        dest.mkdir()
        rolling = dest.name == "rolling"
        items = [{"id": "A"}, {"id": "B"}]
        if rolling and change == "different_ids":
            items.reverse()
        (dest / "items.json").write_text(json.dumps(items))
        (dest / "meta.json").write_text(json.dumps({"layers": 1, "sequences": 2, "widths": [1],
            "unit_kinds": ["dense"], "rolling": {}, "parity": {"ok": True}, "fingerprint": info["fingerprint"]}))
        count = 1 if rolling and change == "short_rows" else 2
        val = -0.0 if rolling and change == "signed_zero" else 0.0
        np.save(dest / "resp_L00.npy", np.full((count, 1), val, dtype=np.float16))
        np.save(dest / "mu.npy", np.array([[np.nan if rolling and change == "mu_nan" else 1.0]], np.float32))
        np.save(dest / "nll_full.npy", np.ones(2, np.float32))

    monkeypatch.setattr(runner, "run_child", capture)
    args = ["rolling-parity", "--model", "toy", "--out", str(out), "--rolling-resident", "--cpu",
            "--probe-corpus", str(corpus)]
    if change in ("short_rows", "different_ids"):
        with pytest.raises(SystemExit, match="shape|items"):
            cli.main(args)
    else:
        assert cli.main(args) == (3 if change == "mu_nan" else 0)
        result = json.loads((out / "parity.json").read_text())
        assert result["bitwise"] == (change != "signed_zero")
