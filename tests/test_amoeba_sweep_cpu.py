"""aMoEba sweep capture on a tiny random-weight DS4.1 (real classes, shrunk config); CPU only."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_deepseek_v41_cpu import Weights, config, hash_metadata  # noqa: E402
from test_deepseek_v41_source_layer_major_cpu import Rows  # noqa: E402

from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model  # noqa: E402
from mlx2.runtime.models.deepseek_v41_engram import EngramHasher  # noqa: E402
from research.amoeba.sweep import capture as C  # noqa: E402
from research.amoeba.sweep.deepseek_v41 import DeepSeekV41Family, route_signal_from_parts  # noqa: E402


def tiny(kind="f32", engram=False):
    kw = dict(index_topk=64, candidate_topk_blocks=64, candidate_block_size=2)
    if kind == "bf16":
        kw.update(hidden_size=32, head_dim=32, q_lora_rank=32, qk_rope_head_dim=16, index_head_dim=32,
                  o_lora_rank=16, moe_intermediate_size=32, simulate_quantization=True, activation_dtype="bfloat16")
    if engram:
        kw.update(engram_layer_ids=(1, 3), engram_num_embeddings=(48, 72), engram_compressed_vocab_size=5)
    c = config(**kw)
    w = Weights(c)
    extra = {}
    if engram:
        rng = np.random.default_rng(1989)
        for layer in c.engram_layer_ids:
            w.data[f"blk.{layer}.engram_q_norm.weight"] = mx.ones((4, c.hidden_size))
            w.data[f"blk.{layer}.engram_k_norm.weight"] = mx.ones((4, c.hidden_size))
            w.data[f"blk.{layer}.engram_kv.weight"] = mx.array(
                rng.normal(0, 0.0004, (5 * c.hidden_size, 6144)).astype(np.float32))
        md = hash_metadata()
        md["deepseek41.engram.token_map"] = [i % 5 for i in range(c.vocab_size)]
        extra = dict(hasher=EngramHasher(md), engram_tables={i: DiscardRows() for i in c.engram_layer_ids})
    model = DeepSeekV41Model(c, w, identity="tiny-sweep@1", index_policy="finite_all_retained", **extra)
    return model


class DiscardRows(Rows):
    def discard_prefetch(self):
        self.pending = None


def family(model, **kw):
    return DeepSeekV41Family(model, None, checkpoint="tiny@deadbeef", fingerprint="deadbeef" * 8, **kw)


def seqs(n=7, vocab=19):
    rng = np.random.default_rng(3)
    return [[int(t) for t in rng.integers(3, vocab, size=int(rng.choice([9, 10, 12])))] for _ in range(n)]


def run(fam, ss, tmp, **kw):
    out, st = tmp / "out", tmp / "state"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "mu.npy", np.zeros((fam.L, max(fam.widths)), np.float32))
    final = C.sweep(fam, ss, ["fit" if i % 2 else "eval" for i in range(len(ss))], out, st, log=lambda m: None,
                    parity=True, **kw)
    return out, final


@pytest.mark.parametrize("kind,engram", [("f32", False), ("bf16", False), ("f32", True), ("bf16", True)])
def test_sweep_final_logits_bitwise_equal_ordinary_forward(tmp_path, kind, engram):
    model = tiny(kind, engram)
    fam = family(model, ffn_texts=1)
    ss = seqs()
    _, final = run(fam, ss, tmp_path)
    assert fam.counters["select_replays"] > 0
    for j, s in enumerate(ss):
        got = fam.final_logits(final.read(np.array([j])))
        ref = fam.reference_logits(mx.array([s], mx.int32))
        np.testing.assert_array_equal(np.array(got).view(np.uint32), np.array(ref).view(np.uint32))


def test_packed_ffn_matches_ordinary_within_tolerance(tmp_path):
    model = tiny("bf16")
    fam = family(model)                      # several texts per FFN call
    ss = seqs()
    _, final = run(fam, ss, tmp_path)
    for j, s in enumerate(ss):
        got = np.array(fam.final_logits(final.read(np.array([j]))))
        ref = np.array(fam.reference_logits(mx.array([s], mx.int32)))
        np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2)


def ordinary_resp(model, ss):
    """Independent oracle: whole-model ordinary forward per text, routed calls observed."""
    c = model.args
    w = model.weights
    original = w.moe
    seen = []

    def moe(x, ids, scores, prefix):
        out = original(x, ids, scores, prefix)
        k = ids.shape[-1]
        sig = [mx.linalg.norm(original(x, ids, scores * (mx.arange(k) == j).astype(scores.dtype), prefix), axis=-1)
               for j in range(k)]
        seen.append((int(prefix.split(".")[1]), np.array(ids), np.array(mx.stack(sig, -1))))
        return out
    w.moe = moe
    try:
        res = np.zeros((c.num_hidden_layers, len(ss), c.n_routed_experts))
        for j, s in enumerate(ss):
            seen.clear()
            model(mx.array([s], mx.int32))
            for l, ids, sig in seen:
                for t in range(1, len(s)):
                    for q in range(ids.shape[1]):
                        res[l, j, ids[t, q]] += sig[t, q]
                res[l, j] /= len(s) - 1
    finally:
        del w.moe
    return res


def test_resp_shapes_widths_and_values(tmp_path):
    model = tiny("f32")
    fam = family(model, ffn_texts=1)
    ss = seqs()
    out, _ = run(fam, ss, tmp_path)
    ref = ordinary_resp(model, ss)
    for l in range(fam.L):
        r = np.load(out / f"resp_L{l:02d}.npy")
        assert r.shape == (len(ss), max(fam.widths)) and r.dtype == np.float16
        np.testing.assert_allclose(r.astype(np.float32), ref[l], rtol=2e-3, atol=2e-3)
    prog = json.loads((out / "progress.json").read_text())
    assert prog["done"] == list(range(fam.L))


def test_resume_matches_uninterrupted(tmp_path):
    ss = seqs()
    a_out, _ = run(family(tiny("bf16"), ffn_texts=1), ss, tmp_path / "a")
    fam = family(tiny("bf16"), ffn_texts=1)
    b = tmp_path / "b"
    out, st = b / "out", b / "state"
    out.mkdir(parents=True)
    np.save(out / "mu.npy", np.zeros((fam.L, max(fam.widths)), np.float32))
    split = ["fit"] * len(ss)
    assert C.sweep(fam, ss, split, out, st, log=lambda m: None, max_layers=2) is None
    assert json.loads((out / "progress.json").read_text())["done"] == [0, 1]
    assert sorted(x.name for x in st.iterdir()) == ["L02"]                        # only the live state kept
    fam2 = family(tiny("bf16"), ffn_texts=1)
    C.sweep(fam2, ss, split, out, st, log=lambda m: None, resume=True)
    for l in range(fam.L):
        np.testing.assert_array_equal(np.load(out / f"resp_L{l:02d}.npy"), np.load(a_out / f"resp_L{l:02d}.npy"))


def test_route_signal_slot_map():
    values = np.array([[2, 0], [1, 2], [0, 1]], np.uint32)
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="stable")
    rows = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)   # rows per original route
    parts = [mx.array(rows[order[flat[order] == e]]) for e in np.unique(flat)]
    sig = np.array(route_signal_from_parts(parts, values, order))
    np.testing.assert_allclose(sig, np.linalg.norm(rows, axis=-1).reshape(values.shape), rtol=1e-6)


def test_unit_rows_matches_carve_definition():
    inds = mx.array([[[0, 2], [1, 2], [2, 3]]], mx.int32)
    sig = mx.array([[[9.0, 9.0], [1.0, 2.0], [3.0, 4.0]]])
    r = np.array(C.unit_rows("experts", {"inds": inds, "signal": sig}, 4))
    np.testing.assert_allclose(r, [[0, 0.5, 2 / 2 + 3 / 2, 2.0]])


def test_reference_index_policy_fails_closed():
    c = config(index_topk=64, candidate_topk_blocks=64)
    model = DeepSeekV41Model(c, Weights(c), identity="tiny@1")
    with pytest.raises(ValueError, match="finite_all_retained"):
        family(model)


def test_state_refuses_growing_shapes(tmp_path):
    st = C.LayerState(tmp_path / "L00", 4)
    st.write(np.array([0, 1]), {"h": mx.zeros((2, 3, 2))})
    with pytest.raises(ValueError, match="non-decreasing"):
        st.write(np.array([2]), {"h": mx.zeros((1, 5, 2))})


@pytest.mark.parametrize("split_planes", [False, True])
@pytest.mark.parametrize("fmt", ["fp32", "bf16_lut"])
def test_engram_span_reader_matches_ordinary_reader(tmp_path, split_planes, fmt):
    from mlx2.runtime.models.deepseek_v41_engram import EngramRows, file_stamp
    from research.amoeba.sweep.deepseek_v41 import read_engram_rows
    rng = np.random.default_rng(5)
    n = 5000
    codes = rng.integers(0, 256, size=(n, 256), dtype=np.uint8)
    codes[(codes & 127) == 127] = 0x38
    scales = rng.integers(110, 140, size=(n, 8), dtype=np.uint8)
    path = tmp_path / "rows"
    if split_planes:
        path.write_bytes(b"\0" * 64 + codes.tobytes() + scales.tobytes())
        kw = dict(offset=64, scale_offset=64 + n * 256)
    else:
        path.write_bytes(b"\0" * 64 + np.concatenate([codes, scales], 1).tobytes())
        kw = dict(offset=64)
    table = EngramRows(path, rows=n, expected_stat=file_stamp(path.stat()), cache_format=fmt, **kw)
    try:
        ids = np.unique(rng.integers(0, n, size=700))
        out = np.zeros((len(ids), 256), np.uint16)
        stats = read_engram_rows(table, ids, out, gap_bytes=2048, span_bytes=1 << 16)
        assert stats["engram_spans"] > 2
        ref = table.read(ids.reshape(-1, 1)).reshape(len(ids), 256)
        got = (out.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(got.view(np.uint32), np.ascontiguousarray(ref, np.float32).view(np.uint32))
    finally:
        table.close()


def test_prefix_parity_against_truncated_ordinary_forward(tmp_path):
    model = tiny("bf16", engram=True)
    fam = family(model, ffn_texts=1)
    out, st = tmp_path / "out", tmp_path / "st"
    out.mkdir()
    rows = C.run_parity(fam, seqs(4), out, st, log=lambda m: None, layers=3)
    assert fam.L == model.args.num_hidden_layers
    assert all(r["bitwise_equal"] for r in rows)
    assert not (out / "resp_L03.npy").exists() and (out / "resp_L02.npy").exists()


def test_fresh_run_never_reuses_stale_state(tmp_path):
    fam = family(tiny("f32"), ffn_texts=1)
    out, st = tmp_path / "out", tmp_path / "st"
    out.mkdir()
    np.save(out / "mu.npy", np.zeros((fam.L, max(fam.widths)), np.float32))
    C.sweep(fam, seqs(5), ["fit"] * 5, out, st, log=lambda m: None, max_layers=1)
    other = [[int(t) % 19 for t in q[::-1]] for q in seqs(5)]
    with pytest.raises(SystemExit, match="another run"):
        C.sweep(fam, other, ["fit"] * 5, out, st, log=lambda m: None, resume=True)
    out2 = tmp_path / "out2"
    out2.mkdir()
    np.save(out2 / "mu.npy", np.zeros((fam.L, max(fam.widths)), np.float32))
    C.sweep(fam, other, ["fit"] * 5, out2, st, log=lambda m: None)          # fresh: re-embeds
    ref = ordinary_resp(fam.m, other)
    np.testing.assert_allclose(np.load(out2 / "resp_L00.npy").astype(np.float32), ref[0], rtol=2e-3, atol=2e-3)


class Killed(Exception):
    pass


def test_kill_inside_a_layer_loses_only_that_layer(tmp_path):
    ss = seqs()
    a_out, _ = run(family(tiny("f32"), ffn_texts=1), ss, tmp_path / "a")
    fam = family(tiny("f32"), ffn_texts=1)
    inner, calls = fam.run_layer, []

    def run_layer(l, carry):
        calls.append(l)
        if l == 2 and calls.count(2) == 2:
            raise SystemExit(143)              # what the SIGTERM handler raises
        return inner(l, carry)
    fam.run_layer = run_layer
    out, st = tmp_path / "b" / "out", tmp_path / "b" / "st"
    out.mkdir(parents=True)
    np.save(out / "mu.npy", np.zeros((fam.L, max(fam.widths)), np.float32))
    split = ["fit"] * len(ss)
    with pytest.raises(SystemExit):
        C.sweep(fam, ss, split, out, st, log=lambda m: None, max_texts=2)
    assert json.loads((out / "progress.json").read_text())["done"] == [0, 1]
    fam2 = family(tiny("f32"), ffn_texts=1)
    C.sweep(fam2, ss, split, out, st, log=lambda m: None, resume=True, max_texts=2)
    for l in range(fam.L):
        np.testing.assert_array_equal(np.load(out / f"resp_L{l:02d}.npy"), np.load(a_out / f"resp_L{l:02d}.npy"))
    assert sorted(x.name for x in st.iterdir()) == [f"L{fam.L:02d}"]


def test_parity_gate():
    from research.amoeba.sweep.parity_check import check
    t = lambda i, d, top=1.0, bit=False: {"text": i, "max_abs": d, "ref_max_abs": 10.0, "bitwise_equal": bit, "top1_agree": top}
    base = {"checkpoint": "x@1", "fingerprint": "1", "layers": 4}
    ex = {**base, "texts": [t(i, 0.0, bit=True) for i in range(3)],
          "counters": {"check_signal_calls": 4, "check_signal_max_rel": 1e-5, "select_replays": 3, "combine_captures": 8}}
    pk = {**base, "texts": [t(i, 0.05) for i in range(3)]}
    assert check(ex, pk) == []
    assert check(ex, {**pk, "texts": [t(0, 0.5), t(1, 0.0), t(2, 0.0)]})
    assert check({**ex, "texts": [t(0, 0.5), t(1, 0.0, bit=True), t(2, 0.0, bit=True)]}, pk)
    assert check({**ex, "counters": {**ex["counters"], "check_signal_max_rel": 0.1}}, pk)
