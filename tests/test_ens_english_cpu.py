"""H44 English edge tile: phrasing, parsing and model shapes (CPU)."""
import random

import mlx.core as mx

from research.ens import english as E
from research.ens import families as F
from research.ens import englishtile as T


def test_phrasings_fit_and_carry_the_numbers():
    rng = random.Random(0)
    for fid in T.FAMILIES:
        fam = F.BY_ID[fid]
        for split in ("train", "held"):
            for r in F.family_rows(fam, split)[:200]:
                for held in (False, True):
                    s = E.phrase(fam, r, rng, held=held)
                    assert len(s.encode()) <= E.MAX_BYTES
                    assert E.from_bytes(E.to_bytes(s)) == s


def test_held_wording_is_disjoint_from_train_wording():
    banks = [E.EDGE_PHRASES, E.HOP_PHRASES, E.STATE_PHRASES, E.BITS_PHRASES, E.ANSWER_FRAMES,
             *E.OP_PHRASES.values(), *E.FRAMES.values()]
    for train, held in banks:
        assert not set(train) & set(held)


def test_true_prompts_parse_to_their_family():
    rng = random.Random(1)
    for fid in T.FAMILIES:
        fam = F.BY_ID[fid]
        for r in F.family_rows(fam, "held")[:50]:
            ids, q, f, a = F.layout(r, "bus")
            assert T.parse(ids) == (fid, q, f, a)
    ids = F.layout(F.family_rows(F.BY_ID["graph.6-2.h1"], "held")[0], "bus")[0]
    bad = list(ids); bad[0] = bad[1]
    assert T.parse(bad) is None
    assert T.parse([0] * F.SEQ_LEN) is None


def test_numbers_read_back():
    for n in range(64):
        assert E.read_number(f"The answer is {E.number_word(n)}.") == n
        assert E.read_number(f"Result: {n}.") == n
    assert E.read_number("no number here") is None


def test_model_shapes():
    r = T.Reader(d=32, enc_layers=1, dec_layers=1)
    b = mx.array([E.to_bytes("Look up key 3, then add 5. What is the result?")] * 2)
    assert r(b).shape == (2, F.SEQ_LEN, 128)
    ren = T.Renderer(d_core=16, hidden=24, emb=8)
    st = mx.zeros((2, 16))
    assert ren.generate(st, n=5).shape == (2, 5)


def test_parse_rejects_unnamed_ids():
    from research.ens import vocab as V
    ids = [V.BOS, V.Q, V.K0, V.tok("ADD"), V.val(3), 100, V.val(4), V.SLOT, V.NEUTRAL, V.ANS]
    assert T.parse(ids + [V.PAD] * (F.SEQ_LEN - len(ids))) is None


def test_core_run_matches_the_core():
    from research.ens.model import BusCall, Core, CoreConfig
    mx.random.seed(0)
    core = Core(CoreConfig(d=32, layers=3, heads=2, ffn=64, bus_width=8))
    r = F.family_rows(F.BY_ID["graph.6-2.h1"], "held")[0]
    ids, q, f, a = F.layout(r, "bus")
    tok = mx.array([ids])
    msg = mx.random.normal((1, 8))
    bus = BusCall("early", mx.array([q]), mx.array([f]), msg)
    want = core(tok, bus)
    got, _ = T.core_run(core, tok, BusCall("early", mx.array([q]), mx.array([f]), msg))
    assert float(mx.max(mx.abs(want - got)).item()) == 0.0
    assert float(mx.max(mx.abs(core(tok) - T.core_run(core, tok, None)[0])).item()) == 0.0
    import pytest
    with pytest.raises(KeyError):
        T.core_run(core, tok, BusCall("nowhere", mx.array([q]), mx.array([f]), msg))


def test_aux_grammar_rows_phrase_and_parse_to_no_dev_family():
    rng = random.Random(3)
    for fam, r in T.aux_rows(rng, 200):
        assert len(E.phrase(fam, r, rng, held=False).encode()) <= E.MAX_BYTES
    assert set(T.AUX_FAMILIES).isdisjoint(T.FAMILIES)


def test_op_guess_rate_bounds():
    fam = F.BY_ID["graph.6-2.h1"]
    for r in F.family_rows(fam, "held")[:20]:
        g = T.op_guess_rate(fam, r)
        assert 1 / fam.z_domain <= g <= 1.0
