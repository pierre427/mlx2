"""CPU tests of the FN language gate (scripts/amoeba_langgate_lib.py, scripts/amoeba_fn_langgate.py): the script
detector, the fastText numpy reader on a synthetic model written here (the real lid.176.bin is never loaded),
the gate policy with a mock LID (Han -> cmn + jpn, fail open, request cue, religion coupling), decode re-detection
with hold-open, oracle streams and gate metrics, the BAU tile groups and the arm masks the scorer builds, memory
accounting, the pre-registered decision rules, and the window end to end on the carve CPU toy (open == full,
receipts, prereg before results, the summary written under a time cap)."""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

import amoeba_langgate_lib as G  # noqa: E402


# ----------------------------------------------------------------------------- scripts

def test_script_counts_runs_and_viet_cue():
    sc = G.script_counts("abc 中文字 ひらがな カタ 한국 عربي हिंदी Привет ơư đ 123 ,。!")
    assert sc["han"] == 3 and sc["kana"] == 6 and sc["hangul"] == 2 and sc["arabic"] == 4
    assert sc["devanagari"] >= 3 and sc["cyrillic"] == 6 and sc["viet"] == 3
    assert sc["latin"] == 3 + 3                                  # abc + the Vietnamese letters
    assert G.script_counts("123 ,.;: 。、！")["letters"] == 0      # digits, punctuation, CJK symbols are no letters


# ----------------------------------------------------------------------------- fastText reader

def _fnv_ref(b):
    h = np.uint32(2166136261)
    with np.errstate(over="ignore"):
        for x in b:
            h = np.uint32(h ^ np.uint32(np.int32(np.int8(np.uint8(x)))))
            h = np.uint32(h * np.uint32(16777619))
    return int(h)


def test_ft_hash_is_fnv1a_with_int8_sign_extension():
    assert G.ft_hash(b"") == 2166136261
    assert G.ft_hash(b"a") == 0xE40C292C                         # the published FNV-1a 32 vector
    for s in (b"<ab>", "<é>".encode(), "中文".encode(), bytes(range(120, 140))):
        assert G.ft_hash(s) == _fnv_ref(s)


def test_ft_subwords_skip_boundary_unigrams_and_keep_code_points():
    assert G.ft_subwords(b"<ab>", 2, 3) == [b"<a", b"<ab", b"ab", b"ab>", b"b>"]
    assert G.ft_subwords(b"<ab>", 1, 3) == [b"<a", b"<ab", b"a", b"ab", b"ab>", b"b", b"b>"]   # never '<' / '>'
    sub = G.ft_subwords("<é>".encode(), 1, 2)
    assert sub == [b"<" + "é".encode(), "é".encode(), "é>".encode()]   # é is one code point (2 bytes)


def write_ft(path, words, labels, dim=4, bucket=10, minn=2, maxn=3, loss=1, quant=False, seed=0):
    """A fastText v12 supervised .bin: words [(w, count)], labels [(label, count)] in descending count order."""
    rng = np.random.default_rng(seed)
    nw, nl = len(words), len(labels)
    b = bytearray(struct.pack("<ii", G.FT_MAGIC, 12))
    b += struct.pack("<12i", dim, 5, 1, 1, 5, 1, loss, 3, bucket, minn, maxn, 100) + struct.pack("<d", 1e-4)
    b += struct.pack("<3i", nw + nl, nw, nl) + struct.pack("<2q", 100, -1)
    for w, c in words:
        b += w.encode() + b"\0" + struct.pack("<qb", c, 0)
    for lb, c in labels:
        b += ("__label__" + lb).encode() + b"\0" + struct.pack("<qb", c, 1)
    b += struct.pack("<?", quant)
    A = rng.standard_normal((nw + bucket, dim)).astype(np.float32)
    b += struct.pack("<2q", nw + bucket, dim) + A.tobytes()
    osz = nl - 1 if loss == 1 else nl
    O = rng.standard_normal((osz, dim)).astype(np.float32)
    b += struct.pack("<?", False) + struct.pack("<2q", osz, dim) + O.tobytes()
    Path(path).write_bytes(bytes(b))
    return A, O


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_fasttext_reader_hidden_and_hs_probabilities(tmp_path):
    words = [("</s>", 9), ("bonjour", 5), ("hello", 4)]
    labels = [("en", 6), ("fr", 3), ("de", 1)]
    A, O = write_ft(tmp_path / "m.bin", words, labels)
    m = G.FastTextLID(tmp_path / "m.bin")
    assert m.labels == ["en", "fr", "de"] and m.loss == "hs"
    # ids: hello (in vocab) = its id + its subwords; zzz (OOV) = subwords only; the end-of-line token = its id
    ids = [2] + [3 + G.ft_hash(s) % 10 for s in G.ft_subwords(b"<hello>", 2, 3)]
    ids += [3 + G.ft_hash(s) % 10 for s in G.ft_subwords(b"<zzz>", 2, 3)] + [0]
    assert np.allclose(m.hidden("hello zzz"), A[ids].mean(0))
    # buildTree on counts (6, 3, 1): node 3 = (left de, right fr); root 4 = (left node 3, right en)
    h = A[ids].mean(0).astype(np.float64)
    f3, f4 = _sig(O[0] @ h), _sig(O[1] @ h)
    lg = lambda x: np.log(x + 1e-5)                                   # noqa: E731  (fastText's std_log)
    want = {"en": np.exp(lg(f4)), "fr": np.exp(lg(1 - f4) + lg(f3)), "de": np.exp(lg(1 - f4) + lg(1 - f3))}
    got = m.probs("hello zzz")
    assert all(abs(got[k] - want[k]) < 1e-6 for k in want)          # float32 hidden and dot, as fastText
    assert abs(sum(got.values()) - 1.0) < 1e-3
    assert m.top("hello zzz", 1)[0][0] == max(want, key=want.get)
    m.close()


def test_fasttext_reader_softmax_and_refusals(tmp_path):
    A, O = write_ft(tmp_path / "s.bin", [("</s>", 2), ("a", 1)], [("en", 2), ("fr", 1)], loss=3)
    m = G.FastTextLID(tmp_path / "s.bin")
    z = O @ m.hidden("a")
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    assert np.allclose([m.probs("a")[k] for k in ("en", "fr")], p)
    write_ft(tmp_path / "q.bin", [("</s>", 2)], [("en", 2), ("fr", 1)], quant=True)
    with pytest.raises(ValueError, match="quantized"):
        G.FastTextLID(tmp_path / "q.bin")
    (tmp_path / "x.bin").write_bytes(b"\0" * 64)
    with pytest.raises(ValueError, match="not a fastText"):
        G.FastTextLID(tmp_path / "x.bin")


def test_load_lid_checks_the_recorded_sha256(tmp_path):
    write_ft(tmp_path / "lid.176.bin", [("</s>", 2)], [("en", 2), ("fr", 1)])
    src = {"files": {"lid.176.bin": {"sha256": "0" * 64, "url": "u"}}, "license": "CC BY-SA 3.0"}
    (tmp_path / "SOURCE.json").write_text(json.dumps(src))
    with pytest.raises(SystemExit, match="sha256"):
        G.load_lid(tmp_path)
    src["files"]["lid.176.bin"]["sha256"] = G.sha256_file(tmp_path / "lid.176.bin")
    (tmp_path / "SOURCE.json").write_text(json.dumps(src))
    m, rec = G.load_lid(tmp_path)
    assert rec["sha256_verified"] and rec["bundled"] is False and rec["license"] == "CC BY-SA 3.0"
    m.close()


# ----------------------------------------------------------------------------- policy

class MockLID:
    """Keyword LID: le/la -> fr, der/und -> de, el/los -> es, na/ya -> sw, ciao -> it; English otherwise; 'xq'
    words make it doubt between en and fr (gated mass 0.3, nothing above 0.5)."""
    KW = {"fr": ("le", "la"), "de": ("der", "und"), "es": ("el", "los"), "sw": ("na", "ya"), "it": ("ciao",)}

    def probs(self, text):
        w = text.lower().split()
        if any(x.startswith("xq") for x in w):
            return {"en": 0.55, "fr": 0.3, "it": 0.15}
        for lb, kws in self.KW.items():
            k = sum(x in kws for x in w)
            if k >= 2:
                return {lb: 0.9, "en": 0.1}
        return {"en": 0.95, "fr": 0.05}

    def top(self, text, k=3):
        return sorted(self.probs(text).items(), key=lambda x: -x[1])[:k]


class MockRel:
    threshold = 0.5

    def opens(self, text, stage="prefill"):
        s = 1.0 if "temple" in text else 0.0
        return s >= self.threshold, s


def pol(**kw):
    return G.GatePolicy(G.GateConfig(**kw), MockLID(), MockRel())


def test_policy_scripts_han_rule_and_latin_lid():
    p = pol()
    assert p.detect("这是一个中文句子")[0] == {"cmn", "jpn"}                     # Han-only opens both
    assert p.detect("これは日本語の文章です")[0] == {"jpn"}                       # Kana: jpn only
    assert p.detect("مرحبا كيف حالك")[0] == {"arb"}
    assert p.detect("नमस्ते आप कैसे हैं")[0] == {"hin"}
    assert p.detect("Привет как дела")[0] == {"rus"}
    assert p.detect("Tôi đang học tiếng Việt ở Hà Nội")[0] >= {"vie"}            # Vietnamese letters
    assert p.detect("le chat et la souris sont dans la maison")[0] == {"fra"}
    assert p.detect("the cat sat on the mat and looked at the dog")[0] == set()
    assert p.detect("中")[0] == set()                                             # one stray character
    assert p.detect("let α, β, γ and θ be angles of the triangle we study")[0] == set()   # maths Greek
    assert p.detect("the space ℝ and the operator ℒ act on 𝐱 and 𝐲 here")[0] == set()   # letterlike symbols
    assert p.detect("ｗｉｎｄｏｗｓ ｓｅｔｔｉｎｇｓ ａｒｅ ｈｅｒｅ")[0] == set()          # fullwidth Latin


def test_policy_fail_open_request_cue_and_religion_coupling():
    p = pol()
    long_doubt = "xqab " + "words of english text that is long enough to judge " * 2
    g, info = p.detect(long_doubt)
    assert g == set(G.LANG10) and info["flags"][0].startswith("uncertain:gated_mass")
    assert pol(on_uncertain="none").detect(long_doubt)[0] == set()
    assert p.detect("xqab short")[0] == set()                                     # too few letters to judge
    g, _ = p.detect("ciao bella ciao amici miei, come state oggi in questa bella giornata")
    assert g == set(G.LANG10)                                                     # uncovered language: fail open
    assert p.detect("한국어 문장입니다")[0] == set(G.LANG10)                         # uncovered script: fail open
    assert p.detect("Please answer in French: what is a cat?")[0] == {"fra"}
    assert pol(request_cue=False).detect("Please answer in French: what is a cat?")[0] == set()
    assert p.detect("the old temple by the river")[0] == {G.RELIGION, "hin"}
    assert pol(religion_couples=()).detect("the old temple by the river")[0] == {G.RELIGION}


def _toks(pieces):
    """Token i is pieces[i]; decode joins them."""
    return list(range(len(pieces))), (lambda ids: "".join(pieces[i] for i in ids))


def test_simulate_decode_redetection_with_hold_open():
    eng = ["the ", "cat ", "sat ", "on ", "the ", "mat ", "and ", "then "]
    fr = ["le ", "chat ", "et ", "la ", "souris ", "le ", "la ", "le "]
    pieces = eng * 2 + fr * 2 + eng * 2
    toks, dec = _toks(pieces)
    p = pol(check_every=4, window_tokens=4, lid_min_letters=4)
    S, log = G.simulate(toks, 8, dec, p)
    fi = G.GATES.index("fra")
    assert not S[:8, fi].any() and log["prefill"]["open"] == []
    # French starts at token 16; checks at t = 11, 15, 19, ...: the window of 15 is English, 19 is French
    first = int(np.flatnonzero(S[:, fi])[0])
    assert first == 19 and log["events"][0]["t"] == 19 and log["events"][0]["open"] == ["fra"]
    assert S[19:, fi].all()                                                       # hold-open through English
    S2, log2 = G.simulate(toks, 8, dec, replace_cfg(p, hold_tokens=6))
    assert S2[19:26, fi].all() and not S2[-1, fi]                                 # closes after 6 quiet rows
    assert any("close" in e for e in log2["events"])
    # oracle: row t needs the label of token t + 1; misses are the rows before the detection
    labels = [frozenset()] * 16 + [frozenset(["fra"])] * 16 + [frozenset()] * 16
    O = G.oracle_stream(labels)
    assert O[15, fi] and not O[14, fi] and not O[31, fi] and O[30, fi]
    m = G.gate_metrics(S, O, 8)
    assert m["missed"] == 19 - 15 and m["missed_rows"] == [15, 16, 17, 18] and m["prefill_exact"]
    assert m["extra"] == (len(pieces) - 2) - 30                                   # scored rows 31..46: held open


def replace_cfg(p, **kw):
    from dataclasses import replace
    return G.GatePolicy(replace(p.cfg, **kw), p.lid, p.rel)


def test_oracle_coupling():
    lab = [frozenset([G.RELIGION])] * 4
    O = G.oracle_stream(lab, ("hin",))
    assert O[:, G.GATES.index("hin")].all() and O[:, G.GATES.index(G.RELIGION)].all()
    assert not G.oracle_stream(lab, ())[:, G.GATES.index("hin")].any()


# ----------------------------------------------------------------------------- tiles

KEYS = ["core", "box:cmn", "box:jpn", "box:fra", "box:hum.religion", "box:sci.astronomy", "box:lang.eng",
        "pair:cmn__jpn", "pair:fra__sci.astronomy", "pair:cmn__hum.religion", "pair:hum.religion__sci.astronomy",
        "group:langs", "group:cjkv", "group:ml"]


def test_gate_groups_bau_rules():
    core, groups, kinds = G.gate_groups(KEYS)
    F = frozenset
    assert set(core) == {"core", "box:sci.astronomy", "box:lang.eng", "pair:hum.religion__sci.astronomy",
                         "group:ml"}
    assert groups[F(["cmn"])] == ["box:cmn"] and groups[F(["fra"])] == ["box:fra", "pair:fra__sci.astronomy"]
    assert groups[F(["cmn", "jpn"])] == ["pair:cmn__jpn"] and groups[F(["cmn", G.RELIGION])] == ["pair:cmn__hum.religion"]
    assert groups[F(G.LANG10)] == ["group:langs"] and groups[F(["cmn", "jpn", "vie"])] == ["group:cjkv"]
    assert kinds["pair:fra__sci.astronomy"] == "pair_mixed" and kinds["pair:cmn__hum.religion"] == "pair_lang_religion"
    core2, groups2, _ = G.gate_groups(KEYS, mixed_pairs="core")
    assert "pair:fra__sci.astronomy" in core2 and groups2[F(["fra"])] == ["box:fra"]
    _, groups3, _ = G.gate_groups(KEYS, pairs="closed")
    assert set(groups3[F()]) == {"pair:cmn__jpn", "pair:fra__sci.astronomy", "pair:cmn__hum.religion"}
    S = np.zeros((3, len(G.GATES)), bool)
    S[1:, G.GATES.index("jpn")] = True
    gs = G.group_streams(S, groups)
    assert gs[F(["cmn", "jpn"])].tolist() == [False, True, True] and not gs[F(["cmn"])].any()
    assert gs[F(G.LANG10)].tolist() == [False, True, True]
    assert not G.group_streams(S, groups3)[F()].any()


def test_gated_bytes_exact_layout():
    W = 128                                                   # 2 layers x 128 units, groups of 64 along F
    idx = {"a": np.arange(0, 64), "b": np.arange(64, 100), "c": np.arange(128 + 64, 128 + 128)}
    ub, gu = G.unit_bytes(2560, 4, 64)
    assert (ub, gu) == (4320, 2880)
    r = G.gated_bytes(idx, W, ["a", "b", "c"])
    assert r["units"] == 64 + 36 + 64 and r["full_down_groups"] == 2
    assert r["approx_bytes"] == r["units"] * 4320 and r["exact_bytes"] == r["units"] * 2880 + 2 * 64 * 1440
    assert G.gated_bytes(idx, W, [])["units"] == 0


# ----------------------------------------------------------------------------- decision rules

def _rows(kinds, auto=0.01, open_=0.0, mixed_gap=0.005):
    rows = {}
    for s, k in kinds.items():
        rows[("open", s)] = {"mean": open_, "ci95": [-1e-4, 1e-4]}
        rows[("auto", s)] = {"mean": auto, "ci95": [auto - 0.01, auto + 0.01]}
        if k == "mixed":
            rows[("auto-oracle", s)] = {"mean": mixed_gap, "ci95": [0.0, 0.01]}
    return rows


def test_decide_rules():
    kinds = {"english": "english", "gsm8k": "gsm8k", "prog.python": "code", "cmn": "lang", "fra": "lang",
             G.RELIGION: "religion", "sci.astronomy": "topic", "mixed.switch": "mixed"}
    pool, gen = {"needed": 1000, "missed": 10}, {"full": 20, "auto": 20}
    gs = {"mixed.switch": {"needed_decode": 100, "missed_decode": 3}}
    dec = G.decide

    def decide(r, p, g, k):
        return dec(r, p, g, k, gate_sets=gs)
    G_decide = decide
    v = G_decide(_rows(kinds), pool, gen, kinds)
    assert v["verdict"] == "pass" and not v["failed"] and "unqualified" in v["label"]
    r = _rows(kinds)
    r[("auto", "cmn")]["mean"] = 0.08                         # within cmn's known served miss (0.084)
    assert G_decide(r, pool, gen, kinds)["verdict"] == "pass"
    r[("auto", "fra")]["mean"] = 0.06                         # fra has no known miss: the +0.05 bar
    assert G_decide(r, pool, gen, kinds)["failed"] == ["fra:auto - full"]
    assert G_decide(_rows(kinds, open_=0.01), pool, gen, kinds)["verdict"] == "INVALID"
    assert G_decide(_rows(kinds, mixed_gap=0.03), pool, gen, kinds)["failed"] == ["mixed.switch:auto - oracle"]
    assert G_decide(_rows(kinds), {"needed": 100, "missed": 6}, gen, kinds)["verdict"] == "fail"
    assert G_decide(_rows(kinds), pool, {"full": 20, "auto": 19}, kinds)["verdict"] == "fail"
    r = _rows(kinds)
    del r[("auto", "english")]
    assert G_decide(r, pool, gen, kinds)["verdict"] == "incomplete"
    assert G_decide(_rows(kinds), pool, None, kinds)["verdict"] == "incomplete"
    gs["mixed.switch"]["missed_decode"] = 6                    # the mixed set's own decode rows over 5%
    assert G_decide(_rows(kinds), pool, gen, kinds)["failed"] == ["mixed.switch:decode missed-open rate"]
    gs["mixed.switch"]["missed_decode"] = 3
    r = _rows(kinds, auto=0.2)
    del r[("open", "english")]                                 # sanity not run: provisional, never "fail"
    v = G_decide(r, pool, gen, kinds)
    assert v["verdict"] == "incomplete" and v["provisional"] and v["failed"]
    assert G_decide(_rows(kinds), {}, gen, kinds)["missing"] == ["gate pool"]


def test_religion_nb_fit_threshold_and_unseen_words():
    rng = np.random.default_rng(0)
    rel = ["god", "prayer", "temple", "faith", "scripture", "monk", "saint"]
    gen = ["market", "river", "engine", "price", "school", "train", "garden", "music"]
    mk = lambda pool, n: " ".join(rng.choice(pool + gen, n))           # noqa: E731
    pos = [mk(rel, 60) for _ in range(40)]
    neg = [" ".join(rng.choice(gen, 60)) for _ in range(40)]
    nb = G.ReligionNB(buckets=1 << 12).fit(pos, neg)
    m = nb.calibrate([mk(rel, 30) for _ in range(20)], [" ".join(rng.choice(gen, 30)) for _ in range(20)], 0.9)
    assert m["val_recall"] >= 0.9 and m["val_fpr"] <= 0.1
    assert nb.score("完全 不同 的 文字") == 0.0                                    # unseen features count 0


# ----------------------------------------------------------------------------- window on the carve toy

mx = pytest.importorskip("mlx.core")


class ToyTok:
    """A reversible character tokenizer over a fixed alphabet of < 64 characters (the toy's vocabulary)."""
    ALPHA = (" \nabcdefghijklmnopqrstuvwxyz.,:中文のд" + "بक" + "ơ" + "?!")

    def __init__(self):
        self.i = {c: k for k, c in enumerate(self.ALPHA)}

    def encode(self, text, add_special_tokens=False):
        return [self.i[c] for c in text if c in self.i]

    def decode(self, ids):
        return "".join(self.ALPHA[i] if i < len(self.ALPHA) else "?" for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **k):
        return self.encode("user: " + messages[-1]["content"] + "\n")


WORDS = {"eng": "the cat sat on a mat and the dog ran far", "fra": "le chat et la souris sont la le",
         "deu": "der hund und der ball und der mann", "spa": "el gato y los perros el sol los",
         "swh": "na ya mtoto na ya mama na ya", "vie": "tơi ơn cơm ơi",
         "cmn": "中文中文 中文", "jpn": "ののの 中の", "rus": "дддд дд", "arb": "بببب بب", "hin": "कककक कक",
         "rel": "the temple and the old temple by the river", "code": "def f x: return x, y.",
         "topic": "stars and planets orbit far suns"}


def _rows_of(kind, n, splits=("fit", "val", "held", "held2"), salt=""):
    out = []
    for sp in splits:
        for i in range(n):
            tag = "".join("abcdefghij"[int(c)] for c in f"{len(out):03d}")       # unique text per row
            out.append({"id": f"{kind}{salt}:{sp}:{i}", "group": f"g{kind}{sp}{i}", "split": sp,
                        "text": (WORDS[kind] + " ") * 30 + tag})
    return out


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def _fixture(tmp_path, h):
    rng = np.random.default_rng(1)
    L, W = h.L, max(h.widths)
    names = ["core", *[f"box:{x}" for x in G.LANG10], f"box:{G.RELIGION}", "box:sci.astronomy", "box:lang.eng",
             "pair:cmn__jpn", "pair:fra__sci.astronomy", "pair:cmn__hum.religion", "group:langs", "group:cjkv"]
    own = rng.integers(0, len(names), size=(L, W))
    own[:, ::3] = 0                                          # a third of the units in the core
    pdir = tmp_path / "partition"
    pdir.mkdir()
    np.savez(pdir / "tiles.npz", layers=np.arange(L), **{k: own == i for i, k in enumerate(names)})
    corp = tmp_path / "corpora"
    corp.mkdir()
    specs = {}
    files = {}
    kmap = {**{L_: L_ for L_ in G.LANG10}, "english": "eng", G.RELIGION: "rel", "prog.python": "code",
            "sci.astronomy": "topic"}
    for name, kind in kmap.items():
        files[name] = _write(corp / f"{name}.jsonl", _rows_of(kind, 8, salt=name))
        specs[name] = f"{name}={files[name]}@24"
    gsm = _write(corp / "gsm.jsonl", [{"question": "the cat has two mats, how many cats", "answer": "two cats"}] * 6)
    specs["gsm8k"] = f"gsm8k=gsm8k:{gsm}@24"
    boxes = {b: {"target_corpus": files.get(b, files["sci.astronomy"]), "baseline_corpus": files["english"]}
             for b in [*G.LANG10, G.RELIGION, "sci.astronomy", "lang.eng"]}
    prep = {"fingerprint": "toy:abc", "checkpoint": "Toy@abc", "boxes": boxes, "nested": {"ancestors": {}}}
    (pdir / "partition.json").write_text(json.dumps(prep))
    bat = tmp_path / "battery"
    bat.mkdir()
    (bat / "battery.meta.json").write_text(json.dumps({"held": {k: {"spec": v} for k, v in specs.items()}}))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"hidden_size": 32, "quantization": {"bits": 4, "group_size": 4}}))
    return tmp_path, model


def _toy_host():
    from research.amoeba.carve.host import Host
    from test_amoeba_carve_neuron_units_cpu import qn_toy
    return Host(model=qn_toy(), tokenizer=ToyTok(), fingerprint="toy:abc", log=lambda *_: None, moe_units="neuron")


@pytest.fixture
def toy_ops(monkeypatch):
    mx.set_default_device(mx.cpu)
    from research.amoeba.carve import ops as O
    monkeypatch.setattr(O, "prepare_wiring", lambda spec: {})
    monkeypatch.setattr(O, "footprint_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(O, "setup_mlx", lambda *a, **k: None)
    monkeypatch.setattr(O, "wire_weights", lambda *a, **k: {"wired": False, "spec": "toy"})
    return O


def _argv(src, model, out, *extra):
    return ["--src", str(src), "--model", str(model), "--out", str(out), "--docs", "4", "--tokens", "120",
            "--prompt-tokens", "8", "--mixed-per-lang", "1", "--topics", "sci.astronomy", "--check-every", "4",
            "--window-tokens", "8", "--gen-tokens", "4", "--rel-window-words", "6", "--rel-neg-per-box", "4",
            "--ce-batch", "3", "--focus", "cmn,hum.religion,mixed.switch", *extra]


def test_scorer_masks_equal_expected_tile_sets(tmp_path, toy_ops):
    import amoeba_box_battery as B
    import amoeba_fn_langgate as LG
    h = _toy_host()
    src, _ = _fixture(tmp_path, h)
    tz = B.Tiles(src / "partition" / "tiles.npz")
    core_keys, groups, _ = G.gate_groups(tz.keys)
    sc = LG.Scorer(h, tz, core_keys, groups, -1, 2, lambda *_: None)
    F = frozenset
    st = {g: np.zeros((2, 5), bool) for g in groups}
    st[F(["cmn"])][:] = True                                  # all open -> folded into the core
    st[F(["fra"])][1, 2:] = True                              # per token -> a box with that token gate
    st[F(["cmn", "jpn"])][1, 2:] = True                       # the same stream -> merged with fra's
    nf, nb = sc.configure(st)
    assert (nf, nb) == (1, 1)
    c = h.ctl
    want_core = tz.mask(*core_keys, *groups[F(["cmn"])])
    for l, n in enumerate(h.widths):
        assert np.array_equal(np.asarray(c.core[l]) > 0, want_core[l, :n])
        box = np.asarray(c.boxes[0]["mask"][l]) > 0
        assert np.array_equal(box, tz.mask(*groups[F(["fra"])], *groups[F(["cmn", "jpn"])])[l, :n])
    assert np.array_equal(np.asarray(c.boxes[0]["token_gate"])[..., 0] > 0, st[F(["fra"])])
    assert c.sink_layers == h.L
    c.reset()


def test_scorer_open_equals_full_and_token_gates_equal_static_masks(tmp_path, toy_ops):
    import amoeba_box_battery as B
    import amoeba_fn_langgate as LG
    h = _toy_host()
    src, _ = _fixture(tmp_path, h)
    tz = B.Tiles(src / "partition" / "tiles.npz")
    core_keys, groups, _ = G.gate_groups(tz.keys)
    sc = LG.Scorer(h, tz, core_keys, groups, -1, 3, lambda *_: None)
    rng = np.random.default_rng(4)
    docs = [rng.integers(0, 40, 12).tolist() for _ in range(5)]
    T = 12
    full = sc.ce(docs)
    ones = [np.ones((T, len(G.GATES)), bool) for _ in docs]
    zeros = [np.zeros((T, len(G.GATES)), bool) for _ in docs]
    assert np.array_equal(sc.ce(docs, ones), full) or np.allclose(sc.ce(docs, ones), full, atol=1e-6)
    closed = sc.ce(docs, zeros)
    assert not np.allclose(closed, full)
    # a constant stream equals its static composition (the generation path's set_state)
    fi = G.GATES.index("fra")
    con = [z.copy() for z in zeros]
    for s in con:
        s[:, fi] = True
    a = sc.ce(docs, con)
    sc.set_state([g for g in groups if g and "fra" in g])
    from research.amoeba.carve.carving import ce_rows
    ids = mx.array(docs)
    ref = np.array(ce_rows(h.logits_at(ids, list(range(1, T - 1))), ids[:, 2:]))
    h.ctl.reset()
    assert np.allclose(a, ref, atol=1e-5)
    # causality: a gate opening at row 6 leaves every target before row 6's prediction as closed's
    sw = [z.copy() for z in zeros]
    for s in sw:
        s[6:, fi] = True
    b = sc.ce(docs, sw)
    assert np.allclose(b[:, :5], closed[:, :5], atol=1e-6)        # targets 2..6 come from rows 1..5
    assert not np.allclose(b[:, 5:], closed[:, 5:])                 # later targets see the opened tiles


def test_window_end_to_end_on_the_toy(tmp_path, toy_ops):
    import amoeba_fn_langgate as LG
    h = _toy_host()
    src, model = _fixture(tmp_path, h)
    out = tmp_path / "out"
    lrec = {"path": "mock", "sha256": "0", "bundled": False}
    r = LG.main(_argv(src, model, out), host_factory=lambda log: h, lid_factory=lambda: (MockLID(), lrec))
    assert r["status"] == "finished"
    s = json.loads((out / "summary.json").read_text())
    pre = json.loads((out / "prereg.json").read_text())
    assert s["prereg_sha256_12"] == G.sha256_file(out / "prereg.json")[:12] and pre["rules"]["target"] == 0.05
    assert s["states"] == {"implemented": True, "qualified": False, "selected": False, "observed_used": False}
    assert s["lid"]["bundled"] is False and s["mlx2_source_revision"]
    res = {(x["arm"], x["set"]): x for x in s["results"]}
    sets = set(s["sets"])
    assert {"english", "gsm8k", "prog.python", "cmn", G.RELIGION, "sci.astronomy", "mixed.request",
            "mixed.switch", "mixed.doc"} <= sets
    for name in sets:
        assert abs(res[("open", name)]["dce"]["all"]["mean"]) < 1e-5          # open == full on every set
        assert ("auto", name) in res and ("oracle", name) in res and ("closed", name) in res
    assert ("auto_nohin", G.RELIGION) in res and ("oracle_han", "cmn") in res and ("auto_nohin", "fra") not in res
    assert "vs_oracle" in res[("auto", "mixed.switch")]
    assert s["verdict"]["verdict"] in ("pass", "fail") and s["verdict"]["label"].startswith("unqualified")
    gens = [json.loads(x) for x in (out / "generations.jsonl").read_text().splitlines()]
    assert {g["arm"] for g in gens} == {"full", "auto"} and s["generations"]["prompts_both_arms"] == len(gens) // 2
    assert (out / "perdoc.jsonl").exists() and (out / "gate.jsonl").exists() and (out / "summary.md").exists()
    g = s["gate"]["per_set"]["mixed.switch"]
    assert g["needed"] > 0 and 0.0 <= g["missed_rate"] <= 1.0
    lines = [json.loads(x) for x in (out / "gate.jsonl").read_text().splitlines()]
    assert all("prefill" in x and "events" in x for x in lines)


def test_window_time_cap_still_writes_the_summary(tmp_path, toy_ops):
    import amoeba_fn_langgate as LG
    h = _toy_host()
    src, model = _fixture(tmp_path, h)
    out = tmp_path / "out"
    lrec = {"path": "mock", "sha256": "0", "bundled": False}
    r = LG.main(_argv(src, model, out, "--time-cap-min", "0"), host_factory=lambda log: h,
                lid_factory=lambda: (MockLID(), lrec))
    s = json.loads((out / "summary.json").read_text())
    assert r["status"] == LG.NO_CELLS and s["results"] == [] and s["not_run"]      # nothing scored: not "finished"
    assert s["verdict"]["missing"] and s["verdict"]["verdict"] != "pass" and any("time cap" in x for x in s["not_run"])


def test_window_refuses_another_model(tmp_path, toy_ops):
    import amoeba_fn_langgate as LG
    h = _toy_host()
    src, model = _fixture(tmp_path, h)
    h.fingerprint = "other"
    with pytest.raises(SystemExit, match="partition of"):
        LG.main(_argv(src, model, tmp_path / "out"), host_factory=lambda log: h,
                lid_factory=lambda: (MockLID(), {"bundled": False}))
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert s["status"].startswith("error")


def test_clock_deadline_night_cutoff():
    import amoeba_fn_langgate as LG
    base = time.mktime((2026, 11, 21, 3, 0, 0, 0, 0, -1))           # 03:00
    assert LG.clock_deadline("06:30", base) - base == 3.5 * 3600
    late = time.mktime((2026, 11, 20, 23, 30, 0, 0, 0, -1))
    assert LG.clock_deadline("06:30", late) - late == 7 * 3600
    after = time.mktime((2026, 11, 21, 6, 40, 0, 0, 0, -1))         # started after the cutoff: nothing starts
    assert LG.clock_deadline("06:30", after) == after
