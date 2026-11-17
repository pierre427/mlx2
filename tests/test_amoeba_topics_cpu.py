"""Amoeba topic map: taxonomy invariants, labeller gates on a synthetic capture, stub recipe (pure
functions), and the torch backend's masking on a tiny random Qwen3 (CPU, no download)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import amoeba_topics_lib as T  # noqa: E402
import amoeba_topics_stub as S  # noqa: E402


# ----------------------------------------------------------------------------- taxonomy

def test_taxonomy_ids_stable_and_wellformed():
    tax = T.Taxonomy()
    d = json.loads(T.TAXO.read_text())
    assert d["schema"] == "mlx2.amoeba-topics.v1"
    for x, n in tax.nodes.items():
        assert all(seg and seg.replace("_", "").isalnum() and seg == seg.lower() for seg in x.split("."))
        if n.get("parent"):
            assert x.startswith(n["parent"] + ".") and n["parent"] in tax.nodes
    for must in ("prog.python", "prog.shell.zsh", "prog.perl", "prog.web.html", "prog.web.css", "prog.cpp",
                 "sci.physics.quantum", "math.algebra.linear_algebra", "lang.fra", "lang.eng", "swe.git",
                 "ml.inference.kv_cache", "ml.inference.quantization", "ml.frameworks.mlx", "ml.frameworks.mlx2"):
        assert must in tax.nodes, must
    assert sum(x.startswith("lang.") for x in tax.nodes) > 7000
    for lab, ids in tax.legacy.items():
        assert all(i in tax.nodes for i in ids), lab
    # the builder reproduces the committed file exactly
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_topics_taxonomy.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_taxonomy_builder_refuses_dropping_ids(tmp_path):
    p = tmp_path / "t.json"
    d = json.loads(T.TAXO.read_text())
    d["nodes"].append({"id": "prog.made_up_old_id", "name": "x", "parent": "prog", "depth": 1})
    p.write_text(json.dumps(d))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_topics_taxonomy.py"), "--out", str(p)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "would disappear" in (r.stdout + r.stderr)


# ----------------------------------------------------------------------------- statistics

def test_cohen_d_and_auc():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 400), rng.normal(0, 1, 400)
    assert 0.8 < T.cohen_d(a, b) < 1.2
    assert 0.7 < T.auc(a, b) < 0.8
    X = np.stack([np.concatenate([a, b]), np.concatenate([b, a])], 1)
    d = T.cohen_d_rows(X, np.arange(400), np.arange(400, 800))
    assert d[0] > 0.8 and d[1] < -0.8
    null = T.perm_null_d(np.concatenate([a, b]), np.arange(400), np.arange(400, 800), 300, rng)
    assert abs(null.mean()) < 0.05 and null.std() < 0.1


def test_twins_keep_per_layer_counts():
    rng = np.random.default_rng(0)
    pool = {0: np.arange(100), 3: np.arange(50, 90)}
    members = {0: np.array([1, 2, 3]), 3: np.array([60, 61])}
    for tw in T.twin_sets(members, pool, 5, rng):
        assert {l: len(c) for l, c in tw.items()} == {0: 3, 3: 2}
        assert set(tw[3]) <= set(range(50, 90))


# ----------------------------------------------------------------------------- labeller on a synthetic capture

def _synthetic_capture(tmp, L=4, N=64, per=20, seed=0, hidden_python=()):
    """Texts for 3 prog leaves + 2 lang leaves; neurons 0-7 (layers 1,2) fire on prog.python only."""
    rng = np.random.default_rng(seed)
    topics = ["prog.python"] * (2 * per) + ["prog.c"] * (2 * per) + ["prog.rust"] * (2 * per) + \
             ["lang.fra"] * (2 * per) + ["lang.deu"] * (2 * per)
    S_ = len(topics)
    items = [{"id": f"t{i}", "topics": [t], "split": "fit" if i % 2 == 0 else "held", "n_tokens": 64}
             for i, t in enumerate(topics)]
    R = rng.gamma(4.0, 0.25, size=(L, S_, N)).astype(np.float32)
    py = np.array([t == "prog.python" for t in topics])
    prog = np.array([t.startswith("prog") for t in topics])
    for l in (1, 2):
        R[l][np.ix_(py, np.arange(8))] *= 2.5             # python-selective neurons
        R[l][np.ix_(prog, np.arange(8, 16))] *= 2.0       # prog-selective (not python-specific)
        if hidden_python:                                  # python neurons hidden in the noise cluster,
            shared_f = rng.gamma(4.0, 0.25, size=S_)       # driven by the same per-text signal as C0
            R[l][np.ix_(py, list(hidden_python))] *= 2.5
            R[l][:, 0:8] *= (1 + 0.3 * shared_f)[:, None]
            R[l][:, list(hidden_python)] *= (1 + 0.3 * shared_f)[:, None]
    for l in range(L):
        np.save(tmp / f"resp_L{l:02d}.npy", R[l].astype(np.float16))
    (tmp / "items.json").write_text(json.dumps(items))
    (tmp / "meta.json").write_text(json.dumps({"checkpoint": "synthetic@00000000", "layers": L, "neurons": N,
                                               "sequences": S_, "mlx2_source_revision": "test", "corpus_sha256": "x"}))
    core = np.zeros((L, N), np.float32)
    core[:, 40:] = 1
    np.savez(tmp / "core.npz", core=core)
    pool = np.argwhere(core == 0)
    lab = np.full(len(pool), 2)
    lab[(np.isin(pool[:, 0], [1, 2])) & (pool[:, 1] < 8)] = 0
    lab[(np.isin(pool[:, 0], [1, 2])) & (pool[:, 1] >= 8) & (pool[:, 1] < 16)] = 1
    np.savez(tmp / "clusters.npz", pool=pool.astype(np.int32), labels=lab.astype(np.int16))
    return tmp


def test_labeller_finds_planted_clusters(tmp_path):
    (tmp_path / "cap").mkdir()
    cap = _synthetic_capture(tmp_path / "cap")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_topics_label.py"), "--capture", str(cap),
                        "--corpus", str(corpus), "--out", str(tmp_path / "lab"), "--arrays", str(tmp_path / "arr"),
                        "--clusters", str(cap / "clusters.npz"), "--core", str(cap / "core.npz"),
                        "--twins", "8", "--perm", "300"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    lab = json.loads((tmp_path / "lab/labels.json").read_text())
    by = {c["id"]: c for c in lab["clusters"]}
    assert [l["topic"] for l in by["pool:C0"]["labels"]] == ["prog.python"]       # finest significant level
    assert [l["topic"] for l in by["pool:C1"]["labels"]] == ["prog"]               # only the branch is significant
    assert by["pool:C2"]["status"] == "unlabelled"                                 # noise cluster stays unlabelled
    # per-neuron d arrays: python neurons have high d for prog.python on both splits
    nodes = json.loads((tmp_path / "arr/neuron_d_nodes.json").read_text())
    dh = np.load(tmp_path / "arr/neuron_d_held.npy")
    k = nodes.index("prog.python")
    assert dh[k, 1, :8].min() > 1.0 and abs(dh[k, 1, 20:40]).mean() < 0.5


# ----------------------------------------------------------------------------- stub recipe

def test_select_prefix_bisection():
    gains = np.array([5, 3, 1, 0.5, 0.2, 0.1, 0.1, 0.05])
    cost = lambda k: gains[:k].sum()
    k, total, evals = S.select_prefix(len(gains), cost, 0.9)
    assert total == pytest.approx(gains.sum())
    assert k == 3 and cost(k) >= 0.9 * total and cost(k - 1) < 0.9 * total
    assert S.select_prefix(0, cost)[0] == 0


def test_split_units_exact_and_invariants():
    L, N = 3, 20
    core = np.zeros((L, N), bool); core[:, :10] = True
    box = np.zeros((L, N), bool); box[:, 10:14] = True
    rest = ~(core | box)
    units = {"core": core, "box": box, "pool": rest}
    stub = np.zeros((L, N), bool); stub[1, [2, 3, 11, 15]] = True
    after, moved = S.split_units(units, stub, "stub:x")
    assert moved == {"core": 2, "box": 1, "pool": 1}
    assert S.check_split(units, after)
    assert not (after["core"] & stub).any() and (after["stub:x"] == stub).all()
    with pytest.raises(ValueError):
        S.split_units({"a": core, "b": core}, stub)
    with pytest.raises(ValueError):
        S.split_units({"core": core}, stub)              # stub takes neurons no unit holds


def test_compartment_twin_matches_counts_and_compartments():
    rng = np.random.default_rng(0)
    L, N = 4, 50
    core = np.zeros((L, N), bool); core[:, :25] = True
    pairs = np.array([[0, 1], [0, 2], [0, 30], [2, 40], [3, 5]])
    ex = np.zeros((L, N), bool); ex[pairs[:, 0], pairs[:, 1]] = True
    tw = S.compartment_twin(pairs, core, ex, rng)
    assert len(tw) == len(pairs)
    for l in range(L):
        a, b = pairs[pairs[:, 0] == l, 1], tw[tw[:, 0] == l, 1]
        assert len(a) == len(b) and core[l, a].sum() == core[l, b].sum()
        assert not ex[l, b].any()


def test_layer_profile():
    prof = S.layer_profile([0.3, -0.1, 0.1], [[0, 1], [2, 3], [4, 5]], 6)
    assert sum(prof) == pytest.approx(1.0) and prof[2] == 0 and prof[0] == pytest.approx(0.375)
    assert sum(S.layer_profile([-1, 0], [[0], [1]], 2)) == 0


# ----------------------------------------------------------------------------- torch backend masking

def test_torch_backend_mask_is_exact(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    m = Qwen3ForCausalLM(cfg)
    m.save_pretrained(tmp_path)
    be = T.TorchBackend.__new__(T.TorchBackend)
    be.torch, be.model = torch, m.eval()
    be.L, be.N = 2, 48
    be._mask = be._cap = be._mu = None
    for l, layer in enumerate(m.model.layers):
        layer.mlp.down_proj.register_forward_pre_hook(be._hook(l))
    ids = [[1, 5, 9, 12, 7, 3, 8, 2]] * 2
    base = be.nll(ids)
    be.set_mask(np.ones((2, 48)))
    assert np.allclose(be.nll(ids), base, atol=0)
    mask = np.ones((2, 48), np.float32); mask[1, [3, 7]] = 0
    be.set_mask(mask)
    masked = be.nll(ids)
    be.set_mask(None)
    with torch.no_grad():
        m.model.layers[1].mlp.up_proj.weight[[3, 7]] = 0       # the same neurons silenced in the weights
    assert np.allclose(be.nll(ids), masked, atol=1e-6)
    st = be.stats(ids)
    assert st[1].shape == (2, 48) and np.allclose(st[1][:, [3, 7]], 0)


def test_monotone_and_bit_exact_helpers():
    assert S.is_monotone([(1, 0.1), (4, 0.3), (8, 0.29995)])
    assert not S.is_monotone([(1, 0.1), (4, 0.3), (8, 0.2)])
    m = np.zeros((2, 5), bool); m[1, [0, 3]] = True
    assert S.masks_match_lists(m, np.array([[1, 0], [1, 3]]))
    assert not S.masks_match_lists(m, np.array([[1, 0]]))
    assert not S.masks_match_lists(m, np.array([[1, 0], [1, 3], [1, 3]]))


def test_registry_functional_gate():
    import amoeba_topics_registry as R
    assert R.functional_pass(None, "x", 0.02) == (None, "not measured")
    fr = {"topics": {"x": {"remove_over_twin": 0.10, "remove_over_twin_max": 0.05},
                     "control(!prog)": {"remove_over_twin": 0.01}}}
    assert R.functional_pass(fr, "x", 0.02)[0] is True
    fr["topics"]["x"]["remove_over_twin_max"] = -0.01                 # one twin costs as much
    assert R.functional_pass(fr, "x", 0.02)[0] is False
    fr["topics"]["x"]["remove_over_twin_max"] = 0.05
    fr["topics"]["control(!prog)"]["remove_over_twin"] = 0.08          # not topic-specific
    assert R.functional_pass(fr, "x", 0.02)[0] is False


def test_split_units_copies_shared_neurons():
    L, N = 2, 10
    core = np.zeros((L, N), bool); core[:, :6] = True
    rest = ~core
    units = {"core": core, "pool": rest}
    stub = np.zeros((L, N), bool); stub[0, [1, 2, 7]] = True
    shared = np.zeros((L, N), bool); shared[0, 2] = True
    after, moved = S.split_units(units, stub, "stub:x", shared=shared)
    assert after["core"][0, 2] and after["stub:x"][0, 2]          # copied, not moved
    assert not after["core"][0, 1] and moved == {"core": 1, "pool": 1}
    assert S.check_split(units, after, shared=shared) and not S.check_split(units, after)


def test_puzzle_places_hidden_neurons(tmp_path):
    (tmp_path / "cap").mkdir()
    cap = _synthetic_capture(tmp_path / "cap", hidden_python=(20, 21, 22, 23))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("")
    py = sys.executable
    r = subprocess.run([py, str(ROOT / "scripts/amoeba_topics_label.py"), "--capture", str(cap), "--corpus", str(corpus),
                        "--out", str(tmp_path / "lab"), "--arrays", str(tmp_path / "arr"), "--clusters", str(cap / "clusters.npz"),
                        "--core", str(cap / "core.npz"), "--twins", "8", "--perm", "100"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    lab = json.loads((tmp_path / "lab/labels.json").read_text())
    assert {c["id"]: c["status"] for c in lab["clusters"]}["pool:C2"] == "unlabelled"
    r = subprocess.run([py, str(ROOT / "scripts/amoeba_topics_puzzle.py"), "--capture", str(cap), "--labels", str(tmp_path / "lab"),
                        "--arrays", str(tmp_path / "arr"), "--core", str(cap / "core.npz"), "--clusters", str(cap / "clusters.npz"),
                        "--out", str(tmp_path / "pz"), "--min-unit", "4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    z = np.load(tmp_path / "pz/puzzle.npz")
    placed = {(int(l), int(n)) for l, n in z["prog.python|pairs"]}
    assert {(l, n) for l in (1, 2) for n in (20, 21, 22, 23)} <= placed
    assert len(placed) <= 12                                   # noise neurons are not swept in
    rep = json.loads((tmp_path / "pz/puzzle.json").read_text())
    assert rep["rounds"][0]["new_members"] >= 8 and rep["units_for_gpu_check"] >= 1


def test_donor_gap_compare(tmp_path):
    rng = np.random.default_rng(0)
    for name, better in (("host", 0.0), ("donor", 0.1)):
        d = tmp_path / name
        d.mkdir()
        rows = []
        for i in range(60):
            topic = "prog.python" if i < 30 else "lang.fra"
            gain = better if topic == "prog.python" else 0.0
            for cond in ("bare", "prefix"):
                rows.append({"id": f"t{i}", "cond": cond, "topics": [topic], "group": f"g{i}",
                             "nats": float(100 * (0.8 - gain + 0.01 * rng.normal())), "bytes": 100, "tokens": 60})
        (d / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (d / "score.json").write_text(json.dumps({"checkpoint": name + "@0", "corpus_sha256": "x"}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_topics_donor_gap.py"), "compare", "--host", str(tmp_path / "host"),
                        "--donor", str(tmp_path / "donor"), "--out", str(tmp_path / "out"), "--boot", "300"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads((tmp_path / "out/donor_gap.json").read_text())
    assert "prog.python" in rep["candidates"] and "lang.fra" not in rep["candidates"]
    assert "prog.python" in rep["donor_better_centred"] and "lang.fra" in rep["host_better_centred"]
    assert not set(rep["donor_better_centred"]) & set(rep["host_better_centred"])


def test_corev2_add_generated(tmp_path):
    import amoeba_corev2_build as B
    assert B.trim_generated("short") is None
    long = "\n".join(f"line {i} of a long verse that goes on" for i in range(60))
    w = B.trim_generated(long)
    assert len(w) <= B.MAX and w.endswith("goes on") and long.startswith(w)
    base = tmp_path / "base.jsonl"
    base.write_text(json.dumps({"id": "cc00000", "level": "L1", "register": "narrative", "text": "x" * 500,
                                "source": "tinystories:val", "group": "ts/0", "license": "l", "split": "fit"}) + "\n")
    g = tmp_path / "gen"
    g.mkdir()
    (g / "receipt.json").write_text(json.dumps({"checkpoint": "A@1", "fingerprint": "1", "mlx2_source_revision": "r",
                                                "sampling": {}, "thinking": "off", "approved": "test", "label": "mg"}))
    rows = [{"id": "gen-song-000", "level": "L3", "register": "song", "form": "f", "theme": "t", "text": long,
             "rejected": None, "license": "mg", "generator": "A@1"},
            {"id": "gen-song-001", "level": "L3", "register": "song", "form": "f", "theme": "t", "text": "tiny",
             "rejected": None, "license": "mg", "generator": "A@1"},
            {"id": "gen-song-002", "level": "L3", "register": "song", "form": "f", "theme": "t", "text": long,
             "rejected": "garbled", "license": "mg", "generator": "A@1"}]
    (g / "generated.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"corpus_path": "p", "corpus_sha256": "s", "items": 1, "built": "b", "pending": "x"}))
    ns = type("A", (), {"out": str(tmp_path / "out"), "add_generated": str(g / "generated.jsonl"), "base": str(base),
                        "manifest": str(man)})
    B.add_generated(ns)
    items = [json.loads(l) for l in open(tmp_path / "out/corpus.jsonl")]
    assert [i["id"] for i in items] == ["cc00000", "cg00000"] and items[1]["label"] == "model-generated"
    m = json.loads(man.read_text())
    assert "pending" not in m and m["items"] == 2 and m["base"]["items"] == 1
    assert m["generated"]["dropped"] == {"under MIN chars after trim": 1, "rejected": 1}
    assert m["cells"]["L3/song"]["sources"] == {"artemis": 1}
    with pytest.raises(SystemExit):
        B.add_generated(ns)


def test_probe_corpus_v5_maps():
    import amoeba_topics_corpus as C
    valid = {n["id"] for n in json.loads(C.TAXO.read_text())["nodes"]}
    for m in (C.BISAC, C.COSMO_AUDIENCE, C.COSMO_FORMAT, C.COREV2_LIT, C.LANG_BOX_ALIASES):
        assert set(m.values()) <= valid
    assert set(C.LANG_BOX_ALIASES) <= valid and not set(C.LANG_BOX_ALIASES) & set(C.LANG_BOX_ALIASES.values())
    assert not {v for k, v in C.LPROJ.items()} & {a[5:] for a in C.LANG_BOX_ALIASES}
    assert C.bisac_node('focusing on "SCIENCE / Chemistry / Organic", for')[0] == "sci.chemistry.organic"
    assert C.bisac_node('on "HISTORY / Europe / France".')[0] == "hum.history"
    assert C.bisac_node('on "SELF-HELP / Motivational".')[0] is None
    assert C.bisac_node("no heading here") == (None, None)


def test_probe_origin_classes():
    import amoeba_topics_corpus as C
    assert C.origin_of("cosmopedia-v2:HISTORY / Ancient") == "synthetic" and C.origin_of("tinystories:val#3") == "synthetic"
    assert C.origin_of("doc:/x.md#2") == "lab" and C.origin_of("gen:arith#1") == "template"
    assert C.origin_of("corev2:pg19:pg1234#cc001") == "natural" and C.origin_of("flores:fra_Latn#5") == "natural"
    assert sorted(C.HELD_PREFERENCE, key=C.HELD_PREFERENCE.get) == ["natural", "lab", "template", "synthetic"]


def test_code_candidates_skip_rez(tmp_path):
    import random as _r
    import amoeba_topics_corpus as C
    sdk = tmp_path / "X.sdk" / "inc"
    sdk.mkdir(parents=True)
    (sdk / "Res.r").write_text("resource 'STR#' (128) {};\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.R").write_text("x <- c(1, 2)\n")
    (tmp_path / "pkg" / "util_test.go").write_text("package x\n")
    keep = C.code_candidates([(tmp_path, "lic")], _r.Random(0), 10, lambda m: None)
    assert [p.name for p, _ in keep["prog.r"]] == ["a.R"]
    assert [p.name for p, _ in keep["swe.testing"]] == ["util_test.go"]


def test_probe_redistribution_classes():
    import amoeba_topics_corpus as C
    assert C.redistribution("Apple SDK framework headers (internal research use)") == "no"
    assert C.redistribution("CC BY-SA 4.0 (FLORES+)") == "share-alike"
    assert C.redistribution("per-package (site-packages METADATA)") == "unresolved"
    assert C.source_revision("cosmopedia-v2:HISTORY") == "HuggingFaceTB/smollm-corpus@3ba9d605"
    assert C.source_revision("file:/nonexistent/x.py") == "file missing at build"
