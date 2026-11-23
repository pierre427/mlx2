"""scripts/amoeba_hy4_cjk_data.py on fake corpus builds (CPU, no model): splits, and no private source name leaks."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SECRET = "SecretCrawl"            # stands in for a private source name; must never appear outside a row's text


@pytest.fixture
def mod(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("amoeba_hy4_cjk_data", ROOT / "scripts/amoeba_hy4_cjk_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    builds = tmp_path / "builds"

    def jl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        op = gzip.open if str(path).endswith(".gz") else open
        with op(path, "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def doc(i, split, private, lang="x"):
        src = f"{SECRET.lower()}-{lang}" if private else f"wikipedia-{lang}"
        return {"id": f"{src}-{i}", "text": f"text {lang} {i} " * 20, "split": split, "origin": "natural",
                "source": f"{src}:{i}", "group": f"{src}:<urn:{i}>", "doc_id": f"{src}-doc-{i}",
                "license": f"ODC-By 1.0 ({SECRET}; crawl terms)" if private else "CC BY-SA 3.0 / GFDL (Wikipedia)",
                "source_revision": f"{SECRET}@abc" if private else "wikimedia@1", "redistribute": "no" if private else "yes"}
    for lang, (topic, build, short) in m.LANG.items():
        jl(builds / build / "corpus.jsonl", [doc(i, sp, i % 2 == 0, lang) for i, sp in
                                             enumerate(["fit"] * 6 + ["val"] * 2 + ["held"] * 2 + ["held2"] * 2)])
    jl(builds / m.ENG / "corpus.jsonl", [doc(i, sp, True, "eng") for i, sp in enumerate(["val", "held"] * 3)])
    jl(builds / m.BOXES / "corpus.jsonl.gz",
       [{**doc(i, sp, False, n), "node": n} for n in ("prog.python", "math.algebra")
        for i, sp in enumerate(["fit"] * 4 + ["val"] * 2 + ["held"] * 2)])
    probes = tmp_path / "probes/topic-probes-e8c22f31a/corpus.jsonl"
    jl(probes, [{"id": f"p{i}", "topics": ["lang.cmn"], "split": "fit", "text": "ui string", "group": "g",
                 "license": "UI strings (internal research use)"} for i in range(3)])
    jl(tmp_path / "probes/topic-probes-9a1f35475/corpus.jsonl",
       [{"id": f"e{i}", "topics": ["hum", "lang.eng"], "split": "held", "text": f"english {i} " * 20, "group": f"w{i}",
         "license": "CC BY-SA 4.0 (Wikipedia)"} for i in range(4)])
    gsm = tmp_path / "gsm8k"
    jl(gsm / "train.jsonl", [{"question": f"q{i}", "answer": f"a{i}"} for i in range(14)])
    jl(gsm / "test.jsonl", [{"question": f"t{i}", "answer": f"b{i}"} for i in range(3)])
    monkeypatch.setattr(m, "CORPUS", builds)
    monkeypatch.setattr(m, "PROBES", probes)
    monkeypatch.setattr(m, "PROBES_ENG", tmp_path / "probes/topic-probes-9a1f35475/corpus.jsonl")
    monkeypatch.setattr(m, "GSM8K", gsm)
    return m


def test_no_private_source_name_outside_the_text(mod, tmp_path):
    out = tmp_path / "out"
    mod.main(["--out", str(out), "--per-class-val", "5", "--per-class-held", "5"])
    n_private = 0
    for f in out.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            fields = json.dumps({k: v for k, v in r.items() if k != "text"})
            if r["private_corpus"]:
                n_private += 1
                assert SECRET.lower() not in fields.lower(), (f.name, fields)
                assert r["license"] == mod.PRIVATE_LICENSE and r["source_revision"] is None
                assert r["group"].startswith("private_corpus:") and r["doc_id"].startswith("private_corpus:")
    assert n_private > 0
    for f in ("inventory.json", "private-corpus-ids.json"):
        assert SECRET.lower() not in (out / f).read_text().lower()
    assert json.loads((out / "private-corpus-ids.json").read_text())["probe_ids"] == ["p0", "p1", "p2"]


def test_splits_stay_apart(mod, tmp_path):
    out = tmp_path / "out"
    mod.main(["--out", str(out)])
    rows = {f.name: [json.loads(x) for x in f.read_text().splitlines()] for f in out.glob("*.jsonl")}
    assert {r["split"] for r in rows["zh.jsonl"]} == {"fit", "held"}
    val = {r["doc_id"] for k in ("removal-target-val.jsonl", "removal-nontarget-val.jsonl") for r in rows[k]}
    held = {r["doc_id"] for k in ("removal-target-held.jsonl", "removal-nontarget-held.jsonl") for r in rows[k]}
    carve = {r["doc_id"] for k in ("zh.jsonl", "ja.jsonl", "hi.jsonl") for r in rows[k]}
    assert not (val & held) and not (val & carve) and not (held & carve)
    with pytest.raises(SystemExit, match="not empty"):
        mod.main(["--out", str(out)])
