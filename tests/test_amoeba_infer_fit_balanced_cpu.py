"""CPU test of scripts/amoeba_infer_fit_balanced.py: per-source balance, short-row joining within a source, held rows
copied unchanged, held groups excluded from fit, rows hitting two boxes skipped, refusal of an existing out dir."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/amoeba_infer_fit_balanced.py"


def _row(i, topics, source, group, words, split="fit"):
    return {"id": f"r{i}", "topics": topics, "split": split, "text": " ".join(f"w{i}x{k}" for k in range(words)),
            "origin": "natural", "source": source, "source_revision": "rev", "group": group, "license": "MIT",
            "redistribute": "yes", "label_method": "keyword", "sha256": "x"}


def _setup(tmp):
    rows, i = [], 0
    for src in ("code", "arxiv"):
        for k in range(40):                                      # long rows, many repos
            rows.append(_row(i, ["ml.inference.kv"], f"{src}:{src}{k}", f"g-{src}{k}", 30)); i += 1
    for k in range(12):                                          # short trace rows: joined three at a time (3 x 10)
        rows.append(_row(i, ["ml.llm.chat"], f"trace:t{k}", f"g-t{k}", 10)); i += 1
    rows.append(_row(i, ["ml.deep_learning.a"], "wiki:s1", "g-s1", 10)); i += 1          # short, then a long row:
    rows.append(_row(i, ["ml.deep_learning.a"], "wiki:l1", "g-l1", 30)); i += 1          # the long one stands alone
    rows.append(_row(i, ["ml.llm.x", "ml.inference.y"], "code:both", "g-both", 30)); i += 1   # two boxes: skipped
    rows.append(_row(i, ["ml.inference.kv"], "code:heldrepo", "g-held", 30)); i += 1          # held group: excluded
    (tmp / "probes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    base = tmp / "base"
    base.mkdir()
    for b in ("ml.inference", "ml.llm", "ml.deep_learning"):
        held = [_row(900 + len(b), [b + ".h"], "code:heldrepo" if b == "ml.inference" else f"code:h{b}",
                     "g-held" if b == "ml.inference" else f"g-h{b}", 30, split="held")]
        (base / f"{b}.jsonl").write_text("\n".join(json.dumps(r) for r in held) + "\n")
    (base / "manifest.json").write_text("{}")


def _run(tmp, out, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--probes", str(tmp / "probes.jsonl"), "--base-fit",
                           str(tmp / "base"), "--out", str(tmp / out), "--min-tokens", "25", "--tokenizer", "words", *extra],
                          capture_output=True, text=True)


def test_balanced_build(tmp_path):
    _setup(tmp_path)
    r = _run(tmp_path, "out", "--per-source", "16")
    assert r.returncode == 0, r.stderr
    inf = [json.loads(x) for x in (tmp_path / "out/ml.inference.jsonl").read_text().splitlines()]
    fit = [x for x in inf if x["split"] == "fit"]
    by = {}
    for x in fit:
        by[x["source"].split(":")[0]] = by.get(x["source"].split(":")[0], 0) + 1
    assert by == {"code": 16, "arxiv": 16}
    assert all("heldrepo" not in x["source"] and "both" not in x["source"] for x in fit)
    assert [x["id"] for x in inf if x["split"] == "held"] == ["r912"]      # held copied unchanged
    llm = [json.loads(x) for x in (tmp_path / "out/ml.llm.jsonl").read_text().splitlines() if '"fit"' in x]
    assert len(llm) == 4 and all(len(x["parts"]) == 3 for x in llm)       # 12 short rows -> 4 joined windows
    assert all(len(x["text"].split()) >= 25 for x in llm)
    m = json.loads((tmp_path / "out/manifest.json").read_text())
    assert m["boxes"]["ml.llm"]["per_source"]["trace"]["joined"] == 4
    dl = [json.loads(x) for x in (tmp_path / "out/ml.deep_learning.jsonl").read_text().splitlines() if '"fit"' in x]
    assert [x["parts"] for x in dl] == [["r93"]]                            # long row alone; short one left unjoined
    assert m["boxes"]["ml.deep_learning"]["per_source"]["wiki"]["short_rows_left_unjoined"] == 1
    assert _run(tmp_path, "out").returncode != 0                           # refuses an existing out dir
