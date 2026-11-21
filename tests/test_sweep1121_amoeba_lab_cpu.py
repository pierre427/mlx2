"""Bug sweep 2026-11-21, lane amoeba-lab (AML2-1 .. AML2-9). CPU only: no model checkpoint, no network."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


# ----------------------------------------------------------------------------- AML2-1: --box-from unit kind

def test_box_from_refuses_other_unit_kind_depth_and_ids():
    import amoeba_cwe_donor as CD
    nunits = np.array([8, 8])
    bj = {"checkpoint": "donor@abc", "moe_units": "expert", "box": [[1, 5, 6], [0, 7]], "twin": [[2], [3]]}
    assert CD.load_box(bj, "donor@abc", "expert", nunits) == (bj["box"], bj["twin"])
    # an expert box in a neuron run would become neurons 1, 5, 6 of expert 0
    with pytest.raises(SystemExit, match="moe_units='expert'"):
        CD.load_box(bj, "donor@abc", "neuron", np.array([8 * 16, 8 * 16]))
    legacy = {k: v for k, v in bj.items() if k != "moe_units"}  # written before the field: an expert box
    assert CD.load_box(legacy, "donor@abc", "expert", nunits)[0] == bj["box"]
    with pytest.raises(SystemExit, match="moe_units"):
        CD.load_box(legacy, "donor@abc", "neuron", nunits)
    with pytest.raises(SystemExit, match="not donor@other"):
        CD.load_box(bj, "donor@other", "expert", nunits)
    with pytest.raises(SystemExit, match="layers"):
        CD.load_box(bj, "donor@abc", "expert", np.array([8, 8, 8]))
    with pytest.raises(SystemExit, match="outside"):
        CD.load_box(bj, "donor@abc", "expert", np.array([8, 7]))


# ----------------------------------------------------------------------------- AML2-2: one book, one side

def test_literary_windows_of_one_book_share_the_split_group():
    import amoeba_corev2_build as CB
    g_narr, c_narr = CB.literary_group("pg19", "test_43845", "narrative")
    g_poem, c_poem = CB.literary_group("pg19", "test_43845", "poetry")
    assert g_narr == g_poem and c_narr != c_poem  # the split unit is the book, the cap unit stays per register
    g_a, c_a = CB.literary_group("gut", "pg20476", 399 // 400)
    g_b, c_b = CB.literary_group("gut", "pg20476", 400 // 400)
    assert g_a == g_b and c_a != c_b
    assert CB.literary_group("pg19", "test_1", "narrative")[0] != g_narr
    # future builds are distinguishable from the ones split by the old groups
    assert "book-groups" in CB.SPLIT_LAW
    src = (ROOT / "scripts" / "amoeba_corev2_build.py").read_text()
    assert '"split_law": SPLIT_LAW' in src


# ----------------------------------------------------------------------------- AML2-3: refusals close run.json

def _overlay_fixture():
    spec = importlib.util.spec_from_file_location("_overlay_tests", ROOT / "tests" / "test_amoeba_donor_overlay_cpu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_donor_overlay_refusal_leaves_a_failed_receipt(tmp_path):
    T = _overlay_fixture()
    probes, mp = T._fixture(tmp_path)
    man = json.loads(mp.read_text())
    man["host_checkpoint"] = "another-host@00000000"
    mp.write_text(json.dumps(man))
    out = tmp_path / "out"
    argv = ["--stage", "arms", "--host", "host", "--donor", "donor", "--out", str(out), "--probes", str(probes),
            "--split-manifest", str(mp), "--arms", "tile_sft"] + T.COMMON
    with pytest.raises(SystemExit) as ei:
        T.DO.run_guarded(argv, loader=T._loader())
    assert "another-host@00000000" in str(ei.value.code)
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed" and "another-host@00000000" in run["reason"]


def test_donor_overlay_refusal_before_the_saver_exists_and_foreign_dirs(tmp_path):
    T = _overlay_fixture()
    probes, mp = T._fixture(tmp_path)
    out = tmp_path / "out"
    argv = ["--stage", "arms", "--host", "host", "--donor", "donor", "--out", str(out), "--probes", str(probes),
            "--split-manifest", str(mp), "--arms", "tile_sft"] + T.COMMON

    def refusing_loader(path):  # stands for a refusal while models load
        T.DO._STATE.pop("save", None)
        raise SystemExit("load refused")

    with pytest.raises(SystemExit):
        T.DO.run_guarded(argv, loader=refusing_loader)
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed" and run["reason"] == "load refused"
    # a non-empty --out is another run's directory: its receipt is not touched
    before = (out / "run.json").read_text()
    with pytest.raises(SystemExit, match="non-empty"):
        T.DO.run_guarded(argv, loader=T._loader())
    assert (out / "run.json").read_text() == before


# ----------------------------------------------------------------------------- AML2-4: donor-trace reuse

def test_donor_traces_are_keyed_on_prompt_format_and_token_cap():
    import amoeba_hub_transfer as HT
    cur = {"task": "gsm549", "prompt_format": "gsm8k5shot", "max_tokens": 1024, "seed": 1}
    assert HT.traces_mismatch(dict(cur, seed=2), cur) == {}  # only the fields a trace depends on
    assert HT.traces_mismatch(dict(cur, prompt_format="chat"), cur) == {"prompt_format": ("chat", "gsm8k5shot")}
    assert set(HT.traces_mismatch(dict(cur, max_tokens=256), cur)) == {"max_tokens"}
    old = {"task": "gsm549", "max_tokens": 1024}  # a run.json from before the option existed
    assert set(HT.traces_mismatch(old, cur)) == {"prompt_format"}
    assert set(HT.traces_mismatch(None, cur)) == set(HT.TRACE_KEYS)


# ----------------------------------------------------------------------------- AML2-5: --values-from identity

def _facts_manifest(**over):
    rm = {"source": {"artifact": "a", "fingerprint": "fp-A"}, "host_tiles_manifest_sha256": "tiles-A",
          "tiles": [{"read_layer": 21, "write_layer": 23}], "value_solver": {"value_templates": 5, "iters": 4}}
    rm.update(over)
    return rm


def test_values_from_refuses_rows_solved_on_another_model():
    import amoeba_det_facts_cpu as DF
    ents = ["France", "Peru", "Kenya"]
    rows = [{"entity": e, "method": "min-norm first-order step on log p(first answer token) x4"} for e in ents]
    kw = dict(src_fp="fp-A", host_sha="tiles-A", read_layer=21, write_layer=23, entities=ents, value_templates=5,
              iters=4)
    assert DF.values_reuse_mismatch(_facts_manifest(), rows, **kw) == ({}, [])
    bad, _ = DF.values_reuse_mismatch(_facts_manifest(), rows, **{**kw, "src_fp": "fp-B"})
    assert bad == {"source.fingerprint": ("fp-A", "fp-B")}
    bad, _ = DF.values_reuse_mismatch(_facts_manifest(), rows, **{**kw, "host_sha": "tiles-B"})
    assert set(bad) == {"host_tiles_manifest_sha256"}
    bad, _ = DF.values_reuse_mismatch(_facts_manifest(), rows, **{**kw, "value_templates": 3, "iters": 6})
    assert set(bad) == {"value_templates", "iters"}
    bad, _ = DF.values_reuse_mismatch(_facts_manifest(), rows, **{**kw, "write_layer": 24})
    assert set(bad) == {"write_layer"}
    bad, _ = DF.values_reuse_mismatch(_facts_manifest(), rows[::-1], **kw)
    assert set(bad) == {"row_order"}
    # a manifest from before value_solver: still readable, iters from the rows, the wordings count flagged
    legacy = _facts_manifest()
    del legacy["value_solver"]
    assert DF.values_reuse_mismatch(legacy, rows, **kw) == ({}, ["value_templates"])
    bad, _ = DF.values_reuse_mismatch(legacy, rows, **{**kw, "iters": 6})
    assert set(bad) == {"iters"}


def test_rows_deleted_manifest_states_its_own_row_counts():
    src = (ROOT / "scripts" / "amoeba_det_facts_cpu.py").read_text()
    assert '"rows": len(keep), "value_rows": int((~nullv[keep]).sum())' in src
    assert 'save("table-rows-deleted", V[keep], K[keep], nullv[keep], [rows_src[i] for i in keep], tile=tile_del)' in src


# ----------------------------------------------------------------------------- AML2-6: shuffled-arm partner

def test_shuffled_generation_partner_falls_back_instead_of_dividing_by_zero():
    import amoeba_bridge_lab as L
    items = [{"id": "it0001", "split": "test", "ids": [1, 40, 41, 42, 43, 44]},
             {"id": "it0002", "split": "test", "ids": [1, 50, 51, 52, 53, 54]},
             {"id": "it0003", "split": "test", "ids": [1, 60, 61, 62, 63, 64, 65, 66, 67]},  # unique length
             {"id": "it0004", "split": "train", "ids": [1, 70, 71, 72, 73, 74, 75, 76, 77]}]
    ids, partner = L.shuf_partner_prompt(items, items[0])
    assert partner == "it0002" and ids == items[1]["ids"]
    ids, partner = L.shuf_partner_prompt(items, items[2])  # the train item of that length is not a partner
    assert partner is None
    assert ids[0] == 1 and sorted(ids) == sorted(items[2]["ids"]) and ids != items[2]["ids"]
    assert L.shuf_partner_prompt(items, items[2])[0] == ids  # fixed per item
    assert L.shuf_partner_prompt(items[:1], items[0])[1] is None  # --n-test 1


# ----------------------------------------------------------------------------- AML2-7: --box-json donor identity

def test_box_json_is_tied_to_the_loaded_donor():
    import amoeba_bridge_lab as L
    width = {3: 64, 5: 64}
    bj = {"box_layers": [3, 5], "box": {"3": [0, 9, 63], "5": [4]}, "donor_checkpoint": "donor@abc",
          "donor_mlp_width": {"3": 64, "5": 64}}
    bad, rec = L.box_json_identity(bj, None, "donor@abc", width)
    assert bad == [] and rec["source"] == "box.json" and "warning" not in rec
    bad, _ = L.box_json_identity(bj, None, "donor@other", width)
    assert len(bad) == 1 and "donor_checkpoint" in bad[0]
    bad, _ = L.box_json_identity(bj, None, "donor@abc", {3: 128, 5: 64})
    assert len(bad) == 1 and "donor_mlp_width layer 3" in bad[0]
    # files written before the identity fields: the run.json beside them decides
    old = {k: v for k, v in bj.items() if k in ("box_layers", "box")}
    bad, rec = L.box_json_identity(old, {"donor_checkpoint": "donor@abc"}, "donor@abc", width)
    assert bad == [] and rec["source"].startswith("run.json")
    bad, _ = L.box_json_identity(old, {"donor_checkpoint": "donor@other"}, "donor@abc", width)
    assert bad and "donor_checkpoint" in bad[0]
    # nothing recorded anywhere: still readable, with a warning for the receipt
    bad, rec = L.box_json_identity(old, None, "donor@abc", width)
    assert bad == [] and rec["source"] == "unrecorded" and rec["warning"]
    bad, _ = L.box_json_identity(old, None, "donor@abc", {3: 32, 5: 64})
    assert bad and "outside" in bad[0]
    src = (ROOT / "scripts" / "amoeba_bridge_lab.py").read_text()
    assert 'box_info = {"donor_checkpoint": Dn["ck"], "host_checkpoint": H["ck"]' in src
    assert '"box_identity": box_identity' in src


# ----------------------------------------------------------------------------- AML2-8: signed answers

@pytest.mark.parametrize("text,exp,ok", [
    ("-36", "36", False), ("The result is -1,250.", "1250", False), ("−36", "36", False),
    ("The answer is **-36**", "36", False), ("1362", "36", False), ("360", "36", False), ("36.5", "36", False),
    ("0.36", "36", False),
    ("$36", "36", True), ("So she has 36.", "36", True), ("36, then", "36", True), ("3 times 12 is **36**", "36", True),
    ("50-36 = 14, so 14 + 22 = 36", "36", True), ("86 - 50 = 36", "36", True), ("1,250 apples", "1250", True),
    ("-36", "-36", True), ("The value is -36.", "-36", True), ("36", "-36", False),
])
def test_whole_number_present_keeps_the_sign(text, exp, ok):
    import amoeba_det_calc_cpu as DC
    assert DC.whole_number_present(text, exp) is ok


def test_subtraction_echo_still_shows_the_operand():
    import amoeba_det_calc_cpu as DC
    assert DC.whole_number_present("50-36", "36")  # a digit before the minus: an operator, not a sign


# ----------------------------------------------------------------------------- AML2-9: wiki crawl resume

GRAPH = {  # category -> (articles, subcategories); "Shared article" sits in all three
    "Root": (["Alpha", "Shared article", "Beta"], ["Sub one", "Sub two"]),
    "Sub one": (["Shared article", "Gamma", "Delta"], []),
    "Sub two": (["Shared article", "Epsilon", "Zeta"], []),
}


def _fake_members(calls):
    def members(cat, cont=None):
        calls.append(cat)
        arts, subs = GRAPH[cat]
        return {"query": {"categorymembers": [{"ns": 0, "title": t} for t in arts]
                          + [{"ns": 14, "title": f"Category:{s}"} for s in subs]}}
    return members


@pytest.fixture
def crawl_mod(monkeypatch):
    import amoeba_category_wikicrawl as WC
    calls = []
    monkeypatch.setattr(WC, "members", _fake_members(calls))
    monkeypatch.setattr(WC, "PACE", 0)
    monkeypatch.setattr(WC, "ROOTS", {"node.a": ["Root"], "node.b": ["Sub two"]})

    def no_network(*a, **k):
        raise AssertionError("network access in a test")
    monkeypatch.setattr(WC.urllib.request, "urlopen", no_network)
    return WC, calls


def test_crawl_cap_counts_distinct_titles(crawl_mod):
    WC, _ = crawl_mod
    # 6 raw members after Root + Sub one, but only 5 distinct: the crawl must go on to Sub two
    got = WC.crawl(["Root"], 1, 6)
    assert got == ["Alpha", "Shared article", "Beta", "Gamma", "Delta", "Epsilon"]
    assert len(set(got)) == 6


def test_crawl_resume_is_keyed_on_settings_and_writes_atomically(crawl_mod, tmp_path, capsys):
    WC, calls = crawl_mod
    out = tmp_path / "titles.json"
    WC.main(["--out", str(out), "--depth", "1", "--per-node", "50"])
    res = json.loads(out.read_text())
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert set(res) == {"node.a", "node.b"} and len(res["node.a"]) == 7
    assert meta["nodes"]["node.a"] == {"depth": 1, "per_node": 50, "roots": ["Root"],
                                       "fetched": meta["nodes"]["node.a"]["fetched"], "titles": 7}
    assert not list(tmp_path.glob("*.tmp"))
    # same settings: nothing is fetched again
    calls.clear()
    WC.main(["--out", str(out), "--depth", "1", "--per-node", "50"])
    assert calls == []
    # other settings: refused, naming the fields, and both files are left as they were
    before = (out.read_bytes(), out.with_suffix(".meta.json").read_bytes())
    with pytest.raises(SystemExit) as ei:
        WC.main(["--out", str(out), "--depth", "2", "--per-node", "50"])
    assert "depth 1 != 2" in str(ei.value) and "node.a" in str(ei.value)
    assert calls == [] and before == (out.read_bytes(), out.with_suffix(".meta.json").read_bytes())
    # edited roots are a setting too
    WC.ROOTS["node.b"] = ["Sub one"]
    with pytest.raises(SystemExit, match="node.b: roots"):
        WC.main(["--out", str(out), "--depth", "1", "--per-node", "50"])


def test_crawl_resume_of_a_file_without_node_records_warns_and_records_it(crawl_mod, tmp_path, capsys):
    WC, calls = crawl_mod
    out = tmp_path / "titles.json"
    out.write_text(json.dumps({"node.a": ["Alpha", "Beta"]}))  # an older result: no per-node record
    out.with_suffix(".meta.json").write_text(json.dumps({"depth": 2, "per_node": 12000, "fetched": "2026-11-18"}))
    WC.main(["--out", str(out), "--depth", "1", "--per-node", "50"])
    assert "WARNING node.a" in capsys.readouterr().out
    res = json.loads(out.read_text())
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert res["node.a"] == ["Alpha", "Beta"] and "Root" not in calls  # kept, not re-crawled
    assert meta["nodes"]["node.a"]["unrecorded"] is True
    assert meta["nodes"]["node.a"]["legacy_meta"] == {"depth": 2, "per_node": 12000, "fetched": "2026-11-18"}
    assert meta["nodes"]["node.b"]["depth"] == 1
