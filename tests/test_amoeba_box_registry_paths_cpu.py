"""Box paths in the amoeba box registry: add, status, select (with approval), and fail-closed resolution."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("amoeba_box_registry", ROOT / "scripts" / "amoeba_box_registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REG", tmp_path)
    rec = {"schema": mod.SCHEMA, "box_id": "b1", "kind": "box", "tags": ["arith"], "family": "arith",
           "origin": {"units_count": 1, "units": [], "heads": [], "donor": {}}, "versions": [], "changelog": []}
    (tmp_path / "b1.json").write_text(json.dumps(rec))
    return mod, tmp_path


def ns(**kw):
    base = dict(box="b1", path=None, donor_effort=None, donor="Qwen3.8-27B@fb827b65", traces="", tensor="",
                params="", note="", implemented=None, qualified=None, observed_used=None, evidence="e",
                approved_by="", checkpoint="host@abc", metrics="{}")
    base.update(kw)
    return argparse.Namespace(**base)


def test_paths_add_select_resolve(reg):
    mod, tmp = reg
    payload = tmp / "xhigh.pt"
    payload.write_bytes(b"x")
    mod.cmd_path_add(ns(path="xhigh", donor_effort="xhigh", tensor=str(payload)))
    mod.cmd_path_add(ns(path="low", donor_effort="low"))  # planned only
    rec = mod.load_box("b1")
    assert rec["paths"]["xhigh"]["status"]["implemented"] and not rec["paths"]["low"]["status"]["implemented"]
    with pytest.raises(LookupError):
        mod.resolve_path(rec)  # paths exist, none selected: refuse
    with pytest.raises(LookupError):
        mod.resolve_path(rec, "low")  # not implemented
    with pytest.raises(SystemExit):
        mod.cmd_path_select(ns(path="xhigh"))  # no approver
    with pytest.raises(SystemExit):
        mod.cmd_path_select(ns(path="low", approved_by="Pierre"))  # planned path
    mod.cmd_path_select(ns(path="xhigh", approved_by="Pierre"))
    rec = mod.load_box("b1")
    assert mod.resolve_path(rec)[0] == "xhigh"
    assert [e["action"] for e in rec["changelog"]].count("path selected as box default") == 1
    mod.cmd_path_evidence(ns(path="xhigh", metrics='{"gsm8k": 0.4}'))
    assert mod.load_box("b1")["paths"]["xhigh"]["ab"][-1]["metrics"] == {"gsm8k": 0.4}


def test_box_without_paths_resolves_to_none(reg):
    mod, _ = reg
    assert mod.resolve_path(mod.load_box("b1")) == (None, None)


def test_planned_path_cannot_be_marked_implemented(reg):
    mod, _ = reg
    mod.cmd_path_add(ns(path="low", donor_effort="low"))
    with pytest.raises(SystemExit):
        mod.cmd_path_status(ns(path="low", implemented="yes"))
