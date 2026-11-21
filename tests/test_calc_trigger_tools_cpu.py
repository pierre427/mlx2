"""Calculator-trigger GSM8K tools: manifest derivation, memwatch wrapper refusals, receipt analysis."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime import amoeba_det_tiles as D

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


manifests = _load("calc_trigger_manifests", "scripts/research/calc_trigger_manifests.py")
analyze = _load("calc_trigger_analyze", "scripts/research/calc_trigger_analyze.py")


def _source(tmp_path, **tile):
    tiles = [{"id": "calculator", "kind": "calculator", "write_mode": "force", "read_layer": 1, "write_layer": 1,
              "gate": {"bias": 0.0, "threshold": 0.9}, **tile}]
    return D.write_manifest(tmp_path / "src", source={"artifact": "a", "fingerprint": "f"}, layers=2, tiles=tiles,
                            tensors={"calculator.probe_w": mx.zeros((4,))})


@pytest.mark.parametrize("parser,want", [("v1", D.PARSER_ID), ("v2", D.PARSER_V2), ("marker", D.PARSER_MARKER)])
def test_derive_changes_only_the_trigger(tmp_path, parser, want):
    src = _source(tmp_path, parser=D.PARSER_V4, trigger_prefilter="gpu", arithmetic_backend="gpu",
                  complete_literal_requests=True, context_tokens=256)
    out = manifests.derive(src, parser, tmp_path / parser)
    m = D.load_manifest(out)
    t = m["tiles"][0]
    assert t["parser"] == want and "trigger_prefilter" not in t and "complete_literal_requests" not in t
    assert m["derived_from"]["manifest_sha256"] == D.load_manifest(src)["_digest"]
    assert (t["gate"].get("kind") == D.MARKER_GATE) == (parser == "marker")
    assert (out.parent / "tiles.safetensors").read_bytes() == (src.parent / "tiles.safetensors").read_bytes()
    with pytest.raises(SystemExit, match="nonempty"):
        manifests.derive(src, parser, tmp_path / parser)


def test_derive_v4_requires_a_v4_source(tmp_path):
    with pytest.raises(SystemExit, match="v4"):
        manifests.derive(_source(tmp_path), "v4", tmp_path / "out")


def test_memwatched_python_accepts_only_module_runs():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/research/memwatched_python.py"), "script.py"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "usage" in r.stderr


def test_slips_find_wrong_steps_only():
    assert analyze.slips("So 48/2 = <<48/2=24>>24 and 12 * 3 = 36.") == []
    s = analyze.slips("She has <<48/2=25>>25, then 12 * 3 = 35 left.")
    assert [x["exact"] for x in s] == ["24", "36"]


def test_analysis_classes_and_pairing():
    off = {"name": "off", "passed": True, "rows": [
        {"test_index": 0, "correct": True, "text": "#### 1"},
        {"test_index": 1, "correct": False, "finish_reason": "stop", "text": "So 12 * 3 = 35.\n#### 35"},
        {"test_index": 2, "correct": False, "finish_reason": "length", "text": "..."},
        {"test_index": 3, "correct": False, "finish_reason": "stop", "text": "It is 7.\n#### 7"}]}
    marker = {"name": "marker", "passed": True, "rows": [
        {"test_index": 0, "correct": False, "finish_reason": "stop", "text": "<<6*7=42>>42\n#### 42", "markers": 1,
         "det_tiles": {"plans": 1, "marker_open": 1, "parse_refused": {"no_operation": 2}}},
        {"test_index": 1, "correct": True, "text": "#### 36", "markers": 1, "det_tiles": {"plans": 1, "marker_open": 1}},
        {"test_index": 2, "correct": True, "text": "#### 2", "det_tiles": {"plans": 0}},
        {"test_index": 3, "correct": True, "text": "#### 3", "det_tiles": {"plans": 0}}]}
    out = analyze.analyze({"arms": [off, marker]})
    assert out["arms"]["off"]["wrong_classes"] == {"arithmetic_slip": 1, "truncated": 1, "other": 1}
    a = out["arms"]["marker"]
    assert a["wrong_classes"] == {"fired": 1} and a["fires"] == 2 and a["questions_fired"] == 2
    assert a["questions_with_markers"] == 2 and a["parse_refused"] == {"no_operation": 2}
    assert a["vs_reference"]["helped"] == [1, 2, 3] and a["vs_reference"]["hurt"] == [0]


def test_tiled_host_route_receipt_carries_the_request_calculator_counters(tmp_path):
    from types import SimpleNamespace
    from mlx2.serving import ServingEngine

    path = _source(tmp_path)
    ts = D.build(path, lambda x: x, source_fingerprint="f", layers_count=2)
    request = {}
    D.processors_for(ts, SimpleNamespace(encode=lambda s, **k: [], decode=lambda i: "", __len__=lambda: 8),
                     request=request)
    adapter = SimpleNamespace(amoeba_blocks=None, amoeba_tiles=SimpleNamespace(identity={"manifest_sha256": "h"}),
                              amoeba_det_tiles=ts)
    receipt = ServingEngine._weight_space_route_receipt(SimpleNamespace(adapter=adapter),
                                                        SimpleNamespace(request=request), {"route": "ordinary"})
    assert receipt["amoeba_tile_set"] == "h"
    assert receipt["amoeba_det_tiles_request"]["calculator"]["plans"] == 0
