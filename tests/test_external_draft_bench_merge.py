"""CPU tests for the rm06 external-draft bench summary and split-run merge."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_external_draft_acceptance.py"


def _load():
    spec = importlib.util.spec_from_file_location("bench_external_draft_acceptance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm(kind, width, tps, repeat, depth=0, ratio=None):
    arm = {"kind": kind, "num_draft": depth, "width": width, "temperature": 0.0, "repeat": repeat,
           "decode_tok_s": tps, "mechanism": {}}
    if kind == "external":
        arm["mechanism"] = {"external_rounds": 5, "proposed_tokens": 15, "mean_acceptance_length": 2.5}
        arm["greedy_prefix_ratio"] = ratio
        arm["greedy_parity"] = 1
    return arm


def _split(module, tmp_path, name, repeat, ext_tps):
    arms = [_arm("ordinary", 1, 50.0, repeat), _arm("external", 1, ext_tps, repeat, 3, 1.0),
            _arm("ordinary", 4, 120.0, repeat), _arm("external", 4, 130.0, repeat, 3, 0.95)]
    data = {"schema": module.SCHEMA, "plan": {"family": "north", "order": []}, "arms": arms, "refused": []}
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_merge_pools_arms_across_splits(tmp_path):
    module = _load()
    a = _split(module, tmp_path, "a.json", 0, 70.0)
    b = _split(module, tmp_path, "b.json", 1, 60.0)
    out = tmp_path / "merged.json"
    assert module.main(["--merge", str(a), str(b), "--out", str(out)]) == 0
    merged = json.loads(out.read_text())
    assert len(merged["arms"]) == 8
    b1 = next(r for r in merged["summary"] if r["width"] == 1)
    assert b1["external_tok_s"] == pytest.approx(65.0)
    assert b1["speedup"] == pytest.approx(1.3)
    assert merged["verdict"]["go"] is True


def test_merge_refused_arm_blocks_go(tmp_path):
    module = _load()
    a = _split(module, tmp_path, "a.json", 0, 70.0)
    data = json.loads(a.read_text())
    data["refused"] = [{"kind": "external", "reason": "mechanism counter is zero"}]
    a.write_text(json.dumps(data))
    out = tmp_path / "merged.json"
    module.main(["--merge", str(a), "--out", str(out)])
    assert json.loads(out.read_text())["verdict"]["go"] is False


def test_merge_refuses_mixed_families(tmp_path):
    module = _load()
    a = _split(module, tmp_path, "a.json", 0, 70.0)
    b = _split(module, tmp_path, "b.json", 1, 70.0)
    data = json.loads(b.read_text())
    data["plan"]["family"] = "laguna"
    b.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        module.main(["--merge", str(a), str(b)])


def test_repeat_offset_labels_split_plan():
    module = _load()
    args = module.build_parser().parse_args(
        ["--family", "north", "--target", "t", "--draft", "d", "--num-draft", "3", "--repeats", "1",
         "--repeat-offset", "2", "--temperatures", "0", "--widths", "1"])
    assert {row[-1] for row in module.plan(args)["order"]} == {2}
