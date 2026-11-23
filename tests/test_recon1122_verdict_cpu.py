"""Exercise the window verdicts with synthetic receipts and a stopped server."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gemma4_d512_decode_ab as gemma
import inkling2_copy_draft_window as ink


def arm(name, tokens=(1, 2, 3)):
    common = {"token_ids": list(tokens), "decode_tok_s": 1, "oracle": {"pass": True}}
    cell = {"cell": 4096, "prompt_sha256": "abc",
            "counter_delta": int(name == "maskless_tk2"),
            "maskless_delta": int(name != "stock"),
            "cold": {**common, "cached_tokens": 0}, "warm": {**common, "cached_tokens": 4096}}
    return {"name": name, "passed_run": True, "checkpoint": "fixture@abc", "cells": [cell]}


def arms(**tokens):
    return [arm(n, tokens.get(n, (1, 2, 3))) for n in ("stock", "maskless", "maskless_tk2")]


def windows(name, index, table):
    for shape in ("cold", "warm"):
        rows = [None] * 3
        rows[index] = table
        gemma.WINDOWS[(name, 4096, shape)] = rows


def test_gemma_verdict_requires_evidence_for_each_cell_and_replay():
    gemma.WINDOWS.clear()
    a = arms()
    assert gemma.verdict(a)["passed"]
    a[2]["cells"][0]["warm"]["cached_tokens"] = 0
    assert not gemma.verdict(a)["passed"]
    a = arms()
    for x in a:
        x["cells"][0]["cold"]["token_ids"] = [11]
    assert not gemma.verdict(a)["passed"]
    a = arms()
    a[1]["cells"].clear()
    assert not gemma.verdict(a)["passed"]
    assert not gemma.verdict(arms()[:2])["passed"]


def test_gemma_verdict_engagement_counters():
    gemma.WINDOWS.clear()
    for name, key, value in (("stock", "maskless_delta", 1), ("maskless", "maskless_delta", 0),
                             ("maskless", "counter_delta", 1), ("maskless_tk2", "counter_delta", 0)):
        a = arms()
        a[[x["name"] for x in a].index(name)]["cells"][0][key] = value
        assert not gemma.verdict(a)["passed"], (name, key)


def test_gemma_tk2_must_equal_maskless_exactly():
    gemma.WINDOWS.clear()
    v = gemma.verdict(arms(maskless_tk2=(1, 9, 3)))
    assert not v["passed"] and any("kernel defect" in f for f in v["failures"])


def test_gemma_maskless_vs_stock_near_tie_passes_high_margin_fails():
    gemma.WINDOWS.clear()
    windows("stock", 1, {2: -0.60, 9: -0.80, 4: -3.0})
    windows("maskless", 1, {9: -0.65, 2: -0.75, 4: -3.0})
    v = gemma.verdict(arms(maskless=(1, 9, 3), maskless_tk2=(1, 9, 3)))
    assert v["passed"], v["failures"]
    assert v["cells"][0]["cold_maskless_vs_stock"]["verdict"] == "near_tie_equivalent"
    gemma.WINDOWS.clear()
    windows("stock", 1, {2: -0.01, 9: -5.0, 4: -6.0})
    windows("maskless", 1, {9: -0.01, 2: -5.0, 4: -6.0})
    v = gemma.verdict(arms(maskless=(1, 9, 3), maskless_tk2=(1, 9, 3)))
    assert not v["passed"] and v["cells"][0]["cold_maskless_vs_stock"]["verdict"] == "high_margin"
    gemma.WINDOWS.clear()
    v = gemma.verdict(arms(maskless=(1, 9, 3), maskless_tk2=(1, 9, 3)))
    assert not v["passed"] and v["cells"][0]["cold_maskless_vs_stock"]["verdict"] == "unclassified"


@pytest.mark.parametrize(("gsm_code", "questions", "rounds", "expected"), [
    (7, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0),
])
def test_copy_window_preserves_gsm8k_failure(tmp_path, monkeypatch, gsm_code, questions, rounds, expected):
    from mlx2.adapters import registry

    monkeypatch.setattr(registry, "inspect_model", lambda path: SimpleNamespace(artifact_fingerprint="fake"))
    monkeypatch.setattr(ink, "revision", lambda path: {"head": "fixture"})
    monkeypatch.setattr(ink.subprocess, "check_output", lambda *a, **kw: "0")
    monkeypatch.setattr(ink, "start_server", lambda a, arm, out, token: (
        SimpleNamespace(poll=lambda: 0), (out / "server.log").open("w"), []))
    monkeypatch.setattr(ink, "wait_ready", lambda *a: {"model": "served"})
    monkeypatch.setattr(ink.Client, "call", lambda self, path: "" if path == "/metrics" else {})
    monkeypatch.setattr(ink, "rewrite_cell", lambda client, name, cap: [{"decode_tps": 0}])

    def run(command, **kwargs):
        directory = Path(command[command.index("--output") + 1])
        directory.mkdir()
        (directory / "receipt.json").write_text(json.dumps({"arms": [{"summary": {"questions": questions, "acceptance": {"verify_rounds": rounds}}}]}))
        return SimpleNamespace(returncode=gsm_code, stdout="", stderr="")

    monkeypatch.setattr(ink.subprocess, "run", run)
    out = tmp_path / "out"
    assert ink.main(["--arms", "dspark_copy", "--model", "fixture", "--output", str(out),
                     "--num-questions", "1", "--port", "0"]) == expected
    assert json.loads((out / "receipt.json").read_text())["arms"][0]["gsm8k_returncode"] == gsm_code


def test_rewrite_divergence_probe_classification():
    import inkling2_rewrite_divergence_probe as probe

    content = [{"id": 5, "logprob": -0.1, "top_logprobs": [{"id": 5, "logprob": -0.1},
                                                           {"id": 6, "logprob": -2.5}]}]
    assert probe.window_at(content, 0) == {5: -0.1, 6: -2.5}
    near = probe.classify(7, {7: -0.60, 8: -0.80}, 8, {8: -0.65, 7: -0.75}, (True, True))
    assert near["verdict"] == "near_tie_equivalent"
    assert probe.classify(7, {7: -0.60, 8: -0.80}, 8, {8: -0.65, 7: -0.75}, (True, False))["verdict"] \
        == "near_tie_oracle_failed"
    assert probe.classify(7, {7: -0.01, 8: -5.0}, 8, {8: -0.01, 7: -5.0}, (True, True))["verdict"] \
        == "high_margin"
    assert probe.classify(7, {7: -0.1, 9: -0.2}, 8, {8: -0.1, 7: -0.3}, (True, True))["verdict"] \
        == "unclassified"
    assert probe.classify(7, None, 8, {8: -0.1}, (True, True))["verdict"] == "unclassified"
