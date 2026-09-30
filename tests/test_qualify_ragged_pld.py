"""CPU tests for scripts/qualify_ragged_pld.py (ragged batched PLD driver).

Only the deterministic tiny CPU model runs here. The tests pin the gates:
declared reference, token versus storage-bit verdicts, coverage refusal,
membership survivors, bounded polling and argument refusals. Real-model
runs need --i-own-the-gpu and belong to the GPU owner.
"""

import json
import subprocess
import sys

import pytest

from scripts import qualify_ragged_pld as Q


def _args(*extra):
    return Q.resolve_args(Q.build_parser(), ["--tiny", "--out", "/dev/null", *extra])


@pytest.fixture(scope="module")
def tiny_record():
    return Q.run_all(_args())


def test_help_is_import_safe():
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, scripts.qualify_ragged_pld; print('mlx.core' in sys.modules)"],
        capture_output=True, text=True, env={"PYTHONPATH": "src:."},
    )
    assert probe.returncode == 0 and probe.stdout.strip() == "False", probe.stderr
    helped = subprocess.run([sys.executable, "scripts/qualify_ragged_pld.py", "--help"],
                            capture_output=True, text=True, env={"PYTHONPATH": "src"})
    assert helped.returncode == 0 and "--i-own-the-gpu" in helped.stdout


def test_tiny_run_reaches_full_coverage_with_a_declared_reference(tiny_record):
    record = tiny_record
    assert record["schema"] == Q.SCHEMA and record["primary_reference"] == "ordinary_b1"
    assert "direct-model" in record["scope"] and "not HTTP" in record["scope"]
    assert all(record["coverage"].values()), record["coverage"]
    assert record["refusals"] == []
    assert record["removed"] == {"lane": 0, "after_tokens": 4}
    kinds = {(c["arm"], c["reference"]): c for c in record["comparisons"]}
    for arm in ("pld_per_lane", "pld_batched"):
        assert kinds[(arm, "ordinary_b1")]["tokens_exact"]
        assert (arm, "ordinary_bN") in kinds  # secondary reference is reported, not swapped in
    assert kinds[("ordinary_bN", "ordinary_b1")]["kind"] == "ordinary geometry (B1 vs BN)"
    assert kinds[("pld_removal", "ordinary_b1")]["tokens_exact"]


def test_bit_divergence_is_never_relabeled_exact(tiny_record):
    record = tiny_record
    primary = [c for c in record["comparisons"]
               if c["reference"] == "ordinary_b1" and c["arm"].startswith("pld")]
    if all(c["bits_exact"] for c in primary):
        assert record["verdict"] == "pass"
    else:
        # The CPU tiny model: verify blocks and batch width change float bits.
        assert record["verdict"] == "token_exact_bits_diverge"
        assert any(c["bit_differences"] or c["incomparable"] for c in primary)
    for c in record["comparisons"]:
        assert c["bits_exact"] == (c["tokens_exact"] and not c["bit_differences"] and not c["incomparable"])


def test_different_covered_lengths_are_incomparable_not_equal():
    lane = {"tokens": [1, 2], "logprob_rows": ["a"], "covered_tokens": 5,
            "final_state": {"status": "complete", "sha256": "x"}}
    other = dict(lane, covered_tokens=6, final_state={"status": "complete", "sha256": "y"})
    result = Q.compare("b", "a", [0], {"a": {0: lane}, "b": {0: other}}, continuation=False)
    assert result["tokens_exact"] and not result["bits_exact"]
    assert result["bit_differences"] == []
    assert result["incomparable"] == ["lane 0: final caches cover 6 vs 5 tokens"]


def test_missing_logprob_rows_are_incomparable():
    lane = {"tokens": [1], "logprob_rows": [], "covered_tokens": 2,
            "final_state": {"status": "complete", "sha256": "x"}}
    result = Q.compare("b", "a", [0], {"a": {0: lane}, "b": {0: dict(lane)}}, continuation=False)
    assert not result["bits_exact"] and "logprob rows unavailable" in result["incomparable"][0]


def _perturb(monkeypatch, arm, lane, edit):
    run = Q.Driver.run

    def perturbed(self, name, lanes, **kwargs):
        out, stats, failures, removed = run(self, name, lanes, **kwargs)
        if name == arm and lane in out and kwargs.get("caches") is None:
            edit(out[lane])
        return out, stats, failures, removed

    monkeypatch.setattr(Q.Driver, "run", perturbed)


def test_pld_token_divergence_is_a_counterexample(monkeypatch):
    _perturb(monkeypatch, "pld_batched", 1, lambda r: r["tokens"].__setitem__(-1, r["tokens"][-1] ^ 1))
    record = Q.run_all(_args())
    assert record["verdict"] == "counterexample"
    bad = [c for c in record["comparisons"] if c["arm"] == "pld_batched" and c["reference"] == "ordinary_b1"]
    assert bad[0]["token_differences"] == ["lane 1: tokens differ at 27"]


def test_membership_survivor_divergence_is_a_counterexample(monkeypatch):
    _perturb(monkeypatch, "pld_removal", 1, lambda r: r["tokens"].__setitem__(0, r["tokens"][0] ^ 1))
    record = Q.run_all(_args())
    assert record["verdict"] == "counterexample"


def test_no_proposals_is_coverage_refused_not_pass():
    record = Q.run_all(_args("--pld-policy", json.dumps({"deferred_admission": True})))
    assert record["verdict"] == "coverage_refused"
    assert not record["coverage"]["proposals"] and not record["coverage"]["rejected_suffix"]


def test_bounded_polling_refuses(monkeypatch):
    record = Q.run_all(_args("--time-limit-s", "1e-9"))
    assert record["verdict"] == "refused"
    assert any("bounded" in r for r in record["refusals"])


def test_early_stop_is_refused(monkeypatch):
    _perturb(monkeypatch, "pld_per_lane", 0, lambda r: r.update(finish_reason=None))
    record = Q.run_all(_args())
    assert record["verdict"] == "refused"
    assert any("pld_per_lane lane 0: early stop" in r for r in record["refusals"])


@pytest.mark.parametrize("argv,message", [
    (["--model", "m"], "--i-own-the-gpu"),
    (["--i-own-the-gpu"], "--model is required"),
    (["--tiny", "--i-own-the-gpu"], "random CPU model"),
    (["--tiny", "--pld-policy", '{"batched_verify": false}'], "set per arm"),
    (["--tiny", "--lanes", "5"], "invalid bounds"),
])
def test_argument_refusals(argv, message, capsys):
    with pytest.raises(SystemExit):
        Q.resolve_args(Q.build_parser(), [*argv, "--out", "/dev/null"])
    assert message in capsys.readouterr().err


def test_unsafe_geometry_is_refused():
    driver = Q.Driver(_args())
    driver.prompts, driver.caps = [[1, 2]], [3]
    with pytest.raises(SystemExit, match="2..4 lanes"):
        driver.check_geometry()
    driver.prompts, driver.caps = [[1, 2], [3, 4]], [3, 600]
    with pytest.raises(SystemExit, match="cap 1..512"):
        driver.check_geometry()


def test_main_writes_a_json_record(tmp_path):
    out = tmp_path / "pld.json"
    code = Q.main(["--tiny", "--out", str(out)])
    record = json.loads(out.read_text())
    assert code == (0 if record["verdict"] == "pass" else 1)
    assert record["identity"]["files"] and record["identity"]["mlx"]["version"]
    lane0 = record["results"]["pld_batched"]["0"]
    assert lane0["final_state"]["status"] == "complete" and lane0["continuation"]["status"] == "complete"
