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
    rows = [{"status": "complete", "sha256": c * 64, "reason": None} for c in "ab"]
    lane = {"tokens": [1, 2], **Q.lane_row_evidence(rows, [1, 2], 2), "covered_tokens": 5,
            "final_state": {"status": "complete", "sha256": "c" * 64, "reason": None}}
    other = dict(lane, covered_tokens=6, final_state={"status": "complete", "sha256": "d" * 64, "reason": None})
    result = Q.compare("b", "a", [0], {"a": {0: lane}, "b": {0: other}}, continuation=False, logprob_rows=2)
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


# ---- repairs after parent review (bounds, disabled continuation) ----

@pytest.mark.parametrize("argv,message", [
    (["--time-limit-s", "nan"], "finite"),
    (["--time-limit-s", "inf"], "finite"),
    (["--continuation-tokens", "513"], "0..512"),
    (["--logprob-rows", "513"], "0..512"),
    (["--prefill-step", "16385"], "--prefill-step 1..16384"),
    (["--prefill-step", "0"], "--prefill-step 1..16384"),
])
def test_nonfinite_or_unbounded_settings_are_refused(argv, message, capsys):
    with pytest.raises(SystemExit):
        Q.resolve_args(Q.build_parser(), ["--tiny", *argv, "--out", "/dev/null"])
    assert message in capsys.readouterr().err


def test_zero_rows_and_continuation_are_explicitly_unavailable_not_exact():
    record = Q.run_all(_args("--logprob-rows", "0", "--continuation-tokens", "0"))
    assert record["verdict"] != "pass"
    lane = record["results"]["pld_batched"]["0"]
    assert lane["continuation"] == {"status": "unavailable", "reason": "disabled (--continuation-tokens 0)"}
    assert lane["logprob_rows_status"] == ["unavailable"]
    primary = [c for c in record["comparisons"] if c["reference"] == "ordinary_b1" and c["arm"].startswith("pld")]
    assert all(not c["bits_exact"] and any("unavailable" in x for x in c["incomparable"]) for c in primary)


def test_no_final_responses_survive_in_the_record(tiny_record):
    for arm in tiny_record["results"].values():
        assert all("_final" not in lane for lane in arm.values())


# ---- requested vs observed compute width, per-lane policies ----

@pytest.mark.parametrize("lanes", [2, 4])
def test_tiny_b2_and_b4_fixtures_engage_their_requested_width(lanes):
    record = Q.run_all(_args("--lanes", str(lanes)))
    assert record["widths"]["requested"] == lanes
    assert record["widths"]["observed_max"]["pld_batched"] == lanes
    assert record["coverage"]["requested_width_engaged"] and all(record["coverage"].values())
    assert record["verdict"] in ("pass", "token_exact_bits_diverge")


def test_default_tiny_fixture_is_reported_as_width_three(tiny_record):
    assert tiny_record["widths"]["requested"] == 3
    assert tiny_record["widths"]["observed_max"]["pld_batched"] == 3
    assert len(tiny_record["lanes"]) == 3


def _fixture(monkeypatch, prompts, caps):
    monkeypatch.setattr(Q, "tiny_prompts", lambda lanes: (prompts, caps))


def test_b4_cohort_that_only_computes_at_width_three_is_refused(monkeypatch):
    repeat, looping = [3, 4, 5, 6, 7] * 4 + [3, 4], [9, 10, 11, 12] * 5 + [9]
    distinct, alt = [20, 21, 22, 23, 24, 25, 26], [30, 31, 32, 33, 34, 35] * 3 + [30]
    _fixture(monkeypatch, [repeat, looping, distinct, alt], [24, 28, 6, 20])
    record = Q.run_all(_args("--lanes", "4"))
    assert record["widths"]["requested"] == 4
    assert record["widths"]["observed_max"]["pld_batched"] == 3
    assert record["coverage"]["batched_width_ge2"]  # the old generic signal still passes
    assert not record["coverage"]["requested_width_engaged"]
    assert record["verdict"] == "coverage_refused"


def test_the_old_width_three_fixture_with_cap_two_is_refused(monkeypatch):
    _fixture(monkeypatch, [[3, 4, 5, 6, 7] * 4 + [3, 4], [9, 10, 11, 12] * 5 + [9],
                           [20, 21, 22, 23, 24, 25, 26]], [24, 28, 2])
    record = Q.run_all(_args())
    assert record["widths"]["observed_max"]["pld_batched"] == 2
    assert record["verdict"] == "coverage_refused"


def test_ordinary_routes_do_not_claim_a_compute_width(tiny_record):
    for arm in ("ordinary_b1", "ordinary_bN"):
        for lane in tiny_record["results"][arm].values():
            assert lane["execution_widths"] == "not reported by the ordinary route"


def test_lane_policies_reach_pld_arms_only(monkeypatch):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    seen = {"pld": [], "ordinary": []}
    pld_insert, ord_insert = PromptLookupBatchGenerator.insert, BatchGenerator.insert

    def spy_pld(self, prompts, **kwargs):
        seen["pld"].append(kwargs.get("prompt_lookup_configs"))
        return pld_insert(self, prompts, **kwargs)

    def spy_ord(self, prompts, **kwargs):
        seen["ordinary"].append("prompt_lookup_configs" in kwargs)
        return ord_insert(self, prompts, **kwargs)

    monkeypatch.setattr(PromptLookupBatchGenerator, "insert", spy_pld)
    monkeypatch.setattr(BatchGenerator, "insert", spy_ord)
    policies = [{"num_draft": 2}, {"deferred_admission": True}]
    record = Q.run_all(_args("--lanes", "2", "--lane-policies", json.dumps(policies)))
    assert record["lane_policies"] == policies
    assert seen["pld"] and all(configs == policies for configs in seen["pld"])
    assert seen["ordinary"] and not any(seen["ordinary"])
    # The deferred (ordinary-admission) lane really proposed nothing.
    assert record["results"]["pld_batched"]["1"]["rounds"]["proposed"] == 0
    plain = Q.run_all(_args("--lanes", "2"))
    for i in ("0", "1"):
        assert record["results"]["ordinary_b1"][i]["tokens"] == plain["results"]["ordinary_b1"][i]["tokens"]


@pytest.mark.parametrize("policies,message", [
    ([{"num_draft": 2}], "one object per lane"),
    ({"num_draft": 2}, "one object per lane"),
    ([{"num_draft": 2}, 5], "lane 1 policy must be an object"),
    ([{"num_draft": 2}, {"batched_verify": False}], "batched_verify is fixed per arm"),
    ([{"num_draft": 2}, {"proposer": "fake"}], "unknown policy keys"),
    ([{"num_draft": 0}, {}], "lane 0 policy"),
])
def test_invalid_lane_policies_fail_closed(policies, message):
    driver = Q.Driver(_args("--lanes", "2", "--lane-policies", json.dumps(policies)))
    with pytest.raises(SystemExit, match=message):
        driver.check_geometry()
