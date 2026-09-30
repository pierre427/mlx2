"""CPU tests for the paired direct-model A/B harness (scripts/paired_direct_ab.py).

Only ``--tiny`` random CPU models run here; real runs need ``--i-own-the-gpu``
and belong to the GPU owner. The tests pin the gates: engagement per arm,
the reclamation-boundary length refusal, the counterexample verdict, the
harness-local reclaim control and the argument refusals.
"""

import json

import pytest

from scripts import paired_direct_ab as H


def _args(*extra):
    return H.resolve_args(H.build_parser(), ["--tiny", "--out", "/dev/null", "--warmups", "0",
                                              "--pairs", "1", *extra])


def test_tiny_qsdpa_pairs_engage_and_match():
    record = H.run_cohort(_args("--mechanism", "qsdpa-tiling"))
    assert record["verdict"] == "pass", record["refusals"] + record["mismatches"]
    assert "direct-model" in record["scope"] and "not HTTP" in record["scope"]
    on = [r for r in record["runs"] if r["arm"] == "on"]
    off = [r for r in record["runs"] if r["arm"] == "off"]
    assert on[0]["counters"]["composed_tiled_calls"] > 0
    assert on[0]["counters"]["composed_tiles"] >= 2
    assert off[0]["counters"]["composed_tiled_calls"] == 0
    assert on[0]["token_sha256"] == off[0]["token_sha256"]
    assert on[0]["logprob_row_sha256"] and on[0]["logprob_row_sha256"] == off[0]["logprob_row_sha256"]
    assert record["protocol"]["kv_bits"] == 8  # one codec, both arms
    for key in ("active_bytes", "cache_bytes", "peak_bytes", "ttft_s", "decode_s"):
        assert key in on[0]
    assert record["identity"]["mlx"]["version"] and record["identity"]["source"]["files"]


def test_tiny_qsdpa_budget_is_restored():
    from mlx2.runtime.models import base

    before = base._QSDPA_SCORES_BUDGET
    H.run_cohort(_args("--mechanism", "qsdpa-tiling"))
    assert base._QSDPA_SCORES_BUDGET == before


def test_qsdpa_refuses_when_tiling_never_engages():
    record = H.run_cohort(_args("--mechanism", "qsdpa-tiling", "--budget-bytes", str(1 << 40)))
    assert record["verdict"] == "refused"
    assert any("tiling not engaged on the on arm" in r for r in record["refusals"])


def test_tiny_reclaim_pairs_engage_and_control_is_harness_local():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    product = ExternalDraftBatchGenerator._reclaim_after_emission
    record = H.run_cohort(_args("--mechanism", "external-reclaim"))
    assert record["verdict"] == "pass", record["refusals"] + record["mismatches"]
    by_arm = {r["arm"]: r for r in record["runs"]}
    assert by_arm["reclaim"]["counters"]["external_allocator_reclaims"] >= 1
    assert by_arm["control"]["counters"]["external_allocator_reclaims"] == 0
    assert by_arm["control"]["counters"]["suppressed_crossings"] >= 1
    assert by_arm["reclaim"]["token_sha256"] == by_arm["control"]["token_sha256"]
    # The control patched one generator instance, never the product class.
    assert ExternalDraftBatchGenerator._reclaim_after_emission is product


def test_reclaim_refuses_output_too_short_to_cross_the_boundary():
    record = H.run_cohort(_args("--mechanism", "external-reclaim", "--max-tokens", "100"))
    assert record["verdict"] == "refused"
    assert all("too short to cross the 256-response" in r for r in record["refusals"])
    assert len(record["refusals"]) == 2


def test_diverging_arm_is_a_counterexample(monkeypatch):
    run_arm = H.run_arm

    def perturbed(cohort, arm):
        record = run_arm(cohort, arm)
        if arm == "on":
            record["_tokens"] = record["_tokens"][:-1] + [record["_tokens"][-1] ^ 1]
        return record

    monkeypatch.setattr(H, "run_arm", perturbed)
    record = H.run_cohort(_args("--mechanism", "qsdpa-tiling"))
    assert record["verdict"] == "counterexample"
    assert record["mismatches"] == ["pair 0 on: tokens differ at 7"]


def test_alternating_order_and_discarded_warmups(monkeypatch):
    seen = []
    run_arm = H.run_arm

    def spy(cohort, arm):
        seen.append(arm)
        return run_arm(cohort, arm)

    monkeypatch.setattr(H, "run_arm", spy)
    args = H.resolve_args(H.build_parser(), ["--tiny", "--out", "/dev/null", "--mechanism",
                                             "qsdpa-tiling", "--warmups", "1", "--pairs", "3"])
    record = H.run_cohort(args)
    assert seen == ["off", "on", "off", "on", "on", "off", "off", "on"]
    assert [r["arm"] for r in record["runs"]] == seen[2:]


@pytest.mark.parametrize("argv,message", [
    (["--mechanism", "qsdpa-tiling", "--model", "m"], "--i-own-the-gpu"),
    (["--mechanism", "qsdpa-tiling", "--i-own-the-gpu"], "--model is required"),
    (["--mechanism", "external-reclaim", "--i-own-the-gpu", "--model", "m"], "explicit --policy"),
    (["--mechanism", "qsdpa-tiling", "--tiny", "--i-own-the-gpu"], "random CPU models"),
    (["--mechanism", "qsdpa-tiling", "--tiny", "--kv-bits", "16"], "quantized KV"),
    (["--mechanism", "external-reclaim", "--tiny", "--kv-bits", "8"], "qsdpa-tiling only"),
    (["--mechanism", "qsdpa-tiling", "--tiny", "--pairs", "0"], "must be positive"),
])
def test_argument_refusals(argv, message, capsys):
    with pytest.raises(SystemExit):
        H.resolve_args(H.build_parser(), [*argv, "--out", "/dev/null"])
    assert message in capsys.readouterr().err


def test_main_writes_the_record_and_exit_code(tmp_path):
    out = tmp_path / "ab.json"
    code = H.main(["--tiny", "--mechanism", "qsdpa-tiling", "--pairs", "1", "--warmups", "0",
                   "--out", str(out)])
    record = json.loads(out.read_text())
    assert code == 0 and record["verdict"] == "pass"
    assert record["schema"] == "mlx2.direct-model.paired-ab.v2"
    assert all("_tokens" not in run for run in record["runs"])


def test_control_that_does_not_suppress_is_refused(monkeypatch):
    """Falsifier: without the harness-local suppression the control reclaims."""
    monkeypatch.setattr(H, "_suppress_reclaim", lambda batch, tally: None)
    record = H.run_cohort(_args("--mechanism", "external-reclaim"))
    assert record["verdict"] == "refused"
    assert record["refusals"] == ["pair 0 control: control arm reclaimed or never crossed the boundary"]


class _StuckGenerator:
    def __init__(self, failures=()):
        self.failures = list(failures)
        self.closed = False

    def insert(self, prompts, **kwargs):
        return [0]

    def next(self):
        return [], []

    def take_lane_failures(self):
        failures, self.failures = self.failures, []
        return failures

    def close(self):
        self.closed = True


@pytest.mark.parametrize("failures,reason", [
    (["lane 0: non-finite logprobs"], "lane failed: lane 0: non-finite logprobs"),
    ((), "lane failed: no response in 5 polls"),
])
def test_dropped_or_stuck_lane_is_refused_not_waited_on(monkeypatch, failures, reason):
    made = []

    def generator(self):
        made.append(_StuckGenerator(failures))
        return made[-1]

    monkeypatch.setattr(H, "IDLE_POLL_LIMIT", 5)
    monkeypatch.setattr(H.Cohort, "generator", generator)
    record = H.run_cohort(_args("--mechanism", "qsdpa-tiling"))
    assert record["verdict"] == "refused"
    assert record["refusals"] == [f"pair 0 off: {reason}", f"pair 0 on: {reason}"]
    assert all(g.closed for g in made)


def test_reclaim_records_draft_engagement_boundaries_and_state():
    record = H.run_cohort(_args("--mechanism", "external-reclaim"))
    assert record["state_comparison"] == {
        "final_target_state": "compared", "final_draft_sidecar": "compared",
    }
    for run in record["runs"]:
        assert run["counters"]["external_rounds"] > 0 and run["counters"]["proposed_tokens"] > 0
        assert [s["emitted"] for s in run["boundary_samples"]] == [256]
        assert run["max_responses_per_poll"] >= 1
    by_arm = {r["arm"]: r for r in record["runs"]}
    assert by_arm["reclaim"]["final_target_state_sha256"] == by_arm["control"]["final_target_state_sha256"]
    assert by_arm["reclaim"]["final_draft_sidecar_sha256"] == by_arm["control"]["final_draft_sidecar_sha256"]


def test_ordinary_fallback_is_not_external_engagement(monkeypatch):
    run_arm = H.run_arm

    def fallback(cohort, arm):
        record = run_arm(cohort, arm)
        record["counters"]["external_rounds"] = record["counters"]["proposed_tokens"] = 0
        return record

    monkeypatch.setattr(H, "run_arm", fallback)
    record = H.run_cohort(_args("--mechanism", "external-reclaim"))
    assert record["verdict"] == "refused"
    assert all("never proposed" in r for r in record["refusals"])


def test_final_state_difference_is_a_counterexample(monkeypatch):
    run_arm = H.run_arm

    def drifted(cohort, arm):
        record = run_arm(cohort, arm)
        if arm == "on":
            record["final_target_state"] = dict(record["final_target_state"], sha256="0" * 64)
        return record

    monkeypatch.setattr(H, "run_arm", drifted)
    record = H.run_cohort(_args("--mechanism", "qsdpa-tiling"))
    assert record["verdict"] == "counterexample"
    assert record["mismatches"] == ["pair 0 on: final_target_state_sha256 differs"]


def test_state_hash_is_none_when_unavailable_and_sensitive_to_bits():
    import mlx.core as mx

    assert H.state_hash(None) is None
    a = [mx.array([1.0, 2.0], dtype=mx.bfloat16)]
    b = [mx.array([1.0, 2.015625], dtype=mx.bfloat16)]
    assert H.state_hash(a) == H.state_hash([mx.array([1.0, 2.0], dtype=mx.bfloat16)])
    assert H.state_hash(a) != H.state_hash(b)


def test_even_pair_counts_use_the_true_median(monkeypatch):
    values = iter([1.0, 3.0, 10.0, 20.0])
    run_arm = H.run_arm

    def timed(cohort, arm):
        record = run_arm(cohort, arm)
        record["ttft_s"] = next(values)
        return record

    monkeypatch.setattr(H, "run_arm", timed)
    args = H.resolve_args(H.build_parser(), ["--tiny", "--out", "/dev/null", "--mechanism",
                                             "qsdpa-tiling", "--warmups", "0", "--pairs", "2"])
    record = H.run_cohort(args)
    # Order off,on,on,off -> off {1, 20}, on {3, 10}.
    assert record["median_by_arm"]["off"]["ttft_s"] == 10.5
    assert record["median_by_arm"]["on"]["ttft_s"] == 6.5


def test_state_status_mismatch_is_a_counterexample(monkeypatch):
    run_arm = H.run_arm

    def degraded(cohort, arm):
        record = run_arm(cohort, arm)
        if arm == "on":
            record["final_target_state"] = {"status": "metadata_unavailable", "sha256": None,
                                            "state_only_sha256": "x", "reason": "no meta_state"}
        return record

    monkeypatch.setattr(H, "run_arm", degraded)
    record = H.run_cohort(_args("--mechanism", "qsdpa-tiling"))
    assert record["verdict"] == "counterexample"
    assert record["mismatches"] == [
        "pair 0 on: final_target_state status metadata_unavailable vs complete"]
    assert record["state_oracle"]["version"] == H.STATE_ORACLE
    assert "not RNG, scheduler or full transaction-state" in record["state_oracle"]["scope"]
