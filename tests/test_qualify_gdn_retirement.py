"""CPU tests for scripts/qualify_gdn_retirement.py.

The synthetic oracle's comparisons are geometry-matched and must be exact
in storage bits; the falsifiers below corrupt retirement and lane filtering
to prove the oracle sees it. The model mode runs the deterministic tiny
Flash-Next-class model only; real B2/B4 cells need --i-own-the-gpu and
belong to the GPU owner. tests/test_short_gdn_rollback.py (allclose, forced
CPU) is not used as exactness evidence anywhere here.
"""

import json
import subprocess
import sys

import pytest

from scripts import qualify_gdn_retirement as R


def _args(*extra):
    return R.resolve_args(R.build_parser(), ["--tiny", "--out", "/dev/null", *extra])


def test_help_is_import_safe():
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, scripts.qualify_gdn_retirement; print('mlx.core' in sys.modules)"],
        capture_output=True, text=True, env={"PYTHONPATH": "src:."},
    )
    assert probe.returncode == 0 and probe.stdout.strip() == "False", probe.stderr


def test_synthetic_oracle_is_exact_and_restores_the_accept_flag():
    from mlx2.runtime.models import qwen3_5

    before = qwen3_5._GDN_ARRAY_ACCEPT
    cases = R.synthetic_oracle()
    assert qwen3_5._GDN_ARRAY_ACCEPT == before
    assert len(cases) == 9 and all(c["exact"] for c in cases), cases
    names = " ".join(c["case"] for c in cases)
    for needle in ("heterogeneous accepted lengths + short-after-wide", "retirement on vs off",
                   "lane removal", "accept implementations agree", "rewind", "per_row_fn"):
        assert needle in names
    retired = [c for c in cases if "retirement" in c["case"]]
    assert all(c["records"]["retire"] < c["records"]["no_retire"] for c in retired)


def test_oracle_catches_retirement_that_touches_recurrent_state(monkeypatch):
    """Retirement runs after the commit trim consumed its record, so the
    property under test is that it leaves state bits alone. A retirement
    that perturbs the recurrent plane by one ulp must fail the oracle."""
    import mlx.core as mx
    from mlx2.runtime.models.cache import ArraysCache

    real = ArraysCache.retire_rollbacks

    def perturbing(self, keep=1):
        dropped = real(self, keep)
        state = self.cache[1]
        self.cache[1] = (state.view(mx.uint32) ^ 1).view(state.dtype)
        return dropped

    monkeypatch.setattr(ArraysCache, "retire_rollbacks", perturbing)
    cases = R.synthetic_oracle()
    failed = [c["case"] for c in cases if not c["exact"]]
    assert "rewind: retirement on vs off (whole batch)" in failed
    assert "per_row_fn: retirement on vs off (whole batch)" in failed


def test_oracle_catches_a_filter_that_keeps_the_wrong_row(monkeypatch):
    from mlx2.runtime.models.cache import ArraysCache

    real = ArraysCache.filter
    monkeypatch.setattr(ArraysCache, "filter", lambda self, rows: real(self, [1 - r for r in rows]))
    cases = R.synthetic_oracle()
    assert [c["case"] for c in cases if not c["exact"]] == [
        "rewind: lane removal filter([1]) vs extract(1)",
        "per_row_fn: lane removal filter([1]) vs extract(1)",
    ]


@pytest.fixture(scope="module")
def tiny_b2():
    return R.model_mode(_args())[1]


def test_tiny_b2_arms_engage_and_match_exactly(tiny_b2):
    record = tiny_b2
    assert record["verdict"] == "pass", record["refusals"] + record["differences"]
    retire, control = record["arms"]["retire"], record["arms"]["control"]
    assert retire["retirement"]["records_dropped"] > 0 and retire["retirement"]["suppressed_calls"] == 0
    assert control["retirement"]["retire_calls"] == 0 and control["retirement"]["suppressed_calls"] > 0
    assert retire["retirement"]["max_live_rollback_records"] < control["retirement"]["max_live_rollback_records"]
    for arm in (retire, control):
        assert set(arm["memory"]) >= {"active_bytes", "cache_bytes", "peak_bytes"}
        for lane in arm["lanes"].values():
            assert lane["mtp"]["draft_proposed"] > 0 and lane["mtp"]["num_draft"] == 2
            assert lane["final_state"]["status"] == "complete"
            assert lane["continuation"]["status"] == "complete"
            assert len(lane["logprob_rows"]) == 16
    assert record["protocol"]["self_mtp"] == {"num_draft": 2, "persistent": True, "rate_gate": False,
                                              "prefill_step_size": 2048}
    assert "not a controlled performance run" in record["protocol"]["timing"]


def test_the_product_retirement_hook_is_restored_even_on_failure(monkeypatch):
    from mlx2.runtime import hybrid_speculative as HS
    from mlx2.runtime.generate import BatchGenerator

    original = HS._retire_committed_rollbacks
    driver = R.RetirementDriver(_args())

    def boom(self, *a, **k):
        raise RuntimeError("insert failed")

    monkeypatch.setattr(BatchGenerator, "insert", boom)
    with pytest.raises(RuntimeError, match="insert failed"):
        driver.run_arm("control")
    assert HS._retire_committed_rollbacks is original


def _perturb(monkeypatch, arm, edit):
    run = R.RetirementDriver.run_arm

    def perturbed(self, name):
        data = run(self, name)
        if name == arm:
            edit(data)
        return data

    monkeypatch.setattr(R.RetirementDriver, "run_arm", perturbed)


def test_token_divergence_between_arms_is_a_counterexample(monkeypatch):
    _perturb(monkeypatch, "control", lambda d: d["lanes"][1]["tokens"].__setitem__(3, 63))
    record = R.model_mode(_args())[1]
    assert record["verdict"] == "counterexample"
    assert record["differences"] == ["lane 1: tokens differ"]


def test_state_bit_divergence_is_a_counterexample(monkeypatch):
    _perturb(monkeypatch, "control", lambda d: d["lanes"][0]["final_state"].update(sha256="0" * 64))
    record = R.model_mode(_args())[1]
    assert record["verdict"] == "counterexample"
    assert "lane 0: final state digest differs" in record["differences"]


@pytest.mark.parametrize("edit,reason", [
    (lambda d: [l["mtp"].update(draft_proposed=0, draft_accepted=0) for l in d["lanes"].values()],
     "retire: no self-MTP proposals"),
    (lambda d: [l["mtp"].update(draft_accepted=l["mtp"]["draft_proposed"]) for l in d["lanes"].values()],
     "retire: no rejected proposals (no rollback)"),
    (lambda d: [l["mtp"].update(observed_widths=[1]) for l in d["lanes"].values()],
     "retire: never ran batched (width >= 2)"),
    (lambda d: d["retirement"].update(records_dropped=0), "retire: no rollback record was retired"),
    (lambda d: d["lanes"][0].update(finish_reason=None), "retire lane 0: early stop"),
])
def test_zero_engagement_is_refused(monkeypatch, edit, reason):
    _perturb(monkeypatch, "retire", edit)
    record = R.model_mode(_args())[1]
    assert record["verdict"] == "refused"
    assert any(r.startswith(reason) for r in record["refusals"]), record["refusals"]


def test_bounded_polling_is_refused():
    record = R.model_mode(_args("--time-limit-s", "1e-9"))[1]
    assert record["verdict"] == "refused" and any("bounded" in r for r in record["refusals"])


@pytest.mark.parametrize("argv,message", [
    (["--model", "m"], "--i-own-the-gpu"),
    (["--i-own-the-gpu"], "--model is required"),
    (["--tiny", "--batch", "3"], "2 or 4"),
    (["--tiny", "--context", "20000"], "--context 8..16384"),
    (["--tiny", "--gen", "600"], "--gen 1..512"),
    (["--tiny", "--i-own-the-gpu"], "random CPU model"),
    (["--synthetic-oracle", "--tiny"], "runs alone"),
])
def test_argument_refusals(argv, message, capsys):
    with pytest.raises(SystemExit):
        R.resolve_args(R.build_parser(), [*argv, "--out", "/dev/null"])
    assert message in capsys.readouterr().err


def test_short_or_missing_prompts_are_refused():
    driver = R.RetirementDriver(_args())
    driver.prompts = driver.prompts[:1]
    with pytest.raises(SystemExit, match="batch must be 2 or 4"):
        driver.check_geometry()
    driver.prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]]
    with pytest.raises(SystemExit, match="8..16384"):
        driver.check_geometry()


def test_main_writes_oracle_and_model_records(tmp_path):
    oracle = tmp_path / "oracle.json"
    assert R.main(["--synthetic-oracle", "--out", str(oracle)]) == 0
    data = json.loads(oracle.read_text())
    assert data["mode"] == "synthetic_oracle" and data["verdict"] == "pass" and "cpu" in data["scope"]
    model = tmp_path / "model.json"
    code = R.main(["--tiny", "--batch", "4", "--out", str(model)])
    data = json.loads(model.read_text())
    assert code == 0 and data["verdict"] == "pass" and data["protocol"]["batch"] == 4
    assert data["identity"]["files"] and data["identity"]["mlx"]["version"]


# ---- repairs after parent review (arm isolation, row status, bounds) ----

def test_arm_isolation_gate_tracks_owners_and_finds_none_alive(tiny_b2):
    check = tiny_b2["arm_isolation"]["control"]
    assert tiny_b2["arm_isolation"]["retire"] is None
    # generator(s), responses, cache entries and their state arrays
    assert check["tracked"] > 20 and check["alive"] == 0 and check["unreferenceable"] == 0
    for arm in tiny_b2["arms"].values():
        assert "_owners" not in arm
        assert all("_final" not in lane for lane in arm["lanes"].values())


def test_a_preceding_arm_owner_kept_alive_is_refused(monkeypatch):
    """The pre-repair shape: the retire arm's final response (and its caches)
    held while control runs. The gate must see it, not just equal tokens."""
    leaked = []
    real = R.RetirementDriver.continuation

    def leaking(self, lane, record, owners=None):
        leaked.append(record.get("_final"))
        return real(self, lane, record, owners)

    monkeypatch.setattr(R.RetirementDriver, "continuation", leaking)
    record = R.model_mode(_args())[1]
    check = record["arm_isolation"]["control"]
    assert check["alive"] > 0
    assert record["verdict"] == "refused"
    assert any(r.startswith("control: preceding arm's tensor owners not released") for r in record["refusals"])
    assert leaked  # still referenced here, which is what the gate detected


def test_a_leaked_generator_is_refused(monkeypatch):
    from mlx2.runtime import generate

    kept = []
    real_init = generate.BatchGenerator.__init__

    def keep(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        kept.append(self)

    monkeypatch.setattr(generate.BatchGenerator, "__init__", keep)
    record = R.model_mode(_args())[1]
    assert record["arm_isolation"]["control"]["alive"] >= 1
    assert record["verdict"] == "refused"


def test_unavailable_logprob_rows_on_both_arms_never_pass(monkeypatch):
    def blank(data):
        for lane in data["lanes"].values():
            lane["logprob_rows"] = [None] * len(lane["logprob_rows"])

    run = R.RetirementDriver.run_arm

    def perturbed(self, name):
        data = run(self, name)
        blank(data)
        return data

    monkeypatch.setattr(R.RetirementDriver, "run_arm", perturbed)
    record = R.model_mode(_args())[1]
    assert record["verdict"] != "pass"
    assert record["verdict"] == "exact_with_unavailable_parts"
    assert all("logprob rows unavailable (missing or undigestable)" in item
               for item in record["incomparable"] if "logprob" in item)
    assert sum("logprob" in item for item in record["incomparable"]) == 2


def test_logprob_rows_disabled_is_explicitly_unavailable():
    record = R.model_mode(_args("--logprob-rows", "0"))[1]
    assert record["verdict"] == "exact_with_unavailable_parts"
    assert any("(disabled (--logprob-rows 0))" in item for item in record["incomparable"])


def test_disabled_continuation_with_a_bounded_early_stop_does_not_crash():
    record = R.model_mode(_args("--continuation-tokens", "0", "--time-limit-s", "1e-9"))[1]
    assert record["verdict"] == "refused"
    for arm in record["arms"].values():
        for lane in arm["lanes"].values():
            assert lane["continuation"] == {"status": "unavailable",
                                            "reason": "disabled (--continuation-tokens 0)"}
            assert "_final" not in lane


@pytest.mark.parametrize("argv,message", [
    (["--time-limit-s", "nan"], "finite"),
    (["--time-limit-s", "inf"], "finite"),
    (["--time-limit-s", "0"], "finite"),
    (["--continuation-tokens", "513"], "0..512"),
    (["--logprob-rows", "513"], "0..512"),
    (["--continuation-tokens", "-1"], "0..512"),
    (["--prefill-step", "16385"], "--prefill-step 1..16384"),
    (["--prefill-step", "0"], "--prefill-step 1..16384"),
])
def test_nonfinite_or_unbounded_settings_are_refused(argv, message, capsys):
    with pytest.raises(SystemExit):
        R.resolve_args(R.build_parser(), ["--tiny", *argv, "--out", "/dev/null"])
    assert message in capsys.readouterr().err


def test_explicit_zero_rows_and_continuation_are_legal():
    args = _args("--logprob-rows", "0", "--continuation-tokens", "0", "--prefill-step", "16384")
    assert (args.logprob_rows, args.continuation_tokens, args.prefill_step) == (0, 0, 16384)
