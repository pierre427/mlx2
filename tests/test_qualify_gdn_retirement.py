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
    # Tiny resolves --prefill-step to 8; the cohort is bound to --batch and
    # no layout key is passed without --mtp-layout.
    assert record["protocol"]["self_mtp"] == {"num_draft": 2, "persistent": True, "rate_gate": False,
                                              "prefill_step_size": 8, "segment_aware_cohort_size": 2}
    assert record["route"]["unbound_keys"] == ["segment_aware_live_tip", "segment_aware_async_qsa_promotion"]
    assert record["route"]["requested_layout"].startswith("environment-resolved")
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


# ---- requested vs observed compute width ----

def test_widths_are_recorded_requested_vs_observed(tiny_b2):
    assert tiny_b2["widths"]["requested"] == 2
    assert tiny_b2["widths"]["observed_max"] == {"retire": 2, "control": 2}


def _cap_width(monkeypatch, width):
    run = R.RetirementDriver.run_arm

    def capped(self, name):
        data = run(self, name)
        for lane in data["lanes"].values():
            lane["mtp"]["observed_widths"] = [w for w in lane["mtp"]["observed_widths"] if w <= width] or [1]
        return data

    monkeypatch.setattr(R.RetirementDriver, "run_arm", capped)


def test_b4_cohort_observed_at_width_two_is_refused(monkeypatch):
    _cap_width(monkeypatch, 2)
    record = R.model_mode(_args("--batch", "4"))[1]
    assert record["verdict"] == "refused"
    assert "retire: observed compute width 2 < requested batch 4" in record["refusals"]
    assert "control: observed compute width 2 < requested batch 4" in record["refusals"]
    assert not any("never ran batched" in r for r in record["refusals"])  # width>=2 alone passed


def test_b2_cohort_demands_width_two(monkeypatch):
    _cap_width(monkeypatch, 1)
    record = R.model_mode(_args())[1]
    assert "retire: observed compute width 1 < requested batch 2" in record["refusals"]


# ---- source-bound route policy and prompt layout ----

@pytest.mark.parametrize("argv,message", [
    (["--async-promotion", "on"], "needs --mtp-layout segmented"),
    (["--mtp-layout", "physical", "--async-promotion", "on"], "needs --mtp-layout segmented"),
    (["--mtp-layout", "physical", "--async-promotion", "off"], "needs --mtp-layout segmented"),
    (["--mtp-layout", "paged"], "invalid choice"),
    (["--prompt-layout", "packed"], "invalid choice"),
])
def test_route_and_prompt_layout_cli_refusals(argv, message, capsys):
    with pytest.raises(SystemExit):
        R.resolve_args(R.build_parser(), ["--tiny", *argv, "--out", "/dev/null"])
    assert message in capsys.readouterr().err


def test_cli_layout_defaults_are_explicit():
    assert (_args().mtp_layout, _args().async_promotion, _args().prompt_layout) == (None, None, "ragged")
    segmented = _args("--mtp-layout", "segmented")
    assert segmented.async_promotion == "off"
    real = R.resolve_args(R.build_parser(), ["--i-own-the-gpu", "--model", "m", "--prompt-ids", "p.json",
                                             "--out", "/dev/null"])
    assert real.prompt_layout is None  # an explicit file is labelled by what it holds


@pytest.mark.parametrize("argv,live_tip,promotion", [
    ([], None, None),
    (["--mtp-layout", "physical"], False, False),
    (["--mtp-layout", "segmented"], True, False),
    (["--mtp-layout", "segmented", "--async-promotion", "on"], True, True),
])
def test_policy_passed_into_the_generator(monkeypatch, argv, live_tip, promotion):
    from mlx2.runtime import hybrid_speculative as HS
    from mlx2.runtime.generate import BatchGenerator

    seen = []

    def capture(self, model, **kwargs):
        seen.append(kwargs["self_mtp"])
        raise RuntimeError("captured")

    original = HS._retire_committed_rollbacks
    monkeypatch.setattr(BatchGenerator, "__init__", capture)
    driver = R.RetirementDriver(_args("--batch", "4", *argv))
    with pytest.raises(RuntimeError, match="captured"):
        driver.run_arm("retire")
    assert HS._retire_committed_rollbacks is original
    (config,) = seen
    assert config["segment_aware_cohort_size"] == 4 and config["prefill_step_size"] == 8
    assert config.get("segment_aware_live_tip") == live_tip
    assert config.get("segment_aware_async_qsa_promotion") == promotion


def test_environment_cannot_move_an_explicit_layout(monkeypatch):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX_LM_SEGMENTED_ASYNC_QSA_PROMOTION", "1")
    physical, unbound = R.self_mtp_config(_args("--mtp-layout", "physical"))
    assert unbound == [] and R.resolve_self_mtp(physical)["layout"] == "physical"
    segmented, _ = R.self_mtp_config(_args("--mtp-layout", "segmented"))
    resolved = R.resolve_self_mtp(segmented)
    assert resolved["layout"] == "segmented" and resolved["async_qsa_promotion"] is False
    # Without --mtp-layout the environment decides, and the record says so.
    default, unbound = R.self_mtp_config(_args())
    resolved = R.resolve_self_mtp(default)
    assert unbound and resolved["layout"] == "segmented" and resolved["async_qsa_promotion"] is True
    assert resolved["environment"]["MLX_LM_SEGMENTED_SELF_MTP"] == "1"
    assert resolved["segment_aware_cohort_size"] == 2  # bound to --batch, not the runtime floor


@pytest.fixture(scope="module")
def tiny_physical_aligned():
    return R.model_mode(_args("--mtp-layout", "physical", "--prompt-layout", "aligned"))[1]


def test_explicit_physical_aligned_cell_records_policy_route_and_prompts(tiny_physical_aligned):
    record = tiny_physical_aligned
    assert record["verdict"] == "pass", record["refusals"] + record["differences"]
    assert record["route"]["requested_layout"] == "physical" and record["route"]["unbound_keys"] == []
    assert record["route"]["observed_routes"] == {"retire": [R.PHYSICAL_ROUTE], "control": [R.PHYSICAL_ROUTE]}
    assert record["prompt_layout"] == {"requested": "aligned", "observed": "aligned", "lengths": [16, 16],
                                       "source": "tiny deterministic constructor (aligned)",
                                       "note": "an aligned cell is not ragged qualification"}
    for arm in record["arms"].values():
        policy = arm["self_mtp"]
        assert policy["passed"] == policy["held_by_generator"] == record["protocol"]["self_mtp"]
        assert policy["passed"]["segment_aware_live_tip"] is False
        assert policy["resolved"]["segment_aware_cohort_size"] == 2
        assert arm["retirement"]["live_rollback_sources"] == ["physical_batch"]
    assert record["protocol"]["ignore_eos"] is False


def _edit_lanes(monkeypatch, edit, arm="retire"):
    _perturb(monkeypatch, arm, lambda d: [edit(l) for l in d["lanes"].values()])


def test_wrong_observed_route_is_refused_for_an_explicit_layout(monkeypatch):
    _edit_lanes(monkeypatch, lambda l: l["mtp"].update(route=R.SEGMENTED_ROUTE))
    record = R.model_mode(_args("--mtp-layout", "physical"))[1]
    assert record["verdict"] == "refused"
    assert "retire lane 0: observed route segmented_self_mtp does not match requested --mtp-layout physical" \
        in record["refusals"]
    assert not any(r.startswith("control lane") and "observed route" in r for r in record["refusals"])


def test_physical_route_without_a_promotion_receipt_is_not_a_segmented_match(monkeypatch):
    _edit_lanes(monkeypatch, lambda l: l["mtp"].update(async_qsa_promotion=None))
    record = R.model_mode(_args("--mtp-layout", "segmented", "--async-promotion", "on",
                                "--prompt-layout", "aligned"))[1]
    assert record["verdict"] == "refused"
    assert any(r.startswith("retire lane 0: observed route continuous_batched_self_mtp does not match")
               for r in record["refusals"]), record["refusals"]


def test_unrequested_route_on_the_default_layout_is_recorded_not_refused(monkeypatch):
    _edit_lanes(monkeypatch, lambda l: l["mtp"].update(route=R.SEGMENTED_ROUTE))
    record = R.model_mode(_args())[1]
    assert not any("observed route" in r for r in record["refusals"])
    assert record["route"]["observed_routes"]["retire"] == [R.SEGMENTED_ROUTE]


@pytest.mark.parametrize("edit,reason", [
    (lambda d: d["self_mtp"]["held_by_generator"].update(segment_aware_cohort_size=2),
     "retire: generator self-MTP config differs"),
    (lambda d: d["self_mtp"]["resolved"].update(segment_aware_cohort_size=2),
     "retire: segment_aware_cohort_size 2 != requested batch 4"),
    (lambda d: d["self_mtp"]["resolved"].update(layout="segmented"),
     "retire: runtime resolved layout segmented != requested physical"),
])
def test_policy_binding_mismatches_are_refused(monkeypatch, edit, reason):
    _perturb(monkeypatch, "retire", edit)
    record = R.model_mode(_args("--batch", "4", "--mtp-layout", "physical", "--prompt-layout", "aligned"))[1]
    assert record["verdict"] == "refused"
    assert any(r.startswith(reason) for r in record["refusals"]), record["refusals"]


def test_segmented_cell_with_zero_retired_records_is_refused():
    """Tiny segmented B2: the hook runs, retires nothing; that is a refusal,
    and the per-row live records are reported as segmented, not physical."""
    record = R.model_mode(_args("--mtp-layout", "segmented"))[1]
    assert record["verdict"] == "refused"
    assert "retire: no rollback record was retired" in record["refusals"]
    assert not any("observed route" in r for r in record["refusals"])
    retire = record["arms"]["retire"]
    assert retire["retirement"]["retire_calls"] > 0 and retire["retirement"]["records_dropped"] == 0
    assert retire["retirement"]["live_rollback_sources"] == ["segmented_rows"]
    assert record["route"]["observed_routes"]["retire"] == [R.SEGMENTED_ROUTE]


def test_prompt_layout_is_validated_and_labelled():
    driver = R.RetirementDriver(_args())
    assert R.prompt_layout(driver.prompts) == "ragged"
    driver.args.prompt_layout = "aligned"
    with pytest.raises(SystemExit, match="--prompt-layout aligned but the prompts are ragged"):
        driver.check_geometry()
    driver.args.prompt_layout = None  # explicit ids: labelled by what they are
    driver.check_geometry()
    driver.prompts = [list(range(1, 17))] * 2
    driver.args.prompt_layout = "ragged"
    with pytest.raises(SystemExit, match="--prompt-layout ragged but the prompts are aligned"):
        driver.check_geometry()
