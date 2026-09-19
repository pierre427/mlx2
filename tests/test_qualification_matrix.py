import importlib.util
import json
import types
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


matrix = load("run_qualification_matrix")
spomin = load("run_spomin_20x20")
prompts = load("build_context_prompts")


def model_fixture():
    return {
        "name": "fixture",
        "tokenizer_renderer": "qwen_direct",
        "arms": [
            {"name": "A", "url": "http://a", "receipt_requirements": [{"path": "cache", "equals": "apcv2"}],
             "qualification_receipt": {"path": "a-route.json", "required_checks": ["runtime_stable"]}},
            {"name": "B", "url": "http://b", "receipt_requirements": [{"path": "cache", "equals": "apcv2"}],
             "qualification_receipt": {"path": "b-route.json", "required_checks": ["runtime_stable"]}},
        ],
        "contexts": [
            {"tokens": 100, "prompt_path": "100.txt", "prompt_sha256": "a", "calibrated_prompt_tokens": 100},
            {"tokens": 200, "prompt_path": "200.txt", "prompt_sha256": "b", "calibrated_prompt_tokens": 200},
            {"tokens": 300, "prompt_path": "300.txt", "prompt_sha256": "c", "calibrated_prompt_tokens": 300},
        ],
        "context": {"runs_per_cell": 3},
        "batch_stress": {"rounds": 20, "width": 20, "prompts": [str(i) for i in range(20)]},
    }


def test_context_order_alternates_arm_after_each_matched_measurement():
    cells = matrix.alternating_cells([model_fixture()])
    observed = [(row["run"], row["arm"], row["context_tokens"]) for row in cells]
    assert observed == [
        (0, "A", 100), (0, "B", 100),
        (1, "A", 100), (1, "B", 100),
        (2, "A", 100), (2, "B", 100),
        (0, "A", 200), (0, "B", 200),
        (1, "A", 200), (1, "B", 200),
        (2, "A", 200), (2, "B", 200),
        (0, "A", 300), (0, "B", 300),
        (1, "A", 300), (1, "B", 300),
        (2, "A", 300), (2, "B", 300),
    ]
    assert all(left["arm"] != right["arm"] for left, right in zip(cells, cells[1:]))
    assert len({row["cell_id"] for row in cells}) == len(cells)


def test_single_run_context_is_explicitly_exploratory():
    model = model_fixture()
    model["context"]["runs_per_cell"] = 1
    manifest = {"schema": matrix.SCHEMA, "thermal": {}, "models": [model]}
    with pytest.raises(ValueError, match="exactly three"):
        matrix.validate_manifest(manifest)
    matrix.validate_manifest(manifest, allow_exploratory_single_run=True)
    cells = matrix.alternating_cells([model])
    assert [(row["arm"], row["context_tokens"]) for row in cells] == [
        (arm, context)
        for context in (100, 200, 300)
        for arm in ("A", "B")
    ]


def test_context_template_requires_prompt_generation_before_cell_materialization():
    model = model_fixture()
    del model["contexts"][0]["prompt_path"]
    with pytest.raises(ValueError, match="build_context_prompts.py"):
        matrix.alternating_cells([model])


def test_single_arm_schedule_has_no_alternating_arm_claim():
    model = model_fixture()
    model["name"] = "ordinary-only"
    model["arm_order_claim"] = "none"
    model["arms"] = model["arms"][:1]
    manifest = {"schema": matrix.SCHEMA, "thermal": {}, "models": [model]}
    matrix.validate_manifest(manifest)
    cells = matrix.alternating_cells([model])
    assert len(cells) == 9
    assert {row["arm"] for row in cells} == {"A"}
    assert {row["arm_order_claim"] for row in cells} == {"none"}
    assert [(row["run"], row["context_tokens"]) for row in cells] == [
        (run, context) for context in (100, 200, 300) for run in range(3)
    ]


def test_suite_override_binds_batch_activation_and_receipt_independently():
    arm = model_fixture()["arms"][0]
    arm["suites"] = {"batch_stress": {
        "activate_command": ["launch", "--max-lanes", "20"],
        "qualification_receipt": {"path": "b20.json", "required_checks": ["batch"]},
        "status_requirements": [{"path": "max_lanes", "equals": 20}],
    }}
    context = matrix.effective_arm(arm, "context")
    batch = matrix.effective_arm(arm, "batch_stress")
    assert "activate_command" not in context
    assert batch["qualification_receipt"]["path"] == "b20.json"
    assert batch["receipt_requirements"] == arm["receipt_requirements"]
    assert batch["status_requirements"][0]["equals"] == 20


def test_comparative_models_still_require_two_arms():
    model = model_fixture()
    model["arms"] = model["arms"][:1]
    manifest = {"schema": matrix.SCHEMA, "thermal": {}, "models": [model]}
    with pytest.raises(ValueError, match="comparative qualification needs at least two"):
        matrix.validate_manifest(manifest)
    model["arm_order_claim"] = "none"
    model["arms"] = model_fixture()["arms"]
    with pytest.raises(ValueError, match="requires exactly one"):
        matrix.validate_manifest(manifest)


def test_batch_stress_is_distinct_twenty_rounds_at_width_twenty():
    cells = matrix.batch_stress_cells([model_fixture()])
    assert len(cells) == 40
    assert all(row["suite"] == "batch_stress" and row["requested_width"] == 20 for row in cells)
    assert {row["run"] for row in cells} == set(range(20))


def test_measured_batch_bodies_declare_one_exact_atomic_cohort():
    bodies = [{"prompt": str(index)} for index in range(20)]
    cohort, measured = matrix.declared_batch_bodies(
        bodies, cell_id="cell-123", width=20
    )

    assert cohort == {"id": "cell-123", "size": 20}
    assert all(body["batch_cohort"] == cohort for body in measured)
    assert all("batch_cohort" not in body for body in bodies)
    with pytest.raises(ValueError, match="request count"):
        matrix.declared_batch_bodies(bodies[:19], cell_id="cell-123", width=20)


def test_manifest_rejects_any_arm_without_canonical_qualification_receipt():
    model = model_fixture()
    del model["arms"][0]["qualification_receipt"]
    manifest = {"schema": matrix.SCHEMA, "thermal": {}, "models": [model]}
    with pytest.raises(ValueError, match="must bind a canonical qualification receipt"):
        matrix.validate_manifest(manifest)


def test_receipt_and_combined_counter_gates_fail_closed():
    receipt = {"cache": "apcv2", "speculation": {"kind": "external_dflash2"}}
    assert matrix.validate_requirements(receipt, [
        {"path": "cache", "equals": "apcv2"},
        {"path": "speculation.kind", "equals": "external_dflash2"},
    ], "receipt")
    before = {"execution": {"ple": {"hits": 3, "misses": 4}, "fallbacks": 0}}
    after = {"execution": {"ple": {"hits": 5, "misses": 5}, "fallbacks": 0}}
    evidence = matrix.validate_counter_deltas(before, after, [
        {"paths": ["execution.ple.hits", "execution.ple.misses"], "delta_gte": 3},
        {"path": "execution.fallbacks", "delta_gte": 0, "delta_lte": 0},
    ])
    assert evidence[0]["delta"] == 3
    with pytest.raises(AssertionError):
        matrix.validate_requirements(receipt, [{"path": "cache", "equals": "legacy"}], "receipt")
    with pytest.raises(AssertionError, match="mechanism counter unavailable") as missing:
        matrix.validate_counter_deltas({}, {}, [{"path": "apcv2.hits", "delta_gte": 1}])
    assert "apcv2.hits" in str(missing.value)


def test_context_arm_uses_fresh_cache_namespace_per_cell():
    arm = {
        "name": "ordinary",
        "activation_key": "ordinary-context",
        "activate_command": [
            "python", "activate_qualification_arm.py", "--cwd", "/repo", "--",
            "/usr/bin/env", "PYTHONPATH=src", "/repo/.venv/bin/python", "-m", "mlx2.server",
            "--cache-dir", "/repo/cache/context",
        ],
    }
    first = matrix.isolated_context_arm(arm, {"cell_id": "cell-a"})
    second = matrix.isolated_context_arm(arm, {"cell_id": "cell-b"})
    assert first["context_cache_dir"] == "/repo/cache/context/cell-cell-a"
    assert second["context_cache_dir"] == "/repo/cache/context/cell-cell-b"
    assert first["activation_key"] != second["activation_key"]
    assert first["activate_command"][4:6] == ["--fresh-cache-dir", "/repo/cache/context/cell-cell-a"]
    assert first["activate_command"][-1] == "/repo/cache/context/cell-cell-a"
    assert arm["activate_command"][-1] == "/repo/cache/context"


def test_matrix_quiescence_waits_for_published_cow_release():
    class Client:
        def __init__(self):
            self.statuses = iter([
                {"inflight": 0, "queue_depth": 0,
                 "apcv2": {"cow": {"active_leases": 1}}},
                {"inflight": 0, "queue_depth": 0,
                 "apcv2": {"cow": {"active_leases": 0}}},
            ])

        def get(self, path):
            assert path == "/v1/status"
            return next(self.statuses)

    clock = iter([0.0, 0.0, 0.1])
    final, evidence = matrix.wait_for_quiescence(
        Client(), sleep=lambda _: None, monotonic=lambda: next(clock)
    )
    assert final["apcv2"]["cow"]["active_leases"] == 0
    assert evidence["passed"] is True
    assert [row["active_cow_leases"] for row in evidence["samples"]] == [1, 0]


def test_matrix_quiescence_fails_closed_on_leak_or_missing_counter():
    class Client:
        def __init__(self, status):
            self.status = status

        def get(self, path):
            assert path == "/v1/status"
            return self.status

    for status in (
        {"inflight": 0, "queue_depth": 0,
         "apcv2": {"cow": {"active_leases": 1}}},
        {"inflight": 0, "queue_depth": 0, "apcv2": {"cow": {}}},
    ):
        clock = iter([0.0, 0.0, 0.01])
        with pytest.raises(AssertionError, match="quiescence"):
            matrix.wait_for_quiescence(
                Client(status), timeout_seconds=0.01,
                poll_interval_seconds=0, sleep=lambda _: None,
                monotonic=lambda: next(clock),
            )


def test_aggregate_qualification_receipt_is_bound_to_active_server(tmp_path):
    status = {"runtime": {"source": "abc"}, "artifact": "weights", "settings": {"mtp": True}}
    report = {"schema": "mlx2.serving-qualification.v1", "passed": True, **status,
              "checks": {"feature_shared_qsa": {"passed": True},
                         "feature_async_promotion": {"passed": True}}}
    receipt = tmp_path / "route.json"
    receipt.write_text(__import__("json").dumps(report))
    arm = {"qualification_receipt": {"path": "route.json", "required_checks": [
        "feature_shared_qsa", "feature_async_promotion"]}}
    evidence = matrix.validate_bound_qualification_receipt(tmp_path / "manifest.json", arm, status)
    assert evidence["passed"] and evidence["required_checks"] == arm["qualification_receipt"]["required_checks"]
    report["checks"]["feature_async_promotion"]["passed"] = False
    receipt.write_text(__import__("json").dumps(report))
    with pytest.raises(AssertionError, match="lacks passed checks"):
        matrix.validate_bound_qualification_receipt(tmp_path / "manifest.json", arm, status)
    with pytest.raises(AssertionError, match="does not match"):
        matrix.validate_bound_qualification_receipt(
            tmp_path / "manifest.json", {**arm}, {**status, "artifact": "other"}
        )


def test_canonical_muse_ordinary_arm_is_full_receipt_bound():
    manifest = json.loads(
        (ROOT / "qualification" / "four-model-experiments.json").read_text()
    )
    muse = next(model for model in manifest["models"] if model["name"] == "muse-glimmer")
    ordinary = next(arm for arm in muse["arms"] if arm["name"] == "ordinary")
    receipt = ordinary["qualification_receipt"]
    assert receipt["path"] == (
        "~/Desktop/mlx2/qualification/runs/macos-26.7/"
        "muse-glimmer/ordinary/route-qualification.json"
    )
    assert receipt["required_checks"] == [
        "apcv2_reuse",
        "apcv2_stores",
        "cache_leases",
        "quiescence",
        "runtime_stable",
    ]


def test_canonical_north_ordinary_arm_is_full_receipt_bound():
    manifest = json.loads(
        (ROOT / "qualification" / "four-model-experiments.json").read_text()
    )
    north = next(
        model for model in manifest["models"] if model["name"] == "north-mini-code"
    )
    assert north["arm_order_claim"] == "none"
    assert [arm["name"] for arm in north["arms"]] == ["ordinary"]
    receipt = north["arms"][0]["qualification_receipt"]
    assert receipt["path"] == (
        "runs/macos-26.7/north-mini-code/ordinary/route-qualification.json"
    )
    assert receipt["required_checks"] == [
        "apcv2_reuse",
        "apcv2_stores",
        "cache_leases",
        "quiescence",
        "runtime_stable",
    ]


def test_spomin_counter_applicability_excludes_long_context_and_stress_only_gates():
    requirements = [
        {"path": "apcv2.hits", "delta_gte": 1},
        {"path": "indexed.engaged", "delta_gte": 1, "min_context_tokens": 32768},
        {"path": "batch.width", "delta_gte": 1, "suite": "batch_stress"},
        {"path": "mtp.batched", "delta_gte": 1},
    ]
    selected = matrix.applicable_requirements(
        requirements, {"suite": "spomin20x20", "context_tokens": 8192}
    )
    assert [row["path"] for row in selected] == ["apcv2.hits", "mtp.batched"]


def test_thermal_gate_requires_nominal_warning_free_settled_samples():
    base = {"thermal_state": 0, "pmset_no_thermal_warning": True,
            "pmset_no_performance_warning": True, "pmset_no_cpu_power_warning": True,
            "battery_temperature_c": 31.0, "virtual_temperature_c": 36.0}
    assert matrix.thermally_stable(base, {})
    assert not matrix.thermally_stable({**base, "thermal_state": 1}, {})
    samples = iter([{**base, "virtual_temperature_c": value} for value in (36.0, 35.8, 35.7)])
    with patch.object(matrix, "sample_thermal", side_effect=lambda command=None: next(samples)):
        observed = matrix.stabilize_thermal({"consecutive_samples": 3,
                                             "sample_interval_seconds": 0,
                                             "max_wait_seconds": 10}, sleep=lambda _: None)
    assert len(observed) == 3


def test_post_thermal_requires_two_consecutive_nonnominal_samples():
    nominal = {"thermal_state": 0, "pmset_no_thermal_warning": True,
               "pmset_no_performance_warning": True, "pmset_no_cpu_power_warning": True,
               "battery_temperature_c": 31.0, "virtual_temperature_c": 36.0}
    bad = {**nominal, "thermal_state": 1}
    with patch.object(matrix, "sample_thermal", side_effect=[bad, nominal]):
        assert matrix.post_thermal_samples({"post_sample_interval_seconds": 0}, sleep=lambda _: None) == [bad, nominal]
    with patch.object(matrix, "sample_thermal", side_effect=[bad, bad]):
        with pytest.raises(AssertionError, match="twice"):
            matrix.post_thermal_samples({"post_sample_interval_seconds": 0}, sleep=lambda _: None)


def test_swap_growth_hard_gate_and_diagnostic_representation():
    baseline = {"used_bytes": 100}
    before = {"used_bytes": 200}
    after = {"used_bytes": 100 + matrix.DEFAULT_SWAP_GROWTH_LIMIT_BYTES + 1}
    with pytest.raises(AssertionError, match="swap grew"):
        matrix.swap_evidence(baseline, before, after)
    evidence = matrix.swap_evidence(baseline, before, after, enforce=False)
    assert not evidence["passed"] and not evidence["enforced"]


def test_matrix_continue_on_error_persists_failure_and_runs_remaining_cells(tmp_path, monkeypatch):
    manifest = {"schema": matrix.SCHEMA, "thermal": {}, "models": [model_fixture()]}
    manifest_path, output = tmp_path / "manifest.json", tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest))
    calls = []
    def fake_cell(_manifest, _model, arm, cell, _thermal, _swap):
        calls.append(cell["cell_id"])
        if len(calls) == 1:
            raise RuntimeError("first cell failed")
        return {"cell_manifest": cell, "server_identity": {"arm": arm["name"]}, "passed": True}
    monkeypatch.setattr(matrix, "context_cell", fake_cell)
    monkeypatch.setattr(matrix, "host_identity", lambda: {
        "platform": "test", "macos": "test", "sw_vers": "test", "uname": "test",
        "python": "test", "mlx": "test", "git_revision": "test",
        "git_status": "", "harness_source": {}, "harness_source_sha256": "test"})
    monkeypatch.setattr(matrix, "sample_swap", lambda command=None: {"used_bytes": 0})
    monkeypatch.setattr(sys, "argv", ["run_qualification_matrix.py", "--manifest", str(manifest_path),
                        "--output", str(output), "--suite", "context", "--continue-on-error"])
    with pytest.raises(SystemExit, match="matrix incomplete"):
        matrix.main()
    report = json.loads(output.read_text())
    assert len(calls) == 18
    assert len(report["cells"]) == 18
    assert sum(not row["passed"] for row in report["cells"].values()) == 1


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        self.kwargs = kwargs
        rendered = "template " + messages[0]["content"]
        return rendered.split() if tokenize else rendered


def test_context_prompt_calibration_hits_exact_rendered_target():
    text = prompts.calibrate(FakeTokenizer(), 128)
    assert prompts.rendered_tokens(FakeTokenizer(), text) == 128
    assert "LADDER_READY" in text
    muse = prompts.calibrate(FakeTokenizer(), 128, "muse_direct")
    assert prompts.rendered_tokens(FakeTokenizer(), muse, "muse_direct") == 128
    north = prompts.calibrate(FakeTokenizer(), 128, "north_direct")
    tokenizer = FakeTokenizer()
    assert prompts.rendered_tokens(tokenizer, north, "north_direct") == 128
    assert tokenizer.kwargs == {
        "reasoning": False,
        "reasoning_effort": "none",
        "skip_thinking": True,
        "tools": None,
    }


def test_generated_manifest_rebases_relative_qualification_receipts(tmp_path):
    manifest_dir = tmp_path / "qualification"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "four-model-experiments.json"
    manifest_bytes = b'{"schema":"mlx2.qualification-matrix.v1"}\n'
    manifest_path.write_bytes(manifest_bytes)
    model = {
        "tokenizer_path": "/absolute/tokenizer",
        "arms": [{"qualification_receipt": {"path": "runs/model/route.json"}}],
    }
    prompts.rebase_model_paths(model, manifest_dir)
    assert model["arms"][0]["qualification_receipt"]["path"] == str(
        (manifest_dir / "runs/model/route.json").resolve()
    )


def test_context_prompt_must_match_frozen_hash_and_calibrated_count():
    text = "frozen prompt"
    import hashlib
    cell = {"context_tokens": 17, "calibrated_prompt_tokens": 17,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
    assert matrix.verify_frozen_prompt(cell, text) == cell["prompt_sha256"]
    with pytest.raises(ValueError, match="hash mismatch"):
        matrix.verify_frozen_prompt(cell, text + " changed")
    with pytest.raises(ValueError, match="generated manifest"):
        matrix.verify_frozen_prompt({"context_tokens": 17}, text)
    with pytest.raises(ValueError, match="do not match"):
        matrix.verify_frozen_prompt({**cell, "calibrated_prompt_tokens": 16}, text)


def test_generated_manifest_must_rebase_relative_spomin_corpus(tmp_path):
    # The generator writes its manifest to a different directory, so portable
    # input artifacts must be resolved before that move.
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir(); output.mkdir()
    relative = source / "corpora" / "frozen.json"
    relative.parent.mkdir(); relative.write_text("{}")
    model = {"tokenizer_path": "/absolute/tokenizer", "spomin20x20": {
        "corpus_path": "corpora/frozen.json", "tokenizer_path": "/absolute/tokenizer",
        "source_reconciliation_tokenizer_path": "/absolute/source-tokenizer"}}
    prompts.rebase_model_paths(model, source)
    assert Path(model["spomin20x20"]["corpus_path"]) == relative.resolve()
    assert model["spomin20x20"]["source_reconciliation_tokenizer_path"] == "/absolute/source-tokenizer"


def test_spomin_rebuild_is_deterministic_and_retains_needles_and_query():
    case = {"case_id": "domain.01", "domain": "domain", "question": "Why?",
            "expected_concepts": [["because"]]}
    first = spomin.prepare_case(case, FakeTokenizer(), 1024)
    second = spomin.prepare_case(case, FakeTokenizer(), 1024)
    assert first["receipt"] == second["receipt"]
    assert first["receipt"]["projected_target_tokens"] <= first["receipt"]["target_limit_tokens"]
    compacted = "\n".join(row["content"] for row in first["compacted_messages"])
    assert all(value in compacted for value in first["needles"].values())
    assert first["sentinel"] in compacted
    assert first["receipt"]["physical_kv_surgery"] is False


def test_manifest_does_not_conflate_spomin_20x20_with_batch_stress():
    manifest = {"schema": matrix.SCHEMA, "thermal": {}, "models": [model_fixture()]}
    matrix.validate_manifest(manifest)
    manifest["models"][0]["batch_stress"]["rounds"] = 19
    with pytest.raises(ValueError, match="batch_stress"):
        matrix.validate_manifest(manifest)


def test_qwen36_mtp2_batch_oracle_accepts_only_uniform_lower_k_receipts():
    manifest = json.loads(
        (ROOT / "qualification" / "qwen36-overnight-experiments.json").read_text()
    )
    model = next(item for item in manifest["models"] if item["name"] == "qwen36-35b-a3b")
    arm = next(item for item in model["arms"] if item["name"] == "mtp2")
    requirements = {
        item["path"]: item for item in matrix.effective_arm(arm, "batch_stress")["receipt_requirements"]
    }

    assert requirements["mtp.route"]["equals"] == "segmented_self_mtp"
    assert requirements["mtp.requested_num_draft"]["equals"] == 2
    assert requirements["mtp.num_draft"]["equals"] == 1
    assert requirements["mtp.admission_stage"]["equals"] == "lower_k"
    assert requirements["mtp.stats.draft_proposed"]["gt"] == 0


def test_resume_skips_only_passed_spomin_cells_and_deduplicates_rows():
    report = {"cells": {"a:full": {"passed": True}, "a:compacted": {"passed": False}},
              "rows": [{"domain": "a", "transcript_arm": "full"},
                       {"domain": "a", "transcript_arm": "compacted"}]}
    assert spomin.completed_cell_ids(report) == {"a:full"}
    assert spomin.rows_for_completed_cells(report) == [{"domain": "a", "transcript_arm": "full"}]


def test_post_thermal_bracket_rejects_a_breached_state():
    sample = {"thermal_state": 0, "pmset_no_thermal_warning": True,
              "pmset_no_performance_warning": True, "pmset_no_cpu_power_warning": True,
              "battery_temperature_c": 31.0, "virtual_temperature_c": 46.0}
    with pytest.raises(AssertionError, match="breached"):
        matrix.validate_post_thermal(sample, {})


def test_loaded_host_service_stops_the_run_unless_quiesce_is_allowed(monkeypatch):
    services = [{"label": "com.example.model-server", "plist": "~/Library/LaunchAgents/com.example.model-server.plist"}]
    calls = []
    state = {"loaded": True}

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1] == "print":
            return types.SimpleNamespace(returncode=0 if state["loaded"] else 1)
        state["loaded"] = command[1] == "bootstrap"
        return types.SimpleNamespace(returncode=0)

    registered = []
    monkeypatch.setattr(matrix.subprocess, "run", fake_run)
    monkeypatch.setattr("atexit.register", registered.append)
    monkeypatch.setattr("signal.signal", lambda *_a: None)
    with pytest.raises(SystemExit, match="--quiesce-host-services"):
        matrix.quiesce_host_services(services, allowed=False)
    assert all(command[1] == "print" for command in calls)  # nothing was touched
    outcome = matrix.quiesce_host_services(services, allowed=True)
    assert outcome == [{"label": "com.example.model-server", "quiesced": True, "restored": None}]
    (restore,) = registered
    restore()
    assert outcome[0]["restored"] is True and state["loaded"] is True
    state["loaded"] = False
    assert matrix.quiesce_host_services(services, allowed=False) == []  # nothing resident: nothing to do


def test_canonical_manifests_name_real_services_selected_arms_and_new_arms():
    four = json.loads((ROOT / "qualification" / "four-model-experiments.json").read_text())
    matrix.validate_manifest(four)
    assert [service["label"] for service in four["host_services"]] == ["com.example.fn-uncensored-mlx-serve"]
    commands = [arm["activate_command"] for model in four["models"] for arm in model["arms"]]
    assert not any("--bootout-label" in command for command in commands)
    muse = next(model for model in four["models"] if model["name"] == "muse-glimmer")
    lookup = next(arm for arm in muse["arms"] if arm["name"] == "prompt-lookup")
    assert "--prompt-lookup" in lookup["activate_command"] and "--external-draft" not in lookup["activate_command"]
    assert {"path": "scheduler.pld_batched_rounds", "delta_gte": 1} in lookup["counter_requirements"]
    north = next(model for model in four["models"] if model["name"] == "north-mini-code")
    required = {item["path"]: item["equals"] for item in north["arms"][0]["status_requirements"] if "equals" in item}
    assert required["settings.thinking_steer.calibration.state"] == "calibrated"
    assert required["settings.thinking_steer.alpha"] == 0.2 and required["settings.thinking_budget"] == 512
    qwen = json.loads((ROOT / "qualification" / "qwen36-overnight-experiments.json").read_text())
    matrix.validate_manifest(qwen, allow_exploratory_single_run=True)
    assert qwen["models"][0]["selected_arm"] == "mtp-artifact-ordinary"
    broken = json.loads(json.dumps(qwen))
    broken["models"][0]["selected_arm"] = "no-such-arm"
    with pytest.raises(ValueError, match="selected_arm"):
        matrix.validate_manifest(broken, allow_exploratory_single_run=True)
