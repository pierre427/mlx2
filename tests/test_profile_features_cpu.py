"""CPU tests for the HTTP A/B profiler's spec plumbing (no servers, no weights)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("profile_features", ROOT / "scripts" / "profile_features.py")
pf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pf)

AB_DIR = ROOT / "qualification" / "runs" / "ab-20261007"
MODELS = Path.home() / "mlx-models"


# ---------------------------------------------------------------- request body


def test_seeded_sampling_is_per_request_and_arm_independent():
    a = pf.Client(1, {"temperature": 0.7, "seed": 11})
    b = pf.Client(2, {"temperature": 0.7, "seed": 11})
    msgs = pf.user("hello")
    body = a.request_body(msgs, max_tokens=8)
    assert body["temperature"] == 0.7
    assert body["seed"] == b.request_body(msgs, max_tokens=8)["seed"]
    assert body["seed"] != a.request_body(pf.user("other"), max_tokens=8)["seed"]
    # An explicit per-call temperature wins over the spec's.
    assert a.request_body(msgs, max_tokens=8, temperature=0.0)["temperature"] == 0.0


def test_default_body_is_greedy_and_unseeded():
    body = pf.Client(1).request_body(pf.user("x"), max_tokens=4)
    assert body["temperature"] == 0.0
    assert "seed" not in body


# ---------------------------------------------------------------- agents


def test_agent_schedule_is_seeded_covers_sessions_and_is_skewed():
    order = pf.agent_schedule(6, 60, 3)
    assert order == pf.agent_schedule(6, 60, 3)
    assert len(order) == 60
    assert sorted(order[:6]) == list(range(6))  # every session's cold turn first
    assert order.count(0) > order.count(5)


def test_agent_value_uses_the_runs_cold_rate():
    reqs = [
        # cold: 2 ms/token
        {"ttft_s": 2.0, "prompt_tokens": 1000, "cached_tokens": 0, "error": None},
        {"ttft_s": 4.0, "prompt_tokens": 2000, "cached_tokens": None, "error": None},
        # warm: 900 of 1100 cached, observed 0.5 s against a 2.2 s cold estimate
        {"ttft_s": 0.5, "prompt_tokens": 1100, "cached_tokens": 900, "error": None},
    ]
    v = pf.agent_value(reqs, resident_byte_seconds=2 * (1 << 30) * 10.0, span_s=10.0)
    assert v["cold_prefill_ms_per_token"] == pytest.approx(2.0)
    assert v["saved_prefill_ms"] == pytest.approx(1800.0)
    assert v["resident_gib_s"] == pytest.approx(20.0)
    assert v["resident_mean_gib"] == pytest.approx(2.0)
    assert v["saved_ms_per_resident_gib_s"] == pytest.approx(90.0)
    assert v["saved_ttft_ms"] == pytest.approx(1700.0)


def test_status_sampler_integrates_trapezoids():
    sampler = pf.StatusSampler(client=None, suffix="x", interval=1)
    sampler.samples = [(0.0, 0.0), (1.0, 2.0), (3.0, 2.0)]
    assert sampler.integral() == (pytest.approx(5.0), pytest.approx(3.0))


# ---------------------------------------------------------------- speculation


def test_speculation_counts_external_and_native():
    ext = {"speculation": {"external_rounds": 10, "accepted": 25}}
    assert pf.speculation_counts(ext) == (10, 25, 35)
    mtp = {"mtp": {"stats": {"cycles": 4, "draft_accepted": 5, "retrieval_accepted": 1,
                             "total_emitted": 10}}}
    assert pf.speculation_counts(mtp) == (4, 6, 10)
    assert pf.speculation_counts({"speculation": None, "mtp": None}) is None
    summary = pf.speculation_summary([{"receipt": ext}, {"receipt": mtp}, {"receipt": {}}])
    assert summary["verify_rounds"] == 14
    assert summary["accepted_per_verify"] == pytest.approx(31 / 14)
    assert summary["tokens_per_verify"] == pytest.approx(45 / 14)


def test_select_timing_histogram_percentiles():
    prefix = "status.execution.qsa_stage1.candidates.select_timing.producers"
    delta = {f"{prefix}.gvr_exact.buckets.le_100": 90,
             f"{prefix}.gvr_exact.buckets.le_2000": 9,
             f"{prefix}.gvr_exact.buckets.le_9000": 1,
             f"{prefix}.radix_exact.buckets.le_500": 4,
             f"{prefix}.gvr_exact.count": 100}
    out = pf.histogram_percentiles(delta, prefix)
    assert out["gvr_exact"] == {"count": 100, "p50_ms": 0.1, "p99_ms": 2.0}
    assert out["radix_exact"]["p99_ms"] == 0.5
    res = {"requests": [{}], "captured": {"select_timing": out}}
    assert pf.metric_of(res, "select_p99") == 2.0


def test_summary_reports_repeatability_across_reps():
    def row(arm, rep, shas):
        return {"arm": arm, "rep": rep, "workloads": {"serial:2": {
            "requests": [{"sha": s, "ttft_s": 0.1, "decode_tps": 10.0} for s in shas],
            "engaged": {}}}}

    results = [row("a", 0, "xy"), row("b", 0, "xz"), row("a", 1, "xy"), row("b", 1, "xq")]
    table = pf.summarize(results, ["a", "b"], ["serial:2"])["workloads"]["serial:2"]
    assert table["a"]["identical_across_reps"] == "2/2"
    assert table["b"]["identical_across_reps"] == "1/2"
    assert table["b"]["identical_vs_base"] == "2/4"
    assert table["a"]["tps_all_median"] == 10.0


# ---------------------------------------------------------------- dry run


def test_workload_checks():
    assert pf._check_workload("agents:6:6000:36", 32768) is None
    assert pf._check_workload("serial:8", None) is None
    assert "unknown workload" in pf._check_workload("nope:1", None)
    assert "max-context" in pf._check_workload("long:32000", 32768)
    assert pf._check_workload("long:16000", 32768) is None


def _fake_qwen38(monkeypatch, tmp_path):
    from mlx2.adapters import qwen38_27b, registry

    model = tmp_path / "target"
    model.mkdir()
    (model / "config.json").write_text("{}")
    draft = tmp_path / "draft"
    draft.mkdir()
    resolution = registry.AdapterResolution(
        qwen38_27b.Qwen3827BAdapter, qwen38_27b.QWEN38_27B, {})
    monkeypatch.setattr(registry, "inspect_model", lambda path: resolution)
    binding = qwen38_27b.Qwen3827BAdapter.default_external_route_binding
    policy = {"draft_model": str(draft), **binding}
    return str(model), policy


def test_dry_run_refuses_block_verification_on_the_default_tree(monkeypatch, tmp_path):
    model, policy = _fake_qwen38(monkeypatch, tmp_path)
    spec = {"model": model, "common_args": ["--max-lanes", "4"],
            "workloads": ["serial:2"],
            "arms": {"tree_block": {"args": ["--external-draft"],
                                    "policy": {**policy, "exact_verification": "block"}},
                     "chain_block": {"args": ["--external-draft"],
                                     "policy": {**policy, "exact_verification": "block",
                                                "batch_size_route": None,
                                                "proposal_composition": False}}}}
    report = pf.dry_run(spec)
    assert not report["ok"]
    assert any("block verification" in e for e in report["arms"]["tree_block"]["errors"])
    chain = report["arms"]["chain_block"]
    assert chain["errors"] == []
    assert chain["route"] == "external_draft"
    assert chain["resolved_policy"]["batch_size_route"] is None
    assert "tree_node_budget_by_lanes" not in chain["resolved_policy"]


def test_dry_run_catches_flags_env_and_server_policy(monkeypatch, tmp_path):
    model, _ = _fake_qwen38(monkeypatch, tmp_path)
    spec = {"model": model, "common_args": [], "workloads": ["short"],
            "arms": {
                "bad_flag": {"args": ["--no-such-flag"]},
                "bad_env": {"args": ["--native-mtp"], "env": {"MLX2_NO_SUCH_KNOB": "1"}},
                "cleared_env": {"args": ["--native-mtp"],
                                "env": {"MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR": "gvr"}},
                "bad_selector": {"args": ["--native-mtp"],
                                 "env": {"MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR": "gvrr"}},
                "bad_retention": {"args": ["--native-mtp"],
                                  "policy": {"apc_retention_policy": "valu"}},
                "handoff_off_route": {"args": ["--ordinary"],
                                      "policy": {"mtp_ordinary_handoff":
                                                 {"enabled": True, "max_mtp_width": 3}}},
                "good": {"args": ["--native-mtp"], "policy": {"apc_retention_policy": "value"},
                         "env": {"MLX2_LOOP_TRACE": "1"}},
            }}
    arms = pf.dry_run(spec)["arms"]
    assert "argparse" in arms["bad_flag"]["errors"][0]
    assert "not read anywhere" in arms["bad_env"]["errors"][0]
    assert "cleared by the model adapter" in arms["cleared_env"]["errors"][0]
    assert any("DIRECT_SELECTOR must be one of" in e for e in arms["bad_selector"]["errors"])
    assert "apc_retention_policy" in arms["bad_retention"]["errors"][0]
    assert arms["handoff_off_route"]["errors"]
    assert arms["good"]["errors"] == []
    assert arms["good"]["resolved_policy"]["apc_retention_policy"] == "value"


def test_dry_run_rejects_unknown_sampling_and_missing_model(tmp_path):
    spec = {"model": str(tmp_path / "missing"), "sampling": {"temprature": 1},
            "workloads": ["short"], "arms": {"a": {}}}
    report = pf.dry_run(spec)
    assert not report["ok"]
    assert len(report["spec_errors"]) == 2


@pytest.mark.skipif(not MODELS.exists() or not AB_DIR.exists(),
                    reason="needs ~/mlx-models and the ab-20261007 specs")
@pytest.mark.parametrize("path", sorted(AB_DIR.glob("*-spec.json")), ids=lambda p: p.name)
def test_ab_20261007_specs_validate(path):
    report = pf.dry_run(json.loads(path.read_text()))
    if report["spec_errors"] and all(
        "has no config.json" in error for error in report["spec_errors"]
    ):
        pytest.skip("requires the model artifacts bound by the retained A/B spec")
    assert report["ok"], json.dumps(report, indent=1, default=str)


def test_dry_run_refuses_env_the_adapter_clears():
    # 2026-10-07: the GVR A/B set MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR in the
    # arm env; Flash-Next's configure_environment cleared it and both arms ran
    # radix.  The dry run must say so before any GPU time is spent.
    problems = pf._check_env(
        {"MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR": "gvr"},
        '"MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR"',
    )
    assert any("cleared by the model adapter" in p for p in problems)
    assert pf._check_env({"MLX2_LOOP_TRACE": "/tmp/t"}, '"MLX2_LOOP_TRACE"') == []
