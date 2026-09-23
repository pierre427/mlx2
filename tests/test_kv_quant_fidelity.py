"""KV-quantization fidelity measurement and the selection gate (CPU)."""

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.approximate_kv import (
    KVQuantizationOperation,
    standard_kv_quantization_operations,
)
from mlx2.runtime.kv_quant_fidelity import (
    BUNDLE_SCHEMA,
    DEFAULT_THRESHOLDS,
    REPORT_SCHEMA,
    evaluate_fidelity_report,
    measure_teacher_forced,
    token_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def harness():
    spec = importlib.util.spec_from_file_location(
        "measure_kv_quant_fidelity", ROOT / "scripts" / "measure_kv_quant_fidelity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operation(name, fingerprint="tiny"):
    descriptor = standard_kv_quantization_operations(group_size=64)[name]
    return KVQuantizationOperation(name, descriptor, adapter_fingerprint=fingerprint)


def test_token_metrics_identity_and_disagreement():
    logits = mx.array([[0.0, 1.0, 3.0, -1.0, 2.0, 0.5]])
    same = token_metrics(logits, logits, mx.array([2]))
    assert float(same["kl"][0]) == pytest.approx(0.0, abs=1e-7)
    assert bool(same["agree"][0]) and float(same["top5_overlap"][0]) == 1.0
    assert float(same["logprob_delta"][0]) == pytest.approx(0.0, abs=1e-6)
    swapped = mx.array([[0.0, 1.0, -1.0, 3.0, 2.0, 0.5]])
    other = token_metrics(logits, swapped)
    assert float(other["kl"][0]) > 0.1 and not bool(other["agree"][0])


def test_teacher_forced_measurement_on_tiny_model_orders_operations():
    model = harness().tiny_model(8.0)
    stream = [(7 * i + 3) % 97 + 1 for i in range(200)]
    q8 = measure_teacher_forced(
        model, stream, context=96, score_tokens=24, operation=operation("kv_q8"),
        prefill_step=40,
    )
    k8v4 = measure_teacher_forced(
        model, stream, context=96, score_tokens=24, operation=operation("kv_k8v4"),
        prefill_step=40,
    )
    for result in (q8, k8v4):
        # Mechanism: both attention planes quantized, exact arm untouched.
        assert result["quantized_planes"] == 2
        assert result["exact_quantized_planes"] == 0
        assert result["scored_tokens"] == 24
        assert 0 < result["kv_bytes_ratio"] < 1
    assert 0 <= q8["kl_mean"] < k8v4["kl_mean"]
    assert k8v4["kl_mean"] > 0
    assert q8["kv_bytes_quant"] > k8v4["kv_bytes_quant"]


def test_measurement_refuses_a_wrong_mechanism_count():
    model = harness().tiny_model(1.0)
    stream = list(range(1, 80))
    from mlx2.runtime.models.cache import make_prompt_cache

    def already_quantized():
        cache = make_prompt_cache(model)
        return [c.to_quantized(group_size=64, bits=8) if hasattr(c, "to_quantized") else c
                for c in cache]

    with pytest.raises(RuntimeError, match="exact arm"):
        measure_teacher_forced(
            model, stream, context=32, score_tokens=4, operation=operation("kv_q8"),
            make_cache=already_quantized,
        )
    with pytest.raises(ValueError, match="shorter"):
        measure_teacher_forced(
            model, stream, context=78, score_tokens=4, operation=operation("kv_q8"),
        )


def passing_report(name="kv_q8", fingerprint="artifact-a"):
    limits = DEFAULT_THRESHOLDS[name]
    entry = {
        "scored_tokens": 256,
        "kl_mean": limits.kl_mean_max / 2,
        "kl_p99": limits.kl_p99_max / 2,
        "top1_agreement": min(1.0, limits.top1_agreement_min + 0.005),
        "kv_bytes_ratio": limits.bytes_ratio_max - 0.02,
        "quantized_planes": 16,
        "exact_quantized_planes": 0,
        "needles": {"total": 9, "exact_hits": 9, "quant_hits": 9, "quant_losses": 0},
    }
    return {
        "schema": REPORT_SCHEMA,
        "device": "gpu",
        "adapter_fingerprint": fingerprint,
        "operation": name,
        "contexts": [dict(entry, context=c) for c in (4096, 16384, 32768, 65536)],
    }


@pytest.mark.parametrize("name", ["kv_q8", "kv_k8v4"])
def test_gate_passes_a_report_inside_the_thresholds(name):
    verdict = evaluate_fidelity_report(
        passing_report(name), operation=name, adapter_fingerprint="artifact-a"
    )
    assert verdict["passed"], verdict["failures"]
    bundle = {"schema": BUNDLE_SCHEMA, "reports": {name: passing_report(name)}}
    assert evaluate_fidelity_report(bundle, operation=name)["passed"]


def mutate(path, value):
    report = passing_report()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return report


@pytest.mark.parametrize(
    "report, reason",
    [
        (mutate(("device",), "cpu"), "device"),
        (mutate(("operation",), "kv_k8v4"), "route selects"),
        (mutate(("adapter_fingerprint",), "artifact-b"), "another adapter"),
        (mutate(("contexts", 1, "kl_mean"), 0.01), "kl_mean"),
        (mutate(("contexts", 2, "kl_p99"), 0.5), "kl_p99"),
        (mutate(("contexts", 0, "top1_agreement"), 0.95), "top1_agreement"),
        (mutate(("contexts", 0, "kv_bytes_ratio"), 0.6), "kv_bytes_ratio"),
        (mutate(("contexts", 3, "quantized_planes"), 0), "mechanism count is 0"),
        (mutate(("contexts", 3, "exact_quantized_planes"), 2), "not exact"),
        (mutate(("contexts", 0, "scored_tokens"), 16), "scored tokens"),
        (mutate(("contexts", 0, "kl_mean"), float("nan")), "kl_mean missing"),
        (mutate(("contexts", 0, "needles", "quant_losses"), 1), "needle losses"),
        (mutate(("contexts",), []), "missing context 4096"),
        ({"schema": "something-else"}, "not a kv-quant"),
        # A malformed user-supplied file fails its check instead of raising.
        (mutate(("contexts", 0, "context"), "4k"), "not a token count"),
        (mutate(("contexts",), 5), "contexts is not a list"),
        (mutate(("contexts", 0, "needles"), [1]), "needles is not a mapping"),
        (mutate(("contexts", 0, "needles", "quant_losses"), float("inf")), "needle losses"),
        (mutate(("contexts", 0, "kl_mean"), 10**400), "kl_mean missing"),
        ({"schema": BUNDLE_SCHEMA, "reports": [1]}, "not a kv-quant"),
    ],
)
def test_gate_fails_closed(report, reason):
    verdict = evaluate_fidelity_report(
        report, operation="kv_q8", adapter_fingerprint="artifact-a"
    )
    assert not verdict["passed"]
    assert any(reason in failure for failure in verdict["failures"]), verdict["failures"]


def test_gate_rejects_unknown_operation_and_missing_bundle_member():
    assert not evaluate_fidelity_report(passing_report(), operation="kv_q3")["passed"]
    bundle = {"schema": BUNDLE_SCHEMA, "reports": {"kv_k8v4": passing_report("kv_k8v4")}}
    assert not evaluate_fidelity_report(bundle, operation="kv_q8")["passed"]


def test_cpu_harness_end_to_end_reports_counters_and_never_qualifies(tmp_path):
    out = tmp_path / "fidelity.json"
    module = harness()
    assert module.main([
        "--cpu-tiny", "--contexts", "48,96", "--score-tokens", "16",
        "--needles", "2", "--decode-tokens", "2", "--repeats", "1",
        "--prefill-step", "32", "--out", str(out),
    ]) == 0
    bundle = json.loads(out.read_text())
    assert bundle["schema"] == BUNDLE_SCHEMA
    assert set(bundle["reports"]) == {"kv_q8", "kv_k8v4"}
    for name, report in bundle["reports"].items():
        assert report["device"] == "cpu" and report["operation"] == name
        for entry in report["contexts"]:
            assert entry["quantized_planes"] > 0 and entry["exact_quantized_planes"] == 0
            assert entry["needles"]["total"] == 2
            assert entry["decode"]["exact_tok_s_median"] > 0
            assert entry["decode"]["quant_tok_s_median"] > 0
        # A CPU tiny report is harness validation, never selection evidence.
        verdict = bundle["verdicts"][name]
        assert not verdict["passed"]
        assert any("device" in failure for failure in verdict["failures"])


def test_gpu_mode_is_gated(capsys):
    module = harness()
    with pytest.raises(SystemExit, match="i-own-the-gpu"):
        module.main(["--model", "/nonexistent"])
    assert module.main(["--model", "/nonexistent", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["contexts"] == [4096, 16384, 32768, 65536]
    assert plan["needle_contexts"] == "4096,32768"


def test_qualifier_observes_fidelity_verdict_and_mtp_lanes():
    spec = importlib.util.spec_from_file_location(
        "qualify_serving", ROOT / "scripts" / "qualify_serving.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    final = {"approximate_kv": {"applied": 2, "mtp_lanes": 2}}
    observed = module.feature_observations(final, {"passed": True})
    assert observed["approximate_kv_fidelity"] == 1
    assert observed["approximate_kv_mtp"] == 2
    assert module.feature_observations(final, {"passed": False})["approximate_kv_fidelity"] == 0
    assert module.feature_observations(final)["approximate_kv_fidelity"] == 0
    assert module.feature_observations({})["approximate_kv_mtp"] == 0


def test_serving_bench_is_gated_and_refuses_null_arms():
    spec = importlib.util.spec_from_file_location(
        "bench_kv_quant_mtp", ROOT / "scripts" / "bench_kv_quant_mtp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit, match="i-own-the-gpu"):
        module.main(["--model", "/nonexistent"])
    assert module.arm_config("mtp_kv_q8", "kv_q8") == (
        True, {"operation": "kv_q8", "enabled": True, "compose_mtp": True}
    )
    assert module.arm_config("ordinary_exact", "kv_q8") == (False, None)
    policy = {"operation": "kv_q8", "enabled": True, "compose_mtp": True}
    good = {
        "arm": "mtp_kv_q8",
        "approximate_kv": {"applied": 1, "mtp_lanes": 1},
        "segmented": {"quantized_kv_segmented_attention_calls": 4,
                      "true_batched_declined": 0},
    }
    module.check_mechanism(good, policy, True)
    for field, value, reason in (
        ("approximate_kv", {"applied": 0, "mtp_lanes": 0}, "applied == 0"),
        ("approximate_kv", {"applied": 1, "mtp_lanes": 0}, "mtp_lanes == 0"),
        ("segmented", {"quantized_kv_segmented_attention_calls": 0}, "never ran"),
        ("segmented", {"quantized_kv_segmented_attention_calls": 4,
                       "true_batched_declined": 2}, "declined"),
    ):
        with pytest.raises(RuntimeError, match=reason):
            module.check_mechanism(dict(good, **{field: value}), policy, True)
    with pytest.raises(RuntimeError, match="exact arm"):
        module.check_mechanism(
            {"arm": "mtp_exact", "approximate_kv": {"applied": 1}}, None, True
        )


def merger():
    spec = importlib.util.spec_from_file_location(
        "merge_kv_quant_fidelity", ROOT / "scripts" / "merge_kv_quant_fidelity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_bundles(contexts_per_part, name="kv_q8"):
    full = passing_report(name)
    full.update(operation_revision="rev", descriptor={"bits": 8}, corpus_sha256="c")
    parts = []
    for contexts in contexts_per_part:
        report = dict(full, contexts=[e for e in full["contexts"] if e["context"] in contexts])
        parts.append({"schema": BUNDLE_SCHEMA, "harness": {"sha256": "h"},
                      "reports": {name: report}, "verdicts": {}})
    return parts


def test_merge_of_per_context_bundles_rederives_the_verdict(tmp_path):
    module = merger()
    parts = split_bundles([(4096,), (16384,), (32768, 65536)])
    # Each part alone is missing required contexts and fails closed.
    for part in parts:
        assert not evaluate_fidelity_report(part, operation="kv_q8")["passed"]
    paths = []
    for index, part in enumerate(parts):
        path = tmp_path / f"part{index}.json"
        path.write_text(json.dumps(part))
        paths.append(str(path))
    out = tmp_path / "merged.json"
    assert module.main([*paths, "--out", str(out)]) == 0
    merged = json.loads(out.read_text())
    assert merged["verdicts"]["kv_q8"]["passed"], merged["verdicts"]
    assert [e["context"] for e in merged["reports"]["kv_q8"]["contexts"]] == [
        4096, 16384, 32768, 65536]
    assert evaluate_fidelity_report(merged, operation="kv_q8",
                                    adapter_fingerprint="artifact-a")["passed"]


def test_merge_fails_closed_on_mismatched_or_duplicate_parts():
    module = merger()
    parts = split_bundles([(4096,), (4096, 16384)])
    with pytest.raises(ValueError, match="measured twice"):
        module.merge(parts)
    parts = split_bundles([(4096,), (16384,)])
    parts[1]["reports"]["kv_q8"]["adapter_fingerprint"] = "artifact-b"
    with pytest.raises(ValueError, match="adapter_fingerprint"):
        module.merge(parts)
    parts = split_bundles([(4096,), (16384,)])
    parts[1]["harness"] = {"sha256": "other"}
    with pytest.raises(ValueError, match="harness"):
        module.merge(parts)
    # Parts may measure different operation subsets (one long context split
    # per operation); each operation is judged on the parts that measured it.
    q8 = split_bundles([(4096, 16384), (32768,)])
    k84 = split_bundles([(4096, 16384), (32768,)], name="kv_k8v4")
    q8[0]["reports"]["kv_k8v4"] = k84[0]["reports"]["kv_k8v4"]
    merged = module.merge([q8[0], q8[1], k84[1]])
    assert merged["verdicts"]["kv_q8"]["passed"], merged["verdicts"]["kv_q8"]
    assert merged["verdicts"]["kv_k8v4"]["passed"], merged["verdicts"]["kv_k8v4"]
    # A failing context in any part fails the merged verdict.
    parts = split_bundles([(4096,), (16384,), (32768,)])
    parts[1]["reports"]["kv_q8"]["contexts"][0]["kl_mean"] = 0.5
    merged = module.merge(parts)
    assert not merged["verdicts"]["kv_q8"]["passed"]
    assert any("context 16384" in f for f in merged["verdicts"]["kv_q8"]["failures"])


def test_needle_answer_len_covers_digit_split_tokenizers():
    module = harness()

    def encode(text):  # Qwen-style: lone leading space, one token per digit
        out = []
        for ch in text:
            out.append(ch)
        return out

    # " 042917." -> 8 tokens; the answer window must hold all six digits.
    assert module.needle_answer_len(encode, "042917") >= 1 + 6


def test_gate_rejects_vacuous_needle_checks():
    report = passing_report()
    report["contexts"][0]["needles"] = {
        "total": 9, "exact_hits": 0, "quant_hits": 0, "quant_losses": 0}
    verdict = evaluate_fidelity_report(report, operation="kv_q8")
    assert not verdict["passed"]
    assert any("vacuous" in f for f in verdict["failures"]), verdict["failures"]
