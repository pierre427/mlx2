"""Host-only campaign launch and attribution guards."""

import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_ast_repair.py"
SPEC = importlib.util.spec_from_file_location("ast_repair_benchmark", PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_ast_arm_binds_execution_policy_flag():
    for arm in ("before", "fixed"):
        assert BENCH.arm_arguments(arm, Path("policy.json")) == [
            "--external-draft", "--execution-policy", "policy.json"]
    assert BENCH.arm_arguments("ordinary", Path("policy.json")) == ["--ordinary"]


def test_reference_stability_is_per_prompt():
    report = {"phases": {f"b1_r{i+1}_{label}": {"decode_tokens_per_second": rate}
                         for label, rates in {"fast": [33, 33.2, 33.1], "slow": [15, 15.1, 15.2]}.items()
                         for i, rate in enumerate(rates)}}
    assert not BENCH.unstable_reference(report, ["fast", "slow"], 3)
    report["phases"]["b1_r2_fast"]["decode_tokens_per_second"] = 23
    assert set(BENCH.unstable_reference(report, ["fast", "slow"], 3)) == {"fast"}
