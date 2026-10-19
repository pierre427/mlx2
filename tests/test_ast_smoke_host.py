"""Host-only guards for the bounded AST diagnostic harness."""

import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / (
    "qualification/runs/ast-smokes-20261018/ast_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("ast_smoke_20261018", PATH)
AST_SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AST_SMOKE)


def test_skip_b4_removes_both_cold_and_warm_phases():
    phases = AST_SMOKE.ast_phase_plan(
        {"short_explain": "prompt"}, skip_b4=True, skip_sampled=True
    )
    assert phases == {"b1": {"short_explain": "prompt"}}


def test_b4_plan_names_both_real_batch_phases():
    phases = AST_SMOKE.ast_phase_plan(
        {"a": "one", "b": "two", "c": "three", "d": "four"}
    )
    assert phases["b4_cold"] is True
    assert phases["b4_warm"] is True
    assert phases["sampled"] is True
