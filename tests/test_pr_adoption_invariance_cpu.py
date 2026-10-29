"""Synthetic diagnostic matrix integrity; no model qualification."""
import copy
import importlib.util
from pathlib import Path

import pytest

from mlx2.generation_trace import GeneratedTokenTrace

_spec = importlib.util.spec_from_file_location("invariance_adoption",
    Path(__file__).parents[1] / "scripts/research/pr_adoption_invariance.py")
matrix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matrix)


def cells():
    trace = GeneratedTokenTrace()
    trace.append(7, 1, 1)
    return [{**shape, "checkpoint": "synthetic@fixture", "source_revision": "a" * 40,
             "numerical_profile": "same-law", "runtime_identity": "build",
             "cache_bytes": 4096, "persistence": True, "completion_tokens": 1,
             "logical_request_id": "tracked-request", "prompt_sha256": "e" * 64,
             "generation_config_sha256": "f" * 64,
             "token_trace": trace.receipt(), "functional_passed": True,
             "route_observed": True, "state_sha256": "b" * 64, "logits_sha256": "c" * 64}
            for shape in matrix.plan()]


def test_complete_exact_matrix_is_not_qualification_and_missing_is_not_pass():
    rows = cells()
    assert len(rows) == 144
    result = matrix.evaluate(rows)
    assert result["exact_matrix_passed"] and not result["qualified"]
    assert not matrix.evaluate(rows[:-1])["exact_matrix_passed"]


@pytest.mark.parametrize("field,value", [
    ("checkpoint", "other@fixture"), ("source_revision", "d" * 40),
    ("cache_bytes", 8192), ("persistence", False), ("numerical_profile", "different-law"),
    ("state_sha256", None), ("logits_sha256", "absent"), ("route_observed", False),
    ("prompt_sha256", "d" * 64), ("generation_config_sha256", "d" * 64),
])
def test_mismatched_bindings_or_absent_proof_are_refused(field, value):
    rows = cells()
    rows[-1][field] = value
    with pytest.raises(ValueError):
        matrix.evaluate(rows)


def test_first_divergence_is_localized_and_state_mismatch_cannot_pass():
    rows = cells()
    rows[-1]["token_trace"]["token_ids"] = [8]
    report = matrix.evaluate(rows)
    assert not report["exact_matrix_passed"]
    assert report["comparisons"][-1]["first_divergence_output_token"] == 0
    rows = cells()
    rows[-1]["state_sha256"] = "d" * 64
    report = matrix.evaluate(rows)
    assert not report["exact_matrix_passed"]
    assert report["comparisons"][-1]["first_divergence_output_token"] is None


def test_duplicate_cell_is_refused():
    with pytest.raises(ValueError, match="duplicate"):
        matrix.evaluate(cells() + [copy.deepcopy(cells()[0])])


@pytest.mark.parametrize("count", [None, True, -1, "1"])
def test_missing_or_noninteger_completion_count_is_refused(count):
    rows = cells()
    rows[-1]["completion_tokens"] = count
    with pytest.raises(ValueError, match="count"):
        matrix.evaluate(rows)
