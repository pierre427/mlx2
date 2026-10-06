"""Import-only continuation receipt accounting regressions."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

from mlx2.runtime.continuation_strategy import ContinuationStrategy
from mlx2.runtime.proposal_providers import (
    CONTINUATION_VERIFICATION_ALGORITHM,
    LONGEST_FIRST_EXACT_PREFIX,
    ContinuationPoolPolicy,
)

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mlx2"
    / "runtime"
    / "external_speculative.py"
)


def _generator_class():
    tree = ast.parse(SOURCE.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ExternalDraftBatchGenerator"
    )


def _generator_method(name):
    """Load one production host method without importing the Metal runtime."""
    generator = _generator_class()
    method = next(
        node
        for node in generator.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    harness = ast.ClassDef(
        name="ReceiptHarness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {"copy": copy}
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return getattr(namespace["ReceiptHarness"], name), method


def _receipt_lane(**updates):
    values = {
        "session_scope_hash": "request-scope",
        "continuation_rounds": 1,
        "continuation_verification_algorithm": LONGEST_FIRST_EXACT_PREFIX,
        "continuation_strategy": "longest_first_exact_prefix_v1",
        "continuation_strategy_launches": 2,
        "continuation_pruned_siblings": 3,
        "continuation_shared_prefix_reused_tokens": 1,
        "continuation_shared_prefix_tokens_reused": 1,
        "continuation_target_rows": 7,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _receipt_generator():
    generator = SimpleNamespace()
    generator.continuation_policy = ContinuationPoolPolicy.from_value({})
    generator.continuation_strategy = ContinuationStrategy(
        algorithm="longest_first_exact_prefix_v1",
        prune_incompatible_siblings=True,
        shared_prefix_reuse=True,
        cache_layout="test-layout",
        routed_experts_per_token=2,
        routed_moe_layers=4,
    )
    generator.draft = SimpleNamespace(
        proposal_pool=SimpleNamespace(receipt=lambda _scope: {"feedback_revision": 1})
    )
    return generator


def test_receipt_separates_configured_policy_from_executed_algorithm():
    receipt_method, _node = _generator_method("_continuation_receipt")
    receipt = receipt_method(_receipt_generator(), _receipt_lane())["continuation_pool"]

    assert (
        receipt["configured_verification_algorithm"]
        == CONTINUATION_VERIFICATION_ALGORITHM
    )
    assert receipt["verification_algorithm"] == LONGEST_FIRST_EXACT_PREFIX
    assert receipt["strategy_algorithm"] == "longest_first_exact_prefix_v1"
    assert receipt["strategy"]["algorithm"] == receipt["strategy_algorithm"]


def test_receipt_publishes_each_pruned_sibling_once():
    receipt_method, _node = _generator_method("_continuation_receipt")
    receipt = receipt_method(
        _receipt_generator(), _receipt_lane(continuation_pruned_siblings=3)
    )["continuation_pool"]

    assert receipt["pruned_siblings"] == 3
    assert receipt["strategy"]["sibling_proposals_pruned"] == 3


def test_pool_round_does_not_precommit_the_pruned_sibling_counter():
    generator = _generator_class()
    writers = []
    for method in generator.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(method):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Attribute)
                and target.attr == "continuation_pruned_siblings"
                for target in targets
            ):
                writers.append(method.name)

    scheduler_keys = [
        node
        for node in ast.walk(generator)
        if isinstance(node, ast.Constant)
        and node.value == "external_continuation_prefix_pruned"
    ]
    assert writers == ["_commit"]
    assert len(scheduler_keys) == 1


def test_permanent_ordinary_fallback_retains_continuation_history():
    """The final target-only receipt stays authoritative after pool execution."""
    generator = _generator_class()
    ordinary_round = next(
        node
        for node in generator.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ordinary_round"
    )
    receipts = []
    for node in ast.walk(ordinary_round):
        if not isinstance(node, ast.Dict):
            continue
        fields = {
            key.value: value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if (
            isinstance(fields.get("current_execution"), ast.Constant)
            and fields["current_execution"].value == "ordinary_target"
            and isinstance(fields.get("ordinary_fallback"), ast.Constant)
            and fields["ordinary_fallback"].value is True
        ):
            receipts.append(node)

    assert len(receipts) == 1
    expansions = [
        value
        for key, value in zip(receipts[0].keys, receipts[0].values, strict=True)
        if key is None
    ]
    assert any(
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "self"
        and value.func.attr == "_continuation_receipt"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "lane"
        for value in expansions
    )
