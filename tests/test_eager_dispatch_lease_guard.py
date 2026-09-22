"""Eager qualifier lease tests never import MLX or read real GPU locks."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "eager_lease_guard", Path(__file__).parents[1] / "scripts/qualify_eager_dispatch.py"
)
qualifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualifier)


@pytest.fixture
def lease(tmp_path, monkeypatch):
    paths = (tmp_path / "lock-a.json", tmp_path / "lock-b.json")
    monkeypatch.setattr(qualifier, "LOCK_RECEIPTS", paths)
    value = dict(campaign_id="campaign", session_id="caller", lease_id="lease", generation=3,
                 owner="agent", resource_key="gpu", mode="exclusive", expires_at=2000)
    for path in paths:
        path.write_text(json.dumps(value))
    return value, paths


def check(value):
    return qualifier._load_lock_receipts(expected=value, session_id="caller", owner="agent", now=1000)


def test_matching_owned_current_lease(lease):
    value, _ = lease
    assert check(value) == [value, value]


@pytest.mark.parametrize("field,replacement", [("session_id", "foreign"), ("owner", "foreign"),
    ("generation", 4), ("campaign_id", "foreign"), ("lease_id", "foreign"),
    ("resource_key", "cpu"), ("mode", "shared"), ("status", "released"),
    ("expires_at", 1000), ("expires_at", None), ("expires_at", float("inf")),
    ("expires_at", True)])
def test_foreign_replaced_expired_or_incomplete_locks_fail(lease, field, replacement):
    value, paths = lease
    # Even two mutually matching global receipts do not establish ownership.
    changed = dict(value, **{field: replacement})
    for path in paths:
        path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError):
        check(value)


@pytest.mark.parametrize("field,replacement", [("session_id", "foreign"), ("owner", "foreign"),
    ("expires_at", 0), ("expires_at", None), ("generation", True)])
def test_independent_expected_receipt_also_must_be_current_and_owned(lease, field, replacement):
    value, _ = lease
    with pytest.raises(RuntimeError):
        check(dict(value, **{field: replacement}))


def test_missing_receipt_fails(lease):
    value, paths = lease
    paths[1].unlink()
    with pytest.raises(RuntimeError, match="missing"):
        check(value)


def test_expiry_rechecked_at_each_forward_before_model_work():
    touched = []
    def expired():
        raise RuntimeError("expired")
    with pytest.raises(RuntimeError, match="expired"):
        qualifier._forward(None, lambda *a, **k: touched.append(True), [], [], lease_check=expired)
    assert touched == []


def test_cli_refuses_foreign_lease_before_tensor_import(lease, monkeypatch, tmp_path):
    value, _ = lease
    expected = tmp_path / "receipt.json"
    expected.write_text(json.dumps(value))
    monkeypatch.setattr(sys, "argv", ["qualify", "--model", "unused", "--output", "unused",
        "--gpu-lease-receipt", str(expected), "--session-id", "foreign", "--owner", "agent"])
    with pytest.raises(RuntimeError, match="another caller"):
        qualifier.main()
