"""Regression tests from the 2026-10-06 sweep: Qwen adapter identity and hooks."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path

import pytest

MODELS = Path.home() / "mlx-models"
QWEN36_27B = MODELS / "Qwen3.6-27B-MLX-8bit"


def _self_stores(func) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and getattr(node.value, "id", None) == "self"
    }


def test_qwen35_122b_lane_policy_defaults_does_not_crash():
    """Serving calls lane_policy_defaults() under the default --lane-matmul
    auto; the 122B adapter inherits it but has no external-draft route."""
    from mlx2.adapters.qwen35_122b import Qwen35122BA10BAdapter as C

    stores = _self_stores(C._init_qwen35_122b)
    assert {"external_policy", "draft_model"} <= stores
    adapter = object.__new__(C)
    for name in stores:
        setattr(adapter, name, None)
    adapter._kernels, adapter._tables = {}, []
    assert adapter.lane_policy_defaults() is None


@pytest.mark.skipif(not QWEN36_27B.is_dir(), reason="Qwen3.6-27B artifact absent")
def test_resaved_qwen36_27b_keeps_its_family(tmp_path):
    """A byte-different but identical Qwen3.6-27B config must not be served
    with the Qwen3.8 identity and Qwen3.8-measured defaults."""
    from mlx2.adapters.registry import inspect_model

    for item in QWEN36_27B.iterdir():
        if item.name != "config.json":
            (tmp_path / item.name).symlink_to(item)
    config = json.loads((QWEN36_27B / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config, indent=1))
    assert inspect_model(QWEN36_27B).descriptor.family == "qwen3.6-27b"
    assert inspect_model(tmp_path).descriptor.family == "qwen3.6-27b"


# rfix-kad 2026-10-07: a chat-template hash is presentation, not weights.
QWEN38_MTP = MODELS / "Qwen3.8-27B-oQ4e-mtp"
QWEN38_DENSE = MODELS / "Qwen3.8-27B-MLX-8bit"


def _with_qwen36_template(source: Path, target: Path) -> Path:
    """``source`` by symlink, with a copy of the Qwen3.6 chat template."""
    for item in source.iterdir():
        if item.name != "chat_template.jinja":
            (target / item.name).symlink_to(item)
    (target / "chat_template.jinja").write_bytes(
        (QWEN36_27B / "chat_template.jinja").read_bytes()
    )
    return target


@pytest.mark.skipif(not QWEN36_27B.is_dir(), reason="Qwen3.6-27B artifact absent")
def test_qwen36_identity_records_its_evidence(tmp_path):
    from mlx2.adapters.qwen36_27b import identity_evidence
    from mlx2.adapters.registry import inspect_model

    attested = inspect_model(QWEN36_27B)
    assert attested.descriptor.metadata["identity_evidence"] == "config_revision"
    assert attested.artifact["identity_evidence"] == "config_revision"
    for item in QWEN36_27B.iterdir():
        if item.name != "config.json":
            (tmp_path / item.name).symlink_to(item)
    config = json.loads((QWEN36_27B / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config, indent=1))
    assert identity_evidence(tmp_path) == "chat_template"
    resaved = inspect_model(tmp_path)
    assert resaved.descriptor.metadata["identity_evidence"] == "chat_template"


@pytest.mark.skipif(
    not (QWEN36_27B.is_dir() and QWEN38_MTP.is_dir()), reason="artifacts absent"
)
def test_qwen38_mtp_artifact_with_qwen36_template_stays_qwen38(tmp_path):
    from mlx2.adapters.qwen36_27b import identity_evidence
    from mlx2.adapters.registry import inspect_model
    from mlx2.contracts import Capability

    path = _with_qwen36_template(QWEN38_MTP, tmp_path)
    assert identity_evidence(path) is None
    resolved = inspect_model(path)
    assert resolved.descriptor.family == "qwen3.8-27b"
    assert Capability.MTP in resolved.descriptor.capabilities
    assert resolved.artifact["has_mtp"] is True


@pytest.mark.skipif(
    not (QWEN36_27B.is_dir() and QWEN38_DENSE.is_dir()), reason="artifacts absent"
)
def test_qwen38_dense_artifact_with_qwen36_template_is_template_evidence(tmp_path):
    from mlx2.adapters.registry import inspect_model

    resolved = inspect_model(_with_qwen36_template(QWEN38_DENSE, tmp_path))
    assert resolved.descriptor.family == "qwen3.6-27b"
    assert resolved.descriptor.metadata["identity_evidence"] == "chat_template"


def test_adapter_descriptor_and_status_carry_the_evidence(monkeypatch, tmp_path):
    from mlx2.adapters import qwen36_27b
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime.models import import_env

    artifact = {
        "identity": {"path": str(tmp_path), "fingerprint": "x", "files": []},
        "config": {}, "has_mtp": False, "weight_map": {}, "mtp_path": None,
        "identity_evidence": "chat_template",
    }
    monkeypatch.setattr(
        qwen36_27b.Qwen3627BAdapter, "artifact_inspector",
        staticmethod(lambda *_a, **_k: dict(artifact)),
    )
    seen = {}

    class Stop(Exception):
        pass

    def stop(_owner):
        raise Stop

    monkeypatch.setattr(import_env, "assert_profile_applied", stop)
    adapter = object.__new__(qwen36_27b.Qwen3627BAdapter)
    with pytest.raises(Stop):
        adapter._init_qwen38(str(tmp_path))
    assert adapter.descriptor.family == "qwen3.6-27b"
    assert adapter.descriptor.metadata["identity_evidence"] == "chat_template"
    monkeypatch.setattr(Qwen3827BAdapter, "diagnostics", lambda self: {"layout": "x"})
    assert adapter.diagnostics()["identity_evidence"] == "chat_template"
