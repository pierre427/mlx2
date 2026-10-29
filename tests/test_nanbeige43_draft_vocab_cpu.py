"""Proposer-only shortlist admission and generic composition declarations."""
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlx2.adapters import nanbeige43_draft_vocab as vocab
from mlx2.runtime.approximate_kv import external_composition_contract


@pytest.fixture(autouse=True)
def binding(record_property):
    record_property("checkpoint", "synthetic-nanbeige43-vocab@sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record_property("source_revision", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    record_property("working_source_sha256", hashlib.sha256(Path(vocab.__file__).read_bytes()).hexdigest())


@pytest.mark.parametrize("value", [{"rows": True}, {"rows": 0}, {"rows": 166100},
                                   {"rows": 100, "coverage_window": 0}, {"rows": 100, "extra": 1}])
def test_invalid_shortlist_refused(value):
    with pytest.raises(ValueError):
        vocab.validate(value)


def test_coverage_includes_every_lane_recent_suffix_and_specials():
    def reason(histories, anchors=(2, 166100), processors=(False, False)):
        return vocab.coverage_reason(anchors, histories, processors, rows=10,
                                     special=166100, vocab=166912, window=3)
    assert reason([[99, 1, 2], [1, 2]]) is None  # old uncovered token is outside window
    assert reason([[1], [1, 99]]) == "coverage"
    assert reason([[1], [1]], processors=(False, True)) == "constraints"
    assert reason(None) == "missing_context"
    assert reason([[1], [1]], anchors=(2, -1)) == "coverage"


def test_guard_restores_head_on_success_and_exception(monkeypatch):
    from mlx2.runtime.drafters import dflash2

    reduced = {"rows": 10, "special": vocab.BASE_TAIL_FROM, "vocab": 166912}
    full_head = object()
    model = SimpleNamespace(lm_head=full_head)
    def configure(owner, rows, **kwargs):
        assert rows == 10 and kwargs["special_from"] == vocab.BASE_TAIL_FROM
        owner._draft_vocab = reduced
        return {"head_rows": 822}
    monkeypatch.setattr(dflash2, "configure_draft_vocab", configure)
    def original(anchors, hidden, cache, count, **options):
        if options.get("fail"):
            raise RuntimeError("graph construction failed")
        return {"reduced": model._draft_vocab is reduced}
    model.start_tree = original
    model.finish_tree = lambda state, **_: state
    receipt = vocab.install(model, {"rows": 10})
    assert model.start_tree([2], None, None, 3, proposal_histories=[[1]],
                            proposal_has_processors=[False])["reduced"]
    assert not model.start_tree([2], None, None, 3, proposal_histories=[[99]],
                                proposal_has_processors=[False])["reduced"]
    assert not model.propose_tree([2], None, None, 3)["reduced"]
    with pytest.raises(RuntimeError):
        model.start_tree([2], None, None, 3, fail=True)
    assert model._draft_vocab is reduced and model.lm_head is full_head
    assert receipt["counts"] == {"reduced": 1, "coverage": 1, "constraints": 0, "missing_context": 2}
    assert "counts" not in model.draft_vocab_receipt


def test_actual_shortlist_retains_basic_character_tail_and_full_head_identity():
    import mlx.core as mx
    from mlx import nn

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        head = nn.Linear(16, 166912, bias=False)
        model = SimpleNamespace(lm_head=head, config=SimpleNamespace(vocab_size=166912),
                                start_tree=lambda *_, **__: None, finish_tree=lambda *_, **__: None)
        receipt = vocab.install(model, {"rows": 32768})
        state = model._draft_vocab
        ids = state["ids"]
        assert int(ids[32768].item()) == vocab.BASE_TAIL_FROM
        assert int(ids[-1].item()) == 166911
        x = mx.ones((1, 16))
        full, reduced = head(x), state["head"](x)
        mx.eval(full, reduced)
        assert mx.allclose(mx.take(full, ids, axis=-1), reduced, atol=1e-5).item()
        assert model.lm_head is head and receipt["special_from"] == 166100
    finally:
        mx.set_default_device(previous)


def test_approximate_external_contract_requires_explicit_declaration_and_revision():
    operation = SimpleNamespace(revision="operation-revision")
    assert external_composition_contract(SimpleNamespace(), operation) is None
    assert external_composition_contract(MagicMock(), operation) is None
    good = {"schema": "mlx2.approximate-external-kv.v1", "operation_revision": operation.revision,
            "apcv2_publication": "skipped", "draft_cache": "exact", "qualified": False}
    adapter = SimpleNamespace(approximate_external_kv_contract=lambda _: dict(good))
    assert external_composition_contract(adapter, operation) == good
    for key, value in (("operation_revision", "other"), ("apcv2_publication", "enabled"),
                       ("draft_cache", "q8"), ("qualified", True)):
        adapter.approximate_external_kv_contract = lambda _, key=key, value=value: {**good, key: value}
        with pytest.raises(ValueError):
            external_composition_contract(adapter, operation)


def test_nanbeige_rejects_foreign_operation_and_preserves_binding_after_namespace():
    from mlx2.adapters.nanbeige43 import Nanbeige43Adapter
    from mlx2.runtime.approximate_kv import (
        KVQuantizationDescriptor,
        KVQuantizationOperation,
    )

    adapter = object.__new__(Nanbeige43Adapter)
    adapter.identity = {"fingerprint": "adapter-A"}
    adapter._nanbeige_checkpoint_fingerprint = "checkpoint"
    adapter.nanbeige_tuning = {"tree": "packed", "q8_ast": True}
    descriptor = KVQuantizationDescriptor(8, 8, 64)
    foreign = KVQuantizationOperation("kv_q8", descriptor, adapter_fingerprint="foreign-B")
    with pytest.raises(ValueError, match="another adapter"):
        adapter.approximate_external_kv_contract(foreign)
    own = KVQuantizationOperation("kv_q8", descriptor, adapter_fingerprint="adapter-A")
    assert adapter.approximate_external_kv_contract(own)["operation_revision"] == own.revision

    adapter.identity = {"fingerprint": "later-external-namespace"}
    assert adapter._tree_provider().operation_revision == own.revision
    assert adapter.approximate_external_kv_contract(own)["operation_revision"] == own.revision


def test_cli_external_approximate_admission_is_capability_and_policy_bound():
    import json

    from mlx2.adapters.nanbeige43 import Nanbeige43Adapter
    from mlx2.server import RouteSelection, approximate_kv_mode, build_parser

    parser = build_parser()
    args = parser.parse_args(["--model", "fixture", "--external-draft", "--qualification-mode",
                              "--approximate-kv", json.dumps({"enabled": True, "operation": "kv_q8"})])
    resolution = SimpleNamespace(adapter_type=Nanbeige43Adapter)
    policy = {"draft_model": "bound-draft", "nanbeige": {"tree": "packed", "q8_ast": True}}
    route = RouteSelection("external_draft", "explicit_flag")
    result = approximate_kv_mode(args, False, route_selection=route,
                                 adapter_resolution=resolution, execution_policy=policy)
    assert result["enabled"] and result["operation"] == "kv_q8"
    for other in ({"draft_model": "bound-draft"}, {"nanbeige": {"q8_ast": False}}):
        with pytest.raises(ValueError):
            approximate_kv_mode(args, False, route_selection=route,
                                 adapter_resolution=resolution, execution_policy=other)
    with pytest.raises(ValueError):
        approximate_kv_mode(args, False, route_selection=route)
