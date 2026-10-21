"""P8: round receipts snapshot lane diagnostics without ``copy.deepcopy``."""

import copy

import mlx.core as mx
import pytest

from mlx2.runtime import external_speculative
from mlx2.runtime.ast_private_state import (
    FrozenReceipt,
    freeze_receipt,
    snapshot_receipt,
)
from tests.test_external_dflash2_cpu import generator, tiny


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_snapshot_matches_deepcopy_and_is_independent_of_the_source():
    source = {
        "hist": {1: 3, "2": 4},
        "rows": [1, (2, 3), {"a": None}],
        "flag": True,
        "ratio": 0.5,
    }
    snap = snapshot_receipt(source)
    assert snap == copy.deepcopy(source)
    source["hist"][1] += 1
    source["rows"][2]["a"] = 9
    assert snap["hist"][1] == 3 and snap["rows"][2]["a"] is None
    frozen = freeze_receipt({"x": [1]})
    assert snapshot_receipt(frozen) is frozen
    assert isinstance(snapshot_receipt({"f": frozen})["f"], FrozenReceipt)

    class Opaque:
        def __init__(self):
            self.value = [1]

    opaque = Opaque()
    copied = snapshot_receipt({"o": opaque})["o"]
    assert copied is not opaque and copied.value == [1]


def test_progressive_round_receipt_takes_no_deepcopy(monkeypatch):
    model, draft = tiny()
    batch = generator(model, draft)
    # Receipt shape only: the progressive branch reads configuration fields.
    batch.progressive_verification_tile = 1
    batch.progressive_protocol_binding = {
        "source_hashes": {},
        "qualification_gate_schema": None,
    }
    try:
        uid = batch.insert([[1, 2, 3]], max_tokens=[4])[0]
        lane = batch.lanes[uid]
        lane.progressive_ast_counterfactual_best_branch_hist = {1: 2}
        lane.progressive_ast_primary_fallback_reasons = {"latch": 1}
        calls = []
        original = copy.deepcopy

        def counted(value, memo=None):
            calls.append(type(value).__name__)
            return original(value, memo)

        monkeypatch.setattr(external_speculative.copy, "deepcopy", counted)
        receipt = batch._verification_receipt(lane)
        assert "progressive_verification" in receipt
        assert calls == []
        ordered = receipt["progressive_verification"]["ordered_alternatives"]
        assert ordered["primary_fallback_reasons"] == {"latch": 1}
        lane.progressive_ast_primary_fallback_reasons["latch"] += 1
        assert ordered["primary_fallback_reasons"] == {"latch": 1}
    finally:
        batch.close()
