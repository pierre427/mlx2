"""Scheduling policy contracts; synthetic CPU fixtures, no model qualification."""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import row_exact_qmv as qmv
from mlx2.runtime.row_exact import gate_rows, planner

mx.set_default_device(mx.cpu)


class PlainMLP(nn.Module):
    def __init__(self, bits=4, dtype=mx.bfloat16):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(512, 640, bias=False, bits=bits, group_size=64)
        self.up_proj = nn.QuantizedLinear(512, 640, bias=False, bits=bits, group_size=64)
        self.set_dtype(dtype)


def test_policy_requires_existing_consumer_and_preserves_default_receipt():
    assert "q4_gate_three_rows" not in FlashNextPolicy().as_dict()
    with pytest.raises(ValueError, match="requires row_exact_verify"):
        FlashNextPolicy(q4_gate_three_rows=True)
    with pytest.raises(ValueError, match="must be boolean"):
        FlashNextPolicy(q4_gate_three_rows=1, row_exact_verify=True)
    policy = FlashNextPolicy(q4_gate_three_rows=True, row_exact_verify=True)
    assert policy.as_dict()["q4_gate_three_rows"] is True


def test_plain_non_gdn_gate_binding_is_scoped_reversible_and_weight_bound():
    model, other = PlainMLP(), PlainMLP()
    handle = gate_rows.install(model)
    assert gate_rows.ATTR not in model.up_proj.__dict__
    assert gate_rows.ATTR not in other.gate_proj.__dict__
    for rows in (1, 2, 4, 8, 16):
        assert gate_rows.selected(model.gate_proj, mx.zeros((rows, 512), mx.bfloat16), rows) is None
    assert gate_rows.selected(model.gate_proj, mx.zeros((3, 512), mx.bfloat16), 3) is handle
    assert gate_rows.selected(model.gate_proj, mx.zeros((3, 512), mx.float16), 3) is None
    handle.set_enabled(False)
    assert gate_rows.selected(model.gate_proj, mx.zeros((3, 512), mx.bfloat16), 3) is None
    handle.set_enabled(True)
    model.gate_proj.scales = model.gate_proj.scales + 1
    with pytest.raises(ValueError, match="weights changed"):
        gate_rows.selected(model.gate_proj, mx.zeros((3, 512), mx.bfloat16), 3)
    handle.remove()
    assert gate_rows.ATTR not in model.gate_proj.__dict__


@pytest.mark.parametrize("bits,dtype", [(5, mx.bfloat16), (4, mx.float16)])
def test_unsupported_gate_binding_refuses(bits, dtype):
    with pytest.raises(ValueError, match="no eligible"):
        gate_rows.install(PlainMLP(bits, dtype))


def test_grouped_consumer_uses_one_existing_launch_and_distinct_cached_plans(monkeypatch):
    monkeypatch.setattr(planner, "_table", dict)
    monkeypatch.setattr(qmv, "decline_reason", lambda *a: None)
    monkeypatch.setattr(qmv, "_group_kernel", lambda *a: object())
    model = PlainMLP()
    handle = gate_rows.install(model)
    # Populate a group verdict on the peer too; research A/B invalidates both.
    model.up_proj.__dict__["_mlx2_row_exact_verdicts"] = {"old": True}
    handle.set_enabled(False)
    assert "_mlx2_row_exact_verdicts" not in model.up_proj.__dict__
    x = mx.zeros((1, 3, 512), mx.bfloat16)
    members = (model.gate_proj, model.up_proj)
    ordinary = qmv._group_plan(members, x, 3)
    handle.set_enabled(True)
    selected = qmv._group_plan(members, x, 3)
    assert dict(ordinary.template)["ROWS"] == 2
    assert dict(selected.template)["ROWS"] == 1
    assert dict(selected.template)["RPS"] == 4
    assert handle.counts["grouped"] == 1
    handle.set_enabled(False)
    assert qmv._group_plan(members, x, 3) is ordinary
    handle.remove()


def test_dense_adapter_finish_load_selection_is_instance_bound():
    from types import SimpleNamespace

    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    adapter = object.__new__(Qwen3827BAdapter)
    adapter.model = SimpleNamespace(language_model=PlainMLP())
    adapter._q4_gate_three_rows_requested = True
    adapter._select_q4_gate_three_rows()
    assert len(adapter.q4_gate_three_rows.modules) == 1
    adapter.q4_gate_three_rows.remove()
    adapter._q4_gate_three_rows_requested = False
    adapter._select_q4_gate_three_rows()
    assert adapter.q4_gate_three_rows is None
