from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.generate import PromptProcessingBatch
from mlx2.runtime.models.qwen3_next import Qwen3NextMLP, Qwen3NextSparseMoeBlock
from mlx2.runtime.models.varlen_dense_mlp import (
    VarlenDenseMLPPolicy,
    VarlenSparseMoEPolicy,
    compact_rows,
    identity,
    install,
    prefill_row_context,
    status,
)
from mlx2.runtime.prefill_plan import execution_identity


def enabled_model(policy=True):
    model = SimpleNamespace(prefill_row_context=lambda *_args, **_kwargs: None)
    handle = install(model, VarlenDenseMLPPolicy.from_value(policy))
    return model, handle


def enabled_sparse_model(policy=True):
    model = SimpleNamespace(prefill_row_context=lambda *_args, **_kwargs: None)
    handle = install(model, VarlenSparseMoEPolicy.from_value(policy))
    return model, handle


def test_policy_is_explicit_bounded_and_default_off():
    assert VarlenDenseMLPPolicy.from_value(None) == VarlenDenseMLPPolicy()
    assert VarlenDenseMLPPolicy.from_value(False) == VarlenDenseMLPPolicy()
    assert VarlenDenseMLPPolicy.from_value(True).enabled is True
    policy = VarlenDenseMLPPolicy.from_value(
        {"minimum_padding_rows": 32, "minimum_padding_fraction": 0.25}
    )
    assert policy == VarlenDenseMLPPolicy(True, 32, 0.25)
    for value in (0, 1, "yes", [], {"unknown": True}):
        with pytest.raises((TypeError, ValueError)):
            VarlenDenseMLPPolicy.from_value(value)
    for value in (
        {"enabled": 1},
        {"minimum_padding_rows": 0},
        {"minimum_padding_rows": True},
        {"minimum_padding_fraction": -0.1},
        {"minimum_padding_fraction": 1.1},
    ):
        with pytest.raises((TypeError, ValueError)):
            VarlenDenseMLPPolicy.from_value(value)
    assert VarlenSparseMoEPolicy.from_value(True).enabled is True
    with pytest.raises(TypeError, match="varlen_sparse_moe"):
        VarlenSparseMoEPolicy.from_value(1)


def test_dense_mlp_compacts_live_rows_and_restores_rectangular_shape():
    mx.random.seed(7)
    layer = Qwen3NextMLP(8, 16)
    layer.eval()
    values = mx.random.normal((2, 4, 8))
    reference = layer(values)
    model, handle = enabled_model()
    with prefill_row_context(model, [2, 4], width=4):
        result = layer(values)
    mx.eval(reference, result)
    assert result.shape == reference.shape
    assert mx.array_equal(result[0, :2], reference[0, :2]).item()
    assert mx.array_equal(result[1], reference[1]).item()
    assert mx.array_equal(result[0, 2:], mx.zeros_like(result[0, 2:])).item()
    receipt = status(handle)
    assert receipt["observed_used"] is True
    assert receipt["qualified"] is False
    assert receipt["counters"]["mlp_compaction_calls"] == 1
    assert receipt["counters"]["mlp_padding_rows_skipped"] == 2
    assert receipt["counters"]["mlp_live_rows"] == 6
    assert compact_rows(values) is None


def test_compacted_rows_reuse_the_installed_tensorfold_dispatch(monkeypatch):
    from mlx2.runtime.models import tensorfold_prefill

    values = mx.arange(2 * 4 * 8).reshape(2, 4, 8)
    layer = Qwen3NextMLP(8, 16)
    layer._prefill_counts = {}
    seen = []

    def fused_mlp(_layer, rows):
        seen.append(tuple(rows.shape))
        return rows

    monkeypatch.setattr(tensorfold_prefill, "fused_mlp", fused_mlp)
    model, handle = enabled_model()
    with prefill_row_context(model, [2, 4], width=4):
        result = layer(values)
    mx.eval(result)
    assert seen == [(1, 6, 8)]
    assert mx.array_equal(result[0, :2], values[0, :2]).item()
    assert mx.array_equal(result[1], values[1]).item()
    assert mx.array_equal(result[0, 2:], mx.zeros_like(result[0, 2:])).item()
    assert status(handle)["counters"]["mlp_scatter_calls"] == 1


def test_sparse_moe_compacts_live_rows_and_keeps_dense_scope_isolated(monkeypatch):
    from mlx2.runtime.models import qwen3_next

    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(qwen3_next, "_MOE_FUSED_GATE_UP", False)
    monkeypatch.setattr(qwen3_next, "_MOE_SHARED_IN_GATHER", False)
    monkeypatch.setattr(qwen3_next, "_MOE_FUSED_EXPERT_MODE", "stock")
    monkeypatch.setattr(qwen3_next, "_MOE_ROUTED_DECODE", "off")
    monkeypatch.setattr(qwen3_next, "_MOE_WINDOW_CONSUMERS", frozenset())
    monkeypatch.setattr(qwen3_next, "_MOE_TOPK_MODE", "off")
    monkeypatch.setattr(qwen3_next, "_MOE_ROUTER_KERNEL", False)
    monkeypatch.setattr(qwen3_next, "_MOE_GATE_COMPILE", False)
    monkeypatch.setattr(qwen3_next, "_COMPILE_GLUE", False)
    monkeypatch.setattr(qwen3_next, "_MOE_WEIGHTED_SUM", False)
    args = SimpleNamespace(
        hidden_size=8,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        norm_topk_prob=True,
        num_experts=4,
        num_experts_per_tok=2,
    )
    mx.random.seed(17)
    block = Qwen3NextSparseMoeBlock(args)
    block.eval()
    values = mx.random.normal((2, 4, 8))
    reference = block(values)
    model, handle = enabled_sparse_model()
    with prefill_row_context(model, [2, 4], width=4):
        assert compact_rows(values) is None
        result = block(values)
    mx.eval(reference, result)
    assert mx.array_equal(result[0, :2], reference[0, :2]).item()
    assert mx.array_equal(result[1], reference[1]).item()
    assert mx.array_equal(result[0, 2:], mx.zeros_like(result[0, 2:])).item()
    receipt = status(handle)
    assert receipt["schema"] == "mlx2.varlen-sparse-moe.v1"
    assert receipt["observed_used"] is True
    assert receipt["counters"]["moe_compaction_calls"] == 1
    assert receipt["counters"]["moe_padding_rows_skipped"] == 2
    assert receipt["counters"]["moe_scatter_calls"] == 1
    assert model._varlen_sparse_moe is handle


def test_cost_threshold_declines_without_changing_mlp_output():
    mx.random.seed(11)
    layer = Qwen3NextMLP(8, 16)
    layer.eval()
    values = mx.random.normal((2, 4, 8))
    reference = layer(values)
    model, handle = enabled_model({"minimum_padding_rows": 3})
    with prefill_row_context(model, [2, 4], width=4):
        result = layer(values)
    mx.eval(reference, result)
    assert mx.array_equal(result, reference).item()
    receipt = status(handle)
    assert receipt["observed_used"] is False
    assert receipt["counters"]["policy_declines"] == 1
    assert receipt["counters"].get("mlp_compaction_calls", 0) == 0


def test_unpadded_scope_declines_and_exception_resets_context():
    values = mx.zeros((2, 4, 8))
    model, handle = enabled_model()
    with prefill_row_context(model, [4, 4], width=4):
        assert compact_rows(values) is None
    assert status(handle)["counters"]["unpadded_declines"] == 1

    with (
        pytest.raises(RuntimeError, match="injected"),
        prefill_row_context(model, [2, 4], width=4),
    ):
        assert compact_rows(values) is not None
        raise RuntimeError("injected")
    assert compact_rows(values) is None


def test_prefill_processing_publishes_exact_chunk_lengths():
    class Model:
        def __init__(self):
            self.scopes = []
            self.calls = []

        @contextmanager
        def prefill_row_context(self, lengths, *, width):
            self.scopes.append((tuple(lengths), width))
            yield

        def __call__(self, tokens, *, cache):
            self.calls.append(tuple(tokens.shape))

    model = Model()
    batch = PromptProcessingBatch(
        model,
        [1, 2],
        [[], []],
        tokens=[[], []],
        prefill_step_size=3,
    )
    batch.prompt([[1, 2], [3, 4, 5, 6]])
    assert model.scopes == [((2, 3), 3), ((0, 1), 1)]
    assert model.calls == [(2, 3), (2, 1)]


def test_identity_separates_apcv2_state_and_excludes_counters():
    model, handle = enabled_model(
        {"minimum_padding_rows": 8, "minimum_padding_fraction": 0.125}
    )
    selected = identity(handle)
    before = execution_identity(varlen=selected)
    handle["counters"]["mlp_compaction_calls"] = 99
    assert before == execution_identity(varlen=identity(handle))
    assert before != execution_identity()
    assert model._varlen_dense_mlp is handle


def test_install_requires_adapter_declared_context():
    with pytest.raises(TypeError, match="does not declare"):
        install(SimpleNamespace(), VarlenDenseMLPPolicy(enabled=True))
