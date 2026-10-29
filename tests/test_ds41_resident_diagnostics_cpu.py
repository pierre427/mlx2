"""Source resident diagnostics cannot imply a selected serving route."""
from types import SimpleNamespace

import pytest

from mlx2.adapters.deepseek_v41_source import (
    DeepSeekV41SourceAdapter, SOURCE_POLICY_CHOICES, source_policy,
)
from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter


def test_actual_source_diagnostics_report_component_only_backend_unavailable(monkeypatch):
    monkeypatch.setattr(DeepSeekV41MLXAdapter,'diagnostics',lambda self:{'qualification':'unqualified'})
    adapter=DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter)
    adapter.identity={'fingerprint':'diagnostic-fixture'}
    adapter.native_admission={}
    adapter.draft_model=None
    adapter.kernel_policy={}
    adapter.weights=SimpleNamespace(expert_math_policy='reference',dense_receipt=lambda:{},
        source_qakv_policy='reference',source_q_b_prefill_policy='reference')
    adapter.model=SimpleNamespace(source_hc_boundary_policy='reference',index_policy='reference',
        selected_kv_policy='reference',source_attention_math_policy='reference',
        scheduled_hc_calls={},hc_coeff_policy='reference')
    diagnostic=adapter.diagnostics()['source_resident_routes']
    assert diagnostic['component_pilot_implemented'] is True
    assert diagnostic['component_pilot_gpu_observed'] is False
    assert diagnostic['serving_backend_available'] is False
    assert diagnostic['selected'] is False and diagnostic['qualified'] is False
    assert 'exact revision-bound cache transaction integration' in diagnostic['missing_serving_requirements']


def test_explicit_bank_policy_preserves_reference_and_requires_preparation():
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    assert SOURCE_POLICY_CHOICES['deepseek_source_resident_gather']==('reference','resident_bank_v1')
    assert 'deepseek_source_resident_gather' not in source_policy()
    with pytest.raises(ValueError,match='requires routed_fused_fp32'):
        source_policy({'deepseek_source_resident_gather':'resident_bank_v1'})
    selected=source_policy({'deepseek_source_resident_gather':'resident_bank_v1',
                            'deepseek_source_expert_math':'routed_fused_fp32'})
    assert selected['deepseek_source_resident_gather']=='resident_bank_v1'
    weights=object.__new__(SourceV41Weights)
    weights.source_resident_gather_policy='resident_bank_v1'
    weights.source_resident_bank_session=None
    with pytest.raises(ValueError,match='explicitly prepared bank session'):
        weights.execution_scope_for_rows(1)
    # Arbitrary-address/native-resource serving remains unavailable; explicit
    # tracked banks are a separate capability with their own source receipts.
    with pytest.raises(ValueError):source_policy({'deepseek_source_resident':'resident_v1'})


def test_probe_refuses_cpu_before_kernel_creation_or_descriptor_access(monkeypatch):
    import mlx.core as mx
    from mlx2.runtime.gguf.source_resident_probe import build
    assert mx.default_device()==mx.cpu
    def forbidden(*args,**kwargs):raise AssertionError('Metal compilation forbidden')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    with pytest.raises(ValueError,match='approved owned Metal'):
        build(None,None,expected_generation=0)
