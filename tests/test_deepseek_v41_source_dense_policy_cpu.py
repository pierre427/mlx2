"""CPU policy/prepared-input contracts; GPU math parity is a separate probe."""
from types import SimpleNamespace
import numpy as np
import pytest
import mlx.core as mx
from mlx2.adapters.deepseek_v41_source import source_policy, SOURCE_POLICY_CHOICES
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights


@pytest.mark.parametrize('key',SOURCE_POLICY_CHOICES)
def test_source_policy_choices_are_explicit_and_reference_default(key,tmp_path):
    assert key not in source_policy()
    for value in SOURCE_POLICY_CHOICES[key]:
        policy={key:value}
        if ((key=='deepseek_source_dspark_prepare'and value!='off')
                or (key=='deepseek_source_ast_kv_preparation'and value!='reference')):
            policy.update(deepseek_source_dspark_prepare='oracle_v1',
                deepseek_source_dspark_target_forward='fully_batched_m1_v1',
                deepseek_source_dspark_target_expert_law='source_ast_m1_rows_v1',
                deepseek_source_expert_math='routed_fused_fp32')
        if ((key=='deepseek_source_dspark_ast'and value!='off')
                or key in ('deepseek_source_dspark_target_expert_law','deepseek_source_dspark_target_forward')):
            oracle=tmp_path/'dspark-oracle.json';oracle.write_text('{}')
            policy['deepseek_source_dspark_oracle']=str(oracle)
            policy['deepseek_source_dspark_ast']='tile3x5_v1'
            if key=='deepseek_source_dspark_target_expert_law'and value in ('source_ast_m1_rows_v1','source_ast_resident_rows_v1','source_ast_hybrid_rows_v1'):
                policy['deepseek_source_expert_math']='routed_fused_fp32'
            if key=='deepseek_source_dspark_target_expert_law'and value=='source_ast_hybrid_rows_v1':
                policy['deepseek_source_dspark_target_forward']='fully_batched_m1_v1'
                policy['deepseek_source_prefill_partition']='canonical_m1_v1'
            if key=='deepseek_source_dspark_target_expert_law'and value=='source_ast_resident_rows_v1':
                policy['deepseek_source_dspark_target_forward']='fully_batched_v1'
        if key=='deepseek_source_woa_storage' and value!='reference':
            policy['deepseek_source_bf16_loads']='woa_f32'
        if key=='deepseek_source_qakv' and value!='reference':
            policy['deepseek_source_dense_qat']='fused_fp32'
        if key=='deepseek_source_ffn_input' and value!='reference':
            policy.update(deepseek_source_dense_qat='fused_fp32',deepseek_source_expert_math='routed_fused_fp32')
        if key=='deepseek_source_shared_middle' and value!='reference':
            policy['deepseek_source_dense_qat']='fused_fp32'
        if key=='deepseek_source_hc_mega' and value!='reference':
            policy['deepseek_source_hc_coefficients']='fused_fp32'
        if key=='deepseek_source_attention_math' and value!='reference':
            policy['deepseek_index_policy']='finite_all_retained'
        if key=='deepseek_source_prefill_batch' and value!='reference':
            policy['deepseek_source_expert_math']='routed_fused_fp32'
        if key=='deepseek_source_resident_prefill' and value!='reference':
            policy.update(deepseek_source_resident_gather='resident_bank_v1',deepseek_source_expert_math='routed_fused_fp32')
        if key=='deepseek_source_resident_gather' and value!='reference':
            policy['deepseek_source_expert_math']='routed_fused_fp32'
        if key=='deepseek_source_prefill_wide_math' and value!='reference':
            policy.update(deepseek_source_expert_math='routed_fused_fp32',deepseek_source_prefill_batch='gate_up')
        if key in ('deepseek_source_frame_reuse','deepseek_source_row_packet',
                   'deepseek_source_native_mx','deepseek_source_prefill_down') and value!='reference':
            policy.update(deepseek_source_expert_math='routed_fused_fp32',deepseek_source_prefill_batch='gate_up')
        if key=='deepseek_source_native_mx' and value!='reference':
            policy['deepseek_source_experts']={'mode':'original_lfu','budget_bytes':75_000_000_000}
        if key=='deepseek_source_expert_io' and value!='reference':
            policy['deepseek_source_experts']={'mode':'original_lfu','budget_bytes':75_000_000_000}
        if key=='deepseek_source_row_packet' and value!='reference':
            policy['deepseek_source_frame_reuse']='prefill_v1'
        if key=='deepseek_source_prefill_readahead' and value!='reference':
            policy.update(deepseek_source_prefill_order='layer_major_v1',
                          deepseek_source_experts={'mode':'original_lfu','budget_bytes':75_000_000_000})
        assert source_policy(policy)[key]==value
    for value in (None,True,1,[],{},'unknown'):
        with pytest.raises(ValueError):source_policy({key:value})


def test_prefill_batch_declines_incompatible_expert_math():
    with pytest.raises(ValueError,match='requires routed_fused_fp32'):
        source_policy({'deepseek_source_prefill_batch':'gate_up'})


def weights(policy='reference'):
    w=SourceV41Weights.__new__(SourceV41Weights)
    w.source_dense_qat_policy=policy;w.source_bf16_loads_policy='reference'
    w._quant={'gate':('gq','gs'),'up':('uq','us')}
    w.scheduled_dense_calls={};w.scheduled_grouped_calls={}
    w._ensure_dense=lambda:None;w._ready=lambda:None
    return w


@pytest.mark.parametrize('policy,qat_calls', [('reference',2),('fused_fp32',1)])
def test_gate_up_prepares_one_identical_activation_only_under_explicit_policy(monkeypatch,policy,qat_calls):
    from mlx2.runtime.models import deepseek_v41_quant as q
    calls=[];projections=[]
    def quant(value,kind):
        calls.append((value,kind));return value+mx.array(1,mx.float32)
    def qmm(value,packed,*,scales,mode):
        projections.append((value,packed,scales,mode));return value*(2 if packed=='gq' else 3)
    monkeypatch.setattr(q,'quantize',quant);monkeypatch.setattr(q,'fused_quantize',quant)
    monkeypatch.setattr(mx,'quantized_matmul',qmm)
    with mx.stream(mx.cpu):
        x=mx.array([[1.,2.,3.]],mx.float32);w=weights(policy);a,b=w.gate_up(x,'gate','up')
        mx.eval(a,b)
        np.testing.assert_array_equal(np.asarray(a),[[4,6,8]])
        np.testing.assert_array_equal(np.asarray(b),[[6,9,12]])
        assert len(calls)==qat_calls
        assert [r[1:] for r in projections]==[('gq','gs','mxfp8'),('uq','us','mxfp8')]
        assert w.scheduled_dense_calls['official_mxfp8_act32']==2
        if policy!='reference':
            assert projections[0][0] is projections[1][0]
            assert w.scheduled_dense_calls['source_dense_gate_up_qat_reused']==1


def test_shared_qat_refuses_to_reuse_across_nonquantized_projection(monkeypatch):
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
    sentinel=(object(),object());seen=[]
    monkeypatch.setattr(NativeV41Weights,'gate_up',lambda self,*a:seen.append(a) or sentinel)
    w=weights('fused_fp32');w._quant.pop('up')
    assert w.gate_up('x','gate','up') is sentinel
    assert seen==[('x','gate','up')]


def test_fused_dense_input_uses_f32_and_reference_preserves_input_dtype(monkeypatch):
    from mlx2.runtime.models import deepseek_v41_quant as q
    seen=[]
    def quant(value,kind):seen.append((value.dtype,kind));return value
    monkeypatch.setattr(q,'quantize',quant);monkeypatch.setattr(q,'fused_quantize',quant)
    with mx.stream(mx.cpu):
        x=mx.array([[1.,2.]],mx.bfloat16)
        assert weights()._dense_input(x).dtype==mx.float32
        assert weights('fused_fp32')._dense_input(x).dtype==mx.float32
    assert seen==[(mx.bfloat16,'fp8'),(mx.float32,'fp8')]


def test_head_direct_and_fallback_are_observable_without_weight_requantization(monkeypatch):
    from mlx2.runtime.gguf import source_dense
    with mx.stream(mx.cpu):
        w=weights();w.source_bf16_loads_policy='direct_f32';w.array=lambda name:mx.array([[1,2],[3,4]],mx.bfloat16)
        x=mx.array([[[2.,3.]]],mx.float32)
        expected=mx.array([[[99.,100.]]]);monkeypatch.setattr(source_dense,'maybe_project',lambda *a:expected)
        assert w.linear('output.weight')(x) is expected
        assert w.scheduled_dense_calls['source_bf16_head_direct_f32']==1
        monkeypatch.setattr(source_dense,'maybe_project',lambda *a:None)
        fallback=w.linear('output.weight')(x);mx.eval(fallback)
        np.testing.assert_array_equal(np.asarray(fallback),[[[8,18]]])
        assert w.scheduled_dense_calls['source_bf16_head_fallback']==1


def test_unhooked_completion_entry_points_keep_synchronous_fallback():
    w=weights();w.source_completion_policy='reference';w._source_completion=None
    with w.execution_scope():assert w._source_completion is None


@pytest.mark.parametrize('rows,selected',[(2,True),(3,True),(7,True),(8,False),(99,False)])
def test_grouped_legacy_selection_retains_stock_for_slower_wide_shapes(monkeypatch,rows,selected):
    from mlx2.runtime.gguf import source_dense_legacy
    calls=[]
    monkeypatch.setattr(source_dense_legacy,'supports',lambda *args:True)
    monkeypatch.setattr(source_dense_legacy,'project',lambda *args:calls.append(args) or 'legacy')
    with mx.stream(mx.cpu):
        w=weights();w.source_bf16_loads_policy='woa_f32'
        w.array=lambda name:mx.array([[1,2],[3,4]],mx.bfloat16)
        x=mx.ones((1,rows,1,2),mx.float32)
        result=w.grouped(x,'wo_a',groups=1)
        if selected:
            assert result=='legacy'
            assert len(calls)==1
            assert w.scheduled_grouped_calls['source_bf16_grouped_legacy_f32']==1
        else:
            mx.eval(result)
            np.testing.assert_array_equal(np.asarray(result),np.broadcast_to([3.,7.],(1,rows,1,2)))
            assert not calls
            assert w.scheduled_grouped_calls['official_bf16_grouped_f32']==1
