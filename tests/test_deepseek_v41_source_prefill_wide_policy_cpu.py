"""Synthetic dtype/shape engagement only; actual BF16 parity is a GPU gate."""
from types import SimpleNamespace
import numpy as np
import pytest
import mlx.core as mx
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights


@pytest.mark.parametrize('rows',[32,33,96,97])
@pytest.mark.parametrize('policy',['reference','bf16_p1'])
def test_wide_policy_only_changes_original_prefill_p1_operand_storage(monkeypatch,rows,policy):
    from mlx2.runtime.gguf import source_middle_fusion,source_mxfp4_pair
    from mlx2.adapters import deepseek_v41_source_math
    monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:True)
    monkeypatch.setattr(source_middle_fusion,'middle',lambda g,u,s,l:mx.ones(g.shape,mx.float32))
    monkeypatch.setattr(deepseek_v41_source_math,'ordered_combine',lambda parts,values,order:mx.zeros((len(values),5120),mx.float32))
    calls=[]
    def qmm(value,weight,*,scales,mode):
        expert,kind=weight;calls.append((expert,kind,value.dtype,value.shape))
        return mx.full((value.shape[0],5120 if kind=='down' else 2304),1.25,dtype=value.dtype)
    monkeypatch.setattr(mx,'quantized_matmul',qmm)
    data={e:{**{p+'.qweight':(e,p) for p in ('gate','up','down')},**{p+'.scales':0 for p in ('gate','up','down')}}for e in range(6)}
    w=SourceV41Weights.__new__(SourceV41Weights);w.source_prefill_wide_dtype=policy
    w.source_prefill_batch_policy='gate_up';w.scheduled_expert_calls={}
    w.expert_stream=SimpleNamespace(packed_buffers=True,kind=lambda key:'original',acquire=lambda prefix,e:data[e],stats={'lookups':0})
    values=np.tile(np.arange(6,dtype=np.uint32),(rows,1));order=np.argsort(values.ravel(),kind='stable')
    with mx.stream(mx.cpu):
        output=w._prefill_batched_gate_up(mx.ones((rows,5120),mx.float32),values,mx.ones((rows,6),mx.float32),'blk.0',order)
        mx.eval(output)
    assert len(calls)==18
    for _,kind,dtype,shape in calls:
        selected=policy=='bf16_p1' and rows >= (33 if kind=='down' else 97)
        assert dtype == (mx.bfloat16 if selected else mx.float32)
        assert shape[0]==rows  # no split/reassociation to enter another row band
    assert w.scheduled_expert_calls.get('official_source_prefill_bf16_p1_gate_up',0)==(12 if policy=='bf16_p1' and rows>=97 else 0)
    assert w.scheduled_expert_calls.get('official_source_prefill_bf16_p1_down',0)==(6 if policy=='bf16_p1' and rows>=33 else 0)
