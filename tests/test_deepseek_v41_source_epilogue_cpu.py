"""Independent CPU composition and ownership checks; Metal checked separately."""
import mlx.core as mx
import numpy as np
import pytest
from mlx2.adapters import deepseek_v41_source_math as assembly
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.runtime.gguf import source_q1,source_middle_fusion as epi,source_down_fusion as down
from mlx2.runtime.models import deepseek_v41_quant as qat
from mlx2.runtime.models.deepseek_v4 import limited_swiglu
from test_deepseek_v41_source_mixed_reuse_cpu import _fixture,_q1_cpu,_repeated_qat_reference


@pytest.mark.parametrize('rows',[1,7,64])
@pytest.mark.parametrize('policy',['epilogue_fused_fp32','routed_fused_fp32'])
@pytest.mark.parametrize('dtype',[mx.float32,mx.bfloat16])
@pytest.mark.parametrize('owned',[False,True])
def test_epilogue_route_matches_independent_mixed_reference(rows,policy,dtype,owned,monkeypatch):
    with mx.stream(mx.cpu):
        w,bank,_,x,ids,scores,acquired,observed,windows=_fixture(rows,dtype,owned)
        expected=_repeated_qat_reference(bank,x,ids,scores);mx.eval(expected)
        def no_metal(*args,**kwargs):raise AssertionError('CPU test reached Metal')
        monkeypatch.setattr(mx.fast,'metal_kernel',no_metal)
        monkeypatch.setattr(qat,'fused_quantize',qat.quantize)
        monkeypatch.setattr(source_q1,'matmul',_q1_cpu)
        def pair(value,mapping,*args):
            return _q1_cpu(value[mapping],*args[:3]),_q1_cpu(value[mapping],*args[3:])
        monkeypatch.setattr(source_q1,'paired_matmul_mapped',pair)
        rounded=lambda v:v.astype(mx.bfloat16).astype(mx.float32)
        def middle(g,u,s,slots):
            return qat.quantize(rounded(limited_swiglu(rounded(g),rounded(u))*s[slots,None]),'fp8')
        monkeypatch.setattr(epi,'middle',middle)
        def down_sum(middles,experts,expert_ids):
            result=mx.zeros((1,x.shape[1]),mx.float32)
            for i in sorted(range(len(expert_ids)),key=lambda i:expert_ids[i]):
                d=experts[i]
                v=(_q1_cpu(middles[i],*(d['down'+f] for f in ('.qweight','.scales','.biases')))
                    if 'down.biases' in d else mx.quantized_matmul(middles[i],d['down.qweight'],scales=d['down.scales'],mode='mxfp4'))
                result=result+rounded(v)
            return result
        monkeypatch.setattr(down,'down_combine',down_sum)
        def combine(parts,values,order):
            packed=mx.concatenate(parts,axis=0);inverse=np.argsort(order)
            results=[]
            for row in range(rows):
                acc=mx.zeros((x.shape[-1],),mx.float32)
                for slot in sorted(range(values.shape[1]),key=lambda j:int(values[row,j])):
                    acc=acc+packed[int(inverse[row*values.shape[1]+slot])]
                results.append(acc)
            return mx.stack(results)
        monkeypatch.setattr(assembly,'ordered_combine',combine)
        w.expert_math_policy=policy
        actual=w.moe(x,ids,scores,'blk.0');mx.eval(actual)
        assert np.array(actual).tobytes()==np.array(expected).tobytes()
        assert acquired==[('blk.0',int(e)) for e in np.unique(ids)]
        assert windows==(['enter','exit'] if owned else []) and len(observed)==1
        assert w.scheduled_expert_calls['official_source_fused_middle']==len(np.unique(ids))
        avoided=int(np.count_nonzero(ids%2))*x.shape[1]*4
        assert w.scheduled_expert_calls['official_source_input_gather_bytes_avoided']==avoided
        assert source_policy({'deepseek_source_expert_math':policy})['deepseek_source_expert_math']==policy


def test_middle_bad_geometry_refuses_before_metal(monkeypatch):
    monkeypatch.setattr(epi,'exp_spelling',lambda:pytest.fail('invalid geometry reached GPU gate'))
    with mx.stream(mx.cpu):
        g=mx.zeros((2,64),mx.float32);s=mx.ones((2,),mx.float32);slots=mx.array([0,1],mx.uint32)
        for args in [(g,g[:1],s,slots),(g.astype(mx.bfloat16),g,s,slots),
                     (g,g,s,slots.astype(mx.int32)),(g[:,:63],g[:,:63],s,slots)]:
            with pytest.raises(ValueError,match='geometry'):
                epi.middle(*args)
