"""Mixed route assembly with hoisted QAT; GPU fusion covered by owned probe."""
import numpy as np
import mlx.core as mx
import pytest

from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.runtime.models import deepseek_v41_quant as qat
from mlx2.runtime.gguf import source_q1
from test_deepseek_v41_source_mixed_reuse_cpu import _fixture, _q1_cpu, _repeated_qat_reference


@pytest.mark.parametrize('value', ['bad', True, 1, {}, []])
def test_source_math_unknown_policy_refused(value):
    with pytest.raises(ValueError, match='expert math'):
        source_policy({'deepseek_source_expert_math':value})


def test_source_math_explicit_and_default_reference():
    assert 'deepseek_source_expert_math' not in source_policy()
    for value in ('reference','hoisted_fused_fp32'):
        assert source_policy({'deepseek_source_expert_math':value})['deepseek_source_expert_math']==value


@pytest.mark.parametrize('rows',[1,7,64])
@pytest.mark.parametrize('dtype',[mx.float32,mx.bfloat16])
@pytest.mark.parametrize('owned',[False,True])
def test_hoisted_mixed_matches_independent_reference(rows,dtype,owned,monkeypatch):
    with mx.stream(mx.cpu):
        weights,bank,_,x,ids,scores,acquired,observed,windows=_fixture(rows,dtype,owned)
        expected=_repeated_qat_reference(bank,x,ids,scores);mx.eval(expected)
        calls=[]
        def simulated_fused(value,kind):
            calls.append(value.shape)
            assert value.dtype==mx.float32
            return qat.quantize(value,kind)
        monkeypatch.setattr(qat,'fused_quantize',simulated_fused)
        monkeypatch.setattr(source_q1,'matmul',_q1_cpu)
        weights.expert_math_policy='hoisted_fused_fp32'
        actual=weights.moe(x,ids,scores,'blk.0');mx.eval(actual)
        assert np.array(actual).tobytes()==np.array(expected).tobytes()
        assert calls[0]==x.shape
        assert len(calls)==1+len(np.unique(ids))
        assert weights.scheduled_expert_calls=={'official_mxfp4_act32_hoisted_fused':1}
        assert acquired==[('blk.0',int(e)) for e in np.unique(ids)]
        assert len(observed)==1
        assert windows==(['enter','exit'] if owned else [])
