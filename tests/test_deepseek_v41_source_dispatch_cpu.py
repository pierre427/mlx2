"""CPU routing/ownership oracles for explicit dispatch candidates; no Metal."""
import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_math as math
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.runtime.gguf import source_q1, source_mxfp4_pair
from mlx2.runtime.models import deepseek_v41_quant as qat
from test_deepseek_v41_source_mixed_reuse_cpu import _fixture, _q1_cpu, _repeated_qat_reference


@pytest.mark.parametrize("rows,topk", [(1,1), (1,6), (7,6), (99,6), (1024,6)])
def test_offsets_preserve_each_route_and_expert_sum_order(rows, topk):
    rng = np.random.default_rng(rows+topk)
    # Include duplicate IDs: each route has its own score/output occurrence.
    ids = rng.integers(0,4,(rows,topk),dtype=np.uint32)
    order = np.argsort(ids.ravel(),kind="stable")
    offsets = math.ordered_offsets(ids,order)
    for row in range(rows):
        expected_slots = sorted(range(topk),key=lambda slot:int(ids[row,slot]))
        for rank,slot in enumerate(expected_slots):
            assert order[offsets[row,rank]] == row*topk+slot


@pytest.mark.parametrize("ids,order", [
    (np.zeros((1,6),np.int32),np.arange(6)),
    (np.zeros(6,np.uint32),np.arange(6)),
    (np.zeros((0,6),np.uint32),np.arange(0)),
    (np.zeros((1,6),np.uint32),np.arange(5)),
])
def test_bad_offset_geometry_refused(ids,order):
    with pytest.raises(ValueError,match="geometry"):
        math.ordered_offsets(ids,order)


@pytest.mark.parametrize("policy", ["ordered_fused_fp32","paired_q1_fused_fp32","paired_fused_fp32"])
@pytest.mark.parametrize("rows", [1,7,64])
@pytest.mark.parametrize("dtype", [mx.float32,mx.bfloat16])
@pytest.mark.parametrize("owned", [False,True])
def test_dispatch_policy_keeps_independent_mixed_reference(policy,rows,dtype,owned,monkeypatch):
    with mx.stream(mx.cpu):
        weights,bank,_,x,ids,scores,acquired,observed,windows = _fixture(rows,dtype,owned)
        expected = _repeated_qat_reference(bank,x,ids,scores);mx.eval(expected)
        def simulated_fused(v,kind):return qat.quantize(v,kind)
        pairs=[]
        def simulated_pair(v,*data):
            pairs.append(v.shape)
            return _q1_cpu(v,*data[:3]),_q1_cpu(v,*data[3:])
        originals=[]
        def simulated_original_pair(v,gq,gs,uq,us,**kwargs):
            originals.append(v.shape)
            return (mx.quantized_matmul(v,gq,scales=gs,mode="mxfp4"),
                    mx.quantized_matmul(v,uq,scales=us,mode="mxfp4"))
        def simulated_combine(parts,values,order):
            # Map output rows to original route occurrences, then use Python's
            # stable expert-ID ordering independently of the offset planner.
            packed=mx.concatenate(parts,axis=0);lookup={int(slot):i for i,slot in enumerate(order)}
            results=[]
            for row in range(len(values)):
                acc=mx.zeros((packed.shape[1],),mx.float32)
                for slot in sorted(range(values.shape[1]),key=lambda s:int(values[row,s])):
                    acc=acc+packed[lookup[row*values.shape[1]+slot]]
                results.append(acc)
            return mx.stack(results)
        monkeypatch.setattr(qat,"fused_quantize",simulated_fused)
        monkeypatch.setattr(source_q1,"matmul",_q1_cpu)
        monkeypatch.setattr(source_q1,"paired_matmul",simulated_pair)
        monkeypatch.setattr(source_mxfp4_pair,"supports",lambda *args:True)
        monkeypatch.setattr(source_mxfp4_pair,"paired_matmul",simulated_original_pair)
        monkeypatch.setattr(math,"ordered_combine",simulated_combine)
        weights.expert_math_policy=policy
        actual=weights.moe(x,ids,scores,"blk.0");mx.eval(actual)
        assert np.array(actual).tobytes()==np.array(expected).tobytes()
        assert acquired==[("blk.0",int(e)) for e in np.unique(ids)]
        assert len(observed)==1 and windows==(["enter","exit"] if owned else [])
        assert len(pairs)==(sum(int(e)%2==1 for e in np.unique(ids)) if policy.startswith("paired") else 0)
        assert len(originals)==(sum(int(e)%2==0 for e in np.unique(ids)) if policy=="paired_fused_fp32" else 0)
        assert source_policy({"deepseek_source_expert_math":policy})["deepseek_source_expert_math"]==policy


def test_pair_geometry_refuses_before_metal(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("bad input reached Metal")
    monkeypatch.setattr(source_q1,"pair_kernel",forbidden)
    with mx.stream(mx.cpu):
        x=mx.zeros((1,64),mx.float32);w=mx.zeros((8,2),mx.uint32)
        s=mx.ones((8,1),mx.float16);b=mx.zeros((8,1),mx.float16)
        for bad in [(w,s.astype(mx.float32),b),(w,s,b[:7]),(w[:7],s[:7],b[:7])]:
            with pytest.raises(ValueError,match="Q1 paired"):
                source_q1.paired_matmul(x,w,s,b,*bad)


def test_changed_original_library_refuses_before_kernel(monkeypatch,tmp_path):
    (tmp_path/"lib").mkdir();(tmp_path/"lib/libmlx.dylib").write_bytes(b"changed stock dispatch")
    source_mxfp4_pair._verify_library.cache_clear()
    monkeypatch.setattr(source_mxfp4_pair,"installed_root",lambda:tmp_path)
    try:
        with pytest.raises(RuntimeError,match="stock dispatch changed"):
            source_mxfp4_pair.kernel_spec(1)
    finally:
        source_mxfp4_pair._verify_library.cache_clear()


@pytest.mark.parametrize("rows",[0,13,16,99])
def test_original_pair_unverified_row_band_refuses(rows):
    with pytest.raises(ValueError,match="unverified"):
        source_mxfp4_pair.descriptor(rows,verified_m5_vector_band=True)
