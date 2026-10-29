"""Interleaved isolated kernels on captured native model operands; owned only."""

import statistics
import time
from dataclasses import replace

import numpy as np

from mlx2.desktop_campaign_worker import atomic
from mlx2.runtime.gguf.packed_metal import PackedMatrix
from mlx2.runtime.gguf import v41_metal,v41_row_exact,v41_dense
from mlx2.runtime.gguf.cooperative_metal import cooperative_dense,cooperative_gate_up
from mlx2.runtime.models.deepseek_v4 import limited_swiglu
from analyze_deepseek_v41_10token import REQUEST


def run(adapter,mx,out,phase):
    model,w=adapter.model,adapter.weights
    captured={};original=w.moe
    def capture(x,ids,scores,prefix,**kwargs):
        if prefix=="blk.0":
            mx.eval(x,ids,scores);captured.update(x=x,ids=ids,scores=scores)
        return original(x,ids,scores,prefix,**kwargs)
    w.moe=capture
    try:
        ids=mx.array([list(adapter.prompt_tokens(REQUEST))],mx.uint32)
        cache=model.make_cache();value=model.prefill_forward(ids,cache);mx.eval(value,[s.state for s in cache])
    finally:w.moe=original
    x,ids,scores=(captured[k] for k in ("x","ids","scores"))
    remap,planes=w.expert_stream.gather("blk.0",ids,row_buffers=True,host_ids=True)
    matrices=tuple(PackedMatrix(mx.concatenate([p.data for p in plane]),
                    replace(plane[0].tensor,shape=(*plane[0].tensor.shape[:2],len(plane)),nbytes=sum(p.data.nbytes for p in plane))) for plane in planes)
    mx.eval([p.data for p in matrices]);gpu_ids=mx.array(remap)
    results={"scope":"captured first-layer ten-token FFN inputs and exact artifact weight bytes; alternating A/B calls; wall plus thread CPU, not hardware GPU timestamps",
             "rows":int(x.shape[0]),"experts":len(planes[0]),"comparisons":[],"passed":False}

    def compare(name,reference,candidate):
        expected=reference();actual=candidate();mx.eval(expected,actual)
        def bytes_(values):
            if isinstance(values,(list,tuple)):return tuple(bytes_(v) for v in values)
            return (values.shape,str(values.dtype),np.array(values).tobytes())
        exact=bytes_(expected)==bytes_(actual)
        times={"reference":[],"candidate":[]};cpus={k:[] for k in times}
        for i in range(6):
            order=(("reference",reference),("candidate",candidate)) if i%2==0 else (("candidate",candidate),("reference",reference))
            for label,fn in order:
                started=time.perf_counter();cpu=time.thread_time();v=fn();mx.eval(v)
                times[label].append(time.perf_counter()-started);cpus[label].append(time.thread_time()-cpu)
        row={"name":name,"strict_bytes_equal":exact,"seconds":times,"thread_cpu_seconds":cpus,
             "median_seconds":{k:statistics.median(v) for k,v in times.items()}}
        results["comparisons"].append(row);atomic(out/"micro.json",results)
        phase("micro_comparison",name_=name,exact=exact,**row["median_seconds"])
        if not exact:raise RuntimeError("captured-operand kernel changed bytes: "+name)

    reference=lambda:v41_metal.moe(x,gpu_ids,scores,*matrices)
    for reuse,groups,simds,broadcast in ((1,6,1,False),(2,6,1,False),(4,6,1,False),(4,10,1,False),(4,10,4,False),(4,10,1,True)):
        compare(f"expert_r{reuse}_g{groups}_s{simds}_broadcast{broadcast}",reference,
                lambda:v41_row_exact.moe(x,remap,scores,*planes,reuse=reuse,group_experts=groups,simd_groups=simds,broadcast=broadcast))
    one_ids,one_planes=w.expert_stream.gather("blk.0",ids[:1],row_buffers=True,host_ids=True)
    compare("decode_parallel_down",lambda:v41_metal.moe_rows(x[:1],mx.array(one_ids),scores[:1],*one_planes),
            lambda:v41_row_exact.moe(x[:1],one_ids,scores[:1],*one_planes,reuse=1))
    a,b=(w.packed("blk.0."+key+".weight") for key in ("attn_q_a","attn_kv"))
    rounded=lambda v:v.astype(mx.bfloat16).astype(mx.float32)
    for shape in (1,10):
        xx=x[:shape]
        compare(f"dense_epilogue_{shape}",lambda:rounded(cooperative_dense(xx,a,rows=4)),lambda:v41_dense.project(xx,a))
        compare(f"dense_pair_{shape}",lambda:(rounded(cooperative_dense(xx,a,rows=4)),rounded(cooperative_dense(xx,b,rows=4))),lambda:v41_dense.project(xx,a,b))
        gate,up=(w.packed("blk.0.ffn_"+kind+"_shexp.weight") for kind in ("gate","up"))
        def shared_reference():
            g,u=cooperative_gate_up(xx,gate,up)
            return rounded(limited_swiglu(rounded(g),rounded(u)))
        compare(f"shared_swiglu_{shape}",shared_reference,lambda:v41_dense.shared_middle(xx,gate,up))
    results["passed"]=True;atomic(out/"micro.json",results)
    return results
