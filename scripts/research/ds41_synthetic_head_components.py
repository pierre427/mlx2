# SPDX-License-Identifier: MIT
"""Actual full-size output-head expressions, synthetic weights only."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import gc,time,statistics
K=5120;N=129280;REPS=7

def run(mx,check_live=None):
 from mlx2.runtime.gguf.source_dense import project
 check=check_live or (lambda:None);check()
 # One BF16 operand and at most one F32 conversion: ~3.97GB, no checkpoint.
 w=mx.full((N,K),0.001,dtype=mx.bfloat16);x=mx.linspace(-1,1,K,dtype=mx.float32).reshape(1,1,K);xb=x.astype(mx.bfloat16)
 mx.eval(w,x,xb);mx.synchronize()
 cases={'head_weight_cast_only':lambda:w.astype(mx.float32),
        'head_cast_each_call_counterfactual':lambda:x@w.astype(mx.float32).T,
        'head_bf16_matvec_dtype_alternative':lambda:(xb@w.T).astype(mx.float32),
        'actual_source_direct_bf16_weight_f32_head':lambda:project(x,w,'head')}
 def once(fn):
  start=time.perf_counter();out=fn();submitted=time.perf_counter();mx.eval(out);mx.synchronize();end=time.perf_counter()
  finite=bool(mx.all(mx.isfinite(out.reshape(-1)[:16])).item())
  return dict(host_graph_submit_seconds=submitted-start,completion_wait_seconds=end-submitted,total_seconds=end-start,sampled_finite=finite)
 result={name:dict(samples=[],warm_compile=once(fn))for name,fn in cases.items()}
 names=tuple(cases)
 for rep in range(REPS):
  for name in names[rep%len(names):]+names[:rep%len(names)]:check();result[name]['samples'].append(once(cases[name]))
 # Avoid two simultaneous2.648GB F32 copies; prepared case is a separatephase.
 del cases;gc.collect();mx.synchronize();mx.clear_cache();check()
 prepared=w.astype(mx.float32);mx.eval(prepared);mx.synchronize()
 result['actual_source_prepared_f32_head']=dict(samples=[],warm_compile=once(lambda:x@prepared.T))
 for _ in range(REPS):check();result['actual_source_prepared_f32_head']['samples'].append(once(lambda:x@prepared.T))
 for value in result.values():
  value['medians']={key:statistics.median(r[key]for r in value['samples'])for key in ('host_graph_submit_seconds','completion_wait_seconds','total_seconds')}
  value['minimum_seconds']=min(r['total_seconds']for r in value['samples']);value['maximum_seconds']=max(r['total_seconds']for r in value['samples'])
 return dict(schema='mlx2.ds41-synthetic-full-head.v1',complete=True,finite_execution=all(r['sampled_finite']for v in result.values()for r in v['samples']),checkpoint_loaded=False,model_constructed=False,shape=[1,1,K,N],bf16_weight_bytes=N*K*2,f32_weight_bytes=N*K*4,maximum_active_weight_inventory_bytes=N*K*6,repetitions=REPS,cases=result,scope='Current reference/woa_f32 Source loader prepares F32 head once, so prepared-F32matvec is actual baseline; cast-per-call is counterfactual only; direct BF16-weight and dtype controls; synthetic constant weights; no output-quality claim; prepared-F32case in separate phase for6GiBcap')
