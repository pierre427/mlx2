# SPDX-License-Identifier: MIT
"""Whole-forward synthetic rotary/mask setup; no model/checkpoint construction."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import gc,time,statistics
from types import SimpleNamespace

def run(mx,check_live=None):
 from mlx2.runtime.models.deepseek_v4 import Rotary
 from mlx2.runtime.models.deepseek_v41_frame_geometry import frame_geometry,phase_operands,local_mask
 from mlx2.runtime.models.deepseek_v41_fused import rotate,rotate_qat
 check=check_live or(lambda:None)
 length=99
 x=mx.full((1,64,length,512),.125,mx.float32);kv=mx.full((1,1,length,512),.25,mx.float32)
 pool=mx.full((1,1,length//2,512),.5,mx.float32);ikey=mx.full((1,1,length//2,128),.25,mx.float32)
 pool1=mx.full((1,1,length,512),.5,mx.float32);ikey1=mx.full((1,1,length,128),.25,mx.float32)
 ropes=[Rotary(64,10000. if i<2 else 160000.)for i in range(40)]
 mx.eval(x,kv,pool,ikey,pool1,ikey1,[r.inv for r in ropes]);mx.synchronize()
 def trial(reuse,phase_only=False):
  counts={};geometry=frame_geometry(SimpleNamespace(source_frame_reuse_policy='prefill_v1'if reuse else'reference'),length,counts)
  values=[];t=time.perf_counter()
  for i,r in enumerate(ropes):
   key=(64,10000. if i<2 else 160000.,'null')
   for inp,kind,inverse in ((x,None,False),(kv,'fp8',False),(x,'fp8',True)):
    phase=phase_operands(geometry,mx,r,key,inp,inverse=inverse)
    if phase_only:
     if phase is None:
      pos=mx.arange(length);angles=pos.astype(mx.float32)[:,None]*r.inv[None]
      if inverse:angles=-angles
      phase=(mx.cos(angles).astype(mx.float32),mx.sin(angles).astype(mx.float32))
     values.extend(phase)
    else:
     values.append(rotate(inp,r,inverse=inverse,phase=phase)if kind is None else rotate_qat(inp,r,kind,inverse=inverse,phase=phase))
   values.append(local_mask(geometry,mx,0,length,length,128))
   if i in (2,8,14,20):
    for inp,kind in ((ikey1 if i==20 else ikey,'index_fp4'),(pool1 if i==20 else pool,'compressed_fp4')):
     pos=tuple(range(length))if i==20 else tuple(range(0,98,2))
     phase=phase_operands(geometry,mx,r,key,inp,positions=pos)
     if phase_only:
      if phase is None:
       angles=mx.array(pos).astype(mx.float32)[:,None]*r.inv[None];phase=(mx.cos(angles).astype(mx.float32),mx.sin(angles).astype(mx.float32))
      values.extend(phase)
     else:values.append(rotate_qat(inp,r,kind,positions=pos,phase=phase))
  submitted=time.perf_counter();mx.eval(*values);mx.synchronize();end=time.perf_counter()
  result=dict(host_submit_seconds=submitted-t,completion_seconds=end-submitted,total_seconds=end-t,counts=counts,roots=len(values))
  del values,geometry;gc.collect();return result
 cells=[(False,True),(True,True),(False,False),(True,False)];data={c:[]for c in cells}
 for c in cells:check();trial(*c)
 for rep in range(7):
  for c in cells[rep%4:]+cells[:rep%4]:check();data[c].append(trial(*c))
 cases=[]
 for (reuse,only),samples in data.items():cases.append(dict(reuse=reuse,phase_only=only,samples=samples,medians={k:statistics.median(s[k]for s in samples)for k in('host_submit_seconds','completion_seconds','total_seconds')},counts=samples[-1]['counts']))
 return dict(schema='mlx2.synthetic-prefill-frame-reuse.v1',complete=True,checkpoint_loaded=False,model_constructed=False,rows=99,layers=40,rotary_calls=128,local_masks=40,repetitions=7,cases=cases,scope='Synthetic independent perlayer inputs; one final fence; actual fusedrotary/QATfunctions and cachephasehelper. Does not include attention/MoE, dependentlayer criticalpath or modelthroughput. Geometryinputreuse/cacheeffects explicit; no broadqualityclaim.')
