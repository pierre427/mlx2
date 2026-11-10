#!/usr/bin/env python3
"""Caller-owned synthetic FP4 exactness/timing probe; no checkpoint load."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import hashlib
import time
import numpy as np


def run_cases(guard):
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v4 import simulate_index_fp4
    from mlx2.runtime.models.deepseek_v4_index_qat import index_fp4,quantize_rotated,reference_rotated
    guard()
    cases=[]
    def compare(label,value,reference,candidate):
        guard();x=mx.array(value);mx.eval(x)
        left=reference(x);right=candidate(x);mx.eval(left,right)
        a,b=np.array(left),np.array(right)
        exact=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32)))
        row=dict(label=label,shape=list(value.shape),bitwise_equal=exact,finite=bool(np.isfinite(b).all()),
                 reference_sha256=hashlib.sha256(a.tobytes()).hexdigest(),candidate_sha256=hashlib.sha256(b.tobytes()).hexdigest())
        if not exact:
            indices=np.flatnonzero(a.view(np.uint32).ravel()!=b.view(np.uint32).ravel())
            i=int(indices[0]);row.update(first_flat_index=i,reference_value=float(a.ravel()[i]),candidate_value=float(b.ravel()[i]),
                                        reference_bits=int(a.view(np.uint32).ravel()[i]),candidate_bits=int(b.view(np.uint32).ravel()[i]),mismatched_elements=len(indices))
        times={}
        for name,fn in [('reference',reference),('candidate',candidate)]:
            samples=[]
            for _ in range(3):
                guard();start=time.perf_counter();result=fn(x);mx.eval(result);samples.append((time.perf_counter()-start)*1000)
            times[name+'_ms']=samples
        row['timing']=times;cases.append(row)
        return exact and row['finite']
    rng=np.random.default_rng(20261024)
    for shape in [(1,128),(1,8,128),(1,64,128,128),(1,64,512,128),(1,64,1024,128)]:
        compare('normalized_hadamard_'+str(shape),rng.normal(size=shape).astype(np.float32),simulate_index_fp4,index_fp4)
    for exponent in [-80,-10,0,10,80]:
        mid=np.array([.25,.75,1.25,1.75,2.5,3.5,5],np.float32)
        values=np.concatenate([np.nextafter(mid,-np.inf),mid,np.nextafter(mid,np.inf),[0.,-0.]])
        groups=np.zeros((8,32),np.float32);groups[:,0]=6
        for i in range(8):groups[i,1:24]=values*(1 if i%2 else -1)
        data=(groups*np.float32(2.**exponent)).reshape(2,128)
        compare('midpoints_scale_'+str(exponent),data,
                lambda x:mx.array(reference_rotated(np.array(x))),quantize_rotated)
    return dict(scope='synthetic FP4 kernels only; no full-model qualification or steady-state throughput claim',
                seed=20261024,passed=all(c['bitwise_equal'] and c['finite'] for c in cases),cases=cases)
