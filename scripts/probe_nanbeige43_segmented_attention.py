"""Import-only diagnostics for the parent's already-owned GPU window."""
import hashlib
import json
import subprocess


def round_reference(values, dtype):
    """Round FP64 oracle values directly, including BF16 midpoint neighbors.

    Selecting adjacent BF16 values in FP64 avoids FP64->FP32->BF16 double
    rounding at a BF16 midpoint. Inputs to this diagnostic must be finite.
    """
    import numpy as np
    values=np.asarray(values,dtype=np.float64)
    if dtype=="float32": return values.astype(np.float32).astype(np.float64)
    if dtype=="float16": return values.astype(np.float16).astype(np.float64)
    if dtype!="bfloat16": raise ValueError(dtype)
    high=values.astype(np.float32).view(np.uint32)>>16
    choices=np.stack([np.maximum(high,1)-1,high,np.minimum(high+1,65535)]).astype(np.uint32)
    candidate=(choices<<16).view(np.float32).astype(np.float64)
    distances=np.abs(candidate-values)
    # Stable even-first ordering resolves exact midpoint ties to even.
    distances=np.where(np.isfinite(distances),distances,np.inf)
    minimum=np.min(distances,axis=0)
    tied=distances==minimum
    preference=np.where(tied,choices&1,2)
    selected=np.argmin(preference,axis=0)
    return np.take_along_axis(candidate,selected[None],axis=0)[0]


def float64_attention_reference(q, keys, values, prefixes, ancestors, scale):
    """Independent NumPy GQA and ancestor-mask oracle over preserved inputs."""
    import numpy as np
    q=np.asarray(q,dtype=np.float64); ancestors=np.asarray(ancestors,dtype=bool)
    outputs=[]
    for lane,prefix in enumerate(prefixes):
        # Independent explicit head mapping: query head h owns KV head h//7.
        k=np.asarray(keys[lane],dtype=np.float64)[0][np.arange(56)//7]
        v=np.asarray(values[lane],dtype=np.float64)[0][np.arange(56)//7]
        scores=np.einsum('hwd,hnd->hwn',q[lane],k)*scale
        visible=np.concatenate([np.ones((q.shape[2],prefix),dtype=bool),ancestors[lane]],axis=1)
        scores=np.where(visible[None],scores,-np.inf)
        weights=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
        weights/=weights.sum(axis=-1,keepdims=True)
        outputs.append(np.einsum('hwn,hnd->hwd',weights,v))
    return np.stack(outputs)


def numerical_metrics(candidate,native,reference,dtype):
    import numpy as np
    candidate=np.asarray(candidate,dtype=np.float64); native=np.asarray(native,dtype=np.float64)
    rounded=round_reference(reference,dtype)
    def ordered_bits(values):
        if dtype=='float32':
            bits=values.astype(np.float32).view(np.uint32).astype(np.int64); sign=1<<31
        elif dtype=='float16':
            bits=values.astype(np.float16).view(np.uint16).astype(np.int64); sign=1<<15
        else:
            bits=(values.astype(np.float32).view(np.uint32)>>16).astype(np.int64); sign=1<<15
        return np.where(bits&sign,sign-(bits&(sign-1)),sign+bits)
    def one(values):
        ulps=np.abs(ordered_bits(values)-ordered_bits(rounded))
        worst_ulp=tuple(int(n) for n in np.unravel_index(np.argmax(ulps),ulps.shape))
        return {'max_abs_vs_float64':float(np.max(np.abs(values-reference))),
                'max_abs_vs_rounded_reference':float(np.max(np.abs(values-rounded))),
                'max_output_ulp_vs_rounded_reference':int(np.max(ulps)),
                'worst_ulp_coordinate':list(worst_ulp),'worst_ulp_value':float(values[worst_ulp]),
                'worst_ulp_rounded_reference':float(rounded[worst_ulp]),
                'rounded_reference_exact_fraction':float(np.mean(values==rounded))}
    worst=tuple(int(n) for n in np.unravel_index(np.argmax(np.abs(candidate-native)),candidate.shape))
    return {'candidate':one(candidate),'native':one(native),
            'worst_native_difference':{'index':list(worst),'candidate':float(candidate[worst]),'native':float(native[worst]),
                'float64_reference':float(reference[worst]),'rounded_reference':float(rounded[worst])},
            'oracle_scope':'FP64 NumPy dot/softmax on exact preserved input values; rounded output reference. Not a mathematical exact-real softmax proof.'}


def run_probe(*, receipt_callback=None, assert_on_failure=True):
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.nanbeige43_segmented_attention import (
        LAW,
        attend_segments,
        source_identity,
    )
    if mx.default_device()!=mx.gpu:
        raise RuntimeError('requires parent-owned GPU window')
    results=[]; trace=[]
    failures=[]
    for dtype in (mx.float32,mx.bfloat16,mx.float16):
        for batch in (2,4):
            for prefix in (0,257,258,4096):
                mx.random.seed(4276)
                w=5; prefixes=[max(0,prefix-i*251) for i in range(batch)]
                q=mx.random.normal((batch,56,w,128)).astype(dtype)
                # Retain original allocated strides/capacity in every lane.
                keys=[mx.random.normal((1,8,p+w+256,128)).astype(dtype)[:,:,:p+w] for p in prefixes]
                values=[mx.random.normal((1,8,p+w+256,128)).astype(dtype)[:,:,:p+w] for p in prefixes]
                parents=[-1,0,0,2,1]
                from mlx2.runtime.attention_tree_executor import _ancestors
                ancestors=mx.array([_ancestors(parents)]*batch,dtype=mx.bool_)
                refs=[]
                for i,p in enumerate(prefixes):
                    mask=mx.concatenate([mx.ones((w,p),dtype=mx.bool_),ancestors[i]],axis=1)[None,None]
                    refs.append(mx.fast.scaled_dot_product_attention(q[i:i+1],keys[i],values[i],scale=128**-.5,mask=mask))
                reference=mx.concatenate(refs,axis=0)
                # Cast BF16 to FP32 before crossing NumPy: values are preserved exactly.
                as_numpy=lambda value: np.asarray(value.astype(mx.float32))
                oracle=float64_attention_reference(as_numpy(q),[as_numpy(k) for k in keys],
                    [as_numpy(v) for v in values],prefixes,np.asarray(ancestors),128**-.5)
                for rows in (1,4):
                    output=attend_segments(q,keys,values,prefixes=prefixes,ancestors=ancestors,scale=128**-.5,shared_kv_rows=rows,dispatch_trace=trace)
                    mx.eval(output,reference)
                    error=float(mx.max(mx.abs(output.astype(mx.float32)-reference.astype(mx.float32))).item())
                    finite=bool(mx.all(mx.isfinite(output)).item())
                    legacy_gate=finite and error<(3e-5 if dtype==mx.float32 else .03)
                    metrics=numerical_metrics(as_numpy(output),as_numpy(reference),oracle,str(dtype).split('.')[-1])
                    # Peer content changes cannot affect lane0 in the SAME shape.
                    changed=list(keys); changed[-1]=changed[-1]*-2
                    isolated=attend_segments(q,changed,values,prefixes=prefixes,ancestors=ancestors,scale=128**-.5,shared_kv_rows=rows)
                    mx.eval(isolated)
                    isolated_exact=bool(mx.array_equal(output[0],isolated[0]).item())
                    result={'dtype':str(dtype),'batch':batch,'prefixes':prefixes,'rows':rows,'error':error,'finite':finite,
                        'lane0_peer_isolation_exact':isolated_exact,'legacy_native_difference_gate_pass':legacy_gate,
                        'independent_oracle':metrics}
                    if not legacy_gate:
                        lane,head,row,dim=metrics['worst_native_difference']['index']
                        if prefixes[lane]+w<=32:
                            result['bounded_counterexample']={'lane':lane,'head':head,'row':row,'dim':dim,
                                'query':as_numpy(q)[lane,head,row].tolist(),
                                'keys':as_numpy(keys[lane])[0,head//7].tolist(),
                                'values':as_numpy(values[lane])[0,head//7].tolist(),
                                'ancestors':np.asarray(ancestors)[lane,row].tolist(),'prefix':prefixes[lane]}
                    results.append(result)
                    if not legacy_gate or not isolated_exact: failures.append(result)
                    if receipt_callback is not None:
                        receipt_callback({'results':list(results),'failures':list(failures),'complete':False,
                            'kernel_sha256':source_identity(),'law':LAW})
    fixture=json.dumps({'seed':4276,'cases':results},sort_keys=True)
    receipt={'checkpoint':'synthetic-segmented-gqa7@sha256:'+hashlib.sha256(fixture.encode()).hexdigest(),
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'kernel_sha256':source_identity(),'law':LAW,'results':results,'dispatch_trace':trace,
            'scope':'synthetic numerics and same-shape lane isolation; not model quality, speed, or qualification',
            'complete':True,'legacy_gates_passed':not failures,'failures':failures}
    if receipt_callback is not None: receipt_callback(receipt)
    if assert_on_failure and failures:
        raise AssertionError('segmented diagnostic legacy/native or isolation gate failed; all metrics captured',failures)
    return receipt


if __name__=='__main__':
    raise SystemExit('Import run_probe() inside the parent-owned GPU campaign; standalone GPU execution refused.')
