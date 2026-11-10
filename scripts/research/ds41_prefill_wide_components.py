# SPDX-License-Identifier: MIT
"""Optional actual-expert BF16-storage oracle; caller owns GPU and paired lease.

No GPU work or MLX import on import. This probes the existing Source BF16
projection boundary, not raw pre-round F32 equivalence or model qualification.
Dispatch provenance: docs/audits/2026-10-25-ds41-prefill-batching/installed-dispatch.json.
No installed kernel source is copied; only public stock QMM is invoked.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import hashlib
import math
from pathlib import Path

ROWS = (12, 13, 16, 32, 33, 64, 65, 96, 97)
LIMIT = 6 << 30


def splitk_hint(rows, output_width, input_width):
    """Pinned0.32.4 CPU dispatch reconstruction, not a GPU engagement receipt."""
    if rows <= 12:
        return {'family_hint':'qmv', 'partitions':None}
    parts=min(input_width//32,max(1,512//(math.ceil(rows/32)*math.ceil(output_width/32))))
    while parts>1 and input_width%(parts*32):
        parts-=1
    return {'family_hint':'legacy_splitk' if parts>1 else 'qmm_dtype_gated',
            'partitions':parts}


def raw_details(np, expected, actual):
    """CPU byte comparison including signed zero; no tolerance admission."""
    expected=np.ascontiguousarray(expected)
    actual=np.ascontiguousarray(actual)
    if expected.shape!=actual.shape or expected.dtype!=np.float32 or actual.dtype!=np.float32:
        raise ValueError('Wide Source oracle requires matching F32 boundary arrays')
    a,b=expected.view(np.uint32).ravel(),actual.view(np.uint32).ravel()
    where=np.flatnonzero(a!=b)
    delta=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    result={'exact':len(where)==0,'mismatched_f32_words':int(len(where)),
            'shape':list(actual.shape),'max_abs_error':float(np.max(delta)),
            'reference_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256(actual.tobytes()).hexdigest()}
    if len(where):
        i=int(where[0]);result['first_raw_mismatch']={
            'flat_index':i,'coordinate':[int(v)for v in np.unravel_index(i,actual.shape)],
            'reference_f32_bits':f'{int(a[i]):08x}','candidate_f32_bits':f'{int(b[i]):08x}',
            'reference_value':float(expected.ravel()[i]),'candidate_value':float(actual.ravel()[i]),
            'absolute_error':float(delta.ravel()[i])}
    return result


def run_wide(mx,np,bank,artifact,out,report,compare,atomic):
    """Two actual original layer0 experts, three projections, nine row widths.

    Nonexact candidate boundaries are optional negative evidence, not a GU/down
    required-gate failure. Memory, ownership and malformed-input failures raise.
    Parent compare performs3 interleaved repetitions only for exact candidates.
    """
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    if 'prefill_wide_bf16' in report:
        raise ValueError('Refusing to overwrite Source wide BF16 evidence')
    if mx.default_device()!=mx.gpu:
        raise RuntimeError('Wide Source probe requires caller-owned Metal GPU')
    selected=[e for e,d in sorted(bank.items()) if 'gate.biases' not in d][:2]
    if not selected:
        raise ValueError('Wide Source probe requires actual original layer0 experts')
    out=Path(out)
    section=report['prefill_wide_bf16']={
        'checkpoint':'DeepSeek-V4.1-Flash-1017-Source@'+artifact['identity']['fingerprint'],
        'scope':'Actual layer0 original MXFP4 expert weights, synthetic BF16-rounded FP8-QAT input; existing BF16 output-boundary comparison only',
        'optional_numerical_experiment':True,'qualified':False,'model_quality_established':False,
        'raw_preround_f32_equivalence_claimed':False,'nax_engagement_measured':False,
        'allocator_ceiling_bytes':LIMIT,'maximum_observed_peak_mlx_bytes':int(mx.get_peak_memory()),
        'expert_ids':selected,'rows':list(ROWS),'cases':[],
        'timing_scope':'Exact variants only;3 interleaved repeats. BF16 prepared excludes input cast; BF16 cast-inclusive includes it. Both include output conversion to F32.',
        'dispatch_hint_scope':'Pinned installed library CPU reconstruction; no shader capture. M12 is vector control; splitK legacy evenBF16. P1 BF16 may admitNAX underhardware/alignment whileTF32offF32doesnot.',
    }
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory())
        section['maximum_observed_peak_mlx_bytes']=max(section['maximum_observed_peak_mlx_bytes'],peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:
            write();raise MemoryError('Source wide BF16 component exceeded6GiB MLX allocator ceiling')
    def raw(fn):
        value=fn();mx.eval(value);result=np.array(value);del value
        return result
    rounded=lambda value:value.astype(mx.bfloat16).astype(mx.float32)
    rng=np.random.default_rng(4102533)
    observe();write()
    for rows in ROWS:
        for width,projections in ((5120,('gate','up')),(2304,('down',))):
            values=rng.normal(0,.25,(rows,width)).astype(np.float32)
            values[:,:8]=np.array([0.,-0.,1.,-1.,.5,-.5,2**-15,-2**-15],np.float32)
            x=fused_quantize(rounded(mx.array(values)),'fp8').astype(mx.float32)
            xb=x.astype(mx.bfloat16);xr=xb.astype(mx.float32);mx.eval(x,xb,xr)
            input_check=raw_details(np,np.array(x),np.array(xr))
            if not input_check['exact']:
                section['input_lossless_failure']={'rows':rows,'width':width,**input_check}
                write();raise ValueError('FP8-QAT input is not losslessly representable as BF16')
            for expert_id in selected:
                data=bank[expert_id]
                for projection in projections:
                    q,s=data[projection+'.qweight'],data[projection+'.scales']
                    n=2304 if projection!='down' else 5120
                    if q.shape!=(n,width//8) or s.shape!=(n,width//32) or q.dtype!=mx.uint32 or s.dtype!=mx.uint8:
                        raise ValueError('Actual original expert geometry mismatch')
                    def reference():return rounded(mx.quantized_matmul(x,q,scales=s,mode='mxfp4'))
                    def prepared():return mx.quantized_matmul(xb,q,scales=s,mode='mxfp4').astype(mx.float32)
                    def inclusive():return mx.quantized_matmul(x.astype(mx.bfloat16),q,scales=s,mode='mxfp4').astype(mx.float32)
                    functions={'reference':reference,'bf16_prepared':prepared,'bf16_cast_inclusive':inclusive}
                    expected=raw(reference)
                    raw_results={name:raw_details(np,expected,expected if name=='reference' else raw(fn))for name,fn in functions.items()}
                    del expected
                    observe()
                    variants=compare(mx,np,functions,repeats=3)
                    row={'expert_id':expert_id,'projection':projection,'rows':rows,'input_width':width,'output_width':n,
                         'dispatch_hint':splitk_hint(rows,n,width),'input_bf16_roundtrip_raw_exact':True,
                         'raw_boundary':raw_results,'variants':variants,
                         'boundary_exact':all(v['exact']for v in raw_results.values())}
                    section['cases'].append(row);observe();write()
            x=None;xb=None;del functions,reference,prepared,inclusive,xr,values
            mx.synchronize()
    section['all_optional_boundaries_exact']=all(c['boundary_exact']for c in section['cases'])
    section['cases_count']=len(section['cases'])
    section['complete']=True
    observe();write()
