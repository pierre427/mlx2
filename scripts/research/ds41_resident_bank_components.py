"""Actual-checkpoint Source bank GU/middle/DOWN functional component; no model load."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from pathlib import Path
import hashlib,json,os

CHECKPOINT='DeepSeek-V4.1-Flash-1017-Source-Hot60GB-Q1g64@927325be8884e00a7bf60b9aa7d540b167a5efec569b9c7c30b1d34c2793d000'
ROOT=Path(str(Path.home() / 'mlx-models/DeepSeek-V4.1-Flash-1017-Source'))
Q1=Path(str(Path.home() / 'mlx-models/DeepSeek-V4.1-Flash-1017-Source-Expert-Q1-g64'))
HOT_SHA='371051887455ccaf61f60c8e3fb74f41c7f53a2f389c88c581991007ff4024e8'
# The pinned hot-set map lives with the private probe evidence; set
# MLX2_DS41_HOT60_MAP to its path (HOT_SHA is checked before use).
HOT=Path(os.environ.get('MLX2_DS41_HOT60_MAP', 'hot60-map.json'))


def plan():
    return dict(schema='mlx2.ds41-resident-bank-components.plan.v1',checkpoint=CHECKPOINT,
        mixtures=[[6,0],[5,1],[3,3],[1,5],[0,6]],hidden=5120,intermediate=2304,topk=6,
        payload_budget_bytes=6<<30,formats=['MXFP4_E2M1_E8M0_group32','affine_Q1_group64'],
        device_router_readback=False,numerical_claim='Compare raw bits first; classify drift independently; unqualified component',
        expert_math='current source_gate_up_fusion.batch_middle + source_down_fusion.down_combine')


def run(mx,check_live=None,*,hotmap_path=None,kernel_variants=(),timing_repeats=0):
    if not callable(check_live)or check_live()is not True:raise ValueError('Source bank component requires live GPU owner')
    if mx.default_device()!=mx.gpu:raise ValueError('Source bank component requires GPU')
    import numpy as np
    from mlx2.runtime.gguf import source_resident_bank_kernels as variants
    if any(v not in variants.VARIANTS or v=='reference'for v in kernel_variants):raise ValueError('Unknown kernel variant')
    if type(timing_repeats)is not int or not 0<=timing_repeats<=20:raise ValueError('Timing pairs must be0..20')
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    from mlx2.adapters.deepseek_v41_source_lfu import load_q1_manifest,q1_read
    from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed,borrow_q1_packed,materialize
    from mlx2.runtime.gguf.source_resident_bank_gather import bank_decode,implementation_identity
    from mlx2.runtime.gguf import source_gate_up_fusion as gu
    from mlx2.runtime.gguf.source_down_fusion import down_combine
    from mlx2.runtime.gguf.source_middle_fusion import exp_spelling
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    path=Path(hotmap_path or HOT)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=HOT_SHA:raise ValueError('Actual component hotmap binding changed')
    hot=json.loads(raw);artifact=inspect_artifact(ROOT)
    if hot['source']!=artifact['identity']['fingerprint']:raise ValueError('Source checkpoint/hotmap mismatch')
    qroot,qrecords,qbinding=load_q1_manifest(Q1,hot['source'])
    binding=dict(source=artifact['identity']['fingerprint'],q1=qbinding,hotmap=hashlib.sha256(raw).hexdigest())
    observed=ROOT.name+'-Hot60GB-Q1g64@'+hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    if observed!=CHECKPOINT:raise ValueError('Actual hybrid checkpoint identity changed: '+observed)
    hotkeys=frozenset(tuple(k)for k in hot['experts'])
    for layer in range(40):
        originals=[e for e in range(384)if(layer,e)in hotkeys]
        quantized=[e for e in range(384)if(layer,e)not in hotkeys]
        if len(originals)>=6 and len(quantized)>=6:break
    else:raise ValueError('No fixed-map layer admits requested mixtures')
    keys=[(layer,e)for e in originals[:6]+quantized[:6]]
    packed={};host_owners=[]
    result=dict(schema='mlx2.ds41-resident-bank-components.v1',checkpoint=CHECKPOINT,plan=plan(),
        implementation_sha256=implementation_identity(),qualified=False,performance_claim=False,model_loaded=False,cases=[])
    for key in keys:
        if check_live()is not True:raise RuntimeError('Source bank component ownership lost')
        host=(read_original_packed(artifact,*key)if key in hotkeys else borrow_q1_packed(q1_read(qroot,qrecords[key])))
        expert=materialize(host);host_owners.append(host);packed[key]=expert
        mx.eval(expert.backing,*expert.values())
    # Exercise the ACTUAL publisher and rebinding code, with multiple banks.
    # This store is only an owned component fixture over the twelve actual fixed-
    # map experts above; it cannot serve a model or acquire unknown experts.
    from collections import OrderedDict
    import threading
    from mlx2.adapters.deepseek_v41_source_bank import build_bank
    fixture=SimpleNamespace(_lock=threading.RLock(),_active=False,_leased={},
        _pending_uploads={},_pending_host={},_completion_collector=None,
        _cache=OrderedDict(packed),_cache_bytes=sum(e.allocation_nbytes for e in packed.values()),
        cache_budget_bytes=6<<30,binding=binding['source'],q1_binding=binding['q1'],hot_binding=binding['hotmap'],
        _ready=lambda:None,pool=SimpleNamespace(counters=lambda:{}),
        pool_bytes={kind:sum(e.allocation_nbytes for e in packed.values()if e.layout.kind==kind)for kind in ('original','q1')})
    fixture.kind=lambda key:packed[key].layout.kind
    fixture._fence=lambda:mx.synchronize()
    generation=73
    published=build_bank(mx,fixture,tuple(packed),ownercheck=check_live,generation=generation,
        budget_bytes=6<<30,max_bank_bytes=2*18_800_640)
    view_names=[];view_checks=[]
    for key,original in packed.items():
        rebound=fixture._cache[key]
        if rebound.layout!=original.layout:raise RuntimeError('Publisher changed physical ExpertLayout')
        for name in original:
            if rebound[name].shape!=original[name].shape or rebound[name].dtype!=original[name].dtype:
                raise RuntimeError('Publisher changed typed projection geometry')
            view_names.append((key,name))
            view_checks.append(mx.all(rebound[name].view(mx.uint8)==original[name].view(mx.uint8)))
    verified=mx.stack(view_checks);mx.eval(verified)
    if not bool(mx.all(verified).item()):raise RuntimeError('Actual builder rebound coefficients changed bits')
    result['actual_bank_publisher']=dict(plan=published.plan.receipt(),rebound_tensor_count=len(view_names),
        all_rebound_views_raw_bit_exact=True,multiple_banks=len(published.banks)>1,
        scope='Actual fixed-map twelve-expert coefficients; production build_bank plus exact rebound typed views')
    banks=published.banks
    try:
        with published.lease(ownercheck=check_live)as lease:
            spelling=exp_spelling();rng=np.random.default_rng(1941)
            raw_x=mx.array(rng.normal(0,.2,(1,5120)).astype(np.float32))
            prepared=fused_quantize(raw_x,'fp8').astype(mx.bfloat16).astype(mx.float32)
            def compare(a,b):
                aa=np.asarray(a.astype(mx.float32));bb=np.asarray(b.astype(mx.float32));err=aa.astype(np.float64)-bb.astype(np.float64)
                finite=bool(np.isfinite(aa).all()and np.isfinite(bb).all())
                return dict(finite=finite,bitwise_equal=bool(np.array_equal(aa.view(np.uint32),bb.view(np.uint32))),
                    max_abs_error=float(np.max(np.abs(err)))if finite else None,rmse=float(np.sqrt(np.mean(err*err)))if finite else None,
                    tolerance_passed=bool(np.allclose(aa,bb,atol=2e-4,rtol=2e-4))if finite else False)
            for original_count in (6,5,3,1,0):
                if check_live()is not True:raise RuntimeError('Source bank component ownership lost')
                expert_ids=originals[:original_count]+quantized[:6-original_count]
                # Nontrivial router order; CPU list defines the fixture, never reads GPU IDs.
                route=[expert_ids[i]for i in (4,0,5,2,1,3)]
                sorted_ids=sorted(route);order=[route.index(e)for e in sorted_ids]
                experts=[packed[(layer,e)]for e in sorted_ids]
                rebound_experts=[fixture._cache[(layer,e)]for e in sorted_ids]
                scores=mx.array([.12,.07,.31,.19,.23,.08],mx.float32)
                reference_middle=gu.batch_middle(prepared,experts,scores,mx.array(order,mx.uint32),exp_spelling=spelling)
                reference=down_combine([reference_middle[i:i+1]for i in range(6)],experts,sorted_ids)
                rebound_middle=gu.batch_middle(prepared,rebound_experts,scores,mx.array(order,mx.uint32),exp_spelling=spelling)
                rebound_output=down_combine([rebound_middle[i:i+1]for i in range(6)],rebound_experts,sorted_ids)
                candidate=bank_decode(prepared,mx.array(route,mx.uint32),scores,lease,layer=layer,
                    expected_generation=generation,check_live=check_live,exp_spelling=spelling,return_middle=True)
                # An independent consumer also exercises output ownership/fences.
                consumed=candidate.output+mx.zeros_like(candidate.output)
                mx.eval(reference,reference_middle,rebound_middle,rebound_output,consumed,candidate.status,candidate.middle,candidate.route_meta)
                status=np.asarray(candidate.status)
                row=dict(mixture=[original_count,6-original_count],route_fixture=route,expected_sorted_ids=sorted_ids,
                    status=status.tolist(),middle=compare(reference_middle,candidate.middle),output=compare(reference,candidate.output),
                    rebound_middle=compare(reference_middle,rebound_middle),rebound_output=compare(reference,rebound_output),
                    raw_reference=np.asarray(reference).tolist(),raw_candidate=np.asarray(candidate.output).tolist())
                row['passed']=bool(not np.any(status)and row['middle']['finite']and row['output']['finite']and row['middle']['tolerance_passed']and row['output']['tolerance_passed']and row['rebound_middle']['bitwise_equal']and row['rebound_output']['bitwise_equal'])
                row['numerical_classification']='observed_raw_bit_exact'if row['middle']['bitwise_equal']and row['output']['bitwise_equal']else'drift_within_tolerance'if row['passed']else'failed'
                row['kernel_variants']=_screen(mx,check_live,kernel_variants,timing_repeats,prepared,
                    mx.array(route,mx.uint32),scores,lease,layer,generation,spelling,candidate,compare)
                result['cases'].append(row)
                if not row['passed']:raise RuntimeError('Actual Source bank FFN component mismatch: '+json.dumps({k:v for k,v in row.items()if not k.startswith('raw_')}))
            # Device refusal states must survive to the whole-forward owner, never zeros.
            valid=[originals[0]]*6
            controls=[('duplicate',valid,generation,4),('invalid',[384,385,386,387,388,389],generation,2),
                      ('generation',originals[:5]+quantized[:1],generation+1,8)]
            absent=next(e for e in range(384)if(layer,e)not in packed)
            controls.append(('missing',[absent]+originals[:5],generation,1))
            result['status_cases']=[]
            for label,route,expected,flag in controls:
                if check_live()is not True:raise RuntimeError('Source bank component ownership lost')
                case=bank_decode(prepared,mx.array(route,mx.uint32),mx.ones((6,),mx.float32),lease,
                    layer=layer,expected_generation=expected,check_live=check_live,exp_spelling=spelling)
                mx.eval(case.output,case.status);status=np.asarray(case.status)
                passed=bool(np.any(status&flag)and not np.isfinite(np.asarray(case.output)).all())
                variant_controls=[]
                for variant in kernel_variants:
                    changed=variants.decode(prepared,mx.array(route,mx.uint32),mx.ones((6,),mx.float32),lease,layer=layer,
                        expected_generation=expected,check_live=check_live,exp_spelling=spelling,variant=variant)
                    mx.eval(changed.output,changed.status);vs=np.asarray(changed.status)
                    okay=bool(np.any(vs&flag)and not np.isfinite(np.asarray(changed.output)).all())
                    variant_controls.append(dict(variant=variant,status=vs.tolist(),passed=okay))
                    if not okay:raise RuntimeError('Variant status failed closed control: '+variant+':'+label)
                result['status_cases'].append(dict(label=label,status=status.tolist(),passed=passed,kernel_variants=variant_controls))
                if not passed:raise RuntimeError('Source bank invalid route did not fail closed: '+label)
            result.update(passed=True,layer=layer,bank_count=len(banks),bank_payload_bytes=sum(bank.nbytes for bank in banks),
                observed_source_binding=binding,case_count=len(result['cases'])+len(result['status_cases']),kernel_variants=list(kernel_variants),timing_repeats=timing_repeats)
            return result
    finally:
        published.close()


def _screen(mx,check_live,names,repeats,x,ids,scores,lease,layer,generation,spelling,reference,compare):
    """Guards outside each pair; full callable+GPU completion, no shader-only claim."""
    import time
    import numpy as np
    from mlx2.runtime.gguf.source_resident_bank_kernels import decode
    rows=[]
    def execute(which,*,diagnostic=False,live=check_live):
        return decode(x,ids,scores,lease,layer=layer,expected_generation=generation,check_live=live,
            exp_spelling=spelling,variant=which,return_middle=diagnostic)
    for name in names:
        changed=execute(name,diagnostic=True);mx.eval(changed.output,changed.middle,changed.status)
        output=compare(reference.output,changed.output);middle=compare(reference.middle,changed.middle)
        status=np.asarray(changed.status)
        row=dict(variant=name,middle=middle,output=output,status=status.tolist(),
            passed=bool(output['bitwise_equal']and middle['bitwise_equal']and not np.any(status)))
        rows.append(row)
        if not row['passed']:raise RuntimeError('Expert variant changed exact arithmetic: '+json.dumps(row))
        if not repeats:continue
        for _ in range(2):
            if check_live()is not True:raise RuntimeError('Kernel warm ownership lost')
            for which in ('reference',name):
                warm=execute(which,live=lambda:True);mx.eval(warm.output,warm.status)
        samples=[]
        for pair in range(repeats):
            if check_live()is not True:raise RuntimeError('Kernel timing ownership lost')
            order=('reference',name)if pair%2==0 else(name,'reference')
            raw=dict(pair=pair,order=list(order))
            for which in order:
                start=time.perf_counter_ns();value=execute(which,live=lambda:True);mx.eval(value.output,value.status)
                raw[which+'_ms']=(time.perf_counter_ns()-start)/1e6
            samples.append(raw)
        a=[r['reference_ms']for r in samples];b=[r[name+'_ms']for r in samples]
        row['timing']=dict(scope='resident complete FFN callable host dispatch+mx.eval(output,status); bankbuild/inputQAT excluded',
            warm_pairs_excluded=2,paired_order='alternatingAB/BA',raw_pairs=samples,
            reference_median_ms=float(np.median(a)),variant_median_ms=float(np.median(b)),
            ratio_of_medians=float(np.median(a)/np.median(b)),performance_claim=False)
    return rows
