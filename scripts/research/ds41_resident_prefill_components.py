"""Actual-checkpoint bounded PP bank functional probe; parent owns GPU."""
from pathlib import Path
import hashlib,json
from ds41_resident_bank_components import CHECKPOINT,ROOT,Q1,HOT,HOT_SHA

VALID_CASES=('m2_mixed','m4_reuse','m8_reuse','m12_reuse','m16_mixed','m32_mixed','m128_q1')
WIDE_CASES=('m13_wide','m16_wide','m32_wide','m64_wide','m96_wide','m128_wide','m32_wide_mixed')
REFUSALS=('original_13','original_16','original_32','original_128','duplicate','invalid','missing','generation')


def plan(*,wide=False,task_policy='padded_v1'):
    from mlx2.runtime.gguf.source_resident_prefill_tasks import validate_policy
    validate_policy(task_policy)
    return dict(task_policy=task_policy,schema='mlx2.ds41-resident-prefill-components.plan.v1',checkpoint=CHECKPOINT,
        valid_cases=list(VALID_CASES)+(list(WIDE_CASES)if wide else[]),refusal_cases=list(REFUSALS[4:]if wide else REFUSALS),rows=[2,4,8,12,16,32,128],
        payload_budget_bytes=6<<30,expert_fixture='12actualoriginal+12actualQ1 fixed-map experts',
        original_occupancy_limit=128 if wide else 12,qualified=False,performance_claim=False,
        reference='existing gu.middle per expert + stock MXFP4/Q1 down + ordered_combine')


def run(mx,check_live=None,*,hotmap_path=None,wide=False,task_policy='padded_v1'):
    if not callable(check_live)or check_live()is not True or mx.default_device()!=mx.gpu:
        raise ValueError('Resident PP component requires live owned GPU')
    import numpy as np
    import threading
    from collections import OrderedDict
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    from mlx2.adapters.deepseek_v41_source_lfu import load_q1_manifest,q1_read
    from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed,borrow_q1_packed,materialize
    from mlx2.adapters.deepseek_v41_source_prefill_bank import build_prefill_bank
    from mlx2.adapters.deepseek_v41_source_math import ordered_combine
    from mlx2.runtime.gguf.source_resident_prefill import bank_prefill,implementation_identity
    from mlx2.runtime.gguf.source_middle_fusion import exp_spelling,middle as separate_middle
    from mlx2.runtime.gguf.source_gate_up_fusion import middle
    from mlx2.runtime.gguf.source_q1 import matmul
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    if wide:
        from mlx2.runtime.gguf.source_resident_prefill_wide import bank_prefill_wide as bank_prefill,implementation_identity
    raw=Path(hotmap_path or HOT).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=HOT_SHA:raise ValueError('PP component hot-map changed')
    hot=json.loads(raw);artifact=inspect_artifact(ROOT);hotkeys=frozenset(tuple(k)for k in hot['experts'])
    if hot['source']!=artifact['identity']['fingerprint']:raise ValueError('PP source binding changed')
    qroot,qrecords,qbinding=load_q1_manifest(Q1,hot['source'])
    binding=dict(source=artifact['identity']['fingerprint'],q1=qbinding,hotmap=HOT_SHA)
    observed=ROOT.name+'-Hot60GB-Q1g64@'+hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    if observed!=CHECKPOINT:raise ValueError('PP hybrid checkpoint label changed')
    for layer in range(40):
        original=[e for e in range(384)if(layer,e)in hotkeys]
        qids=[e for e in range(384)if(layer,e)not in hotkeys]
        if len(original)>=12 and len(qids)>=12:break
    else:raise ValueError('No actual layer admits PP fixture')
    original,qids=original[:12],qids[:12];packed={};host_owners=[]
    for key in [(layer,e)for e in original+qids]:
        if check_live()is not True:raise RuntimeError('PP component ownership lost')
        host=read_original_packed(artifact,*key)if key in hotkeys else borrow_q1_packed(q1_read(qroot,qrecords[key]))
        host_owners.append(host);packed[key]=materialize(host)
        mx.eval(packed[key].backing,*packed[key].values())
    fixture=SimpleNamespace(_lock=threading.RLock(),_active=False,_leased={},_pending_uploads={},_pending_host={},
        _completion_collector=None,_cache=OrderedDict(packed),_cache_bytes=sum(e.allocation_nbytes for e in packed.values()),
        cache_budget_bytes=6<<30,binding=binding['source'],q1_binding=binding['q1'],hot_binding=binding['hotmap'],
        _ready=lambda:None,pool=SimpleNamespace(counters=lambda:{}),
        pool_bytes={kind:sum(e.allocation_nbytes for e in packed.values()if e.layout.kind==kind)for kind in ('original','q1')})
    fixture.kind=lambda key:packed[key].layout.kind;fixture._fence=lambda:mx.synchronize()
    # Force several cohorts and nonzero offsets, exercising the actual publisher.
    bank=build_prefill_bank(mx,fixture,tuple(packed),ownercheck=check_live,generation=91,
        budget_bytes=6<<30,max_bank_bytes=3*18_800_640,staging_bytes=0)
    result=dict(schema='mlx2.ds41-resident-prefill-components.v1',checkpoint=CHECKPOINT,
        implementation_sha256=implementation_identity(),plan=plan(wide=wide,task_policy=task_policy),bank=bank.receipt(),cases=[],refusals=[],
        qualified=False,performance_claim=False,model_loaded=False)
    rng=np.random.default_rng(41032);spelling=exp_spelling()
    def routes(rows,*,fixed=False,q1_only=False):
        values=[]
        for row in range(rows):
            if q1_only:selected=[qids[(row*3+j)%12]for j in range(6)]
            else:
                start=0 if fixed else (row%4)*3
                selected=original[start:start+3]+qids[start:start+3]
            values.append([selected[i]for i in (4,0,5,2,1,3)])
        return np.asarray(values,np.uint32)
    def prepared(rows):
        x=mx.array(rng.normal(0,.2,(rows,5120)).astype(np.float32))
        return fused_quantize(x,'fp8').astype(mx.bfloat16).astype(mx.float32)
    def compare(a,b):
        aa=np.asarray(a);bb=np.asarray(b);finite=bool(np.isfinite(aa).all()and np.isfinite(bb).all())
        err=aa.astype(np.float64)-bb.astype(np.float64)
        return dict(finite=finite,bitwise_equal=bool(np.array_equal(aa.view(np.uint32),bb.view(np.uint32))),
            max_abs_error=float(np.max(np.abs(err)))if finite else None,
            rmse=float(np.sqrt(np.mean(err*err)))if finite else None)
    try:
        with bank.lease(ownercheck=check_live)as lease:
            cases=[('m2_mixed',2,False,False),('m4_reuse',4,True,False),
                ('m8_reuse',8,True,False),('m12_reuse',12,True,False),('m16_mixed',16,False,False),
                ('m32_mixed',32,False,False),('m128_q1',128,False,True)]
            if wide:cases += [('m'+str(n)+'_wide',n,True,False)for n in (13,16,32,64,96,128)]+[('m32_wide_mixed',32,False,False)]
            for label,rows,fixed,qonly in cases:
                if check_live()is not True:raise RuntimeError('PP component ownership lost')
                ids=routes(rows,fixed=fixed,q1_only=qonly);x=prepared(rows)
                if label=='m32_wide_mixed':
                    ids=np.asarray([[original[0],original[1+r%11],original[1+(r+3)%11],qids[r%12],qids[(r+1)%12],qids[(r+2)%12]]for r in range(rows)],np.uint32)
                scores=mx.array(np.tile([.12,.07,.31,.19,.23,.08],(rows,1)).astype(np.float32));flat=ids.reshape(-1)
                order=np.argsort(flat,kind='stable');parts=[];mids=[]
                for expert_id in np.unique(flat):
                    locations=order[flat[order]==expert_id];selected=locations//6;expert=packed[(layer,int(expert_id))]
                    if expert.layout.kind=='original'and len(locations)>12:
                        activation=x[mx.array(selected,mx.uint32)]
                        gate=mx.quantized_matmul(activation,expert['gate.qweight'],scales=expert['gate.scales'],mode='mxfp4')
                        up=mx.quantized_matmul(activation,expert['up.qweight'],scales=expert['up.scales'],mode='mxfp4')
                        # This existing helper binds its served spelling
                        # internally; its public API has no override keyword.
                        if exp_spelling()!=spelling:
                            raise RuntimeError('Reference served sigmoid spelling changed')
                        mid=separate_middle(gate,up,scores.reshape(-1),mx.array(locations,mx.uint32))
                    else:
                        mid=middle(x,mx.array(selected,mx.uint32),scores.reshape(-1)[mx.array(locations,mx.uint32)],expert,exp_spelling=spelling)
                    mids.append(mid)
                    down=(matmul(mid,expert['down.qweight'],expert['down.scales'],expert['down.biases'])
                        if 'down.biases'in expert else mx.quantized_matmul(mid,expert['down.qweight'],scales=expert['down.scales'],mode='mxfp4'))
                    parts.append(down.astype(mx.bfloat16).astype(mx.float32))
                reference=ordered_combine(parts,ids,order);reference_middle=mx.concatenate(mids,axis=0)
                actual=bank_prefill(x,mx.array(ids),scores,lease,layer=layer,expected_generation=91,
                    check_live=check_live,exp_spelling=spelling,return_middle=True,task_policy=task_policy)
                mx.eval(reference,reference_middle,actual.output,actual.middle,actual.status,actual.route_meta)
                row=dict(label=label,rows=rows,status=np.asarray(actual.status).tolist(),
                    middle=compare(reference_middle,actual.middle),output=compare(reference,actual.output),
                    expert_counts=[int(np.sum(flat==e))for e in np.unique(flat)])
                row['passed']=bool(not np.any(np.asarray(actual.status))and row['middle']['bitwise_equal']and row['output']['bitwise_equal'])
                result['cases'].append(row)
                if not row['passed']:
                    error=RuntimeError('Resident PP exact component mismatch: '+json.dumps(row));error.partial_result=result;raise error
            controls=[]
            for count in (()if wide else(13,16,32,128)):controls.append(('original_'+str(count),routes(count,fixed=True),91,128))
            v=routes(2);v[0,1]=v[0,0];controls.append(('duplicate',v,91,4))
            v=routes(2);v[0,0]=384;controls.append(('invalid',v,91,2))
            absent=next(e for e in range(384)if(layer,e)not in packed)
            v=routes(2);v[0,0]=absent;controls.append(('missing',v,91,1))
            controls.append(('generation',routes(2),92,8))
            for label,ids,generation,flag in controls:
                if check_live()is not True:raise RuntimeError('PP component ownership lost')
                actual=bank_prefill(prepared(ids.shape[0]),mx.array(ids),mx.ones(ids.shape,mx.float32),lease,
                    layer=layer,expected_generation=generation,check_live=check_live,exp_spelling=spelling,task_policy=task_policy)
                mx.eval(actual.output,actual.status);status=np.asarray(actual.status)
                passed=bool(np.any(status&flag)and not np.isfinite(np.asarray(actual.output)).all())
                result['refusals'].append(dict(label=label,status=status.tolist(),passed=passed))
                if not passed:
                    error=RuntimeError('Resident PP refusal failed: '+label);error.partial_result=result;raise error
        result.update(passed=True,case_count=len(result['cases'])+len(result['refusals']),layer=layer)
        return result
    except BaseException as error:
        if not hasattr(error,'partial_result'):error.partial_result=result
        raise
    finally:bank.close()
