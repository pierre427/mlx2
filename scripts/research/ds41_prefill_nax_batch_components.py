# SPDX-License-Identifier: MIT
"""Bounded native-law batching oracle over derived actual GU activations.

Caller owns the paired GPU lease, source freeze, expert loader and cleanup.
Batch vs single native partial/raw/boundary/middle equality is mandatory;
stock vs native numerical differences are observations of a distinct profile.
"""
import hashlib
from pathlib import Path
from ds41_prefill_nax_profile_components import PROFILE,LIMIT,details,ordered_partials,time_different_profile
from ds41_prefill_legacy_batch_components import original_ids,derived_rows


def case_plans():
    rows=[]
    for widths in ((13,16,32),(33,34,64),(65,96)):
        for e in (1,3,15):
            counts=tuple(widths[i%len(widths)]for i in range(e))
            rows.append(dict(row_counts=counts,experts=e,partitions=5 if widths[0]==13 else 2,
                             row_tiles=(widths[0]+31)//32))
    return rows


def fused_option(report):
    options=report.get('nax_batch_options',{})
    if not isinstance(options,dict) or set(options)-{'fused_middle'} or type(options.get('fused_middle',False))is not bool:
        raise ValueError('Unknown NAX batch component options')
    return options.get('fused_middle',False)


def adversarial_partials(np,partitions):
    """Synthetic signed-zero/cancellation/midpoint/clamp cases, not activations."""
    if partitions not in (2,5):raise ValueError('Adversarial NAX partitions')
    pattern=np.array([-0.,0.,2**24,-2**24,1.00390625,-1.00390625,10.,10.03125,
                      -10.,-10.03125,88.,-88.,2**-126,-2**-126,1.,-1.],np.float32)
    a=np.tile(pattern,8*2304//len(pattern)).reshape(8,2304)
    gate=np.zeros((partitions,8,2304),np.float32);up=gate.copy()
    gate[0]=a;up[0]=np.roll(a,3,axis=1)
    gate[1]=np.where(np.abs(a)>=2**24,-a,np.copysign(np.float32(0),a));up[1]=np.roll(gate[1],3,axis=1)
    if partitions==5:
        gate[2]=np.float32(2**-20);gate[3]=np.float32(-2**-20)
        up[2]=np.float32(-2**-20);up[3]=np.float32(2**-20)
        gate[:,0,2]=np.array([2**24,1.,-2**24,0.,0.],np.float32)
        up[:,0,2]=np.array([-2**24,-1.,2**24,0.,0.],np.float32)
    scores=np.array([-0.,0.,1.,-1.,.5,-.25,1.0000001192092896,-1.0000001192092896],np.float32)
    slots=np.array([0,1,7,2,6,3,4,5],np.uint32)
    return gate,up,scores,slots


def adversarial_checks(mx,np,reduce_fn,middle_fn,spelling):
    from mlx2.runtime.gguf.source_prefill_legacy_tiles import _reduction_kernel
    rows=[]
    for p in (2,5):
        gp,up,scores,slots=adversarial_partials(np,p)
        gate=mx.array(gp,mx.float32);u=mx.array(up,mx.float32);s=mx.array(scores,mx.float32);ids=mx.array(slots,mx.uint32)
        metadata=mx.array([p,8*2304,2304],mx.int32)
        def reference(parts):return _reduction_kernel()(inputs=[parts,metadata],grid=(2304,8,1),threadgroup=(256,1,1),output_shapes=[(8,2304)],output_dtypes=[mx.float32])[0]
        graw,uraw=reference(gate),reference(u)
        expected=middle_fn(graw.astype(mx.bfloat16).astype(mx.float32),uraw.astype(mx.bfloat16).astype(mx.float32),s,ids)
        actual,ag,au=reduce_fn(gate,u,s,ids,exp_spelling=spelling,return_reduced=True)
        normal=reduce_fn(gate,u,s,ids,exp_spelling=spelling)
        mx.eval(graw,uraw,expected,actual,ag,au,normal)
        checks={'gate_seed_order':details(np,ordered_partials(np,gp),np.asarray(graw)),
                'up_seed_order':details(np,ordered_partials(np,up),np.asarray(uraw)),
                'fused_gate_raw':details(np,np.asarray(graw),np.asarray(ag)),
                'fused_up_raw':details(np,np.asarray(uraw),np.asarray(au)),
                'fused_middle':details(np,np.asarray(expected),np.asarray(actual)),
                'diagnostic_vs_normal':details(np,np.asarray(actual),np.asarray(normal))}
        rows.append(dict(partitions=p,checks=checks,all_exact=all(v['exact']and v['finite']for v in checks.values())))
    return rows


def run_nax_batch(mx,np,capture_path,load_expert,out,report,compare,atomic,*,single=None,batch=None,middle_fn=None,profile_timing=None,plans=None,fused_fn=None,reduce_fn=None):
    from ds41_prefill_legacy_gu_components import capture_module,check_coefficients
    if 'prefill_nax_batch'in report:raise ValueError('Refusing to overwrite native batch evidence')
    if mx.default_device()!=mx.gpu:raise RuntimeError('Caller-owned GPU required')
    fused=fused_option(report);spelling=None
    if fused:
        from mlx2.runtime.gguf.source_prefill_nax_middle import batch_middle as fused_middle,reduce_middle
        from mlx2.runtime.gguf.source_middle_fusion import exp_spelling
        fused_fn=fused_fn or fused_middle;reduce_fn=reduce_fn or reduce_middle;spelling=exp_spelling()
    if single is None or batch is None:
        from mlx2.runtime.gguf.source_prefill_nax_profile import project,batch_project
        single=single or project;batch=batch or batch_project
    if middle_fn is None:
        from mlx2.runtime.gguf.source_middle_fusion import middle
        middle_fn=middle
    timing=profile_timing or time_different_profile
    manifest,captured=capture_module(capture_path).load_capture(capture_path)
    if len(captured)!=4 or {(c['partitions'],c['projection'])for c,_ in captured}!={(p,k)for p in (2,5)for k in ('gate','up')}:
        raise ValueError('Native batch requires four original captured GU controls')
    if any(report.get(k)!=manifest[k]for k in ('checkpoint','source_revision')):raise ValueError('Native batch capture identity mismatch')
    ids,hotsha=original_ids(capture_path,captured);plans=case_plans()if plans is None else plans
    out=Path(out);section=report['prefill_nax_batch']=dict(schema='mlx2.source-nax-batch-components.v1',numerical_profile=PROFILE,
        checkpoint=manifest['checkpoint'],capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        component_source_snapshot_sha256=report.get('source_snapshot_sha256'),capture_path=str(Path(capture_path).resolve()),
        capture_sha256=hashlib.sha256(Path(capture_path).read_bytes()).hexdigest(),hotmap_sha256=hotsha,
        scope='Actual-activation-derived operator cases with original coefficients; crop/repeat rows and synthetic router scores/slots are not whole-model trajectories',
        maximum_allocator_bytes=LIMIT,planned_geometry=plans,reference_controls=[],cases=[],complete=False,candidate_used_by_model=False,
        qualified=False,model_quality_assessed=False,coefficient_sha256={},fused_middle_requested=fused,
        fused_middle_adversarials=[],all_fused_middle_exact=None,served_exp_spelling=spelling)
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Native batch exceeded6GiB allocator bound')
    observe();write();bank={}
    if fused:
        section['fused_middle_adversarials']=adversarial_checks(mx,np,reduce_fn,middle_fn,spelling)
        observe();write()
    for eid in ids:bank[eid]=load_expert(0,eid);observe()
    for c,arrays in captured:
        expert=bank[c['expert']];check_coefficients(np,c,expert)
        activation=arrays['activation']
        if activation.dtype!=np.float32 or not np.isfinite(activation).all() or np.any(activation.view(np.uint32)&0xffff):
            raise ValueError('Captured NAX activation must be finite lossless BF16-grid F32')
        x=mx.array(activation,mx.float32)
        raw=mx.quantized_matmul(x,expert[c['projection']+'.qweight'],scales=expert[c['projection']+'.scales'],mode='mxfp4')
        boundary=raw.astype(mx.bfloat16).astype(mx.float32);mx.eval(raw,boundary)
        checks={'raw':details(np,arrays['reference_preround'],np.asarray(raw)),
                'boundary':details(np,arrays['reference_boundary'],np.asarray(boundary))}
        section['reference_controls'].append(dict(layer=c['layer'],expert=c['expert'],projection=c['projection'],rows=c['rows'],checks=checks))
        if not all(v['exact']for v in checks.values()):write();raise ValueError('Mandatory captured stock replay changed')
        del x,raw,boundary;observe()
    for eid,e in bank.items():
        if 'gate.biases'in e:raise ValueError('Native batch admits original experts only')
        record={}
        for name,shape,dtype in [('gate.qweight',(2304,640),np.uint32),('up.qweight',(2304,640),np.uint32),('gate.scales',(2304,160),np.uint8),('up.scales',(2304,160),np.uint8)]:
            arr=np.asarray(e[name])
            if arr.shape!=shape or arr.dtype!=dtype:raise ValueError('Native batch coefficient geometry')
            record[name]=hashlib.sha256(arr.tobytes()).hexdigest()
        section['coefficient_sha256'][str(eid)]=record
    bases={c['partitions']:a['activation']for c,a in captured if c['projection']=='gate'}
    for index,plan in enumerate(plans):
        counts=tuple(plan['row_counts']);experts=[bank[e]for e in ids[:len(counts)]]
        if (not 1<=len(counts)<=15 or any(type(m)is not int or not 13<=m<=96 for m in counts)
                or len({(5 if m<=32 else 2,(m+31)//32)for m in counts})!=1
                or (plan['partitions'],plan['row_tiles'],plan['experts'])!=(5 if counts[0]<=32 else 2,(counts[0]+31)//32,len(counts))):raise ValueError('Native batch case band')
        rows=[derived_rows(np,bases[plan['partitions']],m,j)for j,m in enumerate(counts)]
        values=np.concatenate(rows,axis=0);x=mx.array(values,mx.float32);slices=[];offset=0
        for m in counts:slices.append(x[offset:offset+m]);offset+=m
        total=sum(counts);score_values=(np.arange(total*2,dtype=np.float32)%29-14)/32
        slot_values=((np.arange(total,dtype=np.uint32)*7+3)%(total*2)).astype(np.uint32)
        scores=mx.array(score_values,mx.float32);slots=mx.array(slot_values,mx.uint32)
        checks={};singles={};batched={};stock={}
        for projection in ('gate','up'):
            one=[single(v,e,projection=projection,return_partials=True,numerical_profile=PROFILE)for v,e in zip(slices,experts)]
            singles[projection]=tuple(mx.concatenate([v[k]for v in one],axis=1 if k==2 else 0)for k in range(3))
            batched[projection]=batch(x,experts,counts,projection=projection,return_partials=True,numerical_profile=PROFILE)
            stock[projection]=mx.concatenate([mx.quantized_matmul(v,e[projection+'.qweight'],scales=e[projection+'.scales'],mode='mxfp4')for v,e in zip(slices,experts)],axis=0)
            mx.eval(*singles[projection],*batched[projection],stock[projection]);observe()
            for k,name in enumerate(('boundary','raw','partials')):
                checks[projection+'_batch_vs_single_'+name]=details(np,np.asarray(singles[projection][k]),np.asarray(batched[projection][k]))
            checks[projection+'_batch_ordered']=details(np,ordered_partials(np,np.asarray(batched[projection][2])),np.asarray(batched[projection][1]))
            checks[projection+'_stock_vs_native_raw']=details(np,np.asarray(stock[projection]),np.asarray(batched[projection][1]))
            checks[projection+'_stock_vs_native_boundary']=details(np,np.asarray(stock[projection].astype(mx.bfloat16).astype(mx.float32)),np.asarray(batched[projection][0]))
        reference_middle=middle_fn(stock['gate'],stock['up'],scores,slots)
        single_middle=middle_fn(singles['gate'][0],singles['up'][0],scores,slots)
        batch_middle=middle_fn(batched['gate'][0],batched['up'][0],scores,slots)
        mx.eval(reference_middle,single_middle,batch_middle);observe()
        checks['batch_vs_single_middle']=details(np,np.asarray(single_middle),np.asarray(batch_middle))
        checks['stock_vs_native_middle']=details(np,np.asarray(reference_middle),np.asarray(batch_middle))
        if fused:
            combined,cgp,cup=fused_fn(x,experts,counts,scores,slots,exp_spelling=spelling,numerical_profile=PROFILE,return_partials=True)
            mx.eval(combined,cgp,cup);observe()
            checks['fused_gate_partials']=details(np,np.asarray(batched['gate'][2]),np.asarray(cgp))
            checks['fused_up_partials']=details(np,np.asarray(batched['up'][2]),np.asarray(cup))
            checks['fused_middle']=details(np,np.asarray(batch_middle),np.asarray(combined))
            del combined,cgp,cup
        engineering=all(v['exact']and v['finite']for k,v in checks.items()if 'stock_vs_native'not in k)
        row=dict(**plan,expert_ids=ids[:len(counts)],checks=checks,batch_addressing_exact=engineering,
                 finite=all(v['finite']for v in checks.values()),activation_derivation='Captured P-band gate activation rows[(row+expert_slot)%capturedM]; independently selected original coefficients',
                 router_inputs='Deterministic synthetic F32 scores with scrambled/duplicate U32 slots; not captured router weights')
        section['cases'].append(row);write()
        fused_adversarial_exact=all(r['all_exact']for r in section['fused_middle_adversarials'])
        if engineering and row['finite'] and fused_adversarial_exact:
            def project_pair(fn):
                if fn=='stock':return middle_fn(*[mx.concatenate([mx.quantized_matmul(v,e[k+'.qweight'],scales=e[k+'.scales'],mode='mxfp4')for v,e in zip(slices,experts)],axis=0)for k in ('gate','up')],scores,slots)
                if fn=='single':return middle_fn(*[mx.concatenate([single(v,e,projection=k,numerical_profile=PROFILE)for v,e in zip(slices,experts)],axis=0)for k in ('gate','up')],scores,slots)
                return middle_fn(*[batch(x,experts,counts,projection=k,numerical_profile=PROFILE)for k in ('gate','up')],scores,slots)
            functions={name:(lambda name=name:project_pair(name))for name in ('stock','single','batch')}
            if fused:functions['fused_middle']=lambda:fused_fn(x,experts,counts,scores,slots,exp_spelling=spelling,numerical_profile=PROFILE)
            row['timing']=timing(mx,np,functions,repeats=3)
        else:row['timing']=None
        del x,slices,rows,values,singles,batched,stock,one,reference_middle,single_middle,batch_middle,scores,slots
        observe();write()
    section.update(status='completed',harness_completed=True,complete=(len(section['cases'])==len(plans)==9),required_reference_controls_exact=True,
                   all_batch_addressing_exact=all(r['batch_addressing_exact']for r in section['cases']),
                   all_finite=all(r['finite']for r in section['cases']))
    if fused:
        section['all_fused_middle_exact']=(len(section['fused_middle_adversarials'])==2 and all(r['all_exact']for r in section['fused_middle_adversarials'])
            and all(r['checks']['fused_middle']['exact']for r in section['cases']) and section['all_batch_addressing_exact'])
        section['all_batch_addressing_exact'] &= section['all_fused_middle_exact']
    write();return section
