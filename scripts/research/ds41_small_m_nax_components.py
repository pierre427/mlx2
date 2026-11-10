# SPDX-License-Identifier: MIT
"""Caller-owned bounded small-M original paired NAX proof; no serving wiring."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import hashlib
from pathlib import Path
import statistics
import time
from ds41_prefill_nax_profile_components import details,LIMIT
from ds41_prefill_legacy_batch_components import original_ids


def case_plans(options=None):
    options={} if options is None else options
    if (not isinstance(options,dict) or set(options)-{'all_single_m'}
            or ('all_single_m' in options and type(options['all_single_m']) is not bool)):
        raise ValueError('Small-M options require only explicit boolean all_single_m')
    single=tuple((m,) for m in (range(8,13) if options.get('all_single_m',False) else (8,10,12)))
    return [dict(partitions=p,numerical_profile=f'small_m_nax_p{p}_ordered_f32_v1',row_counts=counts)
            for p in (1,5)for counts in (*single,(8,10,12),(8,10,12)*5)]


def ordered(np,parts):
    if parts.dtype!=np.float32 or parts.ndim!=3 or parts.shape[0]not in(1,5):raise ValueError('Small-M partial law')
    out=np.zeros(parts.shape[1:],np.float32)
    for part in parts:np.add(out,part,out=out)
    return out


def admitted(checks):
    return all(v['finite']and(v['exact']or k.startswith('stock_vs_native'))for k,v in checks.items())


def finish(section):
    controls=section['reference_controls'];cases=section['cases']
    expected=case_plans(section.get('small_m_options'))
    inventory=[(c['partitions'],tuple(c['row_counts'])) for c in cases]
    complete=inventory==[(c['partitions'],c['row_counts']) for c in expected]
    if len(controls)!=4 or not all(all(v['exact']and v['finite']for v in c['checks'].values())for c in controls):
        raise ValueError('Mandatory four stock controls missing/changed')
    section.update(complete=complete,harness_completed=True,status='completed',required_reference_controls_exact=True,
        verdicts={'all_engineering_exact':complete and all(c['engineering_exact']for c in cases)and all(c['exact']for c in section['adversarials']),
                  'all_finite':complete and all(c['finite']for c in cases)})
    return section


def timed_leaves(mx,np,functions,*,repeats=3):
    """Time real outputs as leaves; no invented baseline concatenate/reorder."""
    if repeats!=3:raise ValueError('Small-M bounded three-repeat timing')
    result={k:{'samples':[],'scope':'Includes host metadata/maps, producer and scored middle; no output concatenation, weight load or router'}for k in functions}
    def invoke(fn):
        mx.synchronize();start=time.perf_counter();value=fn();submitted=time.perf_counter()
        leaves=list(value)if isinstance(value,(list,tuple))else[value]
        mx.eval(*leaves);end=time.perf_counter()
        return leaves,dict(wall_ms=(end-start)*1000,host_ms=(submitted-start)*1000,completion_ms=(end-submitted)*1000)
    for fn in functions.values():
        for _ in range(2):leaves,_=invoke(fn);del leaves
    rng=np.random.default_rng(4102612)
    for _ in range(repeats):
        for key in rng.permutation(list(functions)):
            leaves,sample=invoke(functions[key]);arrays=[np.asarray(v)for v in leaves]
            if not all(np.isfinite(a).all()for a in arrays):raise ValueError('Small-M timing nonfinite')
            digest=hashlib.sha256(b''.join(a.tobytes()for a in arrays)).hexdigest()
            if result[key].setdefault('output_sha256',digest)!=digest:raise ValueError('Small-M timing unstable')
            result[key]['samples'].append(sample);result[key]['output_leaves']=len(leaves)
            del leaves,arrays
    for r in result.values():r['median']={k:statistics.median(x[k]for x in r['samples'])for k in('wall_ms','host_ms','completion_ms')}
    return result


def consumer_adversarials(mx,np,candidate,middle,spelling):
    from ds41_prefill_nax_batch_components import adversarial_partials
    cases=[]
    for p in(1,5):
        g,u,s,ids=adversarial_partials(np,5);g=g[:p].copy();u=u[:p].copy()
        gr=ordered(np,g);ur=ordered(np,u);scores=mx.array(s,mx.float32);slots=mx.array(ids,mx.uint32)
        expected=middle(mx.array(gr),mx.array(ur),scores,slots)
        out,cg,cu=candidate.reduce_middle(mx.array(g),mx.array(u),scores,slots,exp_spelling=spelling,return_reduced=True)
        mx.eval(expected,out,cg,cu)
        checks={k:details(np,a,np.asarray(b))for k,a,b in [('gate',gr,cg),('up',ur,cu),('middle',np.asarray(expected),out)]}
        cases.append(dict(partitions=p,checks=checks,exact=all(v['exact']and v['finite']for v in checks.values())))
    return cases


def run_small_m(mx,np,capture_path,load_expert,out,report,compare,atomic):
    """Same signature as run_nax_batch; controller checks complete/verdicts."""
    from ds41_prefill_legacy_gu_components import capture_module,check_coefficients
    from mlx2.runtime.gguf import source_small_m_nax_pair as candidate
    from mlx2.runtime.gguf.source_gate_up_prefill import pp_middle
    from mlx2.runtime.gguf.source_middle_fusion import middle,exp_spelling
    case_plans(report.get('small_m_options'))
    if mx.default_device()!=mx.gpu:raise RuntimeError('Caller-owned GPU required')
    if 'small_m_nax'in report:raise ValueError('Refusing overwrite small-M evidence')
    manifest,captured=capture_module(capture_path).load_capture(capture_path)
    if len(captured)!=4 or {(c['partitions'],c['projection'])for c,_ in captured}!={(p,k)for p in(2,5)for k in('gate','up')}:
        raise ValueError('Small-M requires four sealed GU captures')
    if any(report.get(k)!=manifest[k]for k in('checkpoint','source_revision')):raise ValueError('Small-M capture identity mismatch')
    ids,hotsha=original_ids(capture_path,captured);out=Path(out);spelling=exp_spelling()
    section=report['small_m_nax']=dict(schema='mlx2.small-M-nax-components.v1',checkpoint=manifest['checkpoint'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        capture_sha256=hashlib.sha256(Path(capture_path).read_bytes()).hexdigest(),hotmap_sha256=hotsha,served_exp_spelling=spelling,
        scope='Actual-capture-derived source activation rows, independent actual original expert coefficients, synthetic route maps/scores; not actual small-M model trajectory',
        small_m_options=dict(report.get('small_m_options',{})),reference_controls=[],cases=[],adversarials=[],complete=False,harness_completed=False,required_reference_controls_exact=False,
        verdicts={},maximum_allocator_bytes=LIMIT,qualified=False,performance_qualified=False,candidate_used_by_model=False,model_quality_assessed=False)
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Small-M exceeded6GiB allocator guard')
    write();bank={}
    for eid in ids:bank[eid]=load_expert(0,eid);observe()
    for c,arrays in captured:
        e=bank[c['expert']];check_coefficients(np,c,e);a=arrays['activation']
        if a.dtype!=np.float32 or not np.isfinite(a).all()or np.any(a.view(np.uint32)&0xffff):raise ValueError('Small-M base activations not finite BF16-grid')
        x=mx.array(a,mx.float32);raw=mx.quantized_matmul(x,e[c['projection']+'.qweight'],scales=e[c['projection']+'.scales'],mode='mxfp4')
        boundary=raw.astype(mx.bfloat16).astype(mx.float32);mx.eval(raw,boundary)
        checks={'raw':details(np,arrays['reference_preround'],np.asarray(raw)),'boundary':details(np,arrays['reference_boundary'],np.asarray(boundary))}
        section['reference_controls'].append(dict(layer=c['layer'],expert=c['expert'],projection=c['projection'],checks=checks))
        if not all(v['exact']and v['finite']for v in checks.values()):write();raise ValueError('Mandatory Source replay changed')
        del x,raw,boundary;observe()
    section['adversarials']=consumer_adversarials(mx,np,candidate,middle,spelling);observe();write()
    base=next(a['activation']for c,a in captured if c['projection']=='gate'and c['partitions']==2)
    x=mx.array(base,mx.float32);mx.eval(x)
    section['source_activation_sha256']=hashlib.sha256(base.tobytes()).hexdigest()
    section['coefficient_sha256']={str(eid):{k:hashlib.sha256(np.asarray(e[k]).tobytes()).hexdigest()for k in('gate.qweight','gate.scales','up.qweight','up.scales')}for eid,e in bank.items()}
    for index,z in enumerate(case_plans(section['small_m_options'])):
        counts=z['row_counts'];es=[bank[i]for i in ids[:len(counts)]];offsets=[0]
        for n in counts:offsets.append(offsets[-1]+n)
        rowmap=np.concatenate([((np.arange(n,dtype=np.uint32)*3+j*7)%len(base))for j,n in enumerate(counts)]).astype(np.uint32)
        total=sum(counts);score_values=(np.arange(total*2,dtype=np.float32)%29-14)/32
        slot_values=((np.arange(total,dtype=np.uint32)*7+3)%(total*2)).astype(np.uint32)
        scores=mx.array(score_values);slots=mx.array(slot_values);profile=z['numerical_profile']
        # Baseline is exactly current homogeneous-M fusedGU worklists. Mixed
        # groups return leaves; reordering/concatenation is outside timing.
        groups=[]
        for m in sorted(set(counts)):
            jobs=[j for j,n in enumerate(counts)if n==m];loc=np.concatenate([np.arange(offsets[j],offsets[j+1],dtype=np.uint32)for j in jobs])
            groups.append((jobs,loc,rowmap[loc].copy(),slot_values[loc].copy()))
        def baseline(diagnostic=False):
            return [pp_middle(x,[es[j]for j in jobs],mx.array(rows),scores[mx.array(inds)],tuple(counts[j]for j in jobs),exp_spelling=spelling,return_projections=diagnostic)
                    for jobs,loc,rows,inds in groups]
        base_outputs=baseline(True)
        mx.eval(*[v for t in base_outputs for v in t]);baseline_arrays=[np.empty((total,2304),np.float32)for _ in range(3)]
        for group,outputs in zip(groups,base_outputs):
            for k,v in enumerate(outputs):baseline_arrays[k][group[1]]=np.asarray(v)
        prepared=candidate.prepare(x,es,counts,rowmap,numerical_profile=profile)
        gp,up=candidate.produce(prepared);one_g=[];one_u=[]
        for j,e in enumerate(es):
            prep=candidate.prepare(x,[e],(counts[j],),rowmap[offsets[j]:offsets[j+1]].copy(),numerical_profile=profile)
            one_g.append(candidate.produce(prep,projection='gate'));one_u.append(candidate.produce(prep,projection='up'))
        single_g=mx.concatenate(one_g,axis=1);single_u=mx.concatenate(one_u,axis=1)
        result,gr,ur=candidate.reduce_middle(gp,up,scores,slots,exp_spelling=spelling,return_reduced=True)
        normal=candidate.middle(x,es,counts,rowmap,scores,slots,numerical_profile=profile,exp_spelling=spelling)
        stock={}
        for projection in('gate','up'):
            stock[projection]=mx.concatenate([mx.quantized_matmul(x[mx.array(rowmap[offsets[j]:offsets[j+1]])],e[projection+'.qweight'],scales=e[projection+'.scales'],mode='mxfp4')for j,e in enumerate(es)],axis=0)
        stock_middle=middle(stock['gate'],stock['up'],scores,slots)
        reference_middle=middle(gr,ur,scores,slots)
        mx.eval(gp,up,single_g,single_u,result,gr,ur,normal,stock['gate'],stock['up'],stock_middle,reference_middle);observe()
        checks={}
        def check(name,a,b):checks[name]=details(np,np.asarray(a),np.asarray(b))
        check('paired_vs_independent_gate_partials',single_g,gp);check('paired_vs_independent_up_partials',single_u,up)
        check('ordered_gate_raw',ordered(np,np.asarray(gp)),gr);check('ordered_up_raw',ordered(np,np.asarray(up)),ur)
        check('same_profile_middle',reference_middle,result);check('normal_vs_diagnostic',result,normal)
        check('baseline_gate_vs_stock',stock['gate'],baseline_arrays[1]);check('baseline_up_vs_stock',stock['up'],baseline_arrays[2]);check('baseline_middle_vs_stock',stock_middle,baseline_arrays[0])
        for k,v in [('gate',gr),('up',ur)]:
            check('stock_vs_native_'+k+'_raw',stock[k],v)
            check('stock_vs_native_'+k+'_boundary',stock[k].astype(mx.bfloat16).astype(mx.float32),v.astype(mx.bfloat16).astype(mx.float32))
        check('stock_vs_native_middle',stock_middle,result)
        row={**z,'row_counts':list(counts),'expert_ids':ids[:len(counts)],'source_rows':len(base),'mapped_rows':rowmap.tolist(),'score_slots':slot_values.tolist(),
             'plan':candidate.plan(counts,numerical_profile=profile),'baseline_GU_dispatches':len(groups),'candidate_major_dispatches':2,
             'checks':checks,'engineering_exact':admitted(checks),'finite':all(v['finite']for v in checks.values()),'timing':None}
        section['cases'].append(row);write()
        if row['engineering_exact']and all(a['exact']for a in section['adversarials']):
            row['timing']=timed_leaves(mx,np,{'current_fused_GU':lambda:baseline(False),'paired_NAX_middle':lambda:candidate.middle(x,es,counts,rowmap,scores,slots,numerical_profile=profile,exp_spelling=spelling)})
        del base_outputs,baseline_arrays,prepared,gp,up,one_g,one_u,single_g,single_u,result,gr,ur,normal,stock,stock_middle,reference_middle
        observe();write()
    finish(section);write();return section
