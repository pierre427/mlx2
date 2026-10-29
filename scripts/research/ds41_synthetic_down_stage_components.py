# SPDX-License-Identifier: MIT
"""Synthetic tiny-row DOWN+ordered assembly costs; parent owns GPU admission.

run(mx,check_live=None)->JSON. No MLX/NumPy import or device work on import.
Primary scope includes fresh CPU planning/validation and activation packing.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import statistics
import time

SEED=41012627
REPEATS=7
BATCH_SIZE=15
WIDTH_SWEEP=(1,3,5,8)
ARRAY_BUDGET=3*1024**3
VARIANTS=('per_expert_stock_concat','old_worklist_mapped','direct_worklist_mapped')
HYBRID_VARIANTS=('hybrid_q1_old_concat','hybrid_q1_direct_concat')
ASSEMBLY_VARIANTS=('assembly_only_concat','assembly_only_direct_mapped')
ASSEMBLY_PROFILES=(('assembly60_ragged',(1,2,3)*20),('assembly124_ragged',(1,2,3)*40+(3,)*4))
PROFILES=(
    ('original15_R2',('original',)*15,(2,)*15),
    ('original15_R6',('original',)*15,(6,)*15),
    ('q1_15_ragged',('q1',)*15,(1,2,3)*5),
    ('mixed15_ragged',('original',)*9+('q1',)*6,(2,3,4,5,6,7,2,3,4,1,2,3,1,2,3)),
)


def _routes(np,rng,counts):
    """Bipartite degree construction: distinct six IDs/token, exact job counts."""
    counts=tuple(counts)
    if (not counts or any(type(n)is not int or n<1 for n in counts)
            or sum(counts)%6 or max(counts)>sum(counts)//6):
        raise ValueError('Synthetic routes require feasible six distinct experts/token')
    rows=sum(counts)//6;capacities=np.full(rows,6,np.int32);matrix=[[]for _ in range(rows)]
    ids=tuple(3+3*i for i in range(len(counts)))
    # Descending degree Havel-Hakimi; random tie keys avoid a sorted route fixture.
    for i in sorted(range(len(counts)),key=lambda i:(-counts[i],i)):
        choices=np.lexsort((rng.random(rows),-capacities))[:counts[i]]
        if np.any(capacities[choices]<=0):raise ValueError('Synthetic route degree construction failed')
        for row in choices:matrix[int(row)].append(ids[i]);capacities[row]-=1
    if np.any(capacities):raise ValueError('Synthetic routes left unfilled slots')
    values=np.asarray([rng.permutation(row)for row in matrix],np.uint32)
    return values,ids


def plan():
    profiles=[]
    for name,kinds,counts in PROFILES:
        rows=sum(counts)//6
        grouped={}
        for kind,count in zip(kinds,counts):grouped.setdefault((kind,count if kind=='original'else 0),[]).append(count)
        profiles.append(dict(name=name,experts=len(counts),token_rows=rows,route_occurrences=sum(counts),
            kinds=list(kinds),row_counts=list(counts),expected_worklist_launches=len(grouped),
            original_row_laws=sorted(set(c for k,c in zip(kinds,counts)if k=='original')),
            q1_rows=sorted(set(c for k,c in zip(kinds,counts)if k=='q1')),
            middle_bytes=sum(counts)*2304*4,contribution_bytes=sum(counts)*5120*4,
            maximum_expert_bindings=max(map(len,grouped.values()))))
    resident=15*(18_800_640+6_635_520)
    bound=resident+256*1024**2
    if bound>=ARRAY_BUDGET:raise ValueError('Synthetic DOWN stage exceeds array budget')
    return dict(schema='mlx2.ds41-synthetic-down-stage-plan.v1',seed=SEED,repeats=REPEATS,
        batch_size=BATCH_SIZE,profiles=profiles,variants=list(VARIANTS),case_count=len(profiles)*(2*len(VARIANTS)+len(ASSEMBLY_VARIANTS))+len(ASSEMBLY_PROFILES)*len(ASSEMBLY_VARIANTS)+2*len(WIDTH_SWEEP)+2+4,
        original_width_sweep=dict(widths=[*WIDTH_SWEEP,15],local_rows=[2,6],total_expert_jobs=15,
            extra_case_count=2*len(WIDTH_SWEEP),scope="Borrowed GU owners; fresh CPU plan/validation/uploads and all DOWN outputs/assembly timed; totalmath fixed"),
        assembly_only_profiles=[dict(name=n,parts=len(c),token_rows=sum(c)//6,route_occurrences=sum(c),mapped_chunks=(len(c)+27)//28,contribution_bytes=sum(c)*5120*4)for n,c in ASSEMBLY_PROFILES],
        resident_expert_payload_bytes=resident,resident_original_experts=15,resident_q1_experts=15,
        conservative_operand_and_scratch_bound_bytes=bound,array_budget_bytes=ARRAY_BUDGET,
        hidden=5120,intermediate=2304,top_k=6,checkpoint=None,
        physical_layouts={'original':'canonical','q1':'native_q1_v1'},
        q1_payload_offsets={'down.qweight':0,'gate.qweight':1474560,'up.qweight':2949120,
            'down.biases':4423680,'down.scales':4792320,'gate.biases':5160960,
            'gate.scales':5529600,'up.biases':5898240,'up.scales':6266880},
        primary_scope='Fresh CPU plan, split/concatenate activation packing, DOWN validation/dispatch, BF16 ownership and ordered assembly; one final output fence',
        borrowed_gu_scope='Same fresh CPU plan/validation/metadata uploads and DOWN/assembly; GU packed producer owners built outside clocks; baseline expert views borrow same owners; no input repack',
        scope_case_counts={'fresh_plan_repacked':12,'borrowed_gu_producer':12,'assembly_only':12,'borrowed_gu_width_sweep':8,'borrowed_gu_grid_stock':2,'borrowed_gu_q1_hybrid':4})


def _live(callback):
    if callback is not None and callback()is False:raise RuntimeError('Synthetic DOWN live guard refused')


def _measure(mx,fn):
    a=time.perf_counter_ns();value=fn();b=time.perf_counter_ns();mx.eval(value);c=time.perf_counter_ns()
    return value,dict(host_graph_submit_seconds=(b-a)/1e9,completion_wait_seconds=(c-b)/1e9,total_seconds=(c-a)/1e9)


def _finite(mx,value):return bool(mx.all(mx.isfinite(value)).item())


def _expert(mx,np,rng,kind):
    from mlx2.adapters.deepseek_v41_source_buffers import HostExpert,canonical_layout,materialize
    if kind=='q1':
        from mlx2.runtime.gguf.source_down_worklist import native_q1_layout
        layout=native_q1_layout()
    else:layout=canonical_layout(kind)
    slab=np.empty(layout.nbytes,np.uint8);host=HostExpert(slab,layout)
    for t in layout.tensors:
        view=slab[t.offset:t.offset+t.nbytes].view(t.dtype).reshape(t.shape)
        if t.name.endswith('qweight'):view[:]=rng.integers(0,2**32,size=t.shape,dtype=np.uint32)
        elif kind=='original':view[:]=119
        elif t.name.endswith('scales'):view[:]=np.float16(1/64)
        else:view[:]=np.float16(-1/128)
    expert=materialize(host);mx.eval(expert.backing,*expert.values());return expert


def _prepare(np,values,kinds,middles,bank,producer=None,batch_size=BATCH_SIZE):
    from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs
    p=plan_jobs(values,kinds,batch_size=batch_size)
    # The parent owns the input middle arena. Each frame rebuilds the ordinary
    # per-expert segment views; selected worklists pack their nonadjacent jobs.
    if producer is not None:
        segments,_=_borrow_producer(p,producer)
    else:
        cursor=0;segments=[]
        for job in p.jobs:
            segments.append(middles[cursor:cursor+job.count]);cursor+=job.count
    order=np.argsort(values.reshape(-1),kind='stable')
    return p,tuple(segments),order


def _borrow_producer(p,producer):
    """Validate this fresh plan against borrowed actual-shaped GU owners."""
    if len(producer)!=len(p.batches):raise ValueError('Synthetic producer batch count mismatch')
    segments=[None]*len(p.jobs);owners=[]
    for batch,(descriptor,owner)in zip(p.batches,producer):
        expected=tuple((p.jobs[i].expert_id,p.jobs[i].count)for i in batch.job_indices)
        if tuple(descriptor)!=expected or owner.shape!=(sum(batch.row_counts),2304):
            raise ValueError('Synthetic producer owner does not match fresh route plan')
        owners.append(owner)
        for i,lo,hi in zip(batch.job_indices,batch.offsets,batch.offsets[1:]):segments[i]=owner[lo:hi]
    if any(x is None for x in segments):raise ValueError('Synthetic producer did not cover all expert jobs')
    return tuple(segments),tuple(owners)


def _per_expert_parts(mx,p,segments,bank):
    from mlx2.runtime.gguf.source_q1 import matmul
    parts=[]
    for job,x in zip(p.jobs,segments):
        expert=bank[job.expert_id]
        if job.kind=='original':
            raw=mx.quantized_matmul(x,expert['down.qweight'],scales=expert['down.scales'],mode='mxfp4')
        else:raw=matmul(x,expert['down.qweight'],expert['down.scales'],expert['down.biases'])
        parts.append(raw.astype(mx.bfloat16).astype(mx.float32))
    return tuple(parts)


def _stage(mx,np,values,kinds,middles,bank,variant,producer=None,batch_size=BATCH_SIZE,traversal='flat'):
    from mlx2.adapters.deepseek_v41_source_math import ordered_combine
    from mlx2.runtime.gguf.source_prefill_combine import combine_mapped
    p,segments,order=_prepare(np,values,kinds,middles,bank,producer,batch_size)
    if p.fallback_job_indices:raise ValueError('Synthetic stage unexpectedly requires fallback')
    if variant=='per_expert_stock_concat':
        return ordered_combine(_per_expert_parts(mx,p,segments,bank),values,order)
    if variant in HYBRID_VARIANTS:
        if producer is None or traversal!='flat':raise ValueError('Synthetic Q1 hybrid requires borrowed GU/default grid')
        if variant=='hybrid_q1_old_concat':
            from mlx2.runtime.gguf.source_down_prefill import pp_down
        else:
            from mlx2.runtime.gguf.source_down_worklist import pp_down
        parts=[None]*len(p.jobs)
        for i,(job,x)in enumerate(zip(p.jobs,segments)):
            if job.kind=='original':
                e=bank[job.expert_id]
                raw=mx.quantized_matmul(x,e['down.qweight'],scales=e['down.scales'],mode='mxfp4')
                parts[i]=raw.astype(mx.bfloat16).astype(mx.float32)
        for batch_index,batch in enumerate(p.batches):
            if batch.kind!='q1':continue
            selected=tuple(bank[p.jobs[i].expert_id]for i in batch.job_indices)
            owned=pp_down(producer[batch_index][1],selected,batch.row_counts)
            for i,lo,hi in zip(batch.job_indices,batch.offsets,batch.offsets[1:]):parts[i]=owned[lo:hi]
        if any(x is None for x in parts):raise ValueError('Synthetic Q1 hybrid left an assembly part missing')
        return ordered_combine(parts,values,order)
    if variant=='old_worklist_mapped':
        from mlx2.runtime.gguf.source_down_prefill import pp_down
    elif variant=='direct_worklist_mapped':
        from mlx2.runtime.gguf.source_down_worklist import pp_down
    else:raise ValueError('Unknown synthetic DOWN variant')
    parts=[];descriptors=[]
    for batch_index,batch in enumerate(p.batches):
        selected=tuple(bank[p.jobs[i].expert_id]for i in batch.job_indices)
        inputs=tuple(segments[i]for i in batch.job_indices)
        packed=producer[batch_index][1]if producer is not None else inputs[0]if len(inputs)==1 else mx.concatenate(inputs,axis=0)
        parts.append(pp_down(packed,selected,batch.row_counts,traversal=traversal)if traversal!='flat'else pp_down(packed,selected,batch.row_counts))
        descriptors.append(tuple((p.jobs[i].expert_id,p.jobs[i].count)for i in batch.job_indices))
    if variant=='direct_worklist_mapped':
        from mlx2.runtime.gguf.source_prefill_combine import prepare_mapped,combine_mapped_prepared
        packet=prepare_mapped(parts,values,order,descriptors)
        return combine_mapped_prepared(parts,packet)
    return combine_mapped(parts,values,order,descriptors)


def _assembly(mx,parts,values,order,descriptors,variant):
    if variant=='assembly_only_concat':
        from mlx2.adapters.deepseek_v41_source_math import ordered_combine
        return ordered_combine(parts,values,order)
    if variant=='assembly_only_direct_mapped':
        from mlx2.runtime.gguf.source_prefill_combine import combine_mapped
        return combine_mapped(parts,values,order,descriptors)
    raise ValueError('Unknown synthetic assembly variant')


def _identify(np,reference,value):
    a=np.asarray(reference);b=np.asarray(value)
    if a.shape!=b.shape or a.dtype!=np.float32 or b.dtype!=np.float32:raise ValueError('Synthetic stage output representation mismatch')
    x=a.view(np.uint32);y=b.view(np.uint32);bad=np.flatnonzero(x.reshape(-1)!=y.reshape(-1))
    return dict(shape=list(b.shape),dtype=str(b.dtype),sha256=hashlib.sha256(b.tobytes()).hexdigest(),
        reference_sha256=hashlib.sha256(a.tobytes()).hexdigest(),raw_bits_exact=not bool(bad.size),
        different_words=int(bad.size),first_mismatch=None if not bad.size else dict(index=list(np.unravel_index(int(bad[0]),a.shape)),reference=float(a.reshape(-1)[bad[0]]),value=float(b.reshape(-1)[bad[0]])))


def run(mx,check_live=None):
    if check_live is not None and not callable(check_live):raise ValueError('check_live must be callable or None')
    p=plan();_live(check_live)
    if mx.default_device()!=mx.gpu:raise ValueError('Synthetic DOWN stage requires caller-owned GPU device')
    import numpy as np
    rng=np.random.default_rng(SEED);banks={}
    for kind in ('original','q1'):
        banks[kind]=[]
        for _ in range(15):_live(check_live);banks[kind].append(_expert(mx,np,rng,kind))
    cases=[];controls=[]
    for name,kinds,counts in PROFILES:
        _live(check_live);values,ids=_routes(np,rng,counts)
        next(v for v in p['profiles']if v['name']==name)
        mapping=dict(zip(ids,kinds));bank={e:banks[k][i]for i,(e,k)in enumerate(zip(ids,kinds))}
        # Dyadic finite inputs stand in for already-produced Source middle.
        # GU/QAT and coefficient construction are intentionally outside stage.
        host=rng.integers(-32,33,size=(sum(counts),2304),dtype=np.int32).astype(np.float32)/np.float32(256)
        middle=mx.array(host);mx.eval(middle)
        reference=None
        for variant in VARIANTS:
            fn=lambda v=values,k=mapping,x=middle,b=bank,w=variant:_stage(mx,np,v,k,x,b,w)
            value,cold=_measure(mx,fn)
            if not _finite(mx,value):raise FloatingPointError('Nonfinite synthetic DOWN warmup:'+name+'/'+variant)
            observed=np.asarray(value)
            if reference is None:reference=observed.copy()
            identity=_identify(np,reference,observed);controls.append(dict(profile=name,variant=variant,**identity))
            cases.append(dict(name=name+'/'+variant,profile=name,variant=variant,fn=fn,warm_compile=cold,identity=identity,samples=[]))
        # Model-real GU producer representation: immutable packed owners are
        # prepared outside clocks; fresh plan/validation and slices stay inside.
        pp,segments,_=_prepare(np,values,mapping,middle,bank);producer=[]
        for batch in pp.batches:
            inputs=tuple(segments[i]for i in batch.job_indices)
            owner=inputs[0]if len(inputs)==1 else mx.concatenate(inputs,axis=0)
            descriptor=tuple((pp.jobs[i].expert_id,pp.jobs[i].count)for i in batch.job_indices)
            producer.append((descriptor,owner))
        producer=tuple(producer);mx.eval(*(owner for _,owner in producer))
        for variant in VARIANTS:
            fn=lambda v=values,k=mapping,x=middle,b=bank,w=variant,g=producer:_stage(mx,np,v,k,x,b,w,g)
            value,cold=_measure(mx,fn)
            if not _finite(mx,value):raise FloatingPointError('Nonfinite borrowed producer warmup')
            identity=_identify(np,reference,np.asarray(value));controls.append(dict(profile=name,variant=variant,scope='borrowed_gu_producer',**identity))
            cases.append(dict(name=name+'/borrowed_gu/'+variant,profile=name,variant=variant,scope='borrowed_gu_producer',fn=fn,warm_compile=cold,identity=identity,samples=[]))
        if name in ('original15_R2','original15_R6'):
            for width in WIDTH_SWEEP:
                pp,segments,_=_prepare(np,values,mapping,middle,bank,batch_size=width);owners=[]
                for batch in pp.batches:
                    inputs=tuple(segments[i]for i in batch.job_indices)
                    owner=inputs[0]if len(inputs)==1 else mx.concatenate(inputs,axis=0)
                    descriptor=tuple((pp.jobs[i].expert_id,pp.jobs[i].count)for i in batch.job_indices)
                    owners.append((descriptor,owner))
                owners=tuple(owners);mx.eval(*(owner for _,owner in owners))
                variant='direct_worklist_mapped'
                fn=lambda v=values,k=mapping,x=middle,b=bank,w=variant,g=owners,e=width:_stage(mx,np,v,k,x,b,w,g,e)
                value,cold=_measure(mx,fn)
                if not _finite(mx,value):raise FloatingPointError('Nonfinite width sweep warmup')
                identity=_identify(np,reference,np.asarray(value));controls.append(dict(profile=name,variant=variant,batch_width=width,scope='borrowed_gu_width_sweep',**identity))
                cases.append(dict(name=name+'/borrowed_gu_E'+str(width)+'/'+variant,profile=name,variant=variant,
                    scope='borrowed_gu_width_sweep',batch_width=width,worklist_launches=len(owners),assembly_parts=len(owners),
                    fn=fn,warm_compile=cold,identity=identity,samples=[]))
        if name in ('original15_R2','original15_R6'):
            variant='direct_worklist_mapped'
            fn=lambda v=values,k=mapping,x=middle,b=bank,w=variant,g=producer:_stage(mx,np,v,k,x,b,w,g,traversal='stock')
            value,cold=_measure(mx,fn)
            if not _finite(mx,value):raise FloatingPointError('Nonfinite stock-grid warmup')
            identity=_identify(np,reference,np.asarray(value));controls.append(dict(profile=name,variant=variant,traversal='stock',scope='borrowed_gu_grid_stock',**identity))
            cases.append(dict(name=name+'/borrowed_gu_stock_E15/'+variant,profile=name,variant=variant,
                scope='borrowed_gu_grid_stock',batch_width=15,traversal='stock',worklist_launches=len(producer),assembly_parts=len(producer),
                fn=fn,warm_compile=cold,identity=identity,samples=[]))
        if name in ('q1_15_ragged','mixed15_ragged'):
            for variant in HYBRID_VARIANTS:
                fn=lambda v=values,k=mapping,x=middle,b=bank,w=variant,g=producer:_stage(mx,np,v,k,x,b,w,g)
                value,cold=_measure(mx,fn)
                if not _finite(mx,value):raise FloatingPointError('Nonfinite Q1 hybrid warmup')
                identity=_identify(np,reference,np.asarray(value));controls.append(dict(profile=name,variant=variant,scope='borrowed_gu_q1_hybrid',**identity))
                cases.append(dict(name=name+'/borrowed_gu_q1_hybrid/'+variant,profile=name,variant=variant,
                    scope='borrowed_gu_q1_hybrid',batch_width=15,assembly_parts=15,
                    fn=fn,warm_compile=cold,identity=identity,samples=[]))
        # Same resident, already-owned expert contributions for both assembly
        # controls. This separate scope isolates concat versus direct mapping.
        pp,segments,order=_prepare(np,values,mapping,middle,bank)
        owned=_per_expert_parts(mx,pp,segments,bank);mx.eval(*owned)
        descriptors=tuple(((job.expert_id,job.count),)for job in pp.jobs)
        for variant in ASSEMBLY_VARIANTS:
            fn=lambda a=owned,v=values,o=order,d=descriptors,w=variant:_assembly(mx,a,v,o,d,w)
            value,cold=_measure(mx,fn)
            if not _finite(mx,value):raise FloatingPointError('Nonfinite assembly warmup')
            identity=_identify(np,reference,np.asarray(value));controls.append(dict(profile=name,variant=variant,**identity))
            cases.append(dict(name=name+'/'+variant,profile=name,variant=variant,fn=fn,warm_compile=cold,identity=identity,samples=[]))
    # Larger pure assembly fixtures isolate the accumulator carries which do
    # not occur with the15-expert whole-DOWN cases. No extra expert payloads.
    for name,counts in ASSEMBLY_PROFILES:
        _live(check_live);values,ids=_routes(np,rng,counts)
        order=np.argsort(values.reshape(-1),kind='stable')
        host=rng.integers(-32,33,size=(sum(counts),5120),dtype=np.int32).astype(np.float32)/np.float32(64)
        arena=mx.array(host);mx.eval(arena);owned=[];cursor=0
        for count in counts:owned.append(arena[cursor:cursor+count]);cursor+=count
        owned=tuple(owned);descriptors=tuple(((e,c),)for e,c in zip(ids,counts));reference=None
        for variant in ASSEMBLY_VARIANTS:
            fn=lambda a=owned,v=values,o=order,d=descriptors,w=variant:_assembly(mx,a,v,o,d,w)
            value,cold=_measure(mx,fn)
            if not _finite(mx,value):raise FloatingPointError('Nonfinite large assembly warmup')
            observed=np.asarray(value)
            if reference is None:reference=observed.copy()
            identity=_identify(np,reference,observed);controls.append(dict(profile=name,variant=variant,**identity))
            cases.append(dict(name=name+'/'+variant,profile=name,variant=variant,fn=fn,warm_compile=cold,identity=identity,samples=[]))
    rotation=[]
    for rep in range(REPEATS):
        indices=list(range(rep%len(cases),len(cases)))+list(range(rep%len(cases)))
        rotation.append([cases[i]['name']for i in indices])
        for i in indices:
            _live(check_live);case=cases[i];value,sample=_measure(mx,case['fn'])
            if not _finite(mx,value):raise FloatingPointError('Nonfinite synthetic DOWN:'+case['name'])
            case['samples'].append(sample)
    rows=[dict(name=c['name'],profile=c['profile'],variant=c['variant'],warm_compile=c['warm_compile'],
        batch_width=c.get('batch_width',BATCH_SIZE),traversal=c.get('traversal','flat'),worklist_launches=c.get('worklist_launches'),assembly_parts=c.get('assembly_parts'),
        output_identity=c['identity'],scope=c.get('scope','assembly_only_preowned_parts' if c['variant']in ASSEMBLY_VARIANTS else 'fresh_plan_repacked_middle_down_assembly'),samples=c['samples'],medians={k:statistics.median(s[k]for s in c['samples'])for k in c['samples'][0]},finite=True)for c in cases]
    for row in rows:
        profile=next((v for v in p['profiles']if v['name']==row['profile']),None)
        if profile is not None:
            row['token_rows']=profile['token_rows'];row['expert_jobs']=profile['experts'];row['route_occurrences']=profile['route_occurrences']
            if row['variant']in VARIANTS:
                worklists=row['worklist_launches']or profile['expected_worklist_launches']
                row['down_operator_calls']=profile['experts']if row['variant']=='per_expert_stock_concat'else worklists
                row['assembly_parts']=profile['experts']if row['variant']=='per_expert_stock_concat'else worklists
        if row['variant']in HYBRID_VARIANTS:
            row['down_operator_calls']=sum(k=='original'for k in profile['kinds'])+1
            row['q1_worklist_calls']=1;row['assembly_parts']=15
            row['combine_packet_preparation']='none; ordered_combine with Q1 owner slices and stock original parts'
        else:
            row['combine_packet_preparation']='inside primary timed closure'if row['variant']=='direct_worklist_mapped'else 'ordinary wrapper inside timed closure'
    comparisons=[]
    for row in rows:
        if row['scope']not in ('borrowed_gu_producer','borrowed_gu_width_sweep','borrowed_gu_grid_stock','borrowed_gu_q1_hybrid'):continue
        for control in ('per_expert_stock_concat','old_worklist_mapped'):
            reference=next(r for r in rows if r['name']==row['profile']+'/borrowed_gu/'+control)
            comparisons.append(dict(case=row['name'],reference=reference['name'],
                ratios_by_repetition=[dict(repetition=i,**{k:a[k]/b[k]for k in a})for i,(a,b)in enumerate(zip(reference['samples'],row['samples']))],
                ratio_definition='same repetition reference_seconds/candidate_seconds; not simultaneous or randomized ABBA'))
    root=Path(__file__).resolve().parents[2]
    paths=['scripts/research/ds41_synthetic_down_stage_components.py','src/mlx2/runtime/gguf/source_down_prefill.py','src/mlx2/runtime/gguf/source_down_worklist.py','src/mlx2/runtime/gguf/source_prefill_combine.py','src/mlx2/runtime/gguf/source_q1.py','src/mlx2/adapters/deepseek_v41_source_math.py','src/mlx2/adapters/deepseek_v41_source_prefill.py','src/mlx2/adapters/deepseek_v41_source_buffers.py']
    return dict(schema='mlx2.ds41-synthetic-down-stage-components.v1',plan=p,complete=True,case_count=len(rows),cases=rows,
        finite_execution=True,paired_comparisons=comparisons,output_controls=controls,all_control_output_bits_exact=all(c['raw_bits_exact']for c in controls),rotated_order=rotation,
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest()for f in paths},
        timing_scope=p['primary_scope']+'; finite/identity readbacks and check_live outside timing. Warm/JIT first call separate. No intermediate eval fence.',
        excluded='GU, shared layer QAT, routing score production, cache admission, real model/weights/startup; no prepared metadata timing shortcut',
        weight_scope='Seeded independent synthetic original canonical and Q1 native_q1_v1 physical-layout experts only; original/Q1/mixed controls share identical borrowed payloads',model_gain_claim=False,performance_qualified=False,quality_claim=False)
