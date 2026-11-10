"""Synthetic original DOWN traversal/assembly factorial; caller owns GPU.

No backend import or device work on import. Ten borrowed-GU stage cases,
five policies for each E15/M2 and E15/M6 fixture. No actual weight file opens.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from pathlib import Path
import hashlib
import statistics
import time

SEED=41012627
REPEATS=7
LIMIT=1<<30
VARIANTS=('per_expert_stock_concat','flat_worklist_mapped','stock_worklist_mapped',
          'flat_worklist_concat','stock_worklist_concat')


def plan():
    payload=15*18_800_640
    bound=payload+(256<<20)
    if bound>=LIMIT:raise ValueError('Original DOWN component bound must be below1GiB')
    return dict(schema='mlx2.ds41-synthetic-original-down-concat.v1',seed=SEED,repeats=REPEATS,
        case_count=10,profiles=[dict(name=f'original15_R{m}',kind='original',experts=15,
            local_rows=m,token_rows=15*m//6,row_counts=[m]*15)for m in (2,6)],
        variants=list(VARIANTS),resident_expert_payload_bytes=payload,
        conservative_operand_and_scratch_bound_bytes=bound,array_budget_bytes=LIMIT,
        scope='Borrowed GU owners; fresh plan/validation/views/DOWN/owner/assembly inside fn; one final eval',
        checkpoint=None,actual_weights_loaded=False)


def _stage(mx,np,values,kinds,middle,bank,producer,variant):
    import ds41_synthetic_down_stage_components as base
    if variant=='per_expert_stock_concat':
        return base._stage(mx,np,values,kinds,middle,bank,'per_expert_stock_concat',producer)
    if variant in ('flat_worklist_mapped','stock_worklist_mapped'):
        return base._stage(mx,np,values,kinds,middle,bank,'direct_worklist_mapped',producer,
                           traversal='stock'if variant.startswith('stock')else'flat')
    if variant not in ('flat_worklist_concat','stock_worklist_concat'):
        raise ValueError('Unknown original DOWN factorial variant')
    from mlx2.runtime.gguf.source_down_worklist import pp_down
    from mlx2.adapters.deepseek_v41_source_math import ordered_combine
    p,segments,order=base._prepare(np,values,kinds,middle,bank,producer)
    if p.fallback_job_indices:raise ValueError('Original DOWN fixture unexpectedly has fallback')
    parts=[None]*len(p.jobs)
    for batch_index,batch in enumerate(p.batches):
        if batch.kind!='original':raise ValueError('Original DOWN factorial excludes Q1')
        experts=tuple(bank[p.jobs[i].expert_id]for i in batch.job_indices)
        owned=pp_down(producer[batch_index][1],experts,batch.row_counts,
                      traversal='stock'if variant.startswith('stock')else'flat')
        # Restore exact ascending job order using already-owned output subviews.
        # No mapped packet or extra activation/weight/owner copy is introduced.
        for job,lo,hi in zip(batch.job_indices,batch.offsets,batch.offsets[1:]):
            parts[job]=owned[lo:hi]
    if any(part is None for part in parts):raise ValueError('Original DOWN contribution missing')
    return ordered_combine(parts,values,order)


def run(mx,check_live=None):
    if check_live is not None and not callable(check_live):raise ValueError('check_live must be callable')
    import ds41_synthetic_down_stage_components as base
    import numpy as np
    if mx.default_device()!=mx.gpu:raise ValueError('Original DOWN stage requires caller-owned GPU')
    recipe=plan();rng=np.random.default_rng(SEED);cases=[];controls=[];setup=[]
    coefficient_start=time.perf_counter();weights=[]
    for _ in range(15):base._live(check_live);weights.append(base._expert(mx,np,rng,'original'))
    coefficient_seconds=time.perf_counter()-coefficient_start
    for profile in recipe['profiles']:
        base._live(check_live);counts=tuple(profile['row_counts']);values,ids=base._routes(np,rng,counts)
        kinds={expert:'original'for expert in ids};bank=dict(zip(ids,weights))
        host=rng.integers(-32,33,size=(sum(counts),2304),dtype=np.int32).astype(np.float32)/np.float32(256)
        middle=mx.array(host);mx.eval(middle)
        begin=time.perf_counter();p,segments,_=base._prepare(np,values,kinds,middle,bank);producer=[]
        for batch in p.batches:
            inputs=tuple(segments[i]for i in batch.job_indices)
            owner=inputs[0]if len(inputs)==1 else mx.concatenate(inputs,axis=0)
            descriptor=tuple((p.jobs[i].expert_id,p.jobs[i].count)for i in batch.job_indices)
            producer.append((descriptor,owner))
        producer=tuple(producer);mx.eval(*(owner for _,owner in producer))
        setup.append(dict(profile=profile['name'],producer_setup_seconds=time.perf_counter()-begin,
                          GU_owner_shapes=[list(owner.shape)for _,owner in producer]))
        reference=None
        for variant in VARIANTS:
            fn=lambda v=values,k=kinds,x=middle,b=bank,g=producer,w=variant:_stage(mx,np,v,k,x,b,g,w)
            value,cold=base._measure(mx,fn)
            if not base._finite(mx,value):raise FloatingPointError('Nonfinite original DOWN control')
            observed=np.asarray(value)
            if reference is None:reference=observed.copy()
            identity=base._identify(np,reference,observed)
            if not identity['raw_bits_exact']:raise ValueError('Original DOWN factorial control changed output bits')
            controls.append(dict(profile=profile['name'],variant=variant,**identity))
            cases.append(dict(name=profile['name']+'/'+variant,profile=profile['name'],variant=variant,
                              warm_compile=cold,output_identity=identity,fn=fn,samples=[]))
    rotation=[]
    for repetition in range(REPEATS):
        order=list(range(repetition%len(cases),len(cases)))+list(range(repetition%len(cases)))
        rotation.append([cases[i]['name']for i in order])
        for i in order:
            base._live(check_live);value,sample=base._measure(mx,cases[i]['fn'])
            if not base._finite(mx,value):raise FloatingPointError('Nonfinite original DOWN timed result')
            cases[i]['samples'].append(sample)
    peak=int(mx.get_peak_memory())
    if peak>=LIMIT:raise MemoryError('Original DOWN component peak reached1GiB')
    rows=[{**{k:v for k,v in case.items()if k!='fn'},'scope':'borrowed_gu_producer',
           'finite':True,'medians':{key:statistics.median(s[key]for s in case['samples'])for key in case['samples'][0]}}
          for case in cases]
    sources=['scripts/research/ds41_synthetic_original_down_concat.py','scripts/research/ds41_synthetic_down_stage_components.py',
             'src/mlx2/runtime/gguf/source_down_worklist.py','src/mlx2/runtime/gguf/source_prefill_combine.py',
             'src/mlx2/adapters/deepseek_v41_source_math.py','src/mlx2/adapters/deepseek_v41_source_prefill.py']
    root=Path(__file__).resolve().parents[2]
    return dict(schema=recipe['schema'],plan=recipe,complete=True,case_count=len(rows),cases=rows,
        output_controls=controls,all_control_output_bits_exact=True,finite_execution=True,rotated_order=rotation,
        peak_mlx_bytes=peak,coefficient_construction_seconds=coefficient_seconds,producer_setups=setup,
        timing_scope='Fresh CPU planning/validation/output views/DOWN/BF16owner/assembly inside fn; one final eval. Same GU owners borrowed across arms.',
        excluded='GU computation, coefficient construction, model/checkpoint, cache admission, finite/identity readbacks, first warm compilation',
        completion_clock='mx.eval includes deferred host graph evaluation/encoding plus GPU execution/wait; not GPU-only',
        weight_scope='15 independent seeded synthetic original canonical payloads, no weight files',
        source_sha256={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest()for rel in sources},
        model_gain_claim=False,quality_claim=False,performance_qualified=False)
