# SPDX-License-Identifier: MIT
"""Small synthetic actual-Source-provider Q1 tiny-DOWN integration proof.

No model/checkpoint/real-weight construction; caller owns GPU/admission.
Only GU producer is replaced temporarily by equivalent preowned output owners.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import statistics

SEED=41012631
REPEATS=7
POLICIES=('reference','q1_tiny_batch_v1')
PROFILES=(
 ('all_tiny_q1',('q1',)*15,(1,2,3)*5),
 ('partial_prefix_q1',('q1',)*12,(1,2,3,4,5,3)*2),
 ('no_tiny_q1',('q1',)*12,(4,5,6)*4),
 ('mixed_tiny',('original',)*9+('q1',)*6,(2,3,4,5,6,7,2,3,4,1,2,3,1,2,3)),
)


def _utilities():
    path=Path(__file__).with_name('ds41_synthetic_down_stage_components.py')
    spec=importlib.util.spec_from_file_location('ds41_synthetic_q1_provider_utilities',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def plan():
    resident=15*(18_800_640+6_635_520);bound=resident+128*1024**2
    return dict(schema='mlx2.ds41-synthetic-q1-provider-plan.v1',seed=SEED,repeats=REPEATS,
        case_count=8,checkpoint=None,resident_expert_payload_bytes=resident,
        conservative_operand_scratch_bound_bytes=bound,array_budget_bytes=1024**3,
        profiles=[dict(name=n,kinds=list(k),row_counts=list(c),expert_jobs=len(c),token_rows=sum(c)//6,
            expected_tiny_jobs=sum(t=='q1'and m<=3 for t,m in zip(k,c)),
            expected_tiny_rows=sum(m for t,m in zip(k,c)if t=='q1'and m<=3))for n,k,c in PROFILES],
        physical_layouts={'original':'canonical','q1':'native_q1_v1'},
        scope='Actual SourceV41Weights._prefill_batched_gate_up; only GU output producer and preowned-resident expert stream are synthetic')


class PreownedStream:
    """Explicit fake resident lease window; no filesystem/seal/cache claim."""
    packed_buffers=True
    def __init__(self,bank):
        self.bank=bank;self.stats={'lookups':0};self.active=False;self.acquired=[]
    def kind(self,key):return self.bank[key[1]].layout.kind
    def acquire(self,prefix,expert):
        if not self.active:raise RuntimeError('Synthetic acquire outside owned window')
        self.acquired.append((prefix,expert));return self.bank[expert]
    @contextmanager
    def window(self):
        if self.active:raise RuntimeError('Nested synthetic provider window')
        self.active=True
        try:yield self
        finally:self.active=False
    def reset(self):
        if self.active:raise RuntimeError('Cannot reset active synthetic stream')
        self.acquired=[];self.stats={'lookups':0}


def _provider(bank,policy):
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    w=SourceV41Weights.__new__(SourceV41Weights)
    w.expert_stream=PreownedStream(bank);w.scheduled_expert_calls={}
    w.source_prefill_down_policy=policy;w.source_prefill_batch_policy='gate_up'
    w.expert_math_policy='routed_fused_fp32';w.source_native_mx_policy='reference'
    w.source_prefill_wide_math='reference';w.source_prefill_wide_dtype='reference'
    w.source_frame_reuse_policy='reference'
    return w


@contextmanager
def _gu_outputs(owners):
    from mlx2.runtime.gguf import source_gate_up_prefill as gu
    previous=gu.pp_middle;calls=[]
    def borrowed(x,experts,rows,scores,counts,*,exp_spelling):
        key=(tuple(id(e)for e in experts),tuple(counts))
        if key not in owners:raise ValueError('Synthetic GU owner does not match actual provider batch')
        owner=owners[key]
        if owner.shape!=(sum(counts),2304)or rows.shape!=(sum(counts),)or scores.shape!=(sum(counts),):
            raise ValueError('Synthetic GU provider row/score geometry mismatch')
        calls.append(key);return owner
    gu.pp_middle=borrowed
    try:yield calls
    finally:gu.pp_middle=previous


def _expected(p,selected):
    jobs=[job for job in p.jobs if selected and job.kind=='q1'and job.count<=3]
    batches=sum(batch.kind=='q1'and any(c<=3 for c in batch.row_counts)for batch in p.batches)if selected else 0
    return dict(jobs=len(jobs),rows=sum(j.count for j in jobs),batches=batches,
        output_views=len(jobs),ordinary_jobs=len(p.jobs)-len(jobs),assembly_parts=len(p.jobs))


def _controls(w,p,calls,expected):
    prefix='official_source_prefill_q1_tiny_down_';c=w.scheduled_expert_calls
    if w.expert_stream.acquired!=[('blk.0',j.expert_id)for j in p.jobs]:raise AssertionError('Synthetic provider acquire census mismatch')
    if len(calls)!=len(p.batches)or w.expert_stream.stats['lookups']!=1:raise AssertionError('Synthetic provider GU/lookup census mismatch')
    for suffix in ('jobs','rows','batches','output_views'):
        if c.get(prefix+suffix,0)!=expected[suffix]:raise AssertionError('Synthetic tiny counter mismatch:'+suffix)
    if c.get('official_source_prefill_ordinary_down_jobs')!=expected['ordinary_jobs']:raise AssertionError('Synthetic ordinary DOWN census mismatch')
    if c.get('official_source_prefill_assembly_parts')!=expected['assembly_parts']:raise AssertionError('Synthetic assembly part census mismatch')
    return dict(exact_acquire_census=True,gu_batches=len(calls),expected=expected,counters=dict(c))


def _execute(mx,util,w,fixture):
    w.expert_stream.reset();w.scheduled_expert_calls={}
    with w.expert_stream.window(),_gu_outputs(fixture['owners'])as calls:
        value,timing=util._measure(mx,lambda:w._prefill_batched_gate_up(fixture['x'],fixture['values'],
            fixture['scores'],'blk.0',fixture['order']))
        controls=_controls(w,fixture['plan'],calls,_expected(fixture['plan'],w.source_prefill_down_policy!='reference'))
    return value,timing,controls


def run(mx,check_live=None):
    if check_live is not None and not callable(check_live):raise ValueError('check_live must be callable or None')
    util=_utilities();util._live(check_live);p=plan()
    if p['conservative_operand_scratch_bound_bytes']>=p['array_budget_bytes']:raise ValueError('Synthetic provider exceeds1GiB')
    if mx.default_device()!=mx.gpu:raise ValueError('Synthetic provider requires caller-owned GPU')
    import numpy as np
    from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs
    rng=np.random.default_rng(SEED);banks={}
    for kind in ('original','q1'):
        banks[kind]=[]
        for _ in range(15):util._live(check_live);banks[kind].append(util._expert(mx,np,rng,kind))
    cases=[]
    for name,kinds,counts in PROFILES:
        values,ids=util._routes(np,rng,counts);mapping=dict(zip(ids,kinds))
        bank={e:banks[k][i]for i,(e,k)in enumerate(zip(ids,kinds))};pp=plan_jobs(values,mapping)
        if pp.fallback_job_indices:raise ValueError('Synthetic provider unexpectedly needs wide GU fallback')
        host=rng.integers(-32,33,size=(sum(counts),2304),dtype=np.int32).astype(np.float32)/np.float32(256)
        middle=mx.array(host);mx.eval(middle);segments={};start=0
        for job in pp.jobs:segments[job.expert_id]=middle[start:start+job.count];start+=job.count
        owners={}
        for batch in pp.batches:
            experts=tuple(bank[pp.jobs[i].expert_id]for i in batch.job_indices)
            inputs=tuple(segments[pp.jobs[i].expert_id]for i in batch.job_indices)
            owner=inputs[0]if len(inputs)==1 else mx.concatenate(inputs,axis=0)
            owners[(tuple(id(e)for e in experts),batch.row_counts)]=owner
        x=mx.zeros((len(values),5120),mx.float32);scores=mx.full(values.shape,1/6,mx.float32)
        mx.eval(x,scores,*owners.values())
        fixture=dict(values=values,order=np.argsort(values.ravel(),kind='stable'),x=x,scores=scores,owners=owners,plan=pp)
        reference=None
        for policy in POLICIES:
            util._live(check_live);w=_provider(bank,policy);value,cold,control=_execute(mx,util,w,fixture)
            if not util._finite(mx,value):raise FloatingPointError('Nonfinite actual-provider synthetic output')
            observed=np.asarray(value)
            if reference is None:reference=observed.copy()
            identity=util._identify(np,reference,observed)
            if not identity['raw_bits_exact']:raise AssertionError('Synthetic actual-provider reference/candidate output differs')
            cases.append(dict(name=name+'/'+policy,profile=name,policy=policy,weights=w,fixture=fixture,
                cold=cold,identity=identity,warm_controls=control,samples=[],controls=[]))
    rotation=[]
    for rep in range(REPEATS):
        indices=list(range(rep%len(cases),len(cases)))+list(range(rep%len(cases)));rotation.append([cases[i]['name']for i in indices])
        for i in indices:
            util._live(check_live);case=cases[i];value,sample,control=_execute(mx,util,case['weights'],case['fixture'])
            if not util._finite(mx,value):raise FloatingPointError('Nonfinite actual-provider steady output')
            case['samples'].append(sample);case['controls'].append(control)
    rows=[dict(name=c['name'],profile=c['profile'],policy=c['policy'],warm_compile=c['cold'],output_identity=c['identity'],
        warm_controls=c['warm_controls'],controls=c['controls'],samples=c['samples'],
        medians={k:statistics.median(s[k]for s in c['samples'])for k in c['samples'][0]},finite=True)for c in cases]
    root=Path(__file__).resolve().parents[2]
    files=['scripts/research/ds41_synthetic_q1_provider.py','scripts/research/ds41_synthetic_down_stage_components.py',
        'src/mlx2/adapters/deepseek_v41_source_weights.py','src/mlx2/adapters/deepseek_v41_source_prefill_down.py',
        'src/mlx2/adapters/deepseek_v41_source_buffers.py','src/mlx2/runtime/gguf/source_down_prefill.py']
    return dict(schema='mlx2.ds41-synthetic-q1-provider.v1',plan=p,complete=True,case_count=len(rows),cases=rows,
        finite_execution=True,all_output_bits_exact=all(c['identity']['raw_bits_exact']for c in cases),all_counter_controls_pass=True,
        rotated_order=rotation,source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest()for f in files},
        timing_scope='Actual Source provider CPU planning/acquire wrappers, selection/prefix slicing, real DOWN math and ordered concat assembly; one final output fence. GU buffer construction/JITwarm/checks outside steady clocks.',
        synthetic_substitutions=['Only GU pp_middle temporarily returns equivalent preowned batch output','Preowned stream replaces real seals/disk/LFU admission; all buffers retained by synthetic fixture'],
        runtime_device_and_identity_guards_unmodified=True,constructor_configuration_validation_bypassed=True,model_gain_claim=False,qualification_claim=False,checkpoint=None)
