"""Unwired attention dataflow equations and replay controls; never GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.models import deepseek_v41_attention_probe as probe
from mlx2.runtime.models.deepseek_v41 import selected_attention


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*a,**kw):raise AssertionError('CPU attention test attempted Metal')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


def fixture(length=7,count=3):
    rng=np.random.default_rng(41+length+count);h,d=2,8
    array=lambda shape:mx.array(rng.integers(-2,3,shape).astype(np.float32)*np.float32(.125))
    q=array((1,h,length,d));local=array((1,1,length,d));pool=array((1,count,d));sink=array((h,))
    local_mask=mx.array(np.tri(length,dtype=bool)[None,None])
    indices=mx.broadcast_to(mx.arange(count,dtype=mx.uint32)[None,None],(1,length,count))
    valid=mx.array(np.arange(count)[None,None]<np.maximum(1,np.arange(length))[None,:,None])
    return(q,local,local_mask,pool,indices,valid,sink)


@pytest.mark.parametrize('length,count',[(1,1),(1,3),(7,3),(7,8)])
def test_reference_diagnostic_tree_matches_owning_cpu_bits(length,count):
    args=fixture(length,count);expected=selected_attention(*args)
    normal=probe.attention(*args,variant='reference')
    diagnostic,stages=probe.attention(*args,variant='reference',return_stages=True)
    assert np.asarray(normal).tobytes()==np.asarray(expected).tobytes()
    assert np.asarray(diagnostic).tobytes()==np.asarray(expected).tobytes()
    assert set(stages)=={'scaled_query','local_scores_raw','pool_scores_raw','local_scores','pool_scores','normalizer','local_pv','pool_pv'}
    assert np.asarray(stages['scaled_query']).tobytes()==np.asarray(args[0]*8**-.5).tobytes()
    assert stages['normalizer'].shape==(1,2,length,1)


@pytest.mark.parametrize('length,count,workspace',[(1,3,16<<20),(7,3,16<<20),(7,8,256)])
def test_broadcast_keeps_cpu_equations_and_removes_python_gather(length,count,workspace,monkeypatch):
    args=fixture(length,count);expected=probe.attention(*args,workspace_bytes=workspace)
    from mlx2.runtime.models import deepseek_v41
    def forbidden(*a,**kw):raise AssertionError('Broadcast path gathered selected rows')
    monkeypatch.setattr(deepseek_v41,'_selected_rows',forbidden)
    actual=probe.attention(*args,variant='broadcast',all_retained=True,workspace_bytes=workspace)
    assert np.asarray(actual).tobytes()==np.asarray(expected).tobytes()


def test_decode_broadcast_keeps_existing_gathered_input(monkeypatch):
    args=fixture(1,3);gathered=args[3][:,None];expected=selected_attention(*args,gathered=gathered)
    def forbidden(*a,**kw):raise AssertionError('Width1 reused gather was replaced')
    monkeypatch.setattr(mx,'broadcast_to',forbidden)
    actual=probe.attention(*args,variant='broadcast',all_retained=True,gathered=gathered)
    assert np.asarray(actual).tobytes()==np.asarray(expected).tobytes()


@pytest.mark.parametrize('length,count',[(1,1),(1,3),(7,3),(7,8)])
@pytest.mark.parametrize('variant',['matmul','flat_pool'])
def test_matmul_cpu_equation_sanity_not_gpu_numerical_qualification(length,count,variant,monkeypatch):
    args=fixture(length,count);expected=selected_attention(*args)
    def forbidden(*a,**kw):raise AssertionError('Contiguous pool path used selected einsum')
    monkeypatch.setattr(mx,'einsum',forbidden)
    actual=probe.attention(*args,variant=variant,all_retained=True)
    np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=2e-6,atol=2e-7)


@pytest.mark.parametrize('count,tile',[(49,41),(99,32)])
def test_actual_PP99_tile_formula_retained(count,tile):
    p=probe.tile_plan(64,99,512,99,count)
    assert p['tile']==tile
    assert sum(b-a for a,b in p['tiles'])==99
    assert p['hypothetical_full_gather_bytes']==99*count*512*4


@pytest.mark.parametrize('values',[(0,1,8,1,1),(65,1,8,1,1),(2,0,8,1,1),(2,129,8,1,1),(2,1,513,1,1),(2,1,8,0,1),(2,1,8,256,1),(2,1,8,1,0),(2,1,8,1,513),(True,1,8,1,1)])
def test_unbounded_geometry_refused(values):
    with pytest.raises(ValueError):probe.tile_plan(*values)


@pytest.mark.parametrize('variant',list(probe.VARIANTS[1:]))
def test_all_retained_contract_is_explicit_and_no_host_readback(variant,monkeypatch):
    args=fixture()
    with pytest.raises(ValueError,match='trusted all-retained'):probe.attention(*args,variant=variant)
    def forbidden(*a,**kw):raise AssertionError('Candidate read MLX input back to host')
    monkeypatch.setattr(np,'asarray',forbidden)
    value=probe.attention(*args,variant=variant,all_retained=True)
    assert value.shape==(1,2,7,8)


@pytest.mark.parametrize('bad',['qtype','qbatch','mask','valid','sinks','indices','gathered','poolsubset','unknown'])
def test_bad_geometry_refused_before_math(bad):
    args=list(fixture());variant='broadcast';extra={}
    if bad=='qtype':args[0]=args[0].astype(mx.float16)
    elif bad=='qbatch':args[0]=mx.zeros((2,2,7,8))
    elif bad=='mask':args[2]=mx.ones((1,7,7),mx.bool_)
    elif bad=='valid':args[5]=args[5].astype(mx.int32)
    elif bad=='sinks':args[6]=mx.zeros((3,))
    elif bad=='indices':args[4]=args[4].astype(mx.float32)
    elif bad=='gathered':extra['gathered']=mx.zeros((1,7,3,8))
    elif bad=='poolsubset':args[3]=mx.zeros((1,4,8))
    else:variant='online'
    with pytest.raises(ValueError):probe.attention(*args,variant=variant,all_retained=True,**extra)


@pytest.fixture
def helper():
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_attention_components.py'
    spec=importlib.util.spec_from_file_location('ds41_attention_components_cpu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_raw_oracle_signed_zero_nonfinite_and_first_difference(helper):
    a=np.array([0.,-0.,np.inf,-np.inf,np.nan],np.float32);b=a.copy();b[0]=-0.
    result=helper.raw_details(np,a,b)
    assert not result['exact']and result['changed_words']==1
    assert result['first_mismatch']['flat_index']==0
    assert result['max_abs_finite_pairs']==0 and result['changed_nonfinite_words']==0
    assert helper.raw_details(np,a,a)['exact']


def sealed_parser(tmp_path,code):
    h=tmp_path/'harness';h.mkdir();parser=h/'attention_probe.py';parser.write_text(code)
    sha=hashlib.sha256(parser.read_bytes()).hexdigest();identity=dict(checkpoint='synthetic@cpu',source_revision='cpu',source_snapshot_sha256='capture-cpu')
    (tmp_path/'spec.json').write_text(json.dumps(dict(identity,harness_sha256={'attention_probe.py':sha})))
    manifest=tmp_path/'attention-capture.json';manifest.write_text(json.dumps(dict(identity,helper_sha256=sha)))
    return manifest,parser


def test_capture_parser_is_sealed_before_execution(helper,tmp_path):
    marker=tmp_path/'executed'
    manifest,parser=sealed_parser(tmp_path,f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    parser.write_text(parser.read_text()+'# mutation\n')
    with pytest.raises(ValueError,match='seal mismatch'):helper.capture_module(manifest)
    assert not marker.exists()


def test_capture_parser_directory_and_symlink_rules(helper,tmp_path):
    manifest,parser=sealed_parser(tmp_path,'flag=41\n')
    module,path=helper.capture_module(tmp_path)
    assert module.flag==41 and path==manifest
    link=tmp_path/'alias.json';link.symlink_to(manifest)
    with pytest.raises(ValueError,match='symlink'):helper.capture_module(link)


def test_split_clocks_are_interleaved_and_synchronize_outside_timer(helper):
    class Clock:
        value=0
        def __call__(self):self.value+=1;return float(self.value)
    events=[];clock=Clock()
    fake=SimpleNamespace(eval=lambda value:events.append(('eval',value)),synchronize=lambda:events.append(('sync',clock.value)))
    functions={k:(lambda k=k:events.append(('fn',k))or k)for k in ('reference','broadcast','matmul')}
    result=helper.timed_interleaved(fake,np,functions,repeats=3,clock=clock)
    timed=[v[1]for v in events if v[0]=='fn'][3:]
    assert timed==['reference','broadcast','matmul','broadcast','matmul','reference','matmul','reference','broadcast']
    for value in result.values():
        assert len(value['samples'])==3
        assert value['median_graph_build_and_submission_seconds']==1
        assert value['median_completion_wait_seconds']==1
        assert value['median_total_seconds']==2


def test_controller_refuses_cpu_before_loading_capture(helper,tmp_path):
    with pytest.raises(RuntimeError,match='caller-owned'):
        helper.run_components(mx,np,tmp_path,tmp_path,{},None,lambda *a:None)


@pytest.mark.parametrize('bad',['raw','stage','reference'])
def test_replay_required_controls_and_optional_timing_admission(helper,tmp_path,monkeypatch,bad):
    # CPU tags exercise controller gates only; no actual capture or GPU claim.
    cases=[]
    for phase,length,frontier in [('pp99',99,0),('decode1',1,100)]:
        for layer in (2,20,21):
            case=dict(layer=layer,phase=phase,length=length,frontier=frontier)
            arrays={'reference_raw':np.full((1,1,1,1),.125,np.float32),
                    'reference_bf16':np.full((1,1,1,1),.125,np.float32)}
            cases.append((case,arrays))
    manifest=dict(checkpoint='synthetic@cpu',source_revision='cpu',source_snapshot_sha256='capture-cpu')
    path=tmp_path/'capture.json';path.write_text('{}')
    parser=SimpleNamespace(load_capture=lambda p:(manifest,cases))
    monkeypatch.setattr(helper,'capture_module',lambda p:(parser,path))
    tag=mx.ones((1,1,1,1));monkeypatch.setattr(helper,'restored_inputs',lambda *a:([tag]*7,None,{'CPU_tag':True}))
    def attention(*a,variant='reference',return_stages=False,**kw):
        value=.25 if (bad=='reference'and variant=='reference')or(bad=='raw'and variant=='matmul')else .125
        output=mx.full((1,1,1,1),value)
        stage=mx.full((1,1,1,1),.5 if bad=='stage'and variant=='matmul'else .125)
        return(output,{'scaled_query':stage})if return_stages else output
    monkeypatch.setattr(probe,'attention',attention)
    monkeypatch.setattr(probe,'geometry',lambda *a,**kw:{'CPU_tag':True,'tile':1,'tiles':[(0,1)]})
    admitted=[]
    def time_functions(_mx,_np,functions,**kw):
        admitted.append(tuple(functions));return{k:{'CPU_tag':True}for k in functions}
    monkeypatch.setattr(helper,'timed_interleaved',time_functions)
    class CPUProxy:
        def __getattr__(self,name):return getattr(mx,name)
        def default_device(self):return mx.gpu  # Only getter overridden; work stays CPU.
        def get_peak_memory(self):return 0
        def synchronize(self):pass
        def clear_cache(self):pass
    report=dict(manifest,source_snapshot_sha256='component-cpu');writes=[]
    atomic=lambda path,value:writes.append(json.loads(json.dumps(value)))
    if bad=='reference':
        with pytest.raises(ValueError,match='owning stock replay'):
            helper.run_components(CPUProxy(),np,path,tmp_path,report,None,atomic)
        assert not admitted and not report['attention_components']['harness_completed']
    else:
        section=helper.run_components(CPUProxy(),np,path,tmp_path,report,None,atomic)
        assert section['complete']and section['harness_completed']and section['required_reference_controls_exact']
        assert len(section['cases'])==6 and len(admitted)==6
        assert all('matmul'not in names for names in admitted)
        assert all(set(probe.VARIANTS)-{'matmul'}==set(names)for names in admitted)
        assert not section['verdicts']['matmul_all_six_exact']

@pytest.mark.parametrize('variant',['matmul_tail_safe','flat_pool_tail_safe'])
def test_partial_tail_alone_keeps_original_gather_einsum(variant,monkeypatch):
    args=fixture(7,3);per_row=probe.tile_plan(2,7,8,7,3)['per_row_workspace_formula_bytes'];workspace=per_row*3
    reference,rs=probe.attention(*args,workspace_bytes=workspace,return_stages=True)
    from mlx2.runtime.models import deepseek_v41
    gather=deepseek_v41._selected_rows;einsum=mx.einsum;gathers=[];equations=[]
    def selected(pool,ids):
        gathers.append(ids.shape[1]);assert ids.shape[1]==1
        return gather(pool,ids)
    def tracked(eq,*values):
        equations.append((eq,values[0].shape[2]));return einsum(eq,*values)
    monkeypatch.setattr(deepseek_v41,'_selected_rows',selected);monkeypatch.setattr(mx,'einsum',tracked)
    value,stages=probe.attention(*args,variant=variant,all_retained=True,workspace_bytes=workspace,return_stages=True)
    assert gathers==[1]
    assert equations==[('bhld,blkd->bhlk',1),('bhlk,blkd->bhld',1)]
    for key in stages:assert np.asarray(stages[key][:,:,6:]).tobytes()==np.asarray(rs[key][:,:,6:]).tobytes()
    assert np.asarray(value[:,:,6:]).tobytes()==np.asarray(reference[:,:,6:]).tobytes()
    np.testing.assert_allclose(np.asarray(value),np.asarray(reference),rtol=2e-6,atol=2e-7)

@pytest.mark.parametrize('variant',['matmul_tail_safe','flat_pool_tail_safe'])
def test_tail_safe_exact_multiple_has_no_reference_gather_or_einsum(variant,monkeypatch):
    args=fixture(6,3);per_row=probe.tile_plan(2,6,8,6,3)['per_row_workspace_formula_bytes']
    from mlx2.runtime.models import deepseek_v41
    def forbidden(*a,**k):raise AssertionError('Exact multiple took tail fallback')
    monkeypatch.setattr(deepseek_v41,'_selected_rows',forbidden);monkeypatch.setattr(mx,'einsum',forbidden)
    value=probe.attention(*args,variant=variant,all_retained=True,workspace_bytes=per_row*3)
    assert value.shape==(1,2,6,8)

@pytest.mark.parametrize('variant',['matmul_tail_safe','flat_pool_tail_safe'])
def test_tail_safe_decode_retains_existing_gather(variant,monkeypatch):
    args=fixture(1,3);gathered=args[3][:,None];expected=selected_attention(*args,gathered=gathered)
    from mlx2.runtime.models import deepseek_v41
    monkeypatch.setattr(deepseek_v41,'_selected_rows',lambda *a:(_ for _ in ()).throw(AssertionError('Existing gather lost')))
    actual=probe.attention(*args,variant=variant,all_retained=True,gathered=gathered)
    assert np.asarray(actual).tobytes()==np.asarray(expected).tobytes()

@pytest.mark.parametrize('count,expected',[(49,['matmul','matmul','reference']),(99,['matmul','matmul','matmul','reference'])])
def test_actual_tail_phases_keep_original_tile_boundaries(count,expected):
    p=probe.tile_plan(64,99,512,99,count)
    assert [probe.tile_variant('matmul_tail_safe',b-a,p['tile'])for a,b in p['tiles']]==expected
    assert sum(b-a for a,b in p['tiles'])==99

def test_bounded_query_row_counts_locate_nonfirst_head_tail(helper):
    x=np.zeros((1,2,99,3),np.float32);y=x.copy();y[0,1,96,2]=1;y[0,0,98,0]=1
    r=helper.raw_details(np,x,y)
    assert r['query_row_axis']==2 and not r['query_rows_truncated']
    assert sum(r['query_row_changed_counts'][:96])==0
    assert r['query_row_changed_counts'][96:]==[1,0,1]
    big=np.zeros((1,1,130,1),np.float32);r=helper.raw_details(np,big,big)
    assert r['query_rows_truncated'] and len(r['query_row_changed_counts'])==128
