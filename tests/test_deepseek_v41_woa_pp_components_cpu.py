"""WoA component options/oracle/call contracts only; no model or GPU work."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/research'))
spec=importlib.util.spec_from_file_location('woa_pp_cpu',ROOT/'scripts/research/ds41_woa_pp_components.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def test_defaults_preserve_exact_only_and_explicit_profile_replicates7():
    assert w.component_options({})==('reference_exact_only',3)
    assert w.component_options({'woa_options':{'numerical_profile':w.NATIVE_PROFILE}})==(w.NATIVE_PROFILE,7)
    assert w.component_options({'woa_options':{'numerical_profile':w.NATIVE_PROFILE,'repeats':5}})==(w.NATIVE_PROFILE,5)


@pytest.mark.parametrize('value',[None,[],{'extra':1},{'repeats':True},{'repeats':2},{'repeats':8},{'repeats':3.0},
    {'numerical_profile':None},{'numerical_profile':True},{'numerical_profile':'bad'},{'numerical_profile':[]}])
def test_malformed_options_refuse(value):
    with pytest.raises(ValueError):w.component_options({'woa_options':value})


def test_bf16_rne_ties_sign_and_exact_grid():
    x=np.array([1.,1.00390625,1.01171875,-1.00390625,0.,-0.],np.float32)
    y=w.round_bf16_f32(np,x)
    np.testing.assert_array_equal(y.view(np.uint32),np.array([1.,1.,1.015625,-1.,0.,-0.],np.float32).view(np.uint32))
    np.testing.assert_array_equal(w.round_bf16_f32(np,y).view(np.uint32),y.view(np.uint32))


@pytest.mark.parametrize('x',[np.array([1],np.float64),np.array([np.inf],np.float32),np.array([np.nan],np.float32)])
def test_bf16_audit_refuses_wrong_dtype_nonfinite(x):
    with pytest.raises(ValueError):w.round_bf16_f32(np,x)


def fixture():
    codes=np.full((10,32),0x38,np.uint8);codes[1::2]=0xb8
    q=codes.view(np.uint32);s=np.full((10,1),127,np.uint8)
    x=np.arange(3*2*32,dtype=np.float32).reshape(1,3,2,32)*.25
    return x,q,s


def test_blocked_fp64_group_addressing_original_coefficients_and_thread_restore():
    from ds41_attention_projection_components import decode_effective_f32
    x,q,s=fixture();coeff=decode_effective_f32(np,q.view(np.uint8),s);calls=[]
    def actual(lo,hi):calls.append((lo,hi));return coeff[lo:hi]
    truth,audit=w.fp64_reference(np,x,q,s,groups=2,block_rows=2,resident_rows=actual)
    expected=np.stack([x[0,:,g].astype(np.float64)@coeff[g*5:(g+1)*5].astype(np.float64).T for g in range(2)],axis=1)[None]
    np.testing.assert_array_equal(truth,expected)
    assert calls==[(0,2),(2,4),(4,5),(5,7),(7,9),(9,10)]
    assert audit['actual_resident_coefficient_blocks_verified']==6
    assert audit['effective_FP8_F32_BF16_roundtrip_exact']and audit['BLAS_threads']==1
    assert len(audit['truth_sha256'])==64


def test_fp64_truth_uses_resident_bf16_conversion_not_unrounded_fp8():
    x,q,s=fixture();q=np.full_like(q,0x01010101);s[:]=0
    truth,audit=w.fp64_reference(np,x,q,s,groups=2,block_rows=3)
    assert np.count_nonzero(truth)==0
    assert audit['effective_FP8_F32_to_resident_BF16_changed_words']==320
    assert not audit['effective_FP8_F32_BF16_roundtrip_exact']


def test_resident_coefficient_mismatch_refuses_mathematical_interpretation():
    x,q,s=fixture()
    with pytest.raises(ValueError,match='resident BF16 coefficient replay'):
        w.fp64_reference(np,x,q,s,groups=2,resident_rows=lambda lo,hi:np.zeros((hi-lo,32),np.float32))


@pytest.mark.parametrize('kind',['groups','rows0','block0','block129','dtype','shape','scale'])
def test_fp64_oracle_geometry_refusals(kind):
    x,q,s=fixture();kw={'groups':2}
    if kind=='groups':kw['groups']=True
    if kind=='rows0':x=x[:,:0]
    if kind=='block0':kw['block_rows']=0
    if kind=='block129':kw['block_rows']=129
    if kind=='dtype':x=x.astype(np.float64)
    if kind=='shape':x=x[:,:,:,:31]
    if kind=='scale':s=s[:,:0]
    with pytest.raises(ValueError):w.fp64_reference(np,x,q,s,**kw)


class Array:
    def __init__(self,name,log):self.name=name;self.log=log
    def astype(self,dtype):self.log.append((self.name,'cast',dtype));return Array(self.name+'->'+dtype,self.log)
    def reshape(self,*shape):self.log.append((self.name,'reshape',shape));return self


def fake_mx(log):
    def einsum(law,x,weight):log.append(('einsum',law,x.name,weight.name));return Array('out',log)
    return SimpleNamespace(float32='f32',bfloat16='bf16',einsum=einsum)


def test_production_reference_repeats_coefficient_cast_inside_every_call():
    log=[];mx=fake_mx(log);x=Array('x',log);weight=Array('resident_bf16',log)
    w.ordinary_grouped(mx,x,weight);w.ordinary_grouped(mx,x,weight)
    assert log.count(('resident_bf16','cast','f32'))==2
    assert log.count(('x','cast','f32'))==2
    assert [v for v in log if v[0]=='einsum']==[('einsum','btgi,goi->btgo','x->f32','resident_bf16->f32')]*2


def test_native_preserves_resident_weight_and_includes_input_output_casts():
    log=[];mx=fake_mx(log);w.native_grouped(mx,Array('x',log),Array('resident_bf16',log))
    assert ('x','cast','bf16')in log and('out','cast','f32')in log
    assert not any(x[0]=='resident_bf16'and x[1]=='cast'for x in log)
    assert ('einsum','btgi,goi->btgo','x->bf16','resident_bf16')in log


@pytest.mark.parametrize('explicit,exact,changed,finite,admitted',[
    (False,False,False,True,False),(True,False,False,True,True),(False,True,False,True,True),
    (True,False,True,True,False),(True,True,False,False,False)])
def test_nonexact_timing_requires_explicit_profile_finite_and_self_stable(explicit,exact,changed,finite,admitted):
    x=np.array([1.],np.float32);y=x.copy()
    if changed:y[0]=2
    if not finite:x[0]=y[0]=np.nan
    r=w.native_timing_gate(np,x,y,profile=w.NATIVE_PROFILE if explicit else'reference_exact_only',owner_exact=exact)
    assert r['timing_admitted']is admitted and r['changed_numerical_profile']is(not exact)


def test_allocator_probe_counts_allocator_not_dram_and_releases_result():
    log=[];active=iter([100,140,100])
    mx=SimpleNamespace(synchronize=lambda:log.append('sync'),clear_cache=lambda:log.append('clear'),
        get_active_memory=lambda:next(active),reset_peak_memory=lambda:log.append('reset'),get_peak_memory=lambda:150,
        eval=lambda x:log.append('eval'))
    r=w.allocator_probe(mx,lambda:object())
    assert r['peak_increment_bytes']==50 and r['active_after_output_release_bytes']==100
    assert log==['sync','clear','reset','eval','sync','clear']
    assert 'not bus/DRAM'in r['scope']


def test_guard_errors_before_capture_or_artifact_work(tmp_path):
    cpu=SimpleNamespace(default_device=lambda:'cpu',gpu='gpu')
    with pytest.raises(RuntimeError,match='caller-owned'):w.run_components(cpu,np,tmp_path,tmp_path,{},None,None)
    gpu=SimpleNamespace(default_device=lambda:'gpu',gpu='gpu')
    with pytest.raises(ValueError,match='prepared-WoB'):w.run_components(gpu,np,tmp_path,tmp_path,{},None,None)
    with pytest.raises(ValueError,match='overwrite'):w.run_components(gpu,np,tmp_path,tmp_path,{'woa_pp_components':{}},None,None)


def test_grouped_query_metadata_axis1_preserves_all_bits_hashes_coordinates():
    from ds41_attention_components import raw_details
    a=np.zeros((1,99,8,4),np.float32);b=a.copy();b[0,11,7,3]=1;b[0,98,0,1]=-0.
    ordinary=raw_details(np,a,b);grouped=w.sequence_details(np,a,b)
    assert ordinary['query_row_axis']==2 and len(ordinary['query_row_changed_counts'])==8
    assert grouped['query_row_axis']==1 and len(grouped['query_row_changed_counts'])==99
    assert grouped['query_row_changed_counts'][11]==grouped['query_row_changed_counts'][98]==1
    for key in('exact','changed_words','reference_sha256','candidate_sha256','first_mismatch','max_abs_finite_pairs','shape'):
        assert grouped[key]==ordinary[key]
    assert sum(grouped['query_row_changed_counts'])==grouped['changed_words']==2


def test_dense_consumer_query_metadata_axis1_and_truncation():
    a=np.zeros((1,130,4),np.float32);b=a.copy();b[0,2,1]=1;b[0,129,0]=2
    result=w.sequence_details(np,a,b)
    assert result['query_row_axis']==1 and result['query_rows_truncated']
    assert len(result['query_row_changed_counts'])==128 and result['query_row_changed_counts'][2]==1
    assert result['changed_words']==2
    with pytest.raises(ValueError,match='sequence diagnostics'):w.sequence_details(np,a[0],b[0])
