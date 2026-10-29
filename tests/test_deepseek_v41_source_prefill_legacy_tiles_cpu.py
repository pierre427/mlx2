"""Legacy F32 split-K source, partition and ownership checks; never GPU."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.gguf import source_prefill_legacy_tiles as tile
from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*args,**kwargs):raise AssertionError('CPU test attempted GPU compilation')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


def packed(kind='original'):
    layout=canonical_layout(kind);types={'<u4':mx.uint32,'|u1':mx.uint8,'<f2':mx.float16}
    fields={t.name:SimpleNamespace(shape=t.shape,dtype=types[t.dtype])for t in layout.tensors}
    return PackedExpert(SimpleNamespace(ndim=1,shape=(layout.nbytes,),dtype=mx.uint8),layout,fields)


@pytest.mark.parametrize('m',[13,16,31,32,33,64,95,96])
def test_actual_split_dispatch_and_complete_partition_domain(m):
    p=tile.plan(m);partitions=5 if m<=32 else 2
    assert p['partitions']==partitions and p['k_partition']*partitions==5120
    assert p['tile']==(32,32,32) and p['warp_shape']==(2,2) and p['k_step']==8
    assert p['threadgroup']==(32,2,2)
    assert tuple(a//b for a,b in zip(p['grid'],p['threadgroup']))==p['threadgroups']
    assert p['threadgroups']==(72,(m+31)//32,partitions)
    assert p['threadgroup_bytes']==9216
    assert p['scratch_bytes']<=3_000_000
    assert p['partial_shape']==(partitions,m,2304)
    assert p['global_weight_repack_bytes']==0


@pytest.mark.parametrize('m',[0,1,12,97,1024,True,13.0,'13'])
def test_invalid_dispatch_domain_refuses(m):
    with pytest.raises(ValueError):tile.plan(m)


@pytest.mark.parametrize('projection',['gate','up'])
def test_one_immutable_backing_projection_offsets(projection):
    e=packed();x=SimpleNamespace(ndim=2,shape=(33,5120),dtype=mx.float32)
    p,offsets=tile.prepare_inputs(x,e,projection)
    assert p['rows']==33
    assert offsets==tuple(e.layout.tensor(projection+'.'+f).offset for f in ('qweight','scales'))


@pytest.mark.parametrize('mutation',['kind','plain','projection','ndim','width','dtype','rows','backing','typed'])
def test_invalid_component_geometry_refuses_before_device(mutation):
    e=packed();x=SimpleNamespace(ndim=2,shape=(33,5120),dtype=mx.float32);projection='gate'
    if mutation=='kind':e=packed('q1')
    elif mutation=='plain':e={}
    elif mutation=='projection':projection='down'
    elif mutation=='ndim':x.ndim=3
    elif mutation=='width':x.shape=(33,5121)
    elif mutation=='dtype':x.dtype=mx.bfloat16
    elif mutation=='rows':x.shape=(97,5120)
    elif mutation=='backing':e.backing.shape=(1,)
    else:e['gate.scales'].dtype=mx.float16
    with pytest.raises(ValueError):tile.prepare_inputs(x,e,projection)


def test_projection_loader_and_f32_k8_helper_are_verbatim_installed():
    _,headers=tile._installed();original=(headers/'fp_quantized.h').read_text()
    start=original.index('template <\n    typename T,\n    const int group_size,\n    const int bits,\n    const bool aligned_N,\n    typename U = T,')
    stop=original.index('\ntemplate <\n    typename T,\n    int group_size,\n    int bits,\n    bool has_global_scale = false,\n    int BM = 32,',start)
    generated=tile._header()
    assert original[start:stop] in generated
    loader_start=original.index('template <\n    typename T,\n    short BROWS,')
    loader_end=original.index('template <typename T, int group_size, int bits, int D>')
    assert original[loader_start:loader_end] in generated
    assert 'BlockMMA<T, U, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>' in generated
    assert (headers/'steel/gemm/mma.h').read_text().splitlines()[0] in generated
    assert 'for (short kk = 0; kk < BK; kk += kFragSize)' in generated


def test_wrapper_only_changes_ownership_offsets_not_operand_dtype():
    s=tile.kernel_spec(33)
    assert 'threadgroup float Xs[32*36]' in s['source']
    assert 'threadgroup float Ws[32*36]' in s['source']
    assert 'const device float* selected_x=x+k_start' in s['source']
    assert 'layout_offsets[0]+k_start/2' in s['source']
    assert 'layout_offsets[1]+k_start/32' in s['source']
    assert 'fp_qmm_t_impl<float,32,4,true,float,false,32,32,32>' in s['source']
    assert 'partials+tid.z*33*2304' in s['source']
    assert 'bfloat' not in s['source'] and 'half' not in s['source']
    r=tile.reduction_spec()
    assert 'float out=0;' in r['source'] and 'out+=C_split[offset];' in r['source']
    assert 'offset+=stride;' in r['source'] and r['compile_options']=={'math_mode':'safe'}


@pytest.mark.parametrize('m',[13,33])
def test_no_partial_rounding_or_regrouped_reduction_and_backing_identity(m,monkeypatch):
    p=tile.plan(m);e=packed();x=mx.full((m,5120),np.float32(1+2**-20))
    a=np.zeros(p['partial_shape'],np.float32)
    a[0,:,0]=np.float32(1+2**-8);a[1,:,0]=np.float32(-1)
    if p['partitions']==5:
        a[0,:,1]=2**25;a[1,:,1]=1;a[2,:,1]=-2**25
    a[:,:,2]=-0.
    partials=mx.array(a);seen={}
    def projection(**kw):seen['projection']=kw;return [partials]
    def reduction(**kw):
        seen['reduction']=kw
        assert kw['inputs'][0] is partials
        assert np.array_equal(np.asarray(kw['inputs'][1]),[p['partitions'],m*2304,2304])
        out=np.zeros((m,2304),np.float32)
        for part in a:out=np.add(out,part,dtype=np.float32)
        return [mx.array(out)]
    monkeypatch.setattr(tile,'_kernel',lambda rows:projection)
    monkeypatch.setattr(tile,'_reduction_kernel',lambda:reduction)
    from mlx2.runtime.gguf import source_dense_legacy
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    rounded,raw,actual=tile.project(x,e,return_partials=True)
    assert actual is partials
    assert seen['projection']['inputs'][0] is x and seen['projection']['inputs'][1] is e.backing
    assert seen['projection']['threadgroup']==(32,2,2)
    host=np.asarray(raw)
    assert np.all(host[:,0]==np.float32(2**-8))
    assert np.all(host[:,1]==0) and np.all(host[:,2].view(np.uint32)==0)
    assert np.array_equal(np.asarray(rounded),host)


def test_cpu_device_refused_before_any_kernel():
    with pytest.raises(ValueError,match='caller-owned'):
        tile.project(mx.zeros((13,5120)),packed())


@pytest.mark.parametrize('rows',[13,16,32,33,64,96])
def test_offline_legacy_compilation_without_gpu(rows,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    source=tmp_path/f'm{rows}.metal';source.write_text(tile.compilation_source(rows))
    include=Path(mx.__file__).parent/'include'
    command=['xcrun','-sdk','macosx','metal','-std=metal3.2','-I',str(include),'-c',str(source),'-o',str(tmp_path/'probe.air')]
    result=subprocess.run(command,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr


def test_offline_ordered_reduction_without_gpu(tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    s=tile.reduction_spec()
    source=tmp_path/'reduce.metal';source.write_text(s['header']+'''
kernel void reduce_probe(const device float* partials [[buffer(0)]],
const device int* metadata [[buffer(1)]],device float* raw [[buffer(2)]],
uint3 thread_position_in_grid [[thread_position_in_grid]]) {
'''+s['source']+'\n}\n')
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-fno-fast-math','-c',str(source),'-o',str(tmp_path/'reduce.air')],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('counts',[(13,31,32),(33,48,64),(65,83,96),(96,)*15])
def test_batch_geometry_matches_adapter_wide_planner(counts):
    from mlx2.adapters.deepseek_v41_source_prefill import Job
    from mlx2.adapters.source_prefill_wide_plan import plan_wide_jobs,partial_offset
    jobs=[];cursor=0
    for expert,count in enumerate(counts):
        locations=np.arange(cursor,cursor+count,dtype=np.uint32)
        jobs.append(Job(expert,locations,locations//6,count,'original'));cursor+=count
    b=plan_wide_jobs(jobs).batches[0];p=tile.batch_plan(counts)
    assert p['row_offsets']==b.row_offsets and p['partial_shape']==b.partial_shape
    assert p['partition_stride']==b.partition_stride and p['grid']==b.stock_threadgrid
    assert p['threadgroup']==b.stock_threadgroup and p['threadgroups']==b.threadgroups
    assert p['partial_bytes']==b.partial_bytes<=26_542_080
    assert p['gathered_input_bytes']==b.gathered_input_bytes<=29_491_200
    seen=set()
    for global_z in range(len(counts)*p['partitions']):
        slot,part=divmod(global_z,p['partitions'])
        for localrow in range(counts[slot]):
            for col in (0,1,2302,2303):
                index=part*p['partition_stride']+(p['row_offsets'][slot]+localrow)*2304+col
                assert index==partial_offset(b,slot,part,localrow,col)
                assert index not in seen;seen.add(index)
                xoffset=p['row_offsets'][slot]*5120+part*p['k_partition']+localrow*5120
                assert xoffset==(p['row_offsets'][slot]+localrow)*5120+part*p['k_partition']
    assert max(seen)<p['partial_bytes']//4


@pytest.mark.parametrize('counts',[(),(13,)*16,(12,),(97,),(True,),(13.,),(13,33),(33,65),(64,96)])
def test_batch_invalid_count_or_mixed_bucket_refuses(counts):
    with pytest.raises(ValueError):tile.batch_plan(counts)


def test_batch_signature_forwarding_only_and_explicit_stock_local_tile():
    assert tile._batch_header().replace('const int M,','const constant int& M,')==tile._header()
    s=tile.batch_kernel_spec(2,3,15)
    assert s['total_buffer_bindings']==18
    assert len(s['input_names'])==17
    assert 'slot=physical.z/2,part=physical.z%2' in s['source']
    assert 'original_m=descriptor[2+slot*4]' in s['source']
    assert 'gathered_x+row_offset*5120+k_start' in s['source']
    assert 'partials+part*total_rows*2304+row_offset*2304' in s['source']
    assert 'uint3(physical.xy,part)' in s['source']
    assert 'bfloat' not in s['source'] and 'fp_qmm_t_splitk(' not in s['source']


@pytest.mark.parametrize('bucket',[(5,1,0),(5,1,16),(5,2,3),(2,1,3),(1,3,3),(2,4,3),(True,1,3)])
def test_batch_unknown_compiled_bucket_refuses(bucket):
    with pytest.raises(ValueError):tile.batch_kernel_spec(*bucket)


def test_batch_descriptor_does_not_slice_or_copy_selected_activation():
    counts=(13,31,32);e=[packed()for _ in counts]
    x=SimpleNamespace(ndim=2,shape=(sum(counts),5120),dtype=mx.float32)
    p,d,retained=tile.prepare_batch_inputs(x,e,counts,'up')
    assert d.dtype==np.uint32 and not d.flags.writeable and d.shape==(1+4*len(counts),)
    assert d[0]==sum(counts)
    for i,(expert,count)in enumerate(zip(e,counts)):
        assert retained[i]is expert
        assert list(d[1+4*i:5+4*i])==[p['row_offsets'][i],count,
            expert.layout.tensor('up.qweight').offset,expert.layout.tensor('up.scales').offset]


def test_batch_identity_ordered_partial_reduction_and_failure_ownership(monkeypatch):
    counts=(13,32);e=[packed(),packed()];p=tile.batch_plan(counts)
    x=mx.full((sum(counts),5120),np.float32(1+2**-20));a=np.zeros(p['partial_shape'],np.float32)
    a[0,:,0]=2**25;a[1,:,0]=1;a[2,:,0]=-2**25
    a[0,:,1]=np.float32(1+2**-8);a[1,:,1]=-1
    a[:,:,2]=-0.
    partials=mx.array(a);seen={}
    def project(**kw):seen['projection']=kw;return [partials]
    def reduce(**kw):
        seen['reduction']=kw
        assert kw['inputs'][0]is partials
        assert list(np.asarray(kw['inputs'][1]))==[5,sum(counts)*2304,2304]
        out=np.zeros((sum(counts),2304),np.float32)
        for part in a:out=np.add(out,part,dtype=np.float32)
        return [mx.array(out)]
    monkeypatch.setattr(tile,'_batch_kernel',lambda *args:project)
    monkeypatch.setattr(tile,'_reduction_kernel',lambda:reduce)
    from mlx2.runtime.gguf import source_dense_legacy
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    rounded,raw,actual=tile.batch_project(x,e,counts,return_partials=True)
    assert actual is partials and seen['projection']['inputs'][0]is x
    assert all(backing is expert.backing for backing,expert in zip(seen['projection']['inputs'][2:],e))
    assert np.all(np.asarray(raw)[:,0]==0)
    assert np.all(np.asarray(raw)[:,1]==np.float32(2**-8))
    assert np.all(np.asarray(raw)[:,2].view(np.uint32)==0)
    assert np.array_equal(np.asarray(rounded),np.asarray(raw))
    def fail(**kw):raise RuntimeError('injected component failure')
    monkeypatch.setattr(tile,'_batch_kernel',lambda *args:fail)
    with pytest.raises(RuntimeError,match='injected'):
        tile.batch_project(x,e,counts)
    assert all(expert.backing is seen['projection']['inputs'][2+i]for i,expert in enumerate(e))


@pytest.mark.parametrize('p,t,e',[(5,1,1),(5,1,3),(5,1,15),(2,2,3),(2,2,15),(2,3,3),(2,3,15)])
def test_offline_batched_legacy_compilation_without_gpu(p,t,e,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    source=tmp_path/'batch.metal';source.write_text(tile.batch_compilation_source(p,t,e))
    include=Path(mx.__file__).parent/'include'
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-I',str(include),'-c',str(source),'-o',str(tmp_path/'batch.air')],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('m',[13,16,32,34,65,96])
def test_optin16x64_keeps_original_partition_law(m):
    baseline=tile.plan(m);candidate=tile.plan(m,tile=(16,64))
    assert candidate['partitions']==baseline['partitions']
    assert candidate['k_partition']==baseline['k_partition']
    assert candidate['tile']==(16,32,64) and candidate['warp_shape']==(2,2)
    assert candidate['k_step']==8 and candidate['threadgroup']==(32,2,2)
    assert candidate['threadgroups']==(36,(m+15)//16,candidate['partitions'])
    assert candidate['grid']==(1152,(m+15)//16*2,candidate['partitions']*2)
    assert candidate['threadgroup_bytes']==11520
    assert candidate['partial_shape']==baseline['partial_shape']
    s=tile.kernel_spec(m,tile=(16,64))
    assert 'threadgroup float Xs[16*36]' in s['source']
    assert 'threadgroup float Ws[64*36]' in s['source']
    assert 'fp_qmm_t_impl<float,32,4,true,float,false,16,32,64>' in s['source']
    assert 'bfloat'not in s['source'] and s['header'].startswith(tile._header())


@pytest.mark.parametrize('bad',[(16,32),(32,64),(8,64),(16,128),[16,64],(True,64),(16.,64),(16,),None])
def test_only_two_explicit_tile_geometries_admitted(bad):
    with pytest.raises(ValueError):tile.plan(13,tile=bad)
    with pytest.raises(ValueError):tile.batch_plan((13,16),tile=bad)
    with pytest.raises(ValueError):tile.batch_kernel_spec(5,1,1,tile=bad)


@pytest.mark.parametrize('counts',[(13,16),(13,32),(33,34,48),(33,64),(65,80),(65,96)])
def test_alt_batch_max_tile_extent_skips_only_whole_inactive_groups(counts):
    p=tile.batch_plan(counts,tile=(16,64));old=tile.batch_plan(counts)
    assert p['m_tiles']==max((m+15)//16 for m in counts)
    assert p['partitions']==old['partitions'] and p['partial_shape']==old['partial_shape']
    assert p['row_offsets']==old['row_offsets'] and p['threadgroups'][0]==36
    covered=[];skipped=[]
    for slot,count in enumerate(counts):
        for ytile in range(p['m_tiles']):
            if ytile*16>=count:
                skipped.append((slot,ytile));continue
            covered.extend((slot,row)for row in range(ytile*16,min((ytile+1)*16,count)))
    assert covered==[(slot,row)for slot,count in enumerate(counts)for row in range(count)]
    assert len(set(covered))==sum(counts)
    s=tile.batch_kernel_spec(p['partitions'],p['m_tiles'],len(counts),tile=(16,64))
    assert 'if(physical.y*16>=original_m)return;'in s['source']
    assert s['source'].index('return;')<s['source'].index('threadgroup float Xs')
    assert 'uint3(physical.xy,part)'in s['source']
    assert s['header'].startswith(tile._batch_header())


def test_optin_tile_forwarding_retains_selected_array_and_backing(monkeypatch):
    e=packed();x=mx.full((13,5120),np.float32(1+2**-20));seen={}
    partials=mx.zeros((5,13,2304));raw=mx.zeros((13,2304))
    def partial_fn(**kw):seen['partial']=kw;return[partials]
    def select(*args):seen['args']=args;return partial_fn
    monkeypatch.setattr(tile,'_kernel',select)
    monkeypatch.setattr(tile,'_reduction_kernel',lambda:lambda **kw:[raw])
    from mlx2.runtime.gguf import source_dense_legacy
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    tile.project(x,e,tile=(16,64))
    assert seen['args']==(13,(16,64))
    assert seen['partial']['inputs'][0]is x and seen['partial']['inputs'][1]is e.backing
    assert seen['partial']['grid']==(1152,2,10)
    counts=(13,32);xb=mx.zeros((45,5120));parts_b=mx.zeros((5,45,2304));raw_b=mx.zeros((45,2304))
    def batch_fn(**kw):seen['batch']=kw;return[parts_b]
    def batch_select(*args):seen['batchargs']=args;return batch_fn
    monkeypatch.setattr(tile,'_batch_kernel',batch_select)
    monkeypatch.setattr(tile,'_reduction_kernel',lambda:lambda **kw:[raw_b])
    tile.batch_project(xb,(e,e),counts,tile=(16,64))
    assert seen['batchargs']==(5,2,2,(16,64))
    assert seen['batch']['inputs'][0]is xb
    assert seen['batch']['grid']==(1152,4,20)


@pytest.mark.parametrize('m',[13,16,32,34,65,96])
def test_offline16x64_single_and15expert_compiles_without_gpu(m,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    p=tile.plan(m,tile=(16,64));b=tile.batch_plan((m,)*15,tile=(16,64))
    sources={'single':tile.compilation_source(m,tile=(16,64)),
        'batch15':tile.batch_compilation_source(b['partitions'],b['m_tiles'],15,tile=(16,64))}
    include=Path(mx.__file__).parent/'include'
    for name,text in sources.items():
        source=tmp_path/(name+'.metal');source.write_text(text)
        result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-I',str(include),'-c',str(source),'-o',str(tmp_path/(name+'.air'))],capture_output=True,text=True,timeout=45)
        assert result.returncode==0,result.stdout+result.stderr
