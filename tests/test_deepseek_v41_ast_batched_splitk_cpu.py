"""Pinned dispatch/body geometry only; no GPU compilation or numerical claim."""
from pathlib import Path
import numpy as np
import pytest
import mlx.core as mx
from mlx2.runtime.gguf import source_ast_batched_splitk as candidate


def test_qk_11_keeps_ordinary16_partitions_and_geometry():
    p=candidate.plan(3,64,11,512,transpose_b=True)
    assert (p['BM'],p['BN'],p['BK'],p['WM'],p['WN'])==(32,16,16,2,2)
    assert (p['partitions'],p['partition_size'],p['k_iterations'])==(16,32,2)
    assert p['grid']==(32,4,96)and p['threadgroup']==(32,2,2)
    assert p['split_shape']==(3,16,64,11)and p['output_shape']==(3,64,11)


@pytest.mark.parametrize('n',[2,3,11,16,17,32,33,127,128,255,256,511,512])
def test_batch_partition_and_output_ownership_has_no_overlap(n):
    p=candidate.plan(16,64,n,512,transpose_b=True)
    seen=set()
    for z in range(16*p['partitions']):
        b,part=divmod(z,p['partitions'])
        assert (b,part)not in seen;seen.add((b,part))
    assert len(seen)==16*p['partitions']
    covered=p['partitions']*p['partition_size']
    assert 0<=512-covered<16*p['partitions']
    assert p['ldB']==512


@pytest.mark.parametrize('args',[(1,64,11,64),(3,32,11,512),(3,64,513,512),(17,64,11,512),(True,64,11,512)])
def test_refuse_outside_bounded_stock_splitk_intersection(args):
    with pytest.raises(ValueError):candidate.plan(*args,transpose_b=True)


def test_actual_installed_body_retained_except_partition_index():
    p=candidate.kernel_specs(3,64,11,512,transpose_b=True)
    base=candidate._installed();stock=(base/'steel/gemm/kernels/steel_gemm_splitk.h').read_text()
    start=stock.index('  using gemm_kernel = GEMMKernel<');stop=stock.index('\n}\n',start)
    body=stock[start:stop].replace('const int tid_z = tid.z;','const int tid_z = tid.z % 16;')
    assert p['split']['source'].endswith(body)
    begin=stock.index('  // Ajust D and C',stock.index('void gemm_splitk_accum('));end=stock.index('\n}\n',begin)
    assert p['accum']['source'].endswith(stock[begin:end])
    assert p['split']['header']==p['accum']['header']
    assert '#include "mlx/'not in p['split']['header']
    assert len(candidate.implementation_identity())==64


def test_scalar_partition_accumulation_order_is_not_reassociated():
    p=candidate.kernel_specs(3,64,11,512,transpose_b=True)
    assert 'for (int i = 0; i < k_partitions; i++)'in p['accum']['source']
    assert 'out += C_split[offset];'in p['accum']['source']
    partial=np.array([1e20,1,-1e20,1],dtype=np.float32)
    expected=np.float32(0)
    for item in partial:expected=np.float32(expected+item)
    assert expected==1 and np.float32((partial[0]+partial[2])+(partial[1]+partial[3]))==2


def test_component_refuses_cpu_before_any_shader_or_array_access(monkeypatch):
    previous=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda*a,**k:pytest.fail('GPU compile forbidden'))
    try:
        with pytest.raises(ValueError,match='parent-owned GPU'):
            candidate.private_matmul(None,None,transpose_b=True,check_live=lambda:True)
        with pytest.raises(ValueError,match='live ownership'):
            candidate.private_matmul(None,None,transpose_b=True,check_live=None)
    finally:mx.set_default_device(previous)


@pytest.mark.parametrize('k',[1,2,11,16,17,127,128,255,511])
def test_pv_regular_preserves_target_single_batch_config_and_tail_flags(k):
    p=candidate.plan(3,64,512,k,transpose_b=False)
    assert candidate.supports(64,512,k)and not candidate.supports_splitk(64,512,k)
    assert p['mode']=='regular'and p['architecture']=='applegpu_g17s'
    assert (p['BM'],p['BN'],p['BK'],p['WM'],p['WN'])==(64,64,16,2,2)
    assert p['has_batch']is False and p['use_out_source']is False
    assert p['align_M']and p['align_N']and p['align_K']==(k%16==0)
    assert p['grid']==(256,2,6)and p['threadgroup']==(32,2,2)
    assert p['ldA']==k and p['ldB']==512


def test_regular_body_is_pinned_unmodified_with_independent_owner_prefix():
    p=candidate.kernel_specs(3,64,512,11,transpose_b=False)
    assert set(p)=={'descriptor','regular'}
    stock=(candidate._installed()/'steel/gemm/kernels/steel_gemm_fused.h').read_text()
    begin=stock.index('  using gemm_kernel = GEMMKernel<');end=stock.index('\n}\n',begin)
    assert p['regular']['source'].endswith(stock[begin:end])
    assert 'const uint3 tid=uint3(physical_tid.x,physical_tid.y,0u);'in p['regular']['source']
    assert 'has_batch=false,use_out_source=false,do_axpby=false;'in p['regular']['source']
    assert 'A=a+size_t(physical_tid.z)*704;'in p['regular']['source']
    assert 'B=b+size_t(physical_tid.z)*5632;'in p['regular']['source']
    assert 'D=result_buffer+size_t(physical_tid.z)*32768;'in p['regular']['source']
    assert p['regular']['source'].index('if (!align_K)')<p['regular']['source'].index('for (int k = 0; k < gemm_k_iterations; k++)')


def test_selected_pv_512_preserves_splitk_dispatch_instead_of_regular():
    p=candidate.plan(3,64,512,512,transpose_b=False)
    assert p['mode']=='splitk'and p['partitions']==2 and p['partition_size']==256


def test_expanded_header_terminates_namespace_comment_before_generated_entry():
    from mlx2.runtime.gguf.source_ast_batched_splitk import kernel_specs
    spec = kernel_specs(3,64,11,512,transpose_b=True)['split']
    combined = spec['header'] + 'kernel void generated_entry() {}\n'
    assert combined.splitlines()[-1] == 'kernel void generated_entry() {}'


def test_generated_output_bindings_do_not_shadow_stock_accumulator_locals():
    import re
    from mlx2.runtime.gguf.source_ast_batched_splitk import kernel_specs
    for n,k,t in ((11,512,True),(512,11,False)):
        specs=kernel_specs(3,64,n,k,transpose_b=t)
        for stage in ('split','accum','regular'):
            if stage not in specs:continue
            spec=specs[stage]
            scalar_locals=set(re.findall(r'\b(?:AccT|OutT|float|int|uint)\s+([a-zA-Z_]\w*)\s*=',spec['source']))
            assert not scalar_locals.intersection(spec['input_names']+spec['output_names'])
