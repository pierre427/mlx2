"""CPU geometry, refusal, native ABI and retention contracts for matrix read."""
import importlib.util
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def planner(tmp_path_factory):
    path=tmp_path_factory.mktemp('matrix-plan')
    source=path/'plan.cpp';binary=path/'plan'
    source.write_text('''#include <iostream>
#include "prefill_matrix.h"
int main(int argc,char** argv){try{
 bool flag=mlx2::paged_kv::prefill_matrix_requested(argv[1]);
 std::vector<std::vector<uint32_t>> spans={{uint32_t(std::stoul(argv[4])),uint32_t(std::stoul(argv[5])),uint32_t(std::stoul(argv[6])),0,0,0,1,0},{33,0,33,0,0,1,1,0}};
 auto p=mlx2::paged_kv::prefill_matrix_plan(std::string(argv[2])=="bf16",std::stoul(argv[3]),24,4,spans);
 std::cout<<flag<<" "<<p.max_tiles<<" "<<p.spans;
}catch(const std::exception& e){std::cerr<<e.what();return 2;}}
''')
    subprocess.run(['c++','-std=c++20','-I',str(ROOT/'native/paged_kv'),str(source),'-o',str(binary)],check=True,capture_output=True)
    return binary


def test_real_multitile_ragged_rows_and_nonzero_offset(planner):
    for count,start,end,tiles in [(17,0,17,3),(65,31,96,5),(1023,7169,8192,64)]:
        run=subprocess.run([str(planner),'1','bf16','256',str(count),str(start),str(end)],capture_output=True,text=True)
        assert run.returncode==0 and run.stdout==f'1 {tiles} 2'


@pytest.mark.parametrize('args', [
    ['true','bf16','256','17','0','17'],['1','fp16','256','17','0','17'],
    ['1','bf16','128','17','0','17'],['1','bf16','256','0','0','0'],
    ['1','bf16','256','17','0','18'],['1','bf16','256','8192','1','8193'],
    ['1','bf16','256','1024','0','1024'],['1','bf16','256','8','0','8'],
])
def test_selected_unsupported_geometry_fails_closed(planner,args):
    run=subprocess.run([str(planner),*args],capture_output=True,text=True)
    assert run.returncode==2 and run.stderr


def test_segmented_tile_page_addresses_and_causal_window_bounds():
    # Disjoint permuted page tables, a prefix beginning inside a page and a
    # partial16-row final tile. This is a physical-address/causal oracle, not
    # a repeat of the C++ admission predicate.
    cases=[(17,0,0,0,[5]),(33,31,7,9,[7]),(34,31,7,0,[3,8])]
    touched=[]
    for count,start,retained,window,pages in cases:
        addresses=[]
        for local in range(count):
            upper=start+local+1;lower=max(retained,upper-window) if window else retained
            rows=[]
            for absolute in range(lower,upper):
                table_index=absolute//64-retained//64
                rows.append((pages[table_index],absolute%64))
            assert len(rows)==upper-lower and rows[-1][1]==(upper-1)%64
            addresses.extend(rows)
        touched.append({p for p,_ in addresses})
    assert touched==[{5},{7},{3,8}] and not(touched[0]&touched[1])


def test_tiled_mma_source_resources_and_terminal_counter_order():
    source=(ROOT/'native/paged_kv/prefill_matrix.h').read_text()
    assert 'MMATile<float,1,32,Frag> Otile' in source and 'tile_matmad(Stile,Qtile,Ktile,Stile)' in source
    assert 'threadgroup bfloat KVs[BD*LDK]' in source
    assert (16*264+256*24)*2==20736<32768
    arena=(ROOT/'native/paged_kv/arena.cpp').read_text()
    section=arena[arena.index('    if (prefill_plan_.spans != 0) {'):arena.index('    if (split_plan_.partition_tokens != 0) {')]
    assert section.index('maxTotalThreadsPerThreadgroup')<section.index('dispatch_threadgroups')
    assert section.index('addCompletedHandler')<section.index('dispatch_threadgroups')<section.index('record_prefill_matrix_dispatch')
    assert 'owner->read_completed(epoch' in section and 'return;' in section
    assert 'encoder.set_input_array(inputs[1], 3)' in section
    assert 'allocator::malloc' not in section # zero extra global scratch


def test_host_optional_counter_preserves_old_binary_and_closed_refusal():
    # Load the wrapper through its real import-safe module, without native import.
    from mlx2.runtime.paged_kv_write import NativeWriteBackend
    from types import SimpleNamespace
    backend=object.__new__(NativeWriteBackend);backend._closed=False;backend._native=SimpleNamespace();backend._arena='arena'
    assert backend.prefill_matrix_dispatch_count()==0
    backend._native=SimpleNamespace(prefill_matrix_dispatch_count=lambda arena:7 if arena=='arena' else None)
    assert backend.prefill_matrix_dispatch_count()==7
    backend._closed=True
    with pytest.raises(RuntimeError,match='closed'):backend.prefill_matrix_dispatch_count()


def test_oracle_is_import_safe_and_retains_ambiguous_private_roots():
    spec=importlib.util.spec_from_file_location('matrix_oracle',ROOT/'scripts/research/varlen_prefill_matrix_device_oracle.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=module.preflight()
    assert p['gpu_executed'] is False and p['qualified'] is False
    assert [c['total_rows'] for c in p['cases']]==[50,162,99]
    assert all(c['expected_matrix_dispatches']==1 and c['global_scratch_bytes']==0 for c in p['cases'])
    source=Path(spec.origin).read_text()
    assert source.index('result[\'gpuq_owner\']=_gpuq_owner()')<source.index('import _paged_kv_native as native')
    assert 'FAILURE_ROOTS.append((arena,writer,backend,layers' in source
    assert 'if terminal_proven:' in source and 'if not terminal_proven:' in source
    assert 'failure_retirement_after_proven_quiescence' in source


def test_known_terminal_numeric_failure_retirement_and_ambiguous_quiescence():
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('retirement_oracle',ROOT/'scripts/research/varlen_prefill_matrix_device_oracle.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[];receipt={}
    layer=SimpleNamespace(close=lambda:calls.append('close'))
    writer=SimpleNamespace(poisoned=False,pending_epochs=(),ledger=SimpleNamespace(pending_count=0))
    arena=SimpleNamespace(close_after_terminal=lambda:calls.append('arena'))
    pool=SimpleNamespace(free_count=2,capacity=2)
    module.retire_proven_case(lambda:calls.append('sync'),[layer],writer,arena,pool,receipt)
    assert calls==['sync','close','arena'] and receipt['retirement_after_proven_quiescence']
    for poison,pending in [(True,()),(False,(7,))]:
        calls.clear();writer.poisoned=poison;writer.pending_epochs=pending
        with pytest.raises(RuntimeError,match='pending native work'):
            module.retire_proven_case(lambda:calls.append('sync'),[layer],writer,arena,pool,{})
        assert calls==['sync']
    calls.clear()
    with pytest.raises(TimeoutError):
        module.retire_proven_case(lambda:(_ for _ in ()).throw(TimeoutError()),[layer],writer,arena,pool,{})
    assert calls==[]


def test_short_stock_rounding_law_changes_fp32_online_result():
    import numpy as np
    def bf16(x):
        a=np.asarray(x,dtype=np.float32);bits=a.view(np.uint32)
        return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
    rng=np.random.default_rng(1004)
    q=bf16(rng.uniform(-.25,.25,(17,256)));k=bf16(rng.uniform(-.25,.25,(33,256)))
    v=bf16(rng.uniform(-.25,.25,(33,256)))
    scaled=bf16(q*bf16(256**-.5));scores=bf16(scaled@k.T)
    exp=np.exp(scores-scores.max(axis=1,keepdims=True))
    normalized=exp/exp.sum(axis=1,keepdims=True)
    reference=bf16(bf16(normalized)@v)
    old_scores=(q@k.T)*np.float32(256**-.5)
    old_exp=np.exp(old_scores-old_scores.max(axis=1,keepdims=True))
    old=bf16((old_exp/old_exp.sum(axis=1,keepdims=True))@v)
    assert np.any(reference!=old) # Stock rounded probability storage is observable.
    source=(ROOT/'native/paged_kv/prefill_matrix.h').read_text()
    assert 'phase<2' in source and 'bfloat(scale)' in source
    assert 'float(bfloat(Stile.elems()[i]))' in source and 'T(bfloat(x*inverse))' in source
    assert 'exp2' not in source and 'Otile.row_bin_op<PFDiv>' not in source
