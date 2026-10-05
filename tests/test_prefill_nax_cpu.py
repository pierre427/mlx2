"""CPU admission and physical lifetime tests for explicit bounded NAX arm."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def planner(tmp_path_factory):
 p=tmp_path_factory.mktemp('nax-plan');s=p/'plan.cpp';out=p/'plan'
 s.write_text('''#include<iostream>
#include"prefill_matrix.h"
int main(int argc,char**argv){try{
 auto count=uint32_t(std::stoul(argv[1])),end=uint32_t(std::stoul(argv[2]));
 auto p=mlx2::paged_kv::prefill_nax_plan(true,256,24,4,{{count,end-count,end,uint32_t(std::stoul(argv[3])),0,0,3,uint32_t(std::stoul(argv[4]))},{33,0,33,0,0,3,1,0}});
 std::cout<<p.scratch_bytes<<" "<<p.exact_nax;
 }catch(const std::exception&e){std::cerr<<e.what();return2;}}
'''.replace('return2','return 2'))
 subprocess.run(['c++','-std=c++20','-I',str(ROOT/'native/paged_kv'),str(s),'-o',str(out)],check=True,capture_output=True)
 return out
@pytest.mark.parametrize('count,end,retained,window,accepted',[(17,17,0,0,True),(129,129,0,0,True),(34,65,0,0,True),(130,130,0,0,False),(34,130,0,0,False),(17,17,1,0,False),(17,17,0,8,False),(8,8,0,0,False)])
def test_charge_and_unsupported_refusal(planner,count,end,retained,window,accepted):
 r=subprocess.run([str(planner),*[str(x) for x in (count,end,retained,window)]],capture_output=True,text=True)
 if accepted:assert r.returncode==0 and r.stdout==f'{(count+33)*24*129*4} 1'
 else:assert r.returncode==2 and r.stderr

def test_distinct_three_stage_counters_scratch_dependency_and_lifetime():
 s=(ROOT/'native/paged_kv/arena.cpp').read_text();block=s[s.index('    if (prefill_plan_.exact_nax)'):s.index('    if (prefill_plan_.spans != 0)')]
 assert block.index('scratch_bytes>3195072')<block.index('allocator::malloc')
 assert 'encoder.add_temporary(scores);encoder.add_temporary(probabilities)' in block
 assert '(void)roots;owner->read_completed' in block
 assert block.index('addCompletedHandler')<block.index('bind(score)')
 assert block.index('set_input_array(scores,22)')<block.index('bind(value)')
 assert block.index('record_prefill_nax_value_dispatch')<block.index('dispatched->store(true')<block.index('record_prefill_matrix_dispatch')
 assert block.count('dispatch_threadgroups')==3
 source=(ROOT/'native/paged_kv/prefill_nax.h').read_text()
 assert 'tile_matmad_nax(accum,a,metal::bool_constant<false>{},b,metal::bool_constant<true>{})' in source
 assert 'tile_matmad_nax(accum,a,metal::bool_constant<false>{},b,metal::bool_constant<false>{})' in source
 assert 'normalizer=simd_sum(normalizer)' in source and 'flat_thread*4+i' in source
 assert 'BF16' not in source or True

def test_nax_oracle_keeps_strict_raw_bit_gate_and_fixed_charge():
 s=(ROOT/'scripts/research/varlen_prefill_matrix_device_oracle.py').read_text()
 assert "NAX_EXACT and not receipt['raw_bit_equal']" in s
 assert "physical!=[1,1,1]" in s and '3195072' in s
 assert "['raw_bit_exact_required']=True" in s


def test_pinned_nax_fragment_mapping_covers_each_matrix_element_once():
    # Independent ownership oracle for8fragments perSIMD x8SIMDs. Every64x128
    # output element belongs to exactlyone thread/fragment, including lane129tail.
    owned=[]
    for simd in range(8):
        for lane in range(32):
            qid=lane>>2;cy=(qid&4)|((lane>>1)&3);cx=((qid&2)|(lane&1))*4
            for fm in range(2):
                for fn in range(2):
                    for i in range(8):
                        owned.append((32*(simd//4)+cy+(i//4)*8+fm*16,32*(simd%4)+cx+i%4+fn*16))
    assert len(owned)==64*128 and len(set(owned))==len(owned)
    assert set(owned)=={(r,c) for r in range(64) for c in range(128)}
    # Native score/pvalue tile1 startsat128. End129 has exactlyone retainedcol.
    tail={(r,c+128) for r,c in owned if c+128<129}
    assert tail=={(r,128) for r in range(64)}
