"""Actual C++ host-law tests for persistent long prefill to Q1 handoff."""
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def test_long_prefill_q1_handoff_cpp():
    source=r'''
#include "native/paged_kv/prefill_matrix.h"
#include "native/paged_kv/q1_stock_long.h"
#include <cassert>
#include <stdexcept>
using namespace mlx2::paged_kv;
int main(){
 auto reject=[](auto fn){try{fn();return false;}catch(const std::invalid_argument&){return true;}};
 validate_long_nax_q1_handoff(false,false,false,false); // ordinary path unchanged
 validate_long_nax_q1_handoff(true,true,false,false); // multirow validates its own plan
 auto plan=q1_stock_long_plan(256,24,{6951,6930});
 assert(plan.partition_tokens==128);
 validate_long_nax_q1_handoff(true,false,plan.partition_tokens!=0,false);
 std::vector<std::vector<uint32_t>> b1{{1,6950,6951,0,0,0,(6951+63)/64,0}};
 assert(bounded_long_scalar_b1(true,true,256,24,4,1,b1,0));
 validate_long_nax_q1_handoff(true,false,false,bounded_long_scalar_b1(true,true,256,24,4,1,b1,0));
 assert(reject([&]{validate_long_nax_q1_handoff(true,false,false,false);}));
 assert(!bounded_long_scalar_b1(false,true,256,24,4,1,b1,0));
 assert(!bounded_long_scalar_b1(true,false,256,24,4,1,b1,0));
 assert(!bounded_long_scalar_b1(true,true,128,24,4,1,b1,0));
 assert(!bounded_long_scalar_b1(true,true,256,16,4,1,b1,0));
 assert(!bounded_long_scalar_b1(true,true,256,24,4,2,b1,0));
 assert(!bounded_long_scalar_b1(true,true,256,24,4,1,b1,128));
 auto bad=b1;bad[0][2]=1024;bad[0][1]=1023;bad[0][6]=(1024+63)/64;
 assert(!bounded_long_scalar_b1(true,true,256,24,4,1,bad,0));
 bad=b1;bad[0][7]=128;assert(!bounded_long_scalar_b1(true,true,256,24,4,1,bad,0));
 bad=b1;bad[0][3]=64;assert(!bounded_long_scalar_b1(true,true,256,24,4,1,bad,0));
 bad=b1;bad[0][6]--;assert(!bounded_long_scalar_b1(true,true,256,24,4,1,bad,0));
 assert(reject([&]{prefill_long_nax_plan(true,256,24,4,b1);}));
}
'''
    with tempfile.TemporaryDirectory() as d:
        cpp=Path(d)/'q1.cpp';out=Path(d)/'q1';cpp.write_text(source)
        subprocess.run(['c++','-std=c++20','-I',str(ROOT),str(cpp),'-o',str(out)],check=True)
        subprocess.run([str(out)],check=True)


def test_native_call_orders_handoff_after_stock_long_plan():
    source=(ROOT/'native/paged_kv/arena.cpp').read_text()
    start=source.index('const bool long_nax_requested=')
    end=source.index('auto as_array =',start)
    admission=source[start:end]
    assert admission.index('if (stock_long) split_plan = q1_stock_long_plan') < admission.index('validate_long_nax_q1_handoff(')
    assert 'bounded_long_scalar_b1(' in admission
    assert 'if(long_nax_requested && !multiquery)' not in admission
