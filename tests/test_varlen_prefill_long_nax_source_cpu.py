"""Host-only boundary checks for the default-off long paged NAX experiment."""
from __future__ import annotations

import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_long_nax_plan_actual_cpp_guards():
    source = r'''
#include "native/paged_kv/prefill_matrix.h"
#include <cassert>
#include <stdexcept>
using namespace mlx2::paged_kv;
int main() {
  auto make=[](unsigned a,unsigned b){return std::vector<std::vector<uint32_t>>{
    {a,0,a,0,0,0,(a+63)/64,0},{b,0,b,0,0,(a+63)/64,(b+63)/64,0}};};
  auto p=prefill_long_nax_plan(true,256,24,4,make(256,8192));
  assert(p.long_nax && !p.exact_nax && p.spans==2 && p.max_tiles==128 && !p.scratch_bytes);
  auto prefixed=make(320,321);prefixed[0][1]=31;prefixed[0][2]=351;
  prefixed[0][6]=(351+63)/64;prefixed[1][5]=prefixed[0][6];
  prefixed[1][1]=7;prefixed[1][2]=328;prefixed[1][6]=(328+63)/64;
  assert(prefill_long_nax_plan(true,256,24,4,prefixed).max_tiles==6);
  auto reject=[](auto fn){try{fn();return false;}catch(const std::invalid_argument&){return true;}};
  assert(reject([&]{prefill_long_nax_plan(true,256,24,4,make(255,256));}));
  assert(reject([&]{prefill_long_nax_plan(true,256,24,4,make(256,8193));}));
  assert(reject([&]{prefill_long_nax_plan(false,256,24,4,make(256,257));}));
  assert(reject([&]{prefill_long_nax_plan(true,128,24,4,make(256,257));}));
  assert(reject([&]{prefill_long_nax_plan(true,256,16,4,make(256,257));}));
  auto bad=make(256,257);bad[1][7]=128;
  assert(reject([&]{prefill_long_nax_plan(true,256,24,4,bad);}));
  bad=make(256,257);bad[1][1]=1;
  assert(reject([&]{prefill_long_nax_plan(true,256,24,4,bad);}));
  assert(!prefill_matrix_requested(nullptr));
  assert(!prefill_matrix_requested("0"));
  assert(prefill_matrix_requested("1"));
  assert(reject([&]{prefill_matrix_requested("yes");}));
}
'''
    with tempfile.TemporaryDirectory() as d:
        cpp=pathlib.Path(d)/'plan.cpp'; binary=pathlib.Path(d)/'plan'
        cpp.write_text(source)
        subprocess.run(['c++','-std=c++20','-I',str(ROOT),str(cpp),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)


def test_long_kernel_has_one_fused_dispatch_and_no_score_probability_scratch():
    arena=(ROOT/'native/paged_kv/arena.cpp').read_text()
    kernel=(ROOT/'native/paged_kv/prefill_long_nax.h').read_text()
    branch=arena.split('if (prefill_plan_.long_nax) {',1)[1].split('if (prefill_plan_.exact_nax)',1)[0]
    assert branch.count('encoder.dispatch_threadgroups(')==1
    assert 'record_prefill_long_nax_dispatch()' in branch
    assert 'read_completed(epoch' in branch
    assert 'mx::allocator::malloc(' not in branch
    assert 'const uint page=pages[table+uint(kb)/2]' in kernel
    assert 'const device bfloat* ktile = K +' in kernel
    assert 'const device bfloat* vtile = V +' in kernel
    assert 'Ktile.load(ktile, BD)' in kernel and 'Vtile.load(vtile, BD)' in kernel
    assert 'Ktile.load_rows(ktile, BD, short(params.kL_rem))' in kernel
    assert 'Vtile.load_rows(vtile, BD, max(short(0), valid_v))' in kernel
    assert 'long_nax_page(' not in kernel
    assert 'threadgroup float s_xchg' in kernel
    assert 'fast::exp2' in kernel


def test_long_page_tile_base_matches_token_address_at_boundaries():
    # The optimized loader is legal only because a 32-token KV tile cannot
    # cross a 64-token page boundary when the retained origin is zero.
    kv_heads, dim = 4, 256
    for end in (256, 257, 320, 321, 4096, 6950, 6929, 8192):
        for kb in range((end + 31) // 32):
            page = kb // 2
            tile_base = (page * kv_heads * 64 + (kv_heads - 1) * 64 +
                         (kb % 2) * 32) * dim
            for local in range(min(32, end - kb * 32)):
                for channel in (0, 127, 128, 255):
                    old = ((page * kv_heads * 64 + (kv_heads - 1) * 64 +
                            (kb * 32 + local) % 64) * dim + channel)
                    assert tile_base + local * dim + channel == old
