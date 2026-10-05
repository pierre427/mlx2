"""Compile actual N20 native geometry guards without importing MLX or using GPU."""

from pathlib import Path
import importlib.util
import subprocess
import pytest

from mlx2.runtime.paged_attention_plan import PagedAttentionPlan, PageHandle, SequenceSpan


ROOT = Path(__file__).resolve().parents[1]


def test_n20_write_and_read_guards_compile_and_reject_bad_geometry(tmp_path):
    source = tmp_path / "n20.cpp"
    source.write_text(r'''
#include "packed_n20.h"
#include "packed_n20_layout.h"
#include "q1_stock_long.h"
#include "prefill_matrix.h"
#include <cassert>
#include <functional>
using namespace mlx2::paged_kv;
static bool bad(const std::function<void()>& f) {
  try { f(); } catch (const std::invalid_argument&) { return true; }
  return false;
}
int main() {
  assert(!packed_n20_requested(nullptr));
  assert(packed_n20_requested("1"));
  assert(bad([]{ packed_n20_requested("yes"); }));
  std::vector<uint32_t> pages(42);for(uint32_t i=0;i<42;i++)pages[i]=i;
  auto p=validate_packed_n20_write({256,257,1},{0,0,2048},{0,0,0},
      {0,4,9},pages,64,514);
  assert(p.rows==514 && p.row_begin.size()==3 && p.row_begin[2]==513);
  assert(bad([]{validate_packed_n20_write({1,1},{0,0},{0,0},{0,1},{0,0},2,2);}));
  assert(bad([]{validate_packed_n20_write({1},{8192},{0},{0},{0},2,1);}));
  assert(bad([]{validate_packed_n20_write({1},{0},{1},{0},{0},2,1);}));
  assert(bad([]{validate_packed_n20_write(std::vector<uint32_t>(21,1),
      std::vector<uint32_t>(21,0),std::vector<uint32_t>(21,0),
      std::vector<uint32_t>(21,0),{0},64,21);}));
  auto q=q1_stock_long_n20_plan(256,24,std::vector<uint32_t>(20,8192));
  assert(q.partition_tokens==128 && q.scratch_bytes<=64*1024*1024);
  assert(!q1_stock_long_n20_singleton_requested(nullptr));
  assert(q1_stock_long_n20_singleton_requested("1"));
  assert(bad([]{q1_stock_long_n20_singleton_requested("yes");}));
  auto q1=q1_stock_long_n20_plan(256,24,{1025});
  assert(q1.partition_tokens==128 && q1.scratch_bytes==24*128*258*4);
  assert(bad([]{q1_stock_long_n20_plan(256,24,{1024});}));
  assert(bad([]{q1_stock_long_n20_plan(128,24,{1025,1026});}));
  assert(bad([]{q1_stock_long_n20_plan(256,24,{1024,1024});}));
  auto spans=std::vector<std::vector<uint32_t>>{{256,0,256,0,0,0,4,0},
                                               {257,0,257,0,0,4,5,0}};
  auto n=prefill_long_n20_plan(true,256,24,4,spans);
  assert(n.long_n20 && n.spans==2 && n.max_tiles==5 && n.scratch_bytes==0);
  spans[1][7]=1;
  assert(bad([&]{prefill_long_n20_plan(true,256,24,4,spans);}));
  // MLX can preserve an arbitrary leading stride on a logical singleton.
  // Every addressed head/channel byte still follows the token-major layout.
  validate_packed_n20_source_layout(1,4,256,std::vector<int64_t>{0,256,1},0,2048);
  validate_packed_n20_source_layout(1,4,256,std::vector<int64_t>{4096,256,1},128,2176);
  validate_packed_n20_source_layout(2,4,256,std::vector<int64_t>{1024,256,1},0,4096);
  assert(bad([]{validate_packed_n20_source_layout(2,4,256,std::vector<int64_t>{0,256,1},0,4096);}));
  assert(bad([]{validate_packed_n20_source_layout(1,4,256,std::vector<int64_t>{0,255,1},0,2048);}));
  assert(bad([]{validate_packed_n20_source_layout(1,4,256,std::vector<int64_t>{0,256,2},0,2048);}));
  assert(bad([]{validate_packed_n20_source_layout(1,4,256,std::vector<int64_t>{0,256,1},1,2048);}));
  assert(bad([]{validate_packed_n20_source_layout(1,4,256,std::vector<int64_t>{0,256,1},128,2175);}));
}
''')
    binary = tmp_path / "n20"
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "native/paged_kv"), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_n20_has_distinct_physical_dispatches_and_terminal_callback():
    native = (ROOT / "native/paged_kv/arena.cpp").read_text()
    assert "record_grouped_n20_write(rows_)" in native
    assert "record_prefill_long_n20_dispatch()" in native
    assert "record_q1_stock_long_n20_partial_dispatch()" in native
    assert "record_q1_stock_long_n20_reduce_dispatch()" in native
    assert "owner->completed(epoch,dispatched->load" in native
    assert "owner->read_completed(epoch,dispatched->load" in native
    assert "MTL::Size(prefill_plan_.max_tiles,query_heads_,prefill_plan_.spans)" in native


def test_q1_oracle_records_actual_singleton_layout_before_submission():
    binding = (ROOT / "native/paged_kv/binding.cpp").read_text()
    oracle = (ROOT / "scripts/research/varlen_packed_n20_q1_device_oracle.py").read_text()
    assert 'diagnostic_n20_source_layout' in binding
    assert 'if(!permit_diagnostic)' in binding
    assert "result['q1_source_layout']" in oracle
    assert 'mx.eval(q1_keys,q1_values)' in oracle
    assert 'private,q1_keys,q1_values,permit_candidate=True' in oracle


def test_n20_oracle_bounds_and_default_off_source_contract():
    path = ROOT / "scripts/research/varlen_packed_n20_device_oracle.py"
    spec = importlib.util.spec_from_file_location("n20_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert [len(case) for case in module.CASES] == [1, 2, 3, 20]
    assert module.case_plan(tuple([256] * 20))["rows"] == 5120
    assert 140000 <= module.case_plan(module.LARGE_CASE)['rows'] <= 163840
    for counts in ((), (255,), tuple([256] * 21), (8193,), (True,)):
        try:
            module.case_plan(counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe oracle counts admitted: {counts}")
    source = path.read_text()
    assert "_gpuq_owner()" in source
    assert "raw_bit_equal=raw" in source
    assert "failure_roots_retained=True" in source
    assert "MLX2_PAGED_PACKED_N20': '1'" in source
    assert "MAX_RSS = 12 << 30" in source
    assert "case['status'] = 'failed'" in source


def test_host_n20_profile_is_distinct_from_existing_b2():
    plan = (ROOT / "src/mlx2/runtime/paged_attention_plan.py").read_text()
    pack = (ROOT / "src/mlx2/runtime/paged_attention_pack.py").read_text()
    backend = (ROOT / "src/mlx2/runtime/qwen3_paged_native_backend.py").read_text()
    assert 'prefill_long_n20_v1' in plan and 'max_work_items!=(20 if' in plan
    assert 'def prepare_staged_token_read_n20(' in pack
    assert 'def append_packed_multirow_n20(' in backend
    assert 'def append_staged_grouped_q1_n20(' in backend
    assert 'def prepare_read_n20(' in backend


def _host_plan(counts, profile):
    spans, pages, row, table = [], [], 0, 0
    for count in counts:
        page_count = (count + 63) // 64
        spans.append(SequenceSpan(row, count, 0, count, 0, 0, table,
                                  page_count, 1, 'causal', None))
        pages.extend(PageHandle(index, 1)
                     for index in range(table, table + page_count))
        row += count
        table += page_count
    return PagedAttentionPlan(
        spans=tuple(spans), page_table=tuple(pages), total_rows=row,
        query_heads=24, kv_heads=4, head_dim=256, dtype='bfloat16',
        pool_capacity=table + 1,
        live_generations={page.page_id: 1 for page in pages},
        max_work_items=20*8192*24,
        max_scratch_bytes=64*1024*1024 if profile == 'q1_long_n20_v1' else 0,
        profile=profile)


def test_n20_host_plan_admits_twenty_real_rows_and_bounded_q1():
    cold = _host_plan((256,) * 20, 'prefill_long_n20_v1')
    assert cold.total_rows == 5120 and len(cold.spans) == 20
    assert cold.spans[-1].visible_bounds(255) == (0, 256)
    # Q1 has one real new row after a long prefix. Build the matching
    # descriptors with their existing 17-page full tables.
    spans, pages = [], []
    for lane in range(20):
        spans.append(SequenceSpan(lane, 1, 1024, 1025, 0, 0,
                                  lane*17, 17, 1, 'causal', None))
        pages.extend(PageHandle(lane*17 + block, 1) for block in range(17))
    q1 = PagedAttentionPlan(
        spans=tuple(spans), page_table=tuple(pages), total_rows=20,
        query_heads=24, kv_heads=4, head_dim=256, dtype='bfloat16',
        pool_capacity=341, live_generations={p.page_id: 1 for p in pages},
        max_work_items=20*8192*24, max_scratch_bytes=64*1024*1024,
        profile='q1_long_n20_v1')
    assert q1.spans[-1].visible_bounds(0) == (0, 1025)
    with pytest.raises(ValueError, match='N20 Q1'):
        PagedAttentionPlan(
            spans=(SequenceSpan(0,2,0,2,0,0,0,1,1),),
            page_table=(PageHandle(0,1),), total_rows=2,
            query_heads=24,kv_heads=4,head_dim=256,dtype='bfloat16',
            pool_capacity=2,live_generations={0:1},
            max_work_items=20*8192*24,max_scratch_bytes=64*1024*1024,
            profile='q1_long_n20_v1')
