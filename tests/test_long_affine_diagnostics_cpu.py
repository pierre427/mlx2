"""Closed-form diagnostics preserve every legacy counter without runtime imports."""
import sys,types,unittest,random
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
for name,path in [('mlx2',ROOT/'src/mlx2'),('mlx2.runtime',ROOT/'src/mlx2/runtime')]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
from mlx2.runtime.paged_attention_plan import PagedAttentionPlan,PageHandle,SequenceSpan
from mlx2.runtime.paged_pack_diagnostics import describe_packed_read,PackedReadWork


def make(counts,starts,profile='prefill_long_n20_v1'):
 spans=[];pages=[];row=0
 for count,start in zip(counts,starts):
  length=(start+count+63)//64
  spans.append(SequenceSpan(row,count,start,start+count,0,0,len(pages),length,1))
  pages.extend(PageHandle(i,1) for i in range(len(pages),len(pages)+length));row+=count
 return PagedAttentionPlan(tuple(spans),tuple(pages),row,24,4,256,'bfloat16',len(pages),{p.page_id:1 for p in pages},(20 if profile=='prefill_long_n20_v1' else 2)*8192*24,0,profile)


def legacy(p):
 serial=longest=slots=0
 for span in p.spans:
  for r in range(span.row_count):
   lo,hi=span.visible_bounds(r);v=hi-lo
   serial+=v*p.query_heads;longest=max(longest,v);slots+=((v+63)//64)*64*p.query_heads
 return PackedReadWork(len(p.spans),p.total_rows,p.query_heads,p.total_rows*p.query_heads,len(p.page_table),len(set(p.page_table)),serial,longest,slots,serial)


class DiagnosticsTests(unittest.TestCase):
 def test_exhaustive_boundaries_and_randomized_equivalence(self):
  rng=random.Random(1004)
  cases=[((256,257),(0,0)),((320,321),(31,7)),((8192,),(0,)),((7000,)*20,(0,)*20)]
  cases += [((256,),(start,)) for start in (0,1,62,63,64,65,127,128,129,7936)]
  cases += [((rng.randrange(256,4097),),(rng.randrange(0,128),)) for _ in range(24)]
  for counts,starts in cases:
   with self.subTest(counts=counts,starts=starts):self.assertEqual(describe_packed_read(make(counts,starts)),legacy(make(counts,starts)))
  p=make((256,257),(0,0),'prefill_long_nax_v1');self.assertEqual(describe_packed_read(p),legacy(p))

 def test_long_never_calls_row_visibility(self):
  p=make((7000,)*20,(0,)*20)
  with patch.object(SequenceSpan,'visible_bounds',side_effect=AssertionError('row scan')):describe_packed_read(p)

 def test_scalar_sliding_semantics_remain_exact(self):
  span=SequenceSpan(0,5,63,68,61,0,0,2,1,'sliding',3)
  p=PagedAttentionPlan((span,),(PageHandle(0,1),PageHandle(1,1)),5,8,2,128,'float16',2,{0:1,1:1})
  old=legacy(p)
  with patch.object(SequenceSpan,'visible_bounds',autospec=True,side_effect=lambda s,r:(max(s.retained_start,s.query_start+r+1-s.window),s.query_start+r+1)) as fn:
   self.assertEqual(describe_packed_read(p),old);self.assertEqual(fn.call_count,5)

 def test_requires_exact_validated_plan(self):
  with self.assertRaises(TypeError):describe_packed_read(types.SimpleNamespace(profile='prefill_long_n20_v1'))

if __name__=='__main__':unittest.main()
