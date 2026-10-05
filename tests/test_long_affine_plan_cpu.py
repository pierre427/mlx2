"""No-runtime equivalence and safety checks for long affine planning."""
import importlib.util
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for name, path in [('mlx2', ROOT/'src/mlx2'), ('mlx2.runtime', ROOT/'src/mlx2/runtime')]:
    module = types.ModuleType(name); module.__path__ = [str(path)]; sys.modules[name] = module
from mlx2.runtime.paged_attention_plan import PagedAttentionPlan, PageHandle, SequenceSpan
from mlx2.runtime.paged_attention_metal import build_paged_read_metadata


def make(counts=(6982,6961), starts=None, profile='prefill_long_n20_v1'):
    starts = starts or (0,)*len(counts)
    spans=[]; pages=[]; row=0
    for count,start in zip(counts,starts):
        end=start+count; length=(end+63)//64
        spans.append(SequenceSpan(row,count,start,end,0,0,len(pages),length,1))
        pages.extend(PageHandle(i,1) for i in range(len(pages),len(pages)+length)); row+=count
    return PagedAttentionPlan(tuple(spans),tuple(pages),row,24,4,256,'bfloat16',len(pages),
                              {p.page_id:1 for p in pages},20*8192*24,0,profile)


class AffinePlanTests(unittest.TestCase):
    def test_affine_visibility_matches_every_legacy_row(self):
        for counts,starts in [((256,257),(0,0)),((320,321),(31,7)),((4096,8192),(63,0)),
                              ((6982,6961),(0,0)),((256,),(7936,)),((7000,)*20,(0,)*20)]:
            p=make(counts,starts)
            legacy_work=0
            for span in p.spans:
                for r in range(span.row_count):
                    lower,upper=span.visible_bounds(r)
                    self.assertLess(lower,upper)
                    self.assertEqual(upper,span.query_start+r+1)
                    legacy_work+=p.query_heads
                self.assertEqual(span.visible_bounds(span.row_count-1)[1],span.kv_end)
            self.assertEqual(legacy_work,p.query_heads*p.total_rows)
            self.assertLessEqual(legacy_work,p.max_work_items)

    def test_long_never_calls_scalar_visibility(self):
        with patch.object(SequenceSpan,'visible_bounds',side_effect=AssertionError('scalar scan')):
            make((7000,)*20)

    def test_page_and_geometry_guards_still_refuse(self):
        p=make()
        changes=[{'live_generations':{}}, {'pool_capacity':1}, {'total_rows':p.total_rows+1},
                 {'page_table':p.page_table[:-1]}, {'max_work_items':1}, {'max_scratch_bytes':1},
                 {'spans':(replace(p.spans[0],query_start=1),)+p.spans[1:]},
                 {'spans':(replace(p.spans[0],retained_start=1),)+p.spans[1:]},
                 {'spans':(replace(p.spans[0],window=2),)+p.spans[1:]},
                 {'spans':(replace(p.spans[0],row_count=255),)+p.spans[1:]}]
        for change in changes:
            with self.subTest(change=tuple(change)),self.assertRaises((ValueError,TypeError)):
                replace(p,**change)

    def test_metadata_omission_preserves_every_other_field(self):
        p=make((320,321),(31,7)); old=build_paged_read_metadata(p)
        new=build_paged_read_metadata(p,omit_long_row_span=True)
        self.assertEqual(new,replace(old,row_span=()))
        self.assertEqual(old.row_span,(0,)*320+(1,)*321)
        self.assertEqual(len(build_paged_read_metadata(make((7000,)*20),omit_long_row_span=True).row_span),0)
        with self.assertRaises(ValueError):build_paged_read_metadata(p,omit_long_row_span=1)

    def test_scalar_plan_and_generic_metadata_remain_unchanged(self):
        span=SequenceSpan(0,2,63,65,61,0,0,2,1,'sliding',2)
        with patch.object(SequenceSpan,'visible_bounds',autospec=True,side_effect=lambda s,r:(max(s.retained_start,s.query_start+r+1-s.window),s.query_start+r+1)) as bounds:
            p=PagedAttentionPlan((span,),(PageHandle(0,1),PageHandle(1,1)),2,8,2,128,'float16',2,{0:1,1:1})
            self.assertEqual(bounds.call_count,2)
        self.assertEqual(build_paged_read_metadata(p).row_span,(0,0))
        with self.assertRaises(ValueError):build_paged_read_metadata(p,omit_long_row_span=True)
        q=make((256,)); q=replace(q,total_rows=1,spans=(replace(q.spans[0],row_count=1,query_start=255),),profile='q1_long_n20_v1',max_scratch_bytes=64*1024*1024)
        self.assertEqual(build_paged_read_metadata(q).row_span,(0,))
        with self.assertRaises(ValueError):build_paged_read_metadata(q,omit_long_row_span=True)

if __name__=='__main__':unittest.main()
