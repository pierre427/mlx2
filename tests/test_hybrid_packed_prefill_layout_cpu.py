"""Stdlib-only addressing checks; no model or kernel numerical claim."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/research/hybrid_packed_prefill_layout.py'
spec = importlib.util.spec_from_file_location('hybrid_packed_prefill_layout', path)
import sys
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
cold_layout = module.cold_layout


class LayoutTests(unittest.TestCase):
    def test_actual_rows_no_rectangular_padding(self):
        p = cold_layout(((7, 32), (9, 96)))
        self.assertEqual(p.total_rows, 128)
        self.assertEqual(p.gdn_parts, ((1, 32, 0), (1, 96, 32)))
        self.assertEqual(p.logit_rows, (31, 127))

    def test_page_and_recurrence_boundaries(self):
        for a, b in ((1,17),(32,96),(63,65),(64,64),(127,129),(128,128),(4096,8192)):
            p = cold_layout(((0,a),(1,b)))
            self.assertEqual(p.total_rows,a+b)
            self.assertEqual(p.lane_position(a-1),(0,a-1))
            self.assertEqual(p.lane_position(a),(1,0))
            self.assertEqual(p.logit_rows,(a-1,a+b-1))

    def test_causal_visibility_has_no_cross_lane_leak(self):
        p = cold_layout(((3,3),(8,2)))
        for query in range(5):
            lane, position = p.lane_position(query)
            for key_lane,length in ((3,3),(8,2)):
                for key in range(length):
                    self.assertEqual(p.causal_key_visible(query,key_lane,key),
                                     lane==key_lane and key<=position)

    def test_singleton(self):
        p=cold_layout(((5,1),))
        self.assertEqual(p.gdn_parts,((1,1,0),))
        self.assertEqual(p.logit_rows,(0,))

    def test_invalid_lanes_refused(self):
        for lanes in ((),((0,0),),((0,True),),((True,2),),((0,2),(0,3)),
                      ((0,2),(1,3),(2,4)),((0,8193),),[(0,2)],((-1,2),)):
            with self.assertRaises(ValueError):cold_layout(lanes)

    def test_total_bound_is_separate(self):
        with self.assertRaises(ValueError):cold_layout(((0,65),(1,64)),max_total_rows=128)

    def test_invalid_query_or_key_refused(self):
        p=cold_layout(((0,2),(1,3)))
        for q in (-1,5,True):
            with self.assertRaises(ValueError):p.lane_position(q)
        for lane,key in ((2,0),(0,2),(1,-1),(True,0)):
            with self.assertRaises(ValueError):p.causal_key_visible(0,lane,key)


if __name__ == '__main__':unittest.main()
