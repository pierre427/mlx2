"""CPU planning/timing-admission fixtures; no GPU/model execution."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('legacy_tuning_cpu',ROOT/'scripts/research/ds41_legacy_tiles_tuning.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_four_actual_plus_ten_derived_cases():
    cases=m.case_inventory();assert len(cases)+4==14
    projections=[c for c in cases if c['kind']=='derived_projection'];middle=[c for c in cases if c['kind']=='derived_middle']
    assert len(projections)==6 and len(middle)==4
    assert {len(c['row_counts'])for c in projections}=={3,15}
    assert {((c['row_counts'][0]+31)//32)for c in projections}=={1,2,3}
    for c in cases:
        assert len({(5 if x<=32 else 2,(x+31)//32)for x in c['row_counts']})==1
    assert (13,16,32)in[c['row_counts']for c in projections]
    assert (33,34,64)in[c['row_counts']for c in projections]
    assert (65,96,65)in[c['row_counts']for c in projections]

@pytest.mark.parametrize('rows',[61,131,226,305])
def test_synthetic_scores_and_nonidentity_valid_slots(rows):
    scores,slots=m.score_inputs(np,rows)
    assert scores.dtype==np.float32 and slots.dtype==np.uint32 and slots.shape==(rows,)
    assert np.isfinite(scores).all()and np.unique(scores).size>10
    assert int(slots.max())<len(scores)<=6144 and not np.array_equal(slots,np.arange(rows))
    a,b=m.score_inputs(np,rows);assert a.tobytes()==scores.tobytes()and b.tobytes()==slots.tobytes()


def test_partial_failure_disables_timing_even_with_matching_final_output():
    invoked=[]
    def compare(mx,np,functions,repeats):
        invoked.extend(functions);assert repeats==3
        return {k:{'synthetic':True}for k in functions}
    functions={'default32':lambda:0,'alt16':lambda:0}
    # Both would return identical final output; an upstream partial mismatch
    # must still prevent timing of the failed exact-law variant.
    checks={'default32':{'exact':True},'alt16':{'exact':False,'raw_exact':True,'boundary_exact':True}}
    timing,ref=m.admitted_timings(None,np,lambda:0,functions,checks,compare)
    assert invoked==['reference','default32']and timing['alt16']is None and timing['default32']is not None


def test_all_optional_negatives_leave_reference_timing_only():
    def compare(mx,np,functions,repeats):
        assert list(functions)==['reference'];return {'reference':{'synthetic':True}}
    timings,ref=m.admitted_timings(None,np,lambda:0,{'default32':lambda:1,'alt16':lambda:2},
        {'default32':{'exact':False},'alt16':{'exact':False}},compare)
    assert timings=={'default32':None,'alt16':None}
