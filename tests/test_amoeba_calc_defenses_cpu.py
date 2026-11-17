"""Content-blind alarms, no premature math cuts, and lane-compaction binding."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/research'))
from amoeba_calc_defenses import CompletionDefenses, phase_budgets
from amoeba_calc_batch import RowBinder
from mlx2.spiral_conclude import SpiralDetector


def test_run_on_requires_minimum_and_current_repetition():
    guard=CompletionDefenses(min_tokens=64)
    for i in range(63):assert guard.feed(i%8) is None
    assert guard.feed(7)=='run_on'
    assert guard.receipt()['run_on']['ngram']==6


def test_short_math_does_not_trigger_defenses():
    guard=CompletionDefenses()
    for i in range(500):assert guard.feed(i%12) is None
    assert guard.receipt()['enabled']


def test_spiral_uses_existing_detector_without_supplying_answer():
    spiral=SpiralDetector(min_tokens=64,min_candidates=3,stall_tokens=16,min_hedges=2,window=64)
    guard=CompletionDefenses(min_tokens=10000,spiral=spiral)
    text=''
    for i in range(1,129):
        if i%16==0:text+='The answer is 5. But maybe another interpretation, perhaps the answer is 4. '
        reason=guard.feed(i,text if i%16==0 else None)
        if reason:break
    assert reason=='spiral'
    assert guard.receipt()['spiral']['fired_at']>=64


def test_budget_counts_the_injected_nudge():
    initial,reserve=phase_budgets(4096,24)
    assert initial+reserve+24==4096 and reserve==512


def test_row_binding_resets_each_forward_after_compaction():
    class Tiles:
        forward_ids={'calculator':1};stash_shape={'calculator':(4,1)}
    class Processor:
        tile=type('Tile',(),{'id':'calculator'})()
        def bind_row(self,row):self.row=row
    tiles=Tiles();binder=RowBinder(tiles);a,b,c,d=[Processor() for _ in range(4)]
    for row,p in enumerate([a,b,c,d]):assert binder.bind(p)==(row,4)
    tiles.forward_ids['calculator']=2;tiles.stash_shape['calculator']=(2,1)
    assert binder.bind(b)==(0,2) and binder.bind(d)==(1,2)
