"""Frame-owned routing metadata; no device imports or inference."""
from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest
from mlx2.adapters.deepseek_v41_source_lfu import LFUSourceExpertStore,LFUIndex

def owner():
 w=LFUSourceExpertStore.__new__(LFUSourceExpertStore);w.index=LFUIndex();w.stats=Counter();w.kind=lambda key:'q1'if key[1]%2 else'original';return w

def test_planner_histogram_and_reference_observe_identical_atlas_counts():
 ids=np.array([[1,0,1],[3,2,0]],np.uint32);a,b=owner(),owner()
 a.observe_routes('blk.4',ids);b.observe_routes('blk.4',ids,histogram=(np.array([0,1,2,3],np.uint32),np.array([2,2,1,1],np.uint32)))
 assert a.index.counts==b.index.counts==Counter({(4,0):2,(4,1):2,(4,2):1,(4,3):1})
 assert a.stats==b.stats==Counter(q1_token_selections=3,original_token_selections=3)

@pytest.mark.parametrize('hist',[([0,0],[1,1]),([384],[2]),([-1],[2]),([1],[0]),([1],[1]),([1,2],[2]),([1.0],[2])])
def test_invalid_frame_histogram_refuses_without_counting(hist):
 w=owner()
 with pytest.raises(ValueError):w.observe_routes('blk.0',np.array([[1,2]],np.uint32),histogram=hist)
 assert not w.index.counts and not w.stats
