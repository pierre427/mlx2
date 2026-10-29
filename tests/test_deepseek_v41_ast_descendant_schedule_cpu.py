"""Pure metadata tests: no MLX import, arrays, model or GPU claim."""
from types import SimpleNamespace
import pytest
from mlx2.adapters.deepseek_v41_ast_descendant_schedule import build_schedule,bind_schedule

PARENTS=(-1,0,1,0)
LOCAL=((0,1,2),(0,1,2,3),(1,2,3,4),(0,1,2,5))
LOGICAL=((0,1,2),(0,1,2),(0,1,2,3),(0,1,2))


def test_shared_prefix_and_delta_masks_separate_physical_rows_from_uses():
    p=build_schedule(PARENTS,LOCAL,2,logical_rows=LOGICAL,prefix_pool_count=2,completed_nodes=(0,2))
    r=p.receipt(head_dim=512,index_head_dim=128)
    assert r['unique_physical_rows']=={'local':6,'pool':4,'index':4}
    assert r['logical_row_uses']=={'local':15,'pool':13,'index':13}
    local={row:run.consumers for run in p.runs if run.plane=='local'for row in range(run.first,run.first+run.count)}
    assert local=={0:0b1011,1:0b1111,2:0b1111,3:0b0110,4:0b0100,5:0b1000}
    pool={row:run.consumers for run in p.runs if run.plane=='pool'for row in range(run.first,run.first+run.count)}
    assert pool=={0:15,1:15,2:15,3:4}
    assert not r['dtype_conversion_saving_claim']and not r['hardware_read_saving_claim']


@pytest.mark.parametrize('rows',[(LOCAL[0],LOCAL[1],LOCAL[2],(0,1,2,3,5)),
    (LOCAL[0],(0,1,3),LOCAL[2],LOCAL[3]),(LOCAL[0],(0,1,2,2,3),LOCAL[2],LOCAL[3])])
def test_local_sibling_future_duplicate_and_incomplete_path_refused(rows):
    with pytest.raises(ValueError):build_schedule(PARENTS,rows,2)


def test_compressed_eligibility_refuses_sibling_row_without_reading_selected_ids():
    bad=LOGICAL[:-1]+((0,1,2,3),)
    with pytest.raises(ValueError,match='ancestry'):
        build_schedule(PARENTS,LOCAL,2,logical_rows=bad,prefix_pool_count=2,completed_nodes=(0,2))


def test_epoch_capability_and_descriptor_staleness_refused():
    active=[True];current=[True]
    def assert_current():
        if not current[0]:raise ValueError('base generation changed')
    segment=SimpleNamespace(base=SimpleNamespace(source=True),local_rows=LOCAL,logical_rows=LOGICAL,
        completed_nodes=(0,2),prefix_local_count=2,prefix_count=2,assert_current=assert_current)
    bound=bind_schedule(segment,PARENTS,epoch_key=('checkpoint','profile',1),check_live=lambda:active[0])
    assert bound.validate(('checkpoint','profile',1))is bound.schedule
    with pytest.raises(ValueError,match='another epoch'):bound.validate(('checkpoint','profile',2))
    active[0]=False
    with pytest.raises(ValueError,match='live'):bound.validate(bound.epoch_key)
    active[0]=True;segment.local_rows=LOCAL[:-1]+((1,2,5),)
    with pytest.raises(ValueError,match='map changed'):bound.validate(bound.epoch_key)
    segment.local_rows=LOCAL;segment.kv=object()
    with pytest.raises(ValueError,match='map changed'):bound.validate(bound.epoch_key)
    del segment.kv
    with pytest.raises(ValueError,match='immutable'):
        bind_schedule(segment,PARENTS,epoch_key=('profile',[]),check_live=lambda:True)
    current[0]=False
    with pytest.raises(ValueError,match='generation'):bound.validate(bound.epoch_key)
