"""Synthetic CPU dispatch receipts; never runs a GPU or invents a crossover."""
from dataclasses import replace

import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_prefill import Job
from mlx2.adapters.deepseek_v41_source_tile_policy import (
    ShapeKey, TileMeasurement, STAGE_COSTS, PROBE_ROWS, TILES,
    select_tile, row_slot_cost, probe_inventory)


def _key(**kwargs):
    args=dict(hardware='M5-test',checkpoint='synthetic-routes@test1',source_revision='base',
              source_snapshot='snapshot1',kind='original',projection='ffn',rows=34,
              layout='mapped_rows',numerical_profile='exact_reference')
    args.update(kwargs)
    return ShapeKey(**args)


def _measurement(**kwargs):
    args=dict(key=_key(),candidate='tile16x32_test',tile=(16,32),
              reference_seconds=(2.,2.2,1.8),candidate_seconds=(1.,1.1,.9),
              evidence='synthetic-comparison@fixture',cost_scope=STAGE_COSTS,numerical_gate_passed=True)
    args.update(kwargs)
    return TileMeasurement(**args)


def test_reference_without_matching_whole_stage_evidence():
    key=_key()
    for measurements in ([], [_measurement(key=_key(rows=1))],
                         [_measurement(key=_key(kind='q1'))],
                         [_measurement(key=_key(source_snapshot='other'))],
                         [_measurement(cost_scope=('gate_up',))],
                         [_measurement(numerical_gate_passed=False)]):
        assert select_tile(key,measurements,minimum_ratio=1.05).candidate=='reference'


def test_exact_shape_evidence_selects_only_with_preregistered_win():
    m=_measurement()
    selected=select_tile(_key(),[m],minimum_ratio=1.05)
    assert (selected.candidate,selected.tile,selected.evidence)==(m.candidate,m.tile,m.evidence)
    assert selected.qualification=='unqualified'
    assert select_tile(_key(),[m],minimum_ratio=2).candidate=='reference'


@pytest.mark.parametrize('field,value',[('hardware','other'),('checkpoint','other@fp'),
 ('source_revision','other'),('projection','down'),('layout','fresh_compact_rows'),
 ('numerical_profile','half_input')])
def test_each_binding_dimension_prevents_evidence_transfer(field,value):
    assert select_tile(_key(**{field:value}),[_measurement()],minimum_ratio=1).candidate=='reference'


@pytest.mark.parametrize('samples', [(),(0.,),(float('nan'),),(float('inf'),),(-1.,),(True,)])
def test_nonpositive_nonfinite_or_bool_measurements_refuse(samples):
    with pytest.raises(ValueError): _measurement(reference_seconds=samples,candidate_seconds=samples)


@pytest.mark.parametrize('ratio',[None,True,.99,float('nan'),float('inf')])
def test_threshold_must_be_explicit_valid_policy(ratio):
    with pytest.raises(ValueError): select_tile(_key(),[],minimum_ratio=ratio)


def test_fixed_tile_capacity_is_not_shared_weight_batching():
    jobs=[Job(i,np.array([i],np.uint32),np.array([0],np.uint32),1,'original')for i in range(6)]
    result=row_slot_cost(jobs,16)
    assert result['active_rows']==6 and result['scheduled_rows']==96
    assert result['useful_row_fraction']==1/16
    assert 'not GPU occupancy' in result['scope']


@pytest.mark.parametrize('rows',PROBE_ROWS)
@pytest.mark.parametrize('tile_rows',[1,2,4,8,16,32])
def test_capacity_tails_independent_cpu_walk(rows,tile_rows):
    j=Job(0,np.arange(rows,dtype=np.uint32),np.arange(rows,dtype=np.uint32),rows,'original')
    result=row_slot_cost([j],tile_rows)
    # Count individual logical tile slots rather than restating ceil formula.
    slots=0
    for begin in range(0,rows,tile_rows):
        slots+=len(list(range(begin,begin+tile_rows)))
    assert result['active_rows']==rows and result['scheduled_rows']==slots


def test_probe_axes_include_negative_controls_layout_tax_and_whole_M():
    probes=probe_inventory()
    assert {p['rows']for p in probes}==set(PROBE_ROWS)
    assert {p['tile']for p in probes}==set(TILES)
    assert {p['input_layout']for p in probes}=={'mapped_rows','fresh_compact_rows'}
    assert all(p['preserve_whole_expert_M'] and p['cost_scope']==STAGE_COSTS for p in probes)
    assert all(p['route']=='decode_control' for p in probes if p['rows']==1)
