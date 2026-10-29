"""Explicit geometry sweep protocol, actual-shape and graph timing guards."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    path=ROOT/'scripts/research'/f'{name}.py'
    s=importlib.util.spec_from_file_location('_geometry_'+name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def configured():
    m=load('ds41_native_mx_geometry_components')
    return dict(geometry_sweep=True,native_mx_options=m.expected_options())


def test_sweep_requires_explicit_flag_and_exact_options():
    m=load('ds41_native_mx_geometry_components');cfg=configured()
    assert m.options(cfg)['actual_rows']==[34,98,1,5]
    cfg.pop('geometry_sweep')
    with pytest.raises(ValueError):m.options(cfg)


@pytest.mark.parametrize('failure',['false','integer','rows','boolrow','tile','booltile','extras','repeats'])
def test_options_fail_closed(failure):
    m=load('ds41_native_mx_geometry_components');c=configured();v=c['native_mx_options']
    if failure=='false':c['geometry_sweep']=False
    elif failure=='integer':c['geometry_sweep']=1
    elif failure=='rows':v['actual_rows']=[1,5,34,98]
    elif failure=='boolrow':v['actual_rows'][2]=True
    elif failure=='tile':v['wide_tiles'].append([8,32])
    elif failure=='booltile':v['wide_tiles'][0][0]=True
    elif failure=='extras':v['force']=True
    else:v['repeats']=3
    with pytest.raises(ValueError):m.options(c)


def test_actual_geometry_count_and_no_synthetic_reshape():
    m=load('ds41_native_mx_geometry_components')
    assert sum(len(m.tiles_for(r))for r in m.ROWS)==12
    assert m.tiles_for(34)==m.tiles_for(98)==((16,32),(16,64),(32,32),(32,64))
    assert m.tiles_for(1)==m.tiles_for(5)==((16,32),(8,32))
    for r in (True,3,32,99):
        with pytest.raises(ValueError):m.tiles_for(r)


def row(m,rows):
    item=dict(capture={'rows':rows},reference_exact=True,coefficient_identity=True,
        coefficient_gate={'eligible':True},half_roundtrip={'finite':True,'bitwise_exact':True},
        fp64_reference={'within_bound':True},variants={})
    for a,b in m.tiles_for(rows):
        item['variants'][f'{a}x{b}']=dict(native_dispatched=True,native_finite=True,native_stable=True,
            canary_passed=True,admitted_raw_exact=True,admitted_owner_exact=True,
            fp64_oracle={'native':{'within_bound':True}})
    return item


@pytest.mark.parametrize('field',['native_dispatched','native_finite','native_stable','canary_passed','admitted_raw_exact','admitted_owner_exact'])
def test_each_variant_must_pass_component_gates(field):
    m=load('ds41_native_mx_geometry_components');cases=[row(m,r)for r in m.ROWS]
    assert all(m.engineering_verdicts(cases).values())
    cases[0]['variants']['16x64'][field]=False
    assert not all(m.engineering_verdicts(cases).values())


def test_missing_variant_and_F64_do_not_pass():
    m=load('ds41_native_mx_geometry_components');cases=[row(m,r)for r in m.ROWS]
    cases[0]['variants'].pop('32x64')
    assert not m.engineering_verdicts(cases)['native_dispatched']
    cases=[row(m,r)for r in m.ROWS];cases[1]['fp64_reference']['within_bound']=False
    assert not m.engineering_verdicts(cases)['actual_fp64_bound']
    assert not any(m.engineering_verdicts([]).values())


def test_controller_seals_geometry_without_weakening_budgets(tmp_path,monkeypatch):
    m=load('probe_ds41_native_mx');monkeypatch.setenv('MLX_ENABLE_TF32','0')
    cfg=dict(schema=m.SCHEMA,out=str(tmp_path),job_id=tmp_path.name,memory_limit_bytes=6<<30,
        host_live_reserve_bytes=9<<30,host_initial_floor_bytes=15<<30,
        numerical_environment={'MLX_ENABLE_TF32':'0'},rate_eligible=False,**configured())
    m.validate_spec(cfg,tmp_path)
    cfg['host_live_reserve_bytes']=8<<30
    with pytest.raises(ValueError):m.validate_spec(cfg,tmp_path)


def test_unit_cli_cannot_enable_unsealed_geometry(tmp_path):
    with pytest.raises(SystemExit):
        load('probe_ds41_native_mx').main(['unit','--output',str(tmp_path),'--geometry-sweep'])


def test_timed_admitted_lane_uses_fresh_graph_capability_without_diagnostics():
    text=(ROOT/'scripts/research/ds41_native_mx_geometry_components.py').read_text()
    tree=ast.parse(text)
    timed=[node for node in ast.walk(tree)if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Subscript)and 'admitted_cast_project_owner_'in ast.unparse(t)for t in node.targets)]
    assert len(timed)==1
    lane=timed[0].value
    assert isinstance(lane,ast.Lambda)
    calls=[ast.unparse(n.func)for n in ast.walk(lane)if isinstance(n,ast.Call)]
    assert calls==['source_native_mx.project_from_admitted_raw']
    assert not any(s in ast.unparse(lane)for s in ['prepare_input','eval','item','isfinite','roundtrip'])


def test_default_worker_untouched_and_no_GPU_import_at_module_import():
    m=load('ds41_native_mx_geometry_components');tree=ast.parse(Path(m.__file__).read_text())
    for n in tree.body:
        if isinstance(n,ast.Import):assert all(not a.name.startswith('mlx')for a in n.names)
        if isinstance(n,ast.ImportFrom):assert not(n.module or '').startswith('mlx')
    # Keep the original shape/control path distinct from this explicit sweep.
    assert load('ds41_native_mx_components').ACTUAL_ROWS==(34,1,3,5,9,32,98)
