"""New component scheduling guards; all GPU execution belongs to the parent."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.fixture
def helper():
    p=Path(__file__).resolve().parents[1]/'scripts/research/ds41_fp8_projection_components.py'
    s=importlib.util.spec_from_file_location('fp8_tiles_components_cpu',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_decode_inventory_is_disjoint_and_exactly_bounded(helper):
    variants=helper.configurations(1)
    assert len(variants)==len({n for n,o in variants})==6
    assert {(o['simdgroups'],o['results_per_simd'])for n,o in variants}=={(s,r)for s in (2,4,8)for r in (4,8)}


def test_pp_inventory_includes_stock_and_only_reviewed_tiles(helper):
    variants=helper.configurations(99)
    assert [o['tile']for n,o in variants]==[(32,32),(64,32),(32,64),(16,64)]


@pytest.mark.parametrize('rows',[0,2,98,100,-1])
def test_other_row_geometries_refuse(helper,rows):
    with pytest.raises(ValueError):helper.configurations(rows)


def test_no_gpu_gate_precedes_capture_and_weight_reads(helper,tmp_path):
    mx=SimpleNamespace(default_device=lambda:'cpu',gpu='gpu')
    with pytest.raises(RuntimeError,match='caller-owned'):
        helper.run_components(mx,np,tmp_path,tmp_path,{},None,None)


def test_existing_evidence_refused_before_parser_or_artifact_work(helper,tmp_path):
    mx=SimpleNamespace(default_device=lambda:'gpu',gpu='gpu')
    with pytest.raises(ValueError,match='overwrite'):
        helper.run_components(mx,np,tmp_path,tmp_path,{'fp8_projection_kernels':{}},None,None)


def test_component_requires_raw_gate_before_both_timing_paths(helper):
    source=Path(helper.__file__).read_text()
    assert "raw_exact=checks['raw']['exact']and checks['raw_then_owner']['exact']"in source
    assert "fused_exact=raw_exact and checks['fused_owner']['exact']"in source
    assert "if raw_exact:functions[name+'_raw_owner']"in source
    assert "if fused_exact:functions[name+'_fused']"in source
    assert "compare(mx,np,functions,repeats=repeats)"in source
    assert 'SourceV41Weights'not in source and 'load_dense'not in source
    assert 'profilerlabel evidence mustcome from parentMetaltrace'in source
    assert "candidate_used_by_model=False"in source


def test_followup_options_preserve_default_and_seal_subset(helper):
    repeats,selected=helper.component_options({})
    assert repeats==3 and len(selected[1])==6 and len(selected[99])==4
    repeats,selected=helper.component_options({'fp8_projection_options':dict(repeats=7,
        pp_variants=['m64_n32','m32_n32'],decode_variants=['s2_r4','s2_r8'])})
    assert repeats==7 and [n for n,o in selected[99]]==['m64_n32','m32_n32']
    assert [n for n,o in selected[1]]==['s2_r4','s2_r8']


@pytest.mark.parametrize('options',[None,{'unknown':1},{'repeats':True},{'repeats':0},{'repeats':16},
    {'pp_variants':[]},{'pp_variants':['s2_r4']},{'decode_variants':['s2_r4','s2_r4']},
    {'decode_variants':'s2_r4'},{'pp_variants':[[]]}])
def test_bad_followup_options_refuse_before_capture(helper,options):
    with pytest.raises(ValueError):helper.component_options({'fp8_projection_options':options})
