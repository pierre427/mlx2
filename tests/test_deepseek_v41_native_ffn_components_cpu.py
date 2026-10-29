"""Complete-stage worker numerical/metadata contracts; CPU-only fixtures."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('_native_ffn_components_cpu', ROOT/'scripts/research/ds41_native_ffn_components.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_options_are_explicit_whole_stage_fixed_repeat_and_geometry():
    assert load().options({'native_ffn_options': {'repeats': 7, 'tile': [16, 32], 'stage': 'complete_original_expert_v1'}})['repeats'] == 7


@pytest.mark.parametrize('change', [{'repeats': 3}, {'repeats': True}, {'tile': [16.,32]},
    {'tile': (16,32)}, {'stage': 'down_only'}, {'extra': 'yes'}])
def test_options_refuse_silent_profile_or_shape_changes(change):
    value = {'repeats': 7, 'tile': [16, 32], 'stage': 'complete_original_expert_v1'}
    value.update(change)
    with pytest.raises(ValueError): load().options({'native_ffn_options': value})


def test_nonexact_real_comparator_key_records_first_consumer_difference():
    module = load()
    reference = {key: np.zeros((2, 2), np.float32) for key in module.FRONTIERS}
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate['gate_raw'][0, 1] = np.float32(1e-7)
    candidate['middle'][1, 0] = .25
    checks = module.compare_frontiers(np, reference, candidate)
    assert module.first_changed(checks, module.FRONTIERS)['frontier'] == 'gate_raw'
    first = module.first_changed(checks, module.CONSUMERS)
    assert first['frontier'] == 'middle' and first['coordinate'] == [1, 0]
    assert module.first_changed(module.compare_frontiers(np, reference, reference), module.FRONTIERS) is None


def good_row():
    return dict(reference_exact=True, coefficient_identity=True, admission={'eligible': True},
        native_finite=True, native_stable=True, all_fp64_bounds_passed=True, zero_aba_passed=True,
        native_stage_dispatched=True, normal_graph_checks={'normal': {'exact': True}},
        assembly_fold_checks={'independent': {'exact': True}}, native_checks={'raw': {'exact': False}})


def test_native_reduction_differences_do_not_automatically_fail_engineering():
    result = load().engineering_verdicts([good_row()])
    assert set(result) == load().VERDICTS and all(result.values())
    assert not any(load().engineering_verdicts([]).values())


@pytest.mark.parametrize('key', ['reference_exact', 'coefficient_identity', 'native_finite', 'native_stable',
    'all_fp64_bounds_passed', 'zero_aba_passed', 'native_stage_dispatched'])
def test_every_required_control_remains_mandatory(key):
    row = good_row()
    row[key] = False
    assert not all(load().engineering_verdicts([row]).values())


@pytest.mark.parametrize('key', ['normal_graph_checks', 'assembly_fold_checks'])
def test_normal_shape_and_independent_fold_control_cannot_be_hidden(key):
    row = good_row()
    row[key]['failed'] = {'exact': False}
    assert not load().engineering_verdicts([row])['finite_stable']


def test_cpu_fold_preserves_ascending_add_order_and_signed_zero():
    module = load()
    parts = np.zeros((2,6,3), np.float32)
    parts[:,0] = 1e10
    parts[:,1] = -1e10
    parts[:,2] = 1.
    assert np.all(module.assembly_fold(np, parts) == 1.)
    assert np.all(module.assembly_fold(np, parts[:,[0,2,1,3,4,5]]) == 0.)
    zero = module.assembly_fold(np, np.full((1,6,3), -0., np.float32))
    assert np.all(zero.view(np.uint32) == 0)


@pytest.mark.parametrize('key', ['shared_gu_input', 'values', 'scores', 'order'])
def test_same_layer_cases_cannot_splice_different_frames(key):
    module = load()
    a = {name: np.zeros((2,2), np.float32) for name in ('shared_gu_input','values','scores','order')}
    b = {name: value.copy() for name,value in a.items()}
    b[key][0,0] = 1.
    with pytest.raises(ValueError, match='splice'):
        module.validate_frame_inventory([({'layer':0},a),({'layer':0},b)])
    module.validate_frame_inventory([({'layer':0},a),({'layer':1},b)])


def test_source_reader_binding_hashes_seals_layout_and_current_original_membership():
    from mlx2.adapters.deepseek_v41_source_buffers import canonical_layout
    from mlx2.adapters.deepseek_v41_source_weights import expert_names
    module = load()
    layout = canonical_layout('original')
    description=[dict(name=t.name,offset=t.offset,dtype=t.dtype,shape=list(t.shape))for t in layout.tensors]
    sha=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    seals={'shard':{'size':1,'inode':2}}
    artifact=dict(identity={'fingerprint':'a'*64},tensors={n:{'file':'shard'}for n in expert_names(0,38)},seals=seals)
    case=dict(layer=0,expert=38,source_binding=dict(source_fingerprint='a'*64,files=['shard'],
        source_seals_sha256=sha(seals),layout_sha256=sha(description)))
    host=SimpleNamespace(layout=layout,data=np.zeros(128,np.uint8))
    assert module.source_coefficients(artifact,case,host,{(0,38)})['source_seals_exact']
    with pytest.raises(ValueError):module.source_coefficients(artifact,case,host,set())
    artifact['seals']['shard']['inode']=3
    with pytest.raises(ValueError):module.source_coefficients(artifact,case,host,{(0,38)})


def test_module_import_has_no_gpu_work_and_timing_calls_only_fresh_graphs():
    module=load()
    tree=ast.parse(Path(module.__file__).read_text())
    for node in tree.body:
        if isinstance(node,ast.Import):assert not any(x.name.startswith('mlx')for x in node.names)
        if isinstance(node,ast.ImportFrom):assert not(node.module or '').startswith('mlx')
    text=Path(module.__file__).read_text()
    assert "'reference_complete': lambda: reference_graph(admission)" in text
    assert "'native_complete': lambda: native_graph(admission)" in text
    assert 'native middle may differ from stock' in text
    assert 'global all-expert output packing/concatenation are excluded' in text
