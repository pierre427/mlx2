"""Inventory/oracle/refusal tests only; never compiles or runs a GPU kernel."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/research/ds41_plan_attention_components.py'
SPEC = importlib.util.spec_from_file_location('ds41_plan_attention_components_test', PATH)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


def test_frozen_source_revision_does_not_need_git_metadata(tmp_path,monkeypatch):
    (tmp_path/'desktop-source.json').write_text(json.dumps({'source_revision':'a'*40,'source_sha256':'b'*64}))
    import subprocess
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:pytest.fail('frozen source must not invoke git'))
    assert suite.source_binding(tmp_path)==('a'*40,'b'*64)


@pytest.mark.parametrize('revision,digest',[('bad','b'*64),('a'*40,'bad')])
def test_malformed_frozen_source_identity_refuses(tmp_path,revision,digest):
    (tmp_path/'desktop-source.json').write_text(json.dumps({'source_revision':revision,'source_sha256':digest}))
    with pytest.raises(ValueError,match='identities'):suite.source_binding(tmp_path)


def test_inventory_has_boundaries_profiles_and_explicit_unqualified_clock():
    p = suite.plan()
    base = {(cell['local_rows'], cell['selected_rows']) for cell in p['cells'] if cell['control'] == 'finite'}
    assert base == {(l, s) for l in (1, 127, 128) for s in (0, 1, 511, 512)}
    assert len(p['cells']) == 15 and p['case_count'] == 150
    assert p['conservative_operand_and_scratch_bound_bytes'] < p['array_budget_bytes'] == 32 << 20
    assert p['variants'] == ['ordinary_reference', 'joint_f32', 'native_hg1', 'native_hg2', 'native_hg4']
    assert p['numerical_acceptance']['exact_reduction_equality_claimed'] is False
    assert not p['qualification'] and not p['performance_qualified'] and not p['model_rate_claim']


def test_missing_callback_refuses_without_touching_backend():
    class Untouchable:
        def __getattr__(self, key): raise AssertionError('Backend touched before callback admission')
    with pytest.raises(ValueError, match='callback'): suite.run(Untouchable())
    for bad in (None, False, 1, object()):
        with pytest.raises(ValueError, match='callback'): suite.run(Untouchable(), bad)


def test_cpu_backend_refuses_before_array_creation_or_shader_import():
    calls = []
    backend = SimpleNamespace(gpu='gpu', default_device=lambda: 'cpu',
        array=lambda *a, **k: (_ for _ in ()).throw(AssertionError('Array creation forbidden')))
    with pytest.raises(ValueError, match='caller-owned GPU'):
        suite.run(backend, lambda: calls.append('checked') or True)
    assert calls == ['checked']


def test_live_guard_must_affirm_ownership_before_backend_probe():
    class Untouchable:
        def __getattr__(self, key): raise AssertionError('Backend touched after refused ownership')
    for answer in (False, None, 0, 'yes'):
        with pytest.raises(RuntimeError, match='did not return True'):
            suite.run(Untouchable(), lambda answer=answer: answer)


@pytest.mark.parametrize('control', ['finite', 'masked_tail', 'masked_nan', 'sink_only'])
def test_fixture_numpy_oracle_is_finite_and_preserves_sink_mass(control):
    cell = dict(local_rows=128 if control != 'sink_only' else 1,
                selected_rows=512 if control != 'sink_only' else 0, control=control)
    host = suite._fixture(cell)
    value = suite.numpy_oracle(host)
    assert value.shape == (1, 64, 1, 512) and value.dtype == np.float32
    assert np.isfinite(value).all()
    diagnostic = suite._diagnostic(value, value)
    assert diagnostic['tolerance_passed'] and diagnostic['full_array_compared']
    assert diagnostic['exact_reduction_equality_claimed'] is False
    if control == 'sink_only': np.testing.assert_array_equal(value, np.zeros_like(value))


def test_diagnostic_detects_nonfinite_and_tolerance_failure_without_exact_claim():
    expected = np.zeros((1, 64, 1, 512), np.float32)
    bad = expected.copy(); bad.flat[-1] = np.nan
    assert not suite._diagnostic(bad, expected)['finite']
    bad.flat[-1] = 1.0
    diagnostic = suite._diagnostic(bad, expected)
    assert diagnostic['finite'] and not diagnostic['tolerance_passed']
    assert diagnostic['largest_error_window'][0]['flat_index'] == bad.size-1
    assert not diagnostic['exact_reduction_equality_claimed']
