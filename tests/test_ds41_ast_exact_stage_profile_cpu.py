"""Synthetic observer contracts; no model or GPU execution."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import mlx.core as mx
import pytest

P = Path(__file__).resolve().parents[1] / 'scripts/research/ds41_ast_exact_stage_profile.py'
spec = importlib.util.spec_from_file_location('exact_stage_profile', P)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class Array:
    pass


def fake_xp():
    calls = []
    return SimpleNamespace(array=Array, eval=lambda *args: calls.append(('eval', args)),
                           synchronize=lambda: calls.append(('sync',))), calls


def test_roots_keep_owned_descriptor_arrays_and_refuse_arbitrary_object_walk():
    a, b = Array(), Array()
    class Descriptor:
        def evaluation_roots(self):
            return [a, {'retained': b}]
    class Opaque:
        @property
        def state(self):
            pytest.fail('Do not inspect arbitrary state properties')
    cycle = [Descriptor(), Opaque()]
    cycle.append(cycle)
    assert profile._roots([a, cycle, {'duplicate': b}], Array) == [a, b]


def test_wrapped_result_is_identical_and_inputs_drain_before_timer():
    xp, calls = fake_xp()
    ticks = iter([10., 12.])
    observer = profile.StageObserver(xp, clock=lambda: next(ticks))
    a, b = Array(), Array()
    result = [b]
    wrapped = observer.wrap(lambda arg: result, 'test')
    assert wrapped(a) is result
    assert calls == [('eval', (a,)), ('sync',), ('eval', (b,)), ('sync',)]
    assert observer.events[0]['seconds'] == 2.
    assert observer.summary()['test']['calls'] == 1


def test_mutated_index_outputs_are_explicit_completion_roots():
    xp, calls = fake_xp()
    observer = profile.StageObserver(xp)
    shared = {}
    a = Array()
    def select(rows):
        rows['indices'] = a
    wrapped = observer.wrap(select, 'index_selection', extra=lambda args, kwargs, result: args[0])
    assert wrapped(shared) is None
    assert ('eval', (a,)) in calls


def test_nested_intervals_are_not_double_counted_in_exclusive_time():
    xp, _ = fake_xp()
    ticks = iter([0., 1., 3., 5.])
    observer = profile.StageObserver(xp, clock=lambda: next(ticks))
    inner = observer.wrap(lambda: Array(), 'inner')
    outer = observer.wrap(lambda: inner(), 'outer')
    outer()
    summary = observer.summary()
    assert summary['outer']['seconds'] == 5.
    assert summary['outer']['exclusive_seconds'] == 3.
    assert summary['inner']['seconds'] == 2.
    assert observer.events[1]['parent'] == 0 and observer.events[1]['depth'] == 1


def test_inherited_method_patch_restores_instance_layout_after_error():
    xp, _ = fake_xp()
    observer = profile.StageObserver(xp)
    class Block:
        def finish(self, value):
            raise RuntimeError('injected')
    block = Block()
    observer.patch(block, 'finish', 'finish', layer=3)
    try:
        with pytest.raises(RuntimeError, match='injected'):
            block.finish(Array())
    finally:
        observer.restore()
    assert 'finish' not in vars(block)
    assert observer.stack == [] and observer.events[0]['completed'] is False


def test_patch_conflict_restores_all_and_reports_concurrent_mutation():
    xp, _ = fake_xp()
    observer = profile.StageObserver(xp)
    original = lambda: None
    owner = SimpleNamespace(call=original)
    observer.patch(owner, 'call', 'call')
    owner.call = lambda: 'foreign'
    with pytest.raises(RuntimeError, match='concurrently'):
        observer.restore()
    assert owner.call is original and not observer.restores


def test_observer_cannot_wrap_an_existing_observer():
    xp, _ = fake_xp()
    observer = profile.StageObserver(xp)
    with pytest.raises(ValueError, match='nest another'):
        observer.wrap(observer.wrap(lambda: None, 'first'), 'second')


def test_run_refuses_cpu_before_adapter_or_evidence_access(tmp_path, monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **kw: pytest.fail('GPU forbidden'))
    try:
        with pytest.raises(ValueError, match='parent-owned GPU'):
            profile.run(None, tmp_path, lambda: True, None, None, None, None, None)
    finally:
        mx.set_default_device(old)


def test_real_stage_entry_points_patch_and_restore_without_running_math(monkeypatch):
    from mlx2.adapters import deepseek_v41_ast_batched_attention_exact as attention
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
    from mlx2.adapters import deepseek_v41_ast_batched_moe as moe
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **kw: pytest.fail('GPU forbidden'))
    original = (attention.attention_half_tile, attention._select, dense.head_tile, moe.moe_tile)
    class Block:
        def finish(self, *args):
            pytest.fail('No model execution')
    block = Block()
    model = SimpleNamespace(_weight_executions=0, layers=[block])
    with pytest.raises(RuntimeError, match='injected'):
        with profile.observe(model, fake_xp()[0]):
            assert moe.moe_tile._ds41_stage_observer
            assert attention.attention_half_tile._ds41_stage_observer
            assert dense.head_tile._ds41_stage_observer
            assert 'finish' in vars(block)
            raise RuntimeError('injected')
    assert original == (attention.attention_half_tile, attention._select, dense.head_tile, moe.moe_tile)
    assert 'finish' not in vars(block)
