"""Timeline lifecycle metadata and completion-boundary checks, no GPU/trace."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import inspect

path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_ast_batched_timeline.py'
spec=importlib.util.spec_from_file_location('tile_trace',path);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_unavailable_notification_is_wall_only_and_does_not_spawn(monkeypatch,tmp_path):
    monkeypatch.setattr(probe,'_StartedNotification',lambda name:(_ for _ in ()).throw(RuntimeError('notification unavailable')))
    monkeypatch.setattr(probe.subprocess,'Popen',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('must not start')))
    trace=probe._Trace(tmp_path,lambda:True);trace.start()
    assert trace.record['available']is False and trace.record['ready']is False
    assert 'wall_only'in trace.record['evidence_scope']


def test_stopped_capture_requires_started_barrier_success_and_actual_bundle(tmp_path):
    trace=probe._Trace(tmp_path,lambda:True);trace.process=SimpleNamespace(poll=lambda:0,returncode=0)
    trace.record['ready']=True
    assert trace.stop()['available']is False
    (tmp_path/'native-ast-metal.trace').mkdir()
    assert trace.stop()['available']is True


def test_frame_brackets_do_not_add_eval_or_sync_to_normal_forward():
    source=inspect.getsource(probe.run)
    body=source[source.index('start_ns=time.perf_counter_ns()'):source.index('end_ns=time.perf_counter_ns()')]
    assert body.count('execute_batched_tree(')==1
    assert 'mx.eval'not in body and 'synchronize'not in body
    assert "timeline_methodology_passed=False"in source
