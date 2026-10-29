"""Native-profile controller refuses invalid ownership/source before GPU work."""
import importlib.util
import json
from pathlib import Path
import pytest


def controller():
    p=Path(__file__).resolve().parents[1]/'scripts/research/probe_ds41_legacy_tiles.py'
    s=importlib.util.spec_from_file_location('_nax_owned_cpu',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


@pytest.mark.parametrize('failure',['ownership','source'])
def test_fail_closed_before_gpu_keeps_atomic_failure(tmp_path,monkeypatch,failure):
    m=controller();(tmp_path/'spec.json').write_text(json.dumps({'experiment':'nax_profile','source_snapshot_sha256':'expected'}))
    def forbidden(*a,**kw):raise AssertionError('Unexpected preparation after refusal')
    monkeypatch.setattr(m,'source_manifest',forbidden)
    if failure=='ownership':
        def reject(*a):raise RuntimeError('paired ownership refused')
        monkeypatch.setattr(m,'check_ownership',reject)
    else:
        monkeypatch.setattr(m,'check_ownership',lambda *a:{'verified':True})
        monkeypatch.setattr(m,'source_manifest',lambda *a:{})
        monkeypatch.setattr(m,'digest',lambda *a:'changed')
    assert m.component(tmp_path)==1
    r=json.loads((tmp_path/'results.json').read_text())
    assert r['passed']is False and r['error_type']=='RuntimeError'


def test_unreleased_capture_refused_before_parser_or_payload_read(tmp_path):
    m=controller();unit=tmp_path/'capture';unit.mkdir();(unit/'release-verified.json').write_text(json.dumps({'owners_absent':False,'source_unchanged':True}))
    source=unit/'capture.json';source.write_text('{}')
    with pytest.raises(ValueError,match='not released'):m.copy_capture(source,tmp_path/'out')
