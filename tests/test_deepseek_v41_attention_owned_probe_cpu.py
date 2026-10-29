"""Fail-closed diagnostic admission before any MLX or GPU import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest


def controller():
    path=Path(__file__).resolve().parents[1]/'scripts/research/probe_ds41_attention.py'
    spec=importlib.util.spec_from_file_location('_attention_owned_cpu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failure',['ownership','source'])
def test_rejects_before_gpu_and_preserves_atomic_failure(tmp_path,monkeypatch,failure):
    m=controller();out=tmp_path/'unit';out.mkdir()
    (out/'spec.json').write_text(json.dumps({'source_snapshot_sha256':'expected'}))
    def forbidden(*a,**kw):raise AssertionError('Reached memory/GPU preparation after refusal')
    monkeypatch.setattr(m,'source_manifest',forbidden)
    if failure=='ownership':
        def reject(spec):raise RuntimeError('Missing paired ownership')
        monkeypatch.setattr(m,'check_ownership',reject)
    else:
        monkeypatch.setattr(m,'check_ownership',lambda spec:{'verified':True})
        monkeypatch.setattr(m,'source_manifest',lambda root:{})
        monkeypatch.setattr(m,'digest',lambda manifest:'changed')
    assert m.component(out)==1
    result=json.loads((out/'results.json').read_text())
    assert result['passed']is False and result['error_type']=='RuntimeError'
    assert ('ownership'in result['error'])if failure=='ownership'else('Frozen source'in result['error'])


@pytest.mark.parametrize('failure',['unreleased','parser_hash','parser_symlink'])
def test_copy_refuses_unreleased_or_unsealed_parser(tmp_path,failure):
    m=controller();unit=tmp_path/'capture';(unit/'harness').mkdir(parents=True)
    parser=unit/'harness/attention_probe.py';body=b'raise AssertionError("Unsealed parser executed")\n'
    parser.write_bytes(body)
    seal=hashlib.sha256(body).hexdigest()
    if failure=='parser_symlink':
        target=unit/'saved.py';target.write_bytes(body);parser.unlink();parser.symlink_to(target)
    (unit/'spec.json').write_text(json.dumps({'harness_sha256':{'attention_probe.py':seal if failure!='parser_hash'else'bad'}}))
    (unit/'results.json').write_text(json.dumps({'status':'completed','passed':True}))
    (unit/'release-verified.json').write_text(json.dumps({'owners_absent':failure!='unreleased','source_unchanged':True}))
    out=tmp_path/'out';out.mkdir()
    with pytest.raises(ValueError):m.copy_capture(unit,out)
    assert not(out/'reference-capture').exists()
