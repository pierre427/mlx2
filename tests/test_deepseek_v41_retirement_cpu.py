"""Source retirement fails closed; all files here are tiny temporary fixtures."""
import importlib.util
import json
from pathlib import Path
import time

import pytest


@pytest.fixture
def retiring(tmp_path,monkeypatch):
    location=Path(__file__).resolve().parents[1]/'scripts/research/retire_deepseek_v41_gguf.py'
    spec=importlib.util.spec_from_file_location('v41_retirement_fixture',location);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'SOURCE_BYTES',8)
    source=tmp_path/m.FILENAME;source.write_bytes(b'original')
    native=tmp_path/'native';native.mkdir();(native/'preserved.safetensors').write_bytes(b'converted')
    receipt=tmp_path/'retirement.json';identity='b'*64
    manifest={'source':{'stat':m.stamp(source)},'verification':{'passed':True}}
    monkeypatch.setattr(m,'check_native',lambda path:({'identity':{'fingerprint':identity}},manifest))
    monkeypatch.setattr(m,'no_open_files',lambda path:None)
    return m,source,native,receipt,identity


def evidence(tmp_path,native,identity):
    p=tmp_path/'runtime';p.mkdir()
    data={
        'spec.json':{'native_mlx':True,'fingerprint':identity,'model':str(native),'checkpoint':'native@'+identity,'source_revision':'c'*40,'source_snapshot_sha256':'d'*64},
        'results.json':{'passed':True,'artifact':{'fingerprint':identity},'ordinary_reference':{'text':'Hello.','finite':True},'APCv2':{'passed':True},'snapshot_restore':{k:True for k in ['logits_bitwise_equal','state_arrays_bitwise_equal','state_metadata_equal','restored_original_frontier']},'cleanup_memory':{'active_bytes':520},'finished_at':time.time()+10},
        'desktop-lease.json':{'release_task':{'released':True},'complete_worker':{'status':'completed'}},
        'paired-locks.json':{'status':'command_completed','command_returncode':0},
    }
    for name,value in data.items():(p/name).write_text(json.dumps(value))
    return p


def test_stage_is_reversible_and_does_not_delete(retiring):
    m,source,native,receipt,identity=retiring
    row=m.stage(source,native,receipt)
    pending=Path(row['pending']);assert not source.exists() and pending.read_bytes()==b'original'
    assert m.restore(receipt)['phase']=='restored'
    assert source.read_bytes()==b'original' and not pending.exists()
    assert (native/'preserved.safetensors').read_bytes()==b'converted'


def test_open_source_refuses_staging(retiring,monkeypatch):
    m,source,native,receipt,_=retiring
    def busy(path):raise RuntimeError('source busy')
    monkeypatch.setattr(m,'no_open_files',busy)
    with pytest.raises(RuntimeError,match='busy'):m.stage(source,native,receipt)
    assert source.exists() and not receipt.exists()


def test_restore_preserves_original_integrity_stat_field_names(retiring):
    m,source,native,receipt,_=retiring
    current=m.stamp(source)
    seal={'sha256':m.SOURCE_SHA,'weight_stat':{'size':8,'mtime_ns':current['mtime_ns'],
        'device':current['dev'],'inode':current['ino']}}
    path=source.parent/'mlx2-v41-integrity.json';path.write_text(json.dumps(seal))
    m.stage(source,native,receipt);m.restore(receipt)
    assert json.loads(path.read_text())==seal


def test_wrong_source_name_or_changed_source_refuses_staging(retiring):
    m,source,native,receipt,_=retiring
    with pytest.raises(ValueError,match='authorized'):m.stage(source.with_name('other.gguf'),native,receipt)
    source.write_bytes(b'changed!')
    with pytest.raises(ValueError,match='changed'):m.stage(source,native,receipt)
    assert source.exists() and not receipt.exists()


@pytest.mark.parametrize('file,key,value',[
    ('results.json','passed',False),
    ('results.json','APCv2',{'passed':False}),
    ('results.json','ordinary_reference',{'text':'wrong','finite':True}),
    ('results.json','snapshot_restore',{}),
    ('results.json','cleanup_memory',{'active_bytes':2<<30}),
    ('results.json','finished_at',0),
    ('desktop-lease.json','release_task',{'released':False}),
    ('paired-locks.json','command_returncode',1),
    ('spec.json','native_mlx',False),
    ('spec.json','fingerprint','e'*64),
])
def test_failed_or_unreleased_runtime_never_deletes(retiring,tmp_path,file,key,value):
    m,source,native,receipt,identity=retiring
    row=m.stage(source,native,receipt);p=evidence(tmp_path,native,identity)
    path=p/file;data=json.loads(path.read_text());data[key]=value;path.write_text(json.dumps(data))
    with pytest.raises(ValueError):m.finalize(receipt,p)
    assert Path(row['pending']).read_bytes()==b'original'
    assert json.loads(receipt.read_text())['phase']=='staged'


def test_changed_staged_inode_never_deletes(retiring,tmp_path):
    m,source,native,receipt,identity=retiring
    row=m.stage(source,native,receipt);pending=Path(row['pending']);pending.unlink();pending.write_bytes(b'original')
    with pytest.raises(ValueError,match='identity changed'):m.finalize(receipt,evidence(tmp_path,native,identity))
    assert pending.exists()


def test_finalize_deletes_only_the_proven_source_and_preserves_provenance(retiring,tmp_path):
    m,source,native,receipt,identity=retiring
    (source.parent/'config.json').write_text('{"model_type":"old"}')
    row=m.stage(source,native,receipt);result=m.finalize(receipt,evidence(tmp_path,native,identity))
    assert result['phase']=='deleted' and not source.exists() and not Path(row['pending']).exists()
    assert (native/'preserved.safetensors').read_bytes()==b'converted'
    assert (source.parent/'config.gguf-retired.json').exists()
    assert json.loads((source.parent/'GGUF-RETIRED.json').read_text())['runtime_validation']['ownership_released']
