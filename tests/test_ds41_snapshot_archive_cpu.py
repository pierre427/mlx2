# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

spec=importlib.util.spec_from_file_location('archive',Path(__file__).resolve().parents[1]/'scripts/research/ds41_snapshot_archive.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.fixture
def frozen(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    (repo/'same.py').write_text('unchanged\n');(repo/'changed.py').write_text('original\n')
    subprocess.run(['git','add','.'],cwd=repo,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','base'],cwd=repo,check=True)
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    source=repo/'qualification/runs/ds41-options-fixture/source';source.mkdir(parents=True)
    files={'same.py':b'unchanged\n','changed.py':b'new bytes\n','new.txt':b'added\n'}
    for name,data in files.items():(source/name).write_bytes(data)
    hashes={name:m.sha(data)for name,data in files.items()}
    (source/'desktop-source.json').write_bytes(m.encode(dict(source_revision=rev,source_sha256=m.manifest_digest(hashes),files=hashes)))
    (source/'admin-token').write_text('never archive this token')
    output=repo/'archive';index=m.pack(repo,repo/'qualification/runs',output)
    return repo,source,output,index


def test_reconstruct_exact_overlay_and_descriptor_excludes_unmanifested(frozen,tmp_path):
    repo,source,archive,index=frozen;row=index['snapshots'][0]
    assert row['overlay_files']==2 and m.verify(repo,archive)['files']==3
    destination=tmp_path/'restored';m.verify(repo,archive,overlay_key=row['overlay_object'],destination=destination)
    assert (destination/'desktop-source.json').read_bytes()==(source/'desktop-source.json').read_bytes()
    assert sorted(p.name for p in destination.iterdir())==['changed.py','desktop-source.json','new.txt','same.py']
    assert all((destination/n).read_bytes()==(source/n).read_bytes()for n in ['changed.py','same.py','new.txt'])
    with pytest.raises(FileExistsError):m.verify(repo,archive,overlay_key=row['overlay_object'],destination=destination)


@pytest.mark.parametrize('damage',['missing','corrupt','tampered_index','tampered_descriptor'])
def test_missing_corrupt_and_tampered_archives_fail_closed(frozen,damage):
    repo,source,archive,index=frozen;objects=m.Objects(archive);key=index['snapshots'][0]['overlay_object'];overlay=json.loads(objects.get(key))
    if damage in ('missing','corrupt'):
        path=objects.path(overlay['overlays']['new.txt'])
        if damage=='missing':path.unlink()
        else:path.write_bytes(b'not gzip')
    elif damage=='tampered_index':
        index['snapshots'][0]['source_snapshot_sha256']='0'*64;(archive/'index.json').write_bytes(m.encode(index))
    else:
        desc=json.loads(objects.get(overlay['descriptor_object']));desc['files']['same.py']='0'*64
        overlay['descriptor_object']=objects.put(m.encode(desc));newkey=objects.put(m.encode(overlay))
        index['snapshots'][0].update(overlay_object=newkey,descriptor_sha256=overlay['descriptor_object']);(archive/'index.json').write_bytes(m.encode(index))
    with pytest.raises((ValueError,OSError)):m.verify(repo,archive)


def test_missing_original_and_sensitive_paths_refused(frozen,tmp_path):
    repo,source,archive,index=frozen;(source/'same.py').unlink()
    with pytest.raises(FileNotFoundError):m.pack(repo,repo/'qualification/runs',repo/'second-archive')
    for path in ('../escape','.git/config','admin-token','x/__pycache__/a.pyc'):
        with pytest.raises(ValueError):m.safe_path(path)
