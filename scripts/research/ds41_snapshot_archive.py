# SPDX-License-Identifier: MIT
"""Content-address frozen DS4.1 sources against their actual Git base.

Only descriptor-declared source files are read. Tokens, process/cache state and
unmanifested freeze attempts are never archived. Verification reconstructs each
manifest byte stream from Git or an authenticated gzip object; originals stay.
"""
from pathlib import Path, PurePosixPath
import argparse
import gzip
import hashlib
import json
import os
import subprocess


def sha(data):return hashlib.sha256(data).hexdigest()
def encode(value):return json.dumps(value,sort_keys=True,indent=2).encode()+b'\n'
def manifest_digest(files):return sha(json.dumps(files,sort_keys=True).encode())


def safe_path(name):
    path=PurePosixPath(name)
    if not name or path.is_absolute()or '..'in path.parts or str(path)!=name:
        raise ValueError('Unsafe source path: '+name)
    forbidden={'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.apcv2','apcv2-state'}
    if any(v in forbidden for v in path.parts)or path.suffix=='.pyc' or path.name in {'admin-token','admin-token.txt','admin-token.json','.DS_Store','.env'}:
        raise ValueError('Secret/cache/transient source path refused: '+name)
    return path


class GitBase:
    def __init__(self,repo,revision):
        self.repo=Path(repo)
        resolved=subprocess.check_output(['git','rev-parse','--verify',revision+'^{commit}'],cwd=repo,text=True).strip()
        if resolved!=revision:raise ValueError('Source revision must be the full actual Git commit')
        self.revision=revision;self.paths={};self.cache={}
        tree=subprocess.check_output(['git','ls-tree','-rz',revision],cwd=repo)
        for item in tree.split(b'\0'):
            if not item:continue
            meta,name=item.split(b'\t',1);mode,kind,oid=meta.decode().split()
            if kind=='blob':self.paths[name.decode()]=oid
        self.process=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    def read(self,name):
        oid=self.paths.get(name)
        if oid is None:return None
        if oid not in self.cache:
            self.process.stdin.write((oid+'\n').encode());self.process.stdin.flush()
            header=self.process.stdout.readline().decode().split()
            if len(header)!=3 or header[0]!=oid or header[1]!='blob':raise ValueError('Missing Git base blob: '+name)
            size=int(header[2]);data=self.process.stdout.read(size)
            if len(data)!=size or self.process.stdout.read(1)!=b'\n':raise ValueError('Truncated Git base blob')
            self.cache[oid]=(data,sha(data))
        return self.cache[oid][0]
    def close(self):
        self.process.stdin.close();self.process.wait()


class Objects:
    def __init__(self,root):self.root=Path(root);self.cache={}
    def path(self,key):
        if len(key)!=64 or any(c not in'0123456789abcdef'for c in key):raise ValueError('Invalid SHA256 object key')
        return self.root/'objects'/key[:2]/(key+'.gz')
    def put(self,data):
        key=sha(data);target=self.path(key)
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb')as out:out.write(gzip.compress(data,compresslevel=9,mtime=0))
        if self.get(key)!=data:raise ValueError('Object hash collision/corruption')
        return key
    def get(self,key):
        if key not in self.cache:
            data=gzip.decompress(self.path(key).read_bytes())
            if sha(data)!=key:raise ValueError('Corrupt source object: '+key)
            self.cache[key]=data
        return self.cache[key]


def descriptor(objects,overlay):
    raw=objects.get(overlay['descriptor_object']);value=json.loads(raw)
    if value['source_revision']!=overlay['source_revision']or value['source_sha256']!=overlay['source_snapshot_sha256']:
        raise ValueError('Overlay descriptor binding mismatch')
    if manifest_digest(value['files'])!=value['source_sha256']:raise ValueError('Descriptor manifest digest mismatch')
    for name,key in value['files'].items():safe_path(name)
    if set(overlay['overlays'])-set(value['files']):raise ValueError('Unmanifested overlay paths')
    return raw,value


def reconstruct(objects,base,overlay,*,destination=None):
    raw,desc=descriptor(objects,overlay);count=0;total=0
    if destination is not None:
        destination=Path(destination)
        if destination.exists()and any(destination.iterdir()):raise FileExistsError('Restore destination must be empty')
        destination.mkdir(parents=True,exist_ok=True)
    for name,expected in desc['files'].items():
        data=objects.get(overlay['overlays'][name])if name in overlay['overlays']else base.read(name)
        if data is None or sha(data)!=expected:raise ValueError('Reconstruction hash mismatch/missing: '+name)
        count+=1;total+=len(data)
        if destination is not None:
            path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            path.chmod(overlay.get('modes',{}).get(name,0o644))
    if destination is not None:(destination/'desktop-source.json').write_bytes(raw)
    return dict(files=count,bytes=total,all_manifest_hashes_verified=True)


def pack(repo,runs,output):
    repo=Path(repo).resolve();runs=Path(runs).resolve();output=Path(output).resolve()
    if output.exists()and any(output.iterdir()):raise FileExistsError('Refusing nonempty source archive output')
    output.mkdir(parents=True,exist_ok=True);objects=Objects(output);bases={};snapshots={};rows=[];excluded=[]
    sources=sorted(runs.glob('ds41-options-*/*/desktop-source.json'))
    if not sources:raise ValueError('No frozen descriptors found')
    try:
        for index,path in enumerate(sources,1):
            if path.parent.name!='source'and not path.parent.name.startswith('rejected-source-attempt-'):continue
            raw=path.read_bytes();desc=json.loads(raw);rev=desc['source_revision'];identity=desc['source_sha256']
            if manifest_digest(desc['files'])!=identity:raise ValueError('Manifest digest mismatch: '+str(path))
            base=bases.setdefault(rev,None)
            if base is None:base=bases[rev]=GitBase(repo,rev)
            overlay=dict(schema='mlx2.frozen-source-overlay.v1',source_revision=rev,source_snapshot_sha256=identity,
                         descriptor_object=objects.put(raw),overlays={},modes={})
            byte_count=0
            for name,expected in desc['files'].items():
                safe_path(name);source=path.parent/name
                if source.is_symlink()or not source.resolve().is_relative_to(path.parent.resolve()):raise ValueError('Linked source refused: '+name)
                data=source.read_bytes();byte_count+=len(data)
                if sha(data)!=expected:raise ValueError('Original frozen source is corrupt: '+str(source))
                original=base.read(name)
                if original is None or sha(original)!=expected:overlay['overlays'][name]=objects.put(data)
                mode=source.stat().st_mode&0o777
                if mode!=0o644:overlay['modes'][name]=mode
            manifest_bytes=encode(overlay);overlay_key=objects.put(manifest_bytes)
            # This performs reconstruction from archive+Git, never trusting the
            # original bytes just read as proof of archive completeness.
            proof=reconstruct(objects,base,overlay)
            snapshots[overlay_key]=dict(overlay_object=overlay_key,**proof)
            rows.append(dict(original_source_path=str(path.parent),source_path=str(path.parent.relative_to(repo)),
                unit_path=str(path.parent.parent.relative_to(repo)),source_revision=rev,source_snapshot_sha256=identity,
                descriptor_sha256=sha(raw),overlay_object=overlay_key,overlay_files=len(overlay['overlays']),**proof))
            if index%10==0:print(json.dumps(dict(event='snapshots_verified',completed=index,total=len(sources))),flush=True)
        for unit in sorted(runs.glob('ds41-options-*')):
            if not unit.is_dir():continue
            for path in sorted(unit.glob('rejected-source-attempt-*')):
                if path.is_dir()and not(path/'desktop-source.json').exists():
                    excluded.append(dict(source_path=str(path.relative_to(repo)),reason='never-executed unstable freeze; no desktop-source manifest'))
        index=dict(schema='mlx2.ds41-source-snapshot-archive.v1',base_revisions=sorted(bases),snapshots=rows,
            unique_overlays=len(snapshots),verified_manifest_files=sum(r['files']for r in rows),
            original_manifest_bytes=sum(r['bytes']for r in rows),unmanifested_rejected_attempts=excluded,
            originals_deleted=False,verification='Every original file hashed; every file independently reconstructed from actual Git base or SHA256-authenticated gzip overlay object and rehashed.',
            restore_command='python scripts/research/ds41_snapshot_archive.py restore --repo REPOSITORY --archive ARCHIVE --overlay OVERLAY_OBJECT_SHA256 --destination EMPTY_DIRECTORY',
            exclusions='Only desktop-source.json-declared files; admin tokens, cache/state/process files and unmanifested originals never copied.')
        index['object_files']=sum(1 for p in(output/'objects').rglob('*.gz'));index['object_compressed_bytes']=sum(p.stat().st_size for p in(output/'objects').rglob('*.gz'))
        temporary=output/'index.json.tmp';temporary.write_bytes(encode(index));os.replace(temporary,output/'index.json')
        print(json.dumps({k:v for k,v in index.items()if k not in('snapshots','unmanifested_rejected_attempts')}),flush=True)
        return index
    finally:
        for base in bases.values():
            if base is not None:base.close()


def verify(repo,archive,*,overlay_key=None,destination=None):
    archive=Path(archive);objects=Objects(archive);bases={};proofs=[]
    try:
        index=json.loads((archive/'index.json').read_text())
        rows=index['snapshots']if overlay_key is None else[r for r in index['snapshots']if r['overlay_object']==overlay_key]
        if not rows:raise ValueError('Overlay not indexed')
        if destination is not None:rows=rows[:1]
        for row in rows:
            overlay=json.loads(objects.get(row['overlay_object']))
            if overlay['source_revision']!=row['source_revision']or overlay['source_snapshot_sha256']!=row['source_snapshot_sha256']or overlay['descriptor_object']!=row['descriptor_sha256']:
                raise ValueError('Archive index binding mismatch')
            rev=overlay['source_revision']
            if rev not in bases:bases[rev]=GitBase(repo,rev)
            proofs.append(reconstruct(objects,bases[rev],overlay,destination=destination))
        return dict(snapshots=len(proofs),files=sum(r['files']for r in proofs),all_manifest_hashes_verified=True)
    finally:
        for base in bases.values():base.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['pack','verify','restore']);parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--runs',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--archive',type=Path);parser.add_argument('--overlay');parser.add_argument('--destination',type=Path);args=parser.parse_args()
    if args.action=='pack':pack(args.repo,args.runs or args.repo/'qualification/runs',args.output or args.repo/'qualification/runs/ds41-options-source-snapshots-20261027')
    else:
        if args.archive is None:parser.error('--archive is required')
        if args.action=='restore'and(args.overlay is None or args.destination is None):parser.error('restore needs --overlay and --destination')
        print(json.dumps(verify(args.repo,args.archive,overlay_key=args.overlay,destination=args.destination)),flush=True)

if __name__=='__main__':main()
