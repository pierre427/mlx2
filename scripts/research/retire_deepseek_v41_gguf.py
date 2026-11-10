"""Two-phase retirement of the one converted GGUF after standalone validation.

Explicit user authorization is required by the caller. Stage is a reversible
rename. Finalize refuses deletion without completed source/output hashes,
standalone ordinary/APCv2 evidence and released GPU ownership receipts.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from pathlib import Path
import argparse,json,subprocess,time

FILENAME='DeepSeek-V4.1-Flash-1017-Maya-XS-Abliterated.gguf'
SOURCE_SHA='f414995b3797f77009634fac995f50834e70393771f9423fca984c892503b42a'
SOURCE_BYTES=365713686528
MANIFEST='mlx2-v41-native-manifest.json'


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def stamp(path):
    s=Path(path).stat(follow_symlinks=False)
    return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def no_open_files(path):
    result=subprocess.run(['/usr/sbin/lsof','-Fpcn','--',str(path)],capture_output=True,text=True)
    if result.returncode not in (0,1):raise RuntimeError('cannot establish source descriptor quiescence')
    if result.stdout.strip():raise RuntimeError('source still has open descriptors: '+result.stdout.strip())


def check_native(path):
    from mlx2.adapters.deepseek_v41_mlx_artifact import inspect_artifact
    artifact=inspect_artifact(path)
    manifest=json.loads((Path(path)/MANIFEST).read_text())
    verify=manifest.get('verification',{})
    if (manifest.get('finished') is not True or verify.get('passed') is not True
            or verify.get('all_file_hashes_checked') is not True
            or verify.get('original_full_sha256_checked') is not True
            or verify.get('original_full_sha256')!=SOURCE_SHA
            or manifest['source'].get('gguf_sha256')!=SOURCE_SHA
            or manifest['source'].get('size')!=SOURCE_BYTES):
        raise ValueError('native artifact lacks completed conversion and full hash proof')
    if verify.get('source_tensor_count')!=1046:raise ValueError('native source inventory coverage is incomplete')
    return artifact,manifest


def validate_runtime(directory,identity):
    directory=Path(directory)
    spec=json.loads((directory/'spec.json').read_text())
    result=json.loads((directory/'results.json').read_text())
    lease=json.loads((directory/'desktop-lease.json').read_text())
    locks=json.loads((directory/'paired-locks.json').read_text())
    if (spec.get('native_mlx') is not True or spec.get('fingerprint')!=identity
            or result.get('passed') is not True or result.get('error')
            or result.get('artifact',{}).get('fingerprint')!=identity
            or result.get('ordinary_reference',{}).get('text','').strip()!='Hello.'
            or result.get('ordinary_reference',{}).get('finite') is not True
            or result.get('APCv2',{}).get('passed') is not True):
        raise ValueError('standalone native runtime and APCv2 evidence did not pass')
    snapshot=result.get('snapshot_restore',{})
    if not all(snapshot.get(k) is True for k in ['logits_bitwise_equal','state_arrays_bitwise_equal','state_metadata_equal','restored_original_frontier']):
        raise ValueError('native ordinary snapshot/replay evidence did not pass')
    if (lease.get('release_task',{}).get('released') is not True
            or lease.get('complete_worker',{}).get('status')!='completed'
            or locks.get('status')!='command_completed' or locks.get('command_returncode')!=0):
        raise ValueError('standalone GPU validation has not released ownership')
    if result.get('cleanup_memory',{}).get('active_bytes',1<<40)>1<<20:
        raise ValueError('standalone validator retained model allocations')
    return dict(directory=str(directory.resolve()),checkpoint=spec['checkpoint'],
                source_revision=spec['source_revision'],source_snapshot_sha256=spec['source_snapshot_sha256'],
                ordinary=True,snapshot_replay=True,APCv2=True,ownership_released=True)


def stage(source,native,receipt):
    source=Path(source).expanduser().absolute();native=Path(native).resolve();receipt=Path(receipt)
    if source.name!=FILENAME or source.is_symlink() or not source.is_file():raise ValueError('source is not the explicitly authorized regular GGUF')
    artifact,manifest=check_native(native)
    current=stamp(source)
    if current!=manifest['source']['stat'] or current['size']!=SOURCE_BYTES:raise ValueError('source changed since full conversion hash verification')
    no_open_files(source)
    pending=source.with_name(source.name+'.pending-native-verification')
    if pending.exists() or pending.is_symlink() or receipt.exists():raise ValueError('refusing to overwrite retirement state')
    record=dict(schema='mlx2.v41-gguf-retirement.v1',phase='prepared',source=str(source),pending=str(pending),
                native=str(native),native_fingerprint=artifact['identity']['fingerprint'],source_sha256=SOURCE_SHA,
                source_before=current,prepared_at=time.time(),conversion=manifest['verification'],
                authorization='User explicitly requested full GGUF-to-MLX conversion followed by safe GGUF deletion.')
    atomic(receipt,record);source.rename(pending);record.update(phase='staged',pending_stamp=stamp(pending),staged_at=time.time());atomic(receipt,record)
    return record


def finalize(receipt,validation):
    receipt=Path(receipt);record=json.loads(receipt.read_text())
    if record.get('phase')!='staged':raise ValueError('source has not been reversibly staged')
    source,pending=Path(record['source']),Path(record['pending'])
    artifact,manifest=check_native(record['native'])
    if artifact['identity']['fingerprint']!=record['native_fingerprint']:raise ValueError('native artifact identity changed since staging')
    if source.exists() or source.is_symlink() or pending.is_symlink() or not pending.is_file() or stamp(pending)!=record['pending_stamp']:
        raise ValueError('staged source identity changed')
    validation=validate_runtime(validation,record['native_fingerprint'])
    # The success receipt must have been generated with the original pathname
    # absent. This is checked at finalization and by the staged timestamp.
    spec=json.loads((Path(validation['directory'])/'spec.json').read_text())
    result=json.loads((Path(validation['directory'])/'results.json').read_text())
    if result.get('finished_at',0)<record['staged_at']:raise ValueError('validation predates source staging')
    if Path(spec['model']).resolve()!=Path(record['native']).resolve():raise ValueError('runtime used a different native path')
    no_open_files(pending)
    record.update(phase='deleting',runtime_validation=validation,deletion_started_at=time.time());atomic(receipt,record)
    pending.unlink()
    record.update(phase='deleted',deleted_at=time.time(),freed_logical_bytes=record['pending_stamp']['size'],
                  source_absent=not source.exists(),pending_absent=not pending.exists())
    atomic(receipt,record)
    config=source.parent/'config.json'
    retired=source.parent/'config.gguf-retired.json'
    if config.is_file() and not config.is_symlink() and not retired.exists():
        config.rename(retired)
        record['retired_source_config']=str(retired);atomic(receipt,record)
    atomic(source.parent/'GGUF-RETIRED.json',record)
    return record


def restore(receipt):
    receipt=Path(receipt);record=json.loads(receipt.read_text());source,pending=Path(record['source']),Path(record['pending'])
    if record.get('phase')!='staged' or source.exists() or pending.is_symlink() or stamp(pending)!=record['pending_stamp']:
        raise ValueError('staged source cannot be safely restored')
    no_open_files(pending);pending.rename(source)
    # Renaming may change ctime; the verified bytes/inode/mtime did not change.
    seal_path=source.parent/'mlx2-v41-integrity.json'
    if seal_path.is_file():
        seal=json.loads(seal_path.read_text());current=stamp(source)
        if seal.get('sha256')!=SOURCE_SHA:raise ValueError('source integrity identity changed')
        current.update(device=current['dev'],inode=current['ino'])
        updated={k:current[k] for k in seal['weight_stat']}
        if updated!=seal['weight_stat']:
            seal['weight_stat']=updated;atomic(seal_path,seal)
    record.update(phase='restored',restored_at=time.time(),restored_stamp=stamp(source));atomic(receipt,record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['stage','finalize','restore'])
    parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--source',type=Path);parser.add_argument('--native',type=Path);parser.add_argument('--validation',type=Path)
    args=parser.parse_args()
    if args.mode=='stage':
        if not args.source or not args.native:parser.error('stage requires source and native')
        result=stage(args.source,args.native,args.receipt)
    elif args.mode=='finalize':
        if not args.validation:parser.error('finalize requires standalone validation directory')
        result=finalize(args.receipt,args.validation)
    else:result=restore(args.receipt)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
