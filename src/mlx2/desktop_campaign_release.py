# SPDX-License-Identifier: MIT
"""Verify release of this campaign's ownership, independent of later waiters.

Read-only CPU evidence checks. A foreign lock holder is never signalled, removed,
joined or treated as this campaign's process. Immutable paired terminal receipts
and CPG release identities remain mandatory for the stronger release verdict.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

LOCKS=(Path('/Users/Shared/mlxuag/gpu.lock'),Path('/tmp/gpu.lock'))


def _number(value):return type(value)in(int,float)and math.isfinite(value)
def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _read(path):return json.loads(Path(path).read_text())


def lock_snapshot(paths=LOCKS):
    rows=[]
    for path in paths:
        path=Path(path)
        try:
            mode=path.lstat()
            if stat.S_ISREG(mode.st_mode):row={'kind':'file','size':mode.st_size}
            elif stat.S_ISDIR(mode.st_mode):
                with (path/'owner.json').open('rb')as stream:raw=stream.read((1<<20)+1)
                row={'kind':'directory','owner':json.loads(raw)if len(raw)<=1<<20 else None}
            else:row={'kind':'unknown'}
        except FileNotFoundError:
            row={'kind':'unknown'}if path.exists()else{'kind':'absent'}
        except (OSError,ValueError):row={'kind':'unknown'}
        rows.append({'path':str(path),**row})
    return rows


def _not_owned(state,owner):
    if state.get('kind')=='absent':return True
    if state.get('kind')=='file':return state.get('size')==0
    foreign=state.get('owner')
    if type(foreign)is str:
        try:foreign=json.loads(foreign)
        except ValueError:return False
    if state.get('kind')!='directory'or type(foreign)is not dict:return False
    # A malformed or partly written owner is unknown, never proof of release.
    for key in ('lease_id','cpg_task','cpg_session'):
        if type(foreign.get(key))is not str or not foreign[key]:return False
    if type(foreign.get('cpg_generation'))is not int or foreign['cpg_generation']<1:return False
    return foreign['lease_id']!=owner['lease_id']and foreign['cpg_task']!=owner['cpg_task']


def validate_release(pair,lease,process,current,process_proof):
    """Strict data contract; callers bind the pair to its immutable receipt."""
    owner=pair.get('owner',{});identity=('cpg_task','cpg_session','cpg_generation','agent_id')
    named=all(type(owner.get(key))is str and bool(owner[key])for key in ('cpg_task','cpg_session','agent_id','lease_id'))and type(owner.get('cpg_generation'))is int and owner['cpg_generation']>=1
    same=named and all(owner.get(key)==lease.get(key)for key in identity)
    release=lease.get('release_task',{});worker=lease.get('complete_worker',{})
    expected={str(path)for path in LOCKS}
    after=pair.get('lock_state_after',[])
    terminal=(pair.get('status')in('command_completed','command_failed','interrupted','error')
        and not pair.get('release_error')and _number(pair.get('finished_at'))
        and _number(pair.get('created_at'))and pair['finished_at']>=pair['created_at'])
    gates=dict(terminal_pair=terminal,CPG_identity=same,
        wrapper_identity=type(process.get('pid'))is int and process.get('pid')==pair.get('pid')==owner.get('pid'),
        CPG_task_released=release.get('released')is True and release.get('task_id')==owner.get('cpg_task')and _number(lease.get('finished_at')),
        CPG_worker_completed=worker.get('status')in('completed','failed')and worker.get('worker_id')==lease.get('cpg_worker')and not worker.get('error'),
        paired_release_states=len(after)==2 and {row.get('path')for row in after}==expected and named and all(_not_owned(row,owner)for row in after),
        current_locks_not_owned=len(current)==2 and {row.get('path')for row in current}==expected and named and all(_not_owned(row,owner)for row in current),
        owned_processes_stopped=type(process_proof)is dict and process_proof.get('pid')==process.get('pid')
            and process_proof.get('all_stopped')is True and process_proof.get('live_members')==[])
    return dict(owned_release_verified=all(gates.values()),release_gates=gates,
        owned_lease_id=owner.get('lease_id'),current_locks=current,owned_process_proof=process_proof)


def _historical_process_proof(unit,pair):
    """Combine original worker group-cleanup receipts with current absence.

    Reused PIDs born after the terminal receipt belong to a different process.
    An unreadable identity or any surviving original group fails closed.
    """
    import psutil
    result=_read(unit/'results.json');cleanup=result.get('cleanup',[])
    server=result.get('server_pid')
    groups={pair.get('pid'),pair.get('child_pid')}
    valid=bool(cleanup)and all(row.get('process_group_absent')is True and type(row.get('process_group'))is int for row in cleanup)
    groups.update(row.get('process_group')for row in cleanup)
    if server is not None:valid=valid and server in groups
    valid=valid and all(type(pid)is int and pid>1 for pid in groups)
    live=[];finished=pair.get('finished_at')
    if not _number(finished):valid=False
    if valid:
        try:
            for member in psutil.process_iter(['pid','create_time','status']):
                try:
                    info=member.info
                    if info['status']==psutil.STATUS_ZOMBIE or info['create_time']>finished:continue
                    if info['pid']in groups or os.getpgid(info['pid'])in groups:
                        live.append({'pid':info['pid'],'created':info['create_time']})
                except (psutil.NoSuchProcess,ProcessLookupError):continue
                except (psutil.AccessDenied,PermissionError):valid=False
        except (OSError,psutil.Error):valid=False
    return dict(pid=pair.get('pid'),all_stopped=valid and not live,live_members=live,
        origin='original worker process-group cleanup plus current original-group absence',groups=sorted(pid for pid in groups if type(pid)is int))


def verify_owned_release(unit,*,process_proof=None,current=None):
    """Verify exact immutable receipt bytes and this unit's released resources."""
    unit=Path(unit).resolve();report=dict(schema='mlx2.owned-campaign-release.v1',owned_release_verified=False,finished_at=time.time())
    try:
        pair=_read(unit/'paired-locks.json')
        attempt=pair.get('attempt_id');immutable=unit/f'paired-locks.attempt-{attempt}.json'
        if type(attempt)is not str or not attempt or '/'in attempt or '\\'in attempt:raise ValueError('invalid paired attempt identity')
        if Path(pair.get('immutable_receipt','')).resolve()!=immutable.resolve():raise ValueError('immutable receipt path mismatch')
        if _read(immutable)!=pair:raise ValueError('mutable and immutable paired receipts differ')
        process=_read(unit/'process.json');lease=_read(unit/'desktop-lease.json')
        if process_proof is None:process_proof=_historical_process_proof(unit,pair)
        verdict=validate_release(pair,lease,process,lock_snapshot()if current is None else current,process_proof)
        report.update(verdict,immutable_pair_path=str(immutable),immutable_pair_sha256=_hash(immutable),
            paired_receipt_sha256=_hash(unit/'paired-locks.json'),desktop_lease_sha256=_hash(unit/'desktop-lease.json'),
            process_receipt_sha256=_hash(unit/'process.json'),
            evidence_sha256={name:_hash(unit/name)for name in ('paired-locks.json','desktop-lease.json','process.json','results.json')if(unit/name).is_file()})
    except (OSError,ValueError,TypeError,KeyError)as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
    return report


def release_evidence(unit):
    """Read original evidence and an optional hash-bound forensic supplement."""
    unit=Path(unit).resolve();path=unit/'release-verified.json';release=_read(path)
    supplement=unit/'release-supplement.json'
    if supplement.exists():
        value=_read(supplement)
        # A stronger false/invalid verdict must never fall back to old absence.
        release['owned_release_verified']=False
        pair_path=Path(value.get('immutable_pair_path',''))
        bound=value.get('evidence_sha256',{})
        evidence_valid=type(bound)is dict and set(bound)=={'paired-locks.json','desktop-lease.json','process.json','results.json'}and all((unit/name).is_file()and _hash(unit/name)==digest for name,digest in bound.items())
        if (value.get('original_release_sha256')==_hash(path)and evidence_valid
                and pair_path.parent.resolve()==unit.resolve()
                and pair_path.is_file()and value.get('immutable_pair_sha256')==_hash(pair_path)):
            verdict=validate_release(_read(pair_path),_read(unit/'desktop-lease.json'),_read(unit/'process.json'),value.get('current_locks',[]),value.get('owned_process_proof'))
            release['owned_release_verified']=value.get('owned_release_verified')is True and verdict['owned_release_verified']is True
            release['release_supplement']=str(supplement)
    return release
