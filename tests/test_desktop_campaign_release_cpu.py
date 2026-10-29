"""Released own lease can coexist with a later valid queued GPU owner."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from mlx2.desktop_campaign_release import LOCKS,release_evidence,validate_release,verify_owned_release


def fixture():
    owner={'lease_id':'own-lease','cpg_task':'own-task','cpg_session':'own-session',
           'cpg_generation':1,'agent_id':'own-agent','pid':100}
    states=[{'path':str(path),'kind':'absent'}for path in LOCKS]
    pair={'owner':owner,'pid':100,'status':'command_completed','created_at':10.,'finished_at':20.,'lock_state_after':states}
    lease={**owner,'finished_at':21.,'cpg_worker':'worker',
           'release_task':{'released':True,'task_id':'own-task'},
           'complete_worker':{'status':'completed','worker_id':'worker'}}
    process={'pid':100};proof={'pid':100,'all_stopped':True,'live_members':[]}
    return pair,lease,process,copy.deepcopy(states),proof


def test_foreign_later_owner_does_not_fail_our_release():
    pair,lease,process,current,proof=fixture()
    other={'lease_id':'next-lease','cpg_task':'next-task','cpg_session':'next-session','cpg_generation':1}
    current=[{'path':str(path),'kind':'directory','owner':other}for path in LOCKS]
    assert validate_release(pair,lease,process,current,proof)['owned_release_verified']


def test_still_owned_lock_cannot_be_relabelled_released():
    pair,lease,process,current,proof=fixture()
    current[0]={'path':str(LOCKS[0]),'kind':'directory','owner':pair['owner']}
    assert not validate_release(pair,lease,process,current,proof)['owned_release_verified']


@pytest.mark.parametrize('defect',['running','release_error','task_not_released','wrong_task','worker_error','wrong_wrapper','live_process','unknown_owner','missing_lock'])
def test_incomplete_or_mismatched_release_fails_closed(defect):
    pair,lease,process,current,proof=fixture()
    if defect=='running':pair['status']='running'
    elif defect=='release_error':pair['release_error']='restore failed'
    elif defect=='task_not_released':lease['release_task']['released']=False
    elif defect=='wrong_task':lease['cpg_task']='foreign'
    elif defect=='worker_error':lease['complete_worker']={'error':'failed'}
    elif defect=='wrong_wrapper':process['pid']=200
    elif defect=='live_process':proof['live_members']=[{'pid':101}]
    elif defect=='unknown_owner':current[0]={'path':str(LOCKS[0]),'kind':'directory','owner':{}}
    elif defect=='missing_lock':current.pop()
    assert not validate_release(pair,lease,process,current,proof)['owned_release_verified']


def write_unit(tmp_path):
    pair,lease,process,current,proof=fixture()
    pair.update(attempt_id='test',immutable_receipt=str(tmp_path/'paired-locks.attempt-test.json'))
    for name,value in [('paired-locks.json',pair),('paired-locks.attempt-test.json',pair),('desktop-lease.json',lease),('process.json',process),('results.json',{})]:
        (tmp_path/name).write_text(json.dumps(value))
    return pair,current,proof


def test_immutable_terminal_receipt_must_match_mutable_alias(tmp_path):
    pair,current,proof=write_unit(tmp_path)
    assert verify_owned_release(tmp_path,current=current,process_proof=proof)['owned_release_verified']
    pair['finished_at']=22.
    (tmp_path/'paired-locks.json').write_text(json.dumps(pair))
    result=verify_owned_release(tmp_path,current=current,process_proof=proof)
    assert not result['owned_release_verified']and 'differ'in result['error']['message']


def test_bound_supplement_preserves_raw_and_rejects_evidence_change(tmp_path):
    pair,current,proof=write_unit(tmp_path)
    raw=tmp_path/'release-verified.json';raw.write_text(json.dumps({'owners_absent':False,'source_unchanged':True}))
    original=raw.read_bytes()
    report=verify_owned_release(tmp_path,current=current,process_proof=proof)
    report['original_release_sha256']=hashlib.sha256(original).hexdigest()
    (tmp_path/'release-supplement.json').write_text(json.dumps(report))
    assert release_evidence(tmp_path)['owned_release_verified']is True
    assert raw.read_bytes()==original
    (tmp_path/'results.json').write_text('{"changed":true}')
    assert release_evidence(tmp_path)['owned_release_verified']is False
