# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np

spec=importlib.util.spec_from_file_location('private_gate_diagnostics',Path(__file__).resolve().parents[1]/'scripts/research/ds41_ast_target_components.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_first_raw_bit_mismatch_and_state_layer_survive_gate_refusal(tmp_path):
    left=np.array([[1.0,2.0]],np.float32);right=left.copy();right[0,1]=np.nextafter(right[0,1],np.float32(3.0))
    mismatch=m.compare_arrays(left,right,atol=1,rtol=1)
    assert mismatch['raw_bits_equal']is False and mismatch['tolerance_valid']is True
    assert mismatch['first_raw_mismatch']['index']==[0,1]
    same=m.compare_arrays(left,left,atol=0,rtol=0)
    assert same['raw_bits_equal']is True and same['first_raw_mismatch']is None
    state=[dict(capacity_equal=True,candidate_capacity=256,ordinary_capacity=256,
        topology_equal=True,private_schema_expected=True,planes=[same]*6)for _ in range(4)]
    state[2]={**state[2],'planes':[same,same,same,mismatch,same,same]}
    controls=[dict(node=0,logits=same,features=mismatch,state=state)]
    path=tmp_path/'diagnostic.json'
    m._private_proof_diagnostic(path,N(identity='synthetic@private-gate'),
        N(transaction=N(provider_revision_token='synthetic-revision',profile='synthetic-private'),receipt={'forward_law':'fully_batched_m1_v1'}),controls)
    record=json.loads(path.read_text());first=record['first_failed_node']['first_failed_state_layer']
    assert record['passed']is False and record['proof_local_admission_granted']is False
    assert record['APCv2_insertions']==0 and first['layer']==2
    assert first['failed_planes'][0]['name']=='pooled'
    assert first['failed_planes'][0]['comparison']['first_raw_mismatch']['index']==[0,1]


def test_nonfinite_failure_remains_json_serializable():
    bad=m.compare_arrays(np.array([np.nan],np.float32),np.array([0],np.float32),atol=0,rtol=0)
    assert bad['finite']is False and bad['maximum_absolute_error']is None
    json.dumps(bad,allow_nan=False)


def test_final_publication_failure_is_saved_before_aggregate_gate(tmp_path):
    path=tmp_path/'private.json'
    mismatch=dict(capacity_equal=False,candidate_capacity=5,ordinary_capacity=256,planes=[])
    m._publication_proof_diagnostic(path,N(identity='synthetic@publication-gate'),
        N(transaction=N(provider_revision_token='synthetic-revision',profile='synthetic-private'),receipt={'forward_law':'fully_batched_m1_v1'}),
        {'private_all16_raw_exact':True,'continuation_exact':False,'isolated':True},
        partial_state=[{'capacity_equal':True}],continuation={'raw_bits_equal':True,'finite':True},
        continuation_state=[mismatch],isolated=[{'metadata_equal':True}])
    saved=json.loads((tmp_path/'private-publication.json').read_text())
    assert saved['failed_gates']==['continuation_exact']and saved['continuation_logits']['raw_bits_equal']is True
    assert saved['first_failed_state_check']['comparison']['candidate_capacity']==5
    assert saved['generic_oracle_admitted']is False and saved['APCv2_insertions']==0
