"""Guard/fixture checks for the owned attention probe, no GPU imports/dispatch."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/research'))
spec=importlib.util.spec_from_file_location('visible_probe',ROOT/'scripts/research/probe_deepseek_v4_visible_attention.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_gpu_flag_without_ownership_fails_before_fixture_or_source(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('source executed before required owner'))
    with pytest.raises(SystemExit): probe.main(['--output',str(tmp_path/'new'),'--run-gpu'])
    assert not (tmp_path/'new').exists()


def test_existing_evidence_not_overwritten(tmp_path):
    with pytest.raises(SystemExit): probe.main(['--output',str(tmp_path)])


def test_working_bound_checked_before_allocating_fixture(monkeypatch):
    monkeypatch.setattr(probe,'fixture',lambda *a,**k:pytest.fail('unbounded allocation'))
    with pytest.raises(RuntimeError,match='memory cap'):
        probe.run_case(probe.PLANS[-1],gpu=False,repetitions=1,guard=lambda:None,max_working_bytes=1024)


def test_fixture_shapes_and_row_prefixes_are_causal():
    for plan in probe.PLANS:
        if plan.get('indexer') or plan['offset']>10000: continue
        data=probe.fixture(plan)
        assert probe.fixture_bytes(data)<=probe.estimated_fixture_bytes(plan)
        for row in (0,plan['length']-1):
            one=probe.row_inputs(data,row)
            assert one['q'].shape==(1,64,1,512)
            assert one['local'].shape[2]<=128
            if plan['ratio']:
                assert one['pooled'].shape[1]==(one['offset']+1)//plan['ratio']
        out=probe.sampled_reference(data,[0,17,63],[0])
        assert out.shape==(1,3,1,512) and np.isfinite(out).all()
