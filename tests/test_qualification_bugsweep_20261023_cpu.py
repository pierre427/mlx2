"""Synthetic CPU regressions for the 2026-10-23 qualification handover."""
import pytest
import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location('bugsweep_' + Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expected_route_cannot_pass_without_any_route_receipt():
    matrix = load('scripts/model_matrix/run_matrix.py')
    cell = {'load_ok': True, 'status': 'completed', 'requests': [
        {'kind': 'code', 'sanity': {'sane': True}, 'mlx2': {}}]}
    result = matrix.verdict_for(cell, {}, 'native_mtp')
    assert result['verdict'] == 'fail'
    assert any('route receipt' in failure for failure in result['failures'])
    assert any('no speculation' in failure for failure in result['failures'])


def test_recorder_binds_only_current_cell_revision_and_evidence(tmp_path, monkeypatch, capsys):
    module = load('scripts/model_matrix/record_tsv.py')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    ledger = tmp_path / 'ledger.tsv'
    ledger.write_text('\t'.join(module.COLUMNS) + '\n')
    monkeypatch.setattr(module, 'TSV', ledger)
    for folder, revision, stamp in [('old', 'a' * 40, '2026-10-21'), ('current', 'b' * 40, '2026-10-22')]:
        path = tmp_path / 'evidence' / 'model' / folder / 'cell.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'route':'ordinary', 'checkpoint':'model@sha256:'+'c'*64,
            'adapter':'StandardDecoderAdapter', 'source_revision':revision,
            'finished_at':stamp, 'verdict':'pass', 'requests':[]}))
    assert module.main(['--evidence-root', str(tmp_path / 'evidence'), '--profile', 'synthetic CPU']) == 0
    result = next(csv.DictReader(io.StringIO('\t'.join(module.COLUMNS) + '\n' + capsys.readouterr().out), delimiter='\t'))
    assert result['source_revision'] == 'b' * 40
    assert result['evidence'] == 'evidence/model/current/cell.json'
    assert 'superseded' in result['reason']


@pytest.mark.skipif(not (ROOT / "qualification/runs/h2h-rapid-vs-mlx2-20261022/h2h_ladder.py").is_file(), reason="Private campaign harness is not distributed")
def test_h2h_route_requires_exact_selection_fingerprint_and_counts_draft_calls():
    module = load('qualification/runs/h2h-rapid-vs-mlx2-20261022/h2h_ladder.py')
    summary = {'route_selection_source':'best_route_default', 'selection_applied':True,
               'selection_fingerprint':None, 'policy':{'drafter':'dflash2'}}
    assert any('fingerprint' in problem for problem in module.check_mlx2_route(summary))
    result = module.speculation_engaged_b4('mlx2', 'plain', [], {'b1_draft_forwards':1})
    assert result['engaged'] is True
