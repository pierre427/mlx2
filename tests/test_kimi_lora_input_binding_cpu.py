"""Consumed-byte provenance checks; stop before ownership or model loading."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts/research'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('kimi_lora_input_binding_runner', SCRIPTS / 'kimi_lora_round.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    payload = (json.dumps(value) + '\n').encode()
    path.write_bytes(payload)
    return payload


@pytest.mark.parametrize('separator', [b'\n', b'\r\n', b'\r'])
def test_raw_hash_preserves_legacy_jsonl_iteration(tmp_path, separator):
    path = tmp_path / 'rows.jsonl'
    payload = separator.join([b'  ', '{"text":"a\u2028b\u2029c"}'.encode(), b'', b'{"n":2}'])
    path.write_bytes(payload)
    expected = [{'text': 'a\u2028b\u2029c'}, {'n': 2}]
    assert runner.read_rows_bound(path) == (expected, digest(payload))
    assert runner.read_rows(path) == expected


def test_single_snapshot_binds_rows_when_path_is_replaced(tmp_path, monkeypatch):
    path = tmp_path / 'rows.jsonl'
    old, new = b'{"n":1}\n', b'{"n":2}\n'
    path.write_bytes(old)
    real_read, calls = Path.read_bytes, []
    def replace_after_read(target):
        payload = real_read(target)
        if target == path:
            calls.append(target)
            target.write_bytes(new)
        return payload
    monkeypatch.setattr(Path, 'read_bytes', replace_after_read)
    assert runner.read_rows_bound(path, expected_sha256=digest(old)) == ([{'n': 1}], digest(old))
    assert calls == [path]
    assert real_read(path) == new


def test_declared_hash_rejected_before_json_parse(tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_bytes(b'not JSON')
    with pytest.raises(ValueError, match='declared input SHA256 mismatch'):
        runner.read_rows_bound(path, expected_sha256='0' * 64)


def test_uppercase_declared_hash_is_normalized(tmp_path):
    path = tmp_path / 'rows.jsonl'
    payload = b'{"n":1}\n'
    path.write_bytes(payload)
    assert runner.read_rows_bound(path, expected_sha256=digest(payload).upper())[1] == digest(payload)


@pytest.mark.parametrize('value', [None, True, 17, '', 'a' * 63, 'a' * 65, 'g' * 64, ' ' * 64])
def test_invalid_declared_hash_rejected(value):
    with pytest.raises(ValueError, match='64 hexadecimal'):
        runner.declared_sha256(value)


def test_selection_receipt_is_opaque_but_hash_bound(tmp_path):
    path = tmp_path / 'selection.receipt'
    payload = b'opaque evidence, no generic JSON schema imposed\r\n'
    path.write_bytes(payload)
    assert runner.data_selection_binding({}) is None
    spec = {'data_selection_receipt': str(path), 'data_selection_receipt_sha256': digest(payload)}
    assert runner.data_selection_binding(spec) == {'path': str(path.resolve()), 'sha256': digest(payload)}
    with pytest.raises(ValueError, match='mismatch'):
        runner.data_selection_binding({**spec, 'data_selection_receipt_sha256': '0' * 64})


@pytest.mark.parametrize('fields', [
    {'data_selection_receipt': 'receipt.json'},
    {'data_selection_receipt_sha256': 'a' * 64},
    {'data_selection_receipt': None, 'data_selection_receipt_sha256': 'a' * 64},
    {'data_selection_receipt': '', 'data_selection_receipt_sha256': 'a' * 64},
    {'data_selection_receipt': 'receipt.json', 'data_selection_receipt_sha256': None},
])
def test_incomplete_or_invalid_selection_pair_rejected(fields):
    with pytest.raises(ValueError):
        runner.data_selection_binding(fields)


def inputs(tmp_path):
    train, cases, loss, selection = [tmp_path / name for name in ('train.jsonl', 'cases.jsonl', 'loss.jsonl', 'selection.json')]
    payloads = {
        train: write_json(train, {'id': 'train-1', 'split': 'train', 'input_ids': [1, 2], 'loss_mask': [0, 1]}),
        cases: write_json(cases, {'id': 'dev-1', 'split': 'dev', 'evaluation_only': True}),
        loss: write_json(loss, {'id': 'loss-1', 'split': 'validation', 'evaluation_only': True, 'never_train': True,
                              'input_ids': [1, 2], 'loss_mask': [0, 1]}),
        selection: write_json(selection, {'source_selection': 'fixture'}),
    }
    spec = {'model': str(tmp_path / 'copy'), 'original_model': str(tmp_path / 'original'),
            'out': str(tmp_path / 'round'), 'mode': 'train', 'source_revision': 'fixture-source',
            'data': str(train), 'eval_cases': str(cases), 'loss_cases': str(loss),
            'data_expected_sha256': digest(payloads[train]), 'data_selection_receipt': str(selection),
            'data_selection_receipt_sha256': digest(payloads[selection])}
    spec_path = tmp_path / 'spec.json'
    payloads[spec_path] = write_json(spec_path, spec)
    return spec_path, spec, payloads


def block_runtime_imports(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'mlx' or name.startswith('mlx.') or name.startswith('mlx2.adapters'):
            raise AssertionError('unexpected model/runtime import: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


@pytest.mark.parametrize('bad_field', ['data_expected_sha256', 'data_selection_receipt_sha256'])
def test_run_rejects_declared_mismatch_before_model_load(tmp_path, monkeypatch, bad_field):
    spec_path, spec, _ = inputs(tmp_path)
    write_json(spec_path, {**spec, bad_field: '0' * 64})
    block_runtime_imports(monkeypatch)
    with pytest.raises(ValueError, match='declared input SHA256 mismatch'):
        runner.run(spec_path)
    assert not (Path(spec['out']) / 'results').exists()


@pytest.mark.parametrize('mode', ['train', 'evaluate'])
def test_declared_data_hash_is_checked_in_every_mode(tmp_path, monkeypatch, mode):
    spec_path, spec, _ = inputs(tmp_path)
    write_json(spec_path, {**spec, 'mode': mode, 'data_expected_sha256': '0' * 64})
    block_runtime_imports(monkeypatch)
    with pytest.raises(ValueError, match='declared input SHA256 mismatch'):
        runner.run(spec_path)


def test_declared_data_hash_requires_valid_value_and_path(tmp_path):
    _, spec, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match='64 hexadecimal'):
        runner.validate_spec({**spec, 'data_expected_sha256': None})
    with pytest.raises(ValueError, match='requires a data path'):
        runner.validate_spec({**spec, 'mode': 'evaluate', 'data': None})


@pytest.mark.parametrize('declare_hashes', [False, True])
def test_run_receipt_binds_consumed_snapshots_without_loading_model(tmp_path, monkeypatch, declare_hashes):
    spec_path, spec, payloads = inputs(tmp_path)
    if not declare_hashes:
        for key in ('data_expected_sha256', 'data_selection_receipt', 'data_selection_receipt_sha256'):
            spec.pop(key)
        payloads[spec_path] = write_json(spec_path, spec)
    real_read, reads = Path.read_bytes, {}
    def replace_after_read(path):
        payload = real_read(path)
        if path in payloads:
            reads[path] = reads.get(path, 0) + 1
            path.write_bytes(b'changed after original bytes were read\n')
        return payload
    monkeypatch.setattr(Path, 'read_bytes', replace_after_read)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: 'fixture-source\n')
    monkeypatch.setattr(runner, 'host_state', lambda: {'fixture': 'CPU only'})
    ownership = ModuleType('mlx2.desktop_campaign_worker')
    def stop_before_runtime(_):
        raise RuntimeError('CPU sentinel before model loading')
    ownership.check_ownership = stop_before_runtime
    monkeypatch.setitem(sys.modules, 'mlx2.desktop_campaign_worker', ownership)
    block_runtime_imports(monkeypatch)
    with pytest.raises(RuntimeError, match='CPU sentinel'):
        runner.run(spec_path)
    receipt = json.loads((Path(spec['out']) / 'results/receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['spec'] == spec
    assert receipt['spec_sha256'] == digest(payloads[spec_path])
    for field, path_key in [('data_sha256', 'data'), ('eval_cases_sha256', 'eval_cases'), ('loss_cases_sha256', 'loss_cases')]:
        path = Path(spec[path_key])
        assert receipt[field] == digest(payloads[path])
        assert reads[path] == 1 and real_read(path) != payloads[path]
    assert reads[spec_path] == 1
    if declare_hashes:
        path = Path(spec['data_selection_receipt'])
        assert receipt['data_selection_receipt']['sha256'] == digest(payloads[path]) and reads[path] == 1
    else:
        assert receipt['data_selection_receipt'] is None
