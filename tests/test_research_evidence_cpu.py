"""Synthetic byte/state and sealed-array protocol checks; no device imports."""
import ast
import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.research_evidence import array_atom, diff_atoms, host_atom, load_array_archive

ROOT = Path(__file__).resolve().parents[1]


def atom(value):
    return array_atom(dtype=str(value.dtype), shape=value.shape, payload=value.tobytes())


def test_raw_bytes_distinguish_signed_zero_and_nan_payloads_without_casts():
    bits = np.array([0, 0x80000000, 0x7fc00001, 0x7fc00002], np.uint32)
    values = bits.view(np.float32)
    assert atom(values) == atom(values.copy())
    assert atom(values[:1]) != atom(values[1:2])
    assert atom(values[2:3]) != atom(values[3:])
    assert host_atom(0.) != host_atom(-0.)
    assert host_atom(True) != host_atom(1)
    assert atom(values).payload == bits.tobytes()


def test_strided_values_and_typed_diff_keep_shape_dtype_and_state_inventory():
    values = np.arange(24, dtype=np.uint32).reshape(4, 6)[:, ::2]
    assert not values.flags.c_contiguous
    assert atom(values) == atom(np.ascontiguousarray(values))
    expected = {'state.keys': atom(values), 'state.cursor': host_atom(4)}
    assert diff_atoms(expected, dict(expected)) == ()
    changed = {'state.keys': atom(values.view(np.float32)), 'state.extra': host_atom(None)}
    assert diff_atoms(expected, changed) == (
        ('state.cursor', 'missing'), ('state.extra', 'unexpected'), ('state.keys', 'dtype'))
    assert diff_atoms({'a': atom(values)}, {'a': atom(values.reshape(2, 6))}) == (('a', 'shape'),)


def test_wrapper_materializes_raw_integer_view_and_preserves_host_scalars():
    # Exercise the real wrappers without importing their MLX-heavy fixture module.
    path = ROOT/'tests/oracle_helpers.py'
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_array_atom', '_host_atom')]
    calls = []
    class Array:
        def __init__(self, bits):
            self.bits = bits
            self.dtype = SimpleNamespace(size=4)
            self.shape = bits.shape
    class Dtype:
        size = 4
        def __str__(self): return 'float32'
    value = Array(np.array([0x80000000, 0x7fc00001], np.uint32))
    value.dtype = Dtype()
    fake = SimpleNamespace(array=Array, uint8=np.uint8, uint16=np.uint16, uint32=np.uint32, uint64=np.uint64,
                           view=lambda v, dt: v.bits.view(dt), eval=lambda v: calls.append(v))
    namespace = {'mx': fake, 'np': np, 'Any': object, 'OracleAtom': type(atom(np.zeros(1, np.float32))),
                 'array_atom': array_atom, 'host_atom': host_atom}
    # Execute only the two named local wrappers against the fake backend.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102
    result = namespace['_array_atom'](value)
    assert len(calls) == 1 and result.payload == value.bits.tobytes()
    assert result.dtype == 'float32' and result.shape == (2,)
    assert namespace['_host_atom'](np.float64(-0.)) == host_atom(-0.)


def test_helper_import_has_no_numpy_or_mlx_dependency():
    code = "import sys; import mlx2.research_evidence; assert not any(k == 'mlx' or k.startswith('mlx.') or k == 'numpy' for k in sys.modules)"
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, timeout=15)


@pytest.fixture
def archive(tmp_path):
    values = np.array([0, 0x80000000, 0x7fc00001], np.uint32).view(np.float32)
    path = tmp_path/'fixture.npz'
    with path.open('xb') as stream:
        np.savez(stream, actual=values)
    return path, values


def load(path, **overrides):
    args = {'specs': {'actual': ((3,), np.float32)}, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'identity': {'checkpoint': 'synthetic@raw-bytes', 'source': 'synthetic-source'},
            'expected_identity': {'checkpoint': 'synthetic@raw-bytes', 'source': 'synthetic-source'},
            'max_archive_bytes': 4096, 'max_array_bytes': 12, 'max_container_bytes': 4096, 'backend': np}
    args.update(overrides)
    return load_array_archive(path, **args)


def test_archive_preserves_bits_and_forbids_pickle(archive, monkeypatch):
    path, values = archive
    real = np.load
    seen = []
    def checked(*args, **kwargs):
        assert kwargs == {'allow_pickle': False}
        seen.append(True)
        return real(*args, **kwargs)
    monkeypatch.setattr(np, 'load', checked)
    restored = load(path)['actual']
    assert seen == [True] and atom(restored) == atom(values)


@pytest.mark.parametrize('bad', ['identity', 'hash', 'shape', 'dtype', 'array_budget', 'file_budget', 'container_budget'])
def test_identity_hash_specs_and_budgets_fail_before_materialization(archive, monkeypatch, bad):
    path, _ = archive
    args = {'identity': {'source': 'other'}} if bad == 'identity' else {
        'hash': {'sha256': '0'*64}, 'shape': {'specs': {'actual': ((4,), np.float32)}},
        'dtype': {'specs': {'actual': ((3,), np.uint32)}},
        'array_budget': {'max_array_bytes': 11}, 'file_budget': {'max_archive_bytes': 8},
        'container_budget': {'max_container_bytes': 8}}[bad]
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('materialization reached'))
    with pytest.raises(ValueError): load(path, **args)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate', 'huge_header', 'object_header', 'truncated_payload'])
def test_all_archive_inventory_and_headers_checked_before_numpy_load(archive, monkeypatch, mutation):
    path, _ = archive
    with zipfile.ZipFile(path) as source:
        data = source.read('actual.npy')
    entries = [('actual.npy', data)]
    if mutation == 'missing': entries = []
    if mutation == 'extra': entries.append(('other.npy', data))
    if mutation == 'duplicate': entries.append(('actual.npy', data))
    if mutation in ('huge_header', 'object_header'):
        body = io.BytesIO()
        np.lib.format.write_array_header_1_0(body, {
            'descr': '|O' if mutation == 'object_header' else '<f4',
            'shape': (3,) if mutation == 'object_header' else (1 << 40,), 'fortran_order': False})
        entries = [('actual.npy', body.getvalue())]
    if mutation == 'truncated_payload': entries = [('actual.npy', data[:-1])]
    with zipfile.ZipFile(path, 'w') as target:
        for name, raw in entries: target.writestr(name, raw)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('materialization reached'))
    with pytest.raises(ValueError): load(path)


def test_archive_rejects_symlink_and_does_not_overwrite(archive):
    path, _ = archive
    link = path.with_name('link.npz')
    link.symlink_to(path)
    with pytest.raises(ValueError, match='symlink'): load(link)
    before = path.read_bytes()
    load(path)
    assert path.read_bytes() == before


def test_archive_refuses_fifo_without_waiting_for_a_writer(tmp_path):
    path = tmp_path/'fifo'
    os.mkfifo(path)
    with pytest.raises(ValueError, match='regular file'):
        load_array_archive(path, specs={'a': ((1,), np.uint32)}, sha256='0'*64,
                           identity={'source': 'synthetic'}, expected_identity={'source': 'synthetic'},
                           max_archive_bytes=4096, max_array_bytes=4, max_container_bytes=4096, backend=np)


def test_archive_replacement_with_fifo_is_nonblocking_and_checked_on_open_fd(archive, monkeypatch):
    from mlx2 import research_evidence
    path, _ = archive
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    real_open = os.open
    def replace_before_open(name, flags):
        assert flags & os.O_NONBLOCK
        path.unlink()
        os.mkfifo(path)
        return real_open(name, flags)
    monkeypatch.setattr(research_evidence.os, 'open', replace_before_open)
    with pytest.raises(ValueError, match='remain a regular file'):
        load(path, sha256=digest)


@pytest.mark.parametrize('mutation', ['encrypted', 'unsupported_compression'])
def test_zip_flags_rejected_before_decode(archive, monkeypatch, mutation):
    path, _ = archive
    raw = bytearray(path.read_bytes())
    central = raw.index(b'PK\x01\x02')
    if mutation == 'encrypted': raw[central+8:central+10] = (1).to_bytes(2, 'little')
    else: raw[central+10:central+12] = (99).to_bytes(2, 'little')
    path.write_bytes(raw)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('materialization reached'))
    with pytest.raises(ValueError, match='encrypted/unsupported'):
        load(path)


@pytest.mark.parametrize('consumer', ['ds41_native_ffn_capture', 'ds41_down_actual_capture'])
def test_sealed_parser_transitively_binds_shared_dependency_before_loading(tmp_path, monkeypatch, consumer):
    from mlx2 import research_evidence
    path = ROOT/'scripts/research'/f'{consumer}.py'
    spec = importlib.util.spec_from_file_location('_a2_'+consumer, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = Path(research_evidence.__file__)
    assert module._EVIDENCE_SHA256 == hashlib.sha256(original.read_bytes()).hexdigest()
    assert module._evidence() is research_evidence
    module._evidence.cache_clear()
    changed = tmp_path/'changed_dependency.py'
    changed.write_bytes(original.read_bytes() + b'\n# changed source\n')
    monkeypatch.setattr(research_evidence, '__file__', str(changed))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('materialization reached'))
    with pytest.raises(ValueError, match='dependency seal'):
        module._evidence()


def test_native_ffn_consumer_uses_shared_raw_diff_for_payload_and_dtype():
    path = ROOT/'scripts/research/ds41_native_ffn_capture.py'
    spec = importlib.util.spec_from_file_location('_a2_ffn_raw', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = np.array([0x80000000, 0x7fc00001], np.uint32).view(np.float32)
    assert module._same(value, value.copy())
    assert not module._same(value, value.view(np.uint32))
    changed = value.copy()
    changed.view(np.uint32)[1] = 0x7fc00002
    assert not module._same(value, changed)
