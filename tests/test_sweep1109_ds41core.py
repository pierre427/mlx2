# SPDX-License-Identifier: MIT
"""Bug sweep 2026-11-09, lane ds41core (DSC-1..DSC-6). CPU only, synthetic tensors/files."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlx2.runtime.immutable_weight_pages import file_stamp


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)

    def forbidden(*args, **kwargs):
        raise AssertionError('synthetic CPU test attempted Metal')
    monkeypatch.setattr(mx.fast, 'metal_kernel', forbidden)
    yield
    mx.set_default_device(previous)


def _renumbered(seal, key='device'):
    return dict(seal, **{key: seal[key] + 4})


def _touch(path, seal, key='mtime_ns'):
    os.utime(path, ns=(seal['mtime_ns'] + 10, seal['mtime_ns'] + 10))


# ------------------------------------------------------------------ DSC-1

def _packed_artifact(root):
    from mlx2.adapters.deepseek_v41_source_buffers import _PARTS, canonical_layout
    layout = canonical_layout('original')
    payload = np.random.default_rng(7).integers(0, 255, layout.nbytes, dtype=np.uint8)
    (root / 'model-00001.safetensors').write_bytes(payload.tobytes())
    tensors, offset = {}, 0
    for part, source_part in _PARTS:
        for field, source_field, dtype in (('qweight', 'weight', 'I8'), ('scales', 'scale', 'F8_E8M0')):
            t = layout.tensor(part + '.' + field)
            shape = (t.shape[0], t.shape[1] * 4) if field == 'qweight' else t.shape
            tensors[f'layers.0.ffn.experts.0.{source_part}.{source_field}'] = dict(
                file='model-00001.safetensors', offset=offset, size=t.nbytes, dtype=dtype, shape=tuple(shape))
            offset += t.nbytes
    return tensors, payload


def test_dsc1_packed_expert_read_survives_renumbered_volume_but_not_edits(tmp_path):
    from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed
    root = tmp_path.resolve()
    tensors, payload = _packed_artifact(root)
    path = root / 'model-00001.safetensors'
    live = file_stamp(path.stat())
    artifact = dict(root=root, tensors=tensors, seals={path.name: _renumbered(live)})
    host = read_original_packed(artifact, 0, 0)
    assert np.array_equal(np.sort(host._data), np.sort(payload))  # every source byte landed
    for key in ('inode', 'size'):
        with pytest.raises(ValueError):
            read_original_packed(dict(artifact, seals={path.name: _renumbered(live, key)}), 0, 0)
    _touch(path, live)
    with pytest.raises(ValueError):
        read_original_packed(dict(artifact, seals={path.name: live}), 0, 0)


def test_dsc1_omlx_read_expert_survives_renumbered_volume_but_not_edits(tmp_path):
    from mlx2.adapters import deepseek_v41_omlx_weights as ow
    path = tmp_path / 'a.safetensors'
    data = np.random.default_rng(1).integers(0, 255, 64, dtype=np.uint8)
    path.write_bytes(data.tobytes())
    live = file_stamp(path.stat())
    dtype = 'U8' if 'U8' in ow.DTYPES else next(iter(ow.DTYPES))
    extent = [('a.safetensors', 0, 64, dtype, (64 // np.dtype(ow.DTYPES[dtype]).itemsize,), 'k')]
    got = ow.read_expert(tmp_path, {'a.safetensors': _renumbered(live)}, extent)
    assert got['k'].view(np.uint8).tobytes() == data.tobytes()
    with pytest.raises(ValueError):
        ow.read_expert(tmp_path, {'a.safetensors': _renumbered(live, 'inode')}, extent)
    _touch(path, live)
    with pytest.raises(ValueError):
        ow.read_expert(tmp_path, {'a.safetensors': live}, extent)


def _safetensors(path):
    data = bytes(range(8))
    header = json.dumps({'t': {'dtype': 'U8', 'shape': [8], 'data_offsets': [0, 8]}}).encode()
    header += b' ' * (-len(header) % 8)
    raw = len(header).to_bytes(8, 'little') + header + data
    path.write_bytes(raw)
    return raw


def test_dsc1_native_reads_survive_renumbered_volume_but_not_edits(tmp_path):
    from mlx2.adapters import deepseek_v41_mlx_weights as nw
    path = tmp_path / 'a.safetensors'
    raw = _safetensors(path)
    seal = nw._stamp(path.stat())
    record = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest(), stat=_renumbered(seal, 'dev'))
    assert nw._verify_file(tmp_path, 'a.safetensors', record) == path
    out = nw.read_native_file(tmp_path, 'a.safetensors', record)
    assert np.asarray(out['t']).tobytes() == bytes(range(8))
    with pytest.raises(ValueError, match='changed'):
        nw._verify_file(tmp_path, 'a.safetensors', dict(record, stat=_renumbered(seal, 'ino')))
    _touch(path, seal)
    with pytest.raises(ValueError, match='changed'):
        nw.read_native_file(tmp_path, 'a.safetensors', dict(record, stat=seal))


def test_dsc1_gguf_v41_inspect_survives_renumbered_volume_but_not_edits(tmp_path, monkeypatch):
    from mlx2.adapters import deepseek_v41_artifact as art
    weight = tmp_path / art.FILENAME
    weight.write_bytes(b'gguf-weight-bytes')
    (tmp_path / 'config.json').write_text(json.dumps({'k': 1}))
    (tmp_path / art.REFERENCE_FILE).write_text('{"tok": 1}')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(art, '_CONFIG', {'k': 1})
    monkeypatch.setattr(art, 'WEIGHT_BYTES', weight.stat().st_size)
    monkeypatch.setattr(art, 'TOKENIZER_SHA256', digest(tmp_path / art.REFERENCE_FILE))

    class Reached(Exception):
        pass

    def gguf(_):
        raise Reached
    monkeypatch.setattr(art, 'GGUFFile', gguf)
    live = art._stamp(weight)
    seal = dict(schema=art._SCHEMA, filename=art.FILENAME, sha256=art.WEIGHT_SHA256,
                weight_stat=_renumbered(live),
                assets={n: digest(tmp_path / n) for n in ('config.json', art.REFERENCE_FILE)})
    (tmp_path / art.INTEGRITY_FILE).write_text(json.dumps(seal))
    with pytest.raises(Reached):  # the seal check passed; the GGUF header is next
        art.inspect_artifact(tmp_path)
    assert art._same_seal(seal, dict(seal, weight_stat=live))
    assert not art._same_seal(seal, dict(seal, weight_stat=_renumbered(live, 'inode')))
    _touch(weight, live)
    (tmp_path / art.INTEGRITY_FILE).write_text(json.dumps(dict(seal, weight_stat=live)))
    with pytest.raises(ValueError, match='no longer matches'):
        art.inspect_artifact(tmp_path)


def test_dsc1_gguf_subset_seal_checks_ignore_only_the_device(tmp_path):
    from mlx2.adapters.deepseek_v41_weights import DeepSeekV41Weights
    from mlx2.runtime.immutable_weight_pages import ImmutablePage, ImmutableWeightPages
    path = tmp_path / 'w.bin'
    path.write_bytes(bytes(256))
    live = file_stamp(path.stat())
    gguf_seal = {k: live[k] for k in ('size', 'mtime_ns', 'inode', 'device')}

    def pages(expected):
        return ImmutableWeightPages(path, {'p': ImmutablePage(0, 64, 64, 'raw')}, binding='b', expected_stat=expected,
                                    cache_bytes=64, materialize=lambda *a: None, evict=lambda *a: None,
                                    synchronize=lambda: None)
    pages(_renumbered(gguf_seal)).close()
    with pytest.raises(ValueError, match='differs'):
        pages(_renumbered(gguf_seal, 'inode'))
    with open(path, 'rb') as stream:
        owner = SimpleNamespace(file=SimpleNamespace(path=path, _file=stream), _source_stat=live,
                                _expected_stat=_renumbered(gguf_seal))
        DeepSeekV41Weights.verify_source(owner)
        owner._expected_stat = _renumbered(gguf_seal, 'size')
        with pytest.raises(Exception, match='changed'):
            DeepSeekV41Weights.verify_source(owner)


# Review follow-up: a stat holding only reboot-volatile keys binds nothing and is refused.

@pytest.mark.parametrize('volatile', [{'device': 1}, {'dev': 1, 'device': 2}])
def test_review_embedding_rows_refuse_a_volatile_only_expected_stat(tmp_path, volatile):
    from mlx2.runtime.gguf.embedding_rows import F16EmbeddingRows
    from mlx2.runtime.gguf.reader import GGUFError, GGUFTensor
    bits = np.random.default_rng(31).integers(0, 65536, (5, 4), dtype=np.uint16)
    path = tmp_path / 'emb.gguf'
    path.write_bytes(bytes(64) + bits.astype('<u2').tobytes())
    tensor = GGUFTensor('token_embd.weight', (4, 5), 1, 0, bits.nbytes)
    live = file_stamp(path.stat())
    F16EmbeddingRows(path, tensor, data_offset=64, expected_stat=_renumbered(live)).close()
    with pytest.raises(GGUFError, match='differs'):
        F16EmbeddingRows(path, tensor, data_offset=64, expected_stat=volatile)


def test_review_immutable_pages_refuse_a_volatile_only_expected_stat(tmp_path):
    from mlx2.runtime.immutable_weight_pages import ImmutablePage, ImmutableWeightPages
    path = tmp_path / 'w.bin'
    path.write_bytes(bytes(256))

    def pages(expected):
        return ImmutableWeightPages(path, {'p': ImmutablePage(0, 64, 64, 'raw')}, binding='b', expected_stat=expected,
                                    cache_bytes=64, materialize=lambda *a: None, evict=lambda *a: None,
                                    synchronize=lambda: None)
    pages(_renumbered(file_stamp(path.stat()))).close()
    with pytest.raises(ValueError, match='differs'):
        pages({'device': 7})


def test_review_v41_verify_source_refuses_a_volatile_only_expected_stat(tmp_path):
    from mlx2.adapters.deepseek_v41_weights import DeepSeekV41Weights
    path = tmp_path / 'w.bin'
    path.write_bytes(bytes(128))
    live = file_stamp(path.stat())
    with open(path, 'rb') as stream:
        owner = SimpleNamespace(file=SimpleNamespace(path=path, _file=stream), _source_stat=live,
                                _expected_stat=_renumbered(live))
        DeepSeekV41Weights.verify_source(owner)
        owner._expected_stat = {'device': live['device'] + 4}
        with pytest.raises(Exception, match='changed'):
            DeepSeekV41Weights.verify_source(owner)
        owner._expected_stat = None  # no admitted stat: only the live bindings are compared
        DeepSeekV41Weights.verify_source(owner)


def test_dsc1_no_strict_seal_compare_left_in_v41_adapters():
    root = Path(__file__).resolve().parents[1] / 'src/mlx2/adapters'
    strict = re.compile(r"(file_stamp|_stamp)\([^()]*(\([^()]*\))?[^()]*\)\s*!=\s*(seal\b|_seal\(|seal\.get\()")
    hits = [f'{p.name}:{n}' for p in sorted(root.glob('deepseek_v41*.py'))
            for n, line in enumerate(p.read_text().splitlines(), 1) if strict.search(line)]
    assert hits == []


# ------------------------------------------------------------------ DSC-2

@pytest.mark.parametrize('total,step', [(22, 14), (21, 13), (21, 14)])
def test_dsc2_wave_with_short_tail_and_rows_after_matches_dependency_ced(total, step):
    import test_deepseek_v41_source_layer_major_ced_cpu as T
    c = T._config()
    wave, _ = T._pair(c, chunk=4)
    plain, _ = T._pair(c, chunk=4, state_only=False)
    ids = T._ids(c, total)
    wc, pc = wave.make_cache(), plain.make_cache()
    actual = T._prompt(wave, wc, ids, step)
    expected = T._prompt(plain, pc, ids, step)
    np.testing.assert_array_equal(T._bits(actual[-1]), T._bits(expected[-1]))
    T._assert_states_equal(wc, pc)
    assert not any(s.pending for s in wc)
    assert wave.scheduled_kv_calls.get('source_layer_major_ced_state_only_chunks', 0) >= 2
    token = mx.array([[1]], mx.int32)
    np.testing.assert_array_equal(T._bits(wave(token, wc)), T._bits(plain(token, pc)))


# ------------------------------------------------------------------ DSC-3

@pytest.mark.parametrize('tier', ['remote_layer', 'ram'])
def test_dsc3_undeclarable_tiers_refuse_at_parse_time(tier):
    from mlx2.adapters.deepseek_v41_source_lfu import parse_policy
    base = dict(mode='original_lfu', budget_bytes=40_000_000_000, storage_tiers='accounting_v1')
    with pytest.raises(ValueError, match='internal_nvme or external'):
        parse_policy(dict(base, tier_declared={'model-00001.safetensors': tier}))
    for ok in ('internal_nvme', 'external'):
        assert parse_policy(dict(base, tier_declared={'model-00001.safetensors': ok}))['tier_declared']


# ------------------------------------------------------------------ DSC-4

def test_dsc4_key_chunked_scores_are_close_but_not_claimed_bitwise():
    from mlx2.runtime.models import deepseek_v41_index_select as sel
    rng = np.random.default_rng(3)
    heads, dim, rows, count, chunk = 4, 8, 3, 39, 21
    q = mx.array(rng.standard_normal((1, heads, rows, dim)).astype(np.float32))
    w = mx.array(rng.standard_normal((1, rows, heads)).astype(np.float32))
    keys = mx.array(rng.standard_normal((1, count, dim)).astype(np.float32))
    ref = mx.sum(mx.maximum(mx.einsum('bhsd,btd->bsht', q, keys), 0) * w[:, :, :, None], 2)
    got = sel.reduced_scores_chunked(q, w, lambda a, b: keys[:, a:b], count, chunk)
    # Same expression per element; the split GEMM may round the last ulp differently.
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=0, atol=2e-6)
    doc = sel.__doc__
    assert 'On CPU the\n    values are bitwise equal' not in doc and 'not a guaranteed bitwise rewrite' in doc


# ------------------------------------------------------------------ DSC-5

def test_dsc5_resident_f32_cache_is_counted_and_cleared_before_park_receipt(monkeypatch):
    from mlx2.adapters import deepseek_v41_mlx_weights as nw
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    name = 'blk.7.ffn_gate_inp.weight'
    weight = mx.array(np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)).astype(mx.bfloat16)
    w = SourceV41Weights.__new__(SourceV41Weights)
    w.source_bf16_loads_policy, w.source_bf16_dense_policy = 'direct_f32', 'resident_f32_v1'
    w._quant, w._dense, w._pending_dense, w.scheduled_dense_calls = {}, {name: weight}, {}, {}
    w._ensure_dense = w._ready = lambda: None
    w.bounds = dict(dense_payload_bytes=100, dense_prepared_bound_bytes=0, dense_staging_bound_bytes=0)
    w.artifact = dict(aliases={name: 'layers.7.ffn.gate.weight', 'output.weight': 'head.weight'},
                      tensors={'layers.7.ffn.gate.weight': dict(shape=[32, 64], dtype='BF16'),
                               'head.weight': dict(shape=[8, 64], dtype='BF16')})
    base = w.resident_bytes()
    assert base == weight.nbytes
    assert w.resume_reservation_bytes() == 100 + 32 * 64 * 4
    with mx.stream(mx.cpu):
        mx.eval(w.linear(name)(mx.ones((1, 1, 64), mx.float32)))
    assert w.resident_bytes() == base + 32 * 64 * 4
    seen = {}
    monkeypatch.setattr(nw.NativeV41Weights, 'park', lambda self: seen.setdefault('resident', self.resident_bytes()))
    w.park()
    assert seen['resident'] == base and not w._dense_f32


# ------------------------------------------------------------------ DSC-6

def test_dsc6_key_chunked_does_not_report_unused_decode_once_residency(monkeypatch):
    from test_deepseek_v41_index_select_cpu import HEADS, _selector
    from mlx2.runtime.models import deepseek_v41 as v41
    from mlx2.runtime.models import deepseek_v41_index_select as sel
    monkeypatch.setattr(v41, 'INDEX_SCORE_SCRATCH_BYTES', HEADS * 4 * 64)
    monkeypatch.setattr(sel, 'INDEX_REDUCED_SCORE_BYTES', 4 * 4096)
    monkeypatch.setattr(v41, 'PACKED_INDEX_DECODE_ONCE', True)
    for tiling in ('key_chunked_v1', 'reference'):
        a, source, shared, x, qr, start = _selector(1000, layer=20, ratio=1, length=70, tiling=tiling, packed=True)
        a.select(x, qr, source, start, shared)
        counters = a.model.scheduled_kv_calls
        if tiling == 'key_chunked_v1':
            assert counters['index_key_chunked_tiles'] > 1
            assert 'packed_index_resident_max_rows' not in counters
            assert counters.get('packed_index_reused_chunks', 0) == 0
        else:
            assert counters['packed_index_resident_max_rows'] == 1000
            assert counters['packed_index_reused_chunks'] > 0
