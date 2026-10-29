"""Source-format and split-plane safety contracts; no device imports."""
import json
from pathlib import Path

import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_artifact import aliases, checked_file, read_headers, read_tensors
from mlx2.runtime.immutable_weight_pages import file_stamp
from mlx2.runtime.models.deepseek_v41_config import DeepSeekV41Config
from mlx2.runtime.models.deepseek_v41_engram import EngramRows, _read_chunk, decode_engram_rows, file_stamp as engram_stamp


def test_source_aliases_cover_required_special_projections():
    values = aliases(DeepSeekV41Config())
    assert values['blk.2.attn_compressor_gate.weight'] == 'layers.2.attn.compressor.wgate.weight'
    assert 'blk.20.attn_compressor_gate.weight' not in values
    assert values['blk.1.engram_kv.weight'] == 'layers.1.engram.wkv.weight'
    assert values['blk.39.ffn_down_shexp.weight'] == 'layers.39.ffn.shared_experts.w2.weight'
    assert len(set(values.values())) == len(values)


@pytest.mark.parametrize('compact', [False, True])
def test_split_engram_matches_interleaved_codes_and_cache(tmp_path, compact):
    rng = np.random.default_rng(41)
    weights = rng.integers(0, 126, (9, 256), dtype=np.uint8)
    scales = np.full((9, 8), 122, np.uint8)
    packed = np.concatenate((weights, scales), axis=1)
    path = tmp_path / 'split.bin'
    path.write_bytes(b'headerxx'+weights.tobytes()+b'padding!'+scales.tobytes())
    table = EngramRows(path, offset=8, scale_offset=8+weights.nbytes+8, rows=9,
        expected_stat=engram_stamp(path.stat()), workers=2, cache_format='bf16_lut' if compact else 'fp32')
    ids = np.array([[5, 2, 5, 1], [7, 1, 0, 8]], np.uint32)
    try:
        actual = table.read(ids)
        assert np.array_equal(actual, decode_engram_rows(packed)[ids])
        before = table.stats['bytes_read']
        assert np.array_equal(table.read(ids), actual)
        assert table.stats['bytes_read'] == before == 6*264
    finally:
        table.close()


def test_split_engram_rejects_overlap_and_short_extent(tmp_path):
    path = tmp_path / 'bad.bin'; path.write_bytes(bytes(600))
    for scale in (128, 590):
        with pytest.raises(ValueError):
            EngramRows(path, offset=0, scale_offset=scale, rows=2, expected_stat=engram_stamp(path.stat()))


def test_source_bounded_read_and_mutation_rejection(tmp_path):
    header = {'layer.weight': {'dtype': 'I8', 'shape': [2, 16], 'data_offsets': [0, 32]}}
    raw = json.dumps(header).encode(); path = tmp_path / 'one.safetensors'
    payload = np.arange(32, dtype=np.uint8)
    path.write_bytes(len(raw).to_bytes(8, 'little')+raw+payload.tobytes())
    seals = {path.name: file_stamp(path.stat())}
    tensors = read_headers(tmp_path, seals)
    artifact = {'root': tmp_path, 'seals': seals, 'tensors': tensors}
    actual = read_tensors(artifact, ['layer.weight'])['layer.weight']
    assert actual.tobytes() == payload.tobytes()
    # uint32 packing preserves the official low-nibble-first byte sequence.
    assert actual.view('<u4').view('u1').tobytes() == payload.tobytes()
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError, match='changed'):
        read_tensors(artifact, ['layer.weight'])


def test_source_rejects_links_and_path_escape(tmp_path):
    (tmp_path/'real').write_bytes(b'x'); (tmp_path/'link').symlink_to(tmp_path/'real')
    for name in ('link', '../real', '/tmp/real'):
        with pytest.raises(ValueError):
            checked_file(tmp_path, name)
