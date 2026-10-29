"""Exact F16 bit-pattern row reads, no MLX or checkpoint execution."""
import os
import numpy as np
import pytest
from mlx2.runtime.gguf.embedding_rows import F16EmbeddingRows, _stamp
from mlx2.runtime.gguf.reader import GGUFTensor, GGUFError


def fixture(tmp_path, width=8, vocab=17):
    bits=np.random.default_rng(72).integers(0,65536,(vocab,width),dtype=np.uint16)
    bits[0,:4]=[0,0x8000,0x7c00,0x7e01]
    path=tmp_path/'embedding.gguf'
    path.write_bytes(b'index'*16 + bits.astype('<u2').tobytes() + b'trailer')
    tensor=GGUFTensor('token_embd.weight',(width,vocab),1,16,bits.nbytes)
    return path,tensor,bits


def test_native_bits_duplicate_order_shape_and_bounded_lru(tmp_path):
    path,tensor,bits=fixture(tmp_path)
    with F16EmbeddingRows(path,tensor,data_offset=64,max_cache_bytes=32) as r:
        ids=np.array([[7,0,7],[16,2,0]])
        out=r.rows(ids)
        np.testing.assert_array_equal(out.view(np.uint16),bits[ids])
        assert r.read_bytes == 4*r.row_bytes
        assert r.cache_bytes == 32
        before=r.read_calls
        np.testing.assert_array_equal(r.rows(np.array([2])).view(np.uint16),bits[[2]])
        assert r.read_calls == before
        assert r.diagnostics()['ple'] == 'not_applicable'
    assert r.closed and r.cache_bytes == 0
    r.close()
    with pytest.raises(GGUFError,match='closed'): r.rows(np.array([0]))


def test_no_lru_still_deduplicates_request_and_preserves_empty_scalar(tmp_path):
    path,tensor,bits=fixture(tmp_path)
    with F16EmbeddingRows(path,tensor,data_offset=64,max_cache_bytes=0) as r:
        np.testing.assert_array_equal(r.rows(np.array([3,3,3])).view(np.uint16),bits[[3,3,3]])
        assert r.read_calls == 1 and r.cache_bytes == 0
        assert r.rows(np.array([],dtype=np.int32)).shape == (0,8)
        assert r.rows(np.array(3)).shape == (8,)


def test_mutation_or_replacement_closes_and_discards_cache(tmp_path):
    path,tensor,_=fixture(tmp_path)
    r=F16EmbeddingRows(path,tensor,data_offset=64)
    r.rows(np.array([0]))
    replacement=path.with_suffix('.replacement');replacement.write_bytes(path.read_bytes())
    os.replace(replacement,path)
    with pytest.raises(GGUFError,match='changed'): r.rows(np.array([0]))
    assert r.closed and r.cache_bytes == 0


def test_pread_partial_and_failure_cleanup(tmp_path,monkeypatch):
    path,tensor,bits=fixture(tmp_path)
    original=os.pread
    monkeypatch.setattr(os,'pread',lambda fd,n,off:original(fd,min(n,3),off))
    with F16EmbeddingRows(path,tensor,data_offset=64) as r:
        np.testing.assert_array_equal(r.rows(np.array([9])).view(np.uint16),bits[[9]])
        assert r.read_calls == 6
    r=F16EmbeddingRows(path,tensor,data_offset=64)
    monkeypatch.setattr(os,'pread',lambda *args:b'')
    with pytest.raises(GGUFError,match='short'): r.rows(np.array([1]))
    assert r.closed


def test_stat_geometry_ids_and_output_bounds(tmp_path):
    path,tensor,_=fixture(tmp_path)
    with pytest.raises(GGUFError,match='stat'):
        F16EmbeddingRows(path,tensor,data_offset=64,expected_stat={'size':1})
    with F16EmbeddingRows(path,tensor,data_offset=64,expected_stat=_stamp(path.stat()),max_output_bytes=32) as r:
        for ids in (np.array([-1]),np.array([17]),np.array([2**64-1],dtype=np.uint64),np.array([1.]),np.array([True])):
            with pytest.raises(GGUFError): r.rows(ids)
        with pytest.raises(GGUFError,match='bounded'): r.rows(np.array([1,2,3]))
        assert not r.closed
    with pytest.raises(GGUFError,match='past file'):
        F16EmbeddingRows(path,tensor,data_offset=10000)
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(OSError):F16EmbeddingRows(link,tensor,data_offset=64)
