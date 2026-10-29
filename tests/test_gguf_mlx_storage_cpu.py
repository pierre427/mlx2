"""Lossless storage/loader contract, including packed bytes and native operands."""
import hashlib
import json
import struct

import numpy as np
import pytest

from mlx2.runtime.gguf import mlx_storage as storage
from mlx2.runtime.gguf.reader import GGUFError


def fixture(path):
    def string(s):
        b=s.encode();return struct.pack('<Q',len(b))+b
    tensors=[('q8',(32,2),8,bytes(range(68))),
             ('iq2',(256,2),16,bytes(range(132))),
             ('q2',(256,2),10,bytes(range(168))),
             ('linear',(4,3),1,np.arange(12,dtype='<f2').tobytes()),
             ('bf16',(4,3),30,(np.arange(12,dtype='<f4').view(np.uint32)>>16).astype('<u2').tobytes()),
             ('token_embd.weight',(4,8),1,np.arange(32,dtype='<f2').tobytes())]
    header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),0)
    offsets=[];end=0
    for name,shape,kind,data in tensors:
        end=(end+31)//32*32;offsets.append(end)
        header+=string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+struct.pack('<IQ',kind,end)
        end+=len(data)
    data=bytearray(end)
    for t,offset in zip(tensors,offsets):data[offset:offset+len(t[3])]=t[3]
    path.write_bytes(header+b' '*(-len(header)%32)+data)
    return tensors


def converted(tmp_path):
    source=tmp_path/'input.gguf';tensors=fixture(source);out=tmp_path/'mlx'
    storage.convert_gguf(source,out,source_sha256=storage.sha256(source),linear_names=['linear','bf16'],shard_bytes=200)
    return source,out,tensors


def test_lossless_cpu_payload_and_independent_safetensors_reader(tmp_path):
    from safetensors import safe_open
    _,out,tensors=converted(tmp_path)
    manifest,_,_=storage.inspect_storage(out,verify_payload=True)
    file=storage.MLXPackedFile(out)
    for name,shape,kind,expected in tensors:
        view=file.packed_view(name)
        assert bytes(view)==expected
        view.release();file.release_mapping()
        r=manifest['tensors'][name]
        assert r['sha256']==hashlib.sha256(expected).hexdigest()
        with safe_open(out/r['shard'],framework='numpy') as reader:
            if kind!=30:assert reader.get_tensor(name).tobytes()==expected
    for name in ['linear','bf16']:
        r=manifest['linear_fp32'][name]
        with safe_open(out/r['shard'],framework='numpy') as reader:
            assert np.array_equal(reader.get_tensor(r['key']),np.arange(12,dtype=np.float32).reshape(3,4))
    file.close()


def test_native_mlx_load_preserves_weights_operand_order_and_file_rows(tmp_path):
    import mlx.core as mx
    from mlx2.runtime.gguf.weights import GGUFWeights
    with mx.stream(mx.cpu):
        source,out,_=converted(tmp_path)
        a=GGUFWeights(source,embedding_policy='file_rows');b=GGUFWeights(out,embedding_policy='file_rows')
        try:
            x=mx.array(np.arange(8,dtype=np.float32).reshape(2,4))
            for name in ['linear','bf16']:
                left,right=a.linear(name)(x),b.linear(name)(x)
                mx.eval(left,right)
                assert np.array_equal(np.array(left),np.array(right))
                assert np.asarray(a._native_linear_fp32[name]).strides == np.asarray(b._native_linear_fp32[name]).strides
            for name in ['q8','q2','iq2']:
                assert np.array_equal(np.array(a.packed(name).data),np.array(b.packed(name).data))
            ids=mx.array([[0,7,0]],dtype=mx.uint32)
            assert np.array_equal(np.array(a.embedding(ids)),np.array(b.embedding(ids)))
            assert 'token_embd.weight' not in b._dense
        finally:a.close();b.close()


def test_mutation_symlinks_partial_and_overwrite_fail_closed(tmp_path):
    source,out,_=converted(tmp_path)
    with pytest.raises(GGUFError,match='nonempty'):
        storage.convert_gguf(source,out,source_sha256=storage.sha256(source))
    manifest=json.loads((out/storage.MANIFEST).read_text())
    p=out/next(iter(manifest['shards']))
    data=p.read_bytes();p.write_bytes(data[:-1]+bytes([data[-1]^1]))
    with pytest.raises(GGUFError,match='changed'):
        storage.inspect_storage(out)
    (out/storage.SEAL).unlink()
    with pytest.raises(GGUFError,match='regular'):
        storage.inspect_storage(out)


def test_failure_before_source_verification_never_publishes_seal(tmp_path):
    source=tmp_path/'input.gguf';fixture(source);out=tmp_path/'mlx'
    with pytest.raises(GGUFError,match='SHA256'):
        storage.convert_gguf(source,out,source_sha256='0'*64)
    assert not (out/storage.SEAL).exists()


def test_storage_inspection_is_gpu_free_and_rejects_manifest_escape(tmp_path,monkeypatch):
    import sys
    _,out,_=converted(tmp_path)
    monkeypatch.setitem(sys.modules,'mlx.core',None)
    storage.inspect_storage(out)
    manifest=json.loads((out/storage.MANIFEST).read_text())
    record=next(iter(manifest['tensors'].values()))
    record['shape'][0] += 1
    (out/storage.MANIFEST).write_text(json.dumps(manifest))
    seal=json.loads((out/storage.SEAL).read_text());seal['manifest_sha256']=storage.sha256(out/storage.MANIFEST)
    (out/storage.SEAL).write_text(json.dumps(seal))
    with pytest.raises(GGUFError,match='geometry'):
        storage.inspect_storage(out)
    with pytest.raises(GGUFError,match='basename'):
        storage.local_file(out,'../escape.safetensors')


def test_dedicated_mlx_registration_migration_is_gpu_free_and_recoverable(tmp_path,monkeypatch):
    from mlx2.adapters import deepseek_v4_gguf_artifact as original
    from mlx2.adapters import deepseek_v4_mlx_artifact as artifact
    from mlx2.adapters.registry import inspect_model
    from mlx2.adapters.deepseek_v4_mlx import DeepSeekV4MLXAdapter
    source,out,_=converted(tmp_path)
    digest=storage.sha256(source)
    monkeypatch.setattr(original,'WEIGHT_SHA256',digest)
    monkeypatch.setattr(original,'validate_header',lambda *_:None)
    (out/'tokenizer.reference.json').write_text('{}')
    config=dict(model_type='deepseek4_gguf',mlx2_weight_format=artifact.FORMAT,source_weight_sha256=digest,packed_manifest=storage.MANIFEST,tokenizer_reference='tokenizer.reference.json')
    (out/'config.json').write_text(json.dumps(config))
    (out/artifact.ARTIFACT_SEAL).write_text(json.dumps(dict(schema='mlx2.deepseek-v4-mlx-integrity.v1',assets=artifact._assets(out),tokenizer_binding={})))
    before=artifact.inspect_artifact(out)['identity']['fingerprint']
    old_assets=[(out/n).read_bytes() for n in ['config.json',artifact.ARTIFACT_SEAL]]
    saved=artifact.atomic_json
    def broken(path,value):
        if path.name==artifact.ARTIFACT_SEAL:raise OSError('seal publication failed')
        return saved(path,value)
    monkeypatch.setattr(artifact,'atomic_json',broken)
    with pytest.raises(OSError):artifact.migrate_registration(out)
    assert old_assets==[(out/n).read_bytes() for n in ['config.json',artifact.ARTIFACT_SEAL]]
    monkeypatch.setattr(artifact,'atomic_json',saved)
    after=artifact.migrate_registration(out)
    assert after['identity']['fingerprint']!=before
    resolution=inspect_model(out)
    assert resolution.adapter_type is DeepSeekV4MLXAdapter
    assert resolution.descriptor.model_type=='deepseek4_mlx'
