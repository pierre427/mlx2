"""Synthetic physical prefix/feature checkpoint source binding, no model GPU."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import mlx.core as mx
import numpy as np
import pytest
from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache

spec=importlib.util.spec_from_file_location('private_ast_prefix',Path(__file__).parents[1]/'scripts/research/ds41_ast_private_prefix.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def make():
    return DeepSeekV41LayerCache(0,xp=mx,identity='synthetic-private-prefix@42',head_dim=2,index_head_dim=1,window_size=4,source=False,history=False,compressed_vocab=0)


def test_physical_prefix_and_feature_shards_roundtrip_and_source_tamper_refuse(tmp_path):
    mx.set_default_device(mx.cpu);state=make();state.offset=4;state.keys=mx.arange(8,dtype=mx.float32).reshape(1,1,4,2)
    logits=mx.zeros((1,1,5),mx.float32);features=mx.arange(24,dtype=mx.float32).reshape(1,4,6)
    binding=dict(prefix_tokens=4,feature_width=6,vocab_size=5,source_snapshot_sha256='synthetic-source-A',prefill_chunk_size=2)
    partition=dict(profile='ordinary_body_chunks_then_B1_final',seconds=0.,spans=[dict(start=0,stop=3),dict(start=3,stop=4)])
    root=tmp_path/'prefix';receipt=probe.save_prefix(root,mx,[state],logits,features,binding,partition,lambda:True)
    model=SimpleNamespace(make_cache=lambda:[make()],args=SimpleNamespace(vocab_size=5))
    restored,a,b,profile=probe.load_prefix(root,mx,model,binding,lambda:True,expected_manifest_sha256=receipt['manifest_sha256'])
    assert restored[0].meta_state==state.meta_state and np.array_equal(np.asarray(restored[0].keys),np.asarray(state.keys))
    assert np.array_equal(np.asarray(a),np.asarray(logits))and np.array_equal(np.asarray(b),np.asarray(features))and profile==partition
    with pytest.raises(FileExistsError):probe.save_prefix(root,mx,[state],logits,features,binding,partition,lambda:True)
    with pytest.raises(ValueError,match='binding differs'):
        probe.load_prefix(root,mx,model,{**binding,'source_snapshot_sha256':'synthetic-source-B'},lambda:True,expected_manifest_sha256=receipt['manifest_sha256'])
    shard=root/'features.safetensors';data=shard.read_bytes();shard.write_bytes(data[:-1]+bytes([data[-1]^1]))
    with pytest.raises(ValueError,match='feature shard SHA'):
        probe.load_prefix(root,mx,model,binding,lambda:True,expected_manifest_sha256=receipt['manifest_sha256'])
