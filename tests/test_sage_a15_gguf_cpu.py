"""CPU conversion regressions; synthetic tensors, no model serving claims."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

gguf = pytest.importorskip('gguf')
safe_open = pytest.importorskip('safetensors').safe_open
spec = importlib.util.spec_from_file_location('sage_converter', Path(__file__).resolve().parents[1] / 'scripts/convert_sage_a15_gguf.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def read(path, key):
    with safe_open(path, framework='numpy') as f:
        return f.get_tensor(key)


def tensor(data, name='fixture'):
    return SimpleNamespace(name=name, data=np.asarray(data, dtype=np.float32), tensor_type=gguf.GGMLQuantizationType.F32)


def test_moe_mapping_and_shared_gate_shape():
    assert c.map_tensor('blk.0.ffn_down_exps.weight', (256, 2048, 512))[:2] == ('language_model.model.layers.0.mlp.switch_mlp.down_proj.weight', (256, 2048, 512))
    assert c.map_tensor('blk.40.ffn_gate_exps.weight', (256, 512, 2048))[0] == 'language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight'
    assert c.map_tensor('blk.0.ffn_gate_inp_shexp.weight', (2048,))[1:] == ((1, 2048), 'shared_gate')
    assert c.map_tensor('blk.40.nextn.eh_proj.weight', (2048, 4096))[0] == 'language_model.mtp.fc.weight'
    for name, shape in [('blk.41.attn_norm.weight', (2048,)), ('blk.3.ssm_a', (32,)), ('blk.0.ffn_down_exps.weight', (256, 512, 2048))]:
        with pytest.raises(ValueError):
            c.map_tensor(name, shape)


def test_expert_axis_survives_bounded_streaming(tmp_path):
    x=np.arange(3*4*8, dtype=np.float32).reshape(3,4,8)
    p=tmp_path/'experts.safetensors'
    c.write_tensor(tensor(x), p, 'w', x.shape, 'identity', rows_per_chunk=3)
    np.testing.assert_array_equal(read(p,'w'),x)


def test_shared_gate_restores_linear_dimension(tmp_path):
    x=np.arange(2048,dtype=np.float32)
    p=tmp_path/'gate.safetensors'
    c.write_tensor(tensor(x),p,'w',(1,2048),'shared_gate')
    np.testing.assert_array_equal(read(p,'w'),x[None])


def test_inverse_gdn_permutation_and_a_log(tmp_path):
    grouped=np.arange(32,dtype=np.float32)/32
    # Forward grouped -> tiled is reshape(key_head, value_per_key).T.
    tiled=grouped.reshape(16,2).T.reshape(-1)
    p=tmp_path/'a.safetensors'
    c.write_tensor(tensor(-np.exp(tiled)),p,'w',(32,),'a_log',rows_per_chunk=7)
    np.testing.assert_allclose(read(p,'w'),grouped,atol=1e-7)
    assert sorted(c.source_row(i,'v_rows_1') for i in range(32)) == list(range(32))
    assert c.source_row(4095,'qkv_rows') == 4095
    assert c.source_row(4096+128,'qkv_rows') == 4096+16*128
    assert sorted(c.source_columns(4096).tolist()) == list(range(4096))


def test_conv_layout_preserves_shifted_norm_convention(tmp_path):
    x=np.arange(8192*4,dtype=np.float32).reshape(8192,4)
    p=tmp_path/'conv.safetensors'
    c.write_tensor(tensor(x),p,'w',(8192,4,1),'conv_rows')
    expected=x[[c.source_row(i,'conv_rows') for i in range(8192)],:,None]
    np.testing.assert_array_equal(read(p,'w'),expected)
    p=tmp_path/'norm.safetensors'
    c.write_tensor(tensor([1.25,0.75]),p,'w',(2,),'identity')
    np.testing.assert_array_equal(read(p,'w'),[1.25,0.75])


def test_nonfinite_and_invalid_a_do_not_publish(tmp_path):
    for data,op in [([np.inf],'identity'),([1.0],'a_log')]:
        p=tmp_path/'bad.safetensors'
        with pytest.raises(ValueError):
            c.write_tensor(tensor(data),p,'w',(1,),op)
        assert not p.exists()
        assert not p.with_suffix('.safetensors.part').exists()


@pytest.mark.parametrize('kind', ['Q4_K', 'Q6_K'])
def test_quantized_experts_match_direct_dequantization(tmp_path, kind):
    import struct
    qtype=getattr(gguf.GGMLQuantizationType,kind)
    block_bytes=gguf.GGML_QUANT_SIZES[qtype][1]
    encoded=np.random.default_rng(17).integers(0,256,size=(2,3,block_bytes),dtype=np.uint8)
    if kind=='Q4_K':
        encoded[...,:2]=np.frombuffer(np.float16(0.5).tobytes(),dtype=np.uint8)
        encoded[...,2:4]=np.frombuffer(np.float16(0.25).tobytes(),dtype=np.uint8)
    else:
        encoded[...,-2:]=np.frombuffer(np.float16(0.25).tobytes(),dtype=np.uint8)
    t=SimpleNamespace(name='experts',data=encoded,tensor_type=qtype)
    p=tmp_path/'quant.safetensors'
    c.write_tensor(t,p,'w',(2,3,256),'identity',rows_per_chunk=2)
    raw=p.read_bytes();size=struct.unpack('<Q',raw[:8])[0]
    expected=gguf.dequantize(encoded,qtype).astype(np.float32)
    assert raw[8+size:]==c._bf16_bytes(expected)


def test_unbound_nonempty_output_is_preserved(tmp_path):
    out=tmp_path/'output';out.mkdir()
    keep=out/'config.json';keep.write_text('preserve me')
    with pytest.raises(ValueError,match='nonempty'):
        c.convert(tmp_path/'source.gguf',out,tmp_path/'support')
    assert keep.read_text()=='preserve me'
