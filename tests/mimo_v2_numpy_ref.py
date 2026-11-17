"""Independent NumPy side of the MiMo-V2 tests (no MLX, no mlx2 imports).

* a minimal safetensors reader/writer;
* E4M3 block dequant, MXFP4 dequant and the TP-sharded fused-qkv split,
  written from the format definitions, not from mlx2's loader;
* a synthetic tiny checkpoint in the real MiMo-V2.6-Pro on-disk format
  (FP8 + per-rank block scales, MXFP4 experts, BF16 the rest) and its float32
  HF state dict for the vendor PyTorch reference.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3() -> np.ndarray:
    out = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 15, b & 7
        out[b] = np.nan if (e == 15 and m == 7) else s * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return out


E4M3 = e4m3()


# ------------------------------------------------------------------ safetensors
def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def read(path, name):
    """``(dtype, shape, values)``: BF16 widened to float32, F8/U8 as uint8 codes."""
    off, h = header(path)
    t = h[name]
    a, b = t["data_offsets"]
    raw = np.fromfile(path, np.uint8, count=b - a, offset=off + a)
    dt, shp = t["dtype"], t["shape"]
    if dt == "BF16":
        return dt, shp, (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shp)
    if dt == "F32":
        return dt, shp, raw.view(np.float32).reshape(shp)
    return dt, shp, raw.reshape(shp)


def write(path, tensors: dict, metadata=None):
    """``tensors``: name -> (dtype string, np array of the storage bytes' dtype)."""
    hdr, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if metadata:
        hdr["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


# ------------------------------------------------------------------ formats
def fp8_dequant(codes, scale_inv, block):
    br, bc = block
    r, c = codes.shape
    return E4M3[codes] * np.repeat(np.repeat(scale_inv.astype(np.float32), br, 0), bc, 1)[:r, :c]


def mxfp4_dequant(packed, scale):
    codes = np.stack([packed & 15, packed >> 4], -1).reshape(*packed.shape[:-1], -1)
    return FP4[codes] * np.repeat(np.exp2(scale.astype(np.float32) - 127), 32, -1)


def split_fused_qkv(dense_rank_blocks, heads, kv, hd, vhd, tp):
    """Rank-concatenated [q_r, k_r, v_r] rows -> global (q, k, v)."""
    q, k, v = heads // tp * hd, kv // tp * hd, kv // tp * vhd
    s = q + k + v
    parts = [dense_rank_blocks[r * s:(r + 1) * s] for r in range(tp)]
    return (np.concatenate([p[:q] for p in parts]), np.concatenate([p[q:q + k] for p in parts]),
            np.concatenate([p[q + k:] for p in parts]))


def fused_qkv_dequant(codes, scale, heads, kv, hd, vhd, tp, block):
    s = (heads // tp) * hd + (kv // tp) * (hd + vhd)
    nb = -(-s // block[0])
    dense = np.concatenate([fp8_dequant(codes[r * s:(r + 1) * s], scale[r * nb:(r + 1) * nb], block)
                            for r in range(tp)])
    return split_fused_qkv(dense, heads, kv, hd, vhd, tp)


# ------------------------------------------------------------------ quantizers (synthetic data)
_FIN = np.where(np.isfinite(E4M3))[0]
_ORDER = _FIN[np.argsort(E4M3[_FIN], kind="stable")]
_SORTED = E4M3[_ORDER]


def to_e4m3(x):
    i = np.clip(np.searchsorted(_SORTED, x), 1, len(_SORTED) - 1)
    lo, hi = _SORTED[i - 1], _SORTED[i]
    return _ORDER[np.where(np.abs(x - lo) <= np.abs(hi - x), i - 1, i)].astype(np.uint8)


def quant_fp8_block(w, block):
    br, bc = block
    r, c = w.shape
    gr, gc = -(-r // br), -(-c // bc)
    scale = np.zeros((gr, gc), np.float32)
    codes = np.zeros((r, c), np.uint8)
    for i in range(gr):
        for j in range(gc):
            t = w[i * br:(i + 1) * br, j * bc:(j + 1) * bc]
            s = np.float32(np.abs(t).max() / 448.0)
            scale[i, j] = s
            codes[i * br:(i + 1) * br, j * bc:(j + 1) * bc] = to_e4m3(t / s)
    return codes, scale


def quant_mxfp4(w):
    r, c = w.shape
    g = w.reshape(r, c // 32, 32)
    e = np.floor(np.log2(np.maximum(np.abs(g).max(-1), 1e-30))) - 2
    q = g / np.exp2(e)[..., None]
    mag = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    idx = np.abs(np.abs(q)[..., None] - mag).argmin(-1)
    code = (idx | ((q < 0) & (idx > 0)) * 8).astype(np.uint8).reshape(r, c)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8), (e + 127).astype(np.uint8)


def bf16(x):
    """Round float32 to bf16 (nearest even); return (uint16 bits, float32 values)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return bits, (bits.astype(np.uint32) << 16).view(np.float32)


# ------------------------------------------------------------------ synthetic tiny checkpoint
def attn_dims(cfg, l):
    if cfg["hybrid_layer_pattern"][l] == 1:
        return (cfg["swa_num_attention_heads"], cfg["swa_num_key_value_heads"], cfg["swa_head_dim"],
                cfg["swa_v_head_dim"], cfg["add_swa_attention_sink_bias"])
    return (cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["v_head_dim"],
            cfg["add_full_attention_sink_bias"])


def build_tiny_checkpoint(out: Path, cfg: dict, tp: int, seed: int = 0):
    """Write ``out`` (config.json, index, two shards) and return the float32 HF state dict."""
    rng = np.random.default_rng(seed)
    block = tuple(cfg["quantization_config"]["weight_block_size"])
    d, V = cfg["hidden_size"], cfg["vocab_size"]
    st, sd = {}, {}

    def put_bf16(name, x):
        bits, val = bf16(x.astype(np.float32))
        st[name] = ("BF16", bits)
        sd[name] = val

    def put_fp8(name, x):
        codes, scale = quant_fp8_block(x.astype(np.float32), block)
        st[name] = ("F8_E4M3", codes)
        st[name + "_scale_inv"] = ("F32", scale)
        sd[name] = fp8_dequant(codes, scale, block)

    def put_mxfp4(name, x):
        packed, scale = quant_mxfp4(x.astype(np.float32))
        st[name] = ("U8", packed)
        st[name.replace(".weight", ".weight_scale")] = ("U8", scale)
        sd[name] = mxfp4_dequant(packed, scale)

    g = lambda *s, std=1.0: rng.standard_normal(s).astype(np.float32) * std
    put_bf16("model.embed_tokens.weight", g(V, d))
    put_bf16("lm_head.weight", g(V, d, std=d ** -0.5))
    put_bf16("model.norm.weight", 1 + g(d, std=0.1))
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        put_bf16(p + "input_layernorm.weight", 1 + g(d, std=0.1))
        put_bf16(p + "post_attention_layernorm.weight", 1 + g(d, std=0.1))
        H, K, hd, vhd, sink = attn_dims(cfg, l)
        q, k, v = g(H * hd, d, std=d ** -0.5), g(K * hd, d, std=d ** -0.5), g(K * vhd, d, std=d ** -0.5)
        if cfg["attention_projection_layout"] == "fused_qkv":
            # TP storage: per rank [q_r, k_r, v_r], each rank quantized on its own block grid.
            qs, ks, vs = (np.split(m, tp) for m in (q, k, v))
            codes, scales, deq = [], [], []
            for r in range(tp):
                c_, s_ = quant_fp8_block(np.concatenate([qs[r], ks[r], vs[r]]), block)
                codes.append(c_)
                scales.append(s_)
                deq.append(fp8_dequant(c_, s_, block))
            st[p + "self_attn.qkv_proj.weight"] = ("F8_E4M3", np.concatenate(codes))
            st[p + "self_attn.qkv_proj.weight_scale_inv"] = ("F32", np.concatenate(scales))
            sd[p + "self_attn.qkv_proj.weight"] = np.concatenate(
                split_fused_qkv(np.concatenate(deq), H, K, hd, vhd, tp))
        else:
            for n, m in (("q", q), ("k", k), ("v", v)):
                put_fp8(p + f"self_attn.{n}_proj.weight", m)
        put_bf16(p + "self_attn.o_proj.weight", g(d, H * vhd, std=(H * vhd) ** -0.5))
        if sink:
            put_bf16(p + "self_attn.attention_sink_bias", g(H, std=1.0))
        if cfg["moe_layer_freq"][l]:
            E, I = cfg["n_routed_experts"], cfg["moe_intermediate_size"]
            put_bf16(p + "mlp.gate.weight", g(E, d, std=0.5))
            bias = g(E, std=0.05)
            st[p + "mlp.gate.e_score_correction_bias"] = ("F32", bias)
            sd[p + "mlp.gate.e_score_correction_bias"] = bias
            for e in range(E):
                q_ = f"{p}mlp.experts.{e}."
                put_mxfp4(q_ + "gate_proj.weight", g(I, d, std=d ** -0.5))
                put_mxfp4(q_ + "up_proj.weight", g(I, d, std=d ** -0.5))
                put_mxfp4(q_ + "down_proj.weight", g(d, I, std=I ** -0.5))
        else:
            I = cfg["intermediate_size"]
            put_fp8(p + "mlp.gate_proj.weight", g(I, d, std=d ** -0.5))
            put_fp8(p + "mlp.up_proj.weight", g(I, d, std=d ** -0.5))
            put_fp8(p + "mlp.down_proj.weight", g(d, I, std=I ** -0.5))
    out.mkdir(parents=True, exist_ok=True)
    names = list(st)
    half = len(names) // 2
    wm = {}
    for i, chunk in enumerate((names[:half], names[half:])):
        fn = f"model_pp0_ep0_shard{i}.safetensors"
        write(out / fn, {n: st[n] for n in chunk})
        wm.update({n: fn for n in chunk})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"tp_size": tp, "save_format": "mxfp4"}, "weight_map": wm}))
    (out / "config.json").write_text(json.dumps(cfg))
    return sd
