"""CPU-only (numpy) AscMLA fp4 latent-KV error estimate. No MLX, no Metal.

Research estimate for docs/research/ascmla-fp4-latent-kv-estimate-2026-11-21.md.
Usage: python -I ascmla_fp4_latent_kv_cpu.py <Xing4.0 bf16 checkpoint dir> <out.json>
Reads three projection tensors per layer and 2048 embedding rows by mmap;
never loads the model.  Original mlx2 code; formats follow arXiv 2611.26106
(AscMLA) and src/mlx2/runtime/models/hy_v4_flash_fp8kv.py.
"""
import json, os, struct, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 2 else os.environ.get(
    "MLX2_XING4_CHECKPOINT", os.path.join(os.path.expanduser("~"), "Models", "Xing4.0-29B-A4B"))
rng = np.random.default_rng(1121)


def tensor(name):
    wm = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
    path = f"{ROOT}/{wm[name]}"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h[name]
    assert meta["dtype"] == "BF16"
    a, b = meta["data_offsets"]
    raw = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + a, shape=((b - a) // 2,))
    return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def rows(name, idx):
    wm = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
    path = f"{ROOT}/{wm[name]}"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h[name]
    a, b = meta["data_offsets"]
    V, D = meta["shape"]
    mm = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + a, shape=(V, D))
    return (np.asarray(mm[idx]).astype(np.uint32) << 16).view(np.float32)


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def rms(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w


def e4m3(x):
    x = np.clip(x, -448, 448)
    a = np.abs(x)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -6)))
    q = 2.0 ** (e - 3)
    return np.round(x / q) * q


E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)


def e2m1(x):
    s = np.sign(x)
    a = np.minimum(np.abs(x), 6.0)
    i = np.abs(a[..., None] - E2M1).argmin(-1)
    return s * E2M1[i]


def mlx2_fp8(lat):  # hy_v4_flash_fp8kv: E4M3 codes, fp32 pow2 scale per 128 tile
    t = lat.reshape(*lat.shape[:-1], -1, 128)
    amax = np.maximum(np.abs(t).max(-1, keepdims=True), 1e-30)
    sc = 2.0 ** np.ceil(np.log2(amax / 448))
    sc = np.where(amax / sc > 448, sc * 2, sc)
    r = (e4m3(t / sc) * sc).reshape(lat.shape)
    return r, r


def nvfp4_shared(lat, block=16):  # one payload, one E4M3 scale per 16, K == V
    t = lat.reshape(*lat.shape[:-1], -1, block)
    sc = e4m3(np.abs(t).max(-1, keepdims=True) / 6)
    sc = np.where(sc == 0, 1, sc)
    r = (e2m1(t / sc) * sc).reshape(lat.shape)
    return r, r


def ascmla(lat, block=16, fit_k=True):
    # V first: one fp32 scale per token, codes from it; then K block scales
    # fitted by least squares to the original latent, stored as E4M3.
    sv = np.abs(lat).max(-1, keepdims=True) / 6
    sv = np.where(sv == 0, 1, sv)
    c = e2m1(lat / sv)
    v = c * sv
    tc = c.reshape(*lat.shape[:-1], -1, block)
    tl = lat.reshape(*lat.shape[:-1], -1, block)
    if fit_k:
        den = (tc * tc).sum(-1, keepdims=True)
        sk = np.where(den > 0, (tc * tl).sum(-1, keepdims=True) / np.maximum(den, 1e-30), 0)
        sk = e4m3(np.maximum(sk, 0))
    else:
        sk = e4m3(np.broadcast_to(sv[..., None], tc.shape[:-1] + (1,)))
    k = (tc * sk).reshape(lat.shape)
    return k, v


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def evaluate(lat, q, pe, qpe, scale, label):
    # lat [T, 512] latent rows (K and V share it in MLA); q [H, Q, 512]
    # absorbed queries; pe [T, 64] rope key (kept exact here: every format
    # stores it at >= 8 bits and the paper stores it E4M3 -- E4M3 pe is
    # applied for the AscMLA arm below).
    ref_s = (np.einsum("hqd,td->hqt", q, lat) + np.einsum("hqd,td->hqt", qpe, pe)) * scale
    ref_p = softmax(ref_s)
    ref_o = np.einsum("hqt,td->hqd", ref_p, lat)
    out = {}
    for name, fn, pe_fmt in (
        ("fp8 (mlx2 Hy4 row: E4M3/128 pow2, pe bf16)", mlx2_fp8, bf16),
        ("fp4 NVFP4-style shared K=V (E4M3/16)", nvfp4_shared, bf16),
        ("fp4 AscMLA (V fp32/token, K E4M3/16 LS-fit, pe E4M3)", ascmla, e4m3),
        ("fp4 AscMLA, pe kept bf16 (mlx2 row convention)", ascmla, bf16),
        ("fp4 AscMLA without K fit", lambda x: ascmla(x, fit_k=False), e4m3),
    ):
        k, v = fn(lat)
        p8 = pe_fmt(pe)
        s = (np.einsum("hqd,td->hqt", q, k) + np.einsum("hqd,td->hqt", qpe, p8)) * scale
        p = softmax(s)
        o = np.einsum("hqt,td->hqd", p, v)
        cos = float(np.mean(np.sum(o * ref_o, -1) / (np.linalg.norm(o, axis=-1) * np.linalg.norm(ref_o, axis=-1))))
        out[name] = {
            "K_rel": rel(k, lat), "V_rel": rel(v, lat),
            "score_rel": rel(s, ref_s),
            "score_abs_mean": float(np.mean(np.abs(s - ref_s))),
            "prob_TV": float(np.mean(0.5 * np.abs(p - ref_p).sum(-1))),
            "out_rel": rel(o, ref_o), "out_cos": cos,
        }
    print(f"\n## {label}")
    for name, m in out.items():
        print(f"{name:58s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return out


def main():
    T, Q, H = 2048, 16, 32
    nope, rope, vdim = 128, 64, 128
    scale = (nope + rope) ** -0.5
    results = {}
    # Synthetic: Gaussian latent with per-channel gain spread (log-normal), heavy-tail channels.
    gain = np.exp(rng.normal(0, 0.6, 512))
    lat = rng.standard_normal((T, 512)) * gain
    lat = lat / np.sqrt(np.mean(lat ** 2, -1, keepdims=True))
    pe = rng.standard_normal((T, rope))
    q = rng.standard_normal((H, Q, 512)) * gain / 4
    qpe = rng.standard_normal((H, Q, rope))
    results["synthetic"] = evaluate(bf16(lat), q, bf16(pe), qpe, scale, "synthetic log-normal-gain latent")

    toks = np.sort(rng.choice(131072, size=T, replace=False))
    emb = rows("model.embed_tokens.weight", toks)
    for layer in (0, 20):
        p = f"model.layers.{layer}.self_attn."
        ln = tensor(f"model.layers.{layer}.input_layernorm.weight")
        wa = tensor(p + "kv_a_proj_with_mqa.weight")
        kvn = tensor(p + "kv_a_layernorm.weight")
        wb = tensor(p + "kv_b_proj.weight").reshape(H, nope + vdim, 512)
        x = bf16(rms(emb, ln))
        kv = x @ wa.T
        lat = bf16(rms(kv[:, :512], kvn))
        pe = bf16(kv[:, 512:])  # unroped; RoPE is a rotation, norm-preserving
        g = rng.standard_normal((H, Q, nope))
        q = np.einsum("hqn,hnd->hqd", g, wb[:, :nope, :])  # absorbed q_nope
        qpe = rng.standard_normal((H, Q, rope))
        results[f"xing_layer{layer}"] = evaluate(
            lat, q, pe, qpe, scale,
            f"Xing4.0 layer {layer}: real kv_a/kv_a_layernorm/kv_b weights, inputs = RMSNorm(real token embeddings)")
    json.dump(results, open(sys.argv[-1], "w"), indent=1)


if __name__ == "__main__":
    main()
