"""Hy4-flash: loading an mlx-lm-style MLX pack (the Nace.AI pack layout), CPU.

Covers each format piece the third-party pack ``nace-ai/Hy4-flash-MLX-4.6bit``
needs (provenance ``provenance/hy-v4-flash.json``): ``switch_mlp`` expert
names with ``kv_b_proj`` stored unfolded, ``q2_0`` routed experts, IQ4_NL
Engram tables in a GGUF file bound by ``engram_gguf_tables``, and an 8-bit
affine LM head under ``enable_lm_head_fp32``.  The oracles are independent
NumPy transcriptions of the formats; no pack code is imported or run.
"""

import json
import struct

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_engram as E  # noqa: E402
from test_hy_v4_flash_cpu import TOKENS, build, checkpoint, tiny_config  # noqa: E402

LEVELS = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


# ------------------------------------------------------------- IQ4_NL + GGUF --

def iq4nl_encode(rows: np.ndarray) -> np.ndarray:
    """Test-side IQ4_NL encoder: ``[n, dim]`` fp32 -> ``uint8 [n, dim/32*18]``."""
    n, dim = rows.shape
    levels = np.array(LEVELS, np.float32)
    blocks = rows.reshape(n, dim // 32, 32)
    amax = np.abs(blocks).max(-1, keepdims=True)
    d = (amax / 127.0).astype(np.float16)
    safe = np.where(d.astype(np.float32) == 0, 1.0, d.astype(np.float32))
    codes = np.abs(blocks[..., None] / safe[..., None] - levels).argmin(-1).astype(np.uint8)
    packed = (codes[..., :16] | (codes[..., 16:] << 4)).astype(np.uint8)
    out = np.concatenate([d.view(np.uint8).reshape(n, dim // 32, 2), packed], axis=-1)
    return out.reshape(n, -1)


def iq4nl_oracle(raw: np.ndarray, dim: int) -> np.ndarray:
    """Scalar transcription of ggml ``dequantize_row_iq4_nl``."""
    out = np.zeros((raw.shape[0], dim), np.float32)
    for r in range(raw.shape[0]):
        for b in range(dim // 32):
            block = raw[r, b * 18:(b + 1) * 18]
            d = float(np.frombuffer(block[:2].tobytes(), "<f2")[0])
            for j in range(16):
                out[r, b * 32 + j] = np.float32(d) * np.float32(LEVELS[block[2 + j] & 15])
                out[r, b * 32 + j + 16] = np.float32(d) * np.float32(LEVELS[block[2 + j] >> 4])
    return out


def write_gguf(path, tensors, *, alignment=32, ggml_type=20, extra_kv=True):
    """Minimal GGUF v3 writer: ``tensors`` maps name -> (packed rows uint8, dim)."""
    def s(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kvs = []
    if extra_kv:
        kvs.append(s("general.name") + struct.pack("<I", 8) + s("tiny"))
        kvs.append(s("tiny.list") + struct.pack("<IIQ", 9, 5, 2) + struct.pack("<ii", 3, 4))
    kvs.append(s("general.alignment") + struct.pack("<II", 4, alignment))
    infos, payload, offset = [], [], 0
    for name, (packed, dim) in tensors.items():
        offset = -(-offset // alignment) * alignment
        infos.append(s(name) + struct.pack("<IQQIQ", 2, dim, packed.shape[0], ggml_type, offset))
        payload.append((offset, packed.tobytes()))
        offset += packed.nbytes
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + b"".join(infos)
    data0 = -(-len(head) // alignment) * alignment
    blob = bytearray(head + b"\0" * (data0 - len(head)))
    for off, raw in payload:
        blob += b"\0" * (data0 + off - len(blob))
        blob += raw
    path.write_bytes(bytes(blob))
    return data0


def test_iq4nl_decoder_matches_scalar_ggml_oracle():
    rng = np.random.default_rng(1)
    rows = (rng.standard_normal((9, 96)) * rng.uniform(0.01, 3, (9, 1))).astype(np.float32)
    rows[0] = 0.0  # zero block: d = 0
    raw = iq4nl_encode(rows)
    assert raw.shape == (9, 54)
    got = E.decode_iq4nl_row(raw, 96)
    np.testing.assert_array_equal(got, iq4nl_oracle(raw, 96))
    assert got.dtype == np.float32 and np.all(got[0] == 0)
    # Approximate: bounded error against the source rows.
    assert np.abs(got - rows).max() <= np.abs(rows).max() * 0.12
    with pytest.raises(ValueError, match="wrong width"):
        E.decode_iq4nl_row(raw[:, :-1], 96)
    bad = raw.copy()
    bad[1, :2] = np.frombuffer(np.float16(np.inf).tobytes(), np.uint8)
    with pytest.raises(ValueError, match="nonfinite"):
        E.decode_iq4nl_row(bad, 96)


def test_gguf_header_parse_and_reader(tmp_path):
    rng = np.random.default_rng(2)
    a = iq4nl_encode(rng.standard_normal((37, 64)).astype(np.float32))
    b = iq4nl_encode(rng.standard_normal((5, 96)).astype(np.float32))
    path = tmp_path / "t.gguf"
    data0 = write_gguf(path, {"a.weight": (a, 64), "b.weight": (b, 96)}, alignment=64)
    tensors = E.gguf_tensors(path)
    assert tensors["a.weight"] == (data0, 20, (64, 37))
    assert tensors["b.weight"][0] % 64 == 0 and tensors["b.weight"][2] == (96, 5)
    layout = E.table_layout(path, "a.weight")
    assert (layout["format"], layout["rows"], layout["dim"], layout["row_bytes"]) == ("iq4_nl", 37, 64, 36)
    reader = E.EngramTableRows(path, "a.weight", table_format="iq4_nl", decode_workers=2, prefill_workers=3)
    try:
        rows = np.array([36, 0, 5, 5, 17])
        np.testing.assert_array_equal(reader.read_host(rows), iq4nl_oracle(a[rows], 64))
        diag = reader.diagnostics()
        assert diag["format"] == "iq4_nl" and diag["approximate"] and not diag["qualified"]
        assert reader.fp8_round_trip_dtype is None  # the round trip stays at runtime
        assert reader.stats["bytes_read"] == 4 * 36
    finally:
        reader.close()
    with pytest.raises(ValueError, match="is iq4_nl, policy asked for bf16"):
        E.EngramTableRows(path, "a.weight", table_format="bf16")
    with pytest.raises(KeyError):
        E.table_layout(path, "missing.weight")


def test_gguf_refuses_other_types_and_truncation(tmp_path):
    raw = iq4nl_encode(np.ones((4, 32), np.float32))
    q8 = tmp_path / "q8.gguf"
    write_gguf(q8, {"t": (raw, 32)}, ggml_type=8)
    with pytest.raises(ValueError, match="only IQ4_NL"):
        E.table_layout(q8, "t")
    short = tmp_path / "short.gguf"
    write_gguf(short, {"t": (raw, 32)})
    short.write_bytes(short.read_bytes()[:-10])
    with pytest.raises(ValueError, match="past the end"):
        E.table_layout(short, "t")
    head = tmp_path / "head.gguf"
    head.write_bytes(b"GGUF" + struct.pack("<IQ", 3, 1))
    with pytest.raises(ValueError, match="truncated"):
        E.gguf_tensors(head)


def test_gguf_bindings_are_exact_and_stay_inside():
    layers = {1: 2, 4: 5}
    ok = {"2": {"file": "e.gguf", "tensor": "x"}, "5": {"file": "e.gguf", "tensor": "y"}}
    assert E.gguf_bindings(ok, layers) == {1: ("e.gguf", "x"), 4: ("e.gguf", "y")}
    with pytest.raises(ValueError, match="exactly the Engram layers"):
        E.gguf_bindings({"2": ok["2"]}, layers)
    with pytest.raises(ValueError, match="within the artifact"):
        E.gguf_bindings({**ok, "5": {"file": "../e.gguf", "tensor": "y"}}, layers)


def test_converter_does_not_offer_iq4nl():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "convert_hy_v4_flash_mlx", Path(__file__).resolve().parents[1] / "scripts" / "convert_hy_v4_flash_mlx.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    with pytest.raises(SystemExit):
        conv.parse_formats("iq4_nl")


# ---------------------------------------------------------------------- q2_0 --

def q2_pack(q: np.ndarray) -> np.ndarray:
    """2-bit codes ``[..., I]`` -> little-endian U32 words, low bits first."""
    q = q.astype(np.uint32).reshape(*q.shape[:-1], -1, 16)
    return (q << (2 * np.arange(16, dtype=np.uint32))).sum(-1).astype(np.uint32)


def q2_planes(rng, E_, O, I, group):
    q = rng.integers(0, 4, (E_, O, I))
    scales = (rng.uniform(0.02, 0.4, (E_, O, I // group)) * rng.choice([-1, 1], (E_, O, I // group))).astype(np.float16)
    dense = scales.astype(np.float32).repeat(group, -1) * (q.astype(np.float32) - 1)
    return q2_pack(q), scales, dense


@pytest.mark.parametrize("group", [32, 64])
def test_q2_zero_bank_matches_dense_oracle(group):
    rng = np.random.default_rng(3)
    E_, O, I = 6, 24, 128
    w, scales, dense = q2_planes(rng, E_, O, I, group)
    weights = {"b.weight": mx.array(w), "b.scales": mx.array(scales)}
    H.q2_0_planes(weights, "b")
    assert weights["b.scales"].dtype == mx.float32
    np.testing.assert_array_equal(np.array(weights["b.biases"]), -np.array(weights["b.scales"]))
    bank = H.Q2ZeroSwitchLinear(I, O, E_, group_size=group)
    bank.load_weights([(k[len("b."):], v) for k, v in weights.items()], strict=True)
    x = rng.standard_normal((5, 1, 1, I)).astype(np.float32)
    idx = rng.integers(0, E_, (5, 3))  # SwitchGLU shapes: x [T, 1, 1, I], indices [T, k]
    for dtype in (mx.float32, mx.bfloat16):
        xin = mx.array(x).astype(dtype)
        got = bank(xin, mx.array(idx))
        assert got.dtype == dtype
        ref = np.einsum("tki,tkoi->tko", np.array(xin.astype(mx.float32))[:, 0, 0][:, None].repeat(3, 1),
                        dense[idx])
        tol = 1e-4 if dtype == mx.float32 else 2e-2
        assert got.shape == (5, 3, 1, O)
        np.testing.assert_allclose(np.array(got.astype(mx.float32))[:, :, 0], ref, rtol=tol, atol=tol * 4)


def test_q2_planes_refuse_stored_biases():
    weights = {"b.weight": mx.zeros((1, 2, 2), mx.uint32), "b.scales": mx.ones((1, 2, 1), mx.float16),
               "b.biases": mx.ones((1, 2, 1), mx.float16)}
    with pytest.raises(ValueError, match="no biases"):
        H.q2_0_planes(weights, "b")


def test_quantization_modes_fail_closed():
    from mlx2.adapters import hy_v4_flash as A

    base = {"group_size": 64, "bits": 4, "mode": "affine"}
    A.check_quantization({**base, "model.layers.1.mlp.switch_mlp.up_proj": {"mode": "q2_0", "bits": 2, "group_size": 64},
                          "model.layers.0.self_attn.kv_b_proj": False})
    with pytest.raises(ValueError, match="routed experts only"):
        A.check_quantization({**base, "lm_head": {"mode": "q2_0", "bits": 2, "group_size": 64}})
    with pytest.raises(ValueError, match="routed experts only"):
        A.check_quantization({**base, "model.dfly_layers.0.mlp.experts.up_proj":
                              {"mode": "q2_0", "bits": 2, "group_size": 64}})
    with pytest.raises(ValueError, match="not implemented"):
        A.check_quantization({**base, "lm_head": {"mode": "mxfp4", "bits": 4, "group_size": 32}})
    with pytest.raises(ValueError, match="affine-quantized"):
        A.check_quantization({**base, "mode": "q2_0"})
    assert A.module_quantization({**base, "model.layers.3.mlp.switch_mlp.down_proj": {"bits": 4},
                                  "x": False}) == {"model.layers.3.mlp.experts.down_proj": {"bits": 4}}


# ----------------------------------------------------------- fp32 LM head --

def test_quantized_lm_head_keeps_the_fp32_contract():
    config = tiny_config()
    args = H.ModelArgs.from_hf_config(config)
    model = H.Model(args)
    rng = np.random.default_rng(5)
    w = mx.array(rng.standard_normal((args.vocab_size, args.hidden_size)).astype(np.float32))
    q, s, b = mx.quantize(w, group_size=32, bits=8, mode="affine")
    model.lm_head = model.lm_head.to_quantized(group_size=32, bits=8)
    model.lm_head.load_weights([("weight", q), ("scales", s.astype(mx.bfloat16)),
                                ("biases", b.astype(mx.bfloat16))], strict=True)
    dense = np.array(mx.dequantize(q, s.astype(mx.bfloat16).astype(mx.float32),
                                   b.astype(mx.bfloat16).astype(mx.float32), group_size=32, bits=8))
    hidden = mx.array(rng.standard_normal((1, 3, args.hidden_size)).astype(np.float32)).astype(mx.bfloat16)
    oracle = np.array(hidden.astype(mx.float32)) @ dense.astype(np.float64).T
    got = model.logits(hidden)
    assert got.dtype == mx.float32
    fp32_err = np.abs(np.array(got) - oracle).max()
    bf16_err = np.abs(np.array(model.lm_head(hidden).astype(mx.float32)) - oracle).max()
    assert fp32_err < 1e-3 < bf16_err
    model.args.enable_lm_head_fp32 = False
    np.testing.assert_array_equal(np.array(model.logits(hidden)),
                                  np.array(model.lm_head(hidden).astype(mx.float32)))


def test_sanitize_renames_switch_mlp_casts_quantized_head_and_refuses_quantized_kv_b():
    config = tiny_config()
    model = H.Model(H.ModelArgs.from_hf_config(config))
    out = model.sanitize({"model.layers.1.mlp.switch_mlp.up_proj.weight": mx.zeros((2, 2), mx.uint32),
                          "lm_head.weight": mx.zeros((4, 2), mx.uint32),
                          "lm_head.scales": mx.zeros((4, 1), mx.bfloat16),
                          "lm_head.biases": mx.zeros((4, 1), mx.bfloat16)})
    assert set(out) == {"model.layers.1.mlp.experts.up_proj.weight", "lm_head.weight",
                        "lm_head.scales", "lm_head.biases"}
    assert out["lm_head.weight"].dtype == mx.uint32
    assert out["lm_head.scales"].dtype == mx.float32 and out["lm_head.biases"].dtype == mx.float32
    with pytest.raises(ValueError, match="quantized kv_b_proj"):
        model.sanitize({"model.layers.0.self_attn.kv_b_proj.scales": mx.zeros((2, 2))})


# ---------------------------------------------------- whole pack, end to end --

def pack_config():
    return tiny_config(routed_expert_hidden_size=32, moe_intermediate_size=32,
                       engram_n_embed_per_ngram=64, engram_fp8_embedding=True)


def _quant(value, bits, group=32):
    q, s, b = mx.quantize(mx.array(value), group_size=group, bits=bits, mode="affine")
    return {"weight": q, "scales": s, "biases": b}, np.array(mx.dequantize(q, s, b, group_size=group, bits=bits))


def write_pack(tmp_path, config, W):
    """An mlx-lm-style pack of ``W``: what the Nace.AI pack's layout looks like.

    Returns ``(pack dir, dequantized float weights for the oracle model)``.
    """
    from safetensors.numpy import save_file

    t = config["text_config"]
    rng = np.random.default_rng(11)
    ref = {k: v for k, v in W.items() if not k.startswith("model.dfly_layers.")}
    pack, quant = {}, {"group_size": 32, "bits": 4, "mode": "affine"}

    def put(stem, value, bits):
        planes, dense = _quant(value, bits)
        for part, arr in planes.items():
            pack[f"{stem}.{part}"] = np.array(arr)
        quant[stem] = {"group_size": 32, "bits": bits, "mode": "affine"}
        ref[stem + ".weight"] = dense

    tables = {}
    for key in list(ref):
        if key.endswith("multi_head_embedding.embedding.weight"):
            raw = iq4nl_encode(ref.pop(key))
            tables[key] = raw
            continue
    for key, value in list(ref.items()):
        if key.endswith("mlp.experts.gate_up_proj"):
            p = key[: -len("experts.gate_up_proj")]
            inter = value.shape[1] // 2
            down = ref.pop(p + "experts.down_proj")
            del ref[key]
            layer = int(key.split(".")[2])
            for role, v in (("gate", value[:, :inter]), ("up", value[:, inter:]), ("down", down)):
                stem = f"{p}switch_mlp.{role}_proj"
                if layer in (1, 2):  # q2_0 banks, as the pack's layers 1-2
                    w, scales, dense = q2_planes(rng, *v.shape, 32)
                    pack[stem + ".weight"], pack[stem + ".scales"] = w, scales
                    quant[stem] = {"mode": "q2_0", "bits": 2, "group_size": 32}
                else:
                    planes, dense = _quant(v, 4)
                    for part, arr in planes.items():
                        pack[f"{stem}.{part}"] = np.array(arr)
                    quant[stem] = {"mode": "affine", "bits": 4, "group_size": 32}
                ref[f"{p}experts_dense.{role}"] = dense
    for key in [k for k in ref if ".experts_dense." in k]:
        p, role = key.split(".experts_dense.")
        if role == "gate":
            ref[p + ".experts.gate_up_proj"] = np.concatenate(
                [ref.pop(key), ref.pop(p + ".experts_dense.up")], axis=1)
            ref[p + ".experts.down_proj"] = ref.pop(p + ".experts_dense.down")
    put("lm_head", W["lm_head.weight"], 8)
    put("model.embed_tokens", W["model.embed_tokens.weight"], 8)
    for key, value in W.items():
        if key.startswith("model.dfly_layers.") or key in ("lm_head.weight", "model.embed_tokens.weight"):
            continue
        if key.endswith("multi_head_embedding.embedding.weight") or ".mlp.experts." in key:
            continue
        stem = key[: -len(".weight")]
        if (key.endswith(("self_attn.q_a_proj.weight", "self_attn.o_proj.weight",
                          "engram_layer.value_proj.weight", "mlp.shared_experts.up_proj.weight"))
                and value.shape[-1] % 32 == 0):
            put(stem, value, 8)
        else:
            pack[key] = value  # kv_b_proj stays unfolded, as in the pack
            if key.endswith("self_attn.kv_b_proj.weight"):
                quant[stem] = False
    save_file({k: v for k, v in pack.items() if ".switch_mlp." not in k}, str(tmp_path / "model-00001.safetensors"))
    save_file({k: v for k, v in pack.items() if ".switch_mlp." in k}, str(tmp_path / "model-experts.safetensors"))
    weight_map = {k: ("model-experts.safetensors" if ".switch_mlp." in k else "model-00001.safetensors")
                  for k in pack}
    bindings, gguf = {}, {}
    for key, raw in tables.items():
        index = int(key.split(".")[2])
        name = f"model.layers.{index}.engram_layer.embedding.weight"
        gguf[name] = (raw, raw.shape[1] // 18 * 32)
        bindings[str(index + 1)] = {"file": "engram-iq4nl.gguf", "tensor": name}
        ref[key] = E.decode_iq4nl_row(raw, raw.shape[1] // 18 * 32)
    write_gguf(tmp_path / "engram-iq4nl.gguf", gguf)
    config = json.loads(json.dumps(config))
    config["text_config"]["engram_gguf_tables"] = bindings
    config["quantization"] = config["quantization_config"] = quant
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tmp_path, config, ref


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    config = pack_config()
    W = checkpoint(config, seed=7, scale=0.2)
    root, pack_cfg, ref = write_pack(tmp_path_factory.mktemp("pack"), config, W)
    return root, pack_cfg, ref


def test_pack_inspects_with_its_gguf_in_the_identity(pack):
    from mlx2.adapters import hy_v4_flash as A

    root, config, _ = pack
    artifact = A.inspect_artifact(root)
    assert artifact["engram_table_formats"] == ["iq4_nl"]
    assert artifact["quantized"] and not artifact["has_dfly"] and not artifact["has_vision"]
    assert "engram-iq4nl.gguf" in {name for name, *_ in artifact["identity"]["files"]}
    # Rewriting the GGUF changes the fingerprint.
    import os

    gguf = root / "engram-iq4nl.gguf"
    stat = gguf.stat()
    os.utime(gguf, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    try:
        assert A.inspect_artifact(root)["identity"]["fingerprint"] != artifact["identity"]["fingerprint"]
    finally:
        os.utime(gguf, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_pack_memory_budget_counts_the_widened_planes(pack):
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.runtime.models.hy_v4_flash_engram import safetensors_header

    root, config, _ = pack
    artifact = A.inspect_artifact(root)
    header = {**safetensors_header(root / "model-experts.safetensors"),
              **safetensors_header(root / "model-00001.safetensors")}
    size = lambda n: header[n]["data_offsets"][1] - header[n]["data_offsets"][0]
    q2 = [k + ".scales" for k, v in config["quantization"].items() if isinstance(v, dict) and v.get("mode") == "q2_0"]
    want = sum(size(n) * 3 for n in q2)  # fp16 -> fp32 scales + fp32 biases
    want += sum(size(f"lm_head.{p}") for p in ("scales", "biases"))  # fp32 (4 B) vs fp32 on disk here
    want -= sum(size(f"lm_head.{p}") for p in ("scales", "biases"))
    assert A.widened_bytes(root, artifact["weight_map"], config["quantization"], lm_head_fp32=True) == want > 0


def test_pack_refuses_unequal_quantization_config_and_missing_gguf(tmp_path, pack):
    import shutil

    from mlx2.adapters import hy_v4_flash as A

    root, config, _ = pack
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    bad = dict(config, quantization_config={"quant_method": "fp8"})
    (copy / "config.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="FP8/compressed-tensors"):
        A.inspect_artifact(copy)
    (copy / "config.json").write_text(json.dumps(config))
    (copy / "engram-iq4nl.gguf").unlink()
    with pytest.raises(ValueError, match="Engram GGUF file"):
        A.inspect_artifact(copy)


def test_pack_loads_and_matches_its_dequantized_float_model(pack):
    from mlx2.adapters import hy_v4_flash as A

    root, config, ref = pack
    artifact = A.inspect_artifact(root)
    with pytest.raises(ValueError, match="no bf16 Engram tables|not in the artifact"):
        A.load_model(root, artifact)
    loaded, drafter, readers = A.load_model(root, artifact, engram_table_format="iq4_nl")
    try:
        assert drafter is None
        layer = loaded.model.layers[1].mlp.experts
        assert isinstance(layer.up_proj, H.Q2ZeroSwitchLinear)
        assert loaded.model.layers[3].mlp.experts.up_proj.bits == 4
        assert loaded.lm_head.scales.dtype == mx.float32
        assert readers[1].format == "iq4_nl" and readers[1].approximate
        assert loaded.model.layers[1].engram_layer.table_round_trip_dtype is None
        oracle_model, _ = build(pack_config(), {**ref, **{k: v for k, v in checkpoint(pack_config(), seed=7, scale=0.2).items()
                                                        if k.startswith("model.dfly_layers.")}})
        got = np.array(loaded(mx.array([TOKENS])))
        want = np.array(oracle_model(mx.array([TOKENS])))
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)
        assert (got.argmax(-1) == want.argmax(-1)).all()
        assert readers[1].stats["unique_rows"] > 0
    finally:
        for r in readers.values():
            r.close()


def test_pack_receipts_label_iq4nl_approximate_and_namespace_it():
    from mlx2.adapters import hy_v4_flash as A

    record = A.engram_format_record("iq4_nl")
    assert record["approximate"] and not record["qualified"] and not record["default"]
    assert A.numerics_layout(A.CACHE_LAYOUT, engram_table_format="iq4_nl").endswith("+engram-iq4_nl")
    assert "no E4M3 round trip" in A.engram_format_record("iq4_nl", "vllm")["relation"]
    meta = A.descriptor_for(engram_table_format="iq4_nl").metadata["engram_tables"]
    assert "unqualified" in meta and "approximate" in meta
    with pytest.raises(ValueError, match="unknown table format"):
        A._fingerprint_engram_formats({"config": {"mlx2_engram_formats": {"iq4_nl": {}}},
                                       "identity": {"path": "/", "files": [], "fingerprint": ""}})
