"""Hy4-flash Engram table formats (fp8_e4m3_row, affine4_g32) on CPU.

Decoders are checked bit for bit against independent float64 NumPy oracles;
the fp8 table against the model's own E4M3 round trip of the bf16 table;
the affine 4-bit table's error is measured and bounded.  Tiny random
checkpoint only (``test_hy_v4_flash_cpu.checkpoint``), no real weights.
"""

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_engram as E  # noqa: E402
from test_hy_v4_flash_cpu import TOKENS, _write_artifact, build, checkpoint, tiny_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = "engram_layer.multi_head_embedding.embedding.weight"


def _converter():
    spec = importlib.util.spec_from_file_location(
        "convert_hy_v4_flash_mlx", ROOT / "scripts" / "convert_hy_v4_flash_mlx.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    return conv


def bf16_bits(x):
    """fp32 -> bf16 bit patterns, round to nearest even."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_value(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def e4m3_oracle(codes):
    """float64 E4M3FN value of each code, from the bit fields."""
    codes = np.asarray(codes, np.int64)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    e, m = (codes >> 3) & 0xF, codes & 0x7
    mag = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e.astype(np.float64) - 7))
    return sign * mag


def write_table(path, name, values, dtype="BF16"):
    """Single-tensor safetensors shard holding one table (BF16 or F32)."""
    data = bf16_bits(values).tobytes() if dtype == "BF16" else np.asarray(values, "<f4").tobytes()
    header = json.dumps({name: {"dtype": dtype, "shape": list(values.shape),
                                "data_offsets": [0, len(data)]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + data)
    return path


def _rows(n=257, dim=96, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32) * rng.uniform(1e-3, 30, (n, 1)).astype(np.float32)
    x[3] = 0.0  # an all-zero row (scale floor)
    x[4, :5] = [448.0, -1e-6, 3e-5, 0.0, -0.0]
    return bf16_value(bf16_bits(x))


# ------------------------------------------------------------- decoders --

@pytest.mark.parametrize("scale_dtype", ["BF16", "F32"])
def test_fp8_row_decoder_matches_numpy_oracle(scale_dtype):
    rng = np.random.default_rng(1)
    dim, n = 96, 64
    codes = rng.integers(0, 256, (n, dim), dtype=np.uint8)
    codes[(codes & 127) == 127] = 0x38  # no NaN codes
    scale = (rng.uniform(1e-6, 2, (n, 1)) * np.sign(rng.standard_normal((n, 1)))).astype(np.float32)
    raw_scale = bf16_bits(scale).view(np.uint8) if scale_dtype == "BF16" else scale.view(np.uint8)
    scale64 = (bf16_value(bf16_bits(scale)) if scale_dtype == "BF16" else scale).astype(np.float64)
    raw = np.concatenate([codes, raw_scale.reshape(n, -1)], axis=1)
    got = E.decode_fp8_row(raw, dim, scale_dtype)
    np.testing.assert_array_equal(got, (e4m3_oracle(codes) * scale64).astype(np.float32))
    for code in (0x7F, 0xFF):
        nan = raw.copy()
        nan[0, 0] = code
        with pytest.raises(ValueError, match="nonfinite"):
            E.decode_fp8_row(nan, dim, scale_dtype)


@pytest.mark.parametrize("scale_dtype", ["BF16", "F32"])
def test_affine4_decoder_matches_numpy_oracle(scale_dtype):
    rng = np.random.default_rng(2)
    dim, n = 96, 64
    q = rng.integers(0, 16, (n, dim)).astype(np.uint32)
    scales = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)
    biases = rng.uniform(-4, 4, (n, 3)).astype(np.float32)
    if scale_dtype == "BF16":
        scales, biases = bf16_value(bf16_bits(scales)), bf16_value(bf16_bits(biases))
        pack = lambda v: bf16_bits(v).view(np.uint8).reshape(n, -1)
    else:
        pack = lambda v: v.view(np.uint8).reshape(n, -1)
    # Word w holds values 8w..8w+7, low nibble first.
    words = np.zeros((n, dim // 8), np.uint32)
    for j in range(dim):
        words[:, j // 8] |= q[:, j] << np.uint32(4 * (j % 8))
    raw = np.concatenate([words.view(np.uint8).reshape(n, -1), pack(scales), pack(biases)], axis=1)
    assert raw.shape[1] == E.packed_row_bytes(E.AFFINE4, dim, scale_dtype)
    got = E.decode_affine4_row(raw, dim, scale_dtype)
    g = np.repeat(np.arange(3), 32)
    # IEEE fp32 multiply then add, each rounded once (the product is exact
    # for BF16 scales: a 4-bit by 8-bit significand).
    product = (q.astype(np.float64) * scales[:, g].astype(np.float64)).astype(np.float32)
    ref = (product.astype(np.float64) + biases[:, g].astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("dtype,mx_dtype", [("BF16", mx.bfloat16), ("F32", mx.float32)])
def test_fp8_row_pack_equals_model_round_trip_bitwise(dtype, mx_dtype):
    conv = _converter()
    x = _rows()
    src = mx.array(bf16_bits(x)).view(mx.bfloat16) if dtype == "BF16" else mx.array(x)
    packed = conv._pack_rows(mx, E.FP8_ROW, src)
    assert packed.shape == (x.shape[0], E.packed_row_bytes(E.FP8_ROW, 96, dtype))
    decoded = mx.array(E.decode_fp8_row(packed, 96, dtype)).astype(mx_dtype)
    reference = H.fp8_round_trip(src.astype(mx_dtype))
    assert mx.array_equal(decoded, reference).item()


def test_affine4_pack_matches_mx_quantize_and_error_is_bounded():
    conv = _converter()
    x = _rows(n=4096, seed=5)
    src = mx.array(bf16_bits(x)).view(mx.bfloat16)
    packed = conv._pack_rows(mx, E.AFFINE4, src)
    assert packed.shape == (4096, 60)
    got = E.decode_affine4_row(packed, 96, "BF16")
    w, s, b = mx.quantize(src, group_size=32, bits=4, mode="affine")
    as_u8 = lambda a: np.array(a.view(mx.uint8)).reshape(4096, -1)
    np.testing.assert_array_equal(packed, np.concatenate([as_u8(w), as_u8(s), as_u8(b)], axis=1))
    # MLX dequantize with fp32 scales/biases computes the same fp32 law (its
    # bf16 CPU kernel instead rounds the product and the sum in bf16).
    cpu = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=32, bits=4)
    np.testing.assert_array_equal(got, np.array(cpu))
    rel = np.linalg.norm(got - x, axis=1) / np.maximum(np.linalg.norm(x, axis=1), 1e-30)
    live = np.linalg.norm(x, axis=1) > 0
    print(f"affine4 row relative L2 error: mean {rel[live].mean():.4f} p99 {np.quantile(rel[live], .99):.4f}")
    assert rel[live].mean() < 0.12 and rel[live].max() < 0.2


# --------------------------------------------------------------- reader --

def test_reader_decodes_each_format_on_the_pool_and_labels_it(tmp_path):
    conv = _converter()
    x = _rows(n=300)
    name = "model.layers.1." + TABLE
    src = write_table(tmp_path / "src.safetensors", name, x)
    rows = np.array([5, 299, 5, 0, 17, 3])
    readers = {"bf16": E.EngramTableRows(src, name, cache_rows=4)}
    for fmt in (E.FP8_ROW, E.AFFINE4):
        out = tmp_path / f"t.{fmt}.safetensors"
        conv.write_packed_table(src, name, out, fmt, chunk_rows=7)  # several chunks
        readers[fmt] = E.EngramTableRows(out, name, cache_rows=4, table_format=fmt)
    try:
        np.testing.assert_array_equal(readers["bf16"].read_host(rows), x[rows])
        src_mx = mx.array(bf16_bits(x)).view(mx.bfloat16)
        fp8 = E.decode_fp8_row(conv._pack_rows(mx, E.FP8_ROW, src_mx), 96, "BF16")
        np.testing.assert_array_equal(readers[E.FP8_ROW].read_host(rows), fp8[rows])
        aff = E.decode_affine4_row(conv._pack_rows(mx, E.AFFINE4, src_mx), 96, "BF16")
        np.testing.assert_array_equal(readers[E.AFFINE4].read_host(rows), aff[rows])
        # Prefill-sized read goes through the prefill pool and still decodes.
        many = np.arange(300).repeat(2)
        np.testing.assert_array_equal(readers[E.AFFINE4].read_host(many), aff[many])
        d = {f: r.diagnostics() for f, r in readers.items()}
        assert d["bf16"]["format"] == "bf16" and d["bf16"]["default"] and not d["bf16"]["approximate"]
        assert d[E.FP8_ROW]["row_bytes"] == 98 and not d[E.FP8_ROW]["approximate"]
        assert d[E.AFFINE4]["row_bytes"] == 60 and d[E.AFFINE4]["approximate"]
        assert all(v["qualified"] is False for v in d.values())
        assert readers[E.FP8_ROW].fp8_round_trip_dtype == mx.bfloat16
        assert readers[E.AFFINE4].fp8_round_trip_dtype is None and readers["bf16"].fp8_round_trip_dtype is None
        header = E.safetensors_header(tmp_path / f"t.{E.AFFINE4}.safetensors")["__metadata__"]
        assert header[E.FORMAT_KEY] == E.AFFINE4 and header["mlx2_engram_group_size"] == "32"
    finally:
        for r in readers.values():
            r.close()
    # A reader must be asked for the format the shard declares.
    with pytest.raises(ValueError, match="policy asked"):
        E.EngramTableRows(src, name, table_format=E.FP8_ROW)
    with pytest.raises(ValueError, match="policy asked"):
        E.EngramTableRows(tmp_path / f"t.{E.FP8_ROW}.safetensors", name)


def test_undeclared_fp8_or_u8_table_is_refused(tmp_path):
    for dtype in ("F8_E4M3", "U8"):
        header = json.dumps({"t": {"dtype": dtype, "shape": [4, 96], "data_offsets": [0, 384]}}).encode()
        path = tmp_path / f"{dtype}.safetensors"
        path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(384))
        with pytest.raises(ValueError, match="row scales"):
            E.EngramTableRows(path, "t")


# -------------------------------------------------------- model exactness --

def _layer_config():
    return tiny_config(engram_fp8_embedding=True, engram_n_embed_per_ngram=192)  # head_dim 96


def test_engram_layer_bf16_fp8_table_is_bit_identical_to_bf16_table(tmp_path):
    from mlx.utils import tree_flatten, tree_unflatten

    conv = _converter()
    args = H.ModelArgs.from_hf_config(_layer_config())
    layer = H.EngramLayer(args, 2)
    rng = np.random.default_rng(7)
    params = [(k, mx.array((rng.standard_normal(v.shape) * 0.3 + (1.0 if "norm" in k else 0.0))
                           .astype(np.float32)).astype(mx.bfloat16))
              for k, v in tree_flatten(layer.parameters()) if v.dtype != mx.int32]
    layer.update(tree_unflatten(params))
    x = _rows(n=200, seed=9)
    name = "model.layers.1." + TABLE
    src = write_table(tmp_path / "src.safetensors", name, x)
    conv.write_packed_table(src, name, tmp_path / "fp8.safetensors", E.FP8_ROW)
    bf16 = E.EngramTableRows(src, name)
    fp8 = E.EngramTableRows(tmp_path / "fp8.safetensors", name, table_format=E.FP8_ROW)
    try:
        streams = mx.array(rng.standard_normal((2, 6, 4, 32)).astype(np.float32)).astype(mx.bfloat16)
        idx = rng.integers(0, 200, (2, 6, 4))
        mask = mx.ones((2, 6), dtype=mx.bool_)
        for precision in ("transformers", "vllm"):
            layer.precision = precision
            layer.table_round_trip_dtype = None
            want = layer(streams, bf16.read(idx.reshape(-1)).reshape(2, 6, 4, 96), mask)
            layer.table_round_trip_dtype = fp8.fp8_round_trip_dtype
            got = layer(streams, fp8.read(idx.reshape(-1)).reshape(2, 6, 4, 96), mask)
            if precision == "transformers":
                assert mx.array_equal(got, want).item()
            else:
                # vLLM reads an FP8 table as decoded and a bf16 table without
                # the round trip: these differ by the round trip, by design.
                assert not mx.array_equal(got, want).item()
    finally:
        bf16.close()
        fp8.close()


def _bind_tables(model, config, W, tmp_path, fmt):
    conv = _converter()
    args = H.ModelArgs.from_hf_config(config)
    readers = {}
    for index in args.engram_layers:
        name = f"model.layers.{index}.{TABLE}"
        src = tmp_path / f"src-{index}.safetensors"
        if not src.exists():
            write_table(src, name, W[name])
        path = src
        if fmt != "bf16":
            path = tmp_path / f"{fmt}-{index}.safetensors"
            conv.write_packed_table(src, name, path, fmt)
        readers[index] = E.EngramTableRows(path, name, table_format=fmt)
    model.bind_engram(readers)
    return readers


def _bf16_model(config, W):
    model, _ = build(config, W)
    model.set_dtype(mx.bfloat16)
    return model


def test_bf16_model_fp8_table_equals_bf16_table_and_affine4_error(tmp_path):
    # head_dim 32; dense MLPs only: MLX's CPU GatherMM (MoE) is fp32-only.
    config = tiny_config(engram_fp8_embedding=True, engram_n_embed_per_ngram=64,
                         mlp_layer_types=["dense"] * 6)
    W = checkpoint(config, seed=2)
    model = _bf16_model(config, W)
    logits, readers = {}, []
    engram = [layer.engram_layer for layer in model.model.layers if "engram_layer" in layer]
    try:
        for fmt in ("bf16", E.FP8_ROW, E.AFFINE4):
            readers += list(_bind_tables(model, config, W, tmp_path, fmt).values())
            logits[fmt] = np.array(model(mx.array([TOKENS]))[0].astype(mx.float32))
            if fmt == "bf16":
                # Control: the references' own disagreement, the E4M3 round
                # trip off (what vLLM does with a bf16 table).
                for layer in engram:
                    layer.fp8 = False
                logits["control"] = np.array(model(mx.array([TOKENS]))[0].astype(mx.float32))
                for layer in engram:
                    layer.fp8 = True
    finally:
        for r in readers:
            r.close()
    np.testing.assert_array_equal(logits[E.FP8_ROW], logits["bf16"])
    base = logits["bf16"].astype(np.float64)
    report = {k: (np.abs(logits[k] - base).mean(), kl(base, logits[k]).mean(),
                  (logits[k].argmax(-1) == base.argmax(-1)).mean()) for k in ("control", E.AFFINE4)}
    for k, (d, k_l, agree) in report.items():
        print(f"bf16 model, {k} vs bf16 table: mean |dlogit| {d:.4f}, mean KL {k_l:.5f} nat, argmax agree {agree:.3f}")
    # Measured (seeds 2/3/4): affine4 mean KL 0.081/0.015/0.056, 2-4x the control.
    assert report[E.AFFINE4][1] < 0.15
    assert report[E.AFFINE4][0] < 6 * report["control"][0]


def kl(p, q):
    """Mean-able per-row KL(p || q) of two logit arrays, in nats."""
    def logsoftmax(z):
        z = np.asarray(z, np.float64)
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))
    a, b = logsoftmax(p), logsoftmax(q)
    return (np.exp(a) * (a - b)).sum(-1)


def test_fp8_table_refused_when_config_has_no_fp8_round_trip(tmp_path):
    config = tiny_config(engram_fp8_embedding=True, engram_n_embed_per_ngram=64)
    W = checkpoint(config, seed=2)
    model, _ = build(config, W)
    model.args.engram_fp8_embedding = False
    with pytest.raises(ValueError, match="engram_fp8_embedding"):
        _bind_tables(model, config, W, tmp_path, E.FP8_ROW)


# ------------------------------------------------- converter and adapter --

FORMATS = "bf16,fp8_e4m3_row,affine4_g32"


def _source(tmp_path):
    """Tiny source with the layer-1 table beside trunk tensors (shard 1) and
    the layer-4 table alone (shard 2), so a resume crosses a finished table."""
    from safetensors.numpy import save_file

    config = tiny_config(routed_expert_hidden_size=32, moe_intermediate_size=32,
                         engram_fp8_embedding=True, engram_n_embed_per_ngram=64)
    W = checkpoint(config, seed=4, scale=0.2)
    src = tmp_path / "src"
    src.mkdir()
    last = f"model.layers.4.{TABLE}"
    shards = {"model-00001.safetensors": {k: v for k, v in W.items() if k != last},
              "model-00002.safetensors": {last: W[last]}}
    for name, tensors in shards.items():
        save_file(tensors, str(src / name))
    (src / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: n for n, t in shards.items() for k in t}}))
    (src / "config.json").write_text(json.dumps(config))
    return config, W, src


def test_converter_writes_formats_and_each_loads(tmp_path):
    from mlx2.adapters import hy_v4_flash as A

    conv = _converter()
    config, W, src = _source(tmp_path)
    out = tmp_path / "out"
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=False, engram_formats=FORMATS)
    artifact = A.inspect_artifact(out)
    declared = artifact["config"]["mlx2_engram_formats"]
    assert sorted(declared) == sorted(FORMATS.split(","))
    name = f"model.layers.1.{TABLE}"
    assert artifact["weight_map"][name] == "engram-1.safetensors" == declared["bf16"][name]
    assert declared[E.FP8_ROW][name] == "engram-1.fp8_e4m3_row.safetensors"
    sizes = {f: (out / declared[f][name]).stat().st_size for f in declared}
    print("tiny table shard bytes:", sizes)
    logits = {}
    for fmt in declared:
        model, _, readers = A.load_model(out, artifact, engram_table_format=fmt)
        try:
            logits[fmt] = np.array(model(mx.array([TOKENS]))[0])
            assert {r.format for r in readers.values()} == {fmt}
        finally:
            for r in readers.values():
                r.close()
    # fp32 tiny model, F32 source tables: the fp8 table's round trip dtype is
    # the activation dtype, so the model skips its own round trip: exact.
    np.testing.assert_array_equal(logits[E.FP8_ROW], logits["bf16"])
    diff = np.abs(logits[E.AFFINE4] - logits["bf16"])
    divergence = kl(logits["bf16"], logits[E.AFFINE4]).mean()
    print(f"fp32 converted model, affine4 vs bf16: max |dlogit| {diff.max():.4f}, "
          f"mean {diff.mean():.4f}, mean KL {divergence:.5f} nat")
    assert divergence < 0.15
    with pytest.raises(ValueError, match="engram_table_format must"):
        A._parse_policy({"engram_table_format": "int2"})
    assert A._parse_policy({})[0].get("engram_table_format") is None  # default bf16


def test_converter_resume_after_failed_table_and_format_mismatch(tmp_path, monkeypatch):
    conv = _converter()
    _, _, src = _source(tmp_path)
    fresh, out = tmp_path / "fresh", tmp_path / "out"
    conv.convert(src, fresh, group_size=32, expert_bits=8, trunk_bits=8, resume=False, engram_formats=FORMATS)
    real = conv._pack_rows
    calls = []

    def flaky(*a, **k):
        calls.append(a[1])
        if len(calls) == 3:  # shard 2's first packed table, header already written
            raise OSError("simulated crash mid-table")
        return real(*a, **k)

    monkeypatch.setattr(conv, "_pack_rows", flaky)
    with pytest.raises(OSError, match="simulated"):
        conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=False, engram_formats=FORMATS)
    monkeypatch.setattr(conv, "_pack_rows", real)
    assert list(out.glob("*.partial"))  # the crash left a partial table
    assert len((out / "convert-progress.jsonl").read_text().splitlines()) == 1
    with pytest.raises(SystemExit, match="engram-formats"):
        conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=True, engram_formats="bf16")
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=True, engram_formats=FORMATS)
    for f in ("config.json", "model.safetensors.index.json"):
        a, b = json.loads((out / f).read_text()), json.loads((fresh / f).read_text())
        a.pop("mlx2_source", None), b.pop("mlx2_source", None)
        assert a == b, f
    for shard in sorted(p.name for p in fresh.glob("engram-*")):
        assert (out / shard).read_bytes() == (fresh / shard).read_bytes(), shard
    assert not list(out.glob("*.partial"))
    before = (out / "model.safetensors.index.json").read_text()
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=True, engram_formats=FORMATS)
    assert (out / "model.safetensors.index.json").read_text() == before


def test_converter_without_bf16_and_default_policy_fails_closed(tmp_path):
    from mlx2.adapters import hy_v4_flash as A

    conv = _converter()
    _, _, src = _source(tmp_path)
    out = tmp_path / "out"
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=False, engram_formats="affine4_g32")
    artifact = A.inspect_artifact(out)
    assert artifact["weight_map"][f"model.layers.1.{TABLE}"] == "engram-1.affine4_g32.safetensors"
    with pytest.raises(ValueError, match="no bf16 Engram tables"):
        A.load_model(out, artifact)
    model, _, readers = A.load_model(out, artifact, engram_table_format="affine4_g32")
    for r in readers.values():
        r.close()
    with pytest.raises(SystemExit, match="engram-formats"):
        conv.parse_formats("bf16,bf16")


def test_adapter_diagnostics_label_the_table_format(tmp_path):
    from tokenizers import Tokenizer, models

    from mlx2.adapters import hy_v4_flash as A

    conv = _converter()
    config, _, src = _source(tmp_path)
    out = tmp_path / "out"
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=False, engram_formats=FORMATS)
    vocab = {f"t{i}": i for i in range(config["text_config"]["vocab_size"])}
    Tokenizer(models.WordLevel(vocab, unk_token="t0")).save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t2"}))
    for fmt, approximate in (("bf16", False), (E.AFFINE4, True)):
        policy = None if fmt == "bf16" else {"engram_table_format": fmt}
        adapter = A.HYV4FlashAdapter(str(out), execution_policy=policy)
        try:
            d = adapter.diagnostics()
            record = d["engram_table_format"]
            assert record["format"] == fmt and record["default"] == (fmt == "bf16")
            assert record["qualified"] is False and record["approximate"] is approximate
            assert all(t["format"] == fmt and t["approximate"] is approximate for t in d["engram"].values())
        finally:
            adapter.close()


def test_resume_of_a_pre_format_progress_file_keeps_its_bf16_tables(tmp_path):
    from mlx2.adapters import hy_v4_flash as A

    conv = _converter()
    _, _, src = _source(tmp_path)
    out = tmp_path / "out"
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=False)
    progress = out / "convert-progress.jsonl"
    first = json.loads(progress.read_text().splitlines()[0])
    for key in ("engram_formats", "tables"):
        first.pop(key)  # as the converter wrote it before --engram-formats
    progress.write_text(json.dumps(first) + "\n")
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=True)
    declared = json.loads((out / "config.json").read_text())["mlx2_engram_formats"]
    assert sorted(declared["bf16"]) == [f"model.layers.{i}.{TABLE}" for i in (1, 4)]
    model, _, readers = A.load_model(out, A.inspect_artifact(out))
    for r in readers.values():
        r.close()


def test_format_shards_enter_the_fingerprint_and_stay_inside(tmp_path):
    from mlx2.adapters import hy_v4_flash as A

    conv = _converter()
    _, _, src = _source(tmp_path)
    out = tmp_path / "out"
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=False, engram_formats=FORMATS)
    before = A.inspect_artifact(out)["identity"]
    assert "engram-1.affine4_g32.safetensors" in {f[0] for f in before["files"]}
    shard = out / "engram-1.affine4_g32.safetensors"
    shard.write_bytes(shard.read_bytes() + b"\0" * 8)
    assert A.inspect_artifact(out)["identity"]["fingerprint"] != before["fingerprint"]
    config = json.loads((out / "config.json").read_text())
    config["mlx2_engram_formats"]["affine4_g32"][f"model.layers.1.{TABLE}"] = "../escape.safetensors"
    (out / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="escaped Engram shard"):
        A.inspect_artifact(out)


def test_fp8_table_dtype_mismatch_and_vllm_labels():
    from mlx2.adapters import hy_v4_flash as A

    config = tiny_config(engram_fp8_embedding=True, engram_n_embed_per_ngram=64,
                         mlp_layer_types=["dense"] * 6)
    W = checkpoint(config, seed=2)
    model, _ = build(config, W)  # fp32 model

    class Bf16Fp8Table:
        rows = None
        fp8_round_trip_dtype = mx.bfloat16

    with pytest.raises(ValueError, match="round trip in"):
        model.bind_engram({i: Bf16Fp8Table() for i in model.args.engram_layers})
    assert A.engram_format_record("fp8_e4m3_row")["approximate"] is False
    assert A.engram_format_record("fp8_e4m3_row", "vllm")["approximate"] is True
    assert A.engram_format_record("affine4_g32")["approximate"] is True
    assert "unqualified; approximate" in A.descriptor_for(engram_table_format="affine4_g32").metadata["engram_tables"]
