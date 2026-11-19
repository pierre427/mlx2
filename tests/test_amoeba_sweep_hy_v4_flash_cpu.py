"""CPU tests of the hy_v4_flash sweep family (research/amoeba/sweep/hy_v4_flash.py) on a tiny synthetic checkpoint.

The checkpoint is tests/test_hy_v4_flash_cpu.py's tiny Hy4-flash (6 layers: indexed and sliding MLA, dense layer 0,
latent MoE, two Engram layers), written as safetensors in bf16 (the real checkpoint's dtype) and in float32, plus an
MLX affine artifact from scripts/convert_hy_v4_flash_mlx.py. The streamed layer-major forward must equal the ordinary
whole-model forward bit for bit: against the family's own reference and against the adapter's ``load_model``.
"""
import importlib.util
import json
import os
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hy4_tiny", ROOT / "tests" / "test_hy_v4_flash_cpu.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

from research.amoeba.sweep import hy_v4_flash as HF

TEXTS = [T.TOKENS, [t + 1 for t in T.TOKENS], list(reversed(T.TOKENS))]


def _write(dirname, config, W, dtype):
    """The tiny checkpoint with real-checkpoint layout: Engram tables alone in their shards, the rest split."""
    dirname.mkdir(parents=True, exist_ok=True)
    tables = sorted(k for k in W if k.endswith(HF.TABLE))
    rest = sorted(k for k in W if k not in tables)
    shards = {f"model-0000{i + 1}.safetensors": {k: W[k]} for i, k in enumerate(tables)}
    half = len(rest) // 2
    shards["model-00008.safetensors"] = {k: W[k] for k in rest[:half]}
    shards["model-00009.safetensors"] = {k: W[k] for k in rest[half:]}
    for name, tensors in shards.items():
        # routed experts stay float32: MLX's CPU GatherMM is float32-only (Metal runs them in bf16)
        arrs = {k: (mx.array(v) if v.dtype.kind in "iu" or ".mlp.experts." in k else mx.array(v).astype(dtype))
                for k, v in tensors.items()}
        mx.save_safetensors(str(dirname / name), arrs)
    (dirname / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: n for n, t in shards.items() for k in t}}))
    (dirname / "config.json").write_text(json.dumps(config))
    return dirname


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    config = T.tiny_config()
    W = T.checkpoint(config)
    root = tmp_path_factory.mktemp("hy4")
    return {"config": config, "W": W, "bf16": _write(root / "bf16", config, W, mx.bfloat16),
            "f32": _write(root / "f32", config, W, mx.float32)}


@pytest.fixture(autouse=True)
def _env():
    with mock.patch.dict(os.environ):         # the family applies the adapter's process profile
        yield


def _streamed(fam, ids, layers=None):
    carry = fam.embed(ids)
    pays = []
    for l in range(layers or fam.L):
        fam.begin_layer(l)
        carry, pay = fam.run_layer(l, carry)
        fam.release_layer(l)
        pays.append(pay)
    return fam.final_logits(carry), carry, pays


def _adapter_logits(path, ids):
    from mlx2.adapters import hy_v4_flash as A
    model, _, readers = A.load_model(path, A.inspect_artifact(path))
    try:
        out = model(ids)
        mx.eval(out)
        return np.array(out.astype(mx.float32))
    finally:
        for r in readers.values():
            r.close()


@pytest.mark.parametrize("kind", ["bf16", "f32"])
@pytest.mark.parametrize("n", [24, 6])        # 24 > candidate_topk 8: the indexer path; 6: dense causal
def test_streamed_equals_ordinary_bitwise(art, kind, n):
    fam = HF.load(art[kind])
    try:
        ids = mx.array([t[:n] for t in TEXTS], dtype=mx.int32)
        got, carry, _ = _streamed(fam, ids)
        ref = fam.reference_logits(ids)
        mx.eval(got, ref)
        assert carry["h"].dtype == (mx.bfloat16 if kind == "bf16" else mx.float32)
        np.testing.assert_array_equal(np.array(got), np.array(ref))
        np.testing.assert_array_equal(np.array(got.astype(mx.float32)), _adapter_logits(art[kind], ids))
        assert fam.counters["layer_loads"] == fam.L       # each layer read once
    finally:
        fam.close()


def test_reference_of_first_layers_equals_streamed_prefix(art):
    fam = HF.load(art["bf16"])
    try:
        ids = mx.array([t[:20] for t in TEXTS], dtype=mx.int32)
        for k in (1, 2, 4):                     # 2: past the first Engram layer (index 1)
            got, _, _ = _streamed(fam, ids, layers=k)
            ref = fam.reference_logits(ids, layers=k)
            np.testing.assert_array_equal(np.array(got), np.array(ref))
    finally:
        fam.close()


def test_engram_args_equal_model_engram_inputs(art):
    fam = HF.load(art["f32"])
    try:
        tokens = np.array([t[:13] for t in TEXTS], np.int64)
        full = fam.model.engram_inputs(tokens, 0, fam.model.make_cache())
        for l in fam.args.engram_layers:
            rows, mask = fam._engram(l, tokens)
            np.testing.assert_array_equal(np.array(rows), np.array(full[l][0]))
            np.testing.assert_array_equal(np.array(mask), np.array(full[l][1]))
    finally:
        fam.close()


@pytest.mark.parametrize("budget_gb", [8.0, 0.0])      # rows in memory, rows in a scratch file
def test_engram_preread_equals_reader(art, tmp_path, budget_gb):
    fam = HF.load(art["bf16"], engram_preread_gb=budget_gb)
    try:
        seqs = [t[:n] for t, n in zip(TEXTS, (24, 17, 9))]
        ids = mx.array([t[:9] for t in TEXTS], dtype=mx.int32)
        carry = fam.embed(ids)
        carry0, _ = fam.run_layer(0, carry)
        l = min(fam.args.engram_layers)
        plain, pay_plain = fam.run_layer(l, carry0)
        rec = fam.prepare_layer(l, seqs, tmp_path / ".prepare")
        assert rec["engram_preread_rows"] > 0 and rec["on_disk"] == (budget_gb == 0.0)
        tokens = np.array([t[:9] for t in TEXTS], np.int64)
        rows = fam._engram_rows(l, tokens).reshape(-1)
        np.testing.assert_array_equal(np.array(fam._preread[l].read(rows)), np.array(fam.model.engram_rows[l].read(rows)))
        pre, pay_pre = fam.run_layer(l, carry0)
        np.testing.assert_array_equal(np.array(pre["h"].astype(mx.float32)), np.array(plain["h"].astype(mx.float32)))
        np.testing.assert_array_equal(np.array(pay_pre["h"].astype(mx.float32)), np.array(pay_plain["h"].astype(mx.float32)))
        with pytest.raises(RuntimeError, match="not pre-read"):
            fam._preread[l].read(np.array([int(fam._preread[l].rows.max()) + 1]))
        assert fam.prepare_layer(0, seqs, tmp_path / ".prepare") is None
        fam.release_layer(l)
        assert l not in fam._preread and not list((tmp_path / ".prepare").glob("engram_*")) if budget_gb == 0.0 else True
    finally:
        fam.close()


def test_preread_failure_leaves_no_scratch_and_reference_budget_refuses(art, tmp_path, monkeypatch):
    fam = HF.load(art["f32"], engram_preread_gb=0.0, reference_gb=1e-9)
    try:
        ids = mx.array([t[:9] for t in TEXTS], dtype=mx.int32)
        with pytest.raises(SystemExit, match="reference_gb"):
            fam.reference_logits(ids, layers=2)

        def broken(*a, **k):
            raise OSError("disk gone")
        monkeypatch.setattr(os, "preadv", broken)
        with pytest.raises(OSError, match="disk gone"):
            fam.prepare_layer(min(fam.args.engram_layers), [t[:9] for t in TEXTS], tmp_path / ".prepare")
        assert not list((tmp_path / ".prepare").glob("engram_*"))
    finally:
        fam.close()


def _oracle_h(fam, l, x_latent, inds):
    """The clamped SwiGLU activation per routed slot, from the layer's own expert weights (float32)."""
    layer = fam._load_layer(l)
    sw = layer.mlp.experts
    lim = layer.mlp.experts.activation.limit
    g = np.array(sw.gate_proj.weight.astype(mx.float32))
    u = np.array(sw.up_proj.weight.astype(mx.float32))
    x = np.array(x_latent.astype(mx.float32))
    idx = np.array(inds)
    gate = np.einsum("btd,btkfd->btkf", x, g[idx])
    up = np.einsum("btd,btkfd->btkf", x, u[idx])
    gate = np.minimum(gate, lim)
    return gate / (1 + np.exp(-gate)) * np.clip(up, -lim, lim)


@pytest.mark.parametrize("sort_min", [0, 1 << 30])     # sorted gather, unsorted gather
def test_neuron_units_in_slot_order(art, monkeypatch, sort_min):
    from mlx2.runtime.models import switch_layers as S
    monkeypatch.setattr(S, "_GATHER_SORT_MIN_ASSIGNMENTS", sort_min)
    fam = HF.load(art["f32"])
    try:
        ids = mx.array([t[:16] for t in TEXTS], dtype=mx.int32)
        carry = fam.embed(ids)
        carry, pay0 = fam.run_layer(0, carry)
        assert set(pay0) == {"act"} and pay0["act"].shape == (3, 16, fam.args.intermediate_size)
        seen = {}
        layer = fam._load_layer(1)
        base = type(layer.mlp.experts)

        def spy(self_, x, indices):
            seen["x"] = x
            return base.__call__(self_, x, indices)
        layer.mlp.experts.__class__ = type("Spy", (base,), {"__call__": spy})
        try:
            _, pay = fam.run_layer(1, carry)
        finally:
            layer.mlp.experts.__class__ = base
        k, F = fam.args.num_experts_per_tok, fam.F
        assert pay["h"].shape == (3, 16, k, F) and pay["F"] == F and fam.widths[1] == fam.args.n_routed_experts * F
        np.testing.assert_allclose(np.array(pay["h"]), _oracle_h(fam, 1, seen["x"], pay["inds"]), rtol=1e-4, atol=1e-5)
        from mlx2.experimental.amoeba.child_rolling import payload_signal
        sig = np.array(payload_signal("neurons", pay, fam.widths[1]))
        assert sig.shape == (3, fam.widths[1]) and sig.max() > 0
    finally:
        fam.close()


def test_expert_units(art):
    fam = HF.load(art["f32"], moe_units="expert")
    try:
        assert fam.kinds[0] == "dense" and set(fam.kinds[1:]) == {"experts"}
        assert fam.widths[1] == fam.args.n_routed_experts
        ids = mx.array([t[:12] for t in TEXTS], dtype=mx.int32)
        carry, _ = fam.run_layer(0, fam.embed(ids))
        _, pay = fam.run_layer(1, carry)
        k = fam.args.num_experts_per_tok
        assert pay["inds"].shape == (3, 12, k) and pay["rows"].shape[:3] == (3, 12, k)
        from mlx2.experimental.amoeba.child_rolling import payload_signal
        assert np.array(payload_signal("experts", pay, fam.widths[1])).shape == (3, fam.widths[1])
    finally:
        fam.close()


def test_quantized_artifact_streamed_equals_adapter(art, tmp_path):
    spec = importlib.util.spec_from_file_location("convert_hy_v4_flash_mlx", ROOT / "scripts" / "convert_hy_v4_flash_mlx.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    config = T.tiny_config(routed_expert_hidden_size=32, moe_intermediate_size=32)
    W = T.checkpoint(config, seed=4, scale=0.2)
    (tmp_path / "src").mkdir()
    src = T._write_artifact(tmp_path / "src", config, W)
    out = tmp_path / "q8"
    conv.convert(src, out, group_size=32, expert_bits=8, trunk_bits=8, resume=False)
    fam = HF.load(out)
    try:
        assert fam.quant is not None
        ids = mx.array([t[:20] for t in TEXTS], dtype=mx.int32)
        got, _, _ = _streamed(fam, ids)
        np.testing.assert_array_equal(np.array(got.astype(mx.float32)), _adapter_logits(out, ids))
        np.testing.assert_array_equal(np.array(got), np.array(fam.reference_logits(ids)))
    finally:
        fam.close()


def test_rolling_driver_parity_and_one_read_per_chunk(art, tmp_path):
    from mlx2.experimental.amoeba import child_rolling as R
    fam = HF.load(art["bf16"])
    try:
        rng = np.random.default_rng(3)
        seqs = [list(rng.integers(4, 90, size=n)) for n in (14, 14, 14, 11, 11, 9)]
        par = R.parity(fam, seqs[:3], None, log=lambda *_: None)
        assert par["ok"] and all(t["bitwise_equal"] for t in par["texts"])
        par2 = R.parity(fam, seqs[:3], 2, log=lambda *_: None)
        assert par2["ok"] and all(t["bitwise_equal"] for t in par2["texts"]) and par2["layers"] == 2
        loads0 = fam.counters["layer_loads"]
        pre = R.Prefetcher(fam.prefetch_ranges)
        try:
            outs = {}
            for chunks, carry in ((1, "resident"), (2, "memmap")):
                d = tmp_path / f"c{chunks}"
                d.mkdir()
                before = fam.counters["layer_loads"]
                rec, mu, nll = R.run(fam, seqs, [True, False, True, True, False, True], d, window=2, chunks=chunks,
                                     carry_mode=carry, prefetch=pre, max_texts=2, log=lambda *_: None)
                assert fam.counters["layer_loads"] - before == fam.L * rec["chunks"]
                outs[chunks] = ([np.load(d / f"resp_L{l:02d}.npy") for l in range(fam.L)], mu, nll)
            assert pre.bytes > 0
        finally:
            pre.close()
        assert loads0 > 0
        for a, b in zip(outs[1][0], outs[2][0]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(outs[1][1], outs[2][1])
        np.testing.assert_array_equal(outs[1][2], outs[2][2])
        assert outs[1][0][1].shape == (6, fam.widths[1]) and np.isfinite(outs[1][2]).all()
    finally:
        fam.close()


def test_prefetch_ranges_cover_layer_tensors_not_engram_tables(art):
    fam = HF.load(art["bf16"])
    try:
        pr = fam.prefetch_ranges
        assert sorted(pr) == list(range(fam.L))
        for l, spans in pr.items():
            total = sum(e - b for _, b, e in spans)
            assert total == sum(fam.tensors[n].end - fam.tensors[n].begin for n in fam._names[l])
            assert not any("model-0000" in f and f.endswith(("1.safetensors", "2.safetensors")) for f, _, _ in spans)
    finally:
        fam.close()


def test_refusals(art, tmp_path):
    import shutil
    fam = HF.load(art["f32"])
    try:
        img = fam.args.image_token_id
        if img is not None:
            with pytest.raises(ValueError, match="image"):
                fam.embed(mx.array([[5, img, 7]], dtype=mx.int32))
        with pytest.raises(ValueError):
            HF.load(art["f32"], moe_units="row")
    finally:
        fam.close()
    partial = tmp_path / "partial"
    shutil.copytree(art["f32"], partial)
    (partial / "model-00009.safetensors").unlink()           # a download still in progress
    with pytest.raises(SystemExit, match="missing"):
        HF.load(partial)
    from mlx2.runtime.models import switch_grouped_experts as G
    with mock.patch.object(G, "MODE", "q4"), pytest.raises(SystemExit, match="grouped-expert"):
        HF.load(art["f32"])


def test_image_tokens_refused_with_image_config(art, tmp_path):
    import shutil
    d = tmp_path / "img"
    shutil.copytree(art["f32"], d)
    cfg = json.loads((d / "config.json").read_text())
    cfg["image_token_id"] = 3
    (d / "config.json").write_text(json.dumps(cfg))
    fam = HF.load(d)
    try:
        with pytest.raises(ValueError, match="image"):
            fam.run_layer(0, {"ids": mx.array([[5, 3, 7]], dtype=mx.int32), "h": mx.zeros((1, 3, 4, 32))})
    finally:
        fam.close()


def test_registered_by_model_type_and_inspected(art):
    from mlx2.experimental.amoeba import child_rolling as R
    from mlx2.experimental.amoeba.inspection import inspect_checkpoint
    from research.amoeba.sweep.family import FAMILIES
    assert FAMILIES["hy_v4_flash"] == "research.amoeba.sweep.hy_v4_flash:load"
    assert R.family_takes_moe_units("hy_v4_flash") and not R.family_takes_moe_units("mimo_v2")
    info = inspect_checkpoint(art["bf16"])
    assert info["sweep_family"] == "hy_v4_flash" and info["residual_streams"] == 4
    lb = info["layer_bytes"]
    assert lb["side_tables"] > 0 and lb["layered"]
    assert info["fingerprint"] == HF.checkpoint_fingerprint(art["bf16"])


def test_plan_gives_hy4_a_read_ahead_window(monkeypatch):
    from mlx2.experimental.amoeba import config as K
    from mlx2.experimental.amoeba import plan as P
    monkeypatch.setattr(P, "physical_gb", lambda: 128.0)
    L, unit = 42, int(7.57e9)
    info = {"path": "/m", "checkpoint": "Hy4-flash@0", "fingerprint": "0" * 64, "adapter": "HYV4FlashAdapter",
            "kind": "moe", "family": "hy-v4-flash", "model_type": "hy_v4_flash", "sweep_family": "hy_v4_flash",
            "layers": L, "hidden": 3072, "intermediate": 13824, "vocab": 120832, "residual_streams": 4,
            "moe": {"experts": 512, "expert_intermediate": 1536, "moe_layers": 41, "top_k": 16},
            "file_backed_ple": False, "weight_bytes": int(445e9),
            "units": {"dense_neurons": 13824, "moe_expert_units": 512 * 41, "moe_neuron_units": 512 * 1536 * 41,
                      "estimate": True},
            "layer_bytes": {"per_layer": [unit] * L, "max_layer": unit, "rest": int(5e9), "side_tables": int(127e9),
                            "experts_per_layer": [unit] * L, "max_layer_experts": unit, "layered": True}}
    st = P.make_plan(info, K.resolve("quick", {}, {}, ""), texts=4017, moe_units="expert")["stream"]
    assert st["family"] == "hy_v4_flash" and st["carry"] == "memmap" and 1 <= st["window"] <= 8


class _CharTok:
    def __init__(self, path):
        pass

    def encode(self, text):
        return [4 + (ord(c) % 90) for c in text]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


@pytest.mark.parametrize("units", ["neuron", "expert"])
def test_child_rolling_main_end_to_end(art, tmp_path, monkeypatch, units):
    from mlx2.experimental.amoeba import child_rolling as R
    monkeypatch.setattr(HF, "_Tok", _CharTok)
    from research.amoeba.carve import ops as O
    monkeypatch.setattr(O, "footprint_watchdog", lambda *a, **k: None)   # it exits the process past its limit
    corpus = tmp_path / "corpus.jsonl"
    rng = np.random.default_rng(5)
    with corpus.open("w") as f:
        for i in range(5):
            text = "".join(chr(97 + int(c)) for c in rng.integers(0, 26, size=14 + i % 2))
            f.write(json.dumps({"id": f"t{i}", "topics": ["x"], "split": "fit" if i % 2 == 0 else "val",
                                "text": text}) + "\n")
    out = tmp_path / "par"
    R.main(["--family", "hy_v4_flash", "--model", str(art["f32"]), "--corpus", str(corpus), "--out", str(out),
            "--window", "2", "--seq-len", "16", "--min-len", "8", "--bs", "2", "--parity-layers", "3",
            "--parity-only", "--moe-units", units, "--cpu"])
    par = json.loads((out / "parity.json").read_text())
    assert par["ok"] and par["layers"] == 3 and json.loads((out / "parity-run.json").read_text())["parity_only"]
    assert not (out / "meta.json").exists()
    cap = tmp_path / "cap"
    R.main(["--family", "hy_v4_flash", "--model", str(art["f32"]), "--corpus", str(corpus), "--out", str(cap),
            "--window", "2", "--seq-len", "16", "--min-len", "8", "--bs", "2", "--moe-units", units, "--cpu"])
    meta = json.loads((cap / "meta.json").read_text())
    assert meta["moe_units"] == units and meta["parity"]["ok"] and meta["family"] == "hy_v4_flash"
    assert meta["weight_reads_observed"]["once_per_chunk"] and meta["rolling"]["prefetched_bytes"] > 0
    E, F = 8, 16
    assert meta["widths"][1] == (E * F if units == "neuron" else E) and meta["unit_kinds"][0] == "dense"
