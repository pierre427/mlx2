"""mlx2.layer_kl: per-layer KL sensitivity (CPU, tiny Hy4-flash fixtures).

The tool is model-agnostic; these tests drive it through the tiny Hy4-flash
checkpoint (test_hy_v4_flash_cpu), its tiny MLX pack layout
(test_hy_v4_flash_mlx_pack_cpu, 4-bit experts, q2_0 on layers 1-2) and the
layer-streaming sweep family (research/amoeba/sweep/hy_v4_flash.py).
"""
import json
import os
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import layer_kl as K  # noqa: E402
from test_hy_v4_flash_cpu import LONG, build, checkpoint, tiny_config  # noqa: E402
from test_hy_v4_flash_mlx_pack_cpu import pack, pack_config  # noqa: E402,F401

TEXTS = [{"id": "a", "split": "dev", "ids": LONG[0:28]},
         {"id": "b", "split": "dev", "ids": LONG[30:54]},
         {"id": "c", "split": "held", "ids": LONG[60:84]}]


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, _ = build(config, W)
    return config, W, model


@pytest.fixture(scope="module")
def packed(pack):  # noqa: F811
    from mlx2.adapters import hy_v4_flash as A

    root, config, ref = pack
    model, _, readers = A.load_model(root, A.inspect_artifact(root), engram_table_format="iq4_nl")
    yield root, config, ref, model
    for r in readers.values():
        r.close()


def _reference(tmp_path, model, name="ref", texts=TEXTS, skip=4):
    out = tmp_path / name
    K.write_reference(out, texts, K.reference_from_model(model), skip=skip,
                      checkpoint="tiny@sha256:" + "0" * 64, source={"kind": "test"})
    return out


def test_model_against_its_own_reference_is_exactly_zero(tmp_path, tiny):
    _, _, model = tiny
    ref = _reference(tmp_path, model)
    meta, texts = K.load_reference(ref)
    s = K.score(model, ref, meta, texts, splits=("dev", "held"))
    assert s.kl_mean == 0.0 and s.top1 == 1.0
    assert s.positions == sum(len(t["ids"]) - 4 for t in TEXTS)
    assert {row["id"] for row in s.per_text} == {"a", "b", "c"}
    dev = K.score(model, ref, meta, texts)
    assert dev.positions == sum(len(t["ids"]) - 4 for t in TEXTS[:2])
    with pytest.raises(ValueError, match="not empty"):
        _reference(tmp_path, model)


def test_tampered_texts_are_refused(tmp_path, tiny):
    _, _, model = tiny
    ref = _reference(tmp_path, model)
    texts = json.loads((ref / "texts.json").read_text())
    texts[0]["ids"][3] += 1
    (ref / "texts.json").write_text(json.dumps(texts))
    with pytest.raises(ValueError, match="sha256"):
        K.load_reference(ref)


def test_pack_scan_ranks_layers_restores_exactly_and_refuses_fixed_formats(tmp_path, packed):
    root, config, ref_weights, model = packed
    oracle, _ = build(pack_config(), {**ref_weights, **{k: v for k, v in checkpoint(pack_config(), seed=7, scale=0.2).items()
                                                        if k.startswith("model.dfly_layers.")}})
    ref = _reference(tmp_path, oracle)
    result = K.scan(model, ref, layers=[3, 4, 5], groups=["mlp.experts"], bits=2, group_size=32)
    base = result["baseline"]["kl_mean"]
    assert base > 0  # the 4-bit pack against its dequantized float model is not exact
    assert result["restore_check"]["exact"]
    for row in result["rows"]:
        assert row["modules"] == [f"model.layers.{row['layer']}.mlp.experts.{p}_proj"
                                  for p in ("gate", "up", "down")] or sorted(row["modules"]) == sorted(
            f"model.layers.{row['layer']}.mlp.experts.{p}_proj" for p in ("gate", "up", "down"))
        assert row["bytes_after"] < row["bytes_before"]
        assert row["delta_kl"] > 0 and np.isfinite(row["kl_mean"])
    assert [r["layer"] for r in K.ranked(result["rows"])] == sorted(
        [3, 4, 5], key=lambda l: next(r["delta_kl"] for r in result["rows"] if r["layer"] == l))
    # The pack's q2_0 banks (layers 1-2) have a fixed format: refused, nothing changed.
    with pytest.raises(ValueError, match="fixed quantized format"):
        K.scan(model, ref, layers=[1], groups=["mlp.experts"], bits=4, group_size=32)
    again = K.scan(model, ref, layers=[3], groups=["mlp.experts"], bits=2, group_size=32)
    assert again["baseline"]["kl_mean"] == base and again["rows"][0]["kl_mean"] == result["rows"][0]["kl_mean"]


def test_cumulative_keeps_each_layer_then_restores(tmp_path, packed):
    _, _, _, model = packed
    ref = _reference(tmp_path, model)
    single = K.scan(model, ref, layers=[3, 4], groups=["mlp.experts"], bits=2, group_size=32)
    joint = K.scan(model, ref, layers=[3, 4], groups=["mlp.experts"], bits=2, group_size=32, cumulative=True)
    assert single["baseline"]["kl_mean"] == 0.0 == joint["baseline"]["kl_mean"]
    assert joint["rows"][0]["kl_mean"] == single["rows"][0]["kl_mean"]
    assert joint["rows"][1]["kl_mean"] != single["rows"][1]["kl_mean"]  # layers 3+4 together
    assert joint["restore_check"]["exact"] and single["restore_check"]["exact"]


def test_dense_modules_are_quantized_by_module_swap_and_restored(tmp_path, tiny):
    _, _, model = tiny
    ref = _reference(tmp_path, model)
    before = model.model.layers[2].self_attn.q_a_proj
    result = K.scan(model, ref, layers=[2], groups=["self_attn.q_a_proj"], bits=4, group_size=32)
    assert result["rows"][0]["delta_kl"] > 0 and result["restore_check"]["exact"]
    assert model.model.layers[2].self_attn.q_a_proj is before


def test_source_checkpoint_weights_go_through_sanitize(tmp_path, packed, tiny):
    from test_hy_v4_flash_cpu import _write_artifact

    _, _, ref_weights, model = packed
    (tmp_path / "src").mkdir()
    source_dir = _write_artifact(tmp_path / "src", pack_config(),
                                 {k: v for k, v in checkpoint(pack_config(), seed=7, scale=0.2).items()})
    ref = _reference(tmp_path, model)
    source = K.checkpoint_source(source_dir, model)
    w = source("model.layers.3.mlp.experts.gate_proj")
    assert tuple(w.shape) == tuple(K._dense_weight(mx, model.model.layers[3].mlp.experts.gate_proj).shape)
    from_source = K.scan(model, ref, layers=[3], groups=["mlp.experts"], bits=2, group_size=32, source=source)
    from_current = K.scan(model, ref, layers=[3], groups=["mlp.experts"], bits=2, group_size=32)
    assert from_source["settings"]["source"] == "checkpoint" and from_source["restore_check"]["exact"]
    assert from_source["rows"][0]["kl_mean"] != from_current["rows"][0]["kl_mean"]


def test_streamed_family_reference_equals_the_ordinary_forward(tmp_path, tiny):
    from research.amoeba.sweep import hy_v4_flash as HF

    from test_amoeba_sweep_hy_v4_flash_cpu import _write

    config, W, model = tiny
    path = _write(tmp_path / "f32", config, W, mx.float32)
    with mock.patch.dict(os.environ):
        family = HF.load(path)
        try:
            streamed = tmp_path / "streamed"
            K.write_reference(streamed, TEXTS, K.reference_from_family(family, TEXTS), skip=4,
                              checkpoint=family.checkpoint, source={"kind": "sweep_family"})
        finally:
            family.close()
    meta, texts = K.load_reference(streamed)
    s = K.score(model, streamed, meta, texts, splits=("dev", "held"))
    assert s.kl_mean < 1e-6 and s.top1 == 1.0


def test_parse_layers():
    assert K.parse_layers("3-5,9") == [3, 4, 5, 9]
    with pytest.raises(ValueError):
        K.parse_layers("3,3")
    with pytest.raises(ValueError):
        K.parse_layers("5-3")


def test_cli_texts_reference_scan_end_to_end(tmp_path):
    import importlib.util

    from test_hy_v4_flash_vision_cpu import _vision_artifact, vision_config, vit_checkpoint

    config = vision_config()
    W = checkpoint(config)
    (tmp_path / "artifact").mkdir()
    artifact = _vision_artifact(tmp_path / "artifact", config, W, vit_checkpoint())
    spec = importlib.util.spec_from_file_location("layer_kl_cli", Path(__file__).resolve().parents[1] / "scripts" / "layer_kl.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    texts = tmp_path / "texts.json"
    # The tiny WordLevel tokenizer maps repository words to few ids; write a varied set by hand.
    texts.write_text(json.dumps(TEXTS))
    with pytest.raises(SystemExit):
        cli.main(["scan", "--model", str(artifact), "--reference", str(tmp_path / "r"), "--out", str(tmp_path / "s"),
                  "--layers", "3", "--groups", "mlp.experts", "--bits", "2"])  # no --i-own-gpu
    assert cli.main(["reference", "--model", str(artifact), "--texts", str(texts), "--out", str(tmp_path / "ref"),
                     "--skip", "4", "--i-own-gpu"]) == 0
    meta = json.loads((tmp_path / "ref" / "reference.json").read_text())
    assert meta["checkpoint"].startswith("artifact@sha256:") and meta["source"]["mlx2"]["commit"]
    assert cli.main(["scan", "--model", str(artifact), "--reference", str(tmp_path / "ref"), "--out", str(tmp_path / "scan"),
                     "--layers", "3-4", "--groups", "self_attn.q_a_proj", "--bits", "4", "--group-size", "32",
                     "--allow-dirty", "--i-own-gpu"]) == 0
    result = json.loads((tmp_path / "scan" / "scan.json").read_text())
    partial = [json.loads(line) for line in (tmp_path / "scan" / "scan.partial.jsonl").read_text().splitlines()]
    assert [p["row"]["layer"] for p in partial] == [3, 4]
    assert result["mlx2"]["commit"] and "tracked_dirty" in result["mlx2"]
    assert result["baseline"]["kl_mean"] == 0.0 and result["restore_check"]["exact"]
    assert [r["layer"] for r in result["rows"]] == [3, 4] and set(result["ranked_layers"]) == {3, 4}
    assert result["checkpoint"] == meta["checkpoint"] and (tmp_path / "scan" / "scan.md").is_file()


def test_gptq_scan_restores_exactly_and_beats_rtn_on_the_pack(tmp_path, packed):
    from mlx2.gptq import GPTQSettings

    _, _, _, model = packed
    ref = _reference(tmp_path, model)
    calib = [{"id": f"c{i}", "split": "dev", "ids": LONG[i * 7:i * 7 + 40]} for i in range(1, 6)]
    settings = GPTQSettings(bits=2, group_size=32, block=32, expert_chunk=4, min_rows=2)
    rtn = K.scan(model, ref, layers=[3, 4], groups=["mlp.experts"], bits=2, group_size=32)
    gptq = K.scan(model, ref, layers=[3, 4], groups=["mlp.experts"], bits=2, group_size=32,
                  quantizer="gptq", calib_texts=calib, gptq_settings=settings)
    assert gptq["restore_check"]["exact"] and gptq["settings"]["quantizer"] == "gptq"
    for row in gptq["rows"]:
        receipts = row["gptq"]
        assert set(receipts) == set(row["modules"])
        assert all(r["method"] == "gptq" and r["calibration_rows"]["total"] > 0 for r in receipts.values())
    assert sum(r["kl_mean"] for r in gptq["rows"]) < sum(r["kl_mean"] for r in rtn["rows"])
    with pytest.raises(ValueError, match="overlap"):
        K.scan(model, ref, layers=[3], groups=["mlp.experts"], bits=2, group_size=32,
               quantizer="gptq", calib_texts=[TEXTS[0]])


def test_source_reads_only_the_requested_groups(tmp_path, packed, monkeypatch):
    from test_hy_v4_flash_cpu import _write_artifact

    _, _, _, model = packed
    (tmp_path / "src").mkdir()
    source_dir = _write_artifact(tmp_path / "src", pack_config(), checkpoint(pack_config(), seed=7, scale=0.2))
    read = []
    original = np.frombuffer

    def counting(raw, *args, **kwargs):
        read.append(len(raw))
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(K.np, "frombuffer", counting)
    source = K.checkpoint_source(source_dir, model, groups=["mlp.experts"])
    source("model.layers.1.mlp.experts.gate_proj")  # layer 1 also holds an Engram table
    weight_map = json.loads((source_dir / "model.safetensors.index.json").read_text())["weight_map"]
    expert_names = [n for n in weight_map if n.startswith("model.layers.1.mlp.experts")]
    assert len(read) == len(expert_names) == 2  # gate_up_proj and down_proj, never the table


def test_dense_source_swap_keeps_the_activation_dtype_and_modes_are_checked(tmp_path, packed):
    _, _, _, model = packed
    ref = _reference(tmp_path, model)
    layer = model.model.layers[3].self_attn
    with pytest.raises(ValueError, match="keeps the module's mode"):
        K.scan(model, ref, layers=[3], groups=["mlp.experts"], bits=4, group_size=32, mode="mxfp4")
    assert model.model.layers[3].mlp.experts.up_proj.mode == "affine"
    # Dense modules the model keeps dense on purpose are skipped (quant_predicate).
    paths = [p for p, *_ in K.module_paths(model, 3, ["self_attn"])]
    assert not any("compressed_indexer.key_" in p for p in paths)


def test_dense_planes_from_a_source_are_cast_to_the_replacement_dtype(tmp_path, tiny):
    _, _, model = tiny
    calls = []

    def source(path):
        calls.append(path)
        return K._dense_weight(mx, model.model.layers[2].self_attn.q_a_proj).astype(mx.float32)

    applied = K.apply_substitution(model, 2, ["self_attn.q_a_proj"], bits=4, group_size=32, source=source)
    try:
        replaced = model.model.layers[2].self_attn.q_a_proj
        assert calls and replaced["scales"].dtype == model.model.layers[2].self_attn.q_a_proj["scales"].dtype
        assert replaced["scales"].dtype == mx.float32  # the tiny model runs fp32: to_quantized's dtype
    finally:
        K.restore(applied)


def test_cumulative_cli_writes_no_ranking(tmp_path):
    import importlib.util

    from test_hy_v4_flash_vision_cpu import _vision_artifact, vision_config, vit_checkpoint

    config = vision_config()
    (tmp_path / "artifact").mkdir()
    artifact = _vision_artifact(tmp_path / "artifact", config, checkpoint(config), vit_checkpoint())
    spec = importlib.util.spec_from_file_location("layer_kl_cli2", Path(__file__).resolve().parents[1] / "scripts" / "layer_kl.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(TEXTS))
    assert cli.main(["reference", "--model", str(artifact), "--texts", str(texts), "--out", str(tmp_path / "ref"),
                     "--skip", "4", "--i-own-gpu"]) == 0
    assert cli.main(["scan", "--model", str(artifact), "--reference", str(tmp_path / "ref"), "--out", str(tmp_path / "scan"),
                     "--layers", "3,4", "--groups", "self_attn.q_a_proj", "--bits", "4", "--group-size", "32",
                     "--cumulative", "--allow-dirty", "--i-own-gpu"]) == 0
    assert "ranked_layers" not in json.loads((tmp_path / "scan" / "scan.json").read_text())


def test_chunked_rtn_planes_equal_mlx_whole_bank():
    rng = np.random.default_rng(9)
    bank = mx.array((rng.standard_normal((10, 8, 64)) * 0.1).astype(np.float32))
    whole = mx.quantize(bank, group_size=32, bits=2, mode="affine")
    chunked = K._rtn_planes(mx, bank, group_size=32, bits=2, mode="affine", chunk=3)
    for a, b in zip(whole, chunked):
        np.testing.assert_array_equal(np.array(a), np.array(b))


def test_streamed_hessians_early_abort_shared_and_sequential(tmp_path, packed):
    from mlx2 import gptq as G

    _, _, _, model = packed
    calib = [{"id": f"c{i}", "split": "dev", "ids": LONG[i * 7:i * 7 + 40]} for i in range(1, 4)]
    experts = model.model.layers[3].mlp.experts
    # Streamed bank Hessian == the row oracle (InputRecorder keeps rows).
    rec_target = [(experts, "gate_proj", experts.gate_proj, True, experts.gate_proj["weight"].shape[0])]
    later = model.model.layers[5].mlp.experts
    seen_later = []
    original_up = later.up_proj

    class Spy:
        def __call__(self, *args, **kwargs):
            seen_later.append(1)
            return original_up(*args, **kwargs)

    object.__setattr__(later, "up_proj", Spy())
    try:
        accs = G.accumulate(rec_target, calib, lambda t: mx.eval(model(mx.array([t["ids"]]))))
    finally:
        vars(later).pop("up_proj", None)
    assert seen_later == []  # every forward stopped at layer 3
    recorders = G.record(model, [(experts, "gate_proj", experts.gate_proj, True)],
                         lambda: [mx.eval(model(mx.array([t["ids"]]))) for t in calib])
    h_rows, counts_rows = recorders[0].hessians(experts.gate_proj["weight"].shape[0])
    h_acc, counts_acc = accs[0].hessians(32)
    assert counts_acc == counts_rows
    E = experts.gate_proj["weight"].shape[0]
    # The row oracle keeps bf16 rows, the accumulator fp32: compare in norm.
    got, want = np.array(h_acc(0, E)), np.array(h_rows(0, E))
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-2
    # The staged GPTQ: up shares gate's Hessian, down is stage 2.
    from mlx2.gptq import GPTQSettings

    applied, receipts = K.gptq_apply(model, 3, ["mlp.experts"], calib, bits=2, group_size=32,
                                     settings=GPTQSettings(bits=2, group_size=32, block=32, expert_chunk=4, min_rows=2))
    try:
        up = "model.layers.3.mlp.experts.up_proj"
        assert receipts[up]["hessian_from"] == "model.layers.3.mlp.experts.gate_proj"
        assert receipts["model.layers.3.mlp.experts.down_proj"]["stage"] == 2
        assert receipts["model.layers.3.mlp.experts.gate_proj"]["stage"] == 1
    finally:
        K.restore(applied)
    assert "up_proj" not in vars(experts) and "gate_proj" not in vars(experts)


def test_calibration_swap_guard_runs_per_text_and_never_masks_a_forward_error(packed, monkeypatch):
    from mlx2.gptq import GPTQSettings

    _, _, _, model = packed
    calib = [{"id": f"c{i}", "split": "dev", "ids": LONG[i * 7:i * 7 + 40]} for i in range(1, 4)]
    settings = GPTQSettings(bits=2, group_size=32, block=32, expert_chunk=4, min_rows=2)
    calls = []
    applied, _ = K.gptq_apply(model, 3, ["mlp.experts"], calib, bits=2, group_size=32,
                              settings=settings, between_texts=lambda: calls.append(1))
    K.restore(applied)
    assert calls  # the guard ran after early-stopped texts

    class Guard(RuntimeError):
        pass

    def guard():
        raise Guard("swap grew")

    def broken(*args, **kwargs):
        raise ValueError("forward failed")

    monkeypatch.setattr(model, "hidden", broken, raising=False)
    with pytest.raises(ValueError, match="forward failed"):
        K.gptq_apply(model, 3, ["mlp.experts"], calib, bits=2, group_size=32,
                     settings=settings, between_texts=guard)
    experts = model.model.layers[3].mlp.experts
    assert "up_proj" not in vars(experts) and "gate_proj" not in vars(experts)
