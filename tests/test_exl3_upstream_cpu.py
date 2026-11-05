import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.exl3_reference import codebook, decode_inner, effective_weight
from mlx2.runtime.trellis.upstream import (
    CODEBOOKS,
    ShardReader,
    load_bank,
    load_linear,
    parse_format,
    plan_checkpoint,
    read_storage,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "scripts/sushi_probe/fixtures"
# The tile fixtures live with the private Sushi probe; the public mirror omits it.
needs_fixtures = pytest.mark.skipif(not FIXTURES.is_dir(), reason="Sushi probe fixtures not present")
# Fixtures without codebook metadata are Sushi's MUL1 converter outputs.
CASES = [
    ("exl3_k2_linear", "mul1", 16),
    ("exl3_k3_linear", "mul1", 16),
    ("exl3_k4_linear", "mul1", 16),
    ("exl3_k2p5_mcg_linear", "mcg", 16),
    ("exl3_k2p5_mcg_w12_linear", "mcg", 12),
    ("exl3_k3_mcg_linear", "mcg", 16),
]
FORMAT = {
    "quant_method": "exl3",
    "version": "0.0.29",
    "bits": 2.08,
    "head_bits": 6,
    "calibration": {"rows": 250, "cols": 2048},
    "out_scales": "always",
    "codebook": "mcg",
}


@needs_fixtures
@pytest.mark.parametrize("name,cb,window", CASES)
def test_host_decoder_matches_exllamav3_tiles(name, cb, window):
    f = mx.load(str(FIXTURES / f"{name}.safetensors"))
    inner = decode_inner(np.asarray(f["trellis"]), cb, window)
    np.testing.assert_array_equal(inner, np.asarray(f["inner"]))
    if window == 16:
        w = effective_weight(np.asarray(f["trellis"]), f["suh"], f["svh"], cb)
        np.testing.assert_allclose(w, np.asarray(f["public"]), atol=2e-3, rtol=1e-2)


@needs_fixtures
def test_signed_trellis_decodes_as_its_bit_pattern():
    f = mx.load(str(FIXTURES / "exl3_k3_mcg_linear.safetensors"))
    signed = np.asarray(f["trellis"]).view(np.int16)
    np.testing.assert_array_equal(
        decode_inner(signed.view(np.uint16)), np.asarray(f["inner"])
    )
    w = effective_weight(signed, f["suh"], f["svh"], "mcg")
    np.testing.assert_allclose(w, np.asarray(f["public"]), atol=2e-3, rtol=1e-2)


def test_unimplemented_codebook_refuses():
    with pytest.raises(ValueError, match="not implemented"):
        codebook("3inst")


def _linear(rng, i, o, words, cb="mcg"):
    return {
        "trellis": mx.array(rng.integers(-32768, 32768, (i // 16, o // 16, words), np.int16)),
        "suh": mx.array(rng.normal(0, 0.01, i).astype(np.float16)),
        "svh": mx.array(np.sign(rng.normal(size=o)).astype(np.float16)),
        cb: mx.array(np.int32(CODEBOOKS[cb] - 2**32 if CODEBOOKS[cb] >= 2**31 else CODEBOOKS[cb])),
    }


def _write(tmp_path, *, experts=4, mutate=None, storage=True, config=None):
    """A small upstream-layout checkpoint: two dense linears, two banks."""
    rng = np.random.default_rng(7)
    tensors = {"model.language_model.norm.weight": mx.ones((256,), mx.bfloat16)}
    modules = {
        "model.language_model.layers.0.self_attn.q_proj": (256, 384, 64),
        "lm_head": (256, 512, 96),
    }
    for proj, shape in (("gate_proj", (256, 128, 32)), ("down_proj", (128, 256, 32))):
        for e in range(experts):
            modules[f"model.language_model.layers.0.mlp.experts.{e}.{proj}"] = shape
    for key, (i, o, words) in modules.items():
        for part, value in _linear(rng, i, o, words).items():
            tensors[f"{key}.{part}"] = value
    if mutate:
        mutate(tensors)
    names = sorted(tensors)
    half = len(names) // 2
    shards = {"model-00001.safetensors": names[:half], "model-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        mx.save_safetensors(str(tmp_path / shard), {k: tensors[k] for k in keys})
        weight_map.update({k: shard for k in keys})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (tmp_path / "config.json").write_text(
        json.dumps(config or {"model_type": "qwen3_5_moe", "quantization_config": FORMAT})
    )
    if storage:
        entries = {
            key: {"quant_format": "exl3", "bits_per_weight": words / 16, "mcg_multiplier": CODEBOOKS["mcg"]}
            for key, (_, _, words) in modules.items()
        }
        (tmp_path / "quantization_config.json").write_text(
            json.dumps({**FORMAT, "tensor_storage": entries})
        )
    return weight_map, tensors


def _plan(tmp_path, weight_map, **kw):
    return plan_checkpoint(tmp_path, weight_map, storage=read_storage(tmp_path), **kw)


def test_plan_stacks_experts_exactly_and_reports_rates(tmp_path):
    weight_map, tensors = _write(tmp_path)
    plan = _plan(tmp_path, weight_map, num_experts=4)
    assert sorted(plan.linears) == ["lm_head", "model.language_model.layers.0.self_attn.q_proj"]
    bank = plan.banks["model.language_model.layers.0.mlp.gate_proj"]
    assert (bank.experts, bank.input_dims, bank.output_dims, bank.words) == (4, 256, 128, 32)
    assert plan.stored == {"model.language_model.norm.weight"}
    receipt = plan.receipt()
    assert receipt["rates"] == {"mcg/k2": 2, "mcg/k4": 1, "mcg/k6": 1}
    assert receipt["unfixtured_words"] == [("mcg", 32), ("mcg", 64), ("mcg", 96)]
    assert receipt["qualification"] == "unqualified"
    read = ShardReader(tmp_path, weight_map)
    stacked = load_bank(read, bank)
    assert stacked["trellis"].dtype == mx.uint16
    assert stacked["trellis"].shape == (4, 16, 8, 32)
    for e in range(4):
        key = f"model.language_model.layers.0.mlp.experts.{e}.gate_proj"
        assert mx.array_equal(stacked["trellis"][e], tensors[f"{key}.trellis"].view(mx.uint16))
        assert mx.array_equal(stacked["suh"][e], tensors[f"{key}.suh"])
    dense = load_linear(read, plan.linears["lm_head"])
    assert dense["trellis"].dtype == mx.uint16 and "bias" not in dense


def _drop(name):
    return lambda t: t.pop(name)


def _add(name, value):
    return lambda t: t.__setitem__(name, value)


Q = "model.language_model.layers.0.self_attn.q_proj"
E1 = "model.language_model.layers.0.mlp.experts.1.gate_proj"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (_drop(f"{Q}.suh"), "incomplete EXL3 module"),
        (_drop(f"{Q}.mcg"), "3INST"),
        (_add(f"{Q}.mul1", mx.array(np.int32(1))), "two codebooks"),
        (_add(f"{Q}.su", mx.zeros((16,), mx.int16)), "legacy"),
        (_add(f"{Q}.bias", mx.zeros((384,), mx.float32)), "bias must be F16"),
        (_add(f"{E1}.bias", mx.zeros((128,), mx.float16)), "cannot carry a bias"),
        (_add(f"{Q}.svh", mx.zeros((383,), mx.float16)), "svh must be F16"),
        (
            _add(f"{Q}.trellis", mx.zeros((16, 24, 63), mx.int16)),
            "word count 63",
        ),
        (
            _add(f"{E1}.trellis", mx.zeros((16, 8, 48), mx.int16)),
            "differ in layout|bits disagree",
        ),
        (
            lambda t: [t.pop(k) for k in list(t) if ".experts.3." in k],
            "has 3 experts",
        ),
        (
            lambda t: [t.pop(k) for k in list(t) if ".experts.2." in k],
            "not contiguous",
        ),
    ],
)
def test_inconsistent_checkpoints_refuse_before_loading(tmp_path, mutate, match):
    weight_map, _ = _write(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, weight_map, num_experts=4)


def test_storage_map_must_agree(tmp_path):
    weight_map, _ = _write(tmp_path)
    q = json.loads((tmp_path / "quantization_config.json").read_text())
    q["tensor_storage"][Q]["mcg_multiplier"] = 1
    (tmp_path / "quantization_config.json").write_text(json.dumps(q))
    with pytest.raises(ValueError, match="multiplier"):
        _plan(tmp_path, weight_map)
    q["tensor_storage"][Q] = {"bits_per_weight": 3}
    (tmp_path / "quantization_config.json").write_text(json.dumps(q))
    with pytest.raises(ValueError, match="bits disagree"):
        _plan(tmp_path, weight_map)


def test_indexed_tensor_absent_from_shard_refuses(tmp_path):
    weight_map, _ = _write(tmp_path)
    weight_map = {**weight_map, "model.language_model.ghost.weight": "model-00001.safetensors"}
    with pytest.raises(ValueError, match="missing from its shard"):
        _plan(tmp_path, weight_map)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"quantization": {"bits": 4, "group_size": 64}}, "also declares MLX"),
        ({"quantization_config": {**FORMAT, "codebook": "3inst"}}, "not implemented"),
        ({"quantization_config": {**FORMAT, "bits": "4"}}, "numeric bits"),
        ({"quantization_config": {**FORMAT, "out_scales": "sometimes"}}, "out_scales"),
    ],
)
def test_format_header_refusals(change, match):
    config = {"model_type": "qwen3_5_moe", "quantization_config": FORMAT, **change}
    with pytest.raises(ValueError, match=match):
        parse_format(config)


def test_declared_codebook_must_match_markers(tmp_path):
    config = {"model_type": "qwen3_5_moe", "quantization_config": {**FORMAT, "codebook": "mul1"}}
    weight_map, _ = _write(tmp_path, config=config, storage=False)
    with pytest.raises(ValueError, match="disagrees with the declared codebook"):
        _plan(tmp_path, weight_map)


def test_registry_refuses_exl3_for_families_without_a_reader(tmp_path):
    from mlx2.adapters.registry import inspect_model

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "quantization_config": FORMAT})
    )
    with pytest.raises(ValueError, match="admits EXL3"):
        inspect_model(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"num_hidden_layers": 48, "hidden_size": 3072},
                "quantization_config": FORMAT,
            }
        )
    )
    with pytest.raises(ValueError, match="122B does not admit EXL3"):
        inspect_model(tmp_path)


def test_qwen36_inspector_plans_and_names_modules(tmp_path):
    from mlx2.adapters import qwen36_35b
    from mlx2.adapters.registry import inspect_model

    text = {
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "num_experts": 4,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    config = {"model_type": "qwen3_5_moe", "text_config": text, "quantization_config": FORMAT}
    _write(tmp_path, config=config)
    with pytest.raises(ValueError, match="topology"):
        inspect_model(tmp_path)  # 4 experts is not the 35B topology
    text["num_experts"] = 256
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="has 4 experts, config says 256"):
        inspect_model(tmp_path)
    _write(tmp_path, experts=256, config=config)
    resolution = inspect_model(tmp_path)
    plan = resolution.artifact["exl3_upstream"]
    assert plan.banks["model.language_model.layers.0.mlp.gate_proj"].experts == 256
    fingerprint = resolution.artifact["identity"]["fingerprint"]
    (tmp_path / "quantization_config.json").write_text(
        (tmp_path / "quantization_config.json").read_text().replace("0.0.29", "0.0.30")
    )
    assert inspect_model(tmp_path).artifact["identity"]["fingerprint"] != fingerprint
    assert qwen36_35b.exl3_module_path(
        "model.language_model.layers.3.self_attn.q_proj"
    ) == "language_model.model.layers.3.self_attn.q_proj"
    assert qwen36_35b.exl3_module_path("lm_head") == "language_model.lm_head"
    with pytest.raises(ValueError, match="no Qwen3.6 module"):
        qwen36_35b.exl3_module_path("model.visual.blocks.0.attn.qkv")


def test_exl3_kernel_assets_bind_into_source_identity():
    import mlx2
    from mlx2.source_dependencies import add_resources

    root = Path(mlx2.__file__).parent
    bound = add_resources(root, {"runtime/models/exl3.py": ""})
    kernels = {k for k in bound if k.startswith("runtime/models/exl3_kernels/")}
    on_disk = {
        f"runtime/models/exl3_kernels/{p.name}"
        for p in (root / "runtime/models/exl3_kernels").iterdir()
        if p.is_file()
    }
    assert kernels == on_disk and any(k.endswith(".metal") for k in kernels)


def test_named_half_gates_cast_to_compute_dtype_and_nothing_else():
    from mlx2.adapters.qwen36_35b import EXL3_HALF_COMPUTE_WEIGHTS
    from mlx2.runtime.trellis.upstream import cast_half_compute

    gate = "language_model.model.layers.0.linear_attn.in_proj_a.weight"
    router = "language_model.model.layers.0.mlp.gate.weight"
    norm = "language_model.model.layers.0.input_layernorm.weight"
    a_log = "language_model.model.layers.0.linear_attn.A_log"
    weights = {
        gate: mx.ones((4, 8), mx.float16),
        router: mx.ones((4, 8), mx.float16),
        norm: mx.ones((8,), mx.float16),
        a_log: mx.ones((4,), mx.float32),
    }
    receipt = cast_half_compute(weights, EXL3_HALF_COMPUTE_WEIGHTS, mx.bfloat16)
    assert receipt["cast"] == 2 and receipt["to"] == "bfloat16"
    assert weights[gate].dtype == weights[router].dtype == mx.bfloat16
    assert weights[norm].dtype == mx.float16 and weights[a_log].dtype == mx.float32
    assert cast_half_compute(weights, EXL3_HALF_COMPUTE_WEIGHTS, mx.float16) == {"cast": 0}


@pytest.mark.parametrize("rows,experts", [(1, 1), (37, 1), (5, 8), (31, 8), (32, 8), (200, 8)])
def test_row_plan_orders_every_row_once_sorted_by_expert(rows, experts):
    from mlx2.runtime.models.exl3 import GROUP_MIN_ROWS, ROWS_PER_BLOCK, row_plan

    rng = np.random.default_rng(rows)
    ids = mx.array(rng.integers(0, experts, rows).astype(np.int32))
    plan = row_plan(ids, experts)
    order = np.asarray(plan.order)
    assert plan.order.dtype == plan.eids.dtype == mx.uint32
    assert sorted(order.tolist()) == list(range(rows))
    np.testing.assert_array_equal(np.asarray(plan.eids), np.asarray(ids))
    wide = experts == 1 and rows > 1 or experts > 1 and rows >= GROUP_MIN_ROWS
    assert plan.rows == (ROWS_PER_BLOCK if wide else 1)
    if plan.rows > 1:
        assert (np.diff(np.asarray(ids)[order]) >= 0).all()


@pytest.mark.parametrize("rows", [1, 5, 40])
def test_rows_law_matches_the_reference_law(rows):
    from mlx2.runtime.models.exl3 import Exl3Linear

    rng = np.random.default_rng(11)
    e, i, o, words = 4, 256, 128, 48
    trellis = mx.array(rng.integers(0, 65536, (e, i // 16, o // 16, words), np.uint16))
    suh = mx.array((rng.normal(0, 1, (e, i)) * 0.03).astype(np.float16))
    svh = mx.array(np.sign(rng.normal(size=(e, o))).astype(np.float16))
    ids = mx.array(rng.integers(0, e, rows).astype(np.uint32))
    x = mx.array(rng.normal(0, 1, (rows, i)).astype(np.float32))
    ref = Exl3Linear(trellis, suh, svh)(x, ids)
    got = Exl3Linear(trellis, suh, svh, law="rows")(x, ids)
    scale = float(mx.max(mx.abs(ref)).item())
    assert float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item()) <= 2e-3 * scale


def test_unknown_law_and_mixed_switch_laws_refuse():
    from mlx2.runtime.models.exl3 import Exl3Linear, Exl3SwitchGLU

    rng = np.random.default_rng(2)
    t = lambda i, o: mx.array(rng.integers(0, 65536, (2, i // 16, o // 16, 32), np.uint16))
    v = lambda n: mx.ones((2, n), mx.float16)
    with pytest.raises(ValueError, match="law"):
        Exl3Linear(t(128, 128), v(128), v(128), law="auto")
    gate = Exl3Linear(t(128, 128), v(128), v(128), law="rows")
    up = Exl3Linear(t(128, 128), v(128), v(128))
    down = Exl3Linear(t(128, 128), v(128), v(128), law="rows")
    with pytest.raises(ValueError, match="row-law"):
        Exl3SwitchGLU(gate, up, down, execution="rows")


def test_nax_dense_law_matches_reference_above_the_row_threshold():
    from mlx2.runtime.models.exl3 import NAX_MIN_ROWS, Exl3DenseLinear
    from mlx2.runtime.models.exl3_reference import decode_inner

    rng = np.random.default_rng(5)
    i, o, words = 256, 384, 64
    t = {
        "trellis": mx.array(rng.integers(0, 65536, (i // 16, o // 16, words), np.uint16)),
        "suh": mx.array((rng.normal(0, 1, i) * 0.03).astype(np.float16)),
        "svh": mx.array(np.sign(rng.normal(size=o)).astype(np.float16)),
    }
    ref, nax = Exl3DenseLinear(**t), Exl3DenseLinear(**t, law="fused_nax")
    ref.eval()
    nax.eval()
    whole = nax.bank.reconstruct(mx.array([0], mx.uint32))
    np.testing.assert_array_equal(np.asarray(whole[0]), decode_inner(np.asarray(t["trellis"])))
    part = nax.bank.reconstruct(mx.array([0], mx.uint32), 8, 128)
    np.testing.assert_array_equal(np.asarray(part[0]), np.asarray(whole[0][:, 128:256]))
    with pytest.raises(ValueError, match="128-column"):
        nax.bank.reconstruct(mx.array([0], mx.uint32), 0, 100)
    x = mx.array(rng.normal(0, 1, (NAX_MIN_ROWS + 3, i)).astype(np.float32))
    got = nax(x)
    assert nax.nax_calls == 1
    scale = float(mx.max(mx.abs(ref(x))).item())
    assert float(mx.max(mx.abs(got - ref(x))).item()) <= 5e-3 * scale


def test_nax_bank_switch_matches_the_reference_switch(monkeypatch):
    from mlx2.runtime.models import exl3

    monkeypatch.setattr(exl3, "NAX_BANK_MIN_ROWS", 16)
    monkeypatch.setattr(exl3, "NAX_BANK_CHUNK", 3)  # several chunks, some empty
    rng = np.random.default_rng(9)
    e, h, i = 8, 256, 128

    def bank(i_dim, o_dim, law):
        t = mx.array(rng.integers(0, 65536, (e, i_dim // 16, o_dim // 16, 32), np.uint16))
        return t, mx.array((rng.normal(0, 1, (e, i_dim)) * 0.03).astype(np.float16)), \
            mx.array(np.sign(rng.normal(size=(e, o_dim))).astype(np.float16))

    tensors = [bank(h, i, None), bank(h, i, None), bank(i, h, None)]
    ref = exl3.Exl3SwitchGLU(*(exl3.Exl3Linear(*t) for t in tensors), execution="reference")
    nax = exl3.Exl3SwitchGLU(*(exl3.Exl3Linear(*t, law="rows") for t in tensors), execution="rows_nax")
    ref.eval()
    nax.eval()
    x = mx.array(rng.normal(0, 1, (6, h)).astype(np.float32))
    ids = mx.array(np.stack([rng.choice([0, 1, 5, 7], 4, replace=False) for _ in range(6)]).astype(np.uint32))
    want, got = ref(x, ids), nax(x, ids)
    assert nax.nax_calls == 1
    scale = float(mx.max(mx.abs(want)).item())
    assert float(mx.max(mx.abs(got - want)).item()) <= 5e-3 * scale
    small = mx.array(np.stack([rng.choice(e, 2, replace=False) for _ in range(3)]).astype(np.uint32))
    nax(x[:3], small)
    assert nax.nax_calls == 1 and nax.rows_calls == 1  # below the threshold: row law
    # The sorted-order switch is, row for row, three per-projection banks.
    layout = exl3.nax_layout(ids.reshape(-1), e)
    rows = mx.repeat(x, 4, axis=0)
    flat_ids = ids.reshape(-1)
    g = exl3.nax_bank(nax.gate_proj, rows, flat_ids, layout)
    u = exl3.nax_bank(nax.up_proj, rows, flat_ids, layout)
    banks = exl3.nax_bank(nax.down_proj, exl3.swiglu(g, u), flat_ids, layout).astype(mx.bfloat16)
    sorted_switch = exl3.nax_switch(nax, x, flat_ids, 4, mx.bfloat16)
    np.testing.assert_array_equal(np.asarray(sorted_switch.view(mx.uint16)), np.asarray(banks.view(mx.uint16)))


def test_nax_fused_is_the_upstream_exl3_default_and_refuses_elsewhere():
    from mlx2.adapters import qwen36_35b

    assert qwen36_35b.DEFAULT_EXL3_EXECUTION == "nax_fused"
    assert qwen36_35b.DEFAULT_EXL3_EXECUTION in qwen36_35b.EXL3_EXECUTIONS
    source = Path(qwen36_35b.__file__).read_text()
    # Non-EXL3 artifacts get no EXL3 law, and only an explicit choice refuses.
    assert 'DEFAULT_EXL3_EXECUTION if upstream_exl3 else "reference"' in source
    assert "if explicit_exl3 and not upstream_exl3:" in source


def test_resident_verify_is_the_nax_fused_default_only():
    # Pierre, 2026-11-05: resident fp16 verify weights default-on under
    # nax_fused; other laws and non-EXL3 artifacts stay "off", and only an
    # explicit choice on a non-EXL3 artifact refuses.
    from mlx2.adapters import qwen36_35b

    assert qwen36_35b.DEFAULT_EXL3_RESIDENT_DENSE == "verify"
    assert qwen36_35b.DEFAULT_EXL3_RESIDENT_DENSE in qwen36_35b.EXL3_RESIDENT_DENSE
    source = Path(qwen36_35b.__file__).read_text()
    assert ('DEFAULT_EXL3_RESIDENT_DENSE if upstream_exl3 and self.exl3_execution == "nax_fused" else "off"'
            in source)
    assert 'if self.exl3_resident_dense != "off" and not upstream_exl3:' in source


def test_external_draft_policy_keeps_the_exl3_law_on_the_target():
    # AST/DFlash2 servers pass the target law beside the drafter contract;
    # it was refused as an unknown drafter key (exl3-fuse-h2h-20261105).
    from mlx2.adapters.qwen36_35b import partition_external_target_policy

    external, target = partition_external_target_policy(
        {"draft_model": "/drafter", "exl3_execution": "nax_fused"})
    assert target == {"exl3_execution": "nax_fused"}
    assert external == {"draft_model": "/drafter"}
