"""Adapter-declared MoE geometries for the exact Nemotron routed-batch strip."""

import copy
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from test_nemotron35_bolt_cpu import target_config

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.models.nemotron_h import (
    NEMOTRON_ROUTED_EXACT_DEFAULT_GEOMETRIES,
    Model,
    ModelArgs,
    NemotronHMoE,
)
from mlx2.runtime.row_exact import canary, ops

SUPER = (4096, 512, 22, 1024, 2688, 5376)


def _refusal(geometry, declared=None, rows=4):
    hidden, experts, top_k, latent, width, shared = geometry
    config = SimpleNamespace(
        hidden_size=hidden, n_routed_experts=experts, num_experts_per_tok=top_k,
        moe_latent_size=latent, moe_intermediate_size=width,
        moe_shared_expert_intermediate_size=shared, n_shared_experts=1,
    )
    owner = SimpleNamespace(training=False, config=config, _relu2_down_policy="auto",
                            switch_mlp=SimpleNamespace(_sort_policy="auto"))
    if declared is not None:
        owner.__dict__["_mlx2_routed_exact_geometries"] = tuple(map(tuple, declared))
    return NemotronHMoE._routed_batch_geometry_refusal(owner, rows)


def test_undeclared_models_keep_exactly_the_super_geometry():
    assert NEMOTRON_ROUTED_EXACT_DEFAULT_GEOMETRIES == (SUPER,)
    assert _refusal(SUPER) is None
    assert _refusal((4096, 240, 9, 1024, 1536, 5376)).startswith("geometry")
    assert "routed_exact_geometries" not in ModelArgs.__dataclass_fields__


def test_a_declaration_admits_only_its_listed_geometries():
    bolt = [(4096, 240, 9, 1024, 1536, 5376), (4096, 496, 9, 1024, 2560, 5376)]
    assert _refusal(bolt[0], bolt) is None and _refusal(bolt[1], [list(bolt[1])]) is None
    assert _refusal((4096, 240, 9, 1024, 2560, 5376), bolt).startswith("geometry")
    # A declaration replaces the default: it does not silently add Super.
    assert _refusal(SUPER, bolt).startswith("geometry")
    assert _refusal(bolt[0], bolt, rows=17).startswith("rows")


def test_bolt_adapter_declares_one_geometry_per_puzzle_block_shape():
    from mlx2.adapters import nemotron35_bolt as bolt

    config = target_config()
    config["llm_config"].update(
        time_step_limit=[0.0, {"__float__": "Infinity"}], moe_latent_size=1024,
        moe_shared_expert_intermediate_size=5376, n_group=1, topk_group=1)
    text = bolt.Nemotron35BoltAdapter._runtime_language_config(config)
    assert "routed_exact_geometries" not in text
    blocks = [b for b in text["block_configs"] if b["block_type"] == "moe"]
    expected = sorted({(4096, b["n_routed_experts"], 9, 1024, b["moe_intermediate_size"], 5376)
                       for b in blocks})
    assert bolt.routed_exact_geometries(text) == expected
    declared = []
    model = SimpleNamespace(declare_routed_exact_geometries=declared.append)
    bolt.Nemotron35BoltAdapter._declare_model_contracts(None, model, text)
    assert declared == [expected]


def test_a_config_supplied_geometry_is_refused():
    # A checkpoint/config dict cannot admit a geometry: the key is not a
    # model argument, and the gate reads only the adapter's declaration.
    mx.set_default_device(mx.cpu)
    blocks = [{"block_type": "moe", "n_routed_experts": 5, "moe_intermediate_size": 6,
               "num_experts_per_tok": 3}] * 2
    smuggled = [[8, 5, 3, 4, 6, 7]]
    args = ModelArgs.from_dict({**_tiny_args_dict(), "block_configs": copy.deepcopy(blocks),
                                "routed_exact_geometries": smuggled})
    assert not hasattr(args, "routed_exact_geometries")
    model = Model(args)
    model.eval()
    mixer = model.layers[0].mixer
    assert mixer._routed_batch_geometry_refusal(3).startswith("geometry")
    with pytest.raises(ValueError, match="geometry"):
        mixer.routed_batch_exact(mx.zeros((1, 3, 8)))
    # The same geometry declared by an adapter is admitted.
    model.declare_routed_exact_geometries(smuggled)
    assert mixer._routed_batch_geometry_refusal(3) is None
    with pytest.raises(ValueError, match="invalid"):
        model.declare_routed_exact_geometries([[8, 5, 3, 4, 6]])


def _tiny_args_dict():
    return dict(  # noqa: C408
        model_type="nemotron_h", vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, max_position_embeddings=128, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, attention_bias=False, mamba_num_heads=2,
        mamba_head_dim=4, mamba_proj_bias=False, ssm_state_size=4, conv_kernel=3,
        n_groups=1, mlp_bias=False, layer_norm_epsilon=1e-5, use_bias=False,
        use_conv_bias=True, hybrid_override_pattern=["E", "E"],
        moe_latent_size=4, moe_shared_expert_intermediate_size=7, n_group=1,
        n_shared_experts=1, topk_group=1, norm_topk_prob=True,
        routed_scaling_factor=1.0, n_routed_experts=6, moe_intermediate_size=6,
        num_experts_per_tok=3,
    )


def _same_bits(a, b):
    return a.shape == b.shape and bool(mx.all(a.view(mx.uint32) == b.view(mx.uint32)).item())


def test_each_declared_puzzle_geometry_runs_its_own_canaries_and_stays_row_exact():
    mx.set_default_device(mx.cpu)
    mx.random.seed(4401)
    canary.CANARIES.reset()
    blocks = [
        {"block_type": "moe", "n_routed_experts": 5, "moe_intermediate_size": 6, "num_experts_per_tok": 3},
        {"block_type": "moe", "n_routed_experts": 7, "moe_intermediate_size": 10, "num_experts_per_tok": 3},
    ]
    declared = [(8, b["n_routed_experts"], 3, 4, b["moe_intermediate_size"], 7) for b in blocks]
    args = ModelArgs.from_dict({**_tiny_args_dict(), "block_configs": blocks,
                                "router_compute_dtype": "float32"})
    model = Model(args)
    model.eval()
    model.declare_routed_exact_geometries(declared)
    assert not any("routed_exact" in key for key, _ in tree_flatten(model.parameters()))
    model._short_segment_policy = "layerwise_short_exact_routed_batch"
    widths = []
    for layer in model.layers:
        mixer = layer.mixer
        widths.append((int(mixer.gate.weight.shape[0]), int(mixer.switch_mlp.fc1.weight.shape[1])))
        assert mixer._routed_batch_geometry_refusal(4) is None
        x = mx.random.normal((1, 4, 8), key=mx.random.key(len(widths))).astype(mx.float32)
        recipe = model._row_exact_recipe(layer)
        assert recipe.name == "moe_exact"
        record = REV.Window(4)
        with REV.window(record):
            out = recipe.run(layer, x, None)
        model._row_exact_close(record)
        reference = ops.per_row(lambda row, layer=layer: layer(row, mask=None, cache=None), x)
        assert _same_bits(out, reference)
    assert widths == [(5, 6), (7, 10)]
    snapshot = canary.CANARIES.snapshot()
    assert not snapshot["failed"] and not snapshot["private"]["failed"]
    router_keys = [entry for entry in canary.CANARIES._results if "router_select" in entry]
    assert {key[-2] for key in router_keys} == {5, 7}
    assert snapshot["private"]["checked"] >= 2
    status = model.row_exact_status()
    assert status["windows"] == 2 and status["windows_exact"] == 2


def test_an_undeclared_puzzle_layer_is_refused_by_the_short_segment():
    blocks = [{"block_type": "moe", "n_routed_experts": 5, "moe_intermediate_size": 6,
               "num_experts_per_tok": 3}] * 2
    mx.set_default_device(mx.cpu)
    args = ModelArgs.from_dict({**_tiny_args_dict(), "block_configs": copy.deepcopy(blocks)})
    model = Model(args)
    model.eval()
    with pytest.raises(ValueError, match="geometry"):
        model.layers[0].mixer.routed_batch_exact(mx.zeros((1, 3, 8)))
