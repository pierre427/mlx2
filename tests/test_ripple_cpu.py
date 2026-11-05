"""Synthetic CPU oracles, not Ripple checkpoint serving qualification.

Source contract: Ripple-65B-Preview-NVFP4@99eea4477a001a41b86914e41feb4b2ed86105e5.
Runtime revision is recorded by the enclosing validation receipt.
"""

import copy
import json
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters.ripple import (
    DESCRIPTOR,
    RippleAdapter,
    parse_policy,
    source_weights,
    tensor_schema,
    validate_config,
    validate_headers,
)
from mlx2.contracts import Capability
from mlx2.conversion.nvfp4 import DTYPE_BYTES, decode_rows
from mlx2.runtime.models.ricursive import Model, ModelArgs, RMSNorm, rotary

FIXTURES = Path(__file__).parent / "fixtures/ripple"


def tiny_args():
    return ModelArgs(hidden_size=16, intermediate_size=32, num_hidden_layers=5,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                     vocab_size=32, sliding_window=4, embed_scale=2.0,
                     ffn_clamp_threshold=0.2)


def torch_reference(tokens, weights, args, torch):
    """Source equations in PyTorch with explicit causal/local attention masks."""
    f = torch.nn.functional
    t = {k: torch.tensor(np.asarray(v.astype(mx.float32))) for k, v in weights.items()}
    dtype = torch.bfloat16 if next(iter(weights.values())).dtype == mx.bfloat16 else torch.float32
    t = {k: v.to(dtype) for k, v in t.items()}

    def norm(x, p):
        y = x.float()
        return (y * torch.rsqrt((y * y).mean(-1, keepdim=True) + args.rms_norm_eps)).to(x.dtype) * t[p + ".weight"]

    def linear(x, p):
        return f.linear(x, t[p + ".weight"])

    h = f.embedding(torch.tensor(tokens, dtype=torch.long), t["model.embed_tokens.weight"]) * args.embed_scale
    seed = h
    b, length, _ = h.shape
    positions = torch.arange(length)
    for index in range(args.num_hidden_layers):
        p = f"model.layers.{index}."
        a = p + "self_attn."
        x = norm(h, p + "input_layernorm")
        q = norm(linear(x, a + "q_proj").reshape(b, length, args.num_attention_heads, args.head_dim), a + "q_norm").transpose(1, 2)
        k = norm(linear(x, a + "k_proj").reshape(b, length, args.num_key_value_heads, args.head_dim), a + "k_norm").transpose(1, 2)
        value_input = torch.lerp(seed, x, torch.sigmoid(t[a + "value_res_lambda"][0]).to(x.dtype))
        v = linear(value_input, a + "v_proj").reshape(b, length, args.num_key_value_heads, args.head_dim).transpose(1, 2)
        mask = positions[:, None] >= positions[None, :]
        if args.layer_types[index] == "sliding_attention":
            inv = 1.0 / args.rope_theta ** (torch.arange(0, args.head_dim, 2).float() / args.head_dim)
            phase = positions[:, None].float() * inv
            phase = torch.cat((phase, phase), dim=-1)
            cos, sin = phase.cos().to(x.dtype).float(), phase.sin().to(x.dtype).float()
            def rotate(y, cos=cos, sin=sin, dtype=x.dtype):
                first, second = y.float().chunk(2, dim=-1)
                return (y.float() * cos + torch.cat((-second, first), dim=-1) * sin).to(dtype)
            q, k = rotate(q), rotate(k)
            mask &= positions[:, None] - positions[None, :] < args.sliding_window
        k = k.repeat_interleave(args.num_attention_heads // args.num_key_value_heads, dim=1)
        v = v.repeat_interleave(args.num_attention_heads // args.num_key_value_heads, dim=1)
        y = f.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, length, -1)
        y = y * torch.sigmoid(linear(x, a + "g_proj").float()).to(y.dtype)
        h = h + norm(linear(y, a + "o_proj"), p + "post_attention_layernorm")
        x = norm(h, p + "pre_feedforward_layernorm")
        gate = linear(x, p + "mlp.gate_proj").clamp(max=args.ffn_clamp_threshold)
        up = linear(x, p + "mlp.up_proj").clamp(-args.ffn_clamp_threshold, args.ffn_clamp_threshold)
        h = h + norm(linear(f.silu(gate) * up, p + "mlp.down_proj"), p + "post_feedforward_layernorm")
    return linear(norm(h, "model.norm"), "lm_head").float().numpy()


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_model_matches_independent_source_equations(dtype):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    mx.random.seed(71)
    args = tiny_args()
    model = Model(args, nvfp4=False)
    for layer, value in zip(model.layers, [-2.75, -.3, .25, 1.1, 4.0], strict=True):
        layer.self_attn.value_res_lambda = mx.array([value])
    model.set_dtype(dtype)
    weights = dict(tree_flatten(model.parameters()))
    # Full five-layer multi-token FP32 oracle plus an exact BF16 one-token
    # rounding oracle. Multi-token BF16 SDPA is checked separately below.
    tokens = [[1]] if dtype == mx.bfloat16 else [[1, 2, 4, 8, 3, 10, 9]]
    actual = np.asarray(model(mx.array(tokens)).astype(mx.float32))
    expected = torch_reference(tokens, weights, args, torch)
    if dtype == mx.bfloat16:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_bf16_attention_cpu_component_rounding_bound():
    torch = pytest.importorskip("torch")
    from mlx2.runtime.models.base import scaled_dot_product_attention

    rng = np.random.default_rng(91)
    q, k, v = [mx.array(rng.normal(size=shape).astype(np.float32)).astype(mx.bfloat16)
               for shape in [(1, 4, 7, 4), (1, 2, 7, 4), (1, 2, 7, 4)]]
    tq, tk, tv = [torch.tensor(np.asarray(x.astype(mx.float32))) for x in (q, k, v)]
    actual = scaled_dot_product_attention(q, k, v, cache=None, scale=.5, mask="causal")
    expected = torch.nn.functional.scaled_dot_product_attention(
        tq, tk.repeat_interleave(2, 1), tv.repeat_interleave(2, 1), is_causal=True).bfloat16()
    # CPU MLX SDPA rounds intermediates differently. Do not promote this
    # component bound into full-model BF16 or token parity.
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected.float().numpy(),
                               rtol=0, atol=0.008)


@pytest.mark.parametrize("chunks", [[1] * 13, [3, 5, 1, 4], [7, 6], [13]])
def test_chunked_prefill_and_decode_match_full_forward_across_window(chunks):
    mx.random.seed(18)
    model = Model(tiny_args(), nvfp4=False)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    expected = model(tokens)
    cache = model.make_cache()
    pieces = []
    offset = 0
    for length in chunks:
        pieces.append(model(tokens[:, offset:offset + length], cache))
        offset += length
    np.testing.assert_allclose(np.asarray(mx.concatenate(pieces, axis=1)), np.asarray(expected), rtol=2e-5, atol=3e-6)
    assert all(c.offset == 13 for c in cache)
    assert [type(c).__name__ for c in cache] == ["RotatingKVCache"] * 4 + ["KVCache"]
    snapshot = copy.deepcopy(cache)
    suffix = mx.array([[15, 16]])
    expected_suffix = model(mx.concatenate((tokens, suffix), axis=1))[:, -2:]
    np.testing.assert_allclose(np.asarray(model(suffix, snapshot)), np.asarray(expected_suffix), rtol=2e-5, atol=3e-6)
    assert all(c.offset == 13 for c in cache)


def test_rotary_lane_offsets_and_source_bf16_rounding():
    x = mx.arange(2 * 3 * 4 * 8, dtype=mx.float32).reshape(2, 3, 4, 8).astype(mx.bfloat16)
    offsets = mx.array([2, 7])
    batched = rotary(x, offsets, 10000)
    separate = mx.concatenate([rotary(x[i:i+1], offset, 10000) for i, offset in enumerate([2, 7])])
    np.testing.assert_array_equal(np.asarray(batched.astype(mx.float32)), np.asarray(separate.astype(mx.float32)))


SOURCE_EMBED_SCALE = 71.55417527999327


def test_bf16_embedding_scale_rounds_once_like_source():
    torch = pytest.importorskip("torch")
    from mlx2.runtime.models.ricursive import scale_embeddings

    rng = np.random.default_rng(5)
    h = mx.array(rng.normal(size=(4, 512)).astype(np.float32)).astype(mx.bfloat16)
    expected = torch.tensor(np.asarray(h.astype(mx.float32))).bfloat16() * SOURCE_EMBED_SCALE
    actual = scale_embeddings(h, SOURCE_EMBED_SCALE)
    assert actual.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), expected.float().numpy())
    # The pre-repair law rounded the scalar to BF16 (71.5) first.
    assert (np.asarray((h * SOURCE_EMBED_SCALE).astype(mx.float32)) != expected.float().numpy()).any()


def test_bf16_one_token_model_oracle_uses_published_embed_scale():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    mx.random.seed(72)
    args = tiny_args()
    args.embed_scale = SOURCE_EMBED_SCALE
    model = Model(args, nvfp4=False)
    for layer, value in zip(model.layers, [-2.75, -.3, .25, 1.1, 4.0], strict=True):
        layer.self_attn.value_res_lambda = mx.array([value])
    model.set_dtype(mx.bfloat16)
    weights = dict(tree_flatten(model.parameters()))
    actual = np.asarray(model(mx.array([[1]])).astype(mx.float32))
    np.testing.assert_array_equal(actual, torch_reference([[1]], weights, args, torch))


@pytest.mark.parametrize("offset", [1000, 5000, 131071])
def test_rotary_matches_source_at_deep_positions(offset):
    torch = pytest.importorskip("torch")
    from mlx2.runtime.models.ricursive import inverse_frequencies

    dims, theta = 128, 10000.0
    source_inv = 1.0 / (theta ** (torch.arange(0, dims, 2).float() / dims))
    np.testing.assert_array_equal(np.asarray(inverse_frequencies(dims, theta)), source_inv.numpy())
    rng = np.random.default_rng(offset)
    x = mx.array(rng.normal(size=(1, 2, 3, dims)).astype(np.float32)).astype(mx.bfloat16)
    positions = torch.arange(offset, offset + 3).float()
    phase = positions[:, None] * source_inv[None, :]
    emb = torch.cat((phase, phase), dim=-1)
    cos, sin = emb.cos().bfloat16().float(), emb.sin().bfloat16().float()
    y = torch.tensor(np.asarray(x.astype(mx.float32)))
    first, second = y.chunk(2, dim=-1)
    expected = (y * cos + torch.cat((-second, first), dim=-1) * sin).bfloat16()
    actual = rotary(x, offset, theta)
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), expected.float().numpy())


def test_rmsnorm_rounds_before_scale():
    norm = RMSNorm(16, 1e-6)
    x = mx.array(np.random.default_rng(7).normal(size=(3, 16))).astype(mx.bfloat16)
    norm.weight = mx.linspace(.6, 1.7, 16).astype(mx.bfloat16)
    y = x.astype(mx.float32)
    expected = (y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + 1e-6)).astype(mx.bfloat16) * norm.weight
    np.testing.assert_array_equal(np.asarray(norm(x).astype(mx.float32)), np.asarray(expected.astype(mx.float32)))


def fixture_shard(tmp_path):
    args = tiny_args()
    schema = tensor_schema({**asdict(args), "quantization_config": {"quant_algo": "NVFP4"}})
    rng = np.random.default_rng(22)
    offset = 0
    header, payload, raw_arrays = {}, [], {}
    for name, (dtype, shape) in schema.items():
        count = int(np.prod(shape))
        if dtype == "BF16":
            raw = (rng.normal(0, .1, count).astype("<f4").view("<u4") >> 16).astype("<u2")
        elif dtype == "F32":
            raw = np.array([.125], "<f4")
        else:
            raw = rng.integers(1, 127 if dtype == "F8_E4M3" else 256, count, dtype=np.uint8)
        raw_arrays[name] = raw.reshape(shape)
        data = raw.tobytes()
        assert len(data) == count * DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))
    weight_map = dict.fromkeys(schema, "model.safetensors")
    artifact = {"identity": {"path": str(tmp_path)}, "weight_map": weight_map}
    return args, schema, artifact, raw_arrays


def test_source_bytes_load_strictly_and_nvfp4_matches_decoded_weight_oracle(tmp_path):
    args, schema, artifact, raw = fixture_shard(tmp_path)
    validate_headers(tmp_path, artifact["weight_map"], schema)
    model = Model(args)
    weights = list(source_weights(artifact))
    assert not any(name.endswith("input_scale") for name, _ in weights)
    model.load_weights(weights, strict=True)
    linear = model.layers[0].mlp.gate_proj
    prefix = "model.layers.0.mlp.gate_proj."
    x = np.random.default_rng(1).normal(size=(2, args.hidden_size)).astype(np.float32)
    dense = decode_rows(raw[prefix + "weight"], raw[prefix + "weight_scale"], raw[prefix + "weight_scale_2"])
    np.testing.assert_allclose(np.asarray(linear(mx.array(x))), x @ dense.T, rtol=2e-5, atol=1e-4)
    assert model(mx.array([[1, 2]])).shape == (1, 2, args.vocab_size)


def test_headers_refuse_wrong_dtype_and_unindexed_tensor(tmp_path):
    _, schema, artifact, _ = fixture_shard(tmp_path)
    wrong = dict(schema)
    wrong["model.embed_tokens.weight"] = ("F32", schema["model.embed_tokens.weight"][1])
    with pytest.raises(ValueError, match="dtype/shape"):
        validate_headers(tmp_path, artifact["weight_map"], wrong)
    artifact["weight_map"].pop("model.norm.weight")
    with pytest.raises(ValueError, match="topology"):
        validate_headers(tmp_path, artifact["weight_map"], schema)


@pytest.mark.parametrize("suffix", ["input_scale", "weight_scale_2", "weight_scale"])
def test_source_loader_rejects_invalid_scale_values(tmp_path, suffix):
    from mlx2.conversion.nvfp4 import read_header

    _, _, artifact, _ = fixture_shard(tmp_path)
    path = tmp_path / "model.safetensors"
    header, payload = read_header(path)
    name = "model.layers.0.mlp.gate_proj." + suffix
    with path.open("r+b") as f:
        f.seek(payload + header[name]["data_offsets"][0])
        f.write(b"\xff" if suffix == "weight_scale" else struct.pack("<f", float("nan")))
    with pytest.raises(ValueError, match="finite"):
        list(source_weights(artifact))


def test_inspected_source_change_is_refused_before_loading(tmp_path):
    _, _, artifact, _ = fixture_shard(tmp_path)
    p = tmp_path / "model.safetensors"
    artifact["identity"]["files"] = [(p.name, p.stat().st_size, p.stat().st_mtime_ns - 1)]
    with pytest.raises(ValueError, match="changed after inspection"):
        list(source_weights(artifact))


def test_registry_and_adapter_import_do_not_load_mlx():
    out = subprocess.run([sys.executable, "-c", ("import sys; "
                          "from mlx2.adapters import ripple, registry; "
                          "assert registry._RESOLVERS['ricursive'] is registry._ripple; "
                          "assert 'mlx.core' not in sys.modules")],
                         capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr


def test_exact_published_config_and_policy_fail_closed():
    config = json.loads((FIXTURES / "config.json").read_text())
    validate_config(config)
    for field, value in [("embed_scale", None), ("rope_theta", 500000), ("use_valres", False),
                         ("head_dim", 64), ("ffn_clamp_threshold", 0), ("quantization", {})]:
        changed = {**config, field: value}
        with pytest.raises(ValueError):
            validate_config(changed)
    for policy in (None, {}, {"activation_precision": "nvfp4"}, {"activation_precision": "bf16", "extra": True}):
        with pytest.raises(ValueError, match="W4A4"):
            parse_policy(policy)
    assert parse_policy({"activation_precision": "bf16"}) == {"activation_precision": "bf16"}
    assert Capability.MTP not in DESCRIPTOR.capabilities
    assert Capability.TOOLS not in DESCRIPTOR.capabilities
    assert not DESCRIPTOR.metadata["qualified"]
    adapter = object.__new__(RippleAdapter)
    assert adapter.execution_config(max_lanes=1, prefill_step=256)["num_draft"] == 0
    assert adapter.execution_config(max_lanes=4, prefill_step=256)["segment_aware_cohort_size"] == 4
    budget = RippleAdapter.planning_cache_budget(config, mtp=False)
    assert budget.global_layers == 16 and budget.sliding_layers == 64
    assert budget.project(16384) > budget.project(8192)
    assert budget.prefill_transient_bytes(8192, 256) > 0


def test_template_and_reasoning_contract():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    from mlx2.adapters.ordinary_text import _template_history

    template = ImmutableSandboxedEnvironment().from_string((FIXTURES / "chat_template.jinja").read_text())
    messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Hi"}]
    rendered = template.render(messages=_template_history(messages), add_generation_prompt=True)
    assert rendered == "<|im_start|>system\nSystem<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    adapter = object.__new__(RippleAdapter)
    parser = adapter.output_parser({"messages": messages})
    assert parser.channel == "content"
    with pytest.raises(ValueError, match="thinking-off"):
        adapter.output_parser({"messages": messages, "enable_thinking": False})
    # Raw prompts have no template control; classifier scoring sets the flag.
    assert adapter.output_parser({"prompt": "x", "enable_thinking": False}).channel == "content"
    assert adapter.render_prompt({"prompt": "x", "enable_thinking": False}) == "x"
    with pytest.raises(ValueError, match="tool calling"):
        adapter.output_parser({"messages": messages, "tools": [{"type": "function", "function": {"name": "f"}}]})


def test_fused_elementwise_option_is_explicit_and_typed():
    from mlx2.adapters.ripple import split_options

    assert split_options(None) == ({}, {})
    assert split_options({"activation_precision": "bf16", "fused_elementwise": True}) == (
        {"activation_precision": "bf16"}, {"fused_elementwise": True})
    with pytest.raises(ValueError, match="must be a bool"):
        split_options({"fused_elementwise": "yes"})
    # Unknown keys are left in the numerics policy, which then fails closed.
    with pytest.raises(ValueError, match="W4A16"):
        parse_policy(split_options({"activation_precision": "bf16", "speed": 1})[0])
