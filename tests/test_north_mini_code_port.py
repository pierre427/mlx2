"""North Mini Code CPU/static serving contracts."""

import copy
import json
import subprocess
import struct
import sys
from pathlib import Path

import pytest

from mlx2.adapters.north_mini_code import (
    CACHE_LAYOUT,
    NORTH_MINI_CODE,
    NorthMiniCodeAdapter,
    inspect_artifact,
    normalize_messages,
    reasoning_policy,
    _expected_weight_headers,
)
from mlx2.adapters.north_output import NorthOutputParser, parse_actions
from mlx2.adapters.north_memory import NorthCacheBudget
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability, StatePlane


def north_config():
    return {
        "architectures": ["Cohere2MoeForCausalLM"],
        "model_type": "cohere2_moe",
        "hidden_size": 2048,
        "head_dim": 128,
        "num_hidden_layers": 49,
        "intermediate_size": 768,
        "prefix_dense_intermediate_size": 3072,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 262144,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_shared_experts": 0,
        "first_k_dense_replace": 1,
        "prefix_dense_sliding_window_pattern": 1,
        "expert_selection_fn": "sigmoid",
        "sliding_window": 4096,
        "rope_theta": 50000,
        "max_position_embeddings": 500000,
        "use_parallel_block": True,
        "use_qk_norm": False,
        "norm_topk_prob": False,
        "tie_word_embeddings": None,
        "rms_norm_eps": 1e-06,
        "layer_norm_eps": 1e-05,
        "layer_types": [
            "full_attention" if i % 4 == 0 else "sliding_attention"
            for i in range(49)
        ],
    }


def artifact(path: Path):
    config = north_config()
    config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    for index in range(1, 49):
        config["quantization"][f"model.layers.{index}.mlp.gate"] = {
            "group_size": 64,
            "bits": 8,
            "mode": "affine",
        }
    (path / "config.json").write_text(json.dumps(config))
    expected = _expected_weight_headers(config)
    offset = 0
    header = {}
    weight_map = {}
    for name, (dtype, shape) in expected.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = elements * (4 if dtype == "U32" else 2)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        weight_map[name] = "model.safetensors"
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    with (path / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.truncate(8 + len(raw) + offset)
    return path


def rewrite_header(path: Path, mutate):
    shard = path / "model.safetensors"
    with shard.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    mutate(header)
    raw = json.dumps(header, separators=(",", ":")).encode()
    payload = max(record["data_offsets"][1] for record in header.values())
    with shard.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.truncate(8 + len(raw) + payload)


def test_import_and_registry_inspection_do_not_import_mlx(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mlx2.adapters.north_mini_code; "
                "import mlx2.adapters.north_output; "
                "assert 'mlx.core' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    resolved = inspect_model(artifact(tmp_path))
    assert resolved.adapter_type is NorthMiniCodeAdapter
    assert resolved.default_route == "ordinary"
    assert resolved.artifact["qualification"] == "pending"


def test_artifact_contract_and_identity(tmp_path):
    root = artifact(tmp_path)
    before = inspect_artifact(root)
    assert before["sliding_layers"] == 36
    assert before["global_layers"] == 13
    assert before["supports_native_mtp"] is False
    (root / "tokenizer_config.json").write_text('{"changed": true}')
    assert inspect_artifact(root)["identity"]["fingerprint"] != before["identity"]["fingerprint"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("hidden_size", 1024),
        ("num_experts", 64),
        ("expert_selection_fn", "softmax"),
        ("use_parallel_block", False),
        ("use_qk_norm", True),
        ("sliding_window", 2048),
    ],
)
def test_wrong_topology_fails_closed_before_tensor_import(tmp_path, key, value):
    root = artifact(tmp_path)
    config = north_config()
    config[key] = value
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="topology"):
        inspect_artifact(root)


def test_wrong_layer_phase_and_embedded_draft_fail_closed(tmp_path):
    root = artifact(tmp_path)
    config = north_config()
    config["layer_types"][0:2] = ["sliding_attention", "full_attention"]
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="layer order"):
        inspect_artifact(root)
    artifact(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"]["mtp.fc.weight"] = "model.safetensors"
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="speculative"):
        inspect_artifact(root)


def test_missing_or_traversing_shard_fails_closed(tmp_path):
    root = artifact(tmp_path)
    (root / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="local safetensors"):
        inspect_artifact(root)


def test_duplicate_config_and_index_keys_fail_closed(tmp_path):
    root = artifact(tmp_path)
    (root / "config.json").write_text('{"model_type":"cohere2_moe","model_type":"cohere2_moe"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        inspect_artifact(root)
    artifact(root)
    (root / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model.safetensors","x":"model.safetensors"}}'
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        inspect_artifact(root)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="local safetensors"):
        inspect_artifact(root)


def test_header_dtype_shape_and_index_mapping_fail_closed(tmp_path):
    root = artifact(tmp_path)
    rewrite_header(
        root,
        lambda header: header["model.norm.weight"].update(dtype="F16"),
    )
    with pytest.raises(ValueError, match="dtype/shape"):
        inspect_artifact(root)

    root = artifact(tmp_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    del index["weight_map"]["model.norm.weight"]
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index/shard mismatch"):
        inspect_artifact(root)


def test_north_cache_budget_is_architecture_bound_monotone_and_ordinary_only():
    budget = NorthCacheBudget.from_config(north_config(), mtp=False)
    points = [budget.project(tokens) for tokens in (0, 4096, 32768, 500000)]
    assert points == sorted(points)
    assert 27 < points[-1] / (1 << 30) < 29
    assert budget.global_layers == 13 and budget.sliding_layers == 36
    assert budget.checkpoint_copies == 4
    # Measured on this model on an M3 Pro, 2026-09-19: the k=2 verify
    # transient is 0.044-0.071 GiB/lane across 1K/4K/16K x 1/2/4 lanes.  The
    # 3.1 that stood here was the dense Qwen3.8-27B figure used as a
    # placeholder.  See provenance/lane-transient-moe.json and
    # tests/test_lane_transient_calibration.py.
    assert budget.transient_gib_per_lane == C.MOE_TRANSIENT_GIB_PER_LANE
    assert budget.transient_gib_per_lane == 0.35
    with pytest.raises(ValueError, match="native MTP"):
        NorthCacheBudget.from_config(north_config(), mtp=True)
    adapter = object.__new__(NorthMiniCodeAdapter)
    adapter.config = north_config()
    assert adapter.cache_budget(mtp=False).project(500000) == points[-1]


def test_descriptor_declares_current_cache_and_no_unimplemented_speculation():
    assert NORTH_MINI_CODE.cache_layout == CACHE_LAYOUT
    assert {
        Capability.APC_V2,
        Capability.LAYERED_CACHE,
        Capability.CONTINUOUS_BATCH,
        Capability.PREFIX_REUSE,
    } <= NORTH_MINI_CODE.capabilities
    assert Capability.MTP not in NORTH_MINI_CODE.capabilities
    assert Capability.EXTERNAL_DRAFT not in NORTH_MINI_CODE.capabilities
    assert StatePlane.DRAFT not in NORTH_MINI_CODE.state_planes
    assert NORTH_MINI_CODE.metadata["qualification"] == "pending"


def test_null_tie_metadata_resolves_to_required_tied_head():
    code = r"""
import json
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx2.runtime.models.cohere2_moe import Model, ModelArgs
cfg = dict(
    model_type="cohere2_moe", hidden_size=16, head_dim=4,
    num_hidden_layers=2, intermediate_size=8,
    prefix_dense_intermediate_size=24, num_attention_heads=4,
    num_key_value_heads=2, vocab_size=32, num_experts=4,
    num_experts_per_tok=2, first_k_dense_replace=1,
    sliding_window=4, layer_types=["full_attention", "sliding_attention"],
    tie_word_embeddings=None,
)
args = ModelArgs.from_dict(cfg)
model = Model(args)
assert args.tie_word_embeddings is True
assert not hasattr(model, "lm_head")
weights = {"model.embed_tokens.weight": mx.zeros((32, 16)),
           "lm_head.weight": mx.ones((32, 16))}
assert "lm_head.weight" not in model.sanitize(weights)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_real_configs_declare_null_tie_and_indexes_omit_head():
    roots = [
        Path.home() / "mlx-models/North-Mini-Code-1.0-mlx-4bit",
        Path.home() / "mlx-models/North-Mini-Code-1.0-mlx-8bit",
    ]
    if not all(root.is_dir() for root in roots):
        pytest.skip("local North artifacts are not available")
    for root in roots:
        config = json.loads((root / "config.json").read_text())
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        assert config.get("tie_word_embeddings") is None
        assert "model.embed_tokens.weight" in index
        assert "lm_head.weight" not in index
        assert inspect_artifact(root)["supports_native_mtp"] is False


def test_registry_and_profile_reject_native_mtp(tmp_path):
    root = artifact(tmp_path)
    assert resolve_adapter(root) is NorthMiniCodeAdapter
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(root, mtp=True)
    assert NorthMiniCodeAdapter.profile_name(False) == "north-mini-code-apcv2-ordinary"
    with pytest.raises(ValueError, match="native MTP"):
        NorthMiniCodeAdapter.profile_name(True)


def test_execution_policy_has_no_hidden_legacy_fallback():
    adapter = object.__new__(NorthMiniCodeAdapter)
    config = adapter.execution_config(max_lanes=20, prefill_step=2048)
    assert config["num_draft"] == 0
    assert config["segment_aware_cohort_size"] == 20
    assert config["segment_aware_live_tip"] is False
    with pytest.raises(ValueError, match="no qualified overrides"):
        NorthMiniCodeAdapter("/missing", execution_policy={"num_draft": 2})


def test_prompt_policy_uses_native_north_controls_only():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages, self.kwargs = messages, kwargs
            return [1, 2, 3]

    adapter = object.__new__(NorthMiniCodeAdapter)
    adapter.tokenizer = Tokenizer()
    assert adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": "hi"}]}
    ) == [1, 2, 3]
    # Unspecified means thinking on, the vendor template's own default.
    assert adapter.tokenizer.kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "reasoning": True,
        "reasoning_effort": "high",
        "skip_thinking": False,
        "tools": None,
    }
    assert adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"}
    ) == [1, 2, 3]
    assert adapter.tokenizer.kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "reasoning": False,
        "reasoning_effort": "none",
        "skip_thinking": True,
        "tools": None,
    }
    adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "ultra"}
    )
    assert adapter.tokenizer.kwargs["reasoning"] is True
    assert adapter.tokenizer.kwargs["skip_thinking"] is False
    # Thinking is the default, as in the vendor template; clients turn it off.
    assert reasoning_policy({}) == ("high", True)
    assert reasoning_policy({"enable_thinking": False}) == ("none", False)
    assert reasoning_policy({"reasoning_effort": "none"}) == ("none", False)
    # Constrained requests think too: the grammar defers past <|END_THINKING|>.
    assert reasoning_policy({"response_format": {"type": "json_object"}}) == ("high", True)
    assert reasoning_policy({"grammar": "ab"}) == ("high", True)
    assert reasoning_policy(
        {"response_format": {"type": "json_object"}, "enable_thinking": False}
    ) == ("none", False)
    assert reasoning_policy({"enable_thinking": True}) == ("high", True)
    assert reasoning_policy({"reasoning_effort": "high"}) == ("high", True)
    assert reasoning_policy(
        {"reasoning_effort": "high", "enable_thinking": False}
    ) == ("high", False)
    assert adapter.output_parser({"messages": [], "enable_thinking": False}).channel == "content"
    assert adapter.output_parser(
        {"messages": [], "reasoning_effort": "high"}
    ).channel == "reasoning_content"


def test_action_processor_is_history_pure_thinking_aware_and_probe_safe():
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.north_mini_code import NorthActionProcessor
    from mlx2.runtime.processor_probe import isolated_logits_processor

    logits = mx.arange(64, dtype=mx.float32)[None, :]
    processor = NorthActionProcessor(
        prompt_length=2,
        action_open=(10, 11),
        thinking_close=(8, 9),
    )

    def finite(history):
        masked = processor(mx.array([1, 2, *history]), logits)
        return set(np.flatnonzero(np.isfinite(np.asarray(masked[0]))).tolist())

    assert finite([]) == set(range(64))
    assert finite([7, 8]) == set(range(64))
    assert finite([7, 8, 9]) == {10}
    assert finite([7, 8, 9, 10]) == {11}
    assert finite([7, 8, 9, 10, 11]) == set(range(64))
    first = processor(mx.array([1, 2, 7, 8, 9]), logits)
    second = processor(mx.array([1, 2, 7, 8, 9]), logits)
    assert mx.array_equal(first, second)
    assert mx.array_equal(
        isolated_logits_processor(processor)(mx.array([1, 2, 7, 8, 9]), logits),
        first,
    )


def test_adapter_action_processors_cover_required_and_named_only():
    class Tokenizer:
        markers = {
            "<|START_ACTION|>": [10, 11],
            "<|END_THINKING|>": [8, 9],
        }

        def encode(self, text, **kwargs):
            return self.markers[text]

    adapter = object.__new__(NorthMiniCodeAdapter)
    adapter.tokenizer = Tokenizer()
    base = {"messages": [], "tools": TOOLS}
    named = {"type": "function", "function": {"name": "echo"}}

    required = adapter.request_logits_processors(
        {**base, "tool_choice": "required", "enable_thinking": False},
        prompt_length=7,
    )[0]
    selected = adapter.request_logits_processors(
        {**base, "tool_choice": named, "enable_thinking": True},
        prompt_length=7,
    )[0]
    assert required.prompt_length == selected.prompt_length == 7
    assert required.action_open == selected.action_open == (10, 11)
    assert required.thinking_close is None
    assert selected.thinking_close == (8, 9)
    assert adapter.request_logits_processors(
        {**base, "tool_choice": "auto"}, prompt_length=7
    ) == ()
    assert adapter.request_logits_processors(
        {**base, "tool_choice": "none"}, prompt_length=7
    ) == ()


def test_message_normalization_is_text_only_and_nonmutating():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "tool_calls": [
                {"function": {"name": "echo", "arguments": '{"value": 2}'}}
            ],
        }
    ]
    original = copy.deepcopy(messages)
    normalized = normalize_messages(messages)
    assert normalized[0]["content"] == "done"
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {"value": 2}
    assert messages == original
    with pytest.raises(ValueError, match="text content only"):
        normalize_messages([{"content": [{"type": "image_url"}]}])


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "required": ["value"],
                "additionalProperties": False,
                "properties": {"value": {"type": "integer"}},
            },
        },
    }
]


@pytest.mark.parametrize("split", [1, 2, 7, 4096])
def test_output_channels_and_actions_are_chunk_safe(split):
    text = (
        "thought<|END_THINKING|><|START_TEXT|>answer<|END_TEXT|>"
        '<|START_ACTION|>[{"tool_call_id":"7","tool_name":"echo",'
        '"parameters":{"value":2}}]<|END_ACTION|><|END_OF_TURN_TOKEN|>'
    )
    parser = NorthOutputParser(chat=True, thinking=True, tools=TOOLS)
    events = []
    for start in range(0, len(text), split):
        events.extend(parser.push(text[start : start + split]))
    events.extend(parser.push("", final=True))
    assert "".join(e.get("reasoning_content", "") for e in events) == "thought"
    assert "".join(e.get("content", "") for e in events) == "answer"
    call = next(e["tool_calls"][0] for e in events if "tool_calls" in e)
    assert call["id"] == "7"
    assert call["function"]["name"] == "echo"
    assert json.loads(call["function"]["arguments"]) == {"value": 2}


def test_incomplete_action_is_only_tolerated_for_length_finish():
    partial = (
        "thought<|END_THINKING|><|START_ACTION|>"
        '[{"tool_name":"echo","parameters":{"value":'
    )
    parser = NorthOutputParser(chat=True, thinking=True, tools=TOOLS)
    events = parser.finish(partial, "length")
    assert "".join(event.get("reasoning_content", "") for event in events) == (
        "thought"
    )
    assert not any(event.get("tool_calls") for event in events)

    parser = NorthOutputParser(chat=True, thinking=True, tools=TOOLS)
    with pytest.raises(ValueError, match="incomplete North action block"):
        parser.finish(partial, "stop")


@pytest.mark.parametrize("split", range(1, 18))
def test_visible_stop_ignores_reasoning_and_is_chunk_safe(split):
    text = (
        "reasoning repeats MLX2_READY"
        "<|END_THINKING|><|START_TEXT|>MLX2_READY trailing"
    )
    parser = NorthOutputParser(
        chat=True, thinking=True, stops=("_READY",)
    )
    events = []
    for start in range(0, len(text), split):
        events.extend(parser.push(text[start : start + split]))
    events.extend(parser.push("", final=True))
    assert "".join(e.get("reasoning_content", "") for e in events) == (
        "reasoning repeats MLX2_READY"
    )
    assert "".join(e.get("content", "") for e in events) == "MLX2"
    assert parser.stopped is True


@pytest.mark.parametrize("split", range(1, 8))
def test_raw_completion_stop_remains_chunk_safe(split):
    parser = NorthOutputParser(chat=False, stops=("_READY",))
    events = []
    text = "MLX2_READY trailing"
    for start in range(0, len(text), split):
        events.extend(parser.push(text[start : start + split]))
    events.extend(parser.push("", final=True))
    assert "".join(e.get("content", "") for e in events) == "MLX2"
    assert parser.stopped is True


@pytest.mark.parametrize(
    "value",
    [
        "{}",
        "[]",
        '[{"tool_name":"missing","parameters":{}}]',
        '[{"tool_name":"echo","parameters":[]}]',
        '[{"tool_name":"echo","parameters":{}}]',
        '[{"tool_name":"echo","parameters":{"value":2,"extra":3}}]',
    ],
)
def test_action_parser_fails_closed(value):
    with pytest.raises((TypeError, ValueError)):
        parse_actions(value, TOOLS)


def test_north_auto_parallel_false_drops_a_second_call():
    from mlx2.adapters.north_output import ACTION_CLOSE, ACTION_OPEN

    parser = NorthOutputParser(
        chat=True,
        thinking=False,
        tools=TOOLS,
        parallel_tool_calls=False,
    )
    action = {"tool_name": "echo", "parameters": {"value": 2}}
    text = ACTION_OPEN + json.dumps([action, action], separators=(",", ":")) + ACTION_CLOSE
    events = parser.push(text, final=True)
    calls = [call for event in events for call in event.get("tool_calls", ())]
    assert len(calls) == 1
    assert parser.tool_call_constraint_truncations == 1


def test_tiny_native_cpu_split_replay_and_layered_cache():
    code = r"""
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx2.runtime.models.cohere2_moe import Model, ModelArgs
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
mx.random.seed(7)
args = ModelArgs(
    hidden_size=16, head_dim=4, num_hidden_layers=2,
    intermediate_size=8, prefix_dense_intermediate_size=24,
    num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
    num_experts=4, num_experts_per_tok=2, first_k_dense_replace=1,
    sliding_window=4, layer_types=["full_attention", "sliding_attention"],
)
model = Model(args)
model.eval()
tokens = mx.array([[1, 2, 3, 4, 5, 6]])
full_cache = model.make_cache()
full = model(tokens, cache=full_cache)
split_cache = model.make_cache()
model(tokens[:, :3], cache=split_cache)
tail = model(tokens[:, 3:], cache=split_cache)
mx.eval(full, tail)
assert float(mx.max(mx.abs(full[:, 3:] - tail))) < 1e-5
assert isinstance(split_cache[0], KVCache)
assert isinstance(split_cache[1], RotatingKVCache)
assert split_cache[0].offset == 6 and split_cache[1].offset == 6
assert model.apc_v2_layout == "north-mini-code-layer-segments-v1"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_tiny_cpu_matches_mined_source_when_checkout_is_available():
    source = Path.home() / "Desktop/mlx-uag/mlx-lm-unified"
    if not source.is_dir():
        pytest.skip("mined source checkout is not available")
    code = rf"""
import sys
sys.path.insert(0, {str(source)!r})
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx.utils import tree_flatten
from mlx2.runtime.models.cohere2_moe import Model as Port, ModelArgs as PortArgs
from mlx_lm.models.cohere2_moe import Model as Ref, ModelArgs as RefArgs
kwargs = dict(
    model_type="cohere2_moe", hidden_size=16, head_dim=4,
    num_hidden_layers=2, intermediate_size=8,
    prefix_dense_intermediate_size=24, num_attention_heads=4,
    num_key_value_heads=2, vocab_size=32, num_experts=4,
    num_experts_per_tok=2, first_k_dense_replace=1,
    sliding_window=4, layer_types=["full_attention", "sliding_attention"],
)
mx.random.seed(9)
# The mined source predates the reference's dense-prefix RoPE (force_rope), so
# compare the math the two still share: a prefix pattern other than 1 leaves
# the dense-prefix layer unrotated in both.
port = Port(PortArgs(**kwargs, prefix_dense_sliding_window_pattern=2))
assert port.model.layers[0].self_attn.rope is None
ref = Ref(RefArgs(**kwargs))
ref.load_weights(tree_flatten(port.parameters()), strict=True)
tokens = mx.array([[1, 2, 3, 4, 5, 6]])
port_full = port(tokens, cache=port.make_cache())
ref_full = ref(tokens, cache=ref.make_cache())
port_cache, ref_cache = port.make_cache(), ref.make_cache()
port(tokens[:, :3], cache=port_cache)
ref(tokens[:, :3], cache=ref_cache)
port_tail = port(tokens[:, 3:], cache=port_cache)
ref_tail = ref(tokens[:, 3:], cache=ref_cache)
mx.eval(port_full, ref_full, port_tail, ref_tail)
assert float(mx.max(mx.abs(port_full - ref_full))) == 0.0
assert float(mx.max(mx.abs(port_tail - ref_tail))) == 0.0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_adapter_close_is_idempotent_for_shared_serving_cleanup(monkeypatch):
    import mlx.core as mx

    clears = []
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(True))
    adapter = object.__new__(NorthMiniCodeAdapter)
    adapter.model = object()
    adapter.tokenizer = object()
    assert adapter.close() is None
    assert adapter.model is None
    assert adapter.tokenizer is None
    assert clears == [True]
    assert adapter.close() is None
    assert clears == [True]


def test_north_declares_grammar_deferral_marker_and_answer_envelope():
    table = {"<|END_THINKING|>": [255011], "<|START_TEXT|>": [255012], "<|END_TEXT|>": [255013]}

    class Tokenizer:
        def encode(self, text, **_kw):
            return table[text]

    adapter = object.__new__(NorthMiniCodeAdapter)
    adapter.tokenizer = Tokenizer()
    assert adapter.thinking_close_token_ids() == (255011,)
    assert adapter.structured_envelope_token_ids() == ((255012,), (255013,))
    table["<|START_TEXT|>"] = [1, 2]  # not atomic: undeclared, framing stays masked
    assert adapter.structured_envelope_token_ids() is None


def test_artifact_check_fails_closed_on_a_missing_norm_selector(tmp_path):
    """An artifact that drops rms_norm_eps must be rejected, not mis-normalized.

    Upstream Cohere2Moe selects the norm *class* from rms_norm_eps (present ->
    RMSNorm eps 1e-6, absent -> mean-centred LayerNorm eps 1e-5).  Both have the
    same parameter shapes, so a config that lost this key would load cleanly and
    serve the wrong normalization in silence.  See
    mlx2/runtime/models/cohere2_moe._norm_layer and tests/test_north_norm_choice.py.
    """
    root = artifact(tmp_path)
    for field in ("rms_norm_eps", "layer_norm_eps"):
        config = json.loads((root / "config.json").read_text())
        assert config[field] is not None
        del config[field]
        (root / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="topology"):
            inspect_artifact(root)
        artifact(tmp_path)


def test_artifact_check_rejects_a_flipped_norm_selector(tmp_path):
    root = artifact(tmp_path)
    config = json.loads((root / "config.json").read_text())
    config["rms_norm_eps"] = 1e-05
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="topology"):
        inspect_artifact(root)


def _tiny_north_kwargs(**overrides):
    kwargs = dict(
        model_type="cohere2_moe", hidden_size=64, head_dim=16,
        num_hidden_layers=8, intermediate_size=32,
        prefix_dense_intermediate_size=96, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=96, rope_theta=50000.0,
        rms_norm_eps=1e-6, sliding_window=6, num_experts=8,
        num_experts_per_tok=2, first_k_dense_replace=1,
        layer_types=[
            "full_attention" if i % 4 == 0 else "sliding_attention"
            for i in range(8)
        ],
    )
    kwargs.update(overrides)
    return kwargs


def test_dense_prefix_layer_is_rotated_like_the_reference():
    # Cohere2Moe ``force_rope``: with prefix_dense_sliding_window_pattern 1 the
    # dense-prefix layer carries RoPE although it is full attention.  North
    # ran its layer 0 unrotated until 2026-09-23; the real artifact's held-out
    # perplexity was 50.8 against 8.9 with the reference layout.
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    rotated = lambda model: [
        layer.self_attn.rope is not None for layer in model.model.layers
    ]
    north = Model(ModelArgs.from_dict(_tiny_north_kwargs(
        prefix_dense_sliding_window_pattern=1
    )))
    assert rotated(north) == [True, True, True, True, False, True, True, True]
    # The field is declared, so from_dict keeps it rather than dropping it.
    other = Model(ModelArgs.from_dict(_tiny_north_kwargs(
        prefix_dense_sliding_window_pattern=2
    )))
    assert rotated(other)[0] is False
    # The reference config class defaults the pattern to 1.
    default = Model(ModelArgs.from_dict(_tiny_north_kwargs()))
    assert rotated(default)[0] is True


def test_tiny_north_matches_the_transformers_reference():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Cohere2MoeForCausalLM"):
        pytest.skip("transformers has no Cohere2Moe reference")
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    kwargs = _tiny_north_kwargs()
    kwargs.pop("model_type")
    config = transformers.Cohere2MoeConfig(
        **kwargs, layer_norm_eps=1e-5, logit_scale=1.0,
        num_shared_experts=0, norm_topk_prob=False,
        expert_selection_fn="sigmoid", use_parallel_block=True,
        use_qk_norm=False, max_position_embeddings=4096,
        prefix_dense_sliding_window_pattern=1, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    reference = transformers.Cohere2MoeForCausalLM(config).eval()
    if not hasattr(reference.model.layers[0].self_attn, "force_rope"):
        pytest.skip("installed transformers predates the dense-prefix RoPE")
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            parameter.copy_(
                torch.randn_like(parameter) * 0.2
                + (1.0 if name.endswith("norm.weight") else 0.0)
            )
    weights = {}
    for key, value in reference.state_dict().items():
        value = value.detach().float().numpy()
        if "rotary_emb" in key or key == "lm_head.weight":
            continue
        if key.endswith(".mlp.experts.gate_up_proj"):
            prefix = key[: -len(".experts.gate_up_proj")]
            half = value.shape[1] // 2
            weights[prefix + ".switch_mlp.gate_proj.weight"] = mx.array(value[:, :half])
            weights[prefix + ".switch_mlp.up_proj.weight"] = mx.array(value[:, half:])
        elif key.endswith(".mlp.experts.down_proj"):
            weights[key.replace(".experts.down_proj", ".switch_mlp.down_proj.weight")] = mx.array(value)
        else:
            weights[key] = mx.array(value)
    port = Model(ModelArgs.from_dict(_tiny_north_kwargs(max_position_embeddings=4096)))
    port.load_weights(list(port.sanitize(weights).items()), strict=True)
    tokens = np.random.RandomState(1).randint(0, 96, (1, 13))
    with torch.no_grad():
        expected = reference(torch.tensor(tokens)).logits.numpy()
    actual = np.array(port(mx.array(tokens)))
    assert np.abs(expected - actual).max() < 1e-4
