"""Synthetic contract/rollback regressions; no real-checkpoint qualification."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.adapters import nanbeige43 as adapter
from mlx2.adapters import nanbeige43_fused as fused
from mlx2.adapters.registry import inspect_model
from mlx2.runtime.attention_tree_executor import AttentionTreeExecutor
from mlx2.runtime.models.standard_decoder import Model, ModelArgs
from mlx2.sampling_defaults import resolve_sampling, vendor_sampling

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def source_receipt(record_property):
    record_property(
        "source_revision",
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    record_property(
        "working_source_sha256",
        hashlib.sha256(
            (ROOT / "src/mlx2/adapters/nanbeige43.py").read_bytes()
            + (ROOT / "src/mlx2/adapters/nanbeige43_fused.py").read_bytes()
        ).hexdigest(),
    )


def tiny(record_property, *, packed=False, dtype=mx.float32):
    mx.random.seed(13)
    args = ModelArgs(
        model_type="llama",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        rms_norm_eps=1e-5,
        vocab_size=32,
        max_position_embeddings=512,
        rope_theta=120000000,
        tie_word_embeddings=False,
    )
    model = Model(args)
    model.set_dtype(dtype)
    nn.quantize(model, bits=8, group_size=64)
    mx.eval(model.parameters())
    from mlx.utils import tree_flatten

    digest = hashlib.sha256()
    for name, value in tree_flatten(model.parameters()):
        digest.update(name.encode())
        digest.update(np.asarray(value.astype(mx.float32)).tobytes())
    record_property(
        "checkpoint", "synthetic-nanbeige43-q8g64@sha256:" + digest.hexdigest()
    )
    if packed:
        fused.install(model)
    return model


def artifact(tmp_path, record_property):
    config = {**adapter.TOPOLOGY, "attention_bias": False, "mlp_bias": False}
    (tmp_path / "config.json").write_text(json.dumps(config))
    names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    (tmp_path / "model.safetensors").write_bytes(b"synthetic-header")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "model.safetensors")})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "enable_thinking preserve_thinking tool_call_format <function="
            }
        )
    )
    info = adapter.inspect_artifact(tmp_path)
    record_property(
        "checkpoint", "synthetic-nanbeige43-artifact@" + info["identity"]["fingerprint"]
    )
    return info


def test_artifact_dispatch_uses_topology_not_directory_name(tmp_path, record_property):
    artifact(tmp_path, record_property)
    result = inspect_model(tmp_path)
    assert result.adapter_type is adapter.Nanbeige43Adapter
    assert result.descriptor.family == "nanbeige4.3"
    assert result.default_route == "ordinary"
    assert not adapter.matches_config({**adapter.TOPOLOGY, "rope_theta": 10000})


def test_artifact_refuses_changed_tensor_math_and_missing_template(
    tmp_path, record_property
):
    artifact(tmp_path, record_property)
    config = json.loads((tmp_path / "config.json").read_text())
    config["attention_bias"] = True
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unbiased"):
        adapter.inspect_artifact(tmp_path)
    config["attention_bias"] = False
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="template"):
        adapter.inspect_artifact(tmp_path)


def test_model_card_sampling_and_explicit_settings(tmp_path, record_property):
    artifact(tmp_path, record_property)
    obj = object.__new__(adapter.Nanbeige43Adapter)
    request = {"tools": [{"type": "function"}]}
    values, receipt = resolve_sampling(
        request, vendor_sampling(obj, request), thinking=True
    )
    assert values["temperature"] == 1.0 and values["top_p"] == 0.95
    assert values["top_k"] == 0 and receipt["profile"] == "agentic"
    request.update(temperature=0, sampling_profile="general")
    values, receipt = resolve_sampling(
        request, vendor_sampling(obj, request), thinking=True
    )
    assert values["temperature"] == 0 and receipt["profile"] == "general"


def test_template_policy_and_tool_history_are_nonmutating(tmp_path, record_property):
    artifact(tmp_path, record_property)
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "f", "arguments": '{"n":9007199254740993}'}}
                ],
            }
        ]
    }
    before = copy.deepcopy(request)
    normalized = adapter._messages(request)
    assert (
        normalized[0]["tool_calls"][0]["function"]["arguments"]["n"] == 9007199254740993
    )
    assert request == before
    assert adapter._prompt_options(request)["preserve_thinking"] is True
    assert (
        adapter._prompt_options({"messages": [{"role": "user", "content": "hi"}]})[
            "preserve_thinking"
        ]
        is False
    )
    assert (
        adapter._prompt_options({"enable_thinking": False})["enable_thinking"] is False
    )
    with pytest.raises(ValueError, match="xml"):
        adapter._prompt_options({"tool_call_format": "json"})


def test_reasoning_and_fragmented_xml_tools(tmp_path, record_property):
    artifact(tmp_path, record_property)
    obj = object.__new__(adapter.Nanbeige43Adapter)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                },
            },
        }
    ]
    parser = obj.output_parser({"messages": [], "tools": tools})
    text = "check</think>\n\n<tool_call><function=lookup><parameter=n>9007199254740993</parameter></function></tool_call>"
    events = []
    for char in text:
        events.extend(parser.push(char))
    events.extend(parser.push("", final=True))
    calls = [call for event in events for call in event.get("tool_calls", [])]
    assert json.loads(calls[0]["function"]["arguments"]) == {"n": 9007199254740993}
    assert "".join(e.get("reasoning_content", "") for e in events) == "check"
    assert not "".join(e.get("content", "") for e in events)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_packed_projection_prefill_and_incremental_reference(record_property, dtype):
    model = tiny(record_property, dtype=dtype)
    cache = model.make_cache()
    tokens = mx.array([[1, 2, 3, 4]])
    reference = model(tokens, cache)
    mx.eval(reference)
    expected_cache = copy.deepcopy(cache)
    expected_next = model(mx.array([[5]]), expected_cache)
    mx.eval(expected_next)
    fused.install(model)
    packed_cache = model.make_cache()
    actual = model(tokens, packed_cache)
    mx.eval(actual)
    next_logits = model(mx.array([[5]]), packed_cache)
    mx.eval(next_logits)
    tol = 0.015 if dtype == mx.bfloat16 else 1e-5
    assert mx.allclose(reference, actual, atol=tol, rtol=tol).item()
    assert mx.allclose(expected_next, next_logits, atol=tol, rtol=tol).item()
    for a, b in zip(packed_cache, expected_cache):
        assert a.offset == b.offset == 5
        assert mx.allclose(a.keys, b.keys, atol=tol, rtol=tol).item()
    assert fused.status(model)["selected"] is True
    assert fused.status(model)["observed_used"] is False


@pytest.mark.parametrize("packed", [False, True])
def test_branched_ast_rows_commit_and_abort_match_serial(record_property, packed):
    model = tiny(record_property, packed=packed)
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache))
    original = copy.deepcopy(cache)
    tokens = [4, 5, 6, 7, 8]
    parents = [-1, 0, 0, 1, 2]
    executor = AttentionTreeExecutor.create()
    logits, features, txn = executor.forward(
        model, tokens, parents, cache, (1,), cached=True
    )
    mx.eval(logits, features)
    for row in range(len(tokens)):
        path = []
        i = row
        while i >= 0:
            path.append(tokens[i])
            i = parents[i]
        ref = copy.deepcopy(original)
        for token in reversed(path):
            expected = model(mx.array([[token]]), ref)
        assert mx.allclose(logits[:, row], expected[:, -1], atol=1e-4, rtol=1e-4).item()
    txn.abort()
    assert [c.offset for c in cache] == [3, 3]
    logits, features, txn = executor.forward(
        model, tokens, parents, cache, (1,), cached=True
    )
    mx.eval(logits, features)
    txn.commit_paths([[0, 2, 4]])
    got = model(mx.array([[9]]), cache)
    ref = copy.deepcopy(original)
    for token in [4, 6, 8, 9]:
        expected = model(mx.array([[token]]), ref)
    assert mx.allclose(got, expected, atol=1e-4, rtol=1e-4).item()
    assert [c.offset for c in cache] == [7, 7]


def test_unsupported_drafters_and_policy_types_fail_before_loading(
    tmp_path, record_property
):
    artifact(tmp_path, record_property)
    assert adapter.split_policy({"nanbeige_fused": True}) == (True, {})
    _, policy = adapter.split_policy(
        {"draft_model": "prompt_lookup", "ast_topology": "3x5"}
    )
    assert policy["num_draft"] == 5
    for policy in [
        {"draft_model": "arbitrary-drafter"},
        {"nanbeige_fused": "yes"},
        {"unknown": 1},
    ]:
        with pytest.raises(ValueError):
            adapter.split_policy(policy)


def test_request_template_controls_survive_validation(tmp_path, record_property):
    artifact(tmp_path, record_property)
    from mlx2.server import validate_request

    request = validate_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {
                "preserve_thinking": True,
                "tool_call_format": "xml",
            },
        }
    )
    assert request["preserve_thinking"] is True and request["tool_call_format"] == "xml"
    with pytest.raises(ValueError, match="boolean"):
        validate_request({"messages": [], "preserve_thinking": "true"})


@pytest.mark.parametrize("packed", [False, True])
def test_prompt_lookup_ast_generator_matches_ordinary_greedy(record_property, packed):
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    model = tiny(record_property, packed=packed)
    prompt = [3, 9, 4, 3, 9, 4, 3, 9, 4, 11, 3, 9]
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache)[:, -1]
    reference = []
    for _ in range(16):
        token = int(mx.argmax(logits).item())
        reference.append(token)
        logits = model(mx.array([[token]]), cache)[:, -1]
    draft = PromptLookupDraftModel(model, num_draft=15)
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="synthetic-nanbeige43-q8g64",
        num_draft=15,
        completion_batch_size=4,
        dynamic_singleton_tree=True,
        dynamic_tree_max_width=4,
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3},
        target_tree_executor_provider=AttentionTreeExecutor.create(),
        memory_headroom=lambda: 1 << 40,
    )
    uid = batch.insert(
        [prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0.0}]
    )[0]
    actual = []
    for _ in range(100):
        _, responses = batch.next()
        actual.extend(r.token for r in responses if r.uid == uid)
        if uid not in batch.lanes:
            break
    assert actual == reference
    assert batch.scheduler_stats["external_tree_rounds"] > 0


def test_template_controls_fail_closed_on_other_adapters(tmp_path, record_property):
    artifact(tmp_path, record_property)
    from mlx2.serving import validate_template_request

    with pytest.raises(ValueError, match="not supported"):
        validate_template_request(object(), {"preserve_thinking": False})
    validate_template_request(
        adapter.Nanbeige43Adapter,
        {"preserve_thinking": False, "tool_call_format": "xml"},
    )
    # The same declaration is published verbatim by /v1/status.
    assert json.loads(json.dumps(adapter.Nanbeige43Adapter.chat_template_options()))[
        "tool_call_format"
    ]["supported"] == ["xml"]


def test_explicit_ast_selects_attention_tree_without_hybrid_dtype_gate(
    tmp_path, record_property
):
    artifact(tmp_path, record_property)
    from types import SimpleNamespace

    from mlx2.server import apply_ast_invocation

    args = SimpleNamespace(
        model=str(tmp_path),
        ast=True,
        ast_drafter="pld",
        ast_topology="3x5",
        ast_graft_pld=False,
        draft_model=None,
        external_draft=False,
        native_mtp=False,
        prompt_lookup=False,
    )
    policy = apply_ast_invocation(
        args, {"nanbeige_fused": True}, inspect_model(tmp_path)
    )
    assert args.external_draft and args.ast_route["tree_tiles"]
    assert policy["draft_model"] == "prompt_lookup" and policy["nanbeige_fused"] is True
    fused_enabled, bound = adapter.split_policy(policy)
    assert fused_enabled and bound["ast_topology"] == "3x5"


def test_sentencepiece_streaming_decodes_space_and_utf8_fallback(
    tmp_path, record_property
):
    artifact(tmp_path, record_property)
    from types import SimpleNamespace

    from mlx2.runtime.tokenizer_utils import SPMStreamingDetokenizer

    tokenizer = SimpleNamespace(
        vocab={
            "▁Hello": 0,
            "▁world": 1,
            "<0xF0>": 2,
            "<0x9F>": 3,
            "<0x98>": 4,
            "<0x80>": 5,
        }
    )
    stream = SPMStreamingDetokenizer(tokenizer)
    for token in [0, 1, 2, 3, 4, 5]:
        stream.add_token(token)
    stream.finalize()
    assert stream.text == "Hello world😀"
