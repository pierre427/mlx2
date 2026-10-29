"""CPU contracts for the 2026-10-28 Qwen 27B / Flash Next / Sage repairs."""

from types import SimpleNamespace

import mlx.core as mx
import pytest


# 1. Packed-expert quantization parameters follow the rename (omlx #4897).

def _packed(prefix, experts=2, inter=4, hidden=8, groups=2):
    return {
        f"{prefix}.experts.gate_up_proj": mx.zeros((experts, 2 * inter, hidden)),
        f"{prefix}.experts.gate_up_proj.scales": mx.ones((experts, 2 * inter, groups)),
        f"{prefix}.experts.gate_up_proj.biases": mx.ones((experts, 2 * inter, groups)),
        f"{prefix}.experts.down_proj": mx.zeros((experts, hidden, inter)),
        f"{prefix}.experts.down_proj.scales": mx.ones((experts, hidden, 1)),
        f"{prefix}.experts.down_proj.biases": mx.ones((experts, hidden, 1)),
    }


def test_packed_expert_quantization_params_split_with_the_weight():
    from mlx2.runtime.models.qwen3_next import rename_packed_experts

    prefix = "language_model.model.layers.0.mlp"
    weights = _packed(prefix)
    rename_packed_experts(weights, [prefix], fuse_gate_up=False)
    assert not [key for key in weights if ".experts." in key]
    for projection in ("gate_proj", "up_proj"):
        for param in ("weight", "scales", "biases"):
            assert weights[f"{prefix}.switch_mlp.{projection}.{param}"].shape[-2] == 4
    for param in ("weight", "scales", "biases"):
        assert f"{prefix}.switch_mlp.down_proj.{param}" in weights


def test_packed_expert_quantization_params_stay_fused_when_gate_up_is_fused():
    from mlx2.runtime.models.qwen3_next import rename_packed_experts

    prefix = "mtp.layers.0.mlp"
    weights = _packed(prefix)
    rename_packed_experts(weights, [prefix], fuse_gate_up=True)
    assert set(weights) == {
        f"{prefix}.switch_mlp.{projection}.{param}"
        for projection in ("gate_up_proj", "down_proj")
        for param in ("weight", "scales", "biases")
    }
    assert weights[f"{prefix}.switch_mlp.gate_up_proj.scales"].shape == (2, 8, 2)


def test_unpacked_prefix_is_left_alone():
    from mlx2.runtime.models.qwen3_next import rename_packed_experts

    weights = {"a.mlp.switch_mlp.gate_proj.weight": mx.zeros((1,))}
    rename_packed_experts(weights, ["a.mlp"], fuse_gate_up=False)
    assert list(weights) == ["a.mlp.switch_mlp.gate_proj.weight"]


# 2. An MTP head deeper than one layer is refused (vllm #62111).

@pytest.mark.parametrize(
    "module, name",
    [
        ("mlx2.runtime.models.qwen38_27b", "MTPModule"),
        ("mlx2.runtime.models.qwen36_35b", "MTPModule"),
        ("mlx2.runtime.models.qwen4_exp", "Qwen4ExpMTP"),
    ],
)
def test_multi_layer_mtp_head_fails_closed(module, name):
    from importlib import import_module

    cls = getattr(import_module(module), name)
    with pytest.raises(ValueError, match="mtp_num_hidden_layers=2"):
        cls(SimpleNamespace(mtp_num_hidden_layers=2))


# 3. Adaptive tile rows decline a tree law that is not size-invariant.

def test_adaptive_rows_pass_through_under_an_invariant_law():
    from mlx2.adapters.qwen38_27b import _bind_adaptive_rows

    kwargs = {}
    _bind_adaptive_rows(kwargs, None, invariant=True)
    assert kwargs == {}
    _bind_adaptive_rows(kwargs, True, invariant=True)
    assert kwargs == {"ast_adaptive_rows": True}


def test_adaptive_rows_default_off_under_a_size_variant_law():
    from mlx2.adapters.qwen38_27b import _bind_adaptive_rows

    kwargs = {}
    _bind_adaptive_rows(kwargs, None, invariant=False)
    assert kwargs == {"ast_adaptive_rows": False}
    kwargs = {}
    _bind_adaptive_rows(kwargs, False, invariant=False)
    assert kwargs == {"ast_adaptive_rows": False}


@pytest.mark.parametrize("policy, caller", [(True, None), (None, True)])
def test_explicit_adaptive_rows_refused_under_a_size_variant_law(policy, caller):
    from mlx2.adapters.qwen38_27b import _bind_adaptive_rows

    kwargs = {} if caller is None else {"ast_adaptive_rows": caller}
    with pytest.raises(ValueError, match="tree-size-invariant"):
        _bind_adaptive_rows(kwargs, policy, invariant=False)


# 4. Tiled prefill admission follows the device, not only the flag.

def test_tiled_prefill_declining_device_admits_nothing(monkeypatch):
    from mlx2.runtime.lane import tiled_prefill

    monkeypatch.setenv(tiled_prefill.ENV, "1")
    monkeypatch.setattr(tiled_prefill, "device_admits", lambda: False)
    leaf = SimpleNamespace()
    assert tiled_prefill.admit_model(SimpleNamespace(named_modules=lambda: [("", leaf)])) is False
    assert leaf._direct_prefill_candidate is False
    status = tiled_prefill.status()
    assert status["selected"] is True and status["device_admitted"] is False


def test_tiled_prefill_admits_on_a_supporting_device(monkeypatch):
    from mlx2.runtime.lane import tiled_prefill

    monkeypatch.setenv(tiled_prefill.ENV, "1")
    monkeypatch.setattr(tiled_prefill, "device_admits", lambda: True)
    leaf = SimpleNamespace()
    assert tiled_prefill.admit_model(SimpleNamespace(named_modules=lambda: [("", leaf)]))
    assert leaf._direct_prefill_candidate is True


# 5. Qwen structural markers are bound to their dedicated tokens.

_SPECIAL = {"<tool_call>": 248058, "</tool_call>": 248059, "<think>": 248068, "</think>": 248069}
_TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [_SPECIAL[text]] if text in _SPECIAL else [1, 2]


def _parser(**kwargs):
    from mlx2.output import TokenBoundOutputParser
    from mlx2.runtime.tool_parsers.qwen3_coder import parse_tool_call

    return TokenBoundOutputParser(
        tokenizer=_Tokenizer(), chat=True, tools=_TOOLS, parse_tool=parse_tool_call, **kwargs
    )


def _feed(parser, segments):
    events = []
    for index, (text, token) in enumerate(segments):
        parser.record_output_token(text, token)
        events += parser.push(text, final=index == len(segments) - 1)
    return events


_CALL = [("\n<function=f>\n</function>\n", 7), ("</tool_call>", 248059), ("", None)]


def _tool_events(events):
    return [event for event in events if "tool_call" in event or "tool_calls" in event]


def test_dedicated_tool_token_opens_a_call():
    parser = _parser()
    events = _feed(parser, [("Sure", 5), (" <tool_call>", 248058), *_CALL])
    assert parser.tool_count == 1, events


def test_spelled_tool_marker_stays_content():
    parser = _parser()
    spelled = [("Use", 5), (" <", 6), ("tool", 7), ("_call", 8), (">", 9)]
    events = _feed(parser, [*spelled, *_CALL])
    assert parser.tool_count == 0
    content = "".join(event.get("content", "") for event in events)
    assert "<tool_call>" in content


def test_spelled_think_close_does_not_end_reasoning():
    parser = _parser(thinking=True)
    _feed(parser, [("a", 5), ("</", 6), ("think", 7), (">", 8), ("b", 9)])
    assert parser.channel == "reasoning_content"
    parser = _parser(thinking=True)
    _feed(parser, [("a", 5), ("</think>", 248069), ("b", 9)])
    assert parser.channel == "content"


def test_text_only_input_and_constrained_choice_keep_text_matching():
    parser = _parser()
    parser.push("<tool_call>\n<function=f>\n</function>\n</tool_call>", final=True)
    assert parser.tool_count == 1
    constrained = _parser(constrained_tools=True)
    assert constrained._special_ids == {}


def test_misshaped_special_segment_falls_back_to_text_matching():
    parser = _parser()
    _feed(parser, [("<tool_call> x", 248058), *_CALL])
    assert parser._token_aware is False
    assert parser.tool_count == 1
