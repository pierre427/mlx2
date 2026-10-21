"""Nemotron-H prompt lookup: row-exact verify through the hybrid transaction.

The rollback machinery existed (``HybridVerifyRows`` record-and-trim), but a
verify block ran the model's block forward: chunked SSM plus multi-row
attention and projections, whose rounding differs from ordinary one-token
decode (CPU fp32 differs by ~4e-7 on the tiny model below), so a near tie
could authorize a token ordinary greedy would not pick.  The model now owns
``pld_verify_forward``: each verify row is the ordinary one-token step (or the
adapter-selected layerwise short-exact recipes of the shared ``row_exact``
package), one lane per forward.  These tests compare logits bit for bit and
greedy output token for token against ordinary decode.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.contracts import Capability  # noqa: E402
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows  # noqa: E402
from mlx2.runtime.models.cache import ArraysCache  # noqa: E402
from mlx2.runtime.pld import PromptLookupBatchGenerator  # noqa: E402
from tests.test_nemotron_external_taps_cpu import tiny_model  # noqa: E402

PROMPTS = [
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3],
    [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8],
    [10, 11, 12, 13, 14, 15, 1, 2, 3],
]
EXPLORE = {
    "num_draft": 4,
    "ngram_min": 2,
    "ngram_max": 3,
    "adaptive": False,
    "deferred_admission": False,
}


def _layerwise(model, policy="layerwise_short_exact"):
    model.set_short_segment_policy(policy)
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.set_conv_policy("fused_short")
            layer.mixer.set_ssm_policy("fused_short")
            layer.mixer.set_projection_policy("short_row_exact")
    return model


@pytest.fixture(scope="module", params=["tokenwise", "layerwise_short_exact"])
def nemotron(request):
    model = tiny_model()
    if request.param == "layerwise_short_exact":
        _layerwise(model)
    mx.eval(model.parameters())
    return request.param, model


def _greedy(model, prompt, tokens):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    produced = []
    for _ in range(tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        produced.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return produced


def _prefilled(model, prompt):
    cache = model.make_cache()
    model(mx.array([prompt]), cache=cache)
    mx.eval([entry.state for entry in cache])
    return cache


def _states(cache):
    values = []
    for entry in cache:
        if isinstance(entry, ArraysCache):
            values.extend(np.asarray(v) for v in entry.cache if v is not None)
        else:
            keys, vals = entry.state
            values.extend(
                (np.asarray(keys)[..., : entry.offset, :], np.asarray(vals)[..., : entry.offset, :])
            )
    return values


def _run(model, prompts, policy, tokens=24):
    generator = PromptLookupBatchGenerator(
        model,
        completion_batch_size=len(prompts),
        prefill_step_size=5,
        prompt_lookup=policy,
    )
    uids = generator.insert(prompts, max_tokens=[tokens] * len(prompts))
    out, finals = {uid: [] for uid in uids}, {}
    for _ in range(4000):
        _prompts, responses = generator.next()
        for response in responses:
            out[response.uid].append(response.token)
            if response.finish_reason:
                finals[response.uid] = response
        if len(finals) == len(uids):
            break
    generator.close()
    return [out[uid] for uid in uids], [finals[uid] for uid in uids], generator


def test_verify_rows_and_committed_state_are_bit_equal_to_ordinary_steps(nemotron):
    path, model = nemotron
    prompt, block, accepted = PROMPTS[0], [3, 4, 5, 6, 7], 3
    verify_cache = _prefilled(model, prompt)
    ordinary_cache = copy.deepcopy(verify_cache)
    owner = HybridVerifyRows([verify_cache])
    transaction = owner.begin([len(block)])
    logits = model.pld_verify_forward(mx.array([block]), cache=transaction.caches)
    mx.eval(logits)
    transaction.commit([accepted])
    ordinary = mx.concatenate(
        [model(mx.array([[token]]), cache=ordinary_cache) for token in block[:accepted]],
        axis=1,
    )
    mx.eval(ordinary)
    status = model.pld_verify_status()
    assert status["max_lanes"] == 1
    assert status[path] > 0
    if path == "tokenwise":
        # Each verify row is the ordinary step: bit-identical on any device.
        assert mx.array_equal(logits[:, :accepted], ordinary).item()
        for actual, expected in zip(_states(verify_cache), _states(ordinary_cache)):
            assert np.array_equal(actual, expected)
        return
    # The selected short-exact Mamba kernels are row-exact on Metal (the
    # native-MTP GPU evidence); their CPU fallbacks round within 1e-6.  The
    # executor's own window must still report no width-dependent fallback.
    assert np.allclose(np.asarray(logits[:, :accepted]), np.asarray(ordinary), atol=1e-6)
    for actual, expected in zip(_states(verify_cache), _states(ordinary_cache)):
        assert np.allclose(actual, expected, atol=1e-6)
    row_exact = model.row_exact_status()
    assert row_exact["windows"] == row_exact["windows_exact"] > 0
    assert not row_exact["failures"]


def test_the_block_forward_is_not_row_exact():
    # Why the model owns the verify: its block forward differs from ordinary.
    model = tiny_model()
    speculative = _prefilled(model, PROMPTS[0])
    ordinary = copy.deepcopy(speculative)
    for entry in speculative:
        if isinstance(entry, ArraysCache):
            entry.start_speculation(16)
    block = mx.array([[3, 4, 5, 6, 7]])
    rows = model(block, cache=speculative)
    steps = mx.concatenate(
        [model(block[:, i : i + 1], cache=ordinary) for i in range(5)], axis=1
    )
    assert not mx.array_equal(rows, steps).item()


def test_multi_lane_verify_rows_are_refused_outside_the_generator():
    model = tiny_model()
    rows = [_prefilled(model, prompt) for prompt in PROMPTS[:2]]
    transaction = HybridVerifyRows(rows).begin([2, 2])
    with pytest.raises(ValueError, match="one lane per forward"):
        model.pld_verify_forward(mx.array([[1, 2], [3, 4]]), cache=transaction.caches)
    transaction.abort()


def test_prompt_lookup_equals_ordinary_greedy_one_lane_per_verify(nemotron):
    _path, model = nemotron
    produced, finals, generator = _run(model, PROMPTS, EXPLORE)
    for prompt, tokens in zip(PROMPTS, produced):
        assert tokens == _greedy(model, prompt, 24)
    stats = generator.scheduler_stats
    assert stats["pld_proposed"] > 0 and stats["pld_rollbacks"] > 0
    assert stats["pld_model_verify_forwards"] > 0
    # pld_verify_max_lanes = 1: three lanes never share a verify forward.
    assert stats["pld_batched_max_width"] == 1
    assert generator.verify_forward == "nemotron_h_row_exact_b1"
    for final in finals:
        assert not any(getattr(c, "speculating", False) for c in final.prompt_cache)


def test_unbatched_per_lane_rounds_replay_through_the_exact_verify(nemotron):
    _path, model = nemotron
    policy = {**EXPLORE, "batched_verify": False}
    produced, _finals, generator = _run(model, PROMPTS[:2], policy)
    for prompt, tokens in zip(PROMPTS, produced):
        assert tokens == _greedy(model, prompt, 24)
    assert generator.scheduler_stats["pld_batched_rounds"] == 0
    assert generator.scheduler_stats["pld_rollbacks"] > 0


def test_default_latched_policy_equals_ordinary_greedy(nemotron):
    from mlx2.server import DEFAULT_PROMPT_LOOKUP_POLICY

    _path, model = nemotron
    proposed = 0
    for head in PROMPTS:
        prompt = head + _greedy(model, head, 30) + head
        produced, _finals, generator = _run(
            model, [prompt], dict(DEFAULT_PROMPT_LOOKUP_POLICY), tokens=40
        )
        assert produced[0] == _greedy(model, prompt, 40)
        proposed += generator.scheduler_stats["pld_proposed"]
    assert proposed > 0


def _adapters():
    from mlx2.adapters.nemotron35_bolt import DESCRIPTOR as BOLT, Nemotron35BoltAdapter
    from mlx2.adapters.nemotron35_lightning import (
        Nemotron35LightningAdapter,
        descriptor_for,
    )
    from mlx2.adapters.nemotron35_super import DESCRIPTOR as SUPER, Nemotron35SuperAdapter
    from mlx2.adapters.nemotron35_super_vl import (
        DESCRIPTOR as SUPER_VL,
        Nemotron35SuperVLAdapter,
    )
    from mlx2.adapters.nemotron3_super import DESCRIPTOR as SUPER3, Nemotron3SuperAdapter

    return {
        "lightning": (Nemotron35LightningAdapter, descriptor_for(has_mtp=False)),
        "lightning_mtp": (Nemotron35LightningAdapter, descriptor_for(has_mtp=True)),
        "nemotron3_super": (Nemotron3SuperAdapter, SUPER3),
        "super_vl_text": (Nemotron35SuperVLAdapter, SUPER_VL),
        "bolt": (Nemotron35BoltAdapter, BOLT),
        "super": (Nemotron35SuperAdapter, SUPER),
    }


@pytest.mark.parametrize(
    ("name", "no_flag_route"),
    [
        # Tokenwise verify measured below ordinary (pld-families-20261021):
        # PLD is selectable but the no-flag route stays ordinary.
        ("lightning", "ordinary"),
        ("lightning_mtp", "ordinary"),
        ("nemotron3_super", "ordinary"),
        ("super_vl_text", "ordinary"),
        # Its PLD route selects the exact short kernels (1.9-2.8x on copy).
        ("bolt", "prompt_lookup"),
        # Its built-in native MTP head stays the default (Pierre 2026-10-18).
        ("super", "native_mtp"),
    ],
)
def test_nemotron_route_selection(name, no_flag_route):
    from mlx2.adapters.registry import AdapterResolution
    from mlx2.server import build_parser, resolve_route_selection

    adapter_type, descriptor = _adapters()[name]
    assert Capability.PROMPT_LOOKUP in descriptor.capabilities
    resolution = AdapterResolution(adapter_type, descriptor, {"fingerprint": "fixture"})
    parser = build_parser()
    selected = resolve_route_selection(
        parser.parse_args(["--model", "fixture"]), None, resolution
    )
    assert selected.route == no_flag_route
    if no_flag_route == "prompt_lookup":
        assert selected.source == "capability_default"
    else:
        assert selected.source == "adapter_default"
    for flag, route in (("--ordinary", "ordinary"), ("--prompt-lookup", "prompt_lookup")):
        selected = resolve_route_selection(
            parser.parse_args(["--model", "fixture", flag]), None, resolution
        )
        assert (selected.route, selected.source) == (route, "explicit_flag")


ROW_EXACT_KERNELS = {
    "mamba_conv": "fused_short",
    "mamba_projection": "short_row_exact",
    "mamba_segment": "layerwise_short_exact_routed_batch",
    "mamba_ssm": "fused_short_gate",
}


@pytest.mark.parametrize("name", ["bolt", "super"])
def test_prompt_lookup_defaults_select_the_exact_short_kernels(name):
    # pld-families-20261021: with these kernels the verify is layerwise
    # short-exact (1.9-2.6x on copy prompts, token-identical to ordinary);
    # without them it is tokenwise.  Only the prompt-lookup route gets them.
    from mlx2.adapters.nemotron35_super import _split_super_execution_policy
    from mlx2.adapters.registry import AdapterResolution
    from mlx2.server import (
        DEFAULT_PROMPT_LOOKUP_POLICY,
        RouteSelection,
        resolve_execution_policy_defaults,
    )

    adapter_type, descriptor = _adapters()[name]
    resolution = AdapterResolution(adapter_type, descriptor, {"fingerprint": "fixture"})
    policy = resolve_execution_policy_defaults(
        None, RouteSelection("prompt_lookup", "capability_default"), resolution
    )
    assert policy == {**ROW_EXACT_KERNELS, "prompt_lookup": DEFAULT_PROMPT_LOOKUP_POLICY}
    adapter_policy = {k: v for k, v in policy.items() if k != "prompt_lookup"}
    split = _split_super_execution_policy(adapter_policy)
    assert split[1:6] == (
        "fused_short", "auto", "fused_short_gate", "short_row_exact",
        "layerwise_short_exact_routed_batch",
    )
    assert split[-1] == {}
    ordinary = resolve_execution_policy_defaults(
        None, RouteSelection("ordinary", "explicit_flag"), resolution
    ) or {}
    assert not set(ordinary) & set(ROW_EXACT_KERNELS)
    # An operator value wins.
    explicit = resolve_execution_policy_defaults(
        {"mamba_segment": "auto"},
        RouteSelection("prompt_lookup", "explicit_flag"),
        resolution,
    )
    assert explicit["mamba_segment"] == "auto"


def test_lightning_parses_and_applies_the_short_exact_kernels(monkeypatch):
    from mlx2.adapters import nemotron35_lightning as lightning
    from mlx2.adapters.nemotron3_super import apply_short_exact_policy

    class _Stop(Exception):
        pass

    def stop(_path):
        raise _Stop

    monkeypatch.setattr(lightning, "inspect_artifact", stop)
    adapter = object.__new__(lightning.Nemotron35LightningAdapter)
    with pytest.raises(_Stop):
        adapter._init_lightning(
            "/unused",
            execution_policy={**ROW_EXACT_KERNELS, "mamba_segment": "layerwise_short_exact"},
        )
    assert adapter.short_exact_policy["mamba_segment"] == "layerwise_short_exact"
    with pytest.raises(ValueError, match="supports num_draft"):
        adapter._init_lightning("/unused", execution_policy={"expert_gather_sort": "auto"})
    with pytest.raises(ValueError, match="mamba_ssm must be one of"):
        adapter._init_lightning("/unused", execution_policy={"mamba_ssm": "fast"})

    model = tiny_model()
    suffix = apply_short_exact_policy(
        model, {**ROW_EXACT_KERNELS, "mamba_segment": "layerwise_short_exact"}
    )
    assert suffix.endswith(":mamba-layerwise-short-exact-v1")
    assert model.short_segment_status()["policy"] == "layerwise_short_exact"
    assert all(
        layer.mixer._ssm_policy == "fused_short_gate"
        for layer in model.layers
        if layer.block_type == "M"
    )
    assert apply_short_exact_policy(tiny_model(), {}) == ""
