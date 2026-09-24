"""CPU-only topology, dispatch and recurrent cache checks for Nemotron 3 Super."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.adapters.registry import inspect_model

ARTIFACT = Path(os.environ.get("MLX2_NEMOTRON3_SUPER_ARTIFACT", "/nonexistent"))


def test_metadata_dispatch_reads_no_mlx_or_payload():
    if not ARTIFACT.is_dir():
        pytest.skip("local 5-bit artifact absent")
    script = """
import sys
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability
r = inspect_model(sys.argv[1])
assert r.default_route == 'ordinary'
assert r.adapter_type.__name__ == 'Nemotron3SuperAdapter'
assert r.artifact['mtp_tensor_count'] == 40
assert Capability.MTP in r.descriptor.capabilities
assert 'mlx.core' not in sys.modules
assert resolve_adapter(sys.argv[1], mtp=True) is r.adapter_type
"""
    proc = subprocess.run([sys.executable, "-c", script, str(ARTIFACT)],
                          capture_output=True, text=True, check=False,
                          env={**os.environ, "PYTHONPATH": "src"})
    assert proc.returncode == 0, proc.stderr


def test_wrong_quantization_fails_before_weight_access(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_h", "num_hidden_layers": 88}))
    with pytest.raises(ValueError, match="topology"):
        inspect_model(tmp_path)


def test_unverified_mtp_depth_fails_even_with_shared_override(monkeypatch):
    from mlx2.adapters.nemotron3_super import Nemotron3SuperAdapter

    monkeypatch.setenv("MLX2_MTP_DEPTH_CAP", "6")
    with pytest.raises(ValueError, match="above 3 is not qualified"):
        Nemotron3SuperAdapter("/nonexistent", execution_policy={"num_draft": 4})


def test_tiny_hybrid_prefill_matches_incremental_cpu():
    import mlx.core as mx

    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    mx.set_default_device(mx.cpu)
    if not (ARTIFACT / "config.json").is_file():
        pytest.skip("local config absent")
    config = json.loads((ARTIFACT / "config.json").read_text())
    config.update(vocab_size=64, hidden_size=16, intermediate_size=16,
                  hybrid_override_pattern="ME*", num_hidden_layers=3,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                  mamba_num_heads=2, mamba_head_dim=8, ssm_state_size=8,
                  conv_kernel=3, n_groups=1, n_routed_experts=2,
                  num_experts_per_tok=1, moe_latent_size=8,
                  moe_intermediate_size=16, moe_shared_expert_intermediate_size=16,
                  num_nextn_predict_layers=0)
    model = Model(ModelArgs.from_dict(config))
    tokens = mx.array([[1, 2, 3]])
    full = model(tokens)
    cache = model.make_cache()
    first = model(tokens[:, :2], cache=cache)
    last = model(tokens[:, 2:], cache=cache)
    mx.eval(full, first, last)
    assert float(mx.max(mx.abs(full - mx.concatenate([first, last], axis=1)))) < 1e-4


def test_cache_budget_charges_recurrent_state():
    from types import SimpleNamespace

    from mlx2.adapters.nemotron3_super_memory import NemotronCacheBudget
    config = SimpleNamespace(hybrid_override_pattern="M" * 40 + "E" * 40 + "*" * 8,
                             num_key_value_heads=2, head_dim=128,
                             mamba_num_heads=128, mamba_head_dim=64,
                             ssm_state_size=128, n_groups=8, conv_kernel=4)
    budget = NemotronCacheBudget.from_config(config)
    assert budget.recurrent_live_bytes > 160 * 1024 * 1024
    assert budget.project(1024) > budget.project(0)
    assert budget.project_pool(1024, resident_lanes=2, running_lanes=1) == 2 * budget.project(1024)


def test_quantized_expert_gather_top22_cpu():
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.models.switch_layers import SwitchMLP

    mx.set_default_device(mx.cpu)
    experts = SwitchMLP(64, 64, 32, activation=nn.ReLU2())
    nn.quantize(experts, group_size=64, bits=5, mode="affine")
    values = experts(mx.ones((1, 1, 64)), mx.arange(22).reshape(1, 1, 22))
    mx.eval(values)
    assert values.shape == (1, 1, 22, 64)
    assert experts.fc1.bits == experts.fc2.bits == 5


def test_quantized_layer_names_match_checkpoint_index_cpu():
    if not ARTIFACT.is_dir():
        pytest.skip("local 5-bit artifact absent")
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten

    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    mx.set_default_device(mx.cpu)
    config = json.loads((ARTIFACT / "config.json").read_text())
    config.update(vocab_size=64, hidden_size=64, intermediate_size=64,
                  hybrid_override_pattern="ME*", num_hidden_layers=3,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                  mamba_num_heads=2, mamba_head_dim=32, ssm_state_size=32,
                  n_groups=1, n_routed_experts=4, num_experts_per_tok=2,
                  moe_latent_size=64, moe_intermediate_size=64,
                  moe_shared_expert_intermediate_size=64,
                  num_nextn_predict_layers=0)
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(model, group_size=64, bits=5, mode="affine")
    parameter_names = {key for key, _ in tree_flatten(model.parameters())}
    indexed_names = json.loads((ARTIFACT / "model.safetensors.index.json").read_text())["weight_map"]
    for tiny_layer, actual_layer in ((0, 0), (1, 1), (2, 7)):
        tiny_prefix = f"backbone.layers.{tiny_layer}."
        actual_prefix = f"backbone.layers.{actual_layer}."
        expected = {key.removeprefix(tiny_prefix) for key in parameter_names if key.startswith(tiny_prefix)}
        actual = {key.removeprefix(actual_prefix) for key in indexed_names if key.startswith(actual_prefix)}
        assert expected == actual


def _tiny_mtp_model():
    import mlx.core as mx

    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    mx.set_default_device(mx.cpu)
    config = json.loads((ARTIFACT / "config.json").read_text())
    config.update(vocab_size=64, hidden_size=16, intermediate_size=16,
                  hybrid_override_pattern="ME*", num_hidden_layers=3,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                  mamba_num_heads=2, mamba_head_dim=8, ssm_state_size=8,
                  conv_kernel=3, n_groups=1, n_routed_experts=2,
                  num_experts_per_tok=1, moe_latent_size=8,
                  moe_intermediate_size=16, moe_shared_expert_intermediate_size=16,
                  num_nextn_predict_layers=1)
    return Model(ModelArgs.from_dict(config))


@pytest.mark.parametrize("kept", [0, 1, 2, 3])
def test_mamba_rewind_matches_independent_prefix_cpu(kept):
    import copy

    import mlx.core as mx

    if not ARTIFACT.is_dir():
        pytest.skip("local config absent")
    model = _tiny_mtp_model()
    base = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), base))
    reference = copy.deepcopy(base)
    candidate = copy.deepcopy(base)
    verify_tokens = mx.array([[4, 5, 6]])
    if kept:
        mx.eval(model(verify_tokens[:, :kept], reference))
    for cache in candidate:
        cache.start_speculation()
    mx.eval(model(verify_tokens, candidate))
    for cache in candidate:
        assert cache.trim(3 - kept) == 3 - kept
        cache.stop_speculation()
    for expected, actual in zip(reference[0].cache, candidate[0].cache):
        mx.eval(expected, actual)
        assert float(mx.max(mx.abs(expected - actual))) < 1e-4
    expected = model(mx.array([[7]]), reference)
    actual = model(mx.array([[7]]), candidate)
    mx.eval(expected, actual)
    assert float(mx.max(mx.abs(expected - actual))) < 1e-4


def test_nemotron_mamba_omits_only_all_live_mtp_mask_cpu(monkeypatch):
    import mlx.core as mx

    from mlx2.runtime.models import nemotron_h

    if not ARTIFACT.is_dir():
        pytest.skip("local config absent")
    model = _tiny_mtp_model()
    observed = []
    original = nemotron_h.ssm_update

    def capture(*args, **kwargs):
        observed.append(args[9])
        return original(*args, **kwargs)

    monkeypatch.setattr(nemotron_h, "ssm_update", capture)
    one = [type(cache).merge([cache]) for cache in model.make_cache()]
    for cache in one:
        cache.prepare(lengths=[3], right_padding=[0])
    mx.eval(model(mx.array([[1, 2, 3]]), cache=one))
    assert observed and all(mask is None for mask in observed)

    observed.clear()
    two = [type(pair[0]).merge(pair) for pair in zip(model.make_cache(), model.make_cache())]
    for cache in two:
        cache.prepare(lengths=[3, 2], right_padding=[0, 1])
    mx.eval(model(mx.array([[1, 2, 3], [1, 2, 0]]), cache=two))
    assert observed and all(mask is not None for mask in observed)


@pytest.mark.parametrize("prompt", [[1, 2, 3, 4], [1, 2, 3, 4, 5]])
def test_nemotron_mtp_prefill_matches_ordinary_chunk_boundaries_cpu(prompt):
    import mlx.core as mx

    from mlx2.runtime.hybrid_speculative import prepare_self_mtp_lane

    if not ARTIFACT.is_dir():
        pytest.skip("local config absent")
    model = _tiny_mtp_model()
    reference = model.make_cache()
    for start in range(0, len(prompt), 2):
        logits = model(mx.array([prompt[start:start + 2]]), cache=reference)
        mx.eval(logits)
    lane, first = prepare_self_mtp_lane(
        mx.array(prompt), model, uid=1, max_tokens=8,
        prompt_cache=None, mtp_state=None, lane_rng=None, num_draft=2,
        sampling_temp=0, sampling_top_p=1, sampling_top_k=0,
        sampling_min_p=0, accept_rule="exact", logits_processors=[],
        prefill_step_size=2, share_qsa_indices=False,
    )
    assert first.token == int(mx.argmax(logits[0, -1]).item())
    for expected, actual in zip(reference, lane.caches.target):
        if hasattr(expected, "cache"):
            for a, b in zip(expected.cache, actual.cache):
                mx.eval(a, b)
                assert float(mx.max(mx.abs(a - b))) < 1e-4
        else:
            for a, b in zip(expected.keys_and_values(), actual.keys_and_values()):
                mx.eval(a, b)
                assert float(mx.max(mx.abs(a - b))) < 1e-4


def test_nemotron_single_lane_mtp_matches_ordinary_state_cpu():
    import mlx.core as mx

    from mlx2.runtime.hybrid_speculative import (
        attach_self_mtp_lanes,
        commit_batched_self_mtp,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )

    if not ARTIFACT.is_dir():
        pytest.skip("local config absent")
    model = _tiny_mtp_model()
    prompt = [1, 2, 3, 4]
    reference = model.make_cache()
    for start in range(0, len(prompt), 2):
        logits = model(mx.array([prompt[start:start + 2]]), cache=reference)
        mx.eval(logits)
    current = int(mx.argmax(logits[0, -1]).item())
    detached, first = prepare_self_mtp_lane(
        mx.array(prompt), model, uid=1, max_tokens=16,
        prompt_cache=None, mtp_state=None, lane_rng=None, num_draft=2,
        sampling_temp=0, sampling_top_p=1, sampling_top_k=0,
        sampling_min_p=0, accept_rule="exact", logits_processors=[],
        prefill_step_size=2, share_qsa_indices=False,
    )
    assert first.token == current
    batch = attach_self_mtp_lanes(model, None, [detached])
    for _ in range(3):
        proposal = propose_batched_self_mtp(model, batch)
        for item in proposal.outputs[0]:
            logits = model(mx.array([[current]]), cache=reference)
            mx.eval(logits)
            current = int(mx.argmax(logits[0, -1]).item())
            assert item.token == current
        commit_batched_self_mtp(
            batch, proposal, emitted_counts=[len(proposal.outputs[0])],
            terminal=[False],
        )
        for expected, batched in zip(reference, batch.caches.target):
            actual = batched.extract(0)
            pairs = (zip(expected.cache, actual.cache) if hasattr(expected, "cache")
                     else zip(expected.keys_and_values(), actual.keys_and_values()))
            for a, b in pairs:
                mx.eval(a, b)
                assert float(mx.max(mx.abs(a - b))) < 1e-4


@pytest.mark.parametrize("force_accept", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
def test_self_mtp_batched_cycles_cpu(force_accept, segmented):
    import mlx.core as mx

    from mlx2.runtime.hybrid_speculative import (
        attach_segmented_self_mtp_lanes,
        attach_self_mtp_lanes,
        commit_batched_self_mtp,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )

    if not ARTIFACT.is_dir():
        pytest.skip("local config absent")
    model = _tiny_mtp_model()
    if force_accept:
        model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
    detached = []
    for uid, prompt in ((1, [1, 2, 3]), (2, [2, 3, 4])):
        lane, _ = prepare_self_mtp_lane(
            mx.array(prompt), model, uid=uid, max_tokens=20,
            prompt_cache=None, mtp_state=None, lane_rng=None, num_draft=2,
            sampling_temp=0, sampling_top_p=1, sampling_top_k=0,
            sampling_min_p=0, accept_rule="exact", logits_processors=[],
            prefill_step_size=2, share_qsa_indices=False,
        )
        detached.append(lane)
    assert len(model.make_mtp_cache()) == 1
    attach = attach_segmented_self_mtp_lanes if segmented else attach_self_mtp_lanes
    batch = attach(model, None, detached)
    for _ in range(3):
        proposal = propose_batched_self_mtp(model, batch)
        assert len(proposal.outputs) == 2
        if force_accept:
            assert proposal.accepted_lengths == (2, 2)
        commit_batched_self_mtp(
            batch, proposal,
            emitted_counts=[len(row) for row in proposal.outputs],
            terminal=[False, False],
        )
    assert all(lane.stats.draft_proposed >= 6 for lane in batch.lanes)
