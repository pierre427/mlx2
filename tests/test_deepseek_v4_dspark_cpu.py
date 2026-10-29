from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.drafters.deepseek_v4_dspark import (
    SIDECAR_SHA256,
    TARGET_SHA256,
    VISION_REVISION,
    DeepSeekV4DSparkConfig,
    DSparkContractError,
    DSparkHistory,
    DSparkStage,
    markov_chain,
    mean_target_hc,
    setup_tokens,
    verify_runtime_binding,
)
from mlx2.runtime.gguf.reader import GGUFTensor
from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Config


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def config():
    return DeepSeekV4DSparkConfig(3, 4, 256, 0, (5, 20, 42))


def index(c):
    meta = {
        "general.architecture": "deepseek4",
        "deepseek4.checkpoint_variant": "vision-exp",
        "general.source.revision": VISION_REVISION,
        "dspark.target_architecture": "deepseek4",
        "dspark.block_size": c.block_size,
        "dspark.markov_rank": c.markov_rank,
        "dspark.noise_token_id": c.noise_token_id,
        "dspark.target_layer_ids": c.target_layers,
    }
    tensors = {
        name: GGUFTensor(name, spec.shape, min(spec.kinds), 0, 1)
        for name, spec in c.tensor_specs().items()
    }
    return meta, tensors


def test_exact_vision_schema_and_shapes():
    c = config()
    metadata, tensors = index(c)
    assert DeepSeekV4DSparkConfig.from_index(metadata, tensors) == c
    assert tensors["mtp.0.main_proj.weight"].shape == (12288, 4096)
    assert tensors["mtp.2.confidence_head.proj.weight"].shape == (4352, 1)


@pytest.mark.parametrize(
    "change",
    [
        {"deepseek4.checkpoint_variant": "0731"},
        {"general.source.revision": "wrong"},
        {"dspark.target_architecture": "deepseek41"},
        {"dspark.target_layer_ids": (5, 5, 42)},
        {"dspark.block_size": 17},
        {"deepseek4.dspark.block_size": 5},
    ],
)
def test_wrong_family_geometry_and_alias_refused(change):
    metadata, tensors = index(config())
    metadata.update(change)
    with pytest.raises(DSparkContractError):
        DeepSeekV4DSparkConfig.from_index(metadata, tensors)


def test_layout_missing_wrong_type_and_orientation_refused():
    c = config()
    _, tensors = index(c)
    key = "mtp.0.attn_q_a.weight"
    wrong = dict(tensors)
    wrong[key] = replace(wrong[key], shape=wrong[key].shape[::-1])
    with pytest.raises(DSparkContractError):
        c.validate_tensors(wrong)
    wrong = dict(tensors)
    del wrong[key]
    with pytest.raises(DSparkContractError):
        c.validate_tensors(wrong)
    wrong = dict(tensors)
    wrong[key] = replace(wrong[key], ggml_type=16)
    with pytest.raises(DSparkContractError):
        c.validate_tensors(wrong)


def test_unverified_sidecar_and_source_refused():
    c = config()
    metadata, _ = index(c)
    metadata["deepseek4.block_count"] = 43
    with pytest.raises(DSparkContractError):
        verify_runtime_binding(
            c,
            target_sha256=TARGET_SHA256,
            sidecar_sha256=None,
            target_metadata=metadata,
            source_snapshot="a" * 64,
        )
    assert SIDECAR_SHA256 in verify_runtime_binding(
        c,
        target_sha256=TARGET_SHA256,
        sidecar_sha256=SIDECAR_SHA256,
        target_metadata=metadata,
        source_snapshot="a" * 64,
    )


def test_noise_positions_and_mean_capture():
    c = config()
    assert setup_tokens(7, 10, c) == ((7, 0, 0, 0), (9, 10, 11, 12, 13))
    assert mean_target_hc(mx.array([[[0.0, 4.0], [2.0, 6.0]]])).tolist() == [[1.0, 5.0]]
    with pytest.raises(DSparkContractError):
        setup_tokens(7, 0, c)


def test_markov_bias_chains_and_confidence_uses_before_norm_hidden():
    w1 = mx.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    w2 = mx.array([[0.0, 2.0], [2.0, 0.0], [0.0, 0.0]])
    hidden = mx.array([[3.0, 4.0], [5.0, 6.0]])
    callbacks = {
        "lookup_w1": lambda token: w1[token],
        "project_w2": lambda state: state @ w2.T,
        "confidence_project": lambda features: (
            features @ mx.array([[1.0], [0.0], [0.0], [1.0]])
        ),
    }
    tokens, confidence = markov_chain(mx.zeros((2, 3)), hidden, 0, **callbacks)
    assert tokens == (1, 0) and confidence == (3.0, 6.0)
    tokens, confidence = markov_chain(
        mx.zeros((2, 3)), hidden, 0, threshold=0.99, **callbacks
    )
    assert tokens == () and confidence == (3.0,)


def test_history_values_restore_and_trim_not_just_metadata():
    h = DSparkHistory("bound", 3, window=3)
    a = mx.arange(8, dtype=mx.float32).reshape(1, 4, 2)
    h.append(a, (a + 10, a + 20, a + 30), 0)
    snapshot = h.snapshot()
    assert snapshot.start_position == 1 and snapshot.next_position == 4
    h.crop(2)
    h.append(a[:, :1] + 99, (a[:, :1] + 109, a[:, :1] + 119, a[:, :1] + 129), 2)
    h.restore(snapshot)
    assert h.main_history.tolist() == a[:, 1:].tolist()
    assert h.stage_kv[0].tolist() == (a + 10)[:, 1:].tolist()
    with pytest.raises(DSparkContractError):
        h.crop(0)
    with pytest.raises(DSparkContractError):
        h.restore(replace(snapshot, identity="foreign"))


class TinyWeights:
    def __init__(self, c):
        self.c = c
        self.data = {}
        rng = np.random.default_rng(99)
        h, d = c.hc_mult, c.hidden_size
        mix = 2 * h + h * h
        shapes = {
            "hc_attn_fn.weight": (mix, h * d),
            "hc_ffn_fn.weight": (mix, h * d),
            "attn_q_a.weight": (c.q_lora_rank, d),
            "attn_q_b.weight": (c.num_attention_heads * c.head_dim, c.q_lora_rank),
            "attn_kv.weight": (c.head_dim, d),
            "attn_output_a.weight": (
                c.o_groups * c.o_lora_rank,
                c.head_dim * c.num_attention_heads // c.o_groups,
            ),
            "attn_output_b.weight": (d, c.o_groups * c.o_lora_rank),
            "ffn_gate_inp.weight": (c.n_routed_experts, d),
            "ffn_gate_shexp.weight": (c.moe_intermediate_size, d),
            "ffn_up_shexp.weight": (c.moe_intermediate_size, d),
            "ffn_down_shexp.weight": (d, c.moe_intermediate_size),
        }
        for name, shape in shapes.items():
            self.data["mtp.0." + name] = mx.array(
                rng.normal(0, 0.1, shape).astype(np.float32)
            )
        for name in ("attn_norm", "ffn_norm"):
            self.data["mtp.0." + name + ".weight"] = mx.ones((d,))
        self.data.update(
            {
                "mtp.0.attn_q_a_norm.weight": mx.ones((c.q_lora_rank,)),
                "mtp.0.attn_kv_a_norm.weight": mx.ones((c.head_dim,)),
                "mtp.0.attn_sinks.weight": mx.full((c.num_attention_heads,), -1.0),
                "mtp.0.exp_probs_b.bias": mx.zeros((c.n_routed_experts,)),
            }
        )
        for name in ("hc_attn", "hc_ffn"):
            self.data["mtp.0." + name + "_base.weight"] = mx.zeros((mix,))
            self.data["mtp.0." + name + "_scale.weight"] = mx.ones((3,))

    def array(self, name):
        return self.data[name]

    def linear(self, name):
        return lambda x: x @ self.data[name].T

    def grouped(self, x, name, groups):
        return mx.einsum(
            "...gi,goi->...go",
            x,
            self.data[name].reshape(groups, self.c.o_lora_rank, -1),
        )

    def moe(self, x, *args):
        return mx.zeros_like(x)


def test_stage_is_noncausal_over_draft_rows():
    c = DeepSeekV4Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        q_lora_rank=4,
        qk_rope_head_dim=2,
        o_groups=1,
        o_lora_rank=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        hc_mult=2,
        num_hash_layers=0,
        compress_ratios=(0,),
        simulate_quantization=False,
    )
    w = TinyWeights(c)
    stage = DSparkStage(c, w, 0)
    h = mx.array(np.random.default_rng(7).normal(size=(1, 2, 2, 8)).astype(np.float32))
    main = mx.ones((1, 1, 8))
    support = mx.zeros((1, 3, 8))
    output = stage(h, main, support, 5)
    changed = mx.concatenate((h[:, :1], h[:, 1:] + 0.7), axis=1)
    other = stage(changed, main, support, 5)
    assert output.shape == h.shape
    assert bool(mx.all(mx.isfinite(output)))
    assert not bool(mx.allclose(output[:, :1], other[:, :1], atol=1e-7))


def test_snapshot_owns_array_objects_against_later_assignment():
    h = DSparkHistory("bound", 3, window=3)
    a = mx.arange(4, dtype=mx.float32).reshape(1, 2, 2)
    h.append(a, (a + 10, a + 20, a + 30), 0)
    saved = h.snapshot()
    h.main_history[0, 0, 0] = -99
    h.stage_kv[0][0, 0, 0] = -88
    assert saved.main_history[0, 0, 0].item() == 0
    assert saved.stage_kv[0][0, 0, 0].item() == 10
    h.restore(saved)
    h.main_history[0, 0, 0] = -77
    assert saved.main_history[0, 0, 0].item() == 0


def test_empty_and_negative_snapshot_metadata_refused():
    h = DSparkHistory("bound", 3)
    state = h.snapshot()
    with pytest.raises(DSparkContractError):
        h.restore(replace(state, start_position=-1, next_position=-1))
    with pytest.raises(DSparkContractError):
        h.restore(replace(state, main_history=mx.zeros((1, 0, 2))))
    with pytest.raises(DSparkContractError):
        h.restore(replace(state, next_position=True))


def test_authoritative_prefill_suffix_can_skip_discarded_projection_rows():
    h = DSparkHistory("bound", 3, window=3)
    last = mx.arange(6, dtype=mx.float32).reshape(1, 3, 2)
    h.append(last, (last + 10, last + 20, last + 30), 0, skipped_prefix=97)
    assert h.start_position == 97 and h.next_position == 100
    assert h.main_history.tolist() == last.tolist()
    h.crop(97)
    h.append(last[:, :1], (last[:, :1] + 10, last[:, :1] + 20, last[:, :1] + 30), 97)
    assert h.start_position == 97 and h.next_position == 98


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_attention_heads", 32),
        ("num_key_value_heads", 2),
        ("simulate_quantization", False),
        ("rope_theta", 160000.0),
        ("swiglu_limit", 20.0),
    ],
)
def test_backbone_target_semantics_fail_closed(field, value):
    target = DeepSeekV4Config(compress_ratios=(0,) * 43)
    config().validate_target(target)
    setattr(target, field, value)
    with pytest.raises(DSparkContractError, match=field):
        config().validate_target(target)


def test_declared_stage_count_cannot_conflict_with_actual_tensors():
    metadata, tensors = index(config())
    metadata["dspark.stage_count"] = 4
    with pytest.raises(DSparkContractError, match="stage count"):
        DeepSeekV4DSparkConfig.from_index(metadata, tensors)


def test_exact_gguf_float32_epsilons_are_accepted_without_changing_math():
    import struct

    target = DeepSeekV4Config(compress_ratios=(0,) * 43)
    target.rms_norm_eps = struct.unpack("<f", struct.pack("<f", 1e-20))[0]
    target.hc_eps = struct.unpack("<f", struct.pack("<f", 1e-6))[0]
    before = (target.rms_norm_eps, target.hc_eps)
    config().validate_target(target)
    assert (target.rms_norm_eps, target.hc_eps) == before


@pytest.mark.parametrize("field,literal", [("rms_norm_eps", 1e-20), ("hc_eps", 1e-6)])
def test_epsilon_gate_has_no_fuzzy_tolerance(field, literal):
    target = DeepSeekV4Config(compress_ratios=(0,) * 43)
    # Both a materially different epsilon and an adjacent storedf32 value
    # must fail; only the exact literal/stored representation is permitted.
    for value in (
        literal * 1.01,
        float(np.nextafter(np.float32(literal), np.float32(np.inf))),
    ):
        setattr(target, field, value)
        with pytest.raises(DSparkContractError, match=field):
            config().validate_target(target)


@pytest.mark.parametrize(
    "threshold,expected", [(0.0, (0, 0, 0)), (0.5, (0,)), (1.0, ())]
)
def test_confidence_prefix_includes_first_and_keeps_equal_boundary(threshold, expected):
    hidden = mx.array([[0.0], [-2.0], [3.0]])
    before = hidden.tolist()
    tokens, confidence = markov_chain(
        mx.zeros((3, 2)),
        hidden,
        0,
        threshold=threshold,
        lookup_w1=lambda token: mx.zeros((1,)),
        project_w2=lambda state: mx.zeros((1, 2)),
        confidence_project=lambda features: features[:, :1],
    )
    assert tokens == expected
    assert len(confidence) == min(len(tokens) + (threshold > 0), 3)
    assert hidden.tolist() == before
