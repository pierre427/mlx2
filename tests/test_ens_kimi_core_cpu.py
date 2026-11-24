"""ENS Kimi Linear core facade (research/ens/kimi_core.py): synthetic CPU oracles on a tiny random Kimi Linear model,
plus the real Kimi tokenizer when its files are present. Not a checkpoint result."""
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters.kimi_linear import parse_policy
from mlx2.runtime.models import moe_nax_gather
from mlx2.runtime.models.kimi_linear import Model, ModelArgs
from research.ens import kimi_core as KC
from research.ens import physics_bus as PB
from research.ens import physics_qa as QA

TOKENIZER_DIR = Path(KC.KIMI_LINEAR_8BIT)


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    previous_mode = moe_nax_gather.set_mode("off")
    mx.set_default_device(mx.cpu)
    mx.random.seed(118)
    yield
    mx.set_default_device(previous)
    moe_nax_gather.set_mode(previous_mode)


def tiny(seed=0):
    args = ModelArgs(model_type="kimi_linear", vocab_size=128, hidden_size=64,
        intermediate_size=64, moe_intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=0, layer_group_size=2, max_position_embeddings=1024,
        rms_norm_eps=1e-5, rope_theta=10000., routed_scaling_factor=2.446,
        head_dim=64, kv_lora_rank=64, qk_rope_head_dim=32, qk_nope_head_dim=32,
        v_head_dim=32, linear_attn_config={"full_attn_layers": [2, 4], "kda_layers": [1, 3],
            "num_heads": 2, "head_dim": 32, "short_conv_kernel_size": 4})
    mx.random.seed(seed)
    m = Model(args)
    m.eval()
    nn.quantize(m, group_size=64, bits=8, class_predicate=lambda _, mod: hasattr(mod, "to_quantized")
                and mod.weight.shape[-1] % 64 == 0)
    m.configure(parse_policy({"mla": "expanded", "mla_flash": "off", "kda": "reference"}))
    mx.eval(m.parameters())
    return m


def plain_logits(m, tokens):
    h = m.model.word_embeddings(tokens)
    for layer in m.layers:
        h = layer(h, mask=m.model.attention_mask(h, None) if layer.is_global else None, cache=None)
    return m.lm_head(m.model.norm(h))


def core(policy="reference"):
    return KC.KimiCore(kda_policy=policy, model=tiny(), tokenizer=None, checkpoint="synthetic-kimi@cpu")


@pytest.mark.parametrize("policy", ["reference", "chunk_wy"])
def test_bypass_is_the_original_model(policy):
    ref = tiny()
    KC.apply_training_policies(ref, policy)
    c = KC.KimiCore(kda_policy=policy, model=tiny(), tokenizer=None, checkpoint="synthetic-kimi@cpu")
    T = mx.array(np.random.default_rng(0).integers(0, 128, (2, 24)))
    assert mx.array_equal(plain_logits(ref, T), c(T)).item()
    # a zero-initialised port: rows before the first mention are the original model exactly
    port = PB.RecordPort(c, 1, rank=4)
    M = mx.array([[9, -1], [5, 12]])
    pl = {s: PB.unit_rms(mx.random.normal((2, 2, 64))) for s in PB.SLOTS}
    a, b = PB.forward(c, T, None), PB.forward(c, T, port, M, pl)
    assert mx.array_equal(a[0, :9], b[0, :9]).item() and mx.array_equal(a[1, :5], b[1, :5]).item()
    assert not mx.allclose(a[0, 9:], b[0, 9:]).item()


def test_port_dims_cover_kda_mla_and_shared_experts():
    c = core()
    port = PB.RecordPort(c, 1, rank=4)
    assert set(port.lora) == {"b2", "b3"}
    kda, mla = c.lora_dims(2), c.lora_dims(1)                  # 0-based: index 2 is KDA, 1 is MLA
    assert {"q_proj", "k_proj", "v_proj", "o_proj", "se_gate_proj", "se_up_proj", "se_down_proj"} <= set(kda)
    assert {"q_proj", "kv_a_proj_with_mqa", "o_proj"} <= set(mla) and "k_proj" not in mla


def test_gradients_reach_the_port_only():
    c = core("chunk_wy")
    port = PB.RecordPort(c, 1, rank=4)
    rng = np.random.default_rng(1)
    T = mx.array(rng.integers(0, 128, (2, 24)))
    Y = mx.array(rng.integers(0, 128, (2, 24)))
    W = mx.zeros((2, 24)).at[:, 14:].add(1.0)
    M = mx.array([[9, -1], [5, 12]])
    R = {s: mx.array([[3, 0], [4, 5]]) for s in PB.SLOTS}

    def lf(p):
        return PB.answer_loss(PB.forward(c, T, p, M, PB.payload_from_ids(c, R)), W, Y)
    loss, g = nn.value_and_grad(port, lf)(port)
    flat = dict(tree_flatten(g))
    assert np.isfinite(loss.item())
    assert all(np.isfinite(np.array(v)).all() for v in flat.values())
    assert float(mx.abs(flat["proj.field.weight"]).sum()) > 0
    assert float(mx.abs(flat["lora.b3.se_down_proj.b.weight"]).sum()) > 0
    assert float(mx.abs(flat["lora.b2.q_proj.b.weight"]).sum()) > 0
    assert float(mx.abs(flat["lora.b3.kv_a_proj_with_mqa.b.weight"]).sum()) > 0     # MLA layer (0-based 3)
    assert not tree_flatten(c.k.trainable_parameters())
    assert c.k.layers[2].attention.calls["chunk_wy"] > 0                            # chunk-WY engaged (T >= 17)


def test_virtual_ids_are_mean_embeddings():
    c = core()
    c.set_virtual([[3, 7], [9, 10, 11]])
    e = c.k.model.word_embeddings
    got = c.embed(mx.array([[128, 129, 5]]))
    want0 = mx.mean(e(mx.array([3, 7])).astype(mx.float32), 0)
    want1 = mx.mean(e(mx.array([9, 10, 11])).astype(mx.float32), 0)
    assert mx.allclose(got[0, 0].astype(mx.float32), want0, atol=1e-2).item()
    assert mx.allclose(got[0, 1].astype(mx.float32), want1, atol=1e-2).item()
    assert mx.array_equal(got[0, 2], e(mx.array([5]))[0]).item()


@pytest.mark.skipif(not (TOKENIZER_DIR / "tiktoken.model").exists(), reason="Kimi tokenizer files absent")
def test_tokenizer_facade_and_mentions():
    from mlx2.runtime.kimi_tokenizer import TikTokenTokenizer
    tk = TikTokenTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    c = KC.KimiCore(model=tiny(), tokenizer=tk, checkpoint="synthetic-kimi@cpu")
    tok = KC.KimiTok(c)
    assert tok.pad == 163839 and tok.t.token_to_id("<|im_end|>") == 163586
    assert tok.encode("It came out in May 2026.<|im_end|>")[-1] == 163586
    q = KC.kimi_chat("Are 'Quantum Ω-Hall Edges' and 'Gravity?' the same?")
    pos = QA.mention_positions(tok, q, ["Quantum Ω-Hall Edges", "Gravity?"])
    ids = tok.t.encode(q).ids
    assert len(pos) == 2 and pos[0] < pos[1]
    for p in pos:
        assert "'" in tok.t.decode(ids[p:p + 1])
    assert tok.t.decode(ids) == q


def test_qwen_resume_accepts_old_args_without_backend():
    old = {"facts": "f", "model": "Qwen/Qwen3-4B", "seed": 0}
    now = {**old, "backend": "qwen3", "kda_policy": "chunk_wy"}
    assert PB._ARG_DEFAULTS | old | PB._FREE == now | PB._FREE


def test_vocab_virtual_ids_with_a_stub_model():
    class Stub:
        vocab_size = 1000
        got = None

        def set_virtual(self, lists):
            Stub.got = lists

    class Tok:
        def encode(self, s):
            return [1, 2] if s == " semiclassical" else [ord(s[1])]
    facts = {"papers": [{"concept": "semiclassical"}, {"concept": "vortex"}]}
    v = PB.Vocab(Tok(), facts, Stub())
    assert v.ids["concept"][v.values["concept"].index("semiclassical")] == 1000 and Stub.got == [[1, 2]]
    with pytest.raises(AssertionError):
        PB.Vocab(Tok(), facts)
