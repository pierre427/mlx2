"""Mechanism assertion: APCv2 hits are nonzero on a hybrid GDN + self-MTP route.

vllm#57616: hybrid GDN + MTP + prefix caching silently recorded zero hits.
This drives the real ServingEngine (mtp=True) with tiny random-weight hybrid
GDN models that carry an MTP head, sends a prompt, then the same prompt with an
extended suffix, and asserts:
  * the second request reused > 0 prompt tokens (receipt + engine counts),
  * the APCv2 lookup counter recorded a hit with cached_tokens > 0,
  * the hit was served from an MTP sidecar (draft state restored, not rebuilt),
  * mtp_sidecar_missing_misses stayed 0,
  * greedy output equals a cold run of the same extended prompt.
CPU only.
"""

import time

import mlx.core as mx
import pytest


from mlx2 import memory, serving
from mlx2.runtime import os_memory
from mlx2.serving import ServingEngine


def tiny_qwen4_mtp():
    from mlx2.runtime.models.qwen4_exp import Model, ModelArgs

    text_config = dict(
        model_type="qwen4_exp_text", hidden_size=32, intermediate_size=0,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=64, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention"],
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, hc_count=2, hc_lowrank=8,
        ple_layer_ids=[1], ple_embed_dim=32, ple_conv_kernel_size=4, ngram_size=3,
        heads_per_ngram=2, ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128, split_ngram_parts=1,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=8, indexer_compress_ratio=2, mtp_num_hidden_layers=1,
        rope_parameters={"type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 0.25},
    )
    mx.random.seed(11)
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    mx.eval(model.parameters())
    return model, 64


def tiny_qwen38_mtp():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=512,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model, 128


class Detok:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = f"{int(token)} "

    def finalize(self):
        pass


class Parser:
    stopped = False
    tool_count = 0

    def push(self, text, final=False):
        return [{"content": text}] if text else []


def make_adapter(model, vocab):
    class Tokenizer:
        vocab_size = vocab
        eos_token_ids = []

        @property
        def detokenizer(self):
            return Detok()

    class Adapter:
        max_context = 512
        identity = {"fingerprint": "tiny-hybrid-mtp"}
        environment = {}
        layout = "tiny-hybrid-mtp-layout"
        tokenizer = Tokenizer()

        def __init__(self, _path):
            self.model = model

        def profile_name(self, mtp):
            return "tiny-mtp" if mtp else "tiny-ordinary"

        def execution_config(self, *, max_lanes, prefill_step):
            return {"persistent": True, "num_draft": 2, "rate_gate": False,
                    "prefill_step_size": prefill_step}

        def prompt_tokens(self, request):
            return list(request["tokens"])

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    return Adapter


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def run(engine, tokens, max_tokens=8):
    job = engine.submit({"tokens": list(tokens), "max_tokens": max_tokens,
                         "temperature": 0})
    text = ""
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], event.get("receipt") or {}, job


def _apcv2(status):
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "apcv2" and isinstance(value, dict):
                    found.update(value)
                walk(value)

    walk(status)
    return found


def _hits(apc):
    lifetime = dict(apc.get("lifetime") or {})
    hits = int(apc.get("hits", 0)) + int(lifetime.get("hits", 0))
    cached = int(apc.get("cached_tokens", 0)) + int(lifetime.get("cached_tokens", 0))
    return hits, cached


def make_engine(model, vocab, **kw):
    engine = ServingEngine("tiny", adapter_factory=make_adapter(model, vocab),
                           qualification_mode=True, mtp=True, max_lanes=1,
                           prefill_step=16, **kw)
    assert engine.ready.wait(60), engine.error
    return engine


@pytest.mark.parametrize("factory", [tiny_qwen4_mtp, tiny_qwen38_mtp],
                         ids=["qwen4_exp_flash_next", "qwen38_hybrid"])
def test_hybrid_gdn_self_mtp_apc_hits_are_nonzero(host, factory):
    model, vocab = factory()
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(96)]
    extended = prompt + [(5 * i + 1) % (vocab - 2) + 1 for i in range(24)]

    warm = make_engine(model, vocab)
    try:
        out1, r1, j1 = run(warm, prompt)
        assert int(j1.cached_tokens or 0) == 0
        out_warm, r2, j2 = run(warm, extended)
        # Multi-turn shape: previous prompt + its generated answer + new turn.
        turn2 = prompt + out1 + [9, 8, 7, 6, 5, 4]
        out_t2, r3, j3 = run(warm, turn2)
        # APC counters reach status through a ~1 s snapshot.
        deadline = time.monotonic() + 10
        while True:
            apc = _apcv2(warm.status())
            if _hits(apc)[0] > 0 or time.monotonic() > deadline:
                break
            time.sleep(0.2)
        counts = dict(warm.counts)
    finally:
        warm.close()

    cold = make_engine(model, vocab)
    try:
        out_cold, _, jc = run(cold, extended)
        assert int(jc.cached_tokens or 0) == 0
    finally:
        cold.close()
    cold = make_engine(model, vocab)
    try:
        out_cold_t2, _, _ = run(cold, turn2)
    finally:
        cold.close()
    # The self-MTP route really ran on the warm-hit request (draft head used).
    assert r2["mtp"]["route"] == "continuous_batched_self_mtp"
    assert r2["mtp"]["stats"]["draft_cycles"] > 0
    assert r2.get("cache_checkpoint_role") == "committed_prompt_boundary"
    assert int(j3.cached_tokens) >= len(prompt) + len(out1) - 1
    assert out_t2 == out_cold_t2

    # Mechanism assertions.
    assert int(j2.cached_tokens) > 0, "second request reused zero tokens"
    assert counts.get("cached_prompt_tokens", 0) > 0
    assert counts.get("mtp_sidecar_missing_misses", 0) == 0
    hits, cached = _hits(apc)
    assert hits > 0 and cached > 0, apc
    # Correctness: warm-hit output equals a cold run.
    assert out_warm == out_cold
