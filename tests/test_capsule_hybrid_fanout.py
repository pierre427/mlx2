"""Cache-capsule fanout must decode exactly what one independent request
decodes (CPU, real ServingEngine over tiny Qwen3.8 models).

The series' experimental round on Qwen3.8-27B saw two of three n=3 samples
end immediately (empty content, finish "stop") whenever the capsule engaged.
The capsule was built from ``apc.lookup(boundary_tokens)``: an exact lookup,
which APCv2 serves one token short (KV trimmed to len-1) or, for GDN hybrid
state that cannot trim, from an older checkpoint.  The siblings had leased
the full committed boundary and prefilled only their one remaining token, so
the capsule that replaced their cache was missing 1..7 prompt tokens.
"""

import mlx.core as mx
import pytest

from mlx2.runtime import apc_v2
from mlx2.server import collect_parallel_samples
from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

CAPSULES = {"enabled": True, "backend": "cpu", "fallback": "cpu"}


def tiny_kv_only(seed=7, vocab=128):
    """The same Qwen3.8 stack with every layer full attention (no GDN)."""
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=vocab, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=1,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=512,
    )
    mx.random.seed(seed)
    model = TextModel(args)
    model.eval()
    return model, vocab


def _model(kind):
    model, vocab = tiny_qwen38_mtp() if kind == "hybrid" else tiny_kv_only()
    # Capsules carry bf16/fp16 KV only; evaluate the cast on this thread so
    # the worker thread never sees a lazy graph bound to another stream.
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model, vocab


def _body(vocab, length=40):
    tokens = mx.random.randint(1, vocab, (length,), key=mx.random.key(3)).tolist()
    return {"tokens": [int(t) for t in tokens], "max_tokens": 8, "temperature": 0}


def _tokens(result):
    return [int(t) for t in result[0]["message"]["content"].split()]


@pytest.mark.parametrize("kind", ["hybrid", "kv"])
def test_capsule_fanout_siblings_decode_the_single_sample_answer(monkeypatch, kind):
    patch_host(monkeypatch)
    model, vocab = _model(kind)
    body = _body(vocab)
    engine = make_engine(model, vocab, mtp=False, max_lanes=4)
    try:
        reference = run(engine, body)
    finally:
        engine.close()
    assert "error" not in reference, reference
    engine = make_engine(model, vocab, mtp=False, max_lanes=4, cache_capsules=CAPSULES)
    try:
        jobs = engine.submit_many([dict(body) for _ in range(3)])
        results = collect_parallel_samples(jobs, body, chat=True)
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert counts.get("cache_capsule_prepared") == 1
    receipts = [result[2]["cache_capsule"] for result in results[1:]]
    assert all(receipt and receipt["status"] == "engaged" for receipt in receipts)
    assert [_tokens(result) for result in results] == [reference["tokens"]] * 3


@pytest.mark.parametrize("kind", ["hybrid", "kv"])
def test_capsule_fanout_under_pressure_spills_stays_exact(monkeypatch, tmp_path, kind):
    """Every APCv2 lookup first spills the oldest unleased entry to disk, as
    admission's pressure eviction did 8 times during the GPU request.  The
    fanout may engage the capsule or fall back, but never decodes wrongly."""
    patch_host(monkeypatch)
    model, vocab = _model(kind)
    body = _body(vocab)
    engine = make_engine(model, vocab, mtp=False, max_lanes=4)
    try:
        reference = run(engine, body)
    finally:
        engine.close()
    original = apc_v2.APCv2.lookup
    spills = []

    def pressured_lookup(self, *args, **kwargs):
        spills.append(self.evict_oldest_unleased())
        return original(self, *args, **kwargs)

    engine = make_engine(
        model, vocab, mtp=False, max_lanes=4, cache_capsules=CAPSULES,
        cache_dir=str(tmp_path), persistent_block_bytes=4096,
    )
    try:
        # Older resident checkpoints for pressure to take first, as on the
        # GPU host where the earlier checks had filled APCv2.
        for seed in (11, 12, 13, 14, 15, 16):
            warm = run(engine, dict(body, tokens=_body(vocab, 40 + seed)["tokens"][seed:], max_tokens=2))
            assert "error" not in warm, warm
        monkeypatch.setattr(apc_v2.APCv2, "lookup", pressured_lookup)
        jobs = engine.submit_many([dict(body) for _ in range(3)])
        results = collect_parallel_samples(jobs, body, chat=True)
    finally:
        engine.close()
    assert any(spills)
    assert [_tokens(result) for result in results] == [reference["tokens"]] * 3
