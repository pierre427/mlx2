"""Prompt-chunk tail skip (MLX2_PREFILL_TAIL_SKIP, TensorFold#2432 concept), CPU."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx import nn

from mlx2.runtime import prefill_tail_skip as pts
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.nemotron_h import Model, ModelArgs
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows


def _tagged(*indices):
    out = []
    for index in indices:
        cache = KVCache()
        if index is not None:
            cache.apcv2_layer_index = index
        out.append(cache)
    return out


def test_last_state_writer_proves_or_declines():
    assert pts.last_state_writer(_tagged(0, 2, 3), 5) == 3
    assert pts.last_state_writer(_tagged(0, 2, 4), 5) is None  # last layer writes
    assert pts.last_state_writer(_tagged(0, None, 3), 5) is None  # missing tag
    assert pts.last_state_writer(_tagged(0, 3, 2), 5) is None  # unordered
    assert pts.last_state_writer(_tagged(0, 0), 5) is None  # duplicate
    assert pts.last_state_writer(_tagged(0, 7), 5) is None  # out of range
    assert pts.last_state_writer([], 5) is None
    assert pts.last_state_writer([None], 5) is None


def test_mode_parsing(monkeypatch):
    monkeypatch.delenv(pts.ENV, raising=False)
    assert pts.tail_skip_mode() == "0"
    for value, want in (("1", "1"), ("head", "head"), ("HEAD", "head"), ("yes", "0")):
        monkeypatch.setenv(pts.ENV, value)
        assert pts.tail_skip_mode() == want


def _nemotron(pattern):
    mx.random.seed(28)
    args = ModelArgs(
        model_type="nemotron_h", vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=len(pattern), max_position_embeddings=128,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, attention_bias=False,
        mamba_num_heads=2, mamba_head_dim=4, mamba_proj_bias=False, ssm_state_size=4,
        conv_kernel=3, n_groups=1, mlp_bias=False, layer_norm_epsilon=1e-5,
        use_bias=False, use_conv_bias=True, hybrid_override_pattern=list(pattern),
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _state(cache):
    out = []
    for entry in cache:
        if isinstance(entry, ArraysCache):
            out.extend(np.array(v) for v in entry.cache if v is not None)
        else:
            out.append(np.array(entry.keys[..., : entry.offset, :]))
            out.append(np.array(entry.values[..., : entry.offset, :]))
    return out


def _chunks(model, chunks, cache, lengths=None):
    outs = []
    for chunk in chunks:
        tokens = mx.array(chunk, dtype=mx.int32)
        with prefill_checkpoint_rows(lengths or [tokens.shape[1]] * tokens.shape[0],
                                     width=tokens.shape[1]) as request:
            logits = model(tokens, cache=cache)
            mx.eval(logits)
        outs.append((logits, request.applied))
    return outs


NEMO_TAIL = ("M", "*", "M", "-", "-")  # last writer = 2, two stateless blocks


def test_nemotron_tier1_keeps_state_bitwise_and_checkpoint_row(monkeypatch):
    rng = np.random.default_rng(0)
    chunks = [rng.integers(0, 16, (1, 9)).tolist(), rng.integers(0, 16, (1, 6)).tolist()]
    with mx.stream(mx.cpu):
        model = _nemotron(NEMO_TAIL)
        monkeypatch.setenv(pts.ENV, "0")
        full_cache = model.make_cache()
        full = _chunks(model, chunks, full_cache)
        monkeypatch.setenv(pts.ENV, "1")
        before = pts.STATS["tier1_engaged"]
        tail_cache = model.make_cache()
        tail = _chunks(model, chunks, tail_cache)
        assert pts.STATS["tier1_engaged"] - before == 2
        for (want, _), (got, applied) in zip(full, tail):
            assert applied == 1 and got.shape == want.shape == (1, 1, 16)
            np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-5)
        for a, b in zip(_state(full_cache), _state(tail_cache)):
            np.testing.assert_array_equal(a, b)
        step = mx.array([[5]], dtype=mx.int32)
        np.testing.assert_array_equal(np.array(model(step, full_cache)),
                                      np.array(model(step, tail_cache)))


def test_nemotron_tier1_uses_template_tags_when_live_cache_lost_them(monkeypatch):
    with mx.stream(mx.cpu):
        model = _nemotron(NEMO_TAIL)
        monkeypatch.setenv(pts.ENV, "1")
        cache = model.make_cache()
        for entry in cache:  # batch merges drop the KV tag
            if hasattr(entry, "apcv2_layer_index") and isinstance(entry, KVCache):
                del entry.apcv2_layer_index
        before = pts.STATS["tier1_engaged"]
        (logits, applied), = _chunks(model, [[[1, 2, 3, 4]]], cache)
        assert applied == 1 and logits.shape == (1, 1, 16)
        assert pts.STATS["tier1_engaged"] - before == 1


def test_nemotron_declines_without_stateless_tail(monkeypatch):
    with mx.stream(mx.cpu):
        model = _nemotron(("-", "M", "*", "-", "M"))  # last block writes state
        monkeypatch.setenv(pts.ENV, "1")
        before = dict(pts.STATS)
        (logits, applied), = _chunks(model, [[[1, 2, 3, 4]]], model.make_cache())
        # The ordinary head trim still narrows to one row; tier 1 did not run.
        assert applied == 1 and logits.shape == (1, 1, 16)
        assert pts.STATS["tier1_engaged"] == before.get("tier1_engaged", 0)
        assert (pts.STATS["declined_no_stateless_tail"]
                == before.get("declined_no_stateless_tail", 0) + 1)


def test_nemotron_declines_on_cache_layout_mismatch(monkeypatch):
    with mx.stream(mx.cpu):
        model = _nemotron(NEMO_TAIL)
        monkeypatch.setenv(pts.ENV, "1")
        tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        before = pts.STATS["declined_cache_layout"]
        with prefill_checkpoint_rows([3], width=3) as request:
            assert pts.claim_stateless_tail(tokens, [KVCache()], model, 5) is None
        assert request.applied == 0
        assert pts.STATS["declined_cache_layout"] == before + 1


def test_refuse_reason_is_counted_and_not_claimed(monkeypatch):
    with mx.stream(mx.cpu):
        model = _nemotron(NEMO_TAIL)
        monkeypatch.setenv(pts.ENV, "1")
        tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        with prefill_checkpoint_rows([3], width=3) as request:
            assert pts.claim_stateless_tail(tokens, model.make_cache(), model, 5,
                                            refuse="row_exact") is None
            assert not pts.claim_adapter_tail(tokens, kind="tail", refuse="reader")
        assert request.applied == 0
        assert pts.STATS["declined_row_exact"] >= 1 and pts.STATS["declined_reader"] >= 1


def test_nemotron_not_claimed_outside_request_or_with_inner_checkpoint(monkeypatch):
    with mx.stream(mx.cpu):
        model = _nemotron(NEMO_TAIL)
        monkeypatch.setenv(pts.ENV, "1")
        before = pts.STATS["tier1_engaged"]
        # No request (decode / verify / drafting forwards): full width.
        logits = model(mx.array([[1, 2, 3, 4]], dtype=mx.int32), cache=model.make_cache())
        assert logits.shape == (1, 4, 16)
        # A lane whose checkpoint row is not the chunk's last: no claim.
        _chunks(model, [[[1, 2, 3, 4]]], model.make_cache(), lengths=[2])
        assert pts.STATS["tier1_engaged"] == before


def test_nemotron_readers_never_engage(monkeypatch):
    """Capture taps and the MTP backbone read every row: full chunk."""
    with mx.stream(mx.cpu):
        model = _nemotron(NEMO_TAIL)
        monkeypatch.setenv(pts.ENV, "1")
        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        before = pts.STATS["tier1_engaged"]
        with prefill_checkpoint_rows([4], width=4):
            hidden, _ = model.mtp_backbone(tokens, model.make_cache())
            assert hidden.shape[1] == 4
        with prefill_checkpoint_rows([4], width=4):
            _, taps = model.forward_with_taps(tokens, model.make_cache(), (1, 3))
            assert taps.shape[1] == 4  # taps concatenate on the last axis
        assert pts.STATS["tier1_engaged"] == before


def test_nemotron_default_off(monkeypatch):
    with mx.stream(mx.cpu):
        model = _nemotron(NEMO_TAIL)
        monkeypatch.delenv(pts.ENV, raising=False)
        before = pts.STATS["tier1_engaged"]
        _chunks(model, [[[1, 2, 3, 4]]], model.make_cache())
        assert pts.STATS["tier1_engaged"] == before


# Gemma 4 (head trim + tier-2 last-layer split) through the adapter wrapper.

GEMMA_LAYERS = ["sliding_attention", "sliding_attention", "full_attention"] * 2
GEMMA_TEXT = dict(
    hidden_size=32, num_hidden_layers=len(GEMMA_LAYERS), intermediate_size=64,
    num_attention_heads=2, head_dim=8, global_head_dim=16, vocab_size=64,
    vocab_size_per_layer_input=64, num_key_value_heads=1, num_global_key_value_heads=1,
    attention_k_eq_v=True, num_kv_shared_layers=0, hidden_size_per_layer_input=0,
    sliding_window=8, layer_types=GEMMA_LAYERS, use_double_wide_mlp=False,
    max_position_embeddings=4096,
)
GEMMA_SPARSE = dict(enable_moe_block=True, num_experts=4, top_k_experts=2,
                    moe_intermediate_size=16)


def _gemma(extra):
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.gemma4.config import TextConfig
    from mlx_vlm.models.gemma4.language import LanguageModel

    from mlx2.adapters.gemma4 import _Gemma4LogitsModel

    class TinyGemma4(nn.Module):
        def __init__(self):
            super().__init__()
            config = TextConfig(**{**GEMMA_TEXT, **extra})
            mx.random.seed(2)
            self.language_model = LanguageModel(config)
            self.config = SimpleNamespace(text_config=config)
            mx.eval(self.parameters())

        def __call__(self, inputs, cache=None, **kwargs):
            return self.language_model(inputs, cache=cache, **kwargs)

    return _Gemma4LogitsModel(TinyGemma4())


@pytest.mark.parametrize("extra", [{}, GEMMA_SPARSE], ids=["dense", "sparse_moe"])
@pytest.mark.parametrize("mode", ["head", "1"])
def test_gemma4_tail_keeps_cache_bitwise_and_checkpoint_row(monkeypatch, extra, mode):
    rng = np.random.default_rng(3)
    # Second chunk crosses the sliding window so rings wrap.
    chunks = [rng.integers(0, 64, (1, 11)).tolist(), rng.integers(0, 64, (1, 7)).tolist()]
    with mx.stream(mx.cpu):
        model = _gemma(extra)
        monkeypatch.setenv(pts.ENV, "0")
        full_cache = model.make_cache()
        full = _chunks(model, chunks, full_cache)
        monkeypatch.setenv(pts.ENV, mode)
        before = dict(pts.STATS)
        tail_cache = model.make_cache()
        tail = _chunks(model, chunks, tail_cache)
        kind = "tail_engaged" if mode == "1" else "head_engaged"
        assert pts.STATS[kind] - before.get(kind, 0) == 2
        split = pts.STATS["gemma4_tail_layer_split"] - before.get("gemma4_tail_layer_split", 0)
        assert split == (2 if mode == "1" else 0)
        for (want, want_applied), (got, applied) in zip(full, tail):
            assert want_applied == 0 and want.shape == (1, len(want[0]), 64)
            assert applied == 1 and got.shape == (1, 1, 64)
            np.testing.assert_allclose(np.array(got[:, -1]), np.array(want[:, -1]),
                                       rtol=1e-4, atol=1e-4)
        for a, b in zip(full_cache, tail_cache):
            np.testing.assert_array_equal(np.array(a.state[0]), np.array(b.state[0]))
            np.testing.assert_array_equal(np.array(a.state[1]), np.array(b.state[1]))
        # The module tree is never mutated by the claimed forward.
        layers = model._model.language_model.model.layers
        assert type(layers[-1]).__name__ == "DecoderLayer"
        step = mx.array([[5]], dtype=mx.int32)
        np.testing.assert_array_equal(np.array(model(step, cache=full_cache)),
                                      np.array(model(step, cache=tail_cache)))


def test_gemma4_kwargs_and_no_request_stay_full(monkeypatch):
    with mx.stream(mx.cpu):
        model = _gemma({})
        monkeypatch.setenv(pts.ENV, "1")
        tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        assert model(tokens, cache=model.make_cache()).shape == (1, 5, 64)
        with prefill_checkpoint_rows([5], width=5) as request:
            out = model(tokens, cache=model.make_cache(), return_hidden=True)
        assert request.applied == 0 and out.shape == (1, 5, 64)


def test_gemma4_positional_second_argument_is_never_claimed(monkeypatch):
    with mx.stream(mx.cpu):
        model = _gemma({})
        monkeypatch.setenv(pts.ENV, "1")
        tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        with prefill_checkpoint_rows([5], width=5) as request:
            with pytest.raises(TypeError):
                model(tokens, model.make_cache(), cache=None)
        assert request.applied == 0


def test_nemotron_wrapped_backbone_is_a_reader(monkeypatch):
    """Calibration swaps in an observing backbone that reads every row."""
    from mlx2.conversion.calibration import _ObservedBackbone

    with mx.stream(mx.cpu):
        model = _nemotron(NEMO_TAIL)
        monkeypatch.setenv(pts.ENV, "1")
        sink = {}
        original = model.backbone
        model.backbone = _ObservedBackbone(original, sink)
        try:
            before = pts.STATS["declined_backbone_wrapped"]
            tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
            with prefill_checkpoint_rows([4], width=4) as request:
                out = model._checkpoint_logits(tokens, model.make_cache())
            assert request.applied == 1 and out.shape == (1, 1, 16)
            assert sink["hidden"].shape[:2] == (1, 4)
            assert pts.STATS["declined_backbone_wrapped"] == before + 1
        finally:
            model.backbone = original


@pytest.mark.parametrize("extra", [{}, GEMMA_SPARSE], ids=["dense", "sparse_moe"])
@pytest.mark.parametrize("mode", ["head", "1"])
def test_gemma4_tail_left_padded_batch(monkeypatch, extra, mode):
    """B=2, left padding [0, 3] on batch KV and rotating caches, two chunks."""
    from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache

    def batch_cache(model):
        text = model._model.config.text_config
        return [BatchKVCache([0, 3]) if kind == "full_attention"
                else BatchRotatingKVCache(text.sliding_window, [0, 3])
                for kind in text.layer_types]

    rng = np.random.default_rng(5)
    chunks = [rng.integers(1, 64, (2, 6)).tolist(), rng.integers(1, 64, (2, 9)).tolist()]
    with mx.stream(mx.cpu):
        model = _gemma(extra)
        monkeypatch.setenv(pts.ENV, "0")
        full_cache = batch_cache(model)
        full = _chunks(model, chunks, full_cache)
        monkeypatch.setenv(pts.ENV, mode)
        tail_cache = batch_cache(model)
        tail = _chunks(model, chunks, tail_cache)
        for (want, _), (got, applied) in zip(full, tail):
            assert applied == 1 and got.shape == (2, 1, 64)
            np.testing.assert_allclose(np.array(got[:, -1]), np.array(want[:, -1]),
                                       rtol=1e-4, atol=1e-4)
        for a, b in zip(full_cache, tail_cache):
            for x, y in zip(a.state[:2], b.state[:2]):
                np.testing.assert_array_equal(np.array(x), np.array(y))
        step = mx.array([[5], [7]], dtype=mx.int32)
        np.testing.assert_array_equal(np.array(model(step, cache=full_cache)),
                                      np.array(model(step, cache=tail_cache)))


def test_family_default_gemma4_on_others_off(monkeypatch):
    monkeypatch.delenv(pts.ENV, raising=False)
    assert pts.tail_skip_mode() == "0"
    assert pts.tail_skip_mode("gemma4") == "1"
    assert pts.tail_skip_mode("nemotron_h") == "0"
    monkeypatch.setenv(pts.ENV, "0")
    assert pts.tail_skip_mode("gemma4") == "0"


def test_gemma4_default_on_when_unset_and_explicit_zero_disables(monkeypatch):
    with mx.stream(mx.cpu):
        model = _gemma({})
        tokens = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
        monkeypatch.delenv(pts.ENV, raising=False)
        before = pts.STATS["tail_engaged"]
        (logits, applied), = _chunks(model, [tokens.tolist()], model.make_cache())
        assert applied == 1 and logits.shape == (1, 1, 64)
        assert pts.STATS["tail_engaged"] == before + 1
        monkeypatch.setenv(pts.ENV, "0")
        (logits, applied), = _chunks(model, [tokens.tolist()], model.make_cache())
        assert applied == 0 and logits.shape == (1, 6, 64)
