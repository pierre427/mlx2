"""Native Gemma 4 tensor math against the pinned mlx-vlm model, same weights.

Tiny random-weight dense and sparse-MoE configs with the served layout
(sliding/full, k_eq_v full layers with their own head width and KV heads,
proportional partial RoPE), prompts longer than the sliding window, mlx2
caches, B=2 left padding, prompt-lookup verify, checkpoint-row head trim,
quantized loading and the vision tower.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)

pytest.importorskip("mlx_vlm")

from mlx2.adapters.gemma4 import _Gemma4LogitsModel, gemma4_tensor_backend  # noqa: E402
from mlx2.runtime.models import gemma4 as native  # noqa: E402
from mlx2.runtime.models.cache import (  # noqa: E402
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402

V, WINDOW = 96, 8
LAYERS = ["sliding_attention"] * 2 + ["full_attention"]
LAYERS = LAYERS * 2
TEXT = dict(
    hidden_size=32,
    num_hidden_layers=len(LAYERS),
    intermediate_size=64,
    num_attention_heads=4,
    head_dim=8,
    global_head_dim=16,
    vocab_size=V,
    vocab_size_per_layer_input=V,
    num_key_value_heads=2,
    num_global_key_value_heads=1,
    attention_k_eq_v=True,
    num_kv_shared_layers=0,
    hidden_size_per_layer_input=0,
    sliding_window=WINDOW,
    layer_types=LAYERS,
    use_double_wide_mlp=False,
    max_position_embeddings=4096,
    use_bidirectional_attention="vision",
    rope_parameters={
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
)
SPARSE = dict(
    enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=16
)
IMAGE, VIDEO = 90, 91
VISION = dict(
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=8,
    global_head_dim=8,
    patch_size=4,
    pooling_kernel_size=2,
    default_output_length=4,
    position_embedding_size=64,
    standardize=True,
)
TOKENS = dict(
    vocab_size=V,
    image_token_id=IMAGE,
    video_token_id=VIDEO,
    audio_token_id=92,
    boi_token_id=93,
    eoi_token_id=94,
    boa_token_id=95,
    eoa_token_id=89,
)
# Longer than the sliding window: every ring wraps during prefill.
PROMPT = [3, 17, 5, 41, 8, 29, 11, 60, 2, 33, 47, 9, 21, 70, 14, 6, 52, 38, 25, 4]


def _pair(extra, *, vision=False, seed=5):
    """(mlx-vlm model, native model) with identical parameters."""
    from mlx_vlm.models.gemma4 import config as vcfg
    from mlx_vlm.models.gemma4.gemma4 import Model as VLMModel

    text = {**TEXT, **extra}
    vlm_config = vcfg.ModelConfig(
        text_config=vcfg.TextConfig(**text),
        vision_config=vcfg.VisionConfig(**VISION),
        **TOKENS,
    )
    mx.random.seed(seed)
    reference = VLMModel(vlm_config)
    # Non-trivial norms, scalars and router scales so every term is exercised.
    params = {
        k: (v + 0.1 * mx.random.normal(v.shape) if v.ndim <= 1 else v)
        for k, v in tree_flatten(reference.parameters())
    }
    reference.load_weights(list(params.items()))
    mx.eval(reference.parameters())
    ours = native.Model(
        native.ModelConfig.from_dict(
            {"text_config": text, "vision_config": VISION, **TOKENS}
        )
    )
    ours.load_weights(list(params.items()), strict=True)
    mx.eval(ours.parameters())
    if not vision:
        return reference, ours
    return reference, ours


@pytest.fixture(scope="module", params=["dense", "sparse_moe"])
def pair(request):
    return _pair(SPARSE if request.param == "sparse_moe" else {})


def _logits(model, tokens, cache, **kwargs):
    out = model(mx.array(tokens), cache=cache, **kwargs)
    return getattr(out, "logits", out)


def _greedy(model, prompt, steps, chunk=None):
    wrapped = _Gemma4LogitsModel(model)
    cache = wrapped.make_cache()
    chunk = chunk or len(prompt)
    for start in range(0, len(prompt), chunk):
        logits = wrapped(mx.array([prompt[start : start + chunk]]), cache=cache)
    out, rows = [], [logits[0, -1]]
    for _ in range(steps):
        token = int(mx.argmax(logits[0, -1]).item())
        out.append(token)
        logits = wrapped(mx.array([[token]]), cache=cache)
        rows.append(logits[0, -1])
    return out, cache, rows


def test_parameter_trees_match_mlx_vlm(pair):
    reference, ours = pair
    theirs = {k: v.shape for k, v in tree_flatten(reference.parameters())}
    mine = {k: v.shape for k, v in tree_flatten(ours.parameters())}
    assert mine == theirs


def test_prefill_logits_cache_and_greedy_tokens_match(pair):
    reference, ours = pair
    tokens_ref, cache_ref, rows_ref = _greedy(reference, PROMPT, 12, chunk=7)
    tokens_ours, cache_ours, rows_ours = _greedy(ours, PROMPT, 12, chunk=7)
    assert tokens_ours == tokens_ref
    for a, b in zip(rows_ref, rows_ours):
        assert mx.allclose(a, b, atol=1e-5, rtol=1e-5).item()
    for a, b in zip(cache_ref, cache_ours):
        assert type(a) is type(b) and a.offset == b.offset
        ka, va = a.state
        kb, vb = b.state
        assert mx.array_equal(ka, kb).item() and mx.array_equal(va, vb).item()


def test_left_padded_batch_matches_mlx_vlm(pair):
    reference, ours = pair
    long, short = PROMPT, PROMPT[5:16]
    pad = len(long) - len(short)
    tokens = [long, [0] * pad + short]

    def run(model):
        cache = [
            BatchKVCache([0, pad])
            if kind == "full_attention"
            else BatchRotatingKVCache(WINDOW, [0, pad])
            for kind in LAYERS
        ]
        rows = [_logits(model, tokens, cache)[:, -1]]
        step = mx.argmax(rows[0], axis=-1)
        for _ in range(6):
            rows.append(_logits(model, step[:, None].tolist(), cache)[:, -1])
            step = mx.argmax(rows[-1], axis=-1)
        return rows

    for a, b in zip(run(reference), run(ours)):
        assert mx.allclose(a, b, atol=1e-5, rtol=1e-5).item()
        assert mx.array_equal(mx.argmax(a, -1), mx.argmax(b, -1)).item()


def test_checkpoint_row_request_narrows_the_head(pair):
    _, ours = pair
    tokens = [PROMPT[:10], PROMPT[3:13]]
    full = _logits(ours, tokens, [None] * len(LAYERS))
    with prefill_checkpoint_rows([10, 7], width=10) as request:
        narrowed = _logits(ours, tokens, None)
    assert request.applied == 1
    assert narrowed.shape == (2, 1, V)
    assert mx.allclose(narrowed[0, 0], full[0, 9], atol=1e-5).item()
    assert mx.allclose(narrowed[1, 0], full[1, 6], atol=1e-5).item()
    # logits_to_keep keeps the last rows, also before the head.
    kept = _logits(ours, tokens, None, logits_to_keep=2)
    assert mx.allclose(kept, full[:, -2:], atol=1e-5).item()


def test_last_layer_tail_equals_the_full_layer_last_row(pair):
    _, ours = pair
    text = ours.language_model.model
    tokens = mx.array([PROMPT[:12]])
    for warm in (0, 9):
        caches = [text.layers[i] for i in range(len(LAYERS))]
        full_cache = ours.make_cache()
        tail_cache = ours.make_cache()
        if warm:
            _logits(ours, [PROMPT[12 : 12 + warm]], full_cache)
            _logits(ours, [PROMPT[12 : 12 + warm]], tail_cache)
        h = text.embed_tokens(tokens) * text.embed_scale
        masks = text._make_masks(h, full_cache)
        x_full = x_tail = h
        for index, layer in enumerate(caches[:-1]):
            x_full, _, _ = layer(x_full, masks[index], full_cache[index])
            x_tail, _, _ = layer(x_tail, masks[index], tail_cache[index])
        last = caches[-1]
        full, _, _ = last(x_full, masks[-1], full_cache[-1])
        tail = last.tail(x_tail, masks[-1], tail_cache[-1])
        assert tail.shape == (1, 1, TEXT["hidden_size"])
        assert mx.allclose(tail[:, 0], full[:, -1], atol=1e-5).item()
        assert mx.array_equal(full_cache[-1].state[0], tail_cache[-1].state[0]).item()


def test_prompt_lookup_verify_matches_mlx_vlm_and_ordinary(pair):
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    reference, ours = pair
    prompts = [PROMPT + PROMPT[:9], [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7]]
    policy = {
        "num_draft": 4,
        "ngram_min": 2,
        "ngram_max": 3,
        "adaptive": False,
        "deferred_admission": False,
    }

    def run(model):
        generator = PromptLookupBatchGenerator(
            _Gemma4LogitsModel(model),
            completion_batch_size=len(prompts),
            prefill_step_size=5,
            prompt_lookup=policy,
        )
        uids = generator.insert(prompts, max_tokens=[16] * len(prompts))
        out = {uid: [] for uid in uids}
        done = set()
        for _ in range(4000):
            _, responses = generator.next()
            for response in responses:
                out[response.uid].append(response.token)
                if response.finish_reason:
                    done.add(response.uid)
            if len(done) == len(uids):
                break
        generator.close()
        return [out[uid] for uid in uids]

    ordinary = [_greedy(ours, prompt, 16)[0] for prompt in prompts]
    assert run(ours) == ordinary
    assert run(reference) == ordinary


@pytest.mark.parametrize("extra", [{}, SPARSE], ids=["dense", "sparse_moe"])
def test_image_and_video_prefill_match_mlx_vlm(extra):
    reference, ours = _pair(extra, vision=True, seed=9)
    pixels = mx.random.uniform(shape=(1, 3, 16, 16), key=mx.random.key(4))
    prompt = [2, 5, 93] + [IMAGE] * 4 + [94, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    types = [[1 if t == IMAGE else 0 for t in prompt]]
    for model in (reference, ours):
        model.eval()

    def run(model, **media):
        cache = _Gemma4LogitsModel(model).make_cache()
        first = _logits(
            model, [prompt], cache, mm_token_type_ids=mx.array(types), **media
        )
        follow = _logits(model, [[int(mx.argmax(first[0, -1]).item())]], cache)
        return first, follow

    for a, b in zip(run(reference, pixel_values=pixels), run(ours, pixel_values=pixels)):
        assert mx.allclose(a, b, atol=1e-5, rtol=1e-5).item()
    frames = mx.random.uniform(shape=(2, 3, 16, 16), key=mx.random.key(6))
    video_prompt = [2] + [VIDEO] * 8 + [7, 11, 13, 17, 19, 23, 29, 31, 37]

    def run_video(model):
        cache = _Gemma4LogitsModel(model).make_cache()
        return _logits(
            model,
            [video_prompt],
            cache,
            pixel_values_videos=frames,
            mm_token_type_ids=mx.array([[2 if t == VIDEO else 0 for t in video_prompt]]),
        )

    assert mx.allclose(run_video(reference), run_video(ours), atol=1e-5).item()
    # The projected-feature cache serves the second request without the tower.
    store = {}

    class Cache:
        def get(self, key):
            return store.get(key)

        def put(self, key, value):
            store[key] = value

    cold = _logits(ours, [prompt], None, pixel_values=pixels, vision_cache=Cache(),
                   _image_key="k", mm_token_type_ids=mx.array(types))
    warm = _logits(ours, [prompt], None, pixel_values=pixels * 0, vision_cache=Cache(),
                   _image_key="k", mm_token_type_ids=mx.array(types))
    assert "k" in store and mx.array_equal(cold, warm).item()


@pytest.mark.parametrize("mode,group", [("affine", 64), ("mxfp8", 32)])
def test_quantized_artifact_loads_like_mlx_vlm(mode, group):
    # Hidden 64 so projections have a quantizable input width.
    reference, ours = _pair({**SPARSE, "hidden_size": 64}, seed=11)
    router = "language_model.model.layers.0.router.proj"
    quant = {"group_size": group, "bits": 8, "mode": mode, router: {"group_size": 64, "bits": 8}}

    def predicate(path, module):
        if path == router:
            return quant[router]
        if path.startswith("vision_tower"):
            return False
        return hasattr(module, "to_quantized") and module.weight.shape[-1] % 64 == 0

    nn.quantize(reference, group_size=group, bits=8, mode=mode, class_predicate=predicate)
    weights = dict(tree_flatten(reference.parameters()))
    native.quantize_like_artifact(ours, quant, weights)
    ours.load_weights(list(weights.items()), strict=True)
    tokens = [PROMPT[:12]]
    a = _logits(reference, tokens, _Gemma4LogitsModel(reference).make_cache())
    b = _logits(ours, tokens, ours.make_cache())
    assert mx.allclose(a, b, atol=1e-5, rtol=1e-5).item()
    assert type(ours.language_model.model.embed_tokens).__name__ == "QuantizedEmbedding"
    assert ours.language_model.model.layers[0].router.proj.mode == "affine"
    with pytest.raises(ValueError, match="unknown modules"):
        native.quantize_like_artifact(ours, {**quant, "no.such.module": False}, weights)


def test_hf_layout_keys_are_sanitized_like_mlx_vlm():
    _, ours = _pair(SPARSE, seed=13)
    weights = dict(tree_flatten(ours.parameters()))
    prefix = "language_model.model.layers.0.experts.switch_glu."
    gate, up = weights.pop(prefix + "gate_proj.weight"), weights.pop(prefix + "up_proj.weight")
    down = weights.pop(prefix + "down_proj.weight")
    hf = {"model." + k.replace("language_model.model.", "language_model."): v
          for k, v in weights.items()}
    hf["model.language_model.layers.0.experts.gate_up_proj"] = mx.concatenate(
        [gate, up], axis=1
    )
    hf["model.language_model.layers.0.experts.down_proj"] = down
    hf["model.language_model.layers.0.self_attn.rotary_emb.inv_freq"] = mx.zeros((2,))
    sanitized = ours.sanitize(hf)
    assert mx.array_equal(sanitized[prefix + "gate_proj.weight"], gate).item()
    assert mx.array_equal(sanitized[prefix + "up_proj.weight"], up).item()
    assert set(sanitized) == set(dict(tree_flatten(ours.parameters())))


def test_unported_topologies_fail_closed():
    for extra, match in (
        ({"num_kv_shared_layers": 2}, "KV-shared"),
        ({"hidden_size_per_layer_input": 8}, "per-layer inputs"),
        ({"enable_moe_block": True}, "MoE block"),
    ):
        with pytest.raises(ValueError, match=match):
            native.Model(native.ModelConfig.from_dict(
                {"text_config": {**TEXT, **extra}, "vision_config": VISION, **TOKENS}
            ))
    with pytest.raises(ValueError, match="audio"):
        native.Model(native.ModelConfig.from_dict(
            {"text_config": TEXT, "vision_config": VISION, "audio_config": {}, **TOKENS}
        ))
    bad_rope = {**TEXT, "rope_parameters": {"full_attention": {"rope_type": "yarn"}}}
    with pytest.raises(ValueError, match="RoPE"):
        native.Model(native.ModelConfig.from_dict(
            {"text_config": bad_rope, "vision_config": VISION, **TOKENS}
        ))


def test_backend_switch_defaults_to_native_and_refuses_unknown():
    assert gemma4_tensor_backend({}) == "native"
    assert gemma4_tensor_backend({"MLX2_GEMMA4_BACKEND": ""}) == "native"
    assert gemma4_tensor_backend({"MLX2_GEMMA4_BACKEND": "native"}) == "native"
    assert gemma4_tensor_backend({"MLX2_GEMMA4_BACKEND": "mlx-vlm"}) == "mlx_vlm"
    with pytest.raises(ValueError, match="MLX2_GEMMA4_BACKEND"):
        gemma4_tensor_backend({"MLX2_GEMMA4_BACKEND": "torch"})


def test_mlx2_caches_are_the_native_cache_types():
    _, ours = _pair({}, seed=3)
    caches = _Gemma4LogitsModel(ours).make_cache()
    assert [type(c) for c in caches] == [
        KVCache if kind == "full_attention" else RotatingKVCache for kind in LAYERS
    ]
    assert [type(c) for c in ours.make_cache()] == [type(c) for c in caches]


def _write_artifact(directory, model, config, *, stale_shard=False):
    import json

    weights = dict(tree_flatten(model.parameters()))
    names = sorted(weights)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half],
              "model-00002-of-00002.safetensors": names[half:]}
    for shard, keys in shards.items():
        mx.save_safetensors(str(directory / shard), {k: weights[k] for k in keys})
    (directory / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: shard for shard, keys in shards.items() for k in keys}}))
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 7, 9]}))
    if stale_shard:
        # Same keys, wrong values: must not be read (not in the index).
        mx.save_safetensors(str(directory / "zz-stale.safetensors"),
                            {k: mx.zeros_like(v) for k, v in weights.items()})


def test_native_loader_matches_mlx_vlm_load_model_from_disk(tmp_path):
    from mlx_vlm.utils import load_model

    reference, _ = _pair({**SPARSE, "hidden_size": 64}, seed=17)
    router = "language_model.model.layers.0.router.proj"
    quant = {"group_size": 64, "bits": 8, "mode": "affine",
             router: {"group_size": 64, "bits": 8}}
    nn.quantize(reference, group_size=64, bits=8, class_predicate=lambda p, m: (
        quant[router] if p == router else
        hasattr(m, "to_quantized") and not p.startswith("vision_tower")
        and m.weight.shape[-1] % 64 == 0))
    config = {"model_type": "gemma4", "text_config": {**TEXT, **SPARSE, "hidden_size": 64},
              "vision_config": VISION, "quantization": quant,
              "quantization_config": quant, **TOKENS}
    _write_artifact(tmp_path, reference, config, stale_shard=True)
    ours = native.load(tmp_path)
    theirs = load_model(tmp_path, lazy=False, strict=True)
    assert ours.config.eos_token_id == theirs.config.eos_token_id == [1, 7, 9]
    tokens = [PROMPT[:12]]
    a = _logits(theirs, tokens, _Gemma4LogitsModel(theirs).make_cache())
    b = _logits(ours, tokens, ours.make_cache())
    assert mx.allclose(a, b, atol=1e-5, rtol=1e-5).item()
    assert mx.allclose(b, _logits(reference, tokens, ours.make_cache()), atol=1e-5).item()
    # quantization_config alone (an HF-style quantized checkpoint) is refused.
    import json
    alone = {k: v for k, v in config.items() if k != "quantization"}
    (tmp_path / "config.json").write_text(json.dumps(alone))
    with pytest.raises(ValueError, match="quantization_config"):
        native.load(tmp_path)


def test_native_model_refuses_unported_speculative_kwargs():
    _, ours = _pair({}, seed=3)
    with pytest.raises(TypeError, match="return_hidden"):
        ours(mx.array([PROMPT[:4]]), return_hidden=True)


def test_adapter_native_backend_wiring(monkeypatch, tmp_path):
    from mlx2.adapters.gemma4 import Gemma431BAdapter

    adapter = object.__new__(Gemma431BAdapter)
    adapter.layout = Gemma431BAdapter.descriptor.cache_layout
    loaded = {}

    class Backend:
        def load_processor(self, path, *, eos_token_ids=None):
            loaded["eos"] = eos_token_ids
            return "processor"

        def load(self, *args, **kwargs):  # the mlx-vlm model must not load
            raise AssertionError("mlx-vlm model loaded on the native backend")

    adapter._vlm_backend = Backend()
    monkeypatch.setattr(native, "load_config", lambda path: {"eos_token_id": [1, 106]})
    monkeypatch.setattr(native, "load", lambda path: "native-model")
    monkeypatch.setenv("MLX2_GEMMA4_BACKEND", "native")
    assert adapter._load_model_and_processor(tmp_path) == ("native-model", "processor")
    assert loaded["eos"] == [1, 106]
    assert adapter.tensor_backend == "native"
    assert adapter.profile_name(False) == "gemma4-native-apcv2-ordinary"
    assert adapter.layout == "gemma4-31b-full-sliding-v1+native-tensors"
    assert adapter.exact_prefix_cascade_contract()["shared_prefix_reuse"]["cache_layout"] == (
        adapter.layout)
    # The explicit mlx-vlm switch loads mlx-vlm's model and keeps its names.
    monkeypatch.setenv("MLX2_GEMMA4_BACKEND", "mlx_vlm")
    reference = object.__new__(Gemma431BAdapter)
    reference.layout = Gemma431BAdapter.descriptor.cache_layout
    reference._vlm_backend = Backend()
    reference.identity = {"config": {}}
    import mlx2.adapters.mlx_vlm as vlm_module
    monkeypatch.setattr(vlm_module, "_load_model",
                        lambda load, path, **kw: ("vlm-model", "vlm-processor"))
    assert reference._load_model_and_processor(tmp_path) == ("vlm-model", "vlm-processor")
    assert reference.tensor_backend == "mlx_vlm"
    assert reference.profile_name(False) == "mlx-vlm-apcv2-ordinary"
    assert reference.layout == "gemma4-31b-full-sliding-v1"


@pytest.mark.parametrize("mode", ["head", "1"])
def test_prompt_chunk_tail_skip_modes_run_on_the_native_model(monkeypatch, pair, mode):
    """MLX2_PREFILL_TAIL_SKIP (adapter wrapper) on the native model: same rows, same cache."""
    from mlx2.runtime import prefill_tail_skip as pts

    _, ours = pair
    model = _Gemma4LogitsModel(ours)
    chunks = [PROMPT[:11], PROMPT[11:18]]

    def run(flag):
        monkeypatch.setenv(pts.ENV, flag)
        cache, rows = model.make_cache(), []
        for chunk in chunks:
            with prefill_checkpoint_rows([len(chunk)], width=len(chunk)) as request:
                logits = model(mx.array([chunk]), cache=cache)
            assert request.applied == 1 and logits.shape == (1, 1, V)
            rows.append(logits[0, -1])
        return rows, cache

    want, want_cache = run("0")  # native head trim through select_checkpoint_rows
    before = dict(pts.STATS)
    got, got_cache = run(mode)
    # The native backend runs its own DecoderLayer.tail (tail_last).
    split = (pts.STATS["gemma4_native_tail_split"]
             - before.get("gemma4_native_tail_split", 0))
    assert split == (2 if mode == "1" else 0)
    assert pts.STATS["gemma4_tail_layer_split"] == before.get("gemma4_tail_layer_split", 0)
    for a, b in zip(want, got):
        assert mx.allclose(a, b, atol=1e-5, rtol=1e-5).item()
    for a, b in zip(want_cache, got_cache):
        assert mx.array_equal(a.state[0], b.state[0]).item()
        assert mx.array_equal(a.state[1], b.state[1]).item()


def _tail_chunks(wrapped, chunks, cache):
    outs = []
    for chunk in chunks:
        tokens = mx.array(chunk, dtype=mx.int32)
        with prefill_checkpoint_rows([tokens.shape[1]] * tokens.shape[0],
                                     width=tokens.shape[1]) as request:
            logits = wrapped(tokens, cache=cache)
            mx.eval(logits)
        outs.append((logits, request.applied))
    return outs


@pytest.mark.parametrize("left_padding", [None, [0, 3]], ids=["b1", "b2_left_padded"])
def test_native_prompt_chunk_tail_skip_default_on(pair, monkeypatch, left_padding):
    """MLX2_PREFILL_TAIL_SKIP unset (Gemma 4 default 1) drives DecoderLayer.tail."""
    import numpy as np

    from mlx2.runtime import prefill_tail_skip as pts
    from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache

    _, ours = pair
    wrapped = _Gemma4LogitsModel(ours)

    def make():
        if left_padding is None:
            return wrapped.make_cache()
        text = ours.config.text_config
        return [BatchKVCache(left_padding) if kind == "full_attention"
                else BatchRotatingKVCache(text.sliding_window, left_padding)
                for kind in text.layer_types]

    rows = 1 if left_padding is None else 2
    rng = np.random.default_rng(11)
    chunks = [rng.integers(1, TEXT["vocab_size"], (rows, 11)).tolist(),
              rng.integers(1, TEXT["vocab_size"], (rows, 7)).tolist()]
    monkeypatch.setenv(pts.ENV, "0")
    full_cache = make()
    full = _tail_chunks(wrapped, chunks, full_cache)
    monkeypatch.delenv(pts.ENV, raising=False)
    before = pts.STATS["gemma4_native_tail_split"]
    tail_cache = make()
    tail = _tail_chunks(wrapped, chunks, tail_cache)
    assert pts.STATS["gemma4_native_tail_split"] - before == 2
    for (want, want_applied), (got, applied) in zip(full, tail):
        # Native always trims the head; the tail additionally claims the request.
        assert want.shape == got.shape == (rows, 1, TEXT["vocab_size"])
        assert applied == 1 and want_applied == 1
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-4)
    for a, b in zip(full_cache, tail_cache):
        for x, y in zip(a.state[:2], b.state[:2]):
            np.testing.assert_array_equal(np.array(x), np.array(y))
    step = mx.array([[5]] * rows, dtype=mx.int32)
    np.testing.assert_array_equal(np.array(wrapped(step, cache=full_cache)),
                                  np.array(wrapped(step, cache=tail_cache)))
