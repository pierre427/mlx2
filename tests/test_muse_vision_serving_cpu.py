"""Muse native vision: independent reference math and paired AST cache state."""
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.muse_glimmer import MUSE_AST_ROUTE, MuseGlimmerAdapter, normalize_external_policy
from mlx2.adapters.muse_glimmer_images import process_images, policy
from mlx2.adapters.muse_vision_config import VisionConfig
from mlx2.runtime.models.muse_vision import VisionModel
from mlx2.runtime.models.muse_multimodal import Model
from test_muse_dflash_assistant_cpu import tiny


@pytest.fixture(autouse=True)
def cpu():
    mx.set_default_device(mx.cpu)


def vision_config():
    return VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=2, patch_size=2, patch_temporal=2, pos_emb_height=2,
        pos_emb_width=2, merge_size=2, layer_types=['window_attention', 'full_attention', 'window_attention'])


def test_tower_matches_transformers_for_multiple_non_square_windowed_images():
    import torch
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel

    torch.manual_seed(74)
    config = vision_config()
    reference = MuseGlimmerVisionModel(MuseGlimmerVisionConfig(**asdict(config))).eval()
    reference.config._attn_implementation = 'eager'
    model = VisionModel(config)
    model.load_weights([(name, mx.array(value.detach().numpy())) for name, value in reference.state_dict().items()], strict=True)
    grid = [[1, 4, 6], [1, 2, 4]]
    pixels = np.random.default_rng(74).normal(size=(32, 24)).astype(np.float32)
    with torch.no_grad():
        expected = reference(torch.from_numpy(pixels), grid_thw=torch.tensor(grid)).last_hidden_state.numpy()
    actual = np.asarray(model(mx.array(pixels), mx.array(grid)))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_preprocessing_matches_primary_processor_patch_order_and_resize(tmp_path):
    import json
    from PIL import Image
    from transformers.models.muse_glimmer.image_processing_muse_glimmer import MuseGlimmerImageProcessor

    (tmp_path / 'processor_config.json').write_text(json.dumps({'image_processor': {
        'patch_size': 14, 'temporal_patch_size': 2, 'merge_size': 2,
        'max_image_tokens': 4096, 'image_mean': [.5] * 3, 'image_std': [.5] * 3,
        'rescale_factor': 1 / 255}}))
    settings = policy(tmp_path)
    images = [Image.fromarray(np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8))
              for seed, shape in [(3, (56, 84, 3)), (4, (101, 213, 3))]]
    pixels, grids = process_images(images, settings)
    # Tensor Lanczos differs from the pinned PIL policy by up to two integer
    # pixel levels on this resized fixture. Compare layout/normalization on
    # shared resized pixels, and retain a separate cross-backend error bound.
    reference = MuseGlimmerImageProcessor()
    tensor_resized = reference(images=images, return_tensors='pt')
    from mlx2.adapters.muse_glimmer_images import smart_resize
    resized = []
    for image in images:
        h, w = smart_resize(image.height, image.width, 28, 4096)
        resized.append(image.resize((w, h), Image.Resampling.LANCZOS))
    expected = reference(images=resized, do_resize=False, return_tensors='pt')
    assert grids == expected['image_grid_thw'].tolist()
    np.testing.assert_allclose(pixels, expected['pixel_values'].numpy(), atol=2e-7, rtol=2e-7)
    assert float(np.abs(pixels - tensor_resized['pixel_values'].numpy()).max()) <= 4 / 255 + 1e-6


def native_pair():
    target, draft = tiny()
    root = {'vision_config': asdict(vision_config()), 'image_token_id': 95,
            'out_hidden_size': 64, 'projector_hidden_size': 16}
    model = Model(target.args, root)
    mx.eval(model.parameters())
    return model, draft.bind(model)


def ordinary(model, prompt, media, limit):
    cache = model.make_cache()
    model.prefill_forward(mx.array([prompt[:-1]]), cache=cache, **media)
    logits = model(mx.array([[prompt[-1]]]), cache=cache)
    output = []
    for _ in range(limit):
        token = int(mx.argmax(logits[0, -1]).item())
        output.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return output


def test_image_ast_isolated_prefill_commits_taps_and_restores_paired_boundary():
    from mlx2.adapters.muse_glimmer import configure_environment
    configure_environment(ast=True)
    target, draft = native_pair()
    adapter = object.__new__(MuseGlimmerAdapter)
    adapter.model, adapter.draft_model = target, draft
    adapter.identity = {'fingerprint': 'synthetic-vision-pair'}
    adapter.external_policy = normalize_external_policy({'draft_model': '/draft',
        'batch_size_route': MUSE_AST_ROUTE, 'num_draft': 5})
    prompts = [[1, 95, 95, 95, 95, 3, 4], [2, 95, 95, 95, 95, 6]]
    media = [{'pixel_values': np.ones((16, 24), np.float32) * value, 'image_grid_thw': [[1, 4, 4]]} for value in [.2, -.3]]
    expected = [ordinary(target, p, m, 12) for p, m in zip(prompts, media)]
    batch = adapter.create_external_batch(completion_batch_size=2, prefill_step_size=2)
    uids = batch.insert(prompts, max_tokens=[12, 12], prefill_inputs=media)
    output, boundaries = {uid: [] for uid in uids}, {}
    for _ in range(100):
        progress, responses = batch.next()
        for event in progress:
            if event.end_of_prompt:
                boundaries[event.uid] = batch.pop_prompt_boundary(event.uid)
        for response in responses:
            output[response.uid].append(response.token)
        if not batch.lanes:
            break
    assert [output[uid] for uid in uids] == expected
    assert batch.scheduler_stats['external_media_prefill_rounds'] == 2
    assert batch.scheduler_stats['external_auto_tree_rounds'] > 0
    assert batch.scheduler_stats['external_auto_chain_rounds'] == 0
    before = target.vision_stats['encoder_calls']
    boundary = boundaries[uids[0]]
    warm = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=2)
    uid, = warm.insert([prompts[0][-1:]], max_tokens=[12],
        all_tokens=[prompts[0][:-1]], caches=[boundary['target_cache']],
        cache_states=[boundary['cache_sidecar']])
    from test_muse_dflash_assistant_cpu import drain
    actual, _ = drain(warm)
    assert actual[uid] == expected[0]
    assert target.vision_stats['encoder_calls'] == before


def test_image_grid_mismatch_fails_before_any_encoder_work():
    target, _ = native_pair()
    with pytest.raises(ValueError, match='differ'):
        target.prefill_forward(mx.array([[1, 95, 3]]), cache=target.make_cache(),
            pixel_values=mx.zeros((16, 24)), image_grid_thw=[[1, 4, 4]])
    assert target.vision_stats['encoder_calls'] == 0


def test_external_media_refuses_undeclared_targets_and_split_prefill():
    from mlx2.adapters.muse_glimmer import configure_environment
    configure_environment(ast=True)
    def batch_for(model, draft):
        adapter = object.__new__(MuseGlimmerAdapter)
        adapter.model, adapter.draft_model = model, draft
        adapter.identity = {'fingerprint': 'synthetic-media-refusal'}
        adapter.external_policy = normalize_external_policy({'draft_model': '/draft',
            'batch_size_route': MUSE_AST_ROUTE, 'num_draft': 3})
        return adapter.create_external_batch(completion_batch_size=1)
    plain, draft = tiny()
    batch = batch_for(plain, draft)
    with pytest.raises(ValueError, match='unavailable'):
        batch.insert([[1, 2, 3]], max_tokens=[4], prefill_inputs=[{'pixel_values': []}])
    assert not batch.lanes
    target, draft = native_pair()
    batch = batch_for(target, draft)
    prompt = [1, 95, 95, 95, 95, 3, 4]
    uid, = batch.insert([prompt], max_tokens=[4], prefill_inputs=[{
        'pixel_values': np.zeros((16, 24), np.float32), 'image_grid_thw': [[1, 4, 4]]}])
    lane = batch.lanes[uid]
    with pytest.raises(ValueError, match='cannot split'):
        batch._prefill(lane, step=2)
    assert list(lane.remaining) == prompt
    assert target.vision_stats['encoder_calls'] == 0


def test_request_expansion_and_media_identity_use_ordered_image_content():
    import base64
    from io import BytesIO
    from PIL import Image
    from transformers import AutoTokenizer

    root = '~/mlx-models-m3-imports-20261021/Muse-Glimmer-30B-mlx-4bit'
    if not Path(root).exists():
        pytest.skip('optional local Muse target absent')
    adapter = object.__new__(MuseGlimmerAdapter)
    adapter.image_policy = policy(root)
    adapter._config = {'image_token_id': 200092}
    adapter.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    def source(color):
        data = BytesIO()
        Image.new('RGB', (56, 84), color).save(data, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()
    def prepare(images):
        return adapter.prepare_multimodal_request({'messages': [{'role': 'user', 'content':
            [{'type': 'image_url', 'image_url': {'url': image}} for image in images]
            + [{'type': 'text', 'text': 'Describe the colors in order.'}]}],
            'reasoning_effort': 'none'})
    red, blue = source('red'), source('blue')
    a, replay, b = prepare([red, blue]), prepare([red, blue]), prepare([blue, red])
    assert a['_mlx2_prompt_tokens'] == b['_mlx2_prompt_tokens']
    assert a['_mlx2_media_fingerprint'] == replay['_mlx2_media_fingerprint']
    assert a['_mlx2_media_fingerprint'] != b['_mlx2_media_fingerprint']
    assert a['_mlx2_prompt_tokens'].count(200092) == 12
    assert a['_mlx2_media_token_end'] < len(a['_mlx2_prompt_tokens']) - 1
    assert adapter.prompt_tokens(a) == a['_mlx2_prompt_tokens']
    with pytest.raises(ValueError, match='without'):
        adapter.validate_prefill_inputs(a, a['_mlx2_prompt_tokens'], None)
    assert adapter.validate_prefill_inputs(a, [7], None) is None


def test_extreme_aspect_ratio_refuses_reference_fallback_token_overflow():
    from mlx2.adapters.muse_glimmer_images import smart_resize
    # Three million source pixels fit the shared image loader's pixel limit,
    # but the reference fallback would expand them past4096merged tokens.
    with pytest.raises(ValueError, match='token bound'):
        smart_resize(3, 1_000_000, 28, 4096)
    h, w = smart_resize(224, 560, 28, 4096)
    assert (h // 28) * (w // 28) <= 4096
