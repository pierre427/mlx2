"""Ling-3.2-air vision tower, projector and processor vs independent CPU references."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx2.adapters import ling32_air_images as images_mod
from mlx2.adapters.ling32_air_images import policy, process_images, smart_resize
from mlx2.runtime.models import ling32_vision as lv
from mlx2.runtime.models.ling32_vision import (
    Projector, VisionConfig, VisionModel, load_vision, sanitize, vision_stats)

ARTIFACT = Path.home() / 'mlx-models/Ling-3.2-air-mlx2-e4r8'
TINY = dict(depth=2, hidden_size=64, intermediate_size=96, num_heads=4, in_channels=3,
            patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64)


@pytest.fixture(autouse=True)
def cpu():
    mx.set_default_device(mx.cpu)


def _cosine_rows(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def _reference_tower(seed=11):
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.manual_seed(seed)
    config = Qwen3VLVisionConfig(**TINY, hidden_act='gelu_pytorch_tanh', out_hidden_size=32,
                                 deepstack_visual_indexes=[])
    config._attn_implementation = 'eager'
    reference = Qwen3VLVisionModel(config).eval()
    with torch.no_grad():
        # Non-trivial norms and biases so a missing affine term would show.
        for name, param in reference.named_parameters():
            if name.endswith('bias'):
                param.copy_(torch.randn_like(param) * 0.1)
            elif 'norm' in name:
                param.copy_(1 + torch.randn_like(param) * 0.1)
    return reference


def _ours_from(reference):
    model = VisionModel(VisionConfig(**TINY))
    weights = {'model.visual.' + name: mx.array(value.detach().numpy())
               for name, value in reference.state_dict().items()
               if not name.startswith('merger.linear_fc') and 'deepstack' not in name}
    sanitized = sanitize(weights, model.config)
    model.load_weights([(k[len('vision_tower.'):], v) for k, v in sanitized.items()], strict=True)
    return model


@pytest.mark.parametrize('grid', [[[1, 4, 6], [1, 8, 4]], [[1, 6, 6]]])
def test_tower_matches_transformers_qwen3vl_without_deepstack(grid):
    import torch

    reference = _reference_tower()
    model = _ours_from(reference)
    total = sum(t * h * w for t, h, w in grid)
    pixels = np.random.default_rng(5).normal(size=(total, 3 * 2 * 16 * 16)).astype(np.float32)
    with torch.no_grad():
        out = reference(torch.from_numpy(pixels), grid_thw=torch.tensor(grid))
        expected = reference.merger.norm(out.last_hidden_state).reshape(-1, 4 * 64).numpy()
    before = dict(vision_stats)
    actual = np.asarray(model(mx.array(pixels), grid))
    assert actual.shape == (total // 4, 256)
    cos = _cosine_rows(actual, expected)
    err = np.abs(actual - expected).max()
    print(f'tower grid={grid} worst_cos={cos.min():.8f} max_abs={err:.3e}')
    assert cos.min() >= 0.9999
    assert err < 1e-4
    assert vision_stats['encoder_calls'] == before['encoder_calls'] + 1
    assert vision_stats['images'] == before['images'] + len(grid)
    assert vision_stats['patches'] == before['patches'] + total
    assert vision_stats['feature_tokens'] == before['feature_tokens'] + total // 4


def test_two_images_attend_separately():
    reference = _reference_tower(seed=3)
    model = _ours_from(reference)
    grid = [[1, 4, 6], [1, 8, 4]]
    rng = np.random.default_rng(9)
    first = rng.normal(size=(24, 1536)).astype(np.float32)
    second = rng.normal(size=(32, 1536)).astype(np.float32)
    joint = np.asarray(model(mx.array(np.concatenate([first, second])), grid))
    alone = np.asarray(model(mx.array(first), [grid[0]]))
    np.testing.assert_allclose(joint[:6], alone, atol=1e-5, rtol=1e-5)


def test_projector_matches_torch_sequential():
    import torch

    torch.manual_seed(2)
    reference = torch.nn.Sequential(torch.nn.Linear(48, 40), torch.nn.GELU(), torch.nn.Linear(40, 40))
    ours = Projector(48, 40)
    ours.load_weights([('linear_proj.' + name, mx.array(value.detach().numpy()))
                       for name, value in reference.state_dict().items()], strict=True)
    x = np.random.default_rng(1).normal(size=(7, 48)).astype(np.float32) * 3
    with torch.no_grad():
        expected = reference(torch.from_numpy(x)).numpy()
    actual = np.asarray(ours(mx.array(x)))
    print(f'projector max_abs={np.abs(actual - expected).max():.3e}')
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def _preprocessor_dir(tmp_path, **overrides):
    config = {'min_pixels': 4096, 'max_pixels': 4194304, 'patch_size': 16, 'temporal_patch_size': 2,
              'merge_size': 2, 'image_mean': [0.48145466, 0.4578275, 0.40821073],
              'image_std': [0.26862954, 0.26130258, 0.27577711],
              'image_processor_type': 'BailingMM2ImageProcessor', **overrides}
    (tmp_path / 'preprocessor_config.json').write_text(json.dumps(config))
    return tmp_path


def _synthetic_images():
    rng = np.random.default_rng(17)
    sizes = [(37, 53), (300, 97), (64, 64), (128, 96)]
    out = [Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)) for h, w in sizes]
    out.append(Image.fromarray(rng.integers(0, 256, size=(45, 70, 4), dtype=np.uint8), mode='RGBA'))
    return out


def test_processor_refuses_combined_patch_count_before_allocating(tmp_path, monkeypatch):
    settings = policy(_preprocessor_dir(tmp_path))
    img = Image.new('RGB', (64, 64))

    _, grids = process_images(iter([img]), settings, max_patches=16)
    assert grids == [[1, 4, 4]]

    def unexpected_resize(*args, **kwargs):
        pytest.fail('patch budget must be checked before resizing any image')

    monkeypatch.setattr(Image.Image, 'resize', unexpected_resize)
    with pytest.raises(ValueError, match=r'16 image patches per request .*32'):
        process_images([img, img], settings, max_patches=16)


def test_processor_matches_transformers_qwen2vl_pil(tmp_path):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    settings = policy(_preprocessor_dir(tmp_path))
    reference = Qwen2VLImageProcessorPil(
        min_pixels=4096, max_pixels=4194304, patch_size=16, merge_size=2, temporal_patch_size=2,
        image_mean=settings['image_mean'], image_std=settings['image_std'])
    imgs = _synthetic_images()
    expected = reference(imgs, return_tensors='np')
    pixels, grids = process_images(imgs, settings)
    assert grids == expected['image_grid_thw'].tolist()
    assert pixels.dtype == np.float32 and pixels.shape == expected['pixel_values'].shape
    err = np.abs(pixels - expected['pixel_values']).max()
    print(f'processor (PIL backend) max_abs={err:.3e}')
    assert err == 0.0


def test_processor_grids_match_transformers_qwen2vl_torchvision(tmp_path):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    settings = policy(_preprocessor_dir(tmp_path))
    reference = Qwen2VLImageProcessor(
        min_pixels=4096, max_pixels=4194304, patch_size=16, merge_size=2, temporal_patch_size=2,
        image_mean=settings['image_mean'], image_std=settings['image_std'])
    imgs = _synthetic_images()
    expected = reference(imgs, return_tensors='np')
    pixels, grids = process_images(imgs, settings)
    assert grids == expected['image_grid_thw'].tolist()
    # Images already at their resized size (64x64, 128x96) skip resampling:
    # layout/normalisation must agree to float32 rounding (torch's fused
    # normalize vs numpy's two-step). Resampled images use a different
    # bicubic kernel (torchvision vs PIL), so only the layout is compared.
    offset = 0
    for image, (_, gh, gw) in zip(imgs, grids):
        count = gh * gw
        if (image.height, image.width) == (gh * 16, gw * 16):
            err = np.abs(pixels[offset:offset + count] - expected['pixel_values'][offset:offset + count]).max()
            print(f'processor (torchvision, no resample) {image.size} max_abs={err:.3e}')
            assert err <= 2e-6
        offset += count


def test_smart_resize_vendor_law():
    assert smart_resize(64, 64, 32, 4096, 4194304) == (64, 64)
    assert smart_resize(37, 53, 32, 4096, 4194304) == (64, 96)
    h, w = smart_resize(5000, 3000, 32, 4096, 4194304)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 4194304
    with pytest.raises(ValueError, match='aspect ratio'):
        smart_resize(10, 2100, 32, 4096, 4194304)


def test_policy_refuses_other_laws(tmp_path):
    good = policy(_preprocessor_dir(tmp_path))
    assert good['version'] == images_mod.PROCESSOR_VERSION and len(good['revision']) == 64
    for overrides in ({'patch_size': 14}, {'max_pixels': 1003520}, {'image_mean': [0.5, 0.5, 0.5]},
                      {'image_processor_type': 'Qwen2VLImageProcessor'}, {'resample': 2},
                      {'do_normalize': False}, {'rescale_factor': 1 / 127.5}):
        directory = tmp_path / str(len(list(tmp_path.iterdir())))
        directory.mkdir()
        with pytest.raises(ValueError, match='preprocessing law'):
            policy(_preprocessor_dir(directory, **overrides))
    with pytest.raises(ValueError, match='preprocessing law'):
        process_images(_synthetic_images()[:1], {**good, 'version': 'other'})


def test_config_refusals():
    geometry = dict(lv.LING32_VISION_GEOMETRY, out_hidden_size=4096, disable_merger_proj=True)
    assert VisionConfig.from_dict(geometry).merged_hidden_size == 4608
    assert VisionConfig.from_dict({**geometry, 'deepstack_visual_indexes': []}).depth == 27
    for bad in ({'depth': 24}, {'hidden_size': 1280}, {'patch_size': 14}, {'num_position_embeddings': 2048},
                {'hidden_act': 'gelu'}, {'model_type': 'qwen3_vl'}):
        with pytest.raises(ValueError, match='geometry'):
            VisionConfig.from_dict({**geometry, **bad})
    with pytest.raises(ValueError, match='DeepStack'):
        VisionConfig.from_dict({**geometry, 'deepstack_visual_indexes': [8, 16, 24]})
    with pytest.raises(ValueError, match='disable_merger_proj'):
        VisionConfig.from_dict({**geometry, 'disable_merger_proj': False})
    with pytest.raises(ValueError, match='gelu_pytorch_tanh'):
        VisionConfig(**{**TINY, 'hidden_act': 'gelu'})


def test_pixel_grid_mismatch_refused():
    model = VisionModel(VisionConfig(**TINY))
    with pytest.raises(ValueError, match='do not match'):
        model(mx.zeros((20, 1536)), [[1, 4, 6]])
    with pytest.raises(ValueError, match=r'\[N, 1536\]'):
        model(mx.zeros((24, 768)), [[1, 4, 6]])
    with pytest.raises(ValueError, match='multiples of the merge size'):
        model(mx.zeros((15, 1536)), [[1, 3, 5]])
    with pytest.raises(ValueError, match='grid_thw'):
        model(mx.zeros((24, 1536)), [4, 6])


def test_sanitize_refuses_unknown_keys():
    with pytest.raises(ValueError, match='Unexpected'):
        sanitize({'model.language_model.embed.weight': mx.zeros((1,))})


@pytest.mark.skipif(not (ARTIFACT / 'vision.safetensors').exists(), reason='Ling-3.2-air artifact absent')
def test_real_checkpoint_strict_load_and_one_image(tmp_path):
    tower, projector = load_vision(ARTIFACT)
    assert tower.patch_embed.proj.weight.dtype == mx.bfloat16
    settings = policy(ARTIFACT)
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    pixels, grids = process_images([image], settings)
    assert grids == [[1, 4, 4]] and pixels.shape == (16, 1536)
    features = tower(mx.array(pixels), grids)
    projected = projector(features)
    mx.eval(features, projected)
    assert features.shape == (4, 4608) and projected.shape == (4, 2560)
    assert features.dtype == mx.bfloat16
    assert bool(mx.all(mx.isfinite(features.astype(mx.float32))))
    assert bool(mx.all(mx.isfinite(projected.astype(mx.float32))))
