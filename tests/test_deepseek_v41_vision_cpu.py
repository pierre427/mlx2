"""Synthetic CPU image/layout/loading contracts; no checkpoint qualification."""
import base64
import io
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.adapters.deepseek_v41_vision import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    ImageInput,
    SourceVision,
    expand_images,
    image_grid,
    image_token_types,
    prepare_request,
    vision_admission_plan,
    vision_tensor_specs,
)


def config():
    return {'image_token_id': 4, 'text_config': {'hidden_size': 8}, 'vision_config': {
        'num_hidden_layers': 1, 'hidden_size': 8, 'num_attention_heads': 2,
        'intermediate_size': 12, 'patch_size': 2, 'rope_theta': 10000,
        'downsample_ratio': 3, 'max_image_tokens': 16, 'min_pixels': 36, 'max_wh_ratio': None}}


def artifact():
    c = config()
    return {'config': c, 'tensors': {n: {'dtype': 'BF16', 'shape': s, 'size': 2*np.prod(s)}
                                    for n, s in vision_tensor_specs(c).items()}}


def test_v41_layout_has_reading_order_and_no_v4_padding():
    assert image_token_types(2, 3).tolist() == [IMAGE_START, IMAGE, IMAGE, IMAGE, IMAGE_NEW_LINE,
                                               IMAGE, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_END]
    image = Image.new('RGB', (6, 6), (255, 0, 127))
    tokens, prepared = expand_images([1, 4, 2, 4, 3], [image, image], config()['vision_config'], 4)
    assert len(prepared) == 2
    assert prepared[0].start == 1 and prepared[1].start == 6
    assert tokens == [1, 4, 4, 4, 4, 2, 4, 4, 4, 4, 3]
    assert prepared[0].patches.shape == (9, 3, 2, 2)
    assert np.all(prepared[0].patches[:, 0] == 1)
    assert np.all(prepared[0].patches[:, 1] == -1)
    with pytest.raises(ValueError, match='placeholder'):
        expand_images([4], [], config()['vision_config'], 4)


@pytest.mark.parametrize('width,height', [(1, 4096), (4096, 1), (544, 544), (1920, 1080), (17, 23)])
def test_official_grid_bound_including_extreme_aspects(width, height):
    v = dict(config()['vision_config'], patch_size=14, max_image_tokens=1024, min_pixels=544*544)
    gh, gw, h, w = image_grid(width, height, v)
    assert h % 14 == w % 14 == 0 and min(h, w) > 0
    assert gh == (h//14 + 2)//3 and gw == (w//14 + 2)//3
    assert gh*(gw+1)+2 <= 1024


def test_vision_admission_validates_all_shapes_and_storage():
    a = artifact()
    plan = vision_admission_plan(a)
    assert plan['tensor_count'] == 18
    assert plan['reservation_bytes'] > sum(r['size'] for r in a['tensors'].values())
    a['tensors']['image_start']['dtype'] = 'F32'
    with pytest.raises(ValueError, match='precision/shape'):
        vision_admission_plan(a)


def test_failed_vision_upload_fence_retains_target_and_blocks_reload(monkeypatch):
    from mlx2.adapters import deepseek_v41_source_artifact
    from mlx2.adapters.deepseek_v41_mlx_weights import _UPLOAD_QUARANTINE
    a = artifact()
    target = SimpleNamespace(args=SimpleNamespace(hidden_size=8))
    owner = SourceVision(a, target, admitted_bytes=vision_admission_plan(a)['reservation_bytes'])
    monkeypatch.setattr(deepseek_v41_source_artifact, 'read_tensors',
        lambda artifact, names: {n: np.zeros(a['tensors'][n]['shape'], np.uint16) for n in names})
    evaluate, synchronize = mx.eval, mx.synchronize
    def fail(*args, **kwargs):
        raise RuntimeError('injected upload fence failure')
    try:
        monkeypatch.setattr(mx, 'eval', fail)
        with pytest.raises(RuntimeError, match='upload fence'):
            owner.load()
        monkeypatch.setattr(mx, 'synchronize', fail)
        with pytest.raises(RuntimeError, match='upload fence'):
            owner.close()
        assert _UPLOAD_QUARANTINE[id(owner)] is owner and owner.target is target
        assert owner._pending_host is not None and owner._pending_arrays
        with pytest.raises(RuntimeError, match='failed-completion'):
            SourceVision(a, target, admitted_bytes=owner.plan['reservation_bytes'])
    finally:
        monkeypatch.setattr(mx, 'eval', evaluate)
        monkeypatch.setattr(mx, 'synchronize', synchronize)
        owner.close()
    assert id(owner) not in _UPLOAD_QUARANTINE and not owner._pending_arrays


def test_source_vision_loader_and_embedding_merge_use_published_names(monkeypatch):
    from mlx2.adapters import deepseek_v41_source_artifact
    a = artifact()
    rng = np.random.default_rng(27)
    arrays = {}
    for name, row in a['tensors'].items():
        values = rng.normal(0, .05, row['shape']).astype(np.float32)
        if 'norm' in name:
            values.fill(1)
        arrays[name] = (values.view(np.uint32) >> 16).astype(np.uint16)
    monkeypatch.setattr(deepseek_v41_source_artifact, 'read_tensors', lambda artifact, names: {n: arrays[n] for n in names})
    target = SimpleNamespace(args=SimpleNamespace(hidden_size=8),
        weights=SimpleNamespace(embedding=lambda tokens: mx.ones((*tokens.shape, 8))), round=lambda x: x)
    owner = SourceVision(a, target, admitted_bytes=vision_admission_plan(a)['reservation_bytes']).load()
    tokens = mx.array([[1, 4, 4, 4, 4, 2]])
    image = ImageInput(1, np.zeros((4, 3, 2, 2), np.float32), 2, 2, image_token_types(1, 1))
    try:
        h, mask = owner(tokens, [SimpleNamespace(offset=0)], [image])
        mx.eval(h)
        assert h.shape == (1, 6, 8)
        assert mask.tolist() == [False, True, True, True, True, False]
        assert np.array_equal(np.asarray(h[0, 0]), np.ones(8))
        assert np.array_equal(np.asarray(h[0, 1]), np.asarray(owner.sentinels[IMAGE_START].astype(mx.float32)))
        assert np.isfinite(np.asarray(h)).all()
        second = ImageInput(6, image.patches, 2, 2, image.types)
        pair, pair_mask = owner(mx.array([[1, 4, 4, 4, 4, 2, 4, 4, 4, 4, 3]]),
                                [SimpleNamespace(offset=0)], [image, second])
        mx.eval(pair)
        np.testing.assert_array_equal(np.asarray(pair[0, 1:5]), np.asarray(pair[0, 6:10]))
        assert pair_mask.sum() == 8
        with pytest.raises(ValueError, match='empty ordinary'):
            owner(tokens, [SimpleNamespace(offset=1)], [image])
        with pytest.raises(ValueError, match='covered exactly'):
            owner(tokens, [SimpleNamespace(offset=0)], [])
    finally:
        owner.close()


def test_source_vision_and_draft_policy_combinations_fail_closed():
    assert source_policy({'deepseek_vision': 'enabled'})['deepseek_vision'] == 'enabled'
    assert source_policy({'draft_model': 'embedded_dspark', 'num_draft': 3})['num_draft'] == 3
    for policy in ({'deepseek_vision': 'enabled', 'draft_model': 'embedded_dspark'},
                   {'deepseek_vision': 'enabled', 'draft_model': 'prompt_lookup'},
                   {'deepseek_vision': 'enabled', 'deepseek_ced': 'dependency_exact'},
                   {'draft_model': 'embedded_dspark', 'deepseek_ced': 'dependency_exact'}):
        with pytest.raises(ValueError):
            source_policy(policy)


def test_image_request_expands_tokens_and_salts_media_identity():
    image = Image.new('RGB', (6, 6), 'red')
    data = io.BytesIO(); image.save(data, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()
    placeholder = '<｜deepseek_image｜>'
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [1, 4, 2] if text.count(placeholder) == 1 else [])
    adapter = SimpleNamespace(vision_weights=object(), draft_model=None, config=config(), max_context=128,
        identity={'fingerprint': 'a'*64}, tokenizer=tokenizer,
        render_prompt=lambda request: str(request['messages'][0]['content']) + 'assistant-prefix')
    request = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Read this'},
                 {'type': 'image_url', 'image_url': {'url': url}}]}]}
    prepared = prepare_request(adapter, request)
    assert prepared['_mlx2_prompt_tokens'] == [1, 4, 4, 4, 4, 2]
    assert prepared['_mlx2_media_token_end'] == 5
    assert prepared['_mlx2_media_fingerprint']
    adapter.identity = {'fingerprint': 'b'*64}
    assert prepare_request(adapter, request)['_mlx2_media_fingerprint'] != prepared['_mlx2_media_fingerprint']


def test_vision_owner_cannot_reload_after_retirement_or_reallocate_while_loaded():
    target = SimpleNamespace()
    a = artifact()
    with pytest.raises(MemoryError, match="reserved"):
        SourceVision(a, target, admitted_bytes=float(vision_admission_plan(a)['reservation_bytes']))
    owner = SourceVision(a, target, admitted_bytes=vision_admission_plan(a)['reservation_bytes'])
    owner.components = object()
    with pytest.raises(ValueError, match="loaded twice"):
        owner.load()
    owner.components = None
    owner.close()
    with pytest.raises(ValueError, match="after close"):
        owner.load()


def test_vision_geometry_uses_the_bound_downsample_ratio():
    c = config()
    c['vision_config']['downsample_ratio'] = 2
    a = {'config': c, 'tensors': {n: {'dtype': 'BF16', 'shape': shape, 'size': int(2*np.prod(shape))}
                                 for n, shape in vision_tensor_specs(c).items()}}
    target = SimpleNamespace(args=SimpleNamespace(hidden_size=8), round=lambda x: x,
                            weights=SimpleNamespace(embedding=lambda ids: mx.ones((*ids.shape, 8))))
    owner = SourceVision(a, target, admitted_bytes=vision_admission_plan(a)['reservation_bytes'])
    owner.components = lambda patches, height, width: mx.ones((4, 8), mx.bfloat16)
    owner.sentinels = {kind: mx.ones((8,), mx.bfloat16) for kind in (IMAGE_START, IMAGE_END, IMAGE_NEW_LINE)}
    types = image_token_types(2, 2)
    tokens = mx.array([[1] + [4]*len(types) + [2]])
    image = ImageInput(1, np.zeros((9, 3, 2, 2), np.float32), 3, 3, types)
    try:
        hidden, mask = owner(tokens, [SimpleNamespace(offset=0)], [image])
        assert hidden.shape == (1, len(types)+2, 8)
        assert sum(mask) == len(types)
    finally:
        owner.close()


def test_source_media_route_policy_admits_only_selected_ordinary_path():
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    from mlx2.serving import media_route_policy
    adapter = object.__new__(DeepSeekV41SourceAdapter)
    assert media_route_policy(adapter, 'ordinary')['media'] == 'route_native'
    for route in ('external_draft', 'native_mtp', 'prompt_lookup'):
        assert media_route_policy(adapter, route)['media'] == 'refused'


def test_source_vision_configuration_publishes_real_capability_only_after_owner_load(monkeypatch):
    from mlx2.adapters import deepseek_v41_vision as vision
    from mlx2.adapters.deepseek_v41_source import DESCRIPTOR, DeepSeekV41SourceAdapter
    from mlx2.contracts import Capability
    adapter = object.__new__(DeepSeekV41SourceAdapter)
    adapter.model = SimpleNamespace()
    adapter.descriptor = DESCRIPTOR
    adapter.vision_weights = None
    events = []

    class Owner:
        def __init__(self, artifact, target, *, admitted_bytes):
            assert target is adapter.model and admitted_bytes == 123
            events.append('construct')

        def load(self):
            assert adapter.vision_weights is self
            assert Capability.VISION not in adapter.descriptor.capabilities
            assert not hasattr(adapter.model, 'vision_inputs')
            events.append('load')
            return self

    monkeypatch.setattr(vision, 'SourceVision', Owner)
    adapter._configure_source_vision({'fixture': True}, 123)
    assert events == ['construct', 'load']
    assert adapter.model.vision_inputs is adapter.vision_weights
    assert Capability.VISION in adapter.descriptor.capabilities
    with pytest.raises(ValueError, match='existing owner'):
        adapter._configure_source_vision({'fixture': True}, 123)
    assert events == ['construct', 'load']


def test_source_vision_load_failure_retains_owner_without_publishing_capability(monkeypatch):
    from mlx2.adapters import deepseek_v41_vision as vision
    from mlx2.adapters.deepseek_v41_source import DESCRIPTOR, DeepSeekV41SourceAdapter
    from mlx2.contracts import Capability
    adapter = object.__new__(DeepSeekV41SourceAdapter)
    adapter.model = SimpleNamespace()
    adapter.descriptor = DESCRIPTOR
    adapter.vision_weights = None
    class Owner:
        def __init__(self, artifact, target, *, admitted_bytes):
            pass
        def load(self):
            raise RuntimeError('fixture upload failure')
    monkeypatch.setattr(vision, 'SourceVision', Owner)
    with pytest.raises(RuntimeError, match='upload failure'):
        adapter._configure_source_vision({}, 123)
    assert isinstance(adapter.vision_weights, Owner)
    assert Capability.VISION not in adapter.descriptor.capabilities
    assert not hasattr(adapter.model, 'vision_inputs')


@pytest.mark.parametrize('defect', ['missing', 'unknown', 'shape'])
def test_source_vision_refuses_incomplete_or_ambiguous_leaf_mapping(monkeypatch, defect):
    from mlx2.runtime.models import deepseek_v4_vision

    original = deepseek_v4_vision.VisionComponents
    def malformed(args):
        components = original(args)
        if defect == 'missing':
            del components.vision.norm.weight
        elif defect == 'unknown':
            components.vision.norm.unexpected = mx.zeros((1,))
        else:
            components.vision.norm.weight = mx.zeros((args.vision_dim + 1,))
        return components
    monkeypatch.setattr(deepseek_v4_vision, 'VisionComponents', malformed)
    a = artifact()
    owner = SourceVision(a, SimpleNamespace(args=SimpleNamespace(hidden_size=8)),
                         admitted_bytes=vision_admission_plan(a)['reservation_bytes'])
    try:
        with pytest.raises(ValueError, match='vision leaf'):
            owner.load()
    finally:
        owner.close()


def test_source_vision_refuses_uploaded_shape_before_leaf_load(monkeypatch):
    from mlx2.adapters import deepseek_v41_source_artifact

    a = artifact()
    monkeypatch.setattr(deepseek_v41_source_artifact, 'read_tensors',
        lambda artifact, names: {name: np.zeros((1,), np.uint16) for name in names})
    owner = SourceVision(a, SimpleNamespace(args=SimpleNamespace(hidden_size=8)),
                         admitted_bytes=vision_admission_plan(a)['reservation_bytes'])
    try:
        with pytest.raises(ValueError, match='uploaded tensor precision/shape'):
            owner.load()
    finally:
        owner.close()
