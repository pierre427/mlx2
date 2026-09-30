"""DeepSeek V4 official N-layout prefill invariants, CPU and no MLX."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from mlx_blocker import block_mlx_imports
from PIL import Image as PILImage

from mlx2.adapters.deepseek_v4_vision_layout import (
    IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_PAD, IMAGE_START,
    MAX_IMAGE_TOKENS, ImagePrefill, build_image_block, expand_image_tokens,
    image_visible, merge_image_embeddings,
    prepare_local_image, visible_window_indices,
)
from mlx2.adapters.deepseek_v4_vision_weights import (
    materialize_vision_weights, plan_vision_weights,
)

MODEL = Path("~/mlx-models/DeepSeek-V4-Flash-Vision-Exp-Q4")
requires_model = pytest.mark.skipif(
    not (MODEL / "model.safetensors.index.json").is_file(),
    reason="optional DeepSeek V4 model fixture is absent",
)


@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    block_mlx_imports(monkeypatch, __name__)


@pytest.mark.parametrize("dimensions", [(512, 512), (1200, 320), (32, 32), (64, 1024)])
def test_image_prefill_budget_and_patch_geometry(tmp_path, dimensions):
    width, height = dimensions
    path = tmp_path / "image.png"
    PILImage.new("RGB", dimensions, color=(255, 0, 0)).save(path)
    prepared = prepare_local_image(path, start_pos=7)
    assert prepared.patches.shape == (prepared.n_vit_h * prepared.n_vit_w, 3, 14, 14)
    assert prepared.patches.dtype == np.float32
    assert len(prepared.types) <= MAX_IMAGE_TOKENS
    assert prepared.types[0] == IMAGE_START  # start 7 is already at compress alignment
    assert prepared.types[-1] == IMAGE_END
    assert (prepared.types == IMAGE).sum() == len(prepared.perm)
    assert sorted(prepared.perm.tolist()) == list(range(len(prepared.perm)))
    assert np.isfinite(prepared.patches).all()


def test_n_layout_interleaves_row_pairs_and_preserves_sentinels():
    types, perm = build_image_block(3, 2, start_pos=0)
    assert types[:4].tolist() == [IMAGE_PAD] * 3 + [IMAGE_START]
    assert types[-1] == IMAGE_END
    assert (types == IMAGE_NEW_LINE).sum() == 3
    assert perm.tolist() == [0, 2, 1, 3, 4, 5]
    assert types.tolist().index(IMAGE_START) % 4 == 3


def test_placeholder_expansion_rejects_missing_media_and_uses_vocab_sentinels(tmp_path):
    path = tmp_path / "image.png"
    PILImage.new("RGB", (70, 70)).save(path)
    with pytest.raises(ValueError, match="placeholder count"):
        expand_image_tokens([10, 11], 11, [], 129280)
    tokens, prepared = expand_image_tokens([10, 11, 12], 11, [path], 129280)
    assert tokens[0] == 10 and tokens[-1] == 12
    assert tokens[1:-1] == (129280 + prepared[0].types).tolist()
    assert prepared[0].start == 1
    assert len(prepared[0].types) <= MAX_IMAGE_TOKENS


def test_source_bound_layout_imports_no_real_mlx():
    script = r'''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import attempted")
sys.meta_path.insert(0, Block())
from mlx2.adapters.deepseek_v4_vision_layout import build_image_block, image_size
assert image_size(512, 512) == (518, 518)
assert build_image_block(3, 2, 0)[0][-1] == 4
assert "mlx.core" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@requires_model
def test_candidate_vit_aligner_names_cover_all_indexed_blocks():
    path = Path(__file__).resolve().parents[1]
    tree = ast.parse((path / "src/mlx2/runtime/models/deepseek_v4_vision.py").read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert {"VisionTower", "Aligner", "VisionComponents", "Block", "Attention"} <= classes
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    for layer in range(32):
        prefix = f"vision.blocks.{layer}."
        for part in ("norm1.weight", "norm2.weight", "attn.wqkv.weight",
                     "attn.wo.weight", "mlp.w1.weight", "mlp.w2.weight"):
            assert prefix + part in index
    assert all(name in index for name in ("vision.patch_embed.proj.weight",
                                          "vision.norm.weight", "aligner.w1.weight",
                                          "aligner.w2.weight"))
    assert "mlx.core" not in sys.modules


def test_fake_aligner_output_merges_before_hyper_connection_and_visibility():
    types, perm = build_image_block(2, 2, start_pos=1)
    image = ImagePrefill(1, np.zeros((36, 3, 14, 14), dtype=np.float32),
                         6, 6, types, perm)
    tokens = np.array([7, *(129280 + types), 8], dtype=np.int64)
    hidden = np.zeros((len(tokens), 3), dtype=np.float32)
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    sentinels = {IMAGE_START: np.full(3, 10), IMAGE_PAD: np.full(3, 20),
                 IMAGE_NEW_LINE: np.full(3, 30), IMAGE_END: np.full(3, 40)}
    merged = merge_image_embeddings(hidden, [image], [features], sentinels)
    assert not hidden.any()  # no in-place mutation of caller state
    np.testing.assert_array_equal(merged[1:1 + len(types)][types == IMAGE], features[perm])
    assert np.all(merged[1:1 + len(types)][types == IMAGE_START] == 10)
    assert np.all(merged[1:1 + len(types)][types == IMAGE_END] == 40)
    left, right = image_visible(tokens, 129280)
    start = 1 + int(np.flatnonzero(types == IMAGE_START)[0])
    end = 1 + int(np.flatnonzero(types == IMAGE_END)[0])
    assert left[start] == 0 and right[start] == end - start
    assert left[end] == end - start and right[end] == 0
    assert left[0] == right[0] == left[-1] == right[-1] == 0
    window = visible_window_indices(len(tokens), 4, left, right)
    assert end in window[start]  # image start can attend through its end
    with pytest.raises(ValueError, match="aligner output"):
        merge_image_embeddings(hidden, [image], [features[:3]], sentinels)


@requires_model
def test_mixed_q4_vision_weight_plan_and_fake_strict_materialization():
    plan = plan_vision_weights(MODEL)
    assert len(plan.tensor_shards) == 529
    assert len(plan.quantized_modules) == 131
    assert {spec["mode"] for spec in plan.quantized_modules.values()} == {"affine"}
    calls = []

    class FakeModel:
        def load_weights(self, pairs, strict):
            calls.append(("load", len(pairs), strict))

    marker = object()
    model, sentinels = materialize_vision_weights(
        plan, model_factory=FakeModel,
        quantize_model=lambda model, modules: calls.append(("quantize", len(modules))),
        read_tensors=lambda path, selected: {name: marker for name in selected},
    )
    assert isinstance(model, FakeModel)
    assert set(sentinels) == {"image_start", "image_end", "image_pad", "image_newline"}
    assert calls == [("quantize", 131), ("load", 525, True)]
    with pytest.raises(ValueError, match="complete plan"):
        materialize_vision_weights(
            plan, model_factory=FakeModel,
            quantize_model=lambda model, modules: None,
            read_tensors=lambda path, selected: {},
        )
    assert "mlx.core" not in sys.modules
