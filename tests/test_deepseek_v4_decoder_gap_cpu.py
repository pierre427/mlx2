"""Pin the image-decoder boundary without importing MLX or loading weights."""

import ast
import json
import subprocess
from pathlib import Path

import pytest

from mlx2.adapters.deepseek_v4_candidate import SOURCE_REVISION, SOURCE_ROOT


MODEL = Path("~/mlx-models/DeepSeek-V4-Flash-Vision-Exp-Q4")
SOURCE = Path(SOURCE_ROOT)

pytestmark = pytest.mark.skipif(
    not all((
        (SOURCE / "mlx_vlm/models/deepseek_v4/language.py").is_file(),
        (SOURCE / "mlx_vlm/models/deepseek_v4/deepseek_v4.py").is_file(),
        (MODEL / "model.safetensors.index.json").is_file(),
    )),
    reason="optional DeepSeek V4 source and model fixtures are absent",
)


def _method(tree, class_name, method_name):
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def test_pinned_decoder_has_embedding_seam_but_no_image_attention_or_gate():
    root = SOURCE
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == SOURCE_REVISION
    language = (root / "mlx_vlm/models/deepseek_v4/language.py").read_text()
    vision_model = (root / "mlx_vlm/models/deepseek_v4/deepseek_v4.py").read_text()
    language_ast = ast.parse(language)
    model_ast = ast.parse(vision_model)

    forward = _method(language_ast, "DeepseekV4Model", "__call__")
    assert "inputs_embeds" in [arg.arg for arg in forward.args.args]
    source = ast.get_source_segment(language, forward)
    assert "self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds" in source
    assert source.index("inputs_embeds") < source.index("mx.broadcast_to(")

    block = _method(language_ast, "DeepseekV4Block", "__call__")
    assert "visible" not in [arg.arg for arg in block.args.args]
    assert "visible" not in ast.get_source_segment(language, block)
    gate = _method(language_ast, "MoEGate", "__init__")
    assert "bias_vl" not in ast.get_source_segment(language, gate)
    assert "bias_vl" not in language

    embeddings = _method(model_ast, "Model", "get_input_embeddings")
    assert "pixel_values" in [arg.arg for arg in embeddings.args.args]
    assert "pixel_values" not in ast.get_source_segment(vision_model, embeddings).split("return", 1)[1]
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    assert sum(name.endswith(".bias_vl") for name in index) == 46
