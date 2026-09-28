"""DeepSeek V4 artifact receipt and fail-closed execution, no real MLX."""

import importlib.abc
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from PIL import Image as PILImage

from mlx2.adapters.deepseek_v4_candidate import DeepSeekV4Candidate, inspect_artifact
from mlx2.adapters._direct_mlx_vlm import load_backend


PATH = "~/mlx-models/DeepSeek-V4-Flash-Vision-Exp-Q4"
ARTIFACT = Path(PATH)
SOURCE_TREE = Path("~/Desktop/mlx-uag/worktrees/agnes-vlm-support")
requires_artifact = pytest.mark.skipif(
    not (ARTIFACT / "model.safetensors.index.json").is_file(),
    reason="optional DeepSeek V4 model fixture is absent",
)


@requires_artifact
def test_complete_local_index_is_inspectable_but_unserved():
    record = inspect_artifact(PATH)
    assert record["indexed_tensors"] == 73677
    assert record["shards"] == 48
    assert all(record["components"].values())
    assert record["mtp_layers"] == [0, 1, 2]
    assert record["components"]["vision_tensors"] == 517
    assert record["serving_route"] is None
    assert record["implementation"] == "direct-mlx-vlm-text-candidate"


@requires_artifact
def test_direct_text_candidate_rejects_embedded_media_and_mtp(tmp_path):
    calls = []
    model = SimpleNamespace(model_type="deepseek_v4", config=SimpleNamespace(model_type="deepseek_v4"))
    backend = SimpleNamespace(
        load=lambda path, strict: (model, object()),
        apply_chat_template=lambda *args, **kwargs: "formatted",
        generate=lambda **kwargs: (calls.append(kwargs) or SimpleNamespace(text="ok")),
    )
    candidate = DeepSeekV4Candidate(PATH, backend=backend)
    assert candidate.generate("hello")["text"] == "ok"
    assert candidate.generate("hello")["route_receipt"]["mtp"] is False
    with pytest.raises(NotImplementedError, match="vision and MTP"):
        candidate.generate("hello", image="image.png")
    with pytest.raises(NotImplementedError, match="vision and MTP"):
        candidate.generate("hello", mtp=True)
    assert len(calls) == 2
    image = tmp_path / "image.png"
    PILImage.new("RGB", (70, 70)).save(image)
    expanded, prepared = candidate.prepare_vision_prefill([3, 99, 4], 99, [image])
    assert expanded[0] == 3 and expanded[-1] == 4
    assert len(prepared) == 1
    assert expanded[1:-1] == (129280 + prepared[0].types).tolist()
    sentinel_vectors = {kind: np.full(3, kind, dtype=np.float32) for kind in (0, 1, 3, 4)}
    aligned = [np.ones((len(prepared[0].perm), 3), dtype=np.float32)]
    merged, visible = candidate.assemble_vision_prefill(
        np.zeros((len(expanded), 3), dtype=np.float32), prepared,
        aligned, sentinel_vectors, expanded,
    )
    assert merged.shape == (len(expanded), 3)
    assert visible[0].shape == visible[1].shape == (len(expanded),)


@requires_artifact
def test_inspection_blocks_mlx_import():
    script = r'''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import attempted")
sys.meta_path.insert(0, Block())
from mlx2.adapters.deepseek_v4_candidate import inspect_artifact
inspect_artifact(sys.argv[1])
assert "mlx.core" not in sys.modules
'''
    proc = subprocess.run([sys.executable, "-c", script, PATH], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(
    not (SOURCE_TREE / "mlx_vlm/models/deepseek_v4").is_dir(),
    reason="optional DeepSeek V4 source fixture is absent",
)
def test_source_revision_guard_stops_before_mlx_import():
    with pytest.raises(RuntimeError, match="revision mismatch"):
        load_backend("~/Desktop/mlx-uag/worktrees/agnes-vlm-support",
                     "0" * 40, ("mlx_vlm/models/deepseek_v4",))
    assert "mlx.core" not in sys.modules
