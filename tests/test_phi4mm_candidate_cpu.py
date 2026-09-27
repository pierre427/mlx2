"""Static complete-component check with all MLX imports blocked."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from mlx2.adapters.phi4mm_candidate import Phi4MMCandidate, inspect_artifact


PATH = "~/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81"


def test_all_modality_components_and_sidecars_present():
    record = inspect_artifact(PATH)
    assert record["indexed_tensors"] == 2047
    assert record["shards"] == 3
    assert all(record["components"].values())
    assert all(record["sidecars"][part]["weights_bytes"] > 0 for part in ("vision", "speech"))
    assert record["serving_route"] is None
    assert record["implementation"] == "direct-mlx-vlm-candidate"


def test_direct_candidate_routes_image_audio_and_lora_without_mlx(tmp_path):
    image = tmp_path / "image.png"; image.write_bytes(b"mock")
    audio = tmp_path / "audio.wav"; audio.write_bytes(b"mock")
    calls = []
    model = SimpleNamespace(model_type="phi4mm", config=SimpleNamespace(model_type="phi4mm"))
    model.set_modality = lambda **kwargs: calls.append(("mode", kwargs))
    backend = SimpleNamespace(
        load=lambda path, strict: (model, object()),
        apply_chat_template=lambda *args, **kwargs: (calls.append(("template", kwargs)) or "formatted"),
        generate=lambda **kwargs: (calls.append(("generate", kwargs)) or SimpleNamespace(text="ok")),
    )
    candidate = Phi4MMCandidate(PATH, backend=backend)
    output = candidate.generate("describe", images=[image], audios=[audio])
    assert output["text"] == "ok"
    assert output["route_receipt"]["lora_mode"] == "both"
    assert calls[0] == ("mode", {"has_image": True, "has_audio": True})
    assert calls[1][1]["num_images"] == calls[1][1]["num_audios"] == 1
    assert calls[2][1]["image"] == [str(image)]
    assert calls[2][1]["audio"] == [str(audio)]


def test_inspection_does_not_import_real_mlx():
    script = r'''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import attempted")
sys.meta_path.insert(0, Block())
from mlx2.adapters.phi4mm_candidate import inspect_artifact
inspect_artifact(sys.argv[1])
assert "mlx.core" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script, PATH], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
