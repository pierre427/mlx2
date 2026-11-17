"""amoeba_box_registry.donor_record accepts a local artifact directory (the 27B q8 build is not a hub snapshot)."""
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_local_artifact_donor(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    reg = importlib.import_module("amoeba_box_registry")
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["X"], "quantization": {"bits": 8},
                                                       "text_config": {"num_hidden_layers": 64, "hidden_size": 5120,
                                                                       "intermediate_size": 17408}}))
    (tmp_path / "model-00001.safetensors").write_bytes(b"abc")
    d = reg.donor_record(str(tmp_path), "q8", fingerprint="2dd8d304136a")
    assert d["artifact"] == f"{tmp_path.name}@2dd8d304"
    assert d["layers"] == 64 and d["intermediate_size"] == 17408 and d["quantization"] == {"bits": 8}
    assert list(d["weights_sha256"]) == ["model-00001.safetensors"]
