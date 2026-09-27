"""CPU-only artifact gates for embedded vision and local VLM candidates."""

import importlib.abc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class _BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import during CPU-only VLM candidate test")
        return None


@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    assert "mlx" not in sys.modules and "mlx.core" not in sys.modules
    monkeypatch.setattr(sys, "meta_path", [_BlockMLX(), *sys.meta_path])


CASES = [
    ("agnes", Path("~/mlx-models/Agnes-3.0-Flash-Preview-MLX-6bit"), 3058),
    ("smolvlm", Path("~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-256M-Video-Instruct/snapshots/067788b187b95ebe7b2e040b3e4299e342e5b8fd"), 471),
    ("qwen2_5_vl", Path("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"), 824),
]


@pytest.mark.parametrize("model_type,path,count", CASES)
def test_complete_local_vision_artifacts(model_type, path, count):
    if not path.is_dir():
        pytest.skip(f"local artifact absent: {path}")
    from mlx2.adapters.pinned_vlm_candidate import descriptor_for, inspect_vision_artifact
    from mlx2.contracts import Capability

    artifact = inspect_vision_artifact(path, expected=model_type)
    descriptor = descriptor_for(model_type)
    assert artifact["tensor_count"] == count
    assert artifact["vision_present"] is True
    assert artifact["qualification"] == "pending"
    assert artifact["selected"] is False
    assert Capability.VISION in descriptor.capabilities
    assert Capability.VIDEO in descriptor.capabilities
    assert Capability.MTP not in descriptor.capabilities
    assert Capability.APC_V2 not in descriptor.capabilities
    assert Capability.PREFIX_REUSE not in descriptor.capabilities
    assert "mlx.core" not in sys.modules


def test_wrong_topology_rejected_before_runtime_import(tmp_path):
    from mlx2.adapters.pinned_vlm_candidate import inspect_vision_artifact

    config = {
        "model_type": "qwen2_5_vl",
        "text_config": {"num_hidden_layers": 36, "hidden_size": 2048},
        "vision_config": {"hidden_size": 1280, "depth": 31},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="vision depth"):
        inspect_vision_artifact(tmp_path, expected="qwen2_5_vl")
    assert "mlx.core" not in sys.modules


def test_qwen_template_retains_vision_frame_markers():
    from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter

    adapter = object.__new__(Qwen25VLCandidateAdapter)
    adapter.processor = SimpleNamespace(
        image_token="<|image_pad|>", video_token="<|video_pad|>"
    )
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe"},
        {"type": "input_image", "image_url": "ignored"},
        {"type": "input_video", "video_url": "ignored"},
    ]}]
    converted = adapter._template_messages(
        messages, ["<|image_pad|>", "<|video_pad|>"],
    )
    assert converted[0]["content"] == [
        {"type": "text", "text": "Describe"},
        {"type": "image"},
        {"type": "video"},
    ]


def test_candidate_mtp_route_fails_closed():
    from mlx2.adapters.agnes_vision import AgnesVisionCandidateAdapter
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter
    from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter

    for adapter in (AgnesVisionCandidateAdapter, SmolVLM2CandidateAdapter,
                    Qwen25VLCandidateAdapter):
        with pytest.raises(ValueError, match="MTP"):
            adapter.profile_name(object(), True)
    assert "mlx.core" not in sys.modules


def test_vision_bridges_are_discoverable_but_not_unqualified_serving_routes():
    from mlx2.adapters.registry import inspect_model, resolve_adapter
    from mlx2.contracts import Capability

    for model_type, path, _count in CASES[1:]:
        if not path.is_dir():
            pytest.skip(f"local artifact absent: {path}")
        result = inspect_model(path)
        assert result.descriptor.model_type == model_type
        assert Capability.APC_V2 in result.descriptor.capabilities
        with pytest.raises(ValueError, match="requires qualification"):
            resolve_adapter(path)
        assert resolve_adapter(path, qualification_mode=True) is result.adapter_type
    assert "mlx.core" not in sys.modules
