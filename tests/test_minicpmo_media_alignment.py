"""MiniCPM-o processor bounds must account for every supplied media input."""

import base64
from io import BytesIO
import wave

import numpy as np
import pytest

from mlx2.adapters.mlx_vlm import MiniCPMOAdapter
from mlx2.adapters.multimodal import MiniCPMOExecutionPolicy


def _adapter(processed):
    class Processor:
        chat_template = "fixture"

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, **kwargs):
            return processed

    adapter = object.__new__(MiniCPMOAdapter)
    adapter.processor = Processor()
    adapter.identity = {"fingerprint": "artifact"}
    adapter.media_policy = MiniCPMOExecutionPolicy(slice_mode=False, chunk_input=False)
    return adapter


def _request(part):
    return {"messages": [{"role": "user", "content": [part, {"type": "text", "text": "describe"}]}]}


def test_image_bound_mismatch_refuses_before_prefill():
    Image = pytest.importorskip("PIL.Image")
    payload = BytesIO()
    Image.new("RGB", (16, 16), "red").save(payload, format="PNG")
    part = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(payload.getvalue()).decode()}}
    processed = {"input_ids": np.array([[1, 2, 3]]), "image_bound": [np.empty((0, 2), dtype=np.int32)], "audio_bounds": [np.empty((0, 2), dtype=np.int32)]}
    with pytest.raises(ValueError, match="image_bound does not match"):
        _adapter(processed).prepare_multimodal_request(_request(part))
    processed["image_bound"] = [np.array([[1, 2]])]
    assert _adapter(processed).prepare_multimodal_request(_request(part))["_mlx2_media_token_end"] == 2


def test_audio_bound_mismatch_refuses_before_prefill():
    payload = BytesIO()
    with wave.open(payload, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * 16)
    part = {"type": "input_audio", "input_audio": {"data": base64.b64encode(payload.getvalue()).decode(), "format": "wav"}}
    processed = {"input_ids": np.array([[1, 2, 3]]), "image_bound": [np.empty((0, 2), dtype=np.int32)], "audio_bounds": [np.empty((0, 2), dtype=np.int32)]}
    with pytest.raises(ValueError, match="audio_bounds does not match"):
        _adapter(processed).prepare_multimodal_request(_request(part))
    processed["audio_bounds"] = [np.array([[1, 2]])]
    assert _adapter(processed).prepare_multimodal_request(_request(part))["_mlx2_media_token_end"] == 2
