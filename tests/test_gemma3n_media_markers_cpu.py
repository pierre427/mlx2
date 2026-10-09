"""Gemma 3n: a literal media placeholder in request text must fail at prepare.

The pinned mlx-vlm Gemma3nProcessor expands *every* occurrence of
``<image_soft_token>`` / ``<audio_soft_token>`` in the rendered prompt,
including text a client typed and string-content system messages.  Prepare
then handed prefill more placeholders than media features, and mlx-vlm's
``merge_multimodal_and_text`` raised ValueError inside the generation worker,
which stopped the whole server (sweep 2026-10-08).  The placeholder count must
match the soft tokens the model will scatter, so the request is a 400.
"""

import base64
import re
import wave
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from mlx2.adapters.mlx_vlm import Gemma3nAdapter
from mlx2.adapters.multimodal import Gemma3nVideoPolicy, MediaFeatureCache

IMAGE, AUDIO = "<image_soft_token>", "<audio_soft_token>"
IMAGE_ID, AUDIO_ID, TEXT_ID = 9, 10, 1
IMAGE_SEQ, AUDIO_SEQ = 4, 3


class _Tokenizer:
    image_token, image_token_id = IMAGE, IMAGE_ID
    audio_token, audio_token_id = AUDIO, AUDIO_ID
    boi_token, eoi_token = "<start_of_image>", "<end_of_image>"
    boa_token, eoa_token = "<start_of_audio>", "<end_of_audio>"


class _Processor:
    """Mirrors the pinned mlx-vlm Gemma3nProcessor placeholder expansion."""

    tokenizer = _Tokenizer()
    chat_template = "stub"
    image_seq_length = IMAGE_SEQ
    audio_seq_length = AUDIO_SEQ
    feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16_000})()

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def __call__(self, text, images=None, audio=None, **kwargs):
        if audio is not None:
            text = text.replace(AUDIO, f"\n\n<start_of_audio>{AUDIO * AUDIO_SEQ}<end_of_audio>\n\n")
        if images is not None:
            text = text.replace(IMAGE, f"\n\n<start_of_image>{IMAGE * IMAGE_SEQ}<end_of_image>\n\n")
        pieces = [piece for piece in re.split(f"({re.escape(IMAGE)}|{re.escape(AUDIO)})", text) if piece]
        ids = [IMAGE_ID if piece == IMAGE else AUDIO_ID if piece == AUDIO else TEXT_ID for piece in pieces]
        types = [1 if token == IMAGE_ID else 3 if token == AUDIO_ID else 0 for token in ids]
        out = {"input_ids": np.array([ids]), "token_type_ids": np.array([types])}
        if images is not None:
            out["pixel_values"] = np.zeros((len(images), 3, 2, 2), np.float32)
        if audio is not None:
            out["input_features"] = np.zeros((len(audio), 4, 2), np.float32)
        return out


def _adapter():
    adapter = object.__new__(Gemma3nAdapter)
    adapter.processor = _Processor()
    adapter.identity = {
        "fingerprint": "artifact",
        "config": {"vision_soft_tokens_per_image": IMAGE_SEQ, "audio_soft_tokens_per_image": AUDIO_SEQ},
    }
    adapter.video_policy = Gemma3nVideoPolicy()
    adapter.media_feature_cache = MediaFeatureCache()
    return adapter


def _image_part():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (9, 9, 9)).save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def _audio_part():
    buffer = BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * 1_600)
    data = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


def _count(prepared, token):
    return sum(1 for value in prepared["_mlx2_prompt_tokens"] if value == token)


def test_gemma3n_prepare_pairs_each_media_input_with_one_placeholder_run():
    prepared = _adapter().prepare_multimodal_request(
        {"messages": [{"role": "user", "content": [
            _image_part(), _image_part(), _audio_part(), {"type": "text", "text": "describe all"},
        ]}]}
    )
    assert _count(prepared, IMAGE_ID) == 2 * IMAGE_SEQ
    assert _count(prepared, AUDIO_ID) == AUDIO_SEQ


@pytest.mark.parametrize(
    ("media", "marker"),
    [(_image_part, IMAGE), (_audio_part, AUDIO)],
    ids=["image", "audio"],
)
def test_gemma3n_refuses_literal_media_placeholder_in_user_text(media, marker):
    # Before the fix prepare returned 2 * seq placeholders for one input;
    # prefill's merge then raised ValueError in the generation worker.
    with pytest.raises(ValueError, match="placeholder"):
        _adapter().prepare_multimodal_request(
            {"messages": [{"role": "user", "content": [
                media(), {"type": "text", "text": f"is this the same as {marker} ?"},
            ]}]}
        )


def test_gemma3n_refuses_literal_image_placeholder_in_string_system_message():
    with pytest.raises(ValueError, match="placeholder"):
        _adapter().prepare_multimodal_request(
            {"messages": [
                {"role": "system", "content": f"Images arrive as {IMAGE} tokens."},
                {"role": "user", "content": [_image_part(), {"type": "text", "text": "describe"}]},
            ]}
        )


def test_gemma3n_refuses_processor_soft_token_drift_from_the_model_config():
    # The model scatters config.vision_soft_tokens_per_image features per
    # image; a processor expanding a different count would fail the same merge.
    adapter = _adapter()
    adapter.identity["config"]["vision_soft_tokens_per_image"] = IMAGE_SEQ + 1
    with pytest.raises(ValueError, match="placeholder"):
        adapter.prepare_multimodal_request(
            {"messages": [{"role": "user", "content": [_image_part()]}]}
        )
