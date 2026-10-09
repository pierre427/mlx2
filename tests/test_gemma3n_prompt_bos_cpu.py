"""Gemma 3n processor-built prompts start with exactly one <bos>.

The Gemma 3n chat template already emits ``<bos>``; the processor's tokenizer
added another because prepare did not pass ``add_special_tokens=False``.  Text
prompts encode the rendered template with ``add_special_tokens=False`` (one
<bos>) and the HF reference ``processor.apply_chat_template(tokenize=True)``
also gives one, so media and text-only content-part requests must match.  A
content list with no media part is a text request: it must not carry a
prefill payload, which serving treats as an isolated whole-prompt media
prefill (sweep 2026-10-08).
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import pytest

from mlx2.adapters.mlx_vlm import Gemma3nAdapter, MiniCPMOAdapter
from mlx2.adapters.multimodal import (
    Gemma3nVideoPolicy,
    MediaFeatureCache,
    MiniCPMOExecutionPolicy,
)
from mlx2.multimodal import MediaValue

BOS, IMAGE_ID = 2, 7
IMAGE = "<image_soft_token>"


class _Tokenizer:
    bos_token = "<bos>"
    bos_token_id = BOS
    image_token, image_token_id = IMAGE, IMAGE_ID
    audio_token, audio_token_id = "<audio_soft_token>", 8

    def encode(self, text, add_special_tokens=False):
        ids = [BOS] if add_special_tokens else []
        for piece in re.split(f"(<bos>|{re.escape(IMAGE)})", text):
            if piece == "<bos>":
                ids.append(BOS)
            elif piece == IMAGE:
                ids.append(IMAGE_ID)
            else:
                ids.extend(1000 + ord(char) for char in piece)
        return ids


class _Processor:
    """Gemma-3n-shaped: the template emits <bos>; the tokenizer adds one by default."""

    tokenizer = _Tokenizer()
    feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16_000})()
    chat_template = "gemma3n"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        rendered = "<bos>"
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "\n".join(part["text"] for part in content if part["type"] == "text")
            rendered += f"<start_of_turn>{message['role']}\n{content}<end_of_turn>\n"
        return rendered + "<start_of_turn>model\n"

    def __call__(self, text, images=None, audio=None, add_special_tokens=True, **kwargs):
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        out = {"input_ids": np.array([ids])}
        if images:
            out["pixel_values"] = np.zeros((len(images), 3, 4, 4), dtype=np.float32)
        return out


def _adapter():
    adapter = object.__new__(Gemma3nAdapter)
    adapter.processor = _Processor()
    adapter.tokenizer = adapter.processor.tokenizer
    adapter.identity = {"fingerprint": "artifact", "config": {"vision_soft_tokens_per_image": 1}}
    adapter.video_policy = Gemma3nVideoPolicy()
    adapter.media_feature_cache = MediaFeatureCache()
    return adapter


def _leading_bos(ids):
    return next(index for index, token in enumerate(ids) if token != BOS)


def test_gemma3n_image_prompt_starts_with_one_bos(monkeypatch):
    image = MediaValue("image", "image/png", "a" * 64, 10, object(), {})
    monkeypatch.setattr("mlx2.adapters.mlx_vlm.resolve_media", lambda *a, **k: image)
    adapter = _adapter()
    prepared = adapter.prepare_multimodal_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ]
        }
    )
    assert _leading_bos(prepared["_mlx2_prompt_tokens"]) == 1


def test_gemma3n_text_only_part_list_is_a_text_request():
    adapter = _adapter()
    text = "What is the capital of France?"
    as_list = {"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}
    as_string = {"messages": [{"role": "user", "content": text}]}
    prepared = adapter.prepare_multimodal_request(as_list)
    # No media part: serving must not see a prefill payload (an empty dict is
    # still "not None" and forces an isolated whole-prompt prefill).
    assert prepared.get("_mlx2_prefill_inputs") is None
    assert prepared.get("_mlx2_media_fingerprint") is None
    assert adapter.prompt_tokens(prepared) == adapter.prompt_tokens(as_string)


def test_minicpmo_text_only_part_list_is_a_text_request():
    calls = []

    class Processor:
        tokenizer = None
        chat_template = "x"

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return {
                "input_ids": np.array([[1, 2, 3]]),
                "image_bound": [np.zeros((0, 2), dtype=np.int32)],
                "audio_bounds": [np.zeros((0, 2), dtype=np.int32)],
            }

    adapter = object.__new__(MiniCPMOAdapter)
    adapter.processor = Processor()
    adapter.identity = {"fingerprint": "artifact"}
    adapter.media_policy = MiniCPMOExecutionPolicy(slice_mode=False)
    prepared = adapter.prepare_multimodal_request(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    )
    assert prepared.get("_mlx2_prefill_inputs") is None
    assert prepared.get("_mlx2_media_fingerprint") is None
    assert calls == []


def test_gemma3n_real_processor_matches_hf_reference_bos(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("mlx_vlm")
    found = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/*"
    ))
    if not found:
        pytest.skip("google/gemma-3n-E2B-it snapshot not present")
    from PIL import Image
    from mlx_vlm.utils import load_processor

    from mlx2.runtime.chat_templates import secure_model_chat_templates

    processor = load_processor(Path(found[0]), True)
    secure_model_chat_templates(processor)
    adapter = object.__new__(Gemma3nAdapter)
    adapter.processor = processor
    adapter.identity = {"fingerprint": "artifact", "config": {}}
    adapter.video_policy = Gemma3nVideoPolicy()
    adapter.media_feature_cache = MediaFeatureCache()
    picture = Image.new("RGB", (64, 64), (9, 99, 9))
    image = MediaValue("image", "image/png", "a" * 64, 10, picture, {})
    monkeypatch.setattr("mlx2.adapters.mlx_vlm.resolve_media", lambda *a, **k: image)
    prepared = adapter.prepare_multimodal_request(
        {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
            {"type": "text", "text": "Describe."},
        ]}]}
    )
    bos = processor.tokenizer.bos_token_id
    ids = prepared["_mlx2_prompt_tokens"]
    reference = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": picture}, {"type": "text", "text": "Describe."},
        ]}],
        tokenize=True, return_dict=True, add_generation_prompt=True,
    )["input_ids"]
    reference = np.asarray(reference).reshape(-1).tolist()
    # The HF reference starts with exactly one <bos>; so must the served ids.
    assert reference[0] == bos and reference[1] != bos
    assert ids[:2] == reference[:2]
