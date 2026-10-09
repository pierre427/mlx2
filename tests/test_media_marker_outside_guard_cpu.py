"""A literal media marker anywhere in a media chat is a client error (400).

Gemma 4's guard (8a8e77959) checked only text parts inside content lists, so
a marker in a string-content message (system, assistant, an earlier user turn)
still ran the processor's replacement iterator dry: StopIteration, which the
server answers as 500 "internal server error".  The pinned Qwen2.5-VL
candidate had no guard and its processor raised IndexError for the same input
(sweep 2026-10-08).  Processors pair every marker in the *rendered* prompt with
one media input, so the rendered marker count must equal the media count.
"""

import base64
import io
import secrets
from pathlib import Path

import numpy as np
import pytest

GEMMA4_31B = Path.home() / "mlx-models" / "gemma-4-31B-MLX-8bit"
QWEN25_VL = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
    / "snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
)
IMAGE_ID, VIDEO_ID = 151655, 151656


def _image_part():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (90, 30, 9)).save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def _cases(marker):
    image = _image_part()
    describe = {"type": "text", "text": "Describe."}
    return {
        "system_string": [
            {"role": "system", "content": f"tag {marker}"},
            {"role": "user", "content": [image, describe]},
        ],
        "assistant_string": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": f"I see {marker}"},
            {"role": "user", "content": [image, describe]},
        ],
        "list_text": [
            {"role": "user", "content": [image, {"type": "text", "text": f"same as {marker}?"}]},
        ],
    }


def _render(messages, typed_marker):
    out = ""
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = "".join(
                part["text"] if part["type"] == "text" else typed_marker(part["type"])
                for part in content
            )
        out += f"<turn>{message['role']}\n{content}<end>\n"
    return out + "<turn>assistant\n"


class _Gemma4Processor:
    """The pinned Gemma 4 processor pairs each marker with the next image."""

    image_token, video_token = "<|image|>", "<|video|>"

    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return _render(messages, typed_marker=None)

    def __call__(self, text, images=None, videos=None, fps=None, return_tensors=None):
        self.calls += 1
        pending = iter(images or ())
        for _ in range(text.count(self.image_token)):
            next(pending)  # StopIteration once the images run out
        return {"input_ids": np.array([[1, 5, 2]]),
                "mm_token_type_ids": np.array([[0, 1, 0]]),
                "pixel_values": np.zeros((1, 3, 2, 2), np.float32)}


class _CandidateProcessor:
    """Qwen2.5-VL / SmolVLM2 shaped: typed media parts render to markers,
    and the processor indexes one grid per marker (IndexError past the end)."""

    def __init__(self, image_token, video_token):
        self.image_token, self.video_token = image_token, video_token
        self.calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return _render(messages, lambda kind: {"image": self.image_token,
                                               "video": self.video_token}[kind])

    def __call__(self, text, images=None, videos=None, **kwargs):
        self.calls += 1
        grids = list(images or ())
        for index in range(text.count(self.image_token)):
            grids[index]
        return {"input_ids": np.array([[1, IMAGE_ID, 2]]),
                "pixel_values": np.zeros((1, 4), np.float32)}


def _fake_gemma4():
    from mlx2.adapters.gemma4 import Gemma431BAdapter
    from mlx2.adapters.multimodal import MediaFeatureCache

    adapter = object.__new__(Gemma431BAdapter)
    adapter.processor = _Gemma4Processor()
    adapter.identity = {"fingerprint": "x"}
    adapter.media_feature_cache = MediaFeatureCache()
    return adapter


def _fake_candidate(adapter_type, image_token, video_token):
    adapter = object.__new__(adapter_type)
    adapter.processor = _CandidateProcessor(image_token, video_token)
    adapter.identity = {"config": {"image_token_id": IMAGE_ID, "video_token_id": VIDEO_ID}}
    adapter._media_proof_key = secrets.token_bytes(32)
    return adapter


def _fake_qwen25():
    from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter

    return _fake_candidate(Qwen25VLCandidateAdapter, "<|image_pad|>", "<|video_pad|>")


def _fake_smolvlm2():
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter

    return _fake_candidate(SmolVLM2CandidateAdapter, "<image>", "<video>")


FAKES = {"gemma4": _fake_gemma4, "qwen25vl": _fake_qwen25, "smolvlm2": _fake_smolvlm2}


@pytest.mark.parametrize("family", sorted(FAKES))
@pytest.mark.parametrize("case", ["system_string", "assistant_string", "list_text"])
def test_literal_image_marker_anywhere_is_refused_before_the_processor(family, case):
    adapter = FAKES[family]()
    marker = adapter.processor.image_token
    with pytest.raises(ValueError, match="placeholder"):
        adapter.prepare_multimodal_request({"messages": _cases(marker)[case]})
    assert adapter.processor.calls == 0


@pytest.mark.parametrize("family", sorted(FAKES))
def test_literal_video_marker_on_an_image_request_is_refused(family):
    # Gemma 4 left a bare video token id in the prompt with no video inputs.
    adapter = FAKES[family]()
    marker = adapter.processor.video_token
    with pytest.raises(ValueError, match="placeholder"):
        adapter.prepare_multimodal_request({"messages": _cases(marker)["system_string"]})


@pytest.mark.parametrize("family", sorted(FAKES))
def test_marker_free_media_request_still_prepares(family):
    adapter = FAKES[family]()
    prepared = adapter.prepare_multimodal_request({"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [_image_part(), {"type": "text", "text": "Describe."}]},
    ]})
    assert prepared["_mlx2_prompt_tokens"] and adapter.processor.calls == 1


# ---- the pinned processors, when their files are present locally ----

@pytest.fixture(scope="module")
def gemma4_adapter():
    if not (GEMMA4_31B / "processor_config.json").is_file():
        pytest.skip("Gemma 4 processor files absent")
    import mlx_vlm.models.gemma4  # noqa: F401  registers the pinned processor
    from mlx_vlm.utils import load_processor

    from mlx2.adapters.gemma4 import Gemma431BAdapter
    from mlx2.adapters.multimodal import MediaFeatureCache

    adapter = object.__new__(Gemma431BAdapter)
    adapter.processor = load_processor(GEMMA4_31B)
    adapter.identity = {"fingerprint": "x"}
    adapter.media_feature_cache = MediaFeatureCache()
    return adapter


@pytest.mark.parametrize("marker", ["<|image|>", "<|video|>"])
@pytest.mark.parametrize("case", ["system_string", "assistant_string", "list_text"])
def test_gemma4_processor_literal_marker_anywhere_is_a_client_error(gemma4_adapter, marker, case):
    with pytest.raises(ValueError, match="placeholder"):
        gemma4_adapter.prepare_multimodal_request({"messages": _cases(marker)[case]})


def test_qwen25_processor_literal_marker_is_a_client_error():
    if not (QWEN25_VL / "config.json").is_file():
        pytest.skip("Qwen2.5-VL processor files absent")
    from mlx2.adapters.mlx_vlm_pin import mlx_vlm_runtime
    from mlx2.adapters.pinned_vlm_candidate import SOURCE_REVISION

    if (mlx_vlm_runtime() or {}).get("revision") != SOURCE_REVISION:
        pytest.skip("the Qwen2.5-VL candidate's pinned mlx-vlm is not importable")
    import mlx_vlm.models.qwen2_5_vl  # noqa: F401  registers the pinned processor
    from mlx_vlm.utils import load_processor

    from mlx2.adapters.multimodal import MediaFeatureCache
    from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter, inspect_artifact

    adapter = object.__new__(Qwen25VLCandidateAdapter)
    adapter.identity = inspect_artifact(QWEN25_VL)
    adapter.processor = load_processor(QWEN25_VL)
    adapter._media_proof_key = secrets.token_bytes(32)
    adapter.media_feature_cache = MediaFeatureCache()
    for case in ("system_string", "assistant_string", "list_text"):
        with pytest.raises(ValueError, match="placeholder"):
            adapter.prepare_multimodal_request(
                {"messages": _cases(adapter.processor.image_token)[case]}
            )
