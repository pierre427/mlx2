import base64
import inspect
import json
import struct
import wave
import zlib
from io import BytesIO

import numpy as np
import pytest

from mlx2.adapters.base import AudioOutput, decoder_input_representations
from mlx2.adapters.mlx_vlm import Gemma3nAdapter
from mlx2.adapters.multimodal import (
    Gemma3nVideoPolicy,
    MediaFeatureCache,
    MiniCPMOExecutionPolicy,
    NativeVideoInput,
)
from mlx2.multimodal import (
    MediaValue,
    decode_data_url,
    decode_image,
    decode_video,
    decode_wav_audio,
    pcm_to_float32,
    resolve_media,
    uniform_frame_indices,
)
from mlx2.openai_compat import responses_to_chat_request
from mlx2.runtime.lora import read_lora_config
from mlx2.runtime.tokenizer_utils import TokenizerWrapper
from mlx2.tool_backend import ConfiguredToolBackend


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"alpha": [0, 1], "beta": [2]}[text]


class _Embedding:
    def __call__(self, tokens):
        table = np.array([[1, 0, 1], [1, 2, 1], [0, 3, 4]], dtype=np.float32)
        return table[tokens]


def test_gemma3n_streaming_detokenizer_decodes_sentencepiece_boundaries():
    class Tokenizer:
        def __init__(self):
            self.vocab = {"<eos>": 0, "Hello": 1, "▁world": 2, "!": 3}
            self.chat_template = None

        def get_vocab(self):
            return self.vocab

        def decode(self, tokens, **_kwargs):
            return "".join(
                self.convert_ids_to_tokens(token).replace("▁", " ")
                for token in tokens
            )

        def convert_ids_to_tokens(self, token):
            return next(piece for piece, index in self.vocab.items() if index == token)

    tokenizer = TokenizerWrapper(
        Tokenizer(),
        detokenizer_class=Gemma3nAdapter.streaming_detokenizer_class(),
        eos_token_ids=[0],
    )
    detokenizer = tokenizer.detokenizer
    for token in (1, 2, 3):
        detokenizer.add_token(token)
    detokenizer.finalize()
    assert detokenizer.text == "Hello world!"


def test_decoder_representation_mean_pools_normalizes_and_truncates():
    adapter = type("Adapter", (), {})()
    adapter.tokenizer = _Tokenizer()
    adapter.model = type(
        "Outer", (), {"model": type("Inner", (), {"embed_tokens": _Embedding()})()}
    )()
    vectors, tokens = decoder_input_representations(
        adapter, ["alpha", "beta"], dimensions=2, array_module=np
    )
    assert tokens == 3
    assert vectors[0] == pytest.approx([2**-0.5, 2**-0.5])
    assert vectors[1] == pytest.approx([0.0, 1.0])


def test_media_data_bounds_wav_metadata_and_uniform_video_sampling():
    encoded = "data:image/png;base64," + base64.b64encode(b"png").decode()
    assert decode_data_url(encoded, max_bytes=3) == ("image/png", b"png")
    with pytest.raises(ValueError, match="byte bound"):
        decode_data_url(encoded, max_bytes=2)

    payload = BytesIO()
    with wave.open(payload, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        stream.writeframes(b"\0\0" * 800)
    audio = decode_wav_audio(payload.getvalue(), "audio/wav", max_seconds=1)
    assert audio.metadata["sample_rate"] == 8_000
    assert audio.metadata["duration_seconds"] == pytest.approx(0.1)
    assert uniform_frame_indices(100, 25, fps=2, max_frames=6) == [0, 20, 40, 59, 79, 99]


def test_malformed_wav_chunk_size_is_a_client_error():
    # A chunk whose size runs past the payload makes stdlib wave raise a bare
    # RuntimeError; the request must fail as invalid media, not as a 500.
    payload = bytes.fromhex(
        "52494646c40f000057415645666d74201000470001000200803e000000fa0000"
        "4700100064617461a00f000001020102"
    )
    source = "data:audio/wav;base64," + base64.b64encode(payload).decode()
    with pytest.raises(ValueError, match="failed to decode WAV audio"):
        resolve_media(source, kind="audio")


def _mp4_data_url(tmp_path, width, height, frames):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height)
    )
    if not writer.isOpened():
        pytest.skip("opencv has no mp4v writer")
    for index in range(frames):
        writer.write(np.full((height, width, 3), index * 20, dtype=np.uint8))
    writer.release()
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()


def test_video_frames_are_bounded_like_images_before_they_are_retained(tmp_path):
    # A small, highly compressible clip can decode to frames far larger than
    # any accepted image; every sampled frame is held on the request thread.
    oversized = _mp4_data_url(tmp_path, 4096, 4096 + 16, 1)
    with pytest.raises(ValueError, match="pixel bound"):
        resolve_media(oversized, kind="video")

    source = _mp4_data_url(tmp_path, 64, 48, 8)
    video = resolve_media(source, kind="video", fps=2.0, max_frames=8)
    assert len(video.value) == 8
    assert video.value[0].shape == (48, 64, 3)

    with pytest.raises(ValueError, match="pixel bound"):
        resolve_media(source, kind="video", max_frame_pixels=64 * 48 - 1)
    with pytest.raises(ValueError, match="byte bound"):
        resolve_media(
            source, kind="video", max_frames=8, max_frame_bytes=8 * 64 * 48 * 3 - 1
        )
    assert (
        inspect.signature(decode_video).parameters["max_frame_pixels"].default
        == inspect.signature(decode_image).parameters["max_pixels"].default
    )


def test_video_frame_bounds_hold_when_the_container_declares_no_size(
    tmp_path, monkeypatch
):
    # The decoded frame, not the container header, is the authority.
    cv2 = pytest.importorskip("cv2")
    source = _mp4_data_url(tmp_path, 64, 48, 8)

    real_capture = cv2.VideoCapture

    class SizelessCapture:
        def __init__(self, *args):
            self._capture = real_capture(*args)

        def get(self, prop):
            if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
                return 0.0
            return self._capture.get(prop)

        def __getattr__(self, name):
            return getattr(self._capture, name)

    monkeypatch.setattr(cv2, "VideoCapture", SizelessCapture)
    with pytest.raises(ValueError, match="pixel bound"):
        resolve_media(source, kind="video", max_frame_pixels=64 * 48 - 1)
    with pytest.raises(ValueError, match="byte bound"):
        resolve_media(
            source, kind="video", max_frames=8, max_frame_bytes=8 * 64 * 48 * 3 - 1
        )
    assert len(resolve_media(source, kind="video", max_frames=8).value) == 8


def _png_header_only(width, height):
    """A PNG whose header declares ``width`` x ``height`` but holds no pixels."""

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0" * 16))
        + chunk(b"IEND", b"")
    )


def test_image_pixel_bound_is_enforced_from_the_header_before_decoding():
    from PIL import Image

    buffer = BytesIO()
    Image.new("L", (4, 3), 0).save(buffer, "PNG")
    image = decode_image(buffer.getvalue(), "image/png")
    assert image.metadata == {"width": 4, "height": 3}
    # The pixel data is missing, so only a header-time bound can answer
    # "pixel bound"; decoding first would fail on the truncated stream.
    with pytest.raises(ValueError, match="pixel bound"):
        decode_image(_png_header_only(5000, 5000), "image/png")
    # Past PIL's own decompression-bomb limit Image.open raises an Exception
    # subclass that is neither OSError nor ValueError.
    with pytest.raises(ValueError, match="pixel bound"):
        decode_image(_png_header_only(14000, 14000), "image/png")
    with pytest.raises(ValueError, match="minimum dimension"):
        decode_image(_png_header_only(2, 2000), "image/png")


def test_gemma3n_native_video_keeps_order_timestamps_and_cache_identity():
    frames = (np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8))
    media = MediaValue(
        "video",
        "video/mp4",
        "a" * 64,
        100,
        frames,
        {
            "source_fps": 25.0,
            "sampled_indices": (0, 25),
            "timestamps_seconds": (0.0, 1.0),
        },
    )
    native = NativeVideoInput.from_media(media, Gemma3nVideoPolicy(max_frames=4))
    inputs = native.processor_inputs("What changes?", "<image_soft_token>")
    assert inputs["images"] == list(frames)
    assert "at 0.000s" in inputs["text"] and "at 1.000s" in inputs["text"]
    assert inputs["video_metadata"]["fingerprint"] == native.fingerprint
    assert [len(batch) for batch in native.batches(1)] == [1, 1]


def test_gemma3n_adapter_reports_real_frame_batches(monkeypatch):
    frames = (
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    )
    media = MediaValue(
        "video",
        "video/mp4",
        "b" * 64,
        100,
        frames,
        {
            "source_fps": 2.0,
            "sampled_indices": (0, 1),
            "timestamps_seconds": (0.0, 0.5),
        },
    )
    monkeypatch.setattr(
        "mlx2.adapters.mlx_vlm.resolve_media", lambda *args, **kwargs: media
    )

    class Processor:
        tokenizer = type(
            "Tokenizer", (), {"image_token": "<image>", "audio_token": "<audio>"}
        )()

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, **kwargs):
            assert len(kwargs["images"]) == 2
            return {
                "input_ids": np.array([[1, 2, 3]]),
                "token_type_ids": np.array([[0, 1, 1]]),
            }

    adapter = object.__new__(Gemma3nAdapter)
    adapter.processor = Processor()
    adapter.identity = {"fingerprint": "artifact"}
    adapter.video_policy = Gemma3nVideoPolicy(frame_batch_size=1)
    prepared = adapter.prepare_multimodal_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "video_url": "data:video/mp4;base64,eA=="},
                        {"type": "text", "text": "What changes?"},
                    ],
                }
            ]
        }
    )
    assert prepared["_mlx2_multimodal_stats"] == {
        "gemma3n_video_requests": 1,
        "gemma3n_video_frames": 2,
        "gemma3n_video_frame_batches": 2,
    }


def test_gemma3n_refuses_wav_audio_at_a_rate_its_extractor_does_not_use():
    # pcm_to_float32 does not resample, so a 44.1 kHz WAV labeled as 16 kHz
    # would reach the encoder as a slowed-down clip with wrong features.
    def wav(rate):
        payload = BytesIO()
        with wave.open(payload, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(b"\0\0" * rate)
        return base64.b64encode(payload.getvalue()).decode()

    processed = []

    class Processor:
        tokenizer = type(
            "Tokenizer", (), {"image_token": "<image>", "audio_token": "<audio>"}
        )()
        feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16_000})()

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, **kwargs):
            processed.append(kwargs)
            return {"input_ids": np.array([[1, 2, 3]])}

    adapter = object.__new__(Gemma3nAdapter)
    adapter.processor = Processor()
    adapter.identity = {"fingerprint": "artifact"}
    adapter.video_policy = Gemma3nVideoPolicy()

    def request(rate):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": wav(rate), "format": "wav"},
                        },
                        {"type": "text", "text": "transcribe"},
                    ],
                }
            ]
        }

    for rate in (44_100, 8_000):
        with pytest.raises(ValueError, match="16000 Hz"):
            adapter.prepare_multimodal_request(request(rate))
    assert processed == []

    adapter.prepare_multimodal_request(request(16_000))
    assert processed[0]["sampling_rate"] == 16_000
    assert processed[0]["audio"][0].shape == (16_000,)


def test_minicpmo_refuses_video_parts_instead_of_misaligning_media():
    # MiniCPM-o has no video path.  Skipping the part left the prompt
    # builder one replacement short, so the request died with StopIteration
    # (or, with a later image, put the image marker at the video's slot).
    Image = pytest.importorskip("PIL.Image")
    from mlx2.adapters.mlx_vlm import MiniCPMOAdapter
    from mlx2.api_resources import CapabilityUnavailable

    buffer = BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    processed = []

    class Processor:
        tokenizer = None
        chat_template = "x"

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, **kwargs):
            processed.append(kwargs)
            return {"input_ids": np.array([[1, 2, 3]])}

    adapter = object.__new__(MiniCPMOAdapter)
    adapter.processor = Processor()
    adapter.identity = {"fingerprint": "artifact"}
    adapter.media_policy = MiniCPMOExecutionPolicy(slice_mode=False)
    video = {"type": "input_video", "video_url": "data:video/mp4;base64,AAAA"}
    image = {"type": "image_url", "image_url": {"url": image_url}}
    text = {"type": "text", "text": "describe"}
    for parts in ([video, text], [video, image, text]):
        with pytest.raises(CapabilityUnavailable, match="video"):
            adapter.prepare_multimodal_request(
                {"messages": [{"role": "user", "content": parts}]}
            )
    assert processed == []

    prepared = adapter.prepare_multimodal_request(
        {"messages": [{"role": "user", "content": [image, text]}]}
    )
    assert prepared["messages"][0]["content"] == "<image>\ndescribe"
    assert len(processed[0]["images"]) == 1


def test_minicpmo_policy_uses_slicing_batching_and_audio_chunk_controls():
    Image = pytest.importorskip("PIL.Image")

    policy = MiniCPMOExecutionPolicy.from_config(
        {
            "batch_vision_input": True,
            "vision_batch_size": 2,
            "slice_mode": True,
            "slice_config": {"max_slice_nums": 4, "scale_resolution": 64},
            "audio_chunk_length": 1.0,
            "chunk_input": True,
            "audio_config": {"sampling_rate": 16_000},
        }
    )
    batches = policy.vision_batches([Image.new("RGB", (256, 64))])
    assert len(batches) >= 2
    assert all(1 <= len(batch) <= 2 for batch in batches)

    payload = BytesIO()
    with wave.open(payload, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * 40_000)
    audio = decode_wav_audio(payload.getvalue(), "audio/wav")
    assert pcm_to_float32(audio).shape == (40_000,)
    assert [len(chunk) for chunk in policy.audio_chunks(audio)] == [16_000, 16_000, 8_000]
    assert policy.receipt()["vision_batch_size"] == 2


def test_media_feature_cache_is_bounded_lru_and_reports_reuse():
    class Feature:
        def __init__(self, size):
            self.nbytes = size

    cache = MediaFeatureCache(max_entries=2, max_bytes=10)
    assert cache.put("a", Feature(4))
    assert cache.put("b", Feature(4))
    assert cache.get("a").nbytes == 4
    assert cache.put("c", Feature(4))
    assert cache.get("b") is None
    assert not cache.put("oversize", Feature(11))
    snapshot = cache.snapshot()
    assert snapshot["entries"] == 2
    assert snapshot["hits"] == 1
    assert snapshot["misses"] == 1
    assert snapshot["evictions"] == 1
    assert snapshot["oversize_skips"] == 1
    cache.clear()
    assert cache.snapshot()["entries"] == 0


def test_audio_output_requires_audio_bytes_and_valid_metadata():
    result = AudioOutput(b"RIFF", "audio/wav", sample_rate=24_000, channels=1)
    assert result.sample_rate == 24_000
    with pytest.raises(ValueError, match="nonempty"):
        AudioOutput(b"", "audio/wav")
    with pytest.raises(ValueError, match="audio MIME"):
        AudioOutput(b"data", "application/octet-stream")
    with pytest.raises(ValueError, match="sample_rate"):
        AudioOutput(b"data", "audio/wav", sample_rate=0)


def test_lora_config_requires_explicit_keys_and_zero_dropout(tmp_path):
    (tmp_path / "adapters.safetensors").write_bytes(b"fixture")
    config = {
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": 4,
            "scale": 8,
            "dropout": 0,
            "keys": ["layers.0.self_attn.q_proj"],
        },
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    _, parsed = read_lora_config(tmp_path)
    assert parsed["keys"] == ("layers.0.self_attn.q_proj",)
    config["lora_parameters"]["dropout"] = 0.1
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="dropout"):
        read_lora_config(tmp_path)


def test_allowlisted_mcp_tool_maps_to_function_and_executor_binding():
    backend = ConfiguredToolBackend(
        {"servers": {"docs": {"server_url": "http://127.0.0.1:9999/mcp"}}}
    )

    class Client:
        url = "http://127.0.0.1:9999/mcp"

        def list_tools(self):
            return [
                {
                    "name": "search",
                    "description": "Search docs",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ]

    backend.clients["docs"] = Client()
    request, options = responses_to_chat_request(
        {
            "input": "find it",
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "docs",
                    "server_url": Client.url,
                    "allowed_tools": ["search"],
                    "require_approval": "never",
                }
            ],
        },
        tool_backend=backend,
    )
    assert request["tools"][0]["function"]["name"] == "mcp__docs__search"
    assert "mcp__docs__search" in options["tool_executors"]
