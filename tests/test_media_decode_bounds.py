"""Host-only regressions for media validation before payload allocation."""

import wave
from io import BytesIO

import pytest
from PIL import Image

from mlx2.multimodal import decode_image, decode_wav_audio, pcm_to_float32


def wav_bytes(*, frames=8, channels=1, width=2):
    payload = BytesIO()
    with wave.open(payload, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(8_000)
        stream.writeframes(b"\0" * (frames * channels * width))
    return payload.getvalue()


def test_image_pixel_limit_precedes_decoding(monkeypatch):
    class OversizedImage:
        size = (5_000, 5_000)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def load(self):
            pytest.fail("oversized image was decoded before enforcing max_pixels")

    monkeypatch.setattr(Image, "open", lambda *_: OversizedImage())
    with pytest.raises(ValueError, match="pixel bound"):
        decode_image(b"compressed image", "image/png", max_pixels=1_000_000)


@pytest.mark.parametrize("error", [Image.DecompressionBombError, Image.DecompressionBombWarning])
def test_pillow_bomb_errors_remain_invalid_media(monkeypatch, error):
    def open_image(*_args):
        raise error("unsafe dimensions")

    monkeypatch.setattr(Image, "open", open_image)
    with pytest.raises(ValueError, match="pixel bound"):
        decode_image(b"compressed image", "image/png")


@pytest.mark.parametrize("channels,width", [(1, 1), (1, 2), (2, 2), (2, 3), (1, 4)])
def test_truncated_wav_is_rejected(channels, width):
    with pytest.raises(ValueError, match="truncated"):
        decode_wav_audio(wav_bytes(channels=channels, width=width)[:-1], "audio/wav")


def test_audio_duration_limit_precedes_pcm_read(monkeypatch):
    monkeypatch.setattr(wave.Wave_read, "readframes", lambda *_: pytest.fail("PCM read before duration validation"))
    with pytest.raises(ValueError, match="duration"):
        decode_wav_audio(wav_bytes(frames=8_000), "audio/wav", max_seconds=0.1)


def test_valid_wav_frame_count_matches_decoded_samples():
    media = decode_wav_audio(wav_bytes(frames=8, channels=2), "audio/wav")
    assert len(pcm_to_float32(media)) == media.metadata["frames"] == 8


def test_valid_image_is_normalized_to_rgb():
    payload = BytesIO()
    Image.new("L", (3, 4), color=128).save(payload, format="PNG")
    media = decode_image(payload.getvalue(), "image/png", max_pixels=12)
    assert media.value.mode == "RGB"
    assert media.metadata == {"width": 3, "height": 4}
