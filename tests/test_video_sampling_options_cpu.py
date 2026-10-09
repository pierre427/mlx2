"""Per-part ``fps``/``max_frames`` on input_video must be honoured or refused.

``_responses_content`` accepted both keys on an input_video part, but every
video adapter samples with its own fixed, fingerprinted policy (Gemma 3n 2 fps
/ 64 frames, Gemma 4 2 fps / 32, LFM2.5-VL and the pinned candidates 1 fps /
16).  A request asking for 0.1 fps and at most 2 frames was served 32 frames
at 2 fps with no error or receipt; the Chat Completions validator let the same
keys through.  No adapter honours them, so the route fails closed with
CapabilityUnavailable (501) (sweep 2026-10-08).
"""

from __future__ import annotations

import pytest

from mlx2.api_resources import CapabilityUnavailable
from mlx2.openai_compat import responses_to_chat_request
from mlx2.server import validate_request

VIDEO = "data:video/mp4;base64,eA=="
REQUESTED = {"fps": 0.1, "max_frames": 2}


def _responses_body(**options):
    return {
        "model": "m",
        "input": [{"role": "user", "content": [
            {"type": "input_video", "video_url": VIDEO, **options},
            {"type": "input_text", "text": "What happens?"},
        ]}],
    }


def _chat_body(**options):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": [
            {"type": "input_video", "video_url": VIDEO, **options},
            {"type": "text", "text": "What happens?"},
        ]}],
    }


@pytest.mark.parametrize("options", [REQUESTED, {"fps": 0.5}, {"max_frames": 2}],
                         ids=["both", "fps", "max_frames"])
def test_responses_video_sampling_options_are_refused(options):
    with pytest.raises(CapabilityUnavailable, match="input_video"):
        responses_to_chat_request(_responses_body(**options))


@pytest.mark.parametrize("options", [REQUESTED, {"fps": 0.5}, {"max_frames": 2}],
                         ids=["both", "fps", "max_frames"])
def test_chat_video_sampling_options_are_refused(options):
    with pytest.raises(CapabilityUnavailable, match="input_video"):
        validate_request(_chat_body(**options), chat=True)


def test_video_without_sampling_options_still_translates():
    request, _ = responses_to_chat_request(_responses_body())
    part = request["messages"][0]["content"][0]
    assert part == {"type": "input_video", "video_url": VIDEO}
    validate_request(_chat_body(), chat=True)


# An explicit null requests no sampling value: the base tree accepted it, so it
# is dropped during normalization rather than refused (review round 1).
NULLS = [{"fps": None}, {"max_frames": None}, {"fps": None, "max_frames": None}]
NULL_IDS = ["fps", "max_frames", "both"]


@pytest.mark.parametrize("options", NULLS, ids=NULL_IDS)
def test_responses_null_video_sampling_options_are_dropped(options):
    request, _ = responses_to_chat_request(_responses_body(**options))
    part = request["messages"][0]["content"][0]
    assert part == {"type": "input_video", "video_url": VIDEO}


@pytest.mark.parametrize("options", NULLS, ids=NULL_IDS)
def test_chat_null_video_sampling_options_are_dropped(options):
    body = validate_request(_chat_body(**options), chat=True)
    assert body["messages"][0]["content"][0] == {
        "type": "input_video", "video_url": VIDEO
    }


@pytest.mark.parametrize("options", [{"fps": None, "max_frames": 2},
                                     {"fps": 0.5, "max_frames": None}],
                         ids=["null_fps", "null_max_frames"])
def test_numeric_option_beside_a_null_is_still_refused(options):
    asked = next(key for key, value in options.items() if value is not None)
    with pytest.raises(CapabilityUnavailable, match=f"input_video {asked} is"):
        responses_to_chat_request(_responses_body(**options))
    with pytest.raises(CapabilityUnavailable, match=f"input_video {asked} is"):
        validate_request(_chat_body(**options), chat=True)
