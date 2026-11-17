import json

from mlx2 import spiral_conclude as sc


def _events(pieces, finish="stop", content=None):
    lines = []
    for p in pieces:
        lines.append({"choices": [{"delta": {"reasoning_content": p}, "finish_reason": None}]})
    if content is not None:
        lines.append({"choices": [{"delta": {"content": content}, "finish_reason": finish}]})
    out = [f"data: {json.dumps(e)}\n".encode() for e in lines] + [b"data: [DONE]\n"]
    return out


class _Stream(list):
    closed = False

    def close(self):
        self.closed = True


def _count(text):
    return len(text.split())


SPIRAL = ("So the answer is 5. But maybe the problem means each, perhaps not. Alternatively it could be 4, "
          "hmm, the wording is ambiguous; suppose instead the answer would be 5. ")


def test_detector_quiet_on_short_trace():
    d = sc.SpiralDetector()
    assert not d.feed(SPIRAL * 3, 300)
    assert d.fired_at is None


def test_detector_fires_on_long_stalled_hedging():
    d = sc.SpiralDetector()
    fired_at = None
    for i in range(1, 200):
        if d.feed(SPIRAL, i * 30):
            fired_at = i * 30
            break
    assert fired_at is not None and fired_at >= sc.DEFAULTS["min_tokens"]
    rec = d.receipt()
    assert rec["fired_at"] == fired_at and rec["modal_candidate"] == "5"
    assert not d.feed(SPIRAL, fired_at + 30)  # fires once


def test_detector_needs_stall_after_first_candidate():
    d = sc.SpiralDetector()
    # computing for a long time, first candidate only near the end: no stall yet
    assert not d.feed("Let me compute. " * 400, 2400)
    assert not d.feed(SPIRAL * 4, 2500)


def test_earlier_candidates_not_recounted_by_tail_rescan():
    d = sc.SpiralDetector()
    d.feed("so the answer is 7.", 10)
    d.feed(" Then more text.", 20)
    d.feed(" answer is 9", 30)
    assert d.candidates == [(10, "7"), (30, "9")]
    assert sc.norm_number("1,250.50") == "1250.5"


def test_continuation_prompt_ends_with_think_close():
    p = sc.continuation_prompt("<prompt><think>\n", "reasoning")
    assert p == "<prompt><think>\nreasoning" + sc.CONCLUDE_SUFFIX
    assert p.endswith("</think>\n\n")


def test_client_no_fire_returns_stream_unchanged():
    calls = []

    def post(url, body, key, timeout, stream):
        calls.append((url, body))
        return _Stream(_events(["short ", "thought"], content="42"))

    out = sc.chat_with_spiral_conclude("http://h", "m", [{"role": "user", "content": "q"}], "P", _count,
                                       post=post)
    assert out["content"] == "42" and out["reasoning"] == "short thought"
    assert out["receipt"]["fired"] is False and out["receipt"]["qualified"] is False
    assert len(calls) == 1 and calls[0][0] == "http://h/v1/chat/completions"


def test_client_fires_stops_stream_and_concludes():
    calls = []
    stream = _Stream(_events([SPIRAL] * 400, content="never"))

    def post(url, body, key, timeout, stream_flag):
        calls.append((url, body))
        if stream_flag:
            return stream
        return {"choices": [{"text": "#### 5", "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}

    det = sc.SpiralDetector(min_tokens=200, stall_tokens=100)
    out = sc.chat_with_spiral_conclude("http://h/", "m", [], "P<think>", _count, detector=det, post=post)
    assert out["receipt"]["fired"] is True and out["receipt"]["conclude_tokens"] == 3
    assert out["content"] == "#### 5" and stream.closed
    url, body = calls[1]
    assert url == "http://h/v1/completions"
    assert body["prompt"] == "P<think>" + out["reasoning"] + sc.CONCLUDE_SUFFIX
    assert body["temperature"] == 0
    assert out["reasoning"].count(SPIRAL) < 400
