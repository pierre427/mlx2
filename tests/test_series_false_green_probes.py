"""CPU-only oracles for the series probes that previously accepted false greens."""

import base64
import importlib.util
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

RUN = Path(__file__).parents[1] / "qualification/runs/series-20260924"


def load(name):
    sys.path.insert(0, str(RUN))
    spec = importlib.util.spec_from_file_location(f"series_false_green_{name}", RUN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PersistenceHTTP:
    def __init__(self, cached_tokens):
        self.events = []
        self.cached_tokens = cached_tokens
        self.resumed = False
        self.restore_polls = 0

    def settled(self):
        after = bool(self.events)
        idle = {"block_bytes": 4 << 20, "parks": int(after), "restores": int(after),
                "resumes": int(after), "restore_failures": 0, "restore_digest_failures": 0,
                "bytes_written": 1024 if after else 0}
        return {"apcv2": {"idle_disk": idle}}

    def chat(self, _prompt, **_kwargs):
        self.events.append("chat")
        if self.resumed:
            assert self.restore_polls >= 2, "chat ran before restore finished"
        return {"status": 200, "body": {
            "choices": [{"message": {"content": "BLOCKS"}}],
            "usage": {"prompt_tokens": 1000, "prompt_tokens_details": {
                "cached_tokens": self.cached_tokens if self.resumed else 0}},
        }}

    def post(self, path, _body):
        if path.endswith("/park"):
            return {"status": 200}
        assert path.endswith("/resume")
        self.resumed = True
        return {"status": 202}

    def get(self, _path):
        if not self.resumed:
            return {"status": 200, "body": {"state": "disk", "covered_tokens": 999}}
        self.restore_polls += 1
        return {"status": 200, "body": {
            "state": "disk" if self.restore_polls == 1 else "resident",
            "covered_tokens": 999,
        }}

    def request(self, method, _path):
        assert method == "DELETE"
        return {"status": 200}


@pytest.mark.parametrize("cached_tokens, expected", [(950, True), (4, False)])
def test_block_persistence_waits_and_requires_substantial_reuse(cached_tokens, expected):
    probe = load("experimental_job")
    http = PersistenceHTTP(cached_tokens)
    result = probe.check_block_persistence(http)
    assert result["engaged"]
    assert result["ok"] is expected
    assert result["evidence"]["required_cached_tokens"] == 900
    assert result["evidence"]["restore_state"] == "resident"
    assert http.events == ["chat", "chat"]


def test_block_persistence_does_not_reask_before_restore(monkeypatch):
    probe = load("experimental_job")

    class StuckRestore(PersistenceHTTP):
        def get(self, path):
            state = super().get(path)
            if self.resumed:
                state["body"]["state"] = "disk"
            return state

    ticks = iter(range(0, 200, 10))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    http = StuckRestore(950)
    result = probe.check_block_persistence(http)
    assert not result["ok"]
    assert result["evidence"]["restore_state"] == "disk"
    assert http.events == ["chat"]


def test_capsule_probe_keeps_per_sample_source_failure_receipt():
    probe = load("experimental_job")
    fallback = {"status": "fallback", "reason": "source_incompatible",
                "source_failure": {"reason": "no_eligible_plane", "plane_reasons": ["bf16_or_fp16_only"]}}

    class HTTP:
        def settled(self):
            return {}

        def chat(self, _prompt, **kwargs):
            if kwargs.get("n") == 3:
                return {"status": 200, "body": {
                    "choices": [{"message": {"content": "answer"}}] * 3,
                    "mlx2": {"samples": [{}, {"cache_capsule": fallback}, {"cache_capsule": fallback}]},
                }}
            return {"status": 200, "body": {"choices": [{"message": {"content": "answer"}}]}}

    result = probe.check_cache_capsules(HTTP())
    assert result["evidence"]["sample_capsule_receipts"] == [None, fallback, fallback]


def test_image_oracle_checks_known_pixels_and_rejects_unrelated_text():
    probe = load("feature_smoke")
    png = base64.b64decode(probe._png_data_url().partition(",")[2])
    assert struct.unpack(">II", png[16:24]) == (32, 32)
    length = struct.unpack(">I", png[33:37])[0]
    raw = zlib.decompress(png[41:41 + length])
    assert raw == (b"\x00" + b"\xff\x00\x00" * 32) * 32

    def reply(text):
        return {"status": 200, "body": {"choices": [{"message": {"content": text},
                                                     "finish_reason": "stop"}]}}

    assert probe._red_image_answer(reply("Red.")).passed
    for text in ("", "blue", "red and blue", "I cannot see an image", "OK"):
        assert not probe._red_image_answer(reply(text)).passed

    class Matrix:
        args = SimpleNamespace(capabilities={"vision"})

        def __init__(self):
            self.http = SimpleNamespace(post=self.post)
            self.results = {}
            self.requests = []

        def post(self, _path, body, **_kwargs):
            self.requests.append(body)
            return reply("blue")

        def check(self, name, function=None, *, applies=True, **_kwargs):
            self.results[name] = function() if applies else None

    matrix = Matrix()
    probe.multimodal_checks(matrix)
    assert not matrix.results["multimodal_image"].passed
    assert matrix.requests[0]["max_tokens"] == 8
    assert "one lowercase color word" in matrix.requests[0]["messages"][0]["content"][1]["text"]
