"""Hy4-flash image requests on a DFly2 (external-draft) server: plain decode (CPU).

Pierre 2026-11-20: DFly2 as the default route and images on the same server.
Real ``ServingEngine`` instances over the tiny vision checkpoint of
``test_hy_v4_flash_vision_cpu``: an image request on the DFly2 server must
equal the same request on an ordinary server token for token and carry the
``media_decode`` plain-decode receipt; a text request on the DFly2 server
still drafts.
"""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import memory, serving  # noqa: E402
from mlx2.runtime import os_memory  # noqa: E402
from test_hy_v4_flash_vision_cpu import _png, _vision_artifact, vtiny  # noqa: E402,F401


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory, vtiny):  # noqa: F811
    config, W, VW, _ = vtiny
    return _vision_artifact(tmp_path_factory.mktemp("hy4vserve"), config, W, VW)


def _engine(path, **kw):
    from mlx2.adapters import hy_v4_flash as A

    engine = serving.ServingEngine(str(path), adapter_factory=A.HYV4FlashAdapter,
                                   qualification_mode=True, mtp=False, max_lanes=1,
                                   prefill_step=16, **kw)
    assert engine.ready.wait(120), engine.error
    return engine


def _collect(job):
    tokens, receipt = [], None
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            return {"error": event["error"]}
        if "logprob" in event:
            tokens.append(event["logprob"]["id"])
        if "receipt" in event:
            receipt = event["receipt"]
        if "finish_reason" in event:
            return {"tokens": tokens, "receipt": receipt or getattr(job, "receipt", None)}


IMAGE = {"messages": [{"role": "user", "content": [
    {"type": "text", "text": "t7 t8"},
    {"type": "image_url", "image_url": {"url": _png(56, 40)}},
    {"type": "text", "text": "t9 t10 t11"}]}],
    "max_tokens": 4, "temperature": 0, "logprobs": True}
TEXT = {"messages": [{"role": "user", "content": "t7 t8 t9 t10 t11 t12 t13 t14 t15 t16 t17 t18"}],
        "max_tokens": 6, "temperature": 0, "logprobs": True}


def test_image_on_the_dfly2_server_equals_ordinary_with_a_plain_decode_receipt(host, artifact):
    ordinary = _engine(artifact)
    try:
        assert ordinary.snapshot["settings"]["route"] == "ordinary"
        reference = _collect(ordinary.submit(dict(IMAGE)))
    finally:
        ordinary.close()
    assert "error" not in reference, reference
    dfly = _engine(artifact, execution_policy={"draft_model": "embedded_dfly", "num_draft": 3})
    try:
        assert dfly.snapshot["settings"]["route"] == "external_draft"
        assert dfly.snapshot["media_route"]["media"] == "plain_decode"
        image = _collect(dfly.submit(dict(IMAGE)))
        warm = _collect(dfly.submit(dict(IMAGE)))
        opted_out_job = dfly.submit({**IMAGE, "speculation": "off"})
        opted_out = _collect(opted_out_job)
        text = _collect(dfly.submit(dict(TEXT)))
        from mlx2.api_resources import CapabilityUnavailable

        # A text request's opt-out has no plain path on this route: refused.
        with pytest.raises(CapabilityUnavailable, match="no per-request speculation opt-out"):
            dfly.submit({**TEXT, "speculation": "off"})
        counts = dict(dfly.counts)
    finally:
        dfly.close()
    assert "error" not in image, image
    assert "error" not in text, text
    assert image["tokens"] == reference["tokens"]
    decode = image["receipt"]["media_decode"]
    assert decode["route"] == "external_draft" and decode["reason"] == "media_request"
    assert decode["speculation"] == "disabled_for_media" and decode["decode"] == "ordinary"
    assert decode["media_prefill"] == "isolated_prefill_forward"
    assert decode["media_prefill_inputs"] == ["image_grid_thw", "pixel_values"]
    # A repeat (APCv2 warm, or re-prefilled when no draft sidecar was kept)
    # and an explicit speculation opt-out decode the same plain tokens.
    for other in (warm, opted_out):
        assert "error" not in other, other
        assert other["tokens"] == reference["tokens"]
        assert other["receipt"]["media_decode"]["decode"] == "ordinary"
    assert decode["plain_only_rounds"] == len(image["tokens"])
    request = opted_out["receipt"].get("speculation_request") or {}
    assert request.get("status") == "plain_decode" and request.get("decode") == "ordinary"
    assert counts.get("media_plain_decode_requests") == 3
    assert "media_decode" not in (text["receipt"] or {})
    # The text request on the same server still speculates.
    assert text["receipt"]["route"] == "external_draft"
    speculation = text["receipt"]["speculation"]
    assert speculation["execution"] == "external_draft_verify" and not speculation["ordinary_fallback"]
    assert speculation["proposed"] > 0
    assert image["receipt"]["speculation"]["ordinary_fallback_reason"] == "media_request"
