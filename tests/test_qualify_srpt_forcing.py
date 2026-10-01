"""The qualifier's prefill-scheduling forcing load (CPU).

``feature_prefill_scheduling`` is required whenever a route selects SRPT
(the native-MTP default at ``--max-lanes`` >= 4), and it passes only when the
scheduler reorders a prefill.  These tests drive the qualifier's forcing load
through the real HTTP handler and a real ServingEngine on a tiny hybrid
model, on the native-MTP and the ordinary route: with SRPT a short prompt
overtakes the older long one; with the order made FIFO the same load records
no reorder and the check fails closed.
"""

import importlib.util
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.request import Request, urlopen

import pytest

from route_harness import PIECES, make_adapter, patch_host, tiny_qwen38_mtp

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qualify_serving_srpt", ROOT / "scripts" / "qualify_serving.py"
)
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)

SRPT = {"order": "srpt", "max_bypass": 3, "one_slice_contention": True}


class ChatTokens:
    """Render chat content as the harness tokenizer's one-char tokens."""

    max_context = 4096

    def prompt_tokens(self, request):
        if "tokens" in request:
            return list(request["tokens"])
        text = " ".join(m["content"] for m in request["messages"])
        return [PIECES.index(ch) for ch in text if ch in PIECES]


def _serve(engine):
    from mlx2.server import handler_for

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def _client(url):
    def get_status():
        with urlopen(url + "/v1/status", timeout=30) as response:
            return json.load(response)

    def post(body, stream=False):
        body = {"temperature": 0, **body}
        if stream:
            body["stream"] = True
        response = urlopen(
            Request(
                url + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=120,
        )
        if stream:
            return response
        with response:
            return json.load(response)

    return get_status, post


@pytest.fixture(scope="module")
def model():
    return tiny_qwen38_mtp()


def _run_forcing(monkeypatch, model, *, fifo=False, mtp=True, lanes=4,
                 segmented=False):
    patch_host(monkeypatch)
    # Another test's adapter may leave the process-wide segmented toggle set;
    # each case here states its own route instead.
    monkeypatch.delenv("MLX_LM_SEGMENTED_SELF_MTP", raising=False)
    if fifo:
        from mlx2.runtime.adaptive_policy import PrefillOrder

        # Arrival order: the same policy object and counters, never a reorder.
        monkeypatch.setattr(
            PrefillOrder,
            "select",
            lambda self, candidates: min(
                range(len(candidates)), key=lambda i: candidates[i].uid
            ),
        )
    from mlx2.serving import ServingEngine

    tiny, vocab = model
    # A real route's slice (2048-8192) dwarfs the hold and cohort prompts;
    # 64 keeps them single-slice here too, so only the long prompt is
    # multi-slice and the only possible reorder is short over long.
    engine = ServingEngine(
        "tiny",
        adapter_factory=make_adapter(
            tiny, vocab, adapter_mixin=ChatTokens,
            # What a segmented adapter (Flash-Next, Nemotron) declares.
            extra=(
                {"segment_aware_live_tip": True, "segment_aware_cohort_size": lanes}
                if segmented else {"segment_aware_live_tip": False}
            ),
        ),
        qualification_mode=True, mtp=mtp, max_lanes=lanes, max_inflight=32,
        prefill_step=64, execution_policy={"prefill_scheduling": SRPT},
    )
    assert engine.ready.wait(120), engine.error
    server, thread, url = _serve(engine)
    try:
        get_status, post = _client(url)
        settings = get_status()["settings"]
        assert settings["prefill_scheduling"] == SRPT
        return qualify.run_prefill_scheduling_forcing(
            get_status=get_status,
            post=post,
            settings=settings,
            nonce="cpu",
            long_text_for=lambda words: "a" * words,
            # Status republishes counters every second; 3 s covers it.
            settle_seconds=3.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        engine.close()


ROUTES = [
    # (native MTP, segmented live tip, --max-lanes, queued short prompts)
    pytest.param(True, False, 4, 1, id="mtp-4-lanes"),
    # The Flash-Next smoke shape: segmented self-MTP at the default 4 lanes.
    pytest.param(True, True, 4, 1, id="segmented-mtp-4-lanes"),
    pytest.param(True, False, 5, 2, id="mtp-5-lanes"),
    pytest.param(False, False, 5, 2, id="ordinary-5-lanes"),
]


@pytest.mark.parametrize("mtp,segmented,lanes,shorts", ROUTES)
def test_forcing_load_observes_an_srpt_reorder(
    monkeypatch, model, mtp, segmented, lanes, shorts
):
    passed, evidence = _run_forcing(
        monkeypatch, model, mtp=mtp, lanes=lanes, segmented=segmented
    )
    assert passed, evidence
    assert evidence["reorders"] >= 1
    assert evidence["short_prompts"] == shorts
    # The gate held every queued request until the hold lane was released:
    # the cohort's two members, the long prompt and the short ones.
    assert evidence["gate_before_release"]["queue_depth"] == 3 + shorts
    assert evidence["long_multi_slice"] is True
    results = evidence["results"]
    assert all("usage" in r for r in results["cohort"])
    # SRPT served the younger short prompts before the older long one.
    assert all(
        short["finished_after_release_seconds"]
        < results["long"]["finished_after_release_seconds"]
        for short in results["shorts"]
    )


@pytest.mark.parametrize("mtp,segmented,lanes,shorts", ROUTES)
def test_forcing_load_fails_closed_without_a_reorder(
    monkeypatch, model, mtp, segmented, lanes, shorts
):
    passed, evidence = _run_forcing(
        monkeypatch, model, fifo=True, mtp=mtp, lanes=lanes, segmented=segmented
    )
    assert not passed
    assert evidence["reorders"] == 0
    assert evidence["failure"] == "the scheduler did not reorder a queued short prompt"
    # The load itself ran exactly as in the passing case: only the order differs.
    assert evidence["gate_before_release"]["queue_depth"] == 3 + shorts
    results = evidence["results"]
    assert all("usage" in r for r in [*results["cohort"], results["long"], *results["shorts"]])


@pytest.mark.parametrize(
    "mtp,lanes,needed", [(True, 3, 4), (False, 4, 5)], ids=["mtp-3", "ordinary-4"]
)
def test_forcing_load_refuses_a_lane_width_it_cannot_gate(mtp, lanes, needed):
    calls = []
    passed, evidence = qualify.run_prefill_scheduling_forcing(
        get_status=lambda: calls.append("status") or {},
        post=lambda *a, **k: calls.append("post"),
        settings={"max_lanes": lanes, "mtp": mtp, "prefill_scheduling": SRPT},
        nonce="x",
    )
    assert not passed and f"max-lanes >= {needed}" in evidence["failure"]
    assert calls == []


def test_reorders_count_overtakes_and_capped_service_only():
    before = {"bypasses": 1, "bypass_forced": 0, "one_slice_clamps": 0}
    assert qualify.prefill_scheduling_reorders(
        before, {"bypasses": 1, "bypass_forced": 0, "one_slice_clamps": 9}
    ) == 0
    assert qualify.prefill_scheduling_reorders(
        before, {"bypasses": 2, "bypass_forced": 1, "one_slice_clamps": 0}
    ) == 2


def test_long_prompt_is_one_slice_plus_margin_within_context():
    assert qualify.srpt_forcing_long_tokens(
        {"prefill_step": 8192, "max_context": 32768}
    ) == 8192 + qualify.SRPT_FORCING_SLICE_MARGIN
    capped = qualify.srpt_forcing_long_tokens({"prefill_step": 8192, "max_context": 8192})
    assert capped == (8192 - qualify.LONG_CONTEXT_HEADROOM
                      - qualify.SRPT_FORCING_COMPLETION_TOKENS)
