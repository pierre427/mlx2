"""The qualifier's rolling-checkpoint cancel-and-retry probe (CPU).

Rolling prefill checkpoints are on by default, and a default option must be
shown working before a route qualifies.  ``feature_apc_rolling_checkpoints``
needs a cancelled prefill that a retry resumes from; this probe produces one
through the real HTTP handler and a real ServingEngine on tiny hybrid and
KV-only models, and fails closed when no rolling checkpoint is published.
"""

import importlib.util
from pathlib import Path

import pytest

from route_harness import make_adapter, patch_host, tiny_qwen38_mtp
from test_qualify_srpt_forcing import ChatTokens, _client, _serve
from test_rolling_default_on_cpu import _kv_only_model

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qualify_serving_rolling", ROOT / "scripts" / "qualify_serving.py"
)
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


class LongChatTokens(ChatTokens):
    max_context = 16384

    def cache_budget(self, mtp):
        # Admission budgets hybrid checkpoints against a cache projection.
        from test_rolling_prefill_checkpoints import _Budget

        return _Budget()


def _probe(monkeypatch, model, *, mtp, policy=None, refuse_rolling=False):
    patch_host(monkeypatch)
    from mlx2.serving import ServingEngine

    if refuse_rolling:
        original = ServingEngine._publish_checkpoint

        def refusing(self, apc, key, tokens, prompt_cache, **kwargs):
            if kwargs.get("retention_role") == "prefill_rolling":
                return False
            return original(self, apc, key, tokens, prompt_cache, **kwargs)

        monkeypatch.setattr(ServingEngine, "_publish_checkpoint", refusing)
    tiny, vocab = model
    engine = ServingEngine(
        "tiny",
        adapter_factory=make_adapter(tiny, vocab, adapter_mixin=LongChatTokens),
        qualification_mode=True, mtp=mtp, max_lanes=1, prefill_step=512,
        execution_policy=policy,
    )
    assert engine.ready.wait(120), engine.error
    server, thread, url = _serve(engine)
    try:
        get_status, post = _client(url)
        settings = get_status()["settings"]
        return settings, qualify.run_rolling_checkpoint_probe(
            get_status=get_status,
            post=post,
            settings=settings,
            nonce="cpu",
            text_for=lambda words: "a" * words,
            settle_seconds=3.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        engine.close()


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "self_mtp"])
def test_probe_observes_the_default_on_a_hybrid_route(monkeypatch, mtp):
    settings, (passed, evidence) = _probe(monkeypatch, tiny_qwen38_mtp(), mtp=mtp)
    assert settings["apc_rolling_checkpoints_default"] is True
    assert passed, __import__("json").dumps(evidence, default=str)
    assert evidence["selection"] == "default"
    assert evidence["abandoned_at"]["processed"] >= 4096
    assert evidence["retry"]["cached_tokens"] == 4096
    assert evidence["retry"]["cache_checkpoint_role"] == "prefill_rolling"
    delta = evidence["counter_delta"]
    assert delta["published"] >= 1 and delta["rolling_hits"] >= 1


def test_probe_observes_the_default_on_a_kv_route(monkeypatch):
    settings, (passed, evidence) = _probe(monkeypatch, _kv_only_model(), mtp=False)
    assert passed, __import__("json").dumps(evidence, default=str)
    assert evidence["counter_delta"]["cancel_published"] == 1
    assert evidence["retry"]["cached_tokens"] >= 4096


def test_probe_fails_closed_when_nothing_is_published(monkeypatch):
    _settings, (passed, evidence) = _probe(
        monkeypatch, tiny_qwen38_mtp(), mtp=False, refuse_rolling=True
    )
    assert not passed
    assert evidence["failure"] == "no rolling checkpoint was published and reused"
    assert evidence["retry"]["cached_tokens"] < 4096


def test_probe_refuses_unselected_rolling_and_a_short_explicit_context():
    """Only an explicit selection can reach the short-context refusal: the
    engine's default steps aside under the same reach rule
    (test_rolling_default_on_cpu.py), so a default never lands here."""
    calls = []
    passed, evidence = qualify.run_rolling_checkpoint_probe(
        get_status=lambda: calls.append("status") or {},
        post=lambda *a, **k: calls.append("post"),
        settings={"max_context": 65536, "prefill_step": 512},
        nonce="x",
    )
    assert not passed and calls == []
    assert evidence["failure"] == "rolling checkpoints are not selected on this route"
    passed, evidence = qualify.run_rolling_checkpoint_probe(
        get_status=lambda: calls.append("status") or {},
        post=lambda *a, **k: calls.append("post"),
        settings={
            "apc_rolling_checkpoints": {"interval_tokens": 8192},
            "max_context": 8192, "prefill_step": 8192,
        },
        nonce="x",
    )
    assert not passed and calls == []
    assert "cannot hold" in evidence["failure"]
    assert evidence["selection"] == "explicit"
    from mlx2.runtime.state_boundaries import rolling_reuse_prompt_tokens

    assert rolling_reuse_prompt_tokens(
        interval_tokens=8192, prefill_step=8192, max_context=8192
    ) is None


def test_probe_separates_interior_and_rolling_boundaries():
    settings = {"apc_rolling_checkpoints": {"interval_tokens": 4096},
                "apc_interior_checkpoints": {"count": 4},
                "prefill_step": 256, "max_context": 262144}
    assert qualify.rolling_probe_prompt_words(settings) > 4 * 4096
    settings["apc_interior_checkpoints"] = {"count": 0}
    assert qualify.rolling_probe_prompt_words(settings) < 2 * 4096
