"""HySparse2 research checkpoints served through mlx2's ordinary machinery (CPU).

A tiny random research checkpoint is saved by the research trainer's own
checkpoint writer and exported by ``scripts/export_hysparse2_serving.py``.
The serving model must reproduce the research model's own cached greedy
decode, and every generic route (APCv2 warm/cold, continuous batching,
prompt lookup with rollback) must emit the ordinary greedy tokens.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest
from hysparse2_fixture import make_export, research_greedy
from route_harness import Detok, Parser, collect, patch_host

from mlx2.adapters.hysparse2 import HySparse2Adapter, inspect_artifact, render_chat
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability

PROMPT = [40, 41, 42, 43, 44, 45, 41, 42, 43, 44, 45, 41, 42, 43, 50, 51, 52, 53, 9]
STEPS = 14


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return make_export(tmp_path_factory.mktemp("hysparse2"))
    finally:
        mx.set_default_device(previous)


@pytest.fixture(scope="module")
def reference(export):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return research_greedy(export, PROMPT, STEPS)
    finally:
        mx.set_default_device(previous)


@pytest.fixture(scope="module")
def copy_case(export, reference):
    """A prompt that already contains the model's own continuation once.

    Prompt lookup then proposes from that earlier occurrence; the longer
    context makes the target accept some drafts and reject others, which
    exercises verification, rollback and the rotating SWA rings (window 3).
    """
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        prompt = PROMPT + reference[0] + PROMPT
        return prompt, research_greedy(export, prompt, 30)[0]
    finally:
        mx.set_default_device(previous)


class FP32Adapter(HySparse2Adapter):
    """The real adapter at FP32, with token-ID I/O for engine comparisons."""

    def __init__(self, path, **kwargs):
        super().__init__(path, execution_policy={"dtype": "float32", **kwargs.get("execution_policy", {})})
        real = self.tokenizer

        class Tokenizer:
            def __getattr__(self, name):
                return getattr(real, name)

            @property
            def detokenizer(self):
                return Detok()

        self.tokenizer = Tokenizer()

    def prompt_tokens(self, request):
        if "tokens" in request:
            return list(request["tokens"])
        return super().prompt_tokens(request)

    def output_parser(self, request):
        return Parser()


def _serving_greedy(model, prompt, steps, *, chunk=5):
    cache = model.make_cache()
    logits = None
    for start in range(0, len(prompt), chunk):
        logits = model(mx.array([prompt[start : start + chunk]]), cache=cache)
    rows, tokens = [logits[0, -1]], []
    for _ in range(steps):
        token = int(mx.argmax(rows[-1]).item())
        tokens.append(token)
        rows.append(model(mx.array([[token]]), cache=cache)[0, -1])
    return tokens, rows, cache


def test_registry_resolves_unqualified_ordinary_export(export):
    resolution = inspect_model(export)
    assert resolution.adapter_type is HySparse2Adapter
    assert resolution.default_route == "ordinary"
    capabilities = resolution.descriptor.capabilities
    assert {Capability.APC_V2, Capability.CONTINUOUS_BATCH, Capability.PROMPT_LOOKUP} <= capabilities
    assert Capability.MTP not in capabilities
    assert resolution.artifact["has_mtp"] is False
    assert resolution.descriptor.metadata["qualification"] == "pending"
    assert resolution.artifact_fingerprint == inspect_artifact(export)["identity"]["fingerprint"]
    with pytest.raises(ValueError, match="no implemented native MTP"):
        resolve_adapter(export, mtp=True)


def test_ast_maps_to_adaptive_prompt_lookup(export):
    from argparse import Namespace

    from mlx2.server import apply_ast_invocation

    args = Namespace(model=str(export), ast=True, ast_drafter=None, ast_topology=None,
                     ast_graft_pld=False, ast_tiles=None, ast_tile_rows=None,
                     draft_model=None, external_draft=False)
    apply_ast_invocation(args, None, inspect_model(export))
    assert args.prompt_lookup is True
    assert not getattr(args, "native_mtp", False)
    assert args.ast_route["tree_tiles"] is False and args.ast_route["drafter"] == "pld"


def test_export_fails_closed_on_changed_inputs(export, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(export, copy)
    state = json.loads((copy / "state.json").read_text())
    state["step"] = 8
    (copy / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="changed since export"):
        inspect_artifact(copy)


def test_chat_uses_training_turn_format_and_rejects_unknown_roles():
    rendered = render_chat([{"role": "user", "content": "Hi"}])
    assert rendered == "user: Hi\n\nassistant:"
    with pytest.raises(ValueError, match="no 'system' role"):
        render_chat([{"role": "system", "content": "x"}, {"role": "user", "content": "Hi"}])


def test_serving_greedy_matches_research_cached_decode(export, reference):
    adapter = HySparse2Adapter(export, execution_policy={"dtype": "float32"})
    try:
        tokens, rows, _ = _serving_greedy(adapter.model, PROMPT, STEPS)
        ref_tokens, ref_rows = reference
        assert tokens == ref_tokens
        worst = max(float(mx.abs(a - b).max()) for a, b in zip(rows, ref_rows))
        assert worst < 1e-4, worst
        # A stateless full forward is the research teacher-forced forward.
        full = adapter.model(mx.array([PROMPT + tokens]))
        teacher, *_ = adapter.model.core(mx.array([PROMPT + tokens]))
        assert float(mx.abs(full - teacher).max()) < 1e-4
        # The tokenizer round-trips text and renders chat without a BOS.
        ids = adapter.prompt_tokens({"messages": [{"role": "user", "content": "def add"}]})
        assert adapter.tokenizer.decode(ids) == "user: def add\n\nassistant:"
    finally:
        adapter.close()


def test_rollback_restores_every_cache_plane(export):
    from mlx2.runtime.models.cache import trim_prompt_cache

    adapter = HySparse2Adapter(export, execution_policy={"dtype": "float32"})
    try:
        model = adapter.model
        tokens, rows, _ = _serving_greedy(model, PROMPT, 6)
        cache = model.make_cache()
        model(mx.array([PROMPT + tokens[:2]]), cache=cache)
        for plane in cache:
            plane.start_speculation(8)
        # Verify three wrong drafts after the committed prefix, then reject them.
        model(mx.array([[tokens[2], 7, 8, 9]]), cache=cache)
        assert trim_prompt_cache(cache, 3) == 3
        for plane in cache:
            plane.stop_speculation()
        replay = model(mx.array([tokens[3:5]]), cache=cache)
        assert float(mx.abs(replay[0, -1] - rows[5]).max()) < 1e-4
        assert int(mx.argmax(replay[0, -1]).item()) == tokens[5]
    finally:
        adapter.close()


def _engine(export, monkeypatch, **kwargs):
    from mlx2.serving import ServingEngine

    patch_host(monkeypatch)
    engine = ServingEngine(
        str(export), adapter_factory=FP32Adapter, qualification_mode=True,
        max_inflight=32, prefill_step=8, **kwargs,
    )
    assert engine.ready.wait(120), engine.error
    if engine.error:
        raise RuntimeError(engine.error)
    return engine


def _request(prompt, steps=STEPS):
    return {"tokens": list(prompt), "max_tokens": steps, "temperature": 0}


def test_engine_ordinary_apcv2_and_batching_emit_research_greedy(export, reference, monkeypatch):
    import threading

    engine = _engine(export, monkeypatch, mtp=False, max_lanes=2)
    try:
        cold = collect(engine.submit(_request(PROMPT)))
        warm = collect(engine.submit(_request(PROMPT)))
        assert "error" not in cold and "error" not in warm, (cold, warm)
        assert cold["tokens"] == reference[0]
        assert warm["tokens"] == reference[0]
        # The warm request restored the committed APCv2 prompt boundary.
        assert cold["receipt"]["cache"] == warm["receipt"]["cache"] == "apcv2"
        assert cold["receipt"]["cached_tokens"] == 0
        assert warm["receipt"]["cached_tokens"] == len(PROMPT) - 1
        assert warm["receipt"]["cache_checkpoint_role"] == "committed_prompt_boundary"
        # Ragged concurrent rows decode in one cohort and match isolation.
        other = PROMPT[3:11]
        alone = collect(engine.submit(_request(other, 9)))
        results = {}

        def run(name, prompt, steps):
            results[name] = collect(engine.submit(_request(prompt, steps)))

        threads = [threading.Thread(target=run, args=("a", PROMPT, STEPS)),
                   threading.Thread(target=run, args=("b", other, 9))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results["a"]["tokens"] == reference[0]
        assert results["b"]["tokens"] == alone["tokens"]
    finally:
        engine.close()


@pytest.mark.parametrize("batched_verify", [False, True])
def test_prompt_lookup_matches_ordinary_greedy(export, copy_case, monkeypatch, batched_verify):
    prompt, expected = copy_case
    engine = _engine(
        export, monkeypatch, mtp=False, prompt_lookup=True, max_lanes=2,
        execution_policy={"prompt_lookup": {
            "num_draft": 4, "ngram_min": 2, "ngram_max": 3, "adaptive": False,
            "batched_verify": batched_verify,
        }},
    )
    try:
        out = collect(engine.submit(_request(prompt, len(expected))))
        assert "error" not in out, out
        assert out["tokens"] == expected
        speculation = out["receipt"]["speculation"]
        # Drafts were verified, some accepted and some rolled back.
        assert speculation["proposed"] > speculation["accepted"] > 0
        assert speculation["verify_path_rounds"] > 0
    finally:
        engine.close()


def test_default_adaptive_prompt_lookup_matches_ordinary_greedy(export, copy_case, monkeypatch):
    from mlx2.server import DEFAULT_PROMPT_LOOKUP_POLICY

    prompt, expected = copy_case
    engine = _engine(
        export, monkeypatch, mtp=False, prompt_lookup=True, max_lanes=1,
        execution_policy={"prompt_lookup": dict(DEFAULT_PROMPT_LOOKUP_POLICY)},
    )
    try:
        out = collect(engine.submit(_request(prompt, len(expected))))
        assert "error" not in out, out
        assert out["tokens"] == expected
    finally:
        engine.close()


def test_ragged_batched_decode_matches_isolated_rows(export):
    """Merged caches of different lengths (left padding) keep per-row math."""
    from mlx2.runtime.generate import _merge_caches

    adapter = HySparse2Adapter(export, execution_policy={"dtype": "float32"})
    try:
        model = adapter.model
        prompts = [PROMPT, PROMPT[4:9], PROMPT[2:]]
        caches, isolated = [], []
        for prompt in prompts:
            tokens, logits, _ = _serving_greedy(model, prompt, 3)
            isolated.append((tokens, logits))
            cache = model.make_cache()
            model(mx.array([prompt]), cache=cache)
            caches.append(cache)
        batch = _merge_caches(caches)
        for step in range(3):
            inputs = mx.array([[tokens[step]] for tokens, _ in isolated])
            logits = model(inputs, cache=batch)
            for row, (_, expected) in enumerate(isolated):
                assert float(mx.abs(logits[row, -1] - expected[step + 1]).max()) < 1e-4
    finally:
        adapter.close()
