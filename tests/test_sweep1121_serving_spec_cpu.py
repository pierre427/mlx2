"""Bug sweep 2026-11-21, lane fix-serving-spec (CPU; no checkpoint, no server).

SRV-1 spiral detector chunking, SRV-2 prompt scoring under deterministic tiles,
SRV-3 null scoring fields, SRV-4 exporter self-match, SPEC-1 Maxwell thinking
toggle, SPEC-2 GSM8K seed law, SPEC-3 Kimi expert-cache partial copy.
"""

import errno
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f"sweep1121_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ SRV-1

REASONING = (
    "The shop sells 46 crates at $400 each, so the answer is $18,400. Hmmm, but the wording is "
    "ambiguous: one interpretation counts the returned crates, and maybe the discount applies "
    "first. Alternatively the answer would be 17,940.50 if the fee is taken off; I am not sure. "
    "Suppose the fee is charged per crate, then => 18,354. Perhaps the natural reading is the "
    "first one, unless the clerk rounds, so #### 18400 it is. Maybes and likelihoods aside."
)


def _feed(pieces):
    from mlx2.spiral_conclude import SpiralDetector

    detector = SpiralDetector()
    for count, piece in enumerate(pieces, start=1):
        detector.feed(piece, count)
    return detector


def _chunkings(text):
    import re

    yield "whole", [text]
    yield "characters", list(text)
    yield "tokens", re.findall(r"\s*(?:[A-Za-z]{1,4}|\d{1,2}|[^\sA-Za-z\d])", text)
    yield "words", re.findall(r"\s*\S+", text)
    for size in (3, 7, 50):
        yield f"blocks_of_{size}", [text[i:i + size] for i in range(0, len(text), size)]


def test_spiral_detector_counts_do_not_depend_on_chunking():
    whole = _feed([REASONING])
    values = [value for _, value in whole.candidates]
    assert values == ["18400", "17940.5", "18354", "18400"]
    assert len(whole.hedges) == 9
    for name, pieces in _chunkings(REASONING):
        assert "".join(pieces) == REASONING, name
        streamed = _feed(pieces)
        assert [value for _, value in streamed.candidates] == values, name
        assert len(streamed.hedges) == len(whole.hedges), name
        assert streamed.modal_candidate() == "18400", name


def test_spiral_detector_one_streamed_answer_is_one_candidate():
    pieces = ["The", " answer", " is", " $", "18", ",", "400", ".", " So", " interpret", "ation", "s",
              " hmm", "m"]
    detector = _feed(pieces)
    # One stated answer and two hedges, first seen at the token that started each.
    assert detector.candidates == [(5, "18400")]
    assert detector.hedges == [10, 13]


# ------------------------------------------------------------------ SRV-3

@pytest.mark.parametrize("body", [
    {"prompt": "hi", "echo": None},
    {"prompt": "hi", "prompt_logprobs": None},
    {"prompt": "hi", "echo": None, "prompt_logprobs": None},
    {"prompt": "hi", "echo": False, "prompt_logprobs": None},
])
def test_null_scoring_fields_are_absent_fields(body):
    from mlx2.prompt_logprobs import extract_prompt_scoring
    from mlx2.server import validate_request

    stripped, spec = extract_prompt_scoring(body, chat=False)
    assert spec is None and stripped == {"prompt": "hi"}
    validate_request(stripped, chat=False)
    assert "prompt" in body and len(body) > 1  # the caller's body is not mutated


def test_body_without_scoring_fields_is_returned_unchanged():
    from mlx2.prompt_logprobs import extract_prompt_scoring

    body = {"prompt": "hi", "stop": None}
    assert extract_prompt_scoring(body, chat=False) == (body, None)
    assert extract_prompt_scoring(body, chat=False)[0] is body


# ------------------------------------------------------------------ SRV-2

class _TileSet:
    """The switch ``DetTileSet`` carries: hooks do nothing while ``enabled`` is False."""

    def __init__(self):
        self.enabled = True
        self.hooked_forwards = 0


def _tiled_adapter(tileset, **extra):
    import mlx.core as mx
    from test_prompt_logprobs_cpu import CharTokenizer, tiny_model

    from mlx2.runtime.models.cache import make_prompt_cache

    model = tiny_model()

    def forward(tokens, cache):
        logits = model(tokens, cache=cache)
        if tileset is not None and tileset.enabled:  # what a fact-table write does to the rows
            tileset.hooked_forwards += 1
            logits = logits + 6.0 * (mx.arange(logits.shape[-1]) % 5 == 0)
        return logits

    return SimpleNamespace(
        model=model, tokenizer=CharTokenizer(), identity={},
        prompt_tokens=lambda request: [ord(c) - 97 for c in request["prompt"]],
        prompt_logprobs_binding=lambda: (forward, make_prompt_cache(model)),
        **({} if tileset is None else {"amoeba_det_tiles": tileset}), **extra)


def test_prompt_scoring_bypasses_deterministic_tiles_and_says_so():
    from mlx2.prompt_logprobs import run_prompt_scoring

    tokens = [ord(c) - 97 for c in "the quick brown fox"if c != " "]
    plain, plain_receipt = run_prompt_scoring(_tiled_adapter(None), tokens, top_k=0, chunk_tokens=4)
    tileset = _TileSet()
    scores, receipt = run_prompt_scoring(_tiled_adapter(tileset), tokens, top_k=0, chunk_tokens=4)
    assert tileset.hooked_forwards == 0 and tileset.enabled is True
    assert scores.logprobs == plain.logprobs
    assert receipt["amoeba_tiles"] == {"amoeba_det_tiles": "bypassed"}
    assert "amoeba_tiles" not in plain_receipt


def test_det_tile_switch_is_restored_when_the_forward_raises():
    from mlx2.prompt_logprobs import PromptScoringBinding, _bind_tiles

    tileset = _TileSet()

    def forward(tokens, cache):
        assert tileset.enabled is False
        raise RuntimeError("forward failed")

    binding = _bind_tiles(SimpleNamespace(amoeba_det_tiles=tileset),
                          PromptScoringBinding(forward, None, "adapter_binding"))
    with pytest.raises(RuntimeError):
        binding.forward(None, None)
    assert tileset.enabled is True


def test_receipt_names_tiles_that_stay_in_the_scoring_forward():
    from mlx2.prompt_logprobs import run_prompt_scoring

    adapter = _tiled_adapter(None, amoeba_tiles=object(), amoeba_blocks=None)
    _scores, receipt = run_prompt_scoring(adapter, [1, 2, 3, 4], top_k=0, chunk_tokens=4)
    assert receipt["amoeba_tiles"] == {"amoeba_tiles": "active_in_scoring_forward"}


def test_serving_scoring_task_bypasses_det_tiles_and_records_it():
    from test_prompt_logprobs_serving_cpu import PROMPT, make_engine

    from mlx2.prompt_logprobs import PromptLogprobsPolicy
    from mlx2.server import wait_prompt_scoring

    policy = PromptLogprobsPolicy(enabled=True, chunk_tokens=4)
    results = []
    tileset = _TileSet()
    for tiles in (None, tileset):
        engine, _model = make_engine(adapter=_tiled_adapter(tiles))
        try:
            results.append(wait_prompt_scoring(engine, {"prompt": PROMPT}, SimpleNamespace(top_k=0), policy))
        finally:
            engine.stop_event.set()
    plain, tiled = results
    assert tiled["scores"].logprobs == plain["scores"].logprobs
    assert tileset.hooked_forwards == 0 and tileset.enabled is True
    assert tiled["receipt"]["amoeba_tiles"] == {"amoeba_det_tiles": "bypassed"}
    assert "amoeba_tiles" not in plain["receipt"]


# ------------------------------------------------------------------ SRV-4

@pytest.fixture
def exporter_repo(tmp_path, monkeypatch):
    exporter = _script("export_public")
    repo = tmp_path / "private"
    files = {
        "scripts/export_public.py": (ROOT / "scripts/export_public.py").read_text(),
        "scripts/public_export_withheld.txt": (ROOT / "scripts/public_export_withheld.txt").read_text(),
        "src/mlx2/plain.py": "VALUE = 1\n",
        "src/mlx2/licensed.py": "# SPDX-License-Identifier: " + exporter.CALCULATOR_LICENSE + "\nVALUE = 2\n",
        "src/mlx2/pointer.py": "LICENSE = 'provenance/LICENSE-calculator.txt'\n",
    }
    for name, content in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (repo / "src/mlx2/latin1.py").write_bytes("# caf\xe9\nVALUE = 3\n".encode("latin-1"))
    exporter.run("git", "init", "-q", str(repo))
    exporter.run("git", "add", ".", cwd=repo)
    exporter.run("git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "fixture", cwd=repo)
    monkeypatch.setattr(exporter, "environment_tokens", lambda _repo: ([], []))
    return exporter, repo


def test_exporter_can_export_itself_without_the_calculator_bundle(exporter_repo, tmp_path):
    exporter, repo = exporter_repo
    out = tmp_path / "out"
    exporter.export(repo, "HEAD", out, include=["scripts/export_public.py", "src/mlx2/plain.py"])
    assert exporter.CALCULATOR_LICENSE in (out / "scripts/export_public.py").read_text()


@pytest.mark.parametrize("name", ["src/mlx2/licensed.py", "src/mlx2/pointer.py"])
def test_exporter_still_refuses_licensed_code_without_the_bundle(exporter_repo, tmp_path, name):
    exporter, repo = exporter_repo
    for include in ([name], ["scripts/export_public.py", name]):
        with pytest.raises(ValueError, match="license bundle"):
            exporter.export(repo, "HEAD", tmp_path / "out", include=include)
    assert not (tmp_path / "out").exists()


def test_exporter_refuses_a_non_utf8_payload_with_its_own_error(exporter_repo, tmp_path):
    exporter, repo = exporter_repo
    with pytest.raises(ValueError, match="non-UTF-8 payload"):
        exporter.export(repo, "HEAD", tmp_path / "out", include=["src/mlx2/latin1.py"])


# ------------------------------------------------------------------ SPEC-1

MESSAGES = [{"role": "user", "content": "How many crates fit on the pallet?"}]


def _served(body):
    """The request as the adapter receives it (server.py folds and drops chat_template_kwargs)."""
    from mlx2.reasoning_options import normalize_reasoning_options

    request = normalize_reasoning_options(body)
    request.pop("chat_template_kwargs", None)
    return request


@pytest.mark.parametrize("control", [
    {"enable_thinking": False},
    {"think": False},
    {"chat_template_kwargs": {"enable_thinking": False}},
    {"reasoning_effort": "none"},
])
def test_maxwell_thinking_off_reaches_template_and_parser(control):
    from mlx2.adapters.maxwell_moe import MaxwellMoeAdapter

    request = _served({"messages": MESSAGES, **control})
    assert MaxwellMoeAdapter._template_kwargs(request) == {"enable_thinking": False}
    assert MaxwellMoeAdapter.thinking_enabled(request) is False


def test_maxwell_thinking_default_and_on():
    from mlx2.adapters.maxwell_moe import DESCRIPTOR, MaxwellMoeAdapter
    from mlx2.contracts import Capability

    default = _served({"messages": MESSAGES})
    assert MaxwellMoeAdapter._template_kwargs(default) == {}
    assert MaxwellMoeAdapter.thinking_enabled(default) is True
    on = _served({"messages": MESSAGES, "enable_thinking": True})
    assert MaxwellMoeAdapter._template_kwargs(on) == {"enable_thinking": True}
    assert MaxwellMoeAdapter.thinking_enabled(on) is True
    # A direct caller's chat_template_kwargs still works, and a raw completion never thinks.
    direct = {"messages": MESSAGES, "chat_template_kwargs": {"enable_thinking": False}}
    assert MaxwellMoeAdapter.thinking_enabled(direct) is False
    assert MaxwellMoeAdapter.thinking_enabled({"prompt": "Crates per pallet:"}) is False
    # The route opens a think channel, so serving must not refuse enable_thinking true.
    assert Capability.REASONING in DESCRIPTOR.capabilities


def test_maxwell_template_receives_the_toggle():
    from mlx2.adapters.maxwell_moe import MaxwellMoeAdapter

    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.update(kwargs)
            return "rendered"

    adapter = object.__new__(MaxwellMoeAdapter)
    adapter.tokenizer = Tokenizer()
    assert adapter.render_prompt(_served({"messages": MESSAGES, "reasoning_effort": "none"})) == "rendered"
    assert seen["enable_thinking"] is False


# ------------------------------------------------------------------ SPEC-2

class _SeedClient:
    def __init__(self):
        self.seeds = []

    def call(self, path, payload=None, **_):
        self.seeds.append(payload["seed"])
        return {"choices": [{"message": {"content": "#### 1"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 2}}


def test_gsm8k_sampled_seed_follows_the_test_index():
    gsm = _script("spec_acceptance_gsm8k")
    test = [{"question": f"Question number {i}: what is {i} plus one?", "answer": f"#### {i + 1}"}
            for i in range(60)]
    sampling = {"temperature": 0.7, "top_p": None, "top_k": None, "seed": 100}

    def seeds(picks):
        client = _SeedClient()
        rows = gsm.run_questions(client, "m", "", [{**test[i], "test_index": i} for i in picks],
                                 chat=True, max_tokens=8, sampling=sampling)
        assert [row["test_index"] for row in rows] == list(picks)
        return client.seeds

    assert seeds(range(60))[57] == 157
    assert seeds([57]) == [157]                 # --indices 57
    assert seeds(range(50, 60))[7] == 157       # --offset 50
    assert seeds([58, 3, 57]) == [158, 103, 157]
    assert gsm.SEED_LAW == "seed_plus_test_index"


# ------------------------------------------------------------------ SPEC-3

def test_kimi_expert_cache_leaves_no_partial_copy(tmp_path, monkeypatch):
    from mlx2.adapters.kimi_k31 import KimiK31LayerStream

    root, cache = tmp_path / "root", tmp_path / "cache"
    source = root / "experts" / "layer_03" / "expert_007.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(bytes(range(256)) * 16)
    stream = object.__new__(KimiK31LayerStream)
    stream.root = root
    stream.enable_expert_cache(cache, max_bytes=1 << 20)
    real = shutil.copyfileobj

    def failing(fi, fo, length=0):
        fo.write(fi.read(1024))
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(shutil, "copyfileobj", failing)
    assert stream._cached_expert_path(3, 7) == source
    assert stream.cache_stats["fallbacks"] == 1
    assert [p for p in cache.rglob("*") if p.is_file()] == []
    # The next attempt, with the volume writable again, caches the whole file.
    monkeypatch.setattr(shutil, "copyfileobj", real)
    cached = stream._cached_expert_path(3, 7)
    assert cached != source and cached.read_bytes() == source.read_bytes()
    assert [p.name for p in cache.rglob("*") if p.is_file()] == ["expert_007.safetensors"]
