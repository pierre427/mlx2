"""Host-only controls for the offline LFM2.5 DSpark evaluation contract.

generate_candidate_dspark is extracted from source by AST and run against
fakes. Nothing here imports mlx, mlx_lm, mlx_vlm or mlx2, reads model, image or
shard files, or says anything about native DSpark acceptance, state or speed.
Run with --noconftest: tests/conftest.py imports mlx.
"""

import __future__
import ast
import builtins
import importlib
import inspect
import json
import sys
import types
from pathlib import Path as RealPath

import pytest

ROOT = RealPath(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "mlx2" / "adapters" / "lfm25_vl.py"
PROVENANCE = ROOT / "provenance" / "lfm25-dspark-eval-contract.json"
FORBIDDEN = frozenset({"mlx", "mlx_lm", "mlx_vlm", "mlx2"})
DUMMIES = ("mlx_vlm", "mlx_vlm.generate", "mlx_vlm.prompt_utils")
STAGES = ("inspect_artifact", "inspect_dspark_artifact", "source", "load",
          "load_candidate_dspark", "template", "generate")
CONTRACT_KEYS = {"scope", "source_revision", "requested_verification_width",
                 "maximum_proposals_per_round", "temperature", "max_tokens"}
RESULT_KEYS = {"text", "finish_reason", "target_fingerprint", "draft_fingerprint",
               "evaluation_contract", "qualified", "selected"}

_REAL_IMPORT = builtins.__import__
_IMPORT_ATTEMPTS = []


def _extract():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    revisions = [ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and [getattr(t, "id", None) for t in node.targets] == ["SOURCE_REVISION"]]
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == "generate_candidate_dspark"]
    assert len(revisions) == 1 and len(defs) == 1
    code = compile(ast.Module(body=defs, type_ignores=[]), str(SOURCE), "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    return code, revisions[0]


CODE, SOURCE_REVISION = _extract()


class Refused(BaseException):
    """Raised by poisoned fakes; not an Exception, so nothing can swallow it."""


class _MetaPathBlocker:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in FORBIDDEN:
            _IMPORT_ATTEMPTS.append(("meta_path", name))
            raise ImportError(f"meta_path guard blocked native import {name}")
        return None


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.partition(".")[0] in FORBIDDEN:
        _IMPORT_ATTEMPTS.append(("builtin", name))
        if not getattr(sys.modules.get(name), "__mlx2_test_dummy__", False):
            raise ImportError(f"builtin guard blocked native import {name}")
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


@pytest.fixture(autouse=True)
def native_guard(monkeypatch):
    assert not [n for n in sys.modules if n.partition(".")[0] in FORBIDDEN]
    _IMPORT_ATTEMPTS.clear()
    monkeypatch.setattr(sys, "meta_path", [_MetaPathBlocker(), *sys.meta_path])
    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    yield
    _IMPORT_ATTEMPTS.clear()


class Harness:
    """Fakes for every dependency of the extracted function, recorded in order."""

    def __init__(self, monkeypatch, *, poisoned=False, fail_at=None, images=()):
        self.calls = []
        self.poisoned, self.fail_at = poisoned, fail_at
        self.target = {"path": "/fake/target", "fingerprint": "target-fp",
                       "config": {"text_config": {}}}
        self.model = types.SimpleNamespace(config=object())
        self.processor, self.draft = object(), object()
        self.draft_record = {"fingerprint": "draft-fp"}
        self.output = types.SimpleNamespace(text="fake text", finish_reason="length")
        harness, allowed = self, frozenset(images)

        class FakePath:
            def __init__(self, value):
                harness._hit("path", value)
                self.value = value

            def expanduser(self):
                return self

            def is_file(self):
                harness.calls.append(("is_file", self.value))
                return self.value in allowed

        modules = {name: types.ModuleType(name) for name in DUMMIES}
        for module in modules.values():
            module.__mlx2_test_dummy__ = True
        modules["mlx_vlm"].load = self._load
        modules["mlx_vlm.generate"].generate = self._generate
        modules["mlx_vlm.prompt_utils"].apply_chat_template = self._template
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        namespace = {
            "__builtins__": builtins, "__name__": "lfm25_vl_extracted",
            "Path": FakePath, "SOURCE_REVISION": SOURCE_REVISION,
            "inspect_artifact": self._inspect_artifact,
            "inspect_dspark_artifact": self._inspect_dspark_artifact,
            "_require_source_revision": self._source,
            "load_candidate_dspark": self._load_candidate_dspark,
        }
        exec(CODE, namespace)
        self.fn = namespace["generate_candidate_dspark"]

    def _hit(self, stage, *args):
        self.calls.append((stage, *args))
        if self.poisoned:
            raise Refused(stage)
        if stage == self.fail_at:
            raise RuntimeError(f"boom {stage}")

    def _inspect_artifact(self, path):
        self._hit("inspect_artifact", path)
        return self.target

    def _inspect_dspark_artifact(self, path, *, target=None):
        self._hit("inspect_dspark_artifact", path, target)
        return {"fingerprint": "inspect-only", "proposal_length": 9, "verification_width": 10}

    def _source(self):
        self._hit("source")
        return {"revision": SOURCE_REVISION, "source": "/fake/source"}

    def _load(self, path, **kwargs):
        self._hit("load", path, kwargs)
        return self.model, self.processor

    def _load_candidate_dspark(self, path, *, target_artifact, target_model):
        self._hit("load_candidate_dspark", path, target_artifact, target_model)
        return self.draft, self.draft_record

    def _template(self, processor, config, prompt, **kwargs):
        self._hit("template", processor, config, prompt, kwargs)
        return "FORMATTED"

    def _generate(self, model, processor, formatted, **kwargs):
        self._hit("generate", model, processor, formatted, kwargs)
        return self.output

    def stages(self):
        return [call[0] for call in self.calls]

    def generate_kwargs(self):
        (call,) = [c for c in self.calls if c[0] == "generate"]
        return call[4]


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)


@pytest.mark.parametrize("name", ["mlx", "mlx.core", "mlx_lm", "mlx_vlm",
                                  "mlx_vlm.generate", "mlx2", "mlx2.adapters.lfm25_vl"])
def test_guards_block_deliberate_native_imports(name):
    with pytest.raises(ImportError, match="builtin guard"):
        exec(f"import {name}", {"__builtins__": builtins})
    with pytest.raises(ImportError, match="meta_path guard"):
        importlib.import_module(name)
    assert ("builtin", name) in _IMPORT_ATTEMPTS
    assert ("meta_path", name.partition(".")[0]) in _IMPORT_ATTEMPTS


def test_dummies_are_explicit_and_unlisted_submodules_stay_blocked(monkeypatch):
    Harness(monkeypatch)
    for name in DUMMIES:
        assert sys.modules[name].__mlx2_test_dummy__ is True
    with pytest.raises(ImportError, match="builtin guard"):
        exec("import mlx_vlm.models.lfm2", {"__builtins__": builtins})
    with pytest.raises(ImportError, match="builtin guard"):
        exec("import mlx.core", {"__builtins__": builtins})


def test_signature_adds_keyword_only_width_default_ten(monkeypatch):
    parameters = inspect.signature(Harness(monkeypatch).fn).parameters
    width = parameters["verification_width"]
    assert width.kind is inspect.Parameter.KEYWORD_ONLY and width.default == 10
    assert parameters["max_tokens"].default == 128
    assert parameters["image"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(parameters) == ["target_path", "draft_path", "prompt", "image",
                                "max_tokens", "verification_width"]


class _IntSubclass(int):
    pass


@pytest.mark.parametrize("width", [True, False, 0, 1, -1, 11, 12, 64, 2.0, 9.0, 10.0,
                                   2.5, None, "10", b"10", [10], _IntSubclass(8)],
                         ids=repr)
@pytest.mark.parametrize("image", [None, "/poison/image.png"])
def test_bad_width_refuses_before_any_lookup_or_dependency(monkeypatch, width, image):
    harness = Harness(monkeypatch, poisoned=True)
    with pytest.raises(ValueError, match="verification_width must be 2..10"):
        harness.fn("/t", "/d", "describe", image=image, max_tokens=16,
                   verification_width=width)
    assert harness.calls == []
    assert _IMPORT_ATTEMPTS == []


@pytest.mark.parametrize("prompt", ["", "   ", None, 3])
def test_prompt_gate_preserved(monkeypatch, prompt):
    harness = Harness(monkeypatch, poisoned=True)
    with pytest.raises(ValueError, match="prompt must be nonempty"):
        harness.fn("/t", "/d", prompt, image="/poison/image.png", verification_width=8)
    assert harness.calls == [] and _IMPORT_ATTEMPTS == []


@pytest.mark.parametrize("max_tokens", [0, -1, 513, True, 1.0, "8", None])
def test_max_tokens_cap_preserved(monkeypatch, max_tokens):
    harness = Harness(monkeypatch, poisoned=True)
    with pytest.raises(ValueError, match="max_tokens must be 1..512"):
        harness.fn("/t", "/d", "describe", image="/poison/image.png",
                   max_tokens=max_tokens, verification_width=8)
    assert harness.calls == [] and _IMPORT_ATTEMPTS == []


@pytest.mark.parametrize("max_tokens", [1, 512])
def test_max_tokens_bounds_still_accepted(monkeypatch, max_tokens):
    harness = Harness(monkeypatch)
    result = harness.fn("/t", "/d", "describe", max_tokens=max_tokens)
    assert harness.generate_kwargs()["max_tokens"] == max_tokens
    assert result["evaluation_contract"]["max_tokens"] == max_tokens


@pytest.mark.parametrize("width", range(2, 11))
def test_supported_widths_forward_requested_size_and_greedy(monkeypatch, width):
    harness = Harness(monkeypatch)
    result = harness.fn("/t", "/d", "describe", max_tokens=64, verification_width=width)
    kwargs = harness.generate_kwargs()
    assert kwargs == {"image": None, "max_tokens": 64, "verbose": False,
                      "draft_model": harness.draft, "draft_kind": "dflash",
                      "draft_block_size": width, "temperature": 0.0}
    assert type(kwargs["draft_block_size"]) is int
    assert type(kwargs["temperature"]) is float and kwargs["temperature"] == 0
    assert result["evaluation_contract"] == {
        "scope": "requested", "source_revision": SOURCE_REVISION,
        "requested_verification_width": width,
        "maximum_proposals_per_round": width - 1,
        "temperature": 0.0, "max_tokens": 64,
    }


def test_default_width_is_ten_with_nine_proposals(monkeypatch):
    harness = Harness(monkeypatch)
    result = harness.fn("/t", "/d", "describe")
    assert harness.generate_kwargs()["draft_block_size"] == 10
    assert harness.generate_kwargs()["temperature"] == 0.0
    assert harness.generate_kwargs()["max_tokens"] == 128
    contract = result["evaluation_contract"]
    assert contract["requested_verification_width"] == 10
    assert contract["maximum_proposals_per_round"] == 9
    assert contract["max_tokens"] == 128


def test_width_eight_has_seven_proposal_ceiling(monkeypatch):
    harness = Harness(monkeypatch)
    contract = harness.fn("/t", "/d", "describe", verification_width=8)["evaluation_contract"]
    assert harness.generate_kwargs()["draft_block_size"] == 8
    assert contract["requested_verification_width"] == 8
    assert contract["maximum_proposals_per_round"] == 7


def test_result_is_requested_only_and_keeps_existing_fields(monkeypatch):
    harness = Harness(monkeypatch)
    result = harness.fn("/t", "/d", "describe", max_tokens=3, verification_width=10)
    assert set(result) == RESULT_KEYS
    assert set(result["evaluation_contract"]) == CONTRACT_KEYS
    assert result["text"] == "fake text" and result["finish_reason"] == "length"
    assert result["target_fingerprint"] == "target-fp"
    assert result["draft_fingerprint"] == "draft-fp"
    assert result["qualified"] is False and result["selected"] is False
    contract = result["evaluation_contract"]
    assert contract["scope"] == "requested"
    assert contract["source_revision"] == SOURCE_REVISION
    # Budget 3 cannot fill a width-10 round; the contract still says only
    # what was asked for and claims nothing about what ran.
    assert contract["requested_verification_width"] == 10
    assert contract["max_tokens"] == 3
    claims = ("observed", "effective", "actual", "accept", "engage", "confidence",
              "parity", "speed", "latency", "throughput", "qualified", "selected")
    assert not [k for k in _keys(contract) if any(c in k for c in claims)]
    assert [k for k in _keys(result) if k in {"qualified", "selected"}] == ["qualified", "selected"]


def test_strict_load_same_bound_target_draft_and_image_forwarding(monkeypatch):
    harness = Harness(monkeypatch, images={"/fake/image.png"})
    harness.fn("/t", "/d", "describe this", image="/fake/image.png",
               max_tokens=32, verification_width=6)
    assert harness.stages() == ["path", "is_file", *STAGES]
    by_stage = {call[0]: call for call in harness.calls}
    assert by_stage["path"][1] == "/fake/image.png"
    assert by_stage["inspect_artifact"][1] == "/t"
    assert by_stage["inspect_dspark_artifact"][1] == "/d"
    assert by_stage["inspect_dspark_artifact"][2] is harness.target
    assert by_stage["load"][1:] == ("/fake/target", {"lazy": False, "strict": True,
                                                     "trust_remote_code": False})
    _, path, artifact, model = by_stage["load_candidate_dspark"]
    assert path == "/d" and artifact is harness.target and model is harness.model
    _, processor, config, prompt, kwargs = by_stage["template"]
    assert processor is harness.processor and config is harness.model.config
    assert prompt == "describe this" and kwargs == {"num_images": 1}
    _, model, processor, formatted, kwargs = by_stage["generate"]
    assert model is harness.model and processor is harness.processor
    assert formatted == "FORMATTED" and kwargs["draft_model"] is harness.draft
    assert kwargs["image"] == "/fake/image.png" and type(kwargs["image"]) is str
    assert kwargs["draft_block_size"] == 6 and kwargs["temperature"] == 0.0
    assert sorted(set(_IMPORT_ATTEMPTS)) == [("builtin", name) for name in DUMMIES]


def test_text_only_requests_zero_images(monkeypatch):
    harness = Harness(monkeypatch)
    harness.fn("/t", "/d", "describe")
    assert harness.stages() == list(STAGES)
    assert harness.calls[STAGES.index("template")][4] == {"num_images": 0}
    assert harness.generate_kwargs()["image"] is None


def test_missing_image_refuses_before_inspection(monkeypatch):
    harness = Harness(monkeypatch)
    with pytest.raises(ValueError, match="image file is missing"):
        harness.fn("/t", "/d", "describe", image="/fake/missing.png", verification_width=4)
    assert harness.stages() == ["path", "is_file"] and _IMPORT_ATTEMPTS == []


@pytest.mark.parametrize("stage", STAGES)
def test_dependency_errors_propagate_and_stop(monkeypatch, stage):
    harness = Harness(monkeypatch, fail_at=stage)
    with pytest.raises(RuntimeError, match=f"boom {stage}"):
        harness.fn("/t", "/d", "describe", verification_width=7)
    assert harness.stages() == list(STAGES[:STAGES.index(stage) + 1])


def test_provenance_binds_source_revision_and_claims_nothing():
    record = json.loads(PROVENANCE.read_text())
    assert record["local_base_revision"] == "87c1ecfb7a195fdf8bd00c4cf79800dc94ee2c86"
    assert record["dependency_semantics"]["revision"] == SOURCE_REVISION
    assert record["mined_code"] is False
    assert record["dependency_semantics"]["source_copied"] is False
    assert record["qualified"] is False and record["selected"] is False
    assert "REQUESTED" in record["requested_not_observed"]
