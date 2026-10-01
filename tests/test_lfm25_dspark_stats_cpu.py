"""Host-only controls for the offline LFM2.5 DSpark per-request counter diagnostic.

generate_candidate_dspark is extracted from source by AST and run against
fakes, including a fake private drafter whose lifetime counters are driven by a
test-local recorder modelled on the pinned source semantics.  Nothing here
imports mlx, mlx_lm, mlx_vlm or mlx2, reads model, image or shard files, or
says anything about native DSpark acceptance, rollback, state or speed.
Run with --noconftest: tests/conftest.py imports mlx.
"""

import __future__
import ast
import builtins
import importlib
import importlib.util
import inspect
import json
import math
import sys
import types
from pathlib import Path as RealPath

import pytest

ROOT = RealPath(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "mlx2" / "adapters" / "lfm25_vl.py"
PROVENANCE = ROOT / "provenance" / "lfm25-dspark-stats.json"
FORBIDDEN = frozenset({"mlx", "mlx_lm", "mlx_vlm", "mlx2"})
DUMMIES = ("mlx_vlm", "mlx_vlm.generate", "mlx_vlm.prompt_utils")
COUNTERS = ("speculative_total_rounds", "speculative_total_accepted",
            "speculative_total_drafted")
ROUNDS, ACCEPTED, DRAFTED = COUNTERS
CONTRACT_KEYS = {"scope", "source_revision", "requested_verification_width",
                 "maximum_proposals_per_round", "temperature", "max_tokens"}
RESULT_KEYS = {"text", "finish_reason", "target_fingerprint", "draft_fingerprint",
               "evaluation_contract", "qualified", "selected"}
STATS_KEYS = {"scope", "attribution", "source_revision", "unit", "available",
              "engaged", "unavailable_reason", "baseline_kind", "rounds",
              "accepted_proposals", "drafted_proposals", "qualified", "selected"}
DELETE = object()

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


def _native_modules():
    return [name for name, module in sys.modules.items()
            if name.partition(".")[0] in FORBIDDEN
            and not getattr(module, "__mlx2_test_dummy__", False)]


@pytest.fixture(autouse=True)
def native_guard(monkeypatch):
    assert not _native_modules()
    _IMPORT_ATTEMPTS.clear()
    monkeypatch.setattr(sys, "meta_path", [_MetaPathBlocker(), *sys.meta_path])
    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    yield
    assert not _native_modules()
    assert all(kind == "builtin" and name in DUMMIES for kind, name in _IMPORT_ATTEMPTS)
    _IMPORT_ATTEMPTS.clear()


class Drafter:
    """Private-drafter fake: counters live in a store; reads are logged, writes refused."""

    def __init__(self, store=None, *, poisoned=False, raise_on=None, refuse_on=None):
        object.__setattr__(self, "_state", {
            "store": dict(store or {}), "reads": [], "poisoned": poisoned,
            "raise_on": raise_on, "refuse_on": refuse_on,
        })

    def __getattribute__(self, name):
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        state = object.__getattribute__(self, "_state")
        state["reads"].append(name)
        if state["poisoned"] or name == state["refuse_on"]:
            raise Refused(f"drafter read {name}")
        if name == state["raise_on"]:
            raise RuntimeError(f"drafter property {name} failed")
        try:
            return state["store"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise Refused(f"foreign drafter write {name}")

    def __delattr__(self, name):
        raise Refused(f"foreign drafter delete {name}")


def _state(drafter):
    return object.__getattribute__(drafter, "_state")


def record_round(drafter, accepted, proposals):
    """Test-local model of the pinned recorder: absent counters start at zero."""
    store = _state(drafter)["store"]
    store.setdefault("accept_lens", []).append(accepted)
    store[ROUNDS] = store.get(ROUNDS, 0) + 1
    store[ACCEPTED] = store.get(ACCEPTED, 0.0) + float(accepted)
    store[DRAFTED] = store.get(DRAFTED, 0) + int(proposals)


def rounds(*pairs):
    def during(drafter):
        for accepted, proposals in pairs:
            record_round(drafter, accepted, proposals)
    return during


def overwrite(**values):
    def during(drafter):
        store = _state(drafter)["store"]
        for name, value in values.items():
            if value is DELETE:
                store.pop(name, None)
            else:
                store[name] = value
    return during


def lifetime(rounds_, accepted, drafted):
    return {ROUNDS: rounds_, ACCEPTED: accepted, DRAFTED: drafted}


class Harness:
    """Fakes for every dependency of the extracted function, recorded in order."""

    def __init__(self, monkeypatch, *, drafter=None, during=None, poisoned=False,
                 fail_at=None, images=()):
        self.calls = []
        self.poisoned, self.fail_at, self.during = poisoned, fail_at, during
        self.target = {"path": "/fake/target", "fingerprint": "target-fp",
                       "config": {"text_config": {}}}
        self.model = types.SimpleNamespace(config=object())
        self.processor = object()
        self.draft = drafter if drafter is not None else Drafter()
        self.draft_record = {"fingerprint": "draft-fp"}
        self.output = types.SimpleNamespace(text="x", finish_reason="stop")
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
        self.modules = modules
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
        return {"fingerprint": "draft-fp"}

    def _source(self):
        self._hit("source")
        return {"revision": SOURCE_REVISION}

    def _load(self, path, **kwargs):
        self._hit("load", path, kwargs)
        return self.model, self.processor

    def _load_candidate_dspark(self, path, *, target_artifact, target_model):
        self._hit("load_candidate_dspark", path, target_artifact, target_model)
        return self.draft, self.draft_record

    def _template(self, processor, config, prompt, *, num_images):
        self._hit("template", processor, config, prompt, num_images)
        return "formatted"

    def _generate(self, model, processor, formatted, **kwargs):
        self.reads_before_generate = list(_state(self.draft)["reads"])
        self._hit("generate", model, processor, formatted, kwargs)
        assert kwargs["draft_model"] is self.draft
        if self.during is not None:
            self.during(self.draft)
        return self.output

    def stages(self):
        return [call[0] for call in self.calls]

    def generate_kwargs(self):
        (call,) = [c for c in self.calls if c[0] == "generate"]
        return call[4]


def run(monkeypatch, *, store=None, during=None, width=10, max_tokens=64, **kw):
    drafter = Drafter(store, raise_on=kw.pop("raise_on", None),
                      refuse_on=kw.pop("refuse_on", None))
    harness = Harness(monkeypatch, drafter=drafter, during=during)
    result = harness.fn("/t", "/d", "hi", max_tokens=max_tokens,
                        verification_width=width, collect_speculative_stats=True, **kw)
    return harness, result


def assert_unavailable(result, reason):
    stats = result["speculative_stats"]
    assert set(stats) == STATS_KEYS
    assert stats["available"] is False and stats["engaged"] is False
    assert stats["unavailable_reason"] == reason
    assert stats["baseline_kind"] is None
    assert stats["rounds"] is None
    assert stats["accepted_proposals"] is None
    assert stats["drafted_proposals"] is None
    assert stats["qualified"] is False and stats["selected"] is False
    assert result["qualified"] is False and result["selected"] is False


def assert_counts(result, expected, *, engaged, baseline_kind="lifetime"):
    stats = result["speculative_stats"]
    assert set(stats) == STATS_KEYS
    assert stats["available"] is True and stats["unavailable_reason"] is None
    assert stats["engaged"] is engaged
    assert stats["baseline_kind"] == baseline_kind
    got = (stats["rounds"], stats["accepted_proposals"], stats["drafted_proposals"])
    assert got == expected
    assert all(type(value) is int for value in got)
    assert stats["scope"] == "request" and stats["attribution"] == "b1-private-drafter"
    assert stats["source_revision"] == SOURCE_REVISION
    assert "not emitted tokens" in stats["unit"]
    assert stats["qualified"] is False and stats["selected"] is False
    assert result["qualified"] is False and result["selected"] is False


# -- native guard controls -------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FORBIDDEN))
def test_builtin_guard_blocks_native_import(name):
    with pytest.raises(ImportError, match="builtin guard"):
        __import__(name)
    assert _IMPORT_ATTEMPTS[-1] == ("builtin", name)
    _IMPORT_ATTEMPTS.clear()


@pytest.mark.parametrize("name", ["mlx.core", "mlx_lm.generate", "mlx_vlm.speculative",
                                  "mlx2.adapters"])
def test_meta_path_guard_blocks_importlib(name):
    with pytest.raises(ImportError, match="meta_path guard"):
        importlib.import_module(name)
    assert ("meta_path", name.partition(".")[0]) in _IMPORT_ATTEMPTS
    _IMPORT_ATTEMPTS.clear()


def test_meta_path_guard_blocks_find_spec():
    with pytest.raises(ImportError, match="meta_path guard"):
        importlib.util.find_spec("mlx_vlm")
    _IMPORT_ATTEMPTS.clear()


def test_guard_lets_ordinary_imports_through():
    assert __import__("json") is json
    assert importlib.import_module("math") is math
    assert not _IMPORT_ATTEMPTS


def test_dummies_are_fileless_specless(monkeypatch):
    harness = Harness(monkeypatch)
    for name, module in harness.modules.items():
        assert module.__spec__ is None and not hasattr(module, "__file__"), name
        assert module.__mlx2_test_dummy__ is True


# -- signature and invalid flag ----------------------------------------------------


def test_signature_keyword_only_default_false(monkeypatch):
    params = inspect.signature(Harness(monkeypatch).fn).parameters
    flag = params["collect_speculative_stats"]
    assert flag.kind is inspect.Parameter.KEYWORD_ONLY
    assert flag.default is False and flag.annotation == "bool"
    assert params["verification_width"].default == 10
    assert params["max_tokens"].default == 128


@pytest.mark.parametrize("flag", [1, 0, "true", "", None, 1.0, [True], types.SimpleNamespace()])
def test_invalid_flag_refused_before_any_inspection(monkeypatch, flag):
    harness = Harness(monkeypatch, poisoned=True, drafter=Drafter(poisoned=True))
    with pytest.raises(ValueError, match="collect_speculative_stats must be a bool"):
        harness.fn("/t", "/d", "hi", image="/poisoned.png",
                   collect_speculative_stats=flag)
    assert harness.calls == []
    assert _state(harness.draft)["reads"] == []
    assert not _IMPORT_ATTEMPTS


def test_bool_flag_passes_gate_into_poisoned_path(monkeypatch):
    # Control: a valid flag reaches the (poisoned) image check.
    harness = Harness(monkeypatch, poisoned=True)
    with pytest.raises(Refused, match="path"):
        harness.fn("/t", "/d", "hi", image="/poisoned.png",
                   collect_speculative_stats=True)
    assert harness.stages() == ["path"]


# -- disabled path: untouched drafter and Stage49 contract -------------------------


@pytest.mark.parametrize("explicit", [False, None])
def test_disabled_never_touches_counters(monkeypatch, explicit):
    harness = Harness(monkeypatch, drafter=Drafter(poisoned=True))
    kwargs = {} if explicit is None else {"collect_speculative_stats": explicit}
    result = harness.fn("/t", "/d", "hi", **kwargs)
    assert _state(harness.draft)["reads"] == []
    assert set(result) == RESULT_KEYS
    assert set(result["evaluation_contract"]) == CONTRACT_KEYS
    assert result["qualified"] is False and result["selected"] is False


def test_default_contract_and_generation_args(monkeypatch):
    harness = Harness(monkeypatch, drafter=Drafter(poisoned=True))
    result = harness.fn("/t", "/d", "hi")
    assert harness.stages() == ["inspect_artifact", "inspect_dspark_artifact",
                                "source", "load", "load_candidate_dspark",
                                "template", "generate"]
    assert result == {
        "text": "x", "finish_reason": "stop",
        "target_fingerprint": "target-fp", "draft_fingerprint": "draft-fp",
        "evaluation_contract": {
            "scope": "requested", "source_revision": SOURCE_REVISION,
            "requested_verification_width": 10, "maximum_proposals_per_round": 9,
            "temperature": 0.0, "max_tokens": 128,
        },
        "qualified": False, "selected": False,
    }
    assert harness.generate_kwargs() == {
        "image": None, "max_tokens": 128, "verbose": False,
        "draft_model": harness.draft, "draft_kind": "dflash",
        "draft_block_size": 10, "temperature": 0.0,
    }


@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("width", [8, 10])
def test_width_forwarding_greedy_and_strict_binding(monkeypatch, width, collect):
    harness = Harness(monkeypatch, images=("/img.png",))
    result = harness.fn("/t", "/d", "hi", image="/img.png", max_tokens=7,
                        verification_width=width, collect_speculative_stats=collect)
    kwargs = harness.generate_kwargs()
    assert kwargs["draft_block_size"] == width and kwargs["temperature"] == 0.0
    assert kwargs["image"] == "/img.png" and kwargs["max_tokens"] == 7
    (load,) = [c for c in harness.calls if c[0] == "load"]
    assert load[1:] == ("/fake/target", {"lazy": False, "strict": True,
                                         "trust_remote_code": False})
    (bind,) = [c for c in harness.calls if c[0] == "load_candidate_dspark"]
    assert bind[2] is harness.target and bind[3] is harness.model
    (inspect_draft,) = [c for c in harness.calls if c[0] == "inspect_dspark_artifact"]
    assert inspect_draft[2] is harness.target
    (template,) = [c for c in harness.calls if c[0] == "template"]
    assert template[4] == 1
    contract = result["evaluation_contract"]
    assert contract["requested_verification_width"] == width
    assert contract["maximum_proposals_per_round"] == width - 1
    assert ("speculative_stats" in result) is collect
    assert set(result) - {"speculative_stats"} == RESULT_KEYS
    assert result["qualified"] is False and result["selected"] is False


@pytest.mark.parametrize("kwargs, match", [
    ({"verification_width": 1}, "verification_width"),
    ({"verification_width": 11}, "verification_width"),
    ({"verification_width": True}, "verification_width"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"max_tokens": 513}, "max_tokens"),
])
def test_stage49_bounds_unchanged_with_flag(monkeypatch, kwargs, match):
    harness = Harness(monkeypatch, poisoned=True)
    with pytest.raises(ValueError, match=match):
        harness.fn("/t", "/d", "hi", collect_speculative_stats=True, **kwargs)
    assert harness.calls == []


@pytest.mark.parametrize("stage", ["inspect_artifact", "load", "load_candidate_dspark",
                                   "template", "generate"])
def test_exceptions_propagate_with_stats_on(monkeypatch, stage):
    harness = Harness(monkeypatch, fail_at=stage, drafter=Drafter(lifetime(1, 0.0, 9)))
    with pytest.raises(RuntimeError, match=f"boom {stage}"):
        harness.fn("/t", "/d", "hi", collect_speculative_stats=True)
    reads = _state(harness.draft)["reads"]
    # Only the pre-generate baseline may have been read; no post read on failure.
    assert reads == (list(COUNTERS) if stage == "generate" else [])


# -- valid per-request deltas ------------------------------------------------------


def test_uninitialized_baseline_counts_from_source_zero(monkeypatch):
    harness, result = run(monkeypatch, during=rounds((9, 9), (0, 9), (4, 9)))
    assert_counts(result, (3, 13, 27), engaged=True, baseline_kind="uninitialized")
    assert harness.reads_before_generate == list(COUNTERS)
    assert _state(harness.draft)["reads"] == list(COUNTERS) * 2


def test_preexisting_lifetime_baseline_is_diffed(monkeypatch):
    store = {**lifetime(7, 20.0, 50), "accept_lens": [1, 2, 3]}

    def during(drafter):
        _state(drafter)["store"]["accept_lens"] = []  # reset() clears history only
        rounds((7, 7), (7, 7))(drafter)

    harness, result = run(monkeypatch, store=store, during=during, width=8)
    assert_counts(result, (2, 14, 14), engaged=True)
    assert _state(harness.draft)["store"]["accept_lens"] == [7, 7]


def test_all_rejected_is_engaged_with_zero_accepted(monkeypatch):
    _, result = run(monkeypatch, store=lifetime(2, 3.0, 18), during=rounds((0, 9), (0, 9)))
    assert_counts(result, (2, 0, 18), engaged=True)


def test_partial_accept(monkeypatch):
    _, result = run(monkeypatch, during=rounds((3, 5), (5, 5), (1, 5)), width=6)
    assert_counts(result, (3, 9, 15), engaged=True, baseline_kind="uninitialized")


def test_proposals_may_exceed_emitted_tokens(monkeypatch):
    # Budget/EOS truncation happens after the source records a round.
    _, result = run(monkeypatch, during=rounds((9, 9)), max_tokens=2)
    assert_counts(result, (1, 9, 9), engaged=True, baseline_kind="uninitialized")
    assert result["text"] == "x"


def test_zero_proposal_round_available_not_engaged(monkeypatch):
    _, result = run(monkeypatch, store=lifetime(4, 6.0, 12), during=rounds((0, 0)),
                    max_tokens=1)
    assert_counts(result, (1, 0, 0), engaged=False)


def test_zero_activity_available_not_engaged(monkeypatch):
    _, result = run(monkeypatch, store=lifetime(5, 3.0, 8))
    assert_counts(result, (0, 0, 0), engaged=False)


def test_int_accepted_counter_accepted(monkeypatch):
    _, result = run(monkeypatch, store=lifetime(1, 2, 9), during=overwrite(
        **lifetime(2, 5, 18)))
    assert_counts(result, (1, 3, 9), engaged=True)


def test_largest_exact_float_accepted(monkeypatch):
    top = 2 ** 53
    _, result = run(monkeypatch, store=lifetime(top, float(top - 2), top),
                    during=overwrite(**lifetime(top + 1, float(top - 1), top + 9)))
    assert_counts(result, (1, 1, 9), engaged=True)


def test_ceiling_exactly_met(monkeypatch):
    _, result = run(monkeypatch, during=rounds((3, 3), (0, 3)), width=4)
    assert_counts(result, (2, 3, 6), engaged=True, baseline_kind="uninitialized")


# -- refused counter evidence --------------------------------------------------------


class IntSub(int):
    pass


class FloatSub(float):
    pass


BAD_VALUES = {
    ROUNDS: [True, False, "3", 3.0, IntSub(3), -1, None],
    ACCEPTED: [True, False, "3", 1.5, -1.0, -1, math.nan, math.inf, -math.inf,
               float(2 ** 53), 1e300, IntSub(3), FloatSub(3.0), None],
    DRAFTED: [True, False, "9", 9.0, IntSub(9), -1, None],
}
BAD_CASES = [(name, value) for name, values in BAD_VALUES.items() for value in values]
BAD_IDS = [f"{name.rsplit('_', 1)[-1]}-{value!r}" for name, value in BAD_CASES]


@pytest.mark.parametrize("name, value", BAD_CASES, ids=BAD_IDS)
def test_malformed_baseline_unavailable(monkeypatch, name, value):
    store = {**lifetime(3, 3.0, 27), name: value}
    harness, result = run(monkeypatch, store=store, during=overwrite(**lifetime(4, 4.0, 36)))
    assert_unavailable(result, "baseline_counter_invalid")
    # A refused baseline is not followed by a post read; generation still ran.
    assert _state(harness.draft)["reads"] == list(COUNTERS)
    assert result["text"] == "x"


@pytest.mark.parametrize("name, value", BAD_CASES, ids=BAD_IDS)
def test_malformed_post_unavailable(monkeypatch, name, value):
    _, result = run(monkeypatch, store=lifetime(3, 3.0, 27), during=overwrite(
        **{**lifetime(4, 4.0, 36), name: value}))
    assert_unavailable(result, "post_counter_invalid")


@pytest.mark.parametrize("missing", [(ROUNDS,), (ACCEPTED,), (DRAFTED,), (ROUNDS, DRAFTED)])
def test_partial_missing_baseline_unavailable(monkeypatch, missing):
    store = {k: v for k, v in lifetime(3, 3.0, 27).items() if k not in missing}
    _, result = run(monkeypatch, store=store, during=rounds((9, 9)))
    assert_unavailable(result, "baseline_counters_missing")


@pytest.mark.parametrize("missing", [COUNTERS, (ROUNDS,), (ACCEPTED,), (DRAFTED,)])
def test_missing_post_unavailable(monkeypatch, missing):
    _, result = run(monkeypatch, store=lifetime(3, 3.0, 27), during=overwrite(
        **{name: DELETE for name in missing}))
    assert_unavailable(result, "post_counters_missing")


def test_uninitialized_without_rounds_is_not_zero_evidence(monkeypatch):
    _, result = run(monkeypatch)
    assert_unavailable(result, "post_counters_missing")


@pytest.mark.parametrize("before, after, reason", [
    (lifetime(1, 5.0, 4), lifetime(2, 5.0, 13), "baseline_accepted_exceeds_drafted"),
    (lifetime(1, 1.0, 9), lifetime(2, 19.0, 18), "post_accepted_exceeds_drafted"),
    (lifetime(5, 3.0, 27), lifetime(4, 3.0, 27), "counter_regression"),
    (lifetime(5, 3.0, 27), lifetime(6, 2.0, 36), "counter_regression"),
    (lifetime(5, 3.0, 27), lifetime(6, 3.0, 26), "counter_regression"),
    (lifetime(1, 0.0, 5), lifetime(2, 3.0, 6), "delta_accepted_exceeds_drafted"),
    (lifetime(5, 3.0, 27), lifetime(5, 3.0, 29), "zero_round_nonzero_delta"),
    (lifetime(5, 3.0, 27), lifetime(5, 4.0, 28), "zero_round_nonzero_delta"),
])
def test_inconsistent_counters_unavailable(monkeypatch, before, after, reason):
    _, result = run(monkeypatch, store=before, during=overwrite(**after))
    assert_unavailable(result, reason)


@pytest.mark.parametrize("width, proposals", [(4, (4,)), (2, (1, 2)), (10, (9, 10))])
def test_requested_ceiling_violation_unavailable(monkeypatch, width, proposals):
    during = rounds(*[(0, count) for count in proposals])
    _, result = run(monkeypatch, store=lifetime(1, 0.0, 1), during=during, width=width)
    assert_unavailable(result, "requested_ceiling_exceeded")


@pytest.mark.parametrize("phase, attr", [("baseline", ROUNDS), ("baseline", DRAFTED),
                                         ("post", ACCEPTED)])
def test_counter_read_exception_unavailable(monkeypatch, phase, attr):
    def during(drafter):
        record_round(drafter, 1, 9)
        if phase == "post":
            _state(drafter)["raise_on"] = attr

    harness, result = run(monkeypatch, store=lifetime(1, 1.0, 9), during=during,
                          raise_on=attr if phase == "baseline" else None)
    assert_unavailable(result, f"{phase}_counter_read_error")
    assert result["text"] == "x"


def test_base_exception_from_counter_read_propagates(monkeypatch):
    with pytest.raises(Refused, match=ACCEPTED):
        run(monkeypatch, store=lifetime(1, 1.0, 9), refuse_on=ACCEPTED)


def test_counter_attributes_only_reads_no_writes(monkeypatch):
    harness, _ = run(monkeypatch, store=lifetime(1, 1.0, 9), during=rounds((2, 9)))
    assert set(_state(harness.draft)["reads"]) == set(COUNTERS)


# -- provenance ----------------------------------------------------------------------


def test_provenance_declares_original_unqualified_diagnostic():
    record = json.loads(PROVENANCE.read_text())
    assert record["mined_code"] is False
    assert record["dependency_semantics"]["source_copied"] is False
    assert record["dependency_semantics"]["revision"] == SOURCE_REVISION
    assert record["qualified"] is False and record["selected"] is False
