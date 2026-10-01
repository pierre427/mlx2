"""Host-only tests for the ragged PLD evidence guards (scripts/qualify_ragged_pld.py).

No mlx, mlx_lm or mlx2 module may load: a meta-path blocker is installed
before the driver and the oracle are imported, and every test checks that
none is present. Records are synthetic host dictionaries. The unchanged
``paired_direct_ab.snapshot_refusal`` is the real oracle check;
``state_digest`` is guarded so that only ``state_digest(None)`` (stdlib-only)
can run. Nothing here executes a model, cache, generator or GPU.

``Driver.continuation`` is EXECUTED with a host substitute for ``run``.
``Driver.run`` row plumbing and ``run_all`` ordering are checked as SOURCE
(AST), not executed: ``run`` imports native modules.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -p no:cacheprovider \
      -o addopts= -q tests/test_ragged_pld_evidence_cpu.py
"""

import ast
import copy
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

NATIVE = frozenset({"mlx", "mlx_lm", "mlx2"})


def _native_loaded():
    return sorted(name for name in sys.modules if name.split(".")[0] in NATIVE)


class _NativeBlocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in NATIVE:
            raise ImportError(f"native import blocked in a host test: {name}")


assert not _native_loaded(), _native_loaded()
sys.meta_path.insert(0, _NativeBlocker())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paired_direct_ab as P
from scripts import qualify_ragged_pld as Q

UNAVAILABLE = P.state_digest(None)  # the real output; returns before any native import
_REAL_STATE_DIGEST = P.state_digest


@pytest.fixture(autouse=True)
def host_only(monkeypatch):
    def guarded(obj):
        if obj is not None:
            raise AssertionError("state_digest over a real object must not run in host tests")
        return _REAL_STATE_DIGEST(None)

    monkeypatch.setattr(P, "state_digest", guarded)
    assert not _native_loaded()
    yield
    assert not _native_loaded(), _native_loaded()


def digest(char):
    return {"status": "complete", "sha256": char * 64, "reason": None}


def lane(tokens=(1, 2), rows="ab", requested=2, final="e", covered=5, cont=(3,), cont_final="f"):
    tokens = list(tokens)
    return {"tokens": tokens, **Q.lane_row_evidence([digest(c) for c in rows], tokens, requested),
            "covered_tokens": covered, "final_state": digest(final),
            "continuation": {"status": "complete", "tokens": list(cont), "final_state": digest(cont_final)}}


def cmp(candidate, reference, rows=2, **kwargs):
    return Q.compare("candidate", "reference", [0], {"candidate": {0: candidate}, "reference": {0: reference}},
                     logprob_rows=rows, **kwargs)


def assert_incomparable(result, needle):
    assert result["tokens_exact"] and not result["bits_exact"], result
    assert result["bit_differences"] == [] and result["token_differences"] == [], result
    assert any(needle in item for item in result["incomparable"]), result


def test_blocker_installed_and_no_native_module_loaded():
    assert isinstance(sys.meta_path[0], _NativeBlocker) or any(
        isinstance(f, _NativeBlocker) for f in sys.meta_path)
    with pytest.raises(ImportError, match="blocked"):
        __import__("mlx.core")
    assert _native_loaded() == []
    assert UNAVAILABLE == {"status": "unavailable", "sha256": None, "reason": "no state returned"}


def test_state_digest_guard_refuses_real_objects():
    with pytest.raises(AssertionError, match="must not run"):
        P.state_digest([object()])


# ---- the four root counterexamples, verbatim legacy records ----

def _root_cases():
    full = {"status": "complete", "sha256": "a" * 64, "reason": None}
    base = {"tokens": [1, 2], "logprob_rows": ["b" * 64, "c" * 64], "logprob_rows_status": ["complete"],
            "covered_tokens": 5, "final_state": full,
            "continuation": {"status": "complete", "tokens": [3], "final_state": full}}
    short = copy.deepcopy(base)
    short["logprob_rows"] = short["logprob_rows"][:1]
    partial = copy.deepcopy(base)
    partial["logprob_rows_status"] = ["metadata_unavailable"]
    invalid = copy.deepcopy(base)
    invalid["continuation"]["final_state"] = {"status": "unavailable", "sha256": None, "reason": "no state"}
    forged = copy.deepcopy(base)
    forged["final_state"] = {"status": "complete", "sha256": None, "reason": None}
    return {"truncated_logprob_rows": (base, short),
            "noncomplete_row_status_with_equal_hashes": (base, partial),
            "identical_unavailable_continuation_states": (invalid, copy.deepcopy(invalid)),
            "identical_complete_without_digest": (forged, copy.deepcopy(forged))}


@pytest.mark.parametrize("name", sorted(_root_cases()))
@pytest.mark.parametrize("bound", [None, 2])
def test_root_counterexamples_no_longer_pass(name, bound):
    reference, candidate = _root_cases()[name]
    result = Q.compare("candidate", "reference", [0], {"candidate": {0: candidate}, "reference": {0: reference}},
                       logprob_rows=bound)
    assert result["tokens_exact"] and not result["bits_exact"]
    assert result["incomparable"]
    rows = next(x for x in result["incomparable"] if "logprob rows unavailable" in x)
    assert ("no protocol logprob row bound" if bound is None else "legacy record without strict row evidence") in rows


def test_root_case_states_are_refused_by_snapshot_refusal_even_with_strict_rows():
    unavailable = lane()
    unavailable["continuation"]["final_state"] = {"status": "unavailable", "sha256": None, "reason": "no state"}
    assert_incomparable(cmp(unavailable, copy.deepcopy(unavailable)), "continuation unavailable")
    forged = lane()
    forged["final_state"] = {"status": "complete", "sha256": None, "reason": None}
    assert_incomparable(cmp(forged, copy.deepcopy(forged)), "final state unavailable")


def test_root_case_rows_are_refused_with_strict_records():
    short = lane()
    short["logprob_row_digests"] = short["logprob_row_digests"][:1]
    short["logprob_rows"] = short["logprob_rows"][:1]
    assert_incomparable(cmp(short, lane()), "candidate: 1 row digests, expected 2")
    meta = {"status": "metadata_unavailable", "sha256": None, "state_only_sha256": "b" * 64,
            "reason": "no meta_state on X"}
    partial = lane()
    partial["logprob_row_digests"][0] = meta
    result = cmp(partial, copy.deepcopy(partial))
    assert_incomparable(result, "candidate: row 0: status 'metadata_unavailable' is not complete")
    assert "reference: row 0" in result["incomparable"][0]


# ---- genuine positive and complete differences ----

def test_genuine_complete_evidence_is_bit_exact():
    result = cmp(lane(), lane())
    assert result == {"arm": "candidate", "reference": "reference", "tokens_exact": True, "bits_exact": True,
                      "token_differences": [], "bit_differences": [], "incomparable": []}


@pytest.mark.parametrize("edit,message", [
    ({"rows": "a9"}, "lane 0: logprob row bits differ from row 1"),
    ({"final": "9"}, "lane 0: final state bits differ"),
    ({"cont_final": "9"}, "lane 0: continuation state bits differ"),
])
def test_complete_wrong_hashes_are_bit_differences(edit, message):
    result = cmp(lane(**edit), lane())
    assert result["tokens_exact"] and not result["bits_exact"]
    assert result["bit_differences"] == [message] and result["incomparable"] == []


def test_token_and_continuation_token_differences_stay_token_failures():
    result = cmp(lane(tokens=(1, 9)), lane())
    assert result["token_differences"] == ["lane 0: tokens differ at 1"] and not result["tokens_exact"]
    result = cmp(lane(cont=(4,)), lane())
    assert result["token_differences"] == ["lane 0: continuation tokens differ"]
    assert result["incomparable"] == [] and not result["bits_exact"]


def test_covered_token_mismatch_is_incomparable_not_a_bit_difference():
    result = cmp(lane(covered=6, final="9"), lane())
    assert_incomparable(result, "lane 0: final caches cover 6 vs 5 tokens")
    result = cmp(lane(covered=None), lane(covered=None))
    assert_incomparable(result, "covered tokens unavailable")
    assert_incomparable(cmp(lane(covered=True), lane(covered=True)), "covered tokens unavailable")


# ---- row count: protocol bound and delivered tokens ----

def test_expected_two_truncated_on_both_arms_is_refused():
    short = lane()
    short["logprob_row_digests"] = short["logprob_row_digests"][:1]
    short["logprob_rows"] = short["logprob_rows"][:1]
    result = cmp(short, copy.deepcopy(short))
    assert_incomparable(result, "candidate: 1 row digests, expected 2")
    assert "reference: 1 row digests, expected 2" in result["incomparable"][0]


def test_extra_rows_are_refused():
    extra = lane()
    extra["logprob_row_digests"].append(digest("z"))
    extra["logprob_rows"].append("z" * 64)
    assert_incomparable(cmp(extra, copy.deepcopy(extra)), "3 row digests, expected 2")


def test_lowered_expected_count_is_refused():
    lowered = lane()
    lowered["logprob_rows_expected"] = 1
    lowered["logprob_row_digests"] = lowered["logprob_row_digests"][:1]
    lowered["logprob_rows"] = lowered["logprob_rows"][:1]
    assert_incomparable(cmp(lowered, copy.deepcopy(lowered)),
                        "declares 1 rows; protocol bound and delivered tokens give 2")


def test_lowered_requested_bound_is_refused_on_record_or_caller():
    record = Q.lane_row_evidence([digest("a")], [1, 2], 1)  # a record claiming a bound of 1
    lowered = {**lane(), **record}
    assert_incomparable(cmp(lowered, copy.deepcopy(lowered)), "requested 1 is not the protocol bound 2")
    assert_incomparable(cmp(lane(), lane(), rows=1), "requested 2 is not the protocol bound 1")
    assert_incomparable(cmp(lane(), lane(), rows=True), "no protocol logprob row bound")


def test_expected_count_is_bounded_by_delivered_tokens():
    record = Q.lane_row_evidence([digest("a"), digest("b")], [1, 2], 16)
    assert record["logprob_rows_requested"] == 16 and record["logprob_rows_expected"] == 2
    one = lane(tokens=(1, 2, 3), rows="a", requested=1)
    assert one["logprob_rows_expected"] == 1
    assert cmp(one, copy.deepcopy(one), rows=1)["bits_exact"]
    long = lane(rows="ab", requested=16)
    assert cmp(long, copy.deepcopy(long), rows=16)["bits_exact"]


def test_disabled_and_missing_rows_are_explicitly_unavailable():
    disabled = lane(rows="", requested=0)
    assert disabled["logprob_rows_status"] == ["unavailable"] and disabled["logprob_rows_expected"] == 0
    assert_incomparable(cmp(disabled, copy.deepcopy(disabled), rows=0), "disabled (--logprob-rows 0)")
    assert_incomparable(cmp(lane(), lane(), rows=0), "disabled (--logprob-rows 0)")
    assert_incomparable(cmp(lane(), lane(), rows=None), "no protocol logprob row bound")
    missing = lane()
    del missing["logprob_row_digests"]
    assert_incomparable(cmp(missing, copy.deepcopy(missing)), "no row digest records")
    empty = lane(tokens=(), rows="")
    assert_incomparable(cmp(empty, copy.deepcopy(empty)), "no delivered tokens")


def test_unavailable_row_slot_from_a_response_without_logprobs_is_refused():
    gap = lane()
    gap["logprob_row_digests"][1] = UNAVAILABLE
    gap["logprob_rows"][1] = None
    gap["logprob_rows_status"] = ["complete", "unavailable"]
    assert_incomparable(cmp(gap, copy.deepcopy(gap)), "row 1: status 'unavailable' is not complete")


def test_legacy_rows_are_incomparable_without_retroactive_upgrade():
    legacy = lane()
    for key in ("evidence", "logprob_rows_requested", "logprob_rows_expected", "logprob_row_digests"):
        del legacy[key]
    assert legacy["logprob_rows"] == ["a" * 64, "b" * 64] and legacy["logprob_rows_status"] == ["complete"]
    assert_incomparable(cmp(legacy, copy.deepcopy(legacy)), "legacy record without strict row evidence")
    assert_incomparable(cmp(legacy, lane()), "candidate: legacy record without strict row evidence")
    old = dict(lane(), evidence="mlx2.ragged-pld.lane-evidence.v1")
    assert_incomparable(cmp(old, copy.deepcopy(old)), "legacy record")


@pytest.mark.parametrize("field,value,message", [
    ("logprob_rows", ["a" * 64, "x" * 64], "logprob_rows hashes contradict the row digests"),
    ("logprob_rows", ["a" * 64], "logprob_rows hashes contradict the row digests"),
    ("logprob_rows_status", ["metadata_unavailable"], "logprob_rows_status contradicts the row digests"),
    ("logprob_rows_status", ["complete", "unavailable"], "logprob_rows_status contradicts the row digests"),
])
def test_duplicate_row_fields_must_match_their_digests(field, value, message):
    bad = dict(lane(), **{field: value})
    assert_incomparable(cmp(bad, copy.deepcopy(bad)), message)


# ---- malformed digests (rows, final state, continuation state) ----

BAD_DIGESTS = {
    "missing_hash": {"status": "complete", "sha256": None, "reason": None},
    "uppercase_hash": {"status": "complete", "sha256": "A" * 64, "reason": None},
    "short_hash": {"status": "complete", "sha256": "a" * 63, "reason": None},
    "nonhex_hash": {"status": "complete", "sha256": "g" * 64, "reason": None},
    "reason": {"status": "complete", "sha256": "a" * 64, "reason": "no meta_state"},
    "no_reason_key": {"status": "complete", "sha256": "a" * 64},
    "extra_field": {"status": "complete", "sha256": "a" * 64, "reason": None, "state_only_sha256": "a" * 64},
    "unavailable": UNAVAILABLE,
    "metadata_unavailable": {"status": "metadata_unavailable", "sha256": None, "state_only_sha256": "a" * 64,
                             "reason": "no meta_state on X"},
    "status_case": {"status": "Complete", "sha256": "a" * 64, "reason": None},
    "not_a_dict": "a" * 64,
    "absent": None,
}


@pytest.mark.parametrize("kind", sorted(BAD_DIGESTS))
@pytest.mark.parametrize("where", ["row", "final", "continuation"])
def test_malformed_digests_never_satisfy_bits_exact_even_when_equal(kind, where):
    bad = lane()
    value = copy.deepcopy(BAD_DIGESTS[kind])
    if where == "row":
        bad["logprob_row_digests"][0] = value
        bad.pop("logprob_rows")  # no duplicate to trip first: the digest itself is refused
        bad.pop("logprob_rows_status")
        needle = "logprob rows unavailable"
    elif where == "final":
        bad["final_state"] = value
        needle = "final state unavailable"
    else:
        bad["continuation"]["final_state"] = value
        needle = "continuation unavailable"
    result = cmp(bad, copy.deepcopy(bad))
    assert_incomparable(result, needle)
    one_sided = cmp(bad, lane())
    assert not one_sided["bits_exact"] and any(needle in x for x in one_sided["incomparable"])


@pytest.mark.parametrize("continuation,message", [
    (None, "absent"),
    ({"status": "unavailable", "reason": "disabled (--continuation-tokens 0)"}, "status 'unavailable'"),
    ({"status": "complete", "tokens": [], "final_state": digest("f")}, "complete without delivered token ids"),
    ({"status": "complete", "tokens": ["3"], "final_state": digest("f")}, "complete without delivered token ids"),
    ({"status": "complete", "tokens": [3]}, "final state: not a digest record"),
    ({"status": "complete", "tokens": [3], "final_state": digest("f"), "reason": "x"}, "complete with a refusal"),
])
def test_incomplete_continuation_records_are_incomparable(continuation, message):
    bad = dict(lane(), continuation=continuation)
    assert_incomparable(cmp(bad, copy.deepcopy(bad)), message)


def test_continuation_check_is_skipped_only_when_disabled_by_the_caller():
    bad = dict(lane(), continuation=None)
    assert cmp(bad, copy.deepcopy(bad), continuation=False)["bits_exact"]


# ---- removed lane (prefix only) and survivors ----

def _removal(prefix, survivor):
    removed = {"tokens": list(prefix)}  # prefix-only: no rows, final or continuation state
    results = {"pld_removal": {0: removed, 1: survivor}, "ordinary_b1": {0: lane(), 1: lane()}}
    return Q.compare("pld_removal", "ordinary_b1", [0, 1], results, prefix_only=(0,),
                     continuation=False, logprob_rows=2)


def test_removed_lane_prefix_is_compared_without_state_and_survivor_is_strict():
    survivor = lane()
    survivor.pop("continuation")
    assert _removal([1], survivor)["bits_exact"]
    assert _removal([1, 2], survivor)["bits_exact"]
    differs = _removal([2], survivor)
    assert differs["token_differences"] == ["lane 0: removed-lane prefix differs"]
    longer = _removal([1, 2, 3], survivor)
    assert longer["token_differences"] == ["lane 0: removed-lane prefix differs"]


def test_empty_removed_prefix_does_not_vacuously_pass():
    result = _removal([], lane())
    assert not result["bits_exact"] and result["incomparable"] == ["lane 0: removed-lane prefix is empty"]


@pytest.mark.parametrize("edit,needle", [
    (lambda r: r.update(final_state={"status": "complete", "sha256": None, "reason": None}), "final state unavailable"),
    (lambda r: r.update(logprob_row_digests=r["logprob_row_digests"][:1]), "logprob rows unavailable"),
    (lambda r: r.pop("evidence"), "legacy record"),
])
def test_survivor_keeps_strict_evidence_requirements(edit, needle):
    survivor = lane()
    edit(survivor)
    result = _removal([1], survivor)
    assert not result["bits_exact"]
    assert any(x.startswith("lane 1: ") and needle in x for x in result["incomparable"]), result


# ---- Driver.continuation: EXECUTED with a host substitute for run ----

class _Final:
    prompt_cache = ("host cache stand-in",)


def _driver(continuation_tokens=2, out=None, failures=()):
    driver = Q.Driver.__new__(Q.Driver)  # no __init__: no model, no mlx
    driver.args = SimpleNamespace(continuation_tokens=continuation_tokens, logprob_rows=2)
    driver.prompts = {0: [7, 8, 9]}
    driver.followup = [5]
    driver.calls = []

    def run(arm, lanes, **kwargs):
        driver.calls.append((arm, lanes, kwargs))
        return {0: dict(out, _final=_Final())}, {}, list(failures), None

    driver.run = run
    return driver


def _lane_record():
    record = lane(covered=4)
    record.pop("continuation")
    record["_final"] = _Final()
    return record


def test_continuation_with_complete_digest_is_complete_and_feeds_the_uncovered_rest():
    driver = _driver(out={"tokens": [3, 4], "final_state": digest("f")})
    result = driver.continuation(0, _lane_record())
    assert result == {"status": "complete", "tokens": [3, 4], "final_state": digest("f")}
    ((arm, lanes, kwargs),) = driver.calls
    assert arm == "ordinary_b1" and lanes == [0]
    assert kwargs["caches"] == [_Final.prompt_cache] and kwargs["all_tokens"] == [[7, 8, 9, 1]]
    assert kwargs["prompts"] == [[2, 5]] and kwargs["caps"] == [2]


@pytest.mark.parametrize("kind", sorted(BAD_DIGESTS))
def test_continuation_with_incomplete_digest_is_unavailable(kind):
    state = copy.deepcopy(BAD_DIGESTS[kind])
    driver = _driver(out={"tokens": [3, 4], "final_state": state})
    result = driver.continuation(0, _lane_record())
    assert result["status"] == "unavailable" and result["reason"].startswith("continuation final state: ")
    assert result["tokens"] == [3, 4] and result["final_state"] == state
    assert Q.continuation_refusal(result) is not None


def test_continuation_without_tokens_is_unavailable():
    result = _driver(out={"tokens": [], "final_state": digest("f")}).continuation(0, _lane_record())
    assert result["status"] == "unavailable" and "no continuation tokens" in result["reason"]


def test_continuation_disabled_failed_or_without_cache_stays_unavailable():
    disabled = _driver(continuation_tokens=0, out={"tokens": [3], "final_state": digest("f")})
    assert disabled.continuation(0, _lane_record()) == {
        "status": "unavailable", "reason": "disabled (--continuation-tokens 0)"}
    assert disabled.calls == []
    failed = _driver(out={"tokens": [3], "final_state": digest("f")}, failures=["bounded: stopped"])
    assert failed.continuation(0, _lane_record()) == {"status": "unavailable", "reason": "bounded: stopped"}
    record = _lane_record()
    record.pop("_final")
    assert _driver(out={"tokens": [3], "final_state": digest("f")}).continuation(0, record)["status"] == "unavailable"


# ---- Driver.run and run_all: SOURCE ORDER (AST), not executed ----

def _function(obj):
    return ast.parse(textwrap.dedent(inspect.getsource(obj))).body[0]


def test_source_run_records_one_row_slot_per_delivered_token():
    run = _function(Q.Driver.run)
    guards = [node for node in ast.walk(run) if isinstance(node, ast.If)
              and "logprob_rows" in ast.unparse(node.test) and "args.logprob_rows" in ast.unparse(node.test)]
    assert len(guards) == 1
    guard = guards[0]
    assert ast.unparse(guard.test) == "len(record['logprob_rows']) < args.logprob_rows"  # no 'row is not None' skip
    assert [ast.unparse(s) for s in guard.body] == [
        "record['logprob_rows'].append(state_digest(None if row is None else [row]))"]


def test_source_run_lane_records_carry_strict_row_evidence():
    run = _function(Q.Driver.run)
    out = [node for node in ast.walk(run) if isinstance(node, ast.Dict)
           and any(isinstance(k, ast.Constant) and k.value == "final_state" for k in node.keys)]
    assert len(out) == 1
    spreads = [ast.unparse(v) for k, v in zip(out[0].keys, out[0].values) if k is None]
    assert spreads == ["lane_row_evidence(record['logprob_rows'], record['tokens'], args.logprob_rows)"]
    keys = {k.value for k in out[0].keys if isinstance(k, ast.Constant)}
    assert not keys & {"logprob_rows", "logprob_rows_status", "evidence", "logprob_row_digests"}


def test_source_run_all_passes_the_protocol_bound_and_digests_before_continuation():
    run_all = _function(Q.run_all)
    calls = [node for node in ast.walk(run_all) if isinstance(node, ast.Call) and ast.unparse(node.func) == "compare"]
    assert len(calls) == 6
    assert all(any(k.arg == "logprob_rows" and ast.unparse(k.value) == "rows" for k in c.keywords) for c in calls)
    assigns = [node for node in ast.walk(run_all) if isinstance(node, ast.Assign)
               and [ast.unparse(t) for t in node.targets] == ["rows"]]
    assert [ast.unparse(a.value) for a in assigns] == ["args.logprob_rows"]
    # final_state is digested inside run (before it returns), and run_all calls
    # continuation (which reuses the cache) only after run for that arm.
    lines = {ast.unparse(n.func): n.lineno for n in ast.walk(run_all) if isinstance(n, ast.Call)}
    assert lines["driver.run"] < lines["driver.continuation"]
    run = _function(Q.Driver.run)
    returns = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Return)]
    digests = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Call)
               and ast.unparse(n) == "state_digest(getattr(final, 'prompt_cache', None))"]
    assert len(digests) == 1 and digests[0] < max(returns)


def test_evidence_version_is_bumped():
    assert Q.SCHEMA == "mlx2.direct-model.ragged-pld.v2"
    assert Q.EVIDENCE == "mlx2.ragged-pld.lane-evidence.v2"
    assert lane()["evidence"] == Q.EVIDENCE


def test_oracle_helpers_are_the_unchanged_reused_ones():
    source = inspect.getsource(Q)
    assert "from scripts.paired_direct_ab import snapshot_refusal" in source
    assert "def snapshot_refusal" not in source and "def state_digest" not in source
    assert P.snapshot_refusal(digest("a")) is None
