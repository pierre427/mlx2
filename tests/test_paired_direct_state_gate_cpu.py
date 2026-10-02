"""Host-only tests for the paired harness ``--require-complete-state`` gate.

No MLX: ``mlx`` and ``mlx_lm`` imports are blocked before the harness loads,
no model (``--tiny`` included) is built, and no cache is digested. The pure
validator is tested directly; ``run_cohort`` runs its real orchestration,
engagement, strict-gate and comparison code over HOST-SUBSTITUTED records
(``Cohort``, ``run_arm``, ``mlx_identity`` and ``source_identity`` replaced;
``mlx2.runtime.generate`` stubbed only for its 256-response interval). These
are substituted host results, not an actual native run or any qualification.

  PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \\
      --noconftest -p no:cacheprovider -o addopts= -q \\
      tests/test_paired_direct_state_gate_cpu.py
"""

import copy
import hashlib
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

BLOCKED = ("mlx", "mlx_lm")


class _BlockMLX:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked in host-only test: {name}")


def _mlx_loaded():
    return sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)


sys.meta_path.insert(0, _BlockMLX())
assert not _mlx_loaded(), _mlx_loaded()

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("paired_direct_ab_strict_cpu",
                                               ROOT / "scripts" / "paired_direct_ab.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

PROMPT = [1 + (i * 7) % 120 for i in range(40)]
COVERED = len(PROMPT) - 1  # a fresh external run keeps the final prompt token as anchor
MECHANISMS = ("qsdpa-tiling", "external-reclaim", "external-prefill-reclaim")


@pytest.fixture(autouse=True)
def _host_only():
    assert not _mlx_loaded()
    yield
    assert not _mlx_loaded(), _mlx_loaded()


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def complete(tag="t"):
    return {"status": "complete", "sha256": _h(tag), "reason": None}


def metadata_unavailable(tag="t"):
    return {"status": "metadata_unavailable", "sha256": None, "state_only_sha256": _h(tag),
            "reason": "no meta_state on mlx2.runtime.cache.KVCache"}


def unavailable():
    return H.state_digest(None)  # the oracle's own output; needs no MLX


def boundary(target=None, sidecar=None, covered=COVERED, tokens=None):
    tokens = PROMPT[:covered] if tokens is None else tokens
    return {"covered_tokens": covered, "tokens_sha256": H._sha(json.dumps(tokens).encode()),
            "target": complete("bt") if target is None else target,
            "sidecar": complete("bs") if sidecar is None else sidecar}


def record(mechanism, arm, *, target=None, sidecar=None, prompt_boundary=None, tokens=None,
           rows=None, failures=(), counters=None, duplicates=True):
    """A run_arm-shaped record that engages its arm unless told otherwise."""
    external = mechanism != "qsdpa-tiling"
    target = complete("target") if target is None else target
    if sidecar is None:
        sidecar = complete("sidecar") if external else unavailable()
    if prompt_boundary is None and mechanism == "external-prefill-reclaim":
        prompt_boundary = boundary()
    if tokens is None:
        tokens = list(range(300 if mechanism == "external-reclaim" else 8))
    if rows is None:
        rows = [] if external else [_h(f"row{i}") for i in range(8)]
    c = {"composed_tiled_calls": 0, "composed_tiles": 0, "external_allocator_reclaims": 0,
         "suppressed_crossings": 0, "external_rounds": 0, "proposed_tokens": 0,
         "accepted_proposals": 0, "prefill_rounds": 0, "external_prefill_allocator_reclaims": 0}
    if arm == "on":
        c.update(composed_tiled_calls=2, composed_tiles=4)
    if external:
        c.update(external_rounds=5, proposed_tokens=10, prefill_rounds=5)
    if arm == "reclaim":
        c["external_allocator_reclaims"] = 1
    if arm == "control":
        c["suppressed_crossings"] = 1
    if arm == "prefill_reclaim":
        c["external_prefill_allocator_reclaims"] = 5
    c.update(counters or {})
    out = {
        "arm": arm, "tokens": len(tokens), "token_sha256": H._sha(json.dumps(tokens).encode()),
        "logprob_row_sha256": rows, "logprob_rows_compared": len(rows),
        "lane_failures": list(failures), "ttft_s": 0.1, "decode_s": 0.2, "decode_tok_s": 10.0,
        "active_bytes": 1, "cache_bytes": 1, "cache_high_bytes": 1, "peak_bytes": 1,
        "boundary_samples": [], "prefill_samples": [], "max_responses_per_poll": 1,
        "final_target_state": target, "final_draft_sidecar": sidecar,
        "prompt_boundary": prompt_boundary, "counters": c, "_tokens": list(tokens),
    }
    if duplicates and isinstance(target, dict) and isinstance(sidecar, dict):
        out["final_target_state_sha256"] = target.get("sha256")
        out["final_draft_sidecar_sha256"] = sidecar.get("sha256")
    return out


def _args(mechanism, *extra, warmups=0, pairs=1):
    return H.resolve_args(H.build_parser(), ["--tiny", "--mechanism", mechanism, "--out",
                                              "/dev/null", "--warmups", str(warmups),
                                              "--pairs", str(pairs), *extra])


def run(monkeypatch, mechanism, make, *, strict, warmups=0, pairs=1, edit_args=None):
    """Real run_cohort over substituted records: ``make(arm, call) -> record``."""
    calls = []

    class SubstitutedCohort:
        def __init__(self, args):
            self.args, self.mx, self.stops, self.prompt = args, None, (), list(PROMPT)
            self.identity = {"model": "host-substituted", "actual_native_run": False}

    def substituted_run_arm(cohort, arm):
        calls.append(arm)
        return copy.deepcopy(make(arm, len(calls) - 1))

    generate = types.ModuleType("mlx2.runtime.generate")
    generate.ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL = 256
    monkeypatch.setitem(sys.modules, "mlx2.runtime.generate", generate)
    monkeypatch.setattr(H, "Cohort", SubstitutedCohort)
    monkeypatch.setattr(H, "run_arm", substituted_run_arm)
    monkeypatch.setattr(H, "mlx_identity", lambda mx: {"substituted": True})
    monkeypatch.setattr(H, "source_identity", lambda: {"substituted": True})
    args = _args(mechanism, *(["--require-complete-state"] if strict else []),
                 warmups=warmups, pairs=pairs)
    if edit_args:
        edit_args(args)
    out = H.run_cohort(args)
    out["_calls"] = calls
    return out


def same(mechanism, **fields):
    return lambda arm, call: record(mechanism, arm, **fields)


# ------------------------------------------------------------ pure helper

def test_oracle_output_shapes_are_the_ones_the_gate_reads():
    source = inspect.getsource(H.state_digest)
    assert 'return {"status": "complete", "sha256": digest, "reason": None}' in source
    assert '"state_only_sha256": digest' in source
    assert H._COMPLETE_FIELDS == {"status", "sha256", "reason"}
    assert H.snapshot_refusal(unavailable()) == "status 'unavailable' is not complete"
    assert H.snapshot_refusal(complete()) is None


@pytest.mark.parametrize("snapshot, expected", [
    (None, "not a digest record (NoneType)"),
    ([complete()], "not a digest record (list)"),
    ({}, "status None is not complete"),
    ({"sha256": _h("t"), "reason": None}, "status None is not complete"),
    ({**complete(), "status": "COMPLETE"}, "is not complete"),
    ({**complete(), "status": ["complete"]}, "is not complete"),
    (metadata_unavailable(), "status 'metadata_unavailable' is not complete"),
    ({**metadata_unavailable(), "status": "complete"}, "unexpected fields state_only_sha256"),
    ({**complete(), "state_only_sha256": _h("t")}, "unexpected fields state_only_sha256"),
    ({**complete(), "extra": 1}, "unexpected fields extra"),
    ({"status": "complete", "sha256": _h("t")}, "complete without reason"),
    ({"status": "complete", "reason": None}, "complete without sha256"),
    ({**complete(), "sha256": None}, "lowercase 64-hex"),
    ({**complete(), "sha256": _h("t").upper()}, "lowercase 64-hex"),
    ({**complete(), "sha256": _h("t")[:63]}, "lowercase 64-hex"),
    ({**complete(), "sha256": "g" + _h("t")[1:]}, "lowercase 64-hex"),
    ({**complete(), "sha256": int(_h("t"), 16)}, "lowercase 64-hex"),
    ({**complete(), "reason": "no meta_state on X"}, "complete with a missing-state reason"),
    ({**complete(), "reason": ""}, "complete with a missing-state reason"),
])
def test_snapshot_refusal_refuses_incomplete_or_malformed(snapshot, expected):
    assert expected in H.snapshot_refusal(snapshot)


def test_required_snapshots_by_mechanism():
    assert H.REQUIRED_STATE == {
        "qsdpa-tiling": ("final_target_state",),
        "external-reclaim": ("final_target_state", "final_draft_sidecar"),
        "external-prefill-reclaim": ("final_target_state", "final_draft_sidecar",
                                     "prompt_boundary.target", "prompt_boundary.sidecar"),
    }
    assert set(H.REQUIRED_STATE) == set(H.ARMS)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_complete_records_pass_every_mechanism(mechanism):
    for arm in H.ARMS[mechanism]:
        assert H.strict_state_refusals(mechanism, record(mechanism, arm), PROMPT) == []


def test_ordinary_sidecar_is_not_required_in_any_status_or_absent():
    for sidecar in (unavailable(), metadata_unavailable()):
        rec = record("qsdpa-tiling", "on", sidecar=sidecar)
        assert H.strict_state_refusals("qsdpa-tiling", rec, PROMPT) == []
    for edit in ({"final_draft_sidecar": None}, {"final_draft_sidecar": "junk"}):
        rec = {**record("qsdpa-tiling", "on"), **edit}
        del rec["final_draft_sidecar_sha256"]
        assert H.strict_state_refusals("qsdpa-tiling", rec, PROMPT) == []
    rec = record("qsdpa-tiling", "on")
    del rec["final_draft_sidecar"]
    assert H.strict_state_refusals("qsdpa-tiling", rec, PROMPT) == []  # duplicate None agrees


@pytest.mark.parametrize("mechanism, edit, expected", [
    ("qsdpa-tiling", {"target": metadata_unavailable()}, "final_target_state: status 'metadata_unavailable'"),
    ("qsdpa-tiling", {"target": unavailable()}, "final_target_state: status 'unavailable'"),
    ("external-reclaim", {"sidecar": unavailable()}, "final_draft_sidecar: status 'unavailable'"),
    ("external-reclaim", {"sidecar": metadata_unavailable()}, "final_draft_sidecar: status 'metadata_unavailable'"),
    ("external-prefill-reclaim", {"target": unavailable()}, "final_target_state: status 'unavailable'"),
    ("external-prefill-reclaim", {"sidecar": metadata_unavailable()}, "final_draft_sidecar: status"),
])
def test_incomplete_required_snapshot_is_refused(mechanism, edit, expected):
    rec = record(mechanism, H.ARMS[mechanism][0], **edit)
    refusals = H.strict_state_refusals(mechanism, rec, PROMPT)
    assert len(refusals) == 1 and expected in refusals[0], refusals


def test_absent_fields_and_bad_records_refuse_without_crashing():
    rec = record("external-reclaim", "reclaim")
    del rec["final_target_state"], rec["final_draft_sidecar"]
    assert H.strict_state_refusals("external-reclaim", rec, PROMPT) == [
        "final_target_state: absent", "final_draft_sidecar: absent"]
    assert H.strict_state_refusals("external-reclaim", None) == ["not a run record (NoneType)"]
    assert "no strict state requirement" in H.strict_state_refusals("other", {})[0]


@pytest.mark.parametrize("key", ["final_target_state", "final_draft_sidecar"])
def test_duplicate_hash_contradiction_is_refused(key):
    rec = record("external-reclaim", "reclaim")
    assert H.strict_state_refusals("external-reclaim", rec, PROMPT) == []
    rec[f"{key}_sha256"] = _h("other")
    assert H.strict_state_refusals("external-reclaim", rec, PROMPT) == [
        f"{key}: {key}_sha256 contradicts the digest"]
    rec[f"{key}_sha256"] = None
    assert H.strict_state_refusals("external-reclaim", rec, PROMPT)
    del rec[f"{key}_sha256"]  # absent duplicate: nothing to contradict
    assert H.strict_state_refusals("external-reclaim", rec, PROMPT) == []


def test_ordinary_sidecar_duplicate_contradiction_is_refused():
    rec = record("qsdpa-tiling", "off")
    rec["final_draft_sidecar_sha256"] = _h("phantom")  # the digest says sha256 None
    expected = ["final_draft_sidecar: final_draft_sidecar_sha256 contradicts the digest (not required complete)"]
    assert H.strict_state_refusals("qsdpa-tiling", rec, PROMPT) == expected
    del rec["final_draft_sidecar"]  # an absent snapshot cannot have a hash either
    assert H.strict_state_refusals("qsdpa-tiling", rec, PROMPT) == expected


def _bad_boundaries():
    good = boundary()
    return [
        (None, "prompt_boundary: absent"),
        ([good], "prompt_boundary: not a mapping (list)"),
        ({k: v for k, v in good.items() if k != "sidecar"}, "prompt_boundary: fields"),
        ({**good, "tokens": PROMPT[:COVERED]}, "prompt_boundary: fields"),
        ({**good, "covered_tokens": 0}, "covered_tokens is not a positive int"),
        ({**good, "covered_tokens": -1}, "covered_tokens is not a positive int"),
        ({**good, "covered_tokens": True}, "covered_tokens is not a positive int"),
        ({**good, "covered_tokens": str(COVERED)}, "covered_tokens is not a positive int"),
        ({**good, "covered_tokens": None}, "covered_tokens is not a positive int"),
        ({**good, "tokens_sha256": good["tokens_sha256"].upper()}, "tokens_sha256 is not lowercase"),
        ({**good, "tokens_sha256": None}, "tokens_sha256 is not lowercase"),
        (boundary(covered=len(PROMPT) + 1, tokens=PROMPT + [0]), "is not the 39-token committed prefix"),
        (boundary(covered=len(PROMPT)), "is not the 39-token committed prefix"),  # anchor included
        (boundary(covered=COVERED - 1), "is not the 39-token committed prefix"),  # shorter prefix
        ({**good, "covered_tokens": COVERED - 1}, "is not the 39-token committed prefix"),
        (boundary(tokens=PROMPT[1:COVERED + 1]), "not the hash of the committed prefix"),
        (boundary(tokens=PROMPT[:COVERED - 1] + [0]), "not the hash of the committed prefix"),
        (boundary(target=unavailable()), "prompt_boundary.target: status 'unavailable'"),
        (boundary(target=metadata_unavailable()), "prompt_boundary.target: status 'metadata_unavailable'"),
        (boundary(sidecar=metadata_unavailable()), "prompt_boundary.sidecar: status"),
        (boundary(sidecar={**complete(), "reason": "x"}), "prompt_boundary.sidecar: complete with a missing"),
        ({**good, "target": None}, "prompt_boundary.target: not a digest record"),
    ]


@pytest.mark.parametrize("bad, expected", _bad_boundaries())
def test_prefill_reclaim_prompt_boundary_is_required_and_well_formed(bad, expected):
    rec = record("external-prefill-reclaim", "reference")
    rec["prompt_boundary"] = bad
    refusals = H.strict_state_refusals("external-prefill-reclaim", rec, PROMPT)
    assert any(expected in r for r in refusals), refusals


def test_boundary_needs_the_pinned_prompt():
    rec = record("external-prefill-reclaim", "reference")
    assert H.strict_state_refusals("external-prefill-reclaim", rec, PROMPT) == []
    assert H.strict_state_refusals("external-prefill-reclaim", rec) == [
        "prompt_boundary: no pinned prompt to bind the boundary against"]
    assert H.strict_state_refusals("external-prefill-reclaim", rec, PROMPT + [5])  # another prompt


def test_missing_boundary_refuses_both_boundary_snapshots():
    rec = record("external-prefill-reclaim", "reference")
    rec["prompt_boundary"] = None
    assert H.strict_state_refusals("external-prefill-reclaim", rec, PROMPT) == [
        "prompt_boundary: absent (no prompt boundary was taken)",
        "prompt_boundary.target: absent", "prompt_boundary.sidecar: absent"]
    del rec["prompt_boundary"]
    assert len(H.strict_state_refusals("external-prefill-reclaim", rec, PROMPT)) == 3


def test_boundary_not_required_off_the_prefill_mechanism():
    rec = record("external-reclaim", "reclaim", prompt_boundary=None)
    assert H.strict_state_refusals("external-reclaim", rec, PROMPT) == []


# ------------------------------------------- substituted run_cohort orchestration

@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_default_mode_keeps_the_diagnostic_verdict(monkeypatch, mechanism):
    # Matching incomplete snapshots still pass on token parity, labelled state_only.
    fields = {"target": metadata_unavailable(), "sidecar": metadata_unavailable()}
    if mechanism == "external-prefill-reclaim":
        fields["prompt_boundary"] = boundary(target=metadata_unavailable(), sidecar=unavailable())
    out = run(monkeypatch, mechanism, same(mechanism, **fields), strict=False)
    assert out["verdict"] == "pass" and out["refusals"] == [] and out["mismatches"] == []
    assert out["state_comparison"] == {"final_target_state": "state_only (metadata unavailable)",
                                       "final_draft_sidecar": "state_only (metadata unavailable)"}
    assert out["strict_state"]["requested"] is False
    assert out["identity"]["actual_native_run"] is False


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_strict_complete_matching_runs_pass(monkeypatch, mechanism):
    out = run(monkeypatch, mechanism, same(mechanism), strict=True, pairs=2)
    assert out["verdict"] == "pass", out["refusals"] + out["mismatches"]
    gate = out["strict_state"]
    assert gate["requested"] is True and gate["refusals"] == []
    assert gate["required_snapshots"] == list(H.REQUIRED_STATE[mechanism])
    assert gate["measured_runs_checked"] == gate["complete_runs"] == 4
    assert "not native, model, transaction or serving qualification" in gate["scope"]
    assert "not RNG, scheduler or full transaction state" in gate["scope"]
    assert gate["not_required"] == (["final_draft_sidecar"] if mechanism == "qsdpa-tiling" else [])
    expected_sidecar = "not required; unavailable" if mechanism == "qsdpa-tiling" else "compared"
    assert out["state_comparison"]["final_draft_sidecar"] == expected_sidecar
    assert out["state_comparison"]["final_target_state"] == "compared"


def test_strict_ordinary_absent_sidecar_passes_and_is_labelled(monkeypatch):
    def make(arm, call):
        rec = record("qsdpa-tiling", arm)
        del rec["final_draft_sidecar"], rec["final_draft_sidecar_sha256"]
        return rec

    out = run(monkeypatch, "qsdpa-tiling", make, strict=True)
    assert out["verdict"] == "pass", out["refusals"]
    assert out["state_comparison"]["final_draft_sidecar"] == "not required; absent or malformed, not compared"


def _incomplete(mechanism, kind):
    bad = metadata_unavailable() if kind == "metadata_unavailable" else unavailable()
    return {"target": bad} if mechanism == "qsdpa-tiling" else {"sidecar": bad}


@pytest.mark.parametrize("mechanism", MECHANISMS)
@pytest.mark.parametrize("kind", ["metadata_unavailable", "unavailable"])
@pytest.mark.parametrize("which", ["one arm", "both arms"])
def test_strict_refuses_incomplete_even_when_tokens_match(monkeypatch, mechanism, kind, which):
    bad_arm = H.ARMS[mechanism][1]

    def make(arm, call):
        hit = which == "both arms" or arm == bad_arm
        return record(mechanism, arm, **(_incomplete(mechanism, kind) if hit else {}))

    default = run(monkeypatch, mechanism, make, strict=False)
    strict = run(monkeypatch, mechanism, make, strict=True)
    assert strict["verdict"] == "refused"
    assert strict["refusals"] and all("strict state" in r for r in strict["refusals"])
    assert len(strict["refusals"]) == (2 if which == "both arms" else 1)
    assert strict["mismatches"] == default["mismatches"]  # counterexamples unchanged
    assert default["refusals"] == []
    assert default["verdict"] == ("pass" if which == "both arms" else "counterexample")
    assert strict["strict_state"]["complete_runs"] == (0 if which == "both arms" else 1)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_strict_complete_mismatch_stays_a_counterexample(monkeypatch, mechanism):
    def make(arm, call):
        return record(mechanism, arm, target=complete(f"target-{arm}"))

    out = run(monkeypatch, mechanism, make, strict=True)
    assert out["verdict"] == "counterexample" and out["refusals"] == []
    assert any("final_target_state_sha256 differs" in m for m in out["mismatches"])


@pytest.mark.parametrize("field", ["target", "sidecar"])
def test_strict_two_arm_boundary_difference_is_detected(monkeypatch, field):
    mechanism = "external-prefill-reclaim"

    def make(arm, call):
        if arm != "prefill_reclaim":
            return record(mechanism, arm)
        return record(mechanism, arm, prompt_boundary=boundary(**{field: complete("other")}))

    out = run(monkeypatch, mechanism, make, strict=True)
    assert out["verdict"] == "counterexample" and out["refusals"] == []
    assert any("prompt boundary differs" in m for m in out["mismatches"])


@pytest.mark.parametrize("both", [False, True])
def test_strict_wrong_boundary_claim_is_refused_even_when_identical(monkeypatch, both):
    mechanism = "external-prefill-reclaim"

    def make(arm, call):
        wrong = both or arm == "prefill_reclaim"
        return record(mechanism, arm, prompt_boundary=boundary(covered=COVERED - 1) if wrong else None)

    out = run(monkeypatch, mechanism, make, strict=True)
    assert out["verdict"] == "refused"
    assert any("committed prefix" in r for r in out["refusals"])
    assert any("prompt boundary differs" in m for m in out["mismatches"]) is not both
    assert run(monkeypatch, mechanism, make, strict=False)["verdict"] == ("pass" if both else "counterexample")


def test_strict_missing_boundary_key_refuses_without_crashing(monkeypatch):
    mechanism = "external-prefill-reclaim"

    def make(arm, call):
        rec = record(mechanism, arm)
        if arm == "prefill_reclaim":
            del rec["prompt_boundary"]
        return rec

    out = run(monkeypatch, mechanism, make, strict=True)
    assert out["verdict"] == "refused"
    assert any("prompt_boundary: absent" in r for r in out["refusals"])
    assert any("prompt boundary differs" in m for m in out["mismatches"])


def test_strict_label_never_says_compared_for_an_invalid_complete_digest(monkeypatch):
    bad = {**complete(), "sha256": complete()["sha256"].upper()}
    out = run(monkeypatch, "external-reclaim", same("external-reclaim", target=bad), strict=True)
    assert out["verdict"] == "refused"
    assert out["state_comparison"]["final_target_state"] == (
        "refused by the strict state gate (incomplete or malformed)")
    assert run(monkeypatch, "external-reclaim", same("external-reclaim", target=bad),
               strict=False)["state_comparison"]["final_target_state"] == "compared"


@pytest.mark.parametrize("mechanism", MECHANISMS)
@pytest.mark.parametrize("bad", [None, {}, {"status": ["complete"]}, "complete"])
def test_strict_malformed_records_refuse_without_crashing(monkeypatch, mechanism, bad):
    def make(arm, call):
        rec = record(mechanism, arm)
        if arm == H.ARMS[mechanism][1]:
            rec["final_target_state"] = bad
            del rec["final_target_state_sha256"]
        return rec

    out = run(monkeypatch, mechanism, make, strict=True)
    assert out["verdict"] == "refused"
    assert any("strict state: final_target_state" in r for r in out["refusals"])
    assert out["state_comparison"]["final_target_state"] == (
        "refused by the strict state gate (incomplete or malformed)")


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_strict_gate_cannot_erase_token_mismatch(monkeypatch, mechanism):
    def make(arm, call):
        n = 300 if mechanism == "external-reclaim" else 8
        tokens = list(range(n)) if arm == H.ARMS[mechanism][0] else list(range(n - 1)) + [999]
        return record(mechanism, arm, tokens=tokens)

    out = run(monkeypatch, mechanism, make, strict=True)
    assert out["verdict"] == "counterexample" and out["refusals"] == []
    assert any("tokens differ at 7" in m or "tokens differ at 299" in m for m in out["mismatches"])

    def incomplete(arm, call):
        return {**make(arm, call), "final_target_state": unavailable(), "final_target_state_sha256": None}

    out = run(monkeypatch, mechanism, incomplete, strict=True)
    assert out["verdict"] == "refused"  # and the token mismatch is still recorded
    assert any("tokens differ at" in m for m in out["mismatches"])


def test_strict_gate_cannot_erase_logprob_mismatch(monkeypatch):
    def make(arm, call):
        rows = [_h(f"row{i}-{arm if i == 3 else ''}") for i in range(8)]
        return record("qsdpa-tiling", arm, rows=rows)

    out = run(monkeypatch, "qsdpa-tiling", make, strict=True)
    assert out["verdict"] == "counterexample" and out["refusals"] == []
    assert any("logprob rows differ" in m for m in out["mismatches"])


@pytest.mark.parametrize("mechanism, arm, edit, reason", [
    ("qsdpa-tiling", "on", {"counters": {"composed_tiled_calls": 0}}, "tiling not engaged on the on arm"),
    ("qsdpa-tiling", "off", {"rows": [_h("r")]}, "fewer logprob rows than requested"),
    ("external-reclaim", "reclaim", {"tokens": list(range(255))}, "too short to cross the 256-response"),
    ("external-reclaim", "control", {"counters": {"suppressed_crossings": 0}}, "control arm reclaimed"),
    ("external-prefill-reclaim", "reference", {"counters": {"proposed_tokens": 0}}, "never proposed"),
    ("external-prefill-reclaim", "prefill_reclaim", {"counters": {"external_prefill_allocator_reclaims": 4}},
     "prefill reclaim ran 4 times over 5 chunks"),
    ("external-reclaim", "control", {"failures": ["lane 0 dropped"]}, "lane failed: lane 0 dropped"),
])
def test_strict_gate_cannot_erase_engagement_or_lane_refusals(monkeypatch, mechanism, arm, edit, reason):
    def make(a, call):
        return record(mechanism, a, **(edit if a == arm else {}))

    for strict in (False, True):
        out = run(monkeypatch, mechanism, make, strict=strict)
        assert out["verdict"] == "refused"
        assert any(reason in r for r in out["refusals"]), out["refusals"]
        assert not any("strict state" in r for r in out["refusals"])


def test_warmups_are_discarded_and_not_checked(monkeypatch):
    mechanism = "external-prefill-reclaim"

    def make(arm, call):
        rec = record(mechanism, arm)
        if call < 2:  # the warmup pair: malformed state, discarded unchecked
            rec.update(final_target_state=None, prompt_boundary=[], final_draft_sidecar=unavailable())
        return rec

    out = run(monkeypatch, mechanism, make, strict=True, warmups=1)
    assert out["_calls"] == ["reference", "prefill_reclaim", "reference", "prefill_reclaim"]
    assert out["verdict"] == "pass", out["refusals"]
    assert out["strict_state"]["measured_runs_checked"] == 2
    assert out["strict_state"]["warmups"] == "discarded diagnostic work; not checked"


@pytest.mark.parametrize("strict", [False, True])
def test_no_measured_runs_is_refused(monkeypatch, strict):
    out = run(monkeypatch, "external-reclaim", same("external-reclaim"), strict=strict,
              edit_args=lambda a: setattr(a, "pairs", 0))
    assert out["verdict"] == "refused" and out["runs"] == []
    if strict:
        assert out["refusals"] == ["strict state: no measured runs to check"]
        assert out["strict_state"]["measured_runs_checked"] == 0
    else:
        assert out["refusals"] == []


def test_cli_flag_defaults_off_and_sets_on():
    assert H.build_parser().get_default("require_complete_state") is False
    assert _args("qsdpa-tiling").require_complete_state is False
    assert _args("qsdpa-tiling", "--require-complete-state").require_complete_state is True
    real = H.resolve_args(H.build_parser(), ["--mechanism", "external-reclaim", "--i-own-the-gpu",
                                              "--model", "/nonexistent", "--policy", "/nonexistent",
                                              "--out", "/dev/null", "--require-complete-state"])
    assert real.require_complete_state is True


def test_long_snapshot_status_keeps_refusal_reason():
    message = H.snapshot_refusal({"status": "x" * 400})
    assert "is not complete" in message
    assert len(message) <= 200
