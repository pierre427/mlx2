from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualify_prefill_handoff", ROOT / "scripts" / "qualify_prefill_handoff.py"
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def _g2_cell(
    *, floor_mean=0.0, floor_max=0.0, arm_mean=0.0, arm_max=0.0, agreement=1.0,
    floor_top1=0.99,
):
    return {
        "floor": {"kl_mean": floor_mean, "kl_max": floor_max, "top1_agreement": floor_top1},
        "arm_teacher_forced": {
            "kl_mean": arm_mean,
            "kl_max": arm_max,
            "top1_agreement": agreement,
        },
    }


def test_g2_uses_absolute_floors_when_floor_is_zero():
    result = HARNESS.evaluate_g2(_g2_cell(arm_mean=0.002, arm_max=0.05, agreement=0.98))
    assert result["pass"] is True
    assert result["mean_kl_limit"] == pytest.approx(0.002)
    assert result["max_step_kl_limit"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("changes", "failed_check"),
    [
        ({"arm_mean": 0.00200001}, "mean_kl"),
        ({"arm_max": 0.05000001}, "max_step_kl"),
        ({"agreement": 0.979999}, "top1_agreement"),
    ],
)
def test_g2_rejects_values_just_over_each_bound(changes, failed_check):
    result = HARNESS.evaluate_g2(_g2_cell(**changes))
    assert result["pass"] is False
    assert result["checks"][failed_check] is False


def test_g2_scales_to_four_times_nonzero_floor():
    result = HARNESS.evaluate_g2(
        _g2_cell(floor_mean=0.01, floor_max=0.02, arm_mean=0.04, arm_max=0.08)
    )
    assert result["pass"] is True
    assert result["mean_kl_limit"] == pytest.approx(0.04)
    assert result["max_step_kl_limit"] == pytest.approx(0.08)


def _divergence(**overrides):
    value = {
        "shared_prefix": True,
        "ref_top2": [7, 9],
        "arm_top2": [9, 7],
        "ref_margin": 0.5,
        "arm_margin": 0.49,
    }
    value.update(overrides)
    return value


def test_near_tie_accepts_same_unordered_top_two_at_margin_ceiling():
    assert HARNESS.near_tie_passes(_divergence(), functional_pass=True) is True


def test_near_tie_rejects_different_unordered_top_two():
    assert (
        HARNESS.near_tie_passes(_divergence(arm_top2=[9, 11]), functional_pass=True)
        is False
    )


def test_near_tie_rejects_margin_just_over_half_nat():
    assert (
        HARNESS.near_tie_passes(_divergence(arm_margin=0.5000001), functional_pass=True)
        is False
    )


def test_near_tie_requires_shared_prefix_and_functional_oracle():
    assert (
        HARNESS.near_tie_passes(_divergence(shared_prefix=False), functional_pass=True)
        is False
    )
    assert HARNESS.near_tie_passes(_divergence(), functional_pass=False) is False


def test_loop_detector_finds_three_occurrences_of_a_32_token_window():
    window = list(range(32))
    assert HARNESS.has_repetition_loop(window + [99] + window + [98] + window)
    assert not HARNESS.has_repetition_loop(window + [99] + window)


def test_kl_helper_matches_direct_probability_calculation():
    ref = np.array([0.0, 1.0, -1.0])
    other = np.array([0.2, 0.4, -0.3])
    p = np.exp(ref - np.log(np.exp(ref).sum()))
    q = np.exp(other - np.log(np.exp(other).sum()))
    expected = float(np.sum(p * np.log(p / q)))
    assert HARNESS.kl_divergence(ref, other) == pytest.approx(expected)
    assert HARNESS.kl_divergence(ref, ref) == pytest.approx(0.0, abs=1e-15)


def test_evidence_directory_refuses_nonempty_without_resume(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "prior.json").write_text("{}")
    with pytest.raises(RuntimeError, match="nonempty evidence"):
        HARNESS.prepare_evidence_dir(evidence, "compare", resume=False)


def test_evidence_directory_resume_requires_matching_step(tmp_path):
    evidence = tmp_path / "evidence"
    HARNESS.prepare_evidence_dir(evidence, "compare", resume=False)
    HARNESS.prepare_evidence_dir(evidence, "compare", resume=True)
    with pytest.raises(RuntimeError, match="another step"):
        HARNESS.prepare_evidence_dir(evidence, "verdict", resume=True)


def test_imported_teacher_forcing_only_forwards_reserved_and_reference_tokens(
    monkeypatch,
):
    class Model:
        def __init__(self):
            self.forwarded = []

        def __call__(self, values, *, cache):
            del cache
            self.forwarded.extend(np.asarray(values).reshape(-1).tolist())
            return np.array([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])

    fake_mx = SimpleNamespace(
        array=np.asarray,
        float32=np.float32,
        eval=lambda _value: None,
    )
    monkeypatch.setattr(
        HARNESS,
        "_handoff",
        lambda: SimpleNamespace(
            prefill_prefix=lambda *_args, **_kwargs: pytest.fail(
                "imported arm must not prefill the prompt prefix"
            )
        ),
    )
    model = Model()
    reference = [4, 5, 6]
    logits = [np.arange(7, dtype=np.float32)] * len(reference)
    HARNESS._teacher_forced_imported(
        model,
        [],
        [1, 2, 3],
        {"prefix_tokens": 2},
        reference,
        logits,
        fake_mx,
    )
    assert model.forwarded == [3, 4, 5]
    assert len(model.forwarded) == 1 + len(reference) - 1


def test_cell_builder_keeps_exact_user_messages_used_for_content():
    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return "<chat>" + messages[0]["content"] + "</chat>"

        def encode(self, text, *, add_special_tokens=False):
            del add_special_tokens
            return [ord(value) for value in text]

        def decode(self, values, **_kwargs):
            return "".join(chr(value) for value in values)

    tokenizer = Tokenizer()
    target = 300
    tokens, needle, messages = HARNESS._build_cell_tokens(
        tokenizer, list(range(65, 91)) * 20, "needle", target, 1
    )
    assert len(tokens) == target
    assert messages == [{"role": "user", "content": messages[0]["content"]}]
    assert "<|MLX2_CONTEXT_FILLER|>" not in messages[0]["content"]
    assert needle in messages[0]["content"]
    assert tokenizer.encode(
        tokenizer.apply_chat_template(messages), add_special_tokens=False
    ) == tokens


def _verdict_cell(cell_id, prompt_tokens, *, g2_pass=True):
    arm_mean = 0.0 if g2_pass else 0.01
    return {
        "cell_id": cell_id,
        "kind": "general",
        "prompt_tokens": prompt_tokens,
        "plan": {"chunk": 512},
        "plan_digest": "plan-digest",
        "manifest": {"token_digest": "token-digest"},
        "needle_ref": None,
        "needle_arm": None,
        "greedy_equal": True,
        **_g2_cell(arm_mean=arm_mean),
    }


def _handoff_contract(monkeypatch):
    monkeypatch.setattr(
        HARNESS,
        "_handoff",
        lambda: SimpleNamespace(
            OPERATION="prefill-handoff-approx-v1",
            OPERATION_REVISION="revision",
            RECORD_SCHEMA="mlx2.approximate-qualification.v1",
        ),
    )


def _comparison(*cells):
    return {
        "artifact_fingerprint": "f" * 64,
        "artifact_name": "artifact",
        "model_checkpoint": "artifact@ffffffff",
        "receiver": {"runtime_digest": "receiver", "host": "m5"},
        "cells": [
            {**cell, "producer": {"runtime_digest": "producer", "host": "m3"}}
            for cell in cells
        ],
    }


def _capability():
    return {"runtime_digest": "receiver", "execution_digest": "execution"}


def _served_cell(cell_id, *, g3=True, g4=True, g5=True):
    return {
        "cell_id": cell_id,
        "plan": {
            "token_ids_match": True,
            "response": {
                "token_digest": "token-digest",
                "plan_digest": "plan-digest",
            },
        },
        "gates": {
            "G3": {"pass": g3},
            "G4": {"pass": g4},
            "G5": {"pass": g5},
        },
    }


def _build(comparison, *, provisional, served=None, g1=True):
    return HARNESS.build_verdict(
        comparison,
        g1_pass=g1,
        g1_details={"pass": g1},
        commit="abc",
        g1_path="g1.json",
        receiver_capability=_capability(),
        receiver_capability_path="receiver.json",
        served=served,
        served_path="served.json" if served else None,
        provisional=provisional,
    )


def test_failed_gate_writes_failed_verdict_and_max_uses_passing_cells(monkeypatch):
    _handoff_contract(monkeypatch)
    comparison = {
        "artifact_fingerprint": "f" * 64,
        "artifact_name": "artifact",
        "model_checkpoint": "artifact@ffffffff",
        "receiver": {"runtime_digest": "receiver", "host": "m5"},
        "cells": [
            {
                **_verdict_cell("short", 1500),
                "producer": {"runtime_digest": "producer", "host": "m3"},
            },
            {
                **_verdict_cell("long", 12000, g2_pass=False),
                "producer": {"runtime_digest": "producer", "host": "m3"},
            },
        ],
    }
    served = {
        "cells": [_served_cell("short"), _served_cell("long")],
    }
    record, cells = _build(comparison, provisional=False, served=served)
    assert record["verdict"] == "failed"
    assert record["gates"]["G2"]["pass"] is False
    assert record["max_prompt_tokens"] == 1500
    assert [cell["pass"] for cell in cells] == [True, False]


def test_provisional_excludes_g5_and_binds_execution_digest(monkeypatch):
    _handoff_contract(monkeypatch)
    comparison = _comparison(_verdict_cell("short", 1500))
    record, cells = _build(comparison, provisional=True)
    assert record["verdict"] == "qualified"
    assert record["provisional"] is True
    assert record["gates"]["G5"]["pass"] is None
    assert cells[0]["gates"]["G5"]["status"] == "pending"
    assert record["receiver"]["execution_digest"] == "execution"


def test_final_requires_served_and_combines_both_g3_g4_evaluations(monkeypatch):
    _handoff_contract(monkeypatch)
    comparison = _comparison(_verdict_cell("short", 1500))
    with pytest.raises(ValueError, match="requires served"):
        _build(comparison, provisional=False)
    served = {"cells": [_served_cell("short", g4=False)]}
    record, cells = _build(comparison, provisional=False, served=served)
    assert record["verdict"] == "failed"
    assert cells[0]["gates"]["G3"]["in_process"]["pass"] is True
    assert cells[0]["gates"]["G3"]["served"]["pass"] is True
    assert cells[0]["gates"]["G4"]["in_process"]["pass"] is True
    assert cells[0]["gates"]["G4"]["served"]["pass"] is False


def _logprob(token_id, first, second, margin=0.2):
    return {
        "id": token_id,
        "logprob": 0.0 if token_id == first else -margin,
        "top_logprobs": [
            {"id": first, "logprob": 0.0},
            {"id": second, "logprob": -margin},
        ],
    }


def test_served_g4_near_tie_uses_logprob_entries_at_different_token_boundary():
    plan = HARNESS._handoff().make_plan(3, 2)
    cell = {
        "kind": "general",
        "needle": None,
        "token_ids": [1, 2, 3],
        "plan": plan,
    }
    ordinary = {
        "content": "same rendered text",
        "finish_reason": "stop",
        "token_logprobs": [_logprob(1, 1, 8), _logprob(2, 2, 3)],
        "mlx2": {},
        "mlx2_fabric": {"approximate": False},
    }
    opted = {
        "content": "same rendered text",
        "finish_reason": "stop",
        "token_logprobs": [
            _logprob(1, 1, 8),
            _logprob(3, 3, 2),
            _logprob(2, 2, 4),
        ],
        "mlx2": {
            "prefill_handoff": {
                "observed_used": True,
                "fidelity": "approximate",
                "record_id": "record",
                "record_state": "provisional",
            }
        },
        "mlx2_fabric": {
            "kind": "approximate_prefill_offload",
            "approximate": True,
            "states": {"observed_used": True, "qualified": False},
            "served_by": {"nid": "receiver"},
            "prefill_handoff": {
                "producer": "producer",
                "record_id": "record",
                "record_state": "provisional",
            },
        },
    }
    gates = HARNESS.evaluate_served_cell(
        cell,
        opted,
        ordinary,
        producer_nid="producer",
        receiver_nid="receiver",
        record_id="record",
        plan_evidence={
            "token_ids_match": True,
            "response": {
                "token_digest": HARNESS._handoff().token_digest([1, 2]),
                "plan_digest": HARNESS._handoff().plan_digest(plan),
            },
        },
    )
    assert gates["G4"]["pass"] is True
    assert gates["G4"]["classification"] == "near_tie"
    assert gates["G4"]["first_divergence"]["index"] == 1
    assert gates["G5"]["pass"] is True


def test_served_g5_fails_tokenization_or_plan_evidence_mismatch():
    plan = HARNESS._handoff().make_plan(3, 2)
    cell = {
        "kind": "general",
        "needle": None,
        "token_ids": [1, 2, 3],
        "plan": plan,
    }
    ordinary = {
        "content": "ok",
        "finish_reason": "stop",
        "token_logprobs": [],
        "mlx2": {},
        "mlx2_fabric": {"approximate": False},
    }
    opted = {
        "content": "ok",
        "finish_reason": "stop",
        "token_logprobs": [],
        "mlx2": {"prefill_handoff": {
            "observed_used": True, "fidelity": "approximate",
            "record_id": "record", "record_state": "provisional",
        }},
        "mlx2_fabric": {
            "kind": "approximate_prefill_offload", "approximate": True,
            "states": {"observed_used": True, "qualified": False},
            "served_by": {"nid": "receiver"},
            "prefill_handoff": {
                "producer": "producer", "record_id": "record",
                "record_state": "provisional",
            },
        },
    }
    gates = HARNESS.evaluate_served_cell(
        cell, opted, ordinary, producer_nid="producer", receiver_nid="receiver",
        record_id="record", plan_evidence={
            "token_ids_match": False,
            "response": {"token_digest": "wrong", "plan_digest": "wrong"},
        },
    )
    assert gates["G5"]["pass"] is False
    assert gates["G5"]["checks"]["token_ids_match"] is False
    assert gates["G5"]["checks"]["token_digest_bound"] is False
    assert gates["G5"]["checks"]["plan_digest_bound"] is False


def test_final_g5_rebinds_served_digests_to_compare_evidence(monkeypatch):
    _handoff_contract(monkeypatch)
    source = _verdict_cell("short", 1500)
    served_cell = _served_cell("short")
    served_cell["plan"]["response"]["token_digest"] = "different"
    record, cells = _build(
        _comparison(source),
        provisional=False,
        served={"cells": [served_cell]},
    )
    assert cells[0]["gates"]["G5"]["pass"] is False
    assert cells[0]["gates"]["G5"]["compare_binding"]["token_digest"] is False
    assert record["gates"]["G5"]["pass"] is False


def test_verdict_step_writes_failed_record_for_any_failed_gate(tmp_path, monkeypatch):
    _handoff_contract(monkeypatch)
    cell = {
        **_verdict_cell("short", 1500),
        "producer": {"runtime_digest": "producer"},
    }
    compare = tmp_path / "compare"
    compare.mkdir()
    (compare / "compare.json").write_text(
        json.dumps(
            {
                "artifact_fingerprint": "f" * 64,
                "artifact_name": "artifact",
                "model_checkpoint": "artifact@ffffffff",
                "mlx2_commit": "abc",
                "max_new": 256,
                "receiver": {"runtime_digest": "receiver"},
                "cells": [cell],
            }
        )
    )
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "cells.json").write_text(
        json.dumps([{"cell_id": "short", "prompt_tokens": 1500}])
    )
    g1 = tmp_path / "g1.json"
    g1.write_text('{"pass": false}')
    capability = tmp_path / "receiver.json"
    capability.write_text(json.dumps(_capability()))
    output = tmp_path / "records"
    HARNESS.run_verdict(
        SimpleNamespace(
            record_dir=output,
            resume=False,
            compare=compare,
            cells=cells,
            commit="abc",
            g1_evidence=g1,
            receiver_capability=capability,
            served=None,
            provisional=True,
        )
    )
    records = [path for path in output.glob("*.json")]
    assert len(records) == 1
    assert json.loads(records[0].read_text())["verdict"] == "failed"
    assert "Verdict: **failed**" in (output / "summary.md").read_text()


def test_g2_top1_is_floor_relative_after_the_post_run_amendment():
    # The measured general-01536 cell: floor 0.977, arm 0.977 -> passes now.
    assert HARNESS.evaluate_g2(_g2_cell(agreement=0.977, floor_top1=0.977))["pass"] is True
    result = HARNESS.evaluate_g2(_g2_cell(agreement=0.9659, floor_top1=0.977))
    assert result["checks"]["top1_agreement"] is False
    assert result["top1_limit"] == pytest.approx(0.967)


def _tie_divergence(**changes):
    value = {
        "index": 48,
        "shared_prefix": True,
        "ref_selected": 26012,
        "arm_selected": 561,
        "ref_top2": [26012, 561],
        "arm_top2": [561, 69208],
        "ref_margin": 0.0,
        "arm_margin": 0.125,
    }
    value.update(changes)
    return value


def test_exact_tie_clause_classifies_the_measured_12k_divergence():
    assert HARNESS.near_tie_passes(_tie_divergence(), functional_pass=True) is True
    # The clause needs an exact zero margin on the tied arm ...
    assert HARNESS.near_tie_passes(_tie_divergence(ref_margin=0.0625), functional_pass=True) is False
    # ... the other arm must have selected one of the tied tokens ...
    assert HARNESS.near_tie_passes(_tie_divergence(arm_selected=69208), functional_pass=True) is False
    # ... and the 0.5-nat ceiling, shared prefix and oracle still apply.
    assert HARNESS.near_tie_passes(_tie_divergence(arm_margin=0.51), functional_pass=True) is False
    assert HARNESS.near_tie_passes(_tie_divergence(shared_prefix=False), functional_pass=True) is False
    assert HARNESS.near_tie_passes(_tie_divergence(), functional_pass=False) is False


def _multi_way(**changes):
    # The measured served general-01536 divergence (token ids from the run),
    # with top-5 windows that would make it a three-way near tie.
    value = {
        "index": 37,
        "shared_prefix": True,
        "ref_selected": 888,
        "arm_selected": 32666,
        "ref_top2": [888, 5875],
        "arm_top2": [32666, 888],
        "ref_margin": 0.0,
        "arm_margin": 0.125,
        "ref_window": [[888, -1.0], [5875, -1.0], [32666, -1.25], [11, -3.0], [12, -3.5]],
        "arm_window": [[32666, -0.9], [888, -1.025], [5875, -1.1], [11, -3.0], [12, -3.5]],
    }
    value.update(changes)
    return value


def test_multi_way_clause_classifies_a_three_way_near_tie():
    assert HARNESS.near_tie_passes(_multi_way(), functional_pass=True) is True
    assert HARNESS.near_tie_passes(_multi_way(), functional_pass=False) is False
    assert HARNESS.near_tie_passes(_multi_way(shared_prefix=False), functional_pass=True) is False


def test_multi_way_clause_needs_both_gaps_within_the_ceiling():
    # The arm's pick sits 0.6 nat under the reference's top: not a near tie.
    far = [[888, -1.0], [5875, -1.0], [32666, -1.6], [11, -3.0], [12, -3.5]]
    assert HARNESS.near_tie_passes(_multi_way(ref_window=far), functional_pass=True) is False


def test_multi_way_clause_fails_closed_when_the_window_cannot_decide():
    # The arm's pick is absent from a window whose last entry is still within
    # 0.5 nat of the top: unknown, so unclassified.
    tight = [[888, -1.0], [5875, -1.0], [7, -1.1], [8, -1.2], [9, -1.3]]
    assert HARNESS.near_tie_passes(_multi_way(ref_window=tight), functional_pass=True) is False
    # Absent from a window that already spans more than 0.5 nat: proven too far.
    wide = [[888, -1.0], [5875, -1.0], [7, -1.1], [8, -1.2], [9, -2.0]]
    assert HARNESS.near_tie_passes(_multi_way(ref_window=wide), functional_pass=True) is False


def _memory_fallback(cell):
    cell["opted_in"] = {
        "mlx2_fabric": {
            "kind": "local",
            "fallback": "ordinary_local",
            "approximate": False,
            "candidates": [
                {
                    "option": "approximate_prefill_offload",
                    "verdict": "fallback",
                    "reason": "prefill handoff cache does not fit ordinary execution headroom "
                    "(required 8098242560 bytes, available 7338114080 bytes)",
                }
            ],
        }
    }
    return cell


def test_producer_memory_fallbacks_lower_the_ceiling(monkeypatch):
    """The 2026-10-18 22:14Z run: the M3 producer could not export 12K and
    24K under ordinary admission; Pierre's amendment caps the record instead
    of failing it."""
    _handoff_contract(monkeypatch)
    comparison = _comparison(
        _verdict_cell("g1536", 1536), _verdict_cell("n8192", 8192),
        _verdict_cell("g12300", 12300), _verdict_cell("n24576", 24576),
    )
    served = {"cells": [
        _served_cell("g1536"), _served_cell("n8192"),
        _memory_fallback(_served_cell("g12300", g5=False)),
        _memory_fallback(_served_cell("n24576", g5=False)),
    ]}
    record, cells = _build(comparison, provisional=False, served=served)
    assert record["verdict"] == "qualified"
    assert record["max_prompt_tokens"] == 8192
    assert record["capacity_ceiling"]["above_ceiling"] == ["g12300", "n24576"]
    assert {c["cell_id"]: c["scope"] for c in cells}["n24576"] == "above_ceiling"


def test_other_g5_failures_still_fail_the_record(monkeypatch):
    _handoff_contract(monkeypatch)
    comparison = _comparison(_verdict_cell("g1536", 1536), _verdict_cell("n8192", 8192))
    # A G5 failure that is not a producer memory refusal (e.g. link gate).
    other = _memory_fallback(_served_cell("n8192", g5=False))
    other["opted_in"]["mlx2_fabric"]["candidates"][0]["reason"] = "link_budget_refused: 9.6 ms > 5 ms"
    record, _ = _build(comparison, provisional=False, served={"cells": [_served_cell("g1536"), other]})
    assert record["verdict"] == "failed"
    # A memory fallback below a passing larger cell caps there; the larger
    # passing cell is out of scope, and nothing above the cap can rescue it.
    comparison = _comparison(
        _verdict_cell("g1536", 1536), _verdict_cell("n8192", 8192), _verdict_cell("g12300", 12300)
    )
    served = {"cells": [
        _served_cell("g1536"), _memory_fallback(_served_cell("n8192", g5=False)), _served_cell("g12300"),
    ]}
    record, _ = _build(comparison, provisional=False, served=served)
    assert record["verdict"] == "qualified" and record["max_prompt_tokens"] == 1536
    # A capacity-limited smallest cell leaves no qualified scope at all.
    served = {"cells": [_memory_fallback(_served_cell("g1536", g5=False)), _served_cell("n8192"), _served_cell("g12300")]}
    record, _ = _build(comparison, provisional=False, served=served)
    assert record["verdict"] == "failed" and record["max_prompt_tokens"] == 0
