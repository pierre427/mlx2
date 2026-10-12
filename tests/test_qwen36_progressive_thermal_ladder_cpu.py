"""CPU/static contracts for the Qwen3.6 progressive thermal ladder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "qualification"
    / "runs"
    / "speculative-composition-qualification-20261012"
    / "diagnose_qwen36_progressive_thermal_ladder.py"
)
pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(), reason="private Qwen3.6 thermal harness is not exported"
)


def _module():
    spec = importlib.util.spec_from_file_location("qwen36_progressive_ladder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "execute": False,
        "max_new_tokens": 128,
        "max_length": 262144,
        "cell_gap_seconds": 5.0,
        "output_dir": None,
        "i_own_the_gpu": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_plan_is_single_session_single_rep_diagnostic_performance_only():
    module = _module()
    plan = module._plan(_args())
    assert plan["schema"] == "mlx2.qwen36-progressive-thermal-context-ladder.v1"
    assert plan["diagnostic_only"] is True
    assert plan["performance_experiment"] is True
    assert plan["qualification_claim"] is False
    assert plan["preflight_run"] is False
    assert plan["production_gate_changed"] is False
    assert plan["measurement"] == {
        "one_model_and_draft_load": True,
        "one_request_at_a_time": True,
        "batched": False,
        "width": 1,
        "measured_repetitions_per_cell": 1,
        "unmeasured_warmup": False,
        "cell_gap_seconds": 5.0,
        "timings": [
            "time_to_first_token_seconds",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "total_wall_seconds",
        ],
        "prefill_rate_definition": "rendered prompt tokens / time to first token",
        "decode_rate_definition": "tokens after first / time from first to final token",
    }
    assert [cell["requested_context_tokens"] for cell in plan["cells"]] == list(
        module.LENGTHS
    )
    assert all(cell["width"] == 1 for cell in plan["cells"])
    assert all(cell["measured_repetitions"] == 1 for cell in plan["cells"])
    assert all(cell["max_new_tokens"] == 128 for cell in plan["cells"])
    assert plan["external_policy"]["proposal_composition"] is False
    assert plan["external_policy"]["pairwise_selection"] == "host"
    assert plan["external_policy"]["progressive_verification_tile"] == 2
    assert plan["external_policy"]["progressive_verification_fidelity"] == (
        "near_tie_bounded"
    )
    assert len(plan["external_policy"]["target_revision"]) == 64
    assert len(plan["external_policy"]["draft_revision"]) == 64


def test_last_rung_reserves_output_headroom_inside_model_context():
    module = _module()
    plan = module._plan(_args())
    last = plan["cells"][-1]
    assert last["requested_context_tokens"] == 262144
    assert last["effective_prompt_tokens"] == 262144 - 256
    assert last["effective_prompt_tokens"] + 128 <= plan["target_max_context"]


def test_plan_binds_portable_three_sample_thermal_policy():
    module = _module()
    thermal = module._plan(_args())["thermal_policy"]
    assert thermal["required_thermal_state"] == 0
    assert thermal["consecutive_samples"] == 3
    assert thermal["sample_interval_seconds"] == 15
    assert thermal["max_temperature_delta_c"] == 0.5


class _Tokenizer:
    eos_token_ids = {2}

    def apply_chat_template(self, messages, *, tokenize, **_kwargs):
        assert tokenize is False
        return "CHAT:" + messages[0]["content"] + ":ASSISTANT"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_prompt_builder_hits_exact_length_and_binds_fresh_needle():
    module = _module()
    tokens, recipe = module._prompt_tokens(_Tokenizer(), 1024, rung=1024)
    assert len(tokens) == 1024
    assert recipe["prompt_tokens"] == 1024
    assert recipe["needle"].startswith("NEEDLE-1024-")
    assert len(recipe["prompt_token_ids_sha256"]) == 64
    decoded = "".join(chr(token) for token in tokens)
    assert recipe["needle"] in decoded


def test_progressive_stats_report_real_engagement_and_acceptance():
    module = _module()
    before = {
        "proposed_tokens": 20,
        "accepted_proposals": 10,
        "external_progressive_verify_rounds": 5,
        "external_progressive_verify_launches": 8,
    }
    after = {
        "proposed_tokens": 32,
        "accepted_proposals": 19,
        "external_progressive_verify_rounds": 9,
        "external_progressive_verify_launches": 15,
        "external_progressive_verify_full_tiles": 3,
    }
    delta = module._numeric_delta(before, after)
    engagement = module._progressive_engagement(delta)
    assert engagement["proposed_tokens"] == 12
    assert engagement["accepted_proposals"] == 9
    assert engagement["acceptance_rate"] == 0.75
    assert engagement["external_progressive_verify_rounds"] == 4
    assert engagement["external_progressive_verify_launches"] == 7
    assert engagement["observed_used"] is True


def test_post_thermal_samples_again_only_after_non_nominal_sample():
    module = _module()
    rows = iter([{"thermal_state": 1}, {"thermal_state": 0}])

    class Matrix:
        @staticmethod
        def sample_thermal(_command):
            return next(rows)

        @staticmethod
        def thermally_stable(sample, _policy):
            return sample["thermal_state"] == 0

    sleeps = []
    result = module._post_thermal(
        Matrix,
        {"sample_interval_seconds": 15},
        sleep=sleeps.append,
    )
    assert result["passed"] is True
    assert len(result["samples"]) == 2
    assert sleeps == [15.0]


def test_execute_fails_closed_without_explicit_gpu_ownership():
    module = _module()
    args = _args(execute=True, output_dir=Path("unused"))
    with pytest.raises(ValueError, match="--i-own-the-gpu"):
        module._execute(args, module._plan(args))
