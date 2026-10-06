"""CPU/static gates for the Muse Glimmer strategy transfer."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.runtime.exact_prefix_cascade import (
    longest_first_paths,
    next_cascade_stage,
)
from mlx2.runtime.prefill_plan import prompt_length_prefill_step


def test_longest_first_is_stable_and_deduplicates_complete_paths():
    paths = [(1,), (2, 3, 4), (5, 6), (2, 3, 4), (7, 8)]
    assert longest_first_paths(paths) == (
        (2, 3, 4),
        (5, 6),
        (7, 8),
        (1,),
    )


def test_accepted_prefix_prunes_impossible_siblings_and_returns_only_suffix():
    paths = [(1, 2, 3, 4), (1, 2, 9), (1, 7, 8, 9), (6, 7)]
    first = next_cascade_stage(paths)
    assert first.path == (1, 2, 3, 4) and first.suffix == first.path
    second = next_cascade_stage(paths, (1, 2), attempted=(0,))
    assert second.path == (1, 2, 9) and second.suffix == (9,)
    assert second.viable_indices == (2,)
    assert set(second.pruned_indices) == {1, 3}


def test_no_sibling_survives_a_nonexistent_prefix():
    assert next_cascade_stage([(1, 2), (1, 3)], (9,)) is None


@pytest.mark.parametrize("paths", [[], [()], [(1, -1)], [(True, 2)]])
def test_invalid_cascade_paths_fail_closed(paths):
    with pytest.raises(ValueError):
        longest_first_paths(paths)


def test_muse_owns_prefill_while_generic_schedule_remains_the_decline_fallback():
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter

    adapter = object.__new__(MuseGlimmerAdapter)
    assert adapter.prefill_step_default() == 2048
    assert prompt_length_prefill_step(32768) == 512
    assert prompt_length_prefill_step(32769) == 2048
    assert prompt_length_prefill_step(65537) == 8192


def test_muse_contract_keeps_proposals_and_multimodal_state_non_authoritative():
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter

    adapter = object.__new__(MuseGlimmerAdapter)
    contract = adapter.exact_prefix_cascade_contract()
    assert contract["verification_order"] == "longest_first"
    assert contract["invalid_sibling_pruning"] is True
    assert contract["accepted_prefix_state"] == "canonical_ordinary_replay"
    assert contract["shared_prefix_reuse"].startswith("suffix_only")
    assert contract["transactional_multirow_state_reuse"] is False
    assert contract["apcv2_publication"] is False
    assert contract["proposal_sources"] == {
        "assistant": {"role": "proposal_only", "implementation": "metadata_only"},
        "dflash2": {
            "role": "proposal_only",
            "implementation": "external_draft_integrated",
        },
    }
    assert contract["state_separation"]["multimodal"] == (
        "not_admitted_by_text_adapter"
    )
    assert contract["qualified"] is contract["selected"] is False
    stage = adapter.plan_exact_prefix_cascade(
        [(1, 2, 3), (1, 7), (1, 2, 9)], (1, 2), attempted=(0,)
    )
    assert stage.path == (1, 2, 9) and stage.suffix == (9,)


def test_static_plan_binds_q8_target_and_proposal_roles(tmp_path):
    target = tmp_path / "target"
    assistant = tmp_path / "assistant"
    dflash = tmp_path / "dflash"
    for path in (target, assistant, dflash):
        path.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "muse_glimmer",
                "text_config": {
                    "num_hidden_layers": 52,
                    "max_position_embeddings": 131072,
                    "sliding_window": 2048,
                },
                "quantization": {"bits": 8, "group_size": 64},
            }
        )
    )
    (assistant / "config.json").write_text(
        json.dumps(
            {
                "model_type": "muse_glimmer_assistant",
                "architectures": ["MuseGlimmerAssistantModel"],
                "block_size": 16,
            }
        )
    )
    (dflash / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["DFlash2DraftModel"],
                "dflash_config": {"block_size": 16},
            }
        )
    )
    out = tmp_path / "plan.json"
    script = Path(__file__).parents[1] / (
        "scripts/research/stage_muse_glimmer_strategy_transfer.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(target),
            "--proposal",
            str(assistant),
            "--proposal",
            str(dflash),
            "--out",
            str(out),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(out.read_text())
    assert plan["target"]["quantization_bits"] == 8
    assert [item["kind"] for item in plan["proposal_sources"]] == [
        "assistant",
        "dflash2",
    ]
    assert all(item["role"] == "proposal_only" for item in plan["proposal_sources"])
    assert plan["candidate"]["transactional_multirow_state_reuse"] is False
    assert plan["gpu_executed"] is False

    wrong_target = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(assistant),
            "--out",
            str(tmp_path / "wrong-target.json"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert wrong_target.returncode != 0
    wrong_proposal = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(target),
            "--proposal",
            str(target),
            "--out",
            str(tmp_path / "wrong-proposal.json"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert wrong_proposal.returncode != 0
