from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research/probe_dflash_frontier_lookahead.py"
SPEC = importlib.util.spec_from_file_location("dflash_frontier_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def identity(**changes):
    values = {
        "request_id": "request-a",
        "generator_revision": "generator-a",
        "context_revision": 7,
        "round_id": 3,
        "parent_path_digest": probe.path_digest((11, 12, 13)),
        "draft_source_revision": "dflash-a",
        "sampler_revision": "sampler-a",
    }
    values.update(changes)
    return probe.FrontierIdentity(**values)


def test_private_extension_promotes_only_at_exact_identity():
    store = probe.PrivateFrontierStore()
    wanted = identity()
    store.put(probe.PrivateExtension(wanted, (21, 22), state=object()))

    claimed = store.claim(wanted)

    assert claimed is not None
    assert claimed.tokens == (21, 22)
    assert len(store) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generator_revision", "generator-b"),
        ("context_revision", 8),
        ("round_id", 4),
        ("parent_path_digest", "different-parent"),
        ("draft_source_revision", "dflash-b"),
        ("sampler_revision", "sampler-b"),
    ],
)
def test_mismatch_discards_all_private_request_state(field, value):
    store = probe.PrivateFrontierStore()
    store.put(probe.PrivateExtension(identity(), (21,), state="private"))
    store.put(
        probe.PrivateExtension(
            identity(parent_path_digest=probe.path_digest((31, 32))),
            (41,),
            state="sibling",
        )
    )

    assert store.claim(identity(**{field: value})) is None
    assert len(store) == 0


def test_other_requests_survive_an_identity_mismatch():
    store = probe.PrivateFrontierStore()
    store.put(probe.PrivateExtension(identity(), (21,), state="request-a"))
    other = identity(request_id="request-b")
    store.put(probe.PrivateExtension(other, (31,), state="request-b"))

    assert store.claim(identity(round_id=100)) is None
    assert store.claim(other).state == "request-b"


def test_source_audit_fails_closed_without_exact_leaf_state_api():
    result = probe.audit_exact_frontier_capability()

    assert result["dflash_next_tile_requires_target_hidden"] is True
    assert result["current_pipeline_begins_after_commit"] is True
    assert result["supports_exact_frontier_lookahead"] is False
    assert "no_exact_uncommitted_leaf_state_api" in result["blockers"]
    assert set(result["source_files"]) == {
        "src/mlx2/runtime/drafters/dflash2.py",
        "src/mlx2/runtime/external_speculative.py",
    }


def test_economics_include_hit_rate_overlap_and_batched_cost():
    result = probe.evaluate_economics(
        width=4,
        hit_rate=0.75,
        single_tile_ms=10.0,
        batched_tile_ms=13.333,
        overlap_fraction=0.5,
        promotion_ms=0.2,
    )

    assert result["expected_delta_ms"] == pytest.approx(0.6335)
    assert result["break_even"] is True
    assert result["performance_claim"] is False


def test_execute_refusal_is_immutable_and_pre_metal(tmp_path, monkeypatch, capsys):
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(probe, "source_revision", lambda: "source-a")

    status = probe.main(
        ["--execute", "--i-own-the-gpu", "--output", str(output)]
    )

    assert status == 2
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "refused_exact_state_unavailable"
    assert receipt["refusal_stage"] == "pre_mlx_import"
    assert receipt["metal_executed"] is False
    with pytest.raises(FileExistsError):
        probe.main(
            ["--execute", "--i-own-the-gpu", "--output", str(output)]
        )
    capsys.readouterr()


def test_dry_run_keeps_reference_and_all_candidate_widths(monkeypatch, capsys):
    monkeypatch.setattr(probe, "source_revision", lambda: "source-a")

    assert probe.main([]) == 0

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["arms"] == ["on_demand", "top_1", "top_2", "top_4"]
    assert receipt["reference_arm"] == "on_demand"
    assert receipt["metal_executed"] is False
    assert receipt["probe_state"] == {
        "cpu_validated": True,
        "implemented": True,
    }
    assert receipt["candidate_state"] == {
        "implemented": False,
        "observed_used": False,
        "performance_claim": False,
        "qualified": False,
        "selected": False,
    }
    assert receipt["candidate_contract"]["apcv2_publication"] is False
