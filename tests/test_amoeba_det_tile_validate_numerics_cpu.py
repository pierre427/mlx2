"""Calculator det tile under the Hy4 fusions: ``validated_numerics`` admission and the
validation script's assessment (scripts/amoeba_det_tile_validate_numerics.py), on CPU.

The script's real run (the Hy4 checkpoint, GPU) is not exercised here; a smoke run on the
tiny random checkpoint covers its plumbing end to end.
"""

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_det_tiles_cpu import (  # noqa: E402
    FIT,
    _char_artifact,
    _manifest,
    tiny,  # noqa: F401  (fixture)
)

from mlx2.adapters import hy_v4_flash as A  # noqa: E402
from mlx2.runtime import amoeba_det_tiles as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("validate_numerics",
                                               ROOT / "scripts" / "amoeba_det_tile_validate_numerics.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
mx.set_default_device(mx.cpu)


@pytest.fixture(scope="module")
def artifact(tiny, tmp_path_factory):  # noqa: F811
    config, W, _, _ = tiny
    return _char_artifact(tmp_path_factory.mktemp("hy4-validate") / "Hy4-tiny", config, W)


def _adapter(artifact, policy):
    return A.HYV4FlashAdapter(str(artifact), execution_policy=policy)


# ------------------------------------------------------------- assessment logic --

def test_assess_identical_scores_pass():
    ref = np.array([-3.0, -0.5, 0.5, 4.0])
    r = V.assess(ref, ref.copy(), 0.0)
    assert r["passed"] and r["changed_rows"] == 0 and r["max_abs_delta"] == 0.0
    assert r["min_margin_changed_rows"] is None and r["fires_ref"] == r["fires_arm"] == 2
    assert r["min_margin_ref"] == pytest.approx(0.5)


def test_assess_small_changes_far_from_threshold_pass():
    ref = np.array([-3.0, -1.0, 1.0, 4.0])
    arm = ref + np.array([1e-3, 0.0, -2e-3, 0.0])
    r = V.assess(ref, arm, 0.0)
    assert r["passed"] and r["changed_rows"] == 2
    assert r["max_abs_delta"] == pytest.approx(2e-3)
    assert r["min_margin_changed_rows"] == pytest.approx(0.998)


@pytest.mark.parametrize("ref,arm,on,off", [([-0.05, -1.0], [0.1, -1.0], 1, 0),
                                           ([0.05, -3.0], [-0.1, -3.0], 0, 1)])
def test_assess_any_flip_fails(ref, arm, on, off):
    r = V.assess(np.array(ref), np.array(arm), 0.0)
    assert not r["passed"] and "flips" in r["reason"]
    assert (r["flips_closed_to_open"], r["flips_open_to_closed"]) == (on, off)


def test_assess_changed_row_inside_the_delta_fails_without_a_flip():
    # Row 0 moves by 0.3 and stays open, but row 1 (changed by 0.01) sits 0.2 from the
    # threshold: closer than the largest change, so a slightly larger change could flip it.
    ref = np.array([5.0, 0.2])
    arm = np.array([5.3, 0.21])
    r = V.assess(ref, arm, 0.0)
    assert r["flips_closed_to_open"] == r["flips_open_to_closed"] == 0
    assert not r["passed"] and "margin" in r["reason"]


def test_assess_rejects_mismatched_or_empty_rows():
    with pytest.raises(ValueError):
        V.assess(np.zeros(3), np.zeros(4), 0.0)
    assert V.assess(np.zeros(0), np.zeros(0), 0.0)["passed"] is False


def test_validated_entries_keep_passing_fit_keys_only():
    arms = {"hc_seam_fused": {"passed": True}, "engram_gate_rounding": {"passed": False},
            "indexer_fused_scores": {"passed": True}, "mla_fused_input_proj": {"passed": True}}
    assert V.validated_entries({"arms": arms}) == {"hc_seam_fused": [True], "indexer_fused_scores": [True]}
    assert V.validated_entries({"arms": {**arms, "all": {"passed": False}}}) == {}
    assert V.validated_entries({"arms": {**arms, "all": {"passed": True}}}) == {
        "hc_seam_fused": [True], "indexer_fused_scores": [True]}


def test_manifest_copy_adds_the_field_and_never_rewrites_the_source(artifact, tmp_path):
    src = Path(_manifest(tmp_path / "src", artifact))
    before = src.read_text()
    with pytest.raises(SystemExit, match="overwrite-source"):
        V.write_manifest_copy(src, src.parent, {"hc_seam_fused": [True]}, "ev.json")
    assert src.read_text() == before
    out = V.write_manifest_copy(src, tmp_path / "copy", {"hc_seam_fused": [True]}, "/runs/ev.json")
    loaded = D.load_manifest(out)      # file digests still check
    assert loaded["validated_numerics"] == {"hc_seam_fused": [True]}
    assert loaded["validated_numerics_evidence"] == "/runs/ev.json"
    with pytest.raises(SystemExit, match="nonempty"):
        V.write_manifest_copy(src, tmp_path / "copy", {}, "ev.json")
    in_place = V.write_manifest_copy(src, src.parent, {"hc_seam_fused": [True]}, "ev.json",
                                     overwrite_source=True)
    assert json.loads(in_place.read_text())["validated_numerics"] == {"hc_seam_fused": [True]}


# --------------------------------------------------------------- adapter admission --

VALIDATED = {"validated_numerics": {"hc_seam_fused": [True], "indexer_fused_scores": [True]},
             "validated_numerics_evidence": "/runs/hy4-calc-fusions/evidence.json"}


def test_listed_values_are_admitted_with_a_receipt(artifact, tmp_path):
    man = _manifest(tmp_path / "v", artifact, extra={"fit_execution_policy": dict(FIT), **VALIDATED})
    adapter = _adapter(artifact, {"amoeba_det_tiles": man, "hc_seam_fused": True,
                                  "indexer_fused_scores": True})
    try:
        det = adapter._det_tiles_receipt()
        assert adapter.amoeba_det_tiles is not None and det["selected"]
        assert det["validated_equivalent"] == [
            "admitted: validated equivalent (hc_seam_fused=true, evidence /runs/hy4-calc-fusions/evidence.json)",
            "admitted: validated equivalent (indexer_fused_scores=true, evidence "
            "/runs/hy4-calc-fusions/evidence.json)"]
    finally:
        adapter.close()
    # Fusions off: the fit numerics themselves, nothing to admit.
    adapter = _adapter(artifact, {"amoeba_det_tiles": man})
    try:
        assert adapter._det_tiles_receipt()["validated_equivalent"] == []
    finally:
        adapter.close()


def test_unlisted_values_keep_the_refusal(artifact, tmp_path):
    man = _manifest(tmp_path / "u", artifact, extra={"fit_execution_policy": dict(FIT), **VALIDATED})
    with pytest.raises(D.DetTileRefused, match="engram_gate_rounding"):
        _adapter(artifact, {"amoeba_det_tiles": man, "engram_gate_rounding": "torch"})
    with pytest.raises(D.DetTileRefused, match="engram_fused_proj"):
        _adapter(artifact, {"amoeba_det_tiles": man, "hc_seam_fused": True, "engram_fused_proj": True})
    # A listed key does not widen the core fit keys.
    with pytest.raises(D.DetTileRefused, match="engram_precision"):
        _adapter(artifact, {"amoeba_det_tiles": man, "hc_seam_fused": True, "engram_precision": "vllm"})


def test_older_manifest_without_the_field_is_unchanged(artifact, tmp_path):
    man = _manifest(tmp_path / "old", artifact)
    with pytest.raises(D.DetTileRefused, match="hc_seam_fused"):
        _adapter(artifact, {"amoeba_det_tiles": man, "hc_seam_fused": True})
    adapter = _adapter(artifact, {"amoeba_det_tiles": man})
    try:
        det = adapter._det_tiles_receipt()
        assert det["selected"] and det["validated_equivalent"] == []
    finally:
        adapter.close()


@pytest.mark.parametrize("extra", [
    {"validated_numerics": {"hc_seam_fused": [True]}},                                   # no evidence
    {"validated_numerics": {"hc_seam_fused": True}, "validated_numerics_evidence": "e"},  # not a list
    {"validated_numerics": {"vision": [False]}, "validated_numerics_evidence": "e"},     # not a fit key
    {"validated_numerics": [["hc_seam_fused", True]], "validated_numerics_evidence": "e"},
])
def test_malformed_field_fails_closed(artifact, tmp_path, extra):
    man = _manifest(tmp_path / "bad", artifact, extra={"fit_execution_policy": dict(FIT), **extra})
    with pytest.raises(D.DetTileRefused, match="validated_numerics"):
        _adapter(artifact, {"amoeba_det_tiles": man})


def test_shipped_manifest_has_no_field_and_still_admits():
    path = A.default_det_tile_manifest("f2a8fa8d2b7f22150ee5bd988c5acc1814d8494aa82ee01aff40a7efc13aad93")
    manifest = D.load_manifest(path)
    assert "validated_numerics" not in manifest
    assert A.det_tile_validated_numerics(manifest) == ({}, None)


# ------------------------------------------------------------------- smoke run --

def test_script_smoke_on_the_tiny_checkpoint(artifact, tmp_path):
    """End to end on CPU: one load, the arms switched in place, evidence and a validated copy.

    On CPU the seam and indexer kernels run the reference ops, so those arms cannot engage
    and fail as not_exercised; the gate-rounding arm is a no-op in fp32 (zero deltas) and
    passes; fusions off is restored at the end."""
    man = _manifest(tmp_path / "tile", artifact)
    out = tmp_path / "run"
    ev = V.main(["--out", str(out), "--model", str(artifact), "--manifest", man, "--base-policy", "{}",
                 "--arms", "off,hc_seam_fused,engram_gate_rounding,mla_fused_input_proj,indexer_fused_scores",
                 "--splits", "val", "--n-pos", "4", "--n-neg", "1", "--max-sequences", "6",
                 "--decode-sequences", "2", "--long-prefix-tokens", "20", "--arith-items", "1", "--max-new", "4",
                 "--device", "cpu", "--no-wire", "--no-watchdog", "--chat-format", "Q: {q}\nA:",
                 "--corpus", str(tmp_path / "no-corpus.json"), "--write-manifest", str(tmp_path / "validated")])
    saved = json.loads((out / "evidence.json").read_text())
    assert saved["schema"] == V.SCHEMA and saved["checkpoint"].startswith("Hy4-tiny@")
    assert saved["mlx2_source_revision"] and saved["arms"]["off"]["prefill_rows"] > 0
    arms = ev["arms"]
    assert arms["hc_seam_fused"]["not_exercised"] == ["hc_seam_metal"] and not arms["hc_seam_fused"]["passed"]
    assert arms["indexer_fused_scores"]["not_exercised"] == ["indexer_fused_calls"]
    gate = arms["engram_gate_rounding"]
    assert gate["passed"] and gate["prefill"]["max_abs_delta"] == 0.0 and gate["decode"]["rows"] > 0
    assert arms["mla_fused_input_proj"]["engaged"]["mla_fused_calls"] > 0
    assert arms["mla_fused_input_proj"]["prefill"]["flips_closed_to_open"] == 0
    assert ev["validated_numerics"] == {"engram_gate_rounding": ["torch"]}
    copy = D.load_manifest(ev["validated_manifest"])
    assert copy["validated_numerics"] == {"engram_gate_rounding": ["torch"]}
    assert copy["validated_numerics_evidence"] == str((out / "evidence.json").resolve())
    with pytest.raises(SystemExit, match="nonempty"):
        V.main(["--out", str(out), "--model", str(artifact)])
