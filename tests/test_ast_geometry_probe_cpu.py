"""Diagnostic wrappers preserve the existing real-tensor CPU AST contract."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/diagnose_ast_geometry.py"
SPEC = importlib.util.spec_from_file_location("ast_geometry_probe", PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_capture_is_localized_to_one_prompt_and_position():
    options = {"skip_writing_prefix_cache": True}
    assert PROBE.capture_options("other", "code", 23, options) == options
    assert PROBE.capture_options("code", "code", None, options) == options
    result = PROBE.capture_options("code", "code", 23, options)
    assert result["logprobs_start"] == result["logprobs_end"] == 23
    assert result["top_logprobs"] == 20 and result["logprobs"]
    assert options == {"skip_writing_prefix_cache": True}
    with pytest.raises(ValueError, match="positive"):
        PROBE.capture_options("code", "code", 0, options)


def test_lazy_install_does_not_import_model_runtime(tmp_path):
    code = (
        "import runpy, sys; "
        f"probe=runpy.run_path({str(PATH)!r}); "
        f"rec=probe['Recorder'](__import__('pathlib').Path({str(tmp_path / 'lazy.jsonl')!r})); "
        "rec.install_lazily(); "
        "assert 'mlx2.runtime.models.base' not in sys.modules; "
        "assert 'mlx2.runtime.external_speculative' not in sys.modules; "
        "rec.output.close()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_redacted_topology_preserves_sharing_and_rank():
    result = PROBE.topology([[91, 72, 83], [91, 72, 44], [91, 35]])
    assert result["paths"] == [[1, 2, 3], [1, 2, 4], [1, 5]]
    assert result["unique_nodes"] == 5
    assert result["path_depths"] == [3, 3, 2]
    assert "91" not in json.dumps(result)


def test_recorder_rejects_tensor_like_values_without_reading(tmp_path):
    class Tensor:
        def __str__(self):
            raise AssertionError("must not serialize tensor")

    with pytest.raises(TypeError, match="non-host"):
        PROBE.host_value(Tensor())
    recorder = PROBE.Recorder(tmp_path / "trace.jsonl", max_events=1)
    row = {"events": []}
    token = recorder.current.set(row)
    try:
        recorder.event("first", shape=[1, 4])
        with pytest.raises(RuntimeError, match="bound exceeded"):
            recorder.event("second")
    finally:
        recorder.current.reset(token)
        recorder.output.close()


def test_instrumented_mixed_topology_keeps_cpu_contract(monkeypatch, tmp_path):
    from mlx2.runtime import external_speculative as ext
    from mlx2.runtime import row_exact_verify as rev
    from mlx2.runtime.proposal_providers import ContinuationDraftModel
    from test_continuation_providers_cpu import (
        test_continuation_ast_rootless_dflash_shape_cross_session_packs,
    )

    cls = ext.ExternalDraftBatchGenerator
    targets = [(cls, name) for name in ("_run_round", "_propose", "_progressive_verify_many",
                                       "_record_ast_geometry", "_commit", "_cohort_target_forward")]
    targets += [(ext, "execute_progressive_target_rows"), (rev, "window"),
                (ContinuationDraftModel, "draft_distributions")]
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    recorder = PROBE.Recorder(tmp_path / "trace.jsonl")
    try:
        recorder.install()
        test_continuation_ast_rootless_dflash_shape_cross_session_packs(monkeypatch)
    finally:
        for owner, name, value in originals:
            setattr(owner, name, value)
        recorder.output.close()
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert rows and all(row["completed"] for row in rows)
    events = [event for row in rows for event in row["events"]]
    assert {"source_slate", "admitted_topology", "target_tile", "tile_group_result",
            "verified_selection", "committed"} <= {event["kind"] for event in events}
    tiles = [event for event in events if event["kind"] == "target_tile"]
    assert all(len(tile["mapping"]) == tile["shape"][0] for tile in tiles)
    assert any(len({item["uid"] for item in tile["mapping"]}) > 1 for tile in tiles)
