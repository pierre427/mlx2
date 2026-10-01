"""Host-only matrix plan and receipt boundaries, plus CPU BF16 conversion."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_xpress_metal_matrix.py"
spec = importlib.util.spec_from_file_location("matrix_cli", SCRIPT)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def args(*extra):
    return matrix.build_parser().parse_args(
        [
            "--model",
            "/missing-target",
            "--draft",
            "/missing-draft",
            "--out",
            "/tmp/matrix-not-written",
            *extra,
        ]
    )


def test_missing_ownership_refuses_before_tensor_imports():
    with pytest.raises(ValueError, match="i-own-the-gpu"):
        matrix.preflight(args())


@pytest.mark.parametrize("timeout", ["0", "901"])
def test_timeout_is_bounded(timeout):
    with pytest.raises(ValueError, match="timeout"):
        matrix.preflight(args("--dry-run", "--timeout-seconds", timeout))


def test_complete_matrix_geometry_and_special_cells():
    plan = matrix.preflight(args("--dry-run"))
    cells = plan["cells"]
    assert len(cells) == 14
    assert {c["context"] for c in cells} == {128, 1024, 4096}
    assert {c["width"] for c in cells} == {1, 2, 4}
    assert {c["passes"] for c in cells} == {1, 3, 6}
    assert {budget for c in cells for budget in c["budgets"]} == {1, 17, 48}
    assert all(len(c["budgets"]) == c["width"] for c in cells)
    assert any(c.get("fault") for c in cells) and any(c.get("cancel") for c in cells)
    assert sum(bool(c.get("french")) for c in cells) == 2
    assert any(c.get("warm") for c in cells) and any(c.get("windows") for c in cells)
    assert plan["draft_attention_windows"] == [32, 128, None]
    assert matrix.resolve_attention_windows(plan, 3) == [32, 128, None]
    assert matrix.resolve_attention_windows(plan, 5) == [32, 128, None, 32, 128]
    assert plan["performance_claim"] is False and plan["qualified"] is False
    assert plan["compute_precision_requested"] == "artifact"


def test_float32_mode_is_explicit_and_choices_fail_closed():
    plan = matrix.preflight(
        args("--dry-run", "--compute-precision", "float32-diagnostic")
    )
    assert (
        plan["compute_precision_requested"] == "float32-diagnostic"
        and plan["qualified"] is False
    )
    with pytest.raises(SystemExit):
        args("--dry-run", "--compute-precision", "float16")


def test_target_row_exact_is_explicit_strict_boolean():
    assert matrix.preflight(args("--dry-run"))["target_verify_row_exact"] is False
    selected = args("--dry-run", "--target-verify-row-exact")
    assert matrix.preflight(selected)["target_verify_row_exact"] is True
    with pytest.raises(SystemExit):
        args("--dry-run", "--target-verify-row-exact", "false")
    selected.target_verify_row_exact = 1
    with pytest.raises(ValueError, match="must be boolean"):
        matrix.preflight(selected)


def test_float32_memory_guard_requires_safe_known_footprint():
    receipt = matrix.guard_float32_footprint(17 << 30, 1 << 30, 24 << 30, 27 << 30)
    assert receipt["required_bytes"] == 22 << 30
    with pytest.raises(ValueError, match="unsafe footprint"):
        matrix.guard_float32_footprint(25 << 30, 1 << 30, 24 << 30, 27 << 30)
    with pytest.raises(ValueError, match="known positive"):
        matrix.guard_float32_footprint(17 << 30, 1 << 30, 24 << 30, 0)


def test_explicit_window_config_must_match_checkpoint_layers(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"draft_attention_windows": [32, 128, None]}))
    plan = matrix.preflight(args("--dry-run", "--config", str(config)))
    assert matrix.resolve_attention_windows(plan, 3) == [32, 128, None]
    with pytest.raises(ValueError, match="explicit window config"):
        matrix.resolve_attention_windows(plan, 5)


@pytest.mark.parametrize(
    "value",
    [[], [True, 32, None], [0, 32, None], [-1, 32, None], [4097, 32, None], "32"],
)
def test_window_config_fails_closed(tmp_path, value):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"draft_attention_windows": value}))
    with pytest.raises(ValueError):
        matrix.preflight(args("--dry-run", "--config", str(config)))


def test_dry_run_cannot_import_tensor_libraries():
    code = """
import sys,runpy
class Guard:
    def find_spec(self,fullname,*args,**kwargs):
        if fullname.startswith(('mlx','numpy','transformers')):
            raise RuntimeError('tensor import forbidden: '+fullname)
sys.meta_path.insert(0,Guard())
sys.argv=[sys.argv[1],'--model','/missing-target','--draft','/missing-draft','--out','/missing-output','--dry-run','--compute-precision','float32-diagnostic']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["will_execute"] is False


def test_float32_diagnostic_cast_preserves_shared_binding_and_source_values_cpu():
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    mx.set_default_device(mx.cpu)
    from test_standard_xpress_serving_cpu import tiny

    target, draft = tiny()
    target.set_dtype(mx.bfloat16)
    draft.set_dtype(mx.bfloat16)
    original = matrix.host_float32(target.model.embed_tokens.weight).copy()
    original_config = draft.config
    matrix.cast_float32_diagnostic(target, draft)
    assert (
        draft.embed_tokens is target.model.embed_tokens
        and draft.lm_head is target.lm_head
    )
    for model in (target, draft):
        assert all(
            value.dtype == mx.float32
            for _, value in tree_flatten(model.parameters())
            if mx.issubdtype(value.dtype, mx.floating)
        )
    np.testing.assert_array_equal(
        matrix.host_float32(target.model.embed_tokens.weight), original
    )
    assert draft.config is original_config
    logits, taps = target.forward_with_taps(
        mx.array([[1, 2]]), target.make_cache(), [0, 2]
    )
    assert logits.dtype == mx.float32 and taps.dtype == mx.float32


def test_header_forecast_matches_parameters_and_refuses_quantization(tmp_path):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    mx.save_safetensors(
        str(artifact / "model.safetensors"),
        {"weight": mx.ones((3, 4), dtype=mx.bfloat16)},
    )
    resident, largest, dtypes = matrix.float32_artifact_forecast([artifact])
    assert (resident, largest, dtypes) == (48, 24, {"BF16": 12})
    (artifact / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    with pytest.raises(ValueError, match="unquantized"):
        matrix.float32_artifact_forecast([artifact])


def test_bfloat16_boundary_and_margin_diagnostics_cpu():
    import mlx.core as mx
    import numpy as np

    mx.set_default_device(mx.cpu)
    values = matrix.host_float32(
        mx.array([2.0, 3.0, 2.5, 0.0, -1.0, 1.0], dtype=mx.bfloat16)
    )
    assert values.dtype == np.float32
    diagnostic = matrix.top_logits(values)
    assert diagnostic["ids"][:2] == [1, 2] and diagnostic["top2_margin"] == 0.5


def test_source_identity_binds_script_and_all_python_sources():
    identity = matrix.source_identity()
    assert len(identity["source_sha256"]) == 64
    assert "scripts/validate_xpress_metal_matrix.py" in identity["files_sha256"]
    assert "src/mlx2/runtime/external_speculative.py" in identity["files_sha256"]


def test_real_cpu_body_only_prefill_observer_skips_missing_logits(monkeypatch):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx2.runtime.models.standard_decoder import Model, ModelArgs

    target = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=8,
            num_hidden_layers=3,
            intermediate_size=12,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=9,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=128,
            rope_theta=10000,
            tie_word_embeddings=False,
        )
    )
    original = target.forward_with_taps
    observed = []

    def spy(*args, **kwargs):
        logits, taps = original(*args, **kwargs)
        observed.append(matrix.prediction_rows(logits))
        return logits, taps

    monkeypatch.setattr(target, "forward_with_taps", spy)
    taps = target.prefill_body(mx.array([[1, 2]]), target.make_cache(), [0, 2])
    assert taps.shape == (1, 2, 16) and observed == [None]
    target.forward_with_taps(mx.array([[1]]), target.make_cache(), [0, 2])
    assert observed[-1] is not None and len(observed[-1][0]) == 1


def test_append_only_recovery_excludes_unused_capacity_but_detects_live_corruption():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx2.runtime.cow_cache import (
        restore_recovery_descriptors,
        snapshot_recovery_descriptors,
    )
    from mlx2.runtime.models.cache import KVCache

    cache = KVCache()
    cache.update_and_fetch(mx.ones((1, 1, 3, 2)), mx.ones((1, 1, 3, 2)))
    before = matrix.cache_diagnostics([cache])[0]
    frozen, sidecar, borrowed = snapshot_recovery_descriptors([cache])
    cache.update_and_fetch(2 * mx.ones((1, 1, 1, 2)), 3 * mx.ones((1, 1, 1, 2)))
    restored, _ = restore_recovery_descriptors(frozen, sidecar, borrowed)
    after = matrix.cache_diagnostics(restored)[0]
    assert restored[0].offset == 3
    assert before["semantic_sha256"] == after["semantic_sha256"]
    assert before["allocated_sha256"] != after["allocated_sha256"]
    restored[0].keys[..., 0, :] = 99
    assert (
        matrix.cache_diagnostics(restored)[0]["semantic_sha256"]
        != before["semantic_sha256"]
    )


def test_rotating_cache_retained_storage_is_semantic():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx2.runtime.models.cache import RotatingKVCache

    cache = RotatingKVCache(max_size=2, keep=0)
    for _ in range(4):
        cache.update_and_fetch(mx.ones((1, 1, 1, 2)), mx.ones((1, 1, 1, 2)))
    before = matrix.cache_diagnostics([cache])[0]["semantic_sha256"]
    cache.keys[..., 0, :] = 99
    assert matrix.cache_diagnostics([cache])[0]["semantic_sha256"] != before


def test_real_cpu_executor_failure_restores_semantics_with_changed_unused_draft_capacity(
    monkeypatch,
):
    import copy

    import mlx.core as mx
    import numpy as np

    mx.set_default_device(mx.cpu)
    from test_standard_xpress_serving_cpu import generator, tiny

    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors

    model, draft = tiny()
    engine = generator(model, draft)
    engine.insert([[1, 2, 3], [3, 2, 1]], max_tokens=[8, 8])
    lanes = list(engine.lanes.values())
    for lane in lanes:
        while lane.anchor is None:
            engine._prefill(lane)
    engine._round(lanes)

    def state(lane):
        branch, _, _ = snapshot_prompt_cache_descriptors(lane.cache)
        logits = matrix.host_float32(model(mx.array([[lane.anchor]]), cache=branch))
        return {
            "history": list(lane.history),
            "rng": copy.deepcopy(lane.rng.snapshot()),
            "anchor": lane.anchor,
            "target": matrix.cache_diagnostics(lane.cache),
            "draft": matrix.cache_diagnostics(lane.draft_cache),
            "tail": matrix.host_float32(lane.tail).copy(),
            "logits": logits.copy(),
        }

    before = [state(lane) for lane in lanes]
    original = model.forward_with_taps

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("after real target write")

    monkeypatch.setattr(model, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="after real target write"):
        engine._round(lanes)
    monkeypatch.setattr(model, "forward_with_taps", original)
    allocated_changed = False
    for lane, old in zip(lanes, before):
        after = state(lane)
        for field in ("history", "rng", "anchor"):
            assert after[field] == old[field]
        for plane in ("target", "draft"):
            assert [c["semantic_sha256"] for c in after[plane]] == [
                c["semantic_sha256"] for c in old[plane]
            ]
            allocated_changed |= any(
                a["allocated_sha256"] != b["allocated_sha256"]
                for a, b in zip(after[plane], old[plane])
            )
        np.testing.assert_array_equal(after["tail"], old["tail"])
        np.testing.assert_array_equal(after["logits"], old["logits"])
    assert allocated_changed
    engine._round(lanes)
