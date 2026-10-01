"""Policy preparation inspects metadata and emits an explicit default-off route."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_xpress_artifact_metadata import artifact  # noqa: F401

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/prepare_parallel_draft_policy.py"
)
spec = importlib.util.spec_from_file_location("parallel_policy_cli", SCRIPT)
policy_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_cli)


def write_target_weights(target, dtype="BF16", second_dtype=None):
    widths = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4}
    first_size = 36 * widths[dtype]
    second_dtype = second_dtype or dtype
    end = first_size + 4 * widths[second_dtype]
    header = {
        "model.embed_tokens.weight": {
            "shape": [9, 4],
            "dtype": dtype,
            "data_offsets": [0, first_size],
        },
        "model.norm.weight": {
            "shape": [4],
            "dtype": second_dtype,
            "data_offsets": [first_size, end],
        },
    }
    raw = json.dumps(header).encode()
    (target / "model.safetensors").write_bytes(
        len(raw).to_bytes(8, "little") + raw + bytes(end)
    )


@pytest.fixture
def metadata_pack(artifact):  # noqa: F811
    draft, target, _, _ = artifact
    config = json.loads((target / "config.json").read_text())
    config.update(
        intermediate_size=7, max_position_embeddings=128, tie_word_embeddings=True
    )
    (target / "config.json").write_text(json.dumps(config))
    write_target_weights(target)
    return target, draft


def invoke(target, draft, out, *extra):
    guard = """
import builtins,runpy,sys
original=builtins.__import__
def safe_import(name,*args,**kwargs):
    if name=='mlx' or name.startswith(('mlx.','transformers','mlx2.runtime.models.')):
        raise AssertionError('live model or tensor import forbidden: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=safe_import
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            guard,
            str(SCRIPT),
            "--target",
            str(target),
            "--draft",
            str(draft),
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )


def test_off_policy_keyset_and_explicit_true(metadata_pack):
    target, draft = metadata_pack
    original = policy_cli.prepare(target, draft)
    assert set(original) == {
        "draft_model",
        "num_draft",
        "draft_revision",
        "target_fingerprint",
    }
    assert policy_cli.prepare(target, draft, target_verify_row_exact=False) == original
    selected = policy_cli.prepare(target, draft, target_verify_row_exact=True)
    assert selected == {**original, "target_verify_row_exact": True}


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_nonboolean_rejected_before_inspection(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        policy_cli.prepare("/missing", "/missing", target_verify_row_exact=value)


@pytest.mark.parametrize("selected", [False, True])
def test_cli_no_live_imports_and_reviewable_summary(metadata_pack, tmp_path, selected):
    target, draft = metadata_pack
    output = tmp_path / "policy.json"
    result = invoke(
        target, draft, output, *(["--target-verify-row-exact"] if selected else [])
    )
    assert result.returncode == 0, result.stderr
    policy = json.loads(output.read_text())
    summary = json.loads(result.stdout)
    assert summary["model_loaded"] is False and summary["qualified"] is False
    if selected:
        assert policy["target_verify_row_exact"] is True
        assert summary["target_verify_row_exact"] == {
            "selected": True,
            "qualified": False,
            "observed_used": False,
        }
    else:
        assert "target_verify_row_exact" not in policy
        assert "target_verify_row_exact" not in summary


@pytest.mark.parametrize(
    "change",
    [
        {"num_experts": 2},
        {"rope_scaling": {}},
        {"quantization": {"bits": 4, "group_size": 64}},
        {"sliding_window": 32},
        {"use_sliding_window": True},
        {"layer_types": ["sliding_attention"] * 3},
    ],
)
def test_unsupported_selected_topology_rejected_tensor_free(
    metadata_pack, tmp_path, change
):
    target, draft = metadata_pack
    config_path = target / "config.json"
    config_path.write_text(
        json.dumps({**json.loads(config_path.read_text()), **change})
    )
    output = tmp_path / "rejected.json"
    result = invoke(target, draft, output, "--target-verify-row-exact")
    assert (
        result.returncode != 0 and "unquantized dense full-attention" in result.stderr
    )
    assert "forbidden" not in result.stderr and not output.exists()


@pytest.mark.parametrize("dtype,second", [("BF16", "F32"), ("I32", None)])
def test_selected_dtype_rejected_tensor_free(metadata_pack, tmp_path, dtype, second):
    target, draft = metadata_pack
    write_target_weights(target, dtype, second)
    output = tmp_path / "rejected.json"
    result = invoke(target, draft, output, "--target-verify-row-exact")
    assert result.returncode != 0 and "homogeneous floating backbone" in result.stderr
    assert "forbidden" not in result.stderr and not output.exists()


def test_selected_custom_decode_env_and_explicit_false_cli_rejected(
    metadata_pack, tmp_path, monkeypatch
):
    target, draft = metadata_pack
    output = tmp_path / "rejected.json"
    monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
    result = invoke(target, draft, output, "--target-verify-row-exact")
    assert result.returncode != 0 and "MLX2_FP_DECODE_KERNEL" in result.stderr
    assert not output.exists()
    invalid = invoke(target, draft, output, "--target-verify-row-exact", "false")
    assert invalid.returncode != 0 and "unrecognized arguments" in invalid.stderr
