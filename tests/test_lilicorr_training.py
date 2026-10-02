"""Frozen teacher capture and separate bounded CPU shadow training."""

import hashlib
import json

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from mlx.utils import tree_flatten
from test_lilicorr_cpu import config, initialized_head

from mlx2.runtime.lilicorr_training import TeacherBuffer, export_shadow, train_shadow


def example():
    rng = np.random.default_rng(27)
    return {
        "candidate_ids": np.array([[0, 1], [2, 3], [4, 5]]),
        "token_embeddings": rng.normal(size=(3, 2, 4)).astype(np.float32),
        "candidate_log_probs": np.array([[-0.8, -1.3]] * 3, np.float32),
        "pass_hidden": rng.normal(size=(3, 4)).astype(np.float32),
        "anchor_hidden": rng.normal(size=(4,)).astype(np.float32),
        "anchor_valid": True,
        "teacher_tokens": [1, 3, 5],
    }


def buffer(**kw):
    return TeacherBuffer(
        config(), target_revision="a" * 40, draft_revision="b" * 64, **kw
    )


def test_rejected_tail_censored_and_teacher_missing_ends_path():
    data = buffer()
    record = example()
    record["first_rejected_position"] = 0
    assert data.add(**record)
    assert data.examples[0].teacher_columns == (1,)
    record = example()
    record["teacher_tokens"] = [1, 8, 5]
    assert data.add(**record)
    assert data.examples[1].teacher_columns == (1,)
    record["teacher_tokens"] = [8, 3, 5]
    assert not data.add(**record)
    assert len(data.examples) == 2


def test_teacher_buffer_copies_freezes_and_enforces_fifo_budget():
    data = buffer(max_examples=1, max_bytes=4096)
    record = example()
    data.add(**record)
    frozen = data.examples[0]
    record["token_embeddings"][:] = 99
    assert not np.all(frozen.token_embeddings == 99)
    assert not frozen.token_embeddings.flags.writeable
    data.add(**example())
    assert len(data.examples) == 1 and data.dropped == 1 and data.nbytes <= 4096
    with pytest.raises(ValueError, match="byte budget"):
        buffer(max_bytes=1).add(**example())


@pytest.mark.parametrize(
    "field", ["token_embeddings", "candidate_log_probs", "pass_hidden", "anchor_hidden"]
)
def test_teacher_cast_overflow_refuses_before_fifo_or_buffer_mutation(field):
    data = buffer(max_examples=1, max_bytes=4096)
    assert data.add(**example())
    original = data.examples[0]
    before = data.nbytes, data.dropped
    record = example()
    record[field] = np.full(
        record[field].shape,
        -1e300 if field == "candidate_log_probs" else 1e300,
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="remain finite"):
        data.add(**record)
    assert data.examples == [original]
    assert (data.nbytes, data.dropped) == before


def test_shadow_loss_decreases_without_touching_source_and_exports_unselected(tmp_path):
    head = initialized_head()
    original = {k: np.asarray(v).copy() for k, v in tree_flatten(head.parameters())}
    data = buffer()
    data.add(**example())
    trained = train_shadow(head, data, steps=25, learning_rate=0.01)
    assert trained.final_loss < trained.initial_loss * 0.8
    for key, value in tree_flatten(head.parameters()):
        np.testing.assert_array_equal(np.asarray(value), original[key])
    manifest = export_shadow(trained, tmp_path / "shadow")
    assert (
        not manifest["qualified"]
        and not manifest["selected"]
        and not manifest["observed_used"]
    )
    tensor = tmp_path / "shadow" / "head.safetensors"
    assert (
        manifest["files"][0]["sha256"]
        == hashlib.sha256(tensor.read_bytes()).hexdigest()
    )
    assert (
        manifest["target_revision"] == "a" * 40
        and manifest["draft_revision"] == "b" * 64
    )
    assert json.loads((tmp_path / "shadow" / "manifest.json").read_text()) == manifest
    assert not (tmp_path / "shadow" / "config.json").exists()
    with pytest.raises(FileExistsError):
        export_shadow(trained, tmp_path / "shadow")


def test_training_refuses_empty_data_and_invalid_budget():
    with pytest.raises(ValueError, match="examples"):
        train_shadow(initialized_head(), buffer(), steps=1)
    data = buffer()
    data.add(**example())
    with pytest.raises(ValueError, match="steps"):
        train_shadow(initialized_head(), data, steps=0)
    with pytest.raises(ValueError, match="pinned"):
        TeacherBuffer(config(), target_revision="main", draft_revision="b" * 64)
    data.config.lilicorr_logit_scale = 2.0
    with pytest.raises(ValueError, match="geometry changed"):
        train_shadow(initialized_head(), data, steps=1)


def test_export_rechecks_geometry_and_trained_parameters_before_publication(tmp_path):
    data = buffer()
    data.add(**example())
    trained = train_shadow(initialized_head(), data, steps=2)
    trained.config.lilicorr_logit_scale = 2.0
    with pytest.raises(ValueError, match="geometry changed"):
        export_shadow(trained, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
    trained.config.lilicorr_logit_scale = 3.0
    trained.head.out_head.weight = trained.head.out_head.weight + 1
    with pytest.raises(ValueError, match="parameters changed"):
        export_shadow(trained, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
