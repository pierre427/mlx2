"""MiniLM Core ML bundle builder contracts that need no Core ML runtime."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlx2.adapters import minilm
from mlx2.runtime import coreml_encoder_build as build

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path(os.environ.get(
    "MLX2_MINILM_REFERENCE_PATH",
    str(Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots" / minilm.MODEL_REVISION),
))


def _manifest(buckets=(32, 64, 128, 256)):
    per_parity = {str(s): {"cases": 3, "min_cosine": 0.9999, "max_abs_error": 0.001} for s in buckets}
    per_plan = {str(s): {"neural_engine": 150, "cpu": 10, "unknown": 0,
                         "neural_engine_compute_fraction": 0.986486} for s in buckets}
    size, digest = minilm.ARTIFACT_FILES["tokenizer.json"]
    return {
        "schema": build.SCHEMA,
        "family": "minilm",
        "source": {
            "model": build.MODEL_ID, "revision": minilm.MODEL_REVISION,
            "artifact_fingerprint": "a" * 64,
            "model_safetensors_sha256": minilm.ARTIFACT_FILES["model.safetensors"][1],
        },
        "tokenizer": {"path": "tokenizer.json", "bytes": size, "sha256": digest},
        "representation": minilm.representation_contract(),
        "precision": "float16",
        "compute_units": "cpu_and_ne",
        "buckets": [
            {"sequence_length": s, "batch": 1, "path": f"models/seq{s}.mlmodelc",
             "tree_sha256": "b" * 64,
             "inputs": {"input_ids": "int32", "attention_mask": "int32"},
             "output": "embedding", "output_shape": [1, 384]}
            for s in buckets
        ],
        "builder": {
            "tool": "mlx2.runtime.coreml_encoder_build", "source_sha256": "c" * 64,
            "coremltools": "9.0", "macos": "26.7 (25G229)", "minimum_deployment_target": "macOS14",
            "graph_layout": "sequence_major", "layer_norm": dict(build.LAYER_NORM_RECORD),
        },
        "build_parity": {
            "reference": "mlx_bert_cpu_float32", "threshold_min_cosine": 0.999,
            "compute_units": ["cpu_only", "cpu_and_ne"], "cases": 3 * len(buckets),
            "per_bucket": per_parity, "min_cosine": 0.9999, "max_abs_error": 0.001, "passed": True,
        },
        "compute_plan": {
            "available": True, "note": build.PLAN_NOTE, "per_bucket": per_plan,
            "placement_gate": build.placement_gate(per_plan),
        },
    }


def test_import_does_not_load_coreml_mlx_or_numpy():
    code = (
        "import sys, mlx2.runtime.coreml_encoder_build\n"
        "print([m for m in ('coremltools', 'mlx', 'mlx.core', 'numpy', 'torch', 'tokenizers')"
        " if m in sys.modules])"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]"


def test_cli_import_is_lazy():
    source = (ROOT / "scripts/build_coreml_minilm.py").read_text()
    head = source.split("def main", 1)[0]
    assert "import mlx" not in head and "coremltools" not in head


@pytest.mark.parametrize("buckets", [(32, 64, 128, 256), (256,), (8, 16, 256)])
def test_valid_buckets(buckets):
    assert build.validate_buckets(buckets) == buckets
    assert build.validate_buckets(list(buckets)) == buckets


@pytest.mark.parametrize("buckets, error", [
    ((), ValueError), ((64, 32, 256), ValueError), ((32, 32, 256), ValueError),
    ((32, 64, 128), ValueError), ((4, 256), ValueError), ((32, 512), ValueError),
    ((True, 256), TypeError), ((32.0, 256), TypeError), ("32,256", TypeError),
    (tuple(range(8, 256, 24)) + (256,), ValueError), (32, TypeError),
])
def test_invalid_buckets(buckets, error):
    with pytest.raises(error):
        build.validate_buckets(buckets)


def test_bucket_for_picks_smallest_fit():
    buckets = (32, 64, 128, 256)
    assert [build.bucket_for(n, buckets) for n in (1, 32, 33, 64, 65, 200, 256)] == [
        32, 32, 64, 64, 128, 256, 256]
    for bad in (0, 257, True, 3.0):
        with pytest.raises(ValueError):
            build.bucket_for(bad, buckets)


def test_prepare_output_dir(tmp_path):
    fresh = build.prepare_output_dir(tmp_path / "a" / "bundle")
    assert fresh.is_dir() and not any(fresh.iterdir())
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build.prepare_output_dir(empty) == empty
    full = tmp_path / "full"
    full.mkdir()
    (full / "old.json").write_text("{}")
    with pytest.raises(FileExistsError):
        build.prepare_output_dir(full)
    plain = tmp_path / "file"
    plain.write_text("x")
    with pytest.raises(FileExistsError):
        build.prepare_output_dir(plain)
    link = tmp_path / "link"
    link.symlink_to(empty)
    with pytest.raises(FileExistsError):
        build.prepare_output_dir(link)


def test_build_refuses_nonempty_output_before_any_work(tmp_path):
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "models").mkdir()
    with pytest.raises(FileExistsError):
        build.build_minilm_bundle(tmp_path / "missing-model", out)
    assert sorted(p.name for p in out.iterdir()) == ["models"]


def test_build_rejects_bad_arguments_before_creating_output(tmp_path):
    out = tmp_path / "bundle"
    with pytest.raises(ValueError):
        build.build_minilm_bundle(tmp_path, out, buckets=(32, 64))
    with pytest.raises(ValueError):
        build.build_minilm_bundle(tmp_path, out, layout="nchw")
    with pytest.raises(ValueError):
        build.build_minilm_bundle(tmp_path, out, parity_texts=["ok", ""])
    assert not out.exists()


def test_fabricated_manifest_validates_and_round_trips(tmp_path):
    manifest = _manifest()
    assert build.validate_manifest(manifest) is manifest
    target = build.write_manifest_atomic(tmp_path, manifest)
    assert target.name == build.MANIFEST_NAME
    assert json.loads(target.read_text()) == manifest
    assert [p.name for p in tmp_path.iterdir()] == [build.MANIFEST_NAME]
    with pytest.raises(FileExistsError):
        build.write_manifest_atomic(tmp_path, manifest)


def _mutations():
    def drop(key):
        return lambda m: m.pop(key)

    def setter(path, value):
        def apply(m):
            node = m
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
        return apply

    return {
        "missing_top_key": drop("compute_plan"),
        "extra_top_key": setter(("neural_engine_observed",), True),
        "schema": setter(("schema",), "mlx2.coreml-encoder.v2"),
        "revision": setter(("source", "revision"), "0" * 40),
        "tokenizer_digest": setter(("tokenizer", "sha256"), "d" * 64),
        "representation": setter(("representation", "qualified"), True),
        "precision": setter(("precision",), "float32"),
        "compute_units": setter(("compute_units",), "all"),
        "bucket_path": lambda m: m["buckets"][0].update(path="models/seq64.mlmodelc"),
        "bucket_digest": lambda m: m["buckets"][1].update(tree_sha256="XYZ"),
        "bucket_shape": lambda m: m["buckets"][2].update(output_shape=[1, 256]),
        "bucket_order": lambda m: m["buckets"].reverse(),
        "last_bucket": lambda m: m["buckets"].pop(),
        "builder_layer_norm": setter(("builder", "layer_norm", "epsilon"), 1e-12),
        "builder_layout": setter(("builder", "graph_layout"), "nchw"),
        "parity_failed": setter(("build_parity", "passed"), False),
        "parity_low": setter(("build_parity", "min_cosine"), 0.99),
        "parity_bucket_low": lambda m: m["build_parity"]["per_bucket"]["64"].update(min_cosine=0.5),
        "parity_bucket_missing": lambda m: m["build_parity"]["per_bucket"].pop("32"),
        "plan_note": setter(("compute_plan", "note"), "observed on ANE"),
        "plan_row_keys": lambda m: m["compute_plan"]["per_bucket"]["32"].pop("unknown"),
        "plan_gate_inconsistent": lambda m: m["compute_plan"]["placement_gate"].update(
            cpu_planned_buckets=[32]),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_manifest_mutations_are_rejected(name):
    manifest = copy.deepcopy(_manifest())
    _mutations()[name](manifest)
    with pytest.raises(ValueError):
        build.validate_manifest(manifest)


def test_manifest_records_unavailable_plan_without_gate():
    manifest = _manifest()
    manifest["compute_plan"] = {"available": False, "note": build.PLAN_NOTE,
                                "per_bucket": {}, "placement_gate": None}
    build.validate_manifest(manifest)


def test_placement_gate_marks_cpu_planned_buckets():
    rows = {"32": {"neural_engine_compute_fraction": 0.2},
            "64": {"neural_engine_compute_fraction": 0.9},
            "256": {"neural_engine_compute_fraction": 1.0}}
    assert build.placement_gate(rows) == {
        "min_neural_engine_compute_fraction": build.MIN_NE_COMPUTE_FRACTION,
        "passed_buckets": [64, 256], "cpu_planned_buckets": [32],
    }


def test_layer_norm_epsilon_is_a_normal_float16():
    eps = build.LN_EPS
    assert np.float16(eps) == eps and np.float16(eps) >= np.finfo(np.float16).tiny
    assert np.float16(build.SOURCE_LN_EPS) == 0  # why the source epsilon is not used
    assert np.float16(build.MASK_NEGATIVE) == build.MASK_NEGATIVE
    assert np.exp(np.float16(build.MASK_NEGATIVE)) == 0


def test_gpu_lock_state(tmp_path):
    assert build.gpu_lock_state(tmp_path / "absent" / "owner.json")["latency_allowed"]
    released = tmp_path / "gpu.lock"
    released.write_text("")
    assert build.gpu_lock_state(released / "owner.json") == {
        "held": False, "label": None, "latency_allowed": True}
    held = tmp_path / "held.lock"
    held.mkdir()
    (held / "owner.json").write_text(json.dumps({"label": "qwen-ladder-128k"}))
    assert build.gpu_lock_state(held / "owner.json")["latency_allowed"] is False
    (held / "owner.json").write_text(json.dumps({"label": "dspark-family-paired-b4"}))
    assert build.gpu_lock_state(held / "owner.json") == {
        "held": True, "label": "dspark-family-paired-b4", "latency_allowed": True}
    (held / "owner.json").write_text("{not json")
    assert build.gpu_lock_state(held / "owner.json")["latency_allowed"] is False


def test_padded_inputs_and_output_checks():
    feed = build.padded_inputs([101, 7592, 102], 8)
    assert feed["input_ids"].dtype == np.int32 and feed["input_ids"].shape == (1, 8)
    assert feed["input_ids"].tolist() == [[101, 7592, 102, 0, 0, 0, 0, 0]]
    assert feed["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]
    for ids in ([], list(range(9))):
        with pytest.raises(ValueError):
            build.padded_inputs(ids, 8)
    unit = np.full((1, 384), 384 ** -0.5)
    for dtype in (np.float16, np.float32):
        assert build.checked_embedding({"embedding": unit.astype(dtype)}).shape == (384,)
    bad = [
        {"embedding": unit.astype(np.float64)},
        {"embedding": unit[:, :256].astype(np.float32)},
        {"embedding": (unit * 2).astype(np.float32)},
        {"embedding": np.full((1, 384), np.nan, np.float32)},
        {"embedding": unit.astype(np.float32), "extra": unit},
    ]
    for result in bad:
        with pytest.raises(ValueError):
            build.checked_embedding(result)


def test_default_parity_texts_are_varied():
    texts = build.DEFAULT_PARITY_TEXTS
    assert len(set(texts)) == len(texts) >= 20
    for text in texts:
        words = text.lower().split()
        # Realistic prose, never repeated-string filler: high lexical variety.
        assert len(set(words)) / len(words) > 0.45, text[:40]


def test_default_parity_texts_cover_every_bucket_and_truncate():
    if not (ARTIFACT / "tokenizer.json").is_file():
        pytest.skip("pinned MiniLM local artifact is absent; never download in tests")
    pytest.importorskip("tokenizers")
    tokenizer_bytes = minilm._read_pinned(ARTIFACT.resolve(), "tokenizer.json")
    long = build.make_tokenizer(tokenizer_bytes, max_length=4096)
    counts = [len(long.encode(text).ids) for text in build.DEFAULT_PARITY_TEXTS]
    natural = [build.bucket_for(min(n, 256), build.DEFAULT_BUCKETS) for n in counts]
    for size in build.DEFAULT_BUCKETS:
        assert natural.count(size) >= 3
    assert sum(n > 256 for n in counts) >= 3


def test_real_bundle_build_two_buckets(tmp_path):
    """Full build path; requires coremltools, MLX and the pinned artifact."""
    pytest.importorskip("coremltools")
    if not (ARTIFACT / "model.safetensors").is_file():
        pytest.skip("pinned MiniLM local artifact is absent; never download in tests")
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    texts = build.DEFAULT_PARITY_TEXTS[:3] + build.DEFAULT_PARITY_TEXTS[-1:]
    manifest = build.build_minilm_bundle(ARTIFACT, tmp_path / "b", buckets=(32, 256), parity_texts=texts)
    assert manifest["build_parity"]["passed"] and manifest["build_parity"]["min_cosine"] >= 0.999
    bundle = tmp_path / "b"
    assert json.loads((bundle / build.MANIFEST_NAME).read_text()) == manifest
    for bucket in manifest["buckets"]:
        assert build.compiled_model_digest(bundle / bucket["path"]) == bucket["tree_sha256"]
